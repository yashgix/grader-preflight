"""Relations: small edits to a gold run whose expected effect on the grader's verdict is known in advance.

Each generator builds candidate variants for one relation. :func:`build_variants` is the single entry point: it
drops variants that equal the gold run and the P5 variant the live environment shows is not inert, and reports
each skip as a note for the task's static entry.
"""

import inspect
import json
import random
import re
from dataclasses import dataclass

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment

from preflight.tau2_adapter import Call, fresh_env, gold_calls, run

AGENT_WRAPPER = "call_discoverable_agent_tool"
USER_WRAPPER = "call_discoverable_user_tool"
NAME_KEY = {AGENT_WRAPPER: "agent_tool_name", USER_WRAPPER: "discoverable_tool_name"}
UNLOCK_AGENT = "unlock_discoverable_agent_tool"
UNLOCK_USER = "unlock_discoverable_user_tool"
UNLOCK_KEY = {UNLOCK_AGENT: "agent_tool_name", UNLOCK_USER: "discoverable_tool_name"}
READ = "read"
WRITE = "write"
KINDS = {"read", "write", "generic", "think"}
MAX_P1_VARIANTS = 5
UNKNOWN_TOOL = "no_such_tool"
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_MISSING = object()


@dataclass(frozen=True)
class Variant:
    relation: str
    calls: tuple[Call, ...]
    expected_pass: bool
    detail: str
    inserted_in_gold: bool = False


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calls_as_dicts(calls) -> list[dict]:
    return [{"name": call.name, "arguments": call.arguments, "requestor": call.requestor} for call in calls]


def _loads(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return _MISSING


def _toolkit_tools(toolkit, discoverable: bool = False) -> dict:
    if toolkit is None:
        return {}
    getter = getattr(toolkit, "get_discoverable_tools" if discoverable else "get_tools", None)
    if callable(getter):
        tools = getter()
        return tools if isinstance(tools, dict) else {}
    if isinstance(toolkit, dict) and not discoverable:
        return toolkit
    return {}


def _tool(toolkit, name: str):
    tools = _toolkit_tools(toolkit)
    if name in tools:
        return tools[name]
    return _toolkit_tools(toolkit, discoverable=True).get(name)


def _kind(tool) -> str:
    declared = getattr(tool, "tool_type", None)
    value = getattr(declared, "value", declared)
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if text in KINDS else "generic"


def _declared_kind(toolkit, name: str) -> str:
    if toolkit is None:
        return "unknown"
    tool = _tool(toolkit, name)
    exists = tool is not None
    has_tool = getattr(toolkit, "has_tool", None)
    if callable(has_tool):
        try:
            exists = bool(has_tool(name)) or exists
        except Exception:
            pass
    if not exists and name not in _toolkit_tools(toolkit, discoverable=True):
        return "unknown"
    tool_type = getattr(toolkit, "tool_type", None)
    if callable(tool_type):
        try:
            declared = tool_type(name)
        except Exception:
            declared = None
        value = getattr(declared, "value", declared)
        text = value.strip().lower() if isinstance(value, str) else ""
        if text in KINDS:
            return text
    return "generic" if tool is not None else "unknown"


def _inner_name(call: Call) -> str | None:
    key = NAME_KEY.get(call.name)
    name = (call.arguments or {}).get(key) if key else None
    return name if isinstance(name, str) else None


def _inner_args(call: Call) -> dict:
    raw = (call.arguments or {}).get("arguments")
    parsed = _loads(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def tool_kind(env: Environment, call: Call) -> str:
    """The declared kind of the tool a call reaches, looking inside the discoverable-tool wrappers."""
    tools = getattr(env, "tools", None)
    user_tools = getattr(env, "user_tools", None)
    inner = _inner_name(call)
    if call.name in NAME_KEY:
        if inner is None:
            return "unknown"
        return _declared_kind(user_tools if call.name == USER_WRAPPER else tools, inner)
    return _declared_kind(user_tools if call.requestor == "user" else tools, call.name)


def tool_error(message) -> bool:
    """Whether a live tool message reports an error, as a flag or as an Error: line from a wrapper."""
    if getattr(message, "error", False):
        return True
    content = getattr(message, "content", None)
    return isinstance(content, str) and content.startswith("Error:")


def _text(value):
    return None if value is None else str(value)


def db_hashes(env: Environment) -> list:
    """The agent and user DB hashes, read from the environment's two documented methods."""
    agent = getattr(env, "get_db_hash", None)
    user = getattr(env, "get_user_db_hash", None)
    agent_value = agent() if callable(agent) else None
    user_value = user() if callable(user) else None
    if not callable(user) and isinstance(agent_value, (tuple, list)) and len(agent_value) == 2:
        return [_text(agent_value[0]), _text(agent_value[1])]
    return [_text(agent_value), _text(user_value)]


def _schema(tool) -> dict:
    """The tool's JSON-schema parameters, preferring the authoritative openai_schema when it has one."""
    candidates = []
    schema = getattr(tool, "openai_schema", None)
    function = schema.get("function") if isinstance(schema, dict) else None
    parameters = (function or {}).get("parameters")
    if isinstance(parameters, dict):
        candidates.append(parameters)
    for attr in ("parameters", "params"):
        value = getattr(tool, attr, None)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        if isinstance(candidate.get("properties"), dict):
            return candidate
    return {}


def _signature(tool) -> tuple[list, list]:
    func = getattr(tool, "func", None)
    if not callable(func):
        return [], []
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return [], []
    names = [
        name
        for name, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    return names, [name for name in names if params[name].default is inspect.Parameter.empty]


def _plain_parameters(tool) -> list:
    """Argument names from a bare parameter mapping that is not a JSON schema."""
    for attr in ("parameters", "params"):
        value = getattr(tool, attr, None)
        if isinstance(value, dict):
            return list(value)
    return []


def _parameters(tool) -> tuple[list, list]:
    """(argument names, required names) from the JSON schema, else the function signature, else the mapping."""
    schema = _schema(tool)
    if schema:
        names = list(schema["properties"])
        required = schema.get("required")
        return names, [name for name in required if isinstance(name, str)] if isinstance(required, list) else []
    names, required = _signature(tool)
    if names:
        return names, required
    names = _plain_parameters(tool)
    return names, names


def _argument_index(gold: list[Call]) -> dict:
    """Argument name -> (first gold call supplying it, value), wrapper JSON arguments included."""
    index = {}
    for position, call in enumerate(gold):
        supplied = dict(call.arguments or {})
        supplied.update(_inner_args(call))
        for name, value in supplied.items():
            index.setdefault(name, (position, value))
    return index


def _unlock_positions(gold: list[Call]) -> dict:
    """Inner tool name -> the first position after the gold call that unlocks it."""
    positions = {}
    for index, call in enumerate(gold):
        key = UNLOCK_KEY.get(call.name)
        name = (call.arguments or {}).get(key) if key else None
        if isinstance(name, str):
            positions.setdefault(name, index + 1)
    return positions


def _read_candidates(env: Environment, gold: list[Call]) -> list[tuple[str, str, int]]:
    """(route, tool name, earliest insert position) for tools the run can reach."""
    tools = getattr(env, "tools", None)
    direct_tools = _toolkit_tools(tools)
    discoverable = _toolkit_tools(tools, discoverable=True)
    unlocked = _unlock_positions(gold)
    candidates = []
    for name in sorted(direct_tools):
        if name in NAME_KEY or name in discoverable or name in unlocked:
            continue
        candidates.append(("direct", name, 0))
    for name, position in sorted(unlocked.items()):
        candidates.append(("agent", name, position))
    return candidates


def _tool_after(domain: str, task: Task, gold: list[Call], position: int, name: str):
    """The named tool in the toolkit as it stands after replaying the gold prefix."""
    try:
        _, env = run(domain, task, list(gold[:position]))
    except Exception:
        return None
    return _tool(getattr(env, "tools", None), name)


def _p1_plans(domain: str, task: Task, gold: list[Call]) -> list[tuple[str, str, int, dict]]:
    """(route, name, insert position, arguments) for reads whose required arguments earlier calls supply."""
    env = fresh_env(domain, task)
    tools = getattr(env, "tools", None)
    supplied = _argument_index(gold)
    plans = []
    for route, name, unlock_at in _read_candidates(env, gold):
        tool = _tool_after(domain, task, gold, unlock_at, name) if route == "agent" else _tool(tools, name)
        if tool is None or _kind(tool) != READ:
            continue
        _, required = _parameters(tool)
        if any(arg not in supplied for arg in required):
            continue
        positions = [supplied[arg][0] for arg in required]
        position = max([unlock_at] + [index + 1 for index in positions])
        if any(supplied[arg][0] >= position for arg in required):
            continue
        arguments = {arg: supplied[arg][1] for arg in required}
        plans.append((route, name, position, arguments))
    plans.sort(key=lambda plan: (plan[0], plan[1], plan[2]))
    return plans


def _insert(gold: list[Call], position: int, call: Call) -> tuple[Call, ...]:
    return tuple(gold[:position]) + (call,) + tuple(gold[position:])


def _inserted_read(route: str, name: str, arguments: dict) -> Call:
    if route == "direct":
        return Call(name, arguments)
    wrapper = AGENT_WRAPPER if route == "agent" else USER_WRAPPER
    return Call(wrapper, {NAME_KEY[wrapper]: name, "arguments": json.dumps(arguments)})


def _ran_cleanly(domain: str, task: Task, calls: tuple[Call, ...], position: int) -> bool:
    try:
        messages, _ = run(domain, task, list(calls[: position + 1]))
    except Exception:  # a call the live env cannot answer is not a candidate
        return False
    return bool(messages) and not tool_error(messages[-1])


def p1_extra_read(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    gold = gold_calls(task)
    plans = _p1_plans(domain, task, gold)
    rng.shuffle(plans)
    in_gold = {_inner_name(call) or call.name for call in gold}
    variants = []
    for route, name, position, arguments in plans:
        calls = _insert(gold, position, _inserted_read(route, name, arguments))
        if not _ran_cleanly(domain, task, calls, position):
            continue
        detail = f"extra {route} read of {name} at position {position}"
        variants.append(Variant("P1", calls, True, detail, name in in_gold))
        if len(variants) == MAX_P1_VARIANTS:
            break
    return variants


def _respell(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_respell(item) for item in value]
    if isinstance(value, dict):
        return {key: _respell(item) for key, item in value.items()}
    return value


def _respell_token(token: str) -> str:
    """A different JSON spelling of the same number: int to float, integral float to int, else add a zero."""
    if "." not in token and "e" not in token and "E" not in token:
        return token + ".0"
    try:
        value = float(token)
    except ValueError:
        return token
    if value.is_integer():
        return str(int(value))
    index = token.lower().find("e")
    mantissa, exponent = (token, "") if index == -1 else (token[:index], token[index:])
    mantissa += "0" if "." in mantissa else ".0"
    return mantissa + exponent


def _respell_json(text: str) -> str:
    """Respell every JSON number token outside strings, keeping the parsed value."""
    if _loads(text) is _MISSING:
        return text
    out = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char in "-0123456789":
            match = _JSON_NUMBER.match(text, index)
            if match:
                out.append(_respell_token(match.group(0)))
                index = match.end()
                continue
        out.append(char)
        index += 1
    respelled = "".join(out)
    return respelled if respelled != text else text


def _respell_call(call: Call) -> Call:
    if not isinstance(call.arguments, dict):
        return call
    arguments = {}
    for name, value in call.arguments.items():
        if name == "arguments" and isinstance(value, str):
            arguments[name] = _respell_json(value)
        else:
            arguments[name] = _respell(value)
    return Call(call.name, arguments, call.requestor)


def _respelled_calls(gold: list[Call]) -> tuple[Call, ...]:
    return tuple(_respell_call(call) for call in gold)


def p3_respelled_numbers(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    gold = gold_calls(task)
    calls = _respelled_calls(gold)
    if canonical_json(calls_as_dicts(calls)) == canonical_json(calls_as_dicts(gold)):
        return []
    return [Variant("P3", calls, True, "numeric arguments respelled (same value, different spelling)")]


def p5_unknown_tool(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    gold = gold_calls(task)
    position = rng.randrange(len(gold) + 1)
    calls = _insert(gold, position, Call(UNKNOWN_TOOL, {}))
    return [Variant("P5", calls, True, f"call to {UNKNOWN_TOOL} inserted at position {position}")]


def f1_empty_run(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    if not gold_calls(task):
        return []
    return [Variant("F1", (), False, "empty run")]


def _calls_of_kind(domain: str, task: Task, gold: list[Call], kind: str) -> list[int]:
    env = fresh_env(domain, task)
    return [position for position, call in enumerate(gold) if tool_kind(env, call) == kind]


def f3_dropped_write(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    gold = gold_calls(task)
    writes = _calls_of_kind(domain, task, gold, WRITE)
    if not writes:
        return []
    gold_state = db_hashes(run(domain, task, list(gold))[1])
    variants = []
    for position in writes:
        calls = tuple(gold[:position] + gold[position + 1 :])
        if db_hashes(run(domain, task, list(calls))[1]) == gold_state:
            continue
        variants.append(Variant("F3", calls, False, f"dropped write {gold[position].name} at position {position}"))
    return variants


def f6_late_verification(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    if domain != "banking_knowledge":
        return []
    gold = gold_calls(task)
    env = fresh_env(domain, task)
    kinds = [tool_kind(env, call) for call in gold]
    variants = []
    for position, call in enumerate(gold):
        if call.name != "log_verification" or WRITE not in kinds[position + 1 :]:
            continue
        calls = tuple(gold[:position] + gold[position + 1 :] + [call])
        variants.append(Variant("F6", calls, False, f"log_verification moved from position {position} to the end"))
    return variants


RELATIONS = {
    "P1": p1_extra_read,
    "P3": p3_respelled_numbers,
    "P5": p5_unknown_tool,
    "F1": f1_empty_run,
    "F3": f3_dropped_write,
    "F6": f6_late_verification,
}


def _p5_inert(domain: str, task: Task, calls: tuple[Call, ...]) -> bool:
    """True when the unknown-tool call leaves both DB hashes alone; a replay that raises is never a pass."""
    position = next(index for index, call in enumerate(calls) if call.name == UNKNOWN_TOOL)
    before = db_hashes(run(domain, task, list(calls[:position]))[1])
    after = db_hashes(run(domain, task, list(calls[: position + 1]))[1])
    return before == after


def build_variants(relation: str, domain: str, task: Task, rng: random.Random) -> tuple[list[Variant], dict]:
    """Variants for one relation with gold-identical ones dropped; notes say what was skipped and why."""
    gold = gold_calls(task)
    variants = RELATIONS[relation](domain, task, rng)
    gold_json = canonical_json(calls_as_dicts(gold))
    kept = [variant for variant in variants if canonical_json(calls_as_dicts(variant.calls)) != gold_json]
    notes = {}
    if len(kept) != len(variants):
        notes["identical_to_gold"] = True
    if relation == "F1" and not gold:
        notes["identical_to_gold"] = True
    if relation == "P3" and not variants:
        if canonical_json(calls_as_dicts(_respelled_calls(gold))) == gold_json:
            notes["identical_to_gold"] = True
    if relation == "P5" and kept and not _p5_inert(domain, task, kept[0].calls):
        notes["p5_not_inert"] = True
        kept = []
    return kept, notes


def _outputs(messages: list, count: int) -> list:
    """The tool messages of the last `count` calls in a run's message list."""
    if not count:
        return []
    start = len(messages) - 2 * count + 1
    return [messages[start + 2 * index] for index in range(count)]


def static_checks(domain: str, task: Task) -> dict:
    """S1 and S2 over the gold run, plus the writes whose removal changes no end state (F3's filter)."""
    gold = gold_calls(task)
    messages, env = run(domain, task, list(gold))
    state = db_hashes(env)
    no_effect = 0
    for position in _calls_of_kind(domain, task, gold, WRITE):
        if db_hashes(run(domain, task, gold[:position] + gold[position + 1 :])[1]) == state:
            no_effect += 1
    return {
        "gold_errors": sum(1 for message in _outputs(messages, len(gold)) if tool_error(message)),
        "gold_changes_db": state != db_hashes(fresh_env(domain, task)),
        "no_effect_writes": no_effect,
    }
