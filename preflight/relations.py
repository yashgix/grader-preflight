"""Relations: small edits to a gold run whose expected effect on the verdict is known in advance."""

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
GIVE_USER_TOOL = "give_discoverable_user_tool"
NAME_KEY = {AGENT_WRAPPER: "agent_tool_name", USER_WRAPPER: "discoverable_tool_name"}
UNLOCK_AGENT = "unlock_discoverable_agent_tool"
UNLOCK_USER = "unlock_discoverable_user_tool"
UNLOCK_KEY = {UNLOCK_AGENT: "agent_tool_name", UNLOCK_USER: "discoverable_tool_name"}
WRAPPER_ARGUMENTS = frozenset({AGENT_WRAPPER, USER_WRAPPER, GIVE_USER_TOOL})
READ = "read"
WRITE = "write"
KINDS = {"read", "write", "generic", "think"}
MAX_P1_VARIANTS = 5
UNKNOWN_TOOL = "no_such_tool"
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[eE][-+]?[0-9]+)?")
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


def _named_tools(tools) -> dict:
    found = {}
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name:
            found[name] = tool
    return found


def _toolkit_tools(toolkit, discoverable: bool = False) -> dict:
    if toolkit is None:
        return {}
    getter = getattr(toolkit, "get_discoverable_tools" if discoverable else "get_tools", None)
    if callable(getter):
        try:
            tools = getter()
        except Exception:
            tools = None
        if isinstance(tools, dict):
            return tools
        if isinstance(tools, (list, tuple)):
            return _named_tools(tools)
    if discoverable:
        return {}
    if isinstance(toolkit, dict):
        return toolkit
    if isinstance(toolkit, (list, tuple)):
        return _named_tools(toolkit)
    return {}


def _kind(tool) -> str:
    declared = getattr(tool, "tool_type", None)
    if callable(declared):
        try:
            declared = declared()
        except Exception:
            declared = None
    value = getattr(declared, "value", declared)
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if text in KINDS else "generic"


def _declared_kind(toolkit, name: str) -> str:
    if toolkit is None or not isinstance(name, str) or not name:
        return "unknown"
    direct = _toolkit_tools(toolkit)
    discoverable = _toolkit_tools(toolkit, discoverable=True)
    exists = name in direct or name in discoverable
    has_tool = getattr(toolkit, "has_tool", None)
    if callable(has_tool):
        try:
            exists = bool(has_tool(name)) or exists
        except Exception:
            pass
    if not exists:
        return "unknown"
    declared = getattr(toolkit, "tool_type", None)
    if callable(declared):
        try:
            declared = declared(name)
        except Exception:
            declared = None
    elif isinstance(declared, dict):
        declared = declared.get(name)
    else:
        declared = None
    value = getattr(declared, "value", declared)
    text = value.strip().lower() if isinstance(value, str) else ""
    if text in KINDS:
        return text
    tool = direct.get(name) or discoverable.get(name)
    return _kind(tool) if tool is not None else "generic"


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
    if callable(schema):
        try:
            schema = schema()
        except Exception:
            schema = None
    function = schema.get("function") if isinstance(schema, dict) else None
    parameters = function.get("parameters") if isinstance(function, dict) else None
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
    target = func if callable(func) else tool if callable(tool) else None
    if target is None:
        return [], []
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return [], []
    names = [
        name
        for name, param in params.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
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
    """(argument names, required names) from the JSON schema, else the callable signature, else the mapping."""
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


def _plan_for(route: str, name: str, tool, supplied: dict):
    _, required = _parameters(tool)
    if any(arg not in supplied for arg in required):
        return None
    position = max([supplied[arg][0] + 1 for arg in required], default=0)
    arguments = {arg: supplied[arg][1] for arg in required}
    return route, name, position, arguments


def _p1_plans(domain: str, task: Task, gold: list[Call]) -> list[tuple[str, str, int, dict]]:
    """(route, name, insert position, arguments) for reads whose required arguments earlier calls supply."""
    env = fresh_env(domain, task)
    tools = getattr(env, "tools", None)
    direct_tools = _toolkit_tools(tools)
    discoverable_tools = _toolkit_tools(tools, discoverable=True)
    supplied = _argument_index(gold)
    plans = []
    for name, tool in direct_tools.items():
        if name in NAME_KEY or name in UNLOCK_KEY or name in discoverable_tools:
            continue
        # A tool's declared kind lives on the toolkit, not necessarily on the Tool object it hands out.
        if _declared_kind(tools, name) != READ:
            continue
        plan = _plan_for("direct", name, tool, supplied)
        if plan:
            plans.append(plan)
    for name, tool in discoverable_tools.items():
        if name in NAME_KEY or name in UNLOCK_KEY:
            continue
        if _declared_kind(tools, name) != READ:
            continue
        plan = _plan_for("agent", name, tool, supplied)
        if plan:
            plans.append(plan)
    plans.sort(key=lambda plan: (plan[0], plan[1], plan[2]))
    return plans


def _insert(gold: list[Call], position: int, call: Call) -> tuple[Call, ...]:
    return tuple(gold[:position]) + (call,) + tuple(gold[position:])


def _inserted_read(route: str, name: str, arguments: dict) -> Call:
    if route == "direct":
        return Call(name, arguments)
    wrapper = AGENT_WRAPPER if route == "agent" else USER_WRAPPER
    return Call(wrapper, {NAME_KEY[wrapper]: name, "arguments": json.dumps(arguments, sort_keys=True)})


def _unlocked_before(gold: list[Call], name: str, position: int) -> bool:
    for call in gold[:position]:
        if call.name == UNLOCK_AGENT and (call.arguments or {}).get("agent_tool_name") == name:
            return True
    return False


def _insert_with_unlock(gold: list[Call], route: str, name: str, position: int, arguments: dict):
    """The read in place, preceded by its unlock when the gold run has not unlocked it before."""
    read = _inserted_read(route, name, arguments)
    if route == "agent" and not _unlocked_before(gold, name, position):
        unlock = Call(UNLOCK_AGENT, {"agent_tool_name": name})
        return tuple(gold[:position]) + (unlock, read) + tuple(gold[position:]), position + 1
    return _insert(gold, position, read), position


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
        calls, read_index = _insert_with_unlock(gold, route, name, position, arguments)
        if not _ran_cleanly(domain, task, calls, read_index):
            continue
        detail = f"extra {route} read of {name} at position {read_index}"
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
    return "".join(out)


def _respell_call(call: Call) -> Call:
    if not isinstance(call.arguments, dict):
        return call
    arguments = {}
    for name, value in call.arguments.items():
        if name == "arguments" and call.name in WRAPPER_ARGUMENTS and isinstance(value, str):
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


def _p5_inert(domain: str, task: Task, calls: tuple[Call, ...]) -> bool:
    """True when the unknown-tool call leaves both DB hashes alone; a replay that raises is never a pass."""
    position = next(index for index, call in enumerate(calls) if call.name == UNKNOWN_TOOL)
    before = db_hashes(run(domain, task, list(calls[:position]))[1])
    after = db_hashes(run(domain, task, list(calls[: position + 1]))[1])
    return before == after


def p5_unknown_tool(domain: str, task: Task, rng: random.Random) -> list[Variant]:
    """The unknown call, kept only when the live DB hashes around it are unchanged."""
    gold = gold_calls(task)
    position = rng.randrange(len(gold) + 1)
    calls = _insert(gold, position, Call(UNKNOWN_TOOL, {}))
    if not _p5_inert(domain, task, calls):
        return []
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
    if relation == "P5" and not variants:
        # P5 builds one candidate; the generator drops it only when the unknown call changed the live DB.
        notes["p5_not_inert"] = True
    return kept, notes


def _outputs(messages: list, count: int) -> list:
    """The tool messages of the last `count` calls in a run's message list."""
    if count <= 0:
        return []
    start = len(messages) - 2 * count + 1
    if start < 0:
        return []
    return [messages[start + 2 * index] for index in range(count) if start + 2 * index < len(messages)]


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
