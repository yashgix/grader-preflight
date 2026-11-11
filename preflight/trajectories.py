"""Phase 3: replay and audit tau2 agent trajectories against the preflight relations.

The build never sees real agent data: `as_simulation` turns a call list into tau2's own results format, whose
answers are known. Real published files go through the same `regrade`, `flags`, `analyze` and `sample` commands
later, outside the build.
"""

import argparse
import inspect
import json
import re
from pathlib import Path

from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

from preflight import suite, tau2_adapter
from preflight.relations import (
    AGENT_WRAPPER,
    READ,
    USER_WRAPPER,
    WRITE,
    _inner_args,
    _inner_name,
    _schema,
    _toolkit_tools,
    calls_as_dicts,
    canonical_json,
    tool_error,
    tool_kind,
)
from preflight.tau2_adapter import Call, env_kwargs, fresh_env, gold_calls, run, tasks

BANK, AIRLINE = "banking_knowledge", "airline"
_CALL_WRAPPERS = frozenset({AGENT_WRAPPER, USER_WRAPPER})
_MESSAGE_CLASSES = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool": ToolMessage,
}
_FACT_KEYS = {
    "date_of_birth": ("date_of_birth", "dob", "birth_date", "birthday"),
    "email": ("email", "email_address"),
    "phone": ("phone", "phone_number", "mobile"),
    "address": ("address", "street_address", "home_address"),
}
_IDENTITY_KEYS = ("user_id", "customer_id", "email", "name", "customer_name", "full_name")
_PUNCTUATION = re.compile(r"[\W_]+")
_VERIFICATION = "log_verification"

# Replays are deterministic and read-only, so a run's live outputs and end state are computed once per process.
_RUNS: dict[str, tuple] = {}
_TOOLKITS: dict[tuple, object] = {}
_TOOL_ENVS: dict[tuple, object] = {}
_TASK_LISTS: dict[str, list] = {}

try:
    _STRICT_REPLAY = "strict_replay" in inspect.signature(EnvironmentEvaluator.calculate_reward).parameters
except (TypeError, ValueError):
    _STRICT_REPLAY = False


def _get(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _text_role(role) -> str:
    value = getattr(role, "value", role)
    return value.strip().lower() if isinstance(value, str) else ""


def _role(message) -> str:
    return _text_role(_get(message, "role"))


def _content(message):
    return _get(message, "content")


def _tool_name(call: Call) -> str:
    return _inner_name(call) or call.name


def _effective_arguments(call: Call) -> dict:
    """A call's arguments, with a wrapper's JSON arguments string unpacked into its inner tool's arguments."""
    arguments = dict(call.arguments or {})
    if call.name not in _CALL_WRAPPERS:
        return arguments
    arguments.pop("arguments", None)
    arguments.update(_inner_args(call))
    return arguments


def _parsed(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _as_ints(value):
    """Every whole-number float turned into an int, everywhere, with booleans left alone."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _as_ints(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_ints(item) for item in value]
    return value


def _normalise_text(value) -> str:
    return _PUNCTUATION.sub("", str(value).casefold())


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    return []


def _task_hash(task) -> str:
    """tau2's own task hash when the helper exists, else its JSON dump, which equal tasks share."""
    try:
        from tau2.utils.utils import get_pydantic_hash
    except ImportError:
        return task.model_dump_json()
    return get_pydantic_hash(task)


def task_matches(task_dict, task) -> bool:
    if not isinstance(task_dict, dict) or task is None:
        return False
    try:
        return _task_hash(Task.model_validate(task_dict)) == _task_hash(task)
    except Exception:
        return False


def restore_ids(messages) -> list:
    """Give every tool message without an id the id of the earliest tool call it answers."""
    pending: list[str] = []
    restored = []
    for message in messages or []:
        copy = dict(message)
        if _text_role(copy.get("role")) == "tool":
            message_id = copy.get("id")
            if not message_id:
                copy["id"] = pending.pop(0) if pending else None
            elif message_id in pending:
                pending.remove(message_id)
            restored.append(copy)
            continue
        for call in copy.get("tool_calls") or []:
            call_id = _get(call, "id")
            if call_id:
                pending.append(call_id)
        restored.append(copy)
    return restored


def calls(messages) -> list[Call]:
    """Every tool call of a message list in order, with the call's own requestor or else the message role."""
    found = []
    for message in messages or []:
        role = _role(message)
        for raw in _get(message, "tool_calls") or []:
            name = _get(raw, "name")
            if not isinstance(name, str):
                continue
            arguments = _get(raw, "arguments") or {}
            requestor = _text_role(_get(raw, "requestor")) or role or "assistant"
            found.append(Call(name, dict(arguments) if isinstance(arguments, dict) else {}, requestor))
    return found


def _as_message(raw):
    if not isinstance(raw, dict):
        return raw
    message_class = _MESSAGE_CLASSES.get(_text_role(raw.get("role")))
    if message_class is None:
        return None
    return message_class.model_validate(raw)


def _messages(raw_messages) -> list:
    converted = [_as_message(raw) for raw in raw_messages or []]
    return [message for message in converted if message is not None]


def _tool_outputs(messages, count: int) -> list:
    if count <= 0:
        return []
    outputs = [message for message in messages or [] if _role(message) == "tool"]
    return outputs[-count:]


def _replay(domain, task, run_calls):
    key = canonical_json([domain, getattr(task, "id", None), calls_as_dicts(run_calls)])
    if key not in _RUNS:
        _RUNS[key] = run(domain, task, list(run_calls))
    return _RUNS[key]


def _live_outputs(domain, task, run_calls) -> list:
    messages, _ = _replay(domain, task, run_calls)
    return _tool_outputs(messages, len(run_calls))


def _grade_messages(domain, task, messages, strict=None):
    """Score a message trajectory with tau2's own, unchanged grader; `strict` only where the version has it."""
    constructor = registry.get_env_constructor(domain)
    arguments = {"env_kwargs": env_kwargs(domain, task)}
    if strict is not None:
        arguments["strict_replay"] = strict
    return EnvironmentEvaluator.calculate_reward(constructor, task, messages, **arguments).reward


def _grade_run(domain, task, run_calls):
    messages, _ = _replay(domain, task, run_calls)
    return _grade_messages(domain, task, messages)


def _grade_recorded(domain, task, run_messages):
    """(mode, reward) for recorded messages: strict, else lenient where v1.0.1 has the flag, else incompatible."""
    converted = _messages(run_messages)
    if not _STRICT_REPLAY:
        try:
            return "strict", _grade_messages(domain, task, converted)
        except Exception:
            return "incompatible", None
    try:
        return "strict", _grade_messages(domain, task, converted, strict=True)
    except Exception:
        pass
    try:
        return "lenient", _grade_messages(domain, task, converted, strict=False)
    except Exception:
        return "incompatible", None


def as_simulation(domain, task, calls, trial=0, trimmed=False) -> dict:
    """Run the calls on a fresh env and return one simulation dict in tau2's results format."""
    live, _ = run(domain, task, list(calls))
    outputs = _tool_outputs(live, len(calls))
    messages = []
    for index, call in enumerate(calls):
        tool_call = {
            "id": f"call_{index}",
            "name": call.name,
            "arguments": call.arguments,
            "requestor": call.requestor,
        }
        messages.append(
            {
                "role": "user" if call.requestor == "user" else "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        )
        output = outputs[index] if index < len(outputs) else None
        tool_message = {
            "role": "tool",
            "id": getattr(output, "id", None) or tool_call["id"],
            "content": _content(output),
            "error": bool(getattr(output, "error", False)),
            "requestor": getattr(output, "requestor", None) or call.requestor,
        }
        if trimmed:
            tool_message.pop("id")
            tool_message.pop("error")
        messages.append(tool_message)
    reward = _grade_messages(domain, task, live)
    return {
        "id": f"{domain}:{task.id}:{trial}",
        "task_id": task.id,
        "trial": trial,
        "messages": messages,
        "reward_info": {"reward": reward, "db_check": {"db_reward": reward}},
    }


def load(path) -> dict:
    """Read a results file into its domain, git commit, embedded tasks and runs (tool-message ids restored)."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    info = data.get("info") or {}
    domain = (info.get("environment_info") or {}).get("domain_name")
    embedded = {}
    for task in data.get("tasks") or []:
        if isinstance(task, dict) and task.get("id") is not None:
            embedded[str(task["id"])] = task
    runs = []
    for simulation in data.get("simulations") or []:
        if not isinstance(simulation, dict):
            continue
        reward_info = simulation.get("reward_info") or {}
        runs.append(
            {
                "id": simulation.get("id"),
                "task_id": simulation.get("task_id"),
                "trial": simulation.get("trial"),
                "messages": restore_ids(simulation.get("messages") or []),
                "recorded_reward": reward_info.get("reward"),
            }
        )
    return {"domain": domain, "git_commit": info.get("git_commit"), "tasks": embedded, "runs": runs}


def _recorded_outputs(messages) -> list:
    return [_content(message) for message in messages or [] if _role(message) == "tool"]


def replay(domain, task, run) -> dict:
    """Replay a run's calls live, compare its outputs, and regrade the recorded messages."""
    run_messages = run.get("messages") or []
    run_calls = calls(run_messages)
    recorded = _recorded_outputs(run_messages)
    replayed = [_content(message) for message in _live_outputs(domain, task, run_calls)]
    mismatched = []
    for index, call in enumerate(run_calls):
        if _tool_name(call) == "KB_search":
            continue
        expected = recorded[index] if index < len(recorded) else None
        actual = replayed[index] if index < len(replayed) else None
        if _parsed(expected) != _parsed(actual):
            mismatched.append({"index": index, "name": _tool_name(call)})
    mode, reward = _grade_recorded(domain, task, run_messages)
    outputs_equal = not mismatched
    return {
        "outputs_equal": outputs_equal,
        "mismatched": mismatched,
        "mode": mode,
        "reward": reward,
        "replay_class": "full" if outputs_equal and mode == "strict" else mode,
    }


def diff_tables(domain, task, run_calls) -> list[str]:
    """The DB tables where the run's end state differs from the gold run's."""
    _, variant = _replay(domain, task, list(run_calls))
    _, gold = _replay(domain, task, gold_calls(task))
    return sorted(suite._db_diff(variant, gold))


def _gold_tool_names(task) -> set[str]:
    return {_tool_name(call) for call in gold_calls(task)}


def p1_trigger(domain, task, run_calls) -> bool:
    """A discoverable READ the gold run never makes, whose live call reported no error."""
    env = _tool_env(domain, task)
    gold_names = _gold_tool_names(task)
    candidates = []
    for index, call in enumerate(run_calls):
        if call.name != AGENT_WRAPPER:
            continue
        inner = _inner_name(call)
        if not inner or inner in gold_names or tool_kind(env, call) != READ:
            continue
        candidates.append(index)
    if not candidates:
        return False
    outputs = _live_outputs(domain, task, run_calls)
    return any(index < len(outputs) and not tool_error(outputs[index]) for index in candidates)


def _spelling_differs(left, right) -> bool:
    """Same value, different JSON spelling: an int against a whole-number float, nested values included."""
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right and isinstance(left, float) != isinstance(right, float)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and any(_spelling_differs(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return all(key in right for key in left) and any(_spelling_differs(left[key], right[key]) for key in left)
    return False


def p3_trigger(task, run_calls) -> bool:
    """A run argument with the same tool and value as a gold argument but a different JSON spelling."""
    gold: dict[str, list[dict]] = {}
    for call in gold_calls(task):
        gold.setdefault(_tool_name(call), []).append(_effective_arguments(call))
    for call in run_calls:
        arguments = _effective_arguments(call)
        for gold_arguments in gold.get(_tool_name(call), []):
            for name, value in arguments.items():
                if name in gold_arguments and _spelling_differs(value, gold_arguments[name]):
                    return True
    return False


def _domain_tasks(domain) -> list:
    if domain not in _TASK_LISTS:
        _TASK_LISTS[domain] = tasks(domain)
    return _TASK_LISTS[domain]


def _tool_env(domain, task):
    key = (domain, getattr(task, "id", None))
    if key not in _TOOL_ENVS:
        _TOOL_ENVS[key] = fresh_env(domain, task)
    return _TOOL_ENVS[key]


def _toolkit(domain, requestor: str):
    key = (domain, requestor)
    if key not in _TOOLKITS:
        try:
            env = registry.get_env_constructor(domain)(**tau2_adapter.ENV_KWARGS.get(domain, {}))
        except Exception:
            known = _domain_tasks(domain)
            env = fresh_env(domain, known[0]) if known else None
        _TOOLKITS[key] = getattr(env, "user_tools" if requestor == "user" else "tools", None) if env else None
    return _TOOLKITS[key]


def _tool_for(domain, call: Call, tool_name: str):
    if call.name == USER_WRAPPER:
        return _toolkit_tools(_toolkit(domain, "user"), discoverable=True).get(tool_name)
    if call.name == AGENT_WRAPPER:
        return _toolkit_tools(_toolkit(domain, "assistant"), discoverable=True).get(tool_name)
    requestor = "user" if call.requestor == "user" else "assistant"
    return _toolkit_tools(_toolkit(domain, requestor)).get(tool_name)


def _integer_parameter(declared) -> bool:
    if not isinstance(declared, dict):
        return False
    kind = declared.get("type")
    if kind == "integer":
        return True
    return isinstance(kind, list) and "integer" in kind


def schema_floats(domain, run_calls) -> list[dict]:
    """Every whole-number float passed to a parameter tau2's own schema declares an integer."""
    found = []
    for index, call in enumerate(run_calls):
        tool_name = _tool_name(call)
        tool = _tool_for(domain, call, tool_name)
        if tool is None:
            continue
        properties = _schema(tool).get("properties") or {}
        for argument, value in _effective_arguments(call).items():
            if isinstance(value, float) and value.is_integer() and _integer_parameter(properties.get(argument)):
                found.append({"index": index, "tool": tool_name, "argument": argument})
    return found


def as_integers(run_calls, found) -> list[Call]:
    """The same calls with exactly the given whole-number floats made int, wrapper argument strings included."""
    result = [Call(call.name, dict(call.arguments or {}), call.requestor) for call in run_calls]
    for item in found:
        index, name = item["index"], item["argument"]
        if index >= len(result):
            continue
        call = result[index]
        arguments = dict(call.arguments or {})
        if call.name in _CALL_WRAPPERS and name not in arguments:
            inner = dict(_inner_args(call))
            if name not in inner:
                continue
            inner[name] = int(inner[name])
            arguments["arguments"] = json.dumps(inner, sort_keys=True)
        elif name in arguments:
            arguments[name] = int(arguments[name])
        else:
            continue
        result[index] = Call(call.name, arguments, call.requestor)
    return result


def _normalised_output(message) -> str:
    return canonical_json(_as_ints(_parsed(_content(message))))


def _state_dump(domain, task, run_calls) -> str:
    _, env = _replay(domain, task, run_calls)
    agent, user = suite._db_pair(env)
    try:
        return canonical_json(_as_ints({"agent": agent, "user": user}))
    except (TypeError, ValueError):
        return repr((agent, user))


def numeric_pair(domain, task, run_calls) -> dict:
    """A run and its integer respelling: same outputs, same state, different reward is an inconsistency."""
    found = schema_floats(domain, run_calls)
    integer_calls = as_integers(run_calls, found)
    float_outputs = _live_outputs(domain, task, run_calls)
    int_outputs = _live_outputs(domain, task, integer_calls)
    positions = {item["index"] for item in found}
    errors_at_found = any(
        index < len(outputs) and tool_error(outputs[index])
        for outputs in (float_outputs, int_outputs)
        for index in positions
    )
    outputs_equal = [_normalised_output(item) for item in float_outputs] == [
        _normalised_output(item) for item in int_outputs
    ]
    states_equal = _state_dump(domain, task, run_calls) == _state_dump(domain, task, integer_calls)
    reward_float = _grade_run(domain, task, run_calls)
    reward_int = _grade_run(domain, task, integer_calls)
    inconsistency = bool(
        not errors_at_found and outputs_equal and states_equal and reward_float != reward_int
    )
    return {
        "found": found,
        "errors_at_found": errors_at_found,
        "outputs_equal": outputs_equal,
        "states_equal": states_equal,
        "reward_float": reward_float,
        "reward_int": reward_int,
        "inconsistency": inconsistency,
    }


def _gold_verification(gold):
    for call in gold:
        if call.name == _VERIFICATION:
            return dict(call.arguments or {})
        if _inner_name(call) == _VERIFICATION:
            return _effective_arguments(call)
    return None


def _run_domain(run):
    domain = run.get("domain")
    if isinstance(domain, str) and domain in tau2_adapter.DOMAINS:
        return domain
    prefix = str(run.get("id") or "").split(":", 1)[0]
    return prefix if prefix in tau2_adapter.DOMAINS else None


def _identity_values(arguments) -> set[str]:
    values = set()
    for key in _IDENTITY_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            values.add(_normalise_text(value))
    return values


def _fact_values(arguments) -> dict[str, str]:
    facts = {}
    for fact, keys in _FACT_KEYS.items():
        for key in keys:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                facts[fact] = _normalise_text(value)
                break
    return facts


def _names_user(tool_calls, identity: set[str]) -> bool:
    if not identity:
        return False
    for call in tool_calls:
        for text in _strings(_get(call, "arguments") or {}):
            normalised = _normalise_text(text)
            if any(value and value in normalised for value in identity):
                return True
    return False


def _access_before_facts(messages, verification) -> bool | None:
    identity = _identity_values(verification)
    facts = _fact_values(verification)
    seen: set[str] = set()
    first_access = False
    facts_then = 0
    user_text = False
    for message in messages or []:
        role = _role(message)
        if role == "user":
            text = _content(message)
            if isinstance(text, str) and text.strip():
                user_text = True
                normalised = _normalise_text(text)
                for name, value in facts.items():
                    if value and value in normalised:
                        seen.add(name)
        elif role == "assistant" and not first_access:
            if _names_user(_get(message, "tool_calls") or [], identity):
                first_access = True
                facts_then = len(seen)
    if not user_text:
        return None
    if not first_access:
        return False
    return facts_then < 2


def late_verification(task, run) -> dict:
    """Banking: did the verification log come after the first write, and did the agent touch the user early?"""
    messages = run.get("messages") or []
    verification = _gold_verification(gold_calls(task))
    if verification is None:
        return {"log_after_first_write": False, "access_before_facts": False}
    domain = _run_domain(run)
    first_write = first_log = None
    if domain is not None:
        env = _tool_env(domain, task)
        for index, call in enumerate(calls(messages)):
            if call.requestor != "assistant":
                continue
            if _tool_name(call) == _VERIFICATION:
                if first_log is None:
                    first_log = index
            elif first_write is None and tool_kind(env, call) == WRITE:
                first_write = index
    log_after_first_write = first_write is not None and (first_log is None or first_log > first_write)
    return {
        "log_after_first_write": log_after_first_write,
        "access_before_facts": _access_before_facts(messages, verification),
    }


def _call_key(call: Call) -> str:
    return canonical_json([_tool_name(call), _as_ints(_effective_arguments(call))])


def _nongold_call(task, run_calls) -> bool:
    gold = {_call_key(call) for call in gold_calls(task)}
    return any(_call_key(call) not in gold for call in run_calls)


def _computed(domain, task, run) -> dict:
    run_calls = calls(run.get("messages") or [])
    graded = replay(domain, task, run)
    computed = {
        "replay_class": graded["replay_class"],
        "reward": graded["reward"],
        "mismatched": graded["mismatched"],
        "n_calls": len(run_calls),
        "nongold_call": _nongold_call(task, run_calls),
        "p1": p1_trigger(domain, task, run_calls),
        "p3": p3_trigger(task, run_calls),
        "diff_tables": diff_tables(domain, task, run_calls),
    }
    if domain == AIRLINE:
        computed["schema_floats"] = schema_floats(domain, run_calls)
        computed["numeric_pair"] = numeric_pair(domain, task, run_calls)
    if domain == BANK:
        computed["late_verification"] = late_verification(task, {**run, "domain": domain})
    return computed


def _record(domain, git_commit, run, task, task_dict) -> dict:
    record = {
        "id": run.get("id"),
        "task_id": run.get("task_id"),
        "trial": run.get("trial"),
        "domain": domain,
        "git_commit": git_commit,
        "task_matches": False,
        "replay_class": None,
        "reward": None,
        "recorded_reward": run.get("recorded_reward"),
        "mismatched": None,
        "p1": None,
        "p3": None,
        "diff_tables": None,
        "n_calls": None,
        "nongold_call": None,
        "schema_floats": None,
        "numeric_pair": None,
        "late_verification": None,
    }
    if task is None or not task_matches(task_dict, task):
        return record
    record["task_matches"] = True
    try:
        record.update(_computed(domain, task, run))
    except Exception as exc:
        lines = str(exc).splitlines()
        record["error"] = {"type": type(exc).__name__, "message": lines[0][:200] if lines else ""}
    return record


def version_label() -> str:
    tag = suite._tau2_version(suite._source_text())
    if tag == "pre-v1.0.1":
        return "old"
    if tag == "v1.0.1":
        return "new"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", tag) if tag else "unknown"


def regrade(paths, out_dir) -> list[Path]:
    """One JSONL record per run and file, labelled with this env's tau2 version."""
    label = version_label()
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in paths:
        result = load(path)
        domain = result.get("domain")
        local = {task.id: task for task in _domain_tasks(domain)} if domain else {}
        records = []
        for run in result.get("runs") or []:
            task_id = run.get("task_id")
            task = local.get(str(task_id)) if task_id is not None else None
            records.append(_record(domain, result.get("git_commit"), run, task, result["tasks"].get(task_id)))
        target = target_dir / f"{Path(path).stem}.{label}.jsonl"
        target.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        written.append(target)
    return written


def flags(paths, out_path) -> dict:
    """The phase 1 G_old tasks each P relation flagged, P1 only where the read is not in the gold run."""
    flagged = {"P1": set(), "P3": set()}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        for variant in data.get("variants") or []:
            relation = variant.get("relation")
            if relation not in flagged or variant.get("flagged") is not True:
                continue
            if relation == "P1" and variant.get("inserted_in_gold") is True:
                continue
            if variant.get("task_id") is not None:
                flagged[relation].add(str(variant["task_id"]))
    payload = {relation: sorted(flagged[relation]) for relation in ("P1", "P3")}
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _regrade_command(args) -> int:
    regrade(args.files, args.out)
    return 0


def _flags_command(args) -> int:
    flags(args.files, args.out)
    return 0


def _analyze_command(args) -> int:
    from preflight import trajectory_analysis

    trajectory_analysis.analyze(args.old, args.new, args.flags, args.pool, args.out)
    return 0


def _sample_command(args) -> int:
    from preflight import trajectory_analysis

    trajectory_analysis.sample(args.old, args.new, args.flags, args.pool, args.files, args.out)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m preflight.trajectories")
    commands = parser.add_subparsers(dest="command", required=True)

    regrade_parser = commands.add_parser("regrade", help="regrade results files in this env, one JSONL per file")
    regrade_parser.add_argument("files", nargs="+")
    regrade_parser.add_argument("--out", required=True)
    regrade_parser.set_defaults(handler=_regrade_command)

    flags_parser = commands.add_parser("flags", help="the tasks phase 1's G_old records flagged")
    flags_parser.add_argument("files", nargs="+")
    flags_parser.add_argument("--out", required=True)
    flags_parser.set_defaults(handler=_flags_command)

    analyze_parser = commands.add_parser("analyze", help="join two regrade directories and write H5 and H7")
    analyze_parser.add_argument("--old", required=True)
    analyze_parser.add_argument("--new", required=True)
    analyze_parser.add_argument("--flags", required=True)
    analyze_parser.add_argument("--pool", required=True)
    analyze_parser.add_argument("--out", required=True)
    analyze_parser.set_defaults(handler=_analyze_command)

    sample_parser = commands.add_parser("sample", help="H6's stratified, blinded packets")
    sample_parser.add_argument("--old", required=True)
    sample_parser.add_argument("--new", required=True)
    sample_parser.add_argument("--flags", required=True)
    sample_parser.add_argument("--pool", required=True)
    sample_parser.add_argument("--files", nargs="+", required=True)
    sample_parser.add_argument("--out", required=True)
    sample_parser.set_defaults(handler=_sample_command)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
