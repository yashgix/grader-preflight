"""Open-loop MDP over banking tool calls for the phase 2 RL preflight."""

import json
from dataclasses import dataclass

from preflight.relations import (
    AGENT_WRAPPER,
    UNLOCK_AGENT,
    UNLOCK_USER,
    USER_WRAPPER,
    WRAPPER_ARGUMENTS,
    WRITE,
    _inner_args,
    _inner_name,
    _inserted_read,
    _p1_plans,
    calls_as_dicts,
    canonical_json,
    tool_kind,
)
from preflight.tau2_adapter import Call, fresh_env, gold_calls, tasks

_WRITE_KEYS: set[str] = set()
_GATED_WRITES: dict[object, list[tuple[Call, list[Call]]]] = {}
_TASK_DOMAINS: dict[object, str] = {}
_UNLOCK_NAMES = {UNLOCK_AGENT, UNLOCK_USER}
_UNLOCK_KEY = {UNLOCK_AGENT: "agent_tool_name", UNLOCK_USER: "discoverable_tool_name"}
_READ_WRAPPERS = {AGENT_WRAPPER, USER_WRAPPER}


@dataclass(frozen=True)
class Action:
    kind: str
    calls: tuple[Call, ...]
    gold_index: int | None = None


def _calls_key(calls) -> str:
    return canonical_json(calls_as_dicts(calls))


def _action_key(action: Action) -> str:
    return _calls_key(action.calls)


def _call_key(call: Call) -> str:
    return canonical_json(calls_as_dicts([call])[0])


def _effective_call_keys(call: Call) -> set[str]:
    """The call itself plus, for a wrapper, the inner tool call it reaches."""
    keys = {_call_key(call)}
    if call.name in WRAPPER_ARGUMENTS:
        inner = _inner_name(call)
        if inner:
            keys.add(_call_key(Call(inner, _inner_args(call), call.requestor)))
    return keys


def _domain_for(task) -> str:
    task_id = getattr(task, "id", None)
    if task_id in _TASK_DOMAINS:
        return _TASK_DOMAINS[task_id]
    declared = getattr(task, "domain", None)
    return declared if isinstance(declared, str) and declared else "banking_knowledge"


def _gold_unlocks(gold: list[Call], name: str, unlock_name: str) -> bool:
    key = _UNLOCK_KEY[unlock_name]
    return any(call.name == unlock_name and (call.arguments or {}).get(key) == name for call in gold)


def _read_actions(domain: str, task, gold: list[Call]) -> list[Action]:
    """Phase 1's P1 read candidates, with an unlock only when the gold run never unlocks that tool."""
    try:
        plans = _p1_plans(domain, task, gold)
    except Exception:
        return []
    actions = []
    seen = set()
    for route, name, position, arguments in plans:
        read = _inserted_read(route, name, arguments)
        if route == "agent" and not _gold_unlocks(gold, name, UNLOCK_AGENT):
            calls = (Call(UNLOCK_AGENT, {"agent_tool_name": name}), read)
        elif route == "user" and not _gold_unlocks(gold, name, UNLOCK_USER):
            calls = (Call(UNLOCK_USER, {"discoverable_tool_name": name}), read)
        else:
            calls = (read,)
        key = _calls_key(calls)
        if key in seen:
            continue
        seen.add(key)
        actions.append(Action("read", calls))
    return actions


def _values_by_name(gold: list[Call]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for call in gold:
        arguments = call.arguments or {}
        for name, value in arguments.items():
            if isinstance(value, str):
                values.setdefault(name, set()).add(value)
        if call.name in WRAPPER_ARGUMENTS:
            for name, value in _inner_args(call).items():
                if isinstance(value, str):
                    values.setdefault(name, set()).add(value)
    return values


def _alternative_value(domain, task, task_values, name: str, current: str) -> str | None:
    values = set(task_values.get(name, set()))
    values.discard(current)
    if values:
        return sorted(values)[0]
    for other in tasks(domain):
        if getattr(other, "id", None) == getattr(task, "id", None):
            continue
        other_values = _values_by_name(gold_calls(other)).get(name, set()) - {current}
        if other_values:
            return sorted(other_values)[0]
    return None


def _id_candidates(call: Call) -> list[tuple[str, str, str]]:
    """String `_id` arguments of a call: (name, current value, outer or inner)."""
    candidates = []
    arguments = call.arguments or {}
    for name, value in arguments.items():
        if name.endswith("_id") and isinstance(value, str):
            candidates.append((name, value, "outer"))
    if call.name in WRAPPER_ARGUMENTS:
        for name, value in _inner_args(call).items():
            if name.endswith("_id") and isinstance(value, str):
                candidates.append((name, value, "inner"))
    candidates.sort(key=lambda item: (item[0], item[2]))
    return candidates


def _replace_inner_arguments(text: str, name: str, new_value: str) -> str | None:
    """Replace one string value inside a JSON arguments text, leaving every other character alone."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get(name), str):
        return None
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        try:
            token, consumed = decoder.raw_decode(text[index:])
        except ValueError:
            return None
        token_end = index + consumed
        if token == name:
            cursor = token_end
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ":":
                cursor += 1
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor < len(text) and text[cursor] == '"':
                    try:
                        _, value_end = decoder.raw_decode(text[cursor:])
                    except ValueError:
                        return None
                    replacement = json.dumps(new_value, ensure_ascii=True)
                    return text[:cursor] + replacement + text[cursor + value_end :]
        index = token_end
    return None


def _altered_call(call: Call, name: str, alternative: str, location: str) -> Call:
    arguments = dict(call.arguments or {})
    if location == "outer":
        arguments[name] = alternative
        return Call(call.name, arguments, call.requestor)
    inner = dict(_inner_args(call))
    inner[name] = alternative
    text = _replace_inner_arguments(str(arguments.get("arguments", "")), name, alternative)
    arguments["arguments"] = text if text is not None else json.dumps(inner, sort_keys=True)
    return Call(call.name, arguments, call.requestor)


def _altered_actions(domain: str, task, gold: list[Call], kinds: list[str]) -> list[Action]:
    """At most two altered copies of each gold write, each swapping one string `_id` argument."""
    task_values = _values_by_name(gold)
    actions = []
    for index, call in enumerate(gold):
        if kinds[index] != WRITE:
            continue
        made = 0
        for name, current, location in _id_candidates(call):
            if made >= 2:
                break
            alternative = _alternative_value(domain, task, task_values, name, current)
            if alternative is None:
                continue
            altered = _altered_call(call, name, alternative, location)
            actions.append(Action("altered", (altered,), index))
            made += 1
    return actions


def _gated_writes(gold: list[Call], kinds: list[str]) -> list[tuple[Call, list[Call]]]:
    gated = []
    for position, call in enumerate(gold):
        if call.name != "log_verification":
            continue
        writes = [gold[later] for later in range(position + 1, len(gold)) if kinds[later] == WRITE]
        gated.append((call, writes))
    return gated


def task_actions(domain: str, task) -> list[Action]:
    """Gold actions first, then phase 1 read candidates, then altered writes, STOP last."""
    task_id = getattr(task, "id", None)
    _TASK_DOMAINS[task_id] = domain
    gold = gold_calls(task)
    if not gold:
        _GATED_WRITES[task_id] = []
        return [Action("stop", ())]
    env = fresh_env(domain, task)
    kinds = [tool_kind(env, call) for call in gold]
    _GATED_WRITES[task_id] = _gated_writes(gold, kinds)
    actions = []
    for index, call in enumerate(gold):
        action = Action("gold", (call,), index)
        if kinds[index] == WRITE:
            _WRITE_KEYS.add(_action_key(action))
        actions.append(action)
    actions.extend(_read_actions(domain, task, gold))
    for action in _altered_actions(domain, task, gold, kinds):
        _WRITE_KEYS.add(_action_key(action))
        actions.append(action)
    actions.append(Action("stop", ()))
    return actions


def horizon(task) -> int:
    return len(gold_calls(task)) + 3


def _needs_unlock(action: Action, actions: list[Action]) -> list[int] | None:
    """Gold unlock indexes a one-call wrapper read waits for; None when it needs no unlock."""
    if action.kind != "read" or len(action.calls) != 1:
        return None
    call = action.calls[0]
    if call.name not in _READ_WRAPPERS:
        return None
    name = _inner_name(call)
    if not name:
        return None
    indices = []
    for index, candidate in enumerate(actions):
        if candidate.kind != "gold":
            continue
        for gold_call in candidate.calls:
            if gold_call.name in _UNLOCK_NAMES and (gold_call.arguments or {}).get(_UNLOCK_KEY[gold_call.name]) == name:
                indices.append(index)
    return indices or None


def allowed(actions: list[Action], made: list[int]) -> list[int]:
    """Every action at most once per episode except STOP; wrapper reads wait for gold unlocks."""
    made_set = set(made)
    result = []
    for index, action in enumerate(actions):
        if action.kind == "stop":
            result.append(index)
            continue
        if index in made_set:
            continue
        unlock_indices = _needs_unlock(action, actions)
        if unlock_indices is not None and not any(item in made_set for item in unlock_indices):
            continue
        result.append(index)
    return result


def _is_write(action: Action) -> bool:
    if action.kind == "altered":
        return True
    if action.kind in ("read", "stop"):
        return False
    return _action_key(action) in _WRITE_KEYS


def _is_gating(call: Call) -> bool:
    return call.name == "log_verification" or _inner_name(call) == "log_verification"


def features(actions: list[Action], i: int, made: list[int], step: int, horizon: int) -> list[float]:
    """The eight shared features, in the order fixed by the pre-registration."""
    action = actions[i]
    made_set = set(made)
    gold_next = next(
        (index for index, item in enumerate(actions) if item.kind == "gold" and index not in made_set),
        None,
    )
    return [
        float(action.kind == "read"),
        float(_is_write(action)),
        float(action.kind == "gold"),
        float(i == gold_next),
        float(action.kind == "altered"),
        float(any(_is_gating(call) for call in action.calls)),
        float(action.kind == "stop"),
        float(step / horizon) if horizon else 0.0,
    ]


def trajectory(actions: list[Action], made: list[int]) -> list[Call]:
    calls = []
    for index in made:
        calls.extend(actions[index].calls)
    return calls


def order_ok(task, calls: list[Call]) -> bool:
    """Whether every gated gold write the trajectory makes comes after its own verification."""
    task_id = getattr(task, "id", None)
    gated = _GATED_WRITES.get(task_id)
    if gated is None:
        gold = gold_calls(task)
        if not gold:
            _GATED_WRITES[task_id] = []
            return True
        env = fresh_env(_domain_for(task), task)
        kinds = [tool_kind(env, call) for call in gold]
        gated = _gated_writes(gold, kinds)
        _GATED_WRITES[task_id] = gated
    if not gated:
        return True
    call_keys = [_effective_call_keys(call) for call in calls]
    for verification, writes in gated:
        verification_keys = _effective_call_keys(verification)
        verification_indices = [index for index, keys in enumerate(call_keys) if keys & verification_keys]
        for write in writes:
            write_keys = _effective_call_keys(write)
            for index, keys in enumerate(call_keys):
                if not keys & write_keys:
                    continue
                if not any(item < index for item in verification_indices):
                    return False
    return True


def oracle(task, actions: list[Action], made: list[int], new_reward) -> dict:
    """Our true success: G_new passes, verification is in order, and no altered write was made."""
    calls = trajectory(actions, made)
    ordered = order_ok(task, calls)
    no_altered = not any(actions[index].kind == "altered" for index in made)
    true_success = bool(new_reward == 1.0 and ordered and no_altered)
    return {"true_success": true_success, "order_ok": ordered, "no_altered": no_altered}
