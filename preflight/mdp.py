"""Open-loop MDP over banking tool calls for the phase 2 RL preflight."""

from dataclasses import dataclass, field

from preflight.relations import (
    UNLOCK_AGENT,
    WRITE,
    _inserted_read,
    _p1_plans,
    _unlocked_before,
    calls_as_dicts,
    canonical_json,
    tool_kind,
)
from preflight.tau2_adapter import Call, fresh_env, gold_calls, tasks

_WRITE_KEYS: set[str] = set()
_GATED_NAMES: dict[object, set[str]] = {}
_TASK_DOMAINS: dict[object, str] = {}


@dataclass(frozen=True)
class Action:
    kind: str
    calls: tuple[Call, ...]
    gold_index: int | None = None
    is_write: bool | None = field(default=None, compare=False, repr=False)


def _calls_key(calls) -> str:
    return canonical_json(calls_as_dicts(calls))


def _action_key(action: Action) -> str:
    return _calls_key(action.calls)


def _domain_for(task) -> str:
    task_id = getattr(task, "id", None)
    if task_id in _TASK_DOMAINS:
        return _TASK_DOMAINS[task_id]
    declared = getattr(task, "domain", None)
    return declared if isinstance(declared, str) and declared else "banking_knowledge"


def _read_actions(domain: str, task, gold: list[Call]) -> list[Action]:
    """Phase 1's P1 read candidates, with an unlock when the gold run has not unlocked one yet."""
    try:
        plans = _p1_plans(domain, task, gold)
    except Exception:
        return []
    actions = []
    seen = set()
    for route, name, position, arguments in plans:
        read = _inserted_read(route, name, arguments)
        if route == "agent" and not _unlocked_before(gold, name, position):
            calls = (Call(UNLOCK_AGENT, {"agent_tool_name": name}), read)
        else:
            calls = (read,)
        key = _calls_key(calls)
        if key in seen:
            continue
        seen.add(key)
        actions.append(Action("read", calls, None, is_write=False))
    return actions


def _alternative_value(domain: str, task, name: str, current: str) -> str | None:
    """Another string value for an argument, from this task's gold run, else other tasks'."""
    values = set()
    for call in gold_calls(task):
        value = (call.arguments or {}).get(name)
        if isinstance(value, str) and value != current:
            values.add(value)
    if values:
        return sorted(values)[0]
    for other in tasks(domain):
        if getattr(other, "id", None) == getattr(task, "id", None):
            continue
        for call in gold_calls(other):
            value = (call.arguments or {}).get(name)
            if isinstance(value, str) and value != current:
                values.add(value)
    return sorted(values)[0] if values else None


def _altered_actions(domain: str, task, gold: list[Call], kinds: list[str]) -> list[Action]:
    """At most two altered copies of each gold write, each swapping one string `_id` argument."""
    actions = []
    for index, call in enumerate(gold):
        if kinds[index] != WRITE:
            continue
        arguments = call.arguments or {}
        made = 0
        for name in sorted(arguments):
            if made >= 2:
                break
            if not name.endswith("_id"):
                continue
            current = arguments.get(name)
            if not isinstance(current, str):
                continue
            alternative = _alternative_value(domain, task, name, current)
            if alternative is None:
                continue
            altered_arguments = dict(arguments)
            altered_arguments[name] = alternative
            altered = Call(call.name, altered_arguments, call.requestor)
            actions.append(Action("altered", (altered,), index, is_write=True))
            made += 1
    return actions


def _gated_names(gold: list[Call], kinds: list[str]) -> set[str]:
    names = set()
    for position, call in enumerate(gold):
        if call.name != "log_verification":
            continue
        for later in range(position + 1, len(gold)):
            if kinds[later] == WRITE:
                names.add(gold[later].name)
    return names


def task_actions(domain: str, task) -> list[Action]:
    """Gold actions first, then phase 1 read candidates, then altered writes, STOP last."""
    _TASK_DOMAINS[getattr(task, "id", None)] = domain
    gold = gold_calls(task)
    env = fresh_env(domain, task)
    kinds = [tool_kind(env, call) for call in gold]
    _GATED_NAMES[getattr(task, "id", None)] = _gated_names(gold, kinds)
    actions = []
    for index, call in enumerate(gold):
        is_write = kinds[index] == WRITE
        action = Action("gold", (call,), index, is_write=is_write)
        if is_write:
            _WRITE_KEYS.add(_action_key(action))
        actions.append(action)
    actions.extend(_read_actions(domain, task, gold))
    for action in _altered_actions(domain, task, gold, kinds):
        _WRITE_KEYS.add(_action_key(action))
        actions.append(action)
    actions.append(Action("stop", (), None, is_write=False))
    return actions


def horizon(task) -> int:
    return len(gold_calls(task)) + 3


def allowed(actions: list[Action], made: list[int]) -> list[int]:
    """Every action at most once per episode except STOP, which is always available."""
    made_set = set(made)
    return [index for index, action in enumerate(actions) if action.kind == "stop" or index not in made_set]


def _is_write(action: Action) -> bool:
    if action.is_write is not None:
        return action.is_write
    if action.kind == "altered":
        return True
    return _action_key(action) in _WRITE_KEYS


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
        float(any(call.name == "log_verification" for call in action.calls)),
        float(action.kind == "stop"),
        float(step / horizon) if horizon else 0.0,
    ]


def trajectory(actions: list[Action], made: list[int]) -> list[Call]:
    calls = []
    for index in made:
        calls.extend(actions[index].calls)
    return calls


def order_ok(task, calls: list[Call]) -> bool:
    """Whether every log_verification in the calls precedes the writes it gates."""
    task_id = getattr(task, "id", None)
    gated = _GATED_NAMES.get(task_id)
    if gated is None:
        gold = gold_calls(task)
        if not gold:
            return True
        env = fresh_env(_domain_for(task), task)
        kinds = [tool_kind(env, call) for call in gold]
        gated = _gated_names(gold, kinds)
        _GATED_NAMES[task_id] = gated
    if not gated:
        return True
    for index, call in enumerate(calls):
        if call.name != "log_verification":
            continue
        if any(earlier.name in gated for earlier in calls[:index]):
            return False
    return True


def oracle(task, actions: list[Action], made: list[int], new_reward) -> dict:
    """Our true success: G_new passes, verification is in order, and no altered write was made."""
    calls = trajectory(actions, made)
    ordered = order_ok(task, calls)
    no_altered = not any(actions[index].kind == "altered" for index in made)
    true_success = bool(new_reward == 1.0 and ordered and no_altered)
    return {"true_success": true_success, "order_ok": ordered, "no_altered": no_altered}
