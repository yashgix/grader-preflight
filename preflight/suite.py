"""Run the relation suite on a live tau2 environment and record per-case evidence."""

import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

from preflight.relations import (
    RELATIONS,
    Variant,
    build_variants,
    calls_as_dicts,
    canonical_json,
    db_hashes,
    static_checks,
    tool_error,
)
from preflight.tau2_adapter import env_kwargs, gold_calls, run, tasks

_MISSING = object()
STATIC_KEYS = ("gold_errors", "gold_changes_db", "no_effect_writes")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_fingerprint(task) -> str:
    return _sha256(canonical_json(task.model_dump(mode="json")))


def gold_fingerprint(task) -> str:
    return _sha256(canonical_json(calls_as_dicts(gold_calls(task))))


def _parse_output(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (TypeError, ValueError):
            return content
    return content


def _normalise_numbers(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [_normalise_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_numbers(item) for key, item in value.items()}
    return value


def output_fingerprint(outputs):
    normalised = [_normalise_numbers(_parse_output(output)) for output in outputs]
    return _sha256(canonical_json(normalised))


def _content(message):
    return getattr(message, "content", None)


def _tool_outputs(messages, count):
    """The tool messages of the last `count` calls in a run's message list."""
    if count <= 0:
        return []
    start = len(messages) - 2 * count + 1
    if start < 0:
        return []
    return [messages[start + 2 * index] for index in range(count) if start + 2 * index < len(messages)]


def _live_call_errors(messages, count):
    return [index for index, message in enumerate(_tool_outputs(messages, count)) if tool_error(message)]


def _error_info(exc):
    text = str(exc)
    first = text.splitlines()[0] if text else ""
    return {"type": type(exc).__name__, "message": first[:200]}


def _reward(domain, task, messages):
    """The DB-state verdict of tau2's unchanged grader, with the env arguments tau2's runner uses."""
    constructor = registry.get_env_constructor(domain)
    return EnvironmentEvaluator.calculate_reward(
        constructor, task, messages, env_kwargs=env_kwargs(domain, task)
    ).reward


def _checks(relation):
    if relation == "P1":
        return {"inserted_ran_cleanly": True}
    if relation == "P5":
        return {"inert": True}
    if relation == "F3":
        return {"end_state_changed": True}
    return {}


def _model_dump(value):
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(mode="json")
        except TypeError:
            try:
                dumped = dump()
            except Exception:
                return None
        except Exception:
            return None
        return dumped if isinstance(dumped, dict) else None
    if isinstance(value, dict):
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {key: item for key, item in attributes.items() if not key.startswith("_")}
    return None


def _as_mapping(value):
    if value is None or value is _MISSING:
        return {}
    if isinstance(value, dict):
        return value
    dumped = _model_dump(value)
    return dumped if isinstance(dumped, dict) else {}


def _toolkit_db(env, attr):
    toolkit = getattr(env, attr, None)
    if toolkit is None:
        return _MISSING
    if isinstance(toolkit, dict):
        if "db" not in toolkit:
            return _MISSING
        dumped = _model_dump(toolkit["db"])
        return dumped if isinstance(dumped, dict) else {}
    db = getattr(toolkit, "db", _MISSING)
    if db is _MISSING or db is None:
        return _MISSING
    dumped = _model_dump(db)
    return dumped if isinstance(dumped, dict) else {}


def _direct_db(env, names):
    for source in (getattr(env, "db", None), env):
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, _MISSING)
            if value is not _MISSING:
                return value
    return _MISSING


def _db_pair(env):
    if env is None:
        return {}, {}
    agent = _toolkit_db(env, "tools")
    user = _toolkit_db(env, "user_tools")
    if agent is not _MISSING or user is not _MISSING:
        return (
            _as_mapping(None if agent is _MISSING else agent),
            _as_mapping(None if user is _MISSING else user),
        )
    agent = _direct_db(env, ("agent_db", "agent"))
    user = _direct_db(env, ("user_db", "user"))
    return (
        _as_mapping(None if agent is _MISSING else agent),
        _as_mapping(None if user is _MISSING else user),
    )


def _same(left, right):
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _db_diff(env, gold_env):
    """Top-level DB field names whose values differ between the variant's state and the gold run's."""
    variant_agent, variant_user = _db_pair(env)
    gold_agent, gold_user = _db_pair(gold_env)
    names = set()
    for variant, gold in ((variant_agent, gold_agent), (variant_user, gold_user)):
        for key in set(variant) | set(gold):
            if key not in variant or key not in gold or not _same(variant[key], gold[key]):
                names.add(key)
    return sorted(names)


def _records_sha256(variants):
    records = [{key: value for key, value in variant.items() if key != "seconds"} for variant in variants]
    records.sort(key=lambda record: (record.get("task_id"), record.get("relation"), record.get("detail")))
    return _sha256("\n".join(canonical_json(record) for record in records))


def _source_text():
    """The environment's SOURCE.txt, which names the tau2 tag it was built from."""
    try:
        return (Path(sys.prefix) / "SOURCE.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _tau2_version(source):
    """The tag named on SOURCE.txt's first line, its second word (`pre-v1.0.1` or `v1.0.1`)."""
    words = source.splitlines()[0].split() if source else []
    return words[1] if len(words) > 1 else None


def _provenance(source):
    return {"source": source, "python": platform.python_version()}


def _selected_relations(relations):
    if relations is None:
        return list(RELATIONS)
    if isinstance(relations, str):
        requested = [item.strip() for item in relations.split(",") if item.strip()]
    else:
        requested = list(relations)
    unknown = sorted(set(requested) - set(RELATIONS))
    if unknown:
        raise ValueError(f"unknown relation(s): {', '.join(unknown)}")
    wanted = set(requested)
    return [relation for relation in RELATIONS if relation in wanted]


def _reward_basis(task):
    criteria = task.evaluation_criteria
    basis = criteria.reward_basis if criteria else []
    if isinstance(basis, str):
        basis = [basis]
    return [getattr(item, "value", item) for item in basis or []]


def _static_entry(domain, task):
    try:
        return dict(static_checks(domain, task))
    except Exception as exc:
        # Keep the record's shape fixed; the checks are reported as unknown beside the error.
        return dict.fromkeys(STATIC_KEYS) | {"static_error": _error_info(exc)}


def _variant_record(domain, task, variant: Variant, gold_state, gold_env):
    record = {
        "task_id": task.id,
        "relation": variant.relation,
        "detail": variant.detail,
        "expected_pass": variant.expected_pass,
        "inserted_in_gold": variant.inserted_in_gold,
        "calls": calls_as_dicts(variant.calls),
        "checks": _checks(variant.relation),
        "live_call_errors": [],
        "output_fingerprint": None,
        "end_state": None,
        "outcome": "error",
        "reward": None,
        "flagged": None,
        "error": None,
        "db_diff": None,
    }
    try:
        messages, env = run(domain, task, list(variant.calls))
        record["live_call_errors"] = _live_call_errors(messages, len(variant.calls))
        outputs = [_content(message) for message in _tool_outputs(messages, len(variant.calls))]
        record["output_fingerprint"] = output_fingerprint(outputs)
        record["end_state"] = {"variant": db_hashes(env), "gold": list(gold_state)}
        reward = float(_reward(domain, task, messages))
    except Exception as exc:
        record["error"] = _error_info(exc)
        return record
    record["outcome"] = "verdict"
    record["reward"] = reward
    record["flagged"] = (reward == 1.0) != bool(variant.expected_pass)
    if record["flagged"]:
        try:
            record["db_diff"] = _db_diff(env, gold_env)
        except Exception:
            record["db_diff"] = []
    return record


def _add_variants(domain, task, selected, seed, record, gold_env, gold_state, variant_records):
    for relation in selected:
        rng = random.Random(f"{seed}:{task.id}:{relation}")
        try:
            variants, notes = build_variants(relation, domain, task, rng)
        except Exception as exc:
            record["static"].setdefault("relation_errors", {})[relation] = _error_info(exc)
            continue
        if notes.get("identical_to_gold"):
            skipped = record["static"].setdefault("identical_to_gold", [])
            skipped.append(relation)
            skipped.sort()
        if notes.get("p5_not_inert"):
            record["static"]["p5_not_inert"] = True
        for variant in variants:
            variant_records.append(_variant_record(domain, task, variant, gold_state, gold_env))


def _task_record(domain, task, selected, seed, variant_records):
    calls = gold_calls(task)
    record = {
        "id": task.id,
        "reward_basis": _reward_basis(task),
        "task_fingerprint": task_fingerprint(task),
        "gold_fingerprint": gold_fingerprint(task),
        "gold_output_fingerprint": None,
        "base_outcome": "error",
        "base_reward": None,
        "static": _static_entry(domain, task),
    }
    gold_env = None
    gold_state = None
    try:
        messages, gold_env = run(domain, task, list(calls))
        gold_state = db_hashes(gold_env)
        outputs = [_content(message) for message in _tool_outputs(messages, len(calls))]
        record["gold_output_fingerprint"] = output_fingerprint(outputs)
        reward = float(_reward(domain, task, messages))
    except Exception as exc:
        record["static"]["base_error"] = _error_info(exc)
    else:
        record["base_outcome"] = "verdict"
        record["base_reward"] = reward
    if record["base_outcome"] == "verdict" and record["base_reward"] == 1.0:
        _add_variants(domain, task, selected, seed, record, gold_env, gold_state, variant_records)
    return record


def run_suite(domain, relations=None, seed=0, limit=None):
    started = time.perf_counter()
    selected = _selected_relations(relations)
    domain_tasks = tasks(domain)
    if limit is not None:
        domain_tasks = domain_tasks[:limit]
    task_records = []
    variant_records = []
    for task in domain_tasks:
        task_records.append(_task_record(domain, task, selected, seed, variant_records))
    source = _source_text()
    return {
        "domain": domain,
        "seed": seed,
        "tau2_version": _tau2_version(source),
        "provenance": _provenance(source),
        "tasks": task_records,
        "variants": variant_records,
        "records_sha256": _records_sha256(variant_records),
        "seconds": time.perf_counter() - started,
    }
