"""Phase 3 analysis: join regrade records, compute H5/H7 and draw H6's blinded sample."""

import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np

BANK = "banking_knowledge"
AIRLINE = "airline"
GRADED_CLASSES = frozenset({"full", "strict", "lenient"})
P3_TABLES = frozenset(
    {
        "credit_card_applications",
        "credit_limit_increase_requests",
        "transaction_disputes",
    }
)
RESAMPLES = 2000
SAMPLE_SEED = 0
SAMPLE_SIZES = (("S1", 8), ("S2", 8), ("S3", 4), ("S4", 4))
_MISSING = object()


def _stem(path) -> str:
    name = Path(path).name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_records(directory) -> list[dict]:
    """Every JSONL record in a regrade directory, tagged with the original file's stem."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        stem = _stem(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            record = dict(record)
            record["_stem"] = stem
            records.append(record)
    return records


def _graded(record) -> bool:
    reward = record.get("reward")
    return (
        record.get("replay_class") in GRADED_CLASSES
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and record.get("error") is None
    )


def _domain_of(record) -> str:
    value = record.get("domain")
    return value if isinstance(value, str) else ""


def _join(old_records, new_records) -> list[tuple[dict, dict]]:
    """Pair old and new records by id; the first new record of an id is the counterpart."""
    new_by_id = {}
    for record in sorted(new_records, key=lambda item: (str(item.get("id")), str(item.get("_stem")))):
        record_id = record.get("id")
        if record_id is not None:
            new_by_id.setdefault(str(record_id), record)
    pairs = []
    for old in old_records:
        record_id = old.get("id")
        if record_id is None:
            continue
        new = new_by_id.get(str(record_id))
        if new is not None:
            pairs.append((old, new))
    pairs.sort(key=lambda pair: str(pair[0].get("id")))
    return pairs


def _pool_ids(path) -> set[str]:
    data = _read_json(path)
    values = data.get("tasks", []) if isinstance(data, dict) else data
    return {str(value) for value in values}


def _flags(path) -> dict[str, set[str]]:
    data = _read_json(path)
    return {key: {str(value) for value in data.get(key, [])} for key in ("P1", "P3")}


def _event(old, new) -> bool:
    try:
        return float(old["reward"]) != float(new["reward"])
    except (KeyError, TypeError, ValueError):
        return False


def _h5_eligible(old, new, pool) -> bool:
    return (
        _domain_of(old) == BANK
        and _domain_of(new) == BANK
        and old.get("task_matches") is True
        and new.get("task_matches") is True
        and str(old.get("task_id")) in pool
        and _graded(old)
        and _graded(new)
    )


def _explained(old, flags) -> bool:
    task_id = str(old.get("task_id"))
    diff = old.get("diff_tables") or []
    tables = {str(table) for table in diff} if isinstance(diff, list) else set()
    if old.get("p1") is True and task_id in flags["P1"] and diff == ["agent_discoverable_tools"]:
        return True
    return (
        old.get("p3") is True
        and task_id in flags["P3"]
        and bool(tables)
        and not (tables - P3_TABLES)
    )


def _bootstrap_interval(events, flags) -> list[float] | None:
    """Task-clustered percentile interval for H5 recall, resampling task ids with replacement."""
    by_task = {}
    for old, new in events:
        task_id = str(old.get("task_id"))
        totals = by_task.setdefault(task_id, [0, 0])
        totals[0] += 1
        if _explained(old, flags):
            totals[1] += 1
    tasks = sorted(by_task)
    if not tasks:
        return None
    rng = np.random.default_rng(0)
    values = []
    for _ in range(RESAMPLES):
        indices = rng.integers(0, len(tasks), size=len(tasks))
        event_count = sum(by_task[tasks[index]][0] for index in indices)
        explained_count = sum(by_task[tasks[index]][1] for index in indices)
        if event_count:
            values.append(explained_count / event_count)
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _baseline(pairs, positive) -> dict:
    pos = [pair for pair in pairs if positive(pair)]
    neg = [pair for pair in pairs if not positive(pair)]

    def rate(items):
        if not items:
            return None
        return sum(1 for old, new in items if _event(old, new)) / len(items)

    positive_rate = rate(pos)
    negative_rate = rate(neg)
    enrichment = None
    if positive_rate is not None and negative_rate not in (None, 0.0):
        enrichment = positive_rate / negative_rate
    return {
        "positive": len(pos),
        "negative": len(neg),
        "events_positive": sum(1 for old, new in pos if _event(old, new)),
        "events_negative": sum(1 for old, new in neg if _event(old, new)),
        "recall": positive_rate,
        "enrichment": enrichment,
    }


def _precision(pairs, key) -> dict:
    triggered = [pair for pair in pairs if pair[0].get(key) is True]
    if not triggered:
        return {"runs": 0, "events": 0, "precision": None}
    events = sum(1 for old, new in triggered if _event(old, new))
    return {"runs": len(triggered), "events": events, "precision": events / len(triggered)}


def _h5_summary(pairs, flags) -> dict:
    events = [(old, new) for old, new in pairs if _event(old, new)]
    explained_count = sum(1 for old, new in events if _explained(old, flags))
    recall = explained_count / len(events) if events else None
    ci = _bootstrap_interval(events, flags) if events else None
    held = None
    if events:
        held = bool(recall >= 0.90 and ci is not None and ci[0] >= 0.75)
    unexplained = []
    for old, new in events:
        if _explained(old, flags):
            continue
        unexplained.append(
            {
                "id": old.get("id"),
                "task_id": old.get("task_id"),
                "diff_tables": old.get("diff_tables"),
                "triggers": {"p1": old.get("p1"), "p3": old.get("p3")},
                "reward_old": old.get("reward"),
                "reward_new": new.get("reward"),
            }
        )
    calls = [
        int(old.get("n_calls"))
        for old, _ in pairs
        if isinstance(old.get("n_calls"), (int, float)) and not isinstance(old.get("n_calls"), bool)
    ]
    median_calls = float(np.median(calls)) if calls else None

    def above_median(pair) -> bool:
        value = pair[0].get("n_calls")
        return (
            median_calls is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > median_calls
        )

    flagged_tasks = flags["P1"] | flags["P3"]
    return {
        "testable": bool(events),
        "runs": len(pairs),
        "events": len(events),
        "explained": explained_count,
        "recall": recall,
        "ci": ci,
        "held": held,
        "unexplained": unexplained,
        "precision": {"p1": _precision(pairs, "p1"), "p3": _precision(pairs, "p3")},
        "baselines": {
            "B1": _baseline(pairs, lambda pair: str(pair[0].get("task_id")) in flagged_tasks),
            "B2": _baseline(
                pairs,
                lambda pair: pair[0].get("nongold_call") is True or pair[1].get("nongold_call") is True,
            ),
            "B3": _baseline(pairs, above_median),
        },
    }


def _h5_document(pairs, flags) -> dict:
    document = _h5_summary(pairs, flags)
    for field, key in (("by_commit", "git_commit"), ("by_stem", "_stem")):
        groups = {}
        for pair in pairs:
            groups.setdefault(str(pair[0].get(key) or "unknown"), []).append(pair)
        document[field] = {
            name: _h5_summary(group, flags) for name, group in sorted(groups.items())
        }
    return document


def _replay_class_counts(records) -> dict:
    counts = {}
    for record in records:
        replay_class = record.get("replay_class")
        name = replay_class if isinstance(replay_class, str) else "missing"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _h7_document(pairs) -> dict:
    airline_new = [
        new
        for old, new in pairs
        if _domain_of(new) == AIRLINE and new.get("task_matches") is True and _graded(new)
    ]
    banking_reward_one = [
        new
        for old, new in pairs
        if _domain_of(new) == BANK
        and new.get("task_matches") is True
        and _graded(new)
        and float(new.get("reward")) == 1.0
    ]
    control_events = sum(
        1
        for old, new in pairs
        if _domain_of(new) == AIRLINE and _graded(old) and _graded(new) and _event(old, new)
    )
    return {
        "airline": {
            "runs": len(airline_new),
            "schema_float_runs": sum(1 for record in airline_new if record.get("schema_floats")),
            "inconsistencies": sum(
                1
                for record in airline_new
                if isinstance(record.get("numeric_pair"), dict)
                and record["numeric_pair"].get("inconsistency") is True
            ),
            "control_events": control_events,
        },
        "banking": {
            "reward_one_runs": len(banking_reward_one),
            "late_log_runs": sum(
                1
                for record in banking_reward_one
                if isinstance(record.get("late_verification"), dict)
                and record["late_verification"].get("log_after_first_write") is True
            ),
            "access_before_facts_runs": sum(
                1
                for record in banking_reward_one
                if isinstance(record.get("late_verification"), dict)
                and record["late_verification"].get("access_before_facts") is True
            ),
        },
    }


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool, int, float)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _write_runs_csv(path, pairs, flags) -> None:
    headers = [
        "id",
        "task_id",
        "trial",
        "domain",
        "git_commit",
        "stem_old",
        "stem_new",
        "task_matches_old",
        "task_matches_new",
        "replay_class_old",
        "replay_class_new",
        "reward_old",
        "reward_new",
        "p1_old",
        "p1_new",
        "p3_old",
        "p3_new",
        "n_calls_old",
        "n_calls_new",
        "nongold_call_old",
        "nongold_call_new",
        "diff_tables_old",
        "diff_tables_new",
        "event",
        "explained",
        "schema_floats_new",
        "inconsistency_new",
        "late_log_new",
        "access_before_facts_new",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for old, new in pairs:
            late = new.get("late_verification") if isinstance(new.get("late_verification"), dict) else {}
            numeric = new.get("numeric_pair") if isinstance(new.get("numeric_pair"), dict) else {}
            writer.writerow(
                [
                    old.get("id"),
                    old.get("task_id"),
                    old.get("trial"),
                    old.get("domain"),
                    old.get("git_commit"),
                    old.get("_stem"),
                    new.get("_stem"),
                    old.get("task_matches"),
                    new.get("task_matches"),
                    old.get("replay_class"),
                    new.get("replay_class"),
                    old.get("reward"),
                    new.get("reward"),
                    old.get("p1"),
                    new.get("p1"),
                    old.get("p3"),
                    new.get("p3"),
                    old.get("n_calls"),
                    new.get("n_calls"),
                    old.get("nongold_call"),
                    new.get("nongold_call"),
                    old.get("diff_tables"),
                    new.get("diff_tables"),
                    _event(old, new),
                    _explained(old, flags),
                    new.get("schema_floats"),
                    numeric.get("inconsistency"),
                    late.get("log_after_first_write"),
                    late.get("access_before_facts"),
                ]
            )


def _write_report(path, h5, h7, pairs) -> None:
    lines = [
        "# Trajectory audit",
        "",
        f"Joined runs: {len(pairs)}.",
        "",
        "## H5 (banking)",
        "",
        f"Testable: {h5['testable']}; runs: {h5['runs']}; events: {h5['events']}; explained: {h5['explained']}.",
    ]
    if h5["testable"]:
        lines.append(f"Recall: {h5['recall']:.4f}; 95% CI: {h5['ci']}; held: {h5['held']}.")
    else:
        lines.append("No events, so recall is undefined and H5 is not testable on this input.")
    lines += ["", "## H7", ""]
    lines.append(
        "Airline: {runs} runs, {schema_float_runs} with schema floats, {inconsistencies} inconsistencies, "
        "{control_events} control events.".format(**h7["airline"])
    )
    lines.append(
        "Banking: {reward_one_runs} reward-1 runs under new, {late_log_runs} with a late log, "
        "{access_before_facts_runs} with early customer access.".format(**h7["banking"])
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(old_dir, new_dir, flags_path, pool_path, out_path):
    """Write h5.json, h7.json, runs.csv and report.md from two regrade directories."""
    old_records = _read_records(old_dir)
    new_records = _read_records(new_dir)
    pool = _pool_ids(pool_path)
    flags = _flags(flags_path)
    pairs = _join(old_records, new_records)
    eligible = [pair for pair in pairs if _h5_eligible(pair[0], pair[1], pool)]
    h5 = _h5_document(eligible, flags)
    h5["replay_classes"] = {
        "old": _replay_class_counts(old_records),
        "new": _replay_class_counts(new_records),
    }
    h5["all_runs"] = {"old": len(old_records), "new": len(new_records), "joined": len(pairs)}
    h7 = _h7_document(pairs)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "h5.json", h5)
    _write_json(out_path / "h7.json", h7)
    _write_runs_csv(out_path / "runs.csv", pairs, flags)
    _write_report(out_path / "report.md", h5, h7, pairs)
    return {"h5": h5, "h7": h7}


def _late_true(record) -> bool:
    late = record.get("late_verification") if isinstance(record.get("late_verification"), dict) else {}
    return late.get("log_after_first_write") is True or late.get("access_before_facts") is True


def _failed_write(domain, task, run) -> bool:
    from preflight import trajectories
    from preflight.relations import tool_error, tool_kind

    env = trajectories._tool_env(domain, task)
    run_calls = trajectories.calls(run.get("messages") or [])
    outputs = trajectories._tool_outputs(run.get("messages") or [], len(run_calls))
    for index, call in enumerate(run_calls):
        if call.requestor == "user" or tool_kind(env, call) != "write":
            continue
        output = outputs[index] if index < len(outputs) else None
        if tool_error(output):
            return True
    return False


def _sampling_key(domain, call) -> str:
    from preflight import trajectories

    name = trajectories._tool_name(call)
    arguments = trajectories._effective_arguments(call)
    tool = trajectories._tool_for(domain, call, name)
    properties = trajectories._schema(tool).get("properties") or {} if tool is not None else {}
    kept = {}
    for argument, value in arguments.items():
        declared = properties.get(argument)
        default = declared.get("default", _MISSING) if isinstance(declared, dict) else _MISSING
        if default is not _MISSING and trajectories._as_ints(value) == trajectories._as_ints(default):
            continue
        kept[argument] = trajectories._as_ints(value)
    return _canonical([name, kept])


def _write_not_in_gold(domain, task, run) -> bool:
    from preflight import trajectories
    from preflight.relations import tool_kind

    env = trajectories._tool_env(domain, task)
    gold_keys = {_sampling_key(domain, call) for call in trajectories.gold_calls(task)}
    for call in trajectories.calls(run.get("messages") or []):
        if call.requestor == "user" or tool_kind(env, call) != "write":
            continue
        if trajectories._tool_name(call) == "log_verification":
            continue
        if _sampling_key(domain, call) not in gold_keys:
            return True
    return False


def _process_flag(domain, task, run, record) -> bool:
    if _late_true(record):
        return True
    if task is None or not domain:
        return False
    try:
        return _failed_write(domain, task, run) or _write_not_in_gold(domain, task, run)
    except Exception:
        return False


def _raw_info(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    info = data.get("info") if isinstance(data, dict) else None
    return info if isinstance(info, dict) else {}


def _find_policy(value) -> str | None:
    if isinstance(value, dict):
        for key in ("policy", "system_prompt", "agent_policy"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text
        for item in value.values():
            found = _find_policy(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_policy(item)
            if found:
                return found
    return None


def _domain_policy(domain) -> str:
    try:
        spec = importlib.util.find_spec("tau2")
    except (ImportError, ValueError):
        return ""
    if spec is None or spec.origin is None:
        return ""
    root = Path(spec.origin).resolve().parent
    names = ("policy.md", "policy.txt", "system_prompt.md", "system_prompt.txt")
    candidates = [root / "domains" / domain / name for name in names]
    candidates += [root / domain / name for name in names]
    candidates += [root / "data" / domain / name for name in names]
    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def _instructions(task) -> str:
    if not isinstance(task, dict):
        return ""
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict) and isinstance(scenario.get("instructions"), str):
        return scenario["instructions"]
    for key in ("instructions", "task_instructions"):
        text = task.get(key)
        if isinstance(text, str):
            return text
    return ""


def _load_source_files(paths) -> dict[str, dict]:
    from preflight import trajectories

    entries = {}
    for path in paths:
        try:
            loaded = trajectories.load(path)
        except Exception:
            continue
        policy = _find_policy(_raw_info(path)) or ""
        domain = loaded.get("domain")
        for run in loaded.get("runs") or []:
            run_id = str(run.get("id"))
            if run_id in entries:
                continue
            entries[run_id] = {
                "run": run,
                "task": loaded.get("tasks", {}).get(str(run.get("task_id"))),
                "domain": domain,
                "policy": policy,
            }
    return entries


def _packet(entry) -> dict:
    domain = entry.get("domain") or ""
    policy = entry.get("policy") or _domain_policy(domain)
    return {
        "task_instructions": _instructions(entry.get("task")),
        "policy": policy,
        "messages": entry.get("run", {}).get("messages") or [],
    }


def _strata(pairs, pool, entries) -> dict[str, set[str]]:
    s1, s2, s3, s4 = set(), set(), set(), set()
    for old, new in pairs:
        if _domain_of(new) != BANK or new.get("task_matches") is not True or not _graded(new):
            continue
        run_id = str(new.get("id"))
        if run_id == "None":
            continue
        reward = float(new["reward"])
        if _h5_eligible(old, new, pool) and _event(old, new):
            s4.add(run_id)
        if reward == 1.0:
            entry = entries.get(run_id)
            flagged = (
                _process_flag(entry.get("domain"), entry.get("task"), entry.get("run"), new)
                if entry
                else _late_true(new)
            )
            (s1 if flagged else s2).add(run_id)
        elif reward == 0.0:
            s3.add(run_id)
    return {"S1": s1 - s4, "S2": s2 - s4, "S3": s3 - s4, "S4": s4}


def sample(old_dir, new_dir, flags_path, pool_path, files, out_path):
    """Draw H6's stratified 24-run sample and write blinded packets plus the key."""
    old_records = _read_records(old_dir)
    new_records = _read_records(new_dir)
    pool = _pool_ids(pool_path)
    _flags(flags_path)  # the flags file is required by the interface; H6 strata use the records themselves
    pairs = _join(old_records, new_records)
    entries = _load_source_files(files)
    strata = _strata(pairs, pool, entries)
    rng = random.Random(SAMPLE_SEED)
    selected = []
    shortfalls = {}
    for name, size in SAMPLE_SIZES:
        ids = sorted(str(item) for item in strata.get(name, set()))
        if len(ids) <= size:
            chosen = ids
        else:
            rng.shuffle(ids)
            chosen = ids[:size]
        if len(chosen) < size:
            shortfalls[name] = size - len(chosen)
        selected.extend((name, run_id) for run_id in chosen)
    rng.shuffle(selected)
    out_path = Path(out_path)
    packets = out_path / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    for old_packet in packets.glob("*.json"):
        old_packet.unlink(missing_ok=True)
    key = []
    for stratum, run_id in selected:
        entry = entries.get(run_id)
        if entry is None:
            continue
        number = len(key) + 1
        packet_name = f"{number:02d}.json"
        _write_json(packets / packet_name, _packet(entry))
        key.append({"packet": number, "run_id": run_id, "stratum": stratum})
    _write_json(out_path / "key.json", key)
    if shortfalls:
        _write_json(out_path / "shortfalls.json", shortfalls)
    return key
