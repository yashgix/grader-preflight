"""Phase 3: synthetic trajectory audit against the two tau2 graders.

Every run is built from tau2's own gold actions with `as_simulation`, so the build never handles real
third-party trajectory files. Each environment builds and regrades its own copies; the parent process joins
the two sides, writes the phase 1-style flags file and runs the H5/H7 analysis.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

# tau2's LLM paths must never reach the network; set before tau2 is imported.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from preflight import trajectories, trajectory_analysis
from preflight.relations import build_variants, f6_late_verification, p3_respelled_numbers
from preflight.tau2_adapter import Call, gold_calls, tasks

BANK = "banking_knowledge"
AIRLINE = "airline"
DOMAINS = (BANK, AIRLINE)
REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "data" / "banking_pool.json"
DEFAULT_OLD = "/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python"
DEFAULT_NEW = "/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python"


def _environment() -> dict:
    env = os.environ.copy()
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _task_dict(task) -> dict:
    dump = getattr(task, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def pool_ids(path=None) -> list[str]:
    path = Path(path) if path is not None else POOL_PATH
    if not path.is_file():
        return [task.id for task in tasks(BANK)]
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("tasks", []) if isinstance(data, dict) else data
    return [str(value) for value in values]


def _write_results(path: Path, domain: str, task_dicts: list[dict], simulations: list[dict]) -> None:
    payload = {
        "info": {"environment_info": {"domain_name": domain}, "git_commit": None},
        "tasks": task_dicts,
        "simulations": simulations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _banking_file(path: Path) -> None:
    by_id = {task.id: task for task in tasks(BANK)}
    task_dicts, simulations = [], []
    for task_id in pool_ids():
        task = by_id.get(str(task_id))
        if task is None:
            continue
        task_dicts.append(_task_dict(task))
        gold = gold_calls(task)
        simulations.append(trajectories.as_simulation(BANK, task, gold, trial=0))
        simulations.append(trajectories.as_simulation(BANK, task, gold, trial=1, trimmed=True))

        p1_variants, _ = build_variants("P1", BANK, task, random.Random(f"0:{task.id}:P1"))
        p1 = next((variant for variant in p1_variants if variant.detail.startswith("extra agent read")), None)
        if p1 is not None:
            simulations.append(trajectories.as_simulation(BANK, task, p1.calls, trial=2))

        p3_variants = p3_respelled_numbers(BANK, task, random.Random(f"0:{task.id}:P3"))
        if p3_variants:
            simulations.append(trajectories.as_simulation(BANK, task, p3_variants[0].calls, trial=3))

        f6_variants = f6_late_verification(BANK, task, random.Random(f"0:{task.id}:F6"))
        if f6_variants:
            simulations.append(trajectories.as_simulation(BANK, task, f6_variants[0].calls, trial=4))

    _write_results(path, BANK, task_dicts, simulations)


def _integer_parameter(declared) -> bool:
    return trajectories._integer_parameter(declared)


def _float_call(domain: str, call: Call) -> Call:
    """The same call with every integer value for an integer-schema parameter turned into a float."""
    tool = trajectories._tool_for(domain, call, trajectories._tool_name(call))
    if tool is None:
        return call
    properties = trajectories._schema(tool).get("properties") or {}
    if call.name in trajectories._CALL_WRAPPERS:
        inner = dict(trajectories._inner_args(call))
        changed = False
        for name, value in list(inner.items()):
            if isinstance(value, int) and not isinstance(value, bool) and _integer_parameter(properties.get(name)):
                inner[name] = float(value)
                changed = True
        if not changed:
            return call
        arguments = dict(call.arguments or {})
        arguments["arguments"] = json.dumps(inner, sort_keys=True)
        return Call(call.name, arguments, call.requestor)
    arguments = dict(call.arguments or {})
    changed = False
    for name, value in list(arguments.items()):
        if isinstance(value, int) and not isinstance(value, bool) and _integer_parameter(properties.get(name)):
            arguments[name] = float(value)
            changed = True
    if not changed:
        return call
    return Call(call.name, arguments, call.requestor)


def float_calls(domain: str, calls: list[Call]) -> list[Call]:
    return [_float_call(domain, call) for call in calls]


def _airline_file(path: Path) -> None:
    task_dicts, simulations = [], []
    for task in tasks(AIRLINE):
        task_dicts.append(_task_dict(task))
        gold = gold_calls(task)
        simulations.append(trajectories.as_simulation(AIRLINE, task, gold, trial=0))
        simulations.append(trajectories.as_simulation(AIRLINE, task, float_calls(AIRLINE, gold), trial=1))
    _write_results(path, AIRLINE, task_dicts, simulations)


def build_synthetic(out_dir) -> None:
    out_dir = Path(out_dir)
    files = out_dir / "files"
    files.mkdir(parents=True, exist_ok=True)
    _banking_file(files / f"{BANK}.json")
    _airline_file(files / f"{AIRLINE}.json")
    trajectories.regrade([files / f"{BANK}.json", files / f"{AIRLINE}.json"], out_dir / "regrade")


def worker(out_dir) -> None:
    build_synthetic(out_dir)


def _write_flags(regrade_dir: Path, path: Path) -> dict:
    records = trajectory_analysis._read_records(regrade_dir)
    p1 = set()
    p3 = set()
    for record in records:
        if record.get("domain") != BANK:
            continue
        if record.get("p1") is True and record.get("reward") == 0.0:
            p1.add(str(record.get("task_id")))
        if record.get("p3") is True and record.get("reward") == 0.0:
            p3.add(str(record.get("task_id")))
    payload = {"P1": sorted(p1), "P3": sorted(p3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _trimmed_equal_gold(new_records: list[dict]) -> int:
    by_key = {}
    for record in new_records:
        if record.get("domain") != BANK:
            continue
        by_key[(record.get("task_id"), record.get("trial"))] = record
    count = 0
    for (task_id, trial), record in by_key.items():
        if trial != 1:
            continue
        gold = by_key.get((task_id, 0))
        if gold is None:
            continue
        left = {key: value for key, value in record.items() if key not in ("id", "trial", "_stem")}
        right = {key: value for key, value in gold.items() if key not in ("id", "trial", "_stem")}
        if left == right:
            count += 1
    return count


def _late_log_runs(new_records: list[dict]) -> int:
    count = 0
    for record in new_records:
        if record.get("domain") != BANK or record.get("trial") != 4:
            continue
        late = record.get("late_verification")
        if isinstance(late, dict) and late.get("log_after_first_write") is True:
            count += 1
    return count


def _metrics(old_records: list[dict], new_records: list[dict], analysis: dict) -> dict:
    h5 = analysis["h5"]
    h7 = analysis["h7"]
    recall = h5.get("recall")
    return {
        "runs_banking": sum(1 for record in new_records if record.get("domain") == BANK),
        "runs_airline": sum(1 for record in new_records if record.get("domain") == AIRLINE),
        "full_replay_new": sum(1 for record in new_records if record.get("replay_class") == "full"),
        "incompatible_old": sum(1 for record in old_records if record.get("replay_class") == "incompatible"),
        "incompatible_new": sum(1 for record in new_records if record.get("replay_class") == "incompatible"),
        "events": int(h5.get("events", 0)),
        "explained": int(h5.get("explained", 0)),
        "recall": float(recall) if recall is not None else 0.0,
        "p1_trigger_runs": sum(1 for record in new_records if record.get("p1") is True),
        "p3_trigger_runs": sum(1 for record in new_records if record.get("p3") is True),
        "trimmed_equal_gold": _trimmed_equal_gold(new_records),
        "schema_float_runs": int(h7["airline"]["schema_float_runs"]),
        "inconsistencies": int(h7["airline"]["inconsistencies"]),
        "airline_control_events": int(h7["airline"]["control_events"]),
        "late_log_runs": _late_log_runs(new_records),
        "errors": sum(1 for record in old_records + new_records if record.get("error") is not None),
    }


def _run_worker(python: str, out_dir: Path) -> None:
    subprocess.run(
        [python, "-B", "-m", "preflight.phase3", "worker", "--out", str(out_dir)],
        cwd=REPO_ROOT,
        env=_environment(),
        check=True,
    )


def run_phase3(out_path, old_python: str = DEFAULT_OLD, new_python: str = DEFAULT_NEW, processes: int = 4) -> dict:
    """Build known-answer runs in each env, regrade them, join them, then write the phase 3 metrics."""
    out_path = Path(out_path).resolve()
    folder = out_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    old_dir = folder / "phase3_old"
    new_dir = folder / "phase3_new"
    _run_worker(old_python, old_dir)
    _run_worker(new_python, new_dir)

    flags_path = folder / "phase3_flags.json"
    _write_flags(old_dir / "regrade", flags_path)
    analysis_dir = folder / "phase3_analysis"
    analysis = trajectory_analysis.analyze(
        old_dir / "regrade", new_dir / "regrade", flags_path, POOL_PATH, analysis_dir
    )

    old_records = trajectory_analysis._read_records(old_dir / "regrade")
    new_records = trajectory_analysis._read_records(new_dir / "regrade")
    metrics = _metrics(old_records, new_records, analysis)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _worker_main(argv) -> int:
    parser = argparse.ArgumentParser(prog="python -m preflight.phase3 worker")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    worker(args.out)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m preflight.phase3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-python", default=DEFAULT_OLD)
    parser.add_argument("--new-python", default=DEFAULT_NEW)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args(argv)
    run_phase3(args.out, args.old_python, args.new_python, args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
