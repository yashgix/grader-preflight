"""Collect the aggregate results of all three phases into results/*.json, from the experiment output folders.

Run with the v1.0.1 interpreter (it re-reads published trajectories and phase 2 records):
    .tau2/v1.0.1/venv/bin/python scripts/collect_results.py --phase1 P1 --phase2 P2 --phase3 P3 --trajectories T

P1: phase 1 run folder (metrics.json); P2: phase 2 full-run folder (runs/, split.json); P3: phase 3 analysis folder
(old/, new/, analysis/, h6-labels/); T: the held-out published trajectory files. Only counts and identifiers of
public submissions are written, never trajectory contents.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preflight import full, trajectories  # noqa: E402
from preflight.tau2_adapter import gold_calls, tasks  # noqa: E402

REPORT_TOOL = "submit_interest_discrepancy_report_7294"


def read(path):
    return json.loads(Path(path).read_text())


def _number(pattern, text):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"phase 1 report has no line matching {pattern!r}")
    return int(match.group(1))


def phase1(folder):
    m = read(folder / "metrics.json")
    text = (folder / "report.md").read_text()
    counts = {
        "comparable_banking_tasks": len(read(ROOT / "data" / "banking_pool.json")["tasks"]),
        "p1_paired_verdicts": _number(r"H1\(a\): \w+ - paired verdicts: (\d+)", text),
        "p3_paired_verdicts": _number(r"H1\(b\): \w+ - paired verdicts: (\d+)", text),
        "f3_variants": _number(r"F3 pass rate NEW: [\d.]+ \(\d+/(\d+);", text),
        "f6_tasks": _number(r"F6 banking pass share NEW: [\d.]+ \(\d+/(\d+) tasks", text),
        "f1_airline_verdicts": _number(r"F1 airline passing tasks NEW: \d+ \((\d+) verdicts", text),
    }
    keys = [
        "attempted_old",
        "attempted_new",
        "errors_old",
        "errors_new",
        "changed_tasks",
        "control_mismatches",
        "p1_flagged_tasks_old",
        "p1_flagged_tasks_new",
        "p3_flagged_tasks_old",
        "p3_flagged_tasks_new",
        "f1_pass_airline_new",
        "f3_pass_rate_new",
        "f6_pass_share_new",
        "h1a_held",
        "h1b_held",
        "h2_held",
    ]
    return {k: m[k] for k in keys} | counts


def phase2(folder):
    train_ids, held_ids = full.check_split(folder)
    metrics, table, per_seed, finds = full.analyze(folder, train_ids, held_ids)
    seeds = {
        c: {
            str(s): {
                k: per_seed[(c, s)][k]
                for k in ("true", "extra_reads", "greedy_reads_after_gold", "greedy_stop_after_gold")
            }
            for s in full.SEEDS
        }
        for c in full.CONDITIONS
    }
    return {
        "metrics": metrics,
        "per_seed": seeds,
        "post_completion_reads_old_minus_new": table[("old-new", "greedy_reads_after_gold")],
        "h3_finds_by_class": Counter(f["violation"] for f in finds),
    }


def reports(calls):
    out = []
    for call in calls:
        if call.name == "call_discoverable_agent_tool" and call.arguments.get("agent_tool_name") == REPORT_TOOL:
            args = call.arguments.get("arguments")
            args = json.loads(args) if isinstance(args, str) else (args or {})
            out.append(
                {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for k, v in args.items()}
            )
    return out


def blind_spot(records_dir, trajectory_dir):
    """Reward-1 runs on tasks whose gold files a discrepancy report, and those whose reports differ from gold."""
    banking = {t.id: t for t in tasks("banking_knowledge")}
    affected = {tid: t for tid, t in banking.items() if reports(gold_calls(t))}
    rewards = {}
    for path in Path(records_dir).glob("*.new.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            rewards[(path.name.split(".")[0], record["id"])] = record.get("reward")
    rows, differing = Counter(), []
    for path in sorted(Path(trajectory_dir).glob("*_banking.json")):
        for run in trajectories.load(path)["runs"]:
            task = affected.get(run["task_id"])
            if task is None or rewards.get((path.stem, run["id"])) != 1.0:
                continue
            rows[run["task_id"]] += 1
            gold, filed = reports(gold_calls(task)), reports(trajectories.calls(run["messages"]))
            if sorted(map(json.dumps, gold)) != sorted(map(json.dumps, filed)):
                gold_by_account = {r["account_id"]: r for r in gold}
                differences = [
                    {"account": r["account_id"], "field": k, "filed": v, "gold": gold_by_account[r["account_id"]][k]}
                    for r in filed
                    if r not in gold and r["account_id"] in gold_by_account
                    for k, v in r.items()
                    if v != gold_by_account[r["account_id"]].get(k)
                ]
                differing.append(
                    {
                        "file": path.stem,
                        "task": run["task_id"],
                        "trial": run["trial"],
                        "gold_reports": len(gold),
                        "filed_reports": len(filed),
                        "differences": differences,
                    }
                )
    return {
        "affected_tasks": sorted(affected),
        "reward1_runs": dict(sorted(rows.items())),
        "reward1_runs_total": sum(rows.values()),
        "reports_differ_from_gold": differing,
    }


def phase3(folder, trajectory_dir):
    h5, h7 = read(folder / "analysis" / "h5.json"), read(folder / "analysis" / "h7.json")
    unexplained = Counter(u["task_id"] for u in h5["unexplained"])
    with open(folder / "analysis" / "runs.csv", newline="") as handle:
        events = Counter(r["task_id"] for r in csv.DictReader(handle) if r["event"] == "True")
    faithful = {}
    for path in sorted((folder / "new").glob("*.new.jsonl")):
        graded = [json.loads(line) for line in path.read_text().splitlines()]
        graded = [r for r in graded if r.get("reward") is not None]
        faithful[path.name.split(".")[0]] = {
            "graded": len(graded),
            "equal_to_recorded": sum(r["reward"] == r["recorded_reward"] for r in graded),
        }
    labels = folder / "h6-labels"
    agreement = {name: read(labels / f"{name}.json") for name in ("agreement", "agreement-v2")}
    adjudicated = read(labels / "adjudicated.json")
    key = {f"{k['packet']:02d}": k["stratum"] for k in read(folder / "h6-sample" / "key.json")}
    confirmed = Counter(key[p] for p, v in adjudicated["pages"].items() if v["outcome"] == "confirmed")
    return {
        "h5": {k: h5[k] for k in ("events", "explained", "recall", "ci", "held", "testable")}
        | {
            "unexplained_by_task": dict(sorted(unexplained.items())),
            "events_by_task": dict(sorted(events.items())),
            "by_file": {s: {"events": v["events"], "explained": v["explained"]} for s, v in h5["by_stem"].items()},
            "p1_trigger_precision": h5["precision"]["p1"],
            "p3_trigger": h5["precision"]["p3"],
            "baseline_task_flag": h5["baselines"]["B1"],
        },
        "files": {
            s: {
                side: {
                    k: v[side][k] for k in ("runs", "task_mismatch", "incompatible", "incompatible_share", "over_20pct")
                }
                for side in ("old", "new")
            }
            for s, v in h5["files"].items()
        },
        "regrade_equals_recorded": faithful,
        "h7": h7,
        "h6": {
            "sample": dict(Counter(key.values())),
            "agreement_per_run": {
                "v1": agreement["agreement"]["violation_per_run"],
                "v2": agreement["agreement-v2"]["violation_per_run"],
            },
            "agreement_items_v2": {
                k: {"agree": v["agree"], "kappa": v["kappa"]} for k, v in agreement["agreement-v2"]["items"].items()
            },
            "confirmed_by_stratum": dict(confirmed),
            "confirmed_hidden_in_reward1": sum(v.get("hidden", False) for v in adjudicated["pages"].values()),
        },
        "blind_spot": blind_spot(folder / "new", trajectory_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--phase1", "--phase2", "--phase3", "--trajectories"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    for name, value in (
        ("phase1", phase1(args.phase1)),
        ("phase2", phase2(args.phase2)),
        ("phase3", phase3(args.phase3, args.trajectories)),
    ):
        (args.out / f"{name}.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        print("wrote", args.out / f"{name}.json")


if __name__ == "__main__":
    main()
