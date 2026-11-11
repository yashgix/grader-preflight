"""Phase 2 full sweep: retail tuning, 10 seeds x 3 graders on the 30 train tasks, baselines, held-out and greedy
evaluation, then the pre-registered measures with seed-level bootstrap intervals (amendments 5 and 6)."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from preflight import phase2
from preflight.mdp import task_actions, trajectory
from preflight.rl import GradeCache
from preflight.tau2_adapter import grade, tasks

BANK, RETAIL = "banking_knowledge", "retail"
SEEDS = tuple(range(10))
CONDITIONS = ("old", "new", "proc")
EPISODES, EVAL_EPISODES, LAST = 3000, 100, 300
TUNE_LRS, TUNE_SEEDS, TUNE_EPISODES, TUNE_TASKS = (0.03, 0.1, 0.3), (0, 1, 2), 2000, 30
RESAMPLES = 10_000


def retail_tuning_ids():
    """30 retail tasks whose empty run fails, drawn with the banking split's seed."""
    eligible = sorted(t.id for t in tasks(RETAIL) if grade(RETAIL, t, []) == 0.0)
    random.Random(phase2.SPLIT_SEED).shuffle(eligible)
    return eligible[:TUNE_TASKS]


def counterfactuals(task_ids, out_path):
    """Per task, this tau2 version's verdict on the empty run, the gold run, and the gold run followed by its first
    1 to 3 extra reads (the spare slots)."""
    known, cache = {t.id: t for t in tasks(BANK)}, GradeCache(BANK)
    rows = {}
    for task_id in task_ids:
        task = known[task_id]
        actions = task_actions(BANK, task)
        gold = [i for i, a in enumerate(actions) if a.kind == "gold"]
        reads = [i for i, a in enumerate(actions) if a.kind == "read" and phase2._extra_reads(actions, [i])]
        row = {"empty": cache(task, [])}  # None if the grader raised
        for k in range(min(3, len(reads)) + 1):
            row[f"gold_plus_{k}_reads"] = cache(task, trajectory(actions, gold + reads[:k]))
        rows[task_id] = row
    Path(out_path).write_text(json.dumps(rows, indent=1, sort_keys=True))


def _worker(python, condition, mode, path, episodes, seed, ids, domain=BANK, lr=phase2.DEFAULT_LR, source=None):
    args = phase2._worker_args(condition, mode, path, episodes, seed, ids, domain, lr, input_path=source)
    phase2._run_worker(python, args)


def _counterfactual_worker(python, out_path, ids):
    command = [python, "-B", "-m", "preflight.full", "counterfactual", str(out_path), *ids]
    subprocess.run(command, cwd=phase2.REPO_ROOT, env=phase2._environment(), check=True)


def tune(folder, python, processes):
    """Retail only, where both graders agree: the learning rate with the best mean true success; a tie keeps the
    smaller rate."""
    ids, out = retail_tuning_ids(), folder / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    def one(job):
        lr, seed = job
        path = out / f"lr{lr}_seed{seed}.json"
        if not path.exists():
            _worker(python, "new", "train", path, TUNE_EPISODES, seed, ids, RETAIL, lr)
        return lr, _rate(phase2._read_json(path)["episodes"][-LAST:], "true_success")

    scores = {}
    for lr, score in phase2._parallel_map(one, [(lr, s) for lr in TUNE_LRS for s in TUNE_SEEDS], processes):
        scores.setdefault(lr, []).append(score)
    means = {lr: float(np.mean(v)) for lr, v in sorted(scores.items())}
    chosen = max(means, key=means.get)  # dict order is ascending, so max keeps the first (smaller) rate on a tie
    phase2._write_json(out / "choice.json", {"tasks": ids, "mean_true_success": means, "chosen_lr": chosen})
    return chosen


def sweep(folder, old_python, new_python, lr, train_ids, held_ids, processes):
    """Every condition and seed with the same settings; G_old runs get true success from v1.0.1 afterwards. A rerun
    skips every file already complete, so a late failure loses nothing."""
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for name, python in (("old", old_python), ("new", new_python)):
        path = folder / f"counterfactual_{name}.json"
        if not path.exists():
            _counterfactual_worker(python, path, train_ids)

    def one(job):
        condition, seed = job
        if condition == "random":
            path = runs / f"random_seed{seed}.json"
            if not path.exists():
                _worker(new_python, "new", "random", path, EPISODES, seed, train_ids)
            return
        python = old_python if condition == "old" else new_python
        trained, evaluated = runs / f"{condition}_seed{seed}.json", runs / f"{condition}_seed{seed}_eval.json"
        if not trained.exists():
            _worker(python, condition, "train", trained, EPISODES, seed, train_ids, lr=lr)
        if not evaluated.exists():
            _worker(python, condition, "eval", evaluated, EVAL_EPISODES, seed, train_ids + held_ids, source=trained)
        for path in (trained, evaluated) if condition == "old" else ():
            if "oracle_grader_calls" not in phase2._read_json(path):  # written by the v1.0.1 fill
                _worker(new_python, "old", "oracle", path, EPISODES, seed, [], source=path)

    phase2._parallel_map(one, [(c, s) for c in (*CONDITIONS, "random") for s in SEEDS], processes)


def _rate(records, key):
    """Mean of a field over the records that have it; NaN when none has it (no data, not zero)."""
    values = [float(r[key]) for r in records if r.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def bootstrap(values):
    """Mean over seeds and its 95% percentile interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(phase2.SPLIT_SEED)
    means = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_measures(run):
    episodes, last = run["episodes"], run["episodes"][-LAST:]
    wrong = (i + 1 for i, e in enumerate(episodes) if e["reward"] == 1.0 and e["true_success"] is False)
    return {
        "proxy": _rate(last, "reward"),
        "true": _rate(last, "true_success"),
        "gap": _rate(last, "reward") - _rate(last, "true_success"),
        "extra_reads": _rate(last, "extra_reads"),
        "stop_without_acting": float(np.mean([not e["calls"] for e in last])),
        "gate": _rate(episodes[1700:2000], "true_success"),
        "first_wrong": next(wrong, None),
    }


def greedy_measures(eval_run, task_ids):
    """The greedy episodes on these tasks: completion, STOP right after the gold steps, later extra reads, and the
    STOP probability at that point."""
    greedy = [e for e in eval_run["episodes"] if e["greedy"] and e["task_id"] in task_ids]
    done = [e for e in greedy if e["gold_done"]]
    return {
        "completed": _rate(greedy, "gold_done"),
        "true": _rate(greedy, "true_success"),
        "stop_after_gold": _rate(done, "stop_after_gold"),
        "reads_after_gold": _rate(done, "reads_after_gold"),
        "p_stop_after_gold": _rate(done, "p_stop_after_gold"),
    }


def violation(episode):
    """Why the oracle rejects a reward-1 episode. Only a late verification alone is a class phase 1 flagged (F6, on
    both graders); a G_new rejection, an altered write, or both checks failing has no phase 1 relation."""
    if episode["new_reward"] != 1.0:
        return "G_new rejects"
    if episode["order_ok"] is False:
        return "order" if episode["no_altered"] else "order and altered"
    return "altered"


def h3_finds(runs):
    """Distinct (task, calls) reward-1 training trajectories the oracle marks wrong, and how many reward-1 episodes
    were left out because their G_new grading raised."""
    finds, skipped = {}, 0
    for run in runs:
        for e in run["episodes"]:
            if e["reward"] != 1.0 or e["true_success"] is not False:
                continue
            if e.get("oracle_error"):
                skipped += 1
                continue
            key = (e["task_id"], json.dumps(e["calls"], sort_keys=True))
            kind = violation(e)
            find = finds.setdefault(
                key,
                {
                    "task_id": e["task_id"],
                    "conditions": [],
                    "violation": kind,
                    "phase1_flagged": kind == "order",
                    "calls": e["calls"],
                },
            )
            if run["condition"] not in find["conditions"]:
                find["conditions"].append(run["condition"])
    return list(finds.values()), skipped


def analyze(folder, train_ids, held_ids):
    load = phase2._read_json
    runs = {(c, s): load(folder / "runs" / f"{c}_seed{s}.json") for c in CONDITIONS for s in SEEDS}
    evals = {(c, s): load(folder / "runs" / f"{c}_seed{s}_eval.json") for c in CONDITIONS for s in SEEDS}
    randoms = [load(folder / "runs" / f"random_seed{s}.json") for s in SEEDS]
    pending = sum(e["true_success"] is None for r in (*runs.values(), *evals.values()) for e in r["episodes"])
    if pending:
        raise ValueError(f"{pending} episodes have no true success: a G_old run was not filled under v1.0.1")
    per_seed = {key: run_measures(run) for key, run in runs.items()}
    for key, run in evals.items():
        held = [e for e in run["episodes"] if e["task_id"] in held_ids and not e["greedy"]]
        per_seed[key].update({"held_true": _rate(held, "true_success"), "held_extra_reads": _rate(held, "extra_reads")})
        per_seed[key].update({f"greedy_{k}": v for k, v in greedy_measures(run, train_ids).items()})
    names = [k for k in per_seed[("new", 0)] if k != "first_wrong"]
    table = {(c, n): bootstrap([per_seed[(c, s)][n] for s in SEEDS]) for c in CONDITIONS for n in names}
    for n in ("extra_reads", "true", "held_true", "held_extra_reads", "greedy_reads_after_gold"):
        table[("old-new", n)] = bootstrap([per_seed[("old", s)][n] - per_seed[("new", s)][n] for s in SEEDS])
    finds, skipped = h3_finds([runs[(c, s)] for c in ("old", "new") for s in SEEDS])
    recall = sum(f["phase1_flagged"] for f in finds) / len(finds) if finds else -1.0  # -1: no finds
    reads, true = table[("old-new", "extra_reads")], table[("old-new", "true")]
    gate = int(table[("new", "gate")][0] > 0.8)
    metrics = {
        "gate_true_success_new": table[("new", "gate")][0],
        "gate_held": gate,
        "h3_finds": len(finds),
        "h3_skipped_grader_errors": skipped,
        "h3_recall": recall,
        "h3_held": int(not finds or recall >= 0.9) if gate else -1,  # -1: not tested, the gate failed
        "h4_extra_read_diff": reads[0],
        "h4_extra_read_diff_low": reads[1],
        "h4_extra_read_diff_high": reads[2],
        "h4_true_diff": true[0],
        "h4_true_diff_low": true[1],
        "h4_true_diff_high": true[2],
        "h4_held": int(reads[2] < 0 and true[1] <= 0) if gate else -1,  # rule in amendment 6; -1: not tested
        "random_true_success": float(np.mean([_rate(r["episodes"][-LAST:], "true_success") for r in randoms])),
        "random_extra_reads": float(np.mean([_rate(r["episodes"][-LAST:], "extra_reads") for r in randoms])),
    }
    for c in CONDITIONS:
        metrics[f"true_success_{c}"] = table[(c, "true")][0]
        metrics[f"extra_reads_{c}"] = table[(c, "extra_reads")][0]
        metrics[f"converged_seeds_{c}"] = sum(per_seed[(c, s)]["true"] > 0.8 for s in SEEDS)
    return metrics, table, per_seed, finds


def _cell(interval):
    return "{:.3f} [{:.3f}, {:.3f}]".format(*interval)


def _outcome(value):
    return {1: "held", 0: "not held", -1: "not tested (the gate failed)"}[value]


def _hypothesis_lines(metrics, table, finds):
    h3 = metrics["h3_held"]
    no_finds = " No finds, so it holds trivially." if not finds and h3 == 1 else ""
    return [
        f"- Gate: G_new true success on episodes 1,701-2,000 = {metrics['gate_true_success_new']:.3f}; "
        f"{_outcome(metrics['gate_held'])} (needs > 0.8).",
        f"- H3: {metrics['h3_finds']} distinct oracle-wrong reward-1 trajectories "
        f"({metrics['h3_skipped_grader_errors']} episodes left out: their G_new grading raised); share in a class "
        f"phase 1 flagged {metrics['h3_recall']:.3f}; {_outcome(h3)} (needs >= 0.9).{no_finds}",
        f"- H4: old - new extra reads {_cell(table[('old-new', 'extra_reads')])} (needs the whole interval below 0); "
        f'true success {_cell(table[("old-new", "true")])} ("no gain": the interval does not lie entirely above 0); '
        f"{_outcome(metrics['h4_held'])}. Held-out, beside it: extra reads "
        f"{_cell(table[('old-new', 'held_extra_reads')])}, true success {_cell(table[('old-new', 'held_true')])}.",
    ]


def _counterfactual_lines(counter):
    names = list(counter)
    rows = sorted({k for table in counter.values() for row in table.values() for k in row})
    lines = ["| Run | tasks | " + " | ".join(names) + " |", "| --- | ---: |" + " --- |" * len(names)]
    shares = {}
    for key in rows:
        covered = [t for t in counter[names[0]] if key in counter[names[0]][t]]
        shares[key] = {n: float(np.mean([counter[n][t][key] == 1.0 for t in covered])) for n in names}
        lines.append(f"| {key} | {len(covered)} | " + " | ".join(f"{shares[key][n]:.3f}" for n in names) + " |")
    with_reads = [k for k in rows if k.startswith("gold_plus_") and k != "gold_plus_0_reads"]
    for n in names:
        gold = shares["gold_plus_0_reads"][n]
        lowest = min(shares[k][n] for k in with_reads) if with_reads else gold
        verdict = "tolerates them (the same pass rate)" if lowest == gold else "punishes them (a lower pass rate)"
        lines.append(f"\nG_{n} {verdict}: gold run {gold:.3f}, with extra reads as low as {lowest:.3f}.")
    lines.append(
        "\nA 0/1 verdict cannot reward the extra reads more than the gold run alone, so it can only tolerate or punish"
        " them; a preference for reads over STOP under a tolerating grader comes from learning, not from the grader."
        " The reads tested are each task's first extra reads in action order, not necessarily the policies' own."
    )
    return lines


def write_report(path, metrics, table, per_seed, finds, tuning, counter, train_ids, held_ids):
    measure_names = [n for c, n in table if c == "new"]
    seeds = sorted({s for _, s in per_seed})
    lines = [
        "# Phase 2 full sweep",
        "",
        "Open-loop linear softmax policy, REINFORCE, over real tau2 tool calls; reward is the grader's verdict. The",
        "oracle (true success) is ours. Exploitability within this action space only; nothing about LLM agents.",
        f"Learning rate {tuning['chosen_lr']}, tuned on retail only (mean true success {tuning['mean_true_success']}).",
        f"{len(seeds)} seeds per grader, {EPISODES} episodes, {len(train_ids)} train and {len(held_ids)} held-out",
        "banking tasks. Means over seeds with 95% bootstrap intervals; every seed is reported, converged or not.",
        "",
        "## By grader",
        "",
        "| Measure | " + " | ".join(CONDITIONS) + " | old - new |",
        "| --- |" + " --- |" * (len(CONDITIONS) + 1),
    ]
    for n in measure_names:
        diff = _cell(table[("old-new", n)]) if ("old-new", n) in table else ""
        lines.append(f"| {n} | " + " | ".join(_cell(table[(c, n)]) for c in CONDITIONS) + f" | {diff} |")
    lines += [
        "",
        "## Every seed",
        "",
        "| Grader | Seed | " + " | ".join(measure_names) + " | first wrong |",
        "| --- | --- |" + " --- |" * (len(measure_names) + 1),
    ]
    for (c, s), m in sorted(per_seed.items()):
        lines.append(f"| {c} | {s} | " + " | ".join(f"{m[n]:.3f}" for n in measure_names) + f" | {m['first_wrong']} |")
    lines += ["", "## Pre-registered hypotheses (G_proc is exploratory and in none of them)", ""]
    lines += _hypothesis_lines(metrics, table, finds)
    lines += ["", "### H3 finds, one by one", ""]
    lines += [
        f"- {'/'.join(f['conditions'])} {f['task_id']}: {f['violation']}, phase 1 flagged: {f['phase1_flagged']}"
        for f in finds
    ] or ["- none"]
    lines += [
        "",
        "## Exploratory (amendment 6, defined after the feasibility trial; not a hypothesis test)",
        "",
        "Greedy policy on the train tasks (rows `greedy_*` above): completion, STOP right after the gold steps, extra",
        "reads after them and the STOP probability there. Old - new post-completion reads: "
        + _cell(table[("old-new", "greedy_reads_after_gold")])
        + ".",
        "",
        "Each version's verdict on the train tasks' gold run followed by extra reads (share passing):",
        "",
    ]
    lines += _counterfactual_lines(counter)
    lines += [
        "",
        "Every grader has the same actions and the same 3 spare slots (horizon = gold length + 3): differences",
        "between graders are reward-induced, while how many reads fit after the task is done is set by the slots.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def check_split(folder):
    """The pre-registered pool and the 30/10 draw, checked before anything runs."""
    pool = phase2.pool_ids()
    if len(pool) != 62:
        raise ValueError(f"the pool has {len(pool)} tasks, not the 62 of amendment 5")
    train_ids, held_ids = phase2.ensure_split(folder, pool)
    if (train_ids, held_ids) != tuple(map(list, phase2.split(pool))):
        raise ValueError(f"{folder / 'split.json'} differs from the pre-registered draw")
    return train_ids, held_ids


def report(folder, out_path):
    """The analysis and report from the records already in the folder."""
    train_ids, held_ids = check_split(folder)
    metrics, table, per_seed, finds = analyze(folder, train_ids, held_ids)
    tuning = phase2._read_json(folder / "tuning" / "choice.json")
    counter = {name: phase2._read_json(folder / f"counterfactual_{name}.json") for name in ("old", "new")}
    write_report(folder / "report.md", metrics, table, per_seed, finds, tuning, counter, train_ids, held_ids)
    phase2._write_json(folder / "h3_finds.json", finds)
    phase2._write_json(out_path, metrics)
    return metrics


def run_full(out_path, old_python, new_python, processes):
    folder = out_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    train_ids, held_ids = check_split(folder)
    lr = tune(folder, new_python, processes)
    sweep(folder, old_python, new_python, lr, train_ids, held_ids, processes)
    return report(folder, out_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "counterfactual":
        counterfactuals(argv[2:], argv[1])
        return 0
    parser = argparse.ArgumentParser(prog="python -m preflight.full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-python", default=phase2.DEFAULT_OLD)
    parser.add_argument("--new-python", default=phase2.DEFAULT_NEW)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--report-only", action="store_true", help="analyse the records already in the folder")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if args.report_only:
        report(out.parent, out)
    else:
        run_full(out, args.old_python, args.new_python, args.processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
