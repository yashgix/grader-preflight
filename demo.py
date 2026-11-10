"""Offline demo: phase 1 relations on a few tasks, or a short phase 2 smoke training run printed as metrics."""

import argparse
import json
import os
import tempfile
from pathlib import Path

# tau2's LLM paths must never reach the network; set before the suite imports tau2.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--domain", default="airline", help="phase 1 domain")
    parser.add_argument("--limit", type=int, default=5, help="tasks to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=("old", "new", "proc"), default="new", help="phase 2 condition")
    parser.add_argument("--episodes", type=int, default=200, help="phase 2 training episodes")
    return parser


def _skip_notes(static: dict) -> str:
    """The reasons a task got no variant for some relation."""
    notes = []
    skipped = static.get("identical_to_gold")
    if isinstance(skipped, list) and skipped:
        notes.append("identical to gold: " + ", ".join(sorted(str(item) for item in skipped)))
    if static.get("p5_not_inert") is True:
        notes.append("P5 not inert")
    written = static.get("no_effect_writes")
    if isinstance(written, int) and not isinstance(written, bool) and written:
        notes.append(f"no-effect writes: {written}")
    return "; ".join(notes)


def _outcome_text(variant: dict) -> str:
    if variant["outcome"] == "error":
        return f"error {variant['error']['type']}"
    return f"reward {variant['reward']:.0f}"


def _print_tasks(result: dict) -> None:
    print(f"domain {result['domain']}, tau2 {result['tau2_version']}, seed {result['seed']}")
    for task in result["tasks"]:
        static = task["static"]
        print(
            f"  {task['id']}: base {task['base_outcome']} reward {task['base_reward']}, "
            f"gold changes db={static.get('gold_changes_db')}, gold errors={static.get('gold_errors')}"
        )
        notes = _skip_notes(static)
        if notes:
            print(f"    no variant: {notes}")


def _print_variants(result: dict) -> None:
    grouped = {}
    for variant in result["variants"]:
        grouped.setdefault(variant["relation"], []).append(variant)
    if not grouped:
        print("no variants: no task had a gold run the grader accepts, or every relation was skipped")
        return
    for relation in sorted(grouped):
        variants = grouped[relation]
        errors = sum(1 for item in variants if item["outcome"] == "error")
        flags = sum(1 for item in variants if item["flagged"] is True)
        print(f"{relation}: {len(variants)} variants, {errors} errors, {flags} flags")
        for variant in variants:
            mark = "FLAG" if variant["flagged"] is True else "ok"
            print(f"  {mark:<4} {variant['task_id']} {_outcome_text(variant)}: {variant['detail']}")


def _print_example(result: dict) -> None:
    if not result["variants"]:
        return
    variant = result["variants"][0]
    print(f"example {variant['relation']} on {variant['task_id']}: {variant['detail']}")
    print(json.dumps(variant["calls"], indent=2, sort_keys=True))
    print(
        f"outcome {variant['outcome']}, reward {variant['reward']}, flagged {variant['flagged']}, "
        f"live call errors {variant['live_call_errors']}"
    )
    print(f"called checks {variant['checks']}, output fingerprint {variant['output_fingerprint']}")
    if variant["db_diff"]:
        print(f"db fields that differ from the gold run: {', '.join(variant['db_diff'])}")


def _phase1(args) -> int:
    from preflight.suite import run_suite

    result = run_suite(args.domain, seed=args.seed, limit=args.limit)
    _print_tasks(result)
    _print_variants(result)
    _print_example(result)
    print(f"records sha256 {result['records_sha256']}")
    return 0


def _phase2(args) -> int:
    """A short smoke training run on the first pool tasks; the metrics printed are the smoke report's own."""
    from preflight.phase2 import DOMAIN, pool_ids, train_run
    from preflight.report import smoke_metrics

    task_ids = pool_ids()[: args.limit]
    if not task_ids:
        print("no banking tasks in the pool")
        return 1
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "run.json"
        payload = train_run(args.condition, args.seed, task_ids, args.episodes, path, domain=DOMAIN)
    metrics = smoke_metrics([payload], episodes=args.episodes)
    print(f"phase 2 demo: condition {args.condition}, {len(task_ids)} tasks, {len(payload['episodes'])} episodes")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return _phase2(args) if args.phase == 2 else _phase1(args)


if __name__ == "__main__":
    raise SystemExit(main())
