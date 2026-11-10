"""Offline demo: a small airline run through the relation suite, printed step by step."""

import argparse
import json
import os

# tau2's LLM paths must never reach the network; set before the suite imports tau2.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__)
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--limit", type=int, default=5, help="tasks to run")
    parser.add_argument("--seed", type=int, default=0)
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
    if variant["db_diff"]:
        print(f"db fields that differ from the gold run: {', '.join(variant['db_diff'])}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from preflight.suite import run_suite

    result = run_suite(args.domain, seed=args.seed, limit=args.limit)
    _print_tasks(result)
    _print_variants(result)
    _print_example(result)
    print(f"records sha256 {result['records_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
