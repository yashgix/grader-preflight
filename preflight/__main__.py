"""Command line interface for the preflight suite and comparison."""

import argparse
import json
import os
from pathlib import Path

# tau2's LLM paths must never reach the network; set before the suite imports tau2.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(args) -> int:
    from preflight.suite import run_suite

    result = run_suite(args.domain, relations=args.relations, seed=args.seed, limit=args.limit)
    _write_json(Path(args.out), result)
    return 0


def _compare(args) -> int:
    from preflight.compare import compare

    compare(args.old_dir, args.new_dir, args.out)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m preflight")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run the relation suite on one domain")
    run_parser.add_argument("--domain", required=True)
    run_parser.add_argument("--relations", default=None, help="comma-separated relation ids")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--out", required=True)
    run_parser.set_defaults(handler=_run)

    compare_parser = commands.add_parser("compare", help="compare two result directories")
    compare_parser.add_argument("old_dir")
    compare_parser.add_argument("new_dir")
    compare_parser.add_argument("--out", required=True)
    compare_parser.set_defaults(handler=_compare)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
