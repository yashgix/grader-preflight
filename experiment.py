"""Run the preflight experiments: phase 1 relations, phase 2 (smoke or full sweep), or phase 3."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

DOMAINS = ("banking_knowledge", "airline", "retail")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OLD = "/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python"
DEFAULT_NEW = "/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python"


def _environment() -> dict:
    env = os.environ.copy()
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _check_result(domain: str, path: Path) -> None:
    """A side that wrote no readable result for a domain stops the run instead of shrinking the comparison."""
    if not path.is_file():
        raise RuntimeError(f"no result file for {domain}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"unreadable result file for {domain}: {path}") from exc
    if not isinstance(data, dict) or data.get("domain") != domain:
        raise RuntimeError(f"result file for {domain} records another domain: {path}")


def _run_side(python: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        path = out_dir / f"{domain}.json"
        path.unlink(missing_ok=True)
        subprocess.run(
            [python, "-B", "-m", "preflight", "run", "--domain", domain, "--seed", "0", "--out", str(path)],
            cwd=REPO_ROOT,
            env=_environment(),
            check=True,
        )
        _check_result(domain, path)


def run(out_path: Path, old_python: str, new_python: str) -> dict:
    """Phase 1: both graders over every domain, then the case-by-case comparison."""
    from preflight.compare import compare

    out_path = Path(out_path).resolve()
    folder = out_path.parent
    old_dir = folder / "results" / "old"
    new_dir = folder / "results" / "new"
    _run_side(old_python, old_dir)
    _run_side(new_python, new_dir)
    compare(old_dir, new_dir, folder / "report.md")
    metrics_path = folder / "metrics.json"
    if out_path != metrics_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.replace(out_path)
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_smoke(
    out_path,
    old_python: str = DEFAULT_OLD,
    new_python: str = DEFAULT_NEW,
    episodes: int | None = None,
    seeds=None,
    processes: int = 4,
) -> dict:
    """Phase 2 smoke run: five train tasks, a few seeds, G_old and G_new."""
    from preflight.phase2 import run_smoke as phase2_run_smoke

    kwargs = {
        "out_path": Path(out_path),
        "old_python": old_python,
        "new_python": new_python,
        "processes": processes,
    }
    if episodes is not None:
        kwargs["episodes"] = episodes
    if seeds is not None:
        kwargs["seeds"] = seeds
    return phase2_run_smoke(**kwargs)


def _parse_seeds(value):
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiment.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--old-python", default=DEFAULT_OLD)
    parser.add_argument("--new-python", default=DEFAULT_NEW)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="phase 2: the full 10-seed sweep instead of the smoke")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.out)
    if args.phase == 1:
        run(out, args.old_python, args.new_python)
        return 0
    if args.full:
        if args.episodes is not None or args.seeds is not None:
            raise SystemExit("--full uses the pre-registered seeds and episodes; drop --episodes and --seeds")
        from preflight.full import run_full

        run_full(out.resolve(), args.old_python, args.new_python, args.processes)
        return 0
    if args.phase == 3:
        from preflight.phase3 import run_phase3

        run_phase3(out.resolve(), args.old_python, args.new_python, args.processes)
        return 0
    seeds = _parse_seeds(args.seeds)
    run_smoke(
        out,
        old_python=args.old_python,
        new_python=args.new_python,
        episodes=args.episodes,
        seeds=seeds,
        processes=args.processes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
