"""Tests for the experiment runner using a fake tau2 interpreter."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_PYTHON = '''#!{python}
import json
import sys
from pathlib import Path

args = sys.argv[1:]
domain = args[args.index("--domain") + 1]
out = Path(args[args.index("--out") + 1])
out.parent.mkdir(parents=True, exist_ok=True)

variant = {{
    "task_id": "t0",
    "relation": "F1",
    "detail": "empty run",
    "expected_pass": False,
    "inserted_in_gold": False,
    "calls": [],
    "checks": {{}},
    "live_call_errors": [],
    "end_state": {{"variant": [None, None], "gold": [None, None]}},
    "outcome": "verdict",
    "reward": 0.0,
    "flagged": True,
    "error": None,
    "db_diff": None,
}}
task = {{
    "id": "t0",
    "reward_basis": ["DB"],
    "task_fingerprint": "task-" + domain,
    "gold_output_fingerprint": "gold-" + domain,
    "base_outcome": "verdict",
    "base_reward": 1.0,
    "static": {{"gold_changes_db": True}},
}}
payload = {{
    "domain": domain,
    "seed": 0,
    "tau2_version": "1.0.1",
    "provenance": {{"source": None, "python": "3.12"}},
    "tasks": [task],
    "variants": [variant],
    "records_sha256": "sha-" + domain,
    "seconds": 0.0,
}}
out.write_text(json.dumps(payload), encoding="utf-8")
'''


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake_python"
        self.fake.write_text(FAKE_PYTHON.format(python=sys.executable), encoding="utf-8")
        self.fake.chmod(0o755)
        self.out = self.root / "run" / "metrics.json"

    def tearDown(self):
        self.temporary.cleanup()

    def run_experiment(self):
        command = [
            sys.executable,
            "experiment.py",
            "--out",
            str(self.out),
            "--old-python",
            str(self.fake),
            "--new-python",
            str(self.fake),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_output_layout_is_under_the_out_folder(self):
        self.run_experiment()
        folder = self.out.parent
        expected = {
            "metrics.json",
            "report.md",
            "results/old/airline.json",
            "results/old/banking_knowledge.json",
            "results/old/retail.json",
            "results/new/airline.json",
            "results/new/banking_knowledge.json",
            "results/new/retail.json",
        }
        files = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(files, expected)
        metrics = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["control_passed"], 1)

    def test_two_runs_give_identical_metrics(self):
        self.run_experiment()
        first = json.loads(self.out.read_text(encoding="utf-8"))
        self.run_experiment()
        second = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
