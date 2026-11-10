"""Tests for the experiment runner using a fake tau2 interpreter."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The fake refuses to write a result unless the offline flag reached it, so the runner's
# environment propagation is what makes the experiment tests pass at all.
FAKE_PYTHON = """#!{python}
import json
import os
import sys
from pathlib import Path

flag = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
if flag != "True":
    sys.exit(f"LITELLM_LOCAL_MODEL_COST_MAP is {{flag!r}}, expected 'True'")

args = sys.argv[1:]
domain = args[args.index("--domain") + 1]
out = Path(args[args.index("--out") + 1])
out.parent.mkdir(parents=True, exist_ok=True)
variant = {
    "task_id": "t0",
    "relation": "F1",
    "detail": "empty run",
    "expected_pass": False,
    "inserted_in_gold": False,
    "outcome": "verdict",
    "reward": 0.0,
    "flagged": True,
}
task = {
    "id": "t0",
    "task_fingerprint": "task-" + domain,
    "gold_output_fingerprint": "gold-" + domain,
    "static": {"gold_changes_db": True},
}
payload = {
    "domain": domain,
    "seed": 0,
    "tasks": [task],
    "variants": [variant],
    "records_sha256": "sha-" + domain,
    "seconds": 0.0,
}
out.write_text(json.dumps(payload), encoding="utf-8")
"""


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake_python"
        # Only the shebang placeholder is substituted; the script's own braces must survive untouched.
        self.fake.write_text(FAKE_PYTHON.replace("{python}", sys.executable), encoding="utf-8")
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

    def test_subprocesses_receive_the_offline_flag(self):
        self.run_experiment()
        metrics = json.loads((self.out.parent / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["attempted_old"], 3)
        self.assertEqual(metrics["attempted_new"], 3)
        self.assertEqual(metrics["errors_old"], 0)
        self.assertEqual(metrics["errors_new"], 0)

    def test_two_runs_give_identical_metrics(self):
        self.run_experiment()
        first = json.loads(self.out.read_text(encoding="utf-8"))
        self.run_experiment()
        second = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
