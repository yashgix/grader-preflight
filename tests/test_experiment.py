"""Tests for the experiment runner: phase 1 compatibility and smoke dispatch."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiment

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_PHASE1 = """#!{python}
import json
import os
import sys
from pathlib import Path

flag = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
if flag != "True":
    sys.exit(f"LITELLM_LOCAL_MODEL_COST_MAP is {flag!r}, expected 'True'")

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
        self.fake.write_text(FAKE_PHASE1.replace("{python}", sys.executable), encoding="utf-8")
        self.fake.chmod(0o755)
        self.out = self.root / "run" / "metrics.json"

    def tearDown(self):
        self.temporary.cleanup()

    def run_phase1(self):
        command = [
            sys.executable,
            "experiment.py",
            "--phase",
            "1",
            "--out",
            str(self.out),
            "--old-python",
            str(self.fake),
            "--new-python",
            str(self.fake),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase1_output_layout_is_under_the_out_folder(self):
        self.run_phase1()
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

    def test_phase1_subprocesses_receive_the_offline_flag(self):
        self.run_phase1()
        metrics = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["attempted_old"], 3)
        self.assertEqual(metrics["attempted_new"], 3)
        self.assertEqual(metrics["errors_old"], 0)
        self.assertEqual(metrics["errors_new"], 0)

    def test_phase1_two_runs_give_identical_metrics(self):
        self.run_phase1()
        first = json.loads(self.out.read_text(encoding="utf-8"))
        self.run_phase1()
        second = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(first, second)

    def test_phase1_dispatches_through_run(self):
        with mock.patch("experiment.run") as run:
            code = experiment.main(["--phase", "1", "--out", str(self.out)])
        self.assertEqual(code, 0)
        run.assert_called_once_with(Path(self.out), experiment.DEFAULT_OLD, experiment.DEFAULT_NEW)

    def test_default_is_the_phase2_smoke_run(self):
        with mock.patch("preflight.phase2.run_smoke", return_value={"episodes": 1}) as smoke:
            code = experiment.main(
                [
                    "--out",
                    str(self.out),
                    "--old-python",
                    "oldpy",
                    "--new-python",
                    "newpy",
                    "--seeds",
                    "0,2",
                    "--episodes",
                    "10",
                    "--processes",
                    "1",
                ]
            )
        self.assertEqual(code, 0)
        smoke.assert_called_once_with(
            out_path=Path(self.out),
            old_python="oldpy",
            new_python="newpy",
            episodes=10,
            seeds=(0, 2),
            processes=1,
        )

    def test_full_flag_is_not_offered(self):
        with mock.patch("preflight.phase2.run_smoke") as smoke:
            with self.assertRaises(SystemExit):
                experiment.main(["--out", str(self.out), "--full"])
        smoke.assert_not_called()

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(SystemExit):
            experiment.main(["--out", str(self.out), "--phase", "3"])

    def test_environment_sets_the_offline_flag(self):
        self.assertEqual(experiment._environment()["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
        self.assertEqual(os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"), "True")


if __name__ == "__main__":
    unittest.main()
