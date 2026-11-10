"""Tests for the preflight command line interface."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import __main__ as cli


class CliTest(unittest.TestCase):
    def test_offline_flag_is_set(self):
        self.assertEqual(os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"), "True")

    def test_run_dispatches_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "nested" / "airline.json"
            with mock.patch("preflight.suite.run_suite", return_value={"domain": "airline"}) as run_suite:
                code = cli.main(
                    [
                        "run",
                        "--domain",
                        "airline",
                        "--relations",
                        "F1",
                        "--seed",
                        "3",
                        "--limit",
                        "2",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            run_suite.assert_called_once_with("airline", relations="F1", seed=3, limit=2)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"domain": "airline"})

    def test_compare_dispatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "report.md"
            with mock.patch("preflight.compare.compare") as compare:
                code = cli.main(["compare", "old", "new", "--out", str(out)])
            self.assertEqual(code, 0)
            compare.assert_called_once_with("old", "new", str(out))

    def test_unknown_relation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "result.json"
            with self.assertRaises(ValueError):
                cli.main(
                    [
                        "run",
                        "--domain",
                        "airline",
                        "--relations",
                        "NOPE",
                        "--limit",
                        "0",
                        "--out",
                        str(out),
                    ]
                )

    def test_run_limit_writes_a_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "airline.json"
            code = cli.main(
                [
                    "run",
                    "--domain",
                    "airline",
                    "--relations",
                    "F1",
                    "--seed",
                    "0",
                    "--limit",
                    "1",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["domain"], "airline")
            self.assertLessEqual(len(data["tasks"]), 1)


if __name__ == "__main__":
    unittest.main()
