"""Focused tests for the smoke report and metrics, using synthetic runs and policies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import report
from preflight.mdp import Action
from preflight.tau2_adapter import Call


def _episode(reward=1.0, true_success=True, extra_reads=0, **extra):
    episode = {
        "task_id": "t0",
        "made": [0],
        "calls": [{"name": "look", "arguments": {}, "requestor": "assistant"}],
        "reward": reward,
        "error": False,
        "extra_reads": extra_reads,
        "new_reward": 1.0,
        "true_success": true_success,
        "order_ok": True,
        "no_altered": True,
        "oracle_error": False,
    }
    episode.update(extra)
    return episode


def _run(condition, seed=0, episodes=None, policy=None, **episode):
    episodes = [_episode(**episode) for _ in range(10)] if episodes is None else episodes
    run = {
        "condition": condition,
        "seed": seed,
        "task_ids": ["t0"],
        "episodes": episodes,
        "grader_calls": 2,
        "errors": sum(1 for item in episodes if item.get("error") or item.get("oracle_error")),
        "seconds": 1.0,
        "episodes_per_second": len(episodes),
    }
    if policy is not None:
        run["policy"] = policy
    return run


class SmokeMetricsTest(unittest.TestCase):
    def test_metrics_have_the_smoke_key_set_and_are_numeric(self):
        runs = [_run("old", 0), _run("new", 0)]
        random_runs = [_run("new", 9)]
        metrics = report.smoke_metrics(runs, random_runs, episodes=2000)
        self.assertEqual(
            set(metrics),
            {
                "episodes",
                "grader_calls_old",
                "grader_calls_new",
                "errors",
                "gate_true_success_new",
                "random_true_success",
                "true_success_old",
                "extra_read_rate_old",
                "extra_read_rate_new",
                "random_extra_read_rate",
                "distinct_trajectories",
            },
        )
        for value in metrics.values():
            self.assertNotIsInstance(value, bool)
            self.assertIsInstance(value, (int, float))

    def test_metrics_are_reproducible(self):
        runs = [_run("old", 0), _run("new", 0)]
        random_runs = [_run("new", 9)]
        self.assertEqual(
            report.smoke_metrics(runs, random_runs, episodes=2000),
            report.smoke_metrics(runs, random_runs, episodes=2000),
        )


class _Task:
    def __init__(self, task_id):
        self.id = task_id


class PerTaskProbabilitiesTest(unittest.TestCase):
    def test_final_policy_probabilities_are_beside_uniform(self):
        actions = [
            Action("gold", (Call("gold_read", {}),), 0),
            Action("read", (Call("extra_read", {}),), None),
            Action("stop", (), None),
        ]
        policy_payload = {
            "w": [0.0] * 8,
            "bias": [[["t0", 1], 10.0]],
        }
        run = _run("new", 0, policy=policy_payload)
        with (
            mock.patch("preflight.mdp.task_actions", return_value=actions),
            mock.patch("preflight.mdp.horizon", return_value=3),
            mock.patch("preflight.tau2_adapter.tasks", return_value=[_Task("t0")]),
        ):
            rows = report.per_task_probabilities([run], domain="banking_knowledge")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["task_id"], "t0")
        self.assertEqual(row["candidates"], 3)
        self.assertAlmostEqual(row["uniform"]["gold"], 1 / 3, places=6)
        self.assertAlmostEqual(row["uniform"]["stop"], 1 / 3, places=6)
        self.assertGreater(row["policy"]["extra_read"], row["uniform"]["extra_read"])


class WriteReportTest(unittest.TestCase):
    def test_report_contains_kept_apart_random_and_per_task_tables(self):
        runs = [_run("old", 0), _run("new", 0)]
        random_runs = [_run("new", 9)]
        per_task = [
            {
                "condition": "new",
                "seed": 0,
                "task_id": "t0",
                "candidates": 3,
                "uniform": {"gold": 1 / 3, "extra_read": 1 / 3, "stop": 1 / 3},
                "policy": {"gold": 0.2, "extra_read": 0.7, "stop": 0.1},
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.md"
            metrics = report.smoke_metrics(runs, random_runs, episodes=2000)
            report.write_report(path, runs, random_runs, metrics=metrics, per_task=per_task)
            text = path.read_text(encoding="utf-8")
        self.assertIn("## Kept-apart measures", text)
        self.assertIn("## Curves", text)
        self.assertIn("## Per-task final policy probabilities", text)
        self.assertIn("| random |", text)
        self.assertIn("t0", text)

    def test_metrics_are_written_as_json_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            report.write_metrics(path, {"a": True, "b": 2.5, "c": 3})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"a": 1, "b": 2.5, "c": 3})


if __name__ == "__main__":
    unittest.main()
