"""Tests for the smoke phase 2 helpers: rewards, oracle fill, random episodes and report metrics."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import phase2, report
from preflight.mdp import Action
from preflight.rl import Policy
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


def _run(condition, seed=0, episodes=None, **episode):
    episodes = [_episode(**episode) for _ in range(10)] if episodes is None else episodes
    return {
        "condition": condition,
        "seed": seed,
        "task_ids": ["t0"],
        "episodes": episodes,
        "grader_calls": 2,
        "errors": sum(1 for item in episodes if item.get("error") or item.get("oracle_error")),
        "seconds": 1.0,
        "episodes_per_second": len(episodes),
    }


class ConditionRewardTest(unittest.TestCase):
    def test_plain_conditions_return_the_cached_reward(self):
        cache = mock.Mock()
        cache.cache.return_value = 1.0
        self.assertEqual(phase2._condition_reward("old", cache, object(), []), 1.0)
        self.assertEqual(phase2._condition_reward("new", cache, object(), []), 1.0)

    def test_proc_adds_the_order_check(self):
        cache = mock.Mock()
        cache.cache.return_value = 1.0
        with mock.patch("preflight.phase2.order_ok", return_value=True):
            self.assertEqual(phase2._condition_reward("proc", cache, object(), []), 1.0)
        with mock.patch("preflight.phase2.order_ok", return_value=False):
            self.assertEqual(phase2._condition_reward("proc", cache, object(), []), 0.0)

    def test_a_grader_error_is_none_not_zero(self):
        cache = mock.Mock()
        cache.cache.return_value = None
        self.assertIsNone(phase2._condition_reward("new", cache, object(), []))


class OracleSetTest(unittest.TestCase):
    def test_old_runs_leave_true_success_unknown(self):
        episode = {"calls": [{"name": "look", "arguments": {}, "requestor": "assistant"}], "made": [0]}
        cache = mock.Mock()
        cache.cache.return_value = 1.0
        phase2._set_oracle(episode, mock.Mock(id="t0"), [], "old", cache)
        self.assertIsNone(episode["true_success"])
        self.assertIsNone(episode["order_ok"])
        self.assertFalse(episode["oracle_error"])

    def test_new_runs_use_the_new_verdict_and_the_oracle_parts(self):
        episode = {"calls": [{"name": "look", "arguments": {}, "requestor": "assistant"}], "made": [0]}
        cache = mock.Mock()
        cache.cache.return_value = 1.0
        oracle = {"true_success": True, "order_ok": True, "no_altered": True}
        with mock.patch("preflight.phase2.oracle", return_value=oracle):
            phase2._set_oracle(episode, mock.Mock(id="t0"), [], "new", cache)
        self.assertTrue(episode["true_success"])
        self.assertTrue(episode["order_ok"])
        self.assertTrue(episode["no_altered"])
        self.assertFalse(episode["oracle_error"])

    def test_a_missing_new_verdict_is_an_oracle_error(self):
        episode = {"calls": [], "made": []}
        cache = mock.Mock()
        cache.cache.return_value = None
        phase2._set_oracle(episode, mock.Mock(id="t0"), [], "new", cache)
        self.assertFalse(episode["true_success"])
        self.assertIsNone(episode["order_ok"])
        self.assertTrue(episode["oracle_error"])


def _actions_for_reads():
    return [
        Action("gold", (Call("gold_read", {}),), 0),
        Action("read", (Call("extra_read", {}),), None),
        Action("read", (Call("gold_read", {}),), None),
        Action("stop", (), None),
    ]


class ExtraReadsTest(unittest.TestCase):
    def test_only_reads_outside_the_gold_run_are_extra(self):
        self.assertEqual(phase2._extra_reads(_actions_for_reads(), [1, 2]), 1)
        self.assertEqual(phase2._extra_reads(_actions_for_reads(), [2]), 0)


def _made_actions():
    return [Action("gold", (Call("a", {}),), 0), Action("stop", (), None)]


class RandomMadeTest(unittest.TestCase):
    def test_a_random_episode_is_deterministic_and_stops_at_stop(self):
        actions = _made_actions()
        first = phase2._random_made(random.Random(7), actions, 5)
        second = phase2._random_made(random.Random(7), actions, 5)
        self.assertEqual(first, second)
        if actions[first[-1]].kind == "stop":
            self.assertEqual(first[-1], 1)
        self.assertLessEqual(len(first), 5)


class PolicyPayloadTest(unittest.TestCase):
    def test_policy_payload_round_trips(self):
        import numpy as np

        policy = Policy(3)
        policy.w = np.array([1.0, -2.0, 0.5])
        policy.bias = {("t0", 0): 0.25, ("t0", 1): -0.5}
        restored = phase2._policy_from_payload(phase2._policy_payload(policy))
        np.testing.assert_array_equal(restored.w, policy.w)
        self.assertEqual(restored.bias, policy.bias)

    def test_missing_policy_is_an_error(self):
        with self.assertRaises(ValueError):
            phase2._policy_from_payload(None)


class SplitTest(unittest.TestCase):
    def test_split_is_written_once_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ids = [f"task_{index:03d}" for index in range(40)]
            train, held_out = phase2.ensure_split(folder, ids)
            self.assertEqual((len(train), len(held_out)), (30, 10))
            data = json.loads((folder / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(data["seed"], phase2.SPLIT_SEED)
            self.assertEqual(phase2.ensure_split(folder, ids), (train, held_out))


class RunPayloadTest(unittest.TestCase):
    def test_payload_counts_errors_and_rates(self):
        cache = mock.Mock(misses=3)
        records = [{"error": False, "oracle_error": False}, {"error": True, "oracle_error": False}]
        payload = phase2._run_payload("new", 0, "banking_knowledge", ["t0"], 2, records, cache, 2.0)
        self.assertEqual(payload["errors"], 1)
        self.assertEqual(payload["grader_calls"], 3)
        self.assertEqual(payload["episodes_per_second"], 1.0)


class OracleFillTest(unittest.TestCase):
    def test_oracle_fill_writes_new_reward_and_oracle_fields(self):
        task = mock.Mock(id="t0")
        calls = [{"name": "look", "arguments": {}, "requestor": "assistant"}]
        payload = {
            "task_ids": ["t0"],
            "episodes": [{"task_id": "t0", "made": [0], "calls": calls, "reward": 1.0, "error": False}],
        }
        cache = mock.Mock()
        cache.cache.return_value = 1.0
        oracle = {"true_success": True, "order_ok": True, "no_altered": True}
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("preflight.phase2._task_list", return_value=[task]),
            mock.patch("preflight.phase2.task_actions", return_value=[Action("gold", (Call("look", {}),), 0)]),
            mock.patch("preflight.phase2.GradeCache", return_value=cache),
            mock.patch("preflight.phase2.oracle", return_value=oracle),
        ):
            path = Path(temporary) / "run.json"
            phase2._write_json(path, payload)
            result = phase2.oracle_fill(path, path)
        episode = result["episodes"][0]
        self.assertEqual(episode["new_reward"], 1.0)
        self.assertTrue(episode["true_success"])
        self.assertTrue(episode["order_ok"])
        self.assertTrue(episode["no_altered"])
        self.assertEqual(result["oracle_grader_calls"], 1)

    def test_oracle_worker_dispatches_to_oracle_fill(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            with mock.patch("preflight.phase2.oracle_fill") as fill:
                phase2.main(
                    [
                        "worker",
                        "--mode",
                        "oracle",
                        "--out",
                        str(path),
                        "--in",
                        str(path),
                        "--task-ids",
                        "t0",
                    ]
                )
            fill.assert_called_once_with(str(path), str(path), phase2.DOMAIN)


class SmokeMetricsTest(unittest.TestCase):
    def test_smoke_metrics_are_numeric_and_reproducible(self):
        runs = [_run("old", 0), _run("new", 0)]
        random_runs = [_run("new", 9)]
        first = report.smoke_metrics(runs, random_runs, episodes=2000)
        second = report.smoke_metrics(runs, random_runs, episodes=2000)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 2000)
        self.assertEqual(first["grader_calls_old"], 2)
        self.assertEqual(first["grader_calls_new"], 2)
        for value in first.values():
            self.assertNotIsInstance(value, bool)
            self.assertIsInstance(value, (int, float))


if __name__ == "__main__":
    unittest.main()
