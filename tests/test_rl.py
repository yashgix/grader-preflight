"""Tests for the phase 2 learner helpers: grader cache, policy, split and training."""

import random
import unittest
from unittest import mock

import numpy as np

from preflight import rl
from preflight.mdp import Action
from preflight.rl import GradeCache, Policy, split, train
from preflight.tau2_adapter import Call


class _Task:
    def __init__(self, task_id="toy", payload=None):
        self.id = task_id
        self.payload = payload or {}

    def model_dump(self, mode="json"):
        return {"id": self.id, **self.payload}


class GradeCacheTest(unittest.TestCase):
    def setUp(self):
        self.task = _Task("task_1", {"x": 1})
        self.calls = [Call("read", {"account_id": "A1"})]

    def test_a_miss_calls_the_grader_once_and_a_hit_does_not(self):
        grader = mock.Mock(return_value=1.0)
        cache = GradeCache("banking_knowledge", grade=grader)
        self.assertEqual(cache.cache(self.task, self.calls), 1.0)
        self.assertEqual(cache.cache(self.task, list(self.calls)), 1.0)
        self.assertEqual(cache.misses, 1)
        grader.assert_called_once_with("banking_knowledge", self.task, self.calls)

    def test_a_grader_exception_is_an_error_cached_without_a_second_call(self):
        grader = mock.Mock(side_effect=RuntimeError("boom"))
        cache = GradeCache("banking_knowledge", grade=grader)
        self.assertIsNone(cache.cache(self.task, self.calls))
        self.assertIsNone(cache.cache(self.task, self.calls))
        self.assertEqual(cache.misses, 1)

    def test_different_tasks_with_the_same_calls_are_separate_entries(self):
        grader = mock.Mock(return_value=1.0)
        cache = GradeCache("banking_knowledge", grade=grader)
        cache.cache(_Task("a"), self.calls)
        cache.cache(_Task("b"), self.calls)
        self.assertEqual(cache.misses, 2)

    def test_two_definitions_of_one_task_id_are_separate_entries(self):
        grader = mock.Mock(return_value=1.0)
        cache = GradeCache("banking_knowledge", grade=grader)
        cache.cache(_Task("a", {"x": 1}), self.calls)
        cache.cache(_Task("a", {"x": 2}), self.calls)
        self.assertEqual(cache.misses, 2)


class PolicyTest(unittest.TestCase):
    def test_starts_uniform_with_zero_weights_and_no_bias(self):
        policy = Policy(2)
        rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        probabilities = policy.probs("t", [0, 1, 2], rows)
        np.testing.assert_allclose(probabilities, [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_equal(policy.w, np.zeros(2))
        self.assertEqual(policy.bias, {})

    def test_a_bias_lifts_its_action(self):
        policy = Policy(2)
        policy.bias[("t", 1)] = 2.0
        probabilities = policy.probs("t", [0, 1], [[0.0, 0.0], [0.0, 0.0]])
        self.assertGreater(probabilities[1], probabilities[0])

    def test_update_raises_the_chosen_actions_probability(self):
        rows = [[1.0, 0.0], [0.0, 1.0]]
        policy = Policy(2)
        before = policy.probs("t", [0, 1], rows)
        policy.update([("t", [0, 1], rows, 1)], 1.0)
        after = policy.probs("t", [0, 1], rows)
        self.assertGreater(after[1], before[1])
        self.assertLess(after[0], before[0])
        self.assertGreater(policy.bias[("t", 1)], 0.0)
        self.assertLess(policy.bias[("t", 0)], 0.0)

    def test_update_accepts_a_position_when_it_is_not_an_action_id(self):
        policy = Policy(1)
        policy.update([("t", [5, 6], [[1.0], [0.0]], 1)], 1.0)
        self.assertGreater(policy.bias[("t", 6)], 0.0)
        self.assertLess(policy.bias[("t", 5)], 0.0)

    def test_zero_advantage_leaves_the_policy_untouched(self):
        policy = Policy(2)
        rows = [[1.0, 0.0], [0.0, 1.0]]
        policy.update([("t", [0, 1], rows, 1)], 0.0)
        np.testing.assert_array_equal(policy.w, np.zeros(2))
        self.assertEqual(policy.bias, {})

    def test_an_empty_step_list_makes_no_change(self):
        policy = Policy(2)
        policy.update([], 1.0)
        np.testing.assert_array_equal(policy.w, np.zeros(2))
        self.assertEqual(policy.bias, {})


class _ToyTask:
    def __init__(self, task_id="toy"):
        self.id = task_id


def _win_actions():
    return [
        Action("gold", (Call("win", {}),), 0, is_write=False),
        Action("stop", (), None, is_write=False),
    ]


def _read_actions():
    return [
        Action("gold", (Call("win", {}),), 0, is_write=False),
        Action("read", (Call("look", {}),), None, is_write=False),
        Action("stop", (), None, is_write=False),
    ]


def _repeat_actions():
    return [
        Action("gold", (Call("look", {}),), 0, is_write=False),
        Action("read", (Call("look", {}),), None, is_write=False),
        Action("stop", (), None, is_write=False),
    ]


def _win_reward(task, calls):
    return 1.0 if any(call.name == "win" for call in calls) else 0.0


def _zero_reward(task, calls):
    return 0.0


def _error_reward(task, calls):
    return None


def _run_train(actions, steps, reward, episodes=20, seed=0, policy=None, task=None):
    task = task or _ToyTask()
    with (
        mock.patch("preflight.rl.task_actions", return_value=actions),
        mock.patch("preflight.rl.horizon", return_value=steps),
    ):
        return train("banking_knowledge", [task], seed=seed, episodes=episodes, reward_fn=reward, policy=policy)


class TrainTest(unittest.TestCase):
    def test_a_rewarded_action_gets_more_likely(self):
        policy, records = _run_train(_win_actions(), 1, _win_reward, episodes=400)
        self.assertEqual(len(records), 400)
        rows = [rl.features(_win_actions(), index, [], 0, 1) for index in (0, 1)]
        probability = policy.probs("toy", [0, 1], rows)[0]
        self.assertGreater(probability, 0.5)

    def test_records_carry_the_episode_summary(self):
        _, records = _run_train(_win_actions(), 1, _win_reward, episodes=10)
        for record in records:
            self.assertEqual(set(record), {"task_id", "made", "reward", "error", "extra_reads"})
            self.assertEqual(record["task_id"], "toy")
            self.assertFalse(record["error"])
            self.assertIn(record["reward"], (0.0, 1.0))
            self.assertEqual(record["extra_reads"], 0)
            self.assertIn(record["made"][-1], (0, 1))

    def test_a_grader_error_is_recorded_with_zero_reward(self):
        policy, records = _run_train(_win_actions(), 1, _error_reward, episodes=5)
        for record in records:
            self.assertTrue(record["error"])
            self.assertEqual(record["reward"], 0.0)
        np.testing.assert_array_equal(policy.w, np.zeros(rl.N_FEATURES))
        self.assertEqual(policy.bias, {})

    def test_zero_reward_leaves_the_weights_at_rest(self):
        policy, _ = _run_train(_win_actions(), 1, _zero_reward, episodes=10)
        np.testing.assert_array_equal(policy.w, np.zeros(rl.N_FEATURES))
        self.assertEqual(policy.bias, {})

    def test_a_read_outside_the_gold_run_counts_as_extra(self):
        policy = Policy(rl.N_FEATURES, lr=0.0)
        policy.bias[("toy", 1)] = 50.0
        _, records = _run_train(_read_actions(), 2, _win_reward, episodes=1, policy=policy)
        self.assertEqual(records[0]["made"][0], 1)
        self.assertEqual(records[0]["extra_reads"], 1)

    def test_a_read_that_repeats_a_gold_tool_is_not_extra(self):
        policy = Policy(rl.N_FEATURES, lr=0.0)
        policy.bias[("toy", 1)] = 50.0
        _, records = _run_train(_repeat_actions(), 2, _win_reward, episodes=1, policy=policy)
        self.assertEqual(records[0]["made"][0], 1)
        self.assertEqual(records[0]["extra_reads"], 0)

    def test_same_arguments_give_the_same_records(self):
        first_policy, first = _run_train(_win_actions(), 1, _win_reward, episodes=25, seed=3)
        second_policy, second = _run_train(_win_actions(), 1, _win_reward, episodes=25, seed=3)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(first_policy.w, second_policy.w)
        self.assertEqual(first_policy.bias, second_policy.bias)

    def test_no_tasks_gives_no_records(self):
        policy, records = train("banking_knowledge", [], seed=0, episodes=5, reward_fn=_win_reward)
        self.assertEqual(records, [])
        np.testing.assert_array_equal(policy.w, np.zeros(rl.N_FEATURES))


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.pool = [f"task_{index:03d}" for index in range(1, 63)]

    def test_split_draws_thirty_and_ten_without_overlap(self):
        train_ids, held_out = split(self.pool)
        self.assertEqual(len(train_ids), 30)
        self.assertEqual(len(held_out), 10)
        self.assertEqual(set(train_ids) & set(held_out), set())
        self.assertEqual(len(set(train_ids) | set(held_out)), 40)

    def test_split_matches_the_preregistered_seeded_draw(self):
        expected = sorted(self.pool)
        random.Random(20261110).shuffle(expected)
        self.assertEqual(split(self.pool), (expected[:30], expected[30:40]))

    def test_split_is_stable_and_does_not_mutate_its_input(self):
        original = list(self.pool)
        first = split(self.pool)
        self.assertEqual(split(list(reversed(self.pool))), first)
        self.assertEqual(self.pool, original)
