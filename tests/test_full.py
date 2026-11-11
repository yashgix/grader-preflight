import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from preflight import full, phase2
from preflight.mdp import Action, features
from preflight.rl import Policy
from preflight.tau2_adapter import Call

CALL = {"name": "x", "arguments": {}, "requestor": "assistant"}


def episode(reward, true_success, calls=(CALL,), extra_reads=0, order_ok=True, no_altered=True, new_reward=None):
    return {
        "task_id": "t1",
        "reward": reward,
        "true_success": true_success,
        "calls": list(calls),
        "extra_reads": extra_reads,
        "order_ok": order_ok,
        "no_altered": no_altered,
        "new_reward": reward if new_reward is None else new_reward,
        "oracle_error": False,
    }


class BootstrapTest(unittest.TestCase):
    def test_constant_values_have_a_point_interval(self):
        self.assertEqual(full.bootstrap([0.5] * 10), (0.5, 0.5, 0.5))

    def test_interval_brackets_the_mean_and_is_deterministic(self):
        values = [0.0, 1.0, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.3, 0.7]
        mean, low, high = full.bootstrap(values)
        self.assertTrue(low < mean < high)
        self.assertEqual(full.bootstrap(values), (mean, low, high))

    def test_no_data_is_nan_not_zero(self):
        self.assertTrue(math.isnan(full._rate([], "true_success")))


class RunMeasuresTest(unittest.TestCase):
    def test_last_300_gate_window_and_first_wrong(self):
        episodes = [episode(0.0, False) for _ in range(1700)]
        episodes += [episode(1.0, True, extra_reads=2) for _ in range(300)]
        episodes += [episode(1.0, False, order_ok=False)] + [episode(0.0, False, calls=()) for _ in range(299)]
        m = full.run_measures({"episodes": episodes})
        self.assertEqual((m["gate"], m["first_wrong"], m["true"]), (1.0, 2001, 0.0))
        self.assertAlmostEqual(m["proxy"], 1 / 300)
        self.assertAlmostEqual(m["stop_without_acting"], 299 / 300)


class FindsTest(unittest.TestCase):
    def test_only_a_late_verification_alone_is_a_flagged_class(self):
        cases = {
            "late": episode(1.0, False, order_ok=False),
            "rejected": episode(1.0, False, order_ok=False, new_reward=0.0),
            "both": episode(1.0, False, order_ok=False, no_altered=False),
            "swapped": episode(1.0, False, no_altered=False),
        }
        for name, case in cases.items():
            case["calls"] = [{**CALL, "name": name}]
        errored = {**episode(1.0, False), "oracle_error": True}
        run = {"condition": "old", "episodes": [*cases.values(), cases["late"], errored, episode(1.0, True)]}
        finds, skipped = full.h3_finds([run, {**run, "condition": "new"}])
        kinds = {f["calls"][0]["name"]: (f["violation"], f["phase1_flagged"]) for f in finds}
        self.assertEqual(
            kinds,
            {
                "late": ("order", True),
                "rejected": ("G_new rejects", False),
                "both": ("order and altered", False),
                "swapped": ("altered", False),
            },
        )
        self.assertEqual(skipped, 2)
        self.assertTrue(all(f["conditions"] == ["old", "new"] for f in finds))


class GreedyMeasuresTest(unittest.TestCase):
    def test_only_greedy_episodes_on_the_given_tasks(self):
        base = {"true_success": True, "gold_done": True, "reads_after_gold": 3, "p_stop_after_gold": 0.1}
        run = {
            "episodes": [
                {**base, "task_id": "t1", "greedy": True, "stop_after_gold": False},
                {**base, "task_id": "t2", "greedy": True, "stop_after_gold": True, "reads_after_gold": 0},
                {**base, "task_id": "t1", "greedy": False, "stop_after_gold": True},
                {**base, "task_id": "t9", "greedy": True, "stop_after_gold": True},
            ]
        }
        m = full.greedy_measures(run, ["t1", "t2"])
        self.assertEqual((m["completed"], m["stop_after_gold"], m["reads_after_gold"]), (1.0, 0.5, 1.5))


class AfterGoldTest(unittest.TestCase):
    def setUp(self):
        self.actions = [Action("gold", (Call("g", {}),), 0), Action("read", (Call("r", {}),)), Action("stop", ())]
        self.task = type("Task", (), {"id": "t"})()
        self.policy = Policy(8)
        self.policy.w[7] = 2.0  # the step feature, so a wrong step index would change the probability

    def after(self, made, steps=4):
        with mock.patch("preflight.phase2.horizon", return_value=steps):
            return phase2.after_gold(self.policy, self.task, self.actions, made)

    def test_stop_or_reads_after_the_gold_steps(self):
        stopped, read = self.after([0, 2]), self.after([0, 1, 2])
        self.assertEqual((stopped["stop_after_gold"], stopped["reads_after_gold"]), (True, 0))
        self.assertEqual((read["stop_after_gold"], read["reads_after_gold"]), (False, 1))
        rows = np.array([features(self.actions, i, [0], 1, 4) for i in (1, 2)])
        logits = rows @ self.policy.w
        self.assertAlmostEqual(stopped["p_stop_after_gold"], float(np.exp(logits[1]) / np.exp(logits).sum()))

    def test_no_choice_left_or_gold_never_done_has_no_measure(self):
        last = self.after([1, 0], steps=2)
        self.assertEqual((last["gold_done"], last["stop_after_gold"], last["p_stop_after_gold"]), (True, None, None))
        self.assertFalse(self.after([1, 2])["gold_done"])


def write_runs(folder, true_by_condition, reads_by_condition, held_id):
    runs = folder / "runs"
    runs.mkdir(parents=True)
    greedy = {
        "greedy": True,
        "gold_done": True,
        "stop_after_gold": True,
        "reads_after_gold": 0,
        "p_stop_after_gold": 0.9,
    }
    for condition in full.CONDITIONS:
        true, reads = true_by_condition[condition], reads_by_condition[condition]
        for seed in full.SEEDS:
            train = [episode(1.0 if true else 0.0, true, extra_reads=reads) for _ in range(2000)]
            held = {**episode(1.0, true, extra_reads=reads), "task_id": held_id, "greedy": False}
            evals = [{**episode(1.0, true), **greedy}, held]
            phase2._write_json(runs / f"{condition}_seed{seed}.json", {"condition": condition, "episodes": train})
            phase2._write_json(runs / f"{condition}_seed{seed}_eval.json", {"condition": condition, "episodes": evals})
    for seed in full.SEEDS:
        phase2._write_json(runs / f"random_seed{seed}.json", {"episodes": [episode(0.0, False)]})


class AnalyzeTest(unittest.TestCase):
    def analyze(self, true_by_condition, reads_by_condition, edit=None):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_runs(folder, true_by_condition, reads_by_condition, "h1")
            if edit:
                edit(folder)
            return full.analyze(folder, ["t1"], ["h1"])[0]

    def test_h4_holds_with_fewer_reads_and_no_gain(self):
        metrics = self.analyze({"old": True, "new": True, "proc": True}, {"old": 0, "new": 3, "proc": 3})
        self.assertEqual((metrics["gate_held"], metrics["h4_held"], metrics["h3_held"]), (1, 1, 1))
        self.assertEqual((metrics["h3_finds"], metrics["h3_recall"], metrics["converged_seeds_old"]), (0, -1.0, 10))

    def test_h4_fails_without_fewer_reads(self):
        metrics = self.analyze({"old": True, "new": True, "proc": True}, {"old": 3, "new": 3, "proc": 3})
        self.assertEqual(metrics["h4_held"], 0)

    def test_a_failed_gate_leaves_h3_and_h4_untested(self):
        metrics = self.analyze({"old": False, "new": False, "proc": False}, {"old": 0, "new": 3, "proc": 3})
        self.assertEqual((metrics["gate_held"], metrics["h3_held"], metrics["h4_held"]), (0, -1, -1))

    def test_an_unfilled_old_run_stops_the_analysis(self):
        def unfill(folder):
            path = folder / "runs" / "old_seed0.json"
            run = json.loads(path.read_text())
            run["episodes"][0]["true_success"] = None
            path.write_text(json.dumps(run))

        with self.assertRaises(ValueError):
            self.analyze({"old": True, "new": True, "proc": True}, {"old": 0, "new": 3, "proc": 3}, unfill)


class EvaluateRunTest(unittest.TestCase):
    def test_evaluation_never_learns_and_repeats_exactly(self):
        task_id = phase2.pool_ids()[0]
        with tempfile.TemporaryDirectory() as temporary:
            trained = Path(temporary) / "train.json"
            phase2.train_run("new", 0, [task_id], 5, trained)
            with mock.patch.object(Policy, "update", side_effect=AssertionError("evaluation must not learn")):
                first = phase2.evaluate_run("new", trained, [task_id], 3, Path(temporary) / "a.json")
                second = phase2.evaluate_run("new", trained, [task_id], 3, Path(temporary) / "b.json")
        self.assertEqual([e["greedy"] for e in first["episodes"]], [True, False, False, False])
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertTrue(all(e["new_reward"] in (0.0, 1.0) for e in first["episodes"]))


class SplitTest(unittest.TestCase):
    def test_a_changed_split_file_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            train, held = full.check_split(folder)
            self.assertEqual((len(train), len(held)), (30, 10))
            phase2._write_json(folder / "split.json", {"seed": 0, "train": held + train[:20], "held_out": train[20:]})
            with self.assertRaises(ValueError):
                full.check_split(folder)


if __name__ == "__main__":
    unittest.main()
