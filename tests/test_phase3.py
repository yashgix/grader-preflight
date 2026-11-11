"""Tests for the phase 3 synthetic experiment: file shapes, simulation selection, flags and metrics."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import phase3, trajectories
from preflight.relations import build_variants, f6_late_verification, p3_respelled_numbers
from preflight.tau2_adapter import Call, gold_calls, tasks

BANK, AIRLINE = phase3.BANK, phase3.AIRLINE
BASE_RECORD_KEYS = {
    "id",
    "task_id",
    "trial",
    "domain",
    "git_commit",
    "task_matches",
    "replay_class",
    "reward",
    "recorded_reward",
    "mismatched",
    "p1",
    "p3",
    "diff_tables",
    "n_calls",
    "nongold_call",
    "schema_floats",
    "numeric_pair",
    "late_verification",
}
METRIC_KEYS = {
    "runs_banking",
    "runs_airline",
    "full_replay_new",
    "incompatible_old",
    "incompatible_new",
    "events",
    "explained",
    "recall",
    "p1_trigger_runs",
    "p3_trigger_runs",
    "trimmed_equal_gold",
    "schema_float_runs",
    "inconsistencies",
    "airline_control_events",
    "late_log_runs",
    "errors",
}


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _analysis(events=3, explained=2, recall=2 / 3, schema_float_runs=4, inconsistencies=1, control_events=0):
    return {
        "h5": {"events": events, "explained": explained, "recall": recall},
        "h7": {
            "airline": {
                "schema_float_runs": schema_float_runs,
                "inconsistencies": inconsistencies,
                "control_events": control_events,
            }
        },
    }


def _old_records():
    return [
        {"id": "a", "domain": BANK, "replay_class": "full"},
        {"id": "b", "domain": BANK, "replay_class": "incompatible", "error": {"type": "ValueError"}},
        {"id": "c", "domain": AIRLINE, "replay_class": "strict"},
    ]


def _new_records():
    return [
        {"id": "a", "domain": BANK, "task_id": "t1", "trial": 0, "replay_class": "full"},
        {"id": "a1", "domain": BANK, "task_id": "t1", "trial": 1, "replay_class": "full"},
        {
            "id": "f",
            "domain": BANK,
            "task_id": "t1",
            "trial": 4,
            "late_verification": {"log_after_first_write": True},
        },
        {"id": "b", "domain": BANK, "task_id": "t2", "trial": 0, "replay_class": "full", "p1": True},
        {"id": "c", "domain": AIRLINE, "replay_class": "lenient", "p3": True},
    ]


class IntegerParameterTest(unittest.TestCase):
    def test_only_integer_schemas_count(self):
        self.assertTrue(phase3._integer_parameter({"type": "integer"}))
        self.assertTrue(phase3._integer_parameter({"type": ["integer", "null"]}))
        self.assertFalse(phase3._integer_parameter({"type": "number"}))
        self.assertFalse(phase3._integer_parameter({"type": "string"}))
        self.assertFalse(phase3._integer_parameter({"type": "boolean"}))
        self.assertFalse(phase3._integer_parameter(None))


class FloatCallTest(unittest.TestCase):
    def test_whole_numbers_of_an_integer_parameter_become_floats(self):
        schema = {"properties": {"amount": {"type": "integer"}, "flag": {"type": "integer"}}}
        call = Call("pay", {"amount": 25, "note": "x", "flag": True})
        with (
            mock.patch.object(trajectories, "_tool_for", return_value=object()),
            mock.patch.object(trajectories, "_schema", return_value=schema),
        ):
            changed = phase3._float_call(AIRLINE, call)
        self.assertEqual(changed, Call("pay", {"amount": 25.0, "note": "x", "flag": True}))
        self.assertIsInstance(changed.arguments["amount"], float)
        self.assertIs(changed.arguments["flag"], True)

    def test_a_wrapper_rewrites_only_the_inner_value(self):
        inner = {"amount": 25, "note": "keep"}
        call = Call(
            "call_discoverable_agent_tool",
            {"agent_tool_name": "pay", "arguments": json.dumps(inner, sort_keys=True)},
        )
        schema = {"properties": {"amount": {"type": "integer"}}}
        with (
            mock.patch.object(trajectories, "_tool_for", return_value=object()),
            mock.patch.object(trajectories, "_schema", return_value=schema),
        ):
            changed = phase3._float_call(BANK, call)
        parsed = json.loads(changed.arguments["arguments"])
        self.assertEqual(parsed, {"amount": 25.0, "note": "keep"})
        self.assertIsInstance(parsed["amount"], float)
        self.assertEqual(json.loads(call.arguments["arguments"])["amount"], 25)

    def test_no_tool_and_no_change_leave_the_call_alone(self):
        call = Call("pay", {"amount": 25})
        with mock.patch.object(trajectories, "_tool_for", return_value=None):
            self.assertEqual(phase3._float_call(AIRLINE, call), call)
        with (
            mock.patch.object(trajectories, "_tool_for", return_value=object()),
            mock.patch.object(trajectories, "_schema", return_value={"properties": {}}),
        ):
            self.assertEqual(phase3._float_call(AIRLINE, call), call)


class PoolTest(unittest.TestCase):
    def test_the_pool_is_the_preregistered_sixty_two_tasks(self):
        ids = phase3.pool_ids()
        self.assertEqual(len(ids), 62)
        self.assertEqual(ids[0], "task_001")
        self.assertEqual(ids[-1], "task_102")


class WriteResultsTest(unittest.TestCase):
    def test_a_results_file_has_tau2_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "airline.json"
            phase3._write_results(path, AIRLINE, [{"id": "t0"}], [{"id": "airline:t0:0"}])
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"info", "tasks", "simulations"})
        self.assertEqual(payload["info"]["environment_info"]["domain_name"], AIRLINE)
        self.assertIsNone(payload["info"]["git_commit"])
        self.assertEqual(payload["tasks"], [{"id": "t0"}])
        self.assertEqual(payload["simulations"], [{"id": "airline:t0:0"}])


class BankingFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        known = {task.id for task in tasks(BANK)}
        cls.task_id = next((task_id for task_id in phase3.pool_ids() if task_id in known), None)
        if cls.task_id is None:
            raise unittest.SkipTest("no pool task in this environment")
        cls.task = next(task for task in tasks(BANK) if task.id == cls.task_id)

    def build(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f"{BANK}.json"
            with mock.patch.object(phase3, "pool_ids", return_value=[self.task_id]):
                phase3._banking_file(path)
            return json.loads(path.read_text(encoding="utf-8"))

    def test_trials_follow_the_generators(self):
        payload = self.build()
        self.assertEqual([task["id"] for task in payload["tasks"]], [self.task_id])
        expected = {0, 1}
        variants, _ = build_variants("P1", BANK, self.task, random.Random(f"0:{self.task.id}:P1"))
        if any(variant.detail.startswith("extra agent read") for variant in variants):
            expected.add(2)
        if p3_respelled_numbers(BANK, self.task, random.Random(f"0:{self.task.id}:P3")):
            expected.add(3)
        if f6_late_verification(BANK, self.task, random.Random(f"0:{self.task.id}:F6")):
            expected.add(4)
        trials = [simulation["trial"] for simulation in payload["simulations"]]
        self.assertEqual(trials, sorted(expected))

    def test_the_trimmed_copy_only_drops_tool_ids_and_errors(self):
        payload = self.build()
        gold, trimmed = payload["simulations"][0], payload["simulations"][1]
        self.assertEqual((gold["trial"], trimmed["trial"]), (0, 1))
        self.assertEqual(gold["id"], f"{BANK}:{self.task_id}:0")
        self.assertEqual(trimmed["id"], f"{BANK}:{self.task_id}:1")
        stripped = [
            {key: value for key, value in message.items() if not (message["role"] == "tool" and key in ("id", "error"))}
            for message in gold["messages"]
        ]
        self.assertEqual(stripped, trimmed["messages"])
        self.assertTrue(all("id" in message and "error" in message for message in gold["messages"] if message["role"] == "tool"))
        self.assertTrue(all("id" not in message and "error" not in message for message in trimmed["messages"] if message["role"] == "tool"))

    def test_the_late_verification_variant_is_flagged(self):
        payload = self.build()
        late = next((simulation for simulation in payload["simulations"] if simulation["trial"] == 4), None)
        if late is None:
            self.skipTest("this task has no late-verification variant")
        run = {"id": late["id"], "task_id": self.task_id, "trial": 4, "domain": BANK, "messages": late["messages"]}
        self.assertTrue(trajectories.late_verification(self.task, run)["log_after_first_write"])


class AirlineFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = next(
            (task for task in tasks(AIRLINE) if phase3.float_calls(AIRLINE, gold_calls(task)) != gold_calls(task)),
            None,
        )

    def test_integer_schema_values_become_floats(self):
        if self.task is None:
            self.skipTest("no airline task with an integer-schema argument")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f"{AIRLINE}.json"
            with mock.patch.object(phase3, "tasks", return_value=[self.task]):
                phase3._airline_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([task["id"] for task in payload["tasks"]], [self.task.id])
        gold, respelled = payload["simulations"]
        self.assertEqual((gold["trial"], respelled["trial"]), (0, 1))
        self.assertEqual(trajectories.calls(gold["messages"]), gold_calls(self.task))
        self.assertEqual(trajectories.calls(respelled["messages"]), phase3.float_calls(AIRLINE, gold_calls(self.task)))
        self.assertNotEqual(gold["messages"], respelled["messages"])
        values = [value for call in trajectories.calls(respelled["messages"]) for value in call.arguments.values()]
        self.assertTrue(any(isinstance(value, float) and value.is_integer() for value in values))


class MetricsTest(unittest.TestCase):
    def test_every_metric_is_counted_once(self):
        metrics = phase3._metrics(_old_records(), _new_records(), _analysis())
        self.assertEqual(set(metrics), METRIC_KEYS)
        self.assertEqual(metrics["runs_banking"], 4)
        self.assertEqual(metrics["runs_airline"], 1)
        self.assertEqual(metrics["full_replay_new"], 3)
        self.assertEqual(metrics["incompatible_old"], 1)
        self.assertEqual(metrics["incompatible_new"], 0)
        self.assertEqual((metrics["events"], metrics["explained"]), (3, 2))
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertEqual((metrics["p1_trigger_runs"], metrics["p3_trigger_runs"]), (1, 1))
        self.assertEqual(metrics["trimmed_equal_gold"], 1)
        self.assertEqual(
            (
                metrics["schema_float_runs"],
                metrics["inconsistencies"],
                metrics["airline_control_events"],
            ),
            (4, 1, 0),
        )
        self.assertEqual(metrics["late_log_runs"], 1)
        self.assertEqual(metrics["errors"], 1)
        for value in metrics.values():
            self.assertNotIsInstance(value, bool)
            self.assertIsInstance(value, (int, float))

    def test_the_same_records_give_the_same_metrics(self):
        first = phase3._metrics(_old_records(), _new_records(), _analysis())
        second = phase3._metrics(_old_records(), _new_records(), _analysis())
        self.assertEqual(first, second)

    def test_no_events_leaves_recall_at_zero(self):
        metrics = phase3._metrics([], _new_records(), _analysis(events=0, explained=0, recall=None))
        self.assertEqual((metrics["events"], metrics["explained"], metrics["recall"]), (0, 0, 0.0))

    def test_trimmed_copies_that_differ_are_not_counted(self):
        new_records = _new_records()
        new_records[1] = {**new_records[1], "replay_class": "strict"}
        self.assertEqual(phase3._trimmed_equal_gold(new_records), 0)

    def test_late_logs_count_only_the_f6_trial(self):
        records = [
            {"domain": BANK, "trial": 4, "late_verification": {"log_after_first_write": True}},
            {"domain": BANK, "trial": 3, "late_verification": {"log_after_first_write": True}},
            {"domain": BANK, "trial": 4, "late_verification": {"log_after_first_write": False}},
            {"domain": BANK, "trial": 4, "late_verification": None},
            {"domain": AIRLINE, "trial": 4, "late_verification": {"log_after_first_write": True}},
        ]
        self.assertEqual(phase3._late_log_runs(records), 1)


class FlagsTest(unittest.TestCase):
    def test_only_losing_flags_become_task_ids(self):
        records = [
            {"domain": BANK, "task_id": "t1", "p1": True, "reward": 0.0},
            {"domain": BANK, "task_id": "t2", "p1": True, "reward": 1.0},
            {"domain": BANK, "task_id": "t3", "p3": True, "reward": 0.0},
            {"domain": AIRLINE, "task_id": "t4", "p1": True, "reward": 0.0},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            regrade = folder / "regrade"
            regrade.mkdir()
            _write_jsonl(regrade / f"{BANK}.old.jsonl", records)
            flags = phase3._write_flags(regrade, folder / "flags.json")
            stored = json.loads((folder / "flags.json").read_text(encoding="utf-8"))
        self.assertEqual(flags, {"P1": ["t1"], "P3": ["t3"]})
        self.assertEqual(stored, flags)


class BuildSyntheticTest(unittest.TestCase):
    def test_both_domains_are_built_and_regraded(self):
        by_id = {task.id: task for task in tasks(BANK)}
        pool = [task_id for task_id in phase3.pool_ids() if task_id in by_id]
        if not pool:
            self.skipTest("no pool task in this environment")
        bank_task = min((by_id[task_id] for task_id in pool), key=lambda task: len(gold_calls(task)))
        airline_candidates = [task for task in tasks(AIRLINE) if gold_calls(task)]
        airline_task = min(airline_candidates, key=lambda task: len(gold_calls(task)))

        def selected(domain):
            return [bank_task] if domain == BANK else [airline_task]

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with (
                mock.patch.object(phase3, "pool_ids", return_value=[bank_task.id]),
                mock.patch.object(phase3, "tasks", side_effect=selected),
            ):
                phase3.build_synthetic(folder)
            files = sorted(path.name for path in (folder / "files").glob("*.json"))
            regraded = sorted((folder / "regrade").glob("*.jsonl"))
            self.assertEqual(files, sorted([f"{AIRLINE}.json", f"{BANK}.json"]))
            self.assertEqual(len(regraded), 2)
            for path in regraded:
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                self.assertTrue(records)
                for record in records:
                    self.assertTrue(BASE_RECORD_KEYS <= set(record) <= BASE_RECORD_KEYS | {"error"})
                    self.assertIsNotNone(record["task_id"])


if __name__ == "__main__":
    unittest.main()
