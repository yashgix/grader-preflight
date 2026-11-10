"""Tests for the suite helpers and a small live run."""

import hashlib
import json
import unittest
from unittest import mock

from preflight.relations import Variant
from preflight.suite import (
    _db_diff,
    _normalise_numbers,
    _records_sha256,
    _tau2_version,
    _variant_record,
    gold_fingerprint,
    output_fingerprint,
    run_suite,
    task_fingerprint,
)
from preflight.tau2_adapter import Call, tasks


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class FingerprintTest(unittest.TestCase):
    def test_numbers_are_normalised_but_booleans_are_not(self):
        self.assertEqual(_normalise_numbers(25), 25.0)
        self.assertIs(_normalise_numbers(True), True)
        self.assertEqual(_normalise_numbers({"a": [1, False]}), {"a": [1.0, False]})

    def test_output_fingerprint_ignores_int_float_spelling(self):
        self.assertEqual(
            output_fingerprint(['{"a": 25, "b": true}']),
            output_fingerprint(['{"a": 25.0, "b": true}']),
        )
        self.assertNotEqual(output_fingerprint(["true"]), output_fingerprint(["1"]))

    def test_task_and_gold_fingerprints_are_sha256(self):
        task = tasks("airline")[0]
        self.assertEqual(len(task_fingerprint(task)), 64)
        self.assertEqual(len(gold_fingerprint(task)), 64)

    def test_task_fingerprint_exact_value(self):
        class _Task:
            def model_dump(self, mode="json"):
                return {"z": 1, "a": {"nested": [2, True]}}

        expected = hashlib.sha256(
            _canonical_json({"z": 1, "a": {"nested": [2, True]}}).encode("utf-8")
        ).hexdigest()
        self.assertEqual(task_fingerprint(_Task()), expected)

    def test_gold_fingerprint_exact_value(self):
        calls = [Call("b", {"x": 1}), Call("a", {"y": 2})]
        expected = hashlib.sha256(
            _canonical_json(
                [
                    {"name": "b", "arguments": {"x": 1}, "requestor": "assistant"},
                    {"name": "a", "arguments": {"y": 2}, "requestor": "assistant"},
                ]
            ).encode("utf-8")
        ).hexdigest()
        with mock.patch("preflight.suite.gold_calls", return_value=calls):
            self.assertEqual(gold_fingerprint(object()), expected)

    def test_output_fingerprint_exact_value_with_non_ascii(self):
        outputs = ['{"b": 1, "a": "é"}', "25"]
        normalized = [{"a": "é", "b": 1.0}, 25.0]
        expected = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
        self.assertEqual(output_fingerprint(outputs), expected)


class VersionTest(unittest.TestCase):
    def test_tau2_version_is_the_second_word_of_the_first_source_line(self):
        self.assertEqual(_tau2_version("tau2-bench v1.0.1 (commit fc0055d)\n"), "v1.0.1")
        self.assertEqual(_tau2_version("tau2-bench pre-v1.0.1\n"), "pre-v1.0.1")

    def test_missing_source_or_short_line_gives_no_version(self):
        self.assertIsNone(_tau2_version(None))
        self.assertIsNone(_tau2_version(""))
        self.assertIsNone(_tau2_version("tau2-bench\n"))


class RecordsHashTest(unittest.TestCase):
    def test_records_hash_is_order_independent_and_ignores_seconds(self):
        left = {"task_id": "t", "relation": "F1", "detail": "a", "seconds": 1}
        right = {"task_id": "u", "relation": "F1", "detail": "b"}
        self.assertEqual(_records_sha256([left, right]), _records_sha256([right, left]))
        self.assertEqual(_records_sha256([left]), _records_sha256([{**left, "seconds": 2}]))
        self.assertEqual(len(_records_sha256([])), 64)

    def test_records_hash_changes_when_a_record_changes(self):
        record = {"task_id": "t", "relation": "F1", "detail": "a"}
        self.assertNotEqual(_records_sha256([record]), _records_sha256([{**record, "reward": 1.0}]))


class VariantRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = "airline"
        cls.task = tasks(cls.domain)[0]

    def test_flagged_verdict_gets_a_diff(self):
        variant = Variant("F1", (), False, "empty run")
        with mock.patch("preflight.suite._reward", return_value=1.0), mock.patch(
            "preflight.suite._db_diff", return_value=["balance"]
        ):
            record = _variant_record(self.domain, self.task, variant, [None, None], None)
        self.assertEqual(record["outcome"], "verdict")
        self.assertEqual(record["reward"], 1.0)
        self.assertTrue(record["flagged"])
        self.assertEqual(record["db_diff"], ["balance"])
        self.assertEqual(record["checks"], {})

    def test_unflagged_verdict_has_no_diff(self):
        variant = Variant("F1", (), False, "empty run")
        with mock.patch("preflight.suite._reward", return_value=0.0), mock.patch(
            "preflight.suite._db_diff"
        ) as diff:
            record = _variant_record(self.domain, self.task, variant, [None, None], None)
        self.assertEqual(record["outcome"], "verdict")
        self.assertFalse(record["flagged"])
        self.assertIsNone(record["db_diff"])
        diff.assert_not_called()

    def test_a_raising_grader_is_an_error_not_a_reward(self):
        variant = Variant("F1", (), False, "empty run")
        with mock.patch("preflight.suite._reward", side_effect=RuntimeError("boom")):
            record = _variant_record(self.domain, self.task, variant, [None, None], None)
        self.assertEqual(record["outcome"], "error")
        self.assertEqual(record["error"], {"type": "RuntimeError", "message": "boom"})
        self.assertIsNone(record["reward"])
        self.assertIsNone(record["flagged"])
        self.assertIsNone(record["db_diff"])

    def test_a_raising_replay_is_an_error_not_a_reward(self):
        variant = Variant("F1", (), False, "empty run")
        with mock.patch("preflight.suite.run", side_effect=RuntimeError("replay")):
            record = _variant_record(self.domain, self.task, variant, [None, None], None)
        self.assertEqual(record["outcome"], "error")
        self.assertEqual(record["error"], {"type": "RuntimeError", "message": "replay"})
        self.assertIsNone(record["reward"])
        self.assertIsNone(record["flagged"])
        self.assertIsNone(record["db_diff"])


class _DB:
    def __init__(self, agent, user):
        self.agent_db = agent
        self.user_db = user


class _Env:
    def __init__(self, agent, user):
        self.db = _DB(agent, user)


class DbDiffTest(unittest.TestCase):
    def test_top_level_names_that_differ(self):
        left = _Env({"a": "x", "b": 1}, {"c": True})
        right = _Env({"a": "x", "b": 2}, {"c": True, "d": None})
        self.assertEqual(_db_diff(left, right), ["b", "d"])

    def test_missing_and_none_differ(self):
        left = _Env({"a": None}, {})
        right = _Env({}, {})
        self.assertEqual(_db_diff(left, right), ["a"])


class RunSuiteTest(unittest.TestCase):
    def test_limit_two_is_deterministic_and_json_serialisable(self):
        first = run_suite("airline", relations=["F1"], seed=0, limit=2)
        second = run_suite("airline", relations=["F1"], seed=0, limit=2)
        json.dumps(first)
        json.dumps(second)
        self.assertEqual(first["records_sha256"], second["records_sha256"])
        first.pop("seconds")
        second.pop("seconds")
        self.assertEqual(first, second)

    def test_base_failure_gets_no_variants(self):
        with mock.patch("preflight.suite._reward", return_value=0.0):
            result = run_suite("airline", relations=["F1"], seed=0, limit=1)
        self.assertEqual(result["variants"], [])
        for task in result["tasks"]:
            self.assertNotEqual(task["base_reward"], 1.0)

    def test_base_run_error_is_recorded_and_gives_no_variants(self):
        with mock.patch("preflight.suite.run", side_effect=RuntimeError("boom")):
            result = run_suite("airline", relations=["F1"], seed=0, limit=1)
        self.assertEqual(result["variants"], [])
        task = result["tasks"][0]
        self.assertEqual(task["base_outcome"], "error")
        self.assertIsNone(task["base_reward"])
        self.assertEqual(task["static"]["base_error"], {"type": "RuntimeError", "message": "boom"})

    def test_relation_error_is_recorded_without_aborting_the_task(self):
        with mock.patch("preflight.suite._reward", return_value=1.0), mock.patch(
            "preflight.suite.build_variants", side_effect=RuntimeError("bad relation")
        ):
            result = run_suite("airline", relations=["F1"], seed=0, limit=1)
        self.assertEqual(result["variants"], [])
        task = result["tasks"][0]
        self.assertEqual(task["base_outcome"], "verdict")
        self.assertEqual(task["base_reward"], 1.0)
        self.assertEqual(
            task["static"]["relation_errors"]["F1"],
            {"type": "RuntimeError", "message": "bad relation"},
        )

    def test_p5_not_inert_reaches_task_static(self):
        with mock.patch("preflight.suite._reward", return_value=1.0), mock.patch(
            "preflight.suite.build_variants", return_value=([], {"p5_not_inert": True})
        ):
            result = run_suite("airline", relations=["P5"], seed=0, limit=1)
        self.assertEqual(result["variants"], [])
        self.assertTrue(result["tasks"][0]["static"].get("p5_not_inert"))

    def test_unknown_relation_is_rejected(self):
        with self.assertRaises(ValueError):
            run_suite("airline", relations=["NOPE"], limit=0)
