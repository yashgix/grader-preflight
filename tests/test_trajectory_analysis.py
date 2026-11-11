"""Tests for the phase 3 trajectory analysis: H5, H7, report output and H6 sampling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import trajectory_analysis

BANK = "banking_knowledge"
AIRLINE = "airline"


def _record(run_id, task_id="task_001", domain=BANK, **overrides):
    record = {
        "id": run_id,
        "task_id": task_id,
        "trial": 0,
        "domain": domain,
        "git_commit": "commit-1",
        "task_matches": True,
        "replay_class": "full",
        "reward": 1.0,
        "recorded_reward": 1.0,
        "mismatched": [],
        "p1": False,
        "p3": False,
        "diff_tables": [],
        "n_calls": 3,
        "nongold_call": False,
        "schema_floats": None,
        "numeric_pair": None,
        "late_verification": None,
        "error": None,
    }
    record.update(overrides)
    return record


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_records(directory, name, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class H5SummaryTest(unittest.TestCase):
    def test_an_explained_event_holds_and_lists_nothing_unexplained(self):
        flags = {"P1": {"task_001"}, "P3": set()}
        old = _record("r1", reward=0.0, p1=True, diff_tables=["agent_discoverable_tools"])
        new = _record("r1", reward=1.0)
        summary = trajectory_analysis._h5_summary([(old, new)], flags)
        self.assertTrue(summary["testable"])
        self.assertEqual((summary["events"], summary["explained"]), (1, 1))
        self.assertEqual(summary["recall"], 1.0)
        self.assertTrue(summary["held"])
        self.assertEqual(summary["unexplained"], [])

    def test_no_events_makes_recall_undefined(self):
        old = _record("r1", reward=1.0)
        new = _record("r1", reward=1.0)
        summary = trajectory_analysis._h5_summary([(old, new)], {"P1": set(), "P3": set()})
        self.assertFalse(summary["testable"])
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["ci"])
        self.assertIsNone(summary["held"])

    def test_half_recall_is_not_held_and_unexplained_is_listed(self):
        flags = {"P1": {"task_001"}, "P3": set()}
        explained = (
            _record("r1", reward=0.0, p1=True, diff_tables=["agent_discoverable_tools"]),
            _record("r1", reward=1.0),
        )
        unexplained = (
            _record("r2", task_id="task_002", reward=0.0),
            _record("r2", task_id="task_002", reward=1.0),
        )
        summary = trajectory_analysis._h5_summary([explained, unexplained], flags)
        self.assertEqual((summary["events"], summary["explained"], summary["recall"]), (2, 1, 0.5))
        self.assertFalse(summary["held"])
        self.assertEqual([item["id"] for item in summary["unexplained"]], ["r2"])

    def test_bootstrap_is_deterministic(self):
        events = [
            (_record(f"r{index}", task_id=f"task_{index}", reward=0.0), _record(f"r{index}", task_id=f"task_{index}"))
            for index in range(5)
        ]
        first = trajectory_analysis._bootstrap_interval(events, {"P1": set(), "P3": set()})
        second = trajectory_analysis._bootstrap_interval(events, {"P1": set(), "P3": set()})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertLessEqual(first[0], first[1])


class PrecisionTest(unittest.TestCase):
    def test_precision_counts_events_among_triggered_runs(self):
        pairs = [
            (_record("a", p1=True, reward=0.0), _record("a", reward=1.0)),
            (_record("b", p1=True, reward=1.0), _record("b", reward=1.0)),
            (_record("c", p1=False, reward=0.0), _record("c", reward=1.0)),
        ]
        self.assertEqual(trajectory_analysis._precision(pairs, "p1"), {"runs": 2, "events": 1, "precision": 0.5})

    def test_precision_is_undefined_without_a_trigger(self):
        pairs = [(_record("a", reward=0.0), _record("a", reward=1.0))]
        self.assertEqual(trajectory_analysis._precision(pairs, "p1"), {"runs": 0, "events": 0, "precision": None})


class BaselineTest(unittest.TestCase):
    def test_baselines_report_rates_and_enrichment(self):
        flags = {"P1": {"task_a"}, "P3": set()}
        pairs = [
            (_record("a1", task_id="task_a", reward=0.0, n_calls=1), _record("a1", task_id="task_a")),
            (_record("a2", task_id="task_a", reward=1.0, n_calls=2), _record("a2", task_id="task_a")),
            (
                _record("b1", task_id="task_b", reward=1.0, nongold_call=True, n_calls=3),
                _record("b1", task_id="task_b"),
            ),
            (_record("b2", task_id="task_b", reward=0.0, n_calls=4), _record("b2", task_id="task_b")),
        ]
        b1 = trajectory_analysis._baseline(pairs, lambda pair: str(pair[0].get("task_id")) in flags["P1"])
        self.assertEqual((b1["positive"], b1["negative"]), (2, 2))
        self.assertEqual((b1["events_positive"], b1["events_negative"]), (1, 1))
        self.assertEqual(b1["recall"], 0.5)
        self.assertEqual(b1["enrichment"], 1.0)

        b2 = trajectory_analysis._baseline(
            pairs,
            lambda pair: pair[0].get("nongold_call") is True or pair[1].get("nongold_call") is True,
        )
        self.assertEqual((b2["positive"], b2["events_positive"]), (1, 0))
        self.assertEqual(b2["enrichment"], 0.0)

        b3 = trajectory_analysis._baseline(
            pairs,
            lambda pair: isinstance(pair[0].get("n_calls"), (int, float)) and pair[0]["n_calls"] > 2.5,
        )
        self.assertEqual((b3["positive"], b3["events_positive"]), (2, 1))
        self.assertEqual(b3["enrichment"], 1.0)

    def test_summary_carries_all_three_baselines(self):
        flags = {"P1": {"task_a"}, "P3": set()}
        pairs = [
            (_record("a1", task_id="task_a", reward=0.0, n_calls=1), _record("a1", task_id="task_a")),
            (_record("b1", task_id="task_b", reward=0.0, n_calls=4), _record("b1", task_id="task_b")),
        ]
        summary = trajectory_analysis._h5_summary(pairs, flags)
        self.assertEqual(set(summary["baselines"]), {"B1", "B2", "B3"})


class DocumentBreakdownTest(unittest.TestCase):
    def test_by_commit_and_by_stem_are_reported(self):
        old_first = _record("r1", task_id="task_001", git_commit="commit-a", reward=0.0)
        new_first = _record("r1", task_id="task_001", git_commit="commit-a")
        old_first["_stem"], new_first["_stem"] = "banking-a", "banking-a"
        old_second = _record("r2", task_id="task_002", git_commit="commit-b")
        new_second = _record("r2", task_id="task_002", git_commit="commit-b")
        old_second["_stem"], new_second["_stem"] = "banking-b", "banking-b"

        document = trajectory_analysis._h5_document(
            [(old_first, new_first), (old_second, new_second)], {"P1": set(), "P3": set()}
        )
        self.assertEqual(document["by_commit"]["commit-a"]["events"], 1)
        self.assertEqual(document["by_commit"]["commit-b"]["events"], 0)
        self.assertIn("banking-a", document["by_stem"])
        self.assertIn("banking-b", document["by_stem"])


class H7Test(unittest.TestCase):
    def test_counts_airline_numeric_and_banking_late_runs(self):
        airline_old = _record("a1", domain=AIRLINE, reward=0.0)
        airline_new = _record(
            "a1",
            domain=AIRLINE,
            schema_floats=[{"index": 0, "tool": "search", "argument": "limit"}],
            numeric_pair={"inconsistency": True},
        )
        bank_old = _record("b1", reward=0.0)
        bank_new = _record(
            "b1",
            reward=1.0,
            late_verification={"log_after_first_write": True, "access_before_facts": True},
        )

        document = trajectory_analysis._h7_document([(airline_old, airline_new), (bank_old, bank_new)])
        self.assertEqual(document["airline"]["runs"], 1)
        self.assertEqual(document["airline"]["schema_float_runs"], 1)
        self.assertEqual(document["airline"]["inconsistencies"], 1)
        self.assertEqual(document["airline"]["control_events"], 1)
        self.assertEqual(document["banking"]["reward_one_runs"], 1)
        self.assertEqual(document["banking"]["late_log_runs"], 1)
        self.assertEqual(document["banking"]["access_before_facts_runs"], 1)

    def test_airline_control_is_zero_when_the_verdicts_agree(self):
        airline_old = _record("a1", domain=AIRLINE, reward=1.0)
        airline_new = _record("a1", domain=AIRLINE, reward=1.0)
        document = trajectory_analysis._h7_document([(airline_old, airline_new)])
        self.assertEqual(document["airline"]["control_events"], 0)


class AnalyzeTest(unittest.TestCase):
    def test_analyze_writes_h5_h7_csv_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = _record("r1", reward=0.0, p1=True, diff_tables=["agent_discoverable_tools"])
            new = _record("r1", reward=1.0)
            _write_records(root / "old", "banking", [old])
            _write_records(root / "new", "banking", [new])
            flags = _write_json(root / "flags.json", {"P1": ["task_001"], "P3": []})
            pool = _write_json(root / "pool.json", {"tasks": ["task_001"]})

            result = trajectory_analysis.analyze(root / "old", root / "new", flags, pool, root / "out")
            h5 = _read_json(root / "out" / "h5.json")
            self.assertEqual(result["h5"]["events"], 1)
            self.assertEqual(h5["explained"], 1)
            self.assertTrue(h5["held"])
            self.assertTrue((root / "out" / "h7.json").is_file())
            self.assertTrue((root / "out" / "runs.csv").is_file())
            self.assertIn("## H5", (root / "out" / "report.md").read_text(encoding="utf-8"))


class StrataTest(unittest.TestCase):
    def _pairs(self):
        rewards = {"s1": (1.0, 1.0), "s2": (1.0, 1.0), "s3": (0.0, 0.0), "s4": (0.0, 1.0)}
        return [(_record(i, reward=o, _stem="f"), _record(i, reward=n, _stem="f")) for i, (o, n) in rewards.items()]

    def test_strata_use_rewards_and_late_verification(self):
        pairs = self._pairs()
        pairs[0][1]["late_verification"] = {"log_after_first_write": True, "access_before_facts": None}
        entries = {
            ("f", run_id): {"domain": "", "task": {"id": "task_001"}, "run": {"id": run_id, "messages": []}}
            for run_id in ("s1", "s2", "s3", "s4")
        }
        strata = trajectory_analysis._strata(pairs, {"task_001"}, entries)
        self.assertEqual(strata["S1"], {("f", "s1")})
        self.assertEqual(strata["S2"], {("f", "s2")})
        self.assertEqual(strata["S3"], {("f", "s3")})
        self.assertEqual(strata["S4"], {("f", "s4")})


class SampleTest(unittest.TestCase):
    def test_sample_writes_blinded_packets_and_a_separate_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_records = [
                _record("s1", reward=1.0),
                _record("s2", reward=1.0),
                _record("s3", reward=0.0),
                _record("s4", reward=0.0),
            ]
            new_records = [
                _record(
                    "s1", reward=1.0, late_verification={"log_after_first_write": True, "access_before_facts": None}
                ),
                _record("s2", reward=1.0),
                _record("s3", reward=0.0),
                _record("s4", reward=1.0),
            ]
            _write_records(root / "old", "banking", old_records)
            _write_records(root / "new", "banking", new_records)
            flags = _write_json(root / "flags.json", {"P1": [], "P3": []})
            pool = _write_json(root / "pool.json", {"tasks": ["task_001"]})
            entries = {
                ("banking", run_id): {
                    "domain": "",
                    "policy": "",
                    "task": {"id": "task_001"},
                    "run": {"id": run_id, "messages": [{"role": "user", "content": "plain"}]},
                }
                for run_id in ("s1", "s2", "s3", "s4")
            }
            out = root / "sample"
            with mock.patch.object(trajectory_analysis, "_load_source_files", return_value=entries):
                key = trajectory_analysis.sample(root / "old", root / "new", flags, pool, [], out)

            self.assertEqual(len(key), 4)
            self.assertEqual({item["stratum"] for item in key}, {"S1", "S2", "S3", "S4"})
            packets = sorted((out / "packets").glob("*.json"))
            self.assertEqual(len(packets), 4)
            for path in packets:
                packet = _read_json(path)
                self.assertEqual(set(packet), {"task_instructions", "policy", "messages"})
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("reward", text)
                self.assertNotIn("stratum", text)
                for _, run_id in entries:
                    self.assertNotIn(run_id, text)
            key_data = _read_json(out / "key.json")
            self.assertEqual({(item["file"], item["run_id"]) for item in key_data}, set(entries))


if __name__ == "__main__":
    unittest.main()
