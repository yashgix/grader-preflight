"""Tests for preflight.compare on small synthetic result files."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from preflight.compare import _load, _records_digest, compare

BANK = "banking_knowledge"
REQUIRED_METRICS = {
    "h1a_held",
    "h1b_held",
    "h2_held",
    "control_passed",
    "control_mismatches",
    "p1_flagged_tasks_old",
    "p1_flagged_tasks_new",
    "p1_in_gold_flagged_tasks_new",
    "p3_flagged_tasks_old",
    "p3_flagged_tasks_new",
    "f1_pass_airline_new",
    "f3_pass_rate_new",
    "f6_pass_share_new",
    "changed_tasks",
    "attempted_old",
    "attempted_new",
    "errors_old",
    "errors_new",
    "unpaired_comparable",
    "records_digest_old",
    "records_digest_new",
}


def task(task_id, *, changes=False, task_fp="task", gold_fp="gold"):
    return {
        "id": task_id,
        "reward_basis": ["DB"],
        "task_fingerprint": task_fp,
        "gold_output_fingerprint": gold_fp,
        "base_outcome": "verdict",
        "base_reward": 1.0,
        "static": {"gold_changes_db": changes},
    }


def task_with_static(task_id, static):
    item = task(task_id)
    item["static"] = static
    return item


def variant(
    task_id,
    relation,
    *,
    detail=None,
    outcome="verdict",
    reward=0.0,
    flagged=False,
    inserted_in_gold=False,
    calls=None,
    output_fingerprint=None,
    db_diff=None,
):
    return {
        "task_id": task_id,
        "relation": relation,
        "detail": detail or f"{relation} {task_id}",
        "expected_pass": relation.startswith("P"),
        "inserted_in_gold": inserted_in_gold,
        "calls": calls or [],
        "checks": {},
        "live_call_errors": [],
        "output_fingerprint": output_fingerprint,
        "end_state": {"variant": [None, None], "gold": [None, None]},
        "outcome": outcome,
        "reward": reward if outcome == "verdict" else None,
        "flagged": flagged if outcome == "verdict" else None,
        "error": None if outcome == "verdict" else {"type": "X", "message": "boom"},
        "db_diff": db_diff,
    }


def record(domain, tasks, variants, sha):
    return {
        "domain": domain,
        "seed": 0,
        "tau2_version": "1.0.1",
        "provenance": {"source": None, "python": "3.12"},
        "tasks": tasks,
        "variants": variants,
        "records_sha256": sha,
        "seconds": 0.0,
    }


def write_side(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    for item in records:
        path = directory / f"{item['domain']}.json"
        path.write_text(json.dumps(item), encoding="utf-8")


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_compare(self, old, new, name="report.md"):
        old_dir = self.root / "old"
        new_dir = self.root / "new"
        write_side(old_dir, old)
        write_side(new_dir, new)
        out = self.root / name
        compare(old_dir, new_dir, out)
        return out, json.loads((self.root / "metrics.json").read_text(encoding="utf-8"))

    def test_p1_pairs_and_in_gold_counted_apart(self):
        old_non = variant("A", "P1", detail="p1 non", reward=1.0, flagged=True)
        new_non = variant("A", "P1", detail="p1 non", reward=0.0, flagged=False)
        old_in = variant("B", "P1", detail="p1 in", reward=1.0, flagged=True, inserted_in_gold=True)
        new_in = variant("B", "P1", detail="p1 in", reward=1.0, flagged=True, inserted_in_gold=True)
        old = [record(BANK, [task("A"), task("B")], [old_non, old_in], "old-sha")]
        new = [record(BANK, [task("A"), task("B")], [new_non, new_in], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(a): held", text)
        self.assertEqual(metrics["h1a_held"], 1)
        self.assertEqual(metrics["p1_flagged_tasks_old"], 2)
        self.assertEqual(metrics["p1_flagged_tasks_new"], 1)
        self.assertEqual(metrics["p1_in_gold_flagged_tasks_new"], 1)

    def test_unpaired_p1_breaks_h1a(self):
        old_non = variant("A", "P1", detail="p1 non", reward=1.0, flagged=True)
        new_non = variant("A", "P1", detail="p1 non", reward=0.0, flagged=False)
        new_extra = variant("A", "P1", detail="p1 extra", reward=0.0, flagged=False)
        old = [record(BANK, [task("A")], [old_non], "old-sha")]
        new = [record(BANK, [task("A")], [new_non, new_extra], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(a): not held", text)
        self.assertEqual(metrics["h1a_held"], 0)
        self.assertEqual(metrics["unpaired_comparable"], 1)

    def test_errors_do_not_make_pairs(self):
        old_variant = variant("A", "P1", reward=1.0, flagged=True)
        new_variant = variant("A", "P1", outcome="error")
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(a): not held", text)
        self.assertIn("paired verdicts: 0", text)
        self.assertEqual(metrics["h1a_held"], 0)
        self.assertEqual(metrics["attempted_new"], 1)
        self.assertEqual(metrics["errors_new"], 1)

    def test_old_only_flags_are_listed_by_pair_and_detail(self):
        old_first = variant("A", "P1", detail="p1 first", reward=1.0, flagged=True)
        new_first = variant("A", "P1", detail="p1 first", reward=0.0, flagged=False)
        old_second = variant("A", "P1", detail="p1 second", reward=0.0, flagged=False)
        new_second = variant("A", "P1", detail="p1 second", reward=1.0, flagged=True)
        old = [record(BANK, [task("A")], [old_first, old_second], "old-sha")]
        new = [record(BANK, [task("A")], [new_first, new_second], "new-sha")]

        out, _ = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        section = text.split("## Flags on OLD and absent on NEW", 1)[1].split("## Hypotheses", 1)[0]
        self.assertIn("p1 first", section)
        self.assertNotIn("p1 second", section)

    def test_p3_pair_holds_h1b(self):
        old_variant = variant("A", "P3", reward=1.0, flagged=True)
        new_variant = variant("A", "P3", reward=0.0, flagged=False)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(b): held", text)
        self.assertIn("OLD flagged paired tasks: 1 (A)", text)
        self.assertEqual(metrics["h1b_held"], 1)

    def test_unpaired_p3_breaks_h1b(self):
        old_variant = variant("A", "P3", reward=1.0, flagged=True)
        new_variant = variant("A", "P3", reward=0.0, flagged=False)
        new_extra = variant("A", "P3", detail="p3 extra", reward=0.0, flagged=False)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant, new_extra], "new-sha")]

        out, metrics = self.run_compare(old, new)
        self.assertIn("H1(b): not held", out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["h1b_held"], 0)
        self.assertEqual(metrics["unpaired_comparable"], 1)

    def test_variants_with_different_calls_are_not_paired(self):
        old_calls = [{"name": "read", "arguments": {"id": 1}, "requestor": "assistant"}]
        new_calls = [{"name": "read", "arguments": {"id": 2}, "requestor": "assistant"}]
        old_variant = variant("A", "P1", detail="p1 read", reward=0.0, flagged=True, calls=old_calls)
        new_variant = variant("A", "P1", detail="p1 read", reward=1.0, flagged=False, calls=new_calls)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        self.assertIn("H1(a): not held", out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["unpaired_comparable"], 2)

    def test_calls_pair_whatever_their_key_order(self):
        old_calls = [{"name": "read", "arguments": {"a": 1, "b": 2}, "requestor": "assistant"}]
        new_calls = [{"requestor": "assistant", "arguments": {"b": 2, "a": 1}, "name": "read"}]
        old_variant = variant("A", "P1", detail="p1 read", reward=0.0, flagged=True, calls=old_calls)
        new_variant = variant("A", "P1", detail="p1 read", reward=1.0, flagged=False, calls=new_calls)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        _, metrics = self.run_compare(old, new)
        self.assertEqual(metrics["unpaired_comparable"], 0)
        self.assertEqual(metrics["h1a_held"], 1)

    def test_a_gold_output_change_alone_marks_the_fixture_changed(self):
        old_variant = variant("A", "P1", reward=0.0, flagged=True)
        new_variant = variant("A", "P1", reward=1.0, flagged=False)
        old = [record(BANK, [task("A", gold_fp="old-output")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A", gold_fp="new-output")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        self.assertEqual(metrics["changed_tasks"], 1)
        self.assertEqual(metrics["h1a_held"], 0)
        self.assertEqual(metrics["unpaired_comparable"], 0)
        self.assertIn("- banking_knowledge/A: gold_output_fingerprint\n", out.read_text(encoding="utf-8"))

    def test_fixture_changed_tasks_are_excluded_from_h1_and_control(self):
        old_variant = variant("X", "F1", reward=1.0, flagged=True)
        new_variant = variant("X", "F1", reward=0.0, flagged=False)
        old = [record("airline", [task("X", task_fp="old-task")], [old_variant], "old-sha")]
        new = [record("airline", [task("X", task_fp="new-task")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertEqual(metrics["changed_tasks"], 1)
        self.assertEqual(metrics["control_passed"], 0)
        self.assertEqual(metrics["control_mismatches"], 1)
        self.assertEqual(metrics["unpaired_comparable"], 0)
        self.assertIn("task_fingerprint", text)

    def test_control_lists_reward_unpaired_and_missing(self):
        old_a = variant("A", "F1", reward=0.0, flagged=False)
        new_a = variant("A", "F1", reward=1.0, flagged=True)
        old_b = variant("B", "F1", reward=0.0, flagged=False)
        old_c = variant("C", "F1", reward=0.0, flagged=False)
        old = [record("airline", [task("A"), task("B"), task("C")], [old_a, old_b, old_c], "old-sha")]
        new = [record("airline", [task("A"), task("C")], [new_a], "new-sha")]

        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("Control: failed", text)
        self.assertIn("missing on new", text)
        self.assertIn("unpaired on old", text)
        self.assertEqual(metrics["control_passed"], 0)
        self.assertEqual(metrics["control_mismatches"], 3)
        self.assertEqual(metrics["unpaired_comparable"], 1)

    def test_control_passes_with_identical_pairs(self):
        old_variant = variant("A", "F1", reward=0.0, flagged=False)
        new_variant = variant("A", "F1", reward=0.0, flagged=False)
        old = [record("airline", [task("A")], [old_variant], "old-sha")]
        new = [record("airline", [task("A")], [new_variant], "new-sha")]

        out, metrics = self.run_compare(old, new)
        self.assertIn("Control: passed", out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["control_passed"], 1)
        self.assertEqual(metrics["control_mismatches"], 0)

    def test_one_sided_banking_task_is_listed(self):
        old_variant = variant("A", "P1", reward=1.0, flagged=True)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]

        out, metrics = self.run_compare(old, [])
        text = out.read_text(encoding="utf-8")
        self.assertIn("only on old", text)
        self.assertEqual(metrics["control_passed"], 1)
        self.assertEqual(metrics["changed_tasks"], 0)
        self.assertEqual(metrics["unpaired_comparable"], 0)
        self.assertEqual(metrics["attempted_old"], 1)
        self.assertEqual(metrics["attempted_new"], 0)

    def test_f1_split_is_per_domain_and_gold_changes_db(self):
        airline_tasks = [task("A", changes=True), task("B", changes=False), task("C", changes=None)]
        airline_variants = [
            variant("A", "F1", reward=1.0, flagged=True),
            variant("B", "F1", reward=1.0, flagged=True),
            variant("C", "F1", reward=1.0, flagged=True),
        ]
        new = [record("airline", airline_tasks, airline_variants, "new-airline")]

        out, _ = self.run_compare([], new)
        text = out.read_text(encoding="utf-8")
        self.assertIn(
            "- new airline gold_changes_db=True: attempted=1 verdicts=1 errors=0 passed_variants=1 passed_tasks=1",
            text,
        )
        self.assertIn(
            "- new airline gold_changes_db=False: attempted=1 verdicts=1 errors=0 passed_variants=1 passed_tasks=1",
            text,
        )
        self.assertIn(
            "- new airline gold_changes_db=unknown: attempted=1 verdicts=1 errors=0 passed_variants=1 passed_tasks=1",
            text,
        )

    def test_static_skips_and_write_counts_are_reported(self):
        first = task_with_static(
            "A",
            {
                "gold_changes_db": True,
                "identical_to_gold": ["P1", "P3"],
                "p5_not_inert": True,
                "no_effect_writes": 2,
            },
        )
        second = task_with_static(
            "B",
            {"gold_changes_db": True, "identical_to_gold": ["P1"], "no_effect_writes": 1},
        )
        airline = task_with_static(
            "X",
            {"gold_changes_db": False, "p5_not_inert": True, "no_effect_writes": 4},
        )
        old = [
            record(BANK, [first, second], [], "old-banking"),
            record("airline", [airline], [], "old-airline"),
        ]

        out, _ = self.run_compare(old, [])
        text = out.read_text(encoding="utf-8")
        self.assertIn("| old | banking_knowledge | P1 | 2 |", text)
        self.assertIn("| old | banking_knowledge | P3 | 1 |", text)
        self.assertIn("| old | banking_knowledge | 1 | 3 |", text)
        self.assertIn("| old | airline | 1 | 4 |", text)

    def test_h2_held_with_enough_f1_f6_and_a_p_flag(self):
        airline_tasks = [task(f"A{index}") for index in range(20)]
        airline_variants = [variant(f"A{index}", "F1", reward=1.0, flagged=True) for index in range(20)]
        airline_variants.append(variant("A0", "F3", reward=0.0, flagged=False))
        f6_variants = [
            variant("B0", "F6", reward=1.0, flagged=True),
            variant("B1", "F6", reward=0.0, flagged=False),
        ]
        p3_variant = variant("R0", "P3", reward=1.0, flagged=True)
        new = [
            record("airline", airline_tasks, airline_variants, "airline-sha"),
            record(BANK, [task("B0"), task("B1")], f6_variants, "banking-sha"),
            record("retail", [task("R0")], [p3_variant], "retail-sha"),
        ]

        out, metrics = self.run_compare([], new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H2: held", text)
        self.assertEqual(metrics["h2_held"], 1)
        self.assertEqual(metrics["f1_pass_airline_new"], 20)
        self.assertEqual(metrics["f3_pass_rate_new"], 0.0)
        self.assertEqual(metrics["f6_pass_share_new"], 0.5)

    def test_h2_is_not_held_when_f3_has_no_verdict(self):
        airline_tasks = [task(f"A{index}") for index in range(20)]
        airline_variants = [variant(f"A{index}", "F1", reward=1.0, flagged=True) for index in range(20)]
        airline_variants.append(variant("A0", "F3", outcome="error"))
        f6_variants = [variant("B0", "F6", reward=1.0, flagged=True)]
        p3_variant = variant("R0", "P3", reward=1.0, flagged=True)
        new = [
            record("airline", airline_tasks, airline_variants, "airline-sha"),
            record(BANK, [task("B0")], f6_variants, "banking-sha"),
            record("retail", [], [p3_variant], "retail-sha"),
        ]

        out, metrics = self.run_compare([], new)
        self.assertIn("H2: not held", out.read_text(encoding="utf-8"))
        self.assertEqual(metrics["h2_held"], 0)
        self.assertEqual(metrics["errors_new"], 1)

    def test_metrics_have_exactly_the_required_numeric_keys(self):
        old_variant = variant("A", "P1", reward=1.0, flagged=True)
        new_variant = variant("A", "P1", reward=0.0, flagged=False)
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        _, metrics = self.run_compare(old, new)
        self.assertEqual(set(metrics), REQUIRED_METRICS)
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float), key)
            self.assertNotIsInstance(value, bool, key)

    def test_h1_line_reports_db_diff_and_output_fingerprint_differences(self):
        old_variant = variant(
            "A",
            "P1",
            reward=1.0,
            flagged=True,
            output_fingerprint="old",
            db_diff=["balance"],
        )
        new_variant = variant("A", "P1", reward=0.0, flagged=False, output_fingerprint="new")
        old = [record(BANK, [task("A")], [old_variant], "old-sha")]
        new = [record(BANK, [task("A")], [new_variant], "new-sha")]

        out, _ = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn(json.dumps(["balance"]), text)
        self.assertIn("output_fingerprints differ: 1", text)

    def test_records_digest_matches_independent_calculation(self):
        directory = self.root / "digest_side"
        directory.mkdir()
        airline = {"domain": "airline", "records_sha256": "aaa"}
        banking = {"domain": "banking_knowledge", "records_sha256": "bbb"}
        (directory / "airline.json").write_text(json.dumps(airline), encoding="utf-8")
        (directory / "banking_knowledge.json").write_text(json.dumps(banking), encoding="utf-8")

        joined = "aaa" + chr(10) + "bbb"
        expected = int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12], 16)
        self.assertEqual(_records_digest(_load(directory)), expected)


if __name__ == "__main__":
    unittest.main()
