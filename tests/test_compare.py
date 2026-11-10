"""Tests for preflight.compare on small synthetic result files."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from preflight.compare import _load, _records_digest, compare


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
):
    return {
        "task_id": task_id,
        "relation": relation,
        "detail": detail or f"{relation} {task_id}",
        "expected_pass": relation.startswith("P"),
        "inserted_in_gold": inserted_in_gold,
        "calls": [],
        "checks": {},
        "live_call_errors": [],
        "end_state": {"variant": [None, None], "gold": [None, None]},
        "outcome": outcome,
        "reward": reward if outcome == "verdict" else None,
        "flagged": flagged if outcome == "verdict" else None,
        "error": None if outcome == "verdict" else {"type": "X", "message": "boom"},
        "db_diff": None,
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
        (directory / f"{item['domain']}.json").write_text(json.dumps(item), encoding="utf-8")


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

    def test_p1_in_gold_on_new_is_ignored_for_h1a(self):
        old = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", reward=1.0, flagged=True)],
                "old-sha",
            )
        ]
        new = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", reward=1.0, flagged=True, inserted_in_gold=True)],
                "new-sha",
            )
        ]
        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(a): held", text)
        self.assertEqual(metrics["h1a_held"], 1)
        self.assertEqual(metrics["p1_flagged_tasks_old"], 1)
        self.assertEqual(metrics["p1_flagged_tasks_new"], 1)
        self.assertEqual(metrics["p1_in_gold_flagged_tasks_new"], 1)

    def test_errors_are_reported_but_not_counted_as_flags(self):
        old = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", outcome="error")],
                "old-sha",
            )
        ]
        new = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", outcome="error")],
                "new-sha",
            )
        ]
        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H1(a): not held", text)
        self.assertEqual(metrics["attempted_old"], 1)
        self.assertEqual(metrics["errors_old"], 1)
        self.assertEqual(metrics["p1_flagged_tasks_old"], 0)
        self.assertEqual(metrics["h1a_held"], 0)

    def test_fixture_changed_tasks_are_excluded_from_h1_and_control(self):
        old = [
            record(
                "banking_knowledge",
                [task("A", task_fp="old-task")],
                [variant("A", "P1", reward=1.0, flagged=True)],
                "old-sha",
            ),
            record(
                "airline",
                [task("X", gold_fp="old-output")],
                [variant("X", "F1", reward=1.0, flagged=True)],
                "old-airline",
            ),
        ]
        new = [
            record(
                "banking_knowledge",
                [task("A", task_fp="new-task")],
                [variant("A", "P1", reward=0.0, flagged=False)],
                "new-sha",
            ),
            record(
                "airline",
                [task("X", gold_fp="new-output")],
                [variant("X", "F1", reward=0.0, flagged=False)],
                "new-airline",
            ),
        ]
        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertEqual(metrics["changed_tasks"], 2)
        self.assertEqual(metrics["h1a_held"], 0)
        self.assertEqual(metrics["control_passed"], 1)
        self.assertIn("task_fingerprint", text)
        self.assertIn("gold_output_fingerprint", text)

    def test_control_failure_lists_the_mismatch(self):
        old = [
            record(
                "airline",
                [task("X")],
                [variant("X", "F1", reward=0.0, flagged=True)],
                "old-sha",
            )
        ]
        new = [
            record(
                "airline",
                [task("X")],
                [variant("X", "F1", reward=1.0, flagged=False)],
                "new-sha",
            )
        ]
        out, metrics = self.run_compare(old, new)
        text = out.read_text(encoding="utf-8")
        self.assertEqual(metrics["control_passed"], 0)
        self.assertEqual(metrics["control_mismatches"], 1)
        self.assertIn("Control: failed", text)
        self.assertIn("airline/X", text)

    def test_metrics_have_exactly_the_required_numeric_keys(self):
        old = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", reward=1.0, flagged=True)],
                "old-sha",
            )
        ]
        new = [
            record(
                "banking_knowledge",
                [task("A")],
                [variant("A", "P1", reward=0.0, flagged=False)],
                "new-sha",
            )
        ]
        _, metrics = self.run_compare(old, new)
        required = {
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
            "records_digest_old",
            "records_digest_new",
        }
        self.assertEqual(set(metrics), required)
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float), key)
            self.assertNotIsInstance(value, bool, key)

    def test_records_digest_is_stable_and_f1_split_is_reported(self):
        old = [
            record(
                "airline",
                [task("A", changes=True), task("B", changes=False)],
                [
                    variant("A", "F1", reward=0.0, flagged=True),
                    variant("B", "F1", reward=0.0, flagged=True),
                ],
                "old-sha",
            )
        ]
        new = [
            record(
                "airline",
                [task("A", changes=True), task("B", changes=False)],
                [
                    variant("A", "F1", reward=0.0, flagged=True),
                    variant("B", "F1", reward=0.0, flagged=True),
                ],
                "new-sha",
            )
        ]
        first, metrics_first = self.run_compare(old, new, name="first.md")
        _, metrics_second = self.run_compare(old, new, name="second.md")
        text = first.read_text(encoding="utf-8")
        self.assertEqual(metrics_first, metrics_second)
        self.assertIn("old-sha", text)
        self.assertIn("gold_changes_db=True", text)
        self.assertIn("gold_changes_db=False", text)
        self.assertIsInstance(metrics_first["records_digest_old"], int)

    def test_records_digest_matches_independent_calculation(self):
        directory = self.root / "digest_side"
        directory.mkdir()
        (directory / "airline.json").write_text(
            json.dumps({"domain": "airline", "records_sha256": "aaa"}), encoding="utf-8"
        )
        (directory / "banking_knowledge.json").write_text(
            json.dumps({"domain": "banking_knowledge", "records_sha256": "bbb"}), encoding="utf-8"
        )
        files = _load(directory)
        expected = int(hashlib.sha256("aaa\nbbb".encode("utf-8")).hexdigest()[:12], 16)
        self.assertEqual(_records_digest(files), expected)

    def test_p1_split_reports_in_gold_and_non_gold_counts(self):
        old = [
            record(
                "banking_knowledge",
                [task("A"), task("B")],
                [
                    variant("A", "P1", detail="p1 a", reward=1.0, flagged=True),
                    variant("B", "P1", detail="p1 b", reward=0.0, flagged=False),
                    variant("A", "P1", detail="p1 c", reward=1.0, flagged=True, inserted_in_gold=True),
                    variant("B", "P1", detail="p1 d", outcome="error", inserted_in_gold=True),
                ],
                "old-sha",
            )
        ]
        out, _ = self.run_compare(old, [])
        text = out.read_text(encoding="utf-8")
        self.assertIn("| old | banking_knowledge | in gold | 2 | 1 | 1 | 1 | 1 |", text)
        self.assertIn("| old | banking_knowledge | not in gold | 2 | 2 | 0 | 1 | 1 |", text)

    def test_f1_split_is_per_domain_and_gold_changes_db(self):
        old = [
            record(
                "airline",
                [task("A", changes=True), task("B", changes=False)],
                [
                    variant("A", "F1", reward=1.0, flagged=True),
                    variant("B", "F1", reward=1.0, flagged=True),
                ],
                "old-airline",
            ),
            record(
                "banking_knowledge",
                [task("C", changes=True)],
                [variant("C", "F1", reward=0.0, flagged=False)],
                "old-banking",
            ),
            record(
                "retail",
                [task("R", changes=None)],
                [variant("R", "F1", reward=0.0, flagged=False)],
                "old-retail",
            ),
        ]
        out, _ = self.run_compare(old, [])
        text = out.read_text(encoding="utf-8")
        self.assertIn(
            "- old airline gold_changes_db=True: attempted=1 verdicts=1 errors=0 "
            "passed_variants=1 passed_tasks=1",
            text,
        )
        self.assertIn(
            "- old airline gold_changes_db=False: attempted=1 verdicts=1 errors=0 "
            "passed_variants=1 passed_tasks=1",
            text,
        )
        self.assertIn(
            "- old banking_knowledge gold_changes_db=True: attempted=1 verdicts=1 errors=0 "
            "passed_variants=0 passed_tasks=0",
            text,
        )
        self.assertIn(
            "- old retail gold_changes_db=unknown: attempted=1 verdicts=1 errors=0 "
            "passed_variants=0 passed_tasks=0",
            text,
        )

    def test_static_skips_and_write_counts_are_reported(self):
        old = [
            record(
                "banking_knowledge",
                [
                    task_with_static(
                        "A",
                        {
                            "gold_changes_db": True,
                            "identical_to_gold": ["P1", "P3"],
                            "p5_not_inert": True,
                            "no_effect_writes": 2,
                        },
                    ),
                    task_with_static(
                        "B",
                        {
                            "gold_changes_db": True,
                            "identical_to_gold": ["P1"],
                            "no_effect_writes": 1,
                        },
                    ),
                ],
                [],
                "old-banking",
            ),
            record(
                "airline",
                [
                    task_with_static(
                        "X",
                        {
                            "gold_changes_db": False,
                            "p5_not_inert": True,
                            "no_effect_writes": 4,
                        },
                    )
                ],
                [],
                "old-airline",
            ),
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
        new = [
            record("airline", airline_tasks, airline_variants, "airline-sha"),
            record(
                "banking_knowledge",
                [task("B0"), task("B1")],
                [
                    variant("B0", "F6", reward=1.0, flagged=True),
                    variant("B1", "F6", reward=0.0, flagged=False),
                ],
                "banking-sha",
            ),
            record(
                "retail",
                [task("R0")],
                [variant("R0", "P3", reward=1.0, flagged=True)],
                "retail-sha",
            ),
        ]
        out, metrics = self.run_compare([], new)
        text = out.read_text(encoding="utf-8")
        self.assertIn("H2: held", text)
        self.assertEqual(metrics["h2_held"], 1)
        self.assertEqual(metrics["f1_pass_airline_new"], 20)
        self.assertEqual(metrics["f3_pass_rate_new"], 0.0)
        self.assertEqual(metrics["f6_pass_share_new"], 0.5)


if __name__ == "__main__":
    unittest.main()
