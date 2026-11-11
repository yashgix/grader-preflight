"""Tests for preflight.trajectories: the results format, the replay classes and the phase 3 triggers."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preflight import trajectories
from preflight.relations import WRITE, build_variants, f6_late_verification, p3_respelled_numbers, tool_kind
from preflight.tau2_adapter import Call, fresh_env, gold_calls, grade, tasks

BANK = "banking_knowledge"
AIRLINE = "airline"
POOL_PATH = Path("data/banking_pool.json")


def _pool_tasks(limit=None):
    """This env's tasks for the pre-registered banking pool, in pool order."""
    if not POOL_PATH.is_file():
        return []
    known = {task.id: task for task in tasks(BANK)}
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))["tasks"]
    picked = [known[str(task_id)] for task_id in pool if str(task_id) in known]
    return picked[:limit] if limit is not None else picked


def _short_task(domain):
    candidates = [task for task in tasks(domain) if gold_calls(task)]
    return min(candidates, key=lambda task: len(gold_calls(task))) if candidates else None


def _verification_task(scan=30):
    """A banking task whose log_verification comes before every write it gates."""
    for task in tasks(BANK)[:scan]:
        gold = gold_calls(task)
        starts = [index for index, call in enumerate(gold) if call.name == "log_verification"]
        if not starts:
            continue
        env = fresh_env(BANK, task)
        kinds = [tool_kind(env, call) for call in gold]
        writes = [index for index, kind in enumerate(kinds) if kind == WRITE and gold[index].name != "log_verification"]
        if writes and min(writes) > starts[0]:
            return task
    return None


def _messages_for(calls):
    """One speaker message and one tool message per call, built without a live env."""
    messages = []
    for index, call in enumerate(calls):
        role = "user" if call.requestor == "user" else "assistant"
        tool_call = {
            "id": f"call_{index}",
            "name": call.name,
            "arguments": call.arguments,
            "requestor": call.requestor,
        }
        messages.append({"role": role, "content": None, "tool_calls": [tool_call]})
        messages.append(
            {
                "role": "tool",
                "id": f"call_{index}",
                "content": "{}",
                "error": False,
                "requestor": call.requestor,
            }
        )
    return messages


def _write_results(path, domain, task_dict, simulations, git_commit="test"):
    payload = {
        "info": {"environment_info": {"domain_name": domain}, "git_commit": git_commit},
        "tasks": [task_dict],
        "simulations": simulations,
    }
    path = Path(path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _integer_value(value, declared):
    return isinstance(value, int) and not isinstance(value, bool) and trajectories._integer_parameter(declared)


def _as_float_calls(domain, run_calls):
    """The same calls with every integer passed to an integer-schema parameter made a whole-number float."""
    result = []
    for call in run_calls:
        tool = trajectories._tool_for(domain, call, trajectories._tool_name(call))
        properties = (trajectories._schema(tool).get("properties") or {}) if tool is not None else {}
        arguments = dict(call.arguments or {})
        if call.name in trajectories._CALL_WRAPPERS:
            inner = dict(trajectories._inner_args(call))
            changed = False
            for name, value in inner.items():
                if _integer_value(value, properties.get(name)):
                    inner[name] = float(value)
                    changed = True
            if changed:
                arguments["arguments"] = json.dumps(inner, sort_keys=True)
        else:
            for name, value in list(arguments.items()):
                if _integer_value(value, properties.get(name)):
                    arguments[name] = float(value)
        result.append(Call(call.name, arguments, call.requestor))
    return result


class _ToolMessage:
    def __init__(self, content="{}", error=False):
        self.content = content
        self.error = error


class FormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = AIRLINE
        cls.task = _short_task(AIRLINE)
        if cls.task is None:
            return
        cls.gold = gold_calls(cls.task)
        cls.simulation = trajectories.as_simulation(cls.domain, cls.task, cls.gold, trial=3)
        cls.trimmed = trajectories.as_simulation(cls.domain, cls.task, cls.gold, trial=1, trimmed=True)

    def setUp(self):
        if getattr(self, "task", None) is None:
            self.skipTest("no airline task with a gold run")

    def test_one_message_pair_per_call(self):
        messages = self.simulation["messages"]
        self.assertEqual(self.simulation["id"], f"{self.domain}:{self.task.id}:3")
        self.assertEqual(self.simulation["task_id"], self.task.id)
        self.assertEqual(self.simulation["trial"], 3)
        self.assertEqual(len(messages), 2 * len(self.gold))
        for index, call in enumerate(self.gold):
            speaker, answer = messages[2 * index], messages[2 * index + 1]
            self.assertEqual(speaker["role"], "user" if call.requestor == "user" else "assistant", call.name)
            (tool_call,) = speaker["tool_calls"]
            self.assertEqual(tool_call["id"], f"call_{index}")
            self.assertEqual(tool_call["name"], call.name)
            self.assertEqual(tool_call["arguments"], call.arguments)
            self.assertEqual(tool_call["requestor"], call.requestor)
            self.assertEqual(answer["role"], "tool")
            self.assertEqual(answer["id"], f"call_{index}")
            self.assertIn("error", answer)
            self.assertIsInstance(answer["requestor"], str)
        reward_info = self.simulation["reward_info"]
        self.assertEqual(reward_info["db_check"]["db_reward"], reward_info["reward"])

    def test_trimmed_copy_drops_id_and_error(self):
        self.assertEqual(len(self.trimmed["messages"]), len(self.simulation["messages"]))
        for full, thin in zip(self.simulation["messages"], self.trimmed["messages"]):
            self.assertEqual(thin["role"], full["role"])
            if full["role"] != "tool":
                self.assertEqual(thin["tool_calls"], full["tool_calls"])
                continue
            self.assertNotIn("id", thin)
            self.assertNotIn("error", thin)
            self.assertEqual(thin["content"], full["content"])

    def test_round_trip_restores_tool_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_results(
                Path(temporary) / "trimmed.json",
                self.domain,
                self.task.model_dump(mode="json"),
                [self.trimmed],
                git_commit="abc123",
            )
            loaded = trajectories.load(path)
        self.assertEqual(loaded["domain"], self.domain)
        self.assertEqual(loaded["git_commit"], "abc123")
        self.assertIn(str(self.task.id), loaded["tasks"])
        (record,) = loaded["runs"]
        self.assertEqual(record["id"], f"{self.domain}:{self.task.id}:1")
        self.assertEqual(record["recorded_reward"], self.trimmed["reward_info"]["reward"])
        tool_messages = [message for message in record["messages"] if message.get("role") == "tool"]
        self.assertEqual(
            [message["id"] for message in tool_messages], [f"call_{index}" for index in range(len(self.gold))]
        )


class RestoreIdsTest(unittest.TestCase):
    def test_a_missing_id_takes_the_earliest_unanswered_call(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "name": "x", "arguments": {}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "name": "y", "arguments": {}}]},
            {"role": "tool", "content": "1"},
            {"role": "tool", "content": "2"},
        ]
        restored = trajectories.restore_ids(messages)
        self.assertEqual([message.get("id") for message in restored[2:]], ["a", "b"])
        self.assertNotIn("id", messages[2])

    def test_an_existing_id_is_kept_and_leaves_the_queue(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "name": "x", "arguments": {}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "name": "y", "arguments": {}}]},
            {"role": "tool", "id": "b", "content": "1"},
            {"role": "tool", "content": "2"},
        ]
        restored = trajectories.restore_ids(messages)
        self.assertEqual([message.get("id") for message in restored[2:]], ["b", "a"])

    def test_an_empty_list_is_unchanged(self):
        self.assertEqual(trajectories.restore_ids([]), [])


class CallsTest(unittest.TestCase):
    def test_requestor_comes_from_the_call_else_the_message_role(self):
        messages = [
            {"role": "user", "content": None, "tool_calls": [{"name": "check", "arguments": {"a": 1}}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "read", "arguments": {}, "requestor": "user"}],
            },
            {"role": "assistant", "content": None, "tool_calls": [{"name": "write", "arguments": {"b": 2}}]},
            {"role": "tool", "content": "{}"},
        ]
        found = trajectories.calls(messages)
        self.assertEqual([call.name for call in found], ["check", "read", "write"])
        self.assertEqual([call.requestor for call in found], ["user", "user", "assistant"])
        self.assertEqual(found[0].arguments, {"a": 1})
        self.assertEqual(trajectories.calls([]), [])


class TaskMatchTest(unittest.TestCase):
    def test_a_task_matches_its_own_dump_and_not_a_changed_one(self):
        task = tasks(AIRLINE)[0]
        data = task.model_dump(mode="json")
        self.assertTrue(trajectories.task_matches(data, task))
        changed = dict(data, id="another_task")
        self.assertFalse(trajectories.task_matches(changed, task))
        self.assertFalse(trajectories.task_matches(None, task))
        self.assertFalse(trajectories.task_matches(data, None))


class GoldReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pool = _pool_tasks(limit=1)
        cls.task = pool[0] if pool else None
        if cls.task is not None:
            cls.gold = gold_calls(cls.task)

    def setUp(self):
        if getattr(self, "task", None) is None:
            self.skipTest("no banking pool")

    def test_a_gold_run_replays_fully_and_is_graded_one(self):
        simulation = trajectories.as_simulation(BANK, self.task, self.gold)
        graded = trajectories.replay(BANK, self.task, {"messages": simulation["messages"]})
        self.assertTrue(graded["outputs_equal"])
        self.assertEqual(graded["mismatched"], [])
        self.assertEqual(graded["mode"], "strict")
        self.assertEqual(graded["replay_class"], "full")
        self.assertEqual(graded["reward"], 1.0)

    def test_the_gold_run_differs_from_itself_in_no_table(self):
        self.assertEqual(trajectories.diff_tables(BANK, self.task, self.gold), [])

    def test_the_gold_run_has_no_p1_trigger(self):
        self.assertFalse(trajectories.p1_trigger(BANK, self.task, self.gold))


class P1GradingTest(unittest.TestCase):
    """An extra agent read is a P1 trigger: punished by the old grader, ignored by the fixed one."""

    @staticmethod
    def _variant(task):
        variants, _ = build_variants("P1", BANK, task, random.Random(f"0:{task.id}:P1"))
        for variant in variants:
            if variant.detail.startswith("extra agent read") and not variant.inserted_in_gold:
                return variant
        return None

    def test_the_fixed_grader_ignores_an_extra_agent_read(self):
        if trajectories.version_label() != "new":
            self.skipTest("the old grader is checked by its own test")
        checked = 0
        for task in _pool_tasks(limit=8):
            variant = self._variant(task)
            if variant is None:
                continue
            self.assertTrue(trajectories.p1_trigger(BANK, task, variant.calls), task.id)
            self.assertEqual(grade(BANK, task, variant.calls), 1.0, f"{task.id}: the fixed grader punished a read")
            checked += 1
        self.assertGreater(checked, 0)

    def test_the_old_grader_punishes_an_extra_agent_read(self):
        if trajectories.version_label() != "old":
            self.skipTest("the fixed grader is checked by its own test")
        punished = []
        for task in _pool_tasks():
            variant = self._variant(task)
            if variant is None or grade(BANK, task, variant.calls) != 0.0:
                continue
            self.assertEqual(
                trajectories.diff_tables(BANK, task, variant.calls),
                ["agent_discoverable_tools"],
                task.id,
            )
            punished.append(task.id)
            break
        self.assertTrue(punished, "the old grader accepted every extra agent read of the pool")


class P3TriggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = None
        for domain in (BANK, AIRLINE, "retail"):
            for task in tasks(domain):
                variants = p3_respelled_numbers(domain, task, random.Random(0))
                if variants:
                    cls.found = (domain, task, variants[0])
                    return

    def test_a_respelled_number_is_a_trigger_and_the_gold_run_is_not(self):
        if self.found is None:
            self.skipTest("no gold run whose numbers can be respelled")
        _, task, variant = self.found
        self.assertTrue(trajectories.p3_trigger(task, variant.calls))
        self.assertFalse(trajectories.p3_trigger(task, gold_calls(task)))

    def test_a_run_without_numbers_has_no_trigger(self):
        for task in tasks(AIRLINE):
            calls = [Call("read", {"reservation_id": "ABC"})]
            self.assertFalse(trajectories.p3_trigger(task, calls))
            return
        self.skipTest("no airline tasks")


class SchemaFloatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = None
        for task in tasks(AIRLINE):
            gold = gold_calls(task)
            if not gold or trajectories.schema_floats(AIRLINE, gold):
                continue
            floated = _as_float_calls(AIRLINE, gold)
            if trajectories.schema_floats(AIRLINE, floated):
                cls.found = (task, gold, floated)
                return

    def test_whole_number_floats_are_found_on_integer_parameters(self):
        if self.found is None:
            self.skipTest("no airline gold run with an integer-schema number")
        _, gold, floated = self.found
        self.assertEqual(trajectories.schema_floats(AIRLINE, gold), [])
        found = trajectories.schema_floats(AIRLINE, floated)
        self.assertTrue(found)
        for item in found:
            self.assertEqual(set(item), {"index", "tool", "argument"})
            self.assertIsInstance(item["index"], int)
            self.assertIsInstance(item["tool"], str)

    def test_as_integers_restores_the_gold_arguments(self):
        if self.found is None:
            self.skipTest("no airline gold run with an integer-schema number")
        _, gold, floated = self.found
        restored = trajectories.as_integers(floated, trajectories.schema_floats(AIRLINE, floated))
        self.assertEqual(len(restored), len(gold))
        self.assertEqual(
            [trajectories._effective_arguments(call) for call in restored],
            [trajectories._effective_arguments(call) for call in gold],
        )


class NumericPairTest(unittest.TestCase):
    """The inconsistency rule, checked on the numbers rather than on what tau2 happens to do."""

    @classmethod
    def setUpClass(cls):
        cls.task, cls.floated = None, []
        for task in tasks(AIRLINE):
            floated = _as_float_calls(AIRLINE, gold_calls(task))
            if trajectories.schema_floats(AIRLINE, floated):
                cls.task, cls.floated = task, floated
                return

    def setUp(self):
        if self.task is None:
            self.skipTest("no airline gold run with an integer-schema number")

    def _outputs(self):
        return [_ToolMessage() for _ in range(8)]

    def _pair(self, outputs, rewards, state="state"):
        with (
            mock.patch("preflight.trajectories._live_outputs", return_value=outputs),
            mock.patch("preflight.trajectories._state_dump", return_value=state),
            mock.patch("preflight.trajectories._grade_run", side_effect=rewards),
        ):
            return trajectories.numeric_pair(AIRLINE, self.task, self.floated)

    def test_a_reward_difference_with_equal_outputs_and_state_is_an_inconsistency(self):
        result = self._pair(self._outputs(), [1.0, 0.0])
        self.assertTrue(result["found"])
        self.assertFalse(result["errors_at_found"])
        self.assertTrue(result["outputs_equal"])
        self.assertTrue(result["states_equal"])
        self.assertEqual((result["reward_float"], result["reward_int"]), (1.0, 0.0))
        self.assertTrue(result["inconsistency"])

    def test_equal_rewards_are_not_an_inconsistency(self):
        self.assertFalse(self._pair(self._outputs(), [1.0, 1.0])["inconsistency"])

    def test_a_tool_error_at_a_respelled_call_is_not_an_inconsistency(self):
        found = trajectories.schema_floats(AIRLINE, self.floated)
        outputs = self._outputs()
        outputs[found[0]["index"]] = _ToolMessage("Error: bad argument", error=True)
        result = self._pair(outputs, [1.0, 0.0])
        self.assertTrue(result["errors_at_found"])
        self.assertFalse(result["inconsistency"])

    def test_a_different_end_state_is_not_an_inconsistency(self):
        with (
            mock.patch("preflight.trajectories._live_outputs", return_value=self._outputs()),
            mock.patch("preflight.trajectories._state_dump", side_effect=["left", "right"]),
            mock.patch("preflight.trajectories._grade_run", side_effect=[1.0, 0.0]),
        ):
            result = trajectories.numeric_pair(AIRLINE, self.task, self.floated)
        self.assertFalse(result["states_equal"])
        self.assertFalse(result["inconsistency"])

    def test_different_outputs_are_not_an_inconsistency(self):
        with (
            mock.patch(
                "preflight.trajectories._live_outputs",
                side_effect=[[_ToolMessage('{"a": 1}')], [_ToolMessage('{"a": 2}')]],
            ),
            mock.patch("preflight.trajectories._state_dump", return_value="state"),
            mock.patch("preflight.trajectories._grade_run", side_effect=[1.0, 0.0]),
        ):
            result = trajectories.numeric_pair(AIRLINE, self.task, self.floated)
        self.assertFalse(result["outputs_equal"])
        self.assertFalse(result["inconsistency"])


class LateVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = _verification_task()
        if cls.task is None:
            return
        cls.gold = gold_calls(cls.task)
        cls.variants = f6_late_verification(BANK, cls.task, random.Random(0))

    def setUp(self):
        if getattr(self, "task", None) is None:
            self.skipTest("no banking task whose verification gates its writes")

    def test_the_gold_run_logs_verification_before_its_writes(self):
        run = {"domain": BANK, "messages": _messages_for(self.gold)}
        found = trajectories.late_verification(self.task, run)
        self.assertFalse(found["log_after_first_write"])
        self.assertIsNone(found["access_before_facts"])

    def test_a_late_verification_variant_is_flagged(self):
        self.assertTrue(self.variants, "the gold run has no gated verification to move")
        run = {"domain": BANK, "messages": _messages_for(self.variants[0].calls)}
        self.assertTrue(trajectories.late_verification(self.task, run)["log_after_first_write"])

    def test_a_task_without_a_log_verification_has_both_flags_false(self):
        found = trajectories.late_verification(tasks(AIRLINE)[0], {"domain": AIRLINE, "messages": []})
        self.assertEqual(found, {"log_after_first_write": False, "access_before_facts": False})


class FlagsTest(unittest.TestCase):
    def test_flags_lists_flagged_tasks_and_skips_reads_the_gold_run_makes(self):
        variants = [
            {"task_id": "task_001", "relation": "P1", "flagged": True, "inserted_in_gold": False},
            {"task_id": "task_002", "relation": "P1", "flagged": True, "inserted_in_gold": True},
            {"task_id": "task_002", "relation": "P1", "flagged": True},
            {"task_id": "task_001", "relation": "P3", "flagged": True},
            {"task_id": "task_003", "relation": "P1", "flagged": False, "inserted_in_gold": False},
            {"task_id": "task_004", "relation": "F1", "flagged": True},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old.json"
            source.write_text(json.dumps({"variants": variants}), encoding="utf-8")
            out = Path(temporary) / "flags.json"
            payload = trajectories.flags([source], out)
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload, written)
        self.assertEqual(written, {"P1": ["task_001", "task_002"], "P3": ["task_001"]})


class RegradeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pool = _pool_tasks(limit=1)
        cls.task = pool[0] if pool else None

    def setUp(self):
        if self.task is None:
            self.skipTest("no banking pool")

    def test_regrade_writes_one_record_per_run_with_the_version_label(self):
        gold = gold_calls(self.task)
        simulation = trajectories.as_simulation(BANK, self.task, gold, trial=0)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = _write_results(folder / "banking.json", BANK, self.task.model_dump(mode="json"), [simulation])
            written = trajectories.regrade([source], folder / "regrade")
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].name, f"banking.{trajectories.version_label()}.jsonl")
            lines = written[0].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["domain"], BANK)
        self.assertEqual(record["task_id"], self.task.id)
        self.assertEqual(record["trial"], 0)
        self.assertTrue(record["task_matches"])
        self.assertEqual(record["replay_class"], "full")
        self.assertEqual(record["reward"], 1.0)
        self.assertEqual(record["recorded_reward"], 1.0)
        self.assertEqual(record["diff_tables"], [])
        self.assertEqual(record["n_calls"], len(gold))
        self.assertFalse(record["p1"])
        self.assertFalse(record["p3"])
        self.assertIsInstance(record["late_verification"], dict)
        self.assertFalse(record["late_verification"]["log_after_first_write"])

    def test_a_run_whose_task_differs_gets_no_verdict(self):
        simulation = {"id": f"{BANK}:{self.task.id}:0", "task_id": self.task.id, "trial": 0, "messages": []}
        data = dict(self.task.model_dump(mode="json"), id="another_task")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = _write_results(folder / "banking.json", BANK, data, [simulation])
            written = trajectories.regrade([source], folder / "regrade")
            record = json.loads(written[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertFalse(record["task_matches"])
        self.assertIsNone(record["replay_class"])
        self.assertIsNone(record["reward"])
        self.assertIsNone(record["p1"])
        self.assertIsNone(record["late_verification"])


class VersionLabelTest(unittest.TestCase):
    def test_the_label_names_the_grader_version(self):
        self.assertIn(trajectories.version_label(), {"old", "new"})


class ReviewRegressionTest(unittest.TestCase):
    def test_task_match_ignores_key_order(self):
        task = next(t for t in tasks(BANK) if t.id == "task_005")
        shuffled = json.loads(json.dumps(json.loads(task.model_dump_json()), sort_keys=True))
        self.assertTrue(trajectories.task_matches(shuffled, task))

    def test_spelling_differs_needs_equal_values(self):
        self.assertTrue(trajectories._spelling_differs([1, 2], [1.0, 2]))
        self.assertFalse(trajectories._spelling_differs([1, 2], [1.0, 3]))
        self.assertFalse(trajectories._spelling_differs({"a": 1, "b": 2}, {"a": 1.0, "b": 3}))

    def test_regrade_refuses_two_files_with_one_name(self):
        task = next(t for t in tasks(BANK) if t.id == "task_005")
        sim = trajectories.as_simulation(BANK, task, gold_calls(task))
        payload = {
            "info": {"environment_info": {"domain_name": BANK}},
            "tasks": [json.loads(task.model_dump_json())],
            "simulations": [sim],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sub in ("a", "b"):
                (root / sub).mkdir()
                (root / sub / "banking.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                trajectories.regrade([root / "a" / "banking.json", root / "b" / "banking.json"], root / "out")


if __name__ == "__main__":
    unittest.main()
