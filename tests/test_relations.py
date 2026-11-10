"""Tests for the relations. Every assertion is structural, so it holds under both grader versions."""

import json
import random
import unittest
from unittest import mock

from preflight.relations import (
    RELATIONS,
    _parameters,
    build_variants,
    calls_as_dicts,
    canonical_json,
    db_hashes,
    p1_extra_read,
    p3_respelled_numbers,
    static_checks,
    tool_error,
    tool_kind,
)
from preflight.tau2_adapter import DOMAINS, Call, fresh_env, gold_calls, run, tasks

NAME_KEY = {
    "call_discoverable_agent_tool": "agent_tool_name",
    "call_discoverable_user_tool": "discoverable_tool_name",
}
DECLARED = {"read", "write", "generic", "think"}


def variants_of(relation, domain, task, seed=0):
    return build_variants(relation, domain, task, random.Random(f"{seed}:{task.id}:{relation}"))


def toolkit_names(env, attr):
    toolkit = getattr(env, attr, None)
    if toolkit is None:
        return []
    getter = getattr(toolkit, "get_tools", None)
    if callable(getter):
        return list(getter())
    if isinstance(toolkit, dict):
        return list(toolkit)
    return [tool.name for tool in toolkit]


def numbers(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for key in sorted(value) for number in numbers(value[key])]
    if isinstance(value, list):
        return [number for item in value for number in numbers(item)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return numbers(parsed) if isinstance(parsed, (dict, list)) else []
    return []


def call_numbers(calls):
    return [number for call in calls for number in numbers(call.arguments)]


def same_calls(left, right):
    return canonical_json(calls_as_dicts(left)) == canonical_json(calls_as_dicts(right))


def read_positions(gold, calls):
    """Positions of the inserted read in a variant that is the gold run plus one inserted block."""
    positions = []
    for index in range(len(calls)):
        for size in (1, 2):
            if index + size > len(calls):
                continue
            if same_calls(calls[:index], gold[:index]) and same_calls(calls[index + size :], gold[index:]):
                positions.append(index + size - 1)
    if not positions:
        raise AssertionError("the variant is not the gold run plus one inserted block")
    return positions


def argument_name_values(calls):
    values = {}
    for call in calls:
        arguments = dict(call.arguments or {})
        raw = arguments.pop("arguments", None)
        pairs = list(arguments.items())
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                pairs += list(parsed.items())
        for name, value in pairs:
            values.setdefault(name, set()).add(canonical_json(value))
    return values


def short_task_with_calls(domain, limit=10):
    candidates = [task for task in tasks(domain) if gold_calls(task)][:limit]
    return min(candidates, key=lambda task: len(gold_calls(task))) if candidates else None


def find_task_with_a_write(domain="airline", max_calls=8, scan=25):
    for task in tasks(domain)[:scan]:
        gold = gold_calls(task)
        if not gold or len(gold) > max_calls:
            continue
        env = fresh_env(domain, task)
        if any(tool_kind(env, call) == "write" for call in gold):
            return task
    return None


def find_verification_task(scan=30):
    domain = "banking_knowledge"
    for task in tasks(domain)[:scan]:
        gold = gold_calls(task)
        positions = [index for index, call in enumerate(gold) if call.name == "log_verification"]
        if not positions:
            continue
        env = fresh_env(domain, task)
        if any(tool_kind(env, later) == "write" for later in gold[positions[0] + 1 :]):
            return task
    return None


class _HashEnv:
    """A stand-in for a live environment: only the two documented hash methods."""

    def __init__(self, agent_hash, user_hash):
        self.agent_hash = agent_hash
        self.user_hash = user_hash

    def get_db_hash(self):
        return self.agent_hash

    def get_user_db_hash(self):
        return self.user_hash


class _Message:
    def __init__(self, content=None, error=False):
        self.content = content
        self.error = error


class RelationsTableTest(unittest.TestCase):
    def test_all_relations_are_registered(self):
        self.assertEqual(sorted(RELATIONS), ["F1", "F3", "F6", "P1", "P3", "P5"])


class ToolKindTest(unittest.TestCase):
    def test_unknown_names_are_unknown(self):
        env = fresh_env("airline", tasks("airline")[0])
        self.assertEqual(tool_kind(env, Call("no_such_tool", {})), "unknown")
        wrapped = Call("call_discoverable_agent_tool", {"agent_tool_name": "no_such_tool", "arguments": "{}"})
        self.assertEqual(tool_kind(env, wrapped), "unknown")

    def test_declared_tools_resolve(self):
        for domain in DOMAINS:
            env = fresh_env(domain, tasks(domain)[0])
            agent_tools = [name for name in toolkit_names(env, "tools") if name not in NAME_KEY]
            for name in agent_tools[:3]:
                self.assertIn(tool_kind(env, Call(name, {})), DECLARED, f"{domain} {name}")
            user_tools = [name for name in toolkit_names(env, "user_tools") if name not in NAME_KEY]
            for name in user_tools[:3]:
                self.assertIn(tool_kind(env, Call(name, {}, "user")), DECLARED, f"{domain} {name}")

    def test_wrapper_resolves_to_the_inner_tool(self):
        pairs = [("call_discoverable_agent_tool", "tools"), ("call_discoverable_user_tool", "user_tools")]
        for domain in DOMAINS:
            env = fresh_env(domain, tasks(domain)[0])
            for wrapper, target in pairs:
                names = toolkit_names(env, target)
                if wrapper not in names:
                    continue
                inner = next((name for name in names if name != wrapper), None)
                if inner is None:
                    continue
                requestor = "user" if target == "user_tools" else "assistant"
                wrapped = Call(wrapper, {NAME_KEY[wrapper]: inner, "arguments": "{}"})
                self.assertEqual(tool_kind(env, wrapped), tool_kind(env, Call(inner, {}, requestor)))


class ParametersTest(unittest.TestCase):
    def test_openai_schema_supplies_the_required_list(self):
        class _Tool:
            parameters = {"a": "first", "b": "second"}
            openai_schema = {
                "function": {"parameters": {"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a"]}}
            }

        self.assertEqual(_parameters(_Tool()), (["a", "b"], ["a"]))

    def test_a_plain_mapping_falls_back_to_the_signature(self):
        def _call(account_id, note="none"):
            return account_id, note

        class _Tool:
            parameters = {"account_id": "the account", "note": "an optional note"}
            func = staticmethod(_call)

        self.assertEqual(_parameters(_Tool()), (["account_id", "note"], ["account_id"]))

    def test_nothing_known_means_every_argument_is_required(self):
        class _Tool:
            params = {"account_id": "the account"}

        self.assertEqual(_parameters(_Tool()), (["account_id"], ["account_id"]))

    def test_a_bound_method_supplies_its_signature(self):
        class _Discoverable:
            def get_details(self, account_id, note="none"):
                return account_id, note

        self.assertEqual(_parameters(_Discoverable().get_details), (["account_id", "note"], ["account_id"]))


class P1CandidateTest(unittest.TestCase):
    def test_failing_candidates_do_not_take_a_variant_slot(self):
        plans = [("direct", f"read{index}", 0, {}) for index in range(8)]
        failing = {"read0", "read1", "read2"}

        def ran_cleanly(domain, task, calls, position):
            return calls[position].name not in failing

        with mock.patch("preflight.relations.gold_calls", return_value=[Call("write_a", {})]):
            with mock.patch("preflight.relations._p1_plans", return_value=plans):
                with mock.patch("preflight.relations._ran_cleanly", side_effect=ran_cleanly):
                    variants = p1_extra_read("airline", object(), random.Random(0))

        self.assertEqual(len(variants), 5)
        for variant in variants:
            self.assertNotIn(variant.calls[0].name, failing)

    def test_candidates_are_shuffled_with_the_seed(self):
        plans = [("direct", f"read{index}", 0, {}) for index in range(8)]
        expected = list(plans)
        random.Random(0).shuffle(expected)

        with mock.patch("preflight.relations.gold_calls", return_value=[Call("write_a", {})]):
            with mock.patch("preflight.relations._p1_plans", return_value=list(plans)):
                with mock.patch("preflight.relations._ran_cleanly", return_value=True):
                    variants = p1_extra_read("airline", object(), random.Random(0))

        inserted = [variant.calls[0].name for variant in variants]
        self.assertEqual(inserted, [plan[1] for plan in expected[:5]])

    def test_an_already_unlocked_read_gets_no_second_unlock(self):
        gold = [
            Call("unlock_discoverable_agent_tool", {"agent_tool_name": "get_details"}),
            Call("get_details", {"account_id": "A1"}),
        ]
        plans = [("agent", "get_details", 1, {"account_id": "A1"})]
        with mock.patch("preflight.relations.gold_calls", return_value=gold):
            with mock.patch("preflight.relations._p1_plans", return_value=plans):
                with mock.patch("preflight.relations._ran_cleanly", return_value=True):
                    variants = p1_extra_read("banking_knowledge", object(), random.Random(0))

        self.assertEqual(len(variants), 1)
        calls = variants[0].calls
        self.assertEqual(len(calls), len(gold) + 1)
        self.assertEqual(sum(1 for call in calls if call.name == "unlock_discoverable_agent_tool"), 1)
        inserted = calls[1]
        self.assertEqual(inserted.name, "call_discoverable_agent_tool")
        self.assertEqual(inserted.arguments["agent_tool_name"], "get_details")
        self.assertEqual(json.loads(inserted.arguments["arguments"]), {"account_id": "A1"})
        self.assertTrue(variants[0].inserted_in_gold)

    def test_an_unseen_discoverable_read_gets_its_unlock_first(self):
        gold = [Call("get_account", {"account_id": "A1"})]
        plans = [("agent", "get_details", 1, {"account_id": "A1"})]
        with mock.patch("preflight.relations.gold_calls", return_value=gold):
            with mock.patch("preflight.relations._p1_plans", return_value=plans):
                with mock.patch("preflight.relations._ran_cleanly", return_value=True):
                    variants = p1_extra_read("banking_knowledge", object(), random.Random(0))

        self.assertEqual(len(variants), 1)
        calls = variants[0].calls
        self.assertEqual(len(calls), len(gold) + 2)
        self.assertEqual(calls[1].name, "unlock_discoverable_agent_tool")
        self.assertEqual(calls[1].arguments, {"agent_tool_name": "get_details"})
        self.assertEqual(calls[2].name, "call_discoverable_agent_tool")
        self.assertFalse(variants[0].inserted_in_gold)


class P1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = "airline"
        cls.task = short_task_with_calls(cls.domain)
        if cls.task is None:
            return
        cls.gold = gold_calls(cls.task)
        cls.variants, cls.notes = variants_of("P1", cls.domain, cls.task)

    def setUp(self):
        if getattr(self, "task", None) is None:
            self.skipTest("no airline task with a gold run")

    def test_at_most_five_variants_that_expect_a_pass(self):
        self.assertLessEqual(len(self.variants), 5)
        for variant in self.variants:
            self.assertEqual(variant.relation, "P1")
            self.assertTrue(variant.expected_pass)
            self.assertIn(len(variant.calls), (len(self.gold) + 1, len(self.gold) + 2))
            self.assertFalse(same_calls(variant.calls, self.gold))

    def test_inserted_call_runs_cleanly(self):
        if not self.variants:
            self.skipTest("no P1 candidate ran cleanly on this task")
        variant = self.variants[0]
        clean = []
        for index in read_positions(self.gold, variant.calls):
            try:
                messages, _ = run(self.domain, self.task, list(variant.calls[: index + 1]))
            except Exception:
                clean.append(False)
                continue
            clean.append(bool(messages) and not tool_error(messages[-1]))
        self.assertTrue(any(clean), "no prefix through a possible inserted call ran cleanly")

    def test_inserted_arguments_come_from_earlier_calls(self):
        if not self.variants:
            self.skipTest("no P1 candidate ran cleanly on this task")
        variant = self.variants[0]
        accepted = False
        for index in read_positions(self.gold, variant.calls):
            inserted = variant.calls[index]
            if inserted.name in NAME_KEY:
                try:
                    inner = json.loads(inserted.arguments["arguments"])
                except (TypeError, ValueError, KeyError):
                    continue
            else:
                inner = inserted.arguments
            if not isinstance(inner, dict):
                continue
            available = argument_name_values(variant.calls[:index])
            if all(canonical_json(value) in available.get(name, set()) for name, value in inner.items()):
                accepted = True
                break
        self.assertTrue(accepted, "inserted arguments were not filled from same-named earlier arguments")

    def test_same_seed_gives_the_same_variants(self):
        again, _ = variants_of("P1", self.domain, self.task)
        self.assertEqual(
            canonical_json([calls_as_dicts(variant.calls) for variant in again]),
            canonical_json([calls_as_dicts(variant.calls) for variant in self.variants]),
        )


class P3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = None
        for domain in DOMAINS:
            for task in tasks(domain):
                if not call_numbers(gold_calls(task)):
                    continue
                variants, notes = variants_of("P3", domain, task)
                if variants:
                    cls.found = (domain, task)
                    cls.variants = variants
                    cls.notes = notes
                    break
            if cls.found:
                break

    def test_one_variant_with_the_same_numbers(self):
        if not self.found:
            self.skipTest("no gold run whose numbers can be respelled")
        domain, task = self.found
        gold = gold_calls(task)
        self.assertEqual(self.notes, {})
        self.assertEqual(len(self.variants), 1)
        variant = self.variants[0]
        self.assertTrue(variant.expected_pass)
        self.assertEqual(len(variant.calls), len(gold))
        self.assertFalse(same_calls(variant.calls, gold))
        self.assertEqual(call_numbers(variant.calls), call_numbers(gold))

    def test_no_variant_when_nothing_changes(self):
        for domain in DOMAINS:
            for task in tasks(domain):
                if call_numbers(gold_calls(task)):
                    continue
                variants, notes = variants_of("P3", domain, task)
                self.assertEqual(variants, [])
                self.assertTrue(notes.get("identical_to_gold"))
                return
        self.skipTest("every gold run has numeric arguments")

    def test_numbers_inside_a_wrapper_arguments_string_are_respelled(self):
        arguments = json.dumps({"account_id": "A1", "amount": 25, "note": "25"}, sort_keys=True)
        gold = [Call("call_discoverable_agent_tool", {"agent_tool_name": "get_details", "arguments": arguments})]
        with mock.patch("preflight.relations.gold_calls", return_value=gold):
            variants = p3_respelled_numbers("banking_knowledge", object(), random.Random(0))

        self.assertEqual(len(variants), 1)
        self.assertTrue(variants[0].expected_pass)
        respelled = variants[0].calls[0].arguments["arguments"]
        self.assertIsInstance(respelled, str)
        self.assertNotEqual(respelled, arguments)
        self.assertEqual(json.loads(respelled), json.loads(arguments))
        self.assertIn("25.0", respelled)
        self.assertIn('"25"', respelled)

    def test_a_direct_tools_arguments_string_is_not_respelled(self):
        arguments = json.dumps({"amount": 25, "note": "25"}, sort_keys=True)
        gold = [Call("direct_tool", {"account_id": "A1", "arguments": arguments, "amount": 25})]
        with mock.patch("preflight.relations.gold_calls", return_value=gold):
            variants = p3_respelled_numbers("banking_knowledge", object(), random.Random(0))

        self.assertEqual(len(variants), 1)
        self.assertTrue(variants[0].expected_pass)
        respelled = variants[0].calls[0].arguments
        self.assertEqual(respelled["arguments"], arguments)
        self.assertEqual(respelled["amount"], 25.0)


class P5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = "airline"
        cls.task = short_task_with_calls(cls.domain)
        cls.variants, cls.notes = variants_of("P5", cls.domain, cls.task) if cls.task else ([], {})

    def test_at_most_one_unknown_call(self):
        if self.task is None:
            self.skipTest("no airline task with a gold run")
        self.assertLessEqual(len(self.variants), 1)
        if not self.variants:
            self.assertTrue(self.notes.get("p5_not_inert"))
            return
        variant = self.variants[0]
        self.assertTrue(variant.expected_pass)
        self.assertFalse(variant.inserted_in_gold)
        inserted = [call for call in variant.calls if call.name == "no_such_tool"]
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].arguments, {})
        self.assertEqual(inserted[0].requestor, "assistant")


class P5InertnessTest(unittest.TestCase):
    def _build(self, before, after):
        responses = [([], _HashEnv(*before)), ([], _HashEnv(*after))]
        with mock.patch("preflight.relations.gold_calls", return_value=[Call("some_call", {})]):
            with mock.patch("preflight.relations.run", side_effect=responses):
                return build_variants("P5", "airline", object(), random.Random(0))

    def test_inert_unknown_call_keeps_the_variant(self):
        variants, notes = self._build(("agent", "user"), ("agent", "user"))
        self.assertEqual(len(variants), 1)
        self.assertEqual(notes, {})

    def test_agent_only_mutation_drops_the_variant_and_sets_note(self):
        variants, notes = self._build(("agent", "user"), ("agent-changed", "user"))
        self.assertEqual(variants, [])
        self.assertTrue(notes.get("p5_not_inert"))

    def test_user_only_mutation_drops_the_variant_and_sets_note(self):
        variants, notes = self._build(("agent", "user"), ("agent", "user-changed"))
        self.assertEqual(variants, [])
        self.assertTrue(notes.get("p5_not_inert"))

    def test_a_failed_inertness_check_is_not_reported_as_inert(self):
        with mock.patch("preflight.relations.gold_calls", return_value=[Call("some_call", {})]):
            with mock.patch("preflight.relations.run", side_effect=RuntimeError("replay")):
                with self.assertRaises(RuntimeError):
                    build_variants("P5", "airline", object(), random.Random(0))


class F1Test(unittest.TestCase):
    def test_empty_run_is_one_failing_variant(self):
        task = short_task_with_calls("airline")
        if task is None:
            self.skipTest("no airline task with a gold run")
        variants, notes = variants_of("F1", "airline", task)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].calls, ())
        self.assertFalse(variants[0].expected_pass)
        self.assertEqual(notes, {})

    def test_no_variant_when_the_gold_run_is_empty(self):
        for domain in DOMAINS:
            for task in tasks(domain):
                if gold_calls(task):
                    continue
                variants, notes = variants_of("F1", domain, task)
                self.assertEqual(variants, [])
                self.assertTrue(notes.get("identical_to_gold"))
                return
        self.skipTest("no task with an empty gold run")


class F3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = "airline"
        cls.task = find_task_with_a_write(cls.domain)

    def test_only_writes_that_change_the_end_state_are_dropped(self):
        if self.task is None:
            self.skipTest("no short airline task with a write")
        gold = gold_calls(self.task)
        env = fresh_env(self.domain, self.task)
        writes = [index for index, call in enumerate(gold) if tool_kind(env, call) == "write"]
        variants, notes = variants_of("F3", self.domain, self.task)
        checks = static_checks(self.domain, self.task)
        self.assertEqual(notes, {})
        self.assertLessEqual(len(variants), len(writes))
        self.assertEqual(len(variants) + checks["no_effect_writes"], len(writes))
        for variant in variants:
            self.assertFalse(variant.expected_pass)
            self.assertEqual(len(variant.calls), len(gold) - 1)


class F6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = "banking_knowledge"
        cls.task = find_verification_task()

    def test_log_verification_moves_to_the_end(self):
        if self.task is None:
            self.skipTest("no banking task with a gated gold run")
        gold = gold_calls(self.task)
        variants, notes = variants_of("F6", self.domain, self.task)
        self.assertEqual(notes, {})
        self.assertGreaterEqual(len(variants), 1)
        for variant in variants:
            self.assertFalse(variant.expected_pass)
            self.assertEqual(len(variant.calls), len(gold))
            self.assertEqual(variant.calls[-1].name, "log_verification")

    def test_other_domains_have_no_late_verification_variants(self):
        task = short_task_with_calls("airline")
        if task is None:
            self.skipTest("no airline task with a gold run")
        self.assertEqual(variants_of("F6", "airline", task), ([], {}))


class StaticChecksTest(unittest.TestCase):
    def test_static_checks_shape(self):
        checked = 0
        for domain in DOMAINS:
            for task in tasks(domain)[:3]:
                try:
                    checks = static_checks(domain, task)
                except Exception:  # a task the live env cannot replay is not this test's subject
                    continue
                self.assertEqual(set(checks), {"gold_errors", "gold_changes_db", "no_effect_writes"})
                self.assertIsInstance(checks["gold_errors"], int)
                self.assertIsInstance(checks["gold_changes_db"], bool)
                self.assertIsInstance(checks["no_effect_writes"], int)
                checked += 1
                break
        if not checked:
            self.skipTest("no task replayed cleanly in any domain")

    def test_gold_errors_counts_flagged_and_wrapper_errors(self):
        messages = [_Message(), _Message(error=True), _Message(), _Message(content="Error: no such tool")]
        with mock.patch("preflight.relations.gold_calls", return_value=[Call("a", {}), Call("b", {})]):
            with mock.patch("preflight.relations.run", return_value=(messages, _HashEnv("agent", "user"))):
                checks = static_checks("airline", tasks("airline")[0])

        self.assertEqual(checks["gold_errors"], 2)
        self.assertTrue(checks["gold_changes_db"])
        self.assertEqual(checks["no_effect_writes"], 0)


class DbHashesTest(unittest.TestCase):
    def test_initial_state_has_two_entries(self):
        hashes = db_hashes(fresh_env("airline", tasks("airline")[0]))
        self.assertEqual(len(hashes), 2)
        self.assertTrue(all(entry is None or isinstance(entry, str) for entry in hashes))


if __name__ == "__main__":
    unittest.main()
