"""Tests for the phase 2 MDP: actions, features, trajectory, ordering and oracle."""

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from preflight import mdp
from preflight.mdp import (
    Action,
    allowed,
    features,
    horizon,
    oracle,
    order_ok,
    task_actions,
    trajectory,
)
from preflight.relations import WRAPPER_ARGUMENTS
from preflight.rl import split
from preflight.tau2_adapter import Call, gold_calls, tasks

_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _json_number_tokens(text: str) -> list[str]:
    tokens = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char in "-0123456789":
            match = _JSON_NUMBER.match(text, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
        index += 1
    return tokens


def _value_number_tokens(value) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, float):
        return [repr(value)]
    if isinstance(value, list):
        return [token for item in value for token in _value_number_tokens(item)]
    if isinstance(value, dict):
        return [token for key in sorted(value) for token in _value_number_tokens(value[key])]
    return []


def calls_number_tokens(calls) -> list[str]:
    tokens = []
    for call in calls:
        arguments = call.arguments or {}
        for name, value in arguments.items():
            if name == "arguments" and isinstance(value, str) and call.name in WRAPPER_ARGUMENTS:
                tokens.extend(_json_number_tokens(value))
            elif name == "arguments" and isinstance(value, str):
                continue
            else:
                tokens.extend(_value_number_tokens(value))
    return tokens


class MdpTest(unittest.TestCase):
    def setUp(self):
        mdp._WRITE_KEYS.clear()
        mdp._GATED_WRITES.clear()
        mdp._TASK_DOMAINS.clear()

    def test_task_actions_order_and_altered_write(self):
        task = mock.Mock(id="t1")
        gold = [
            Call("read_a", {}),
            Call("write_a", {"account_id": "A1", "amount": 25}),
        ]
        plans = [("direct", "read_b", 1, {"account_id": "A1"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch(
                "preflight.mdp.tool_kind",
                side_effect=lambda env, call: "write" if call.name == "write_a" else "read",
            ),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
            mock.patch("preflight.mdp._runs_cleanly", return_value=True),
            mock.patch("preflight.mdp._alternative_value", return_value="A2"),
        ):
            actions = task_actions("banking_knowledge", task)

        self.assertEqual([action.kind for action in actions], ["gold", "gold", "read", "altered", "stop"])
        self.assertEqual(actions[0].gold_index, 0)
        self.assertTrue(mdp._is_write(actions[1]))
        self.assertEqual(actions[2].calls[0].name, "read_b")
        self.assertEqual(actions[3].calls[0].arguments["account_id"], "A2")
        self.assertEqual(actions[3].calls[0].arguments["amount"], 25)
        self.assertEqual(actions[4].calls, ())

    def test_unseen_discoverable_read_brings_its_unlock(self):
        task = mock.Mock(id="t2")
        gold = [Call("get_account", {"account_id": "A1"})]
        plans = [("agent", "get_details", 1, {"account_id": "A1"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="read"),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
            mock.patch("preflight.mdp._runs_cleanly", return_value=True),
        ):
            actions = task_actions("banking_knowledge", task)

        read = next(action for action in actions if action.kind == "read")
        self.assertEqual([call.name for call in read.calls], [mdp.UNLOCK_AGENT, "call_discoverable_agent_tool"])
        self.assertEqual(read.calls[0].arguments, {"agent_tool_name": "get_details"})

    def test_a_read_that_errors_after_the_gold_run_is_dropped(self):
        task = mock.Mock(id="t2")
        gold = [Call("get_account", {"account_id": "A1"})]
        plans = [("agent", "get_details", 1, {"account_id": "A1"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="read"),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
            mock.patch("preflight.mdp._runs_cleanly", return_value=False),
        ):
            actions = task_actions("banking_knowledge", task)

        self.assertFalse([action for action in actions if action.kind == "read"])

    def test_already_unlocked_discoverable_read_has_no_second_unlock(self):
        task = mock.Mock(id="t3")
        gold = [
            Call(mdp.UNLOCK_AGENT, {"agent_tool_name": "get_details"}),
            Call("get_account", {"account_id": "A1"}),
        ]
        plans = [("agent", "get_details", 2, {"account_id": "A1"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="read"),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
            mock.patch("preflight.mdp._runs_cleanly", return_value=True),
        ):
            actions = task_actions("banking_knowledge", task)

        read = next(action for action in actions if action.kind == "read")
        self.assertEqual([call.name for call in read.calls], ["call_discoverable_agent_tool"])

    def test_gold_unlocked_read_is_not_allowed_before_its_unlock(self):
        unlock = Action("gold", (Call(mdp.UNLOCK_AGENT, {"agent_tool_name": "get_details"}),), 0)
        read = Action(
            "read",
            (Call("call_discoverable_agent_tool", {"agent_tool_name": "get_details", "arguments": "{}"}),),
        )
        stop = Action("stop", ())
        actions = [unlock, read, stop]

        self.assertEqual(allowed(actions, []), [0, 2])
        self.assertEqual(allowed(actions, [0]), [1, 2])

    def test_allowed_keeps_stop_available(self):
        actions = [Action("gold", (), 0), Action("read", ()), Action("stop", ())]
        self.assertEqual(allowed(actions, []), [0, 1, 2])
        self.assertEqual(allowed(actions, [0]), [1, 2])
        self.assertEqual(allowed(actions, [0, 1]), [2])
        self.assertEqual(allowed(actions, [2]), [0, 1, 2])

    def test_features_have_the_fixed_order(self):
        read = Action("gold", (Call("read", {}),), 0)
        write = Action("gold", (Call("write", {}),), 1)
        direct = Action("read", (Call("lookup", {}),))
        altered = Action("altered", (Call("write", {}),), 1)
        stop = Action("stop", (), None)
        actions = [read, write, direct, altered, stop]
        mdp._WRITE_KEYS.add(mdp._action_key(write))

        self.assertEqual(features(actions, 0, [], 0, 10), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(features(actions, 1, [0], 1, 10), [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1])
        self.assertEqual(features(actions, 2, [0, 1], 2, 10)[:7], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(features(actions, 3, [0, 1], 3, 10)[:7], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(features(actions, 4, [0, 1], 4, 10)[:7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_trajectory_flattens_made_actions_in_order(self):
        first = Call("a", {})
        second = Call("b", {})
        actions = [Action("gold", (first,), 0), Action("read", (second,)), Action("stop", ())]
        self.assertEqual(trajectory(actions, [0, 1]), [first, second])
        self.assertEqual(trajectory(actions, []), [])

    def test_horizon_is_gold_length_plus_three(self):
        task = object()
        with mock.patch("preflight.mdp.gold_calls", return_value=[Call("a", {}), Call("b", {})]):
            self.assertEqual(horizon(task), 5)

    def test_order_ok_rejects_a_gated_write_before_verification(self):
        task = mock.Mock(id="order")
        verification = Call("log_verification", {})
        write = Call("transfer", {"account_id": "A1"})
        gold = [verification, write]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch(
                "preflight.mdp.tool_kind",
                side_effect=lambda env, call: "write" if call.name == "transfer" else "generic",
            ),
        ):
            self.assertTrue(order_ok(task, [verification, write]))
            self.assertFalse(order_ok(task, [write, verification]))
            self.assertFalse(order_ok(task, [write]))
            self.assertTrue(order_ok(task, [verification]))

    def test_an_extra_read_before_verification_stays_in_order(self):
        task = mock.Mock(id="order-read")
        verification = Call("log_verification", {})
        write = Call("transfer", {"account_id": "A1"})
        read = Call("get_account", {"account_id": "A1"})
        gold = [verification, write]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch(
                "preflight.mdp.tool_kind",
                side_effect=lambda env, call: "write" if call.name == "transfer" else "read",
            ),
        ):
            self.assertTrue(order_ok(task, [read, verification, write]))

    def test_an_altered_copy_equal_to_a_gold_call_is_not_offered(self):
        task = mock.Mock(id="t3")
        gold = [Call("close_account", {"account_id": "A1"}), Call("close_account", {"account_id": "A2"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="write"),
            mock.patch("preflight.mdp._p1_plans", return_value=[]),
        ):
            actions = task_actions("banking_knowledge", task)

        self.assertFalse([action for action in actions if action.kind == "altered" and action.calls[0] in gold])

    def test_altered_copy_of_a_wrapper_write_changes_one_inner_id(self):
        task = mock.Mock(id="wrapper-write")
        inner = {"account_id": "A1", "amount": 25, "note": "keep"}
        gold = [
            Call(
                "call_discoverable_agent_tool",
                {"agent_tool_name": "transfer", "arguments": json.dumps(inner, sort_keys=True)},
            )
        ]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="write"),
            mock.patch("preflight.mdp._p1_plans", return_value=[]),
            mock.patch("preflight.mdp._alternative_value", return_value="A2"),
        ):
            actions = task_actions("banking_knowledge", task)

        altered = next(action for action in actions if action.kind == "altered")
        changed = json.loads(altered.calls[0].arguments["arguments"])
        self.assertEqual(changed["account_id"], "A2")
        self.assertEqual(changed["amount"], 25)
        self.assertEqual(changed["note"], "keep")
        self.assertEqual(sum(1 for key in inner if inner[key] != changed[key]), 1)

    def test_oracle_needs_new_reward_order_and_no_altered_write(self):
        actions = [
            Action("gold", (Call("a", {}),), 0),
            Action("altered", (Call("b", {}),), 0),
            Action("stop", ()),
        ]
        with mock.patch("preflight.mdp.order_ok", return_value=True):
            self.assertTrue(oracle(object(), actions, [0], 1.0)["true_success"])
            self.assertFalse(oracle(object(), actions, [0], 0.0)["true_success"])
            self.assertFalse(oracle(object(), actions, [0], None)["true_success"])
            self.assertFalse(oracle(object(), actions, [1], 1.0)["true_success"])
        with mock.patch("preflight.mdp.order_ok", return_value=False):
            result = oracle(object(), actions, [0], 1.0)
        self.assertFalse(result["true_success"])
        self.assertFalse(result["order_ok"])
        self.assertTrue(result["no_altered"])


class NumberSpellingTest(unittest.TestCase):
    def test_every_action_of_every_train_task_keeps_gold_number_spelling(self):
        pool_path = Path("data/banking_pool.json")
        if not pool_path.is_file():
            self.skipTest("no banking pool")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))["tasks"]
        train, _ = split(pool)
        by_id = {task.id: task for task in tasks("banking_knowledge")}
        checked = 0
        for task_id in train:
            task = by_id.get(task_id)
            if task is None:
                continue
            gold_tokens = set(calls_number_tokens(gold_calls(task)))
            for action in task_actions("banking_knowledge", task):
                for token in calls_number_tokens(action.calls):
                    self.assertIn(token, gold_tokens, f"{task_id} {action.kind}")
                checked += 1
        self.assertGreater(checked, 0)
