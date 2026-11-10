"""Tests for the phase 2 MDP: actions, features, trajectory, ordering and oracle."""

import unittest
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
from preflight.tau2_adapter import Call


class MdpTest(unittest.TestCase):
    def setUp(self):
        mdp._WRITE_KEYS.clear()
        mdp._GATED_NAMES.clear()
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
            mock.patch("preflight.mdp._alternative_value", return_value="A2"),
        ):
            actions = task_actions("banking_knowledge", task)

        self.assertEqual([action.kind for action in actions], ["gold", "gold", "read", "altered", "stop"])
        self.assertEqual(actions[0].gold_index, 0)
        self.assertFalse(actions[0].is_write)
        self.assertTrue(actions[1].is_write)
        self.assertEqual(actions[2].calls[0].name, "read_b")
        self.assertEqual(actions[3].calls[0].arguments["account_id"], "A2")
        self.assertEqual(actions[3].calls[0].arguments["amount"], 25)
        self.assertIsNone(actions[4].calls)

    def test_unseen_discoverable_read_brings_its_unlock(self):
        task = mock.Mock(id="t2")
        gold = [Call("get_account", {"account_id": "A1"})]
        plans = [("agent", "get_details", 1, {"account_id": "A1"})]
        with (
            mock.patch("preflight.mdp.gold_calls", return_value=gold),
            mock.patch("preflight.mdp.fresh_env", return_value=object()),
            mock.patch("preflight.mdp.tool_kind", return_value="read"),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
        ):
            actions = task_actions("banking_knowledge", task)

        read = next(action for action in actions if action.kind == "read")
        self.assertEqual([call.name for call in read.calls], [mdp.UNLOCK_AGENT, "call_discoverable_agent_tool"])
        self.assertEqual(read.calls[0].arguments, {"agent_tool_name": "get_details"})

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
            mock.patch("preflight.mdp.tool_kind", side_effect=lambda env, call: "read"),
            mock.patch("preflight.mdp._p1_plans", return_value=plans),
        ):
            actions = task_actions("banking_knowledge", task)

        read = next(action for action in actions if action.kind == "read")
        self.assertEqual([call.name for call in read.calls], ["call_discoverable_agent_tool"])

    def test_allowed_keeps_stop_available(self):
        actions = [Action("gold", (), 0), Action("read", (), None), Action("stop", (), None)]
        self.assertEqual(allowed(actions, []), [0, 1, 2])
        self.assertEqual(allowed(actions, [0]), [1, 2])
        self.assertEqual(allowed(actions, [0, 1]), [2])
        self.assertEqual(allowed(actions, [2]), [0, 1, 2])

    def test_features_have_the_fixed_order(self):
        actions = [
            Action("gold", (Call("read", {}),), 0, is_write=False),
            Action("gold", (Call("write", {}),), 1, is_write=True),
            Action("read", (Call("lookup", {}),), None, is_write=False),
            Action("altered", (Call("write", {}),), 1, is_write=True),
            Action("stop", (), None, is_write=False),
        ]
        self.assertEqual(features(actions, 0, [], 0, 10), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(features(actions, 1, [0], 1, 10), [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1])
        self.assertEqual(features(actions, 2, [0, 1], 2, 10)[:7], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(features(actions, 3, [0, 1], 3, 10)[:7], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(features(actions, 4, [0, 1], 4, 10)[:7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_trajectory_flattens_made_actions_in_order(self):
        first = Call("a", {})
        second = Call("b", {})
        actions = [Action("gold", (first,), 0), Action("read", (second,), None), Action("stop", (), None)]
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
            self.assertTrue(order_ok(task, [verification]))

    def test_oracle_needs_new_reward_order_and_no_altered_write(self):
        actions = [
            Action("gold", (Call("a", {}),), 0),
            Action("altered", (Call("b", {}),), 0),
            Action("stop", (), None),
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
