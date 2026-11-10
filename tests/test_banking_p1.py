"""A bounded live check that P1 reaches banking's discoverable reads the way an agent calls them."""

import json
import random
import unittest

from preflight.relations import AGENT_WRAPPER, UNLOCK_AGENT, _p1_plans, build_variants, tool_kind
from preflight.tau2_adapter import fresh_env, gold_calls, tasks

DOMAIN = "banking_knowledge"


def _task_with_unlocks(scan=40):
    """The shortest banking gold run that unlocks a discoverable tool."""
    candidates = [task for task in tasks(DOMAIN)[:scan] if any(c.name == UNLOCK_AGENT for c in gold_calls(task))]
    return min(candidates, key=lambda task: len(gold_calls(task))) if candidates else None


def _inserted(gold, calls):
    """(position, block): the calls a variant inserts, found by matching the gold run around them."""
    size = len(calls) - len(gold)
    for position in range(len(gold) + 1):
        if tuple(calls[:position]) != tuple(gold[:position]):
            continue
        if tuple(calls[position + size :]) != tuple(gold[position:]):
            continue
        return position, tuple(calls[position : position + size])
    return None, ()


class BankingP1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = _task_with_unlocks()
        if cls.task is None:
            return
        cls.gold = gold_calls(cls.task)
        cls.variants, _ = build_variants("P1", DOMAIN, cls.task, random.Random(f"0:{cls.task.id}:P1"))

    def setUp(self):
        if getattr(self, "task", None) is None:
            self.skipTest("no banking task unlocks a discoverable tool")
        if not self.variants:
            self.skipTest("no P1 candidate ran cleanly on this task")

    def test_discoverable_reads_are_candidates(self):
        agent = [name for route, name, _, _ in _p1_plans(DOMAIN, self.task, self.gold) if route == "agent"]
        self.assertTrue(agent, "no discoverable read had its required arguments in the gold run")

    def test_every_inserted_call_is_a_declared_read(self):
        env = fresh_env(DOMAIN, self.task)
        for variant in self.variants:
            position, block = _inserted(self.gold, variant.calls)
            self.assertIsNotNone(position, variant.detail)
            self.assertIn(len(block), (1, 2), variant.detail)
            self.assertEqual(tool_kind(env, block[-1]), "read", variant.detail)

    def test_an_inserted_agent_read_brings_its_unlock(self):
        checked = 0
        for variant in self.variants:
            position, block = _inserted(self.gold, variant.calls)
            if position is None or block[-1].name != AGENT_WRAPPER:
                continue
            checked += 1
            read = block[-1]
            self.assertIsInstance(read.arguments.get("arguments"), str)
            self.assertIsInstance(json.loads(read.arguments["arguments"]), dict)
            inner = read.arguments["agent_tool_name"]
            unlocked = [
                call.arguments.get("agent_tool_name")
                for call in variant.calls[:position]
                if call.name == UNLOCK_AGENT
            ]
            if len(block) == 2:
                self.assertEqual(block[0].name, UNLOCK_AGENT)
                self.assertEqual(block[0].arguments, {"agent_tool_name": inner})
                self.assertNotIn(inner, unlocked)
            else:
                self.assertIn(inner, unlocked)
        self.assertTrue(checked, "no inserted read went through the discoverable wrapper")


if __name__ == "__main__":
    unittest.main()
