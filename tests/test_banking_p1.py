"""Live check that P1 reaches banking's discoverable reads, including one the gold run never uses."""

import json
import random
import unittest

from preflight.relations import AGENT_WRAPPER, UNLOCK_AGENT, _p1_plans, build_variants, tool_kind
from preflight.tau2_adapter import fresh_env, gold_calls, tasks

DOMAIN = "banking_knowledge"
TASK_ID = "task_005"
UNSEEN_READ = "get_user_dispute_history_7291"


def _task():
    return next(task for task in tasks(DOMAIN) if task.id == TASK_ID)


def _inserted(gold, calls):
    """(position, block): the calls a variant inserts, found by matching the gold run around them."""
    size = len(calls) - len(gold)
    if size <= 0:
        return None, ()
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
        cls.task = _task()
        cls.gold = gold_calls(cls.task)
        cls.variants, _ = build_variants("P1", DOMAIN, cls.task, random.Random(f"0:{cls.task.id}:P1"))

    def test_the_unseen_read_is_a_candidate(self):
        agent_reads = [name for route, name, _, _ in _p1_plans(DOMAIN, self.task, self.gold) if route == "agent"]
        self.assertIn(UNSEEN_READ, agent_reads)

    def test_every_inserted_call_is_a_declared_read(self):
        env = fresh_env(DOMAIN, self.task)
        self.assertTrue(self.variants, "P1 produced no variant on task_005")
        for variant in self.variants:
            position, block = _inserted(self.gold, variant.calls)
            self.assertIsNotNone(position, variant.detail)
            self.assertIn(len(block), (1, 2), variant.detail)
            self.assertEqual(tool_kind(env, block[-1]), "read", variant.detail)

    def test_p1_inserts_a_read_the_gold_run_does_not_use(self):
        unlisted = []
        for variant in self.variants:
            position, block = _inserted(self.gold, variant.calls)
            if position is None or not block or variant.inserted_in_gold:
                continue
            unlisted.append((variant, position, block))
        self.assertTrue(unlisted, "P1 produced no read outside the gold run")
        for variant, position, block in unlisted:
            read = block[-1]
            if read.name != AGENT_WRAPPER:
                continue
            self.assertIsInstance(read.arguments.get("arguments"), str, variant.detail)
            self.assertIsInstance(json.loads(read.arguments["arguments"]), dict, variant.detail)
            inner = read.arguments.get("agent_tool_name")
            unlocked = [
                call.arguments.get("agent_tool_name")
                for call in variant.calls[:position]
                if call.name == UNLOCK_AGENT
            ]
            if len(block) == 2:
                self.assertEqual(block[0].name, UNLOCK_AGENT, variant.detail)
                self.assertEqual(block[0].arguments, {"agent_tool_name": inner}, variant.detail)
                self.assertNotIn(inner, unlocked, variant.detail)
            else:
                self.assertIn(inner, unlocked, variant.detail)


if __name__ == "__main__":
    unittest.main()
