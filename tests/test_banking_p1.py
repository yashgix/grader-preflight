"""A bounded live check that P1 reaches banking's discoverable reads the way an agent calls them."""

import json
import random
import unittest

from preflight.relations import (
    AGENT_WRAPPER,
    UNLOCK_AGENT,
    _inner_name,
    _p1_plans,
    build_variants,
    tool_error,
    tool_kind,
)
from preflight.tau2_adapter import fresh_env, gold_calls, run, tasks

DOMAIN = "banking_knowledge"


def _task_with_unlocks(scan=40):
    """The shortest banking gold run that unlocks a discoverable tool and has an agent read candidate."""
    candidates = []
    for task in tasks(DOMAIN)[:scan]:
        gold = gold_calls(task)
        if not any(call.name == UNLOCK_AGENT for call in gold):
            continue
        if any(route == "agent" for route, _, _, _ in _p1_plans(DOMAIN, task, gold)):
            candidates.append(task)
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


def _inserted_read(gold, calls):
    """(position, read): where a variant inserts its block, and the read call inside it."""
    position, block = _inserted(gold, calls)
    if position is None or not block:
        return None, None
    return position, block[-1]


def _unseen_agent_reads(task, gold):
    """Discoverable reads the gold run never reaches, the case amendment 4 is about."""
    used = {_inner_name(call) or call.name for call in gold}
    return [
        name
        for route, name, _, _ in _p1_plans(DOMAIN, task, gold)
        if route == "agent" and name not in used
    ]


def _task_with_an_unseen_read(scan=40):
    """The first banking task whose P1 variants include one on a read the gold run does not use."""
    for task in tasks(DOMAIN)[:scan]:
        gold = gold_calls(task)
        if not gold:
            continue
        try:
            if not _unseen_agent_reads(task, gold):
                continue
            variants, _ = build_variants("P1", DOMAIN, task, random.Random(f"0:{task.id}:P1"))
        except Exception:  # a task the live env cannot replay is not this test's subject
            continue
        unseen = []
        for variant in variants:
            _, read = _inserted_read(gold, variant.calls)
            if read is not None and read.name == AGENT_WRAPPER and not variant.inserted_in_gold:
                unseen.append(variant)
        if unseen:
            return task, gold, unseen
    return None, [], []


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


class UnseenReadP1Test(unittest.TestCase):
    """Amendment 4: P1 must reach a discoverable read the gold run never uses, not only a repeated one."""

    @classmethod
    def setUpClass(cls):
        cls.task, cls.gold, cls.variants = _task_with_an_unseen_read()

    def setUp(self):
        if self.task is None:
            self.skipTest("no banking task yields a P1 variant on a read the gold run does not use")

    def test_the_variant_calls_an_unseen_read_through_the_wrapper(self):
        for variant in self.variants:
            position, read = _inserted_read(self.gold, variant.calls)
            self.assertIsNotNone(position, variant.detail)
            self.assertEqual(read.name, AGENT_WRAPPER, variant.detail)
            self.assertIsInstance(read.arguments.get("arguments"), str, variant.detail)
            self.assertIsInstance(json.loads(read.arguments["arguments"]), dict, variant.detail)
            self.assertFalse(variant.inserted_in_gold, variant.detail)

    def test_the_unlock_and_the_read_run_cleanly(self):
        for variant in self.variants:
            position, block = _inserted(self.gold, variant.calls)
            self.assertIsNotNone(position, variant.detail)
            inner = block[-1].arguments["agent_tool_name"]
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
            messages, _ = run(DOMAIN, self.task, list(variant.calls[: position + len(block)]))
            self.assertFalse(tool_error(messages[-1]), variant.detail)


if __name__ == "__main__":
    unittest.main()
