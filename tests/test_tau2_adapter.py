import unittest

from preflight.tau2_adapter import DOMAINS, fresh_env, gold_calls, grade, run, tasks


def writes_db(domain, task):
    return run(domain, task, gold_calls(task))[1].get_db_hash() != fresh_env(domain, task).get_db_hash()


class AdapterTest(unittest.TestCase):
    def test_replaying_the_trajectory_rebuilds_the_live_state(self):
        for domain in DOMAINS:
            for task in tasks(domain)[:5]:
                messages, live = run(domain, task, gold_calls(task))
                replayed = fresh_env(domain, task)
                init = task.initial_state
                replayed.set_state(
                    initialization_data=init.initialization_data if init else None,
                    initialization_actions=init.initialization_actions if init else None,
                    message_history=messages,
                )
                self.assertEqual(replayed.get_db_hash(), live.get_db_hash(), f"{domain} {task.id}")

    def test_gold_run_scores_one(self):
        for domain in DOMAINS:
            task = tasks(domain)[0]
            self.assertEqual(grade(domain, task, gold_calls(task)), 1.0, f"{domain} {task.id}")

    def test_empty_run_fails_when_the_gold_run_writes(self):
        for domain in DOMAINS:
            task = next(t for t in tasks(domain) if writes_db(domain, t))
            self.assertEqual(grade(domain, task, []), 0.0, f"{domain} {task.id}")


if __name__ == "__main__":
    unittest.main()
