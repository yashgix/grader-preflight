"""Minimal reproduction: tau2's banking grader accepts an interest-discrepancy report with wrong numbers.

Task `task_096` asks the agent to correct two savings accounts' interest and file one discrepancy report per account.
The report tool returns "Submitted Successfully" but stores nothing in the database the grader compares, so the
grader can see that the report tool was used, not what the reports say, how many were filed or for which account. No published trajectory or model call is needed: this
uses only the task's own gold actions.

Run with either pinned tau2 interpreter:  .tau2/v1.0.1/venv/bin/python examples/report_blind_spot.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

logger.remove()  # tau2 logs its setup and every tool call; the result is three lines

from preflight import suite  # noqa: E402
from preflight.tau2_adapter import Call, gold_calls, grade, run, tasks  # noqa: E402

DOMAIN, TASK = "banking_knowledge", "task_096"
REPORT = "submit_interest_discrepancy_report_7294"
# Values like those a real agent filed (gold: Bronze actual 2.55%, Gold Plus expected 6.85%).
WRONG = {"sav_ni73k9m4p2_bronze": {"actual_apy": 2.40}, "sav_ni73k9m4p2_goldplus": {"expected_apy": 6.90}}


def is_report(call: Call) -> bool:
    return call.name == "call_discoverable_agent_tool" and call.arguments.get("agent_tool_name") == REPORT


def with_wrong_numbers(calls: list[Call]) -> list[Call]:
    changed = []
    for call in calls:
        if is_report(call):
            args = json.loads(call.arguments["arguments"])
            args.update(WRONG[args["account_id"]])
            call = Call(call.name, {**call.arguments, "arguments": json.dumps(args)}, call.requestor)
        changed.append(call)
    return changed


def check() -> dict:
    task = next(t for t in tasks(DOMAIN) if t.id == TASK)
    gold = gold_calls(task)
    wrong = with_wrong_numbers(gold)
    _, wrong_env = run(DOMAIN, task, wrong)
    _, gold_env = run(DOMAIN, task, gold)
    reports = [i for i, call in enumerate(gold) if is_report(call)]
    first_report_only = [call for i, call in enumerate(gold) if i not in reports[1:]]
    return {
        "gold": grade(DOMAIN, task, gold),
        "wrong_report_numbers": grade(DOMAIN, task, wrong),
        "db_tables_that_differ": suite._db_diff(wrong_env, gold_env),
        "first_report_only": grade(DOMAIN, task, first_report_only),
        "no_report_at_all": grade(DOMAIN, task, [c for c in gold if not is_report(c)]),
    }


def main() -> None:
    result = check()
    print(f"gold run:                          reward {result['gold']}")
    print(
        f"same run, wrong report numbers:    reward {result['wrong_report_numbers']}"
        f"  (DB tables that differ from gold: {result['db_tables_that_differ'] or 'none'})"
    )
    print(f"same run, second report left out:  reward {result['first_report_only']}")
    print(f"same run, both reports left out:   reward {result['no_report_at_all']}")


if __name__ == "__main__":
    main()
