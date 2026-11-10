"""Run tool calls on a live τ²-bench environment and score them with τ²-bench's own, unchanged DB-state grader."""

from dataclasses import dataclass

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

try:  # v1.0.1 builds banking's read-log allowlist (#329) in its runner, not in the grader
    from tau2.runner.build import _derive_read_log_allowlist
except ImportError:
    _derive_read_log_allowlist = None

DOMAINS = ("banking_knowledge", "airline", "retail")
# The default banking variant embeds the knowledge base through a paid API; grep keeps it offline.
ENV_KWARGS = {"banking_knowledge": {"retrieval_variant": "grep_only"}}


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict
    requestor: str = "assistant"


def tasks(domain: str) -> list[Task]:
    """Tasks the DB-state grader scores: those with DB in their reward basis."""
    return [
        t
        for t in registry.get_tasks_loader(domain)()
        if t.evaluation_criteria and "DB" in t.evaluation_criteria.reward_basis
    ]


def gold_calls(task: Task) -> list[Call]:
    return [Call(a.name, dict(a.arguments), a.requestor) for a in task.evaluation_criteria.actions or []]


def env_kwargs(domain: str, task: Task) -> dict:
    """The environment arguments tau2's own runner uses for this task."""
    kwargs = dict(ENV_KWARGS.get(domain, {}))
    if domain == "banking_knowledge" and _derive_read_log_allowlist:
        kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
    return kwargs


def fresh_env(domain: str, task: Task) -> Environment:
    env = registry.get_env_constructor(domain)(**env_kwargs(domain, task))
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=list(init.message_history or []) if init else [],
    )
    return env


def run(domain: str, task: Task, calls: list[Call]) -> tuple[list[Message], Environment]:
    """Execute calls in order on a live env; return the message trajectory (live tool outputs) and the env."""
    env = fresh_env(domain, task)
    init = task.initial_state
    messages: list[Message] = list(init.message_history or []) if init else []
    for i, call in enumerate(calls):
        tool_call = ToolCall(id=f"call_{i}", name=call.name, arguments=call.arguments, requestor=call.requestor)
        sender = UserMessage if call.requestor == "user" else AssistantMessage
        messages += [sender(role=call.requestor, content=None, tool_calls=[tool_call]), env.get_response(tool_call)]
    return messages, env


def grade(domain: str, task: Task, calls: list[Call]) -> float:
    messages, _ = run(domain, task, calls)
    constructor = registry.get_env_constructor(domain)
    return EnvironmentEvaluator.calculate_reward(
        constructor, task, messages, env_kwargs=env_kwargs(domain, task)
    ).reward
