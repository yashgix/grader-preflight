"""Phase 2 learner: a memoising grader cache, a linear softmax policy and REINFORCE training."""

import hashlib
import random

import numpy as np

from preflight import tau2_adapter
from preflight.mdp import allowed, features, horizon, task_actions, trajectory
from preflight.relations import _inner_name, calls_as_dicts, canonical_json

N_FEATURES = 8
DEFAULT_LR = 0.1
SPLIT_SEED = 20261110
TRAIN_SIZE = 30
HELD_OUT_SIZE = 10


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(task) -> str:
    task_id = getattr(task, "id", None)
    return task_id if isinstance(task_id, str) else repr(task_id)


def _task_fingerprint(task) -> str:
    """Phase 1's task fingerprint, with a fallback for objects that cannot dump themselves."""
    dump = getattr(task, "model_dump", None)
    if callable(dump):
        try:
            return _sha256(canonical_json(dump(mode="json")))
        except Exception:
            pass
    return _sha256(canonical_json(_identity(task)))


class GradeCache:
    """One real grader call per distinct (task, ordered calls); a grader exception is cached as None."""

    def __init__(self, domain, grade=None):
        self.domain = domain
        self.grade = grade if grade is not None else tau2_adapter.grade
        self.misses = 0
        self._rewards = {}

    def cache(self, task, calls) -> float | None:
        key = (_identity(task), _task_fingerprint(task), canonical_json(calls_as_dicts(calls)))
        if key in self._rewards:
            return self._rewards[key]
        self.misses += 1
        try:
            reward = self.grade(self.domain, task, list(calls))
        except Exception:
            reward = None
        self._rewards[key] = reward
        return reward


def _chosen_position(action_ids, chosen):
    """The chosen action's position; `chosen` is an action id, or a position when it is not one."""
    for position, action_id in enumerate(action_ids):
        if action_id == chosen:
            return position
    try:
        position = int(chosen)
    except (TypeError, ValueError):
        return None
    return position if 0 <= position < len(action_ids) else None


class Policy:
    """Linear softmax over the shared action features, plus a bias per (task, action)."""

    def __init__(self, n_features, lr=DEFAULT_LR):
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.w = np.zeros(self.n_features)
        self.bias = {}

    def _rows(self, action_ids, rows):
        array = np.asarray(rows, dtype=float)
        if array.size == 0:
            return np.zeros((0, self.n_features))
        if array.ndim == 1:
            array = array.reshape(len(action_ids), self.n_features)
        return array

    def _logits(self, task_id, action_ids, rows):
        array = self._rows(action_ids, rows)
        if array.shape[0] == 0:
            return np.zeros(0)
        logits = array @ self.w
        for position, action_id in enumerate(action_ids):
            logits[position] += self.bias.get((task_id, action_id), 0.0)
        return logits

    def probs(self, task_id, action_ids, rows) -> np.ndarray:
        logits = self._logits(task_id, action_ids, rows)
        if logits.size == 0:
            return logits
        weights = np.exp(logits - np.max(logits))
        return weights / np.sum(weights)

    def update(self, steps, advantage) -> None:
        """One gradient-ascent step on advantage * log prob of each chosen action."""
        if advantage == 0:
            return
        grad_w = np.zeros(self.n_features)
        grad_bias = {}
        for task_id, action_ids, rows, chosen in steps:
            array = self._rows(action_ids, rows)
            if array.shape[0] == 0:
                continue
            position = _chosen_position(action_ids, chosen)
            if position is None:
                continue
            probabilities = self.probs(task_id, action_ids, rows)
            grad_w += array[position] - probabilities @ array
            for index, action_id in enumerate(action_ids):
                key = (task_id, action_id)
                indicator = 1.0 if index == position else 0.0
                grad_bias[key] = grad_bias.get(key, 0.0) + indicator - probabilities[index]
        self.w = self.w + self.lr * float(advantage) * grad_w
        for key, value in grad_bias.items():
            self.bias[key] = self.bias.get(key, 0.0) + self.lr * float(advantage) * value


def _gold_names(actions) -> set[str]:
    names = set()
    for action in actions:
        if action.kind != "gold":
            continue
        for call in action.calls:
            names.add(_inner_name(call) or call.name)
    return names


def _read_names(action) -> set[str]:
    return {_inner_name(call) or call.name for call in action.calls}


def _episode(policy, rng, task, actions, steps_limit, gold_names):
    task_id = getattr(task, "id", None)
    made = []
    update_steps = []
    extra_reads = 0
    for step in range(steps_limit):
        candidates = allowed(actions, made)
        if not candidates:
            break
        rows = [features(actions, index, made, step, steps_limit) for index in candidates]
        probabilities = policy.probs(task_id, candidates, rows)
        position = rng.choices(range(len(candidates)), weights=[float(value) for value in probabilities])[0]
        index = candidates[position]
        update_steps.append((task_id, candidates, rows, index))
        made.append(index)
        action = actions[index]
        if action.kind == "read" and not (_read_names(action) & gold_names):
            extra_reads += 1
        if action.kind == "stop":
            break
    return made, update_steps, extra_reads


def train(domain, tasks, seed, episodes, reward_fn, policy=None):
    """REINFORCE over episodes drawn uniformly from `tasks`; returns the policy and per-episode records.

    A reward of None is a grader error: it counts as 0 for learning and is recorded with its error flag.
    """
    rng = random.Random(seed)
    policy = policy if policy is not None else Policy(N_FEATURES, lr=DEFAULT_LR)
    prepared = []
    for task in list(tasks):
        actions = task_actions(domain, task)
        prepared.append((task, actions, horizon(task), _gold_names(actions)))
    history = []
    records = []
    for _ in range(episodes):
        if not prepared:
            break
        task, actions, steps_limit, gold_names = prepared[rng.randrange(len(prepared))]
        made, update_steps, extra_reads = _episode(policy, rng, task, actions, steps_limit, gold_names)
        value = reward_fn(task, trajectory(actions, made))
        error = value is None
        reward = 0.0 if error else float(value)
        baseline = sum(history) / len(history) if history else 0.0
        policy.update(update_steps, reward - baseline)
        history.append(reward)
        records.append(
            {
                "task_id": getattr(task, "id", None),
                "made": made,
                "reward": reward,
                "error": error,
                "extra_reads": extra_reads,
            }
        )
    return policy, records


def split(pool_ids):
    """The pre-registered draw: sort, shuffle with seed 20261110, take 30 train then 10 held-out."""
    ids = sorted(pool_ids)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    drawn = ids[: TRAIN_SIZE + HELD_OUT_SIZE]
    return drawn[:TRAIN_SIZE], drawn[TRAIN_SIZE:]
