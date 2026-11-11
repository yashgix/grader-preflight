"""Phase 2: train a small REINFORCE policy against tau2 graders in two environments."""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from preflight import report
from preflight.envs import DEFAULT_NEW, DEFAULT_OLD
from preflight.mdp import actions_fingerprint, allowed, features, horizon, oracle, order_ok, task_actions, trajectory
from preflight.relations import _inner_name, calls_as_dicts
from preflight.rl import DEFAULT_LR, N_FEATURES, GradeCache, Policy, _episode, _gold_names, split, train
from preflight.tau2_adapter import Call, tasks

DOMAIN = "banking_knowledge"
REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "data" / "banking_pool.json"

SMOKE_SEEDS = (0, 1, 2)
SMOKE_EPISODES = 2000
SPLIT_SEED = 20261110


def _environment():
    env = os.environ.copy()
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding="utf-8")


def pool_ids(path=None):
    path = Path(path) if path is not None else POOL_PATH
    if not path.is_file():
        return [task.id for task in tasks(DOMAIN)]
    data = _read_json(path)
    values = data.get("tasks", []) if isinstance(data, dict) else data
    return [str(value) for value in values]


def _task_list(domain, task_ids):
    by_id = {task.id: task for task in tasks(domain)}
    return [by_id[str(task_id)] for task_id in task_ids if str(task_id) in by_id]


def ensure_split(folder, ids):
    path = Path(folder) / "split.json"
    if path.is_file():
        try:
            data = _read_json(path)
            train = [str(item) for item in data["train"]]
            held_out = [str(item) for item in data["held_out"]]
            if train:
                return train, held_out
        except (OSError, ValueError, KeyError, TypeError):
            pass
    train, held_out = split([str(item) for item in ids])
    _write_json(path, {"seed": SPLIT_SEED, "train": list(train), "held_out": list(held_out)})
    return list(train), list(held_out)


def _task_ids_from_arg(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _worker_args(condition, mode, out_path, episodes, seed, task_ids, domain=DOMAIN, lr=DEFAULT_LR, input_path=None):
    args = [
        "--condition",
        condition,
        "--mode",
        mode,
        "--out",
        str(out_path),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--task-ids",
        ",".join(str(item) for item in task_ids),
        "--domain",
        domain,
        "--lr",
        str(lr),
    ]
    if input_path is not None:
        args.extend(["--in", str(input_path)])
    return args


def _run_worker(python, args):
    subprocess.run(
        [python, "-B", "-m", "preflight.phase2", "worker", *args],
        cwd=REPO_ROOT,
        env=_environment(),
        check=True,
    )


def _calls_from_dicts(values):
    calls = []
    for value in values or []:
        calls.append(
            Call(
                str(value.get("name", "")),
                dict(value.get("arguments") or {}),
                str(value.get("requestor", "assistant")),
            )
        )
    return calls


def _extra_reads(actions, made):
    gold_names = _gold_names(actions)
    count = 0
    for index in made:
        action = actions[index]
        if action.kind != "read":
            continue
        names = {_inner_name(call) or call.name for call in action.calls}
        if not names & gold_names:
            count += 1
    return count


def _condition_reward(condition, cache, task, calls):
    reward = cache.cache(task, calls)
    if reward is None:
        return None
    if condition == "proc":
        return 1.0 if float(reward) == 1.0 and order_ok(task, calls) else 0.0
    return float(reward)


def _new_reward(condition, cache, task, calls):
    if condition in ("new", "proc"):
        return cache.cache(task, calls)
    return None


def _episode_payload(task, actions, made, reward, error, new_reward, extra_reads=None, oracle_pending=False):
    calls = trajectory(actions, made)
    if extra_reads is None:
        extra_reads = _extra_reads(actions, made)
    result = oracle(task, actions, made, new_reward)
    true_success = None if oracle_pending else result["true_success"]
    order, altered = result["order_ok"], result["no_altered"]
    oracle_error = new_reward is None and not oracle_pending
    return {
        "task_id": getattr(task, "id", None),
        "made": list(made),
        "calls": calls_as_dicts(calls),
        "reward": 0.0 if reward is None else float(reward),
        "error": bool(error),
        "extra_reads": int(extra_reads),
        "new_reward": None if new_reward is None else float(new_reward),
        "true_success": true_success,
        "order_ok": order,
        "no_altered": altered,
        "oracle_error": oracle_error,
    }


def _set_oracle(episode, task, actions, condition, cache):
    calls = _calls_from_dicts(episode.get("calls") or [])
    raw = _new_reward(condition, cache, task, calls)
    episode["new_reward"] = None if raw is None else float(raw)
    episode.update(oracle(task, actions, episode["made"], raw))
    pending = raw is None and condition == "old"  # filled later by oracle_fill in the v1.0.1 env
    if pending:
        episode["true_success"] = None
    episode["oracle_error"] = raw is None and not pending


def _random_made(rng, actions, steps_limit):
    made = []
    for _ in range(steps_limit):
        candidates = allowed(actions, made)
        if not candidates:
            break
        index = candidates[rng.randrange(len(candidates))]
        made.append(index)
        if actions[index].kind == "stop":
            break
    return made


def _policy_payload(policy):
    bias = [
        [[str(key[0]), int(key[1])], float(value)]
        for key, value in sorted(policy.bias.items(), key=lambda item: (str(item[0][0]), int(item[0][1])))
    ]
    return {"w": [float(value) for value in policy.w], "bias": bias}


def _policy_from_payload(payload):
    if not payload:
        raise ValueError("run has no policy")
    policy = Policy(len(payload.get("w") or []))
    policy.w = np.asarray(payload.get("w") or [], dtype=float)
    policy.bias = {(str(key[0]), int(key[1])): float(value) for key, value in payload.get("bias", [])}
    return policy


def _run_payload(
    condition, seed, domain, task_ids, episodes, records, cache, seconds, actions_by_id, policy=None, mode="train"
):
    payload = {
        "action_fingerprints": {task_id: actions_fingerprint(actions) for task_id, actions in actions_by_id.items()},
        "condition": condition,
        "seed": int(seed),
        "domain": domain,
        "task_ids": list(task_ids),
        "mode": mode,
        "episodes_requested": int(episodes),
        "episodes": records,
        "grader_calls": int(cache.misses),
        "errors": sum(1 for record in records if record.get("error") or record.get("oracle_error")),
        "seconds": float(seconds),
        "episodes_per_second": float(len(records) / seconds) if seconds > 0 else 0.0,
    }
    if policy is not None:
        payload["policy"] = _policy_payload(policy)
    return payload


def train_run(condition, seed, task_ids, episodes, out_path, domain=DOMAIN, lr=DEFAULT_LR):
    started = time.perf_counter()
    task_list = _task_list(domain, task_ids)
    if not task_list:
        raise ValueError("no tasks")
    cache = GradeCache(domain)

    def reward_fn(task, calls):
        return _condition_reward(condition, cache, task, calls)

    policy = Policy(N_FEATURES, lr=lr)
    policy, records = train(domain, task_list, seed=seed, episodes=episodes, reward_fn=reward_fn, policy=policy)
    by_id = {task.id: task for task in task_list}
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    for record in records:
        task = by_id[record["task_id"]]
        actions = actions_by_id[task.id]
        record["calls"] = calls_as_dicts(trajectory(actions, record["made"]))
        _set_oracle(record, task, actions, condition, cache)
    seconds = time.perf_counter() - started
    payload = _run_payload(condition, seed, domain, task_ids, episodes, records, cache, seconds, actions_by_id, policy)
    _write_json(out_path, payload)
    return payload


def random_run(condition, seed, task_ids, episodes, out_path, domain=DOMAIN):
    started = time.perf_counter()
    task_list = _task_list(domain, task_ids)
    if not task_list:
        raise ValueError("no tasks")
    cache = GradeCache(domain)
    rng = random.Random(f"{seed}:random")
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    records = []
    for _ in range(episodes):
        task = task_list[rng.randrange(len(task_list))]
        actions = actions_by_id[task.id]
        made = _random_made(rng, actions, horizon(task))
        calls = trajectory(actions, made)
        value = _condition_reward(condition, cache, task, calls)
        raw = _new_reward(condition, cache, task, calls)
        pending = condition == "old" and raw is None
        records.append(
            _episode_payload(task, actions, made, value, value is None, raw, _extra_reads(actions, made), pending)
        )
    seconds = time.perf_counter() - started
    payload = _run_payload(
        condition, seed, domain, task_ids, episodes, records, cache, seconds, actions_by_id, mode="random"
    )
    _write_json(out_path, payload)
    return payload


def greedy_made(policy, task, actions):
    """The most likely action at every step, no sampling and no learning."""
    made, steps = [], horizon(task)
    for step in range(steps):
        candidates = allowed(actions, made)
        rows = [features(actions, index, made, step, steps) for index in candidates]
        made.append(candidates[int(np.argmax(policy.probs(task.id, candidates, rows)))])
        if actions[made[-1]].kind == "stop":
            break
    return made


def after_gold(policy, task, actions, made):
    """What an episode does once every gold action is made: STOP at once or not, its later extra reads, and the
    policy's STOP probability at that point (None when the gold actions were never all made, or were finished on
    the last step, which leaves no choice)."""
    gold = {i for i, a in enumerate(actions) if a.kind == "gold"}
    done = next((p for p in range(len(made)) if gold <= set(made[: p + 1])), None)
    later, steps = made[done + 1 :] if done is not None else [], horizon(task)
    if done is None or done + 1 >= steps:
        none = {"stop_after_gold": None, "reads_after_gold": None, "p_stop_after_gold": None}
        return {"gold_done": done is not None, **none}
    state = made[: done + 1]
    candidates = allowed(actions, state)
    rows = [features(actions, index, state, done + 1, steps) for index in candidates]
    stop = next(i for i, index in enumerate(candidates) if actions[index].kind == "stop")
    return {
        "gold_done": True,
        "stop_after_gold": bool(later) and actions[later[0]].kind == "stop",
        "reads_after_gold": _extra_reads(actions, later),
        "p_stop_after_gold": float(policy.probs(task.id, candidates, rows)[stop]),
    }


def evaluate_run(condition, run_path, task_ids, episodes_per_task, out_path, domain=DOMAIN):
    """A trained policy without learning: per task, its greedy episode, then sampled episodes."""
    started = time.perf_counter()
    trained = _read_json(run_path)
    policy = _policy_from_payload(trained["policy"])
    task_list = _task_list(domain, task_ids)
    cache = GradeCache(domain)
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    records = []
    for task in task_list:
        actions, rng = actions_by_id[task.id], random.Random(f"{trained['seed']}:eval:{task.id}")
        runs = [greedy_made(policy, task, actions)]
        runs += [
            _episode(policy, rng, task, actions, horizon(task), _gold_names(actions))[0]
            for _ in range(episodes_per_task)
        ]
        for number, made in enumerate(runs):
            calls = trajectory(actions, made)
            reward = _condition_reward(condition, cache, task, calls)
            new = _new_reward(condition, cache, task, calls)
            record = _episode_payload(
                task, actions, made, reward, reward is None, new, oracle_pending=condition == "old"
            )
            record["greedy"] = number == 0
            record.update(after_gold(policy, task, actions, made))
            records.append(record)
    seconds = time.perf_counter() - started
    payload = _run_payload(
        condition, trained["seed"], domain, task_ids, len(records), records, cache, seconds, actions_by_id, mode="eval"
    )
    _write_json(out_path, payload)
    return payload


def oracle_fill(input_path, out_path=None, domain=DOMAIN):
    data = _read_json(input_path)
    episode_ids = {str(item.get("task_id")) for item in data.get("episodes", []) if item.get("task_id") is not None}
    task_ids = data.get("task_ids") or sorted(episode_ids)
    task_list = _task_list(domain, task_ids)
    by_id = {task.id: task for task in task_list}
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    recorded = data.get("action_fingerprints", {})
    for task_id, actions in actions_by_id.items():
        if recorded.get(task_id) != actions_fingerprint(actions):
            raise ValueError(f"{task_id}: the action list differs between tau2 versions, so indexes cannot be reused")
    cache = GradeCache(domain)
    for episode in data.get("episodes", []):
        task = by_id.get(str(episode.get("task_id")))
        if task is None:
            continue
        calls = _calls_from_dicts(episode.get("calls") or [])
        raw = cache(task, calls)
        episode["new_reward"] = None if raw is None else float(raw)
        episode.update(oracle(task, actions_by_id[task.id], episode.get("made", []), raw))
        episode["oracle_error"] = raw is None
    data["oracle_grader_calls"] = int(cache.misses)
    data["oracle_errors"] = sum(1 for episode in data.get("episodes", []) if episode.get("oracle_error"))
    data["errors"] = sum(
        1 for episode in data.get("episodes", []) if episode.get("error") or episode.get("oracle_error")
    )
    _write_json(out_path or input_path, data)
    return data


def _parallel_map(function, items, processes):
    items = list(items)
    if processes <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(processes)) as pool:
        return list(pool.map(function, items))


def run_smoke(
    out_path,
    old_python=DEFAULT_OLD,
    new_python=DEFAULT_NEW,
    episodes=SMOKE_EPISODES,
    seeds=SMOKE_SEEDS,
    limit=5,
    processes=4,
):
    out_path = Path(out_path).resolve()
    folder = out_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    ids = pool_ids()
    train_ids, _ = ensure_split(folder, ids)
    smoke_ids = list(train_ids[:limit])
    run_dir = folder / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    def one(seed):
        old_path = run_dir / f"old_seed{seed}.json"
        new_path = run_dir / f"new_seed{seed}.json"
        random_path = run_dir / f"random_seed{seed}.json"
        _run_worker(old_python, _worker_args("old", "train", old_path, episodes, seed, smoke_ids))
        _run_worker(
            new_python,
            _worker_args("new", "oracle", old_path, episodes, seed, smoke_ids, input_path=old_path),
        )
        _run_worker(new_python, _worker_args("new", "train", new_path, episodes, seed, smoke_ids))
        _run_worker(new_python, _worker_args("new", "random", random_path, episodes, seed, smoke_ids))
        return old_path, new_path, random_path

    pairs = _parallel_map(one, seeds, processes)
    old_runs = [_read_json(pair[0]) for pair in pairs]
    new_runs = [_read_json(pair[1]) for pair in pairs]
    random_runs = [_read_json(pair[2]) for pair in pairs]
    runs = old_runs + new_runs
    metrics = report.smoke_metrics(runs, random_runs, episodes=episodes)
    per_task = report.per_task_probabilities(runs, domain=DOMAIN)
    report.write_report(folder / "report.md", runs, random_runs, metrics=metrics, per_task=per_task, full=False)
    report.write_metrics(out_path, metrics)
    return metrics


smoke = run_smoke


def _worker_main(argv):
    parser = argparse.ArgumentParser(prog="python -m preflight.phase2 worker")
    parser.add_argument("--condition", choices=("old", "new", "proc"), default="new")
    parser.add_argument("--mode", choices=("train", "random", "oracle", "eval"), default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--in", dest="input", default=None)
    parser.add_argument("--episodes", type=int, default=SMOKE_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--domain", default=DOMAIN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args(argv)
    task_ids = _task_ids_from_arg(args.task_ids)
    if not task_ids and args.mode in ("train", "random"):
        task_ids = pool_ids()
    if args.mode == "train":
        train_run(args.condition, args.seed, task_ids, args.episodes, args.out, args.domain, args.lr)
    elif args.mode == "random":
        random_run(args.condition, args.seed, task_ids, args.episodes, args.out, args.domain)
    elif args.mode == "eval":
        evaluate_run(args.condition, args.input, task_ids, args.episodes, args.out, args.domain)
    else:
        oracle_fill(args.input or args.out, args.out, args.domain)
    return 0


def _parse_seeds(value, default):
    if value is None:
        return tuple(default)
    return tuple(int(item) for item in value.split(",") if item.strip())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m preflight.phase2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-python", default=DEFAULT_OLD)
    parser.add_argument("--new-python", default=DEFAULT_NEW)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args(argv)
    episodes = args.episodes if args.episodes is not None else SMOKE_EPISODES
    seeds = _parse_seeds(args.seeds, SMOKE_SEEDS)
    smoke(
        args.out,
        old_python=args.old_python,
        new_python=args.new_python,
        episodes=episodes,
        seeds=seeds,
        processes=args.processes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
