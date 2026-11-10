'''Phase 2: train a small REINFORCE policy against tau2 graders in two environments.'''

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
from preflight.mdp import allowed, horizon, oracle, order_ok, task_actions, trajectory
from preflight.relations import _inner_name, calls_as_dicts
from preflight.rl import DEFAULT_LR, N_FEATURES, GradeCache, Policy, _episode, _gold_names, split, train
from preflight.tau2_adapter import Call, tasks

DOMAIN = 'banking_knowledge'
REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / 'data' / 'banking_pool.json'
DEFAULT_OLD = '/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python'
DEFAULT_NEW = '/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python'
SMOKE_SEEDS = (0, 1, 2)
FULL_SEEDS = tuple(range(10))
SMOKE_EPISODES = 2000
FULL_EPISODES = 3000
HELDOUT_EPISODES = 100
SPLIT_SEED = 20261110


def _environment():
    env = os.environ.copy()
    env['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def _read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding='utf-8')


def pool_ids(path=None):
    path = Path(path) if path is not None else POOL_PATH
    if not path.is_file():
        return [task.id for task in tasks(DOMAIN)]
    data = _read_json(path)
    values = data.get('tasks', []) if isinstance(data, dict) else data
    return [str(value) for value in values]


def _task_list(domain, task_ids):
    by_id = {task.id: task for task in tasks(domain)}
    return [by_id[str(task_id)] for task_id in task_ids if str(task_id) in by_id]


def ensure_split(folder, ids):
    path = Path(folder) / 'split.json'
    if path.is_file():
        try:
            data = _read_json(path)
            train = [str(item) for item in data['train']]
            held_out = [str(item) for item in data['held_out']]
            if train:
                return train, held_out
        except (OSError, ValueError, KeyError, TypeError):
            pass
    train, held_out = split([str(item) for item in ids])
    _write_json(path, {'seed': SPLIT_SEED, 'train': list(train), 'held_out': list(held_out)})
    return list(train), list(held_out)


def _task_ids_from_arg(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def _worker_args(condition, mode, out_path, episodes, seed, task_ids, domain=DOMAIN, lr=DEFAULT_LR,
                 input_path=None, episodes_per_task=None):
    args = [
        '--condition', condition,
        '--mode', mode,
        '--out', str(out_path),
        '--episodes', str(episodes),
        '--seed', str(seed),
        '--task-ids', ','.join(str(item) for item in task_ids),
        '--domain', domain,
        '--lr', str(lr),
    ]
    if input_path is not None:
        args.extend(['--in', str(input_path)])
    if episodes_per_task is not None:
        args.extend(['--episodes-per-task', str(episodes_per_task)])
    return args


def _run_worker(python, args):
    subprocess.run(
        [python, '-B', '-m', 'preflight.phase2', 'worker', *args],
        cwd=REPO_ROOT,
        env=_environment(),
        check=True,
    )


def _calls_from_dicts(values):
    calls = []
    for value in values or []:
        calls.append(
            Call(
                str(value.get('name', '')),
                dict(value.get('arguments') or {}),
                str(value.get('requestor', 'assistant')),
            )
        )
    return calls


def _extra_reads(actions, made):
    gold_names = _gold_names(actions)
    count = 0
    for index in made:
        action = actions[index]
        if action.kind != 'read':
            continue
        names = {_inner_name(call) or call.name for call in action.calls}
        if not names & gold_names:
            count += 1
    return count


def _condition_reward(condition, cache, task, calls):
    reward = cache.cache(task, calls)
    if reward is None:
        return None
    if condition == 'proc':
        return 1.0 if float(reward) == 1.0 and order_ok(task, calls) else 0.0
    return float(reward)


def _new_reward(condition, cache, task, calls):
    if condition in ('new', 'proc'):
        return cache.cache(task, calls)
    return None


def _episode_payload(task, actions, made, reward, error, new_reward, extra_reads=None, oracle_pending=False):
    calls = trajectory(actions, made)
    if extra_reads is None:
        extra_reads = _extra_reads(actions, made)
    if oracle_pending:
        true_success = None
        order = None
        altered = None
        oracle_error = False
    elif new_reward is None:
        true_success = False
        order = None
        altered = None
        oracle_error = True
    else:
        result = oracle(task, actions, made, new_reward)
        true_success = result['true_success']
        order = result['order_ok']
        altered = result['no_altered']
        oracle_error = False
    return {
        'task_id': getattr(task, 'id', None),
        'made': list(made),
        'calls': calls_as_dicts(calls),
        'reward': 0.0 if reward is None else float(reward),
        'error': bool(error),
        'extra_reads': int(extra_reads),
        'new_reward': None if new_reward is None else float(new_reward),
        'true_success': true_success,
        'order_ok': order,
        'no_altered': altered,
        'oracle_error': oracle_error,
    }


def _set_oracle(episode, task, actions, condition, cache):
    calls = _calls_from_dicts(episode.get('calls') or [])
    raw = _new_reward(condition, cache, task, calls)
    episode['new_reward'] = None if raw is None else float(raw)
    if raw is None and condition == 'old':
        episode['true_success'] = None
        episode['order_ok'] = None
        episode['no_altered'] = None
        episode['oracle_error'] = False
        return
    if raw is None:
        episode['true_success'] = False
        episode['order_ok'] = None
        episode['no_altered'] = None
        episode['oracle_error'] = True
        return
    result = oracle(task, actions, episode['made'], raw)
    episode.update(result)
    episode['oracle_error'] = False


def _random_made(rng, actions, steps_limit):
    made = []
    for _ in range(steps_limit):
        candidates = allowed(actions, made)
        if not candidates:
            break
        index = candidates[rng.randrange(len(candidates))]
        made.append(index)
        if actions[index].kind == 'stop':
            break
    return made


def _policy_payload(policy):
    bias = [
        [[str(key[0]), int(key[1])], float(value)]
        for key, value in sorted(policy.bias.items(), key=lambda item: (str(item[0][0]), int(item[0][1])))
    ]
    return {'w': [float(value) for value in policy.w], 'bias': bias}


def _policy_from_payload(payload):
    if not payload:
        raise ValueError('run has no policy')
    policy = Policy(len(payload.get('w') or []))
    policy.w = np.asarray(payload.get('w') or [], dtype=float)
    policy.bias = {(str(key[0]), int(key[1])): float(value) for key, value in payload.get('bias', [])}
    return policy


def _run_payload(condition, seed, domain, task_ids, episodes, records, cache, seconds, policy=None, mode='train'):
    payload = {
        'condition': condition,
        'seed': int(seed),
        'domain': domain,
        'task_ids': list(task_ids),
        'mode': mode,
        'episodes_requested': int(episodes),
        'episodes': records,
        'grader_calls': int(cache.misses),
        'errors': sum(1 for record in records if record.get('error') or record.get('oracle_error')),
        'seconds': float(seconds),
        'episodes_per_second': float(len(records) / seconds) if seconds > 0 else 0.0,
    }
    if policy is not None:
        payload['policy'] = _policy_payload(policy)
    return payload


def train_run(condition, seed, task_ids, episodes, out_path, domain=DOMAIN, lr=DEFAULT_LR):
    started = time.perf_counter()
    task_list = _task_list(domain, task_ids)
    if not task_list:
        raise ValueError('no tasks')
    cache = GradeCache(domain)

    def reward_fn(task, calls):
        return _condition_reward(condition, cache, task, calls)

    policy = Policy(N_FEATURES, lr=lr) if float(lr) != float(DEFAULT_LR) else None
    policy, records = train(domain, task_list, seed=seed, episodes=episodes, reward_fn=reward_fn, policy=policy)
    by_id = {task.id: task for task in task_list}
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    for record in records:
        task = by_id[record['task_id']]
        actions = actions_by_id[task.id]
        record['calls'] = calls_as_dicts(trajectory(actions, record['made']))
        _set_oracle(record, task, actions, condition, cache)
    seconds = time.perf_counter() - started
    payload = _run_payload(condition, seed, domain, task_ids, episodes, records, cache, seconds, policy)
    _write_json(out_path, payload)
    return payload


def random_run(condition, seed, task_ids, episodes, out_path, domain=DOMAIN):
    started = time.perf_counter()
    task_list = _task_list(domain, task_ids)
    if not task_list:
        raise ValueError('no tasks')
    cache = GradeCache(domain)
    rng = random.Random(f'{seed}:random')
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    records = []
    for _ in range(episodes):
        task = task_list[rng.randrange(len(task_list))]
        actions = actions_by_id[task.id]
        made = _random_made(rng, actions, horizon(task))
        calls = trajectory(actions, made)
        value = _condition_reward(condition, cache, task, calls)
        raw = _new_reward(condition, cache, task, calls)
        pending = condition == 'old' and raw is None
        records.append(
            _episode_payload(task, actions, made, value, value is None, raw, _extra_reads(actions, made), pending)
        )
    seconds = time.perf_counter() - started
    payload = _run_payload(condition, seed, domain, task_ids, episodes, records, cache, seconds, mode='random')
    _write_json(out_path, payload)
    return payload


def baseline_run(mode, seed, task_ids, out_path, domain=DOMAIN):
    started = time.perf_counter()
    task_list = _task_list(domain, task_ids)
    if not task_list:
        raise ValueError('no tasks')
    cache = GradeCache(domain)
    records = []
    for task in task_list:
        actions = task_actions(domain, task)
        if mode == 'gold':
            made = [index for index, action in enumerate(actions) if action.kind == 'gold']
        else:
            made = [index for index, action in enumerate(actions) if action.kind == 'stop'][:1]
        calls = trajectory(actions, made)
        value = _condition_reward('new', cache, task, calls)
        raw = _new_reward('new', cache, task, calls)
        records.append(_episode_payload(task, actions, made, value, value is None, raw, _extra_reads(actions, made)))
    seconds = time.perf_counter() - started
    payload = _run_payload('new', seed, domain, task_ids, len(task_list), records, cache, seconds, mode=mode)
    _write_json(out_path, payload)
    return payload


def eval_run(condition, seed, task_ids, episodes_per_task, input_path, out_path, domain=DOMAIN):
    started = time.perf_counter()
    source = _read_json(input_path)
    policy = _policy_from_payload(source.get('policy'))
    task_list = _task_list(domain, task_ids)
    cache = GradeCache(domain)
    rng = random.Random(f'{seed}:eval:{condition}')
    records = []
    for task in task_list:
        actions = task_actions(domain, task)
        steps = horizon(task)
        gold_names = _gold_names(actions)
        for _ in range(episodes_per_task):
            made, _, extra = _episode(policy, rng, task, actions, steps, gold_names)
            calls = trajectory(actions, made)
            value = _condition_reward(condition, cache, task, calls)
            raw = _new_reward(condition, cache, task, calls)
            pending = condition == 'old' and raw is None
            records.append(_episode_payload(task, actions, made, value, value is None, raw, extra, pending))
    seconds = time.perf_counter() - started
    payload = _run_payload(
        condition, seed, domain, task_ids, len(records), records, cache, seconds, mode='eval'
    )
    _write_json(out_path, payload)
    return payload


def oracle_fill(input_path, out_path=None, domain=DOMAIN):
    data = _read_json(input_path)
    episode_ids = {str(item.get('task_id')) for item in data.get('episodes', []) if item.get('task_id') is not None}
    task_ids = data.get('task_ids') or sorted(episode_ids)
    task_list = _task_list(domain, task_ids)
    by_id = {task.id: task for task in task_list}
    actions_by_id = {task.id: task_actions(domain, task) for task in task_list}
    cache = GradeCache(domain)
    for episode in data.get('episodes', []):
        task = by_id.get(str(episode.get('task_id')))
        if task is None:
            continue
        calls = _calls_from_dicts(episode.get('calls') or [])
        raw = cache.cache(task, calls)
        episode['new_reward'] = None if raw is None else float(raw)
        if raw is None:
            episode['true_success'] = False
            episode['order_ok'] = None
            episode['no_altered'] = None
            episode['oracle_error'] = True
        else:
            result = oracle(task, actions_by_id[task.id], episode.get('made', []), raw)
            episode.update(result)
            episode['oracle_error'] = False
    data['oracle_grader_calls'] = int(cache.misses)
    data['oracle_errors'] = sum(1 for episode in data.get('episodes', []) if episode.get('oracle_error'))
    data['errors'] = sum(
        1 for episode in data.get('episodes', []) if episode.get('error') or episode.get('oracle_error')
    )
    _write_json(out_path or input_path, data)
    return data


def _parallel_map(function, items, processes):
    items = list(items)
    if processes <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(processes)) as pool:
        return list(pool.map(function, items))


def _last_true_success(run, count):
    episodes = run.get('episodes', [])[-count:]
    if not episodes:
        return 0.0
    return sum(1.0 if episode.get('true_success') else 0.0 for episode in episodes) / len(episodes)


def _tuning_check(new_python, run_dir):
    try:
        retail = [task.id for task in tasks('retail')[:5]]
    except Exception:
        return DEFAULT_LR
    if not retail:
        return DEFAULT_LR
    best_lr = DEFAULT_LR
    best_score = -1.0
    for lr in (0.05, 0.1, 0.2):
        path = run_dir / f'tune_lr{lr}.json'
        try:
            _run_worker(
                new_python,
                _worker_args('new', 'train', path, 200, 0, retail, domain='retail', lr=lr),
            )
            score = _last_true_success(_read_json(path), 50)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_lr = lr
    return best_lr


def run_smoke(out_path, old_python=DEFAULT_OLD, new_python=DEFAULT_NEW, episodes=SMOKE_EPISODES,
              seeds=SMOKE_SEEDS, limit=5, processes=4):
    out_path = Path(out_path).resolve()
    folder = out_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    ids = pool_ids()
    train_ids, _ = ensure_split(folder, ids)
    smoke_ids = list(train_ids[:limit])
    run_dir = folder / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)

    def one(seed):
        old_path = run_dir / f'old_seed{seed}.json'
        new_path = run_dir / f'new_seed{seed}.json'
        random_path = run_dir / f'random_seed{seed}.json'
        _run_worker(old_python, _worker_args('old', 'train', old_path, episodes, seed, smoke_ids))
        _run_worker(
            new_python,
            _worker_args('new', 'oracle', old_path, episodes, seed, smoke_ids, input_path=old_path),
        )
        _run_worker(new_python, _worker_args('new', 'train', new_path, episodes, seed, smoke_ids))
        _run_worker(new_python, _worker_args('new', 'random', random_path, episodes, seed, smoke_ids))
        return old_path, new_path, random_path

    pairs = _parallel_map(one, seeds, processes)
    old_runs = [_read_json(pair[0]) for pair in pairs]
    new_runs = [_read_json(pair[1]) for pair in pairs]
    random_runs = [_read_json(pair[2]) for pair in pairs]
    runs = old_runs + new_runs
    metrics = report.smoke_metrics(runs, random_runs, episodes=episodes)
    report.write_report(folder / 'report.md', runs, random_runs, metrics=metrics, full=False)
    report.write_metrics(out_path, metrics)
    return metrics


def run_full(out_path, old_python=DEFAULT_OLD, new_python=DEFAULT_NEW, episodes=FULL_EPISODES,
             seeds=FULL_SEEDS, processes=4):
    out_path = Path(out_path).resolve()
    folder = out_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    ids = pool_ids()
    train_ids, held_out = ensure_split(folder, ids)
    run_dir = folder / 'runs'
    heldout_dir = folder / 'heldout'
    run_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)
    lr = _tuning_check(new_python, run_dir)

    def one(seed):
        old_path = run_dir / f'old_seed{seed}.json'
        new_path = run_dir / f'new_seed{seed}.json'
        proc_path = run_dir / f'proc_seed{seed}.json'
        random_path = run_dir / f'random_seed{seed}.json'
        old_held = heldout_dir / f'old_seed{seed}.json'
        new_held = heldout_dir / f'new_seed{seed}.json'
        proc_held = heldout_dir / f'proc_seed{seed}.json'
        _run_worker(old_python, _worker_args('old', 'train', old_path, episodes, seed, train_ids, lr=lr))
        _run_worker(
            new_python,
            _worker_args('new', 'oracle', old_path, episodes, seed, train_ids, input_path=old_path),
        )
        _run_worker(new_python, _worker_args('new', 'train', new_path, episodes, seed, train_ids, lr=lr))
        _run_worker(new_python, _worker_args('proc', 'train', proc_path, episodes, seed, train_ids, lr=lr))
        _run_worker(
            old_python,
            _worker_args(
                'old', 'eval', old_held, len(held_out) * HELDOUT_EPISODES, seed, held_out,
                lr=lr, input_path=old_path, episodes_per_task=HELDOUT_EPISODES,
            ),
        )
        _run_worker(
            new_python,
            _worker_args(
                'new', 'oracle', old_held, len(held_out) * HELDOUT_EPISODES, seed, held_out,
                input_path=old_held,
            ),
        )
        _run_worker(
            new_python,
            _worker_args(
                'new', 'eval', new_held, len(held_out) * HELDOUT_EPISODES, seed, held_out,
                lr=lr, input_path=new_path, episodes_per_task=HELDOUT_EPISODES,
            ),
        )
        _run_worker(
            new_python,
            _worker_args(
                'proc', 'eval', proc_held, len(held_out) * HELDOUT_EPISODES, seed, held_out,
                lr=lr, input_path=proc_path, episodes_per_task=HELDOUT_EPISODES,
            ),
        )
        _run_worker(new_python, _worker_args('new', 'random', random_path, episodes, seed, train_ids))
        return old_path, new_path, proc_path, random_path, old_held, new_held, proc_held

    pairs = _parallel_map(one, seeds, processes)
    runs = []
    heldout_runs = []
    random_runs = []
    for pair in pairs:
        runs.extend([_read_json(pair[0]), _read_json(pair[1]), _read_json(pair[2])])
        heldout_runs.extend([_read_json(pair[4]), _read_json(pair[5]), _read_json(pair[6])])
        random_runs.append(_read_json(pair[3]))
    gold_path = run_dir / 'gold.json'
    empty_path = run_dir / 'empty.json'
    _run_worker(
        new_python,
        _worker_args('new', 'gold', gold_path, len(train_ids), 0, train_ids),
    )
    _run_worker(
        new_python,
        _worker_args('new', 'empty', empty_path, len(train_ids), 0, train_ids),
    )
    gold_runs = [_read_json(gold_path)]
    empty_runs = [_read_json(empty_path)]
    metrics = report.full_metrics(runs, random_runs, gold_runs, empty_runs, heldout_runs, episodes=episodes)
    report.write_report(
        folder / 'report.md', runs, random_runs, gold_runs, empty_runs, heldout_runs, metrics=metrics, full=True
    )
    report.write_metrics(out_path, metrics)
    return metrics


smoke = run_smoke
full = run_full


def _worker_main(argv):
    parser = argparse.ArgumentParser(prog='python -m preflight.phase2 worker')
    parser.add_argument('--condition', choices=('old', 'new', 'proc'), default='new')
    parser.add_argument('--mode', choices=('train', 'random', 'gold', 'empty', 'eval', 'oracle'), default='train')
    parser.add_argument('--out', required=True)
    parser.add_argument('--in', dest='input', default=None)
    parser.add_argument('--episodes', type=int, default=SMOKE_EPISODES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task-ids', default='')
    parser.add_argument('--domain', default=DOMAIN)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--episodes-per-task', type=int, default=HELDOUT_EPISODES)
    args = parser.parse_args(argv)
    task_ids = _task_ids_from_arg(args.task_ids)
    if not task_ids and args.mode in ('train', 'random', 'gold', 'empty', 'eval'):
        task_ids = pool_ids()
    if args.mode == 'train':
        train_run(args.condition, args.seed, task_ids, args.episodes, args.out, args.domain, args.lr)
    elif args.mode == 'random':
        random_run(args.condition, args.seed, task_ids, args.episodes, args.out, args.domain)
    elif args.mode in ('gold', 'empty'):
        baseline_run(args.mode, args.seed, task_ids, args.out, args.domain)
    elif args.mode == 'eval':
        eval_run(
            args.condition, args.seed, task_ids, args.episodes_per_task, args.input, args.out, args.domain
        )
    else:
        oracle_fill(args.input or args.out, args.out, args.domain)
    return 0


def _parse_seeds(value, default):
    if value is None:
        return tuple(default)
    return tuple(int(item) for item in value.split(',') if item.strip())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'worker':
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser(prog='python -m preflight.phase2')
    parser.add_argument('--out', required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--old-python', default=DEFAULT_OLD)
    parser.add_argument('--new-python', default=DEFAULT_NEW)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seeds', default=None)
    parser.add_argument('--processes', type=int, default=4)
    args = parser.parse_args(argv)
    episodes = args.episodes
    if episodes is None:
        episodes = FULL_EPISODES if args.full else SMOKE_EPISODES
    seeds = _parse_seeds(args.seeds, FULL_SEEDS if args.full else SMOKE_SEEDS)
    if args.full:
        full(
            args.out,
            old_python=args.old_python,
            new_python=args.new_python,
            episodes=episodes,
            seeds=seeds,
            processes=args.processes,
        )
    else:
        smoke(
            args.out,
            old_python=args.old_python,
            new_python=args.new_python,
            episodes=episodes,
            seeds=seeds,
            processes=args.processes,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
