'''Deterministic smoke report and metrics for the phase 2 preflight.'''

import json
from pathlib import Path

import numpy as np

CHECKPOINTS = (100, 250, 500, 1000, 1500, 2000)
CONDITIONS = ('old', 'new')


def _episodes(run):
    return run.get('episodes') or []


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(records, key):
    records = list(records)
    if not records:
        return 0.0
    return sum(1.0 if record.get(key) else 0.0 for record in records) / len(records)


def _known_rate(records, key):
    values = [bool(record.get(key)) for record in records if isinstance(record.get(key), bool)]
    if not values:
        return 0.0
    return _mean(values)


def _extra_rate(records):
    records = list(records)
    if not records:
        return 0.0
    return sum(float(record.get('extra_reads', 0) or 0) for record in records) / len(records)


def _last_n(run, count):
    if count <= 0:
        return []
    return _episodes(run)[-count:]


def _gate_window(run):
    episodes = _episodes(run)
    if len(episodes) >= 2000:
        return episodes[1700:2000]
    return episodes[-300:]


def _condition_runs(runs, condition):
    return [run for run in runs if run.get('condition') == condition]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _distinct_count(runs):
    keys = set()
    for run in runs:
        for episode in _episodes(run):
            keys.add((episode.get('task_id'), _canonical(episode.get('calls') or [])))
    return len(keys)


def _numeric(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            result[key] = int(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


def smoke_metrics(runs, random_runs=(), episodes=2000):
    runs = list(runs)
    random_runs = list(random_runs)
    old = _condition_runs(runs, 'old')
    new = _condition_runs(runs, 'new')
    old_last = [episode for run in old for episode in _last_n(run, 300)]
    new_last = [episode for run in new for episode in _last_n(run, 300)]
    gate = [episode for run in new for episode in _gate_window(run)]
    random_episodes = [episode for run in random_runs for episode in _episodes(run)]
    metrics = {
        'episodes': int(episodes),
        'grader_calls_old': sum(int(run.get('grader_calls', 0)) for run in old),
        'grader_calls_new': sum(int(run.get('grader_calls', 0)) for run in new),
        'errors': sum(
            1
            for run in runs + random_runs
            for episode in _episodes(run)
            if episode.get('error') or episode.get('oracle_error')
        ),
        'gate_true_success_new': _rate(gate, 'true_success'),
        'random_true_success': _rate(random_episodes, 'true_success'),
        'true_success_old': _rate(old_last, 'true_success'),
        'extra_read_rate_old': _extra_rate(old_last),
        'extra_read_rate_new': _extra_rate(new_last),
        'random_extra_read_rate': _extra_rate(random_episodes),
        'distinct_trajectories': _distinct_count(runs),
    }
    return _numeric(metrics)


def _policy_from_payload(payload):
    if not payload:
        raise ValueError('run has no policy')
    from preflight.rl import Policy

    policy = Policy(len(payload.get('w') or []))
    policy.w = np.asarray(payload.get('w') or [], dtype=float)
    policy.bias = {(str(key[0]), int(key[1])): float(value) for key, value in payload.get('bias', [])}
    return policy


def _tool_names(calls):
    from preflight.relations import _inner_name

    names = set()
    for call in calls:
        if call.name in ('unlock_discoverable_agent_tool', 'unlock_discoverable_user_tool'):
            continue
        names.add(_inner_name(call) or call.name)
    return names


def _gold_tool_names(actions):
    from preflight.relations import _inner_name

    names = set()
    for action in actions:
        if action.kind != 'gold':
            continue
        for call in action.calls:
            if call.name in ('unlock_discoverable_agent_tool', 'unlock_discoverable_user_tool'):
                continue
            names.add(_inner_name(call) or call.name)
    return names


def per_task_probabilities(runs, domain='banking_knowledge'):
    '''Final-policy action-group probabilities per task, beside the uniform start.'''
    from preflight.mdp import allowed, features, horizon, task_actions
    from preflight.tau2_adapter import tasks

    rows = []
    by_id = {task.id: task for task in tasks(domain)}
    for run in runs:
        if not run.get('policy'):
            continue
        policy = _policy_from_payload(run.get('policy'))
        condition = run.get('condition')
        seed = run.get('seed')
        for task_id in run.get('task_ids') or []:
            task = by_id.get(str(task_id))
            if task is None:
                continue
            actions = task_actions(domain, task)
            candidates = allowed(actions, [])
            if not candidates:
                continue
            steps = horizon(task)
            feature_rows = [features(actions, index, [], 0, steps) for index in candidates]
            probabilities = policy.probs(str(task_id), candidates, feature_rows)
            gold_names = _gold_tool_names(actions)
            policy_sums = {'gold': 0.0, 'extra_read': 0.0, 'stop': 0.0}
            uniform_sums = {'gold': 0.0, 'extra_read': 0.0, 'stop': 0.0}
            uniform = 1.0 / len(candidates)
            for position, index in enumerate(candidates):
                action = actions[index]
                group = None
                if action.kind == 'gold':
                    group = 'gold'
                elif action.kind == 'stop':
                    group = 'stop'
                elif action.kind == 'read':
                    names = _tool_names(action.calls)
                    if names and not (names & gold_names):
                        group = 'extra_read'
                if group is None:
                    continue
                policy_sums[group] += float(probabilities[position])
                uniform_sums[group] += uniform
            rows.append({
                'condition': condition,
                'seed': int(seed),
                'task_id': str(task_id),
                'candidates': int(len(candidates)),
                'uniform': uniform_sums,
                'policy': policy_sums,
            })
    return rows


def _run_sort_key(run):
    order = {'old': 0, 'new': 1}
    return (order.get(str(run.get('condition')), 9), int(run.get('seed', 0)))


def _random_episodes(random_runs):
    return [episode for run in random_runs for episode in _episodes(run)]


def _summary(label, records):
    new_reward_values = [float(episode.get('new_reward')) for episode in records if episode.get('new_reward') is not None]
    return {
        'label': label,
        'episodes': len(records),
        'reward': _mean(float(episode.get('reward', 0.0) or 0.0) for episode in records),
        'new_reward': _mean(new_reward_values),
        'order_ok': _known_rate(records, 'order_ok'),
        'no_altered': _known_rate(records, 'no_altered'),
        'true_success': _rate(records, 'true_success'),
    }


def _condition_summaries(runs, random_runs):
    summaries = []
    for condition in CONDITIONS:
        condition_runs = _condition_runs(runs, condition)
        if not condition_runs:
            continue
        records = [episode for run in condition_runs for episode in _last_n(run, 300)]
        summaries.append(_summary(condition, records))
    if random_runs:
        summaries.append(_summary('random', _random_episodes(random_runs)))
    return summaries


def _checkpoints(runs):
    maximum = max((len(_episodes(run)) for run in runs), default=0)
    points = [point for point in CHECKPOINTS if point <= maximum]
    return points or [maximum]


def _window_mean(run, checkpoint, key):
    episodes = _episodes(run)[:checkpoint]
    if key == 'reward':
        return _mean(float(episode.get('reward', 0.0) or 0.0) for episode in episodes)
    if key == 'true_success':
        return _rate(episodes, 'true_success')
    return _extra_rate(episodes)


def _condition_curve(runs, condition, checkpoint, key):
    values = [_window_mean(run, checkpoint, key) for run in _condition_runs(runs, condition)]
    return _mean(values)


def _format_rate(value):
    return f'{float(value):.4f}'


def write_report(path, runs, random_runs=(), metrics=None, per_task=None, full=False):
    runs = list(runs)
    random_runs = list(random_runs)
    if metrics is None:
        metrics = smoke_metrics(runs, random_runs)
    if per_task is None:
        per_task = per_task_probabilities(runs)
    lines = [
        '# Phase 2 preflight',
        '',
        'A small, open-loop REINFORCE policy over tau2 tool calls. The oracle is ours and is logged, never used as reward.',
        'The policy observes only the calls made so far and the step number. Every call is a real tau2 tool call.',
        'Extra lookups use horizon steps equally in every condition; this report shows them beside the learned curves.',
        '',
        '## Runs',
        '',
        '| Condition | Seed | Episodes | Seconds | Episodes/s | Grader calls | Grader calls/ep | Errors | '
        'True success (last 300) | Extra reads (last 300) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for run in sorted(runs, key=_run_sort_key):
        episodes = _episodes(run)
        last = episodes[-300:]
        calls = int(run.get('grader_calls', 0) or 0)
        calls_per_episode = calls / len(episodes) if episodes else 0.0
        lines.append(
            '| {} | {} | {} | {:.2f} | {:.2f} | {} | {:.4f} | {} | {:.4f} | {:.4f} |'.format(
                run.get('condition'),
                run.get('seed'),
                len(episodes),
                float(run.get('seconds', 0.0) or 0.0),
                float(run.get('episodes_per_second', 0.0) or 0.0),
                calls,
                calls_per_episode,
                int(run.get('errors', 0) or 0),
                _rate(last, 'true_success'),
                _extra_rate(last),
            )
        )
    if random_runs:
        random_episodes = _random_episodes(random_runs)
        lines.extend([
            '',
            '## Baselines',
            '',
            '| Baseline | Episodes | Proxy reward | True success | Extra reads |',
            '| --- | ---: | ---: | ---: | ---: |',
            '| random | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
                len(random_episodes),
                _mean(float(episode.get('reward', 0.0) or 0.0) for episode in random_episodes),
                _rate(random_episodes, 'true_success'),
                _extra_rate(random_episodes),
            ),
        ])
    summaries = _condition_summaries(runs, random_runs)
    lines.extend([
        '',
        '## Kept-apart measures',
        '',
        '| Condition | Episodes | Training reward | G_new verdict | Order check | No altered write | True success |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for summary in summaries:
        lines.append(
            '| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                summary['label'],
                summary['episodes'],
                summary['reward'],
                summary['new_reward'],
                summary['order_ok'],
                summary['no_altered'],
                summary['true_success'],
            )
        )
    lines.extend([
        '',
        '## Curves',
        '',
        '| Condition | Episodes | Proxy reward | True success | Extra reads |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for condition in CONDITIONS:
        if not _condition_runs(runs, condition):
            continue
        for checkpoint in _checkpoints(runs):
            lines.append(
                '| {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
                    condition,
                    checkpoint,
                    _condition_curve(runs, condition, checkpoint, 'reward'),
                    _condition_curve(runs, condition, checkpoint, 'true_success'),
                    _condition_curve(runs, condition, checkpoint, 'extra'),
                )
            )
    if random_runs:
        random_episodes = _random_episodes(random_runs)
        lines.append(
            '| random | - | {:.4f} | {:.4f} | {:.4f} |'.format(
                _mean(float(episode.get('reward', 0.0) or 0.0) for episode in random_episodes),
                _rate(random_episodes, 'true_success'),
                _extra_rate(random_episodes),
            )
        )
    lines.extend([
        '',
        '## Per-task final policy probabilities',
        '',
        '| Condition | Seed | Task | Candidates | Uniform gold | Uniform extra reads | Uniform STOP | '
        'Final gold | Final extra reads | Final STOP |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in per_task:
        lines.append(
            '| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                row['condition'],
                row['seed'],
                row['task_id'],
                row['candidates'],
                row['uniform']['gold'],
                row['uniform']['extra_read'],
                row['uniform']['stop'],
                row['policy']['gold'],
                row['policy']['extra_read'],
                row['policy']['stop'],
            )
        )
    lines.extend([
        '',
        '## Gate',
        '',
        'G_new true success on episodes 1,701 to 2,000: {}.'.format(
            _format_rate(metrics.get('gate_true_success_new', 0.0))
        ),
        '',
        '## Scope',
        '',
        'Exploitability within this action space. Nothing here is a claim about LLM agents.',
        'Timings appear in this report only; metrics are numbers only and deterministic.',
        '',
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines), encoding='utf-8')


def write_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + chr(10), encoding='utf-8')
