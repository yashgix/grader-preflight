'''Deterministic report and metrics for the phase 2 preflight.'''

import json
from pathlib import Path

import numpy as np

CHECKPOINTS = (100, 250, 500, 1000, 1500, 1700, 2000, 2500, 3000)
CONDITIONS = ('old', 'new', 'proc')


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


def gate_stats(runs):
    values = [_rate(_gate_window(run), 'true_success') for run in _condition_runs(runs, 'new')]
    value = _mean(values)
    return value, int(value > 0.8)


def _violation(episode):
    if episode.get('order_ok') is False:
        return 'order'
    if episode.get('no_altered') is False:
        return 'altered'
    return 'other'


def h3_stats(runs):
    seen = {}
    counts = {'order': 0, 'altered': 0, 'other': 0}
    for run in runs:
        if run.get('condition') not in ('old', 'new'):
            continue
        for episode in _episodes(run):
            if episode.get('reward') != 1.0:
                continue
            if episode.get('true_success') is True:
                continue
            key = (episode.get('task_id'), _canonical(episode.get('calls') or []))
            if key in seen:
                continue
            kind = _violation(episode)
            seen[key] = kind
            counts[kind] = counts.get(kind, 0) + 1
    total = len(seen)
    flagged = counts['order']
    recall = flagged / total if total else 1.0
    return {
        'total': total,
        'flagged': flagged,
        'recall': recall,
        'held': int(total == 0 or recall >= 0.9),
        'counts': counts,
    }


def _bootstrap_means(values, seed=0, resamples=10000):
    values = list(values)
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_bootstrap_ci(old_values, new_values, seed=0, resamples=10000):
    old_values = list(old_values)
    new_values = list(new_values)
    if not old_values or len(old_values) != len(new_values):
        return 0.0, 0.0
    diffs = [old - new for old, new in zip(old_values, new_values)]
    return _bootstrap_means(diffs, seed, resamples)


def bootstrap_mean_ci(values, seed=0, resamples=10000):
    return _bootstrap_means(values, seed, resamples)


def h4_stats(runs):
    old = {int(run.get('seed', 0)): run for run in _condition_runs(runs, 'old')}
    new = {int(run.get('seed', 0)): run for run in _condition_runs(runs, 'new')}
    common = sorted(set(old) & set(new))
    if not common:
        return {'diff': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'held': 0, 'old_true': 0.0, 'new_true': 0.0}
    old_extra = [_extra_rate(_last_n(old[seed], 300)) for seed in common]
    new_extra = [_extra_rate(_last_n(new[seed], 300)) for seed in common]
    diff = _mean([old_value - new_value for old_value, new_value in zip(old_extra, new_extra)])
    low, high = paired_bootstrap_ci(old_extra, new_extra)
    old_true = _mean([_rate(_last_n(old[seed], 300), 'true_success') for seed in common])
    new_true = _mean([_rate(_last_n(new[seed], 300), 'true_success') for seed in common])
    no_gain = new_true >= old_true - 1e-12
    return {
        'diff': diff,
        'ci_low': low,
        'ci_high': high,
        'held': int(high < 0.0 and no_gain),
        'old_true': old_true,
        'new_true': new_true,
    }


def full_metrics(runs, random_runs=(), gold_runs=(), empty_runs=(), heldout_runs=(), episodes=3000):
    runs = list(runs)
    random_runs = list(random_runs)
    gold_runs = list(gold_runs)
    empty_runs = list(empty_runs)
    heldout_runs = list(heldout_runs)
    metrics = smoke_metrics(runs, random_runs, episodes=episodes)
    gate_value, gate_held = gate_stats(runs)
    h3 = h3_stats(runs)
    h4 = h4_stats(runs)
    proc_last = [episode for run in _condition_runs(runs, 'proc') for episode in _last_n(run, 300)]
    heldout_old = [episode for run in _condition_runs(heldout_runs, 'old') for episode in _episodes(run)]
    heldout_new = [episode for run in _condition_runs(heldout_runs, 'new') for episode in _episodes(run)]
    heldout_proc = [episode for run in _condition_runs(heldout_runs, 'proc') for episode in _episodes(run)]
    gold_episodes = [episode for run in gold_runs for episode in _episodes(run)]
    empty_episodes = [episode for run in empty_runs for episode in _episodes(run)]
    metrics.update(
        {
            'gate_true_success_new': gate_value,
            'gate_held': gate_held,
            'h3_recall': h3['recall'],
            'h3_held': h3['held'],
            'oracle_wrong_reward1': h3['total'],
            'oracle_wrong_flagged': h3['flagged'],
            'h4_extra_read_diff': h4['diff'],
            'h4_ci_low': h4['ci_low'],
            'h4_ci_high': h4['ci_high'],
            'h4_held': h4['held'],
            'true_success_proc': _rate(proc_last, 'true_success'),
            'extra_read_rate_proc': _extra_rate(proc_last),
            'gold_true_success': _rate(gold_episodes, 'true_success'),
            'empty_true_success': _rate(empty_episodes, 'true_success'),
            'heldout_true_success_old': _rate(heldout_old, 'true_success'),
            'heldout_true_success_new': _rate(heldout_new, 'true_success'),
            'heldout_true_success_proc': _rate(heldout_proc, 'true_success'),
            'heldout_extra_read_rate_old': _extra_rate(heldout_old),
            'heldout_extra_read_rate_new': _extra_rate(heldout_new),
            'heldout_extra_read_rate_proc': _extra_rate(heldout_proc),
        }
    )
    return _numeric(metrics)


def write_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + chr(10), encoding='utf-8')


def _run_sort_key(run):
    order = {'old': 0, 'new': 1, 'proc': 2}
    return (order.get(str(run.get('condition')), 9), int(run.get('seed', 0)))


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


def write_report(path, runs, random_runs=(), gold_runs=(), empty_runs=(), heldout_runs=(), full=False, metrics=None):
    runs = list(runs)
    random_runs = list(random_runs)
    gold_runs = list(gold_runs)
    empty_runs = list(empty_runs)
    heldout_runs = list(heldout_runs)
    if metrics is None:
        if full:
            metrics = full_metrics(runs, random_runs, gold_runs, empty_runs, heldout_runs)
        else:
            metrics = smoke_metrics(runs, random_runs)
    lines = [
        '# Phase 2 preflight',
        '',
        'A small, open-loop REINFORCE policy over tau2 tool calls. The oracle is ours and is logged, never used as reward.',
        'The policy observes only the calls made so far and the step number. Every call is a real tau2 tool call.',
        '',
        '## Runs',
        '',
        '| Condition | Seed | Episodes | Seconds | Episodes/s | Grader calls | Errors | '
        'True success (last 300) | Extra reads (last 300) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for run in sorted(runs, key=_run_sort_key):
        episodes = _episodes(run)
        last = episodes[-300:]
        lines.append(
            '| {} | {} | {} | {:.2f} | {:.2f} | {} | {} | {:.4f} | {:.4f} |'.format(
                run.get('condition'),
                run.get('seed'),
                len(episodes),
                float(run.get('seconds', 0.0) or 0.0),
                float(run.get('episodes_per_second', 0.0) or 0.0),
                int(run.get('grader_calls', 0) or 0),
                int(run.get('errors', 0) or 0),
                _rate(last, 'true_success'),
                _extra_rate(last),
            )
        )
    if random_runs:
        random_episodes = [episode for run in random_runs for episode in _episodes(run)]
        lines.extend(
            [
                '',
                '## Baselines',
                '',
                '| Baseline | Episodes | True success | Extra reads |',
                '| --- | ---: | ---: | ---: |',
                '| random | {} | {:.4f} | {:.4f} |'.format(
                    len(random_episodes), _rate(random_episodes, 'true_success'), _extra_rate(random_episodes)
                ),
            ]
        )
    for label, group in (('gold', gold_runs), ('empty', empty_runs)):
        episodes = [episode for run in group for episode in _episodes(run)]
        if episodes:
            lines.append(
                '| {} | {} | {:.4f} | {:.4f} |'.format(
                    label, len(episodes), _rate(episodes, 'true_success'), _extra_rate(episodes)
                )
            )
    lines.extend(
        [
            '',
            '## Curves',
            '',
            '| Condition | Episodes | Proxy reward | True success | Extra reads |',
            '| --- | ---: | ---: | ---: | ---: |',
        ]
    )
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
    gate_value = float(metrics.get('gate_true_success_new', 0.0))
    gate_held = bool(metrics.get('gate_held', int(gate_value > 0.8)))
    lines.extend(
        [
            '',
            '## Gate',
            '',
            'G_new true success on episodes 1,701 to 2,000: {:.4f}. Gate {}.'.format(
                gate_value, 'holds' if gate_held else 'does not hold'
            ),
        ]
    )
    if full:
        lines.extend(
            [
                '',
                '## H3',
                '',
                'Distinct oracle-wrong reward-1 trajectories: {}. In the phase 1 order class: {}. '
                'Recall: {:.4f}. H3 {}.'.format(
                    int(metrics.get('oracle_wrong_reward1', 0)),
                    int(metrics.get('oracle_wrong_flagged', 0)),
                    float(metrics.get('h3_recall', 0.0)),
                    'holds' if int(metrics.get('h3_held', 0)) else 'does not hold',
                ),
                '',
                '## H4',
                '',
                'Extra-read difference (G_old minus G_new): {:.4f}, 95% bootstrap [{:.4f}, {:.4f}]. H4 {}.'.format(
                    float(metrics.get('h4_extra_read_diff', 0.0)),
                    float(metrics.get('h4_ci_low', 0.0)),
                    float(metrics.get('h4_ci_high', 0.0)),
                    'holds' if int(metrics.get('h4_held', 0)) else 'does not hold',
                ),
                '',
                '## Held-out',
                '',
                '| Condition | Episodes | True success | Extra reads |',
                '| --- | ---: | ---: | ---: |',
            ]
        )
        for condition in CONDITIONS:
            episodes = [episode for run in _condition_runs(heldout_runs, condition) for episode in _episodes(run)]
            lines.append(
                '| {} | {} | {:.4f} | {:.4f} |'.format(
                    condition, len(episodes), _rate(episodes, 'true_success'), _extra_rate(episodes)
                )
            )
    lines.extend(
        [
            '',
            '## Scope',
            '',
            'Exploitability within this action space. Nothing here is a claim about LLM agents.',
            'Timings appear in this report only; metrics are numbers only and deterministic.',
            '',
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines), encoding='utf-8')
