"""Fill docs/template.html with values and charts computed from results/*.json, writing docs/index.html.

Every measured number on the page comes from the results files, so the page can't drift from the data (fixed design
facts such as the seed count are written in the template). Placeholders look like {{name}}; an unknown or unused name is
an error.

    python3 scripts/build_site.py
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "var(--series-1)", "var(--series-2)"


def load(name):
    return json.loads((REPO / "results" / f"{name}.json").read_text())


NAMES = {
    "gpt55_banking": "GPT-5.5",
    "opus47_banking": "Claude Opus 4.7",
    "opus5_banking": "Claude Opus 5",
    "qwen38max_banking": "Qwen3.8-max",
}


def difference_text(row):
    """'expected APY 4.25 (correct 4.275)' for each field a filed report got wrong."""
    return "; ".join(
        f"{d['field'].replace('_apy', ' APY')} {d['filed']:g} (correct {d['gold']:g})" for d in row["differences"]
    )


def pct(x):
    return f"{round(100 * x)}%"


def num(x, places=2):
    return f"{x:.{places}f}".replace("-", "−")


def interval(low, high, places=2):
    return f"[{num(low, places)}, {num(high, places)}]"


def fix_bars(p1):
    """Phase 1: tasks each should-pass change was wrongly failed on, old grader versus fixed."""
    rows = []
    for change, key, total in (
        ("Extra lookup", "p1", p1["comparable_banking_tasks"]),
        ("Number respelled", "p3", p1["p3_paired_verdicts"]),
    ):
        for side, label in (("old", "old grader"), ("new", "fixed grader")):
            n = p1[f"{key}_flagged_tasks_{side}"]
            rows.append(
                f'<div class="row"><div class="name">{change}<small>{label}</small></div>'
                f'<div class="track" title="{change}, {label}: {n} of {total} tasks">'
                f'<div class="fill {side}" style="width:{100 * n / total:.1f}%"></div></div>'
                f'<div class="val">{n} / {total}</div></div>'
            )
    return "".join(rows)


def seeds_chart(p2):
    """True success over the last 300 training episodes, one dot per seed and grader."""
    width, height, left, top, bottom = 640, 250, 44, 30, 36
    plot_h = height - top - bottom
    step = (width - left - 16) / 10
    y = lambda v: top + plot_h * (1 - v)  # noqa: E731
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="True success per seed, '
        'old grader versus fixed grader">'
    ]
    for v in (0, 0.5, 1):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
            f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
        )
    parts.append(
        f'<line class="threshold" x1="{left}" x2="{width - 8}" y1="{y(0.8):.1f}" y2="{y(0.8):.1f}"/>'
        f'<text class="tick" x="{width - 10}" y="{y(0.8) - 6:.1f}" text-anchor="end">converged above 0.8</text>'
    )
    for s in range(10):
        cx = left + step * (s + 0.5)
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{height - 14}" text-anchor="middle">{s}</text>')
        for condition, color, dx in (("old", ORANGE, -6), ("new", BLUE, 6)):
            v = p2["per_seed"][condition][str(s)]["true"]
            label = "old grader" if condition == "old" else "fixed grader"
            parts.append(
                f'<circle cx="{cx + dx:.1f}" cy="{y(v):.1f}" r="5.5" fill="{color}" class="dot">'
                f"<title>Seed {s}, {label}: {v:.2f}</title></circle>"
            )
    parts.append(
        f'<text class="tick" x="{left + (width - left) / 2:.0f}" y="{height - 1}" text-anchor="middle">seed</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def events_chart(p3):
    """Counterfactual grading errors per task, split into explained and unexplained."""
    events, unexplained = p3["h5"]["events_by_task"], p3["h5"]["unexplained_by_task"]
    rows = sorted(events, key=lambda t: (-events[t], t))
    row_h, left, width, top = 19, 74, 640, 8
    scale = (width - left - 40) / max(events.values())
    height = top + row_h * len(rows) + 8
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Grading errors per task: '
        'explained by phase 1 versus not">'
    ]
    for i, task in enumerate(rows):
        total, missed = events[task], unexplained.get(task, 0)
        y = top + i * row_h
        explained_w, missed_w = (total - missed) * scale, missed * scale
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y + 12}" text-anchor="end">{task.replace("task_", "task ")}</text>'
        )
        if explained_w:
            parts.append(
                f'<rect x="{left}" y="{y + 2}" width="{max(explained_w - 2, 1):.1f}" height="12" rx="3" '
                f'fill="{BLUE}"><title>{task}: {total - missed} explained</title></rect>'
            )
        if missed_w:
            parts.append(
                f'<rect x="{left + explained_w:.1f}" y="{y + 2}" width="{max(missed_w - 2, 1):.1f}" height="12" '
                f'rx="3" fill="{ORANGE}"><title>{task}: {missed} not explained</title></rect>'
            )
        parts.append(f'<text class="value" x="{left + total * scale + 6:.1f}" y="{y + 12}">{total}</text>')
    parts.append("</svg>")
    return "".join(parts)


def values():
    p1, p2, p3 = load("phase1"), load("phase2"), load("phase3")
    m2, h5, h7, h6, spot = p2["metrics"], p3["h5"], p3["h7"], p3["h6"], p3["blind_spot"]
    banking_files = [s for s in p3["files"] if s.endswith("banking")]
    shares = [p3["files"][s]["old"]["incompatible_share"] for s in banking_files]
    faithful = [p3["regrade_equals_recorded"][s] for s in banking_files]
    reads = p2["post_completion_reads_old_minus_new"]
    out = {
        "p1_attempted": f"{p1['attempted_old']:,}",
        "p1_errors": str(p1["errors_old"] + p1["errors_new"]),
        "p1_comparable": str(p1["comparable_banking_tasks"]),
        "p1_p1_old": str(p1["p1_flagged_tasks_old"]),
        "p1_p1_new": str(p1["p1_flagged_tasks_new"]),
        "p1_p1_pairs": str(p1["p1_paired_verdicts"]),
        "p1_p3_tasks": str(p1["p3_paired_verdicts"]),
        "p1_p3_old": str(p1["p3_flagged_tasks_old"]),
        "p1_p3_new": str(p1["p3_flagged_tasks_new"]),
        "p1_f3_variants": str(p1["f3_variants"]),
        "p1_f3_pass": pct(p1["f3_pass_rate_new"]),
        "p1_f6_tasks": str(p1["f6_tasks"]),
        "p1_f6_pass": pct(p1["f6_pass_share_new"]),
        "p1_f1_airline": str(p1["f1_pass_airline_new"]),
        "p1_f1_verdicts": str(p1["f1_airline_verdicts"]),
        "p1_control": str(p1["control_mismatches"]),
        "p1_changed": str(p1["changed_tasks"]),
        "p2_gate": num(m2["gate_true_success_new"], 3),
        "p2_h3_finds": str(m2["h3_finds"]),
        "p2_h3_recall": pct(m2["h3_recall"]),
        "p2_reads_diff": num(m2["h4_extra_read_diff"]),
        "p2_reads_ci": interval(m2["h4_extra_read_diff_low"], m2["h4_extra_read_diff_high"]),
        "p2_true_diff": num(m2["h4_true_diff"], 3),
        "p2_true_ci": interval(m2["h4_true_diff_low"], m2["h4_true_diff_high"], 3),
        "p2_true_points": str(round(-100 * m2["h4_true_diff"])),
        "p2_converged_old": str(m2["converged_seeds_old"]),
        "p2_converged_new": str(m2["converged_seeds_new"]),
        "p2_post_reads": num(reads[0]),
        "p2_post_reads_ci": interval(reads[1], reads[2]),
        "p3_events": str(h5["events"]),
        "p3_explained": str(h5["explained"]),
        "p3_unexplained": str(h5["events"] - h5["explained"]),
        "p3_recall": num(h5["recall"]),
        "p3_recall_pct": pct(h5["recall"]),
        "p3_ci": interval(*h5["ci"]),
        "p3_missed_tasks": ", ".join(t.replace("task_", "") for t in sorted(h5["unexplained_by_task"])),
        "p3_missed_task_count": str(len(h5["unexplained_by_task"])),
        "p3_event_tasks": str(len(h5["events_by_task"])),
        "p3_eligible": str(h5["baseline_task_flag"]["positive"] + h5["baseline_task_flag"]["negative"]),
        "p3_p1_precision_runs": str(h5["p1_trigger_precision"]["runs"]),
        "p3_trigger_agree": str(h5["p1_trigger_precision"]["runs"] - h5["p1_trigger_precision"]["events"]),
        "p3_p3_runs": str(h5["p3_trigger"]["runs"]),
        "p3_p3_events": str(h5["p3_trigger"]["events"]),
        "p3_b1_enrichment": num(h5["baseline_task_flag"]["enrichment"]),
        "p3_banking_files": str(len(banking_files)),
        "p3_runs_per_file": str(p3["files"][banking_files[0]]["new"]["runs"]),
        "p3_incompatible_range": f"{pct(min(shares))}–{pct(max(shares))}",
        "p3_faithful_min": str(min(f["equal_to_recorded"] for f in faithful)),
        "p3_faithful_max": str(max(f["equal_to_recorded"] for f in faithful)),
        "p3_faithful_of": str(max(f["graded"] for f in faithful)),
        "p3_airline_runs": str(h7["airline"]["runs"]),
        "p3_airline_floats": str(h7["airline"]["schema_float_runs"]),
        "p3_reward1": str(h7["banking"]["reward_one_runs"]),
        "p3_late_logs": str(h7["banking"]["late_log_runs"]),
        "p3_access_early": str(h7["banking"]["access_before_facts_runs"]),
        "h6_sample": str(sum(h6["sample"].values())),
        "h6_reward1": str(h6["sample"]["S1"] + h6["sample"]["S2"]),
        "h6_kappa_v1": num(h6["agreement_per_run"]["v1"]["kappa"]),
        "h6_kappa_v2": num(h6["agreement_per_run"]["v2"]["kappa"]),
        "h6_agree_v2": str(h6["agreement_per_run"]["v2"]["agree"]),
        "h6_hidden": str(h6["confirmed_hidden_in_reward1"]),
        "spot_tasks": str(len(spot["affected_tasks"])),
        "spot_task_list": ", ".join(t.replace("task_", "") for t in spot["affected_tasks"]),
        "spot_reward1": str(spot["reward1_runs_total"]),
        "spot_differ": str(len(spot["reports_differ_from_gold"])),
        "spot_rows": "".join(
            f"<tr><td>{NAMES[d['file']]}</td><td>{d['task'].replace('task_', '')}</td><td>{d['trial']}</td>"
            f'<td class="num">{d["gold_reports"]} → {d["filed_reports"]}</td><td>{difference_text(d)}</td></tr>'
            for d in spot["reports_differ_from_gold"]
        ),
        "chart_fix": fix_bars(p1),
        "chart_seeds": seeds_chart(p2),
        "chart_events": events_chart(p3),
    }
    return out


def render():
    template = (REPO / "docs" / "template.html").read_text()
    data = values()
    used = set(re.findall(r"\{\{(\w+)\}\}", template))
    if unknown := used - data.keys():
        raise SystemExit(f"template names with no value: {sorted(unknown)}")
    if unused := data.keys() - used:
        raise SystemExit(f"values the template never uses: {sorted(unused)}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: data[m.group(1)], template)


def main():
    (REPO / "docs" / "index.html").write_text(render())
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
