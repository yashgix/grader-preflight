"""Compare two result directories: relation tables, pre-registered checks, control and metrics."""

import hashlib
import json
from pathlib import Path

BANK = "banking_knowledge"
CONTROL_DOMAINS = frozenset({"airline", "retail"})
P_RELATIONS = frozenset({"P1", "P3", "P5"})
IN_GOLD = "in gold"
NOT_IN_GOLD = "not in gold"
F1_LABELS = {True: "True", False: "False", None: "unknown"}
F1_ORDER = {True: 0, False: 1, None: 2}


def _load(directory):
    """Every JSON result file in a directory, each tagged with its file name."""
    files = []
    for path in sorted(Path(directory).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data["_file"] = path.name
            files.append(data)
    return files


def _variants(files):
    for data in files:
        domain = data.get("domain")
        for variant in data.get("variants") or []:
            if isinstance(variant, dict):
                yield domain, variant


def _tasks(files):
    found = {}
    for data in files:
        domain = data.get("domain")
        for task in data.get("tasks") or []:
            if isinstance(task, dict):
                found[(domain, task.get("id"))] = task
    return found


def _row():
    return {
        "attempted": 0,
        "attempted_tasks": set(),
        "verdicts": 0,
        "verdict_tasks": set(),
        "errors": 0,
        "error_tasks": set(),
        "flagged_variants": 0,
        "flagged_tasks": set(),
    }


def _count(row, variant):
    task_id = variant.get("task_id")
    row["attempted"] += 1
    row["attempted_tasks"].add(task_id)
    outcome = variant.get("outcome")
    if outcome == "verdict":
        row["verdicts"] += 1
        row["verdict_tasks"].add(task_id)
        if variant.get("flagged") is True:
            row["flagged_variants"] += 1
            row["flagged_tasks"].add(task_id)
    elif outcome == "error":
        row["errors"] += 1
        row["error_tasks"].add(task_id)


def _relation_rows(files):
    """Attempted, verdicts, errors and flags per domain and relation."""
    rows = {}
    for domain, variant in _variants(files):
        _count(rows.setdefault((domain, variant.get("relation")), _row()), variant)
    return rows


def _p1_rows(files):
    """The P1 counts split by whether the inserted tool also appears in the gold run."""
    rows = {}
    for domain, variant in _variants(files):
        if variant.get("relation") != "P1":
            continue
        label = IN_GOLD if variant.get("inserted_in_gold") is True else NOT_IN_GOLD
        _count(rows.setdefault((domain, label), _row()), variant)
    return rows


def _flagged(rows, domain, relation):
    row = rows.get((domain, relation))
    return set(row["flagged_tasks"]) if row else set()


def _without_changed(task_ids, domain, changed):
    return {task_id for task_id in task_ids if (domain, task_id) not in changed}


def _outcome_count(files, domain, relation, outcome, changed, inserted_in_gold=None):
    count = 0
    for variant_domain, variant in _variants(files):
        if variant_domain != domain or variant.get("relation") != relation:
            continue
        if (variant_domain, variant.get("task_id")) in changed:
            continue
        if inserted_in_gold is not None and (variant.get("inserted_in_gold") is True) != inserted_in_gold:
            continue
        if variant.get("outcome") == outcome:
            count += 1
    return count


def _changed_tasks(old_tasks, new_tasks):
    changed = {}
    for key in sorted(set(old_tasks) & set(new_tasks), key=lambda item: (str(item[0]), str(item[1]))):
        old = old_tasks[key]
        new = new_tasks[key]
        parts = []
        if old.get("task_fingerprint") != new.get("task_fingerprint"):
            parts.append("task_fingerprint")
        if old.get("gold_output_fingerprint") != new.get("gold_output_fingerprint"):
            parts.append("gold_output_fingerprint")
        if parts:
            changed[key] = parts
    return changed


def _variant_key(domain, variant):
    return (domain, variant.get("task_id"), variant.get("relation"), str(variant.get("detail")))


def _control_mismatches(old_files, new_files, old_tasks, new_tasks, changed):
    old_map = {_variant_key(domain, variant): variant for domain, variant in _variants(old_files)}
    new_map = {_variant_key(domain, variant): variant for domain, variant in _variants(new_files)}
    identical = {
        key
        for key in set(old_tasks) & set(new_tasks)
        if key[0] in CONTROL_DOMAINS and key not in changed
    }
    mismatches = []
    for key in sorted(set(old_map) | set(new_map), key=lambda item: tuple(str(part) for part in item)):
        if (key[0], key[1]) not in identical:
            continue
        old = old_map.get(key)
        new = new_map.get(key)
        if old is None or new is None:
            missing = "old" if old is None else "new"
            mismatches.append(f"{key[0]}/{key[1]} {key[2]} '{key[3]}': missing on {missing}")
            continue
        old_pair = (old.get("outcome"), old.get("reward"))
        new_pair = (new.get("outcome"), new.get("reward"))
        if old_pair != new_pair:
            mismatches.append(f"{key[0]}/{key[1]} {key[2]} '{key[3]}': old={old_pair} new={new_pair}")
    return mismatches


def _records_digest(files):
    entries = sorted(
        (
            (str(data.get("domain")), data.get("_file", ""), str(data.get("records_sha256", "")))
            for data in files
        ),
        key=lambda entry: (entry[0], entry[1]),
    )
    text = "\n".join(entry[2] for entry in entries)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _f1_row():
    return {"attempted": 0, "verdicts": 0, "errors": 0, "passed_variants": 0, "passed_tasks": set()}


def _f1_split(files, tasks):
    """F1 counts per domain, split by whether the gold run changes the DB."""
    split = {}
    for domain, variant in _variants(files):
        if variant.get("relation") != "F1":
            continue
        task = tasks.get((domain, variant.get("task_id")))
        static = task.get("static") if isinstance(task, dict) else None
        value = static.get("gold_changes_db") if isinstance(static, dict) else None
        bucket = value if isinstance(value, bool) else None
        row = split.setdefault((domain, bucket), _f1_row())
        row["attempted"] += 1
        outcome = variant.get("outcome")
        if outcome == "verdict":
            row["verdicts"] += 1
            if variant.get("reward") == 1.0:
                row["passed_variants"] += 1
                row["passed_tasks"].add(variant.get("task_id"))
        elif outcome == "error":
            row["errors"] += 1
    return split


def _static_counts(files):
    """Per side: relations skipped as identical to gold, tasks whose P5 call was not inert, no-effect writes."""
    counts = {"identical_to_gold": {}, "p5_not_inert": {}, "no_effect_writes": {}}
    for data in files:
        domain = data.get("domain")
        for task in data.get("tasks") or []:
            static = task.get("static") if isinstance(task, dict) else None
            if not isinstance(static, dict):
                continue
            skipped = static.get("identical_to_gold")
            if isinstance(skipped, list):
                for relation in skipped:
                    key = (domain, relation)
                    counts["identical_to_gold"][key] = counts["identical_to_gold"].get(key, 0) + 1
            if static.get("p5_not_inert") is True:
                counts["p5_not_inert"][domain] = counts["p5_not_inert"].get(domain, 0) + 1
            written = static.get("no_effect_writes")
            if isinstance(written, int) and not isinstance(written, bool):
                counts["no_effect_writes"][domain] = counts["no_effect_writes"].get(domain, 0) + written
    return counts


def _h2(variants):
    f3 = [(domain, variant) for domain, variant in variants if variant.get("relation") == "F3"]
    f3_verdicts = [variant for _, variant in f3 if variant.get("outcome") == "verdict"]
    f3_passes = sum(1 for variant in f3_verdicts if variant.get("reward") == 1.0)
    f3_errors = sum(1 for _, variant in f3 if variant.get("outcome") == "error")
    f3_rate = f3_passes / len(f3_verdicts) if f3_verdicts else 0.0

    f1_airline = [
        (domain, variant)
        for domain, variant in variants
        if domain == "airline" and variant.get("relation") == "F1"
    ]
    f1_tasks = {
        variant.get("task_id")
        for _, variant in f1_airline
        if variant.get("outcome") == "verdict" and variant.get("reward") == 1.0
    }
    f1_errors = sum(1 for _, variant in f1_airline if variant.get("outcome") == "error")

    f6 = [
        (domain, variant)
        for domain, variant in variants
        if domain == BANK and variant.get("relation") == "F6"
    ]
    f6_attempted_tasks = {variant.get("task_id") for _, variant in f6}
    f6_verdicts = [variant for _, variant in f6 if variant.get("outcome") == "verdict"]
    f6_tasks = {variant.get("task_id") for variant in f6_verdicts}
    f6_passes = {variant.get("task_id") for variant in f6_verdicts if variant.get("reward") == 1.0}
    f6_errors = sum(1 for _, variant in f6 if variant.get("outcome") == "error")
    f6_error_tasks = {variant.get("task_id") for _, variant in f6 if variant.get("outcome") == "error"}
    f6_share = len(f6_passes) / len(f6_tasks) if f6_tasks else 0.0

    p_flags = sum(
        1
        for _, variant in variants
        if variant.get("relation") in P_RELATIONS
        and variant.get("outcome") == "verdict"
        and variant.get("flagged") is True
    )
    p_errors = sum(
        1
        for _, variant in variants
        if variant.get("relation") in P_RELATIONS and variant.get("outcome") == "error"
    )
    held = f3_rate <= 0.01 and len(f1_tasks) >= 20 and f6_share >= 0.5 and p_flags > 0
    return {
        "held": held,
        "f3_rate": f3_rate,
        "f3_passes": f3_passes,
        "f3_verdicts": len(f3_verdicts),
        "f3_errors": f3_errors,
        "f1_tasks": len(f1_tasks),
        "f1_errors": f1_errors,
        "f6_share": f6_share,
        "f6_passes": len(f6_passes),
        "f6_tasks": len(f6_tasks),
        "f6_attempted_tasks": len(f6_attempted_tasks),
        "f6_errors": f6_errors,
        "f6_error_tasks": len(f6_error_tasks),
        "p_flags": p_flags,
        "p_errors": p_errors,
    }


def _names(task_ids):
    return ", ".join(sorted(str(task_id) for task_id in task_ids)) or "none"


def _table_row(*cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _sorted_rows(rows):
    return sorted(rows.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))


def _sorted_f1(split):
    return sorted(split.items(), key=lambda item: (str(item[0][0]), F1_ORDER[item[0][1]]))


def compare(old_dir, new_dir, out_path):
    """Write a Markdown report and metrics.json beside it; return the metrics object."""
    old_files = _load(old_dir)
    new_files = _load(new_dir)
    old_rows = _relation_rows(old_files)
    new_rows = _relation_rows(new_files)
    old_p1_rows = _p1_rows(old_files)
    new_p1_rows = _p1_rows(new_files)
    old_tasks = _tasks(old_files)
    new_tasks = _tasks(new_files)
    changed = _changed_tasks(old_tasks, new_tasks)

    old_all = list(_variants(old_files))
    new_all = list(_variants(new_files))

    old_p1 = _without_changed(_flagged(old_rows, BANK, "P1"), BANK, changed)
    new_p1_all = _without_changed(_flagged(new_rows, BANK, "P1"), BANK, changed)
    new_p1_in_gold = _without_changed(_flagged(new_p1_rows, BANK, IN_GOLD), BANK, changed)
    new_p1_non_gold = _without_changed(_flagged(new_p1_rows, BANK, NOT_IN_GOLD), BANK, changed)
    h1a_held = bool(old_p1) and not new_p1_non_gold

    old_p1_errors = _outcome_count(old_files, BANK, "P1", "error", changed)
    new_p1_errors = _outcome_count(new_files, BANK, "P1", "error", changed)
    new_p1_non_gold_errors = _outcome_count(new_files, BANK, "P1", "error", changed, False)
    new_p1_in_gold_errors = _outcome_count(new_files, BANK, "P1", "error", changed, True)

    old_p3 = _without_changed(_flagged(old_rows, BANK, "P3"), BANK, changed)
    new_p3 = _without_changed(_flagged(new_rows, BANK, "P3"), BANK, changed)
    h1b_held = bool(old_p3) and not new_p3
    old_p3_errors = _outcome_count(old_files, BANK, "P3", "error", changed)
    new_p3_errors = _outcome_count(new_files, BANK, "P3", "error", changed)

    h2 = _h2(new_all)
    mismatches = _control_mismatches(old_files, new_files, old_tasks, new_tasks, changed)
    control_passed = not mismatches
    f1_old = _f1_split(old_files, old_tasks)
    f1_new = _f1_split(new_files, new_tasks)
    static_old = _static_counts(old_files)
    static_new = _static_counts(new_files)

    lines = [
        "# Preflight comparison",
        "",
        f"OLD: `{Path(old_dir)}`",
        f"NEW: `{Path(new_dir)}`",
        "",
        "## Fixture changed",
        "",
    ]
    if changed:
        for (domain, task_id), parts in sorted(
            changed.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
        ):
            lines.append(f"- {domain}/{task_id}: {', '.join(parts)}")
    else:
        lines.append("None.")

    lines += ["", "## Per domain and relation", ""]
    lines.append(
        "| Side | Domain | Relation | Attempted | Verdicts | Errors | Flagged variants | Flagged tasks | "
        "Attempted tasks | Verdict tasks | Error tasks |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for side, rows in (("old", old_rows), ("new", new_rows)):
        for (domain, relation), row in _sorted_rows(rows):
            lines.append(
                _table_row(
                    side,
                    domain,
                    relation,
                    row["attempted"],
                    row["verdicts"],
                    row["errors"],
                    row["flagged_variants"],
                    len(row["flagged_tasks"]),
                    len(row["attempted_tasks"]),
                    len(row["verdict_tasks"]),
                    len(row["error_tasks"]),
                )
            )

    lines += ["", "## P1 split by inserted_in_gold", ""]
    lines.append(
        "| Side | Domain | Inserted in gold | Attempted | Verdicts | Errors | Flagged variants | Flagged tasks | "
        "Attempted tasks | Verdict tasks | Error tasks |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for side, rows in (("old", old_p1_rows), ("new", new_p1_rows)):
        for (domain, label), row in _sorted_rows(rows):
            lines.append(
                _table_row(
                    side,
                    domain,
                    label,
                    row["attempted"],
                    row["verdicts"],
                    row["errors"],
                    row["flagged_variants"],
                    len(row["flagged_tasks"]),
                    len(row["attempted_tasks"]),
                    len(row["verdict_tasks"]),
                    len(row["error_tasks"]),
                )
            )

    old_flags = {
        (domain, variant.get("task_id"), variant.get("relation"))
        for domain, variant in old_all
        if variant.get("outcome") == "verdict" and variant.get("flagged") is True
    }
    new_flags = {
        (domain, variant.get("task_id"), variant.get("relation"))
        for domain, variant in new_all
        if variant.get("outcome") == "verdict" and variant.get("flagged") is True
    }
    missing_flags = sorted(old_flags - new_flags, key=lambda item: tuple(str(part) for part in item))
    lines += ["", "## Flags on OLD and absent on NEW", ""]
    if missing_flags:
        for domain, task_id, relation in missing_flags:
            lines.append(f"- {domain}/{task_id} {relation}")
    else:
        lines.append("None.")

    lines += ["", "## Hypotheses", ""]
    lines.append(
        f"H1(a): {'held' if h1a_held else 'not held'} - OLD P1 flagged banking tasks: "
        f"{len(old_p1)} ({_names(old_p1)}), errors: {old_p1_errors}; NEW P1 flagged non-in-gold tasks: "
        f"{len(new_p1_non_gold)} ({_names(new_p1_non_gold)}), errors: {new_p1_non_gold_errors}; "
        f"NEW P1 in-gold flagged tasks: {len(new_p1_in_gold)} ({_names(new_p1_in_gold)}), errors: "
        f"{new_p1_in_gold_errors}; NEW P1 flagged tasks all: {len(new_p1_all)}; NEW P1 errors all: "
        f"{new_p1_errors}."
    )
    lines.append(
        f"H1(b): {'held' if h1b_held else 'not held'} - OLD P3 flagged banking tasks: "
        f"{len(old_p3)} ({_names(old_p3)}), errors: {old_p3_errors}; NEW P3 flagged banking tasks: "
        f"{len(new_p3)} ({_names(new_p3)}), errors: {new_p3_errors}."
    )
    lines.append(
        f"H2: {'held' if h2['held'] else 'not held'} - F3 pass rate NEW: "
        f"{h2['f3_rate']:.4f} ({h2['f3_passes']}/{h2['f3_verdicts']}; {h2['f3_errors']} errors); "
        f"F1 airline passing tasks NEW: {h2['f1_tasks']} ({h2['f1_errors']} errors); "
        f"F6 banking pass share NEW: {h2['f6_share']:.4f} "
        f"({h2['f6_passes']}/{h2['f6_tasks']}; {h2['f6_errors']} errors on {h2['f6_error_tasks']} tasks; "
        f"{h2['f6_attempted_tasks']} attempted tasks); "
        f"P-relation flagged variants NEW: {h2['p_flags']} ({h2['p_errors']} errors)."
    )

    lines += ["", "## Control", "", f"Control: {'passed' if control_passed else 'failed'}"]
    lines.append(f"Mismatches: {len(mismatches)}.")
    for mismatch in mismatches:
        lines.append(f"- {mismatch}")

    lines += ["", "## F1 split by domain and gold_changes_db", ""]
    for side, split in (("old", f1_old), ("new", f1_new)):
        if not split:
            lines.append(f"- {side}: no F1 variants")
        for (domain, bucket), row in _sorted_f1(split):
            parts = [
                f"attempted={row['attempted']}",
                f"verdicts={row['verdicts']}",
                f"errors={row['errors']}",
                f"passed_variants={row['passed_variants']}",
                f"passed_tasks={len(row['passed_tasks'])}",
            ]
            lines.append(f"- {side} {domain} gold_changes_db={F1_LABELS[bucket]}: " + " ".join(parts))

    lines += ["", "## Relations skipped as identical to gold", ""]
    lines.append("| Side | Domain | Relation | Tasks |")
    lines.append("| --- | --- | --- | ---: |")
    for side, counts in (("old", static_old), ("new", static_new)):
        skipped = counts["identical_to_gold"]
        if not skipped:
            lines.append(_table_row(side, "-", "-", 0))
        for (domain, relation), count in _sorted_rows(skipped):
            lines.append(_table_row(side, domain, relation, count))

    lines += ["", "## Static counts", ""]
    lines.append("| Side | Domain | p5_not_inert tasks | no_effect_writes |")
    lines.append("| --- | --- | ---: | ---: |")
    for side, counts in (("old", static_old), ("new", static_new)):
        domains = set(counts["p5_not_inert"]) | set(counts["no_effect_writes"])
        if not domains:
            lines.append(_table_row(side, "-", 0, 0))
        for domain in sorted(domains, key=str):
            lines.append(
                _table_row(
                    side,
                    domain,
                    counts["p5_not_inert"].get(domain, 0),
                    counts["no_effect_writes"].get(domain, 0),
                )
            )

    lines += ["", "## Records", ""]
    for side, files in (("old", old_files), ("new", new_files)):
        if not files:
            lines.append(f"- {side}: no JSON files")
        for data in sorted(files, key=lambda item: (str(item.get("domain")), item.get("_file", ""))):
            lines.append(
                f"- {side} {data.get('domain')}/{data.get('_file')}: {data.get('records_sha256')}"
            )

    metrics = {
        "h1a_held": int(h1a_held),
        "h1b_held": int(h1b_held),
        "h2_held": int(h2["held"]),
        "control_passed": int(control_passed),
        "control_mismatches": len(mismatches),
        "p1_flagged_tasks_old": len(old_p1),
        "p1_flagged_tasks_new": len(new_p1_all),
        "p1_in_gold_flagged_tasks_new": len(new_p1_in_gold),
        "p3_flagged_tasks_old": len(old_p3),
        "p3_flagged_tasks_new": len(new_p3),
        "f1_pass_airline_new": h2["f1_tasks"],
        "f3_pass_rate_new": h2["f3_rate"],
        "f6_pass_share_new": h2["f6_share"],
        "changed_tasks": len(changed),
        "attempted_old": sum(row["attempted"] for row in old_rows.values()),
        "attempted_new": sum(row["attempted"] for row in new_rows.values()),
        "errors_old": sum(row["errors"] for row in old_rows.values()),
        "errors_new": sum(row["errors"] for row in new_rows.values()),
        "records_digest_old": _records_digest(old_files),
        "records_digest_new": _records_digest(new_files),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
