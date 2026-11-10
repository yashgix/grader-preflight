"""Compare two result directories case by case: relation tables, pre-registered checks, control and metrics."""

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


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def _changed_tasks(old_tasks, new_tasks):
    changed = {}
    common = set(old_tasks) & set(new_tasks)
    for key in sorted(common, key=lambda item: (str(item[0]), str(item[1]))):
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


def _one_sided_tasks(old_tasks, new_tasks):
    sides = {}
    for key in set(old_tasks) - set(new_tasks):
        sides[key] = "old"
    for key in set(new_tasks) - set(old_tasks):
        sides[key] = "new"
    return sides


def _variant_key(domain, variant):
    return (
        domain,
        variant.get("task_id"),
        variant.get("relation"),
        variant.get("detail"),
        _canonical_json(variant.get("calls") or []),
    )


def _variant_sort_key(key):
    return tuple(str(part) for part in key)


def _variant_groups(files):
    groups = {}
    for domain, variant in _variants(files):
        key = _variant_key(domain, variant)
        groups.setdefault(key, []).append(variant)
    for variants in groups.values():
        variants.sort(key=_canonical_json)
    return groups


def _pair_variants(old_files, new_files, comparable):
    old_groups = _variant_groups(old_files)
    new_groups = _variant_groups(new_files)
    pairs = []
    unpaired = []
    for key in sorted(set(old_groups) | set(new_groups), key=_variant_sort_key):
        task_key = (key[0], key[1])
        if task_key not in comparable:
            continue
        old_list = old_groups.get(key, [])
        new_list = new_groups.get(key, [])
        paired = min(len(old_list), len(new_list))
        for index in range(paired):
            pairs.append(
                {
                    "domain": key[0],
                    "task_id": key[1],
                    "relation": key[2],
                    "detail": key[3],
                    "old": old_list[index],
                    "new": new_list[index],
                }
            )
        for variant in old_list[paired:]:
            unpaired.append({"domain": key[0], "side": "old", "variant": variant})
        for variant in new_list[paired:]:
            unpaired.append({"domain": key[0], "side": "new", "variant": variant})
    return pairs, unpaired


def _records_digest(files):
    entries = sorted(
        ((str(data.get("domain")), data.get("_file", ""), str(data.get("records_sha256", ""))) for data in files),
        key=lambda entry: (entry[0], entry[1]),
    )
    text = chr(10).join(entry[2] for entry in entries)
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
    """The four parts of H2 on the new side; a part with no verdict behind it is not held."""
    f3 = [variant for _, variant in variants if variant.get("relation") == "F3"]
    f3_verdicts = [variant for variant in f3 if variant.get("outcome") == "verdict"]
    f3_passes = sum(1 for variant in f3_verdicts if variant.get("reward") == 1.0)
    f3_rate = f3_passes / len(f3_verdicts) if f3_verdicts else 0.0
    f3_held = bool(f3_verdicts) and f3_rate <= 0.01

    f1_airline = [variant for domain, variant in variants if domain == "airline" and variant.get("relation") == "F1"]
    f1_verdicts = [variant for variant in f1_airline if variant.get("outcome") == "verdict"]
    f1_tasks = {variant.get("task_id") for variant in f1_verdicts if variant.get("reward") == 1.0}

    f6 = [variant for domain, variant in variants if domain == BANK and variant.get("relation") == "F6"]
    f6_attempted_tasks = {variant.get("task_id") for variant in f6}
    f6_verdicts = [variant for variant in f6 if variant.get("outcome") == "verdict"]
    f6_verdict_tasks = {variant.get("task_id") for variant in f6_verdicts}
    f6_passes = {variant.get("task_id") for variant in f6_verdicts if variant.get("reward") == 1.0}
    f6_share = len(f6_passes) / len(f6_verdict_tasks) if f6_verdict_tasks else 0.0
    f6_held = bool(f6_verdict_tasks) and f6_share >= 0.5

    p_flags = sum(
        1
        for _, variant in variants
        if variant.get("relation") in P_RELATIONS
        and variant.get("outcome") == "verdict"
        and variant.get("flagged") is True
    )
    p_errors = sum(
        1 for _, variant in variants if variant.get("relation") in P_RELATIONS and variant.get("outcome") == "error"
    )
    return {
        "held": f3_held and len(f1_tasks) >= 20 and f6_held and p_flags > 0,
        "f3_rate": f3_rate,
        "f3_passes": f3_passes,
        "f3_verdicts": len(f3_verdicts),
        "f3_errors": sum(1 for variant in f3 if variant.get("outcome") == "error"),
        "f3_held": f3_held,
        "f1_tasks": len(f1_tasks),
        "f1_verdicts": len(f1_verdicts),
        "f1_errors": sum(1 for variant in f1_airline if variant.get("outcome") == "error"),
        "f6_share": f6_share,
        "f6_held": f6_held,
        "f6_passes": len(f6_passes),
        "f6_attempted_tasks": len(f6_attempted_tasks),
        "f6_verdict_tasks": len(f6_verdict_tasks),
        "f6_errors": sum(1 for variant in f6 if variant.get("outcome") == "error"),
        "f6_error_tasks": len({variant.get("task_id") for variant in f6 if variant.get("outcome") == "error"}),
        "p_flags": p_flags,
        "p_errors": p_errors,
    }


def _db_diff_counts(pairs):
    counts = {}
    for pair in pairs:
        diff = pair["old"].get("db_diff")
        fields = sorted(str(field) for field in diff) if isinstance(diff, list) else []
        label = _canonical_json(fields)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _format_db_diff_sets(counts):
    if not counts:
        return "none"
    return ", ".join(f"{label}: {counts[label]}" for label in sorted(counts))


def _h1(relation, pairs, unpaired):
    group = []
    for pair in pairs:
        if pair["domain"] != BANK or pair["relation"] != relation:
            continue
        if relation == "P1" and (
            pair["old"].get("inserted_in_gold") is True or pair["new"].get("inserted_in_gold") is True
        ):
            continue
        group.append(pair)
    considered = [
        pair for pair in group if pair["old"].get("outcome") == "verdict" and pair["new"].get("outcome") == "verdict"
    ]
    old_flags = [pair for pair in considered if pair["old"].get("flagged") is True]
    new_flags = [pair for pair in considered if pair["new"].get("flagged") is True]
    bad_unpaired = [
        item
        for item in unpaired
        if item["domain"] == BANK
        and item["variant"].get("relation") == relation
        and not (relation == "P1" and item["variant"].get("inserted_in_gold") is True)
    ]
    old_errors = sum(1 for pair in group if pair["old"].get("outcome") == "error")
    new_errors = sum(1 for pair in group if pair["new"].get("outcome") == "error")
    fp_diff = sum(
        1 for pair in considered if pair["old"].get("output_fingerprint") != pair["new"].get("output_fingerprint")
    )
    return {
        "held": bool(considered) and bool(old_flags) and not new_flags and not bad_unpaired,
        "paired": len(considered),
        "old_flag_tasks": {pair["task_id"] for pair in old_flags},
        "new_flag_tasks": {pair["task_id"] for pair in new_flags},
        "unpaired": bad_unpaired,
        "old_errors": old_errors,
        "new_errors": new_errors,
        "fp_diff": fp_diff,
        "db_diff_counts": _db_diff_counts(old_flags),
    }


def _names(task_ids):
    return ", ".join(sorted(str(task_id) for task_id in task_ids)) or "none"


def _h1_line(label, data):
    held = "held" if data["held"] else "not held"
    old_names = _names(data["old_flag_tasks"])
    new_names = _names(data["new_flag_tasks"])
    diff_sets = _format_db_diff_sets(data["db_diff_counts"])
    paired = data["paired"]
    old_flags = len(data["old_flag_tasks"])
    new_flags = len(data["new_flag_tasks"])
    unpaired = len(data["unpaired"])
    old_errors = data["old_errors"]
    new_errors = data["new_errors"]
    fingerprint_diff = data["fp_diff"]
    return (
        f"{label}: {held} - paired verdicts: {paired}; "
        f"OLD flagged pairs: {old_flags} ({old_names}); "
        f"NEW flagged pairs: {new_flags} ({new_names}); "
        f"unpaired comparable: {unpaired}; "
        f"OLD errors: {old_errors}; NEW errors: {new_errors}; "
        f"output_fingerprints differ: {fingerprint_diff}; "
        f"OLD db_diff sets: {diff_sets}."
    )


def _h2_line(h2):
    held = "held" if h2["held"] else "not held"
    f3_part = "part held" if h2["f3_held"] else "part not held"
    f6_part = "part held" if h2["f6_held"] else "part not held"
    f3_rate = h2["f3_rate"]
    f3_passes = h2["f3_passes"]
    f3_verdicts = h2["f3_verdicts"]
    f3_errors = h2["f3_errors"]
    f1_tasks = h2["f1_tasks"]
    f1_verdicts = h2["f1_verdicts"]
    f1_errors = h2["f1_errors"]
    f6_share = h2["f6_share"]
    f6_passes = h2["f6_passes"]
    f6_verdict_tasks = h2["f6_verdict_tasks"]
    f6_attempted_tasks = h2["f6_attempted_tasks"]
    f6_errors = h2["f6_errors"]
    f6_error_tasks = h2["f6_error_tasks"]
    p_flags = h2["p_flags"]
    p_errors = h2["p_errors"]
    return (
        f"H2: {held} - F3 pass rate NEW: {f3_rate:.4f} "
        f"({f3_passes}/{f3_verdicts}; {f3_errors} errors; {f3_part}); "
        f"F1 airline passing tasks NEW: {f1_tasks} "
        f"({f1_verdicts} verdicts, {f1_errors} errors); "
        f"F6 banking pass share NEW: {f6_share:.4f} "
        f"({f6_passes}/{f6_verdict_tasks} tasks with a verdict; "
        f"{f6_attempted_tasks} attempted; {f6_errors} errors on {f6_error_tasks} tasks; "
        f"{f6_part}); "
        f"P-relation flagged variants NEW: {p_flags} ({p_errors} errors)."
    )


def _control_mismatches(old_tasks, new_tasks, changed, pairs, unpaired):
    mismatches = []
    for domain in sorted(CONTROL_DOMAINS):
        keys = sorted(
            {key for key in set(old_tasks) | set(new_tasks) if key[0] == domain},
            key=lambda key: str(key[1]),
        )
        for key in keys:
            if key not in old_tasks:
                mismatches.append(f"{domain}/{key[1]}: missing on old")
            elif key not in new_tasks:
                mismatches.append(f"{domain}/{key[1]}: missing on new")
            elif key in changed:
                parts = ", ".join(changed[key])
                mismatches.append(f"{domain}/{key[1]}: fixture changed ({parts})")
        for item in unpaired:
            if item["domain"] != domain:
                continue
            variant = item["variant"]
            detail = variant.get("detail")
            task_id = variant.get("task_id")
            relation = variant.get("relation")
            side = item["side"]
            mismatches.append(f"{domain}/{task_id} {relation} {detail!r}: unpaired on {side}")
        for pair in pairs:
            if pair["domain"] != domain:
                continue
            old = pair["old"]
            new = pair["new"]
            if old.get("outcome") != new.get("outcome") or old.get("reward") != new.get("reward"):
                old_pair = (old.get("outcome"), old.get("reward"))
                new_pair = (new.get("outcome"), new.get("reward"))
                task_id = old.get("task_id")
                relation = old.get("relation")
                detail = old.get("detail")
                mismatches.append(f"{domain}/{task_id} {relation} {detail!r}: old={old_pair} new={new_pair}")
    return mismatches


def _table_row(*cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _sorted_rows(rows):
    return sorted(rows.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))


def _sorted_f1(split):
    return sorted(split.items(), key=lambda item: (str(item[0][0]), F1_ORDER[item[0][1]]))


def _relation_table(side, rows):
    lines = []
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
    return lines


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
    one_sided = _one_sided_tasks(old_tasks, new_tasks)
    comparable = {key for key in set(old_tasks) & set(new_tasks) if key not in changed}
    pairs, unpaired = _pair_variants(old_files, new_files, comparable)
    new_all = list(_variants(new_files))

    old_p1_all = _without_changed(_flagged(old_p1_rows, BANK, IN_GOLD), BANK, changed) | _without_changed(
        _flagged(old_p1_rows, BANK, NOT_IN_GOLD), BANK, changed
    )
    new_p1_all = _without_changed(_flagged(new_p1_rows, BANK, IN_GOLD), BANK, changed) | _without_changed(
        _flagged(new_p1_rows, BANK, NOT_IN_GOLD), BANK, changed
    )
    new_p1_in_gold = _without_changed(_flagged(new_p1_rows, BANK, IN_GOLD), BANK, changed)
    old_p3 = _without_changed(_flagged(old_rows, BANK, "P3"), BANK, changed)
    new_p3 = _without_changed(_flagged(new_rows, BANK, "P3"), BANK, changed)

    h1a = _h1("P1", pairs, unpaired)
    h1b = _h1("P3", pairs, unpaired)
    h2 = _h2(new_all)
    mismatches = _control_mismatches(old_tasks, new_tasks, changed, pairs, unpaired)
    control_passed = not mismatches
    control_text = "passed" if control_passed else "failed"
    f1_old = _f1_split(old_files, old_tasks)
    f1_new = _f1_split(new_files, new_tasks)
    static_old = _static_counts(old_files)
    static_new = _static_counts(new_files)
    old_only_flags = [
        pair
        for pair in pairs
        if pair["old"].get("outcome") == "verdict"
        and pair["old"].get("flagged") is True
        and pair["new"].get("outcome") == "verdict"
        and pair["new"].get("flagged") is not True
    ]
    old_only_flags.sort(
        key=lambda pair: (str(pair["domain"]), str(pair["task_id"]), str(pair["relation"]), str(pair["detail"]))
    )

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
        for (domain, task_id), parts in sorted(changed.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
            joined = ", ".join(parts)
            lines.append(f"- {domain}/{task_id}: {joined}")
    else:
        lines.append("None.")

    lines += ["", "## One-sided tasks", ""]
    if one_sided:
        for (domain, task_id), side in sorted(one_sided.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
            lines.append(f"- {domain}/{task_id}: only on {side}")
    else:
        lines.append("None.")

    lines += ["", "## Unpaired comparable variants", ""]
    if unpaired:
        for item in unpaired:
            variant = item["variant"]
            detail = variant.get("detail")
            side = item["side"]
            domain = item["domain"]
            task_id = variant.get("task_id")
            relation = variant.get("relation")
            lines.append(f"- {side} {domain}/{task_id} {relation} {detail!r}")
    else:
        lines.append("None.")

    lines += ["", "## Per domain and relation", ""]
    lines.append(
        "| Side | Domain | Relation | Attempted | Verdicts | Errors | Flagged variants | Flagged tasks | "
        "Attempted tasks | Verdict tasks | Error tasks |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines += _relation_table("old", old_rows)
    lines += _relation_table("new", new_rows)

    lines += ["", "## P1 split by inserted_in_gold", ""]
    lines.append(
        "| Side | Domain | Inserted in gold | Attempted | Verdicts | Errors | Flagged variants | "
        "Flagged tasks | Attempted tasks | Verdict tasks | Error tasks |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines += _relation_table("old", old_p1_rows)
    lines += _relation_table("new", new_p1_rows)

    lines += ["", "## Flags on OLD and absent on NEW", ""]
    if old_only_flags:
        for pair in old_only_flags:
            domain = pair["domain"]
            task_id = pair["task_id"]
            relation = pair["relation"]
            detail = pair["detail"]
            lines.append(f"- {domain}/{task_id} {relation} {detail!r}")
    else:
        lines.append("None.")

    lines += ["", "## Hypotheses", ""]
    lines.append(_h1_line("H1(a)", h1a))
    lines.append(_h1_line("H1(b)", h1b))
    lines.append(_h2_line(h2))

    lines += ["", "## Control", "", f"Control: {control_text}"]
    lines.append(f"Mismatches: {len(mismatches)}.")
    for mismatch in mismatches:
        lines.append(f"- {mismatch}")

    lines += ["", "## F1 split by domain and gold_changes_db", ""]
    for side, split in (("old", f1_old), ("new", f1_new)):
        if not split:
            lines.append(f"- {side}: no F1 variants")
        for (domain, bucket), row in _sorted_f1(split):
            attempted = row["attempted"]
            verdicts = row["verdicts"]
            errors = row["errors"]
            passed_variants = row["passed_variants"]
            passed_tasks = len(row["passed_tasks"])
            parts = [
                f"attempted={attempted}",
                f"verdicts={verdicts}",
                f"errors={errors}",
                f"passed_variants={passed_variants}",
                f"passed_tasks={passed_tasks}",
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
            domain = data.get("domain")
            file_name = data.get("_file")
            digest = data.get("records_sha256")
            lines.append(f"- {side} {domain}/{file_name}: {digest}")

    metrics = {
        "h1a_held": int(h1a["held"]),
        "h1b_held": int(h1b["held"]),
        "h2_held": int(h2["held"]),
        "control_passed": int(control_passed),
        "control_mismatches": len(mismatches),
        "p1_flagged_tasks_old": len(old_p1_all),
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
        "unpaired_comparable": len(unpaired),
        "records_digest_old": _records_digest(old_files),
        "records_digest_new": _records_digest(new_files),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return metrics
