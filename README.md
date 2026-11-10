# grader-preflight

An LLM-free preflight for graders of stateful tool environments. It takes a known-good run, makes small changes that
should not matter (an extra read, `25` written as `25.0`, a call to a tool that does not exist) and changes that
should (a dropped write, an empty run, verification logged after the writes it gates), then checks that the grader's
verdict moves the right way. No language model is called: every variant is built from the gold run, from the
environment's own tool metadata, and from the live DB state.

It targets τ²-bench's grader before and after the v1.0.1 fixes (`pre-v1.0.1` at `b51a6d6`, `v1.0.1` at `fc0055d`):
does it find the bugs that were fixed by hand?

**Status:** pre-registered and built; the experiment has not been run. The hypotheses, thresholds, seeds and the four
amendments that fix the protocol are in [PREREGISTRATION.md](PREREGISTRATION.md), committed before any experiment.

## Scope

Only the environment-state terms of the reward are exercised: every verdict quoted here is the DB-state verdict of
`EnvironmentEvaluator.calculate_reward`, called unchanged. The COMMUNICATE, NL_ASSERTION and ACTION terms are out of
scope. Tasks are those with `DB` in their `reward_basis`, in `banking_knowledge` (the `grep_only` retrieval variant, so
it runs offline), `airline` and `retail`. Telecom is not part of phase 1.

On v1.0.1 banking logs a read to the hashed agent DB only when the read is on a per-task allowlist that τ²-bench's
runner builds. The adapter derives the same allowlist with tau2's own `_derive_read_log_allowlist` and passes it to the
live environment and to the grader, so banking is graded in the configuration τ²-bench actually runs.

## Relations

| Id | Change to the gold run | Expected verdict |
| --- | --- | --- |
| P1 | insert one call to a declared read tool, its arguments taken from earlier calls in the run | pass |
| P3 | respell numeric arguments with the same value (`25` ↔ `25.0`, wrapper JSON included) | pass |
| P5 | insert a call to `no_such_tool` with `{}` arguments | pass |
| F1 | run nothing | fail |
| F3 | drop one write whose removal changes the live end state | fail |
| F6 | move a banking `log_verification` after the writes it gates | fail |

A flag is a variant whose verdict differs from the one above.

- **P1** uses the assistant toolkit's direct reads plus its discoverable reads, and a discoverable read is a candidate
  whether or not the gold run has unlocked it. A tool counts as a read when the toolkit's declared `tool_type(name)`
  says so. A discoverable read is called the way an agent calls it, through `call_discoverable_agent_tool` with a JSON
  `arguments` string; when the run has not unlocked that tool before the insertion point, its
  `unlock_discoverable_agent_tool` call goes right before the read, so a variant is the gold run plus the read, or plus
  the unlock and the read. Required argument names come from the tool's JSON schema, with the decorated signature
  settling optionality when only a bare parameter mapping is available; each required value is taken from the
  same-named argument of an earlier call (inside wrapper `arguments` strings too) and the read goes right after the
  last call that supplied one. The variant is kept only if replaying the prefix through the inserted read ends in a
  tool message that is not an error. At most five variants per task: candidates are shuffled with the seed and tried
  in that order, so a candidate that fails its replay never takes a slot. `inserted_in_gold` marks a variant whose
  inserted tool also appears in the gold run; H1(a) counts those apart on both sides, because v1.0.1 keeps logging
  reads the gold run itself requires by design.
- **P3** respells every numeric argument, integral `int` to `float` and integral `float` to `int`, inside wrapper
  `arguments` strings too, by rewriting the JSON number tokens so the parsed value is unchanged. Booleans are not
  numbers, and there is no variant when nothing changes (`identical_to_gold`).
- **P5** is kept only when the agent and user DB hashes right after the inserted call equal those right before it.
  Otherwise the task gets no P5 variant and `p5_not_inert` in its static entry; if the validation replay itself raises,
  the exception is recorded in `relation_errors` and the variant is not built, because a check that did not complete
  is not a check that passed.
- **F3** skips writes whose removal changes no end state; they are counted as `no_effect_writes`.
- **F6** exists only for banking tasks whose `log_verification` is followed by a write.

No variant ever equals the gold run (an empty-gold task gets no F1, for instance); each skipped relation is recorded
as `identical_to_gold`.

Two static checks are recorded per task instead of as variants: `gold_errors` (S1: gold calls whose live output is an
error, including wrapper outputs starting with `Error:`) and `gold_changes_db` (S2: the gold run leaves the DB
different from a fresh environment).

## Running

Everything runs in the two read-only environments; nothing is installed and no key is present. Both entry points set
`LITELLM_LOCAL_MODEL_COST_MAP=True` before tau2 is imported.

```sh
OLD=/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python
NEW=/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python

# one domain on one version ($OLD is the grader before the fixes, $NEW the grader after them)
$OLD -m preflight run --domain airline --seed 0 --out results/old/airline.json

# all three domains on both sides, then the comparison
for domain in banking_knowledge airline retail; do
  $OLD -m preflight run --domain "$domain" --seed 0 --out "results/old/$domain.json"
  $NEW -m preflight run --domain "$domain" --seed 0 --out "results/new/$domain.json"
done
$OLD -m preflight compare results/old results/new --out results/report.md  # also writes results/metrics.json
```

`--relations F1,F3` restricts a run to some relations and `--limit N` to the first N tasks, which is enough for a
smoke test. A run is deterministic for a seed: the same seed gives the same variants and the same `records_sha256`,
timings excluded.

The whole experiment is one command:

```sh
/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python experiment.py --out run/metrics.json
```

It runs the suite for all three domains under both environments (subprocesses, `--old-python` and `--new-python`
override the defaults) and writes everything inside the folder that holds the output path: `run/results/old/*.json`,
`run/results/new/*.json`, `run/report.md`, and the metrics at the output path itself. Two runs give identical metrics.
The project itself recomputes nothing: the independent check of amendment 2 works from the per-case files the run
leaves behind.

Tests and the demo run in either environment:

```sh
$OLD -m unittest
$NEW -m unittest
$OLD demo.py --limit 5
```

CI runs `ruff check` and `ruff format --check` on every push; the unit tests need tau2 and run in the environments
above.

## Output

Each result file holds `domain`, `seed`, `tau2_version`, `provenance` (the env's `SOURCE.txt` and the Python version),
the task records, the variant records, a `records_sha256` over the variants and the elapsed `seconds`. `tau2_version`
is the tag named on the first line of `SOURCE.txt` (its second word).

A task record holds `id`, `reward_basis`, its three fingerprints (`task_fingerprint`, `gold_fingerprint`,
`gold_output_fingerprint`), the base run's `base_outcome` and `base_reward`, and a `static` entry with `gold_errors`,
`gold_changes_db`, `no_effect_writes`, the relations skipped as `identical_to_gold`, `p5_not_inert` when the unknown
call was not inert, and `base_error`, `static_error` or `relation_errors` when a replay, a static check or a generator
raised.

A variant record holds `task_id`, `relation`, `detail`, `expected_pass`, `inserted_in_gold`, the exact `calls`, the
validity `checks` that were applied, `live_call_errors` (indexes of calls the live env answered with an error), the
variant and gold `end_state` hashes, its `outcome` (`verdict` or `error`), `reward`, `flagged`, `error`, and, when
flagged, the top-level DB fields that differ from the gold run (`db_diff`).

The comparison writes a Markdown report and a numbers-only `metrics.json` beside it:

- per domain and relation, on each side: attempted, verdicts, errors, flagged variants and flagged tasks;
- the same counts for P1 split into `inserted_in_gold` (`in gold`) and the rest (`not in gold`);
- the tasks whose fixture changed, with which fingerprint differs (`task_fingerprint` or `gold_output_fingerprint`);
- the flags present on the old grader and absent on the new one, by task and relation;
- the H1(a), H1(b) and H2 checks of the pre-registration, each with the numbers behind it;
- the airline/retail control, with its mismatches when it fails;
- F1 counts per side, domain and `gold_changes_db` (`unknown` when a task has no such static entry);
- the relations skipped as identical to gold, counted per domain and relation, and the `p5_not_inert` and
  `no_effect_writes` counts per domain;
- the `records_sha256` of every input file.

`metrics.json` is a flat object of numbers only (0/1 for held, never booleans) with a fixed key set: `h1a_held`,
`h1b_held`, `h2_held`, `control_passed`, `control_mismatches`, `p1_flagged_tasks_old`, `p1_flagged_tasks_new`,
`p1_in_gold_flagged_tasks_new`, `p3_flagged_tasks_old`, `p3_flagged_tasks_new`, `f1_pass_airline_new`,
`f3_pass_rate_new`, `f6_pass_share_new`, `changed_tasks`, `attempted_old`, `attempted_new`, `errors_old`, `errors_new`,
`records_digest_old`, `records_digest_new`. A side's `records_digest` is the integer value of the first 12 hex digits
of the SHA-256 of its files' `records_sha256` values, sorted by domain and joined by newlines, so a repeated run is
compared on every variant case and not only on totals; a task's base run and static checks sit outside that digest and
are checked from the per-case files instead.

How the pieces fit, and what the checks do not cover, is in [docs/DESIGN.md](docs/DESIGN.md).

## About

Epsilab's site is titled "Epsilab | Long-Horizon RL Environments for Agent Training" (<https://epsilab.com/>) and the
team describes itself as "building high-fidelity RL environments for real-world agents."
(<https://epsilab.com/blog/hello-from-the-training-grounds>). The same post lists planned "Deep dives on how we build
environments and verifiers."

This project is an inference from that work rather than a company claim: a grader is the part of an environment that
decides whether a rollout succeeded, so a cheap check on it is worth having before training against it.
