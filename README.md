# grader-preflight

An LLM-free preflight for graders of stateful tool environments. It takes a known-good ("gold") run, changes it in
ways that should not matter (an extra read, `25` written as `25.0`, a call to a tool that does not exist) and in ways
that should (a dropped write, an empty run, verification logged after the writes it gates), then checks that the
grader's verdict moves the right way. No language model is called: each variant is built from the gold run, from the
environment's own tool metadata and from the live DB state.

It targets τ²-bench's grader before and after the v1.0.1 fixes (`pre-v1.0.1` at `b51a6d6`, `v1.0.1` at
`fc0055d`). The hypotheses, thresholds, seeds and the amendments that fix the protocol are in
[PREREGISTRATION.md](PREREGISTRATION.md), committed before any experiment. How the pieces fit, and what the checks do
not cover, is in [docs/DESIGN.md](docs/DESIGN.md). Results are the report and `metrics.json` that a comparison
writes (see [Output](#output)).

## Scope

Only the environment-state terms of the reward are exercised: every verdict quoted here is the DB-state verdict of
`EnvironmentEvaluator.calculate_reward`, called unchanged, on tasks whose `reward_basis` contains `DB`, in
`banking_knowledge` (the `grep_only` retrieval variant, so it runs offline), `airline` and `retail`. The COMMUNICATE,
NL_ASSERTION and ACTION terms are out of scope, and telecom is not part of phase 1.

Every environment is built through `preflight/tau2_adapter.py`, the only place tau2's constructors are called. On
v1.0.1 banking logs a read to the hashed agent DB only when the read is on a per-task allowlist that τ²-bench's runner
builds; the adapter derives the same allowlist with tau2's own `_derive_read_log_allowlist` and passes it to the live
environment and to the grader, so banking is graded in the configuration τ²-bench actually runs.

## Relations

| Id | Change to the gold run | Expected verdict |
| --- | --- | --- |
| P1 | insert one call to a declared read tool, its arguments taken from earlier calls in the run | pass |
| P3 | respell numeric arguments with the same value (`25` ↔ `25.0`, discoverable-wrapper `arguments` strings included) | pass |
| P5 | insert a call to `no_such_tool` with `{}` arguments | pass |
| F1 | run nothing | fail |
| F3 | drop one write whose removal changes the live end state | fail |
| F6 | move a banking `log_verification` after the writes it gates | fail |

A flag is a variant whose verdict differs from the one above. A task gets variants only when its gold run scores 1;
a task whose gold run raises or scores 0 stays in the results as "gold not accepted".

- **P1** inserts one declared read. Candidates are the assistant toolkit's direct read tools and its discoverable read
  tools, the kind read from the toolkit's `tool_type(name)`. Required argument names come from the tool's JSON schema
  (`openai_schema` preferred); a discoverable read is a plain bound method with no schema, so its signature settles
  which arguments are required (no default). Each required value is taken from the same-named argument of an earlier
  call, inside wrapper `arguments` strings too, and the read goes right after the last call that supplied one. A
  discoverable read is called through `call_discoverable_agent_tool` with a JSON `arguments` string; when the run has
  not unlocked that tool earlier, its `unlock_discoverable_agent_tool` call goes right before the read, so a variant is
  the gold run plus the read, or plus the unlock and the read. A discoverable read counts whether the gold run uses it
  or not. Candidates are shuffled with the seed, with no priority by kind or route, and the first five whose replayed
  prefix through the inserted call ends in a tool message that is not an error are kept. `inserted_in_gold` marks a
  variant whose inserted tool also appears in the gold run; H1(a) counts those apart on both sides, because v1.0.1
  keeps logging reads the gold run itself requires by design.
- **P3** respells every numeric argument with the same value, integral `int` to `float` and integral `float` to `int`,
  so the parsed value is unchanged and the spelling differs. Booleans are not numbers. Number tokens inside a JSON
  `arguments` string are rewritten only for the discoverable wrappers (`call_discoverable_agent_tool`,
  `call_discoverable_user_tool`, `give_discoverable_user_tool`); a string argument named `arguments` on any other tool
  is left as it is. Nothing to respell means no variant (`identical_to_gold`).
- **P5** inserts `no_such_tool` with `{}` arguments, as the assistant, at a position drawn with the seed. The variant
  is kept only when the agent and user DB hashes right after the inserted call equal those right before it; otherwise
  the task records `p5_not_inert` and gets no P5 variant. If that validation replay raises, the exception is recorded
  in `relation_errors` for the relation and no variant is built: a check that did not complete is not a check that
  passed.
- **F1** is the empty run. It is dropped when the gold run is empty too, since then it is the gold run itself.
- **F3** drops one write per variant, and only writes whose removal changes the live end state; the others are counted
  in `static.no_effect_writes`.
- **F6** exists only for banking tasks whose `log_verification` is followed by a write, and moves that call to the
  end.

No variant ever equals the gold run (amendment 3); each skipped relation is recorded in the task's
`static.identical_to_gold`.

Two checks are recorded per task instead of as variants: `gold_errors` (S1: gold calls whose live output is an error,
including wrapper outputs starting with `Error:`) and `gold_changes_db` (S2: the gold run leaves the DB different
from a fresh environment).

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
The repo is not written to while it runs, and `experiment.py` recomputes nothing: the independent check of amendment 2
works from the per-case files the run leaves behind.

Tests and the demo run in either environment:

```sh
$OLD -m unittest
$NEW -m unittest
$OLD demo.py --limit 5
```

CI runs `ruff check` and `ruff format --check` on every push; the unit tests need tau2 and run in the environments
above.

## Output

Each result file holds `domain`, `seed`, `tau2_version`, `provenance` (the env's `SOURCE.txt` text and the Python
version), the task records, the variant records, a `records_sha256` over the variants and the elapsed `seconds`.
`tau2_version` is the tag named on the first line of `SOURCE.txt` (its second word).

A task record holds `id`, `reward_basis`, its three fingerprints (`task_fingerprint`, `gold_fingerprint`,
`gold_output_fingerprint`), the base run's `base_outcome` and `base_reward`, and a `static` entry with `gold_errors`,
`gold_changes_db`, `no_effect_writes`, the relations skipped as `identical_to_gold`, `p5_not_inert` when the unknown
call was not inert, and `base_error`, `static_error` or `relation_errors` when a replay, a static check or a generator
raised. A task fingerprint is the SHA-256 of the whole task's canonical JSON; a gold output fingerprint is the
SHA-256 of the gold run's tool outputs, each parsed as JSON when it parses, with every number read as a float, so `25`
and `25.0` match.

A variant record holds `task_id`, `relation`, `detail`, `expected_pass`, `inserted_in_gold`, the exact `calls`, the
validity `checks` that were applied, `live_call_errors` (indexes of calls the live env answered with an error), its
`output_fingerprint` (its live tool outputs, fingerprinted exactly like the gold output, null when the run raised),
the variant and gold `end_state` hashes, its `outcome` (`verdict` or `error`), `reward`, `flagged`, `error`, and, when
flagged, the top-level DB fields that differ from the gold run (`db_diff`). An exception while replaying or grading
is an `outcome` of `error` with `reward` and `flagged` null; it is never counted as a reward or a flag, so every count
reports attempted = verdicts + errors.

The comparison pairs cases before it counts anything: an OLD and a NEW variant are a pair when domain, `task_id`,
`relation`, `detail` and the canonical JSON of `calls` are all equal, and only pairs on comparable tasks (present on
both sides with identical fixtures) count toward the hypotheses. It writes a Markdown report and a numbers-only
`metrics.json` beside it:

- per domain and relation, on each side: attempted, verdicts, errors, flagged variants and flagged tasks;
- the same counts for P1 split into `inserted_in_gold` (`in gold`) and the rest (`not in gold`);
- the tasks whose fixture changed, with which fingerprint differs (`task_fingerprint` or `gold_output_fingerprint`),
  and tasks present on one side only;
- comparable variants with no counterpart ("unpaired"), listed one by one;
- the flags present on the old grader and absent on the new one, one line per pair (a pair flagged on OLD whose NEW
  counterpart is a verdict and not flagged), giving domain, task, relation and `detail`;
- one line each for H1(a), H1(b) and H2 of the pre-registration, with the numbers behind them:
  H1(a) holds when a comparable banking P1 pair whose inserted tool is not in the gold run is flagged on the old
  grader and no such pair is flagged on the new one, the new grader produced at least one verdict in that group, and
  no unpaired comparable variant belongs to it; H1(b) is the same for P3; H2 holds when F3 passes in at most 1% of
  its verdicts, at least 20 airline tasks pass an empty run on the new grader, F6 passes on at least half of the
  banking tasks with an F6 variant, and at least one P-relation variant is flagged. A part with no verdict behind it
  is not held.
- the airline/retail control, with its mismatches when it fails: every task on both sides with an identical fixture,
  every variant paired, and each pair with the same outcome and reward;
- F1 counts per side, domain and `gold_changes_db` (`unknown` when a task has no such static entry);
- the relations skipped as identical to gold, counted per domain and relation, and the `p5_not_inert` and
  `no_effect_writes` counts per domain;
- the `records_sha256` of every input file.

`metrics.json` is a flat object of numbers only (0/1 for held, never booleans) with a fixed key set: `h1a_held`,
`h1b_held`, `h2_held`, `control_passed`, `control_mismatches`, `p1_flagged_tasks_old`, `p1_flagged_tasks_new`,
`p1_in_gold_flagged_tasks_new`, `p3_flagged_tasks_old`, `p3_flagged_tasks_new`, `f1_pass_airline_new`,
`f3_pass_rate_new`, `f6_pass_share_new`, `changed_tasks`, `attempted_old`, `attempted_new`, `errors_old`, `errors_new`,
`unpaired_comparable`, `records_digest_old`, `records_digest_new`. A side's `records_digest` is the integer value of
the first 12 hex digits of the SHA-256 of its files' `records_sha256` values, sorted by domain and joined by
newlines, so a repeated run is compared on every variant case and not only on totals.

## Reading the results

H2 is one hypothesis: it is held only if all four of its parts hold, so a part can hold (F6, say) while H2 is not
held. The control covers airline and retail only; the changed-fixture tasks are all banking and matter only to H1. A
phase 1 run says nothing about RL or held-out tasks, which are phase 2 of the pre-registration.

## Known limits

- Only the DB-state term decides a verdict; a task whose reward also depends on communication or assertions is
  covered only for its state term.
- Tasks whose base run does not score 1 get no variants and stay in the results as "gold not accepted".
- A relation that cannot be built on a task yields no variant: no write, no gated verification, no read whose
  required arguments appear in the run, an empty gold run. Counts therefore differ from task to task.
- F1 cannot separate an empty run from a gold run that does not change the DB, so its counts are reported split by
  `gold_changes_db`; empty-gold tasks get no F1 at all.
- `end_state` is a pair of DB hashes; `db_diff` names top-level fields that differ, not the values inside them.
- `records_sha256` and the side digests cover the variant records only: a repeated run that differs in a task's base
  outcome, its gold output fingerprint or its static checks keeps the same digest, so per-case repeatability is
  checked from the per-case files as well.
- Tool outputs are read as every second message from the end of a trajectory, which is how the adapter appends them.
- The airline and retail control assumes those domains grade identically in both versions; a mismatch is treated as a
  fault in this harness, not as a finding.
- `experiment.py` passes on whatever the subprocesses wrote; it does not recompute the metrics.

## About

Epsilab's site is titled "Epsilab | Long-Horizon RL Environments for Agent Training" (<https://epsilab.com/>) and the
team describes itself as "building high-fidelity RL environments for real-world agents."
(<https://epsilab.com/blog/hello-from-the-training-grounds>). The same post lists planned "Deep dives on how we build
environments and verifiers."

This project is an inference from that work rather than a company claim: a grader is the part of an environment that
decides whether a rollout succeeded, so a cheap check on it is worth having before training against it.
