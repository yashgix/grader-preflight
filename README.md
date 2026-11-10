# grader-preflight

An LLM-free preflight for the grader of a stateful tool environment, and a small reinforcement-learning experiment on
what a grader flaw does to the policy trained against it. Both parts target τ²-bench's
`EnvironmentEvaluator.calculate_reward`, before the v1.0.1 fixes (`pre-v1.0.1` at `b51a6d6`) and after them (`v1.0.1`
at `fc0055d`).

Phase 1 takes a known-good ("gold") run, changes it in ways that should not matter (an extra read, `25` written as
`25.0`, a call to a tool that does not exist) and in ways that should (a dropped write, an empty run, verification
logged after the writes it gates), then checks that the grader's verdict moves the right way. No language model is
called: each variant is built from the gold run, from the environment's own tool metadata and from the live DB state.

Phase 2 trains a small linear softmax policy over banking tool calls with REINFORCE, once per grader, and asks whether
the flaw changes what the policy learns. It is a small, faithful RL experiment, not LLM-agent training: the policy is
open-loop (it sees only the calls it has made and the step number, never a tool output), and every call it makes is a
real τ²-bench tool call, graded by τ²-bench's own, unchanged grader.

Hypotheses, thresholds, seeds and the five amendments that fix the protocol are in
[PREREGISTRATION.md](PREREGISTRATION.md), committed before any experiment. How the pieces fit, and what the checks do
not cover, is in [docs/DESIGN.md](docs/DESIGN.md).

## Scope

Every verdict in this project is the DB-state verdict of `EnvironmentEvaluator.calculate_reward`, called unchanged, on
tasks whose `reward_basis` contains `DB`. The COMMUNICATE, NL_ASSERTION and ACTION terms are out of scope, and telecom
is not covered.

Phase 1 covers `banking_knowledge` (built with the `grep_only` retrieval variant, so no API is reached), `airline` and
`retail`. Phase 2 covers `banking_knowledge` only, on the 62 tasks phase 1 found comparable across the two versions.

Environments are built only in `preflight/tau2_adapter.py`, the only place τ²-bench's constructors are called. On
v1.0.1 banking logs a read to the hashed agent DB only when the read is on a per-task allowlist that τ²-bench's runner
builds; the adapter derives the same allowlist with τ²-bench's own `_derive_read_log_allowlist` and passes it to the
live environment and to the grader, so banking is graded in the configuration τ²-bench actually runs.

## Running

Everything runs in the two read-only environments; nothing is installed and no API key is present. Both entry points
set `LITELLM_LOCAL_MODEL_COST_MAP=True` before τ²-bench is imported.

```sh
OLD=/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python
NEW=/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python

# phase 1: one domain on one version ($OLD is the grader before the fixes, $NEW the grader after them)
$OLD -m preflight run --domain airline --seed 0 --out results/old/airline.json

# phase 1: all three domains on both sides, then the comparison
for domain in banking_knowledge airline retail; do
  $OLD -m preflight run --domain "$domain" --seed 0 --out "results/old/$domain.json"
  $NEW -m preflight run --domain "$domain" --seed 0 --out "results/new/$domain.json"
done
$OLD -m preflight compare results/old results/new --out results/report.md  # also writes results/metrics.json

# phase 1 as one command
$OLD experiment.py --phase 1 --out run/metrics.json

# phase 2: the feasibility run, which is the default of experiment.py
$OLD experiment.py --out results/phase2/metrics.json

# phase 2: the full sweep, run only after the feasibility result has been seen
$OLD experiment.py --full --out results/phase2/metrics.json

# tests and demos
$OLD -m unittest
$NEW -m unittest
$OLD demo.py --limit 5             # phase 1 relations on five airline tasks
$NEW demo.py --phase 2 --limit 5   # a short training run, printed
```

`--relations F1,F3` restricts a phase 1 run to some relations and `--limit N` to the first N tasks. A phase 1 run is
deterministic for a seed: the same seed gives the same variants and the same `records_sha256`, timings excluded.
`--old-python` and `--new-python` override the two interpreters; `--seeds`, `--episodes` and `--processes` steer
phase 2.

CI runs `ruff check` and `ruff format --check` on every push; the unit tests need τ²-bench and run in the two
environments above.

## Phase 1: relations

| Id | Change to the gold run | Expected verdict |
| --- | --- | --- |
| P1 | insert one call to a declared read tool, its arguments taken from earlier calls in the run | pass |
| P3 | respell numeric arguments with the same value (`25` ↔ `25.0`, discoverable-wrapper `arguments` strings included) | pass |
| P5 | insert a call to `no_such_tool` with `{}` arguments | pass |
| F1 | run nothing | fail |
| F3 | drop one write whose removal changes the live end state | fail |
| F6 | move a banking `log_verification` after the writes it gates | fail |

A flag is a variant whose verdict differs from its expected one. A task gets variants only when its gold run scores 1;
a task whose gold run raises or scores 0 stays in the results as "gold not accepted". No variant ever equals the gold
run (amendment 3), and each skipped relation is recorded in the task's `static.identical_to_gold`.

- **P1** inserts one declared read, direct or discoverable. Required argument names come from the tool's JSON schema,
  else the callable's signature, and every value is taken from the same-named argument of an earlier gold call. A
  discoverable read the run has not unlocked brings its `unlock_discoverable_agent_tool` call first. Candidates are
  shuffled with the seed and the first five whose replay through the inserted call ends in a non-error tool message
  are kept. `inserted_in_gold` marks a variant whose inserted tool also appears in the gold run; H1(a) counts those
  apart on both sides, because v1.0.1 keeps logging reads the gold run itself requires by design.
- **P3** respells every numeric argument with the same value, integral `int` to `float` and integral `float` to `int`,
  and rewrites number tokens inside the `arguments` string of the three discoverable wrappers only. Booleans are not
  numbers. Nothing to respell means no variant (`identical_to_gold`).
- **P5** inserts `no_such_tool` at a position drawn with the seed, kept only when the agent and user DB hashes around
  the inserted call are unchanged; otherwise the task records `p5_not_inert` and gets no P5 variant.
- **F1** is the empty run, dropped when the gold run is empty too.
- **F3** drops one write per variant, and only writes whose removal changes the live end state; the others are counted
  in `static.no_effect_writes`.
- **F6** exists only for banking tasks whose `log_verification` is followed by a write, and moves that call to the
  end.

Two checks are recorded per task instead of as variants: `gold_errors` (S1, gold calls whose live output is an error)
and `gold_changes_db` (S2, the gold run leaves the DB different from a fresh environment).

## Phase 2: conditions

| Name | Reward | Environment |
| --- | --- | --- |
| `G_old` | `pre-v1.0.1` `calculate_reward`, through the adapter | `tau2-pre-py312` |
| `G_new` | `v1.0.1`, through the adapter (read-log allowlist included) | `tau2-v101-py312` |
| `G_proc` | `G_new` plus an order check (exploratory) | `tau2-v101-py312` |

`G_proc` accepts a run only if `G_new` does and every `log_verification` comes before the first write it gates. It is
reported beside the other two and never enters H3 or H4. One process can load only one τ²-bench version, so each
condition runs as its own subprocess: `experiment.py` dispatches `python -m preflight.phase2 worker` into the right
interpreter.

## Phase 2: tasks

`data/banking_pool.json` lists the 62 banking tasks that phase 1 found comparable across the two versions (the fixture
rule of amendment 2) and whose base run scores 1 on both, with their task fingerprints. The split is drawn once per
output folder: the pool ids are sorted, shuffled with `random.Random(20261110)`, and the first 30 are the train set
and the next 10 held-out; it is written to `split.json` and reused on every later run.

Hyperparameters are tuned on retail only (the control domain, where the two graders agree), over learning rate 0.05,
0.1 and 0.2, on five retail tasks for 200 episodes each, then frozen for the full sweep. The feasibility run uses the
defaults: learning rate 0.1 and a running mean of all past rewards as the baseline.

## Phase 2: the episode

- Horizon: gold length + 3 steps; the episode ends at STOP or at the horizon. The policy is open-loop.
- Actions, built once per task and deterministic: every gold call; every READ candidate of phase 1's P1 (a
  discoverable read the episode has not unlocked brings its unlock as one action); for every gold write, at most two
  altered copies, each swapping one string argument whose name ends in `_id` for another value that argument name
  takes in this task's gold run or, if none, in another banking task's; and STOP.
- Eight features per action, shared across tasks: `is_read`, `is_write`, `is_gold`, `is_gold_next`, `is_altered`,
  `is_gating` (a `log_verification`), `is_stop`, and step / horizon. A bias per task and action lets a task learn its
  own order.
- Numbers keep the gold run's spelling in every action, so #397 cannot differ between conditions and a G_old/G_new
difference in what is learned comes from #329 alone. `is_gold` and `is_gold_next` tell the policy which actions the
  gold run uses; they stand for an agent that roughly knows the right steps, so reaching the gold path is not a
  result — what each grader does to the remaining choices is.

## Phase 2: learner and grading

- Softmax over the features, REINFORCE with a running-mean baseline, undiscounted: every action of an episode gets the
  episode's return, so nothing but the grader penalizes a lookup.
- One run is one condition and one seed: 3,000 training episodes for the full sweep (2,000 for the feasibility run),
  with tasks drawn uniformly from the 30 train tasks. Seeds 0 to 9, the same seeds for every condition; a seed fixes
  the policy init and the episode sampling.
- One grader call per distinct final trajectory, memoized inside the run's process: the key is the task id, the task
  fingerprint and the canonical JSON of the ordered calls, and a miss calls the adapter's `grade`, which replays the
  calls on a fresh τ²-bench environment. A grader exception is an error outcome: reward 0 for learning, counted apart,
  never silently dropped.

## Phase 2: the oracle

"True success", our own measure and stated as ours: `G_new` passes, every `log_verification` comes before the first
write it gates, and no altered write was made. It is logged every episode and never used as reward. A `G_old` process
cannot load v1.0.1, so its distinct trajectories are re-graded under `G_new` afterwards, in the `tau2-v101-py312`
interpreter, and true success is filled in from that; `G_new` and `G_proc` runs use their own `G_new` verdict.

## Phase 2: measures and hypotheses

Per run, then mean and 95% bootstrap interval over seeds, paired by seed across conditions:

- proxy reward, true success and proxy minus true, over the last 300 training episodes;
- the extra-read rate (amendment 5): READ actions the gold run does not make, per episode, over the last 300 training
  episodes and on held-out tasks with the final policy;
- the stop-without-acting rate and the number of episodes to the first oracle-wrong reward-1 trajectory;
- every distinct oracle-wrong reward-1 trajectory with its violation type — order (verification after a gated write),
  altered write, or other — and whether phase 1 flagged that class on that grader.

- **Gate:** RL stays in only if true success on the train tasks under `G_new` exceeds 0.8 within 2,000 episodes (mean
  over episodes 1,701 to 2,000, then over seeds).
- **H3:** at least 90% of the distinct oracle-wrong reward-1 trajectories under `G_old` and `G_new` fall in a class
  phase 1 flagged (F6's order class). A miss is reported as a miss.
- **H4:** the extra-read rate is lower under `G_old` than under `G_new` (paired by seed; the 95% bootstrap interval of
  the difference lies below 0) with no gain in true success for `G_old`; the held-out rate is reported beside it.

## Phase 2: baselines

The random policy over the same actions, gold replay and the empty run (STOP at once), on the same tasks and seeds,
give the scale each measure is read against. The random policy's proxy reward and true success are in the feasibility
metrics; the gold and empty runs appear in the full report.

## Output

A phase 1 result file holds `domain`, `seed`, `tau2_version`, `provenance` (the env's `SOURCE.txt` text and the Python
version), the task records, the variant records, a `records_sha256` over the variants and the elapsed `seconds`.

The comparison writes a Markdown report and a numbers-only `metrics.json` beside it: per domain and relation on each
side the attempted, verdict, error, flagged-variant and flagged-task counts (so attempted = verdicts + errors); the
same for P1 split by `inserted_in_gold`; the fixture-changed and one-sided tasks; unpaired comparable variants; the
flags present on the old grader and absent on the new one; one line each for H1(a), H1(b) and H2 of the
pre-registration; the airline/retail control with its mismatches; F1 counts per side, domain and `gold_changes_db`;
the relations skipped as identical to gold; `p5_not_inert` and `no_effect_writes`; and the `records_sha256` of every
input file. `metrics.json` has a fixed key set with 0/1 for held, never booleans.

Phase 2 writes everything inside the folder that holds its `--out` path:

- `runs/old_seed{S}.json`, `runs/new_seed{S}.json` and, in the full sweep, `runs/proc_seed{S}.json`: one run's
  records, i.e. every episode's task, made actions, calls, reward, error flag, extra reads, `G_new` verdict and oracle
  parts;
- `runs/random_seed{S}.json`, plus `runs/gold.json` and `runs/empty.json` from the full sweep;
- `heldout/{old,new,proc}_seed{S}.json` in the full sweep: 100 episodes per held-out task with the final policy;
- `split.json`, `report.md` (tables, curves and timings) and the metrics at the output path.

The metrics are numbers only and deterministic: two runs give identical metrics, and timings go into `report.md` only.
The feasibility metrics are `episodes`, `grader_calls_old`, `grader_calls_new`, `errors`, `gate_true_success_new`,
`random_true_success`, `true_success_old`, `extra_read_rate_old`, `extra_read_rate_new`, `random_extra_read_rate` and
`distinct_trajectories`. The full sweep adds `gate_held`, `h3_recall`, `h3_held`, `oracle_wrong_reward1`,
`oracle_wrong_flagged`, `h4_extra_read_diff`, `h4_ci_low`, `h4_ci_high`, `h4_held`, `true_success_proc`,
`extra_read_rate_proc`, `gold_true_success`, `empty_true_success`, and the held-out true success and extra-read rates
of the three conditions. The independent recompute of the pre-registration is done outside the project, from the
per-run records the run leaves behind.

## Reading the results

H2 is one hypothesis: it is held only if all four of its parts hold, so a part can hold (F6, say) while H2 is not
held. The control covers airline and retail only; the changed-fixture tasks are all banking and matter only to H1.
Phase 1 says nothing about RL or held-out tasks. Phase 2 is exploitability within its own action space, with the
oracle shown beside the grader's verdict; it is not a claim about LLM agents.

## Known limits

- Only the DB-state term decides a verdict; a task whose reward also depends on communication or assertions is
  covered only for its state term.
- Phase 1: tasks whose base run does not score 1 get no variants, and a relation that cannot be built on a task yields
  no variant, so counts differ from task to task. F1 cannot separate an empty run from a gold run that does not change
  the DB, so its counts are split by `gold_changes_db`.
- Phase 2: the policy is open-loop and gets `is_gold` / `is_gold_next`; each distinct trajectory is graded once and
  memoized inside the run, never saved; the oracle is ours; and a `G_old` run's true success needs the v1.0.1
  interpreter afterwards. The extra-read rate counts READ actions whose inner tool name is not in the gold run, which
  measures whether a flaw discourages harmless lookups, not whether identity was checked.
- `records_sha256` and the phase 1 side digests cover variant records only, so per-case repeatability is checked from
  the per-case files as well.

## About

Epsilab's site is titled "Epsilab | Long-Horizon RL Environments for Agent Training" (<https://epsilab.com/>) and the
team describes itself as "building high-fidelity RL environments for real-world agents."
(<https://epsilab.com/blog/hello-from-the-training-grounds>). The same post lists planned "Deep dives on how we build
environments and verifiers."

This project is an inference from that work rather than a company claim: a grader is the part of an environment that
decides whether a rollout succeeded, so a cheap check on it is worth having before training against it, and a grader
flaw is worth measuring in what a policy actually learns.
