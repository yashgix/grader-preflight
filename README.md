# grader-preflight

An LLM-free preflight for the grader of a stateful tool environment, a small reinforcement-learning experiment on
what a grader flaw does to the policy trained against it, and an offline audit of agent trajectories. All three
target τ²-bench's `EnvironmentEvaluator.calculate_reward`, before the v1.0.1 fixes (`pre-v1.0.1` at `b51a6d6`) and
after them (`v1.0.1` at `fc0055d`).

Phase 1 takes a known-good (gold) run, changes it in ways that should not matter (an extra read, `25` written as
`25.0`, a call to a tool that does not exist) and in ways that should (a dropped write, an empty run, verification
logged after the writes it gates), then checks that the grader's verdict moves the right way. No language model is
called: each variant is built from the gold run, from the environment's own tool metadata and from the live DB state.

Phase 2 trains a small linear softmax policy over banking tool calls with REINFORCE, once per grader, and asks whether
the flaw changes what the policy learns. It is a small, faithful RL experiment, not LLM-agent training: the policy is
open-loop (it sees only the calls it has made and the step number, never a tool output), and every call it makes is a
real τ²-bench tool call, graded by τ²-bench's own, unchanged grader. `python experiment.py --phase 2 --out PATH` runs
the smoke run and `python experiment.py --full --out PATH` the full sweep (amendment 6).

Phase 3 replays agent runs in τ²-bench's own results format (`info`, `tasks`, `simulations`), regrades every run under
both graders and checks it against phase 1's relations: whether each G_old/G_new verdict change falls in a class phase
1 already flags (H5), what the fixed grader still misses (H6), and whether airline's numeric spelling is a grading
inconsistency (H7). **This build never sees real agent data.** Every results file it writes is synthetic, built from
τ²-bench's own gold actions with `as_simulation`, so its correct answers are known. Real published files are analysed
later, outside the build, with the commands under [Real data](#real-data-documented-not-run-here); no third-party
trajectory file is added to this repo.

Hypotheses, thresholds, seeds and the seven dated amendments that fix the protocol are in
[PREREGISTRATION.md](PREREGISTRATION.md), committed before any experiment. How the pieces fit, and what the checks do
not cover, is in [docs/DESIGN.md](docs/DESIGN.md).

## Scope

Every verdict in this project is the DB-state verdict of `EnvironmentEvaluator.calculate_reward`, called unchanged, on
tasks whose `reward_basis` contains `DB`. The COMMUNICATE, NL_ASSERTION and ACTION terms are out of scope, and telecom
is not covered.

Phase 1 covers `banking_knowledge` (built with the `grep_only` retrieval variant, so no API is reached), `airline` and
`retail`. Phase 2 covers `banking_knowledge` only, on the 62 tasks phase 1 found comparable across the two versions;
this build runs the phase 2 smoke on the first five train tasks. Phase 3 covers `banking_knowledge` and `airline`, on
synthetic runs only.

Environments are built only in `preflight/tau2_adapter.py`, the only place τ²-bench's constructors are called. On
v1.0.1 banking logs a read to the hashed agent DB only when the read is on a per-task allowlist that τ²-bench's runner
builds; the adapter derives the same allowlist with τ²-bench's own `_derive_read_log_allowlist` and passes it to the
live environment and to the grader, so banking is graded in the configuration τ²-bench actually runs.

## Running

Everything runs in the two read-only environments; nothing is installed and no API key is present. Every entry point
sets `LITELLM_LOCAL_MODEL_COST_MAP=True` before τ²-bench is imported.

```sh
OLD=/Users/Shared/arcturus-exec/envs/tau2-pre-py312/bin/python
NEW=/Users/Shared/arcturus-exec/envs/tau2-v101-py312/bin/python

# phase 1: one domain on one version ($OLD is the grader before the fixes, $NEW the grader after them)
$OLD -m preflight run --domain airline --seed 0 --out results/old/airline.json

# phase 1: all three domains on both sides, then the comparison
for domain in banking_knowledge airline retail; do
  $OLD -m preflight run --domain $domain --seed 0 --out results/old/$domain.json
  $NEW -m preflight run --domain $domain --seed 0 --out results/new/$domain.json
done
$OLD -m preflight compare results/old results/new --out results/report.md  # also writes results/metrics.json

# phase 1 as one command
$OLD experiment.py --phase 1 --out run/metrics.json

# phase 2: the smoke run
$OLD experiment.py --phase 2 --out results/phase2/metrics.json

# phase 2: the full 10-seed sweep
$OLD experiment.py --full --out results/full/metrics.json

# phase 3 (also the default phase): the synthetic trajectory audit, offline
$OLD experiment.py --out results/phase3/metrics.json

# tests and demos
$OLD -m unittest
$NEW -m unittest
$OLD demo.py --limit 5             # phase 1 relations on five airline tasks
$NEW demo.py --phase 2 --limit 5   # a short training run, printed
$NEW demo.py --phase 3 --limit 5   # five synthetic banking runs, regraded here
```

`--relations F1,F3` restricts a phase 1 run to some relations and `--limit N` to the first N tasks. A phase 1 run is
deterministic for a seed: the same seed gives the same variants and the same `records_sha256`, timings excluded.
`--old-python` and `--new-python` override the two interpreters; `--seeds`, `--episodes` and `--processes` steer the
phase 2 smoke. `--full` refuses `--episodes` and `--seeds`, because the sweep's settings are pre-registered.

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
a task whose gold run raises or scores 0 stays in the results as `gold not accepted`. No variant ever equals the gold
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
| `G_proc` | `G_new` plus an order check (exploratory, full sweep only) | `tau2-v101-py312` |

The smoke runs `G_old` and `G_new` only. `G_proc` accepts a run only if `G_new` does and every `log_verification`
comes before the first write it gates. It is part of the full sweep and never enters this build's smoke report.
One process can load only one τ²-bench version, so each condition runs as its own subprocess: `experiment.py`
dispatches `python -m preflight.phase2 worker` into the right interpreter.

## Phase 2: tasks

`data/banking_pool.json` lists the 62 banking tasks that phase 1 found comparable across the two versions (the fixture
rule of amendment 2) and whose base run scores 1 on both. The split is pre-registered: the pool ids are sorted,
shuffled with `random.Random(20261110)`, and the first 30 are the train set and the next 10 held-out. It is written
once to `split.json` and reused on every later run; no process recomputes the pool from τ²-bench data because one
process can load only one version.

The smoke uses the first five train tasks from that split, seeds 0 to 2, the two conditions above and the random
baseline. Hyperparameters are frozen at the defaults for the smoke (learning rate 0.1, running mean of all past
rewards as the baseline); the pre-registered retail-only tuning belongs to the full sweep (`--full`).

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

- Softmax over the features, REINFORCE with a running-mean baseline, undiscounted: every action of an episode gets
  the episode's return, so no step cost or discount penalizes a lookup. Extra lookups do use up horizon steps, equally
  in every condition.
- The smoke runs 2,000 training episodes per condition and seed, with tasks drawn uniformly from the five smoke train
  tasks. The full sweep uses 3,000 episodes and ten seeds; this build's report is the smoke.
- One grader call per distinct final trajectory, memoized inside the run's process: the key is the task id, the task
  fingerprint and the canonical JSON of the ordered calls, and a miss calls the adapter's `grade`, which replays the
  calls on a fresh τ²-bench environment. A grader exception is an error outcome: reward 0 for learning, counted apart,
  never silently dropped.

## Phase 2: the oracle

True success, our own measure and stated as ours: `G_new` passes, every `log_verification` comes before the first
write it gates, and no altered write was made. It is logged every episode and never used as reward. A `G_old` process
cannot load v1.0.1, so its distinct trajectories are re-graded under `G_new` afterwards, in the `tau2-v101-py312`
interpreter; `G_new` uses its own `G_new` verdict.

For each gold `log_verification`, its gated writes are the gold writes after it. `order_ok` holds when every such gated
write that the trajectory makes comes after that verification; extra reads never count, and calls are matched exactly
(name, arguments and requestor, so a wrapper's inner tool counts).

## Phase 2: smoke report and metrics

The smoke is feasibility evidence only, never hypothesis evidence. `report.md` shows:

- wall time and episodes per second for each run, and grader calls per episode after memoization;
- curves of proxy reward, true success and extra-read rate against episodes, beside the random policy on the same
  tasks and seeds;
- the gate measure: mean true success under `G_new` on episodes 1,701 to 2,000;
- kept apart per condition and for the random policy: the training reward (that condition's own grader), the `G_new`
  verdict, the order check, the no-altered-write check and true success;
- per task, the final policy's probabilities of the gold actions, extra reads and STOP beside the uniform start;
- a scope line saying that extra lookups use horizon steps equally in every condition.

The numbers-only metrics are deterministic and contain exactly:

`episodes`, `grader_calls_old`, `grader_calls_new`, `errors`, `gate_true_success_new`, `random_true_success`,
`true_success_old`, `extra_read_rate_old`, `extra_read_rate_new`, `random_extra_read_rate`,
`distinct_trajectories`.

Timings appear in `report.md` only, never in the metrics. The full sweep (`--full`) writes its own report and metrics
under [preflight/full.py](preflight/full.py) with the amendment 5-6 measures.

## Phase 2: baselines

The smoke runs the random policy over the same actions, tasks and seeds as the learned policies. Gold replay and the
empty run (STOP at once) are part of the full sweep, not of the smoke.

## Phase 2: output

The phase 2 smoke writes everything inside the folder that holds its `--out` path:

- `split.json`: the pre-registered train/held-out split;
- `runs/old_seed{S}.json`, `runs/new_seed{S}.json` and `runs/random_seed{S}.json` for seeds 0 to 2: one run's
  records, i.e. every episode's task, made actions, calls, reward, error flag, extra reads, `G_new` verdict and oracle
  parts;
- `report.md` (tables, curves and timings) and the metrics at the output path.

Phase 1 output is unchanged.

## Phase 3: trajectory audit

`preflight/trajectories.py` reads a τ²-bench `Results` file, replays each run's calls on a live environment in the
current version, compares the recorded tool outputs with the live ones, regrades the recorded messages and computes
phase 1's triggers on the run's own calls. `preflight/trajectory_analysis.py` joins the two versions' records and
writes the H5/H7 documents and H6's blinded packets. `preflight/phase3.py` runs the whole pipeline on synthetic runs.

Per run, a regrade record holds its replay class (`full` when every replayed output matches the recorded one and the
strict regrade works, `strict` when only the recorded-messages regrade works, `lenient` when only
`strict_replay=False` does — v1.0.1 only — and `incompatible` when neither), this version's verdict, the recorded
one, the DB tables where the run's end state differs from the gold run's, the P1 and P3 triggers, whether any call is
not in the gold run, airline's schema floats with the integer/float pair, and banking's late-verification evidence. A
tool call the live environment answers with an error is a fact about the run; a replay or grading exception is
recorded with its type and never stops the file.

H5 (banking) asks whether every G_old/G_new verdict difference is explained by a phase 1 trigger whose G_old DB diff
lies only in that relation's tables; unexplained events are listed one by one with their triggers. H6's 24-run sample
is stratified by rule and blinded: a packet holds the task instructions, the policy the agent saw and the messages,
never a reward, stratum, flag or run id. H7 is descriptive: airline's numeric-representation inconsistencies and
banking's late-verification evidence beside the graders' verdicts.

### Phase 3: the synthetic experiment

`experiment.py --phase 3` (the default phase) builds, in each environment, one synthetic results file per domain from
τ²-bench's own gold actions:

- banking, for every pool task: the gold run, a `trimmed=True` copy of it (tool-message `id` and `error` dropped, as
  τ²-bench's `trim_trajectories.py` does), the first agent-read P1 variant, the P3 variant when there is one, and the
  F6 late-verification variant when there is one;
- airline, for every task: the gold run and a copy whose whole-number integers for parameters τ²-bench's schema
  declares `integer` are floats.

Each environment builds what it grades, so recorded outputs match by construction; the parent process writes the
flags from the old side's records, joins the two sides and writes the metrics. Every file stays inside the folder that
holds `--out`; no network and no LLM is reached.

The metrics are exact and identical across two runs:

`runs_banking`, `runs_airline`, `full_replay_new`, `incompatible_old`, `incompatible_new`, `events`, `explained`,
`recall`, `p1_trigger_runs`, `p3_trigger_runs`, `trimmed_equal_gold`, `schema_float_runs`, `inconsistencies`,
`airline_control_events`, `late_log_runs`, `errors`.

Expected shape, not a threshold: gold runs give no H5 events; P1 variants on tasks phase 1 flagged give explained
events; airline control events are 0; every F6 variant has `late_log_runs` true; a trimmed copy's new-side record
equals its gold copy's apart from `id`, `trial` and the file stem.

## Real data (documented, not run here)

Nothing below is executed by this build, and nothing about a publisher's files is hard-coded: the paths are
arguments. The published results files stay outside this repo, are never redistributed, and only aggregate counts are
reported (amendment 7).

```sh
# in each environment: replay, regrade and trigger every run of every published file
$OLD -m preflight.trajectories regrade PUBLISHED/*.json --out regrade/old
$NEW -m preflight.trajectories regrade PUBLISHED/*.json --out regrade/new

# phase 1's G_old per-case records -> the banking tasks each P relation flagged
$OLD -m preflight.trajectories flags results/old/*.json --out flags.json

# join the two sides: H5, H7, runs.csv and a short report
$OLD -m preflight.trajectories analyze --old regrade/old --new regrade/new --flags flags.json \
  --pool data/banking_pool.json --out analysis

# H6's stratified, blinded 24-run sample
$OLD -m preflight.trajectories sample --old regrade/old --new regrade/new --flags flags.json \
  --pool data/banking_pool.json --files PUBLISHED/*.json --out h6
```

## Reading the results

H2 is one hypothesis: it is held only if all four of its parts hold, so a part can hold (F6, say) while H2 is not
held. The control covers airline and retail only; the changed-fixture tasks are all banking and matter only to H1.
Phase 1 says nothing about RL or held-out tasks. Phase 2 is exploitability within its own action space, with the
oracle shown beside the grader's verdict; it is not a claim about LLM agents. Phase 3's H5 and H7 are measurements on
whatever files they are given, and on this build the files are synthetic, so its numbers are evidence that the
pipeline works, not evidence about τ²-bench's published runs. H3, H4 and the seed-level bootstrap intervals belong
to the full sweep, not to the smoke.

## Known limits

- Only the DB-state term decides a verdict; a task whose reward also depends on communication or assertions is
  covered only for its state term.
- Phase 1: tasks whose base run does not score 1 get no variants, and a relation that cannot be built on a task yields
  no variant, so counts differ from task to task. F1 cannot separate an empty run from a gold run that does not change
  the DB, so its counts are split by `gold_changes_db`.
- Phase 2: the policy is open-loop and gets `is_gold` / `is_gold_next`; each distinct trajectory is graded once and
  memoized inside the run, never saved; the oracle is ours; and a `G_old` run's true success needs the v1.0.1
  interpreter afterwards. The extra-read rate counts READ actions whose inner tool name is not in the gold run, which
  measures whether a flaw discourages harmless lookups, not whether identity was checked. The smoke is not the full
  sweep: full-sweep baselines, held-out evaluation and H3/H4 are not computed here.
- Phase 3: this build has no real trajectory file, so every phase 3 number comes from synthetic runs built from gold
actions and says nothing about published agents. `access_before_facts` is `None` on synthetic runs, because they
  carry no user message with text. A run's end state is compared as two DB hashes and a list of top-level table names,
  not values inside them. H6 has no labelers in this build; the packets and the key are produced, the labels are not.
- `records_sha256` and the phase 1 side digests cover variant records only, so per-case repeatability is checked from
  the per-case files as well.

## About

Why this project: a grader is the part of an environment that decides whether a rollout succeeded, so a cheap check
on it is worth having before training against it, a grader flaw is worth measuring in what a policy actually learns,
and a grader flaw is worth auditing in runs that are already published.
