# Design

## Scope

Every verdict in this project is the DB-state verdict of `EnvironmentEvaluator.calculate_reward`, called unchanged with
the environment arguments τ²-bench's runner uses for the task. Tasks are those with `DB` in `reward_basis`. The
COMMUNICATE, NL_ASSERTION and ACTION terms are never the subject of a variant. Phase 1 covers `banking_knowledge`
(built with the `grep_only` retrieval variant, so no API is reached), `airline` and `retail`. Phase 2 covers banking
only, over the tasks phase 1 found comparable across the two tags. Phase 3 covers banking and airline: its tests and
synthetic experiment use trajectory files built from τ²-bench's own gold actions, and the published leaderboard runs go
through the same `regrade`, `analyze` and `sample` commands (their aggregate results are in `results/phase3.json`).

Environments are built only in `preflight/tau2_adapter.py`, never through τ²-bench's constructors directly. On v1.0.1
banking logs a read to the hashed agent DB only when the read is on a per-task allowlist built by τ²-bench's runner.
The adapter derives it with τ²-bench's own `_derive_read_log_allowlist` and passes it to the live environment and to
the grader.

## Module map

| Module | Job |
| --- | --- |
| `preflight/tau2_adapter.py` | fixed: `Call`, `tasks`, `gold_calls`, `env_kwargs`, `fresh_env`, `run`, `grade`. Replays calls on a live env and scores the trajectory |
| `preflight/relations.py` | `tool_kind`, `Variant`, the six generators in `RELATIONS`, `build_variants`, `static_checks` |
| `preflight/suite.py` | `run_suite`: base runs, variants, per-case records, fingerprints, provenance, `records_sha256` |
| `preflight/compare.py` | `compare`: case-by-case pairing, report and metrics from two result folders, standard library only |
| `preflight/mdp.py` | `Action`, `task_actions`, `horizon`, `allowed`, `features`, `trajectory`, `order_ok`, `oracle` |
| `preflight/rl.py` | `GradeCache`, `Policy`, `train`, `split`, plus `N_FEATURES` and `DEFAULT_LR` |
| `preflight/phase2.py` | `run_smoke`, `pool_ids`, `ensure_split`, and the worker CLI (`train`, `random`, `oracle`, `eval`) |
| `preflight/full.py` | the phase 2 full sweep: retail tuning, ten seeds, counterfactuals, H3/H4 with bootstrap intervals |
| `preflight/report.py` | `smoke_metrics`, `per_task_probabilities`, `write_report`, `write_metrics` |
| `preflight/trajectories.py` | `as_simulation`, `load`, `restore_ids`, `calls`, `task_matches`, `replay`, `diff_tables`, `p1_trigger`, `p3_trigger`, `schema_floats`, `as_integers`, `numeric_pair`, `late_verification`, `regrade`, `flags` |
| `preflight/trajectory_analysis.py` | `analyze` and `sample`: join two regrade folders, H5 and H7, `runs.csv`, `report.md`, H6's blinded packets |
| `preflight/phase3.py` | `build_synthetic`, `worker`, `run_phase3`. The synthetic trajectory audit and its metrics |
| `preflight/envs.py` | the two pinned interpreters: `TAU2_OLD_PYTHON` / `TAU2_NEW_PYTHON`, else the venvs `scripts/setup-tau2.sh` builds |
| `preflight/__main__.py` | CLI for the phase 1 `run` and `compare` commands |
| `experiment.py` | one command per phase: the phase 1 suite under both environments, the phase 2 smoke or full sweep, and the phase 3 audit (the default) |
| `demo.py` | a few tasks through the phase 1 relations, a short phase 2 training run, or a few synthetic phase 3 runs, printed |
| `examples/report_blind_spot.py` | the minimal reproduction of the discrepancy-report blind spot, from task 096's gold run |
| `scripts/setup-tau2.sh` | builds both pinned τ²-bench environments from τ²-bench's lockfile plus `env/requirements-extra.txt` |
| `scripts/collect_results.py` | aggregates the three phases' outputs into `results/*.json` (counts only, no trajectory contents) |
| `scripts/build_site.py` | fills `docs/template.html` from `results/*.json` into `docs/index.html`. `tests/test_site.py` keeps them in step |

## Phase 1

### Base run

Each task's gold run is replayed once on a fresh environment and graded. Only a task whose base verdict is 1 gets
variants. A base run that raises is recorded as `base_outcome: "error"` with `static.base_error`, keeps the record's
shape, and gets no variants.

### Resolving a call to a tool

`tool_kind(env, call)` answers what a call reaches, not what it is named. `call_discoverable_agent_tool` reaches the
tool named by `agent_tool_name` in `env.tools`, `call_discoverable_user_tool` the tool named by
`discoverable_tool_name` in `env.user_tools`. Any other name is looked up in `env.tools` for an assistant call and in
`env.user_tools` for a user call. Existence is settled by `has_tool(name)` when the toolkit has one, and the declared
kind by the toolkit's `tool_type(name)` when that is callable, else by the `Tool` object's own `tool_type`. The kind is
read as a lowercase string (an enum's `value` when it is one). Anything outside `read`, `write`, `generic` and `think`
counts as `generic`, and a name in neither toolkit is `unknown`.

A live tool message counts as an error when it has `error=True` or when its content is a string starting with
`Error:` — the second form is how the discoverable wrappers report a failure.

### Relations and their validity filters

The generators in `preflight/relations.py` build candidates. `build_variants` is the only entry point the suite uses,
and it drops any variant whose calls equal the gold calls (amendment 3), reporting each skip.

**P1, an extra read.** `_p1_plans` collects the assistant toolkit's direct reads plus its discoverable reads, skipping
the wrapper and unlock names, keeping only tools whose declared `tool_type(name)` says `read`. Required argument names
come from the tool's JSON schema, preferring `openai_schema["function"]["parameters"]`. A bare parameter mapping is
not a schema, and the callable's signature then says which arguments are required. Each required name must be
supplied by an earlier gold call, wrapper `arguments` strings included, and the read goes right after the last call
that supplied one. A discoverable read is called through `call_discoverable_agent_tool` with a JSON `arguments`
string. When the run has not unlocked it before, `unlock_discoverable_agent_tool` goes right before the read. The
variant is kept only if replaying the prefix through the inserted read ends in a tool message that is not an error.
Candidates are sorted by route, name and position and then shuffled with `rng`, with no priority by kind or route, so
the draw is reproducible and a read that fails its replay never takes one of the five slots. `inserted_in_gold` is
true when the inserted (inner) tool also appears in the gold run. H1(a) counts those variants apart on both sides.

**P3, respelled numbers.** Every numeric argument is respelled with the same value, integral `int` to `float` and
integral `float` to `int`. The JSON `arguments` string of a discoverable wrapper is rewritten token by token over its
number literals. Only the three wrappers (`call_discoverable_agent_tool`, `call_discoverable_user_tool`,
`give_discoverable_user_tool`) get that treatment. A string argument named `arguments` on any other tool is left
alone, and booleans are not numbers. Nothing to respell means no variant, recorded as `identical_to_gold`.

**P5, an unknown tool.** The position is drawn with `rng` from `0..len(gold)`. The variant is kept only when the agent
and user DB hash pair right after the inserted call equals the pair right before it. Otherwise the task's `static`
carries `p5_not_inert`. If that validation replay raises, the exception travels out of the generator and the suite
records it in `static.relation_errors` under `P5`, with no variant built: a check that did not complete is never
reported as passed.

**F1, an empty run.** Dropped when the gold run is also empty, since then it is the gold run itself.

**F3, a dropped write.** Only calls of kind `write`. The gold end state is replayed once. Each write is dropped in turn
and the variant kept only when the resulting agent and user hash pair differs from the gold run's. A write whose
removal changes nothing is counted in `static_checks` as `no_effect_writes`.

**F6, late verification.** Banking only, one variant per `log_verification` that is followed by at least one write,
with that call moved to the end of the run.

A variant record's `checks` names the validity check that was applied to it (`inserted_ran_cleanly` for P1, `inert` for
P5, `end_state_changed` for F3, empty otherwise). Candidates that failed their check never reach a record.

### Static checks

`static_checks(domain, task)` replays the gold run and every write-removed run, and returns `gold_errors` (S1), `gold_changes_db` (S2) and `no_effect_writes` (F3's filter). A static check that raises keeps the task record's shape:
the three values are null and the exception sits beside them in `static_error`.

### Errors are outcomes, not verdicts

A replay or grading exception on a variant, or on a base run, produces `outcome = "error"` with
`error = {"type": <class name>, "message": <first line, at most 200 chars>}`, `reward = null` and `flagged = null`. A
tool call the live environment answers with an error output is a fact about that call, recorded in `live_call_errors`
and in S1, not an error outcome. At task level, a base run that raises gets a `static.base_error` and no variants, and
a generator that raises is recorded in `static.relation_errors` under its relation id. Neither aborts the run.

### Records and fingerprints

- `task_fingerprint`: SHA-256 of the canonical JSON of `task.model_dump(mode="json")`.
- `gold_fingerprint`: SHA-256 of the canonical JSON of the gold calls.
- `gold_output_fingerprint` and a variant's `output_fingerprint`: SHA-256 over the tool outputs, each parsed as JSON
  where it parses, with every number that is not a boolean turned into a float, in canonical JSON. `25` and `25.0`
  therefore match, which matters because #397 changes number echoes on purpose.

Canonical JSON is `json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. `end_state` is
`{"variant": [agent hash, user hash], "gold": [agent hash, user hash]}`. `db_diff` names the top-level agent and user
DB fields whose values differ from the gold run's.

### Determinism and the comparison

Each task and relation gets `random.Random(f"{seed}:{task.id}:{relation}")`, so a seed fixes every sample and one
task's draw cannot shift another's. Variant records are serialised without `seconds`, sorted by
(`task_id`, `relation`, `detail`) and joined by newlines. `records_sha256` is the SHA-256 of that text, and the side
digest is the integer value of the first 12 hex digits of the SHA-256 of the files' hashes.

`compare(old_dir, new_dir, out_path)` reads only the JSON files and pairs cases before it counts anything: an OLD and a
NEW variant are a pair when domain, `task_id`, `relation`, `detail` and the canonical JSON of `calls` are equal. Only
pairs on comparable tasks count toward the hypotheses. A task is comparable when it is on both sides and its
`task_fingerprint` and `gold_output_fingerprint` agree. A variant on a comparable task with no counterpart is
unpaired. Errors never count as verdicts: every count reports attempted = verdicts + errors. The control passes when,
on airline and retail, every task is on both sides with an identical fixture, every variant is paired and each pair
has the same outcome and reward.

## Phase 2

The smoke is the first five train tasks, three seeds, `G_old` and `G_new`, 2,000 episodes per run, and the random
baseline. The full sweep (`preflight/full.py`, `experiment.py --full`) adds `G_proc`, ten seeds, retail tuning,
held-out and greedy evaluation, counterfactuals and H3/H4.

### The MDP (`preflight/mdp.py`)

- `Action(kind, calls, gold_index)` is frozen. `kind` is `"gold"`, `"read"`, `"altered"` or `"stop"`. `calls` is a
tuple of `Call` (two for a discoverable read that brings its unlock, empty for STOP). `gold_index` is the gold position
of a gold call or of the gold write an altered copy comes from.
- `task_actions(domain, task)` is deterministic: gold calls first in gold order, then the P1 read candidates, then the
altered writes, then STOP last. Reads are deduplicated by their calls and, when a discoverable read has not been
unlocked before its insertion point, come as an unlock plus the wrapper call.
- Altered writes: for each gold write, at most two copies, each swapping one string argument whose name ends in `_id`
for another value that argument name takes in this task's gold run or, if none, in another banking task's gold run
(sorted, first value). Numbers and every other argument are copied from the gold call, so #397 cannot differ between
conditions. For a discoverable wrapper the `_id` value lives inside the wrapper's JSON `arguments` string and only
that one value is rewritten there.
- `horizon(task)` is gold length + 3. `allowed(actions, made)` keeps every action at most once per episode but always
keeps STOP. A wrapper read of a tool the gold run unlocks waits for that exact gold unlock, while a read whose action
already contains its own unlock is available immediately.
- `features(actions, i, made, step, horizon)` returns the eight shared features in the fixed order `is_read`,
`is_write`, `is_gold`, `is_gold_next` (the earliest gold action not made), `is_altered`, `is_gating` (the action
contains a `log_verification`), `is_stop`, `step / horizon`. `trajectory` flattens the made actions in order.
- `order_ok(task, calls)`: for every gold `log_verification`, each of its own gated writes (the gold writes after that
verification) that appears in the trajectory must come after that verification. Extra reads never count. Calls are
matched with exact `Call` equality, inner wrapper tools included.
- `oracle` returns `true_success = (new_reward == 1.0 and order_ok and no altered write was made)`, with `order_ok`
and `no_altered` beside it. A `None` verdict (a grader error) gives false.

### The learner (`preflight/rl.py`)

- `GradeCache(domain, grade=tau2_adapter.grade)` keys a real grader call by (task id, task fingerprint, canonical JSON
of the ordered calls). A miss calls `grade` once and counts in `misses`. Instances are callable as
`cache(task, calls)`. A grader exception is cached as `None` and asked for no second time. The cache lives in one run's
process, i.e. one τ²-bench version and one condition.
- `Policy(n_features, lr=0.1)` holds `w` (zeros) and `bias` (a dict keyed by `(task id, action index)`). `probs` is a
softmax over `rows @ w` plus the bias of each candidate. `update(steps, advantage)` takes one gradient-ascent step on
`advantage * log prob` of each chosen action, i.e. `chosen row - sum(p_i * row_i)` for `w` and `indicator - p_i` for
each bias.
- `train(domain, tasks, seed, episodes, reward_fn, policy=None)`: actions, horizon and gold tool names are prepared
once per task. Each episode draws a task uniformly with `random.Random(seed)`, samples each action from the policy,
ends at STOP or the horizon, grades the final trajectory and updates on `reward - running mean of past rewards`. A
`None` reward counts as 0 for learning and is recorded with `error = True`. Records hold `task_id`, `made`, `reward`,
`error` and `extra_reads` (READ actions whose inner tool names are not in the gold run). Same arguments, same records.
- `split(pool_ids)`: sort, shuffle with `random.Random(20261110)`, take the first 30 as train and the next 10 as
held-out.

### Orchestration (`preflight/phase2.py`, `preflight/full.py`)

- `experiment.py` runs `python -B -m preflight.phase2 worker` in the condition's interpreter, so each process loads
exactly one τ²-bench version and sets `LITELLM_LOCAL_MODEL_COST_MAP=True` before importing it.
- Worker modes: `train` (REINFORCE, then before writing the file it fills `calls` and the oracle parts), `random`
(uniform over allowed actions), `eval` (a trained policy without learning: per task one greedy episode then sampled
episodes) and `oracle` (re-grade a finished run's distinct trajectories under `G_new` and fill `new_reward`,
`true_success`, `order_ok`, `no_altered`, `oracle_error`). The oracle fill refuses a file whose action fingerprints do
not match this version's action lists, so action indexes can never be reused across versions.
- `pool_ids()` reads `data/banking_pool.json` and falls back to every banking task when the file is missing.
`ensure_split` writes `split.json` once and reuses it. `full.check_split` refuses a split that differs from the
pre-registered draw.
- `run_smoke` runs old train, oracle-fills the old run in the v1.0.1 worker, runs new train and new random for each
seed, then writes the report and the metrics.

### Report and metrics (`preflight/report.py`)

- `smoke_metrics` is computed from the per-run episode records only: grader calls old and new, errors across runs and
the random baseline, gate true success over episodes 1,701 to 2,000 under `G_new`, random true success, old true
success over the last 300 episodes, extra-read rates over the last 300 episodes for old, new and random, and distinct
trajectories over old and new.
- `per_task_probabilities` returns, per task and condition, the final policy's probabilities of gold actions, extra
reads and STOP beside the uniform-start probabilities.
- `write_report` lays out the per-run table with timings and episodes per second, grader calls per episode, the random
baseline row, the kept-apart measures, the curves at fixed checkpoints, per-task final policy probabilities, the gate
measure, the horizon note and a scope line. Timings appear there and nowhere else.
- `write_metrics` writes numbers only: booleans become 0/1, so two runs of the smoke give identical metrics.

### Stated limitation: gold information

`is_gold` and `is_gold_next` tell the policy which actions the gold run uses, standing for an agent that roughly knows
the right steps. Reaching the gold path is therefore not a result. What each grader does to the remaining choices —
extra reads, order, altered writes — is.

## Phase 3

### Records and replay (`preflight/trajectories.py`)

- `as_simulation(domain, task, calls, trial=0, trimmed=False)` replays the calls with the adapter's `run`, takes the
  last `len(calls)` tool messages and writes one message per call plus its live output: the call as
  `{"id", "name", "arguments", "requestor"}` inside `tool_calls`, the output as
  `{"role": "tool", "id", "content", "error", "requestor"}`. `reward_info` holds the adapter's grade and the same
  value as `db_check.db_reward`. `trimmed=True` drops `id` and `error` from tool messages, which is what τ²-bench's
  `trim_trajectories.py` does. The simulation id is `f"{domain}:{task.id}:{trial}"`.
- `load(path)` returns the domain from `info.environment_info.domain_name`, the git commit, the embedded tasks keyed by
  id, and the runs with tool-message ids restored. `restore_ids` gives a tool message without an id the id of the
  earliest not-yet-answered tool call before it. `calls` reads every tool call in order, taking the requestor from the
  call and else from the message role.
- `task_matches(task_dict, task)` compares `get_pydantic_hash` of `Task.model_validate(task_dict)` with the local
  task's. A mismatch leaves the record with `task_matches: false` and nothing else computed.
- `replay(domain, task, run)` recomputes the run's live outputs once per process (keyed by domain, task id and the
  canonical JSON of the calls), compares them with the recorded outputs — JSON-equal where both parse, `KB_search`
  excluded because its retrieval differs — and regrades the recorded messages: strict first, `strict_replay=False`
  only where v1.0.1 has that parameter, else `incompatible`. `replay_class` is `full` when the outputs match and the
  strict regrade works, else the mode. Grades use the adapter's `env_kwargs`, so τ²-bench's read-log allowlist is
  passed on v1.0.1 and banking is graded as τ²-bench runs it. A run that raises is recorded with its exception type
  and first line and never stops its file.
- `diff_tables` uses `suite._db_diff` between the env after the run's calls and the env after the gold calls. There is
  no second copy of the DB-diff logic.
- `p1_trigger` is a `call_discoverable_agent_tool` whose inner tool's `tool_kind` is READ, whose inner name the gold
  run never calls, and whose live output is not an error. `p3_trigger` is a run call with the same tool (inner tool
  for wrappers) as a gold call and an argument with the same value but a different JSON spelling, wrapper argument
  strings unpacked first.
- `schema_floats` resolves each call's tool through the toolkits and reports every whole-number float passed for a
  parameter whose `openai_schema` type is `integer`. `as_integers` rewrites exactly those values, inside a wrapper's
  JSON `arguments` string when that is where the value lives.
- `numeric_pair` compares a run with its integer respelling: outputs equal after whole-number floats become ints
  everywhere, end states equal under the same normalization, and both rewards. `inconsistency` is true only when
  neither copy errored at those calls, both comparisons hold and the rewards differ.
- `late_verification` needs the gold `log_verification`: `log_after_first_write` when the first assistant WRITE (by
  `tool_kind`, `log_verification` itself excluded) precedes the first `log_verification`, or no verification exists at
  all. `access_before_facts` compares the first assistant call naming the user (id, email or name from the gold
  verification) with how many of the four facts the user's own messages already contain, matched after case folding
  and removing spaces and punctuation. It is `None` when the run has no user message with text, which is the case for
  every synthetic run. Both are false when the gold run has no `log_verification`.
- `regrade(paths, out_dir)` writes one JSONL record per run and file, labelled `old` or `new` from the environment's
  `SOURCE.txt` the same way phase 1 reads its version. The record holds the ids, `task_matches`, `replay_class`,
  `reward`, `recorded_reward`, `mismatched`, `p1`, `p3`, `diff_tables`, `n_calls`, `nongold_call` (some call, inner
  tool for wrappers and numbers normalized, is not in the gold run), `schema_floats` and `numeric_pair` on airline,
  `late_verification` on banking, and `error` when something raised.
- `flags(phase1_files, out_path)` turns phase 1's G_old per-case files into `{"P1": [...], "P3": [...]}`: a task is
  listed when a variant of that relation is flagged, P1 only where `inserted_in_gold` is false. The `analyze` and
  `sample` subcommands import `preflight.trajectory_analysis` lazily, so the replay module never needs numpy.

### Join, H5 and H7 (`preflight/trajectory_analysis.py`)

- Records of the two sides are paired by `id`, the first new-side record of an id being the counterpart. A pair enters
  H5 when both sides are banking, both carry `task_matches`, the task is in the pool and both sides have a strict or
  lenient grade. A pair is an event when the two rewards differ.
- Explained: a P1 trigger on a task phase 1 flagged, with the old side's `diff_tables` exactly
  `["agent_discoverable_tools"]`, or a P3 trigger on a flagged task with a non-empty old diff whose tables all lie in
  the three P3 tables. Recall is `explained / events`, its interval a task-clustered percentile bootstrap (2,000
  resamples, `numpy.random.default_rng(0)`, task ids resampled with replacement). With no events, `recall`, `ci` and
  `held` are null and `testable` is false. `held = recall >= 0.90 and ci[0] >= 0.75`.
- Also written: counts per replay class on both sides, the unexplained events one by one with their id, task, diff
  tables and triggers, each trigger's precision (events among runs carrying it), and three baselines — B1 a run on a
  flagged task, B2 any call not in gold, B3 above the median call count — each with the event rate on the positive and
  negative side and their enrichment, null when the negative rate is 0 or missing. Every H5 summary is repeated per
  git commit and per file stem.
- H7 is descriptive. Airline: runs under new with schema floats, numeric-pair inconsistencies, and the control events
  (a verdict difference between the two sides). Banking: reward-1 runs under new with a late log, and with early
  customer access. `analyze` writes `h5.json`, `h7.json`, `runs.csv` (one row per joined pair) and a short
  `report.md`.

### H6 sampling (`trajectory_analysis.sample`)

- A run is S4 when its pair is an H5 event, S1 when its new-side reward is 1 and any automated flag fires (a failed
  WRITE, `late_verification` true, or a WRITE not in gold after dropping arguments equal to the tool's schema default
  and normalizing numbers), S2 when its reward is 1 with none, and S3 when its reward is 0. S4 is taken out of S1-S3, so a
  run is drawn once. Ids are sorted and drawn with `random.Random(0)`, all of them when a stratum is short, and any
  shortfall is written to `shortfalls.json`.
- The draw is shuffled with the same rng and written as `packets/NN.json`: the task instructions, the policy text the
  file carries when it has one (else the domain's policy file), and the messages. `key.json` holds the packet number,
  the run id and the stratum, and nothing in a packet names a reward, a stratum, a flag or an id.

### Published runs

The leaderboard trajectory files stay outside this repository and are never redistributed. Paths are arguments, and
only counts, identifiers and the few report values that differ from gold are written (amendment 7 names the held-out
files and the rule that chose them). With `OLD` and `NEW` the two interpreters, `PUBLISHED` the downloaded files and
`PHASE1` a phase 1 run folder, run from an empty working folder:

```sh
$OLD -m preflight.trajectories regrade PUBLISHED/*.json --out old        # replay, regrade and triggers per run
$NEW -m preflight.trajectories regrade PUBLISHED/*.json --out new
$OLD -m preflight.trajectories flags PHASE1/results/old/banking_knowledge.json --out flags.json
$NEW -m preflight.trajectories analyze --old old --new new --flags flags.json \
  --pool data/banking_pool.json --out analysis                           # H5, H7, runs.csv, report
$NEW -m preflight.trajectories sample --old old --new new --flags flags.json \
  --pool data/banking_pool.json --files PUBLISHED/*_banking.json --out h6-sample   # H6's blinded sample
```

H6's labels (`h6-labels/`: two model annotators' answers, their agreement and the adjudication) were produced outside
this code, so `scripts/collect_results.py --phase3 .` needs that folder too. The rest of `results/phase3.json` comes
from the commands above.

### The synthetic experiment (`preflight/phase3.py`)

- `experiment.py --phase 3` (the default) dispatches `python -m preflight.phase3 worker` into each environment, as phase
  1 does, so one process loads one τ²-bench version and every recorded output comes from the env that grades it.
- Banking: for every pool task, the gold run, a `trimmed=True` copy, the first P1 variant whose detail starts with
  `extra agent read`, the P3 variant when the task has one, and the F6 late-verification variant when it has one, as
  trials 0-4. Airline: every task's gold run and a copy whose integers for integer-schema parameters are floats.
- The parent writes the synthetic files in each env, regrades them there, writes the flags from the old side's
  records (`p1` or `p3` true with a reward of 0), joins the two sides with `trajectory_analysis.analyze` and writes the
  metrics. The run is deterministic: the flags and the metrics are recomputed from per-run records, so two runs give
  the same numbers.

## Known limits

- Only the DB-state term decides a verdict. A task whose reward also depends on communication or assertions is
  covered only for its state term.
- Tasks whose base run does not score 1 get no variants and stay in the results as `gold not accepted`. A relation
  that cannot be built yields no variant, so counts differ from task to task.
- F1 cannot separate an empty run from a gold run that does not change the DB, so its counts are split by
  `gold_changes_db`. Empty-gold tasks get no F1 at all.
- `end_state` is a pair of DB hashes. `db_diff` names top-level fields, not the values inside them.
- `records_sha256` and the side digests cover variant records only, so per-case repeatability is checked from the
  per-case files as well.
- The airline and retail control assumes those domains grade identically in both versions. A mismatch is treated as a
  fault in this harness, not as a finding.
- Phase 2's policy is open-loop, its action set is built once per task, its grader cache is per run and never saved,
  and its oracle is ours. A G_old run needs the v1.0.1 interpreter to learn its true success, so the pipeline
  dispatches that step as a separate process. The smoke is not the full sweep: held-out evaluation, G_proc, H3 and H4
  are not computed by `--phase 2`.
- Phase 3's synthetic experiment (`experiment.py --phase 3`) shows that the pipeline works, not what published agents
  did. The published-run results come from the same commands on the leaderboard files, which are not in this
  repository. `access_before_facts` is `None` on synthetic runs because they carry no user message with text. A run's
  end state is compared as two DB hashes and a list of top-level table names. H6's labels come from outside this code
  (two independent language-model annotators, then adjudication, under amendments 8 and 9), and no population rate can be
  read off a stratified sample. A file whose incompatible share is high is reported, not tuned around.
