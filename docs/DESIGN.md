# Design

## Scope

Every verdict in this project is the DB-state verdict of `EnvironmentEvaluator.calculate_reward`, called unchanged with
the environment arguments τ²-bench's runner uses for the task. Tasks are those with `DB` in `reward_basis`; the
COMMUNICATE, NL_ASSERTION and ACTION terms are never the subject of a variant. Phase 1 covers `banking_knowledge`
(built with the `grep_only` retrieval variant, so no API is reached), `airline` and `retail`. Phase 2 covers banking
only, over the tasks phase 1 found comparable across the two tags. This build implements the phase 2 smoke only; the
later full sweep is described in the pre-registration and is not implemented here.

Environments are built only in `preflight/tau2_adapter.py`, never through τ²-bench's constructors directly. On v1.0.1
banking logs a read to the hashed agent DB only when the read is on a per-task allowlist built by τ²-bench's runner;
the adapter derives it with τ²-bench's own `_derive_read_log_allowlist` and passes it to the live environment and to
the grader.

## Module map

| Module | Job |
| --- | --- |
| `preflight/tau2_adapter.py` | fixed: `Call`, `tasks`, `gold_calls`, `env_kwargs`, `fresh_env`, `run`, `grade`; replays calls on a live env and scores the trajectory |
| `preflight/relations.py` | `tool_kind`, `Variant`, the six generators in `RELATIONS`, `build_variants`, `static_checks` |
| `preflight/suite.py` | `run_suite`: base runs, variants, per-case records, fingerprints, provenance, `records_sha256` |
| `preflight/compare.py` | `compare`: case-by-case pairing, report and metrics from two result folders, standard library only |
| `preflight/mdp.py` | `Action`, `task_actions`, `horizon`, `allowed`, `features`, `trajectory`, `order_ok`, `oracle` |
| `preflight/rl.py` | `GradeCache`, `Policy`, `train`, `split`; `N_FEATURES` and `DEFAULT_LR` |
| `preflight/phase2.py` | `run_smoke`, `pool_ids`, `ensure_split`, and the worker CLI (`train`, `random`, `oracle`) |
| `preflight/report.py` | `smoke_metrics`, `per_task_probabilities`, `write_report`, `write_metrics` |
| `preflight/__main__.py` | CLI for the phase 1 `run` and `compare` commands |
| `experiment.py` | the phase 1 suite under both environments and the phase 2 smoke run |
| `demo.py` | a few airline tasks through the relation suite, or a short phase 2 training run, printed |

## Phase 1

### Base run

Each task's gold run is replayed once on a fresh environment and graded. Only a task whose base verdict is 1 gets
variants; a base run that raises is recorded as `base_outcome: "error"` with `static.base_error`, keeps the record's
shape, and gets no variants.

### Resolving a call to a tool

`tool_kind(env, call)` answers what a call reaches, not what it is named. `call_discoverable_agent_tool` reaches the
tool named by `agent_tool_name` in `env.tools`, `call_discoverable_user_tool` the tool named by
`discoverable_tool_name` in `env.user_tools`; any other name is looked up in `env.tools` for an assistant call and in
`env.user_tools` for a user call. Existence is settled by `has_tool(name)` when the toolkit has one, and the declared
kind by the toolkit's `tool_type(name)` when that is callable, else by the `Tool` object's own `tool_type`. The kind is
read as a lowercase string (an enum's `value` when it is one); anything outside `read`, `write`, `generic` and `think`
counts as `generic`, and a name in neither toolkit is `unknown`.

A live tool message counts as an error when it has `error=True` or when its content is a string starting with
`Error:` — the second form is how the discoverable wrappers report a failure.

### Relations and their validity filters

The generators in `preflight/relations.py` build candidates; `build_variants` is the only entry point the suite uses,
and it drops any variant whose calls equal the gold calls (amendment 3), reporting each skip.

**P1, an extra read.** `_p1_plans` collects the assistant toolkit's direct reads plus its discoverable reads, skipping
the wrapper and unlock names, keeping only tools whose declared `tool_type(name)` says `read`. Required argument names
come from the tool's JSON schema, preferring `openai_schema["function"]["parameters"]`; a bare parameter mapping is
not a schema, and the callable's signature then says which arguments are required. Each required name must be
supplied by an earlier gold call, wrapper `arguments` strings included, and the read goes right after the last call
that supplied one. A discoverable read is called through `call_discoverable_agent_tool` with a JSON `arguments`
string; when the run has not unlocked it before, `unlock_discoverable_agent_tool` goes right before the read. The
variant is kept only if replaying the prefix through the inserted read ends in a tool message that is not an error.
Candidates are sorted by route, name and position and then shuffled with `rng`, with no priority by kind or route, so
the draw is reproducible and a read that fails its replay never takes one of the five slots. `inserted_in_gold` is
true when the inserted (inner) tool also appears in the gold run; H1(a) counts those variants apart on both sides.

**P3, respelled numbers.** Every numeric argument is respelled with the same value, integral `int` to `float` and
integral `float` to `int`; the JSON `arguments` string of a discoverable wrapper is rewritten token by token over its
number literals. Only the three wrappers (`call_discoverable_agent_tool`, `call_discoverable_user_tool`,
`give_discoverable_user_tool`) get that treatment; a string argument named `arguments` on any other tool is left
alone, and booleans are not numbers. Nothing to respell means no variant, recorded as `identical_to_gold`.

**P5, an unknown tool.** The position is drawn with `rng` from `0..len(gold)`. The variant is kept only when the agent
and user DB hash pair right after the inserted call equals the pair right before it; otherwise the task's `static`
carries `p5_not_inert`. If that validation replay raises, the exception travels out of the generator and the suite
records it in `static.relation_errors` under `P5`, with no variant built: a check that did not complete is never
reported as passed.

**F1, an empty run.** Dropped when the gold run is also empty, since then it is the gold run itself.

**F3, a dropped write.** Only calls of kind `write`. The gold end state is replayed once; each write is dropped in turn
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
a generator that raises is recorded in `static.relation_errors` under its relation id; neither aborts the run.

### Records and fingerprints

- `task_fingerprint`: SHA-256 of the canonical JSON of `task.model_dump(mode="json")`.
- `gold_fingerprint`: SHA-256 of the canonical JSON of the gold calls.
- `gold_output_fingerprint` and a variant's `output_fingerprint`: SHA-256 over the tool outputs, each parsed as JSON
  where it parses, with every number that is not a boolean turned into a float, in canonical JSON. `25` and `25.0`
  therefore match, which matters because #397 changes number echoes on purpose.

Canonical JSON is `json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. `end_state` is
`{"variant": [agent hash, user hash], "gold": [agent hash, user hash]}`; `db_diff` names the top-level agent and user
DB fields whose values differ from the gold run's.

### Determinism and the comparison

Each task and relation gets `random.Random(f"{seed}:{task.id}:{relation}")`, so a seed fixes every sample and one
task's draw cannot shift another's. Variant records are serialised without `seconds`, sorted by
(`task_id`, `relation`, `detail`) and joined by newlines; `records_sha256` is the SHA-256 of that text, and the side
digest is the integer value of the first 12 hex digits of the SHA-256 of the files' hashes.

`compare(old_dir, new_dir, out_path)` reads only the JSON files and pairs cases before it counts anything: an OLD and a
NEW variant are a pair when domain, `task_id`, `relation`, `detail` and the canonical JSON of `calls` are equal. Only
pairs on comparable tasks count toward the hypotheses; a task is comparable when it is on both sides and its
`task_fingerprint` and `gold_output_fingerprint` agree. A variant on a comparable task with no counterpart is
unpaired. Errors never count as verdicts: every count reports attempted = verdicts + errors. The control passes when,
on airline and retail, every task is on both sides with an identical fixture, every variant is paired and each pair
has the same outcome and reward.

## Phase 2

This build implements the smoke only: the first five train tasks, three seeds, `G_old` and `G_new`, 2,000 episodes per
run, and the random baseline. The later full sweep (`G_proc`, ten seeds, held-out evaluation, H3 and H4, gold and
empty baselines) is in the pre-registration and is not implemented in this code.

### The MDP (`preflight/mdp.py`)

- `Action(kind, calls, gold_index)` is frozen; `kind` is `"gold"`, `"read"`, `"altered"` or `"stop"`; `calls` is a
tuple of `Call` (two for a discoverable read that brings its unlock, empty for STOP); `gold_index` is the gold position
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
keeps STOP; a wrapper read of a tool the gold run unlocks waits for that exact gold unlock, while a read whose action
already contains its own unlock is available immediately.
- `features(actions, i, made, step, horizon)` returns the eight shared features in the fixed order `is_read`,
`is_write`, `is_gold`, `is_gold_next` (the earliest gold action not made), `is_altered`, `is_gating` (the action
contains a `log_verification`), `is_stop`, `step / horizon`. `trajectory` flattens the made actions in order.
- `order_ok(task, calls)`: for every gold `log_verification`, each of its own gated writes (the gold writes after that
verification) that appears in the trajectory must come after that verification. Extra reads never count. Calls are
matched with exact `Call` equality, inner wrapper tools included.
- `oracle` returns `true_success = (new_reward == 1.0 and order_ok and no altered write was made)`, with `order_ok`
and `no_altered` beside it; a `None` verdict (a grader error) gives false.

### The learner (`preflight/rl.py`)

- `GradeCache(domain, grade=tau2_adapter.grade)` keys a real grader call by (task id, task fingerprint, canonical JSON
of the ordered calls). A miss calls `grade` once and counts in `misses`; instances are callable as
`cache(task, calls)`. A grader exception is cached as `None` and asked for no second time. The cache lives in one run's
process, i.e. one τ²-bench version and one condition.
- `Policy(n_features, lr=0.1)` holds `w` (zeros) and `bias` (a dict keyed by `(task id, action index)`). `probs` is a
softmax over `rows @ w` plus the bias of each candidate; `update(steps, advantage)` takes one gradient-ascent step on
`advantage * log prob` of each chosen action, i.e. `chosen row - sum(p_i * row_i)` for `w` and `indicator - p_i` for
each bias.
- `train(domain, tasks, seed, episodes, reward_fn, policy=None)`: actions, horizon and gold tool names are prepared
once per task; each episode draws a task uniformly with `random.Random(seed)`, samples each action from the policy,
ends at STOP or the horizon, grades the final trajectory and updates on `reward - running mean of past rewards`. A
`None` reward counts as 0 for learning and is recorded with `error = True`. Records hold `task_id`, `made`, `reward`,
`error` and `extra_reads` (READ actions whose inner tool names are not in the gold run). Same arguments, same records.
- `split(pool_ids)`: sort, shuffle with `random.Random(20261110)`, take the first 30 as train and the next 10 as
held-out.

### Orchestration (`preflight/phase2.py`)

- `experiment.py` runs `python -B -m preflight.phase2 worker` in the condition's interpreter, so each process loads
exactly one τ²-bench version and sets `LITELLM_LOCAL_MODEL_COST_MAP=True` before importing it.
- Worker modes in this build: `train` (REINFORCE, then before writing the file it fills `calls` and the oracle parts),
`random` (uniform over allowed actions), and `oracle` (re-grade a finished run's distinct trajectories under `G_new`
and fill `new_reward`, `true_success`, `order_ok`, `no_altered`, `oracle_error`).
- Conditions in the smoke are `G_old` and `G_new`. `G_proc` is defined in the pre-registration as exploratory and is
not run in this build. A `G_old` run's oracle is filled afterwards, in the v1.0.1 interpreter.
- `pool_ids()` reads `data/banking_pool.json` and falls back to every banking task when the file is missing;
`ensure_split` writes `split.json` once and reuses it. The smoke uses the first five train tasks, seeds 0 to 2, and
2,000 episodes per run.
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
the right steps. Reaching the gold path is therefore not a result; what each grader does to the remaining choices —
extra reads, order, altered writes — is.

## Known limits

- Only the DB-state term decides a verdict; a task whose reward also depends on communication or assertions is
  covered only for its state term.
- Tasks whose base run does not score 1 get no variants and stay in the results as `gold not accepted`; a relation
  that cannot be built yields no variant, so counts differ from task to task.
- F1 cannot separate an empty run from a gold run that does not change the DB, so its counts are split by
  `gold_changes_db`; empty-gold tasks get no F1 at all.
- `end_state` is a pair of DB hashes; `db_diff` names top-level fields, not the values inside them.
- `records_sha256` and the side digests cover variant records only, so per-case repeatability is checked from the
  per-case files as well.
- The airline and retail control assumes those domains grade identically in both versions; a mismatch is treated as a
  fault in this harness, not as a finding.
- Phase 2's policy is open-loop, its action set is built once per task, its grader cache is per run and never saved,
  and its oracle is ours. A G_old run needs the v1.0.1 interpreter to learn its true success, so the pipeline
  dispatches that step as a separate process. This build is the smoke only; held-out evaluation, G_proc, H3 and H4
  are not computed here.
