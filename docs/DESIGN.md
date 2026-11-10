# Design

## Scope

Every verdict in this project is the DB-state verdict of `EnvironmentEvaluator.calculate_reward`, called unchanged.
Tasks are those with `DB` in `reward_basis`; the COMMUNICATE, NL_ASSERTION and ACTION terms are never the subject of a
variant. Phase 1 covers `banking_knowledge` (built with the `grep_only` retrieval variant, so no API is reached),
`airline` and `retail`.

## Module map

| Module | Job |
| --- | --- |
| `preflight/tau2_adapter.py` | fixed: `Call`, `tasks`, `gold_calls`, `env_kwargs`, `fresh_env`, `run`, `grade`; replays calls on a live env and scores the trajectory |
| `preflight/relations.py` | `tool_kind`, `Variant`, the six generators in `RELATIONS`, `build_variants`, `static_checks` |
| `preflight/suite.py` | `run_suite`: base runs, variants, per-case records, fingerprints, provenance, `records_sha256` |
| `preflight/compare.py` | `compare`: report and metrics from two result folders, standard library only |
| `preflight/__main__.py` | CLI for `run` and `compare` |
| `experiment.py` | the suite under both environments, then the comparison |
| `demo.py` | a few airline tasks, printed |

## Resolving a call to a tool

`tool_kind(env, call)` answers what a call reaches, not what it is named. `call_discoverable_agent_tool` reaches the
tool named by `agent_tool_name` in `env.tools`, `call_discoverable_user_tool` the tool named by
`discoverable_tool_name` in `env.user_tools`; any other name is looked up in `env.tools` for an assistant call and in
`env.user_tools` for a user call. Existence is settled by `has_tool(name)` when the toolkit has one, and the declared
kind by the toolkit's `tool_type(name)` when that is callable, else by the `Tool` object's own `tool_type`. The kind is
read as a lowercase string (an enum's `value` when it is one); anything outside `read`, `write`, `generic` and `think`
counts as `generic`, and a name in neither toolkit is `unknown`. Toolkits are read whether the environment exposes them
as a dict or as a sequence of tools with `.name`.

A live tool message counts as an error when it has `error=True` or when its content is a string starting with
`Error:` — the second form is how the discoverable wrappers report a failure.

## Relations and their validity filters

The generators in `preflight/relations.py` build candidates; `build_variants` is the only entry point the suite uses,
and it applies the rules that make an expected verdict defensible.

**P1, an extra read.** `_p1_plans` collects the assistant toolkit's direct reads plus its discoverable reads, skipping
the wrapper names and the unlock names. A tool is a read when the toolkit's declared `tool_type(name)` says so: the
`Tool` object a toolkit hands out need not carry the kind. A discoverable read is a candidate whether or not the gold
run has unlocked it, because after the #329 fix a read is logged only when it is on the allowlist τ²-bench's runner
derives from the gold run; a read the gold run already uses therefore carries no information about the fix, and one it
never unlocks is the interesting case. Required argument names come from the tool's JSON schema, preferring the
authoritative `openai_schema["function"]["parameters"]` when it carries a `properties` mapping, then `parameters` or
`params` when one of those is a full schema. A bare parameter mapping is not a schema: the decorated function's
signature then says which arguments are required and which are optional, and only when neither is available is every
mapped name treated as required. Each required name must be supplied by an earlier gold call, wrapper `arguments`
strings included; the insertion point is after the last call that supplied one, and never before the point where the
run unlocks the tool. A discoverable read is called through `call_discoverable_agent_tool` with a JSON `arguments`
string; when the run has not unlocked that tool before the insertion point, `unlock_discoverable_agent_tool` is
inserted right before the read (`_insert_with_unlock`), so the variant is the gold run plus one block. The variant is
kept only if replaying the prefix through the inserted read ends in a tool message that is not an error. Candidates
are shuffled with `rng` and tried in that order, so a read that fails its replay never takes one of the five slots; the
first five that run cleanly are the variants. `inserted_in_gold` is true when the inserted tool (the inner tool, for a
wrapper) also appears in the gold run, and H1(a) counts those variants apart on both sides.

**P3, respelled numbers.** Every numeric argument is respelled with the same value, integral `int` to `float` and
integral `float` to `int`; a wrapper's JSON `arguments` string is rewritten token by token over its number literals, so
the parsed value is unchanged and the string form differs. Booleans are left alone. Nothing to respell means no
variant, recorded as `identical_to_gold`. This is the #397 check: a grader that hashes number echoes rather than
values rejects the same state written differently.

**P5, an unknown tool.** The position is drawn with `rng` from `0..len(gold)`. The variant is kept only when the agent
and user DB hash pair right after the inserted call equals the pair right before it; otherwise the generator returns
nothing and the task's `static` carries `p5_not_inert`. If that validation replay raises, the exception travels out of
the generator and `build_variants` and the suite records it in the task's `static.relation_errors` under `P5`, with no
variant built: a check that did not complete is never reported as passed.

**F1, an empty run.** Dropped when the gold run is also empty, since then it is the gold run itself; the task records
`identical_to_gold`.

**F3, a dropped write.** Only calls of kind `write`. The gold end state is replayed once; each write is dropped in turn
and the variant kept only when the resulting agent and user hash pair differs from the gold run's. A write whose
removal changes nothing has nothing for the grader to catch: it gets no variant and is counted in `static_checks` as
`no_effect_writes`. This is amendment 1.

**F6, late verification.** Banking only, one variant per `log_verification` that is followed by at least one write, with
that call moved to the end of the run.

`build_variants` drops any variant whose calls equal the gold calls (amendment 3) and reports each skip; the suite
keeps the relation ids sorted in the task's `identical_to_gold` list. A P5 relation that yields nothing is reported as
`p5_not_inert`, because the generator builds one candidate and drops it only when the unknown call moved the live DB.
`static_checks` replays the gold run and every write-removed run to produce `gold_errors` (S1), `gold_changes_db` (S2)
and `no_effect_writes`.

A variant record's `checks` names the validity check that was applied to it (`inserted_ran_cleanly` for P1, `inert` for
P5, `end_state_changed` for F3, empty otherwise). Candidates that failed their check never reach a record: they are
dropped inside the generator or `build_variants`, and the reason survives as `p5_not_inert` or `no_effect_writes` in the
task's static entry, or as a `relation_errors` entry when the check itself raised.

## Errors are outcomes, not verdicts

A replay or grading exception on a variant, or on a base run, produces `outcome = "error"` with
`error = {"type": <class name>, "message": <first line, at most 200 chars>}`, `reward = null` and `flagged = null`. A
variant that raised while replaying carries no `end_state`; one that raised only in the grader keeps the hash pair it
reached. Errors are never counted as rewards or flags: the comparison reports attempted = verdicts + errors for every
domain and relation. A tool call the live environment answers with an error output is a fact about that call, recorded
in `live_call_errors` and in S1, not an error outcome. Both rules are amendment 2.

At task level, a base run that raises gets a `static.base_error` and no variants, and a generator that raises is
recorded in `static.relation_errors` under its relation id; neither aborts the task or the domain run. A static check
that raises keeps the task record's shape: `gold_errors`, `gold_changes_db` and `no_effect_writes` are then null and
the exception sits beside them in `static_error`.

## Fixtures: three fingerprints

- `task_fingerprint`: SHA-256 of the canonical JSON of `task.model_dump(mode="json")`.
- `gold_fingerprint`: SHA-256 of the canonical JSON of the gold calls.
- `gold_output_fingerprint`: SHA-256 of the gold run's tool outputs, each parsed as JSON where it parses, with every
  number that is not a boolean turned into a float, in canonical JSON. `25` and `25.0` therefore match, which matters
  because #397 changes number echoes on purpose.

Canonical JSON is `json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. `compare` marks a task
"fixture changed" when its task fingerprint or its gold output fingerprint differs, naming which, and leaves it out of
H1 and out of the control. The comparison is per task rather than per file because `v1.0.1` edits banking's shared
`db.json`.

## Determinism and record hashing

Each task and relation gets `random.Random(f"{seed}:{task.id}:{relation}")`, so a seed fixes every sample and one
task's list of tasks cannot shift another's draw. Variant records are serialised without `seconds`, sorted by
(`task_id`, `relation`, `detail`) and joined by newlines; `records_sha256` is the SHA-256 of that text. The comparison
reduces each side's per-file hashes to `records_digest`: the integer value of the first 12 hex digits of the SHA-256 of
the files' `records_sha256` values, sorted by domain (and file name) and joined by newlines. `records_sha256` covers
the variant records only, so matching digests show that both runs produced the same variants and the same verdicts;
they say nothing about a task's base outcome, its static checks or its fingerprints, which the pre-registration's
per-case repeatability is checked against outside the project, from the per-case files a run leaves behind.

## Comparison

`compare(old_dir, new_dir, out_path)` reads only the JSON files, writes the Markdown report to `out_path`, writes
`metrics.json` beside it, and returns the metrics object. The report holds, in order:

1. the old and new directories, then the fixture-changed tasks with the fingerprint that differs;
2. a table per domain and relation on each side: attempted, verdicts, errors, flagged variants and flagged tasks;
3. the same table for P1 split by `inserted_in_gold` (`in gold`, `not in gold`);
4. the flags present on OLD and absent on NEW, by task and relation;
5. one line each for H1(a), H1(b) and H2, starting `held` or `not held`, with the numbers behind them;
6. the control line and its mismatches;
7. F1 counts per side, domain and `gold_changes_db` (`unknown` when the task has no boolean static entry), with
   attempted, verdicts, errors, passed variants and passed tasks;
8. the relations skipped as identical to gold, counted per side, domain and relation;
9. `p5_not_inert` tasks and `no_effect_writes` per side and domain;
10. the `records_sha256` of every input file.

Only verdicts count toward held or not held; errors are printed beside them. The checks themselves are as
pre-registered:

- H1(a): OLD P1 flags at least one banking task and NEW flags none, on both sides counting only variants without
  `inserted_in_gold`, and ignoring tasks whose fixture changed. NEW must have produced at least one verdict in that
  group: a run where every comparable NEW variant errored has not shown a fix.
- H1(b): the same for P3, likewise with at least one NEW verdict.
- H2: F3 pass rate at most 1%, at least 20 airline tasks where an empty run passes, F6 passing on at least half of the
  banking tasks that have an F6 variant, and at least one flagged P-relation variant. A part with no verdict behind it
  is not held.
- Control: on airline and retail tasks with an identical fixture, every variant matched by (task, relation, `detail`)
  has the same outcome and the same reward on both sides. Any mismatch fails the control and invalidates the run until
  it is explained; a variant present on one side only is listed as missing.

`metrics.json` is a flat object of numbers only, never booleans, with a fixed key set: `h1a_held`, `h1b_held`,
`h2_held`, `control_passed`, `control_mismatches`, `p1_flagged_tasks_old`, `p1_flagged_tasks_new`,
`p1_in_gold_flagged_tasks_new`, `p3_flagged_tasks_old`, `p3_flagged_tasks_new`, `f1_pass_airline_new`,
`f3_pass_rate_new`, `f6_pass_share_new`, `changed_tasks`, `attempted_old`, `attempted_new`, `errors_old`, `errors_new`,
`records_digest_old`, `records_digest_new`. The key set is fixed so that published numbers stay comparable, and a test
asserts it exactly.

## Known limits

- Only the DB-state term decides a verdict; a task whose reward also depends on communication or assertions is covered
  only for its state term.
- Tasks whose base run does not score 1 get no variants and stay in the results as "gold not accepted".
- A relation that cannot be built on a task yields no variant: no write, no gated verification, no read whose required
  arguments appear in the run, an empty gold run. Counts therefore differ from task to task.
- F1 cannot separate an empty run from a gold run that does not change the DB, so its counts are reported split by
  `gold_changes_db`; empty-gold tasks get no F1 at all (amendment 3).
- `end_state` is a pair of DB hashes; `db_diff` names top-level fields that differ, not the values inside them.
- `tau2_version` is the second word of the first line of the environment's `SOURCE.txt`, and is null when the file is
  missing or that line is too short.
- `records_sha256` covers variant records only: a repeated run that differs in a task's base outcome, gold output
  fingerprint or static checks keeps the same digest, so the digest is not a full per-case repeatability check.
- The static checks replay the gold run once per write, and every P1 candidate replays a prefix, so a suite run costs
  several replays per task.
- The airline and retail control assumes those domains grade identically in both versions; a mismatch is treated as a
  fault in this harness, not as a finding.
- `experiment.py` passes on whatever the subprocesses wrote; it does not recompute the metrics. The independent
  recompute of amendment 2 is done outside the project, from the per-case files the run leaves behind.
