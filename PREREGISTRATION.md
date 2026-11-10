# Pre-registration

Committed before any relation, baseline or RL run. Any later change is added under "Amendments" with its date and
reason; nothing above that heading is edited after this commit.

## Question

Can an LLM-free preflight, built only from a known-good run and the environment's own tool metadata, find the grading
bugs that τ²-bench's maintainers later fixed by hand? And do such bugs change what a policy learns?

## Grader under test

`tau2.evaluator.evaluator_env.EnvironmentEvaluator.calculate_reward`, called unchanged, at:

| Name | Version | Commit |
|---|---|---|
| G_old | tag `pre-v1.0.1` | `b51a6d69e26f` |
| G_new | tag `v1.0.1` | `fc0055dc4e0a` |

Each version runs with its own task data. Only the DB and ENV_ASSERTION terms are in scope (the terms
`calculate_reward` computes). COMMUNICATE, NL_ASSERTION and ACTION terms are out of scope: phase 1 uses tasks whose
`reward_basis` has DB in it, and the report says the verdict is the DB-state verdict. Banking uses the
`grep_only` retrieval variant (offline). No LLM is called and no API key is present.

Domains: banking_knowledge, airline, retail. Telecom is out of phase 1.

## Already seen before this commit (disclosure)

- The documented fixes in tau2's CHANGELOG (#329, #397, #402, #403, #374, #273) and the static count in the
  research notes (airline 23 of 50 tasks need no DB change).
- An environment smoke test that replayed gold actions with `make_tool_call` (no grader call): airline 24 of 50 gold
  runs leave the DB unchanged; retail 18 gold calls raise an error and 11 of 114 leave the DB unchanged; banking 4 of
  97 (G_old) and 7 of 97 (G_new) leave it unchanged. These are environment facts, not results.
- **Correction to the earlier plan:** the replay crash on a hallucinated tool call (#273) was fixed in `2be6916`
  (29 Apr 2026), before `pre-v1.0.1`. It cannot differ between G_old and G_new. It is tested only if a third version
  (the parent of `2be6916`) is installed, and reported separately.

## Method

**Base run.** For each task, the gold actions run in order on a live environment; each call and its live tool output
become a message trajectory, which the grader scores. A task whose base run does not score 1 is reported as "gold not
accepted" and gets no variants.

**Relations.** Each variant has an expected verdict. A *flag* is a variant whose verdict differs from it.

| Id | Variant | Expected |
|---|---|---|
| P1 | Insert one call to a declared-READ tool (for banking, a discoverable tool declared READ, called the way an agent calls it), with arguments taken from the run's own entities | pass |
| P3 | Respell numeric arguments (`25` ↔ `25.0`; the same JSON number) | pass |
| P5 | Insert one call to a tool name that does not exist | pass |
| F1 | Empty run | fail |
| F3 | Drop one WRITE call (each in turn) | fail |
| F6 | Move a gating write (banking `log_verification`) after the writes it gates | fail |
| S1 | Every gold action runs without error (static) | no error |
| S2 | The gold run changes the DB (static) | changes |

Variants are drawn with seed 0; at most 5 P1 variants per task.

**Triage.** Every flag is published with its DB diff. After the run, each is labelled by hand "real bug", "intended
design" or "unclear", with a one-line reason.

## Hypotheses

**H1, recall on documented bugs (the pilot test).**
- (a) P1 flags at least one banking task on G_old and none on G_new (#329). P1 variants whose inserted tool already
  appears in the task's gold run are counted apart, because v1.0.1 keeps logging reads the gold requires by design.
- (b) P3 flags at least one banking task on G_old and none on G_new (#397).
- (c) Only with the third version: P5 flags on it and not on G_old (#273).

Expected misses, stated now: #374 (wrong gold value) and #403 (tool output order) need the policy's meaning; #402 has
the same symptom as #329 and is not counted on its own.

**H2, the two sides of end-state hashing (G_new).** F3 passes in at most 1% of its variants; F1 passes on at least 20
airline tasks; F6 passes on at least half of the banking tasks with a gold `log_verification`; some P-relation flags at
least one task in some domain.

**H3, RL finds nothing the preflight missed (phase 2).** Over 10 seeds, every distinct trajectory with reward 1 that
the oracle marks wrong falls in a flagged class (recall at least 90%). Any RL find outside them is reported as a miss.

**H4, a false reject shapes behaviour (phase 2).** Trained against G_old, policies end with a lower verification-read
rate than against G_new (paired by seed, 95% bootstrap interval excludes 0), with no gain in true success. Also
measured on held-out tasks.

## Phase 2 design (fixed now, run only after the pilot passes)

- Seeds 0-9, the same seeds for every grader. Banking split: 30 train and 10 held-out tasks, drawn with seed 20261110
  from the tasks whose base run scores 1 on both versions. Hyperparameters are tuned on retail only.
- Policy: softmax over shared action features plus a per-task bias; REINFORCE with a running baseline; numpy only.
  Reward is the grader's verdict at the end, with no step cost.
- True success (our oracle, stated as ours): G_new passes, verification comes before the writes it gates, and no
  altered write was made. Logged every episode, never used as reward.
- RL stays in the project only if true success on the train tasks under G_new exceeds 0.8 within 2,000 episodes.

## Pilot rule (Tue 10 Nov, 16:00 ET)

Continue if H1(a) holds and at least one of H1(b), H1(c) holds, and the banking suite runs in under 30 minutes. Stop
claiming the method if the base run is not working, or H1(a) fails; the fallback is a short written note. A miss is
reported as a miss.

## Amendments

### 1. Validity filters and controls (9 Nov 2026, before any relation run)

Reason: a review of the method before the build found variants whose expected verdict was not justified, and no
control that would expose a broken harness.

- **F3 counts only effective writes.** A dropped write whose removal leaves the live end state (agent and user DB
  hash) the same as the gold run's has nothing for the grader to catch; such variants are reported as "no-effect
  write" with their count and are not F3 variants. (Retail's gold has 18 calls that raise an error.)
- **P5 must be inert on the live env.** A P5 variant counts only if the agent and user DB hashes right after the
  inserted call equal those right before it; otherwise it is reported as "not inert".
- **Negative control.** τ²-bench's changelog says grading outside banking_knowledge did not change in 1.0.1. On
  airline and retail, every variant's verdict must be identical under G_old and G_new. Any difference is a harness
  fault: the run is invalid until it is explained.
- **Changed task data counted apart.** Tasks whose gold run differs between the two versions (for example task_074
  and tasks 077-086) are listed separately and left out of the H1 counts, because a verdict change there may come
  from the data, not the grader.
- **F1 split.** F1 results are reported separately for tasks whose gold run changes the DB and tasks whose gold run
  does not (where an end-state grader cannot tell an empty run apart).
- **Reproducibility.** Each result file records the tau2 commit, the `uv.lock` hash, the Python version and the
  seed. The suite is run twice per version; the two runs must be identical apart from timings.

### 2. Errors, per-case evidence and fixture identity (9 Nov 2026, before any relation run)

Reason: an independent review of amendment 1 found that a grading or replay exception would have been scored as
reward 0 (a false flag), that a totals-only repeatability check can hide changes that cancel out, and that the
cross-version control needs a stated test of "identical fixture". Checking the data showed that `v1.0.1` edits
banking's shared `db.json` (answer-leaking transaction descriptions removed, two rows swapped for #403), so the test
must be per task, not per file.

- **Errors are outcomes, not verdicts.** Each variant (and each base run) ends as either a verdict (reward 0 or 1) or
  an error (the grader or replay raised), recorded with its exception type and first line. Errors are never counted
  as rewards or flags and never dropped: every count reports attempted = verdicts + errors, per domain and relation.
  A tool call that the live environment answers with an error output is a recorded fact about that call, not an
  error outcome.
- **Fixture identity, per task.** A task is comparable across versions only if (a) its definition is identical
  (canonical JSON of the whole task) and (b) the gold run's tool outputs are identical once JSON is parsed and every
  number is read as a float (so `25` and `25.0` match; #397 changes number echoes on purpose). Tasks that fail
  either part are listed as "fixture changed", with the part that differed, and are left out of H1 and the control.
  They are still run and reported per version.
- **Per-case evidence.** Every variant record keeps its exact calls, its validity checks (P1 inserted call ran
  cleanly; P5 inert; F3 end state changed), the live call errors, the end-state hashes of the variant and the gold
  run, the outcome, reward, flag and DB diff. Each result file records the tau2 source, `uv.lock` hash, Python
  version, seed, and the task's `reward_basis`.
- **Per-case repeatability.** Records are serialised deterministically (sorted keys, no whitespace, sorted by
  domain, task, relation and detail; timings left out) and hashed with SHA-256. The full hash is kept with the
  evidence; a 48-bit integer prefix goes into the numbers-only metrics, so the required two identical runs cover
  every case, not only totals.
- **Independent check.** The aggregate metrics are recomputed from the per-case records by a check that does not use
  the project's own `compare` code, before any result is reported.
- **Protocol frozen.** After this amendment the protocol changes only on new evidence, through a further dated
  amendment that states that evidence.
