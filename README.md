# grader-preflight

Can you catch mistakes in the grader that scores an AI agent before anyone trains or ranks agents with it?

**Write-up: [yashgix.github.io/grader-preflight](https://yashgix.github.io/grader-preflight/)** (the same page is in
[`docs/index.html`](docs/index.html))

Agents that handle customer requests, write code or run workflows are trained and ranked by automatic graders: code
that decides whether a task was done. In [τ²-bench](https://github.com/sierra-research/tau2-bench), a widely used
benchmark for customer-service agents, the agent calls tools that change a database, and the grader compares the final
database with the one a known-correct run produces. When that grader is wrong, training rewards the wrong behaviour and
leaderboards rank agents wrongly.

grader-preflight tests a grader with no AI model in the loop. It takes each task's known-correct run, makes small
changes whose right verdict is known (an extra lookup should still pass; a dropped write should fail), runs them
through the real environment and the unchanged grader, and flags every wrong verdict. It was checked three ways, each
phase pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) before its data was run. One of the five
pre-registered hypotheses held; the other four are reported as not held.

## Results

| Phase | Question | Result |
|---|---|---|
| 1 | Does it find the banking grading bugs τ²-bench's authors fixed in v1.0.1? | Yes: an extra lookup was wrongly failed on 43 of 62 comparable tasks before the fix, 0 after; a respelled number on 25 of 33, then 0. Controls: 0 mismatches. |
| 2 | Does such a bug change what a learning agent does? | Not as pre-registered: H3 and H4 failed. Exploratory: greedy policies trained on the buggy grader made 1.8 fewer checks after finishing the task (95% interval 0.8–2.7); 3 of 10 buggy-grader seeds collapsed (true success 19–47%), so mean true success fell 21 points (interval 0.4–43). |
| 3 | Does it explain real grading errors in published frontier-agent runs? | Not fully: H5 failed. The old grader would wrongly fail 118 runs by GPT-5.5, Claude Opus 4.7, Claude Opus 5 and Qwen3.8-max (counterfactual: the runs were made on the fixed version); phase 1's flags cover 87 (recall 0.74, 95% interval 0.48–0.94), short of the pre-registered 0.90. All 31 misses are the same bug, on 6 tasks where none of phase 1's five sampled extra reads used the tool these agents read. |

**New finding: the fixed grader can't see interest-discrepancy reports.** The report tool writes to a database table
that doesn't exist and reports success anyway, so the grader only sees that the tool was used: not what the reports
say, how many were filed, or for which account. A run whose reports carry wrong interest rates, or that files one of
two required reports, still scores full marks. This affects 5 banking tasks; in the published runs, 3 of the 41
full-score runs on them filed wrong numbers (one then filed correct reports as well). It reproduces on both versions
tested, and v2.0.0's tool still writes to the undefined table (from reading its source; not run). I found no report
of it in τ²-bench's issues or changelog (searched 11 Nov 2026).

```console
$ .tau2/v1.0.1/venv/bin/python examples/report_blind_spot.py
gold run:                          reward 1.0
same run, wrong report numbers:    reward 1.0  (DB tables that differ from gold: none)
same run, second report left out:  reward 1.0
same run, both reports left out:   reward 0.0
```

The phase 1 defects are Sierra's, fixed in their pull requests #329 and #397; everything else was found here. Phase
3's check for hidden mistakes in conversations used two language-model annotators, with disagreements settled by a
third model pass, and is model-assisted, not human-validated; the blind-spot evidence above does not depend on it.

## Run it

Needs git, [uv](https://docs.astral.sh/uv/) and network access for the first setup.

```sh
make setup      # builds .tau2/pre-v1.0.1 and .tau2/v1.0.1 from τ²-bench's own lockfile, about a minute
make check      # lint, format and the full test suite under both grader versions, as CI runs them
make example    # the blind spot, in a second
make phase1     # phase 1 on both versions, a few minutes
make phase3     # the phase 3 pipeline on synthetic runs built from gold actions
```

Phase 1's metrics should match [`results/phase1.json`](results/phase1.json). Phase 2's full sweep
(`experiment.py --full`) takes about 1.5 hours on 8 cores. Phase 3 on published runs needs τ²-bench's leaderboard
trajectory files, downloaded separately; the commands are in [`docs/DESIGN.md`](docs/DESIGN.md#phase-3) and their
aggregate output is [`results/phase3.json`](results/phase3.json). To use other interpreters, set `TAU2_OLD_PYTHON`
and `TAU2_NEW_PYTHON` (or `make OLD=… NEW=…`).

## How it works

- `preflight/tau2_adapter.py` runs tool calls on a live τ²-bench environment and scores them with τ²-bench's own,
  unchanged grader. Each grader version lives in its own environment; one process loads one version and results are
  joined afterwards.
- `preflight/relations.py` builds the six changes (P1, P3, P5 should pass; F1, F3, F6 should fail); `suite.py` and
  `compare.py` run and compare them (phase 1).
- `mdp.py`, `rl.py`, `phase2.py` and `full.py` train small REINFORCE policies on real tool calls, rewarded by one grader
  version (phase 2).
- `trajectories.py` and `trajectory_analysis.py` replay published runs, regrade them under both versions and check them
  against the phase 1 relations (phase 3).
- `scripts/collect_results.py` writes the aggregate results; `scripts/build_site.py` builds the write-up from them, and
  a test fails if the page and the data disagree.

Design, data flow and known limits: [`docs/DESIGN.md`](docs/DESIGN.md).

## Limits

Phase 3's old-grader verdicts are counterfactual: the published runs were made on the fixed version, and 22–30% of
them can't be replayed on the old code. Four high-scoring agents, one domain, and only the 1.x grader is run. Phase 2's
policy doesn't see tool outputs, so it shows how a reward flaw shapes learning, not how a language model would behave.
The conversation annotation is small (24 runs) and model-assisted.

## About

An independent research project by Yash Gupta, not affiliated with Sierra. τ²-bench is MIT-licensed and is installed
from its official repository, not copied here. The published leaderboard runs are used for aggregate analysis only and
are not redistributed: their licence is not stated (τ²-bench issue #612).

Shared to show the work; not licensed for reuse.
