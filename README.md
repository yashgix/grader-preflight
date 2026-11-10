# grader-preflight

An LLM-free preflight for graders of stateful tool environments. It takes a known-good run, makes small changes that
should not matter (an extra read, `25` written as `25.0`, a call to a tool that does not exist) and changes that should
(drop a write, do nothing, verify identity after acting), and checks that the grader's verdict moves the right way.

It is tested on τ²-bench's grader before and after the July 2026 fixes (`pre-v1.0.1` and `v1.0.1`): does it find the
bugs that were later fixed by hand?

**Status:** pre-registered, not yet run. The hypotheses, thresholds and seeds are in
[PREREGISTRATION.md](PREREGISTRATION.md), committed before any experiment.

Built with Claude Code and the Arcturus agent pipeline, reviewed with Codex.
