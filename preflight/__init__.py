"""LLM-free preflight: edited gold runs checked against a live tool environment's own grader.

Phase 2 adds an open-loop MDP over banking tool calls and a small REINFORCE learner over it. Phase 3 audits agent
trajectories in tau2's own results format, on synthetic runs built from tau2's gold actions, whose answers are known.
"""
