# M4.8 Work Prompt

Implement a controlled complex-environment milestone that maps benchmark-constrained cognitive parameters into survival-oriented behavior in MiniGrid or an equivalent partially observable grid world.

Requirements:
- Explicitly route cognitive parameters into action scoring for resource gathering, threat avoidance, and information search.
- Add partially reliable cue sources so `source_precision_gain` can be measured behaviorally.
- Add compute or attention-budget constraints to test graceful degradation under scarce processing resources.
- Produce mapping, transfer, ablation, stress, and acceptance artifacts.
- Enforce these acceptance rules:
  - different parameter settings must yield statistically separable behavioral strategies,
  - parameter rankings inferred from Confidence Database and IGT must align with controlled-environment behavior with Spearman rho above 0.6,
  - under tighter compute budgets the DOTTORE agent must degrade selectively rather than uniformly,
  - high and low `source_precision_gain` settings must diverge significantly under both helpful and misleading cues.
- Open-world tool environments are deferred to M5; M4.8 is the bridge milestone that validates cross-context style stability before real tools.
- Regress M4.4 and M3.6 acceptance.
