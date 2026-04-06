# M4.8 Milestone Specification

## Title

`M4.8: Controlled Complex Environment Validation`

## Scope

- Replace direct open-world tooling with a controlled but behaviorally rich environment such as MiniGrid or an equivalent partially observable grid world.
- Cover resource gathering, threat avoidance, and information search under one survival-oriented task family.
- Add controllable cue sources so source-trust parameters can be tested under partially reliable hints.
- Add explicit compute or attention-budget constraints to measure graceful degradation.
- Produce environment mapping, ablation, stress, transfer, and acceptance artifacts.

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`
- `artifact_freshness`
- `parameter_projection`
- `style_observability`
- `cross_context_rank_consistency`
- `budgeted_graceful_degradation`
- `source_cue_sensitivity`
