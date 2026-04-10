# M4.8 to M4.9 Boundary Update - Representation-Level Reconstruction

## Goal

Carry M4.8's causal-memory result forward into M4.9 by making recall a representation-level reconstruction replacing string assembly.

## Boundary

- M4.8 established that memory can change behavior.
- M4.9 raises the acceptance bar: the recalled artifact must perturb predictive-coding priors through `reconstructed_state_vector`, `prior_projection`, `prior_delta`, and `residual_prior`.
- `content` and `competing_interpretations` remain audit metadata only; they are not valid evidence of successful reconstruction.
- Old decision helpers such as `memory_bias`, `pattern_bias`, and action-specific rollups are not M4.9 acceptance evidence unless the representational pathway is shown to remain causal when those helpers are isolated away.

## Acceptance Direction

- Acceptance must be framed in terms of prior perturbation, residual interference, and donor-biased downstream behavior.
- Gate 4 and Gate 5 evidence must be taken from final `prediction_after_memory` and downstream action ranking, not only intermediate `prior_projection`.
- A/B misinformation-style tests should include a causal ablation where the representational prior path is disabled; if the same policy flip remains, the result does not count as M4.9 evidence.

## Out Of Scope

- Procedural step-outline recall remains legacy retrieval behavior for now and is not M4.9 acceptance evidence.
- String similarity, audit prose, or alternative explanation text do not satisfy the milestone.

## Status

M4.9 remains `NOT_ISSUED` until all gates in `prompts/m49_acceptance_criteria.md` pass.
