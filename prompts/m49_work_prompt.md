# M4.9 Work Prompt - Representation-Level Recall Reconstruction

## Goal

Replace string-assembled recall with a reconstruction artifact that directly perturbs predictive-coding priors and measurably changes the next decision.

## Context

M4.8 proved that memory causally changes behavior. M4.9 raises the bar: recall can no longer be an explanatory string such as `alternative recall via ...`. The retrieval result must become a representational object that:

- reconstructs a state-space prior,
- expresses donor influence as dimension-level injections,
- preserves protected anchors as dimension-level stabilizers,
- carries residual competition forward as measurable interference.

The downstream consequence must be observable in both `predict()` and final action selection.

## Implementation Plan

### 1. Replace text reconstruction with state reconstruction

- Update `build_recall_artifact()` so the primary output is a `reconstructed_state_vector` plus `latent_perturbation`.
- Keep any textual field strictly as audit metadata; no downstream component should depend on prose content.

### 2. Map protected anchors and donor traces into latent dimensions

- Encode protected anchors as `protected_anchor_biases` over concrete state dimensions.
- Encode donor contribution as `donor_injections`, with each record showing source memory, target dimension, and injected amount.
- Express blocked or invalid donors structurally in `donor_blocks`.

### 3. Make competition winner-take-most, not winner-take-all

- Run competition as a softmax over cue-match plus salience and accessibility.
- Promote the top memory to primary recall.
- Preserve losing candidates as `residual_prior` so they remain causally active and testable.

### 4. Feed reconstructed priors into predictive coding

- `SegmentAgent._build_memory_context()` must surface `prior_projection`, `prior_delta`, `residual_prior`, and residual gain.
- `GenerativeWorldModel.predict()` must consume those fields and expose them in `last_prediction_details`.
- A donor-biased recall must numerically shift the next prediction toward donor-aligned dimensions.

### 5. Add a minimal misinformation / DRM-style acceptance test

- Build a target memory and a competing donor memory over the same retrieval cues.
- Control condition: the target memory alone yields the target-aligned action.
- Interference condition: donor wins recall competition, target remains as residual prior, and the final decision shifts toward the donor action.
- Assert both the representational shift and the downstream policy flip.
- Keep isolation harnesses only as mechanism evidence; final acceptance must also include default-path donor/no-donor evidence.

## Constraints

- Do NOT accept string similarity as evidence of successful reconstruction.
- Do NOT implement competition by generating multiple explanation strings.
- Do NOT add a second downstream reasoning path that ignores the reconstructed prior.
- Residual competition must remain observable after primary selection.

## Out of Scope

- Procedural step-outline recall remains out of scope for M4.9 acceptance until it emits non-empty representational fields instead of a parallel text-only branch.
- Until then it must be tagged as `legacy_non_acceptance`, not silently presented as if it were a representational M4.9 path.
- Acceptance cannot be signed off by donor-biased audit strings alone; the effect must survive isolation from legacy decision shortcuts.

## Acceptance Criteria

See `prompts/m49_acceptance_criteria.md`.
