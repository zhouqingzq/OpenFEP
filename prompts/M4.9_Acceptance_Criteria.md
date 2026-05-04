# M4.9 Acceptance Criteria - Representation-Level Recall Reconstruction

## Gate 1: Recall artifact is representational, not explanatory text

- `build_recall_artifact()` emits `reconstructed_state_vector` and `latent_perturbation`.
- The artifact also carries `residual_prior`, `winner_take_most_weights`, `protected_anchor_biases`, and `donor_injections`.
- `content` may remain as audit metadata, but downstream logic must not depend on free-form explanation strings.

## Gate 2: Donor and protected anchors operate as dimensional bias injections

- `donor_blocks` and `protected_anchors` are expressed as dimension-level injections or protections, not sentence fragments.
- `protected_anchor_biases` pull the reconstructed prior back toward protected dimensions of the winning memory.
- `donor_injections` expose which donor or residual source moved which dimension and by how much.

## Gate 3: Competing memories use winner-take-most with measurable residual interference

- Recall competition is implemented as a softmax over cue-match plus salience and accessibility, not by generating alternate explanation strings.
- The top memory becomes the primary recall source.
- Losers remain in `residual_prior` with nonzero residual weight and produce observable downstream interference.

## Gate 4: Predictive coding prior is directly perturbed by recall

- The next `predict()` call consumes `prior_projection` and `prior_delta` derived from recall.
- `world_model.last_prediction_details` exposes `prior_projection`, `prior_delta`, and `residual_prior`.
- If donor competition wins, the predicted state shifts toward donor-aligned dimensions in a way that can be asserted numerically.

## Gate 5: Minimal DRM / misinformation reproduction exists

- Construct an A/B interference pair with one target memory and one donor memory.
- Control condition: only the target memory is available, recall stays target-aligned, and the downstream decision follows the target direction.
- Interference condition: donor salience, cue-match, or recency wins recall competition, target remains as residual prior, and the downstream decision shifts toward the donor action.
- Mechanism evidence from isolation harnesses is allowed as supporting evidence, but it does not sign off M4.9 on its own.
- Default-path acceptance evidence must show the donor/no-donor contrast under normal decision settings and must attribute the final policy change to representational recall rather than to legacy helper terms alone.

## Gate 6: Acceptance prompts reflect the new milestone boundary

- [prompts/m48_work_prompt.md](/E:/workspace/segments/prompts/m48_work_prompt.md) describes M4.9 as representation-level reconstruction replacing string assembly.
- [prompts/m49_acceptance_criteria.md](/E:/workspace/segments/prompts/m49_acceptance_criteria.md) defines acceptance in terms of prior perturbation, residual interference, and donor-biased downstream behavior.
- Procedural recall must be explicitly marked as `legacy_non_acceptance` until it emits genuine representational artifacts.

## Not in scope

- No reliance on string similarity as evidence of successful reconstruction.
- No acceptance based only on audit snapshots without a causal downstream effect.
- No second explanatory text branch standing in for memory competition.
- No milestone sign-off based only on experimental or mechanism-only evidence when default-path acceptance is still missing.

## 2026-04-10 Status Alignment

- M4.9 acceptance remains `NOT_ISSUED` until all six gates pass.
