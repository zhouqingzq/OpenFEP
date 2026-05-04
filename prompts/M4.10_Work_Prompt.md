# M4.10 Work Prompt - Encoding and Consolidation as Dynamics, Not Keyword Tables

## Goal

Move memory encoding and consolidation out of the keyword-table / template-string regime and into a resource-limited, error-driven, vector-space dynamic process. After M4.10, salience must come from FEP prediction error plus an attention budget, and semantic memory must be a geometric abstraction over episodic clusters with replay that actually re-encodes episodes - not a printf template.

## Context

M4.8 and M4.9 established that memory causally changes behavior and that recall is a representational object, not a string. The remaining "not a database" gap lives upstream of recall:

- `segmentum/memory_encoding.py` still decides what gets encoded via hand-coded `THREAT_KEYWORDS`, `SOCIAL_KEYWORDS`, `REWARD_KEYWORDS`, `FIRST_PERSON_TOKENS`, `TASK_ONLY_KEYWORDS`, etc. Salience is keyword overlap in disguise.
- `segmentum/memory_consolidation.py` still manufactures semantic memories through template strings such as `"Semantic skeleton from N episodes: ..."`, `"Inferred pattern from N related memories"`, and `"Replay hypothesis from ..."`. Consolidation is text generation, not dynamics.

M4.10 makes encoding and consolidation behave like a resource-limited predictive-coding system: surprising events get encoded because prediction error and arousal overwhelm a finite attention budget, and semantic memory emerges as the centroid of episodic clusters whose residuals are folded back into reconstruction error. Replay is implemented by actually feeding sampled episodes back through the encoder, producing a second-pass prediction error that modulates their salience and retention.

## Implementation Plan

### 1. Replace keyword-table salience with error-driven encoding strength

- In `segmentum/memory_encoding.py`, introduce an `EncodingDynamics` layer whose output is an `encoding_strength` scalar computed as a function of:
  - `prediction_error` obtained from the FEP layer (`fep.py`'s free-energy / prediction-error diff that the project already exposes),
  - `surprise` (Bayesian surprise on the posterior update from the same FEP step),
  - `arousal` (from interoception / homeostasis),
  - `available_attention_budget` (see step 2).
- `encoding_strength` must be strictly a function of these dynamical signals; it must not read any keyword set.
- Retire `THREAT_KEYWORDS`, `SOCIAL_KEYWORDS`, `REWARD_KEYWORDS`, `OUTCOME_POSITIVE_KEYWORDS`, and `OUTCOME_NEGATIVE_KEYWORDS` as primary salience sources. Any remaining reference to these sets must be behind a `FallbackHeuristic` shim that only fires when the dynamical path is unavailable, and every entry it emits must be tagged `encoding_source="heuristic"` in the audit log so the heuristic share is auditable.
- The dynamical path must emit `encoding_source="dynamics"` and must be the default for all production ticks.

### 2. Finite attention budget with competition across events

- Each tick gets a finite `attention_budget` (per-tick scalar, sourced from attention / homeostasis state).
- Candidate events in the same tick compete for budget in proportion to their raw `(prediction_error, surprise, arousal)` product.
- When the budget is exhausted, low-salience events are dropped (not encoded) or encoded at a reduced fidelity (`encoding_strength` shrunk toward zero). This must naturally reproduce the "flat events forgotten, surprising events retained" curve without any keyword lookup.
- The per-tick budget bookkeeping (requested, granted, denied) must be logged on the episode's encoding audit record.

### 3. First-person / task-only tokens demoted to labelled fallback

- `FIRST_PERSON_TOKENS` and `TASK_ONLY_KEYWORDS` may remain in the file, but only as a fallback heuristic used when `EncodingDynamics` cannot score an input (for example, missing FEP signals during a bootstrap tick).
- Every invocation of this fallback path must set `encoding_source="heuristic"` and must be counted in a per-run histogram. Acceptance will check that heuristic usage is below a hard ceiling on the default path.

### 4. Semantic memory as centroid + variance shrinkage, not a template string

- In `segmentum/memory_consolidation.py`, replace the two template-string generators (`"Semantic skeleton from {N} episodes: ..."` at line 524 and `"Inferred pattern from {support_count} related memories"` at line 600) with a real vector-space construction:
  - Collect the embedding/state vectors of the clustered episodes.
  - Compute the cluster centroid `c` and the residuals `r_i = x_i - c`.
  - Store the semantic memory as `{centroid: c, residual_norm_mean: ..., residual_norm_var: ..., support_ids: [...]}`.
  - The residuals must be folded back into the episodes as a reconstruction-error field, so that each source episode knows how well its cluster's semantic summary reconstructs it.
- Text content on the semantic memory, if present at all, must be audit metadata only. No downstream component (recall, prediction, decision) may read it.
- Consolidation must therefore be differentiable in principle - centroid / residual is a real vector operation - not a `f"..."` format string.

### 5. Replay as genuine re-encoding

- Replace the `"Replay hypothesis from ..."` path (around line 664) with a replay routine that, during offline ticks, samples episodes, re-feeds them through the same `EncodingDynamics` used at step 1, and records:
  - `replay_prediction_error_second_pass`,
  - a `salience_delta` equal to the difference between first-pass and second-pass encoding strength,
  - the resulting `retention_adjustment` applied to the episode and, if any, to its parent semantic cluster.
- Replay must therefore change memory state through the same dynamical operator that originally encoded the episode. No replay path may construct its effect from a string.
- Replay's effect on semantic memory must route through the centroid / residual update from step 4, not through a second template string.

### 6. Audit fields and acceptance logging

- Every encoded episode must carry:
  - `encoding_source` in `{"dynamics", "heuristic"}`,
  - `encoding_strength`,
  - `fep_prediction_error`, `surprise`, `arousal`, `attention_budget_granted`,
  - for replay-touched episodes: `replay_second_pass_error`, `salience_delta`.
- Every semantic memory must carry:
  - `centroid`, `residual_norm_mean`, `residual_norm_var`, `support_ids`,
  - `consolidation_source="dynamics"`.
- These fields are what the acceptance gates read. They must be populated on the default path (normal agent tick), not only in isolated harnesses.

## Constraints

- Do NOT rely on keyword-table overlap as a primary salience signal on the default path.
- Do NOT leave template-string semantic memories on the default path; they are acceptable only as explicitly tagged `legacy_non_acceptance` records during migration, and must not be produced by normal consolidation after M4.10.
- Do NOT introduce a parallel text-only replay path that bypasses re-encoding.
- Do NOT sign off M4.10 on mechanism-only harnesses; acceptance must show the dynamical path active during a normal agent run.

## Out of Scope

- Learning an end-to-end differentiable encoder on top of FEP. M4.10 only requires that the encoding pathway be a function of dynamical signals and that consolidation operate in vector space. Full gradient-based training of the encoder is future work.
- Replacing the episodic store's on-disk schema. Only the content of the `encoding_source`, `encoding_strength`, `centroid`, and related fields changes; the store layout can remain.

## Acceptance Criteria

See `prompts/m410_acceptance_criteria.md`.
