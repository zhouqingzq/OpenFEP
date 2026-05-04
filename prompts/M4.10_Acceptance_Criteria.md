# M4.10 Acceptance Criteria - Encoding and Consolidation as Dynamics

## Gate 1: Encoding strength is a function of dynamical signals, not keyword overlap

- `segmentum/memory_encoding.py` exposes an `EncodingDynamics` entry point whose `encoding_strength` is computed from `prediction_error`, `surprise`, `arousal`, and `available_attention_budget`.
- No primary code path reads `THREAT_KEYWORDS`, `SOCIAL_KEYWORDS`, `REWARD_KEYWORDS`, `OUTCOME_POSITIVE_KEYWORDS`, or `OUTCOME_NEGATIVE_KEYWORDS` to compute salience on the default agent tick.
- A static check (grep or AST walk) shows that all remaining references to those keyword sets sit inside a `FallbackHeuristic` shim and are guarded by an `encoding_source="heuristic"` tag.

## Gate 2: Attention budget produces human-like retention curves without keyword lookup

- Each tick records `attention_budget_total`, `attention_budget_granted_per_event`, and `attention_budget_denied_per_event` in the episode encoding audit.
- In a mixed-salience run where several low-error and a few high-error events arrive in the same tick, the high-error events must be encoded (`encoding_strength` > threshold) and the low-error events must be either dropped or encoded at reduced strength purely because of budget competition.
- A control run with the budget raised to "unlimited" must encode all of them, proving the selectivity in the normal run came from budget dynamics, not from keyword filters.

## Gate 3: First-person / task-only heuristics are fallback-only and auditable

- `FIRST_PERSON_TOKENS` and `TASK_ONLY_KEYWORDS` are only consulted through the fallback shim.
- Every event they score carries `encoding_source="heuristic"` in its audit record.
- On a default agent run of at least N ticks (N defined by the harness, not zero), the fraction of encoded episodes with `encoding_source="heuristic"` must be at or below a configured ceiling, and the remainder must carry `encoding_source="dynamics"`.

## Gate 4: Semantic memories are centroids + variance, not template strings

- `segmentum/memory_consolidation.py` no longer emits `"Semantic skeleton from N episodes: ..."` or `"Inferred pattern from N related memories"` on the default consolidation path. Any remaining occurrence must be explicitly tagged `legacy_non_acceptance` and must not be produced by normal consolidation.
- Every semantic memory produced on the default path exposes `centroid`, `residual_norm_mean`, `residual_norm_var`, `support_ids`, and `consolidation_source="dynamics"`.
- Each source episode linked to a semantic memory exposes a reconstruction-error field (`semantic_reconstruction_error`) derived from `x_i - centroid`.
- Downstream recall, prediction, and decision code reads `centroid` / residual fields, not the semantic memory's text content.

## Gate 5: Replay is re-encoding, not string generation

- `"Replay hypothesis from ..."` is removed from the default replay path. Any remaining occurrence must be explicitly tagged `legacy_non_acceptance`.
- Replay ticks sample episodes and route them back through the same `EncodingDynamics` used at initial encoding, producing `replay_second_pass_error` and `salience_delta` on the episode's audit record.
- Replay's effect on a parent semantic memory must be observable as an update of `centroid`, `residual_norm_mean`, or `residual_norm_var` - not as a new text field.
- An acceptance test must show that running replay on a cluster with artificially drifted episodes shifts the semantic memory's centroid measurably toward the drifted episodes and leaves a corresponding salience_delta trail on those episodes.

## Gate 6: Default-path evidence, not only harnesses

- Acceptance evidence must include a normal agent run (default path, not a mechanism-only harness) that produces episodes tagged `encoding_source="dynamics"` and semantic memories tagged `consolidation_source="dynamics"`.
- Mechanism-only harnesses are allowed as supporting evidence for Gates 2 and 5, but they do not sign off M4.10 on their own.
- A heuristic-share histogram from the default run must be attached to the acceptance report.

## Gate 7: Milestone documents reflect the new boundary

- `prompts/m410_work_prompt.md` describes M4.10 as the move from keyword tables / template strings to resource-limited, error-driven dynamics.
- `prompts/m410_acceptance_criteria.md` (this file) defines acceptance in terms of encoding_source tagging, attention budget competition, centroid-based semantic memory, and real re-encoding replay.
- Any section of earlier milestone documents that described encoding or consolidation in keyword / template terms must be explicitly marked as superseded by M4.10.

## Not in scope

- No acceptance based on keyword-table overlap, however cleverly reweighted.
- No acceptance based on inspection of template strings like "Semantic skeleton from ..." or "Replay hypothesis ...".
- No acceptance from isolated harnesses alone when the default agent tick has not exercised the dynamical path.
- No sign-off if more than the configured ceiling of default-path episodes are `encoding_source="heuristic"`.

## 2026-04-11 Status Alignment

- M4.10 acceptance remains `NOT_ISSUED` until all seven gates pass.
