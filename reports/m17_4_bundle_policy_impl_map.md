# M17.4 Bundle Policy Implementation Map

## Purpose

This note translates the M17.4 milestone contract into concrete codebase
landing points. The goal is to avoid a vague "add bundle reasoning somewhere"
implementation that quietly collapses back into single-item top-k behavior.

The hard requirement remains:

```text
at least one bounded downstream decision must be bundle-required
AND every single member must remain below the single-trigger threshold
AND the best-single counterfactual must be audited as non-triggering
```

## Existing Landing Points

### 1. Coarse retrieval surface

Current row-level retrieval already exists and should remain the stage-1 owner:

- `segmentum/dialogue/runtime/m14_7_recall_scoring.py`
  - `score_recall_candidate(...)`
- `segmentum/dialogue/runtime/mvp_loop.py`
  - `lexical_recall_short_term_candidates(...)`
  - `retrieve_memories_for_guidance(...)`

Current behavior:

- row-level lexical/precision/recency scoring exists
- retrieval outputs evidence cards
- no bundle owner exists
- no best-single counterfactual is computed

### 2. Memory gate

The existing deterministic write gate is the correct place to host M17.2
bundle sidecar fields, but not the place to perform free-form grouping:

- `segmentum/dialogue/runtime/m14_7_memory_gate.py`
  - `MemoryWriteIntent`
  - `MemoryGateDecision`
  - `MemoryGate.evaluate(...)`
  - `memory_gate_event(...)`

Current behavior:

- deterministic item-level write score
- item-level factor audit
- no bundle support fields

### 3. Memory-EFE consumer

The lightest existing reply-policy consumer is:

- `segmentum/dialogue/runtime/m13_memory_efe.py`
  - `normalize_expectations_for_efe(...)`
  - `evaluate_memory_efe(...)`
  - `_choose_reply_angle_bias(...)`
  - `merge_memory_efe_guidance_into_control(...)`

Current behavior:

- traceable expectation filtering already depends on `bound_memory_ids` and
  `evidence_refs`
- reply-angle bias already exists as a bounded downstream signal
- this is the best first place to consume a compact bundle summary without
  exposing raw bundle internals to prompts

### 4. Consolidation consumer

The best first memory-side consumer is:

- `segmentum/dialogue/runtime/m15_consolidation.py`
  - `_merge_rows(...)`
  - `_promote_stm(...)`
  - `_abstract_paths(...)`

Current behavior:

- deterministic duplicate merge exists
- path abstraction already aggregates multiple episodes
- no explicit bundle-required gate exists

### 5. Runtime wiring

The orchestration owner is:

- `segmentum/dialogue/runtime/mvp_loop.py`
  - retrieval call sites
  - `_evaluate_memory_gate(...)`
  - `_run_consolidation_cycle(...)`
  - memory-EFE evaluation flow

This is where bundle assembly should be called and where bundle audit events
should be appended.

## Recommended New Modules

Create:

```text
segmentum/dialogue/runtime/m17_bundle_features.py
segmentum/dialogue/runtime/m17_bundle_policy.py
tests/test_m17_4_bundle_policy.py
fixtures/m17_4/**
```

Recommended split:

- `m17_bundle_features.py`
  - small dataclasses
  - normalization helpers
  - support aggregation math
  - redundancy / contradiction penalties

- `m17_bundle_policy.py`
  - bounded grouping
  - bundle decision evaluation
  - best-single counterfactual comparison
  - audit event builders

## Phase Order

### Phase 1. Expose scored row features without changing behavior

Goal:

```text
keep current retrieval outputs intact
but expose deterministic row features that a bundle owner can consume
```

Changes:

1. Extend `m14_7_recall_scoring.py` with a compact structure, for example:

```text
ScoredRecallCandidate
  memory_id
  score
  lexical_overlap_norm
  salience_factor
  precision_factor
  recency_factor
  value_factor
  evidence_refs
```

2. Add helper:

```text
explain_recall_candidate(...)
```

3. Update `retrieve_memories_for_guidance(...)` and
   `lexical_recall_short_term_candidates(...)` to optionally attach bundle-safe
   metadata on evidence cards, for example:

```text
_m17_item_support
_m17_evidence_refs
_m17_bound_memory_ids
```

Cut line:

- do not change ranking order in this phase
- do not change prompt-visible memory text

### Phase 2. Add bundle math in observe-only mode

Goal:

```text
compute bundle summaries and best-single counterfactuals
without changing downstream decisions yet
```

Changes in `m17_bundle_features.py`:

- `ScoredMemoryEvidence`
- `MemoryEvidenceBundle`
- `aggregate_memory_bundle_support(...)`
- `redundancy_penalty(...)`
- `contradiction_penalty(...)`

Changes in `m17_bundle_policy.py`:

- `assemble_memory_evidence_bundles(...)`
- `best_single_counterfactual(...)`
- `evaluate_bundle_decision(...)`
- `bundle_decision_event(...)`

Bundle assembly should only use structured linkage:

- `prediction_id`
- `expectation_id`
- `episode_id`
- `bound_memory_ids`
- `evidence_refs`

Not allowed:

- raw-text semantic clustering
- LLM-generated grouping
- duplicate evidence inflation

Observe-only output should include:

```text
bundle_id
member_memory_ids
aggregated_support
max_single_support
synergy_margin
bundle_required
best_single_counterfactual_would_trigger
```

### Phase 3. Wire first downstream consumers

Recommended first consumer order:

1. `reply_policy_bias`
2. `memory_consolidation_candidate`
3. `memory_revision_candidate`

Reason:

- `reply_policy_bias` already has a bounded advisory surface in
  `m13_memory_efe.py`
- `consolidation` already groups evidence deterministically
- `revision` is strongest semantically and should come after the audit path is
  proven

#### 3A. Reply policy bias

Add a compact optional bundle input to `evaluate_memory_efe(...)`, for example:

```text
bundle_summaries: list[MemoryEvidenceBundle]
```

Use only compact fields to bias:

- `reply_angle_bias`
- `reason_codes`
- `evidence_refs`

Do not pass raw member texts into prompts.

Recommended first rule:

```text
if a bundle_required repair-oriented bundle exists for the traceable expectation
then allow a bounded shift toward clarify_or_repair bias
only when best-single counterfactual would not trigger
```

#### 3B. Consolidation candidate

Use bundle summaries inside `m15_consolidation.py` to decide whether multiple
 weak rows should be consolidated because they jointly support a stable pattern.

Best landing points:

- `_promote_stm(...)`
- `_abstract_paths(...)`

Recommended first rule:

```text
promote/abstract only when bundle_required is true
or when the path is explicitly labeled top-k-equivalent and treated as legacy
```

### Phase 4. Runtime audit wiring

`mvp_loop.py` should own:

- calling the bundle policy after coarse retrieval
- storing recent bundle audit events
- appending bundle decisions to the same audit flow as M14.7 and M15 events

Recommended event tail:

```text
state["bundle_policy_audit_tail"]
```

Recommended state diagnostics:

```text
state["bundle_policy_linkage_diagnostics"]
  retrieval_eligible_count
  bundle_linkable_count
  unlinked_count
```

Recommended event types:

- `BundleDecisionEvent`
- `BundleDecisionSuppressedEvent`

Minimum event fields:

```text
bundle_id
member_memory_ids
aggregated_support
max_single_support
synergy_margin
best_single_memory_id
best_single_support
best_single_counterfactual_would_trigger
consumer_kind
reason_codes
```

## Minimal Viable First Implementation

If we want the smallest change that can still honestly claim non-top-k behavior,
build this exact vertical slice:

1. expose row-level support features from M14.7 retrieval
2. assemble bundles in observe-only mode
3. compute best-single counterfactual
4. wire bundle summaries into `m13_memory_efe.evaluate_memory_efe(...)`
5. allow only `reply_policy_bias` to change when `bundle_required == true`
6. audit every changed decision with best-single comparison
7. compare it in replay against both `best_single_memory_baseline` and
   `naive_additive_topn_baseline`
8. fail the claim if linkage diagnostics show the feature is simply never
   entering bundle-linkable states

This is smaller than trying to change write gate, consolidation, and revision
all at once.

## Test Map

### New focused unit tests

Add to `tests/test_m17_4_bundle_policy.py`:

```text
test_bundle_requires_multiple_unique_memories
test_bundle_rejects_duplicate_evidence_inflation
test_bundle_rejects_when_best_single_would_trigger
test_bundle_synergy_margin_must_exceed_floor
test_bundle_contradiction_penalty_suppresses_conflicting_members
test_bundle_decision_emits_counterfactual_audit_fields
```

### Existing suites to extend

- `tests/test_m14_7_memory_gate_decay_recall.py`
  - add factor-breakdown coverage
  - add bundle-safe row feature coverage

- `tests/test_mvp_dialogue_runtime.py`
  - add a small vertical slice proving reply policy changes only under
    `bundle_required`

- future `tests/test_m17_3_replay_harness.py`
  - compare bundle policy vs best-single baseline on same fixture

## Fixture Shape

The first acceptance fixture should be intentionally narrow:

```text
memory A: weak evidence for "user wants implementation detail"
memory B: weak evidence for "user dislikes abstract-only replies"
memory C: weak evidence for "same technical topic continuity"
```

Requirements:

- each memory alone must stay below the single trigger threshold
- the bundle must cross the bundle threshold
- removing any one member should break the bundle-required condition
- best-single counterfactual must not trigger
- naive additive top-N without structured penalties should also fail
- reply-policy bias should change only in the full bundle condition

## Cut Lines

Stop the implementation and label it incomplete if any of these happen:

1. bundle assembly depends on free-text clustering instead of structured links
2. duplicate evidence refs can push a bundle over threshold
3. a bundle-triggered decision still fires under the best-single counterfactual
4. replay shows `bundle_required_decision_rate == 0`
5. the only observed gains come from item ranking or larger top-N retrieval

## Recommendation

Build `reply_policy_bias` first, not memory revision.

That path is:

- already bounded
- already auditable
- easier to compare in replay
- less likely to contaminate memory state while the anti-top-k contract is still
  being proven

Once replay shows real `bundle_required` wins over `best_single_memory_baseline`,
expand the same bundle owner to consolidation and memory revision.
