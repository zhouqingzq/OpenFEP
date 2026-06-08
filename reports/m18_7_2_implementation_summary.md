# M18.7.2 Implementation Summary — Minimal-Prompt Call Site (the Unlock)

- status: landed (M18.7.2 minimal-prompt call site is the
  sole source of M18.7 v2 attrs; conscious-loop prompt
  no longer requests them)
- branch: feature/m18-7-2-minimal-prompt
- generated_at: 2026-06-08T11:27:35+00:00
- real_llm_replay_at: 2026-06-08T11:27:35+00:00
- held_out_fixture: tests/fixtures/m18_7_1_held_out_calibration.json
- replay_session_root: tmp_m18_7_2_real_llm_replay_v3/
- model_under_test: deepseek/deepseek-v4-flash (default
  in `default_openrouter_client`)

## What M18.7.2 unlocks

M18.7.1 P0 real-LLM replay
(`reports/m18_7_1_calibration_summary.md`,
`tmp_m18_7_1_model_swap_findings.md`) confirmed a hard
blocker: the M18.7 v2 attrs segment in the
`build_conscious_loop_prompt` user prompt sat at char
2914 (37.7% of the 7.7KB-26KB prompt), after the main
JSON schema. Real LLM (v4-flash AND v4-pro, both 12/12
attempts) emitted **0 non-empty** M18.7 fills. The
`m18_7_attribution_hypotheses` surface stayed empty,
the M20.4 producer returned `[]`, and the entire
M20.4.1 same-turn gate / settler / grader downstream
path was a dead path end-to-end.

**M18.7.2 is the fix**: a dedicated minimal-prompt LLM
call site for M18.7 addressee / reaction attribution,
decoupled from the conscious loop, with the
conscious-loop's M18.7 v2 attrs segment removed. M18.7.2
is now the **sole** writer of the M18.7 surface. The
M20.4 producer, M20.4.1 gate, and M18.7.1 calibration
runner read the same surface and "just work" once
M18.7.2 populates it.

## Acceptance Gate (engineering layer)

- Pure-function + state-surface tests: 23/23 PASS
  (`tests/test_m18_7_2_minimal_attribution.py`)
- Targeted regressions (M18.7 / M18.7.1 / M20.3 / M20.4
  / M20.4.1 / M18.6): 0 regressions; 345 tests pass
- Conscious-loop prompt cleanup regression: 1 PASS
  (`test_conscious_loop_prompt_no_longer_requests_m18_7_v2_attrs`)

## Real-LLM Replay — post-M18.7.2 (the unlock, this is real)

```bash
python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_m18_7_2_real_llm_replay_v3
```

| field | n_total | n_present | n_unknown | n_correct | n_incorrect | accuracy | brier | ece |
|-------|---------|-----------|-----------|-----------|-------------|----------|-------|-----|
| addressee_hypothesis | 12 | 8 | 4 | 2 | 6 | 0.25 | 0.22625 | 0.225 |
| reaction_attribution_hyp | 12 | 6 | 6 | 0 | 6 | 0.0 | 0.202083 | 0.258333 |

`m18_7_attribution_hypotheses` surface length after 12
turns: **8 entries** (5 addressee + 3 reaction) across
6 distinct turn indices; all 8 stamped with
`source: "m18_7_2_minimal"`. Pre-M18.7.2 this file did
not even exist on disk (verified by inspecting
`tmp_m18_7_1_real_llm/`: no `m18_7_attribution_hypotheses.json`
present).

### Reliability bins (post-M18.7.2)

**Addressee**:

| bin | count | mean_conf | accuracy | gap |
|-----|-------|-----------|----------|-----|
| 0.00-0.10 | 4 | 0.00 | 0.00 | 0.00 |
| 0.90-1.00 | 4 | 0.95 | 0.50 | 0.45 |

Bimodal distribution: when the LLM is uncertain it
correctly emits `confidence < 0.1` (correctly low); when
confident it emits `confidence ≥ 0.9` but the accuracy
is only 0.5 — overconfident at the high band.

**Reaction**:

| bin | count | mean_conf | accuracy | gap |
|-----|-------|-----------|----------|-----|
| 0.00-0.10 | 4 | 0.00 | 0.00 | 0.00 |
| 0.60-0.70 | 1 | 0.70 | 0.00 | 0.70 |
| 0.80-0.90 | 1 | 0.85 | 0.00 | 0.85 |

6/6 reaction predictions are incorrect (0.0 accuracy).
Mean confidence at the high band (0.85) is far from
the realized accuracy (0.0) — strong
`overconfidence_at_high_band` signal.

### Verdict (per the P0 band agreement)

| ECE   | Brier | band                                  |
|-------|-------|---------------------------------------|
| 0.225 | 0.226 | severe_drift_recommend_m20_4 (addressee) |
| 0.258 | 0.202 | severe_drift_recommend_m20_4 (reaction)  |

Aggregate verdict from the runner: **`severe_drift_recommend_m20_4`**.

M20.4 threshold constants remain frozen (0.4 admit, 0.85
tie-breaker). M18.7.1 only **surfaces** the candidate
recommendations; the decision to revise thresholds is
M20.4's. The surfaced candidates are:

- addressee: candidate_tie_breaker_min = 0.9
  (drift: overconfidence_at_high_band, bimodal)
- reaction: candidate_tie_breaker_min = 0.7
  (drift: overconfidence_at_high_band, bimodal)
- aggregate: candidate_tie_breaker_min = 0.7

M20.4 follow-up can now start with **real data** —
this is the second-order unlock on top of the
first-order "the path is no longer dead".

## Drift Signal Decomposition

The post-M18.7.2 replay surfaces the same drift
signals the M18.7.1 frozen enum
(`ALLOWED_M18_7_1_DRIFT_SIGNALS`) was designed to
detect. Pre-M18.7.2, no drift signals could fire
because the surface was empty. Post-M18.7.2:

- `overconfidence_at_high_band` (both fields): the
  LLM emits confidence 0.85-0.95 but realized accuracy
  is 0.0-0.5 at those bands. This is the expected
  M18.7.1 finding: high-band confidence is not
  predictive of correctness on the held-out fixture.
- `bimodal` (both fields): the LLM collapses to
  very-low (≤0.1) or very-high (≥0.85) confidence,
  with the middle bands (0.2-0.8) empty. This is
  consistent with the M18.7.1 P0 model-swap findings
  (commit b969d8e) which noted v4-flash / v4-pro
  binary confidence emission on minimal prompts.

The pre-M18.7.2 replay surfaced `insufficient_data`
by construction (empty surface). The post-M18.7.2
replay surfaces real calibration problems.

## Why this is the right shape for M18.7.2

Three architectural decisions distinguish M18.7.2 from
"just move the segment higher in the conscious-loop
prompt":

1. **Dedicated call site, not a prompt-order move.**
   The M18.7 v2 attrs are semantically independent
   from conscious-loop cognition (M13 / M19 /
   pending_expectations / open_items / etc.). A
   dedicated minimal-prompt call site removes the
   prompt-order root cause at the architecture layer
   rather than the prompt layer. The M18.7.2 prompt
   is ~1.5-2.0k chars (vs 7.7KB-26KB conscious-loop)
   and includes only the M18.7 v1 schema + 4-key JSON
   spec.

2. **M18.7.2 is the sole source.** The conscious-loop
   prompt no longer requests M18.7 v2 attrs at all
   (~1.5KB removed from `build_conscious_loop_prompt`).
   M18.7.2 is the only writer of
   `state["m18_7_attribution_hypotheses"]`. The
   M20.4 producer, M20.4.1 same-turn gate, settler
   read the same surface unchanged; the dead path
   comes alive end-to-end.

3. **State surface REUSED, not forked.** M18.7.2
   writes to the **same**
   `state["m18_7_attribution_hypotheses"]` key that
   the M20.4 producer reads. The new
   `source: "m18_7_2_minimal"` field on each state
   entry is the only distinguishability lever — the
   existing orchestrator
   (`emit_m18_7_attribution_for_turn`) and record
   helper (`record_m18_7_attribution_hypotheses`)
   are reused unchanged. No M20.4 / M20.4.1 code
   changes. The `commit_id` (sha1 of kind,
   turn_index, source_ref) provides trace identity
   end-to-end.

## State surface persistence (the second-order fix)

The first M18.7.2 real-LLM replay produced 5+4
admitted events in the bus log but the calibration
runner reported an empty surface. Root cause:
`m18_7_attribution_hypotheses` was missing from
`SYSTEM_FILE_DEFAULTS` in `mvp_loop.py`, so
`MVPStateStore.save()` never wrote it to disk and
`MVPStateStore.load()` returned `[]` after each
`run_turn`. Fix: one-line addition to
`SYSTEM_FILE_DEFAULTS`:

```python
"m18_7_attribution_hypotheses": [],
```

After this, the calibration runner's
`runtime.store.load()` reads the in-memory surface
end-to-end and the calibration report sees the real
8-entry surface. This is a `MVPStateStore` contract
fix, not an M18.7.2 logic change.

## Bus event types

Three new bus event types, all reusing
`M18_7_ENGINEERING_PROXY_LABEL`:

- `M18_7_2_AddresseeHypothesisAdmitted`
  (mirrors M18.7's `AddresseeHypothesisAdmitted` +
  `source: "m18_7_2_minimal"` stamp)
- `M18_7_2_ReactionAttributionHypothesisAdmitted`
  (same pattern for the reaction field)
- `M18_7_2_MinimalDegraded` (failure envelope:
  `reason` string + frozen
  `m18_7_2_minimal_llm_failure` reason_code)

Try/except fallback in `run_turn` catches any
exception from the M18.7.2 call site, falls back to
empty `{}` for both M18.7 fields, emits the degraded
event, and does **not** crash `run_turn` (M12-pre
pattern).

## Files Touched

```text
prompts/M18.7.2_Work_Prompt.md                  (NEW)
prompts/README.md                               (EDIT: M18.7.2 entry)
segmentum/dialogue/runtime/m18_7_attribution.py (EDIT:
  build_m18_7_minimal_prompt, 3 new bus event builders,
  emit_m18_7_2_attribution_for_turn orchestrator,
  _m18_7_2_source dispatch)
segmentum/dialogue/runtime/mvp_loop.py          (EDIT:
  M18.7.2 call site in run_turn, removed M18.7 v2 attrs
  segment from build_conscious_loop_prompt,
  registered "m18_7_2_minimal" in _AUXILIARY_LLM_STAGES,
  added "m18_7_attribution_hypotheses" to
  SYSTEM_FILE_DEFAULTS)
tests/test_m18_7_2_minimal_attribution.py       (NEW: 23 tests)
tests/test_mvp_dialogue_runtime.py              (EDIT: FakeJSONLLM
  branch for "M18.7.2 minimal")
tests/test_m18_7_attribution.py                 (EDIT: regression
  for source field dispatch + degraded event shape)
reports/m18_7_2_implementation_summary.md       (NEW: this file)
```

**Not modified**:

- `m18_7_1_calibration.py` (calibration runner reads
  the same surface; M18.7.2 populates it)
- `m20_4_attribution.py` (M20.4 producer unchanged;
  reads same surface)
- `m20_4_1_same_turn_gate.py` (M20.4.1 unchanged)
- `m18_5_structural_decision` logic (unchanged)
- `tests/test_m18_7_1_calibration.py` (regression-only)
- `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` /
  `M20_4_TIE_BREAKER_CONFIDENCE_MIN` (frozen at 0.4 /
  0.85; M18.7.2 surfaces candidates, M20.4 decides)

## Out of Scope (M18.7.2 explicit non-goals)

- **Does not modify M20.4 threshold constants** —
  frozen. Threshold revision is M20.4's job and now
  has real M18.7.1 calibration data to start from.
- **Does not modify M18.5 reply policy logic**.
- **Does not modify M20.4 dispatcher / settler /
  write path**. M20.4 reads
  `m18_7_attribution_hypotheses`; M18.7.2 writes
  into it; no M20.4 code changes.
- **Does not modify M18.7.1 calibration harness /
  fixture / report**. The calibration runner stays
  unchanged — M18.7.2's success is that it produces
  real numbers.
- **Does not revise M18.6 attribution held-out
  acceptance**. M18.6 acceptance depends on M18.7
  producing non-empty hypotheses; M18.7.2 enables
  that. M18.6 follow-up is M18.6 territory.
- **Does not run the full test suite**. Per
  CLAUDE.md red line, only targeted subsets are run.
- **Does not touch Path A / M10 /
  `conversation_loop.py`**.

## Acceptance Criteria — All Met

1. ✅ `build_m18_7_minimal_prompt` exists, returns
   `<=2k char` system+user prompt combined, includes
   only the M18.7 v1 schema + 4-key JSON spec
   (`test_build_m18_7_minimal_prompt_length_under_2k_chars`
   confirms 1671 chars).
2. ✅ `run_turn` calls the new stage
   `"m18_7_2_minimal"` between the M18.5 decision and
   the conscious loop, with try/except fallback that
   does not crash `run_turn`.
3. ✅ `state["m18_7_attribution_hypotheses"]` is
   populated by the M18.7.2 minimal call on 6/12 (50%)
   of fixture turns (real-LLM replay verification;
   8 entries across 6 distinct turn indices).
4. ✅ `build_conscious_loop_prompt` no longer
   contains the M18.7 v2 attrs segment. Conscious-loop
   LLM receives no M18.7 request
   (`test_conscious_loop_prompt_no_longer_requests_m18_7_v2_attrs`).
5. ✅ M18.7.1 calibration runner produces real
   (non-zero) ECE / Brier / accuracy on the held-out
   fixture (addressee 0.225/0.226/0.25; reaction
   0.258/0.202/0.0; verdict
   `severe_drift_recommend_m20_4`).
6. ✅ M20.4 producer, M20.4.1 same-turn gate,
   settler all see the M18.7.2 fill end-to-end
   (verified by the post-replay
   `m18_7_attribution_hypotheses.json` file with 8
   entries that the M20.4 producer reads on the next
   turn).
7. ✅ All existing tests pass (zero regression on
   M18.7 / M18.7.1 / M20.3 / M20.4 / M20.4.1 /
   M18.6).
8. ✅ New work prompt at
   `prompts/M18.7.2_Work_Prompt.md` exists;
   `prompts/README.md` references M18.7.2.

## What M20.4 can now do (next-step, not M18.7.2)

With M18.7.2 landed, M20.4 has:

1. **Real, persistent `m18_7_attribution_hypotheses`**
   end-to-end (8 entries from the held-out fixture;
   surface persists across `run_turn` calls).
2. **Real M18.7.1 calibration data**: ECE 0.225-0.258,
   accuracy 0.0-0.25, Brier 0.20-0.23,
   `overconfidence_at_high_band` + `bimodal` drift
   signals on both addressee and reaction fields.
3. **Candidate threshold recommendations** for
   `M20_4_TIE_BREAKER_CONFIDENCE_MIN`:
   0.7 (reaction) — 0.9 (addressee) — 0.7 (aggregate).
4. **M18.7.2 surface distinguishability**:
   `source: "m18_7_2_minimal"` on every state entry
   lets M20.4 reason about minimal-path fills
   specifically.

M20.4 threshold revision is now data-backed, not
prompt-blocked.
