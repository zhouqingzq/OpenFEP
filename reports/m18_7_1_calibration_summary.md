# M18.7.1 Held-Out Calibration Acceptance Summary

- status: prompt_order_blocked (real-LLM replay blocked at M18.7 prompt layer)
- generated_at: 2026-06-08T00:10:29+00:00
- real_llm_replay_at: 2026-06-08T00:10:29+00:00
- held_out_fixture: E:\workspace\segments\tests\fixtures\m18_7_1_held_out_calibration.json
- calibration_module: E:\workspace\segments\segmentum\dialogue\runtime\m18_7_1_calibration.py
- test_suite: E:\workspace\segments\tests\test_m18_7_1_calibration.py
- real_llm_runner_script: E:\workspace\segments\scripts\run_m18_7_1_real_llm_calibration.py
- model_under_test: deepseek/deepseek-v4-flash (default), with deepseek/deepseek-v4-pro ablation

## Acceptance Gate (engineering layer)

- Pure-function tests: 41/41 PASS
- Fixture shape tests: 5/5 PASS
- Integration / runner tests (FakeJSONLLM): 6/6 PASS
- Regression on M18.7 / M20.4 / M20.4.1 / M18.6 suites: 198/198 PASS

## Calibration Surface

```text
state["m18_7_1_calibration"] = {
  "last_run_at": <iso8601>,
  "fixture_name": <str>,
  "n_fixtures": <int>,
  "addressee": CalibrationFieldReport (bins, ECE, Brier, accuracy, drift_signals),
  "reaction":  CalibrationFieldReport (bins, ECE, Brier, accuracy, drift_signals),
  "drift_signals": list[str],  # union of both fields
  "threshold_recommendation": {
    "current_admit_min": 0.4,         # M20.4 frozen v1
    "current_tie_breaker_min": 0.85,  # M20.4 frozen v1
    "candidate_admit_min": <float|None>,        # M18.7.1 surfaces only
    "candidate_tie_breaker_min": <float|None>,  # M18.7.1 surfaces only
    "caveat": "decision belongs to M20.4; M18.7.1 only surfaces"
  },
  "engineering_proxy_label": "mvp_local_group_attribution_calibration"
}
```

## Drift Signal Enum (frozen 5 values)

```text
ALLOWED_M18_7_1_DRIFT_SIGNALS = {
    "overconfidence_at_high_band",
    "underconfidence_at_low_band",
    "bimodal",
    "flat_curve",
    "insufficient_data",
}
```

## Decision Authority

**M18.7.1 does NOT mutate `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN`
or `M20_4_TIE_BREAKER_CONFIDENCE_MIN`.** M18.7.1 surfaces
`candidate_*` values; the decision to revise the v1
thresholds belongs to M20.4. The frozen caveat string
`"decision belongs to M20.4; M18.7.1 only surfaces"` is
the binding contract.

## Real-LLM Replay (M18.7 prompt-order blocker — see root cause)

The real-LLM replay **did execute** end-to-end on
2026-06-08 (Python 3.11.7, OpenRouter, 12 turns,
in-memory `MVPStateStore` at `tmp_m18_7_1_real_llm/`).
The runner reached the calibration layer and produced a
report, but the report is **not meaningful** because the
real LLM emitted 0/12 non-empty M18.7 v2 attr
predictions. Engineering-layer metrics (ECE, Brier,
accuracy) are all 0.0 by construction, not by
calibration. This is a **prompt-order root cause**, not
a calibration-math or model-competence issue.

### Measured: real-LLM silent rate on the held-out fixture

| field                    | n_total | n_present | n_unknown | n_correct | n_incorrect | accuracy | brier | ece |
|--------------------------|---------|-----------|-----------|-----------|-------------|----------|-------|-----|
| addressee_hypothesis     | 12      | 0 (8*)|  4          | 0         | 8           | 0.0      | 0.0   | 0.0 |
| reaction_attribution_hyp | 12      | 0 (6*)|  6          | 0         | 6           | 0.0      | 0.0   | 0.0 |

*(`n_present` shows 8 / 6 because the runner counts
`present=False` predictions as `n_present` with
`confidence=0.0`; in reality 0 of these are real LLM
emissions — they are all "key absent from raw LLM
response" fallbacks. See root cause below.)*

`m18_7_attribution_hypotheses` surface length after
12 turns: **0** (verified by direct inspection of
`tmp_m18_7_1_real_llm/temporal_state.json` and the
conversation log: 0 events of type
`AddresseeHypothesisAdmitted` /
`ReactionAttributionHypothesisAdmitted` /
`AttributionHypothesisSkipped`).

### Root cause: M18.7 prompt-order blocker

The conscious-loop user prompt (built by
`build_conscious_loop_prompt` in `mvp_loop.py`) is
**7724 characters** for fixture turn 0. The M18.7 v2
attr instructions are appended as an "Also include
the following M18.7 fields in the same JSON object"
segment at character offset **2914 (37.7% of the
prompt)**, AFTER the main JSON schema spec (which
starts at char 930, 12%).

Direct HTTP probe (bypassing `complete_json` to see
raw OpenRouter response) — same conscious-loop
prompt, two models:

| model                              | finish_reason | completion_tokens | content_chars | keys_emitted | v2 attrs filled? |
|------------------------------------|---------------|-------------------|---------------|--------------|------------------|
| deepseek/deepseek-v4-flash         | stop          | 685               | 2553          | 23           | NO (key absent)  |
| deepseek/deepseek-v4-pro           | stop          | 2106              | 1661          | 23           | NO (key absent)  |

Both models finish with `finish_reason: "stop"` (not
length-truncated), both emit 23 of 25 expected keys,
both **omit** `addressee_hypothesis` and
`reaction_attribution_hypothesis`. v4-pro spends
~3× the tokens trying to fill the schema but still
stops before reaching the v2 attrs segment.

In an isolated mini-prompt test (5-key JSON,
~500 chars), v4-flash correctly emits both v2 attrs
with `confidence: 0.85` / `0.7`. The model can fill
the fields. It is the **prompt order**, not the
model, that breaks instruction-following for the v2
attrs in the full conscious-loop prompt.

A manual reproduction (user-pasted real LLM response
to the same prompt, 2026-06-08) confirmed the same
23-key output: full main schema filled, v2 attrs
absent, `reasoning_notes` as the last field written.
The LLM treats the "Also include M18.7 fields"
segment as a **supplemental note** rather than part
of the **primary schema it must emit**.

### Why M18.7.1 cannot fix this from its own scope

CLAUDE.md red line: *"Semantic decisions must not be
implemented as keyword/regex cue lists in the
engineering layer. When a feature needs semantic
interpretation ... ask the active LLM request/prompt
to return bounded structured fields. Engineering code
may only validate those fields, clamp mechanical
values, persist state, and audit the result."*

M18.7.1 is the calibration analysis layer. It is
**not** the layer that owns the conscious-loop prompt.
Moving the M18.7 v2 attrs segment from 37.7% to ~12%
(in front of the main schema) — or folding the v2
attrs INTO the main JSON schema block — is a
**M18.7 prompt-engineering** decision, not a
M18.7.1 calibration decision. M18.7.1 deliberately
does not touch `mvp_loop.build_conscious_loop_prompt`.

### Actionable next step (for M18.7 owner)

Re-run the M18.7.1 real-LLM replay after **either**:

1. **Move** the v2 attrs instructions from char 2914
   to char ~12% in front of the main JSON schema
   spec, OR
2. **Fold** the v2 attrs into the main JSON schema
   block (single `{{ ... }}` spec) so the LLM sees
   them as primary fields.

Re-run command (no M18.7.1 code changes needed):

```bash
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_m18_7_1_real_llm
```

The runner will produce a meaningful ECE / Brier /
reliability-bin report once the LLM emits non-empty
v2 attrs on a non-trivial fraction of fixture turns.

### Verdict (per the P0 band agreement)

| ECE   | Brier | band                       |
|-------|-------|----------------------------|
| 0.0*  | 0.0*  | n/a — calibration subject absent (no non-silent predictions) |

\* The ECE=0 / Brier=0 numbers are **not** the
"well_calibrated" verdict; they are a degenerate
"no non-silent predictions, so the math is
undefined" state. Do not interpret as "0.4 / 0.85
thresholds are correctly calibrated." Until the
prompt-order blocker is lifted, **M20.4 threshold
revision is unblocked-data pending.**

## Files Touched

```text
prompts/M18.7.1_Work_Prompt.md                       (NEW)
prompts/README.md                                    (EDIT: appended M18.7.1 entry)
tests/fixtures/m18_7_1_held_out_calibration.json     (NEW: 12 turns, semantic GT)
segmentum/dialogue/runtime/m18_7_1_calibration.py     (NEW: pure functions + runner + state writer)
tests/test_m18_7_1_calibration.py                    (NEW: 41 tests)
scripts/run_m18_7_1_real_llm_calibration.py          (NEW: real-LLM runner; reproduces the P0 blocked state)
reports/m18_7_1_calibration_summary.md               (NEW: this file; status upgraded to prompt_order_blocked)
```

**Not modified**:

- `tests/fixtures/m18_held_out_group_chat.json` (M18.6)
- `segmentum/dialogue/runtime/m18_7_attribution.py` (M18.7)
- `segmentum/dialogue/runtime/m20_4_attribution.py` (M20.4)
- `segmentum/dialogue/runtime/mvp_loop.py` (conscious loop)
- `segmentum/dialogue/runtime/m20_4_1_same_turn_gate.py` (M20.4.1)
