# M18.7.1 Held-Out Calibration Acceptance Summary

- status: STRUCTURAL (real-LLM replay pending)
- generated_at: 2026-06-07T00:00:00+00:00
- held_out_fixture: E:\workspace\segments\tests\fixtures\m18_7_1_held_out_calibration.json
- calibration_module: E:\workspace\segments\segmentum\dialogue\runtime\m18_7_1_calibration.py
- test_suite: E:\workspace\segments\tests\test_m18_7_1_calibration.py

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

## Real-LLM Replay (extension action, not acceptance gate)

The first successful real-LLM replay of the held-out
fixture will populate this section with the actual ECE /
Brier / per-bin accuracy / drift signals observed
against a real conscious-loop LLM call. Until then, this
report remains at the structural acceptance gate.

## Files Touched

```text
prompts/M18.7.1_Work_Prompt.md                       (NEW)
prompts/README.md                                    (EDIT: appended M18.7.1 entry)
tests/fixtures/m18_7_1_held_out_calibration.json     (NEW: 12 turns, semantic GT)
segmentum/dialogue/runtime/m18_7_1_calibration.py     (NEW: pure functions + runner + state writer)
tests/test_m18_7_1_calibration.py                    (NEW: 41 tests)
reports/m18_7_1_calibration_summary.md               (NEW: this file)
```

**Not modified**:

- `tests/fixtures/m18_held_out_group_chat.json` (M18.6)
- `segmentum/dialogue/runtime/m18_7_attribution.py` (M18.7)
- `segmentum/dialogue/runtime/m20_4_attribution.py` (M20.4)
- `segmentum/dialogue/runtime/mvp_loop.py` (conscious loop)
- `segmentum/dialogue/runtime/m20_4_1_same_turn_gate.py` (M20.4.1)
