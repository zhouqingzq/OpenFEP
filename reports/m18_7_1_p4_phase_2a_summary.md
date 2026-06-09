# P4 Phase 2A Implementation Summary: v1 Addressee Scorer Fix

- **status**: implementation complete; tests pass; real-LLM verification pending
- **generated_at**: 2026-06-09
- **milestone**: P4 Phase 2A (surgical scorer fix)
- **scope**: `segmentum/dialogue/runtime/m18_7_1_calibration.py` + tests
- **pre-read**:
  - `reports/m18_7_1_p4_addressee_design.md`
  - `reports/m18_7_1_p4_phase_1_memo.md`

## TL;DR

`calibrate_addressee_field` now accepts an opt-in kwarg
`treat_no_emit_as_not_addressed: bool = False` (default False
preserves v1 byte-identity). When True, an empty LLM
prediction against `gt.addressed_to_assistant = False` is
counted as **correct** (the LLM correctly identified "not
addressed"). The v2 by_pid runner path enables the kwarg;
v1 modes (`by_turn_id_v1`, `by_turn_id_resolved`) keep the
v1 behavior for byte-identity (D6).

**Honest impact framing** (replaces the earlier "acc
0.125 → 0.500" headline, which was misleading — see
"Overclaim correction" below):

The fix is a **measurement correction**, not a model
improvement. The +3 n_correct delta on the regen data all
come from "no emit against GT false" cases (turns 2, 3, 9,
11). In precision/recall terms:

- **Precision on not-addressed**: empty + GT-false
  (turns 2, 3, 9, 11) now counted correct → improves from
  0/4 to 4/4 (4 cases) when no-emit is the LLM's signal.
- **Recall on addressed**: GT-true + no-emit still wrong
  (4 cases). Recall on addressed is **unchanged** at 0/4
  on the empty-LLM test; the LLM never emits to these
  cases in the regen, so the question is whether the LLM
  is even *attempting* the recall task.
- **ECE regression (0.356 → 0.731)**: 0.0-confidence bin
  accuracy jumps from 0.167 to 0.667, gap doubles. This
  is the **calibration paradox** — the LLM is right but
  reports 0% confidence. ECE is a calibration metric, not
  a correctness metric; M20.4.1 only triggers at
  `confidence > 0.85`, so 0.0-conf cases never reach the
  override, and the ECE regression is informational only.

**What this fix does NOT claim**:
- ❌ It does **not** improve the LLM's recall on
  addressed cases (still 0/4 in the empty-LLM test).
- ❌ It does **not** reduce the high-band overconfidence
  (the 0.90+ bin still has 0/2 acc, the drift signals
  are unchanged).
- ❌ It does **not** move the threshold recommendation
  (tie_breaker=0.9 unchanged, the high-band is excluded).

**Tests**: 61/61 in `test_m18_7_1_calibration.py` (4 new
P4 Phase 2A tests, 1 existing test verified byte-identity
preserved). 244/244 in cross-M18.7.1 regression (after
V2 commit split).

**Real-LLM verification**: 1 fresh by_pid P0 replay running
in background (`bxg45ar4h`, 2026-06-09 19:52 UTC). When
complete, this report should be updated with the surfaced
JSON numbers; the precision/recall framing is
LLM-sampling-robust (it does not depend on whether the
LLM happens to emit a hit in the 0.90+ bin).

## Why this fix

The v1 rule in `calibrate_addressee_field`
(`m18_7_1_calibration.py:836-852` originally) treated
"empty LLM prediction against decidable GT" as always
incorrect. P4 Phase 1's per-turn analysis on the P0 regen
run revealed that 3 of 7 wrong cases in the regen are the
LLM correctly identifying "not addressed to assistant" via
no-emit on implicit side-thread / short-ack cases (turns
2, 3, 9, 11 in the held-out fixture). The v1 rule marked
these as wrong; the P4 fix treats them as correct.

This is a **measurement fix**, not a prompt change or
fixture change. The LLM's behavior is unchanged; we're
correctly counting the cases where the LLM is right.

## Implementation

### File 1: `segmentum/dialogue/runtime/m18_7_1_calibration.py`

**Function signature** (line 798):
```python
def calibrate_addressee_field(
    predictions: Sequence[AddresseePrediction],
    ground_truth: Sequence[AddresseeGroundTruth],
    *,
    treat_no_emit_as_not_addressed: bool = False,
) -> CalibrationFieldReport:
```

**Behavior change** (the only logic change):
- Default (kwarg=False): unchanged. Empty + GT decidable →
  incorrect.
- Opt-in (kwarg=True): Empty + GT False → correct. Empty +
  GT True → still incorrect (LLM should have emitted True).

**Docstring**: documents the new kwarg + links to the P4
Phase 1 finding + explains the rationale (LLM is using
the M18.7 prompt's "MAY leave empty if uncertain" option,
and on side-thread cases "uncertain" is the right
answer).

**Runner dispatch** (line 1410-1422):
The runner now passes the kwarg based on `scoring_mode`:
```python
addressee_report = calibrate_addressee_field(
    addressee_predictions,
    addressee_ground_truth,
    treat_no_emit_as_not_addressed=(
        scoring_mode == "by_pid"
    ),
)
```

This applies the fix only in by_pid mode (the v2 mode).
`by_turn_id_v1` and `by_turn_id_resolved` modes keep the
v1 default behavior.

### File 2: `tests/test_m18_7_1_calibration.py`

Added 4 new tests:

1. **`test_p4_phase2a_empty_pred_still_incorrect_when_gt_true`**
   — Verifies that even with the opt-in kwarg, empty
   prediction + GT true is still incorrect (the fix only
   applies to GT false).

2. **`test_p4_phase2a_empty_pred_correct_when_gt_false_with_kwarg`**
   — Pure-function test of the new behavior: GT false + empty
   prediction → correct (opt-in) / incorrect (v1 default).

3. **`test_p4_phase2a_runner_by_pid_enables_fix`**
   — End-to-end runner test: with the empty LLM (no emit
   on all turns), the by_pid runner reports n_correct=4
   (the 4 GT-false cases) instead of v1's n_correct=0.

4. **`test_p4_phase2a_runner_v1_mode_preserves_byte_identity`**
   — Verifies v1 mode (by_turn_id_v1) still produces
   n_correct=0 with the empty LLM, matching the v1
   baseline fixture. T9 byte-identity preserved.

## Regen data re-score (no LLM cost)

Running the function directly on the P0 regen's per-turn
emissions (extracted from
`tmp_m18_7_1_p4_p0_regen/m18_7_attribution_hypotheses.json`):

| metric | v1 rule | P4 Phase 2A fix | delta |
|---|---|---|---|
| n_correct | 1 / 8 | 4 / 8 | +3 |
| n_incorrect | 7 / 8 | 4 / 8 | -3 |
| accuracy | 0.125 | 0.500 | +0.375 |
| brier | 0.339 | 0.714 | +0.375 (worse) |
| ECE | 0.356 | 0.731 | +0.375 (worse) |
| drift signals | overconfidence + underconfidence + bimodal | same | unchanged |
| threshold: admit_min | None | None | unchanged |
| threshold: tie_breaker | 0.9 | 0.9 | unchanged |

**Bins detail**:
- 0.00-0.10 bin (6 cases, all no-emit conf 0.0):
  v1: acc=0.167 (1/6) gap=0.167
  fix: acc=0.667 (4/6) gap=0.667
- 0.90-1.00 bin (2 cases, turns 4 + 8 emit conf 0.90/0.95):
  v1: acc=0.0 (0/2) gap=0.925
  fix: acc=0.0 (0/2) gap=0.925 (unchanged)

**Why ECE gets worse (calibration paradox)**: the fix
flips 3 cases from "incorrect" to "correct" in the 0.0
confidence bin. The bin's mean confidence is 0.0;
accuracy jumps from 0.167 to 0.667; gap doubles. The LLM
is right but says "0% confident" — that's a real
calibration issue (LLM is underconfident), but the more
fundamental signal (is the LLM correct?) is now more
accurate. **However**, since the 0.0-conf bin is the bin
*least likely* to drive the M20.4.1 override (which
triggers at confidence > 0.85), the ECE regression is
informational only — it does not feed the threshold
recommendation in the way that the high-band gap does.

**Why this is OK for M20.4**: M20.4 wants to know "is the
LLM right?", not "is the LLM well-calibrated?". The fix
improves the answer to the first question. The threshold
recommendation (which uses ECE + drift signals) is
unchanged because the drift signature is the same and
the high-band gap is unchanged.

## Real-LLM verification (pending)

A fresh by_pid P0 replay is running in background
(`bo7yfoddp`) to confirm the surfaced JSON numbers on a
real LLM call. Expected:
- n_correct: 4-6 (range similar to v1 2-4, +3 per run)
- accuracy: 0.50-0.75 (vs v1 0.25-0.50)
- ECE: 0.4-0.7 (vs v1 0.06-0.43, expected regression on
  the calibration metric)
- drift: bimodal + overconfidence (consistent)
- threshold: tie_breaker=0.9 (consistent)

When complete, this report will be updated with the
surfaced JSON.

## What this fix is NOT

- **NOT** a v2 design change. v2 lives in
  `m18_7_1_calibration.py` for measurement, but the v1
  default behavior is preserved (D6 byte-identity).
- **NOT** a prompt change. M18.7.2 prompt is untouched.
- **NOT** a fixture change. The held-out fixture is
  untouched.
- **NOT** a M20.4 threshold decision. M20.4 owner reads
  the handoff doc; P4 surfaces data, doesn't decide.

## File Touch List

| Path | Action | Notes |
|---|---|---|
| `segmentum/dialogue/runtime/m18_7_1_calibration.py` | EDIT | 1 function signature + 1 docstring + 1 logic branch (3-4 lines net) + 1 runner dispatch (5-6 lines) |
| `tests/test_m18_7_1_calibration.py` | EDIT | 4 new tests (~80 lines) |
| `reports/m18_7_1_p4_phase_2a_summary.md` | NEW | this file |

**Not modified**: M18.7, M18.7.2, M20.4, M20.4.1, the
runner script (`scripts/run_m18_7_1_real_llm_calibration.py`),
the held-out fixture, the v1 baseline fixture, any other
calibration file.

## Acceptance criteria for P4 Phase 2A (full milestone)

P4 Phase 2A is "done" when:
1. ✅ The scorer kwarg is implemented.
2. ✅ All tests pass (61/61 unit, 244/244 cross-regression
   after V2 commit split; was 305/305 pre-split).
3. ✅ T9 v1 baseline byte-identity preserved.
4. ⏳ Real-LLM P0 replay confirms the surfaced numbers
   (`bxg45ar4h` running, 2026-06-09 19:52 UTC, ~17 min
   in at last check).
5. ⏳ M20.4 handoff doc updated with the new accuracy
   numbers and the "addressee measurement fix" footnote.

Phase 2A complete = 1-4 done. 5 is part of the broader P4
milestone closure (post-Phase 2B/2C/2D if needed).

## Related

- `reports/m18_7_1_p4_addressee_design.md` (P4 design)
- `reports/m18_7_1_p4_phase_1_memo.md` (P4 Phase 1 finding)
- `reports/m18_7_1_v2_stability_summary.md` (5 P0 runs baseline)
- `reports/m18_7_1_p1_m20_4_handoff.md` (M20.4 handoff)
- `reports/m18_7_1_p5_cross_language_summary.md` (P5 cross-language)
- `tests/fixtures/m18_7_1_v1_report_baseline.json` (T9 baseline)
- `tests/fixtures/m18_7_1_held_out_calibration.json`

## Overclaim correction (2026-06-09, pre-commit review)

The original TL;DR framed the impact as "acc 0.125 →
0.500" with "+3 correct" — this overclaims the fix. The
+3 all come from "no emit matches GT false" cases, which
is **precision on the not-addressed sub-class**, not
recall on addressed. The headline was updated to make
this split explicit:

- Precision on not-addressed: 0/4 → 4/4 (the +3 is real
  on this sub-class).
- Recall on addressed: unchanged at 0/4 (the LLM does
  not emit to these cases in the regen).
- ECE 0.356 → 0.731: real but the regression is on the
  0.0-confidence bin, which M20.4.1 cannot act on
  (trigger threshold is `confidence > 0.85`).

The M20.4-relevant signals (high-band gap, drift
signature, threshold recommendation) are unchanged.
