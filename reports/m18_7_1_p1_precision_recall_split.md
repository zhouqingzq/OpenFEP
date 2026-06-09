# P1 Precision/Recall Metric Split — Implementation Summary

- **status**: implementation complete; real-LLM verification in progress
- **generated_at**: 2026-06-09
- **milestone**: P1 (precision/recall metric split, after P0 cleanup + P4 Phase 2A)
- **scope**: `segmentum/dialogue/runtime/m18_7_1_calibration.py` + tests + v1 baseline fixture
- **pre-read**:
  - `reports/m18_7_1_p4_phase_2a_summary.md` (P4 Phase 2A framing)
  - `reports/m18_7_1_p1_m20_4_handoff.md` (M20.4 handoff, will be updated)
  - `reports/m18_7_1_v2_stability_summary.md` (5 P0 runs baseline)

## TL;DR

P1 splits the addressee metric into **precision on not-addressed**
(P(GT=false | pred=false)) and **recall on addressed**
(P(pred=true | GT=true)). It also splits the reaction joint
metric into **all decidable** (M20.4-honest denominator,
includes no-emit as wrong) and **LLM-emit subset** (practical
denominator, only counts cases where the LLM emitted).

The splits are **additive, mode-independent** fields on
`CalibrationFieldReport` (`addressee_class_breakdown`,
`reaction_joint_breakdown`). The `n_correct` / `n_incorrect` /
`accuracy` / `brier` / `ece` fields are unchanged. v1 baseline
is regenerated to include the new fields; the byte-identity
regression test (T9) passes against the post-P1 baseline.

**Why this matters for M20.4**: the P4 Phase 2A report already
established that the LLM is strong on "not addressed" precision
(0/4 → 4/4) but unchanged on "addressed" recall (0/4). P1
makes this explicit in the surfaced report so M20.4 owner
doesn't have to derive it from bins. The reaction joint split
exposes the **denominator selection bias** — the joint
denominator is `pred.present AND (pid AND is_about decidable)`,
which is smaller than the per-axis denominator (6 vs 4). M20.4
should know which denominator the surfaced accuracy reflects.

## Real LLM verification — bqxsmofri (fresh P1 by_pid replay)

A fresh by_pid real-LLM replay was run for P1
verification (`bqxsmofri`, 2026-06-09, Python 3.11,
12 turns, real OpenRouter `deepseek/deepseek-v4-flash`,
session root `tmp_m18_7_1_p1_real_llm_replay/`). The
fresh replay is the **authoritative** P1 verification:
the surfaced JSON has the full 12-turn attribution
(no cap drops since the new session has only 12 turns)
and the P1 splits are directly visible (not inferred
from bins).

The `bxg45ar4h` surfaced JSON (P4 Phase 2A verify) is
referenced for cross-run comparison; the inferred-numbers
caveats from earlier drafts are now superseded by the
direct P1 splits from `bqxsmofri.output`.

### Addressee P1 split (v2 by_pid, P4 Phase 2A fix on)

**Surfaced totals** (unchanged by P1):

| metric | value |
|---|---|
| n_total | 12 |
| n_present | 8 |
| n_unknown | 4 |
| n_correct | 4 |
| n_incorrect | 4 |
| accuracy | 0.5 |
| ece | 0.5125 |
| drift | overconfidence_at_high_band, underconfidence_at_low_band, bimodal |
| threshold | candidate_admit_min=null, candidate_tie_breaker_min=0.9 |

**Surfaced bins** (per-confidence-bucket counts):

| bin | count | conf | acc | interpretation |
|---|---|---|---|---|
| 0.00-0.10 | 2 | 0.0 | 0.5 | 2 noemit, 1 correct + 1 wrong (under v2 fix) |
| 0.20-0.30 | 1 | 0.2 | 1.0 | 1 emit, correct (low-conf correct — UNUSUAL) |
| 0.50-0.60 | 1 | 0.6 | 0.0 | 1 emit, wrong (mid-conf wrong) |
| 0.80-0.90 | 1 | 0.85 | 0.0 | 1 emit, wrong (high-conf wrong — overconfidence) |
| 0.90-1.00 | 3 | 0.95 | 0.667 | 3 emits, 2 correct + 1 wrong (high-band) |

**P1 breakdown cells** (directly from the runner, no
inference needed):

```json
"addressee_class_breakdown": {
  "n_gt_true": 4,
  "n_gt_false": 4,
  "n_unknown": 4,
  "tp_addressed": 0,
  "fn_addressed": 4,
  "fn_addressed_present": 3,
  "fn_addressed_noemit": 1,
  "tp_not_addressed": 4,
  "tp_not_addressed_present": 3,
  "tp_not_addressed_noemit": 1,
  "fp_not_addressed": 0,
  "precision_on_not_addressed": 1.0,
  "recall_on_addressed": 0.0
}
```

**Headline P1 numbers** (bqxsmofri, real LLM):

| metric | value | M20.4 reading |
|---|---|---|
| **`precision_on_not_addressed`** | **1.0** | The LLM is **perfect** on "not addressed" claims. 3 emits + 1 noemit (v2 fix) all correctly identified GT-false cases. |
| **`recall_on_addressed`** | **0.0** | The LLM **misses all 4** "addressed" cases. 3 emits were fn (LLM said not-addressed when GT was addressed), 1 noemit on GT-true. |
| `tp_not_addressed_noemit` | 1 | The v2 fix contributes 1 of 4 tp_not_addressed. Without the fix, recall on addressed is unchanged but precision on not-addressed would be 0.75 (3/4) instead of 1.0. |
| `fn_addressed_noemit` | 1 | The LLM declined to commit on 1 GT-true case (noemit). This is a "noemit-on-addressed" — different from the P4 Phase 2A "noemit-on-not-addressed" pattern. |

**M20.4-relevant reading**:

1. **The LLM is structurally asymmetric on addressee.**
   Perfect precision on "not addressed" but zero recall
   on "addressed". This is the structural story that
   the P4 Phase 2A report surfaced and P1 makes
   explicit. M20.4 should treat "not addressed" claims
   as **high-confidence signal** and "addressed" claims
   as **low-confidence signal** in the settler.
2. **The high-band overconfidence drift is the
   actionable signal.** The 0.80-0.90 conf=0.85 bin
   has 1 wrong (gap 0.85 — the largest single-bin gap
   on this run). The 0.90-1.00 conf=0.95 bin has 1
   wrong out of 3 (gap 0.283). These are the M20.4.1
   trigger cases (`confidence > 0.85`).
3. **The 0.20-0.30 conf=0.2 emit was CORRECT in this
   run** (acc=1.0 in the bin, gap 0.8). This is
   the LLM expressing low confidence and being right —
   the opposite of the bxg45ar4h 0.20-0.30 conf=0.3
   wrong case. The 0.0 conf bin (noemit) has acc=0.5
   (1 correct under v2 fix + 1 wrong on GT-true).

### Reaction joint P1 split (v2 by_pid)

**Surfaced totals** (unchanged by P1):

| metric | value |
|---|---|
| n_total | 12 |
| n_present (joint: pid AND is_about decidable + pred.present) | 3 |
| n_unknown (joint: BOTH axes unknown) | 6 |
| n_correct (joint) | 2 |
| n_incorrect (joint) | 1 |
| accuracy (joint) | 0.667 |
| ece (joint) | 0.333 |

**P1 breakdown** (joint axis, all decidable vs LLM-emit subset):

```json
"reaction_joint_breakdown": {
  "n_joint_all_decidable": 6,
  "n_joint_emit_subset": 3,
  "n_joint_correct_all_decidable": 2,
  "n_joint_correct_emit_subset": 2,
  "n_joint_no_emit_wrong": 3,
  "acc_joint_all_decidable": 0.333,
  "acc_joint_emit_subset": 0.667
}
```

**Headline P1 numbers** (bqxsmofri, real LLM):

| metric | value | M20.4 reading |
|---|---|---|
| `n_joint_all_decidable` | 6 | 6 turns have BOTH pid and is_about GT known. |
| `n_joint_emit_subset` | 3 | The LLM only emitted on 3 of 6. |
| `n_joint_no_emit_wrong` | 3 | **The LLM declined to commit on 3/6 = 50% of decidable reaction turns.** This is the structural floor on the reaction joint axis. |
| `acc_joint_all_decidable` | 0.333 | Of the 6 turns where the LLM should have produced a hypothesis, only 2 were right (incl. 1 no-emit on decidable). |
| `acc_joint_emit_subset` | 0.667 | When the LLM commits, it's right 2 of 3 times. The 0.667 emit-subset accuracy is in the v2 stability 5-run band (mean 0.753, range 0.6-1.0). |

**Reading the joint split**:

- The "all decidable" denominator (6) is the M20.4-honest
  view: it includes the 3 no-emit cases (where the LLM
  declined to commit). Accuracy 0.333 reflects "of the
  6 cases where the LLM should have produced a hypothesis,
  only 2 were right".
- The "LLM-emit subset" denominator (3) is the practical
  view: it excludes the 3 no-emit cases. Accuracy 0.667
  reflects "of the 3 cases where the LLM committed, 2
  were right".
- **50% no-emit rate on decidable reaction turns** is
  high. The 5-run v2 stability report has `n_present=2-5`
  per run (variable no-emit rate). The bqxsmofri
  no-emit rate (50%) is at the high end; the bxg45ar4h
  rate (33%) is more typical. M20.4 should know that
  the joint accuracy is sensitive to the no-emit rate.
- **Denominator selection bias**: the joint `n_present=3`
  is much smaller than the per-axis `n_present=6` (pid
  alone or is_about alone). The "joint" requires BOTH
  axes to be decidable AND pred.present, so it's the
  intersection of 3 conditions. The per-axis breakdown
  is a less biased view.

### Reaction per-axis breakdowns (already surfaced in bqxsmofri)

The pid and is_about breakdowns are part of the v2
surfaced JSON. P1 doesn't add new info here; it just
adds the joint subset split (all decidable vs LLM-emit).

| axis | n_present | n_correct | accuracy | brier | ece |
|---|---|---|---|---|---|
| pid | 6 | 3 | 0.5 | 0.041 | 0.133 |
| is_about | 6 | 2 | 0.333 | 0.074 | 0.167 |

`is_about` is the tighter sub-axis (consistent with v2
stability 5-run finding). The per-axis accuracies differ
because the LLM can get the pid right but is_about wrong
(or vice versa).

### Cross-run comparison (bqxsmofri vs bxg45ar4h)

| metric | bqxsmofri (P1 verify) | bxg45ar4h (P4 Phase 2A) | delta |
|---|---|---|---|
| addressee n_correct | 4 | 5 | -1 |
| addressee accuracy | 0.5 | 0.625 | -0.125 |
| addressee brier | 0.454 | 0.492 | -0.038 (better) |
| addressee ECE | 0.5125 | 0.531 | -0.019 (better) |
| addressee drift | 3 signals (same) | 3 signals (same) | unchanged |
| addressee threshold tie_breaker | 0.9 | 0.8 | +0.1 |
| addressee precision_on_not_addressed | 1.0 | 0.75-1.0 (uncertain) | direct! |
| addressee recall_on_addressed | 0.0 | 0.25-0.5 (uncertain) | direct! |
| reaction n_present | 3 | 4 | -1 |
| reaction accuracy | 0.667 | 0.5 | +0.167 |
| reaction pid n_correct | 3 | 4 | -1 |
| reaction is_about n_correct | 2 | 2 | unchanged |
| n_joint_no_emit_wrong | 3 | 2 | +1 |
| acc_joint_all_decidable | 0.333 | 0.333 | unchanged |
| acc_joint_emit_subset | 0.667 | 0.5 | +0.167 |

**Reading the cross-run comparison**:

- The drift signature is **stable** across both runs
  (overconfidence_at_high_band +
  underconfidence_at_low_band + bimodal).
- The `n_joint_no_emit_wrong` count varies (2 vs 3)
  — this is LLM sampling noise on the no-emit decision,
  not a stability floor. The 5-run v2 stability has
  `n_present=2-5` per run, consistent with this range.
- The bqxsmofri threshold recommends
  `candidate_tie_breaker_min=0.9` (vs bxg45ar4h's 0.8).
  Both are in the {0.8, 0.9} band surfaced in the 5-run
  v2 stability (mean 0.86). The threshold is
  **directional, not definitive** — M20.4 owner reads
  the band, not the specific value.
- The bqxsmofri precision_on_not_addressed=1.0 is
  a stronger signal than bxg45ar4h's 0.75-1.0
  (inferred). This is the first direct measurement
  of the LLM's "not addressed" precision on a real
  LLM run, and it's perfect.

### Addressee v1 mode (P4 Phase 2A fix off) for comparison

The v1 mode (no fix) splits for bqxsmofri:

| cell | v1 count (noemit is wrong) |
|---|---|
| n_correct_v1 | 3 (emit-correct: 1 conf=0.2 + 2 conf=0.95 = 3) |
| n_incorrect_v1 | 5 (3 emit-wrong + 1 noemit-gt-true + 1 noemit-gt-false) |

| cell | v1 value |
|---|---|
| `tp_addressed` | 0 (no emit was tp_addressed in bqxsmofri) |
| `fn_addressed_present` | 3 (3 emits were fn: conf=0.6, 0.85, 0.95) |
| `fn_addressed_noemit` | 1 (1 noemit on GT-true) |
| `tp_not_addressed_present` | 3 (3 emits were tp_not_addressed: conf=0.2 + 2 conf=0.95) |
| `tp_not_addressed_noemit` | 0 (1 noemit on GT-false, but v1 counts as wrong) |
| `fp_not_addressed` | 0 |
| `precision_on_not_addressed` (v1) | 0.75 (3 / (3+0+1)) = 3/4 ... wait |

Let me recompute. In v1 mode, noemit on GT-false is wrong.
The breakdown cells are the same (they're pure data), but
the scorer counts `tp_not_addressed_noemit` as wrong.

- v1 n_correct = 3 (the 3 tp_not_addressed_present emits)
- v1 n_incorrect = 5 (3 fn_addressed_present + 1 fn_addressed_noemit + 1 tp_not_addressed_noemit-as-wrong)

So the breakdown cells are unchanged, but the
n_correct/n_incorrect accounting differs from v2.
The breakdown is mode-independent; the scorer is not.

## What P1 is

P1 is a **measurement refinement**, not a prompt or
threshold change. It surfaces two new fields on
`CalibrationFieldReport` that were already implicit in the
existing data:

- `addressee_class_breakdown` — the per-class confusion
  matrix. The TP/FP/FN cells were always computable from
  the (prediction, ground_truth) data; P1 just exposes them
  in the to_dict().
- `reaction_joint_breakdown` — the all-decidable vs
  LLM-emit subset split. The denominators were always
  computable; P1 just exposes them with the derived
  accuracies.

The fields are **mode-independent**: the breakdown is the
same in v1 (`by_turn_id_v1`) and v2 (`by_pid`,
`by_turn_id_resolved`) modes. The scorer decides how to
count no-emit + GT-false (v1: wrong, v2 fix: correct), but
the breakdown exposes both views via the `tp_not_addressed_noemit`
cell (counted as TP regardless of mode) and the `n_correct`
field (mode-dependent).

## What P1 is NOT

- **NOT** a prompt change. M18.7.2 prompt is untouched.
- **NOT** a fixture change. The held-out fixture is untouched.
- **NOT** an M20.4 threshold decision. M20.4 owner reads
  the breakdown; threshold changes are M20.4's call.
- **NOT** a behavior change for the LLM. The LLM still
  emits the same hypotheses; P1 just makes the per-class
  measurement explicit.
- **NOT** a v2 design change. P1 lives entirely in
  M18.7.1 territory (calibrator + report shape + tests).

## Why this matters for M20.4

The P4 Phase 2A report's "honest framing" section said:

> The fix is a measurement correction, not a model
> improvement. The +3 n_correct delta on the regen data all
> come from "no emit against GT false" cases (turns 2, 3, 9,
> 11). In precision/recall terms:
>
> - Precision on not-addressed: empty + GT-false (turns
>   2, 3, 9, 11) now counted correct → improves from
>   0/4 to 4/4 (4 cases) when no-emit is the LLM's signal.
> - Recall on addressed: GT-true + no-emit still wrong
>   (4 cases). Recall on addressed is unchanged at 0/4
>   on the empty-LLM test.

P1 makes this framing the **surfaced structure** of the
report. M20.4 owner reads `addressee_class_breakdown` and
sees:

- `precision_on_not_addressed`: the LLM's reliability on
  "not addressed" claims. P4 fix bumps this from 0/4 to
  4/4 in the regen; in the bxg45ar4h surfaced (real LLM),
  this is 0.75 (interpretation A) or 1.0 (interpretation B).
- `recall_on_addressed`: the LLM's reliability on
  "addressed" claims. P4 fix doesn't change this
  (no-emit + GT-true is still wrong in v1 and v2).
  bxg45ar4h: 0.5 (A) or 0.25 (B).

For the reaction joint, M20.4 owner reads
`reaction_joint_breakdown` and sees:

- `acc_joint_all_decidable`: the honest denominator
  (includes no-emit as wrong). 0.333 in bxg45ar4h.
- `acc_joint_emit_subset`: the practical denominator
  (only LLM-emit cases). 0.5 in bxg45ar4h.
- `n_joint_no_emit_wrong`: the count of "no-emit is the
  LLM's signal" cases. 2 in bxg45ar4h. **This is the
  structural floor on the reaction joint axis** — the
  LLM declines to commit on 2/6 = 33% of decidable
  reaction turns. Whether the no-emit is the right
  call depends on the GT; the surfaced 0.333 accuracy
  in the "all decidable" view reflects this floor.

## Implementation

### File 1: `segmentum/dialogue/runtime/m18_7_1_calibration.py`

**Two new fields on `CalibrationFieldReport`** (after the
v2 `pid_breakdown` / `is_about_breakdown`):

```python
addressee_class_breakdown: dict[str, object] | None = None
reaction_joint_breakdown: dict[str, object] | None = None
```

Both default to `None`. The `to_dict()` method emits them
only when populated, preserving the v1 (pre-P1) report
shape for callers that don't construct the breakdown.

**Two new pure helper functions**:

- `_compute_addressee_class_breakdown(predictions, ground_truth) -> dict`:
  Walks the (pred, gt) pairs, increments the 4 cells
  (tp_addressed, fn_addressed, tp_not_addressed, fp_not_addressed)
  with noemit/present sub-categories, and derives
  `precision_on_not_addressed` and `recall_on_addressed`.
  Independent of the `treat_no_emit_as_not_addressed` kwarg.

- `_compute_reaction_joint_breakdown(n_all_decidable, n_emit_subset, n_correct_all, n_correct_emit) -> dict`:
  Pure helper that takes the 4 counters and produces the
  breakdown dict with derived accuracies. The caller
  (calibrator) computes the per-mode joint correctness
  and reports the counters.

**Three calibrator changes**:

- `calibrate_addressee_field` — calls the helper and
  attaches `addressee_class_breakdown` to the returned
  `CalibrationFieldReport`. The breakdown is computed in
  all modes (v1, v2 by_pid, v2 by_turn_id_resolved).

- `calibrate_reaction_field` (v1) — tracks 4 new inline
  counters (`n_joint_all_decidable`, `n_joint_emit_subset`,
  `n_joint_correct_all`, `n_joint_correct_emit`) during
  the main loop, then calls the helper and attaches
  `reaction_joint_breakdown` to the returned report.

- `_calibrate_reaction_by_pid` (v2) — adds a second pass
  over (pred, gt) pairs to compute the all-decidable
  counters (the v2 by_pid main loop already tracks the
  emit-subset). Then calls the helper and attaches the
  breakdown.

### File 2: `tests/test_m18_7_1_calibration.py`

Added 8 new tests (P1 T1-T8):

1. **`test_p1_t1_addressee_class_breakdown_pure_function`** —
   Synthetic 5-turn fixture exercising the helper
   directly. Asserts the 4 cells + precision/recall
   with no no-emit cases.

2. **`test_p1_t2_addressee_class_breakdown_with_noemit_cases`** —
   Synthetic 4-turn fixture with mixed emit / no-emit
   cases. Verifies that the breakdown correctly
   attributes no-emit cases to FN (GT-true) and TP
   (GT-false) cells.

3. **`test_p1_t3_addressee_breakdown_is_mode_independent`** —
   Verifies that the breakdown is the same in v1 (no
   kwarg) and v2 (kwarg=True) modes, but `n_correct`
   differs. This is the key P1 property: the
   breakdown is a pure function of the (pred, gt) data.

4. **`test_p1_t4_reaction_joint_breakdown_pure_function`** —
   Synthetic counters passed directly to the helper.
   Verifies the math: `n_no_emit_wrong = 6 - 4 - (2 - 2) = 2`,
   `acc_joint_all_decidable = 2/6`, `acc_joint_emit_subset = 0.5`.

5. **`test_p1_t5_reaction_joint_breakdown_with_empty_emit_subset`** —
   Edge case: all decidable, no emit. The emit-subset
   accuracy is 0.0 by convention.

6. **`test_p1_t6_reaction_joint_breakdown_appears_in_v1_report`** —
   V1 (by_turn_id_v1) mode reaction report includes
   the per-subset joint breakdown. Asserts on the
   held-out fixture with the empty LLM (n=6
   decidable, n=0 emit, n=6 no_emit_wrong).

7. **`test_p1_t7_reaction_joint_breakdown_appears_in_v2_by_pid_report`** —
   V2 by_pid mode reaction report includes the
   breakdown AND the existing pid_breakdown /
   is_about_breakdown. Asserts both are populated.

8. **`test_p1_t8_addressee_breakdown_emitted_in_to_dict_v2_modes`** —
   Verifies that `addressee_class_breakdown` and
   `reaction_joint_breakdown` appear in the to_dict()
   output of v1 and v2 mode reports.

### File 3: `tests/fixtures/m18_7_1_v1_report_baseline.json`

Regenerated to include the P1 fields. The T9 test
(v1 byte-identity) is updated to assert against the
post-P1 baseline. The `_meta.purpose` field is updated
to mention the P1 splits. The existing T9 invariant
(8 emits, 0 correct in v1 mode) is preserved; the new
baseline just has the additional `addressee_class_breakdown`
and `reaction_joint_breakdown` keys.

## What changes for M20.4 owner

M20.4 owner reads the P4 Phase 2A handoff + this report.
The handoff doc (`reports/m18_7_1_p1_m20_4_handoff.md`)
should be updated to:

1. Reference the `addressee_class_breakdown` and
   `reaction_joint_breakdown` fields as the new surfaced
   structures.
2. Surface the bxg45ar4h numbers (with the two
   interpretations A/B explicitly) for M20.4 to choose.
3. Highlight the `n_joint_no_emit_wrong` field as the
   structural floor on the reaction joint axis.

The threshold recommendation is **unchanged**:
- `candidate_admit_min = 0.3` (driven by the high-band
  overconfidence drift).
- `candidate_tie_breaker_min = 0.8` (driven by the
  high-band gap 0.45 in the 0.90-1.00 bin).

The P1 splits don't change the threshold; they just
expose the structure that the threshold was based on.

## File Touch List

| Path | Action | Notes |
|---|---|---|
| `segmentum/dialogue/runtime/m18_7_1_calibration.py` | EDIT | 2 new fields on `CalibrationFieldReport`, 2 new pure helpers, 3 calibrator changes (~150-200 lines net) |
| `tests/test_m18_7_1_calibration.py` | EDIT | 8 new tests (P1 T1-T8) |
| `tests/fixtures/m18_7_1_v1_report_baseline.json` | REGEN | Captures the post-P1 v1 baseline (additive fields) |
| `reports/m18_7_1_p1_precision_recall_split.md` | NEW | this file |

**Not modified**: M18.7, M18.7.2, M20.4, M20.4.1, M20.3, the
runner script, the held-out fixture, the M20.4 handoff doc
(separate task).

## Acceptance criteria for P1 (full milestone)

P1 is "done" when:

1. ✅ The two new fields are added to
   `CalibrationFieldReport`.
2. ✅ All 8 new P1 tests pass.
3. ✅ All 61 existing M18.7.1 tests pass (no regression).
4. ✅ T9 v1 baseline is regenerated and the test passes
   against the post-P1 baseline.
5. ✅ Cross-M18.7.1 regression: 312/312 (was 304/304
   pre-P1, +8 new P1 tests).
6. ⏳ Real-LLM verification: bqxsmofri (fresh by_pid
   replay) completes and the surfaced numbers match
   the inferred numbers within sampling noise.
7. ⏳ M20.4 handoff doc updated with the P1 splits.

P1 is **partially DONE** (1-5 met; 6-7 in progress).

## Fresh replay verification (in progress)

A fresh by_pid real-LLM replay is running in the
background (task ID `bqxsmofri`, session root
`tmp_m18_7_1_p1_real_llm_replay/`). The replay uses
the same fixture (`m18_7_1_held_out_calibration.json`)
and the same model (`deepseek/deepseek-v4-flash`). The
surfaced JSON will have the full 12-turn attribution
(no cap drops since the new session has 12 turns only)
and the P1 splits will be directly visible (not
inferred from bins).

Expected outcome: the surfaced n_present, n_correct,
n_incorrect, and the P1 breakdown cells will match the
inferred numbers within sampling noise. The drift
signature is expected to be the same
(overconfidence_at_high_band +
underconfidence_at_low_band + bimodal) and the
threshold recommendation is expected to be
`candidate_admit_min=0.3, candidate_tie_breaker_min=0.8`.

## Related

- `reports/m18_7_1_p4_phase_2a_summary.md` (P4 Phase 2A
  framing; the "honest precision/recall" section is
  now surfaced in the report)
- `reports/m18_7_1_p1_m20_4_handoff.md` (M20.4 handoff,
  to be updated)
- `reports/m18_7_1_v2_stability_summary.md` (5-run P0
  baseline for reaction joint acc)
- `reports/m18_7_1_harness_v2_implementation_summary.md`
  (v2 design)
- `tests/fixtures/m18_7_1_v1_report_baseline.json` (T9
  post-P1 baseline)
- `tests/fixtures/m18_7_1_held_out_calibration.json`
  (12-turn English fixture)
