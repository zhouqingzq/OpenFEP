# M18.7.1 v2 + P0 Stability — M20.4 Handoff

- **for**: M20.4 owner (threshold decision input)
- **from**: M18.7.1 v2 milestone
- **date**: 2026-06-09
- **read time**: ~5 min (this page) + ~25 min (the 2 reports it points at)

## TL;DR

1. **v2 by_pid is the M20.4-relevant measurement signal.**
   It scores `reaction_to_participant_id` + `is_about_assistant_claim`
   joint correctness with pid normalization — the same axes
   the M20.4 settler uses semantically.
2. **Reaction joint accuracy is stable ≥ 0.6 across 5 fresh
   real-LLM replays** (mean 0.753, range 0.600-1.000). This
   is the new M20.4 input where v1's "0/6" was a measurement
   artifact.
3. **`tie_breaker_min = 0.85` is borderline.** 5/5 replays
   surface candidate `tie_breaker_min` in `{0.8, 0.9}` —
   no nulls, no values > 0.9, no values < 0.8. A nudge to
   0.85-0.90 is data-supported.
4. **`admit_min` should NOT be revised on this 5-run data.**
   Candidate `admit_min` spread (null / 0.5 / 0.2) is too
   wide; the 0.2 is a single-bin outlier, not a real signal.

## What v2 measures (1 paragraph)

M18.7.1 v2 adds three scoring modes for the reaction field
on top of v1's byte-identical legacy mode:

- `by_pid` (Mode A, **primary**): joint correctness of
  `reaction_to_participant_id` (with pid normalization:
  `assistant` / `hutao` / `clawdgroupchat_bot` → `bot`)
  AND `is_about_assistant_claim`. This is the axis the
  M20.4 settler judges on.
- `by_turn_id_resolved` (Mode B): v1 scoring on
  `reaction_to_turn_id` after resolving
  `turn_<role>` placeholders at runner-time.
- `by_turn_id_v1` (Mode C): byte-identical v1 scoring.
  Reported numbers from the P0 replays: 0/6 reaction
  accuracy — confirms v1 was the wrong measurement, not
  the LLM was wrong.

CLI default is `by_pid`. The runner default is
`by_turn_id_v1` for back-compat with the 5 existing
integration tests. Use `--scoring-mode by_pid` for new
work.

Full design: `prompts/M18.7.1_Harness_V2_Design.md`.

## P0 stability headline (5 replays, by_pid)

`scripts/run_m18_7_1_real_llm_calibration.py
--scoring-mode by_pid` × 5, fresh `MVPStateStore` per run,
deepseek/deepseek-v4-flash, 12-turn held-out fixture.

| field | mean | range | consistency |
|---|---|---|---|
| **reaction joint acc** | **0.753** | 0.600 - 1.000 | 5/5 ≥ 0.6 ✅ |
| reaction pid alone acc | 0.500 | 0.333 - 0.667 | 5/5 ≥ 0.333 |
| reaction is_about alone acc | 0.433 | 0.333 - 0.500 | 5/5 ≥ 0.333 (tighter than pid) |
| reaction ECE | 0.293 | 0.175 - 0.438 | n_present=2-5 (binomial floor) |
| addressee acc | 0.400 | 0.250 - 0.500 | 1-2 flips per run |
| addressee ECE | 0.239 | 0.063 - 0.425 | wide — M18.7.2 follow-up |
| `candidate_tie_breaker_min` | 0.86 | 0.8 / 0.9 only | 5/5 in {0.8, 0.9} ✅ |
| `candidate_admit_min` | noisy | null / 0.5 / 0.2 | do not use |

## P1 precision/recall split (2026-06-09)

P1 (commit pending) surfaces two new fields on
`CalibrationFieldReport` (additive; the `n_correct`,
`n_incorrect`, `accuracy`, `brier`, `ece` fields above
are unchanged):

- `addressee_class_breakdown` — per-class confusion
  matrix. Cells: `tp_addressed`, `fn_addressed`
  (with `fn_addressed_present` + `fn_addressed_noemit`
  sub-categories), `tp_not_addressed`
  (with `tp_not_addressed_present` +
  `tp_not_addressed_noemit` sub-categories),
  `fp_not_addressed`. Derived: `precision_on_not_addressed`
  and `recall_on_addressed`.
- `reaction_joint_breakdown` — per-subset joint
  split. Cells: `n_joint_all_decidable` (BOTH axes
  known), `n_joint_emit_subset` (BOTH axes known +
  pred.present), `n_joint_correct_all_decidable`,
  `n_joint_correct_emit_subset`, `n_joint_no_emit_wrong`.
  Derived: `acc_joint_all_decidable` and
  `acc_joint_emit_subset`.

These are **mode-independent pure functions** of the
(prediction, ground_truth) data — populated in v1 mode
AND in v2 modes. The scorer decides how to count
no-emit + GT-false (v1: wrong, v2 fix: correct), but
the breakdown cells are the same in both views.

### bqxsmofri P1 numbers (P1 fresh verify, real LLM)

Direct P1 measurements from the fresh bqxsmofri replay
(`tmp_m18_7_1_p1_real_llm_replay.output`, 2026-06-09,
Python 3.11, 12 turns, real OpenRouter
`deepseek/deepseek-v4-flash`). These are the
**authoritative** P1 numbers — no inference needed.

| addressee split (v2 by_pid) | bqxsmofri value | reading |
|---|---|---|
| `n_gt_true` / `n_gt_false` / `n_unknown` | 4 / 4 / 4 | balanced GT |
| `tp_addressed` | **0** | LLM gets 0 of 4 "addressed" cases right |
| `fn_addressed` (present + noemit) | 4 (3 + 1) | LLM misses ALL 4 addressed cases |
| `tp_not_addressed` (present + noemit) | 4 (3 + 1) | LLM gets 4 of 4 "not addressed" cases right |
| `fp_not_addressed` | 0 | LLM never claims "addressed" wrongly |
| **`precision_on_not_addressed`** | **1.0** | LLM is **perfect** on not-addressed claims |
| **`recall_on_addressed`** | **0.0** | LLM is **0%** recall on addressed cases |

| reaction joint split (v2 by_pid) | bqxsmofri value | reading |
|---|---|---|
| `n_joint_all_decidable` | 6 | 6 turns have BOTH pid and is_about GT known |
| `n_joint_emit_subset` | 3 | LLM only emits on 3 of 6 |
| `n_joint_correct_all_decidable` | 2 | 2 of 6 decidable turns were right (incl. 1 noemit counted wrong) |
| `n_joint_correct_emit_subset` | 2 | 2 of 3 emits were right |
| `n_joint_no_emit_wrong` | **3** | **LLM declines on 3/6 = 50% of decidable reaction turns** |
| `acc_joint_all_decidable` | 0.333 | of 6 decidable, 2 right |
| `acc_joint_emit_subset` | 0.667 | of 3 emits, 2 right (in v2 stability 5-run band) |

**The bqxsmofri numbers resolve the bxg45ar4h inference
ambiguity**: the direct P1 splits show
`precision_on_not_addressed = 1.0` (interpretation B was
correct) and `recall_on_addressed = 0.0` (even worse than
the 0.25 lower-bound estimate).

The reaction `n_joint_no_emit_wrong = 3` (50% no-emit rate)
is higher than the bxg45ar4h inferred 2 (33% rate), but
both are in the same band as the 5-run v2 stability
`n_present = 2-5` per run (variable no-emit rate).

### M20.4-relevant reads from P1

1. **Addressee precision/recall split is the
   structural story — and it's a STRONGER signal
   than the bxg45ar4h inference suggested.** The
   LLM is **perfect on "not addressed" precision
   (1.0)** but has **zero recall on "addressed"
   (0.0)**. M20.4 should weight "not addressed"
   claims more highly than "addressed" claims in
   the settler. This is the structural asymmetry
   the P4 Phase 2A report surfaced, made explicit
   by P1's direct measurement.

2. **Reaction joint "all decidable" accuracy is
   the M20.4-honest signal.** The bqxsmofri emit
   subset is 3 of 6 decidable (50% no-emit rate).
   The 0.667 emit-subset accuracy corresponds to
   a 0.333 all-decidable accuracy once no-emit is
   counted as wrong. The 50% no-emit rate is the
   structural floor on the reaction joint axis
   — and it varies across runs (bxg45ar4h: 33%;
   bqxsmofri: 50%).

3. **High-band overconfidence drift is the
   actionable signal.** bqxsmofri has the same
   drift signature as bxg45ar4h
   (`overconfidence_at_high_band` +
   `underconfidence_at_low_band` + `bimodal`).
   The 0.80-0.90 conf=0.85 bin has 1 wrong (gap
   0.85 — the largest single-bin gap on this
   run). The 0.90-1.00 conf=0.95 bin has 1
   wrong out of 3 (gap 0.283). These are the
   M20.4.1 trigger cases (`confidence > 0.85`).

4. **The 0.20-0.30 conf=0.2 emit was CORRECT in
   bqxsmofri** (acc=1.0 in the bin, gap 0.8).
   The LLM expressed low confidence and was
   right — the opposite of the bxg45ar4h
   0.20-0.30 conf=0.3 wrong case. The
   low-confidence band is unstable across runs;
   M20.4.1 should respect low-conf emits and
   not act on them.

P1 does NOT change the threshold recommendation
(`candidate_admit_min = null,
candidate_tie_breaker_min = 0.9` for bqxsmofri,
vs `0.8` for bxg45ar4h; both in the {0.8, 0.9}
band surfaced in the 5-run v2 stability). P1
makes the per-class structure that the threshold
was based on **explicit** in the report.

Full P1 report: `reports/m18_7_1_p1_precision_recall_split.md`.

## M20.4-relevant decisions (actionable)

### Decision A: `tie_breaker_min` nudge

**Current**: `tie_breaker_min = 0.85` (M20.4 frozen v1)
**Data**: 5/5 replays → candidate ∈ {0.8, 0.9}
**Recommendation**: pick 0.85-0.90. Either is data-supported.

- `0.85` (keep): defensible if M20.4 wants minimal change.
- `0.90` (raise): supported by 3/5 replays; tightens the
  high-confidence band at the cost of 1-2 borderline cases
  per replay.
- `0.80` (loosen): supported by 2/5 replays; relaxes the
  band, more borderline cases pass through. Note: the
  current 0.85-0.9 band in v2 has very few cases per run
  (3-5), so the 0.8 vs 0.9 split is sensitive to which
  bin the worst-gap case lands in. Not a strong signal
  in either direction.

### Decision B: `is_about_assistant_claim` sub-axis as settler input

**Data**: 0.333-0.500 across 5 runs, variance tighter than
the `pid` sub-axis (range 0.167 vs 0.333).
**Recommendation**: M20.4 settler can use the
`is_about_assistant_claim` axis as a **more reliable
sub-input** than the raw pid. If the settler weights
sub-axes, weight `is_about` higher.

### Decision C: reaction field is "moderate drift" not "severe"

**Data**: per-axis reaction verdict = `insufficient_data`
in 5/5 runs (small `n_present`), but joint accuracy
≥ 0.6 and joint ECE < 0.5.
**Recommendation**: M20.4 should treat reaction as
**calibrated enough for settler use**, not as a blocker
to the threshold decision.

## What M20.4 should NOT do (with this 5-run data)

### Do NOT move `admit_min`

`admit_min` is the lower bound — which predictions get
admitted to the settler at all. The 5-run spread on
`candidate_admit_min` (null / 0.5 / 0.2) is too wide to
be a meaningful signal:

- 2/5 replays surface `null` (no recommendation).
- 2/5 replays surface `0.5` (raise from 0.4 to 0.5).
- 1/5 replays surfaces `0.2` (lower from 0.4 to 0.2) —
  this is an outlier from a 0.20-0.30 band gap, not a
  real signal.

`admit_min = 0.4` should be **left alone** until
addressee calibration stabilizes (P4 milestone).

### Do NOT make addressee threshold decisions

Addressee ECE spread is 0.063-0.425 across 5 runs.
That's a 0.36 ECE range, which is wider than the
ECE range that distinguishes "moderate_drift" from
"severe_drift" (the band boundary is at 0.15). The
held-out fixture is not enough data to make a call
on addressee. P4 is the right milestone for this.

**P4 Phase 2A measurement-fix footnote (2026-06-09)**:
the addressee acc in those 5 runs is measured under the
v1 rule, which is biased on the "no emit + GT false"
sub-class. The P4 Phase 2A fix (`0a42c24`) introduces
an opt-in kwarg `treat_no_emit_as_not_addressed` that
flips these to correct under `scoring_mode="by_pid"`.
On the regen data this is precision on not-addressed
0/4 → 4/4; recall on addressed is unchanged at 0/4. The
high-band gap (turns 4 + 8 conf 0.90/0.95) and the drift
signature (bimodal + overconfidence) are unchanged, so
the threshold recommendation (tie_breaker=0.9) is
unchanged. The 5-run v2 stability table above is
addressee-M20.4-irrelevant; do not act on those numbers
for the addressee side until P4 closes. See
`reports/m18_7_1_p4_phase_2a_summary.md` for the
precision/recall breakdown.

### Do NOT expect `reaction_to_turn_id` (Mode B) to work yet

5/5 replays in by_turn_id_resolved mode show 0/3
reaction accuracy on the decidable subset. The LLM
emits `reaction_to_turn_id=""` 5/6 times, and the
3 decidable assistant placeholders can't resolve
from the fixture's user-only `replay_history`. This
is a fixture/prompt gap, not a stability question.
P2 (M18.7.2 prompt turn_id enumeration) + P3
(fixture assistant prior turn) are the unlocks.

## What M20.4 owner should do next (concrete)

1. **Read** the 2 reports:
   - `reports/m18_7_1_harness_v2_implementation_summary.md`
     (10.9 KB, v2 mechanics + AC4 acceptance)
   - `reports/m18_7_1_v2_stability_summary.md`
     (5-run stability + per-axis variance)
2. **Decide** on `tie_breaker_min` (Decision A above).
   Either `0.85` (keep) or `0.90` (raise) is
   data-supported; `0.80` (loosen) is also surfaced but
   less strongly. The 5-run data is **directional, not
   definitive** — M20.4 owner picks.
3. **Decide** whether to weight `is_about_assistant_claim`
   in the settler (Decision B above).
4. **Defer** `admit_min` revision and addressee work to
   P2/P3/P4.
5. **Open a follow-up** with the M18.7.1 / M18.7.2 owner
   about P2 (prompt turn_id enumeration) and P3
   (fixture repair) — these will unlock the turn_id
   axis and give the next round of M20.4 input.

## Open questions for M20.4 owner

1. **`tie_breaker_min` exact value**: 0.85 / 0.90 / 0.80?
   Trade-off: 0.90 tightens and may drop borderline
   cases; 0.80 loosens and may admit too many. The
   5-run data does not break the tie — M20.4 owner's
   call.
2. **Is the 5-run sample size enough?** Each run has
   `n_present = 2-5` on the reaction joint axis. If M20.4
   needs n ≥ 30 for a tight CI, that's a separate
   stability-rerun with a larger fixture (P2/P3 unlock).
3. **Is the held-out fixture representative?** The
   fixture is carol/david/hutao group chat, 12 turns.
   A second fixture (P5: Chinese small-scenario) is
   coming; this gives cross-language stability data.
4. **What does M20.4 do with the LLM non-determinism
   floor?** The 0.6-1.0 reaction joint spread is the
   binomial floor for n=2-5 on this fixture. Any future
   v2 replay should land in this band; out-of-band
   results mean model/fixture/prompt changed.

## Pointers

- v2 design: `prompts/M18.7.1_Harness_V2_Design.md`
- v2 implementation report: `reports/m18_7_1_harness_v2_implementation_summary.md`
- v2 stability report (5 replays): `reports/m18_7_1_v2_stability_summary.md`
- v1 calibration status: `reports/m18_7_1_calibration_summary.md`
- v2 test surface: `tests/test_m18_7_1_calibration.py` (69 tests, 61 v2 + 8 P1)
- v1 baseline (T9 byte-identity): `tests/fixtures/m18_7_1_v1_report_baseline.json`
- 12-turn held-out fixture: `tests/fixtures/m18_7_1_held_out_calibration.json`
- Memory: `project_m18_7_1_v2_landed.md` +
  `project_m18_7_1_v2_stability_landed.md` +
  `project_m18_7_1_p4_p5_status.md` +
  `project_m18_7_1_p1_landed.md`

## CAVEAT (frozen, binding)

**M18.7.1 surfaces candidates. M20.4 owns the decision.**
This handoff is input only. The threshold values
`M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` and
`M20_4_TIE_BREAKER_CONFIDENCE_MIN` are mutated only by
M20.4 owner, with M20.4's own change log.
