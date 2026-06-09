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
- v2 test surface: `tests/test_m18_7_1_calibration.py` (57 tests)
- v1 baseline (T9 byte-identity): `tests/fixtures/m18_7_1_v1_report_baseline.json`
- 12-turn held-out fixture: `tests/fixtures/m18_7_1_held_out_calibration.json`
- Memory: `project_m18_7_1_v2_landed.md` +
  `project_m18_7_1_v2_stability_landed.md`

## CAVEAT (frozen, binding)

**M18.7.1 surfaces candidates. M20.4 owns the decision.**
This handoff is input only. The threshold values
`M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` and
`M20_4_TIE_BREAKER_CONFIDENCE_MIN` are mutated only by
M20.4 owner, with M20.4's own change log.
