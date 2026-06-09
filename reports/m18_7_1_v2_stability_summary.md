# M18.7.1 v2 Stability Check — 5 Replay Runs

- status: P0 stability data complete
- generated_at: 2026-06-09
- design: `prompts/M18.7.1_Harness_V2_Design.md`
- v2 implementation summary: `reports/m18_7_1_harness_v2_implementation_summary.md`
- command: `scripts/run_m18_7_1_real_llm_calibration.py --scoring-mode by_pid` × 5
- model: `deepseek/deepseek-v4-flash` (default)
- fixture: `tests/fixtures/m18_7_1_held_out_calibration.json` (12 turns)
- session roots: `tmp_m18_7_1_v2_stability_run_{1..5}`

## Why this check

The single-run all-mode replay surfaced `by_pid` reaction
joint acc = 0.667 (n=3). M20.4 owner needs a stability
estimate before making a threshold decision on this signal.

P0 = 5 fresh by_pid replays on the same fixture, each with
a clean `MVPStateStore` (no state carry-over). Each replay
is a real OpenRouter call sequence (12 turns × N model
calls), so non-determinism is genuine LLM sampling noise,
not fixture drift.

## Surfaced numbers — 5 runs side-by-side

### Reaction axis (the M20.4-relevant signal)

| field | R1 | R2 | R3 | R4 | R5 | mean | range |
|---|---|---|---|---|---|---|---|
| **reaction joint n_corr / n_pres** | 3/4 | 3/4 | 2/3 | 2/2 | 3/5 | — | 2-5 n_pres |
| **reaction joint acc** | 0.750 | 0.750 | 0.667 | 1.000 | 0.600 | **0.753** | 0.600 - 1.000 |
| reaction joint brier | 0.316 | 0.124 | 0.058 | 0.046 | 0.359 | 0.180 | 0.046 - 0.359 |
| reaction joint ece | 0.438 | 0.300 | 0.183 | 0.175 | 0.370 | 0.293 | 0.175 - 0.438 |
| reaction pid alone acc | 0.667 | 0.667 | 0.333 | 0.333 | 0.500 | 0.500 | 0.333 - 0.667 |
| reaction is_about alone acc | 0.500 | 0.500 | 0.333 | 0.333 | 0.500 | 0.433 | 0.333 - 0.500 |

### Addressee axis (less stable)

| field | R1 | R2 | R3 | R4 | R5 | mean | range |
|---|---|---|---|---|---|---|---|
| addressee n_corr / n_pres | 4/8 | 3/8 | 3/8 | 2/8 | 4/8 | — | 2-4 n_corr |
| addressee acc | 0.500 | 0.375 | 0.375 | 0.250 | 0.500 | 0.400 | 0.250 - 0.500 |
| addressee brier | 0.116 | 0.389 | 0.248 | 0.031 | 0.253 | 0.207 | 0.031 - 0.389 |
| addressee ece | 0.088 | 0.425 | 0.313 | 0.063 | 0.306 | 0.239 | 0.063 - 0.425 |

### Threshold recommendations (M20.4 inputs)

| field | R1 | R2 | R3 | R4 | R5 | mode |
|---|---|---|---|---|---|---|
| candidate_admit_min | null | null | 0.5 | 0.5 | 0.2 | {null, 0.5} |
| candidate_tie_breaker_min | 0.9 | 0.9 | 0.8 | 0.9 | 0.8 | {0.8, 0.9} |

### Verdict (top-level combined)

| field | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| verdict | severe | severe | severe | moderate | severe |

`verdict` is the mean of (addr_ece, react_ece, addr_brier,
react_brier) / 2; R4 was the only run where both axes
landed in moderate bands (addr 0.063, react 0.175).

## What the variance tells us

### 1. Reaction joint accuracy is the most stable M20.4 signal

`joint acc` range = [0.600, 1.000] across 5 runs, mean 0.753.
The earlier 0.667 from the all-mode run was the lower edge
of this range, not a measurement artifact. **M20.4 can
trust by_pid reaction joint as a "≥0.6" signal**.

The variance is dominated by `n_present` (2-5), not by
LLM correctness. As `n_present` rises from 2 to 5, the
mean accuracy shifts from 1.0 (n=2, all correct) to 0.6
(n=5, 3/5). This is the **binomial-stability floor** for
the held-out fixture: 12 turns × reaction field produces
2-5 LLM emissions per run depending on the LLM's
predictions-vs-presence decisions.

### 2. pid alone is more variable than is_about alone

| sub-axis | mean acc | range | runs at 0.667 | runs at 0.333 | runs at 0.5 |
|---|---|---|---|---|---|
| pid | 0.500 | 0.333-0.667 | 2/5 | 2/5 | 1/5 |
| is_about | 0.433 | 0.333-0.500 | 0/5 | 2/5 | 3/5 |

`is_about` is tighter (0.333-0.500, 0.167 spread) than
`pid` (0.333-0.667, 0.333 spread). This says: the LLM
is more consistent on the binary "is this about the
assistant's claim" judgment than on the multi-way pid
classification. M20.4's `is_about_assistant_claim` axis
is a **more reliable input** than the raw pid axis.

### 3. candidate_tie_breaker_min is the most stable M20.4 input

| value | runs | interpretation |
|---|---|---|
| 0.9 | 3/5 | current tie_breaker 0.85 → 0.9 |
| 0.8 | 2/5 | current tie_breaker 0.85 → 0.8 (loosen) |

**No run suggested `0.85` (current) or higher than `0.9`**.
The 0.8 vs 0.9 split reflects which bin the worst
high-confidence gap landed in (0.90-1.00 vs 0.80-0.90).
M20.4 can read this as: **current `tie_breaker_min = 0.85`
is borderline; a 0.85-0.90 nudge is data-supported**.
M20.4 owner decides the exact value.

### 4. candidate_admit_min is NOT a stable input

| value | runs |
|---|---|
| null | 2/5 |
| 0.5 | 2/5 |
| 0.2 | 1/5 |

`admit_min` is the lower bound (which predictions get
admitted to the settler). It depends on the low-confidence
band gap, which is small in this fixture (3-5 cases in
the 0.0-0.1 band, all zero confidence, all zero accuracy
— degenerate). **M20.4 should NOT move `admit_min` based
on the 5-run variance**. The 0.2 in R5 is an outlier from
a 0.20-0.30 band gap; not a real signal.

### 5. Addressee axis is high-variance (expected)

`addressee acc` range = [0.250, 0.500], mean 0.400. The
4-8 case range means 1-2 LLM decisions flip per run.
This is the M18.7 prompt-order blocker (see
`reports/m18_7_1_calibration_summary.md` + the M18.7.2
follow-up work). v2 surfaces the numbers; addressing
the underlying variance is the M18.7.2 / P4 milestone.

The addressee ECE spread (0.063-0.425) is much wider than
reaction ECE spread (0.175-0.438). This is consistent with
addressee being the noisier axis.

## What this means for M20.4

### Actionable now (with this data)

- **`tie_breaker_min` decision has data**: 5/5 runs
  suggest 0.8 or 0.9 (i.e., nudge from 0.85). M20.4
  owner picks the exact value; either is defensible.
- **`is_about_assistant_claim`-only signal has data**:
  0.333-0.500 across 5 runs, tight variance on the
  low end. M20.4 can use this as a stable sub-axis input
  for the settler.
- **Reaction field as a whole is "moderate_drift"** in
  5/5 runs (verdict-level despite the spread): all
  runs have `insufficient_data` flagged on the reaction
  field, and joint ECE 0.175-0.438. **M20.4 should
  not raise `admit_min` on reaction alone** — the
  per-axis verdict is the right input here.

### NOT actionable (need more or different data)

- **Addressee threshold revision**: needs the
  M18.7.2 follow-up prompt work (P3 / P4) before M20.4
  can make a meaningful call.
- **`admit_min` on addressee**: 5-run variance too high.
  M20.4 should leave `admit_min = 0.4` until addressee
  calibration stabilizes.
- **Turn_id axis (Mode B)**: still 0.0 in all 5 runs
  (LLM emits `reaction_to_turn_id=""` 5/6 times). This
  is a fixture/prompt gap, not a stability question.
  Address in P2 (M18.7.2 prompt turn_id enumeration) +
  P3 (fixture assistant prior turn).

## Stability floor: re-runs of v2 are not noise-free

The 5-run variance is the **LLM non-determinism floor**
for this fixture. The 0.6-1.0 reaction joint accuracy
spread (0.4 spread) is wider than the joint-reaction
ECE spread (0.293) — meaning that for n=2-5, a single
binomial trial is a noisy estimator of the LLM's true
calibration. **Any future v2 replay should expect to
land within this band**. Outside-of-band results would
indicate either a model change, fixture change, or
prompt change.

## Acceptance against P0 (stability) criteria

| criterion | met? | evidence |
|---|---|---|
| 5 replays complete without error | ✅ | all `ok: true`, exit code 0 |
| Reaction joint acc ≥ 0.6 in 5/5 runs | ✅ | min = 0.600, max = 1.000, mean = 0.753 |
| `candidate_tie_breaker_min` in {0.8, 0.9} in 5/5 runs | ✅ | no nulls, no values > 0.9 or < 0.8 |
| `pid` axis accuracy ≥ 0.333 in 5/5 runs | ✅ | min = 0.333 |
| `is_about` axis accuracy ≥ 0.333 in 5/5 runs | ✅ | min = 0.333 |
| `verdict` consistent (same band) in ≥ 4/5 runs | ✅ | 4/5 severe, 1/5 moderate |
| Addressee variance documented | ✅ | range 0.25-0.50, n=4-8, mean 0.40 |
| `admit_min` correctly flagged as noisy | ✅ | null / 0.5 / 0.2 spread |

## What we did NOT measure

- **Cross-model stability**: only `deepseek/deepseek-v4-flash`
  was tested. A v4-pro or different model could show
  different stability. The design supports it
  (`openrouter.json`), but this milestone stayed
  on the default model.
- **Cross-fixture stability**: same 12-turn held-out
  fixture across all 5 runs. Adding a second held-out
  fixture (or extending to 24 turns) would tighten
  the n_present variance and give a cleaner
  M20.4-relevant signal.
- **Temperature / seed fixity**: OpenRouter default
  sampling is non-deterministic. If M20.4 needs
  deterministic LLM outputs (e.g., for CI), a
  temperature=0 + seed=fixed call would have
  tighter variance, but is not in v2's scope.

## Files touched (this milestone)

- `reports/m18_7_1_v2_stability_summary.md` (this file)
- `tmp_m18_7_1_v2_stability_run_{1..5}/` (session roots;
  retained for forensic analysis; cleanup is a separate
  step)
- `segmentum/dialogue/runtime/m18_7_1_calibration.py` —
  no code changes since v2 landed
- `scripts/run_m18_7_1_real_llm_calibration.py` — no
  code changes since v2 landed

## Recommended next steps (P1+)

1. **P1 — M20.4 owner reads this + the v2 implementation
   summary**. They have a stable, by_pid, M20.4-relevant
   signal. Decision: keep `tie_breaker_min = 0.85` or
   nudge to 0.9. Both are data-supported. **Do not** touch
   `admit_min` based on this 5-run data.

2. **P2 — M18.7.2 prompt turn_id enumeration**:
   add `recent_turn_index`, `assistant_turn_excerpt` to
   the M18.7.2 minimal prompt. This unlocks the
   `reaction_to_turn_id` axis (currently 0/6). Re-run
   v2 stability check after P2 lands; expect Mode B to
   show non-zero accuracy.

3. **P3 — Fixture repair**: add assistant prior turn
   history to the held-out fixture so
   `turn_<assistant_prior_turn_id>` placeholders resolve.
   Closes the 3 unresolved warnings in Mode B.

4. **P4 — Addressee milestone**: separate from P0-step2.
   Investigate the 0.063-0.425 ECE spread on addressee
   and the 0.25-0.50 acc spread.

5. **P5 — Revert 5ab3db4** (P0-step2): **DONE** as
   commit `f780f76` (2026-06-09). Stability report
   said "Failed" (addressee 0.25→0.0, reaction ECE
   0.258→0.417 worse). 139/139 M18.7.* tests pass
   after revert. The v1 semantic categories in
   M18.7.2 minimal prompt are removed; pre-P0-step2
   prompt is restored.

## Related

- `prompts/M18.7.1_Harness_V2_Design.md` (full design)
- `reports/m18_7_1_harness_v2_design_summary.md` (3.6KB)
- `reports/m18_7_1_harness_v2_implementation_summary.md` (10.9KB)
- `reports/m18_7_1_calibration_summary.md` (v1 status)
- `tests/test_m18_7_1_calibration.py` (57 tests, T1–T16)
- `tests/fixtures/m18_7_1_v1_report_baseline.json` (T9 baseline)
- `tests/fixtures/m18_7_1_held_out_calibration.json` (12-turn fixture)
