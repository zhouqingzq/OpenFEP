# M18.7.2 v2 P0-8 — 5-Run Stability Rerun Report

- status: experimental (validates v2 prompt revision stability
  on bqxsmofri; this is the M18.7.2 v2 follow-up to P0-7)
- branch: main
- generated_at: 2026-06-10T17:30:00+00:00
- held_out_fixture: tests/fixtures/m18_7_1_held_out_calibration.json
- scoring_mode: by_pid (v2)
- model_under_test: anthropic/claude-sonnet-4.6
  (real LLM via `default_openrouter_client`)

## Question this report answers

> "Is the M18.7.2 v2 prompt revision (commit 479c3e2)
> stable across 5 runs on the bqxsmofri fixture? Does
> the R1 single-replay's `recall_on_addressed = 0.5`
> reproduce, or was it a fluke?"

**TL;DR: v2 lifts the recall band ceiling (0.25 → 0.5)
and floor (0.0 → 0.25) but is NOT a stable fix. Recall
varies 0.25-0.5 across 5 runs. The verdict signal
(`severe_drift_recommend_m20_4`) is unchanged in 5/5
runs. P0-4 fired once (R2) and admitted 2 dir=True
predictions (R4, R5) in 5 runs.**

## 5-run stability findings (bqxsmofri, real LLM, by_pid)

### Addressee (n_present=8 in all 5)

| run | acc | brier  | ece   | verdict                                |
|-----|-----|--------|-------|----------------------------------------|
| R1  | 0.75 | 0.254 | 0.275 | severe_drift_recommend_m20_4          |
| R2  | 0.625| 0.496 | 0.531 | severe_drift_recommend_m20_4          |
| R3  | 0.625| 0.566 | 0.581 | severe_drift_recommend_m20_4          |
| R4  | 0.75 | 0.263 | 0.306 | severe_drift_recommend_m20_4          |
| R5  | 0.75 | 0.491 | 0.506 | severe_drift_recommend_m20_4          |
| **mean** | **0.700** | **0.414** | **0.440** | **5/5 severe_drift** |

### Addressee P1 split (the headline signal)

| run | precision_on_not_addressed | recall_on_addressed | tp_addr | tp_not_addr | fp_not_addr |
|-----|----------------------------|---------------------|---------|-------------|-------------|
| R1  | 1.0 | **0.5** | 2 | 4 | 0 |
| R2  | 1.0 | 0.25 | 1 | 4 | 0 |
| R3  | 1.0 | 0.25 | 1 | 4 | 0 |
| R4  | 1.0 | **0.5** | 2 | 4 | 0 |
| R5  | 1.0 | **0.5** | 2 | 4 | 0 |
| **5/5** | **1.0 (5/5)** | **0.25-0.5** | **8 total** | **20 total** | **0 total** |

### Reaction joint (n_present 4-6)

| run | acc | brier | ece | joint_all_dec | joint_emit | no_emit_wrong | acc_joint_all |
|-----|-----|-------|-----|---------------|------------|---------------|---------------|
| R1  | 0.5 | 0.150 | 0.350 | 6 | 4 | 2 | 0.333 |
| R2  | 0.5 | 0.226 | 0.408 | 6 | 6 | 0 | 0.500 |
| R3  | 0.4 | 0.197 | 0.200 | 6 | 5 | 1 | 0.333 |
| R4  | 0.5 | 0.296 | 0.463 | 6 | 4 | 2 | 0.333 |
| R5  | 0.5 | 0.174 | 0.338 | 6 | 4 | 2 | 0.333 |
| **mean** | **0.48** | **0.209** | **0.352** | **6 (5/5)** | **4.6** | **1.4** | **0.367** |

### M20.4 per-sub-class counters (5-run totals)

| counter                                        | total | per-run                       |
|------------------------------------------------|-------|-------------------------------|
| producer_admit_total                           | -     | (varies)                      |
| producer_admit_addressee_directed_total        | 2     | [0, 0, 0, 1, 1]               |
| producer_admit_addressee_not_directed_total    | -     | (varies)                      |
| producer_reject_low_confidence_addressee_directed_total | 1 | [0, 1, 0, 0, 0] |
| producer_reject_low_confidence_addressee_not_directed_total | - | (varies) |
| producer_reject_disclosure_total               | -     | (varies)                      |
| write_path_skip_addressee_directed_low_confidence_total | 0 | [0, 0, 0, 0, 0] |
| tie_breaker_engaged_addressee_directed_total   | 0     | [0, 0, 0, 0, 0]               |

**P0-4 fired 1/5 in v2 (R2 only)** — same frequency as
v1 P0-7 (R2 only). P0-4 also **admitted 2 dir=True
predictions** in 5 runs (R4, R5; the first time we
see dir=True admits on this fixture under v2). P0-5
and P0-6 are still **dormant 5/5** (0 fires; 0
write-path skips).

### Threshold recommendation

| run | candidate_admit_min | candidate_tie_breaker_min |
|-----|---------------------|---------------------------|
| R1  | null | 0.8 |
| R2  | 0.3 | null |
| R3  | null | null |
| R4  | null | 0.8 |
| R5  | null | 0.8 |

`candidate_tie_breaker_min = 0.8` in 3/5 (R1, R4, R5)
— same band as v1 P0-7 (0.8-0.9 in 5/5).
`candidate_admit_min` is too sparse to surface
consistently.

## Comparison vs v1 P0-7 5-run band (M20.4 P0-7 stability)

| metric                       | v1 P0-7 (5-run) | v2 P0-8 (5-run) | delta |
|------------------------------|-----------------|-----------------|-------|
| addr acc (mean)              | 0.600           | 0.700           | **+0.100** |
| addr brier (mean)            | 0.423           | 0.414           | -0.009 |
| addr ece (mean)              | 0.458           | 0.440           | -0.018 |
| addr recall_on_addressed band | 0.0-0.25       | 0.25-0.5        | **floor +0.25, ceiling +0.25** |
| addr recall_on_addressed (3/5 mode) | 0.25     | 0.5             | **+0.25** |
| addr precision_on_not_addressed | 1.0 (5/5)   | 1.0 (5/5)       | unchanged |
| verdict: severe_drift        | 5/5             | 5/5             | unchanged |
| P0-4 fires (5-run total)     | 1               | 1               | unchanged |
| P0-4 admits dir=True (5-run total) | 0        | 2               | **+2** |
| P0-5/P0-6 fires (5-run total) | 0              | 0               | unchanged |

**Two signals improved**:

1. **Recall band ceiling + floor both lifted by 0.25**.
   v1: 0.0 (R1 outlier) + 0.25 (4/5). v2: 0.25 (2/5)
   + 0.5 (3/5). The v2 prompt revision is moving
   the LLM toward more dir=True predictions, but
   inconsistently.
2. **P0-4 admits dir=True for the first time** (2 in
   5 runs). On v1 P0-7 5 runs, dir=True was never
   admitted (0 in 5). v2's 2 admits (R4, R5) are at
   conf 0.7+, which is the P0-4 admit threshold.
   This confirms v2 is producing dir=True predictions
   that pass the P0-4 admit gate.

**Three signals unchanged**:

1. **precision_on_not_addressed = 1.0 in 5/5** (v1
   and v2). The LLM remains perfect on "not addressed"
   claims.
2. **`severe_drift_recommend_m20_4` verdict in 5/5**
   (v1 and v2). ECE 0.44 mean > 0.15 threshold.
3. **P0-4 fires 1/5 runs** (R2 only, both v1 and v2).
   P0-4 still dormant 4/5 — the LLM is not flooding
   the producer with low-conf dir=True predictions.

## What v2 R2-R5 confirms vs contradicts

**Confirms (R1 + R4 + R5)**:
- v2 prompt can lift `recall_on_addressed` to 0.5 on
  bqxsmofri (3/5 runs). The strong-signal
  enumeration works when the LLM attends to it.
- `precision_on_not_addressed = 1.0` is preserved.
- The drift signature is structurally unchanged but
  the high-band accuracy improved (R1, R4 are the
  cleaner runs).

**Contradicts (R2 + R3)**:
- v2 prompt is NOT a stable fix. R2 and R3 dropped
  back to 0.25 recall, same as v1 P0-7 mean.
- v2 R1's `recall_on_addressed = 0.5` was not a
  fluke of 1/5; 3/5 runs hit it. But it's also not
  a stable 5/5.
- The drift verdict is unchanged. v2 R2 and R3 are
  WORSE on ECE than v1 P0-7 mean (0.531, 0.581 vs
  0.458). v2 can hurt when the LLM attends to the
  strong-signal list and over-applies it.

**Net assessment**:
- v2 is a **noisy improvement** on recall ceiling,
  not a stable fix. The structural issue (LLM is
  under-confident on `addressed_to_assistant=True`)
  is partially addressed by the strong-signal list
  but the LLM is not consistently attending to it.
- 3/5 runs at 0.5 recall is a meaningful signal
  that v2 is the right direction, but 2/5 runs at
  0.25 (same as v1 mean) means v2 cannot replace
  v1 as the production prompt without further
  refinement.

## What v2 R2-R5 does NOT cover

- **Cross-language**: P0-8 is bqxsmofri (English) only.
  P5 Chinese fixture is separate.
- **Cross-model**: anthropic/claude-sonnet-4.6 only.
  P0-7 used deepseek/deepseek-v4-flash. v2 behavior
  may differ on other models.
- **M20.4 threshold revision**: v2 does not change the
  P0-4 (0.7) / P0-5 (0.9) / P0-6 (0.95) thresholds.
  The 2 dir=True admits in R4, R5 are at conf 0.7+;
  if the 0.25-recall runs had emitted dir=True at
  conf 0.7+, the P0-4 admit would have surfaced them
  too.
- **Reaction-axis improvement**: v2 reaction n_present
  4-6 (vs v1 P0-7 3-5), acc 0.4-0.5 (vs v1 0.167-0.333
  on joint_all_decidable). v2 is slightly better on
  reaction joint but still ECE 0.35 mean > 0.15.

## Recommendation

**v2 prompt is the right direction but not yet ready
to replace v1 as the production default.** Three
options for the next step:

1. **Keep v2 as the prompt, accept the 0.25-0.5
   recall band as the new noisy normal.** M20.4's
   P0-4/P0-5/P0-6 sub-class split is the safety net;
   0.25 floor is not zero, so the LLM is still
   catching some dir=True cases per run. This is the
   "ship and observe" path.

2. **v3 prompt revision: add a 4th inline example
   covering the "implicit directive" case
   (turn 4: 'Someone from the team is reading this').
   This is the dominant false-negative in v1 P0-7
   (1 fn_addressed_noemit per run was the floor).
   v3 would target the 0.25-recall floor specifically.**

3. **v2 + post-LLM filter: when the LLM emits
   `addressed_to_assistant=False` on a turn where
   the user_text contains a directive pattern
   ("can you", "could you", "could you please",
   "?"), down-weight the prediction by 0.1.
   This is an M18.7.1 measurement-side adjustment,
   not a prompt change.**

**M20.4 owner decides**. M18.7.2 v2 is a data point,
not a decision.

## CAVEAT: on-disk surface vs harness prediction count

The on-disk `m18_7_attribution_hypotheses.json` file
shows fewer addressee entries (4-5 per run) than the
harness's `addressee_predictions` list (8 on
decidable turns). This is the rolling-window cap
`M18_7_STATE_SURFACE_CAP=8` evicting early-turn addr
entries before the surface was persisted at end-of-run.
The harness's view is the **correct** one for
scoring — it reads the surface during each turn.
The on-disk file is the final-8 snapshot. (See
`reports/m18_7_2_v2_prompt_replay_summary.md` for the
full investigation log.)

## CAVEAT: P0-8 is a single-fixture, single-model 5-run

This 5-run is on bqxsmofri (English) with
claude-sonnet-4.6. v2 stability on other fixtures
(P5 Chinese, future M18.7.2 variants) is unverified.
P0-8 R1-R5 are the first 5-run v2 stability data;
cross-fixture / cross-model validation is a separate
milestone.

## Files

- Run outputs: `tmp_m18_7_2_v2_p0_8_run_{1..5}.output`
  (R1 is in `reports/m18_7_2_v2_p0_8_r1_snapshot.json`
  for the v2 R1 commit)
- Session roots: `tmp_m18_7_2_v2_p0_8_run_{1..5}/`
- Helper: `segmentum/tools/extract_p0_7_run_summary.py`
  (reused; reads by_pid reports the same way)
- Commit: 479c3e2 (v2 prompt implementation)
- Commit: 89f319a (v2 R1 report)

## Related

- [[project-m18-7-2-v2-r1-landed]] — v2 R1 single
  replay (recall 0.5, acc 0.75, ECE 0.275; 1-run).
- [[project-m18-7-1-p1-landed]] — P1 precision/recall
  split definition.
- `reports/m20_4_p0_7_stability_rerun.md` — v1 P0-7
  5-run stability verdict (the reference band).
- `reports/m18_7_2_v2_prompt_replay_summary.md` —
  v2 R1 report (single-run).
