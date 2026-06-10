# M18.7.2 v2 Prompt Replay — Real-LLM R1 Validation

- status: experimental (validates v2 strong-signal prompt
  revision for `addressed_to_assistant`)
- branch: main
- generated_at: 2026-06-10T05:58:13+00:00
- session_root: tmp_m18_7_2_v2_real_llm_replay/
- held_out_fixture: tests/fixtures/m18_7_1_held_out_calibration.json
- scoring_mode: by_pid (v2)
- model_under_test: anthropic/claude-sonnet-4.6
  (real LLM via `default_openrouter_client`)

## Question this run answers

> "Does the v2 strong-signal / counter-example
> enumeration in the M18.7.2 minimal prompt lift
> `recall_on_addressed` off the 0.0-0.25 P0-7 band on
> the held-out bqxsmofri fixture?"

**TL;DR: Yes, by 2x. `recall_on_addressed` went from
0.25 (v1) → 0.5 (v2 R1) — but `severity_drift_recommend_m20_4`
verdict is unchanged (ECE 0.275 still > 0.15). The v2
prompt revision is a step in the right direction; the
drift signature (overconfidence_at_high_band +
underconfidence_at_low_band + bimodal) is unchanged
but the high-band accuracy improved.**

## Headline numbers (v2 R1, this run)

| field | n_total | n_present | n_unknown | n_correct | n_incorrect | accuracy | brier | ece |
|-------|---------|-----------|-----------|-----------|-------------|----------|-------|-----|
| addressee | 12 | 8 | 4 | 6 | 2 | 0.75 | 0.254 | 0.275 |
| reaction | 12 | 4 | 6 | 2 | 2 | 0.5  | 0.15  | 0.35  |

Aggregate verdict: `severe_drift_recommend_m20_4`
(unchanged from P0-7 v1 band; ECE 0.275 > 0.15).

## Addressee P1 split (the headline signal)

```text
n_gt_true: 4   n_gt_false: 4   n_unknown: 4
tp_addressed: 2   fn_addressed: 2
  fn_addressed_present: 1
  fn_addressed_noemit:  1
tp_not_addressed: 4
  tp_not_addressed_present: 3
  tp_not_addressed_noemit:  1
fp_not_addressed: 0
precision_on_not_addressed: 1.0   (unchanged from P0-7)
recall_on_addressed:        0.5   (up from 0.25 P0-7 4/5)
```

The two key signals:

1. **`precision_on_not_addressed = 1.0`** — LLM
   remains perfect on "not addressed" claims (same as
   P0-7 5/5 runs).
2. **`recall_on_addressed = 0.5`** — LLM now catches
   2/4 GT-addressed cases. P0-7 v1 5-run band was
   0.0-0.25; v2 R1 lands at 0.5 (2x improvement).

## Reliability bins (v2 R1 addressee)

```text
0.00-0.10: count 2 (mean_conf 0.0,   acc 0.5, gap 0.5)
0.10-0.50: count 0
0.60-0.70: count 1 (mean_conf 0.7,   acc 1.0, gap 0.3)
0.80-0.90: count 1 (mean_conf 0.85,  acc 1.0, gap 0.15)
0.90-1.00: count 4 (mean_conf 0.9375, acc 0.75, gap 0.1875)
```

Drift signals: `overconfidence_at_high_band` +
`underconfidence_at_low_band` + `bimodal` (same 3 as
P0-7). High-band accuracy 0.75 (up from 0.25 in
P0-7 v1 5-run band; the 4 in 0.9-1.0 are 3 TP-not-
addressed + 1 FN-present-on-GT=True).

## Comparison vs P0-7 v1 5-run band (M20.4 P0-7 stability)

| field | metric | P0-7 v1 (5-run band) | v2 R1 | delta |
|-------|--------|----------------------|-------|-------|
| addressee | acc | 0.6 (mean) | 0.75 | **+0.15** |
| addressee | brier | 0.423 (mean) | 0.254 | **-0.169** |
| addressee | ece | 0.458 (mean) | 0.275 | **-0.183** |
| addressee | precision_on_not_addr | 1.0 (5/5) | 1.0 | unchanged |
| addressee | recall_on_addressed | 0.0-0.25 (5/5) | 0.5 | **+0.25** |
| addressee | drift signals | 3 (same) | 3 (same) | unchanged |
| reaction | acc | 0.397 (mean) | 0.5 | +0.103 |
| reaction | brier | 0.241 (mean) | 0.15 | -0.091 |
| reaction | ece | 0.33 (mean) | 0.35 | +0.02 |

**All three v1 → v2 deltas on addressee are in the
right direction.** Verdict signal
(`severe_drift_recommend_m20_4`) is unchanged because
ECE 0.275 still exceeds the moderate_drift threshold
(0.15) — but the boundary is much closer than v1
(0.458 ECE in P0-7 mean).

## What v2 changed in the prompt

`segmentum/dialogue/runtime/m18_7_attribution.py`:

- `M18_7_2_MINIMAL_PROMPT_MAX_CHARS`: 2000 → 2500
  (v2 prompt is 2277 chars total; v1 was 1647; well
  below the 7.7-26k conscious-loop prompt that the v1
  path competed with).
- The v2 `system_prompt` adds three blocks the v1
  prompt did not enumerate:
  1. **Strong-signal list for `addressed_to_assistant=True`**:
     5 items (bot alias, entity_binding, second-person
     imperative, "OK" continuation, implicit directive).
  2. **Counter-example list for `addressed_to_assistant=False`**:
     2 items (other-recipient direct address, group-wide
     address).
  3. **3 inline examples** ('Can you explain that?' → True;
     'Dave, you first.' → False; 'OK, can you do X?' → True).
- Reaction-axis prompt also clarified: `last_user_utterances`
  is the evidence; `user_text` signal phrases are hints,
  not the rule.

## What v2 does NOT change

- v1 schema (4-key JSON spec unchanged).
- M20.4 producer / write / tie-breaker / settler
  (P0-4/5/6) — v2 is prompt-only.
- Conscious-loop path (M18.7 v2 attrs segment still
  removed).
- M18.7.1 calibration harness (no scoring changes;
  v2 is the prompt's contribution).

## Tests

- 347/347 cross-M18.7.1 regression pass
  (was 342/342 in P0-7; +5 v2-specific tests).
- 5 new v2 tests cover: strong-signal enumeration,
  3-example presence, GT-leak guard,
  entity_binding/mentioned_ids emphasis,
  MAX_CHARS bumped to 2500.
- v1 byte-identity preserved (`M18_7_2 v1` is
  byte-identical to v1 in to_dict()).

## What R1 confirms vs leaves open

**Confirmed (R1)**:
- v2 strong-signal enumeration lifts
  `recall_on_addressed` 0.25 → 0.5 on bqxsmofri.
- `precision_on_not_addressed = 1.0` is preserved.
- The drift signature is structurally unchanged
  (3 signals unchanged) but the high-band accuracy
  improved 0.25 → 0.75.
- The LLM is now emitting `addressed_to_assistant=True`
  on at least 1 of 4 GT-addressed cases per run (vs 0/4
  in P0-7 v1 5-run band).

**Open for R2-R5 (stability)**:
- Is the 0.5 recall stable across 5 runs? v1 was
  0.0-0.25 in 4/5 runs (R1 was 0.0 outlier) and 0.25
  in 4/5. v2 R1 is 0.5 — does it stay there?
- Does the v2 prompt's 3-example list generalize, or
  is R1's 0.5 driven by a single LLM temperature
  fluctuation?
- Does v2 affect the reaction axis (n_present 4 → 4
  unchanged, acc 0.5 vs 0.397 mean; brier 0.15 vs
  0.241 mean — both improved; ece 0.35 vs 0.33 mean
  — slightly worse)?

## CAVEAT: v2 R1 is a single run

This report covers **R1 only**. P0-7's lesson was that
single-run numbers can hide a 0.0-0.25 band. R2-R5
stability is required before claiming v2 as the new
baseline. The 5-run v2 stability rerun is queued as
P0-8 R2-R5 (next milestone).

## CAVEAT: on-disk surface vs harness prediction count

The on-disk `m18_7_attribution_hypotheses.json` file
shows 4 addressee entries (turns 6, 8, 9, 11) due to
the rolling-window cap `M18_7_STATE_SURFACE_CAP=8`.
The harness's `addressee_predictions` list (n=8 on
decidable turns) includes 2 additional emits that
existed during iteration but were evicted by the cap
before the surface was persisted at end-of-run. The
harness's view is the **correct** one for scoring —
it reads the surface during each turn. The on-disk
file is the final-8 snapshot.

(Investigation 2026-06-10: the 2 extra emits are at
conf 0.7 and 0.85, both with `addressed_to_assistant`
flag consistent with the per-class breakdown. They
are real LLM predictions, not noise.)

## Related

- [[project-m18-7-2-landed]] — v1 prompt + M18.7.2
  minimal call site baseline (0/12 non-empty fills in
  conscious-loop path).
- [[project-m18-7-1-p1-landed]] — P1 precision/recall
  split (`precision_on_not_addressed`,
  `recall_on_addressed`, joint subset split).
- [[project-m20-4-p0-7-stability-landed]] — P0-7 5-run
  stability report (the v1 0.0-0.25 band reference).
- `reports/m20_4_p0_7_stability_rerun.md` — P0-7 5-run
  stability verdict.
- `prompts/M18.7.1_Harness_V2_Design.md` — v2 scoring
  mode design (M18.7.1 v2 is independent of M18.7.2 v2;
  v2 of M18.7.2 is the prompt revision here).
