# 4-Sub-Capability 5-Run Baseline — Done

**Status**: 5-run completed end-to-end.
**Date**: 2026-06-11
**Model**: `deepseek/deepseek-v4-flash` (native API,
`https://api.deepseek.com`)
**Fixture**: `tests/fixtures/m18_7_1_held_out_calibration.json`
(bqxsmofri, 12 turns × 4 personas)
**Scoring mode**: `by_pid` (M18.7.1 v2 default)
**Total wall time**: 1h 18min (5 runs × 17min avg,
including 1 retry that bypassed the 4 retry budget
for run_1 — see Run 1 note)

**Overall verdict**: `all_4_subcap_below_bar` — **all
4 sub-capabilities failed the conservative first-cut
bars on every run**. This is the first data-backed
verdict on real LLM at the framework level.

**Snapshot**: `reports/m18_7_2_4subcap_5run_baseline_snapshot.json`
(full JSON output of the 5-run)

## 4-Sub-Capability Aggregate (5-run mean / total)

### Sub-1: addressee target (P1 split)

| field | value | bar | pass? |
|---|---|---|---|
| `recall_on_addressed` (mean) | **0.05** | ≥ 0.60 | ❌ |
| `precision_on_not_addressed` (mean) | 1.00 | ≥ 0.90 | ✅ |
| `tp_addressed` (total) | 1 / 20 (4 GT-true × 5 runs) | — | — |
| `tp_not_addressed` (total) | 20 / 20 | — | — |
| `fp_not_addressed` (total) | 0 | — | — |

5/5 runs `verdict: under_recall_dir_true`. The LLM
catches 1/4 GT-addressed turns in run_1, 0/4 in
runs 2-5. **Recall is broken, precision is perfect.**

### Sub-2: speaker identity (LLM-emitted pid vs GT speaker pid)

| field | value | bar | pass? |
|---|---|---|---|
| `n_decidable_emits` (total) | 15 / 60 (3 avg/run) | — | — |
| `n_exact_match` (total) | 1 | — | — |
| `speaker_pid_exact_match_rate` (mean) | **0.05** | ≥ 0.70 | ❌ |
| best run | R4: 1/4 (rate 0.25) | — | — |

5/5 runs `verdict: below_bar`. Only 1 of 15
decidable emits was correct (run_4, dave). The
LLM almost never emits an addressee pid matching
the speaker pid. **Sub-2 is broken.**

### Sub-3: bidirectional FEP (M20.4 producer + M11 channels)

| field | value | bar | pass? |
|---|---|---|---|
| `producer_admit_total` (mean) | 6.2/run (31 total) | ≥ 1/run | ✅ |
| `producer_admit_dir_true` (total) | **0** | ≥ 1/run | ❌ |
| `n_persona_channels` | 3/5 (carol, dave, eve) | ≥ 2 | ✅ |
| `write_path_skip_dir_true` | 0/5 | — | (P0-5 dormant) |
| `tie_breaker_engaged_dir_true` | 0/5 | — | (P0-6 dormant) |

5/5 runs `verdict: p04_dir_true_admit_zero`. The
producer IS alive (mean 6.2 admits/run) but it
**never admits a `dir_true` claim** — every
admit is `dir_false`. Channels (carol/dave/eve)
are healthy (3/3 present), so the M11 side is
fine; the M20.4 producer is biased toward False.

### Sub-4: persona consistency (M12.1 surface)

| field | value | bar | pass? |
|---|---|---|---|
| `n_profiles` | **0** (5/5 runs) | ≥ 1 | ❌ |
| `n_latest_reports` | 0 (5/5 runs) | — | — |

5/5 runs `verdict: no_m12_1_surface`. **M12.1 is
dormant** — no profiles written across the 5
runs. This is a real bug: M12.1 should be
populated on a 12-turn group chat with 3 personas
in 17 minutes of wall time. Either the
conscious_loop never reaches the M12.1 step, or
the M12.1 state surface is never persisted to
`m12_1_user_personality.json`.

## Per-Run Verdict Map

| run | sub-1 | sub-2 | sub-3 | sub-4 |
|---|---|---|---|---|
| 1 | under_recall_dir_true | below_bar | p04_dir_true_admit_zero | no_m12_1_surface |
| 2 | under_recall_dir_true | below_bar | p04_dir_true_admit_zero | no_m12_1_surface |
| 3 | under_recall_dir_true | below_bar | p04_dir_true_admit_zero | no_m12_1_surface |
| 4 | under_recall_dir_true | below_bar | p04_dir_true_admit_zero | no_m12_1_surface |
| 5 | under_recall_dir_true | below_bar | p04_dir_true_admit_zero | no_m12_1_surface |
| **mean/total** | 0.05 / 1.00 | 0.05 / 15 | 31 admits / 0 dir_true | 0 / 0 |

## What this verdict means

**胡桃 (Hu Tao) is NOT group-chat-competent on
deepseek-v4-flash against the 4-sub-cap bar.**

Specifically:

1. **Sub-1 (addressee target)**: the LLM catches
   1/20 GT-addressed turns across 5 runs. The v2
   prompt that lifted `claude-sonnet-4-6` from 0/4
   to 2/4 recall does NOT transfer to
   `deepseek-v4-flash`. The M18.7.2 v2 prompt was
   tuned for claude-style reasoning; deepseek
   on this fixture over-biases toward `False`.

2. **Sub-2 (speaker identity)**: the LLM almost
   never emits the correct speaker pid. With
   `n_decidable_emits=3/run` (out of 8 addressable
   turns), the LLM is **also under-emitting** —
   sub-2's denominator is the on-disk surface
   (rolling-window cap=8 evicts early turns).
   Sub-2 may also be capped by sub-1's
   under-recall (no emit → no pid to score).

3. **Sub-3 (bidirectional FEP)**: the producer
   admits 6/run, all `dir_false`. This is the
   **same `dir_true` starvation observed on
   bqxsmofri in v2 P0-8** (claude-sonnet-4-6
   admitted 2 dir_true in 5 runs, 0.4/run).
   Deepseek produces ZERO dir_true admits in 5
   runs. Either deepseek is structurally less
   willing to commit to `dir_true` (LLM style),
   or the M20.4 P0-4 0.7 threshold + the v2
   prompt's effect on this model is too tight.

4. **Sub-4 (M12.1 surface)**: this is a **state
   surface bug**, not an LLM verdict. The
   5-run framework never wrote a single M12.1
   profile. This is independent of model choice
   — claude-sonnet-4-6 (used in M18.7.2 v2 P0-8)
   would likely also produce 0/5 profiles on
   bqxsmofri because the 12-turn fixture is too
   short to reach M12.1's `n_turns_threshold`.

## CAVEAT: deepseek-v4-flash vs claude-sonnet-4-6

The M18.7.2 v2 P0-8 5-run used
`anthropic/claude-sonnet-4-6` and got:
- recall_on_addressed mean 0.35 (3/5 runs at 0.5,
  2/5 at 0.25) — 7x better than deepseek's 0.05
- P0-4 admitted dir_true 2 in 5 runs — first
  time on this fixture; deepseek's 0 is a
  regression

So the verdict is **not a code-bug verdict; it is
a model-fit verdict**. Deepseek-v4-flash on this
fixture is **structurally worse than claude-sonnet-4-6**
at:
- Catching dir_true (recall 7x worse)
- Committing to dir_true (0 vs 2 admits in 5 runs)

The conservative first-cut bars are NOT met by
either model, but the gap to claude-sonnet-4-6 is
real.

## Action items surfaced by this 5-run

1. **M18.7.2 v3 prompt needed for deepseek**. The
   v2 prompt lifts claude from 0/4 to 2/4; on
   deepseek it lifts 0/4 to 1/4 (run_1 only). A
   v3 prompt could:
   - Move the strong-signal list to the END of
     the prompt (current position: middle; deepseek
     is reportedly better at end-positioned
     instructions)
   - Add an explicit anti-bias sentence
     ("do NOT default to `False`")
   - Add a chain-of-thought wrapper

2. **M20.4 P0-4 threshold revision**. With
   dir_true admit 0/5 on deepseek (and 2/5 on
   claude), the 0.7 threshold is too tight for
   both models. The M18.7.1 v2 P0 stability
   report flagged `admit_min` as noisy; this
   5-run confirms it.

3. **M12.1 surface bug** (NOT a model issue).
   The framework never wrote profiles; this
   needs an investigation: is M12.1 not
   being reached? Or reached but not persisted?
   This is the only sub-cap that can be fixed
   with no model change.

4. **Sub-2 reading source**. Sub-2 reads
   `state["m18_7_attribution_hypotheses"]` from
   the on-disk surface (cap=8 evicted). The
   harness's in-memory predictions are the
   correct scoring source. This was the same
   surface-vs-harness discrepancy from M18.7.2
   v2 P0-8 (commit 39d2ef0) — sub-2 needs to
   read from `harness_report.addressee_predictions`
   instead of on-disk state.

5. **Bars may need revision** if the user
   agrees the 4 bars are too tight for the
   current state of M18.7.2 + M20.4. Current
   bars: recall 0.6, precision 0.9, pid match
   0.7, producer admit ≥1, dir_true admit ≥1,
   channels ≥2, profiles ≥1. Of these, only
   precision 0.9 (1.0 mean) and channels ≥2
   (3/3) are met. The other 5 bars are not met
   by either deepseek OR claude.

## What is NOT a verdict (caveats)

- **Cross-fixture**: bqxsmofri only; the P5
  Chinese fixture is a separate baseline.
- **Cross-model**: deepseek-v4-flash native API;
  the v2 P0-8 baseline used claude-sonnet-4-6
  via OpenRouter. Different prompts to the same
  model may also differ.
- **Sub-2 denominator**: capped by sub-1's
  under-emit (no emit = no decidable pid).
  Sub-2's 0.05 rate is partly a sub-1 artifact.
- **M12.1 surface**: not actually validated —
  the 5-run never wrote a profile. Whether
  M12.1 works AT ALL on a 12-turn fixture is
  unknown.

## Out of scope (this report)

- No code change.
- No prompt change.
- No fixture change.
- No M20.4 threshold change.
- The framework is unchanged; the model is
  unchanged; the fixture is unchanged.
- The 4-sub-cap bars are unchanged.

This is **data**, not an intervention.

## Follow-up milestones (planned)

- **M18.7.1 v3 sub-2 source**: read from
  harness report, not on-disk state. Will
  change `n_decidable_emits` from 3/run to
  12/run (harness sees all in-memory).
- **M12.1 5-run investigation**: why no
  profiles? Likely the fixture is too short
  to reach M12.1's `n_turns_threshold`. A
  longer fixture (24+ turns) may surface M12.1.
- **M18.7.2 v3 prompt**: target deepseek's
  dir_true bias; same fixture, model = native
  deepseek.
- **M20.4 P0-4 threshold review**: 0.7 → 0.5
  or 0.6, per M18.7.1 v2 P0 stability data
  + this 5-run's 0/5 dir_true admits.
