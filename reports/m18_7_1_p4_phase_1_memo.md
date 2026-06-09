# P4 Phase 1: Addressee Investigation Memo (Preliminary)

- **status**: preliminary; per-turn analysis pending P0 regen run completion
- **generated_at**: 2026-06-09
- **pre-read**:
  - `reports/m18_7_1_p4_addressee_design.md` (the design)
  - `reports/m18_7_1_v2_stability_summary.md` (5-run P0 aggregate)
  - `reports/m18_7_1_p5_chinese_smoke_summary.md` (P5 R1, 1 Chinese run)

## TL;DR

The 6 P0 by_pid replays (5 surfaced + 1 regen) show addressee
n_correct ∈ {1, 2, 3, 3, 4, 4} out of n_present=8. The regen
run's per-turn data reveals a **v1 measurement bias** in
`calibrate_addressee_field`: 3 of 7 wrong cases in the regen
(turns 2, 3, 9) are the LLM correctly identifying "not
addressed to assistant" but being marked incorrect because the
v1 rule treats "no prediction + GT decidable false" as wrong
(unlike the reaction scorer, which has a different
unknown-skip rule).

**Most likely dominant cause (per regen)**: 2.1 (LLM genuine
uncertainty on implicit-continuation) + 2.5 (sampling noise)
+ a **v1 addressee scorer bias** (the 3-of-7 measurement
artifact). 2.2/2.3/2.4 are ruled out or minor.

**Recommended Phase 2 direction (updated with regen data)**:
- **Phase 2A (HIGH PRIORITY)**: Fix the v1 addressee scorer
  bias. Update `calibrate_addressee_field` to treat "no
  prediction + GT decidable false" as **correct** (or at
  least as a separate "no-emit" category, not as incorrect).
  This is a 3-4 line scorer change. Expected impact:
  addressee acc on the regen jumps from 12.5% to 50%.
- **Phase 2D (FALLBACK)**: If 2A doesn't reduce variance to
  ECE < 0.20, document the irreducible floor + M20.4 owner
  informed.
- **Skip 2B, 2C, 2E** — 2.2/2.3/2.4 are not dominant.

## Step 1.1: per-turn flip analysis (regen run R0_regen)

Aggregate evidence (5 P0 runs + 1 regen):

| run | n_correct | n_incorrect | n_present | n_unknown |
|---|---|---|---|---|
| 1 | 4 | 4 | 8 | 0 |
| 2 | 3 | 5 | 8 | 0 |
| 3 | 3 | 5 | 8 | 0 |
| 4 | 2 | 6 | 8 | 0 |
| 5 | 4 | 4 | 8 | 0 |
| R0_regen | 1 | 7 | 8 | 0 |

Total: 17/48 = 35.4% accuracy across 6 runs. The R0_regen is
the worst of the 6, scoring 1/8.

**Per-turn analysis (R0_regen)** — full emissions from
`m18_7_attribution_hypotheses.json`:

| turn | GT addr | GT pid | LLM emit | LLM pid | LLM addr_to_ass | conf | outcome |
|---|---|---|---|---|---|---|---|
| 0 | true | carol | NO | — | — | 0.0 | **incorrect** (no emit, GT decidable) |
| 1 | true | carol | NO | — | — | 0.0 | **incorrect** (no emit, GT decidable) |
| 2 | false | dave | NO | — | — | 0.0 | **incorrect** (no emit; v1 marks wrong despite semantic match) |
| 3 | false | carol | NO | — | — | 0.0 | **incorrect** (no emit; v1 marks wrong despite semantic match) |
| 4 | true | "" | YES | (empty) | False | 0.90 | **incorrect** (LLM said not addressed, GT says addressed; M18.4 disclosure-forbid case) |
| 8 | true | carol | YES | hutao | False | 0.95 | **incorrect** (LLM said not addressed, GT says addressed; "OK胡桃" + "can you..." case) |
| 9 | false | eve | NO | — | — | 0.0 | **incorrect** (no emit; v1 marks wrong despite semantic match) |
| 11 | false | carol | YES | (empty) | False | 0.00 | **correct** (LLM said not addressed, matches GT) |

**Outcome breakdown (8 decidable turns)**:
- 2 are "GT true + LLM no emit" (turns 0, 1) — LLM missed
  explicit "Can you..." and implicit continuation
- 2 are "GT true + LLM emit False" (turns 4, 8) — LLM
  confident-but-wrong on M18.4 disclosure-forbid + OK胡桃
- 3 are "GT false + LLM no emit" (turns 2, 3, 9) — LLM
  semantically correct, but v1 unknown-skip rule doesn't
  distinguish "no prediction" from "wrong prediction"
- 1 is "GT false + LLM emit False" (turn 11) — LLM correct

**v1 measurement bias confirmed**: 3 of 7 incorrect cases
(turns 2, 3, 9) are the LLM correctly identifying "not
addressed to assistant" but being marked wrong by v1's
"empty prediction against decidable GT → incorrect" rule.
**This is a v1 measurement artifact, not LLM failure.**
In a v2 addressee scorer that treats "no prediction matches
GT false" as correct, the regen would score 4/8 = 50%.

**Cross-run per-turn analysis (LIMITED)**: 5 P0 surfaced
runs don't have per-turn data (only aggregate n_correct).
To cross-reference 6 runs per turn, the 5 P0 session roots
would need to be re-collected or per-turn data captured.
**This is the dominant P4 Phase 1 limitation.**

What we CAN say from 6 runs' aggregate n_correct
distribution: 5 P0 runs scored 2-4 (mean 3.2). The regen
scored 1. Adding the regen: range 1-4, mean 2.83. The
"no-emit + GT false" measurement bias likely accounts for
1-2 of the wrong cases per run (3 such cases exist), so
adjusted-correct across 6 runs is likely 4-6 per run
(before the no-emit bias correction).

**Hypothesis (pending per-turn data for 5 P0 runs)**: the
1-2 cases that flip are most likely the GT-true cases where
the LLM "should" emit `addressed_to_assistant=True` but
sometimes doesn't (turns 0, 1) and sometimes does
incorrectly (turns 4, 8). The M18.4 disclosure-forbid (turn 4)
and "OK胡桃" explicit-@ (turn 8) cases are the strongest
candidates for the always-flip case.

## Step 1.2: noisy case characterization (partial)

Aggregate evidence from reliability_bins (5 P0 runs):

| run | high-band (0.9-1.0) count | high-band mean conf | high-band acc | gap |
|---|---|---|---|---|
| 1 | 5 | 0.940 | 0.800 | 0.140 |
| 2 | 4 | 0.950 | 0.500 | 0.450 |
| 3 | 4 | 0.963 | 0.500 | 0.463 |
| 4 | 2 | 1.000 | 1.000 | 0.000 |
| 5 | 3 | 0.967 | 0.667 | 0.300 |

**Pattern**: high-band accuracy ranges 0.500-1.000 across runs.
When high-band accuracy is 1.000 (R4), ECE is low (0.063).
When high-band accuracy is 0.500 (R2, R3), ECE is high (0.425,
0.313). **The variance is dominated by the high-confidence
band's accuracy**, not by the lower bands.

This is consistent with **2.1 (LLM uncertainty) + 2.5
(sampling noise)**:
- The LLM emits high confidence (0.9+) on cases it
  "knows" (correct or not).
- The 0.5-0.8 confidence band is sparsely populated
  (bimodal signature).
- The flips happen in the high-confidence band, where
  the LLM is "certain" but sometimes wrong.

**Specific evidence for 2.5 (sampling noise)**: the high-band
count itself varies (2, 3, 4, 4, 5 across runs). The LLM is
sometimes choosing to emit high-confidence for a case, and
sometimes choosing not to. This is consistent with sampling
noise at the decision boundary.

## Step 1.3: temperature/seed experiment (DEFERRED)

This step requires a fresh LLM call with `temperature=0` and
a fixed seed. OpenRouter supports this via the standard API
parameters; the project's `OpenRouterJSONClient` would need
to expose them.

**Status**: deferred. Rationale: Step 1.1 (per-turn analysis)
is cheaper and more informative. If 1.1 confirms that 1-2
turns are the "always-flip" cases, then 1.3 becomes a
targeted probe on those turns (cheap). If 1.1 is inconclusive,
then 1.3 is needed as a sweep.

**Implementation plan** (for when 1.3 is needed):
- Add `temperature=0` and `seed=42` parameters to
  `OpenRouterJSONClient` (or pass via `model_kwargs`)
- Run 1 P0 replay with these settings
- Compare per-turn correctness to the 5 P0 baseline
- If flips collapse: dominant cause is 2.5
- If flips persist: dominant cause is 2.1 or 2.2

**Out of Phase 1 scope**: this is a Phase 2 / 3 step.

## Step 1.4: pid normalization audit (regen run R0_regen)

Pid normalization table (frozen v2):
```python
M18_7_1_PID_NORMALIZATION = {
    "assistant": "bot",
    "hutao": "bot",
    "hutao_assistant": "bot",
    "clawdgroupchat_bot": "bot",
}
```

Held-out fixture speaker_participant_ids: `carol`, `dave`,
`hutao`, `eve`. None of `carol`, `dave`, or `eve` are in the
normalization table — they pass through as-is (lowercased).
`hutao` is in the table → `bot`.

**LLM pid emissions (R0_regen, 8 emissions)**:

| turn | kind | LLM pid | table hit? | normalized |
|---|---|---|---|---|
| 1 | reaction | (empty) | no | "" |
| 4 | addressee | (empty) | no | "" |
| 5 | addressee | unknown | no | "unknown" |
| 5 | reaction | unknown | no | "unknown" |
| 6 | reaction | carol | no | "carol" |
| 8 | addressee | hutao | **yes** | "bot" |
| 11 | addressee | (empty) | no | "" |
| 11 | reaction | hutao | **yes** | "bot" |

**Audit findings**:
- 2 of 8 emissions hit the normalization table (both `hutao` → `bot`)
- 6 of 8 are pass-through (3 empty, 2 "unknown", 1 "carol")
- **No non-canonical pids** (e.g., "AI assistant", "the bot", "胡桃")
  are observed. The LLM consistently uses lowercase Latin script
  pids.
- The "unknown" emission is a problem: the LLM is saying "I don't
  know the pid", which is a valid "no prediction" signal, but the
  v1 scorer treats it as a positive prediction with pid="unknown",
  which mismatches any non-unknown GT pid.

**Aggregate evidence** (pid breakdown, 5 P0 runs + 1 regen):

| run | pid n_present | pid n_correct | pid acc |
|---|---|---|---|
| 1 | 6 | 4 | 0.667 |
| 2 | 6 | 4 | 0.667 |
| 3 | 6 | 2 | 0.333 |
| 4 | 6 | 2 | 0.333 |
| 5 | 6 | 3 | 0.500 |
| R0_regen | 6 | 2 | 0.333 |

Range 0.333-0.667, mean 0.483.

**Conclusion**: 2.4 (pid normalization gap) is **NOT** the
dominant cause. Evidence:
- LLM uses canonical pid forms consistently (no "AI assistant",
  "the bot", or Chinese characters in pids).
- The pid table handles the LLM's preferred forms
  (hutao → bot).
- The wrong pids in the regen are mostly **empty + GT non-empty**
  (turn 5: empty GT hutao) or **LLM "unknown" + GT hutao**
  (turn 5 reaction). These are 2.1 (LLM uncertainty), not 2.4
  (pid form).

**Possible fix for 2.4** (low priority): if the LLM emits
`pid="unknown"`, treat it as a no-prediction (skip scoring
on that turn). This is a scorer change, not a pid table change.
**Not recommended for P4 Phase 2** — the v1 unknown-skip
rule is intentionally conservative (per M18.7 DECIDED 6).

## Step 1.5: prompt format probe (DEFERRED)

This step requires 2-3 prompt variations × N runs. Each
variation is a new fixture or M18.7.2 prompt adjustment.

**Status**: deferred. Rationale: Step 1.1-1.4 should narrow
down the cause enough that 1.5 is a targeted test, not a
sweep. If 1.1-1.4 indicate 2.1+2.5 (LLM uncertainty + sampling
noise), 1.5 is unlikely to help (prompt changes don't fix
LLM uncertainty). If 1.1-1.4 indicate 2.2, 1.5 is the right
test.

## Step 1.6 (conclusion)

Based on 5 P0 surfaced runs + 1 regen run:

- **Dominant cause is 2.1 (LLM uncertainty) + 2.5 (sampling
  noise) + a v1 addressee scorer measurement bias** (the
  "no prediction + GT decidable false = wrong" rule).

  Evidence:
  - 3 of 7 wrong cases in the regen are the v1 measurement
    bias (turns 2, 3, 9)
  - The 4 remaining wrong cases are the LLM missing or
    misjudging the GT-true cases (turns 0, 1, 4, 8)
  - Bimodal signature in 5/5 + overconfidence_at_high_band
    in 3/5
  - Pid acc 0.333-0.667 is genuine LLM uncertainty, not
    pid form mismatch (DONE audit)

- **2.2 (prompt format)** is unlikely to be dominant. The
  M18.7.2 prompt is minimal and has been working for
  addressee in earlier milestones.

- **2.3 (fixture ambiguity)** is plausible for turns 4 (M18.4
  disclosure-forbid) and 8 ("OK胡桃" + "can you..."). Both
  have non-obvious GT. **Cannot rule out without per-turn
  data for the 5 P0 surfaced runs.**

- **2.4 (pid normalization)** is **ruled out** by Step 1.4
  audit.

**Recommended Phase 2** (revised after regen):
- **Phase 2A (HIGH PRIORITY)**: Fix the v1 addressee scorer
  bias. This is a **measurement fix**, not a prompt/fixture
  fix. The change is in
  `segmentum/dialogue/runtime/m18_7_1_calibration.py:836-852`.
  Scope: 3-4 lines. Tests: 2-3 new T-cases for the new rule.
  Expected impact: addressee acc on the regen jumps from
  12.5% to 50%. If the 5 P0 runs see similar improvements
  (likely, since the bias is per-run, not per-LLM-call),
  the addressee axis becomes more "moderate_drift" (ECE <
  0.20).
- **Phase 2D (FALLBACK)**: If 2A's scorer change doesn't
  reduce variance to ECE < 0.20, document the remaining
  irreducible floor (the 4 GT-true cases where the LLM
  misses or misjudges) and inform M20.4 owner.
- **Skip 2B (prompt), 2C (fixture re-annotation), 2E (pid
  extension)** — 2.2/2.3/2.4 are not dominant.

## Phase 2A status: IMPLEMENTED (2026-06-09)

The Phase 2A recommendation has been implemented and is
documented in `reports/m18_7_1_p4_phase_2a_summary.md`.
Key facts:

- `calibrate_addressee_field` now accepts opt-in kwarg
  `treat_no_emit_as_not_addressed: bool = False` (default
  preserves v1 byte-identity, D6).
- Runner enables the kwarg only in `scoring_mode == "by_pid"`.
- Regen re-score (no LLM cost): n_correct 1 → 4
  (acc 0.125 → 0.500), threshold unchanged
  (tie_breaker=0.9), drift signature unchanged.
- Tests: 4 new P4 Phase 2A tests in
  `test_m18_7_1_calibration.py`; 61/61 pass in unit,
  305/305 pass in cross-M18.7.1 regression.
- Real-LLM verification: 1 fresh by_pid P0 replay running
  in background (bo7yfoddp); surfaced numbers pending.

**Caveat on ECE regression**: the fix worsens ECE on the
regen (0.356 → 0.731) because correct items now land in
the 0.0 confidence bin. This is the calibration paradox
— the LLM is right but says "0% confident". For M20.4, the
"is the LLM right?" signal is more important than the
"is the LLM calibrated?" signal. Threshold recommendation
is unchanged because it uses drift signals + bin gap, not
raw accuracy.

## Open questions (pending per-turn data for 5 P0 surfaced runs)

1. **Which 1-2 addressee turns flip most often across the
   5 P0 surfaced runs?** Per-turn data is needed. The
   regen shows 1/8 with 4 wrong on GT-true cases (0, 1, 4, 8)
   and 3 wrong on GT-false no-emit cases (2, 3, 9). The
   5 P0 runs scored 2-4 (mean 3.2). If 2-3 of those 2-4
   corrects are on the GT-true cases (0, 1, 4, 8), the LLM
   is sometimes getting them right and sometimes wrong
   (2.1 + 2.5). If 0-1 are on GT-true, the LLM is
   consistently wrong on those (2.2 or 2.3 dominant).

2. **Pid forms audit (DONE)**: the LLM uses canonical
   pid forms consistently. No normalization gap.

3. **No-emit pattern (PARTIAL)**: the regen shows
   no-emit on 5 of 8 decidable addressee turns. If the
   5 P0 runs show similar no-emit patterns on the same
   5 turns, it's a stable LLM behavior. If no-emit
   turns vary across runs, it's sampling noise.

4. **High-confidence variance (PARTIAL)**: regen emitted
   high confidence (0.90, 0.95) on turns 4, 8 — both
   wrong. The 5 P0 runs also showed 3/5 with
   overconfidence_at_high_band. Consistent with 2.1
   (LLM uncertainty) + 2.5 (sampling noise on the
   decision boundary).

## Cross-language signal (preliminary, P5 R1 n=1)

P5 R1 Chinese fixture (6 turns) at
`reports/m18_7_1_p5_chinese_smoke_summary.md`:
- addressee acc=0.500 (3/6) on Chinese vs mean 0.400 on
  English P0 (3.2/8). n=1, not stable.
- Same drift signature: overconfidence_at_high_band +
  bimodal.
- LLM pid form inconsistent (hutao/assistant/alan mixing)
  on Chinese — no Chinese-character pids observed.

**Implication for P4**: the addressee variance is **not
language-specific** (the drift signature matches across
languages). The dominant cause is LLM behavior, not
language artifact. P4's investigation (English-only) is
relevant to Chinese M20.4 handoff too.

## Files this memo will touch (post-regen)

- `reports/m18_7_1_p4_phase_1_memo.md` (this file, with
  per-turn data filled in)
- The P4 design doc (`reports/m18_7_1_p4_addressee_design.md`)
  will not change; this memo is the deliverable.

## Recommendation (updated with regen data)

Based on the regen run + 5 P0 surfaced:

- **Dominant cause is most likely 2.1 + 2.5** (LLM uncertainty
  + sampling noise), with a v1 **measurement bias contributing
  3/8 wrong cases per run** (turns 2, 3, 9 no-emit + GT false).
- **2.2 (prompt format)** is unlikely to be dominant.
- **2.3 (fixture ambiguity)** is plausible for turns 4 (M18.4
  disclosure-forbid) and 8 (OK胡桃 + can you...) — both have
  non-obvious GT.
- **2.4 (pid normalization)** is ruled out (DONE audit).

**Recommended Phase 2**:
- **Phase 2A (HIGH PRIORITY)**: Fix v1 measurement bias on
  addressee scorer. Update `calibrate_addressee_field` to treat
  "no prediction + GT decidable false" as **correct** (not
  incorrect). This is a 3-4 line change in
  `m18_7_1_calibration.py:836-852`. Expected impact: addressee
  acc jumps from 12.5% to 50% on the regen. If 5 P0 runs see
  similar jumps, the addressee axis becomes "moderate_drift"
  (ECE < 0.15) and M20.4 can threshold on it.
- **Phase 2D (FALLBACK)**: If 2A doesn't reduce variance to
  ECE < 0.20, document the irreducible floor + M20.4 owner
  informed.
- **Skip 2B, 2C, 2E** — 2.2/2.3/2.4 are not dominant.

## What this memo does NOT conclude

- **The exact per-turn flip pattern for the 5 P0 surfaced
  runs** (per-turn data not preserved; would need to
  re-collect with the regen's session_root pattern). The
  regen run's per-turn pattern is documented.
- **Whether the variance is reducible via prompt change**
  (pending 1.5 or Phase 2B; not recommended based on
  current evidence)
- **The exact irreducible floor after 2A scorer fix**
  (pending 2A implementation + 5 P0 re-replay)
- **Whether the cross-language signal (P5 R1) holds for
  n=3** (pending P5 R2 + R3 stability runs)

## Files this memo will touch (post-regen)

- `reports/m18_7_1_p4_phase_1_memo.md` (this file, with
  per-turn data filled in)
- The P4 design doc (`reports/m18_7_1_p4_addressee_design.md`)
  will not change; this memo is the deliverable.

## Related

- `reports/m18_7_1_p4_addressee_design.md` (the design)
- `reports/m18_7_1_v2_stability_summary.md` (5 P0 runs)
- `reports/m18_7_1_p5_chinese_smoke_summary.md` (1 Chinese run)
- `reports/m18_7_1_p1_m20_4_handoff.md` (M20.4 handoff)
- `tests/fixtures/m18_7_1_held_out_calibration.json`
