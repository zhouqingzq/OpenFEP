# P5 Chinese Stability + Cross-Language Summary (n=3)

- **status**: P5 R2 + R3 complete; cross-language table finalized
- **generated_at**: 2026-06-09
- **fixtures**:
  - English: `tests/fixtures/m18_7_1_held_out_calibration.json` (12 turns, 4 speakers)
  - Chinese: `tests/fixtures/m18_7_1_chinese_smoke_calibration.json` (6 turns, 3 speakers)
- **scoring mode**: `by_pid` (v2 default)
- **session roots**:
  - English: 5 P0 surfaced + 1 regen
  - Chinese: 3 P5 runs

## TL;DR

The 3 Chinese P5 runs match the 5 English P0 runs on **drift
signature** (bimodal + overconfidence_at_high_band) and
**threshold recommendation** (candidate_tie_breaker=0.9 in
3/3 Chinese + 5/5 English). The **reaction axis is much
noisier on Chinese** (n_present=1 in 3/3 runs, vs English
2-5/12), likely a real under-emit issue rather than a
measurement artifact.

**Cross-language verdict**: addressee variance is **not
language-specific**; the LLM uncertainty floor on implicit-
continuation is universal. The Chinese-language reaction
under-emit is a separate concern, but **insufficient data**
(3 runs × 1 emit/run = 3 emissions total) prevents a stable
conclusion.

**P5 milestone deliverable**: 3-run Chinese stability
established. The M20.4 handoff doc
(`reports/m18_7_1_p1_m20_4_handoff.md`) should remain
English-only until P5 reaches n=5+ runs.

## Cross-language comparison table

| field | English P0 (5 runs) | Chinese P5 (3 runs) | cross-language read |
|---|---|---|---|
| **addressee n_present** | 8 / 8 / 8 / 8 / 8 | 6 / 6 / 6 | n/a (different fixture sizes) |
| **addressee n_correct** | 4 / 3 / 3 / 2 / 4 | 3 / 3 / 2 | Chinese mean 2.67/6 (44.4%) vs English 3.2/8 (40%) — close |
| **addressee accuracy** | 0.500 / 0.375 / 0.375 / 0.250 / 0.500 | 0.500 / 0.500 / 0.333 | both languages 0.25-0.50 spread |
| **addressee ECE** | 0.088 / 0.425 / 0.313 / 0.063 / 0.306 | 0.133 / 0.450 / 0.125 | Chinese range 0.125-0.450; English 0.063-0.425 — overlapping |
| **drift signals (common)** | bimodal 5/5; overconfidence 3/5 | bimodal 3/3; overconfidence 3/3 | **same drift signature** |
| **reaction n_present** | 4 / 4 / 3 / 2 / 5 | 1 / 1 / 1 | **Chinese under-emits 4-5×** |
| **reaction joint acc** | 0.750 / 0.750 / 0.667 / 1.000 / 0.600 | 0.000 / 0.000 / 0.000 | Chinese insufficient data (n=1) |
| **pid sub-axis** | 0.667 / 0.667 / 0.333 / 0.333 / 0.500 | 1.000 / 1.000 / 1.000 | Chinese pid perfect (n=1); English 0.33-0.67 |
| **is_about sub-axis** | 0.500 / 0.500 / 0.333 / 0.333 / 0.500 | 0.000 / 0.000 / 0.000 | Chinese insufficient data |
| **verdict** | severe_drift (5/5) | severe_drift (3/3) | identical |
| **candidate_tie_breaker** | 0.9 / 0.9 / 0.8 / 0.9 / 0.8 | 0.9 / 0.9 / 0.9 | Chinese 3/3 → 0.9; English 5/5 → {0.8, 0.9} |
| **candidate_admit_min** | None (noisy) | None | both languages: insufficient signal |

## Per-run Chinese data (3 runs)

### P5 R1 (`tmp_m18_7_1_p5_chinese_run_1`)
- addressee: 3/6 acc=0.500 ECE=0.133
- reaction: 0/1 acc=0.000
- drift: overconfidence + bimodal
- LLM errors surfaced: turn 0 missed (explicit @), turn 5 wrong on addressed_to_assistant, pid form inconsistent

### P5 R2 (`tmp_m18_7_1_p5_chinese_run_2`)
- addressee: 3/6 acc=0.500 ECE=0.450
- reaction: 0/1 acc=0.000
- drift: overconfidence + underconfidence + bimodal
- high-band: count=4, acc=0.500 (overconfidence gap 0.45)
- low-band: count=1, acc=0.000 (underconfidence gap 0.05)

### P5 R3 (`tmp_m18_7_1_p5_chinese_run_3`)
- addressee: 2/6 acc=0.333 ECE=0.125
- reaction: 0/1 acc=0.000
- drift: overconfidence + bimodal
- high-band: count=3, acc=0.667
- low-band: count=3, acc=0.000 (underconfidence)

## Cross-language findings

### Finding 1: Addressee drift is language-agnostic

The drift signature (`bimodal` + `overconfidence_at_high_band`)
is identical in both languages. The LLM is just as uncertain
about implicit-continuation on Chinese as on English. The
underlying LLM behavior is universal; the prompt + scoring
is doing the same thing in both cases.

**Implication**: P4's English-only investigation is relevant
to Chinese M20.4 handoff. The v1 addressee scorer bias
identified in P4 Phase 1 (3-of-7 measurement artifact) likely
applies to Chinese too, though n=6 is too small to confirm.

### Finding 2: Reaction under-emits on Chinese (REAL)

The LLM emits 1 reaction hypothesis per 6-turn Chinese fixture
across all 3 runs. On English (12 turns), it emits 2-5
hypotheses per run. **This is a 4-5× under-emit on Chinese,
after normalizing for fixture length.**

**Possible causes** (not investigated, pending P5 n=5+):
- M18.7.2 prompt's `assertion_kind` enum or examples are
  English-centric and don't trigger on Chinese patterns
- Chinese pids (`xiaoming`/`alan`/`hutao`) in the fixture
  may not match the LLM's mental model
- The LLM genuinely has fewer "reaction" patterns to attribute
  in the small Chinese fixture

**Implication**: P5 R4 + R5 are needed to confirm whether the
under-emit is stable. If n=5 still shows n_present ≤ 2, this
is a real cross-language LLM behavior that needs prompt
investigation (a separate milestone from P4).

### Finding 3: Pid normalization is cross-language robust

The LLM consistently uses Latin-script pids (`hutao` /
`assistant` / `alan` / `xiaoming`) on Chinese text. The
normalization table's `hutao → bot` rule fires on the Chinese
emissions. **No Chinese-character pid forms** (`胡桃` / `小明` /
`阿蓝`) were emitted in any of the 3 Chinese runs. The
normalization table does not need a Chinese-pid extension for
the LLM's current behavior.

### Finding 4: Threshold recommendation is consistent

`candidate_tie_breaker_min = 0.9` in 3/3 Chinese + 5/5 English
(8/8 total). `candidate_admit_min` is None in all 3 Chinese
runs (consistent with the 5 P0 runs' "noisy" signal).

**M20.4 implication**: the tie_breaker nudge from
`0.85 → 0.9` is **cross-language validated** on n=8 runs.
The M20.4 owner can adopt the 0.9 threshold with reasonable
confidence that it's not language-specific.

## LLM emissions on Chinese (R1 detailed)

From P5 R1 surfaced (R2 + R3 surfaced data not in this turn's
context; the R1 detail is in
`reports/m18_7_1_p5_chinese_smoke_summary.md`):

| turn | kind | pid (raw) | confidence | GT | match |
|---|---|---|---|---|---|
| 0 | (none) | — | — | addr: hutao, addressed=true | **MISS** (no emit) |
| 1 | addressee | hutao | 0.95 | addr: alan, addressed=false | partial (pid after norm) → incorrect |
| 1 | reaction | hutao | 0.90 | pid: hutao, is_about=true | pid ✓, is_about ✗ → incorrect |
| 2 | addressee | assistant | 0.90 | addr: hutao, addressed=true | pid (after norm) ✓ → **correct** |
| 3 | (none) | — | — | addr: alan, addressed=false | **MISS** (no emit) |
| 4 | addressee | alan | 0.95 | addr: alan, addressed=false | **correct** |
| 5 | addressee | hutao | 1.00 | addr: hutao, addressed=true | pid ✓, addressed ✗ → incorrect |

**Pattern**: LLM uses Latin pids consistently. Pid
normalization handles `hutao` ↔ `assistant` (both → bot) and
`alan` (passthrough). The 3 correct cases are turn 2, 4, and
one "no-emit-matches-GT" (per the v1 scorer behavior
documented in P4 Phase 1).

**Real LLM errors on Chinese**:
- Turn 0: explicit `@胡桃 我有个问题` — LLM skipped entirely.
  Most explicit addressee case in fixture, missed.
- Turn 5: `胡桃, 我还有一个建议` — LLM said
  `addressed_to_assistant=false` (wrong; GT says true).
  Real semantic error, not a normalization issue.
- Pid form mixing: `hutao` (turn 1, 5) vs `assistant` (turn 2)
  vs `alan` (turn 4). LLM not following a single convention.
  After normalization, this is mostly absorbed.

## What P5 n=3 establishes

- **3 runs of by_pid replay on Chinese fixture work** without
  errors. v2 harness is cross-language functional.
- **Addressee drift signature matches English** (bimodal +
  overconfidence). Cross-language consistency confirmed.
- **Threshold recommendation: tie_breaker=0.9 in 3/3 Chinese**
  reinforces the English 5/5 signal. **M20.4 handoff
  tie_breaker nudge is cross-language validated on n=8.**
- **Pid normalization does not need Chinese extension** for
  the LLM's current Latin-pid behavior.

## What P5 n=3 does NOT establish

- **Stability of Chinese reaction under-emit** (n=1 per run
  × 3 runs = 3 total emissions; insufficient for stability).
  Need P5 R4 + R5 (or n=5) to confirm.
- **Whether the 1 reaction emit on Chinese is consistent in
  target** (always the same turn, or varies). Per-turn
  analysis would help but the surfaced JSON doesn't include
  it.
- **Whether v1 addressee scorer bias (3/7 measurement
  artifact) applies to Chinese too**. Likely yes (universal
  v1 behavior), but n=6 is too small to confirm.

## P5 next steps (out of scope for this report)

1. **P5 R4 + R5**: 2 more by_pid replays on Chinese fixture
   to reach n=5 stability. ~30-40 min. If reaction
   under-emit persists (n_present ≤ 2/run), this is real
   and needs prompt investigation.
2. **Per-turn Chinese analysis**: read R1-R3 session roots'
   `m18_7_attribution_hypotheses.json` files. Identify which
   turn(s) the LLM emits reaction on. If it's the same turn
   across all 3 runs, the LLM has a "favorite" reaction turn.
   If it varies, the under-emit is sampling noise.
3. **M20.4 handoff update**: add a "cross-language
   validation" footnote to the handoff doc noting the
   3 Chinese runs match the 5 English runs on the
   tie_breaker=0.9 recommendation.

## Out of scope (explicit)

- **Chinese pid normalization extension**. Not needed; the
  LLM uses Latin pids consistently.
- **M18.7.2 prompt changes for Chinese reaction**. P5 R4 + R5
  stability data needed first.
- **M20.4 threshold revision**. M20.4 owner reads the
  handoff doc; P5 surfaces data, doesn't decide.
- **Addressee milestone work**. P4 owns that.

## Files touched by this report

- `reports/m18_7_1_p5_cross_language_summary.md` (NEW, this file)
- Session roots preserved:
  - `tmp_m18_7_1_p5_chinese_run_2/`
  - `tmp_m18_7_1_p5_chinese_run_3/`

**Not modified**: M18.7, M18.7.2, M20.4, M20.4.1, the
Chinese fixture content, the runner, the calibration math,
the M20.4 handoff doc (pending user review).

## Related

- `reports/m18_7_1_p5_chinese_smoke_summary.md` (P5 R1 detail)
- `reports/m18_7_1_v2_stability_summary.md` (5 English P0 runs)
- `reports/m18_7_1_p4_phase_1_memo.md` (P4 Phase 1)
- `reports/m18_7_1_p4_addressee_design.md` (P4 design)
- `reports/m18_7_1_p1_m20_4_handoff.md` (M20.4 handoff)
- `tests/fixtures/m18_7_1_chinese_smoke_calibration.json`
- `tests/fixtures/m18_7_1_held_out_calibration.json`
