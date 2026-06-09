# P4 Addressee Milestone — Design

- **status**: design (Phase 0 only; implementation phases are scoped but not scheduled)
- **generated_at**: 2026-06-09
- **generated_by**: M18.7.1 v2 + P0 stability follow-up
- **pre-read**:
  - `reports/m18_7_1_harness_v2_implementation_summary.md`
  - `reports/m18_7_1_v2_stability_summary.md`
  - `reports/m18_7_1_p1_m20_4_handoff.md`

## 1. Problem statement

The addressee axis on the held-out fixture
(`tests/fixtures/m18_7_1_held_out_calibration.json`,
12 turns) shows high variance across 5 fresh real-LLM
replays with `--scoring-mode by_pid`:

| run | n_correct / n_present | acc | brier | ece | drift signals |
|---|---|---|---|---|---|
| 1 | 4/8 | 0.500 | 0.116 | 0.088 | bimodal |
| 2 | 3/8 | 0.375 | 0.389 | 0.425 | overconfidence_at_high_band, underconfidence_at_low_band, bimodal |
| 3 | 3/8 | 0.375 | 0.248 | 0.313 | overconfidence_at_high_band, bimodal |
| 4 | 2/8 | 0.250 | 0.031 | 0.063 | bimodal |
| 5 | 4/8 | 0.500 | 0.253 | 0.306 | overconfidence_at_high_band, underconfidence_at_low_band, bimodal |

Variance summary:
- **n_correct**: 2-4 out of 8 → 1-2 cases flip per run
- **ECE**: 0.063-0.425 (0.36 spread) → wider than the
  ECE band boundary (0.15) that distinguishes
  moderate_drift from severe_drift
- **drift signals**: `bimodal` in 5/5 runs;
  `overconfidence_at_high_band` in 3/5

This means M20.4 cannot make a threshold decision on the
addressee axis using the by_pid stability data. M20.4
owner reads the addressee axis as "not actionable" and
defers. **P4 is the milestone that makes addressee
actionable.**

### What P4 is NOT

- P4 is **not** a revert of `5ab3db4` (P0-step2 prompt
  addition). That commit was a fix attempt that targeted
  the wrong axis. P0-step2 does not measurably help
  the by_pid signal. **P5 (revert 5ab3db4) is closed** as
  commit `f780f76` (2026-06-09), per the P0 priority
  cleanup; the v1 semantic categories in the M18.7.2
  minimal prompt are removed.
- P4 is **not** an M20.4 threshold decision. P4 makes
  addressee measurable, not threshold-acted-on. The
  M20.4 owner still owns the threshold.

## 2. Possible root causes

The high variance could be any combination of:

### 2.1 LLM genuine uncertainty (irreducible)

The LLM might not be able to reliably determine
"addressed to assistant" on certain turns — e.g.,
implicit-continuation cases that depend on conversation
history. If so, the variance is irreducible without
fine-tuning, and P4's deliverable is "document the
irreducible floor; recommend M20.4 not gate on
addressee alone".

**Investigation cue**: if the LLM emits
`confidence < 0.5` on the cases that flip, it's
genuine uncertainty. If it emits `confidence > 0.8`
on cases that flip, it's confident-but-wrong.

### 2.2 Prompt format confusion

The M18.7.2 prompt asks for `addressed_to_assistant`
and `addressee_participant_id` as two separate fields.
The relationship between them may confuse the LLM. The
held-out fixture has cases where one is "obvious" and
the other is "implicit", and the LLM might pick the
wrong one.

**Investigation cue**: if flipping a field-order
or adding an example to the M18.7.2 prompt
reduces variance, this is the cause.

### 2.3 Held-out fixture ambiguity

The held-out fixture's `ground_truth.addressed_to_assistant`
might be ambiguous on the cases that flip most. E.g.,
"my previous question still stands" (turn 1) is
implicit-continuation with medium confidence.

**Investigation cue**: re-annotate the 2-3 cases that
flip most often with a panel of 2-3 raters. If the
raters disagree, the GT is ambiguous, not the LLM.

### 2.4 Pid normalization gap

`M18_7_1_PID_NORMALIZATION` only handles English
surface forms (`assistant` / `hutao` /
`hutao_assistant` / `clawdgroupchat_bot` → `bot`).
If the LLM emits pids in forms not in the table (e.g.,
"AI assistant" or "the bot"), the score is "incorrect"
by the v2 by_pid scorer, but it's actually a surface-
form mismatch the v2 harness can't normalize.

**Investigation cue**: log all LLM-emitted pids across
the 5 runs; check if any non-canonical pids appear.
If so, add them to the normalization table (or
document as not normalizable).

### 2.5 LLM temperature / sampling noise (irreducible)

`deepseek/deepseek-v4-flash` (and most LLM providers)
default to non-zero temperature. Some cases may be
on the decision boundary (e.g., "yeah" → could be
addressed to assistant or not), and sampling noise
flips them.

**Investigation cue**: run the same prompt with
`temperature=0` and a fixed seed. If the variance
collapses, it's sampling noise. If it doesn't, it's
genuine LLM uncertainty (2.1) or something else.

## 3. Investigation plan (Phase 1 — first deliverable)

Phase 1 is the first P4 deliverable. It produces a memo
that identifies the dominant root cause (or "mixed
causes") and recommends the implementation phase.

### Step 1.1: identify which addressee cases flip most

Cross-reference the 5 by_pid runs by `turn_index`.
For each turn, count how many runs got it right:

```
turn | R1 | R2 | R3 | R4 | R5 | flip rate
  0  |  ? |  ? |  ? |  ? |  ? | ?
  ...
  11 |  ? |  ? |  ? |  ? |  ? | ?
```

Cases with flip rate ≥ 2/5 are the "noisy" cases.
The investigation focuses on these.

**Output**: 1 table in the P4 memo.

### Step 1.2: characterize the noisy cases

For each noisy case:
- Re-read the GT annotation `note` field
- Re-read the `text` field
- Check: is the GT clear, or could a reasonable
  annotator have picked the other label?
- Log: the LLM's emitted `addressed_to_assistant` and
  `confidence` per run (need to re-run with extra
  logging or read `m18_7_attribution_hypotheses.json`
  in the session roots — these were already saved)

**Output**: per-noisy-case analysis in the P4 memo.

### Step 1.3: temperature / seed experiment

Re-run the by_pid replay once with
`temperature=0, seed=42` (or whatever the OpenRouter
client supports for fixed seed). Compare the addressee
variance to the 5-run baseline.

**Output**: a comparison table. If variance collapses,
the dominant cause is 2.5 (sampling noise). If not,
it's 2.1, 2.2, 2.3, or 2.4.

### Step 1.4: pid normalization audit

Log all unique `addressee_participant_id` values
emitted by the LLM across the 5 runs. Compare to
`M18_7_1_PID_NORMALIZATION` keys + GT pid forms.

**Output**: a "LLM-emitted pids vs table" diff. If
any LLM pid is not normalizable, the issue is 2.4.

### Step 1.5: prompt format probe

If Steps 1.1-1.4 don't converge on a single cause,
construct 2-3 minimal prompt variations:
- Original prompt
- Swapped field order
- One extra example added (from the high-confidence
  cases of the held-out fixture)

Re-run by_pid with each variation; compare addressee
variance. If a variation reduces variance, cause 2.2
is contributing.

**Output**: variance comparison across prompt
variations. This step is only needed if 1.1-1.4
don't converge.

### Step 1.6: write the Phase 1 memo

The P4 Phase 1 memo answers:
- Which is the dominant root cause?
- Is the variance reducible at all, or irreducible?
- If reducible, what is the recommended Phase 2
  implementation strategy?
- If irreducible, what is the documentation strategy
  (and does M20.4 owner need to know)?

**Output**: `reports/m18_7_1_p4_phase_1_memo.md`.

## 4. Implementation phases (after Phase 1 memo)

The P4 implementation is split into 2-3 phases. The
exact scope depends on the Phase 1 findings.

### Phase 2A — pid normalization extension (if 2.4 is dominant)

- Add new pid entries to `M18_7_1_PID_NORMALIZATION`
  for any LLM-emitted form not currently in the table.
- Update test cases for the new entries.
- Re-run the by_pid stability check (3-5 fresh
  replays) to confirm variance reduction.

**Scope**: small (1-2 hours). Pure data, no prompt work.

### Phase 2B — prompt format adjustment (if 2.2 is dominant)

- Adjust the M18.7.2 minimal prompt: field order
  change, one example added, or both.
- Run M18.7.2 tests (no LLM) to ensure no regression
  on the existing 23 tests.
- Re-run by_pid stability check (3-5 replays).

**Scope**: medium (1-2 days). Touches M18.7.2 prompt +
tests. **Not** M18.7 prompt or M18.7.1 calibration math.

### Phase 2C — fixture re-annotation (if 2.3 is dominant)

- Re-annotate the noisy cases with a panel of 2-3
  raters (the user + a colleague, or a 2-3 day
  spread of single-rater self-reannotation).
- Update the held-out fixture with the consensus GT.
- Add a `gt_annotator` and `gt_agreement` field to
  the fixture for traceability.
- Re-run by_pid stability check.

**Scope**: medium (2-3 days). Touches fixture only.
No code change.

### Phase 2D — irreducible noise documentation (if 2.1 / 2.5 is dominant)

- Document the irreducible floor in the P4 memo.
- Recommend M20.4 not gate on addressee alone.
- Add a `drift_signals` documentation note to the
  v2 stability report.

**Scope**: small (2-4 hours). Documentation only.

### Phase 3 — verify

After Phase 2 (whichever combination), re-run the
by_pid stability check (5 replays, same fixture).
Expected outcome:
- If Phase 2A-2C was the right fix: addressee ECE
  spread narrows from 0.36 to ≤ 0.20.
- If Phase 2D is the right call: addressee ECE spread
  is documented; M20.4 owner told.

Update the stability summary report with the post-P4
numbers; update the M20.4 handoff doc with the
post-P4 actionable input.

## 5. Out of scope (explicit)

- **M20.4 threshold revision**. P4 makes addressee
  measurable; the threshold decision stays with M20.4.
- **Reaction axis work**. Reaction axis is stable in
  P0 (5/5 ≥ 0.6). P4 is addressee-only.
- **Turn_id axis (Mode B)**. Still 0.0; covered by
  P2 (M18.7.2 prompt turn_id enumeration) + P3
  (fixture repair).
- **M18.7 prompt re-ordering**. The 37.7%-offset
  v2 attrs problem from the v1 summary is a separate
  concern. P4 does not change M18.7.
- **`M18_7_1_PID_NORMALIZATION` extension for Chinese**.
  This is a P5 concern (the Chinese small-scenario
  fixture may surface new pid forms). P4 only handles
  the English held-out fixture.

## 6. Acceptance criteria (for the full P4 milestone)

P4 is "done" when **all** of the following are met:

1. **Phase 1 memo exists** at
   `reports/m18_7_1_p4_phase_1_memo.md` and identifies
   the dominant root cause (or "mixed / irreducible").
2. **Addressee ECE spread ≤ 0.20** across 5 fresh
   by_pid replays on the held-out fixture (vs 0.36
   pre-P4).
3. **Addressee n_correct range ≤ 2** (vs 2-4 pre-P4).
4. **No regression on P0 stability metrics**: reaction
   joint acc still ≥ 0.6 in 5/5 runs.
5. **M20.4 handoff doc updated** with post-P4 numbers
   and a recommendation on whether to gate on addressee.

If the dominant root cause is irreducible (2.1 / 2.5),
AC2-AC3 are replaced with: "irreducible floor
documented + M20.4 owner informed + verdict in
stability report is honest about it".

## 7. Estimated work

- **Phase 1 (investigation)**: 2-3 days
- **Phase 2 (implementation)**: 1-3 days
  (depending on which sub-phase is needed)
- **Phase 3 (verify)**: 1-2 days (mostly waiting for
  LLM runs)
- **Total**: 4-8 days = ~1-2 weeks

## 8. Out of milestone but related

- **P5 (Chinese small-scenario E2E)** is running in
  parallel. P5 may surface cross-language addressee
  variance that informs P4's Phase 1.
- **P2 (M18.7.2 prompt turn_id enumeration)** is
  independent of P4. P2 unlocks the turn_id axis;
  P4 is the addressee axis.
- **P5 (revert 5ab3db4)** is **closed** as commit
  `f780f76` (2026-06-09); combined with the P0 git-state
  cleanup. The v1 semantic categories in the M18.7.2
  minimal prompt are removed. Pre-P0-step2 prompt is
  restored.

## 9. Open questions for the user

1. **Should Phase 1 (investigation) start before
   Phase 0 (this design doc) is approved?** I assume
   yes — Phase 1 is read-only analysis of the 5
   existing P0 replay outputs, no code changes.
2. **Is the addressee ECE spread of 0.36 acceptable
   if it's irreducible?** If so, P4 closes as
   Phase 2D (documentation) and M20.4 doesn't gate
   on addressee. The M20.4 handoff doc would note
   "addressee not gated; admit/tie_breaker use
   reaction only".
3. **Is the held-out fixture re-annotation (Phase 2C)
   in scope?** This requires user time for 2-3 day
   spread re-annotation. If not in scope, P4 cannot
   fix cause 2.3.
4. **Should the pid normalization extension
   (Phase 2A) touch the Chinese fixture pids**?
   P5 may surface new pid forms; P4 could extend the
   table in one shot. Or P4 stays English-only and
   P5 gets a separate pid table extension.

## 10. Files this milestone will touch

Pre-P4 design (this doc):
- `reports/m18_7_1_p4_addressee_design.md` (this file)

Phase 1 investigation:
- `reports/m18_7_1_p4_phase_1_memo.md` (output)

Phase 2 (whichever combination):
- `segmentum/dialogue/runtime/m18_7_1_calibration.py`
  (pid normalization table extension, if 2A)
- `segmentum/dialogue/runtime/m18_7_2_minimal_attribution.py`
  (prompt adjustment, if 2B)
- `tests/fixtures/m18_7_1_held_out_calibration.json`
  (re-annotation, if 2C)
- New tests for any change

Phase 3 (verify):
- `reports/m18_7_1_v2_stability_summary.md` (update)
- `reports/m18_7_1_p1_m20_4_handoff.md` (update)

**Touch list respects**: no changes to M20.4,
M20.4.1, M20.3, M18.5, M18.7 prompt template,
or the calibration math.

## 11. Why P4 matters

The addressee axis is the **primary input** to the
M20.4 producer (M20.4 = "do we admit this attribution
as a commitment?"). If addressee is noisy, the
producer is noisy, and the settler downstream is
noisy. M20.4 owner reads P0 stability and says
"addressee is not actionable" — that's the
engineering honest answer, but it's not the
**final** answer. P4 is the milestone that turns
"not actionable" into "actionable with a known
floor". Until P4 lands, M20.4 thresholds are
necessarily under-specified on the addressee axis.

## Related

- `prompts/M18.7.1_Harness_V2_Design.md`
- `reports/m18_7_1_harness_v2_implementation_summary.md`
- `reports/m18_7_1_v2_stability_summary.md`
- `reports/m18_7_1_p1_m20_4_handoff.md`
- `tests/fixtures/m18_7_1_held_out_calibration.json`
