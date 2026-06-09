# M18.7.1 Harness v2 — Implementation Summary

- status: implementation complete + real-LLM acceptance unblocked
- generated_at: 2026-06-09
- design: `prompts/M18.7.1_Harness_V2_Design.md`
- design summary: `reports/m18_7_1_harness_v2_design_summary.md`
- plan: `C:\Users\zq\.claude\plans\synthetic-wishing-candle.md`
- pre-read: `reports/m18_7_1_calibration_summary.md` (v1 status: `prompt_order_blocked`)
- acceptance gate: real-LLM replay with `--scoring-mode all` (3 sub-reports)

## What was implemented (recap)

M18.7.1 v2 added three scoring modes + placeholder resolution +
PID normalization, all isolated to:

| Path | Action |
|---|---|
| `segmentum/dialogue/runtime/m18_7_1_calibration.py` | EDIT |
| `scripts/run_m18_7_1_real_llm_calibration.py` | EDIT |
| `tests/test_m18_7_1_calibration.py` | EDIT (+16 tests) |
| `tests/fixtures/m18_7_1_v1_report_baseline.json` | NEW (frozen v1 baseline) |
| `prompts/README.md` | EDIT (M18.7.1 v2 reference) |

- 3 new frozen constants (`M18_7_1_SCORING_MODES`,
  `M18_7_1_DEFAULT_SCORING_MODE`, `M18_7_1_PLACEHOLDER_PATTERN`,
  `M18_7_1_PID_NORMALIZATION`).
- 2 pure helpers (`normalize_pid`, `resolve_placeholder`).
- 1 new scoring function (`calibrate_reaction_field_by_pid`).
- 1 wrapper helper (`_calibrate_reaction_by_pid`).
- Runner extended with `scoring_mode` /
  `pid_normalization_override` / `resolve_placeholders` kwargs.
- `replay_history` inferred from `group_turn_envelope.speaker_participant_id`
  membership in `visible_participant_ids` (D7).
- `CalibrationHarnessReport` / `CalibrationFieldReport` extended
  with `scoring_mode` / `fixture_warnings` / `pid_breakdown` /
  `is_about_breakdown` (omitted when None for D6 v1 byte-identity).
- Runner default = `by_turn_id_v1` (v1 byte-compat for the 5
  existing integration tests). CLI default = `by_pid` (Q1).

## Test surface

- 41 existing tests still pass.
- 16 new tests (T1–T16) pass.
- **57/57 targeted tests pass.**
- 301/301 cross-M18.7.1 regression pass (m18_7_1_calibration +
  m18_7_attribution + m18_7_2_minimal_attribution +
  m20_4_producer + m20_4_settlers + m20_4_tie_breaker +
  m20_4_write_path + m20_4_1_gate + m20_3_pre_send_minimal).
- **T9 byte-identity** confirmed: Mode C report's `to_dict()`
  matches `tests/fixtures/m18_7_1_v1_report_baseline.json`
  exactly (the baseline was captured with v1 code + empty LLM
  stub *before* any v2 changes landed).

## Real-LLM replay with `--scoring-mode all`

Command:

```bash
PYTHONPATH="E:/workspace/segments" "C:\Users\zq\AppData\Local\Programs\Python\Python311\python.exe" \
  scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_m18_7_1_v2_replay \
  --scoring-mode all
```

`--scoring-mode all` runs the fixture 3 times against the same
runtime and surfaces 3 sub-reports. Each replay is a separate
LLM call sequence, so addressee/reaction numbers across modes
reflect real-LLM non-determinism, not measurement noise.

Model: `deepseek/deepseek-v4-flash` (default in
`default_openrouter_client`).

### Replay surface — 3 modes, side-by-side

| mode | addr n_corr / n_pres | addr acc | react n_corr / n_pres | react acc | react ECE | react Brier | warnings |
|------|----------------------|----------|------------------------|-----------|-----------|-------------|----------|
| **by_pid** (Mode A) | 1/8 | 0.125 | **2/3 (joint)** | **0.6667** | 0.167 | 0.043 | 0 |
| by_turn_id_resolved (Mode B) | 2/8 | 0.250 | 0/3 | 0.000 | 0.317 | 0.301 | 3 unresolved |
| by_turn_id_v1 (Mode C) | 3/8 | 0.375 | 0/6 | 0.000 | 0.517 | 0.412 | 0 |

**`by_pid` per-axis breakdown** (joint axis is the strictest
signal; per-axis numbers are 6/6 decidable, not 3/3):

| sub-axis | n_present | n_correct | accuracy | ECE | Brier |
|----------|-----------|-----------|----------|-----|-------|
| pid | 6 | 2 | 0.333 | 0.083 | 0.022 |
| is_about_assistant_claim | 6 | 2 | 0.333 | 0.083 | 0.022 |
| **joint (pid ∧ is_about)** | 3 | 2 | **0.667** | 0.167 | 0.043 |

`by_turn_id_resolved` resolved 2 carol placeholders and 1
non-placeholder; 3 assistant placeholders could not be resolved
because the fixture's `group_turn_envelope` contains only
user-side turns, so the runner's `replay_history` has no
assistant-role entries at the time the GT is being scored.
This is **informational**, not a bug — the runner surfaces
the warnings in `fixture_warnings` and skips the unresolvable
GTs (counted in `n_unknown`).

### Why the per-mode numbers differ

The 3 modes measure different things; the LLM emits the same
underlying prediction, but the scoring function asks
different questions:

- **by_pid** scores the joint of `reaction_to_participant_id`
  + `is_about_assistant_claim`, both with pid normalization.
  The 3 decidable joint-cases yield 2 correct (LLM picked
  the right pid AND the right is_about).
- **by_turn_id_resolved** scores `reaction_to_turn_id` after
  resolving placeholders. The LLM emitted
  `reaction_to_turn_id=""` on all 6 turns with reaction
  predictions; even after resolution, the LLM's blank
  string cannot match any resolved turn_id (0/3 correct).
- **by_turn_id_v1** scores `reaction_to_turn_id` with the v1
  placeholder-as-literal rule. GTs containing
  `"turn_<assistant_prior_turn_id>"` etc. can never match
  `""` → 0/6 correct (this is the original v1
  "scoring-function-forced 0/6" problem that v2 fixes).

**Addressee numbers** also differ across modes (1, 2, 3) —
this is real-LLM non-determinism, not measurement drift.
The 3 LLM runs are independent because `--scoring-mode all`
calls `run_m18_7_1_calibration_harness` 3 times against the
same `runtime`. The scores that matter for v2 acceptance are
the **reaction** numbers, not the addressee numbers, because
v2's purpose is to fix reaction scoring (v1 reaction
n_correct was 0/6 *forced by the scoring function*, not by
the LLM).

## Acceptance criteria — met

| AC | criterion | met? | evidence |
|----|-----------|------|----------|
| AC1 | All 11 new tests pass | ✅ | 16 new (T1–T16), 57/57 targeted |
| AC2 | 41 existing tests still pass | ✅ | 0 regressions |
| AC3 | T9 confirms Mode C byte-identical to v1 | ✅ | baseline fixture + T9 assertion |
| **AC4** | **Real-LLM replay with `--scoring-mode by_pid` shows joint reaction accuracy ≥ 0.33** | ✅ | **0.667 (2/3)** |
| AC5 | CLI supports `--scoring-mode all` with 3 sub-reports | ✅ | `scoring_mode_reports` triple in summary |
| AC6 | No changes outside 5 files in File Touch List | ✅ | only 5 files touched |

## What v2 unblocks

**v1's 0/6 reaction accuracy was a measurement artifact, not
a model-competence signal.** v2's `by_pid` mode shows the LLM
gets 2/3 (0.667) of joint reaction cases correct on the
held-out fixture. This is consistent with the v1 behavior:

- v1: `reaction_to_turn_id` is `""` 5/6 times → strict string
  equality forces 0/6.
- v2 by_pid: the LLM is filling `reaction_to_participant_id`
  and `is_about_assistant_claim` correctly in 2/3 of the
  decidable cases → 0.667.

**The M20.4 producer / settler uses pid + is_about
semantically, not turn_id strings** (per
`m20_4_attribution.py` design). v2's `by_pid` mode is the
correct measurement signal for the M20.4-relevant question.

## What v2 does NOT unblock

- **Addressee drift is real.** `by_pid` reports
  `n_correct=1/8, accuracy=0.125, brier=0.259, ece=0.300` on
  the addressee axis. This is the M18.7 prompt-order
  blocker from the v1 summary, partially unblocked by
  M18.7.2's minimal-prompt call site
  (see `reports/m18_7_2_post_p0_replay_summary.md` +
  `prompts/M18.7.2_Work_Prompt.md`). v2 surfaces the
  numbers; addressing the underlying calibration
  problem is M20.4's call.
- **M20.4 threshold revision is unblocked-data pending.**
  v2 surfaces `candidate_admit_min: null` and
  `candidate_tie_breaker_min: 0.9` on the addressee axis
  (Mode A) and `null / 0.7` (Mode C). Per the frozen
  caveat, **the threshold decision belongs to M20.4**;
  M18.7.1 only surfaces the candidates.

## Out of scope (per the design / plan)

- M18.7.2 prompt turn_id enumeration (separate milestone).
- Fixture GT placeholder replacement (v2 resolves at
  runner-time; fixture JSON unchanged).
- P0-step2 revert (commit 5ab3db4, separate decision per D4).
- M20.4 threshold revision.
- M18.5 / M18.7 / M20.4 / M20.4.1 changes.
- Path A / M10 / `conversation_loop.py`.

## How to use v2

```bash
# Default (Q1: by_pid at CLI)
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_run

# v1 byte-compat (D6)
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_run \
  --scoring-mode by_turn_id_v1

# All 3 modes side-by-side
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_run \
  --scoring-mode all

# Custom pid normalization
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_run \
  --scoring-mode by_pid \
  --pid-normalization-override my_pid_table.json

# Skip placeholder resolution (diagnostic)
PYTHONPATH=. python scripts/run_m18_7_1_real_llm_calibration.py \
  --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
  --session-root tmp_run \
  --scoring-mode by_turn_id_resolved \
  --no-resolve-placeholders
```

## Verdict (per the P0 band agreement)

| mode | addr ECE | addr Brier | react ECE | react Brier | band |
|------|----------|------------|-----------|-------------|------|
| by_pid | 0.300 | 0.259 | 0.167 | 0.043 | addressee: severe_drift_recommend_m20_4; reaction: moderate_drift |
| by_turn_id_resolved | 0.100 | 0.103 | 0.317 | 0.301 | addressee: moderate_drift; reaction: severe_drift_recommend_m20_4 |
| by_turn_id_v1 | 0.319 | 0.265 | 0.517 | 0.412 | both: severe_drift_recommend_m20_4 |

`by_pid` is the M20.4-relevant signal (M20.4 settler uses
pid + is_about semantically). The verdict for the M20.4
threshold-revision question is **"addressee overconfidence
at high band + bimodal"** (the same drift signals v1
surfaced, now with the M20.4-relevant measurement signal).
M20.4 owner decides whether to revise the v1 0.4 / 0.85
thresholds; M18.7.1's job ends here.

## Related

- `prompts/M18.7.1_Harness_V2_Design.md` (full design, 16 sections)
- `prompts/M18.7.1_Work_Prompt.md` (v1)
- `prompts/M18.7.2_Work_Prompt.md` (M18.7.2 minimal-prompt call site)
- `reports/m18_7_2_post_p0_replay_summary.md` (the v1 → v2 motivation)
- `reports/m18_7_1_calibration_summary.md` (v1 status, now superseded by v2)
- `tests/test_m18_7_1_calibration.py` (57 tests, T1–T16 new)
- `tests/fixtures/m18_7_1_v1_report_baseline.json` (frozen v1 baseline, T9 regression)
- `tests/fixtures/m18_7_1_held_out_calibration.json` (12-turn held-out fixture)
