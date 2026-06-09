# M18.7.1 Harness v2 — Design Summary

- full design: `prompts/M18.7.1_Harness_V2_Design.md`
- status: design draft, awaiting approval
- at: 2026-06-09
- pre-read: `reports/m18_7_2_post_p0_replay_summary.md`

## One-paragraph summary

After the post-P0 replay showed P0-step2 prompt fix did
not help, the unfixed problem is the measurement layer:
M18.7.1 scores reaction on strict string equality of
`reaction_to_turn_id` while the LLM emits
`reaction_to_turn_id=""` 5/6 times, and the fixture GT
uses unresolvable placeholder strings
`"turn_<assistant_prior_turn_id>"`. v2 adds three
scoring modes — `by_pid` (primary, scores
`reaction_to_participant_id` + `is_about_assistant_claim`
joint correctness), `by_turn_id_resolved` (Mode B with
placeholder resolution at runner-time), and
`by_turn_id_v1` (byte-identical legacy) — plus a
participant-id normalization table that collapses
`assistant` / `hutao` / `clawdgroupchat_bot` to a
canonical `bot` form. v2 lives entirely in
`m18_7_1_calibration.py` + the runner script + tests;
no changes to M18.7, M18.7.2, M20.4, M20.4.1, M18.5.

## What's in the design

16 sections, ~410 lines:

0. Why this design exists (the 3 root causes)
1. Goals (G1-G5)
2. Non-goals (NG1-NG5)
3. Scoring modes (A: by_pid, B: by_turn_id_resolved, C: by_turn_id_v1)
4. Placeholder resolution (4.1 pattern, 4.2 mechanism, 4.3 turn_id format)
5. Participant id normalization (5.1 problem, 5.2 table, 5.3 location, 5.4 extensibility)
6. Report shape (6.1-6.5 per-mode)
7. CLI / runner arg changes
8. Test plan (T1-T10)
9. Acceptance criteria (AC1-AC6)
10. Out of scope (OOS1-OOS4)
11. Open design questions (Q1-Q5)
12. Risk assessment (R1-R4)
13. Estimated work (lines + scope)
14. Timeline guess (no SLA)
15. Why v2 is the right next step
16. Decision points for the user (D1-D4)

## Key design decisions

- **Mode A (`by_pid`) is the primary v2 mode.** It
  scores what the LLM actually emits and is
  M20.4-relevant (M20.4 settler uses pid + is_about
  semantically, not turn_id strings).

- **Mode C (`by_turn_id_v1`) is byte-identical to v1.**
  No risk of v2 silently shifting v1 report numbers.

- **Placeholder resolution is a runner-time pure
  function over `replay_history`.** No fixture
  content change required; the runner resolves
  `"turn_<assistant_prior_turn_id>"` against the
  actual prior assistant turn at replay time.

- **PID normalization table is a Python constant
  with 4-5 entries.** `assistant` / `hutao` /
  `clawdgroupchat_bot` / `hutao_assistant` →
  `bot`. Other names lowercase passthrough.
  Override via JSON file is future-proofing.

- **All v2 changes are isolated to M18.7.1 files.**
  CLAUDE.md's "no new fields without bump" is
  honored by the explicit `scoring_mode` field
  and the v1-byte-identical Mode C.

## Open questions for the user

- **Q1**: Default to `by_pid` in v2 or stay on v1?
  Recommend: `by_pid`.
- **Q3**: Joint empty-skip rule? Recommend: skip
  if both empty on a sub-axis.
- **D1**: Approve / reject the design.
- **D2-D4**: Q1, Q3, pid table entries, P0-step2
  revert question (out of v2 scope).

## Estimated work

~400-550 lines, all in M18.7.1 territory
(`m18_7_1_calibration.py`,
`scripts/run_m18_7_1_real_llm_calibration.py`,
`tests/test_m18_7_1_calibration.py`).

## Why v2 is the right next step

P0 prompt fix failed. Remaining problems are
measurement + inputs. Measurement is the cheaper,
lower-risk, higher-information step. Once v2 is in,
subsequent prompt and fixture changes can be
evaluated correctly. v2 requires no LLM call
during development (all pure functions +
FakeJSONLLM tests).
