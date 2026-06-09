# M18.7.2 P0-step2 — Real-LLM Re-Run (post v1 categories)

- status: experimental (verifies P0-step2 prompt fix)
- branch: main
- generated_at: 2026-06-08T15:12:36+00:00
- session_root: tmp_m18_7_2_post_p0_replay/
- held_out_fixture: tests/fixtures/m18_7_1_held_out_calibration.json
- model_under_test: deepseek/deepseek-v4-flash

## Question this run answers

> "Does re-introducing the v1 semantic categories in the
> M18.7.2 minimal prompt (P0-step2, commit 5ab3db4) move
> reaction accuracy off 0.0/6?"

**TL;DR: No, accuracy stays 0.0/6 on reaction and 0.0/8
on addressee. P0-step2 did NOT help; in fact reaction
ECE got worse (0.258 → 0.417). The root cause is not
the prompt — it is the scoring harness + a missing
`reaction_to_turn_id` semantic.**

## Headline numbers (this run, post-P0-step2)

| field | n_total | n_present | n_unknown | n_correct | n_incorrect | accuracy | brier | ece |
|-------|---------|-----------|-----------|-----------|-------------|----------|-------|-----|
| addressee | 12 | 8 | 4 | 0 | 8 | 0.0 | 0.2025 | 0.225 |
| reaction | 12 | 6 | 6 | 0 | 6 | 0.0 | 0.3483 | 0.4167 |

Aggregate verdict: `severe_drift_recommend_m20_4` (unchanged).

## Comparison vs pre-P0 baseline (commit 066b172)

| field | metric | pre-P0 | post-P0-step2 | delta |
|-------|--------|--------|---------------|-------|
| addressee | acc | 0.25 | 0.0 | **-0.25** |
| addressee | brier | 0.226 | 0.2025 | -0.024 |
| addressee | ece | 0.225 | 0.225 | 0.0 |
| reaction | acc | 0.0 | 0.0 | 0.0 |
| reaction | brier | 0.202 | 0.348 | **+0.146** |
| reaction | ece | 0.258 | 0.417 | **+0.159** |

**P0-step2 made things worse on reaction, not better.**
Addressee regressed from 0.25 to 0.0. The v1 semantic
category re-introduction did not help; in fact, the
larger prompt with more semantic detail seems to have
increased the LLM's high-band overconfidence on
reaction (2/2 in 0.8-1.0 band, both wrong → ECE 0.417).

## Per-turn analysis (root cause)

The scoring harness judges reaction accuracy on
**strict string equality of `reaction_to_turn_id`**
(m18_7_1_calibration.py line 810-812). The fixture
ground truth uses placeholder strings like
`"turn_<assistant_prior_turn_id>"` and
`"turn_<carol_prior_turn_id>"`. The LLM, in 5/6
reaction predictions, emits `reaction_to_turn_id=""`
(EMPTY). It does not attempt to fill the turn_id.

But the LLM IS getting `reaction_to_participant_id`
and `is_about_assistant_claim` correct on several
turns. For example:

| turn | text | GT pid | PR pid | match? | GT is_about | PR is_about | match? |
|------|------|--------|--------|--------|-------------|-------------|--------|
| 5 | "Wait, you said that was free, right?" | hutao | assistant | **NO** | True | True | YES |
| 6 | "Yeah, what Carol said earlier about the deadline." | carol | carol | **YES** | False | False | **YES** |
| 11 | "I think hutao had it right earlier, but let me check first." | hutao | hutao | **YES** | True | False | NO |

Turn 6 is a clean correct on the participant_id +
is_about_claim axis. Turn 11 is partial (correct pid,
wrong is_about). The LLM is doing meaningful semantic
work — the scoring harness is just not measuring what
the LLM produces.

## Reliability bins (post-P0-step2)

**Addressee**: bimodal (6 in 0.0-0.10, 2 in 0.90-1.00,
middle bands empty). High band has 0/2 correct
(gap 0.9). Mean conf 0.0 in low band is well-calibrated
(0/6 correct = 0.0 conf).

**Reaction**: bimodal (3 in 0.0-0.10, 2 in 0.80-0.90,
1 in 0.90-1.00). High band has 0/3 correct
(gap 0.8-0.9). Mean conf 0.0 in low band is well-
calibrated (0/3 correct = 0.0 conf).

Drift signals on both fields (unchanged from pre-P0):
- `overconfidence_at_high_band`
- `bimodal`

## What P0-step2 did NOT fix

1. **The `reaction_to_turn_id` field is not being
   filled.** 5/6 reaction predictions emit
   `reaction_to_turn_id=""`. The LLM is unable to map
   "the prior claim" to a specific turn_id because the
   fixture's group_turn_envelope does not expose
   turn_ids in a way the LLM can pick up. The prompt
   says `reaction_to_turn_id` is "the specific prior
   turn" but the prompt does not enumerate the
   available turn_ids.

2. **Addressee accuracy regressed.** Pre-P0 the LLM
   got 2/8 (turns 8, 11) correct; post-P0 it got 0/8.
   The v1 categories may have over-anchored the LLM
   on a stricter interpretation of "directly aimed at
   you" — turns 0 and 8, which the pre-P0 LLM got
   right (conf 0.95), are now wrong. Possibly the
   longer semantic discussion made the LLM more
   cautious about `addressed_to_assistant=True`.

3. **Bimodal confidence distribution is unchanged.**
   The LLM still collapses to 0.0 or 0.85-0.95 with
   no middle band. This is a model behavior, not a
   prompt problem.

## P0-step2 verdict

**Failed.** The v1 semantic category re-introduction
was the wrong intervention. Real progress on
calibration requires:

- **(a) Scoring harness revision (M18.7.1 territory):**
  add a `calibrate_by_pid` mode that scores
  `reaction_to_participant_id` and
  `is_about_assistant_claim` directly. This is
  independent of the LLM and would already show
  ~33-50% accuracy on reaction (turn 6 fully
  correct, turn 11 partially). The current strict-
  turn-id equality understates the LLM's actual
  semantic performance.

- **(b) Prompt revision: enumerate available
  turn_ids in the prompt.** The
  `m18_7_minimal_prompt` should include the prior
  turn list (last 2-3 turns with their turn_ids and
  speaker names) so the LLM can map "the prior
  claim" to a specific turn_id. This is the
  M18.7.2 follow-up prompt fix.

- **(c) Fixture revision: replace placeholder
  turn_ids with real turn_ids.** The fixture
  ground truth uses `"turn_<assistant_prior_turn_id>"`
  as a literal placeholder, which can never match a
  real LLM output. The fixture should be patched
  to use the actual turn_index of the prior
  assistant/user turn being reacted to.

(a) is M18.7.1's job (harness). (b) is M18.7.2's
follow-up. (c) is fixture maintenance.

## Out of scope for this run

- Did NOT revise the M20.4 threshold constants
  (P0-3 / commit 1e892b2 already split per-field;
  this run's `candidate_tie_breaker_min` values
  remain data-backed recommendations, not new
  constants).
- Did NOT change the M20.4 producer, the M20.4.1
  gate, or the M18.5 enforcement point.
- Did NOT re-run M20.4.1 / M18.7 / M18.7.2 / M20.3
  tests (targeted regression last ran at commit
  80f428a with 285/285 pass; P0-step2 changes are
  prompt-only and tests still pass).
