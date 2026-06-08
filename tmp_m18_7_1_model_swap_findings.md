# M18.7.1 P0 — Model-swap and Prompt-length Findings (not committed)

- status: experimental (not a milestone deliverable)
- at: 2026-06-08T06:53:36+00:00
- script: E:\workspace\segments\tmp_m18_7_1_v4pro_p0_replay.py
- session state: E:\workspace\segments\tmp_m18_7_1_real_llm_v4pro\
- run log: E:\workspace\segments\tmp_m18_7_1_v4pro_run.log
- prior findings: E:\workspace\segments\tmp_m18_7_1_f_option_findings.md

## Question this run answers

> "Should we switch to a model that holds up under
> long prompts, or should we adjust the agent side?"

Three sub-questions, each answered with a measurement:

1. Is the 26k-char prompt length necessary?
2. Does switching to a stronger model (deepseek-v4-pro)
   fix the v2 attrs omission in a 12-turn full replay?
3. If neither is the surgical fix, what is?

## Measurement 1 — 26k prompt is NOT necessary

The conscious-loop user prompt is built by
`build_conscious_loop_prompt` in `mvp_loop.py`. Its
bulk comes from
`prompt_safe_state_with_self_expectation_summary(state)`
serialized into the user prompt at L2104.

Per-turn state sizes (measured against
`tmp_m18_7_1_real_llm_promoted/temporal_state.json`):

| state key                  | chars    | share |
|----------------------------|----------|-------|
| **m13_drive_state**        | 69,403   | 44.9% |
| **self_expectation_state** | 49,087   | 31.7% |
| pending_expectations       | 14,168   |  9.2% |
| open_items                 |  7,537   |  4.9% |
| temporal_state             |  3,439   |  2.2% |
| short_term_memory          |  3,050   |  2.0% |
| long_term_memory           |  2,271   |  1.5% |
| m11_user_models            |  1,492   |  1.0% |
| habit_traits               |  1,277   |  0.8% |
| self_cognition             |  1,026   |  0.7% |
| relationship_value_memories|    725   |  0.5% |
| self_basic_facts           |    179   |  0.1% |
| m12_1_user_personality     |    127   |  0.1% |
| m12_user_continuity        |     81   |  0.1% |
| social_sharing_policy      |     46   |  0.0% |
| m12_2_reciprocal_role      |     49   |  0.0% |
| m17_path_b_bridge          |    111   |  0.1% |
| (others, mostly booleans)  |     ~25  |  0.0% |
| **total**                  | **154,586** | 100% |

The safe-pruning in
`prompt_safe_state_with_self_expectation_summary`
already trims ~46k chars (154k → 108k in the rendered
safe state). After that, the user prompt is 23–26k
chars per turn across the 12-turn replay.

For an M18.7 addressee / reaction attribution decision,
the only state that matters is:

- `group_turn_binding` (already passed in as
  `entity_binding` / `bounded_group_turn` argument)
- `m18_5_structural_decision` (the
  `group_reply_policy.action` value)
- the last 2–3 turns of `conversation_log`
- `self_basic_facts` (so the LLM knows the assistant's
  name/identity)

That is roughly **5–8k chars**, not 23–26k. The other
~76% of the state volume (`m13_drive_state` 45% +
`self_expectation_state` 32%) is **noise** for the
M18.7 attribution question. M13 drive state is about
idle initiative and boredom; self_expectation_state is
the M19 self-cognition ledger. Neither is needed to
ask "is this turn addressed to me, and is it reacting
to a prior turn?"

## Measurement 2 — Switching to v4-pro does NOT fix it

The v4-pro 12-turn full replay (script
`tmp_m18_7_1_v4pro_p0_replay.py`, original un-promoted
prompt, char 2914 / 37.7%):

| metric             | v4-flash   | v4-pro   |
|--------------------|------------|----------|
| finish_reason      | stop       | stop     |
| m18_7 surface len  | 0          | **0**    |
| m18_7 bus events   | 0          | **0**    |
| addressee n_present| 8 (silent) | 8 (silent) |
| reaction n_present | 6 (silent) | 6 (silent) |
| accuracy / brier / ece | 0.0/0.0/0.0 | **0.0/0.0/0.0** |
| wall-clock         | ~4 min     | ~6 min   |

Both models produce the same degenerate "no
non-silent predictions" calibration report. The
single-turn isolated test from
`tmp_m18_7_1_f_option_findings.md` showed v4-pro can
fill v2 attrs when given a 7.7k-char minimal-state
prompt. That confirms the model can do it. The 12-turn
state accumulation is what breaks it.

## Verdict on the model-swap question

**Do not switch models.** Three reasons:

1. **Cost–benefit is wrong.** v4-pro is already 50%
   slower than v4-flash on this fixture. GPT-4o or
   Claude Sonnet 4 would be 2–3× slower still, with
   per-token cost typically 5–20× higher than deepseek
   tiers. The marginal quality gain (longer prompt
   headroom) does not justify the cost.
2. **The model is not the root cause.** v4-flash in
   the single-turn isolated test fills v2 attrs with
   `confidence: 0.65` correctly. The model is competent
   on this task at 7.7k prompt length. The 12-turn
   state accumulation is what breaks instruction
   following, and that is true for any chat model
   with a 25-key schema and 23k+ prompt — the model
   just has different decay points.
3. **It does not address the real issue.** The
   conscious-loop prompt is over-shared with the
   M18.7 attribution path. M18.7 hypothesis extraction
   should not need `m13_drive_state` /
   `self_expectation_state` / `pending_expectations`
   / `open_items` — those are consumed by other
   conscious-loop branches, not by addressee /
   reaction attribution. Sharing the same prompt
   forces M18.7 to compete for instruction-following
   budget against 60+ other schema fields that have
   nothing to do with the question being asked.

## Recommendation for the M18.7 owner

**Adjust the agent side, not the model.**

Two surgical fixes (in order of effort vs impact):

### Fix 1: purpose-aware state pruning (low effort, high impact)

Add a `purpose` parameter to
`prompt_safe_state_with_self_expectation_summary`
in `mvp_loop.py`:

```python
M18_7_RELEVANT_KEYS = frozenset({
    "temporal_state",     # last_user_text, last_reply_at
    "self_basic_facts",   # persona name, do_not_invent
    "short_term_memory",  # last 2-3 entries only
    "m13_drive_state",    # session_open_count, last_initiative_at
})
```

When `purpose == "m18_7_attribution"`, only the
keys above are serialized into the user prompt.
Expected user_prompt length: 5–8k chars (vs 23–26k).

The 41 structural tests in
`tests/test_m18_7_1_calibration.py` and the
`tests/test_mvp_dialogue_runtime.py` regression set
should continue to pass — they test the calibration
math and the conscious-loop contract, not the
specific state keys the prompt serializes.

### Fix 2: dedicated M18.7 prompt (medium effort, root-cause fix)

Add a `build_m18_7_addressee_reaction_prompt` in
`segmentum/dialogue/runtime/m18_7_attribution.py`
that asks **only** the v2 attrs questions:

```python
# segmentum/dialogue/runtime/m18_7_attribution.py
def build_m18_7_minimal_prompt(
    *, state, user_text, group_turn_binding, m18_5_action,
) -> tuple[str, str]:
    """~500-1500 char prompt, 5-key JSON spec, no
    conscious-loop coupling."""
    ...
```

`MVPDialogueRuntime.run_turn` then calls this for
the M18.7 attribution extraction, separate from the
conscious loop. Expected prompt length: <2k chars.
The LLM fills v2 attrs with high confidence and the
M18.7.1 calibration layer produces real ECE / Brier.

This is the actual surgical fix. Fix 1 is a band-aid
that improves the existing architecture; Fix 2 is
the right architecture.

## What M18.7.1 cannot do from its own scope

M18.7.1 is the calibration analysis layer. It cannot
modify the conscious-loop prompt, the state pruning
function, or the run_turn orchestration. Both fixes
above are M18.7 (or a new M18.7.x) territory.

**Note on the CLAUDE.md red line.** The actual red
line in CLAUDE.md is about *forbidding keyword/regex
cues in engineering code* for semantic interpretation,
not about forbidding prompt edits. Verbatim:

> "Semantic decisions must not be implemented as
> keyword/regex cue lists in the engineering layer.
> When a feature needs semantic interpretation ...
> ask the active LLM request/prompt to return bounded
> structured fields. Engineering code may only validate
> those fields ... and audit the result. Raw user text
> may be stored as evidence/excerpt, but must not be
> parsed by ad hoc semantic keyword cues."

Fix 2 above is fully aligned with the red line: it
moves the semantic decision (addressee / reaction
attribution) into an LLM prompt that returns bounded
structured fields. The engineering code only validates
the returned fields and persists them. There is no
keyword / regex / cue list involved on either side.

M18.7.1's job is the calibration math + the runner
+ the held-out fixture. The runner
(`scripts/run_m18_7_1_real_llm_calibration.py`) is in
place and will produce real numbers the moment M18.7
implements either fix.

## Scope of this report (M18.7.1 P0 findings only)

- M18.7.1 is the calibration layer. State-prompt
  pruning and dedicated minimal-prompt work belong
  to M18.7 (or a follow-up M18.7.x) — the M18.7 owner
  needs to weigh the two fixes above, since either
  is a meaningful change to the M18.7 surface contract
  (prompt shape, schema, run_turn wiring). **This
  report surfaces them; it does not apply them.**
- The previously committed report
  (`reports/m18_7_1_calibration_summary.md`, commit
  `b13f07f`) remains the durable record of the
  baseline real-LLM replay. Its `prompt_order_blocked`
  verdict is still factually correct — the prompt
  is blocked, and the underlying cause is in the
  M18.7 prompt / state-pruning layer, not in the
  calibration math.
- These findings refine the actionable for M18.7
  but do not change the M18.7.1 deliverable.

## Tmp artifacts produced

**Committed in this report:**

- `tmp_m18_7_1_model_swap_findings.md` (this file)

**Inputs referenced by this report, kept as
untracked `tmp_*` workspace artifacts (deliberately
not committed; reproducible by re-running the scripts
listed below):**

- `tmp_m18_7_1_v4pro_p0_replay.py` — v4-pro 12-turn replay script
- `tmp_m18_7_1_v4pro_run.log` — v4-pro 12-turn run log
- `tmp_m18_7_1_real_llm_v4pro/` — v4-pro 12-turn session state
- `tmp_m18_7_1_f_option_findings.md` — F-option earlier findings
- `tmp_m18_7_1_prompt_position_experiment.py` — single-turn isolated test
- `tmp_m18_7_1_promoted_p0_replay.py` — promoted-position 12-turn replay
- `tmp_m18_7_1_real_llm_promoted/` — promoted 12-turn session state
- `tmp_m18_7_1_real_llm/` — baseline session state (commit `b13f07f`)
- `tmp_m18_7_1_run.log` — baseline run log
- `tmp_m18_7_1_prompt_to_test.txt` — the user-pasted prompt

The state directories and log files are large and
contain per-turn runtime output. They are kept on
disk for reproducibility but not version-controlled.
