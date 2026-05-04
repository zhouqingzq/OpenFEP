# M10.0 Acceptance Report — Self-Initiated Exploration Agenda

**Milestone:** M10.0
**Source prompt:** [prompts/M10.0_Work_Prompt.md](../prompts/M10.0_Work_Prompt.md)
**Report date:** 2026-05-04 (revised v2 after second remediation)
**Verdict:** **ACCEPT** (P1/P2 issues from v1 review resolved)

## Remediation summary (v2)

| Issue | Severity | Fix |
|-------|----------|-----|
| Budget was lifetime exhaustion | P1 | `budget_remaining` now resets to `1.0 - total_budget_cost` each turn from fresh base 1.0; integration passes `budget_spent=0.0` each turn |
| Citation audit wire was dead code | P2 | `agent.latest_citation_audit` stored after generation+audit; `_produce_self_thought_events_for_turn` reads it for next-turn trigger. Note: audit runs after generation, so detection is always one turn delayed (inherent to the architecture). |

## Current test results

```text
M10.0: 24/24 passed (including per-turn budget reset test)
M6+M9 regression: 154/154 passed
```

## Verification checkpoints

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SelfThoughtEvent enters bus | PASS | `COGNITIVE_EVENT_TYPES`, factory, consumers registered |
| Consumed by CognitiveLoop | PASS | `_derive_self_agenda` reads `SelfThoughtEvent` from events |
| Cooldown starts and decays | PASS | `cooldown=3` on self-thought consumed, decrements per turn, blocks producer while active |
| Budget per-turn (not lifetime) | PASS | `budget_remaining` resets from 1.0 each turn; integration passes fresh `budget_spent=0.0` |
| Dedupe by gap_id | PASS | `prior_gap_ids` derived from previous `active_exploration_target` |
| Resolved gaps removed | PASS | Only gaps still in current gap list are carried forward |
| 8 triggers detectable | PASS | All trigger types have detection logic; 7 of 8 have real data sources in integration |
| Exploration respects budget | PASS | Per-turn cap enforced by `LoopControl.should_produce()` |
| No fabricated evidence | PASS | `evidence_event_ids` hardcoded to empty tuple |
| No direct prompt mutation | PASS | Forbidden keys excluded; frozen dataclass |

## Known boundaries (not blocking)

1. **Citation audit is one-turn-delayed**: The audit runs after generation; self-thought production reads the *previous* turn's audit from `agent.latest_citation_audit`. On the very first turn, no prior audit exists, so `citation_audit_failure` can only trigger from turn 2 onward. This is an inherent architectural constraint, not a bug.

2. **`commitment_tension` channel not standardized**: Relies on `channels.get("commitment_tension")` which observers may not emit.

3. **`open_uncertainty_duration` is approximate**: Estimated as `self_thought_count + 1` rather than per-gap age tracking. Adequate for MVP; per-gap age tracking could be added in M11.

## Key files

| Component | Path |
|-----------|------|
| SelfThoughtEvent | `segmentum/cognitive_events.py` |
| SelfAgenda + derivation | `segmentum/cognitive_state.py` |
| Producer + LoopControl + Policy | `segmentum/exploration.py` |
| Conversation integration | `segmentum/dialogue/conversation_loop.py` |
| Tests (24) | `tests/test_m10_0_self_thought.py` |

## Conclusion

**ACCEPT.** Cooldown cycle, per-turn budget, dedupe, gap resolution, and citation audit wiring all verified in cross-turn integration tests. The system can initiate bounded clarification/repair from internal uncertainty without bypassing the message bus, state ownership, memory evidence, or response contract.
