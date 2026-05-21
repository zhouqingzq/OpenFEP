# M14.3 Proactive Alignment

M14.3 makes Path B proactive delivery use one auditable target-selection path.
The runtime no longer treats a vague `open_items[].next_check` token as enough
to produce a visible proactive message.

## Pipeline

Idle or outbox delivery now follows this order:

1. Environment or idle tick gathers structural signals.
2. Idle path builds a compact context and retrieves memories first.
3. M13.6 evaluates memory EFE with retrieved evidence.
4. M14.0 applies idle drive rules against that M13.6 snapshot.
5. M14.3 target selection accepts only scheduled outreach, M13.6
   `memory_efe_outreach`, correction follow-up, or evidence-backed boredom
   exploration.
6. M13.3 creates or drains one proposal and M13.3 delivery assessment remains
   the only visible-text gate.

## Removed Shortcut

`open_item_next_check` remains only as a legacy compatibility trigger behind
`initiative.legacy_vague_open_item_proactive=false` by default. Vague values
such as `later`, `regular`, `someday`, and empty strings are diagnostic-only.

## Traceable Expectation

A traceable expectation is an open loop with an id, concrete source kind, and
evidence refs or bound memory ids. User-facing diagnostics should describe this
operationally as an unresolved expectation with evidence, not as subjective
desire.

## Suppression Codes

Pre-proposal codes include `initiative_disabled`, `not_opted_in`,
`cooldown_active`, `session_limit_reached`, `delivery_channel_unavailable`,
`idle_time_too_short`, `opponent_strength_pre_block`,
`context_assessment_unsafe`, and `no_traceable_proactive_target`.

Post-generation codes include `delivery_assessor_reject`,
`delivery_assessor_low_confidence`, and `empty_generation`.

M13.6 uses `memory_efe_opponent_risk` for its own risk penalty so it does not
collide with delivery-stage safety diagnostics.

## M14.2 Boundary

M14.2 still owns event durability and outbox lifecycle. M14.3 only aligns which
outbox or memory target is allowed to become a proposal, and adds daemon health
visibility for event acknowledgement ratio. Crash and idempotency boundaries
remain the M14.2 JSONL event, intent, and outbox files.

