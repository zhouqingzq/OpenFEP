"""M20.2 §5 + §7 routing: `same_turn` is advisory only.

`same_turn` corrections are advisory by M20.2 §7. The write path
MUST refuse to write a non-advisory field. M20.2.1 v1 scope:
`m13_drive_state` writes a `recent_action_trace` priority nudge
(advisory field) and emits an `M13PullNudge` event.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
    build_correction_rejected_event,
    record_active_commitment_event,
)

from ._common import emit_routed, is_advisory_compatible
from ._write_paths import run_m20_2_1_write_path


def route_same_turn(
    decision: GradedCorrectionDecision,
    *,
    state: dict,
    bus: list,
    owner_state_snapshot: Mapping[str, Any] | None = None,
    non_advisory_fields: tuple[str, ...] = (),
    commitment: ActiveCommitment | None = None,
) -> dict[str, Any]:
    """Emit `GradedCorrectionRouted` if advisory-compatible, else `CorrectionRejected`."""
    if not is_advisory_compatible(decision, non_advisory_fields=non_advisory_fields):
        # Re-route as a rejection. The decision is rewritten in place
        # only for the audit envelope; the dispatcher is pure and
        # does not see this.
        rejected_decision = GradedCorrectionDecision(
            commit_id=decision.commit_id,
            correction_level=decision.correction_level,
            routed_owner_id=decision.routed_owner_id,
            reason_codes=("same_turn_not_advisory",),
            evidence_refs=decision.evidence_refs,
            magnitude_before=decision.magnitude_before,
            magnitude_after=decision.magnitude_before,
            outcome=decision.outcome,
            at=decision.at,
            turn_index=decision.turn_index,
            engineering_proxy_label=decision.engineering_proxy_label,
            deferred=True,
        )
        event = build_correction_rejected_event(rejected_decision)
        bus.append(event)
        record_active_commitment_event(state, event)
        return event
    event = emit_routed(decision, state=state, bus=bus)
    if commitment is not None:
        run_m20_2_1_write_path(
            level="same_turn",
            owner_id=decision.routed_owner_id,
            decision=decision,
            commitment=commitment,
            state=state,
            bus=bus,
        )
    return event
