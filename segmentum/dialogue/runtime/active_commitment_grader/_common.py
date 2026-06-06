"""Shared helpers for the M20.2 routing stubs."""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    GradedCorrectionDecision,
    build_graded_correction_routed_event,
    record_active_commitment_event,
)


def emit_routed(
    decision: GradedCorrectionDecision,
    *,
    state: dict,
    bus: list,
) -> dict[str, Any]:
    """Append the `GradedCorrectionRouted` audit event to bus and state.

    Returns the event dict.
    """
    event = build_graded_correction_routed_event(decision)
    bus.append(event)
    record_active_commitment_event(state, event)
    return event


def is_advisory_compatible(
    decision: GradedCorrectionDecision,
    *,
    non_advisory_fields: tuple[str, ...] = (),
) -> bool:
    """Return True if a `same_turn` decision is advisory-compatible.

    M20.2 §7: `same_turn` corrections MUST be advisory only. If the
    owner's write path tries to write a non-advisory field, the
    existing owner write path MUST refuse and M20.2 emits
    `CorrectionRejected` with `reason_code = "same_turn_not_advisory"`.
    M20.2.1 may read `non_advisory_fields` from the owner write
    function's signature. In M20.2 the routing stubs default to
    `non_advisory_fields = ()` (no rejection path is exercised).
    """
    if decision.correction_level != "same_turn":
        return True
    return len(non_advisory_fields) == 0


def owner_state_for(
    owner_id: str,
    state: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Return the read-only owner-state snapshot the dispatcher reads."""
    if not isinstance(state, Mapping):
        return {}
    return state
