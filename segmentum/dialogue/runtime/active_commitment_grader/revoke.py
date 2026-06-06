"""M20.2 §5 routing: `revoke` to owner revocation path.

M20.2 ships the audit envelope; M20.2.1 wires the real write
path. v1 scope: `self_cognition_calibrated_tendencies`. Other
`revoke`-accepting owners (`mismatch_memory_fast`,
`self_repair_expectation`, `outreach_intent_registry`) remain
no-op for the write path; the `GradedCorrectionRouted` event
still fires.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
)

from ._common import emit_routed
from ._write_paths import run_m20_2_1_write_path


def route_revoke(
    decision: GradedCorrectionDecision,
    *,
    state: dict,
    bus: list,
    owner_state_snapshot: Mapping[str, Any] | None = None,
    commitment: ActiveCommitment | None = None,
) -> dict[str, Any]:
    """Emit `GradedCorrectionRouted` and apply the v1-scope write path."""
    event = emit_routed(decision, state=state, bus=bus)
    if commitment is not None:
        run_m20_2_1_write_path(
            level="revoke",
            owner_id=decision.routed_owner_id,
            decision=decision,
            commitment=commitment,
            state=state,
            bus=bus,
        )
    return event
