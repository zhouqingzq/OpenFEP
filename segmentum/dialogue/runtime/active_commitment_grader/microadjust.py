"""M20.2 §5 routing: `microadjust` to m13_drive_state traction bump.

M20.2 ships the audit envelope; M20.2.1 wires the real write
path. v1 scope: `m13_drive_state`. Other `microadjust`-accepting
owners (`m15_episode_ledger`, `mismatch_memory_fast`,
`user_prediction_ledger`, `memory_dynamics_control_guidance`,
`group_addressee_graph`) remain no-op for the write path; the
`GradedCorrectionRouted` event still fires.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
)

from ._common import emit_routed
from ._write_paths import run_m20_2_1_write_path


def route_microadjust(
    decision: GradedCorrectionDecision,
    *,
    state: dict,
    bus: list,
    owner_state_snapshot: Mapping[str, Any] | None = None,
    commitment: ActiveCommitment | None = None,
) -> dict[str, Any]:
    """Emit `GradedCorrectionRouted` and apply the v1-scope write path.

    `commitment` is the originating `ActiveCommitment` reconstructed
    from observability by the dispatcher. The write path needs
    it to read dispatch context (action, user_id, observable_payload).
    When absent (e.g. legacy observability without a stored
    commitment), the write path is skipped; the audit envelope
    still fires.
    """
    event = emit_routed(decision, state=state, bus=bus)
    if commitment is not None:
        run_m20_2_1_write_path(
            level="microadjust",
            owner_id=decision.routed_owner_id,
            decision=decision,
            commitment=commitment,
            state=state,
            bus=bus,
        )
    return event
