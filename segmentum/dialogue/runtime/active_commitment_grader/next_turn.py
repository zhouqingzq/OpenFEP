"""M20.2 §5 routing: `next_turn` to m13_drive_state traction bump.

M20.2 ships the audit envelope; M20.2.1 wires the real write
path. v1 scope: `m13_drive_state` (per the M20.2.1 milestone
decision). All other `next_turn`-accepting owners remain no-op
for the write path; the `GradedCorrectionRouted` event still
fires.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
)

from ._common import emit_routed
from ._write_paths import run_m20_2_1_write_path


def route_next_turn(
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
            level="next_turn",
            owner_id=decision.routed_owner_id,
            decision=decision,
            commitment=commitment,
            state=state,
            bus=bus,
        )
    return event
