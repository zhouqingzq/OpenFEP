"""M20.2 §5 routing stub: `expire` is no-routing.

`expire` corrections emit no `GradedCorrectionRouted` event (the
level is non-routing by M20.2 §5). The dispatcher routes `expire`
through `build_correction_deferred_event` and never calls any
owner write function. This module exists for symmetry with the
other routing levels; its body is intentionally a no-op.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import GradedCorrectionDecision


def route_expire(
    decision: GradedCorrectionDecision,
    *,
    state: dict,
    bus: list,
    owner_state_snapshot: Mapping[str, Any] | None = None,
) -> None:
    """`expire` is a no-routing level. The dispatcher emits the
    `CorrectionDeferred` audit event; this stub does nothing.
    """
    return None
