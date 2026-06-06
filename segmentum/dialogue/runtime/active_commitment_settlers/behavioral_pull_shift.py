"""M20.1 §5d reference: BehavioralPullShiftSilentSettler.

The `behavioral_pull_shift` observable carries the `silent` hint in v1
because the M13.0 traction update flow is its own concern. The M20.1
silent settler does NOT interpret the observable payload. It records
the absence of settlement in the owner observability surface and
returns `NoSettlement` with `settler_silent_carry_forward`.

The scheduler treats this as a valid terminal state for that attempt
and emits a `NoSettlementMade` audit event.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
)


class BehavioralPullShiftSilentSettler:
    """Reference silent settler for `behavioral_pull_shift`."""

    SETTLER_TYPE: str = "silent"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_m13_drive"

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> NoSettlement:
        return NoSettlement(
            commit_id=commitment.commit_id,
            reason_code="settler_silent_carry_forward",
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=commitment.engineering_proxy_label,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
        )
