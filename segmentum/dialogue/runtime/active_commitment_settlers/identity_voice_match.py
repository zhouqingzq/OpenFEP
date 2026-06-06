"""M20.1 §5b reference: IdentityVoiceMatchLLMJudgeSettler.

Wraps the existing M19.x `surface_consistency_verification` LLM call.
The LLM call itself is performed in `mvp_loop.py`; this settler only
reads the bounded audit envelope from `observation_context` and
maps the LLM outcome to a M20.1 outcome.

M20.1 does not call an LLM directly from this settler. The LLM
self-audit is already produced by the conscious loop. The settler
only reads its result.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
)

from ._common import bounded_evidence_refs, clamp_to_unit_interval


# Map M19.x surface_intent_outcome -> M20.1 outcome.
_M19_TO_M20: dict[str, str] = {
    "consistent": "confirmed",
    "drifted_intent": "violated",
    "drifted_self_id": "violated",
    "drifted_voice": "violated",
    "ambiguous": "ambiguous",
}


class IdentityVoiceMatchLLMJudgeSettler:
    """Reference LLM-judge settler for `identity_voice_match`."""

    SETTLER_TYPE: str = "llm_judge"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_identity_voice"

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        surface = observation_context.get("surface_consistency_verification")
        if not isinstance(surface, Mapping):
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        surface_outcome = str(surface.get("surface_intent_outcome", "") or "").lower()
        outcome = _M19_TO_M20.get(surface_outcome)
        if outcome is None:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        observation_evidence_refs = list(surface.get("evidence_refs", [])) if isinstance(surface.get("evidence_refs"), list) else None
        evidence_refs = bounded_evidence_refs(
            list(commitment.evidence_refs),
            observation_evidence_refs,
        )
        if not evidence_refs:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        # Binary observable: magnitude is 1.0 on confirmed/violated.
        # For `ambiguous`, the magnitude falls back to 0.5
        # (magnitude_defaulted) since the LLM did not commit.
        if outcome == "ambiguous":
            magnitude = 0.5
            reason_codes = ("settler_llm_judge", "magnitude_defaulted")
        else:
            magnitude = 1.0
            reason_codes = ("settler_llm_judge",)

        # LLM confidence is preserved in the audit envelope but does
        # NOT become magnitude. Magnitude is the binary ε-size.
        llm_confidence = clamp_to_unit_interval(
            float(surface.get("confidence", 0.0) or 0.0)
        )
        _ = llm_confidence  # captured for future M20.2 promotion logic

        return SettledValue(
            commit_id=commitment.commit_id,
            outcome=outcome,
            magnitude=magnitude,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )
