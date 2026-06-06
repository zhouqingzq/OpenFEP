"""M20.1 §5a reference: ExpectationOutcomeMatchDeterministicSettler.

Reads the current turn's M19.0 `self_expectation_outcome_results`
rows (provided by the scheduler in `observation_context`). Looks up
the row whose `source_expectation_id` and `target_context` match the
commitment's `observable_payload`. Maps the LLM outcome to one of
the M20.1 §2 outcomes.

This settler is a *pure* function of (commitment, observation_context).
It MUST NOT call an LLM.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
)

from ._common import bounded_evidence_refs


# Map the M19.0 outcome_results status enum to M20.1 outcomes.
# `confirmed` / `violated` / `uncertain` map 1:1. M19.0 does not
# produce `ambiguous` for outcome_results (the LLM's `ambiguous`
# appears in surface_consistency_verification instead).
_M19_TO_M20: dict[str, str] = {
    "confirmed": "confirmed",
    "violated": "violated",
    "uncertain": "uncertain",
}


class ExpectationOutcomeMatchDeterministicSettler:
    """Reference deterministic settler for `expectation_outcome_match`."""

    SETTLER_TYPE: str = "deterministic"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_self_expectation"

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload)
        source_expectation_id = str(payload.get("source_expectation_id", "") or "")
        target_context = str(payload.get("target_context", "") or "")
        if not source_expectation_id or not target_context:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        rows = observation_context.get("self_expectation_outcome_results")
        if not isinstance(rows, list):
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        match = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("source_expectation_id") != source_expectation_id:
                continue
            if row.get("target_context") != target_context:
                continue
            match = row
            break

        if match is None:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        m19_status = str(match.get("status", "") or "")
        outcome = _M19_TO_M20.get(m19_status)
        if outcome is None:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        # Binary observable (§3a scale=1.0): magnitude is 1.0 on
        # confirmed/violated, magnitude_defaulted falls back to 0.5 if
        # no numeric value. We do not need a numeric committed_value
        # for binary, so we use 1.0.
        magnitude = 1.0
        observation_evidence_refs = list(match.get("evidence_refs", [])) if isinstance(match.get("evidence_refs"), list) else None
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

        return SettledValue(
            commit_id=commitment.commit_id,
            outcome=outcome,
            magnitude=magnitude,
            evidence_refs=evidence_refs,
            reason_codes=("settler_deterministic",),
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )
