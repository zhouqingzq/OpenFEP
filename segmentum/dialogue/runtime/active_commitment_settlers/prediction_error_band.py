"""M20.1 §5a reference: PredictionErrorBandDeterministicSettler.

Reads the current turn's M13.2 `prediction_settlements` rows from
`observation_context` (rows are dicts with `prediction_id` and a
band-shaped value). Compares the actual band to the expected band in
the commitment's `observable_payload`. Returns `confirmed` if
matched, `violated` otherwise, `uncertain` if the actual band is
missing.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
    compute_magnitude,
)

from ._common import bounded_evidence_refs


_BAND_VALUES: frozenset[str] = frozenset({"low", "med", "high"})


def _band_to_int(band: str) -> float:
    return {"low": 0.0, "med": 1.0, "high": 2.0}.get(band, 0.0)


class PredictionErrorBandDeterministicSettler:
    """Reference deterministic settler for `prediction_error_band`."""

    SETTLER_TYPE: str = "deterministic"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_prediction_lock"

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload)
        prediction_id = str(payload.get("prediction_id", "") or "")
        expected_band = str(payload.get("band", "") or "")
        if not prediction_id or expected_band not in _BAND_VALUES:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        rows = observation_context.get("prediction_settlements")
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
            if isinstance(row, Mapping) and row.get("prediction_id") == prediction_id:
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

        actual_band = str(match.get("band", "") or "")
        if actual_band not in _BAND_VALUES:
            # Band missing: emit `uncertain` (per §5a).
            evidence_refs = bounded_evidence_refs(
                list(commitment.evidence_refs),
                list(match.get("evidence_refs", [])) if isinstance(match.get("evidence_refs"), list) else None,
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
                outcome="uncertain",
                magnitude=0.5,
                evidence_refs=evidence_refs,
                reason_codes=("settler_deterministic", "magnitude_defaulted"),
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
            )

        outcome = "confirmed" if actual_band == expected_band else "violated"
        # band delta: {0, 1, 2} -> / 2.0 (§3a scale=1.0; raw delta / scale
        # is in [0, 1]).
        magnitude, extra_codes = compute_magnitude(
            observable="prediction_error_band",
            observable_payload=payload,
            committed_value=_band_to_int(actual_band),
            expected_value=_band_to_int(expected_band),
        )
        reason_codes = ("settler_deterministic",) + tuple(extra_codes)
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
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )
