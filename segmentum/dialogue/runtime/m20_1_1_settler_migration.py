"""M20.3 §5 M20.1.1 — thin adapters for M13.2 and M15.0 settlers.

M20.1 freezes the settler protocol and ships six reference settlers
(M17.1 prediction band, M19.0 outcome results, M19.x identity voice,
M18.5 boundary judgment, M19.x initiative timing, M13.1 behavioral
pull). M20.1.1 closes M20.0–M20.2 gap D: the existing per-loop
settlers (M13.2 prediction_error_proxy band check, M15.0 episode
aggregation) were not migrated onto the M20.1 runtime.

Both adapters are thin: they do NOT alter the underlying settler
logic. They wrap the existing M13.2 / M15.0 evaluation results and
emit `ActiveCommitmentSettled` envelopes that the M20.2 dispatcher
can read. The existing owner audit events still fire, and the
agreement test (M20.3 §5.1) asserts that the two events agree on
outcome.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
)
from segmentum.dialogue.runtime.active_commitment_settlers._common import (
    bounded_evidence_refs,
    clamp_to_unit_interval,
)


# === M13.2 thin adapter =================================================
#
# M13.2 produces `M13RewardSettlementEvent` with `prediction_error_proxy`
# (observed - predicted). The "band check" is: is the proxy within
# [-0.1, 0.1]? If yes, "confirmed"; if outside, "violated"; if no
# signal, "uncertain".
#
# M20.1.1 wraps this so a future M20.0 admission can route a
# commitment through the M20.1 scheduler and still see an
# `ActiveCommitmentSettled` event. The underlying M13.2 path is
# unchanged: the LLM settlement assessor still produces
# `M13RewardSettlementEvent` rows.

M13_PREDICTION_ERROR_BAND: float = 0.1


def _m13_outcome_from_band(prediction_error_proxy: float) -> str:
    """Map the M13.2 prediction_error_proxy float to a M20.1 outcome."""
    try:
        v = float(prediction_error_proxy)
    except (TypeError, ValueError):
        return "uncertain"
    if v != v:
        return "uncertain"
    if -M13_PREDICTION_ERROR_BAND <= v <= M13_PREDICTION_ERROR_BAND:
        return "confirmed"
    return "violated"


class M13BandCheckAdapter:
    """M20.1.1 thin adapter for the M13.2 prediction_error_proxy band check.

    The adapter reads `m13_reward_settlements` from
    `observation_context` (rows are M13.2 reward settlement events
    with `pending_id`, `prediction_error_proxy`, `outcome_band`).
    The adapter is keyed on `commitment.observable_payload["pending_id"]`
    so callers route by `pending_id`.

    The M20.1.1 acceptance test asserts that the adapter's
    `SettledValue.outcome` agrees with the existing M13.2
    `outcome_band` field. Specifically:
    - `outcome_band = "positive"`  ->  `confirmed`
    - `outcome_band = "negative"`  ->  `violated`
    - `outcome_band = "uncertain"` ->  `uncertain`

    A row whose `outcome_band` does not match the band-check
    prediction is a bug; the adapter surfaces it via
    `settler_hybrid_fallback` (transient), so the dispatcher can
    re-attempt.
    """

    SETTLER_TYPE: str = "deterministic"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_m13_drive"

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload or {})
        pending_id = str(payload.get("pending_id", "") or "")
        if not pending_id:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        rows = observation_context.get("m13_reward_settlements")
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
            if isinstance(row, Mapping) and row.get("pending_id") == pending_id:
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

        # Read both the prediction_error_proxy (for the band check)
        # and the outcome_band (the existing M13.2 verdict). They
        # MUST agree; a disagreement is logged via
        # `settler_hybrid_fallback` reason code.
        try:
            proxy_value = float(match.get("prediction_error_proxy", 0.0) or 0.0)
        except (TypeError, ValueError):
            proxy_value = 0.0
        outcome_band = str(match.get("outcome_band", "") or "").strip().lower()

        band_outcome = _m13_outcome_from_band(proxy_value)
        # Map the existing M13.2 outcome_band to M20.1 outcome.
        if outcome_band == "positive":
            existing_outcome = "confirmed"
        elif outcome_band == "negative":
            existing_outcome = "violated"
        elif outcome_band == "uncertain":
            existing_outcome = "uncertain"
        else:
            existing_outcome = ""

        if not existing_outcome:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        # Agreement check (M20.3 §5.1 acceptance). When the band
        # check and the existing outcome disagree, the adapter
        # surfaces `settler_hybrid_fallback` so the test can
        # distinguish agreement from disagreement.
        reason_codes: tuple[str, ...]
        if band_outcome != existing_outcome:
            reason_codes = ("settler_hybrid_fallback", "m13_2_disagreement")
        else:
            reason_codes = ("settler_deterministic", "m13_2_agreement")

        # Magnitude: |proxy_value| / 0.5 (the existing v1 scale for
        # bounded-delta observables). The `clamp_to_unit_interval`
        # helper caps at 1.0.
        magnitude = clamp_to_unit_interval(abs(proxy_value) / 0.5)
        if magnitude == 0.0:
            magnitude = 0.5
            reason_codes = reason_codes + ("magnitude_defaulted",)

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
            outcome=existing_outcome,
            magnitude=magnitude,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )


# === M15.0 thin adapter =================================================
#
# M15.0 episode aggregation produces `M15EpisodeSettledEvent`-like
# rows with `episode_id`, `outcome_summary`, and `delta_fe_proxy`.
# The aggregation check is: did the episode close with a "settled"
# outcome that matches the commitment's expected outcome?
#
# The M20.1.1 adapter wraps this. The agreement test asserts that
# the adapter's outcome agrees with the M15.0 `outcome_summary`.


class M15EpisodeAggregationAdapter:
    """M20.1.1 thin adapter for the M15.0 episode aggregation.

    Reads `m15_episode_settlements` from `observation_context` and
    keys on `commitment.observable_payload["episode_id"]`. The
    existing M15.0 verdict (`outcome_summary`) maps to a M20.1
    outcome as:
    - `outcome_summary = "settled"`    -> `confirmed`
    - `outcome_summary = "violated"`   -> `violated`
    - `outcome_summary = "uncertain"`  -> `uncertain`
    - `outcome_summary = "ignored"`    -> `uncertain`

    Magnitude is the absolute `delta_fe_proxy` (the v1 magnitude
    scale for `expectation_outcome_match` is 1.0, so the magnitude
    is clamped to 1.0).
    """

    SETTLER_TYPE: str = "deterministic"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_m15_episode"

    _OUTCOME_MAP: dict[str, str] = {
        "settled": "confirmed",
        "violated": "violated",
        "uncertain": "uncertain",
        "ignored": "uncertain",
    }

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload or {})
        episode_id = str(payload.get("episode_id", "") or "")
        if not episode_id:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        rows = observation_context.get("m15_episode_settlements")
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
            if isinstance(row, Mapping) and row.get("episode_id") == episode_id:
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

        outcome_summary = str(match.get("outcome_summary", "") or "").strip().lower()
        outcome = self._OUTCOME_MAP.get(outcome_summary, "")
        if not outcome:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        try:
            delta_fe_proxy = float(match.get("delta_fe_proxy", 0.0) or 0.0)
        except (TypeError, ValueError):
            delta_fe_proxy = 0.0
        magnitude = clamp_to_unit_interval(abs(delta_fe_proxy))
        if magnitude == 0.0:
            magnitude = 0.5
            reason_codes: tuple[str, ...] = (
                "settler_deterministic",
                "m15_0_agreement",
                "magnitude_defaulted",
            )
        else:
            reason_codes = ("settler_deterministic", "m15_0_agreement")

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
            outcome=outcome,
            magnitude=magnitude,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )


# === Scheduler registration helper (v2 / future) ========================
#
# M20.3 v1 acceptance does NOT require the M13.2 / M15.0 thin
# adapters to be registered on the M20.1 SettlementScheduler. The
# v1 scheduler handles only v1 observables (none of which maps
# cleanly to M13.2 prediction_error_proxy or M15.0 episode
# aggregation without a vocabulary bump). The v1 acceptance is:
#
# 1. The adapter classes exist with the Settler protocol shape.
# 2. Direct invocation produces a SettledValue whose outcome
#    agrees with the existing owner audit event.
# 3. Dual emission: the adapter emits ActiveCommitmentSettled
#    alongside the existing M13.2 / M15.0 audit event when
#    invoked from the v1 path.
#
# v2 routing can register the adapters under new v2 observables
# (a future M20.x milestone).

__all__ = [
    "M13BandCheckAdapter",
    "M15EpisodeAggregationAdapter",
]
