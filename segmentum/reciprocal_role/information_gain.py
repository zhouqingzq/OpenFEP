"""Deterministic, safety-leaning candidate ranking for M12.2."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .hyperparams import BAND_ORDER
from .reciprocal_model import InformationGainCandidate
from .safety_linter import apply_safety_linter


def rank_information_gain_candidates(
    candidates: Sequence[InformationGainCandidate],
) -> tuple[InformationGainCandidate, ...]:
    allowed, _findings = apply_safety_linter(candidates)
    typed = tuple(item for item in allowed if isinstance(item, InformationGainCandidate) and not item.blocked_by_safety)
    ranked = sorted(
        typed,
        key=lambda candidate: (
            BAND_ORDER.get(candidate.risk_band, 9),
            -BAND_ORDER.get(candidate.expected_gain_band, 0),
            0 if candidate.source == "explicit_user_request" else 1,
            -_latest_turn_number(candidate),
            candidate.candidate_id,
        ),
    )
    return tuple(ranked)


def no_action_candidate(*, candidate_id: str = "candidate:no_action") -> InformationGainCandidate:
    return InformationGainCandidate(
        candidate_id=candidate_id,
        kind="no_action",
        target_axis="persona_about_user",
        plain_action="No extra question is useful here.",
        expected_gain_band="low",
        risk_band="low",
        consent_requirement="none",
        source="safety_fallback",
    )


def rank_or_no_action(candidates: Sequence[InformationGainCandidate]) -> tuple[InformationGainCandidate, ...]:
    ranked = rank_information_gain_candidates(candidates)
    if not ranked:
        return (no_action_candidate(),)
    if all(candidate.expected_gain_band == "low" for candidate in ranked):
        return (no_action_candidate(),)
    return ranked


def lower_candidate_priority(candidate: InformationGainCandidate) -> InformationGainCandidate | None:
    if candidate.expected_gain_band == "high":
        return replace(candidate, expected_gain_band="medium")
    if candidate.expected_gain_band == "medium":
        return replace(candidate, expected_gain_band="low")
    return None


def _latest_turn_number(candidate: InformationGainCandidate) -> int:
    latest = -1
    for ref in candidate.evidence_refs:
        turn_id = ref.turn_id.casefold().lstrip("t")
        try:
            latest = max(latest, int(turn_id))
        except ValueError:
            continue
    return latest
