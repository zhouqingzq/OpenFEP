"""M12.2 compact evidence cards and reply-hint reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .information_gain import lower_candidate_priority
from .plain_language_linter import lint_text
from .reciprocal_model import EvidenceRef, InformationGainCandidate, ReciprocalRoleModel
from .turn_assessment import ReplyPolicyHint


@dataclass(frozen=True)
class ReciprocalEvidenceCard:
    source: str
    kind: str
    content_summary: str
    confidence_band: str
    priority: str
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "kind": self.kind,
            "content_summary": self.content_summary,
            "confidence_band": self.confidence_band,
            "priority": self.priority,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
        }

    def prompt_safe_dict(self) -> dict[str, object]:
        return self.to_dict()


def evidence_cards_from_candidates(
    candidates: Sequence[InformationGainCandidate],
    *,
    model: ReciprocalRoleModel | None = None,
) -> tuple[ReciprocalEvidenceCard, ...]:
    cards: list[ReciprocalEvidenceCard] = []
    for candidate in candidates:
        if candidate.kind == "no_action" or candidate.blocked_by_safety:
            continue
        kind = {
            "ask_question": "safe_question",
            "clarify_self": "second_order_clarity",
            "demonstrate_consistency": "second_order_clarity",
            "defer": "boundary",
        }.get(candidate.kind, "reciprocal_uncertainty")
        text = candidate.plain_action
        if lint_text(text, section="m12_2_card"):
            continue
        cards.append(
            ReciprocalEvidenceCard(
                source="m12_2_reciprocal_role",
                kind=kind,
                content_summary=text,
                confidence_band=_confidence_for_candidate(candidate, model=model),
                priority=candidate.expected_gain_band,
                evidence_refs=candidate.evidence_refs,
            )
        )
    return tuple(cards[:4])


def evidence_cards_from_model(model: ReciprocalRoleModel) -> tuple[ReciprocalEvidenceCard, ...]:
    cards: list[ReciprocalEvidenceCard] = []
    for point in model.unresolved_uncertainty_points[:4]:
        if point.status != "open" or lint_text(point.plain_question, section="m12_2_card"):
            continue
        cards.append(
            ReciprocalEvidenceCard(
                source="m12_2_reciprocal_role",
                kind="reciprocal_uncertainty",
                content_summary=point.plain_question,
                confidence_band="low",
                priority=point.expected_gain_band,
                evidence_refs=point.evidence_refs,
            )
        )
    return tuple(cards[:4])


def prompt_safe_cards(cards: Sequence[ReciprocalEvidenceCard]) -> tuple[dict[str, object], ...]:
    return tuple(card.prompt_safe_dict() for card in cards)


def hints_from_candidates(candidates: Sequence[InformationGainCandidate], *, source: str) -> tuple[ReplyPolicyHint, ...]:
    hints: list[ReplyPolicyHint] = []
    for candidate in candidates:
        if candidate.kind == "no_action":
            hints.append(
                ReplyPolicyHint(
                    hint_id=f"hint:{candidate.candidate_id}",
                    kind="no_action",
                    plain_reason="No extra question is useful here.",
                    target_axis=candidate.target_axis,
                    priority="low",
                    evidence_refs=candidate.evidence_refs,
                    source=source,
                )
            )
            continue
        kind = {
            "ask_question": "ask_clear_question",
            "clarify_self": "clarify_persona_stance",
            "demonstrate_consistency": "clarify_persona_stance",
            "defer": "maintain_boundary",
        }.get(candidate.kind, "acknowledge_uncertainty")
        hints.append(
            ReplyPolicyHint(
                hint_id=f"hint:{candidate.candidate_id}",
                kind=kind,  # type: ignore[arg-type]
                plain_reason=candidate.plain_action,
                target_axis=candidate.target_axis,
                priority=candidate.expected_gain_band,
                evidence_refs=candidate.evidence_refs,
                claim_id=candidate.claim_id,
                topic_label=candidate.topic_label,
                source=source,
            )
        )
    return tuple(hints)


def reconcile_hints(
    volatile_hints: Sequence[ReplyPolicyHint],
    durable_hints: Sequence[ReplyPolicyHint],
    *,
    durable_ran: bool,
) -> tuple[ReplyPolicyHint, ...]:
    if not durable_ran:
        return tuple(volatile_hints)
    durable_keys = {
        key
        for hint in durable_hints
        for key in ((hint.claim_id and f"claim:{hint.claim_id}"), (hint.topic_label and f"topic:{hint.topic_label}"))
        if key
    }
    merged = list(durable_hints)
    for hint in volatile_hints:
        if _is_relationship_value_hint(hint):
            merged.append(hint)
            continue
        keys = {key for key in ((hint.claim_id and f"claim:{hint.claim_id}"), (hint.topic_label and f"topic:{hint.topic_label}")) if key}
        if keys & durable_keys:
            continue
        lowered = _lower_hint_priority(hint)
        if lowered is not None:
            merged.append(lowered)
    return tuple(sorted(merged, key=lambda hint: ({"high": 0, "medium": 1, "low": 2}.get(hint.priority, 9), hint.hint_id)))


def reconcile_candidates(
    volatile_candidates: Sequence[InformationGainCandidate],
    durable_candidates: Sequence[InformationGainCandidate],
    *,
    durable_ran: bool,
) -> tuple[InformationGainCandidate, ...]:
    if not durable_ran:
        return tuple(volatile_candidates)
    durable_keys = {
        key
        for candidate in durable_candidates
        for key in ((candidate.claim_id and f"claim:{candidate.claim_id}"), (candidate.topic_label and f"topic:{candidate.topic_label}"))
        if key
    }
    merged = list(durable_candidates)
    for candidate in volatile_candidates:
        keys = {key for key in ((candidate.claim_id and f"claim:{candidate.claim_id}"), (candidate.topic_label and f"topic:{candidate.topic_label}")) if key}
        if keys & durable_keys:
            continue
        lowered = lower_candidate_priority(candidate)
        if lowered is not None:
            merged.append(lowered)
    return tuple(merged)


def _lower_hint_priority(hint: ReplyPolicyHint) -> ReplyPolicyHint | None:
    if _is_relationship_value_hint(hint):
        return hint
    if hint.priority == "high":
        return replace(hint, priority="medium")
    if hint.priority == "medium":
        return replace(hint, priority="low")
    return None


def _is_relationship_value_hint(hint: ReplyPolicyHint) -> bool:
    return hint.source == "relationship_value_memory" or hint.kind == "apply_relationship_value_constraint"


def _confidence_for_candidate(candidate: InformationGainCandidate, *, model: ReciprocalRoleModel | None) -> str:
    if model is not None and candidate.claim_id:
        for claim in model.all_claims():
            if claim.claim_id == candidate.claim_id:
                return claim.confidence_band
    return "low" if candidate.target_axis == "user_about_persona" else "medium"
