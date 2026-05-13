"""Volatile per-turn reciprocal-role assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .reciprocal_model import EvidenceRef, InformationGainCandidate, TargetAxis, UncertaintyPoint

ObservedUserProbe = Literal["none", "mild", "explicit", "adversarial", "boundary_test"]
HintKind = Literal["ask_clear_question", "clarify_persona_stance", "acknowledge_uncertainty", "maintain_boundary", "no_action"]


@dataclass(frozen=True)
class ReplyPolicyHint:
    hint_id: str
    kind: HintKind
    plain_reason: str
    target_axis: TargetAxis
    priority: str = "low"
    evidence_refs: tuple[EvidenceRef, ...] = ()
    claim_id: str = ""
    topic_label: str = ""
    source: str = "volatile"

    def to_dict(self) -> dict[str, object]:
        return {
            "hint_id": self.hint_id,
            "kind": self.kind,
            "plain_reason": self.plain_reason,
            "target_axis": self.target_axis,
            "priority": self.priority,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "claim_id": self.claim_id,
            "topic_label": self.topic_label,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReplyPolicyHint":
        return cls(
            hint_id=str(payload.get("hint_id", "")),
            kind=_hint_kind(payload.get("kind")),
            plain_reason=str(payload.get("plain_reason", "")),
            target_axis=_axis(payload.get("target_axis")),
            priority=str(payload.get("priority", "low")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            claim_id=str(payload.get("claim_id", "")),
            topic_label=str(payload.get("topic_label", "")),
            source=str(payload.get("source", "volatile")),
        )


@dataclass(frozen=True)
class ReciprocalTurnAssessment:
    turn_id: str
    observed_user_probe: ObservedUserProbe = "none"
    persona_about_user_uncertainty_band: str = "high"
    user_about_persona_uncertainty_band: str = "high"
    top_uncertainty_points: tuple[UncertaintyPoint, ...] = ()
    top_gain_candidates: tuple[InformationGainCandidate, ...] = ()
    reply_policy_hints: tuple[ReplyPolicyHint, ...] = ()
    safety_findings: tuple[Mapping[str, object], ...] = ()
    insufficient_evidence: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "observed_user_probe": self.observed_user_probe,
            "persona_about_user_uncertainty_band": self.persona_about_user_uncertainty_band,
            "user_about_persona_uncertainty_band": self.user_about_persona_uncertainty_band,
            "top_uncertainty_points": [point.to_dict() for point in self.top_uncertainty_points],
            "top_gain_candidates": [candidate.to_dict() for candidate in self.top_gain_candidates],
            "reply_policy_hints": [hint.to_dict() for hint in self.reply_policy_hints],
            "safety_findings": [dict(item) for item in self.safety_findings],
            "insufficient_evidence": self.insufficient_evidence,
        }


def assess_turn_light(
    *,
    turn_id: str,
    user_text: str,
    current_turn_quotes: Mapping[str, str] | None = None,
) -> ReciprocalTurnAssessment:
    """Small volatile estimator.

    The estimator uses dialogue-act patterns plus quote evidence. It never
    writes durable state and it returns insufficient evidence on cold starts.
    """
    text = str(user_text or "").strip()
    quote_refs = _quote_refs(turn_id, current_turn_quotes or ({"q_current": text} if text else {}))
    if not text:
        return ReciprocalTurnAssessment(turn_id=turn_id)
    probe = _probe_level(text)
    if probe == "none" and len(text) < 24:
        return ReciprocalTurnAssessment(turn_id=turn_id, insufficient_evidence=True)
    points: list[UncertaintyPoint] = []
    candidates: list[InformationGainCandidate] = []
    hints: list[ReplyPolicyHint] = []
    if probe in {"explicit", "adversarial", "boundary_test"}:
        points.append(
            UncertaintyPoint(
                point_id=f"uncertainty:{turn_id}:persona_legibility",
                target_axis="user_about_persona",
                plain_question="The user may be checking what the persona can honestly say about itself.",
                why_it_matters_internal="direct persona probe observed",
                expected_gain_band="high",
                risk_band="low",
                evidence_refs=quote_refs,
            )
        )
        candidates.append(
            InformationGainCandidate(
                candidate_id=f"candidate:{turn_id}:clarify_self",
                kind="clarify_self",
                target_axis="user_about_persona",
                plain_action="Give a direct, bounded explanation of what is known and what is still uncertain.",
                expected_gain_band="high",
                risk_band="low",
                evidence_refs=quote_refs,
                source="explicit_user_request" if probe == "explicit" else "implicit_probe",
                topic_label="persona_legibility",
            )
        )
        hints.append(
            ReplyPolicyHint(
                hint_id=f"hint:{turn_id}:clarify_self",
                kind="clarify_persona_stance",
                plain_reason="A direct explanation of the boundary may make the answer easier to trust.",
                target_axis="user_about_persona",
                priority="high",
                evidence_refs=quote_refs,
                topic_label="persona_legibility",
            )
        )
    else:
        points.append(
            UncertaintyPoint(
                point_id=f"uncertainty:{turn_id}:current_goal",
                target_axis="persona_about_user",
                plain_question="The exact kind of help the user wants may still be unclear.",
                why_it_matters_internal="turn has task ambiguity but no durable evidence yet",
                expected_gain_band="medium",
                risk_band="low",
                evidence_refs=quote_refs,
            )
        )
        candidates.append(
            InformationGainCandidate(
                candidate_id=f"candidate:{turn_id}:ask_goal",
                kind="ask_question",
                target_axis="persona_about_user",
                plain_action="Ask one short question about the kind of answer the user wants.",
                expected_gain_band="medium",
                risk_band="low",
                evidence_refs=quote_refs,
                source="implicit_probe",
                topic_label="current_goal",
            )
        )
    return ReciprocalTurnAssessment(
        turn_id=turn_id,
        observed_user_probe=probe,
        persona_about_user_uncertainty_band="medium",
        user_about_persona_uncertainty_band="high" if probe != "none" else "medium",
        top_uncertainty_points=tuple(points[:2]),
        top_gain_candidates=tuple(candidates[:2]),
        reply_policy_hints=tuple(hints),
        insufficient_evidence=False,
    )


def _probe_level(text: str) -> ObservedUserProbe:
    folded = text.casefold()
    question_like = "?" in text or "？" in text or any(term in folded for term in ("how", "what", "why", "whether", "你", "吗"))
    persona_refs = any(term in folded for term in ("you", "persona", "model me", "trust", "memory", "consistent", "honest", "你", "人格", "记得", "一致", "信任"))
    testing_refs = any(term in folded for term in ("test", "prove", "trap", "lying", "are you sure", "测试", "证明", "骗", "确定"))
    boundary_refs = any(term in folded for term in ("secret", "private", "boundary", "consent", "隐私", "边界", "秘密"))
    if boundary_refs and persona_refs:
        return "boundary_test"
    if testing_refs and persona_refs:
        return "adversarial"
    if question_like and persona_refs:
        return "explicit"
    if persona_refs:
        return "mild"
    return "none"


def _quote_refs(turn_id: str, quotes: Mapping[str, str]) -> tuple[EvidenceRef, ...]:
    return tuple(EvidenceRef(ref_id=f"{turn_id}:{quote_id}") for quote_id, quote in sorted(quotes.items()) if str(quote).strip())


def _refs(value: object) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(EvidenceRef.from_any(item) for item in value if str(item or "").strip() or isinstance(item, Mapping))


def _hint_kind(value: object) -> HintKind:
    text = str(value or "no_action")
    return text if text in {"ask_clear_question", "clarify_persona_stance", "acknowledge_uncertainty", "maintain_boundary", "no_action"} else "no_action"  # type: ignore[return-value]


def _axis(value: object) -> TargetAxis:
    text = str(value or "persona_about_user")
    return text if text in {"persona_about_user", "user_about_persona"} else "persona_about_user"  # type: ignore[return-value]
