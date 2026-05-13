"""M12.1 evidence-card conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from segmentum.memory_evidence import MemoryEvidence

from .hyperparams import CARD_CONFIDENCE_PRIORITY, DEFAULT_HYPERPARAMS, M121Hyperparams, PERMITTED_PRIORITY, SECTION_KINDS
from .personality_profile import InsufficientEvidence, PersonalityProfile

CONFIDENCE_FLOATS = {
    "low": DEFAULT_HYPERPARAMS.card_confidence_float_low,
    "med": DEFAULT_HYPERPARAMS.card_confidence_float_med,
    "high": DEFAULT_HYPERPARAMS.card_confidence_float_high,
}


@dataclass(frozen=True)
class PersonalityEvidenceCard:
    memory_id: str
    memory_class: str
    content_summary: str
    confidence_band: str
    permitted_use: str
    section_kind: str
    why_retrieved: str

    def prompt_safe_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "memory_class": self.memory_class,
            "content_summary": self.content_summary,
            "confidence_band": self.confidence_band,
            "permitted_use": self.permitted_use,
            "section_kind": self.section_kind,
            "why_retrieved": self.why_retrieved,
        }

    def to_memory_evidence(self) -> MemoryEvidence:
        return MemoryEvidence(
            memory_id=self.memory_id,
            memory_class="inferred_hypothesis",
            source_turn_id="",
            source_utterance_id="",
            speaker="user_personality",
            status="hypothesis",
            confidence=CONFIDENCE_FLOATS.get(self.confidence_band, CONFIDENCE_FLOATS["low"]),
            retrieval_score=0.0,
            cue_match=self.why_retrieved,
            salience=0.0,
            value_score=0.0,
            decay_state="active",
            conflict_status="none",
            permitted_use=self.permitted_use,  # type: ignore[arg-type]
            content_summary=self.content_summary,
        )

    def to_dict(self) -> dict[str, object]:
        return self.prompt_safe_dict()


def evidence_cards_from_personality_profile(
    profile: PersonalityProfile,
    *,
    report_status: str | None = None,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[PersonalityEvidenceCard, ...]:
    cards: list[PersonalityEvidenceCard] = []
    restrict_to_strategy = _section_insufficient(profile, "step_4") or _section_insufficient(profile, "step_7")
    for section_kind in SECTION_KINDS:
        section = profile.section_for(section_kind)
        if section is None:
            continue
        if isinstance(section, InsufficientEvidence):
            content = "insufficient_evidence"
            permitted = "strategy_only"
            confidence = "low"
        else:
            content = _section_summary(section_kind, section.to_dict())
            permitted = "strategy_only" if restrict_to_strategy else "cautious_hypothesis"
            confidence = str(section.to_dict().get("confidence_band", "low"))
        if not content:
            continue
        card = PersonalityEvidenceCard(
            memory_id=f"m12_1:{profile.user_id}:{section_kind}",
            memory_class="inferred_hypothesis",
            content_summary=content[: hyperparams.max_summary_chars],
            confidence_band=confidence,
            permitted_use=permitted,
            section_kind=section_kind,
            why_retrieved="active_personality_profile",
        )
        cards.append(card)
    cards.sort(
        key=lambda card: (
            PERMITTED_PRIORITY.get(card.permitted_use, 9),
            CARD_CONFIDENCE_PRIORITY.get(card.confidence_band, 9),
            card.section_kind,
        )
    )
    return tuple(cards[: hyperparams.max_card_count])


def prompt_safe_cards(cards: Sequence[PersonalityEvidenceCard]) -> tuple[dict[str, str], ...]:
    return tuple(card.prompt_safe_dict() for card in cards if card.permitted_use != "forbidden")


def _section_insufficient(profile: PersonalityProfile, section_kind: str) -> bool:
    return isinstance(profile.section_for(section_kind), InsufficientEvidence)


def _section_summary(section_kind: str, payload: dict[str, object]) -> str:
    if section_kind == "step_1":
        return str(payload.get("summary", ""))
    if section_kind == "step_2":
        rows = payload.get("evidence_items", [])
        if isinstance(rows, list) and rows:
            first = rows[0]
            return str(first.get("content_summary", "")) if isinstance(first, dict) else ""
    if section_kind == "step_3":
        return str(payload.get("default_interpretation", ""))
    if section_kind == "step_4":
        row = payload.get("about_self")
        return str(row.get("content_summary", "")) if isinstance(row, dict) else ""
    if section_kind == "step_5":
        return str(payload.get("threat_response", ""))
    if section_kind == "step_6":
        return str(payload.get("recurring_loop_summary", ""))
    if section_kind == "step_7":
        stages = payload.get("stages", [])
        if isinstance(stages, list) and stages:
            first = stages[0]
            return str(first.get("content_summary", "")) if isinstance(first, dict) else ""
    if section_kind == "step_8":
        values = payload.get("soft_spots", [])
        if isinstance(values, list) and values:
            return str(values[0])
    return ""
