"""Prompt-safe M11 evidence-card conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from segmentum.memory_evidence import MemoryEvidence

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams
from .prediction_ledger import PredictionEntry
from .user_model import UserHypothesis, UserModel

CONFIDENCE_FLOATS = {"low": 0.33, "med": 0.66, "high": 0.9}
PERMITTED_PRIORITY = {"explicit_fact": 0, "cautious_hypothesis": 1, "strategy_only": 2, "forbidden": 3}
CONFIDENCE_PRIORITY = {"high": 0, "med": 1, "low": 2}


@dataclass(frozen=True)
class UserModelEvidenceCard:
    memory_id: str
    memory_class: str
    content_summary: str
    confidence_band: str
    source_reliability: float
    source_reliability_label: str
    permitted_use: str
    contradiction_status: str
    why_retrieved: str

    def to_memory_evidence(self) -> MemoryEvidence:
        return MemoryEvidence(
            memory_id=self.memory_id,
            memory_class=self.memory_class,
            source_turn_id="",
            source_utterance_id="",
            speaker="user_model",
            status="hypothesis" if self.memory_class == "inferred_hypothesis" else "asserted",
            confidence=CONFIDENCE_FLOATS.get(self.confidence_band, CONFIDENCE_FLOATS["low"]),
            retrieval_score=0.0,
            cue_match=self.why_retrieved,
            salience=0.0,
            value_score=0.0,
            decay_state="active",
            conflict_status="unresolved" if self.contradiction_status == "unresolved" else "none",
            permitted_use=self.permitted_use,
            content_summary=self.content_summary,
        )

    def prompt_safe_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "memory_class": self.memory_class,
            "content_summary": self.content_summary,
            "confidence_band": self.confidence_band,
            "source_reliability": self.source_reliability_label,
            "permitted_use": self.permitted_use,
            "contradiction_status": self.contradiction_status,
            "why_retrieved": self.why_retrieved,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.prompt_safe_dict(),
            "source_reliability_float": round(self.source_reliability, DEFAULT_HYPERPARAMS.float_round_digits),
        }


def evidence_cards_from_user_model(
    model: UserModel,
    *,
    recent_judgments: Sequence[PredictionEntry] = (),
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[UserModelEvidenceCard, ...]:
    cards: list[UserModelEvidenceCard] = []
    for hypothesis in model.all_hypotheses():
        reliability = float(model.source_reliability_by_domain.get(hypothesis.domain, hyperparams.prior_mean))
        cards.append(_card_from_hypothesis(hypothesis, reliability))
    for judgment in recent_judgments:
        cards.append(_card_from_judgment(judgment, hyperparams.prior_mean))
    cards.sort(
        key=lambda card: (
            PERMITTED_PRIORITY.get(card.permitted_use, 9),
            CONFIDENCE_PRIORITY.get(card.confidence_band, 9),
            -card.source_reliability,
            card.memory_id,
        )
    )
    return tuple(cards)


def prompt_safe_cards(cards: Sequence[UserModelEvidenceCard]) -> tuple[dict[str, str], ...]:
    return tuple(card.prompt_safe_dict() for card in cards if card.permitted_use != "forbidden")


def _card_from_hypothesis(hypothesis: UserHypothesis, reliability: float) -> UserModelEvidenceCard:
    contradiction = "unresolved" if hypothesis.contradiction_refs else "none"
    return UserModelEvidenceCard(
        memory_id=hypothesis.hypothesis_id,
        memory_class="inferred_hypothesis" if hypothesis.permitted_use != "explicit_fact" else "user_assertion",
        content_summary=hypothesis.content_summary[:120],
        confidence_band=hypothesis.confidence_band,
        source_reliability=reliability,
        source_reliability_label=_reliability_label(reliability),
        permitted_use=hypothesis.permitted_use,
        contradiction_status=contradiction,
        why_retrieved="active_user_model_hypothesis",
    )


def _card_from_judgment(judgment: PredictionEntry, reliability: float) -> UserModelEvidenceCard:
    return UserModelEvidenceCard(
        memory_id=f"judgment:{judgment.prediction_id}:{judgment.turn_id}",
        memory_class="semantic",
        content_summary=judgment.observed_outcome_summary[:120],
        confidence_band=judgment.calibration_need_band,
        source_reliability=reliability,
        source_reliability_label=_reliability_label(reliability),
        permitted_use="strategy_only",
        contradiction_status="none" if judgment.validation_status != "violated" else "unresolved",
        why_retrieved=f"recent_prediction_{judgment.validation_status}",
    )


def _reliability_label(value: float) -> str:
    if value >= 0.7:
        return "high"
    if value >= 0.45:
        return "med"
    return "low"
