"""M12.0 identity continuity evidence cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from segmentum.memory_evidence import MemoryEvidence

from .hyperparams import DEFAULT_HYPERPARAMS, M12Hyperparams
from .identity_conflict_detector import ConflictRecord
from .identity_profile import IdentityProfile

CONFIDENCE_FLOATS = {"low": 0.33, "med": 0.66, "high": 0.9}
PERMITTED_PRIORITY = {"explicit_fact": 0, "cautious_hypothesis": 1, "strategy_only": 2, "forbidden": 3}
STATE_PRIORITY = {"corroborated": 0, "asserted": 1, "conflicted": 2, "unverified": 3, "retracted": 4}
CONFIDENCE_PRIORITY = {"high": 0, "med": 1, "low": 2}


@dataclass(frozen=True)
class IdentityEvidenceCard:
    memory_id: str
    memory_class: str
    content_summary: str
    confidence_band: str
    identity_state: str
    permitted_use: str
    why_retrieved: str

    def prompt_safe_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "memory_class": self.memory_class,
            "content_summary": self.content_summary,
            "confidence_band": self.confidence_band,
            "identity_state": self.identity_state,
            "permitted_use": self.permitted_use,
            "why_retrieved": self.why_retrieved,
        }

    def to_memory_evidence(self) -> MemoryEvidence:
        return MemoryEvidence(
            memory_id=self.memory_id,
            memory_class=self.memory_class,
            source_turn_id="",
            source_utterance_id="",
            speaker="user_continuity",
            status=self.identity_state,
            confidence=CONFIDENCE_FLOATS.get(self.confidence_band, 0.33),
            retrieval_score=0.0,
            cue_match=self.why_retrieved,
            salience=0.0,
            value_score=0.0,
            decay_state="active",
            conflict_status="unresolved" if self.identity_state == "conflicted" else "none",
            permitted_use=self.permitted_use,  # type: ignore[arg-type]
            content_summary=self.content_summary,
        )


def cards_to_prompt_safe_memory_evidence(
    *,
    profile: IdentityProfile,
    open_conflicts: Sequence[ConflictRecord],
    hyperparams: M12Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[IdentityEvidenceCard, ...]:
    cards: list[IdentityEvidenceCard] = []
    if profile.aliases_observed:
        latest_alias = profile.aliases_observed[-1]
        cards.append(
            IdentityEvidenceCard(
                memory_id=f"profile:{profile.user_id}",
                memory_class="user_assertion",
                content_summary=f"User may use alias '{latest_alias.alias_text}'",
                confidence_band=profile.binding_confidence_band,
                identity_state=profile.identity_state,
                permitted_use="explicit_fact" if profile.identity_state == "corroborated" else "cautious_hypothesis",
                why_retrieved="active_identity_profile",
            )
        )
    for conflict in open_conflicts:
        cards.append(
            IdentityEvidenceCard(
                memory_id=conflict.conflict_id,
                memory_class="inferred_hypothesis",
                content_summary=f"Alias '{conflict.asserted_alias}' needs identity verification",
                confidence_band="high" if conflict.severity_band == "major" else "med",
                identity_state="conflicted",
                permitted_use="strategy_only",
                why_retrieved="open_identity_conflict",
            )
        )
    cards = _lint_user_facing_tokens(cards, hyperparams=hyperparams)
    cards.sort(
        key=lambda item: (
            PERMITTED_PRIORITY.get(item.permitted_use, 9),
            STATE_PRIORITY.get(item.identity_state, 9),
            CONFIDENCE_PRIORITY.get(item.confidence_band, 9),
            item.memory_id,
        )
    )
    return tuple(cards)


def _lint_user_facing_tokens(
    cards: Sequence[IdentityEvidenceCard],
    *,
    hyperparams: M12Hyperparams,
) -> list[IdentityEvidenceCard]:
    cleaned: list[IdentityEvidenceCard] = []
    forbidden = tuple(token.lower() for token in hyperparams.forbidden_user_facing_tokens)
    for card in cards:
        lower = card.content_summary.lower()
        if any(token in lower for token in forbidden):
            raise ValueError("identity evidence card contains forbidden user-facing token")
        cleaned.append(card)
    return cleaned
