from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .narrative_types import SemanticEvidence, SemanticGrounding


_NEGATION_MARKERS: tuple[str, ...] = (
    "not",
    "never",
    "fictional",
    "poster",
    "quote",
    "quoted",
    "pretend",
    "pretended",
    "imagined",
    "just a word",
    "training slogan",
    "wasn't real",
    "didn't happen",
    "no one actually",
    "bu shi",
    "mei you",
    "不是",
)

_SEMANTIC_CUES: dict[str, tuple[str, ...]] = {
    "resource_gain": (
        "food",
        "resource",
        "berries",
        "supplies",
        "found provisions",
        "shared meal",
        "safe resources",
        "meal",
        "stockpile",
        "shelter cache",
        "foraged",
        "找到",
        "食物",
        "吃的",
    ),
    "predator_attack": (
        "predator",
        "attack",
        "ambush",
        "pounced",
        "near miss",
        "unsafe crossing",
        "wounded",
        "injured",
        "trap snapped",
        "claws",
        "fangs",
        "追赶",
        "攻击",
        "受伤",
    ),
    "witnessed_death": (
        "poison",
        "toxic",
        "contamin",
        "fatal",
        "death",
        "corpse",
        "body collapsed",
        "died",
        "dead",
        "尸体",
        "死亡",
        "中毒",
        "毒蘑菇",
        "死去",
        "看到一个人",
    ),
    "social_exclusion": (
        "betray",
        "excluded",
        "rejected",
        "abandoned",
        "ignored",
        "mocked",
        "shunned",
        "humiliat",
        "deceived",
        "lied to",
        "trust broken",
        "left me outside",
        "turned away",
        "孤立",
        "背叛",
        "欺骗",
    ),
    "rescue": (
        "rescue",
        "rescued",
        "saved",
        "protect",
        "helped",
        "help",
        "supported",
        "comforted",
        "encouraged",
        "welcomed",
        "accepted",
        "invited",
        "listened",
        "cared",
        "stayed with me",
        "shared",
        "cooperate",
        "friend",
        "ally",
        "stayed nearby",
        "safe contact",
        "repair",
        "帮助",
        "保护",
        "救了",
        "接纳",
    ),
    "exploration": (
        "explore",
        "explored",
        "mapped",
        "map",
        "search",
        "searched",
        "question",
        "experiment",
        "adapted",
        "adapt",
        "curious",
        "trail",
        "unfamiliar",
        "new signals",
        "probe",
        "learned the route",
        "pattern",
        "figure out",
        "探索",
        "地图",
        "适应",
    ),
    "uncertainty": (
        "unclear",
        "uncertain",
        "didn't know",
        "unpredictable",
        "ambiguous",
        "mixed signals",
        "contradictory",
        "fuzzy",
        "maybe",
        "perhaps",
        "不确定",
        "模糊",
    ),
}

_PARAPHRASE_CUES: dict[str, tuple[str, ...]] = {
    "predator_attack": ("trap", "snare", "menace", "closing in", "stalked"),
    "rescue": ("stood by me", "kept me safe", "made room for me", "reconnected"),
    "social_exclusion": ("frozen out", "shut out", "turned cold", "left behind"),
    "resource_gain": ("provisions", "stocked up", "supplies held", "cache"),
    "exploration": ("charted", "surveyed", "tested", "followed a trail"),
}

_IMPLICIT_CUES: dict[str, tuple[str, ...]] = {
    "predator_attack": ("danger rose", "something hunted", "trap snapped again"),
    "rescue": ("felt safe enough", "someone stayed", "I could reconnect"),
    "social_exclusion": ("trust collapsed", "no one answered", "I was outside"),
    "resource_gain": ("energy returned", "we could eat", "shelter held"),
    "exploration": ("new route", "uncertain terrain", "mapped the area"),
}

_DIRECTION_BY_EVENT: dict[str, str] = {
    "predator_attack": "threat",
    "witnessed_death": "threat",
    "social_exclusion": "threat",
    "rescue": "social",
    "resource_gain": "resource",
    "exploration": "exploration",
    "uncertainty": "uncertainty",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _match_count(text: str, cues: tuple[str, ...]) -> int:
    return sum(1 for cue in cues if cue and cue in text)


@dataclass(slots=True)
class GroundingResult:
    grounding: SemanticGrounding
    semantic_direction_scores: dict[str, float]
    conflict_cues: list[str] = field(default_factory=list)
    event_structure_signals: list[str] = field(default_factory=list)
    surface_adversarial_risk: float = 0.0
    low_signal: bool = False
    uncertainty_cues: float = 0.0
    matched_event_type: str = "unknown_event"


class SemanticGrounder:
    """Derive deterministic semantic grounding from open narrative text."""

    def ground_episode(
        self,
        *,
        episode_id: str,
        text: str,
        metadata: Mapping[str, object] | None = None,
    ) -> GroundingResult:
        metadata = dict(metadata or {})
        lowered = text.casefold()
        lexical_hits = {name: _match_count(lowered, cues) for name, cues in _SEMANTIC_CUES.items()}
        paraphrase_hits = {name: _match_count(lowered, cues) for name, cues in _PARAPHRASE_CUES.items()}
        implicit_hits = {name: _match_count(lowered, cues) for name, cues in _IMPLICIT_CUES.items()}
        negation_hits = _match_count(lowered, _NEGATION_MARKERS)

        direction_scores = {
            "threat": 0.0,
            "social": 0.0,
            "resource": 0.0,
            "exploration": 0.0,
            "uncertainty": 0.0,
        }
        evidence: list[SemanticEvidence] = []
        motifs: list[str] = []
        supporting_segments: list[str] = []
        event_support: dict[str, float] = {}

        for event_type in sorted(_SEMANTIC_CUES):
            lexical = float(lexical_hits.get(event_type, 0))
            paraphrase = float(paraphrase_hits.get(event_type, 0))
            implicit = float(implicit_hits.get(event_type, 0))
            if lexical <= 0.0 and paraphrase <= 0.0 and implicit <= 0.0:
                continue
            strength = lexical + paraphrase * 0.85 + implicit * 0.70
            direction = _DIRECTION_BY_EVENT.get(event_type, event_type)
            direction_scores[direction] = direction_scores.get(direction, 0.0) + strength
            event_support[event_type] = strength
            motifs.append(event_type)

            if lexical > 0:
                evidence.append(
                    SemanticEvidence(
                        evidence_id=f"{episode_id}:{event_type}:surface",
                        source_type="surface",
                        label=event_type,
                        strength=_clamp(lexical / 3.0),
                        matched_text=[
                            cue for cue in _SEMANTIC_CUES[event_type] if cue and cue in lowered
                        ][:4],
                        direction=direction,
                        metadata={"surface_hits": int(lexical)},
                    )
                )
            if paraphrase > 0:
                evidence.append(
                    SemanticEvidence(
                        evidence_id=f"{episode_id}:{event_type}:paraphrase",
                        source_type="paraphrase",
                        label=event_type,
                        strength=_clamp(paraphrase / 3.0),
                        matched_text=[
                            cue for cue in _PARAPHRASE_CUES[event_type] if cue and cue in lowered
                        ][:4],
                        direction=direction,
                        metadata={"paraphrase_hits": int(paraphrase)},
                    )
                )
            if implicit > 0:
                evidence.append(
                    SemanticEvidence(
                        evidence_id=f"{episode_id}:{event_type}:implicit",
                        source_type="implicit",
                        label=event_type,
                        strength=_clamp(implicit / 3.0),
                        matched_text=[
                            cue for cue in _IMPLICIT_CUES[event_type] if cue and cue in lowered
                        ][:4],
                        direction=direction,
                        metadata={"implicit_hits": int(implicit)},
                    )
                )

        conflict_cues: list[str] = []
        if any(token in lowered for token in ("but", "however", "yet", "although", "但是", "却", "但")):
            conflict_cues.append("contrastive_connector")
        if any(token in lowered for token in ("at first", "later", "once", "now", "起初", "后来")):
            conflict_cues.append("temporal_shift")

        uncertainty_cues = float(lexical_hits.get("uncertainty", 0)) + float(
            "but" in lowered or "however" in lowered or "但是" in lowered
        )
        surface_adversarial_risk = min(
            1.0,
            negation_hits * 0.22 + max(0.0, sum(direction_scores.values()) - 5.0) * 0.05,
        )
        direction_scores["threat"] = max(0.0, direction_scores["threat"] - negation_hits * 0.70)
        direction_scores["social"] = max(0.0, direction_scores["social"] - negation_hits * 0.45)
        direction_scores["resource"] = max(0.0, direction_scores["resource"] - negation_hits * 0.25)
        direction_scores["exploration"] = max(
            0.0, direction_scores["exploration"] - negation_hits * 0.40
        )
        direction_scores["uncertainty"] = max(direction_scores["uncertainty"], uncertainty_cues)
        low_signal = sum(direction_scores.values()) <= 1.15 and uncertainty_cues >= 0.0

        matched_event_type = "unknown_event"
        if event_support:
            matched_event_type = sorted(event_support.items(), key=lambda item: (-item[1], item[0]))[0][0]

        event_structure_signals: list[str] = []
        if direction_scores["threat"] > 0.0:
            event_structure_signals.append("threat_signal")
        if direction_scores["social"] > 0.0:
            event_structure_signals.append("social_signal")
        if direction_scores["resource"] > 0.0:
            event_structure_signals.append("resource_signal")
        if direction_scores["exploration"] > 0.0:
            event_structure_signals.append("exploration_signal")
        if low_signal:
            event_structure_signals.append("weak_semantic_support")

        for item in evidence:
            supporting_segments.extend(item.matched_text)
        grounding = SemanticGrounding(
            episode_id=episode_id,
            motifs=sorted(set(motifs)),
            evidence=evidence,
            semantic_direction_scores={key: round(value, 6) for key, value in direction_scores.items()},
            lexical_surface_hits={key: int(value) for key, value in lexical_hits.items()},
            paraphrase_hits={key: int(value) for key, value in paraphrase_hits.items()},
            implicit_hits={key: int(value) for key, value in implicit_hits.items()},
            supporting_segments=supporting_segments[:12],
            provenance={
                "metadata": metadata,
                "matched_event_type": matched_event_type,
                "negation_hits": int(negation_hits),
                "surface_adversarial_risk": round(surface_adversarial_risk, 6),
                "conflict_cues": list(conflict_cues),
            },
            low_signal=low_signal,
        )
        return GroundingResult(
            grounding=grounding,
            semantic_direction_scores=dict(grounding.semantic_direction_scores),
            conflict_cues=conflict_cues,
            event_structure_signals=event_structure_signals,
            surface_adversarial_risk=surface_adversarial_risk,
            low_signal=low_signal,
            uncertainty_cues=uncertainty_cues,
            matched_event_type=matched_event_type,
        )
