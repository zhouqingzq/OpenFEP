from __future__ import annotations

from typing import Mapping

from .utils import clamp as _clamp

# Context field written by DialogueWorld.current_turn (and M5.3+ generators): plain text, no six-channel encoding.
PRIOR_SELF_BODY_KEY = "prior_self_body"
PRIOR_SELF_QUESTION_RELEVANCE_SELF_DELTA = 0.04


def _prior_self_looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith("?") or stripped.endswith("？")


DIALOGUE_STATE_VECTOR_ORDER: tuple[str, ...] = (
    "semantic_content",
    "topic_novelty",
    "emotional_tone",
    "conflict_tension",
    "relationship_depth",
    "hidden_intent",
)


def dialogue_observation_to_memory_fields(
    observation: Mapping[str, float],
    dialogue_context: Mapping[str, object] | None = None,
) -> dict[str, float]:
    emotional = _clamp(float(observation.get("emotional_tone", 0.5)))
    conflict = _clamp(float(observation.get("conflict_tension", 0.0)))
    novelty = _clamp(float(observation.get("topic_novelty", 0.5)))
    relationship = _clamp(float(observation.get("relationship_depth", 0.5)))
    intent = _clamp(float(observation.get("hidden_intent", 0.5)))
    semantic = _clamp(float(observation.get("semantic_content", 0.5)))
    raw_valence = (emotional * 2.0) - 1.0
    valence = max(-1.0, min(1.0, raw_valence * (1.0 - 0.6 * conflict)))
    relevance_self = abs(emotional - 0.5) * 2.0
    if dialogue_context:
        prior = dialogue_context.get(PRIOR_SELF_BODY_KEY)
        if isinstance(prior, str) and _prior_self_looks_like_question(prior):
            relevance_self = _clamp(relevance_self + PRIOR_SELF_QUESTION_RELEVANCE_SELF_DELTA)
    return {
        "valence": round(valence, 6),
        "arousal": round(max(conflict, novelty, abs(emotional - 0.5) * 2.0), 6),
        "relevance_threat": round(conflict * (0.5 + intent * 0.5), 6),
        "relevance_social": round(relationship * emotional, 6),
        "relevance_reward": round(semantic * (1.0 - conflict) * emotional, 6),
        "relevance_self": round(relevance_self, 6),
        "novelty": round(novelty, 6),
    }


def encode_dialogue_state_vector(observation: Mapping[str, float]) -> list[float]:
    return [
        _clamp(float(observation.get(channel, 0.5 if channel != "conflict_tension" else 0.0)))
        for channel in DIALOGUE_STATE_VECTOR_ORDER
    ]


def dialogue_state_vector_metadata() -> dict[str, object]:
    return {
        "vector_type": "dialogue_observation",
        "channel_order": list(DIALOGUE_STATE_VECTOR_ORDER),
        "index_map": {channel: index for index, channel in enumerate(DIALOGUE_STATE_VECTOR_ORDER)},
    }
