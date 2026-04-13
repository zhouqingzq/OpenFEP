from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ObservabilityTier(IntEnum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True, slots=True)
class DialogueChannelSpec:
    name: str
    tier: ObservabilityTier
    precision_floor: float
    precision_ceiling: float
    default_precision: float
    description: str


DIALOGUE_CHANNELS: tuple[DialogueChannelSpec, ...] = (
    DialogueChannelSpec(
        name="semantic_content",
        tier=ObservabilityTier.HIGH,
        precision_floor=0.60,
        precision_ceiling=0.90,
        default_precision=0.75,
        description="What the interlocutor actually said.",
    ),
    DialogueChannelSpec(
        name="topic_novelty",
        tier=ObservabilityTier.HIGH,
        precision_floor=0.60,
        precision_ceiling=0.90,
        default_precision=0.75,
        description="Embedding-distance proxy for how novel the topic is.",
    ),
    DialogueChannelSpec(
        name="emotional_tone",
        tier=ObservabilityTier.MEDIUM,
        precision_floor=0.25,
        precision_ceiling=0.50,
        default_precision=0.35,
        description="Estimated valence/arousal from sentiment-like cues.",
    ),
    DialogueChannelSpec(
        name="conflict_tension",
        tier=ObservabilityTier.MEDIUM,
        precision_floor=0.25,
        precision_ceiling=0.50,
        default_precision=0.35,
        description="Estimated disagreement/confrontation pressure.",
    ),
    DialogueChannelSpec(
        name="relationship_depth",
        tier=ObservabilityTier.LOW,
        precision_floor=0.05,
        precision_ceiling=0.20,
        default_precision=0.10,
        description="Slow latent estimate of trust/intimacy with partner.",
    ),
    DialogueChannelSpec(
        name="hidden_intent",
        tier=ObservabilityTier.LOW,
        precision_floor=0.05,
        precision_ceiling=0.20,
        default_precision=0.10,
        description="Latent estimate of interlocutor unstated motives.",
    ),
)

DIALOGUE_CHANNEL_NAMES: tuple[str, ...] = tuple(ch.name for ch in DIALOGUE_CHANNELS)
_BY_NAME: dict[str, DialogueChannelSpec] = {spec.name: spec for spec in DIALOGUE_CHANNELS}


def get_channel_spec(name: str) -> DialogueChannelSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown dialogue channel: {name}") from exc


def get_tier_bounds(tier: ObservabilityTier) -> tuple[float, float]:
    specs = [spec for spec in DIALOGUE_CHANNELS if spec.tier == tier]
    if not specs:
        raise ValueError(f"no dialogue channels for tier={tier}")
    return (min(spec.precision_floor for spec in specs), max(spec.precision_ceiling for spec in specs))


def is_precision_anomalous(channel: str, precision: float) -> bool:
    spec = get_channel_spec(channel)
    return precision < spec.precision_floor or precision > spec.precision_ceiling
