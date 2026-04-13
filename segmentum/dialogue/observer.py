from __future__ import annotations

from datetime import datetime

from .channel_registry import DIALOGUE_CHANNELS, DIALOGUE_CHANNEL_NAMES, DialogueChannelSpec
from .observation import DialogueObservation
from .signal_extractors import (
    ConflictTensionExtractor,
    EmotionalToneExtractor,
    HiddenIntentExtractor,
    RelationshipDepthExtractor,
    SemanticContentExtractor,
    SignalExtractor,
    TopicNoveltyExtractor,
)


def _default_extractors() -> dict[str, SignalExtractor]:
    return {
        "semantic_content": SemanticContentExtractor(),
        "topic_novelty": TopicNoveltyExtractor(),
        "emotional_tone": EmotionalToneExtractor(),
        "conflict_tension": ConflictTensionExtractor(),
        "relationship_depth": RelationshipDepthExtractor(),
        "hidden_intent": HiddenIntentExtractor(),
    }


class DialogueObserver:
    def __init__(
        self,
        extractors: dict[str, SignalExtractor] | None = None,
        channel_registry: tuple[DialogueChannelSpec, ...] = DIALOGUE_CHANNELS,
    ) -> None:
        self.extractors = extractors or _default_extractors()
        self.channel_registry = channel_registry
        missing = sorted(set(DIALOGUE_CHANNEL_NAMES) - set(self.extractors))
        if missing:
            raise ValueError(f"missing extractors for channels: {missing}")

    def observe(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
        session_id: str,
        turn_index: int,
        speaker_uid: int,
        timestamp: datetime | None = None,
    ) -> DialogueObservation:
        channels: dict[str, float] = {}
        for channel in DIALOGUE_CHANNEL_NAMES:
            channels[channel] = float(
                self.extractors[channel].extract(
                    current_turn=current_turn,
                    conversation_history=conversation_history,
                    partner_uid=partner_uid,
                    session_context=session_context,
                )
            )
        return DialogueObservation(
            channels=channels,
            raw_text=current_turn,
            speaker_uid=speaker_uid,
            turn_index=turn_index,
            session_id=session_id,
            timestamp=timestamp,
        )
