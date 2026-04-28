from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Mapping

from .channel_registry import DIALOGUE_CHANNELS, DIALOGUE_CHANNEL_NAMES, DialogueChannelSpec
from .types import TranscriptUtterance
from .observation import DialogueObservation
from .signal_extractors import (
    ConflictTensionExtractor,
    EmotionalToneExtractor,
    HiddenIntentExtractor,
    LLMChannelExtractor,
    RelationshipDepthExtractor,
    SemanticContentExtractor,
    SignalExtractor,
    TopicNoveltyExtractor,
)


def normalize_conversation_history(
    conversation_history: Sequence[str | TranscriptUtterance | Mapping[str, object]],
) -> list[str]:
    """M5.3: accept legacy list[str] or role-tagged utterances; extractors still see flat text."""
    normalized: list[str] = []
    for item in conversation_history:
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            text = item.get("text", "")
        else:
            text = ""
        text = str(text).strip()
        if text:
            normalized.append(text)
    return normalized


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
        llm_extractor: LLMChannelExtractor | None = None,
    ) -> None:
        self.extractors = extractors or _default_extractors()
        self.channel_registry = channel_registry
        self.llm_extractor = llm_extractor
        #: Source of the most recent observation channels ("rule" or "llm").
        self.last_channel_source: str = "rule"
        missing = sorted(set(DIALOGUE_CHANNEL_NAMES) - set(self.extractors))
        if missing:
            raise ValueError(f"missing extractors for channels: {missing}")

    def observe(
        self,
        current_turn: str,
        conversation_history: Sequence[str | TranscriptUtterance | Mapping[str, object]],
        partner_uid: int,
        session_context: dict[str, object],
        session_id: str,
        turn_index: int,
        speaker_uid: int,
        timestamp: datetime | None = None,
    ) -> DialogueObservation:
        flat_history = normalize_conversation_history(conversation_history)
        channels: dict[str, float] = {}

        # Try structured LLM extraction first; fall back to rule-based on failure.
        if self.llm_extractor is not None:
            llm_channels = self.llm_extractor.extract_all(
                current_turn=current_turn,
                conversation_history=flat_history,
            )
            if llm_channels and all(ch in llm_channels for ch in DIALOGUE_CHANNEL_NAMES):
                channels = dict(llm_channels)
                self.last_channel_source = "llm"
        if not channels:
            self.last_channel_source = "rule"
            for channel in DIALOGUE_CHANNEL_NAMES:
                channels[channel] = float(
                    self.extractors[channel].extract(
                        current_turn=current_turn,
                        conversation_history=flat_history,
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
