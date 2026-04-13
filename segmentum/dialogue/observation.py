from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from segmentum.io_bus import BusSignal

from .channel_registry import DIALOGUE_CHANNEL_NAMES, get_channel_spec


@dataclass(slots=True)
class DialogueObservation:
    channels: dict[str, float]
    raw_text: str
    speaker_uid: int
    turn_index: int
    session_id: str
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.channels) - set(DIALOGUE_CHANNEL_NAMES))
        if unknown:
            raise ValueError(f"unknown dialogue channels: {unknown}")
        missing = sorted(set(DIALOGUE_CHANNEL_NAMES) - set(self.channels))
        if missing:
            raise ValueError(f"missing dialogue channels: {missing}")
        for channel, value in self.channels.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"channel {channel} must be in [0,1], got {value}")

    def to_bus_signals(self, source_id: str = "dialogue_world") -> list[BusSignal]:
        signals: list[BusSignal] = []
        for channel in DIALOGUE_CHANNEL_NAMES:
            value = float(self.channels[channel])
            spec = get_channel_spec(channel)
            signals.append(
                BusSignal(
                    channel=channel,
                    raw_value=value,
                    normalized_value=value,
                    confidence=spec.default_precision,
                    source_id=source_id,
                    source_type="dialogue_world",
                    role="exteroceptive",
                    provenance={
                        "tier": int(spec.tier),
                        "session_id": self.session_id,
                        "turn_index": int(self.turn_index),
                        "speaker_uid": int(self.speaker_uid),
                    },
                )
            )
        return signals
