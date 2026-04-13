from __future__ import annotations

from dataclasses import dataclass

from .channel_registry import DIALOGUE_CHANNELS


@dataclass(slots=True)
class ChannelPrecisionBounds:
    bounds: dict[str, tuple[float, float]]

    @classmethod
    def from_dialogue_channels(cls) -> "ChannelPrecisionBounds":
        return cls(
            {
                ch.name: (float(ch.precision_floor), float(ch.precision_ceiling))
                for ch in DIALOGUE_CHANNELS
            }
        )

    def clamp(self, channel: str, precision: float) -> float:
        floor, ceiling = self.bounds[channel]
        return max(floor, min(ceiling, precision))

    def is_anomalous(self, channel: str, precision: float) -> bool:
        floor, ceiling = self.bounds[channel]
        return precision < floor or precision > ceiling

    def anomaly_report(self, precisions: dict[str, float]) -> dict[str, str]:
        report: dict[str, str] = {}
        for channel, value in precisions.items():
            if channel not in self.bounds:
                continue
            floor, ceiling = self.bounds[channel]
            if value < floor:
                if channel in {"hidden_intent", "relationship_depth"}:
                    report[channel] = "naive"
                elif channel in {"emotional_tone", "conflict_tension"}:
                    report[channel] = "numb"
                else:
                    report[channel] = "underconfident"
            elif value > ceiling:
                if channel == "hidden_intent":
                    report[channel] = "paranoid"
                elif channel in {"emotional_tone", "conflict_tension"}:
                    report[channel] = "anxious"
                else:
                    report[channel] = "overconfident"
        return report
