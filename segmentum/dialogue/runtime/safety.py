from __future__ import annotations

from dataclasses import dataclass, field

from ..channel_registry import DIALOGUE_CHANNELS
from ..precision_bounds import ChannelPrecisionBounds


@dataclass
class SafetyCheck:
    passed: bool
    channel: str = ""
    reason: str = ""
    severity: str = "ok"  # "ok", "warning", "blocked"
    suggested_fix: str | None = None


def _build_default_blocked_topics() -> list[str]:
    return [
        "self_harm",
        "suicide",
        "violence_against_children",
    ]


class SafetyLayer:
    def __init__(self, blocked_topics: list[str] | None = None) -> None:
        self._blocked_topics: list[str] = (
            blocked_topics if blocked_topics is not None else _build_default_blocked_topics()
        )
        self._bounds: ChannelPrecisionBounds = ChannelPrecisionBounds.from_dialogue_channels()

    def check_precision_health(self, precisions: dict[str, float]) -> list[SafetyCheck]:
        report = self._bounds.anomaly_report(precisions)
        checks: list[SafetyCheck] = []
        for channel, label in report.items():
            spec_lookup = {ch.name: ch for ch in DIALOGUE_CHANNELS}
            spec = spec_lookup.get(channel)
            ceiling = spec.precision_ceiling if spec else 1.0
            value = precisions.get(channel, 0.0)
            severity = "warning"
            if channel == "hidden_intent" and value > ceiling + 0.10:
                severity = "blocked"
            checks.append(
                SafetyCheck(
                    passed=False,
                    channel=channel,
                    reason=f"{channel} precision {value:.3f} outside bounds [{spec.precision_floor if spec else 0:.2f}, {ceiling:.2f}]: {label}",
                    severity=severity,
                    suggested_fix=(
                        "Consider triggering sleep consolidation to restore precision balance."
                        if severity == "blocked"
                        else None
                    ),
                )
            )
        return checks

    def check_response(self, text: str) -> SafetyCheck:
        text_lower = text.lower()
        for topic in self._blocked_topics:
            if topic.lower() in text_lower:
                return SafetyCheck(
                    passed=False,
                    channel="content",
                    reason=f"Response matched blocked topic pattern: {topic}",
                    severity="blocked",
                    suggested_fix="[Response filtered by safety policy]",
                )
        return SafetyCheck(passed=True)

    def enforce(self, text: str, precisions: dict[str, float] | None = None) -> tuple[str, list[SafetyCheck]]:
        checks: list[SafetyCheck] = []
        if precisions:
            checks.extend(self.check_precision_health(precisions))
        content_check = self.check_response(text)
        checks.append(content_check)
        if not content_check.passed:
            return (content_check.suggested_fix or "[Response filtered]", checks)
        return (text, checks)
