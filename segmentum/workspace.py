from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .attention import AttentionTrace


ACTION_HINTS = {
    "danger": {"hide": 0.22, "exploit_shelter": 0.14, "scan": -0.08, "forage": -0.10},
    "novelty": {"scan": 0.22, "forage": 0.06, "hide": -0.04},
    "food": {"forage": 0.20, "rest": -0.03},
    "shelter": {"exploit_shelter": 0.18, "hide": 0.08},
    "temperature": {"thermoregulate": 0.22, "rest": 0.04},
    "social": {"seek_contact": 0.22, "hide": -0.05},
}


@dataclass(frozen=True)
class WorkspaceContent:
    channel: str
    salience: float
    observation_value: float
    prediction_value: float
    error_value: float
    source: str = "attention"
    action_hints: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "salience": self.salience,
            "observation_value": self.observation_value,
            "prediction_value": self.prediction_value,
            "error_value": self.error_value,
            "source": self.source,
            "action_hints": {str(key): float(value) for key, value in self.action_hints.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WorkspaceContent":
        action_hints = payload.get("action_hints", {})
        return cls(
            channel=str(payload.get("channel", "")),
            salience=float(payload.get("salience", 0.0)),
            observation_value=float(payload.get("observation_value", 0.0)),
            prediction_value=float(payload.get("prediction_value", 0.0)),
            error_value=float(payload.get("error_value", 0.0)),
            source=str(payload.get("source", "attention")),
            action_hints=(
                {str(key): float(value) for key, value in action_hints.items()}
                if isinstance(action_hints, Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class GlobalWorkspaceState:
    tick: int
    capacity: int
    broadcast_contents: tuple[WorkspaceContent, ...]
    suppressed_channels: tuple[str, ...]
    broadcast_intensity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "capacity": self.capacity,
            "broadcast_contents": [content.to_dict() for content in self.broadcast_contents],
            "suppressed_channels": list(self.suppressed_channels),
            "broadcast_intensity": self.broadcast_intensity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "GlobalWorkspaceState | None":
        if not payload:
            return None
        raw_contents = payload.get("broadcast_contents", [])
        return cls(
            tick=int(payload.get("tick", 0)),
            capacity=max(1, int(payload.get("capacity", 1))),
            broadcast_contents=tuple(
                WorkspaceContent.from_dict(item)
                for item in raw_contents
                if isinstance(item, Mapping)
            ),
            suppressed_channels=tuple(
                str(item) for item in payload.get("suppressed_channels", [])
            ),
            broadcast_intensity=float(payload.get("broadcast_intensity", 0.0)),
        )


class GlobalWorkspace:
    def __init__(
        self,
        *,
        capacity: int = 2,
        enabled: bool = True,
        action_bias_gain: float = 0.35,
        memory_gate_gain: float = 0.08,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.enabled = bool(enabled)
        self.action_bias_gain = float(action_bias_gain)
        self.memory_gate_gain = float(memory_gate_gain)
        self.last_state: GlobalWorkspaceState | None = None

    def broadcast(
        self,
        *,
        tick: int,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
        attention_trace: AttentionTrace | None,
    ) -> GlobalWorkspaceState | None:
        if not self.enabled:
            self.last_state = None
            return None
        if attention_trace is None:
            ranked_channels = sorted(
                set(observation) | set(prediction) | set(errors),
                key=lambda channel: (-abs(float(errors.get(channel, 0.0))), channel),
            )
            selected = tuple(ranked_channels[: self.capacity])
            dropped = tuple(ranked_channels[self.capacity :])
            salience_scores = {channel: abs(float(errors.get(channel, 0.0))) for channel in ranked_channels}
        else:
            selected = attention_trace.allocation.selected_channels[: self.capacity]
            dropped = attention_trace.allocation.dropped_channels
            salience_scores = dict(attention_trace.salience_scores)

        contents = []
        for channel in selected:
            contents.append(
                WorkspaceContent(
                    channel=channel,
                    salience=float(salience_scores.get(channel, 0.0)),
                    observation_value=float(observation.get(channel, 0.0)),
                    prediction_value=float(prediction.get(channel, 0.0)),
                    error_value=float(errors.get(channel, 0.0)),
                    action_hints=dict(ACTION_HINTS.get(channel, {})),
                )
            )
        intensity = 0.0
        if contents:
            intensity = sum(content.salience for content in contents) / max(1, len(contents))
        state = GlobalWorkspaceState(
            tick=int(tick),
            capacity=self.capacity,
            broadcast_contents=tuple(contents),
            suppressed_channels=tuple(str(item) for item in dropped),
            broadcast_intensity=float(round(intensity, 6)),
        )
        self.last_state = state
        return state

    def action_bias(self, action: str, state: GlobalWorkspaceState | None = None) -> float:
        state = state or self.last_state
        if not self.enabled or state is None:
            return 0.0
        bias = 0.0
        for content in state.broadcast_contents:
            hint = float(content.action_hints.get(action, 0.0))
            if hint == 0.0:
                continue
            salience_scale = max(0.05, min(1.0, content.salience + abs(content.error_value)))
            bias += hint * salience_scale * self.action_bias_gain
        return round(bias, 6)

    def memory_threshold_delta(self, state: GlobalWorkspaceState | None = None) -> float:
        state = state or self.last_state
        if not self.enabled or state is None or not state.broadcast_contents:
            return 0.0
        peak_signal = max(
            abs(content.error_value) + content.salience
            for content in state.broadcast_contents
        )
        return -round(min(0.18, max(0.0, peak_signal * self.memory_gate_gain)), 6)

    def report_focus(self, state: GlobalWorkspaceState | None = None) -> list[str]:
        state = state or self.last_state
        if state is None:
            return []
        return [content.channel for content in state.broadcast_contents]

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "enabled": self.enabled,
            "action_bias_gain": self.action_bias_gain,
            "memory_gate_gain": self.memory_gate_gain,
            "last_state": self.last_state.to_dict() if self.last_state is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "GlobalWorkspace":
        if not payload:
            return cls()
        workspace = cls(
            capacity=int(payload.get("capacity", 2)),
            enabled=bool(payload.get("enabled", True)),
            action_bias_gain=float(payload.get("action_bias_gain", 0.35)),
            memory_gate_gain=float(payload.get("memory_gate_gain", 0.08)),
        )
        workspace.last_state = GlobalWorkspaceState.from_dict(
            payload.get("last_state") if isinstance(payload.get("last_state"), Mapping) else None
        )
        return workspace
