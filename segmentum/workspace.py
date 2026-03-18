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
    "stress": {"rest": 0.12, "hide": 0.10, "forage": -0.10},
    "maintenance": {"rest": 0.18, "exploit_shelter": 0.08},
    "conflict": {"scan": 0.05, "hide": -0.04},
}

MAINTENANCE_HINTS = {
    "danger": ("resource_guard", "hide"),
    "stress": ("stress_relief", "rest"),
    "temperature": ("thermal_rebalance", "thermoregulate"),
    "food": ("energy_recovery", "forage"),
    "shelter": ("shelter_security", "exploit_shelter"),
}


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class WorkspaceContent:
    channel: str
    salience: float
    observation_value: float
    prediction_value: float
    error_value: float
    source: str = "attention"
    action_hints: dict[str, float] = field(default_factory=dict)
    carry_over_strength: float = 0.0
    age: int = 0
    report_accessible: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "salience": _round(self.salience),
            "observation_value": _round(self.observation_value),
            "prediction_value": _round(self.prediction_value),
            "error_value": _round(self.error_value),
            "source": self.source,
            "action_hints": {str(key): _round(value) for key, value in self.action_hints.items()},
            "carry_over_strength": _round(self.carry_over_strength),
            "age": int(self.age),
            "report_accessible": bool(self.report_accessible),
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
            carry_over_strength=float(payload.get("carry_over_strength", 0.0)),
            age=int(payload.get("age", 0)),
            report_accessible=bool(payload.get("report_accessible", True)),
        )


@dataclass(frozen=True)
class GlobalWorkspaceState:
    tick: int
    capacity: int
    latent_candidates: tuple[WorkspaceContent, ...]
    attended_candidates: tuple[WorkspaceContent, ...]
    broadcast_contents: tuple[WorkspaceContent, ...]
    suppressed_candidates: tuple[WorkspaceContent, ...]
    carry_over_contents: tuple[WorkspaceContent, ...]
    suppressed_channels: tuple[str, ...]
    broadcast_intensity: float
    persistence_horizon: int
    replacement_pressure: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "capacity": self.capacity,
            "latent_candidates": [content.to_dict() for content in self.latent_candidates],
            "attended_candidates": [content.to_dict() for content in self.attended_candidates],
            "broadcast_contents": [content.to_dict() for content in self.broadcast_contents],
            "suppressed_candidates": [content.to_dict() for content in self.suppressed_candidates],
            "carry_over_contents": [content.to_dict() for content in self.carry_over_contents],
            "suppressed_channels": list(self.suppressed_channels),
            "broadcast_intensity": _round(self.broadcast_intensity),
            "persistence_horizon": int(self.persistence_horizon),
            "replacement_pressure": _round(self.replacement_pressure),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "GlobalWorkspaceState | None":
        if not payload:
            return None

        def _contents(key: str) -> tuple[WorkspaceContent, ...]:
            raw_contents = payload.get(key, [])
            return tuple(
                WorkspaceContent.from_dict(item)
                for item in raw_contents
                if isinstance(item, Mapping)
            )

        broadcast_contents = _contents("broadcast_contents")
        suppressed_candidates = _contents("suppressed_candidates")
        latent_candidates = _contents("latent_candidates")
        attended_candidates = _contents("attended_candidates")
        carry_over_contents = _contents("carry_over_contents")
        if not latent_candidates:
            latent_candidates = broadcast_contents + suppressed_candidates
        if not attended_candidates:
            attended_candidates = broadcast_contents

        return cls(
            tick=int(payload.get("tick", 0)),
            capacity=max(1, int(payload.get("capacity", 1))),
            latent_candidates=latent_candidates,
            attended_candidates=attended_candidates,
            broadcast_contents=broadcast_contents,
            suppressed_candidates=suppressed_candidates,
            carry_over_contents=carry_over_contents,
            suppressed_channels=tuple(str(item) for item in payload.get("suppressed_channels", [])),
            broadcast_intensity=float(payload.get("broadcast_intensity", 0.0)),
            persistence_horizon=int(payload.get("persistence_horizon", 0)),
            replacement_pressure=float(payload.get("replacement_pressure", 0.0)),
        )


class GlobalWorkspace:
    def __init__(
        self,
        *,
        capacity: int = 2,
        enabled: bool = True,
        action_bias_gain: float = 0.35,
        memory_gate_gain: float = 0.08,
        persistence_ticks: int = 2,
        carry_over_decay: float = 0.82,
        carry_over_min_salience: float = 0.12,
        report_carry_over: bool = True,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.enabled = bool(enabled)
        self.action_bias_gain = float(action_bias_gain)
        self.memory_gate_gain = float(memory_gate_gain)
        self.persistence_ticks = max(0, int(persistence_ticks))
        self.carry_over_decay = float(carry_over_decay)
        self.carry_over_min_salience = float(carry_over_min_salience)
        self.report_carry_over = bool(report_carry_over)
        self.last_state: GlobalWorkspaceState | None = None

    def _build_content(
        self,
        *,
        channel: str,
        salience: float,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
        source: str,
        carry_over_strength: float = 0.0,
        age: int = 0,
        report_accessible: bool = True,
    ) -> WorkspaceContent:
        return WorkspaceContent(
            channel=channel,
            salience=_round(max(0.0, salience)),
            observation_value=float(observation.get(channel, 0.0)),
            prediction_value=float(prediction.get(channel, 0.0)),
            error_value=float(errors.get(channel, 0.0)),
            source=source,
            action_hints=dict(ACTION_HINTS.get(channel, {})),
            carry_over_strength=_round(max(0.0, carry_over_strength)),
            age=max(0, int(age)),
            report_accessible=bool(report_accessible),
        )

    def _latent_candidates(
        self,
        *,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
        salience_scores: Mapping[str, float],
    ) -> tuple[WorkspaceContent, ...]:
        channels = sorted(set(observation) | set(prediction) | set(errors) | set(salience_scores))
        contents = [
            self._build_content(
                channel=channel,
                salience=float(salience_scores.get(channel, abs(float(errors.get(channel, 0.0))))),
                observation=observation,
                prediction=prediction,
                errors=errors,
                source="latent",
            )
            for channel in channels
        ]
        return tuple(sorted(contents, key=lambda item: (-item.salience, item.channel)))

    def _carry_over_candidates(
        self,
        *,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
    ) -> tuple[WorkspaceContent, ...]:
        if (
            self.last_state is None
            or self.persistence_ticks <= 0
            or not self.last_state.broadcast_contents
        ):
            return ()
        carried: list[WorkspaceContent] = []
        for content in self.last_state.broadcast_contents:
            age = int(content.age) + 1
            if age > self.persistence_ticks:
                continue
            decayed_salience = max(0.0, content.salience * (self.carry_over_decay ** age))
            if decayed_salience < self.carry_over_min_salience:
                continue
            carried.append(
                self._build_content(
                    channel=content.channel,
                    salience=decayed_salience,
                    observation=observation,
                    prediction=prediction,
                    errors=errors,
                    source="carry_over",
                    carry_over_strength=decayed_salience,
                    age=age,
                    report_accessible=self.report_carry_over,
                )
            )
        return tuple(sorted(carried, key=lambda item: (-item.salience, item.channel)))

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
            salience_scores = {
                channel: abs(float(errors.get(channel, 0.0)))
                for channel in sorted(set(observation) | set(prediction) | set(errors))
            }
            attended_channels = tuple(
                sorted(salience_scores, key=lambda channel: (-salience_scores[channel], channel))[
                    : self.capacity
                ]
            )
        else:
            salience_scores = dict(attention_trace.salience_scores)
            attended_channels = attention_trace.allocation.selected_channels

        latent_candidates = self._latent_candidates(
            observation=observation,
            prediction=prediction,
            errors=errors,
            salience_scores=salience_scores,
        )
        latent_index = {content.channel: content for content in latent_candidates}
        carry_over_candidates = self._carry_over_candidates(
            observation=observation,
            prediction=prediction,
            errors=errors,
        )
        carry_over_index = {content.channel: content for content in carry_over_candidates}

        attended_candidates = tuple(
            self._build_content(
                channel=channel,
                salience=float(latent_index.get(channel, carry_over_index.get(channel)).salience)
                if channel in latent_index or channel in carry_over_index
                else 0.0,
                observation=observation,
                prediction=prediction,
                errors=errors,
                source="attention",
            )
            for channel in attended_channels
            if channel in latent_index or channel in carry_over_index
        )

        competitive_pool: list[WorkspaceContent] = []
        for content in attended_candidates:
            competitive_pool.append(content)
        for content in carry_over_candidates:
            if content.channel not in {item.channel for item in competitive_pool}:
                competitive_pool.append(content)
        competitive_pool.sort(key=lambda item: (-item.salience, item.channel))

        winners = tuple(competitive_pool[: self.capacity])
        winner_channels = {content.channel for content in winners}
        carry_over_contents = tuple(
            content for content in winners if content.source == "carry_over"
        )
        suppressed_candidates = tuple(
            content
            for content in latent_candidates + carry_over_candidates
            if content.channel not in winner_channels
        )
        suppressed_channels = tuple(content.channel for content in suppressed_candidates)

        intensity = 0.0
        if winners:
            intensity = sum(content.salience for content in winners) / len(winners)
        replacement_pressure = 0.0
        if suppressed_candidates:
            replacement_pressure = max(content.salience for content in suppressed_candidates)

        state = GlobalWorkspaceState(
            tick=int(tick),
            capacity=self.capacity,
            latent_candidates=latent_candidates,
            attended_candidates=attended_candidates,
            broadcast_contents=winners,
            suppressed_candidates=suppressed_candidates,
            carry_over_contents=carry_over_contents,
            suppressed_channels=suppressed_channels,
            broadcast_intensity=_round(intensity),
            persistence_horizon=self.persistence_ticks,
            replacement_pressure=_round(replacement_pressure),
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
            persistence_bonus = 1.0 + min(0.25, content.carry_over_strength * 0.25)
            bias += hint * salience_scale * persistence_bonus * self.action_bias_gain
        return _round(bias)

    def memory_threshold_delta(self, state: GlobalWorkspaceState | None = None) -> float:
        state = state or self.last_state
        if not self.enabled or state is None or not state.broadcast_contents:
            return 0.0
        peak_signal = max(
            abs(content.error_value) + content.salience + content.carry_over_strength
            for content in state.broadcast_contents
        )
        return -_round(min(0.18, max(0.0, peak_signal * self.memory_gate_gain)))

    def report_focus(self, state: GlobalWorkspaceState | None = None) -> list[str]:
        state = state or self.last_state
        if state is None:
            return []
        return [
            content.channel
            for content in state.broadcast_contents
            if content.report_accessible
        ]

    def conscious_report_payload(self, state: GlobalWorkspaceState | None = None) -> dict[str, object]:
        state = state or self.last_state
        if state is None:
            return {
                "broadcast_contents": [],
                "carry_over_contents": [],
                "accessible_channels": [],
                "suppressed_channels": [],
                "leakage_checked": True,
            }
        accessible = [
            content.channel for content in state.broadcast_contents if content.report_accessible
        ]
        carry_over = [
            content.channel for content in state.carry_over_contents if content.report_accessible
        ]
        return {
            "broadcast_contents": accessible,
            "carry_over_contents": carry_over,
            "accessible_channels": accessible,
            "suppressed_channels": list(state.suppressed_channels),
            "leakage_checked": not bool(set(accessible) & set(state.suppressed_channels)),
        }

    def maintenance_signal(self, state: GlobalWorkspaceState | None = None) -> dict[str, object]:
        state = state or self.last_state
        if state is None or not state.broadcast_contents:
            return {
                "priority_gain": 0.0,
                "active_tasks": [],
                "recommended_action": "",
            }
        tasks: list[str] = []
        action_scores: dict[str, float] = {}
        total_gain = 0.0
        for content in state.broadcast_contents:
            task_hint = MAINTENANCE_HINTS.get(content.channel)
            gain = min(0.35, content.salience + max(0.0, content.observation_value * 0.25))
            total_gain += gain
            if task_hint is not None and task_hint[0] not in tasks:
                tasks.append(task_hint[0])
                action_scores[task_hint[1]] = action_scores.get(task_hint[1], 0.0) + gain
        recommended_action = ""
        if action_scores:
            recommended_action = sorted(
                action_scores,
                key=lambda key: (-action_scores[key], key),
            )[0]
        return {
            "priority_gain": _round(total_gain / max(1, len(state.broadcast_contents))),
            "active_tasks": tasks,
            "recommended_action": recommended_action,
        }

    def metacognitive_review_signal(self, state: GlobalWorkspaceState | None = None) -> dict[str, object]:
        state = state or self.last_state
        if state is None:
            return {
                "review_gain": 0.0,
                "conflict_channels": [],
            }
        conflict_channels = [
            content.channel
            for content in state.broadcast_contents
            if content.channel in {"conflict", "stress", "social", "danger"}
            and (abs(content.error_value) >= 0.12 or content.salience >= 0.20)
        ]
        gain = sum(
            min(0.30, content.salience + abs(content.error_value))
            for content in state.broadcast_contents
            if content.channel in conflict_channels
        )
        return {
            "review_gain": _round(gain / max(1, len(conflict_channels) or 1)),
            "conflict_channels": conflict_channels,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "enabled": self.enabled,
            "action_bias_gain": self.action_bias_gain,
            "memory_gate_gain": self.memory_gate_gain,
            "persistence_ticks": self.persistence_ticks,
            "carry_over_decay": self.carry_over_decay,
            "carry_over_min_salience": self.carry_over_min_salience,
            "report_carry_over": self.report_carry_over,
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
            persistence_ticks=int(payload.get("persistence_ticks", 2)),
            carry_over_decay=float(payload.get("carry_over_decay", 0.82)),
            carry_over_min_salience=float(payload.get("carry_over_min_salience", 0.12)),
            report_carry_over=bool(payload.get("report_carry_over", True)),
        )
        workspace.last_state = GlobalWorkspaceState.from_dict(
            payload.get("last_state") if isinstance(payload.get("last_state"), Mapping) else None
        )
        return workspace
