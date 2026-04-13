from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .environment import clamp


THREAT_CHANNELS = frozenset({"danger", "threat", "hazard"})
SOCIAL_CHANNELS = frozenset({"social", "trust"})
CONTAMINATION_CHANNELS = frozenset({"food", "hazard"})


@dataclass(frozen=True, slots=True)
class AttentionAllocation:
    selected_channels: tuple[str, ...]
    dropped_channels: tuple[str, ...]
    weights: dict[str, float]
    bottleneck_load: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_channels": list(self.selected_channels),
            "dropped_channels": list(self.dropped_channels),
            "weights": {str(key): float(value) for key, value in self.weights.items()},
            "bottleneck_load": float(self.bottleneck_load),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "AttentionAllocation":
        if not payload:
            return cls(
                selected_channels=(),
                dropped_channels=(),
                weights={},
                bottleneck_load=0.0,
            )
        return cls(
            selected_channels=tuple(str(item) for item in payload.get("selected_channels", [])),
            dropped_channels=tuple(str(item) for item in payload.get("dropped_channels", [])),
            weights={
                str(key): float(value)
                for key, value in dict(payload.get("weights", {})).items()
                if isinstance(value, (int, float))
            },
            bottleneck_load=float(payload.get("bottleneck_load", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    tick: int
    salience_scores: dict[str, float]
    allocation: AttentionAllocation

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": int(self.tick),
            "salience_scores": {
                str(key): float(value) for key, value in self.salience_scores.items()
            },
            "allocation": self.allocation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "AttentionTrace | None":
        if not payload:
            return None
        return cls(
            tick=int(payload.get("tick", 0)),
            salience_scores={
                str(key): float(value)
                for key, value in dict(payload.get("salience_scores", {})).items()
                if isinstance(value, (int, float))
            },
            allocation=AttentionAllocation.from_dict(payload.get("allocation")),
        )


class AttentionBottleneck:
    def __init__(
        self,
        *,
        capacity: int = 3,
        enabled: bool = True,
        novelty_weight: float = 0.2,
        threat_weight: float = 0.35,
        surprise_weight: float = 0.45,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.enabled = bool(enabled)
        self.novelty_weight = float(novelty_weight)
        self.threat_weight = float(threat_weight)
        self.surprise_weight = float(surprise_weight)

    def score_channels(
        self,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
        narrative_priors: Mapping[str, float],
        memory_context: Mapping[str, object] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        trauma_bias = max(0.0, float(narrative_priors.get("trauma_bias", 0.0)))
        contamination_sensitivity = max(
            0.0,
            float(narrative_priors.get("contamination_sensitivity", 0.0)),
        )
        trust_prior = float(narrative_priors.get("trust_prior", 0.0))
        controllability_prior = float(narrative_priors.get("controllability_prior", 0.0))
        memory_context = memory_context or {}
        aggregate = memory_context.get("aggregate", {})
        if not isinstance(aggregate, Mapping):
            aggregate = {}
        chronic_threat_bias = max(0.0, float(aggregate.get("chronic_threat_bias", 0.0)))
        protected_anchor_bias = max(0.0, float(aggregate.get("protected_anchor_bias", 0.0)))
        sensitive_channels = {
            str(item) for item in memory_context.get("sensitive_channels", []) if str(item)
        }
        attention_biases = memory_context.get("attention_biases", {})
        if not isinstance(attention_biases, Mapping):
            attention_biases = {}

        configured_threat_channels = set(THREAT_CHANNELS)
        configured_social_channels = set(SOCIAL_CHANNELS)
        configured_novelty_channels = {
            str(item)
            for item in memory_context.get("novelty_channels", [])
            if str(item)
        }
        configured_threat_channels.update(
            str(item) for item in memory_context.get("threat_channels", []) if str(item)
        )
        configured_social_channels.update(
            str(item) for item in memory_context.get("social_channels", []) if str(item)
        )

        all_channels = sorted(set(observation) | set(prediction) | set(errors))
        for channel in all_channels:
            observed_value = float(observation.get(channel, 0.0))
            predicted_value = float(prediction.get(channel, 0.0))
            error_value = abs(float(errors.get(channel, observed_value - predicted_value)))
            score = error_value * self.surprise_weight

            if "novel" in channel or channel in configured_novelty_channels:
                score += observed_value * self.novelty_weight
                score += max(0.0, controllability_prior) * (
                    0.14 + observed_value * 0.24
                )
            if channel in configured_threat_channels:
                score += observed_value * self.threat_weight
                score += trauma_bias * (0.38 + observed_value * 0.40)
                score += chronic_threat_bias * (0.15 + observed_value * 0.20)
            if channel in CONTAMINATION_CHANNELS:
                score += contamination_sensitivity * observed_value * 0.25
            if channel in configured_social_channels:
                score += max(0.0, trust_prior) * (0.16 + observed_value * 0.34)
                score += max(0.0, -trust_prior) * (1.0 - observed_value) * 0.10
            if channel == "shelter":
                score += max(0.0, -controllability_prior) * (1.0 - observed_value) * 0.10
            if channel in sensitive_channels:
                score += 0.08 + (protected_anchor_bias * 0.12)
            score += max(0.0, float(attention_biases.get(channel, 0.0)))

            scores[channel] = round(max(0.0, score), 6)

        return scores

    def allocate(
        self,
        observation: Mapping[str, float],
        prediction: Mapping[str, float],
        errors: Mapping[str, float],
        narrative_priors: Mapping[str, float],
        tick: int,
        memory_context: Mapping[str, object] | None = None,
    ) -> AttentionTrace:
        salience_scores = self.score_channels(
            observation=observation,
            prediction=prediction,
            errors=errors,
            narrative_priors=narrative_priors,
            memory_context=memory_context,
        )
        ranked_channels = sorted(
            salience_scores,
            key=lambda channel: (-salience_scores[channel], channel),
        )
        if not self.enabled:
            selected = tuple(ranked_channels)
            dropped = ()
        else:
            selected = tuple(ranked_channels[: self.capacity])
            dropped = tuple(ranked_channels[self.capacity :])

        weights = {
            channel: (1.0 if channel in selected else 0.35)
            for channel in ranked_channels
        }
        bottleneck_load = 0.0
        if self.capacity > 0:
            bottleneck_load = round(min(1.0, len(ranked_channels) / float(self.capacity)), 6)

        return AttentionTrace(
            tick=int(tick),
            salience_scores=salience_scores,
            allocation=AttentionAllocation(
                selected_channels=selected,
                dropped_channels=dropped,
                weights=weights,
                bottleneck_load=bottleneck_load,
            ),
        )

    def filter_observation(
        self,
        observation: Mapping[str, float],
        allocation: AttentionAllocation,
        prediction: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        filtered: dict[str, float] = {}
        prediction = prediction or {}
        selected = set(allocation.selected_channels)
        all_channels = sorted(set(observation) | set(prediction) | set(allocation.weights))
        for channel in all_channels:
            observed_value = float(observation.get(channel, 0.0))
            if channel in selected or not self.enabled:
                filtered[channel] = observed_value
                continue
            anchor = float(prediction.get(channel, 0.5))
            filtered[channel] = clamp(anchor + (observed_value - anchor) * 0.35)
        return filtered

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "enabled": bool(self.enabled),
            "novelty_weight": float(self.novelty_weight),
            "threat_weight": float(self.threat_weight),
            "surprise_weight": float(self.surprise_weight),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "AttentionBottleneck":
        if not payload:
            return cls()
        return cls(
            capacity=int(payload.get("capacity", 3)),
            enabled=bool(payload.get("enabled", True)),
            novelty_weight=float(payload.get("novelty_weight", 0.2)),
            threat_weight=float(payload.get("threat_weight", 0.35)),
            surprise_weight=float(payload.get("surprise_weight", 0.45)),
        )
