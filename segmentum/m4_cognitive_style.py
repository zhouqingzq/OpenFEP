from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math
import random

from .action_schema import ActionSchema, action_name, ensure_action_schema


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CognitiveStyleParameters:
    schema_version: str = "m4.cognitive_style.v1"
    uncertainty_sensitivity: float = 0.65
    error_aversion: float = 0.7
    exploration_bias: float = 0.55
    attention_selectivity: float = 0.6
    confidence_gain: float = 0.7
    update_rigidity: float = 0.65
    resource_pressure_sensitivity: float = 0.75

    def __post_init__(self) -> None:
        for field_name in (
            "uncertainty_sensitivity",
            "error_aversion",
            "exploration_bias",
            "attention_selectivity",
            "confidence_gain",
            "update_rigidity",
            "resource_pressure_sensitivity",
        ):
            object.__setattr__(self, field_name, _clamp01(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CognitiveStyleParameters":
        return cls(**{key: payload[key] for key in payload if key in cls.__dataclass_fields__})

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "schema_version": "m4.cognitive_style.v1",
            "type": "object",
            "required": [
                "schema_version",
                "uncertainty_sensitivity",
                "error_aversion",
                "exploration_bias",
                "attention_selectivity",
                "confidence_gain",
                "update_rigidity",
                "resource_pressure_sensitivity",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "uncertainty_sensitivity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "error_aversion": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "exploration_bias": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "attention_selectivity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence_gain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "update_rigidity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "resource_pressure_sensitivity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    energy: float
    budget: float
    stress: float
    time_remaining: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _clamp01(self.energy))
        object.__setattr__(self, "budget", _clamp01(self.budget))
        object.__setattr__(self, "stress", _clamp01(self.stress))
        object.__setattr__(self, "time_remaining", _clamp01(self.time_remaining))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResourceSnapshot":
        return cls(
            energy=float(payload.get("energy", 0.0)),
            budget=float(payload.get("budget", 0.0)),
            stress=float(payload.get("stress", 0.0)),
            time_remaining=float(payload.get("time_remaining", 0.0)),
        )


@dataclass(frozen=True)
class CandidateScore:
    action: dict[str, Any]
    total_score: float
    expected_confidence: float
    expected_prediction_error: float
    update_magnitude: float
    resource_penalty: float
    uncertainty_bonus: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionLogRecord:
    schema_version: str
    tick: int
    seed: int
    task_context: dict[str, Any]
    observation_evidence: dict[str, float]
    candidate_actions: list[dict[str, Any]]
    resource_state: dict[str, float]
    internal_confidence: float
    selected_action: str
    prediction_error: float
    update_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionLogRecord":
        return cls(
            schema_version=str(payload.get("schema_version", "m4.decision_log.v1")),
            tick=int(payload.get("tick", 0)),
            seed=int(payload.get("seed", 0)),
            task_context=dict(payload.get("task_context", {})),
            observation_evidence={
                str(key): float(value) for key, value in dict(payload.get("observation_evidence", {})).items()
            },
            candidate_actions=list(payload.get("candidate_actions", [])),
            resource_state={str(key): float(value) for key, value in dict(payload.get("resource_state", {})).items()},
            internal_confidence=float(payload.get("internal_confidence", 0.0)),
            selected_action=str(payload.get("selected_action", "")),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            update_magnitude=float(payload.get("update_magnitude", 0.0)),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "schema_version": "m4.decision_log.v1",
            "type": "object",
            "required": [
                "schema_version",
                "tick",
                "seed",
                "task_context",
                "observation_evidence",
                "candidate_actions",
                "resource_state",
                "internal_confidence",
                "selected_action",
                "prediction_error",
                "update_magnitude",
            ],
        }


def default_behavior_mapping_table() -> dict[str, dict[str, str]]:
    return {
        "resource_conservation": {
            "primary_parameter": "resource_pressure_sensitivity",
            "observable": "shift toward low-cost actions under high pressure",
        },
        "directed_exploration": {
            "primary_parameter": "exploration_bias",
            "observable": "scan or inspect wins more often under elevated uncertainty",
        },
        "confidence_sharpening": {
            "primary_parameter": "confidence_gain",
            "observable": "confidence increases with stronger evidence separation",
        },
        "update_rigidity": {
            "primary_parameter": "update_rigidity",
            "observable": "prediction error produces smaller update magnitudes",
        },
        "attention_selectivity": {
            "primary_parameter": "attention_selectivity",
            "observable": "evidence-weighted actions outscore distractor actions",
        },
        "error_avoidance": {
            "primary_parameter": "error_aversion",
            "observable": "high-risk actions are penalized after expected error increases",
        },
    }


class CognitiveParameterBridge:
    def __init__(self, parameters: CognitiveStyleParameters) -> None:
        self.parameters = parameters

    def _resource_pressure(self, resource_state: ResourceSnapshot) -> float:
        scarcity = 1.0 - ((resource_state.energy + resource_state.budget + resource_state.time_remaining) / 3.0)
        return _clamp01((scarcity + resource_state.stress) / 2.0)

    def score_action(
        self,
        action: ActionSchema,
        *,
        evidence_strength: float,
        uncertainty: float,
        expected_error: float,
        resource_state: ResourceSnapshot,
    ) -> CandidateScore:
        evidence_strength = _clamp01(evidence_strength)
        uncertainty = _clamp01(uncertainty)
        expected_error = _clamp01(expected_error)
        resource_pressure = self._resource_pressure(resource_state)
        action_label = action_name(action)
        cost = _clamp01(float(action.cost_estimate) + sum(max(0.0, float(v)) for v in action.resource_cost.values()) * 0.15)

        exploration_bonus = self.parameters.exploration_bias * uncertainty
        if action_label in ("scan", "inspect", "query"):
            exploration_bonus += 0.18 * uncertainty
        if action_label in ("rest", "conserve"):
            exploration_bonus -= 0.10 * uncertainty

        attention_bonus = self.parameters.attention_selectivity * evidence_strength
        if action_label in ("commit", "choose_right", "choose_left"):
            attention_bonus += 0.20 * evidence_strength

        resource_penalty = self.parameters.resource_pressure_sensitivity * resource_pressure * (0.35 + cost)
        if action_label in ("rest", "conserve"):
            resource_penalty *= 0.35

        error_penalty = self.parameters.error_aversion * expected_error * 0.7
        if action_label in ("rest", "conserve", "scan"):
            error_penalty *= 0.6

        base_score = attention_bonus + exploration_bonus - error_penalty - resource_penalty
        if action_label == "commit" and evidence_strength >= 0.75 and uncertainty <= 0.25:
            base_score += self.parameters.confidence_gain * 0.28
        if action_label in ("rest", "conserve") and resource_pressure > 0.55:
            base_score += self.parameters.resource_pressure_sensitivity * 0.30
        if action_label in ("scan", "inspect") and uncertainty > 0.6:
            base_score += self.parameters.uncertainty_sensitivity * 0.15

        confidence = _clamp01(
            0.30
            + self.parameters.confidence_gain * evidence_strength * 0.62
            - self.parameters.uncertainty_sensitivity * uncertainty * 0.20
            - expected_error * 0.10
        )
        update_magnitude = _clamp01(expected_error * (1.0 - self.parameters.update_rigidity * 0.75))
        return CandidateScore(
            action=action.to_dict(),
            total_score=round(base_score, 6),
            expected_confidence=round(confidence, 6),
            expected_prediction_error=round(expected_error, 6),
            update_magnitude=round(update_magnitude, 6),
            resource_penalty=round(resource_penalty, 6),
            uncertainty_bonus=round(exploration_bonus, 6),
        )

    def decide(
        self,
        *,
        tick: int,
        seed: int,
        task_context: dict[str, Any],
        observation_evidence: dict[str, float],
        actions: list[ActionSchema],
        resource_state: ResourceSnapshot,
    ) -> DecisionLogRecord:
        uncertainty = _clamp01(observation_evidence.get("uncertainty", 0.5))
        evidence_strength = _clamp01(observation_evidence.get("evidence_strength", 0.5))
        expected_error = _clamp01(observation_evidence.get("expected_error", 0.5))
        scores = [
            self.score_action(
                ensure_action_schema(action),
                evidence_strength=evidence_strength,
                uncertainty=uncertainty,
                expected_error=expected_error,
                resource_state=resource_state,
            )
            for action in actions
        ]
        rng = random.Random(seed * 1000 + tick)
        winner = max(
            scores,
            key=lambda item: (
                item.total_score,
                item.expected_confidence,
                -item.resource_penalty,
                rng.random() * 1e-6,
                str(item.action["name"]),
            ),
        )
        return DecisionLogRecord(
            schema_version="m4.decision_log.v1",
            tick=tick,
            seed=seed,
            task_context=dict(task_context),
            observation_evidence={key: round(float(value), 6) for key, value in observation_evidence.items()},
            candidate_actions=[score.to_dict() for score in scores],
            resource_state=resource_state.to_dict(),
            internal_confidence=winner.expected_confidence,
            selected_action=str(winner.action["name"]),
            prediction_error=winner.expected_prediction_error,
            update_magnitude=winner.update_magnitude,
        )


def canonical_action_schemas() -> list[ActionSchema]:
    return [
        ActionSchema(name="scan", cost_estimate=0.35, resource_cost={"tokens": 0.15}),
        ActionSchema(name="commit", cost_estimate=0.45, resource_cost={"tokens": 0.20}),
        ActionSchema(name="rest", cost_estimate=0.08, resource_cost={"tokens": 0.02}),
    ]


def run_cognitive_style_trial(
    parameters: CognitiveStyleParameters,
    *,
    seed: int = 41,
    stress: bool = False,
    ablate_resource_pressure: bool = False,
) -> dict[str, Any]:
    active_parameters = parameters
    if ablate_resource_pressure:
        active_parameters = CognitiveStyleParameters.from_dict(
            {**parameters.to_dict(), "resource_pressure_sensitivity": 0.0}
        )
    bridge = CognitiveParameterBridge(active_parameters)
    sequence = [
        (
            {"phase": "ambiguity_probe"},
            {"evidence_strength": 0.28, "uncertainty": 0.84, "expected_error": 0.42},
            ResourceSnapshot(energy=0.78, budget=0.82, stress=0.24, time_remaining=0.90),
        ),
        (
            {"phase": "commit_window"},
            {"evidence_strength": 0.87, "uncertainty": 0.18, "expected_error": 0.15},
            ResourceSnapshot(energy=0.74, budget=0.75, stress=0.28, time_remaining=0.70),
        ),
        (
            {"phase": "pressure_spike"},
            {"evidence_strength": 0.36, "uncertainty": 0.52, "expected_error": 0.39},
            ResourceSnapshot(energy=0.22 if stress else 0.32, budget=0.18 if stress else 0.28, stress=0.82, time_remaining=0.21),
        ),
        (
            {"phase": "recovery_probe"},
            {"evidence_strength": 0.51, "uncertainty": 0.31, "expected_error": 0.27},
            ResourceSnapshot(energy=0.56, budget=0.47, stress=0.36, time_remaining=0.45),
        ),
    ]
    logs = [
        bridge.decide(
            tick=index,
            seed=seed,
            task_context=context,
            observation_evidence=evidence,
            actions=canonical_action_schemas(),
            resource_state=resource_state,
        )
        for index, (context, evidence, resource_state) in enumerate(sequence, start=1)
    ]
    patterns = reconstruct_behavior_patterns(logs)
    return {
        "parameters": active_parameters.to_dict(),
        "logs": [record.to_dict() for record in logs],
        "patterns": patterns,
        "summary": {
            "selected_actions": [record.selected_action for record in logs],
            "mean_confidence": round(sum(record.internal_confidence for record in logs) / len(logs), 6),
            "mean_update_magnitude": round(sum(record.update_magnitude for record in logs) / len(logs), 6),
            "pattern_count": len(patterns),
        },
    }


def reconstruct_behavior_patterns(records: list[DecisionLogRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record)
        for record in records
    ]
    patterns: list[dict[str, Any]] = []
    if any(
        record.selected_action == "scan" and record.observation_evidence.get("uncertainty", 0.0) >= 0.75
        for record in normalized
    ):
        patterns.append(
            {"label": "directed_exploration", "evidence": "scan selected during high uncertainty"}
        )
    if any(
        record.selected_action == "rest"
        and ResourceSnapshot.from_dict(record.resource_state).energy <= 0.35
        and ResourceSnapshot.from_dict(record.resource_state).budget <= 0.30
        for record in normalized
    ):
        patterns.append(
            {"label": "resource_conservation", "evidence": "rest selected under elevated resource pressure"}
        )
    if any(
        record.selected_action == "commit"
        and record.internal_confidence >= 0.63
        and record.update_magnitude <= 0.20
        for record in normalized
    ):
        patterns.append(
            {"label": "confidence_sharpening", "evidence": "commit selected with high confidence and bounded update"}
        )
    return patterns


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
