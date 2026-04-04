from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
import math
import random

from .action_schema import ActionSchema, action_name, ensure_action_schema


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


PARAMETER_REFERENCE: dict[str, dict[str, Any]] = {
    "uncertainty_sensitivity": {
        "range": [0.0, 1.0],
        "default": 0.65,
        "physical_meaning": "Sensitivity to ambiguity and incomplete evidence.",
        "decision_path": "Higher values depress confidence and favor inspect-like actions under uncertainty.",
        "observable_relationship": "Mapped through high-uncertainty confidence drop and inspect/scan preference.",
    },
    "error_aversion": {
        "range": [0.0, 1.0],
        "default": 0.70,
        "physical_meaning": "Penalty applied to options with elevated expected failure cost.",
        "decision_path": "Higher values suppress risky actions and increase conservative recovery choices after error signals.",
        "observable_relationship": "Mapped through risky-action rejection and post-error conservative switching.",
    },
    "exploration_bias": {
        "range": [0.0, 1.0],
        "default": 0.55,
        "physical_meaning": "Preference for information-seeking or novel actions.",
        "decision_path": "Higher values increase query/scan/inspect selection in ambiguous contexts.",
        "observable_relationship": "Mapped through unknown-option choice rate and lower repeated-choice streaks.",
    },
    "attention_selectivity": {
        "range": [0.0, 1.0],
        "default": 0.60,
        "physical_meaning": "Degree to which attention concentrates on the strongest evidence channels.",
        "decision_path": "Higher values improve evidence-to-choice alignment and suppress distractor actions.",
        "observable_relationship": "Mapped through dominant-feature attention share and evidence-aligned action wins.",
    },
    "confidence_gain": {
        "range": [0.0, 1.0],
        "default": 0.70,
        "physical_meaning": "Amplification from clean evidence separation into internal confidence.",
        "decision_path": "Higher values raise confidence and commit rate when evidence becomes decisive.",
        "observable_relationship": "Mapped through confidence-vs-evidence slope and high-evidence commit rate.",
    },
    "update_rigidity": {
        "range": [0.0, 1.0],
        "default": 0.65,
        "physical_meaning": "Resistance to changing internal state after observed error.",
        "decision_path": "Higher values reduce learning-step magnitude and prolong strategy persistence after error.",
        "observable_relationship": "Mapped through lower update magnitude and slower post-error switching.",
    },
    "resource_pressure_sensitivity": {
        "range": [0.0, 1.0],
        "default": 0.75,
        "physical_meaning": "Pressure response to low energy, low budget, high stress, or little time.",
        "decision_path": "Higher values favor low-cost recovery and conservation actions under scarcity.",
        "observable_relationship": "Mapped through low-cost-action rate and conservation trigger threshold under pressure.",
    },
    "virtual_prediction_error_gain": {
        "range": [0.0, 1.0],
        "default": 0.68,
        "physical_meaning": "Weight placed on imagined or counterfactual prediction-error signals.",
        "decision_path": "Higher values amplify caution when simulated losses conflict with direct evidence.",
        "observable_relationship": "Mapped through conflict-condition avoidance and counterfactual-loss driven conservative shifts.",
    },
}


TRAIN_PROFILE_SEEDS = [11, 12, 13, 14]
EVAL_PROFILE_SEEDS = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]


@dataclass(frozen=True)
class CognitiveStyleParameters:
    schema_version: str = "m4.cognitive_style.v2"
    uncertainty_sensitivity: float = 0.65
    error_aversion: float = 0.70
    exploration_bias: float = 0.55
    attention_selectivity: float = 0.60
    confidence_gain: float = 0.70
    update_rigidity: float = 0.65
    resource_pressure_sensitivity: float = 0.75
    virtual_prediction_error_gain: float = 0.68

    def __post_init__(self) -> None:
        for field_name in PARAMETER_REFERENCE:
            object.__setattr__(self, field_name, _clamp01(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CognitiveStyleParameters":
        normalized = dict(payload)
        if "schema_version" not in normalized:
            normalized["schema_version"] = cls().schema_version
        for field_name, spec in PARAMETER_REFERENCE.items():
            normalized.setdefault(field_name, spec["default"])
        return cls(**{key: normalized[key] for key in cls.__dataclass_fields__ if key in normalized})

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "schema_version": cls().schema_version,
            "type": "object",
            "required": ["schema_version", *PARAMETER_REFERENCE.keys()],
            "properties": {
                "schema_version": {"type": "string"},
                **{
                    name: {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": spec["default"]}
                    for name, spec in PARAMETER_REFERENCE.items()
                },
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
    expected_value: float
    expected_confidence: float
    expected_prediction_error: float
    expected_prediction_error_vector: dict[str, float]
    update_magnitude: float
    resource_penalty: float
    uncertainty_bonus: float
    resource_cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionLogRecord:
    schema_version: str
    tick: int
    timestamp: str
    seed: int
    task_context: dict[str, Any]
    percept_summary: dict[str, Any]
    observation_evidence: dict[str, float]
    prediction_error_vector: dict[str, float]
    attention_allocation: dict[str, float]
    candidate_actions: list[dict[str, Any]]
    parameter_snapshot: dict[str, Any]
    resource_state: dict[str, float]
    internal_confidence: float
    selected_action: str
    result_feedback: dict[str, Any]
    model_update: dict[str, Any]
    prediction_error: float
    update_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionLogRecord":
        prediction_error_vector = {
            str(key): float(value)
            for key, value in dict(payload.get("prediction_error_vector", {})).items()
        }
        if not prediction_error_vector:
            scalar_error = float(payload.get("prediction_error", 0.0))
            prediction_error_vector = {
                "direct_error": scalar_error,
                "virtual_error": scalar_error,
                "signed_total": scalar_error,
            }
        result_feedback = dict(payload.get("result_feedback", {}))
        if not result_feedback:
            result_feedback = {
                "observed_outcome": "legacy_unknown",
                "reward": round(1.0 - prediction_error_vector["signed_total"], 6),
                "counterfactual_warning": prediction_error_vector["virtual_error"] > prediction_error_vector["direct_error"],
            }
        model_update = dict(payload.get("model_update", {}))
        if not model_update:
            update_magnitude = float(payload.get("update_magnitude", 0.0))
            model_update = {
                "magnitude": update_magnitude,
                "strategy_shift": round(update_magnitude * 0.5, 6),
                "confidence_delta": round(-prediction_error_vector["signed_total"] * 0.1, 6),
            }
        return cls(
            schema_version=str(payload.get("schema_version", "m4.decision_log.v3")),
            tick=int(payload.get("tick", 0)),
            timestamp=str(
                payload.get(
                    "timestamp",
                    datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(timespec="seconds"),
                )
            ),
            seed=int(payload.get("seed", 0)),
            task_context=dict(payload.get("task_context", {})),
            percept_summary=dict(payload.get("percept_summary", payload.get("task_context", {}))),
            observation_evidence={str(key): float(value) for key, value in dict(payload.get("observation_evidence", {})).items()},
            prediction_error_vector=prediction_error_vector,
            attention_allocation={str(key): float(value) for key, value in dict(payload.get("attention_allocation", {})).items()},
            candidate_actions=list(payload.get("candidate_actions", [])),
            parameter_snapshot=dict(payload.get("parameter_snapshot", {})),
            resource_state={str(key): float(value) for key, value in dict(payload.get("resource_state", {})).items()},
            internal_confidence=float(payload.get("internal_confidence", 0.0)),
            selected_action=str(payload.get("selected_action", "")),
            result_feedback=result_feedback,
            model_update=model_update,
            prediction_error=float(payload.get("prediction_error", prediction_error_vector["signed_total"])),
            update_magnitude=float(payload.get("update_magnitude", model_update.get("magnitude", 0.0))),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "schema_version": "m4.decision_log.v3",
            "type": "object",
            "required": [
                "schema_version",
                "tick",
                "timestamp",
                "seed",
                "task_context",
                "percept_summary",
                "observation_evidence",
                "prediction_error_vector",
                "attention_allocation",
                "candidate_actions",
                "parameter_snapshot",
                "resource_state",
                "internal_confidence",
                "selected_action",
                "result_feedback",
                "model_update",
                "prediction_error",
                "update_magnitude",
            ],
            "properties": {
                "candidate_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "action",
                            "expected_value",
                            "expected_confidence",
                            "expected_prediction_error",
                            "resource_cost",
                        ],
                    },
                }
            },
        }


def parameter_reference_markdown() -> str:
    lines = ["# M4.1 Parameter Reference", ""]
    for parameter_name, spec in PARAMETER_REFERENCE.items():
        lines.append(f"## `{parameter_name}`")
        lines.append(f"- Range: `{spec['range'][0]}..{spec['range'][1]}`")
        lines.append(f"- Default: `{spec['default']}`")
        lines.append(f"- Physical meaning: {spec['physical_meaning']}")
        lines.append(f"- Decision path: {spec['decision_path']}")
        lines.append(f"- Observable relationship: {spec['observable_relationship']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


class CognitiveParameterBridge:
    def __init__(self, parameters: CognitiveStyleParameters) -> None:
        self.parameters = parameters

    def _resource_pressure(self, resource_state: ResourceSnapshot) -> float:
        scarcity = 1.0 - ((resource_state.energy + resource_state.budget + resource_state.time_remaining) / 3.0)
        return _clamp01((scarcity + resource_state.stress) / 2.0)

    def _prediction_error_vector(
        self,
        *,
        evidence_strength: float,
        uncertainty: float,
        expected_error: float,
        imagined_risk: float,
    ) -> dict[str, float]:
        direct_error = _clamp01(expected_error * (0.55 + uncertainty * 0.45))
        virtual_error = _clamp01(
            expected_error * 0.35
            + imagined_risk * (0.30 + self.parameters.virtual_prediction_error_gain * 0.70)
            + uncertainty * self.parameters.virtual_prediction_error_gain * 0.25
            - evidence_strength * 0.14
        )
        signed_total = _clamp01((direct_error + virtual_error) / 2.0)
        return {
            "direct_error": round(direct_error, 6),
            "virtual_error": round(virtual_error, 6),
            "signed_total": round(signed_total, 6),
        }

    def _attention_allocation(
        self,
        *,
        evidence_strength: float,
        uncertainty: float,
        expected_error: float,
        imagined_risk: float,
    ) -> dict[str, float]:
        raw = {
            "evidence": 0.24 + self.parameters.attention_selectivity * evidence_strength * 0.75,
            "uncertainty": 0.12 + self.parameters.uncertainty_sensitivity * uncertainty * 0.60,
            "error": 0.12 + self.parameters.error_aversion * expected_error * 0.62,
            "counterfactual": 0.10 + self.parameters.virtual_prediction_error_gain * imagined_risk * 0.65,
        }
        total = sum(raw.values()) or 1.0
        return {key: round(value / total, 6) for key, value in raw.items()}

    def score_action(
        self,
        action: ActionSchema,
        *,
        evidence_strength: float,
        uncertainty: float,
        expected_error: float,
        resource_state: ResourceSnapshot,
        imagined_risk: float = 0.0,
    ) -> CandidateScore:
        evidence_strength = _clamp01(evidence_strength)
        uncertainty = _clamp01(uncertainty)
        expected_error = _clamp01(expected_error)
        imagined_risk = _clamp01(imagined_risk)
        resource_pressure = self._resource_pressure(resource_state)
        label = action_name(action)
        cost = _clamp01(float(action.cost_estimate) + sum(max(0.0, float(v)) for v in action.resource_cost.values()) * 0.18)
        prediction_error_vector = self._prediction_error_vector(
            evidence_strength=evidence_strength,
            uncertainty=uncertainty,
            expected_error=expected_error,
            imagined_risk=imagined_risk,
        )
        direct_error = prediction_error_vector["direct_error"]
        virtual_error = prediction_error_vector["virtual_error"]

        inspect_actions = {"scan", "inspect", "query"}
        conservative_actions = {"rest", "conserve", "recover", "scan", "inspect", "query", "plan"}
        risky_actions = {"commit", "guess", "retry"}

        exploration_bonus = self.parameters.exploration_bias * uncertainty * 0.28
        if label in inspect_actions:
            exploration_bonus += 0.08 + self.parameters.exploration_bias * (0.20 + uncertainty * 0.48)
        if label in {"rest", "conserve"}:
            exploration_bonus -= 0.08 * uncertainty
            exploration_bonus -= self.parameters.exploration_bias * max(0.0, uncertainty - 0.35) * 0.18
        if label in {"commit", "plan", "recover"}:
            exploration_bonus -= self.parameters.exploration_bias * uncertainty * 0.12

        evidence_bonus = self.parameters.attention_selectivity * evidence_strength * 0.34
        if label == "commit":
            evidence_bonus += evidence_strength * 0.30 + self.parameters.confidence_gain * 0.16
        if label == "plan":
            evidence_bonus += self.parameters.attention_selectivity * max(0.0, 0.60 - uncertainty) * 0.18
        if label == "recover":
            evidence_bonus += self.parameters.error_aversion * expected_error * 0.16

        resource_penalty = self.parameters.resource_pressure_sensitivity * resource_pressure * (0.18 + cost)
        if label in {"rest", "conserve", "recover"}:
            resource_penalty *= 0.40
        if label == "retry":
            resource_penalty *= 1.18
        if label == "commit":
            resource_penalty *= 1.05

        direct_penalty = self.parameters.error_aversion * direct_error * 0.62
        virtual_penalty = self.parameters.virtual_prediction_error_gain * max(0.0, virtual_error - direct_error * 0.45) * 0.55
        if label in conservative_actions:
            direct_penalty *= 0.62
            virtual_penalty *= 0.60
        if label in risky_actions:
            direct_penalty *= 1.12
            virtual_penalty *= 1.18
            direct_penalty += self.parameters.error_aversion * expected_error * 0.18

        score = evidence_bonus + exploration_bonus - resource_penalty - direct_penalty - virtual_penalty
        if label in inspect_actions and uncertainty >= 0.60:
            score += self.parameters.uncertainty_sensitivity * 0.24
        if label == "commit" and evidence_strength >= 0.72 and uncertainty <= 0.30:
            score += self.parameters.confidence_gain * 0.30
        if label == "commit":
            score -= self.parameters.uncertainty_sensitivity * uncertainty * 0.22
        if label == "recover" and expected_error >= 0.50:
            score += (self.parameters.error_aversion + self.parameters.update_rigidity) * 0.18
        if label in {"recover", "rest", "conserve"}:
            score += self.parameters.error_aversion * expected_error * 0.14
        if label in {"rest", "conserve"} and resource_pressure >= 0.58:
            score += self.parameters.resource_pressure_sensitivity * 0.34
        if label == "retry":
            score += (1.0 - self.parameters.error_aversion) * max(0.0, 0.55 - expected_error) * 0.18
        if label == "guess":
            score += (1.0 - self.parameters.error_aversion) * 0.06
            score -= uncertainty * 0.10
        if label == "commit" and virtual_error > direct_error:
            score -= self.parameters.virtual_prediction_error_gain * (virtual_error - direct_error) * 0.48
        if label in conservative_actions and virtual_error > direct_error:
            score += self.parameters.virtual_prediction_error_gain * (virtual_error - direct_error) * 0.18

        confidence = _clamp01(
            0.22
            + self.parameters.confidence_gain * evidence_strength * 0.72
            - self.parameters.uncertainty_sensitivity * uncertainty * 0.24
            - direct_error * 0.10
            - self.parameters.virtual_prediction_error_gain * virtual_error * 0.08
        )
        if label in {"rest", "recover"}:
            confidence = _clamp01(confidence + self.parameters.error_aversion * 0.05)
        if label == "guess":
            confidence = _clamp01(confidence - 0.10)

        update_magnitude = _clamp01(prediction_error_vector["signed_total"] * (1.0 - self.parameters.update_rigidity * 0.72))
        expected_value = _clamp01(0.45 + evidence_strength * 0.30 - direct_error * 0.18 - resource_penalty * 0.12)
        return CandidateScore(
            action=action.to_dict(),
            total_score=round(score, 6),
            expected_value=round(expected_value, 6),
            expected_confidence=round(confidence, 6),
            expected_prediction_error=round(prediction_error_vector["signed_total"], 6),
            expected_prediction_error_vector=prediction_error_vector,
            update_magnitude=round(update_magnitude, 6),
            resource_penalty=round(resource_penalty, 6),
            uncertainty_bonus=round(exploration_bonus, 6),
            resource_cost=round(cost, 6),
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
        imagined_risk = _clamp01(observation_evidence.get("imagined_risk", expected_error * 0.5))
        attention_allocation = self._attention_allocation(
            evidence_strength=evidence_strength,
            uncertainty=uncertainty,
            expected_error=expected_error,
            imagined_risk=imagined_risk,
        )
        scores = [
            self.score_action(
                ensure_action_schema(action),
                evidence_strength=evidence_strength,
                uncertainty=uncertainty,
                expected_error=expected_error,
                resource_state=resource_state,
                imagined_risk=imagined_risk,
            )
            for action in actions
        ]
        rng = random.Random(seed * 1000 + tick)
        winner = max(
            scores,
            key=lambda item: (
                item.total_score,
                item.expected_confidence,
                item.expected_value,
                -item.resource_penalty,
                rng.random() * 1e-6,
                str(item.action["name"]),
            ),
        )
        dominant_signal = max(attention_allocation, key=attention_allocation.get)
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(seed * 97) + tick * 11)
        reward = round(1.0 - winner.expected_prediction_error - winner.resource_penalty * 0.18, 6)
        outcome = "stabilized" if reward >= 0.58 else "fragile" if reward >= 0.35 else "lossy"
        return DecisionLogRecord(
            schema_version="m4.decision_log.v3",
            tick=tick,
            timestamp=timestamp.isoformat(timespec="seconds"),
            seed=seed,
            task_context=dict(task_context),
            percept_summary={
                "dominant_signal": dominant_signal,
                "evidence_band": "high" if evidence_strength >= 0.70 else "medium" if evidence_strength >= 0.40 else "low",
                "uncertainty_band": "high" if uncertainty >= 0.60 else "medium" if uncertainty >= 0.30 else "low",
                "pressure_band": "high" if self._resource_pressure(resource_state) >= 0.60 else "medium" if self._resource_pressure(resource_state) >= 0.35 else "low",
            },
            observation_evidence={key: round(float(value), 6) for key, value in observation_evidence.items()},
            prediction_error_vector=winner.expected_prediction_error_vector,
            attention_allocation=attention_allocation,
            candidate_actions=[
                {
                    **score.to_dict(),
                    "expected_prediction_error": score.expected_prediction_error,
                    "resource_cost": score.resource_cost,
                }
                for score in scores
            ],
            parameter_snapshot=self.parameters.to_dict(),
            resource_state=resource_state.to_dict(),
            internal_confidence=winner.expected_confidence,
            selected_action=str(winner.action["name"]),
            result_feedback={
                "observed_outcome": outcome,
                "reward": reward,
                "counterfactual_warning": winner.expected_prediction_error_vector["virtual_error"] > winner.expected_prediction_error_vector["direct_error"],
            },
            model_update={
                "magnitude": winner.update_magnitude,
                "strategy_shift": round(winner.update_magnitude * (1.0 - self.parameters.update_rigidity * 0.28), 6),
                "confidence_delta": round(winner.expected_confidence - 0.5, 6),
            },
            prediction_error=winner.expected_prediction_error,
            update_magnitude=winner.update_magnitude,
        )


def canonical_action_schemas() -> list[ActionSchema]:
    return [
        ActionSchema(name="scan", cost_estimate=0.18, resource_cost={"tokens": 0.05}),
        ActionSchema(name="inspect", cost_estimate=0.20, resource_cost={"tokens": 0.06}),
        ActionSchema(name="query", cost_estimate=0.22, resource_cost={"tokens": 0.08}),
        ActionSchema(name="commit", cost_estimate=0.42, resource_cost={"tokens": 0.16}),
        ActionSchema(name="plan", cost_estimate=0.16, resource_cost={"tokens": 0.04}),
        ActionSchema(name="recover", cost_estimate=0.14, resource_cost={"tokens": 0.03}),
        ActionSchema(name="conserve", cost_estimate=0.10, resource_cost={"tokens": 0.02}),
        ActionSchema(name="rest", cost_estimate=0.08, resource_cost={"tokens": 0.02}),
        ActionSchema(name="guess", cost_estimate=0.06, resource_cost={"tokens": 0.02}),
        ActionSchema(name="retry", cost_estimate=0.24, resource_cost={"tokens": 0.08}),
    ]


SCENARIO_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "ambiguity_probe": [
        {
            "phase": "ambiguity_probe",
            "evidence": {"evidence_strength": 0.24, "uncertainty": 0.88, "expected_error": 0.36, "imagined_risk": 0.20},
            "resource": {"energy": 0.74, "budget": 0.76, "stress": 0.28, "time_remaining": 0.82},
        },
        {
            "phase": "ambiguity_probe",
            "evidence": {"evidence_strength": 0.30, "uncertainty": 0.80, "expected_error": 0.32, "imagined_risk": 0.24},
            "resource": {"energy": 0.78, "budget": 0.70, "stress": 0.24, "time_remaining": 0.74},
        },
        {
            "phase": "ambiguity_probe",
            "evidence": {"evidence_strength": 0.20, "uncertainty": 0.92, "expected_error": 0.40, "imagined_risk": 0.30},
            "resource": {"energy": 0.72, "budget": 0.84, "stress": 0.22, "time_remaining": 0.88},
        },
    ],
    "commit_window": [
        {
            "phase": "commit_window",
            "evidence": {"evidence_strength": 0.88, "uncertainty": 0.16, "expected_error": 0.12, "imagined_risk": 0.08},
            "resource": {"energy": 0.76, "budget": 0.72, "stress": 0.20, "time_remaining": 0.76},
        },
        {
            "phase": "commit_window",
            "evidence": {"evidence_strength": 0.82, "uncertainty": 0.22, "expected_error": 0.16, "imagined_risk": 0.10},
            "resource": {"energy": 0.70, "budget": 0.68, "stress": 0.24, "time_remaining": 0.70},
        },
        {
            "phase": "commit_window",
            "evidence": {"evidence_strength": 0.92, "uncertainty": 0.12, "expected_error": 0.10, "imagined_risk": 0.06},
            "resource": {"energy": 0.80, "budget": 0.74, "stress": 0.18, "time_remaining": 0.78},
        },
    ],
    "error_hazard": [
        {
            "phase": "error_hazard",
            "evidence": {"evidence_strength": 0.34, "uncertainty": 0.40, "expected_error": 0.86, "imagined_risk": 0.22},
            "resource": {"energy": 0.62, "budget": 0.58, "stress": 0.54, "time_remaining": 0.54},
        },
        {
            "phase": "error_hazard",
            "evidence": {"evidence_strength": 0.38, "uncertainty": 0.34, "expected_error": 0.80, "imagined_risk": 0.18},
            "resource": {"energy": 0.58, "budget": 0.62, "stress": 0.58, "time_remaining": 0.50},
        },
        {
            "phase": "error_hazard",
            "evidence": {"evidence_strength": 0.28, "uncertainty": 0.46, "expected_error": 0.90, "imagined_risk": 0.25},
            "resource": {"energy": 0.60, "budget": 0.56, "stress": 0.60, "time_remaining": 0.48},
        },
    ],
    "pressure_spike": [
        {
            "phase": "pressure_spike",
            "evidence": {"evidence_strength": 0.34, "uncertainty": 0.48, "expected_error": 0.38, "imagined_risk": 0.34},
            "resource": {"energy": 0.28, "budget": 0.26, "stress": 0.82, "time_remaining": 0.24},
        },
        {
            "phase": "pressure_spike",
            "evidence": {"evidence_strength": 0.30, "uncertainty": 0.56, "expected_error": 0.42, "imagined_risk": 0.30},
            "resource": {"energy": 0.22, "budget": 0.24, "stress": 0.86, "time_remaining": 0.20},
        },
        {
            "phase": "pressure_spike",
            "evidence": {"evidence_strength": 0.42, "uncertainty": 0.44, "expected_error": 0.34, "imagined_risk": 0.26},
            "resource": {"energy": 0.30, "budget": 0.22, "stress": 0.78, "time_remaining": 0.22},
        },
    ],
    "counterfactual_conflict": [
        {
            "phase": "counterfactual_conflict",
            "evidence": {"evidence_strength": 0.68, "uncertainty": 0.22, "expected_error": 0.18, "imagined_risk": 0.86},
            "resource": {"energy": 0.56, "budget": 0.62, "stress": 0.42, "time_remaining": 0.48},
        },
        {
            "phase": "counterfactual_conflict",
            "evidence": {"evidence_strength": 0.72, "uncertainty": 0.18, "expected_error": 0.14, "imagined_risk": 0.92},
            "resource": {"energy": 0.58, "budget": 0.64, "stress": 0.36, "time_remaining": 0.52},
        },
        {
            "phase": "counterfactual_conflict",
            "evidence": {"evidence_strength": 0.64, "uncertainty": 0.26, "expected_error": 0.20, "imagined_risk": 0.84},
            "resource": {"energy": 0.54, "budget": 0.58, "stress": 0.44, "time_remaining": 0.46},
        },
    ],
    "recovery_probe": [
        {
            "phase": "recovery_probe",
            "evidence": {"evidence_strength": 0.44, "uncertainty": 0.30, "expected_error": 0.48, "imagined_risk": 0.20},
            "resource": {"energy": 0.34, "budget": 0.32, "stress": 0.66, "time_remaining": 0.28},
        },
        {
            "phase": "recovery_probe",
            "evidence": {"evidence_strength": 0.50, "uncertainty": 0.28, "expected_error": 0.42, "imagined_risk": 0.18},
            "resource": {"energy": 0.36, "budget": 0.30, "stress": 0.62, "time_remaining": 0.26},
        },
        {
            "phase": "recovery_probe",
            "evidence": {"evidence_strength": 0.40, "uncertainty": 0.34, "expected_error": 0.52, "imagined_risk": 0.24},
            "resource": {"energy": 0.30, "budget": 0.28, "stress": 0.70, "time_remaining": 0.24},
        },
    ],
}


def _pressure_value(resource_state: dict[str, float] | ResourceSnapshot) -> float:
    snapshot = resource_state if isinstance(resource_state, ResourceSnapshot) else ResourceSnapshot.from_dict(resource_state)
    scarcity = 1.0 - ((snapshot.energy + snapshot.budget + snapshot.time_remaining) / 3.0)
    return _clamp01((scarcity + snapshot.stress) / 2.0)


def _jitter(value: float, rng: random.Random, width: float) -> float:
    return _clamp01(value + rng.uniform(-width, width))


def _sample_trial_sequence(*, seed: int, stress: bool, episodes_per_family: int = 3) -> list[tuple[dict[str, Any], dict[str, float], ResourceSnapshot]]:
    rng = random.Random(seed * 177 + (19 if stress else 0))
    sampled: list[tuple[dict[str, Any], dict[str, float], ResourceSnapshot]] = []
    for family_name, variants in SCENARIO_LIBRARY.items():
        picks = [variants[index % len(variants)] for index in range(episodes_per_family)]
        rng.shuffle(picks)
        for sample_index, template in enumerate(picks, start=1):
            evidence = {
                key: _jitter(float(value), rng, 0.04 if key != "imagined_risk" else 0.06)
                for key, value in template["evidence"].items()
            }
            resource = {
                key: _jitter(float(value), rng, 0.05)
                for key, value in template["resource"].items()
            }
            if stress and family_name in {"pressure_spike", "recovery_probe"}:
                resource["energy"] = _clamp01(resource["energy"] - 0.08)
                resource["budget"] = _clamp01(resource["budget"] - 0.06)
                resource["stress"] = _clamp01(resource["stress"] + 0.08)
                resource["time_remaining"] = _clamp01(resource["time_remaining"] - 0.06)
            sampled.append(
                (
                    {
                        "phase": family_name,
                        "variant": template["phase"],
                        "family_index": sample_index,
                        "stress_mode": stress,
                    },
                    {key: round(value, 6) for key, value in evidence.items()},
                    ResourceSnapshot(**resource),
                )
            )
    rng.shuffle(sampled)
    return sampled


def reconstruct_behavior_patterns(records: list[DecisionLogRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record) for record in records]
    patterns: list[dict[str, Any]] = []
    if any(record.selected_action in {"scan", "inspect", "query"} and record.observation_evidence.get("uncertainty", 0.0) >= 0.75 for record in normalized):
        patterns.append({"label": "directed_exploration", "evidence": "inspect-like action selected during high uncertainty"})
    if any(
        record.selected_action in {"rest", "recover", "conserve"}
        and ResourceSnapshot.from_dict(record.resource_state).energy <= 0.38
        and ResourceSnapshot.from_dict(record.resource_state).budget <= 0.35
        for record in normalized
    ):
        patterns.append({"label": "resource_conservation", "evidence": "recovery action selected under elevated resource pressure"})
    if any(record.selected_action == "commit" and record.internal_confidence >= 0.60 and record.update_magnitude <= 0.22 for record in normalized):
        patterns.append({"label": "confidence_sharpening", "evidence": "commit selected with high confidence and bounded update"})
    if any(
        record.prediction_error_vector.get("virtual_error", 0.0) > record.prediction_error_vector.get("direct_error", 0.0)
        and record.selected_action != "commit"
        for record in normalized
    ):
        patterns.append({"label": "counterfactual_avoidance", "evidence": "imagined-loss conflict shifted choice away from commit"})
    return patterns


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _proportion(records: list[DecisionLogRecord], predicate: Callable[[DecisionLogRecord], bool]) -> float | None:
    if not records:
        return None
    return sum(1.0 for record in records if predicate(record)) / len(records)


def _metric_result(*, value: float | None, sample_size: int, min_samples: int) -> dict[str, Any]:
    return {
        "value": _round_float(value) if sample_size >= min_samples and value is not None else None,
        "sample_size": sample_size,
        "min_samples": min_samples,
        "insufficient_data": sample_size < min_samples or value is None,
    }


def _metric_evaluator(metric_name: str) -> Callable[[list[DecisionLogRecord | dict[str, Any]]], dict[str, Any]]:
    def _evaluate(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, Any]:
        return compute_observable_metrics(records)[metric_name]

    _evaluate.__name__ = f"evaluate_{metric_name}"
    return _evaluate


def observable_metrics_registry() -> dict[str, dict[str, Any]]:
    return {
        "uncertainty_confidence_drop_rate": {
            "metric_id": "uncertainty_confidence_drop_rate",
            "parameter": "uncertainty_sensitivity",
            "description": "High uncertainty lowers internal confidence.",
            "formula": "mean(uncertainty * (1 - internal_confidence) | uncertainty >= 0.6)",
            "depends_on": ["observation_evidence.uncertainty", "internal_confidence"],
            "evaluator": _metric_evaluator("uncertainty_confidence_drop_rate"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "high_uncertainty_inspect_ratio": {
            "metric_id": "high_uncertainty_inspect_ratio",
            "parameter": "uncertainty_sensitivity",
            "description": "Ambiguous contexts increase the score advantage of inspect-like candidates over decisive ones.",
            "formula": "mean(logistic(2 * (best_inspect_score - best_decisive_score)) | uncertainty >= 0.6)",
            "depends_on": ["observation_evidence.uncertainty", "candidate_actions"],
            "evaluator": _metric_evaluator("high_uncertainty_inspect_ratio"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "high_expected_error_rejection_rate": {
            "metric_id": "high_expected_error_rejection_rate",
            "parameter": "error_aversion",
            "description": "High expected error increases the score gap between protective and risky actions.",
            "formula": "mean(logistic(2 * (best_protective_score - best_risky_score)) | expected_error >= 0.65)",
            "depends_on": ["observation_evidence.expected_error", "candidate_actions"],
            "evaluator": _metric_evaluator("high_expected_error_rejection_rate"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "post_error_conservative_shift": {
            "metric_id": "post_error_conservative_shift",
            "parameter": "error_aversion",
            "description": "Error signals trigger protective recovery choices.",
            "formula": "P(selected_action in {rest,recover,conserve} | prediction_error >= 0.45)",
            "depends_on": ["prediction_error", "selected_action"],
            "evaluator": _metric_evaluator("post_error_conservative_shift"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "novel_action_ratio": {
            "metric_id": "novel_action_ratio",
            "parameter": "exploration_bias",
            "description": "Information-seeking actions are favored in uncertainty.",
            "formula": "P(selected_action in {scan,inspect,query} | uncertainty >= 0.5)",
            "depends_on": ["observation_evidence.uncertainty", "selected_action"],
            "evaluator": _metric_evaluator("novel_action_ratio"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 4,
        },
        "choice_repeat_suppression": {
            "metric_id": "choice_repeat_suppression",
            "parameter": "exploration_bias",
            "description": "In medium uncertainty, inspect-like candidates maintain a larger score advantage over direct commit as exploration rises.",
            "formula": "mean(logistic(2 * (best_inspect_score - commit_score)) | uncertainty >= 0.5)",
            "depends_on": ["observation_evidence.uncertainty", "candidate_actions"],
            "evaluator": _metric_evaluator("choice_repeat_suppression"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 4,
        },
        "dominant_attention_share": {
            "metric_id": "dominant_attention_share",
            "parameter": "attention_selectivity",
            "description": "Attention concentrates on the strongest evidence feature.",
            "formula": "mean(max(attention_allocation.values()))",
            "depends_on": ["attention_allocation"],
            "evaluator": _metric_evaluator("dominant_attention_share"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 6,
        },
        "evidence_aligned_choice_rate": {
            "metric_id": "evidence_aligned_choice_rate",
            "parameter": "attention_selectivity",
            "description": "Selective attention creates a larger concentration gap between dominant and secondary channels.",
            "formula": "mean(1 - normalized_entropy(attention_allocation))",
            "depends_on": ["attention_allocation"],
            "evaluator": _metric_evaluator("evidence_aligned_choice_rate"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 6,
        },
        "confidence_evidence_slope": {
            "metric_id": "confidence_evidence_slope",
            "parameter": "confidence_gain",
            "description": "Confidence rises with stronger evidence separation.",
            "formula": "mean(evidence_strength * internal_confidence)",
            "depends_on": ["observation_evidence.evidence_strength", "internal_confidence"],
            "evaluator": _metric_evaluator("confidence_evidence_slope"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 6,
        },
        "high_evidence_commit_rate": {
            "metric_id": "high_evidence_commit_rate",
            "parameter": "confidence_gain",
            "description": "Commit wins more often once evidence becomes strong.",
            "formula": "P(selected_action == commit | evidence_strength >= 0.7)",
            "depends_on": ["observation_evidence.evidence_strength", "selected_action"],
            "evaluator": _metric_evaluator("high_evidence_commit_rate"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "mean_update_inverse": {
            "metric_id": "mean_update_inverse",
            "parameter": "update_rigidity",
            "description": "Prediction errors yield smaller internal updates.",
            "formula": "1 - mean(model_update.magnitude)",
            "depends_on": ["model_update.magnitude"],
            "evaluator": _metric_evaluator("mean_update_inverse"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 6,
        },
        "strategy_persistence_after_error": {
            "metric_id": "strategy_persistence_after_error",
            "parameter": "update_rigidity",
            "description": "After error, action policy changes more slowly.",
            "formula": "mean(1 - model_update.strategy_shift | prediction_error >= 0.45)",
            "depends_on": ["model_update.strategy_shift", "prediction_error"],
            "evaluator": _metric_evaluator("strategy_persistence_after_error"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "high_pressure_low_cost_ratio": {
            "metric_id": "high_pressure_low_cost_ratio",
            "parameter": "resource_pressure_sensitivity",
            "description": "Low-cost actions dominate under resource pressure.",
            "formula": "P(selected_action in {rest,conserve,recover,scan} | pressure >= 0.6)",
            "depends_on": ["resource_state", "selected_action"],
            "evaluator": _metric_evaluator("high_pressure_low_cost_ratio"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "recovery_trigger_rate": {
            "metric_id": "recovery_trigger_rate",
            "parameter": "resource_pressure_sensitivity",
            "description": "Recovery triggers when energy and time are low.",
            "formula": "P(selected_action in {rest,recover,conserve} | energy <= 0.35 or time_remaining <= 0.3)",
            "depends_on": ["resource_state", "selected_action"],
            "evaluator": _metric_evaluator("recovery_trigger_rate"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "conflict_avoidance_shift": {
            "metric_id": "conflict_avoidance_shift",
            "parameter": "virtual_prediction_error_gain",
            "description": "Imagined loss signals bias decisions away from direct commit.",
            "formula": "P(selected_action != commit | virtual_error > direct_error)",
            "depends_on": ["prediction_error_vector.virtual_error", "prediction_error_vector.direct_error", "selected_action"],
            "evaluator": _metric_evaluator("conflict_avoidance_shift"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
        "counterfactual_loss_sensitivity": {
            "metric_id": "counterfactual_loss_sensitivity",
            "parameter": "virtual_prediction_error_gain",
            "description": "Counterfactual loss increases conservative choices.",
            "formula": "mean(max(virtual_error - direct_error, 0) * conservative_choice)",
            "depends_on": ["prediction_error_vector", "selected_action"],
            "evaluator": _metric_evaluator("counterfactual_loss_sensitivity"),
            "direction": "higher_means_higher_parameter",
            "min_samples": 3,
        },
    }


def default_behavior_mapping_table() -> dict[str, dict[str, Any]]:
    registry = observable_metrics_registry()
    return {
        metric_name: {
            "primary_parameter": spec["parameter"],
            "observable": spec["description"],
            "formula": spec["formula"],
            "evaluator": getattr(spec["evaluator"], "__name__", str(spec["evaluator"])),
            "min_samples": spec["min_samples"],
            "depends_on": spec["depends_on"],
        }
        for metric_name, spec in registry.items()
    }


def observable_parameter_contracts() -> dict[str, dict[str, Any]]:
    registry = observable_metrics_registry()
    contracts: dict[str, dict[str, Any]] = {}
    for parameter_name in PARAMETER_REFERENCE:
        metrics = [
            {
                "metric": metric_name,
                "description": spec["description"],
                "formula": spec["formula"],
                "depends_on": spec["depends_on"],
                "direction": spec["direction"],
                "evaluator": spec["evaluator"],
                "min_samples": spec["min_samples"],
            }
            for metric_name, spec in registry.items()
            if spec["parameter"] == parameter_name
        ]
        contracts[parameter_name] = {
            "physical_meaning": PARAMETER_REFERENCE[parameter_name]["physical_meaning"],
            "observables": metrics,
        }
    return contracts


def _candidate_action_score(record: DecisionLogRecord, action_names: set[str]) -> float | None:
    matching_scores = [
        float(candidate.get("total_score", 0.0))
        for candidate in record.candidate_actions
        if isinstance(candidate, dict)
        and isinstance(candidate.get("action"), dict)
        and str(candidate["action"].get("name", "")) in action_names
    ]
    return max(matching_scores) if matching_scores else None


def _normalized_attention_entropy(attention_allocation: dict[str, float]) -> float | None:
    values = [float(value) for value in attention_allocation.values() if float(value) > 0.0]
    if len(values) <= 1:
        return 0.0 if values else None
    entropy = -sum(value * math.log(value, 2) for value in values)
    return entropy / math.log(len(values), 2)


def compute_observable_metrics(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = [record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record) for record in records]

    inspect_actions = {"scan", "inspect", "query"}
    protective_actions = {"rest", "conserve", "recover"}
    conservative_actions = protective_actions | {"scan", "inspect", "query", "plan"}
    low_cost_actions = {"rest", "conserve", "recover", "scan"}
    risky_actions = {"commit", "guess", "retry"}

    high_uncertainty = [record for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.60]
    medium_uncertainty = [record for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.50]
    high_error = [record for record in normalized if record.observation_evidence.get("expected_error", 0.0) >= 0.65]
    high_pred_error = [record for record in normalized if record.prediction_error >= 0.45]
    high_evidence = [record for record in normalized if record.observation_evidence.get("evidence_strength", 0.0) >= 0.70]
    high_pressure = [record for record in normalized if _pressure_value(record.resource_state) >= 0.60]
    low_resource = [
        record
        for record in normalized
        if record.resource_state.get("energy", 0.0) <= 0.35 or record.resource_state.get("time_remaining", 0.0) <= 0.30
    ]
    conflict_cases = [
        record
        for record in normalized
        if record.prediction_error_vector.get("virtual_error", 0.0) > record.prediction_error_vector.get("direct_error", 0.0)
    ]
    inspect_margin_cases = [
        record
        for record in high_uncertainty
        if _candidate_action_score(record, inspect_actions) is not None and _candidate_action_score(record, risky_actions) is not None
    ]
    expected_error_margin_cases = [
        record
        for record in high_error
        if _candidate_action_score(record, conservative_actions) is not None and _candidate_action_score(record, risky_actions) is not None
    ]
    attention_entropy_values = [
        1.0 - entropy
        for record in normalized
        for entropy in [_normalized_attention_entropy(record.attention_allocation)]
        if entropy is not None
    ]
    medium_uncertainty_margin_cases = [
        record
        for record in medium_uncertainty
        if _candidate_action_score(record, inspect_actions) is not None and _candidate_action_score(record, {"commit"}) is not None
    ]

    results = {
        "uncertainty_confidence_drop_rate": _metric_result(
            value=_mean(
                [
                    record.observation_evidence.get("uncertainty", 0.0) * (1.0 - record.internal_confidence)
                    for record in high_uncertainty
                ]
            ),
            sample_size=len(high_uncertainty),
            min_samples=3,
        ),
        "high_uncertainty_inspect_ratio": _metric_result(
            value=_mean(
                [
                    logistic(
                        2.0
                        * (
                            _candidate_action_score(record, inspect_actions)
                            - _candidate_action_score(record, risky_actions)
                        )
                    )
                    for record in inspect_margin_cases
                ]
            ),
            sample_size=len(inspect_margin_cases),
            min_samples=3,
        ),
        "high_expected_error_rejection_rate": _metric_result(
            value=_mean(
                [
                    logistic(
                        2.0
                        * (
                            _candidate_action_score(record, conservative_actions)
                            - _candidate_action_score(record, risky_actions)
                        )
                    )
                    for record in expected_error_margin_cases
                ]
            ),
            sample_size=len(expected_error_margin_cases),
            min_samples=3,
        ),
        "post_error_conservative_shift": _metric_result(
            value=_proportion(high_pred_error, lambda record: record.selected_action in protective_actions),
            sample_size=len(high_pred_error),
            min_samples=3,
        ),
        "novel_action_ratio": _metric_result(
            value=_proportion(medium_uncertainty, lambda record: record.selected_action in inspect_actions),
            sample_size=len(medium_uncertainty),
            min_samples=4,
        ),
        "choice_repeat_suppression": _metric_result(
            value=_mean(
                [
                    logistic(
                        2.0
                        * (
                            _candidate_action_score(record, inspect_actions)
                            - _candidate_action_score(record, {"commit"})
                        )
                    )
                    for record in medium_uncertainty_margin_cases
                ]
            ),
            sample_size=len(medium_uncertainty_margin_cases),
            min_samples=4,
        ),
        "dominant_attention_share": _metric_result(
            value=_mean([max(record.attention_allocation.values()) for record in normalized if record.attention_allocation]),
            sample_size=len(normalized),
            min_samples=6,
        ),
        "evidence_aligned_choice_rate": _metric_result(
            value=_mean(attention_entropy_values),
            sample_size=len(attention_entropy_values),
            min_samples=6,
        ),
        "confidence_evidence_slope": _metric_result(
            value=_mean(
                [record.observation_evidence.get("evidence_strength", 0.0) * record.internal_confidence for record in normalized]
            ),
            sample_size=len(normalized),
            min_samples=6,
        ),
        "high_evidence_commit_rate": _metric_result(
            value=_proportion(high_evidence, lambda record: record.selected_action == "commit"),
            sample_size=len(high_evidence),
            min_samples=3,
        ),
        "mean_update_inverse": _metric_result(
            value=(1.0 - _mean([record.model_update.get("magnitude", 0.0) for record in normalized])) if normalized else None,
            sample_size=len(normalized),
            min_samples=6,
        ),
        "strategy_persistence_after_error": _metric_result(
            value=_mean([1.0 - record.model_update.get("strategy_shift", 0.0) for record in high_pred_error]),
            sample_size=len(high_pred_error),
            min_samples=3,
        ),
        "high_pressure_low_cost_ratio": _metric_result(
            value=_proportion(high_pressure, lambda record: record.selected_action in low_cost_actions),
            sample_size=len(high_pressure),
            min_samples=3,
        ),
        "recovery_trigger_rate": _metric_result(
            value=_proportion(low_resource, lambda record: record.selected_action in {"rest", "recover", "conserve"}),
            sample_size=len(low_resource),
            min_samples=3,
        ),
        "conflict_avoidance_shift": _metric_result(
            value=_proportion(conflict_cases, lambda record: record.selected_action != "commit"),
            sample_size=len(conflict_cases),
            min_samples=3,
        ),
        "counterfactual_loss_sensitivity": _metric_result(
            value=_mean(
                [
                    max(
                        0.0,
                        record.prediction_error_vector.get("virtual_error", 0.0)
                        - record.prediction_error_vector.get("direct_error", 0.0),
                    )
                    * (1.0 if record.selected_action in conservative_actions else 0.0)
                    for record in conflict_cases
                ]
            ),
            sample_size=len(conflict_cases),
            min_samples=3,
        ),
    }
    return results


def metric_values_from_payload(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric_name, payload in metrics.items():
        if isinstance(payload, dict) and not payload.get("insufficient_data", False) and payload.get("value") is not None:
            values[metric_name] = float(payload["value"])
        elif isinstance(payload, (int, float)):
            values[metric_name] = float(payload)
    return values


def audit_observable_contracts(*, seeds: list[int] | None = None) -> dict[str, Any]:
    active_seeds = seeds or [41, 42, 43]
    registry = observable_metrics_registry()
    baseline = CognitiveStyleParameters()
    baseline_trial = run_cognitive_style_trial(baseline, seed=active_seeds[0])
    baseline_metrics = baseline_trial["observable_metrics"]
    stress_parameters = {
        parameter_name
        for parameter_name, probe in parameter_probe_registry().items()
        if probe.get("stress", False)
    }

    per_metric: dict[str, dict[str, Any]] = {}
    per_parameter: dict[str, dict[str, Any]] = {
        parameter_name: {
            "observable_count": 0,
            "informative_observables": 0,
            "direction_mismatches": [],
            "uninformative_metrics": [],
        }
        for parameter_name in PARAMETER_REFERENCE
    }
    direction_mismatch_count = 0
    uninformative_metric_count = 0

    for metric_name, spec in registry.items():
        parameter_name = str(spec["parameter"])
        stress = parameter_name in stress_parameters
        low = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.0})
        high = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 1.0})
        low_value, _ = _mean_trial_metric(low, metric_name, seeds=active_seeds, stress=stress)
        high_value, _ = _mean_trial_metric(high, metric_name, seeds=active_seeds, stress=stress)
        delta = None if low_value is None or high_value is None else high_value - low_value
        direction_matches = bool(
            delta is not None
            and (
                (spec["direction"] == "higher_means_higher_parameter" and delta > 0.0)
                or (spec["direction"] == "lower_means_higher_parameter" and delta < 0.0)
            )
        )
        informative = bool(delta is not None and abs(delta) >= 0.05)
        baseline_payload = baseline_metrics.get(metric_name, {})
        entry = {
            "metric": metric_name,
            "parameter": parameter_name,
            "baseline_value": _round_float(baseline_payload.get("value")),
            "baseline_insufficient_data": bool(baseline_payload.get("insufficient_data", False)),
            "low_value": _round_float(low_value),
            "high_value": _round_float(high_value),
            "delta": _round_float(delta),
            "direction": spec["direction"],
            "direction_matches": direction_matches,
            "informative": informative,
            "stress_mode": stress,
        }
        per_metric[metric_name] = entry
        per_parameter[parameter_name]["observable_count"] += 1
        if informative:
            per_parameter[parameter_name]["informative_observables"] += 1
        else:
            per_parameter[parameter_name]["uninformative_metrics"].append(metric_name)
            uninformative_metric_count += 1
        if not direction_matches:
            per_parameter[parameter_name]["direction_mismatches"].append(metric_name)
            direction_mismatch_count += 1

    return {
        "metric_count": len(registry),
        "registry_executable": metrics_have_executable_registry(registry, sample_records=baseline_trial["logs"]),
        "direction_mismatch_count": direction_mismatch_count,
        "uninformative_metric_count": uninformative_metric_count,
        "per_metric": per_metric,
        "per_parameter": per_parameter,
    }


def metrics_have_executable_registry(
    registry: dict[str, dict[str, Any]] | None = None,
    sample_records: list[DecisionLogRecord | dict[str, Any]] | None = None,
) -> bool:
    active_registry = registry or observable_metrics_registry()
    records = sample_records
    if records is None:
        records = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)["logs"]
    required_keys = {"metric_id", "parameter", "depends_on", "evaluator", "formula", "direction", "min_samples"}
    for metric_name, spec in active_registry.items():
        if not required_keys <= set(spec.keys()):
            return False
        evaluator = spec["evaluator"]
        if not callable(evaluator) or spec["min_samples"] <= 0:
            return False
        try:
            payload = evaluator(records)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if {"value", "sample_size", "min_samples", "insufficient_data"} - set(payload):
            return False
        if int(payload["min_samples"]) != int(spec["min_samples"]):
            return False
        if metric_name != spec["metric_id"]:
            return False
    return True


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
            {
                **parameters.to_dict(),
                "resource_pressure_sensitivity": 0.0,
                "schema_version": CognitiveStyleParameters().schema_version,
            }
        )
    bridge = CognitiveParameterBridge(active_parameters)
    sequence = _sample_trial_sequence(seed=seed, stress=stress, episodes_per_family=3)
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
    observable_metrics = compute_observable_metrics(logs)
    metric_values = metric_values_from_payload(observable_metrics)
    return {
        "parameters": active_parameters.to_dict(),
        "logs": [record.to_dict() for record in logs],
        "patterns": patterns,
        "observable_metrics": observable_metrics,
        "observable_metric_values": metric_values,
        "summary": {
            "selected_actions": [record.selected_action for record in logs],
            "mean_confidence": round(sum(record.internal_confidence for record in logs) / len(logs), 6),
            "mean_update_magnitude": round(sum(record.update_magnitude for record in logs) / len(logs), 6),
            "pattern_count": len(patterns),
            "unique_action_count": len(set(record.selected_action for record in logs)),
            "scenario_families": sorted({record.task_context.get("phase", "") for record in logs}),
            "observable_metrics_with_data": sorted(metric_values.keys()),
        },
    }


def compute_trial_variation(
    reference_trial: dict[str, Any],
    comparison_trial: dict[str, Any],
) -> dict[str, Any]:
    reference_actions = reference_trial["summary"]["selected_actions"]
    comparison_actions = comparison_trial["summary"]["selected_actions"]
    differing_positions = sum(
        1
        for left, right in zip(reference_actions, comparison_actions)
        if left != right
    )
    reference_metrics = reference_trial.get("observable_metric_values", {})
    comparison_metrics = comparison_trial.get("observable_metric_values", {})
    shared = sorted(set(reference_metrics) & set(comparison_metrics))
    metric_delta = round(
        sum(abs(float(reference_metrics[name]) - float(comparison_metrics[name])) for name in shared) / len(shared),
        6,
    ) if shared else 0.0
    return {
        "differing_positions": differing_positions,
        "action_overlap_ratio": round(
            sum(1 for left, right in zip(reference_actions, comparison_actions) if left == right) / max(1, len(reference_actions)),
            6,
        ),
        "mean_metric_delta": metric_delta,
        "varies": differing_positions > 0 and metric_delta >= 0.01,
    }


def _mean_trial_metric(
    parameters: CognitiveStyleParameters,
    metric_name: str,
    *,
    seeds: list[int],
    stress: bool = False,
) -> tuple[float | None, list[str]]:
    values: list[float] = []
    action_digest: list[str] = []
    for seed in seeds:
        trial = run_cognitive_style_trial(parameters, seed=seed, stress=stress)
        metric_payload = trial["observable_metrics"][metric_name]
        if not metric_payload["insufficient_data"] and metric_payload["value"] is not None:
            values.append(float(metric_payload["value"]))
        action_digest.append(",".join(trial["summary"]["selected_actions"][:6]))
    return (_mean(values), action_digest)


def parameter_probe_registry() -> dict[str, dict[str, Any]]:
    return {
        "uncertainty_sensitivity": {"metric": "uncertainty_confidence_drop_rate", "expectation": "higher", "min_effect": 0.10},
        "error_aversion": {"metric": "high_expected_error_rejection_rate", "expectation": "higher", "min_effect": 0.10},
        "exploration_bias": {"metric": "novel_action_ratio", "expectation": "higher", "min_effect": 0.08},
        "attention_selectivity": {"metric": "dominant_attention_share", "expectation": "higher", "min_effect": 0.05},
        "confidence_gain": {"metric": "high_evidence_commit_rate", "expectation": "higher", "min_effect": 0.08},
        "update_rigidity": {"metric": "mean_update_inverse", "expectation": "higher", "min_effect": 0.08},
        "resource_pressure_sensitivity": {"metric": "recovery_trigger_rate", "expectation": "higher", "min_effect": 0.10, "stress": True},
        "virtual_prediction_error_gain": {"metric": "counterfactual_loss_sensitivity", "expectation": "higher", "min_effect": 0.10},
    }


def parameter_intervention_sensitivity_matrix(*, seeds: list[int] | None = None) -> dict[str, dict[str, Any]]:
    active_seeds = seeds or [41, 42, 43]
    baseline = CognitiveStyleParameters()
    matrix: dict[str, dict[str, Any]] = {}
    for parameter_name, probe in parameter_probe_registry().items():
        low = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.0})
        high = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 1.0})
        stress = bool(probe.get("stress", False))
        low_value, low_actions = _mean_trial_metric(low, probe["metric"], seeds=active_seeds, stress=stress)
        high_value, high_actions = _mean_trial_metric(high, probe["metric"], seeds=active_seeds, stress=stress)
        delta = None if low_value is None or high_value is None else high_value - low_value
        identifiable = bool(
            delta is not None
            and (
                (probe["expectation"] == "higher" and delta >= probe["min_effect"])
                or (probe["expectation"] == "lower" and delta <= -probe["min_effect"])
            )
        )
        matrix[parameter_name] = {
            "parameter": parameter_name,
            "analysis_type": "intervention_sensitivity",
            "target_metric": probe["metric"],
            "expectation": probe["expectation"],
            "minimum_effect": probe["min_effect"],
            "low_parameter_value": 0.0,
            "high_parameter_value": 1.0,
            "low_observed_value": _round_float(low_value),
            "high_observed_value": _round_float(high_value),
            "delta": _round_float(delta),
            "identifiable": identifiable,
            "insufficient_data": low_value is None or high_value is None,
            "seeds": list(active_seeds),
            "stress_mode": stress,
            "low_selected_actions": low_actions,
            "high_selected_actions": high_actions,
        }
    return matrix


def parameter_causality_matrix(*, seeds: list[int] | None = None) -> dict[str, dict[str, Any]]:
    return parameter_intervention_sensitivity_matrix(seeds=seeds)


def parameter_identifiability_probe(*, seeds: list[int] | None = None) -> dict[str, Any]:
    matrix = parameter_intervention_sensitivity_matrix(seeds=seeds)
    baseline = run_cognitive_style_trial(CognitiveStyleParameters(), seed=(seeds or [41])[0])
    return {
        "analysis_type": "intervention_sensitivity",
        "contracts": observable_parameter_contracts(),
        "identifiable": {name: payload["identifiable"] for name, payload in matrix.items()},
        "baseline": baseline["summary"],
        "probes": matrix,
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_unit_interval(value: Any) -> bool:
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


def _payload_from_record(record: DecisionLogRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, DecisionLogRecord):
        return record.to_dict()
    return dict(record)


def _timestamp_is_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def audit_decision_log(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, Any]:
    required_fields = DecisionLogRecord.schema()["required"]
    missing_counts = {field_name: 0 for field_name in required_fields}
    invalid_value_counts = {field_name: 0 for field_name in required_fields}
    semantic_invalid_counts = {
        "parameter_snapshot_incomplete": 0,
        "parameter_snapshot_out_of_range": 0,
        "selected_action_unknown": 0,
        "selected_action_not_in_candidates": 0,
        "attention_allocation_not_normalized": 0,
        "prediction_error_mismatch": 0,
        "update_magnitude_mismatch": 0,
    }
    parameter_snapshot_complete_records = 0
    valid_records = 0
    valid_action_names = {action.name for action in canonical_action_schemas()}
    parameter_fields = CognitiveStyleParameters.schema()["required"]
    for record in records:
        payload = _payload_from_record(record)
        invalid = False
        for field_name in required_fields:
            value = payload.get(field_name)
            if value in (None, "", [], {}):
                missing_counts[field_name] += 1
                invalid = True

        if not _timestamp_is_valid(payload.get("timestamp")):
            invalid_value_counts["timestamp"] += 1
            invalid = True
        if not isinstance(payload.get("tick"), int) or int(payload["tick"]) <= 0:
            invalid_value_counts["tick"] += 1
            invalid = True
        if not isinstance(payload.get("seed"), int):
            invalid_value_counts["seed"] += 1
            invalid = True
        if not _is_unit_interval(payload.get("internal_confidence")):
            invalid_value_counts["internal_confidence"] += 1
            invalid = True
        if not _is_unit_interval(payload.get("prediction_error")):
            invalid_value_counts["prediction_error"] += 1
            invalid = True
        if not _is_unit_interval(payload.get("update_magnitude")):
            invalid_value_counts["update_magnitude"] += 1
            invalid = True

        snapshot = payload.get("parameter_snapshot", {})
        if isinstance(snapshot, dict) and all(field in snapshot for field in parameter_fields):
            parameter_snapshot_complete_records += 1
        else:
            semantic_invalid_counts["parameter_snapshot_incomplete"] += 1
            invalid = True
        if not isinstance(snapshot, dict) or any(not _is_unit_interval(snapshot.get(field_name)) for field_name in PARAMETER_REFERENCE):
            invalid_value_counts["parameter_snapshot"] += 1
            semantic_invalid_counts["parameter_snapshot_out_of_range"] += 1
            invalid = True

        resource_state = payload.get("resource_state")
        if not isinstance(resource_state, dict) or any(not _is_unit_interval(resource_state.get(key)) for key in ("energy", "budget", "stress", "time_remaining")):
            invalid_value_counts["resource_state"] += 1
            invalid = True

        observation_evidence = payload.get("observation_evidence")
        if not isinstance(observation_evidence, dict) or not observation_evidence or any(not _is_unit_interval(value) for value in observation_evidence.values()):
            invalid_value_counts["observation_evidence"] += 1
            invalid = True

        prediction_error_vector = payload.get("prediction_error_vector")
        if (
            not isinstance(prediction_error_vector, dict)
            or any(key not in prediction_error_vector for key in ("direct_error", "virtual_error", "signed_total"))
            or any(not _is_unit_interval(prediction_error_vector.get(key)) for key in ("direct_error", "virtual_error", "signed_total"))
        ):
            invalid_value_counts["prediction_error_vector"] += 1
            invalid = True
        elif abs(float(prediction_error_vector["signed_total"]) - float(payload["prediction_error"])) > 1e-6:
            semantic_invalid_counts["prediction_error_mismatch"] += 1
            invalid = True

        attention_allocation = payload.get("attention_allocation")
        if not isinstance(attention_allocation, dict) or not attention_allocation or any(not _is_unit_interval(value) for value in attention_allocation.values()):
            invalid_value_counts["attention_allocation"] += 1
            invalid = True
        else:
            attention_total = sum(float(value) for value in attention_allocation.values())
            if abs(attention_total - 1.0) > 1e-3:
                semantic_invalid_counts["attention_allocation_not_normalized"] += 1
                invalid = True

        candidate_actions = payload.get("candidate_actions")
        candidate_action_names: set[str] = set()
        if not isinstance(candidate_actions, list) or not candidate_actions:
            invalid_value_counts["candidate_actions"] += 1
            invalid = True
        else:
            candidate_actions_valid = True
            for candidate in candidate_actions:
                if not isinstance(candidate, dict):
                    candidate_actions_valid = False
                    break
                action_payload = candidate.get("action")
                action_name_value = action_payload.get("name") if isinstance(action_payload, dict) else None
                if not isinstance(action_name_value, str) or not action_name_value:
                    candidate_actions_valid = False
                    break
                candidate_action_names.add(action_name_value)
                if any(
                    not _is_finite_number(candidate.get(key))
                    for key in ("expected_value", "expected_confidence", "expected_prediction_error", "resource_cost")
                ):
                    candidate_actions_valid = False
                    break
            if not candidate_actions_valid:
                invalid_value_counts["candidate_actions"] += 1
                invalid = True

        selected_action = payload.get("selected_action")
        if not isinstance(selected_action, str) or selected_action not in valid_action_names:
            invalid_value_counts["selected_action"] += 1
            semantic_invalid_counts["selected_action_unknown"] += 1
            invalid = True
        elif candidate_action_names and selected_action not in candidate_action_names:
            semantic_invalid_counts["selected_action_not_in_candidates"] += 1
            invalid = True

        result_feedback = payload.get("result_feedback")
        if (
            not isinstance(result_feedback, dict)
            or not isinstance(result_feedback.get("observed_outcome"), str)
            or not result_feedback.get("observed_outcome")
            or not _is_finite_number(result_feedback.get("reward"))
            or not isinstance(result_feedback.get("counterfactual_warning"), bool)
        ):
            invalid_value_counts["result_feedback"] += 1
            invalid = True

        model_update = payload.get("model_update")
        if (
            not isinstance(model_update, dict)
            or not _is_unit_interval(model_update.get("magnitude"))
            or not _is_unit_interval(model_update.get("strategy_shift"))
            or not _is_finite_number(model_update.get("confidence_delta"))
        ):
            invalid_value_counts["model_update"] += 1
            invalid = True
        elif abs(float(model_update["magnitude"]) - float(payload["update_magnitude"])) > 1e-6:
            semantic_invalid_counts["update_magnitude_mismatch"] += 1
            invalid = True

        if not invalid:
            valid_records += 1
    total_records = len(records)
    invalid_records = total_records - valid_records
    invalid_rate = round((invalid_records / total_records) if total_records else 0.0, 6)
    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "invalid_rate": invalid_rate,
        "missing_field_counts": missing_counts,
        "invalid_value_counts": invalid_value_counts,
        "semantic_invalid_counts": semantic_invalid_counts,
        "parameter_snapshot_complete_records": parameter_snapshot_complete_records,
        "parameter_snapshot_complete_rate": round((parameter_snapshot_complete_records / total_records) if total_records else 0.0, 6),
    }


PROFILE_REGISTRY: dict[str, CognitiveStyleParameters] = {
    "high_exploration_low_caution": CognitiveStyleParameters(
        uncertainty_sensitivity=0.84,
        error_aversion=0.24,
        exploration_bias=0.92,
        attention_selectivity=0.48,
        confidence_gain=0.42,
        update_rigidity=0.28,
        resource_pressure_sensitivity=0.30,
        virtual_prediction_error_gain=0.22,
    ),
    "low_exploration_high_caution": CognitiveStyleParameters(
        uncertainty_sensitivity=0.34,
        error_aversion=0.92,
        exploration_bias=0.16,
        attention_selectivity=0.70,
        confidence_gain=0.66,
        update_rigidity=0.86,
        resource_pressure_sensitivity=0.90,
        virtual_prediction_error_gain=0.88,
    ),
    "balanced_midline": CognitiveStyleParameters(
        uncertainty_sensitivity=0.55,
        error_aversion=0.45,
        exploration_bias=0.60,
        attention_selectivity=0.58,
        confidence_gain=0.58,
        update_rigidity=0.45,
        resource_pressure_sensitivity=0.52,
        virtual_prediction_error_gain=0.46,
    ),
}


BLIND_CLASSIFICATION_FEATURES = [
    "confidence_evidence_slope",
    "conflict_avoidance_shift",
    "counterfactual_loss_sensitivity",
]


def _feature_vector(metrics: dict[str, Any], feature_names: list[str] | None = None) -> dict[str, float]:
    metric_values = metric_values_from_payload(metrics)
    active_features = feature_names or BLIND_CLASSIFICATION_FEATURES
    return {name: float(metric_values[name]) for name in active_features if name in metric_values}


def _prototype_distance(features: dict[str, float], prototype: dict[str, float]) -> float:
    shared = sorted(set(features) & set(prototype))
    if not shared:
        return float("inf")
    return sum(abs(features[name] - prototype[name]) for name in shared) / len(shared)


def _profile_prototypes(*, seeds: list[int]) -> dict[str, dict[str, float]]:
    prototypes: dict[str, dict[str, float]] = {}
    for profile_name, parameters in PROFILE_REGISTRY.items():
        feature_values: dict[str, list[float]] = {name: [] for name in BLIND_CLASSIFICATION_FEATURES}
        for seed in seeds:
            trial = run_cognitive_style_trial(
                parameters,
                seed=seed,
                stress=profile_name == "low_exploration_high_caution",
            )
            features = _feature_vector(trial["observable_metrics"])
            for name, value in features.items():
                feature_values[name].append(value)
        prototypes[profile_name] = {
            name: round(sum(values) / len(values), 6)
            for name, values in feature_values.items()
            if values
        }
    return prototypes


def classify_profile_from_metrics(
    metrics: dict[str, Any],
    prototypes: dict[str, dict[str, float]] | None = None,
) -> str:
    from .m41_blind_classifier import classify_profile_from_metrics as _classify_profile_from_metrics
    from .m41_blind_classifier import train_blind_classifier

    classifier_artifact = None
    if prototypes is not None:
        classifier_artifact = train_blind_classifier()
        classifier_artifact["class_centroids"] = prototypes
    return _classify_profile_from_metrics(metrics, classifier_artifact=classifier_artifact)


def blind_classification_experiment(
    *,
    train_seeds: list[int] | None = None,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from .m41_blind_classifier import blind_classification_experiment as _blind_classification_experiment

    return _blind_classification_experiment(train_seeds=train_seeds, eval_seeds=eval_seeds)


def synthetic_profile_distinguishability_benchmark(
    *,
    train_seeds: list[int] | None = None,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from .m41_blind_classifier import synthetic_profile_distinguishability_benchmark as _benchmark

    return _benchmark(train_seeds=train_seeds, eval_seeds=eval_seeds)


def validate_acceptance_report(report: dict[str, Any]) -> dict[str, Any]:
    required_gates = {
        "g1_schema_completeness",
        "g2_trial_variability",
        "g3_observability",
        "g4_intervention_sensitivity",
        "g5_log_completeness",
        "g6_stress_behavior",
        "r1_report_structure",
    }
    errors: list[str] = []
    gates = report.get("gates", {})
    missing = sorted(required_gates - set(gates))
    if missing:
        errors.append(f"missing_gates:{missing}")
    for gate_name in required_gates & set(gates):
        gate = gates[gate_name]
        if "passed" not in gate:
            errors.append(f"{gate_name}:missing_passed")
        evidence = gate.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            errors.append(f"{gate_name}:missing_evidence")
    if not isinstance(report.get("failed_gates"), list):
        errors.append("failed_gates_not_list")
    findings = report.get("findings")
    if not isinstance(findings, list):
        errors.append("findings_not_list")
    status = report.get("status")
    if status not in {"PASS", "FAIL"}:
        errors.append("status_invalid")
    blocking_failures = [
        gate_name
        for gate_name, gate in gates.items()
        if isinstance(gate, dict) and gate.get("blocking") and gate.get("passed") is False
    ]
    if blocking_failures and status != "FAIL":
        errors.append(f"status_mismatch:blocking_gate_failed:{sorted(blocking_failures)}")
    return {"valid": not errors, "errors": errors}


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
