from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import math
import random

from .action_schema import ActionSchema, action_name, ensure_action_schema


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


@dataclass(frozen=True)
class CognitiveStyleParameters:
    schema_version: str = "m4.cognitive_style.v2"
    uncertainty_sensitivity: float = 0.65
    error_aversion: float = 0.7
    exploration_bias: float = 0.55
    attention_selectivity: float = 0.6
    confidence_gain: float = 0.7
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
        if "virtual_prediction_error_gain" not in normalized:
            normalized["virtual_prediction_error_gain"] = PARAMETER_REFERENCE["virtual_prediction_error_gain"]["default"]
        if "schema_version" not in normalized:
            normalized["schema_version"] = cls().schema_version
        return cls(**{key: normalized[key] for key in normalized if key in cls.__dataclass_fields__})

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
        result_feedback = dict(payload.get("result_feedback", {}))
        model_update = dict(payload.get("model_update", {}))
        if not prediction_error_vector:
            scalar_error = float(payload.get("prediction_error", 0.0))
            prediction_error_vector = {
                "direct_error": scalar_error,
                "virtual_error": scalar_error,
                "signed_total": scalar_error,
            }
        if not result_feedback:
            result_feedback = {
                "observed_outcome": "legacy_unknown",
                "reward": round(1.0 - float(payload.get("prediction_error", 0.0)), 6),
            }
        if not model_update:
            update_magnitude = float(payload.get("update_magnitude", 0.0))
            model_update = {
                "magnitude": update_magnitude,
                "strategy_shift": round(update_magnitude * 0.5, 6),
                "confidence_delta": round(-float(payload.get("prediction_error", 0.0)) * 0.1, 6),
            }
        return cls(
            schema_version=str(payload.get("schema_version", "m4.decision_log.v3")),
            tick=int(payload.get("tick", 0)),
            timestamp=str(payload.get("timestamp", datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(timespec="seconds"))),
            seed=int(payload.get("seed", 0)),
            task_context=dict(payload.get("task_context", {})),
            percept_summary=dict(payload.get("percept_summary", payload.get("task_context", {}))),
            observation_evidence={
                str(key): float(value) for key, value in dict(payload.get("observation_evidence", {})).items()
            },
            prediction_error_vector=prediction_error_vector,
            attention_allocation={
                str(key): float(value) for key, value in dict(payload.get("attention_allocation", {})).items()
            },
            candidate_actions=list(payload.get("candidate_actions", [])),
            parameter_snapshot=dict(payload.get("parameter_snapshot", {})),
            resource_state={str(key): float(value) for key, value in dict(payload.get("resource_state", {})).items()},
            internal_confidence=float(payload.get("internal_confidence", 0.0)),
            selected_action=str(payload.get("selected_action", "")),
            result_feedback=result_feedback,
            model_update=model_update,
            prediction_error=float(payload.get("prediction_error", prediction_error_vector.get("signed_total", 0.0))),
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


def observable_metrics_registry() -> dict[str, dict[str, Any]]:
    return {
        "uncertainty_confidence_drop_rate": {
            "parameter": "uncertainty_sensitivity",
            "description": "High uncertainty lowers internal confidence.",
            "formula": "mean(high_uncertainty * (1 - internal_confidence))",
            "depends_on": ["observation_evidence.uncertainty", "internal_confidence"],
            "direction": "higher_means_higher_parameter",
        },
        "high_uncertainty_inspect_ratio": {
            "parameter": "uncertainty_sensitivity",
            "description": "Ambiguous contexts increase inspect-like actions.",
            "formula": "P(selected_action in {scan,inspect,query} | uncertainty >= 0.6)",
            "depends_on": ["observation_evidence.uncertainty", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "high_expected_error_rejection_rate": {
            "parameter": "error_aversion",
            "description": "Riskier actions are rejected when expected error is high.",
            "formula": "P(selected_action not in {commit,guess,retry} | expected_error >= 0.45)",
            "depends_on": ["observation_evidence.expected_error", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "post_error_conservative_shift": {
            "parameter": "error_aversion",
            "description": "Error signals trigger conservative follow-up choices.",
            "formula": "P(selected_action in {rest,recover,scan,plan} | prediction_error >= 0.4)",
            "depends_on": ["prediction_error", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "novel_action_ratio": {
            "parameter": "exploration_bias",
            "description": "Information-seeking actions are favored in uncertainty.",
            "formula": "P(selected_action in {scan,inspect,query} | uncertainty >= 0.5)",
            "depends_on": ["observation_evidence.uncertainty", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "choice_repeat_suppression": {
            "parameter": "exploration_bias",
            "description": "Repeated action streaks shorten when exploration rises.",
            "formula": "1 / max_repeat_streak(selected_action)",
            "depends_on": ["selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "dominant_attention_share": {
            "parameter": "attention_selectivity",
            "description": "Attention concentrates on the strongest evidence feature.",
            "formula": "mean(max(attention_allocation.values()))",
            "depends_on": ["attention_allocation"],
            "direction": "higher_means_higher_parameter",
        },
        "evidence_aligned_choice_rate": {
            "parameter": "attention_selectivity",
            "description": "Chosen actions align with the dominant evidence channel.",
            "formula": "P(selected_action matches percept_summary.dominant_signal)",
            "depends_on": ["selected_action", "percept_summary.dominant_signal"],
            "direction": "higher_means_higher_parameter",
        },
        "confidence_evidence_slope": {
            "parameter": "confidence_gain",
            "description": "Confidence rises with stronger evidence separation.",
            "formula": "mean(evidence_strength * internal_confidence)",
            "depends_on": ["observation_evidence.evidence_strength", "internal_confidence"],
            "direction": "higher_means_higher_parameter",
        },
        "high_evidence_commit_rate": {
            "parameter": "confidence_gain",
            "description": "Commit wins more often once evidence becomes strong.",
            "formula": "P(selected_action == commit | evidence_strength >= 0.7)",
            "depends_on": ["observation_evidence.evidence_strength", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "mean_update_inverse": {
            "parameter": "update_rigidity",
            "description": "Prediction errors yield smaller internal updates.",
            "formula": "1 - mean(model_update.magnitude)",
            "depends_on": ["model_update.magnitude"],
            "direction": "higher_means_higher_parameter",
        },
        "strategy_persistence_after_error": {
            "parameter": "update_rigidity",
            "description": "After error, action policy changes more slowly.",
            "formula": "mean(1 - model_update.strategy_shift)",
            "depends_on": ["model_update.strategy_shift"],
            "direction": "higher_means_higher_parameter",
        },
        "high_pressure_low_cost_ratio": {
            "parameter": "resource_pressure_sensitivity",
            "description": "Low-cost actions dominate under resource pressure.",
            "formula": "P(selected_action in {rest,conserve,recover,scan} | pressure >= 0.6)",
            "depends_on": ["resource_state", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "recovery_trigger_rate": {
            "parameter": "resource_pressure_sensitivity",
            "description": "Recovery triggers when energy and time are low.",
            "formula": "P(selected_action in {rest,recover,conserve} | energy <= 0.35 or time_remaining <= 0.3)",
            "depends_on": ["resource_state", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "conflict_avoidance_shift": {
            "parameter": "virtual_prediction_error_gain",
            "description": "Imagined loss signals bias decisions away from direct commit.",
            "formula": "P(selected_action != commit | virtual_error > direct_error)",
            "depends_on": ["prediction_error_vector.virtual_error", "prediction_error_vector.direct_error", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
        "counterfactual_loss_sensitivity": {
            "parameter": "virtual_prediction_error_gain",
            "description": "Counterfactual loss increases conservative choices.",
            "formula": "mean((virtual_error - direct_error)_+ * conservative_choice)",
            "depends_on": ["prediction_error_vector", "selected_action"],
            "direction": "higher_means_higher_parameter",
        },
    }


def default_behavior_mapping_table() -> dict[str, dict[str, Any]]:
    registry = observable_metrics_registry()
    return {
        metric_name: {
            "primary_parameter": spec["parameter"],
            "observable": spec["description"],
            "formula": spec["formula"],
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
            }
            for metric_name, spec in registry.items()
            if spec["parameter"] == parameter_name
        ]
        contracts[parameter_name] = {
            "physical_meaning": PARAMETER_REFERENCE[parameter_name]["physical_meaning"],
            "observables": metrics,
        }
    return contracts


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
            expected_error * 0.45
            + imagined_risk * (0.35 + self.parameters.virtual_prediction_error_gain * 0.65)
            + uncertainty * self.parameters.virtual_prediction_error_gain * 0.20
            - evidence_strength * 0.12
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
            "evidence": 0.30 + self.parameters.attention_selectivity * evidence_strength * 0.55,
            "uncertainty": 0.16 + self.parameters.uncertainty_sensitivity * uncertainty * 0.32,
            "error": 0.14 + self.parameters.error_aversion * expected_error * 0.34,
            "counterfactual": 0.10 + self.parameters.virtual_prediction_error_gain * imagined_risk * 0.34,
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
        action_label = action_name(action)
        cost = _clamp01(float(action.cost_estimate) + sum(max(0.0, float(v)) for v in action.resource_cost.values()) * 0.15)
        prediction_error_vector = self._prediction_error_vector(
            evidence_strength=evidence_strength,
            uncertainty=uncertainty,
            expected_error=expected_error,
            imagined_risk=imagined_risk,
        )
        virtual_error = prediction_error_vector["virtual_error"]
        direct_error = prediction_error_vector["direct_error"]
        conservative_actions = {"rest", "conserve", "recover", "scan", "inspect", "query", "plan"}

        exploration_bonus = self.parameters.exploration_bias * uncertainty
        if action_label in ("scan", "inspect", "query"):
            exploration_bonus += 0.18 * uncertainty
        if action_label == "query":
            exploration_bonus += self.parameters.exploration_bias * 0.12
        if action_label == "plan":
            exploration_bonus += self.parameters.attention_selectivity * max(0.0, 0.40 - uncertainty) * 0.18
        if action_label in ("rest", "conserve"):
            exploration_bonus -= 0.10 * uncertainty

        attention_bonus = self.parameters.attention_selectivity * evidence_strength
        if action_label in ("commit", "choose_right", "choose_left"):
            attention_bonus += 0.20 * evidence_strength
        if action_label == "recover":
            attention_bonus += self.parameters.error_aversion * expected_error * 0.16
            attention_bonus += self.parameters.update_rigidity * 0.10
        if action_label == "plan":
            attention_bonus += self.parameters.update_rigidity * evidence_strength * 0.10

        resource_penalty = self.parameters.resource_pressure_sensitivity * resource_pressure * (0.35 + cost)
        if action_label in ("rest", "conserve"):
            resource_penalty *= 0.35
        if action_label == "retry":
            resource_penalty *= 1.15
        if action_label == "recover":
            resource_penalty *= 0.80

        error_penalty = self.parameters.error_aversion * direct_error * 0.72
        virtual_penalty = self.parameters.virtual_prediction_error_gain * max(0.0, virtual_error - direct_error * 0.50) * 0.58
        if action_label in ("rest", "conserve", "scan", "query", "recover"):
            error_penalty *= 0.65
            virtual_penalty *= 0.72
        if action_label in conservative_actions and virtual_error > direct_error:
            attention_bonus += self.parameters.virtual_prediction_error_gain * (virtual_error - direct_error) * 0.28
        if action_label == "guess":
            error_penalty *= 1.35
            virtual_penalty *= 1.25
            if uncertainty > 0.55:
                error_penalty += uncertainty * 0.10
        if action_label == "retry":
            error_penalty *= 1.15
            virtual_penalty *= 1.10
            error_penalty += resource_pressure * expected_error * 0.18
        if action_label == "recover":
            error_penalty *= 0.72

        base_score = attention_bonus + exploration_bonus - error_penalty - virtual_penalty - resource_penalty
        if action_label == "commit" and evidence_strength >= 0.75 and uncertainty <= 0.25 and virtual_error <= 0.35:
            base_score += self.parameters.confidence_gain * 0.28
        if action_label in ("rest", "conserve") and resource_pressure > 0.55:
            base_score += self.parameters.resource_pressure_sensitivity * 0.30
        if action_label in ("scan", "inspect") and uncertainty > 0.6:
            base_score += self.parameters.uncertainty_sensitivity * 0.15
        if action_label == "query" and uncertainty > 0.65:
            base_score += self.parameters.exploration_bias * 0.18
        if action_label == "plan" and evidence_strength >= 0.50 and uncertainty <= 0.50:
            base_score += self.parameters.attention_selectivity * 0.16
        if action_label == "recover" and expected_error >= 0.40:
            recovery_support = (
                self.parameters.error_aversion
                + self.parameters.update_rigidity
                + self.parameters.resource_pressure_sensitivity
            ) / 3.0
            base_score += recovery_support * 0.26 + self.parameters.resource_pressure_sensitivity * resource_pressure * 0.16
            base_score -= (1.0 - recovery_support) * 0.24
        if action_label == "retry" and expected_error >= 0.40:
            base_score += (1.0 - self.parameters.error_aversion) * 0.12
            base_score += (1.0 - self.parameters.resource_pressure_sensitivity) * 0.10
            base_score -= self.parameters.error_aversion * 0.12 + self.parameters.resource_pressure_sensitivity * resource_pressure * 0.12
        if action_label == "commit" and virtual_error > direct_error:
            base_score -= self.parameters.virtual_prediction_error_gain * (virtual_error - direct_error) * 0.44

        confidence = _clamp01(
            0.30
            + self.parameters.confidence_gain * evidence_strength * 0.62
            - self.parameters.uncertainty_sensitivity * uncertainty * 0.20
            - direct_error * 0.10
            - self.parameters.virtual_prediction_error_gain * virtual_error * 0.08
        )
        if action_label == "recover":
            confidence = _clamp01(confidence + self.parameters.error_aversion * 0.06)
        if action_label == "guess":
            confidence = _clamp01(confidence - 0.08)
        update_magnitude = _clamp01(prediction_error_vector["signed_total"] * (1.0 - self.parameters.update_rigidity * 0.75))
        expected_value = _clamp01(
            0.50 + evidence_strength * 0.35 - direct_error * 0.18 - resource_penalty * 0.10 - virtual_penalty * 0.12
        )
        return CandidateScore(
            action=action.to_dict(),
            total_score=round(base_score, 6),
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
        timestamp = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() + tick + seed
        prediction_error_vector = winner.expected_prediction_error_vector
        reward = round(1.0 - winner.expected_prediction_error - winner.resource_penalty * 0.15, 6)
        return DecisionLogRecord(
            schema_version="m4.decision_log.v3",
            tick=tick,
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds"),
            seed=seed,
            task_context=dict(task_context),
            percept_summary={
                "dominant_signal": dominant_signal,
                "evidence_band": "high" if evidence_strength >= 0.7 else "medium" if evidence_strength >= 0.4 else "low",
                "uncertainty_band": "high" if uncertainty >= 0.6 else "medium" if uncertainty >= 0.3 else "low",
            },
            observation_evidence={key: round(float(value), 6) for key, value in observation_evidence.items()},
            prediction_error_vector=prediction_error_vector,
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
                "observed_outcome": "stabilized" if reward >= 0.55 else "fragile" if reward >= 0.35 else "lossy",
                "reward": reward,
                "counterfactual_warning": prediction_error_vector["virtual_error"] > prediction_error_vector["direct_error"],
            },
            model_update={
                "magnitude": winner.update_magnitude,
                "strategy_shift": round(winner.update_magnitude * (1.0 - self.parameters.update_rigidity * 0.30), 6),
                "confidence_delta": round(winner.expected_confidence - 0.5, 6),
            },
            prediction_error=winner.expected_prediction_error,
            update_magnitude=winner.update_magnitude,
        )


def canonical_action_schemas() -> list[ActionSchema]:
    return [
        ActionSchema(name="scan", cost_estimate=0.35, resource_cost={"tokens": 0.15}),
        ActionSchema(name="query", cost_estimate=0.22, resource_cost={"tokens": 0.08}),
        ActionSchema(name="commit", cost_estimate=0.45, resource_cost={"tokens": 0.20}),
        ActionSchema(name="plan", cost_estimate=0.18, resource_cost={"tokens": 0.05}),
        ActionSchema(name="recover", cost_estimate=0.16, resource_cost={"tokens": 0.03}),
        ActionSchema(name="rest", cost_estimate=0.08, resource_cost={"tokens": 0.02}),
        ActionSchema(name="guess", cost_estimate=0.06, resource_cost={"tokens": 0.02}),
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
            {
                **parameters.to_dict(),
                "resource_pressure_sensitivity": 0.0,
                "schema_version": CognitiveStyleParameters().schema_version,
            }
        )
    bridge = CognitiveParameterBridge(active_parameters)
    sequence = [
        (
            {"phase": "ambiguity_probe"},
            {"evidence_strength": 0.28, "uncertainty": 0.84, "expected_error": 0.42, "imagined_risk": 0.48},
            ResourceSnapshot(energy=0.78, budget=0.82, stress=0.24, time_remaining=0.90),
        ),
        (
            {"phase": "commit_window"},
            {"evidence_strength": 0.87, "uncertainty": 0.18, "expected_error": 0.15, "imagined_risk": 0.12},
            ResourceSnapshot(energy=0.74, budget=0.75, stress=0.28, time_remaining=0.70),
        ),
        (
            {"phase": "pressure_spike"},
            {"evidence_strength": 0.36, "uncertainty": 0.52, "expected_error": 0.39, "imagined_risk": 0.41},
            ResourceSnapshot(energy=0.22 if stress else 0.32, budget=0.18 if stress else 0.28, stress=0.82, time_remaining=0.21),
        ),
        (
            {"phase": "counterfactual_conflict"},
            {"evidence_strength": 0.67, "uncertainty": 0.24, "expected_error": 0.21, "imagined_risk": 0.79},
            ResourceSnapshot(energy=0.58, budget=0.63, stress=0.41, time_remaining=0.49),
        ),
        (
            {"phase": "recovery_probe"},
            {"evidence_strength": 0.51, "uncertainty": 0.31, "expected_error": 0.27, "imagined_risk": 0.22},
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
    metrics = compute_observable_metrics(logs)
    return {
        "parameters": active_parameters.to_dict(),
        "logs": [record.to_dict() for record in logs],
        "patterns": patterns,
        "observable_metrics": metrics,
        "summary": {
            "selected_actions": [record.selected_action for record in logs],
            "mean_confidence": round(sum(record.internal_confidence for record in logs) / len(logs), 6),
            "mean_update_magnitude": round(sum(record.update_magnitude for record in logs) / len(logs), 6),
            "pattern_count": len(patterns),
        },
    }


def _probe_score_delta(
    parameters: CognitiveStyleParameters,
    *,
    action_name_a: str,
    action_name_b: str,
    context: dict[str, float],
    resource_state: ResourceSnapshot,
) -> float:
    actions = {action.name: action for action in canonical_action_schemas()}
    bridge = CognitiveParameterBridge(parameters)
    score_a = bridge.score_action(actions[action_name_a], resource_state=resource_state, **context)
    score_b = bridge.score_action(actions[action_name_b], resource_state=resource_state, **context)
    return round(score_a.total_score - score_b.total_score, 6)


def parameter_probe_registry() -> dict[str, dict[str, Any]]:
    moderate_resources = ResourceSnapshot(energy=0.72, budget=0.76, stress=0.25, time_remaining=0.83)
    hard_resources = ResourceSnapshot(energy=0.18, budget=0.22, stress=0.82, time_remaining=0.20)
    return {
        "uncertainty_sensitivity": {
            "probe_type": "score_margin",
            "context": {"evidence_strength": 0.22, "uncertainty": 0.92, "expected_error": 0.35, "imagined_risk": 0.18},
            "resource_state": moderate_resources,
            "target": "scan_minus_commit_margin",
            "actions": ("scan", "commit"),
            "expectation": "higher",
        },
        "error_aversion": {
            "probe_type": "score_margin",
            "context": {"evidence_strength": 0.28, "uncertainty": 0.44, "expected_error": 0.86, "imagined_risk": 0.20},
            "resource_state": moderate_resources,
            "target": "recover_minus_guess_margin",
            "actions": ("recover", "guess"),
            "expectation": "higher",
        },
        "exploration_bias": {
            "probe_type": "score_margin",
            "context": {"evidence_strength": 0.22, "uncertainty": 0.92, "expected_error": 0.35, "imagined_risk": 0.18},
            "resource_state": moderate_resources,
            "target": "query_minus_rest_margin",
            "actions": ("query", "rest"),
            "expectation": "higher",
        },
        "attention_selectivity": {
            "probe_type": "attention_share",
            "context": {"evidence_strength": 0.98, "uncertainty": 0.06, "expected_error": 0.08, "imagined_risk": 0.02},
            "resource_state": moderate_resources,
            "target": "attention_allocation.evidence",
            "expectation": "higher",
        },
        "confidence_gain": {
            "probe_type": "commit_confidence",
            "context": {"evidence_strength": 0.92, "uncertainty": 0.08, "expected_error": 0.05, "imagined_risk": 0.01},
            "resource_state": moderate_resources,
            "target": "commit.expected_confidence",
            "expectation": "higher",
        },
        "update_rigidity": {
            "probe_type": "recover_update_magnitude",
            "context": {"evidence_strength": 0.28, "uncertainty": 0.44, "expected_error": 0.86, "imagined_risk": 0.20},
            "resource_state": moderate_resources,
            "target": "recover.update_magnitude",
            "expectation": "lower",
        },
        "resource_pressure_sensitivity": {
            "probe_type": "trial_metric",
            "trial_kwargs": {"stress": True},
            "target": "high_pressure_low_cost_ratio",
            "metric": "high_pressure_low_cost_ratio",
            "expectation": "higher",
        },
        "virtual_prediction_error_gain": {
            "probe_type": "score_margin",
            "context": {"evidence_strength": 0.72, "uncertainty": 0.22, "expected_error": 0.15, "imagined_risk": 0.96},
            "resource_state": moderate_resources,
            "target": "recover_minus_commit_margin",
            "actions": ("recover", "commit"),
            "expectation": "higher",
        },
    }


def parameter_causality_matrix(*, seed: int = 41) -> dict[str, dict[str, Any]]:
    baseline = CognitiveStyleParameters()
    actions = {action.name: action for action in canonical_action_schemas()}
    matrix: dict[str, dict[str, Any]] = {}
    for parameter_name, probe in parameter_probe_registry().items():
        low = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.0})
        high = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 1.0})
        low_value = 0.0
        high_value = 0.0
        low_trial: dict[str, Any] | None = None
        high_trial: dict[str, Any] | None = None
        if probe["probe_type"] == "score_margin":
            low_value = _probe_score_delta(
                low,
                action_name_a=probe["actions"][0],
                action_name_b=probe["actions"][1],
                context=dict(probe["context"]),
                resource_state=probe["resource_state"],
            )
            high_value = _probe_score_delta(
                high,
                action_name_a=probe["actions"][0],
                action_name_b=probe["actions"][1],
                context=dict(probe["context"]),
                resource_state=probe["resource_state"],
            )
        elif probe["probe_type"] == "attention_share":
            low_value = CognitiveParameterBridge(low)._attention_allocation(**probe["context"])["evidence"]
            high_value = CognitiveParameterBridge(high)._attention_allocation(**probe["context"])["evidence"]
        elif probe["probe_type"] == "commit_confidence":
            low_value = CognitiveParameterBridge(low).score_action(
                actions["commit"],
                resource_state=probe["resource_state"],
                **probe["context"],
            ).expected_confidence
            high_value = CognitiveParameterBridge(high).score_action(
                actions["commit"],
                resource_state=probe["resource_state"],
                **probe["context"],
            ).expected_confidence
        elif probe["probe_type"] == "recover_update_magnitude":
            low_value = CognitiveParameterBridge(low).score_action(
                actions["recover"],
                resource_state=probe["resource_state"],
                **probe["context"],
            ).update_magnitude
            high_value = CognitiveParameterBridge(high).score_action(
                actions["recover"],
                resource_state=probe["resource_state"],
                **probe["context"],
            ).update_magnitude
        elif probe["probe_type"] == "trial_metric":
            trial_kwargs = dict(probe.get("trial_kwargs", {}))
            low_trial = run_cognitive_style_trial(low, seed=seed, **trial_kwargs)
            high_trial = run_cognitive_style_trial(high, seed=seed, **trial_kwargs)
            low_value = float(low_trial["observable_metrics"][probe["metric"]])
            high_value = float(high_trial["observable_metrics"][probe["metric"]])
        identifiable = high_value > low_value if probe["expectation"] == "higher" else high_value < low_value
        matrix[parameter_name] = {
            "parameter": parameter_name,
            "probe_type": probe["probe_type"],
            "target": probe["target"],
            "expectation": probe["expectation"],
            "low_parameter_value": 0.0,
            "high_parameter_value": 1.0,
            "low_observed_value": round(float(low_value), 6),
            "high_observed_value": round(float(high_value), 6),
            "delta": round(float(high_value) - float(low_value), 6),
            "identifiable": bool(identifiable),
            "low_selected_actions": low_trial["summary"]["selected_actions"] if low_trial is not None else [],
            "high_selected_actions": high_trial["summary"]["selected_actions"] if high_trial is not None else [],
        }
    return matrix


def reconstruct_behavior_patterns(records: list[DecisionLogRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record)
        for record in records
    ]
    patterns: list[dict[str, Any]] = []
    if any(
        record.selected_action in {"scan", "query"} and record.observation_evidence.get("uncertainty", 0.0) >= 0.75
        for record in normalized
    ):
        patterns.append({"label": "directed_exploration", "evidence": "inspect-like action selected during high uncertainty"})
    if any(
        record.selected_action in {"rest", "recover"}
        and ResourceSnapshot.from_dict(record.resource_state).energy <= 0.35
        and ResourceSnapshot.from_dict(record.resource_state).budget <= 0.30
        for record in normalized
    ):
        patterns.append({"label": "resource_conservation", "evidence": "recovery action selected under elevated resource pressure"})
    if any(
        record.selected_action == "commit"
        and record.internal_confidence >= 0.60
        and record.update_magnitude <= 0.20
        for record in normalized
    ):
        patterns.append({"label": "confidence_sharpening", "evidence": "commit selected with high confidence and bounded update"})
    if any(
        record.prediction_error_vector.get("virtual_error", 0.0) > record.prediction_error_vector.get("direct_error", 0.0)
        and record.selected_action != "commit"
        for record in normalized
    ):
        patterns.append({"label": "counterfactual_avoidance", "evidence": "imagined-loss conflict shifted choice away from commit"})
    return patterns


def _mean(values: list[float], default: float = 0.0) -> float:
    return round(sum(values) / len(values), 6) if values else default


def compute_observable_metrics(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, float]:
    normalized = [
        record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record)
        for record in records
    ]
    selected_actions = [record.selected_action for record in normalized]
    max_repeat = 0
    streak = 0
    previous = None
    for action in selected_actions:
        streak = streak + 1 if action == previous else 1
        max_repeat = max(max_repeat, streak)
        previous = action

    high_uncertainty = [record for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.6]
    high_error = [record for record in normalized if record.observation_evidence.get("expected_error", 0.0) >= 0.35]
    high_evidence = [record for record in normalized if record.observation_evidence.get("evidence_strength", 0.0) >= 0.7]
    high_pressure = [
        record
        for record in normalized
        if (
            1.0
            - (
                (
                    record.resource_state.get("energy", 0.0)
                    + record.resource_state.get("budget", 0.0)
                    + record.resource_state.get("time_remaining", 0.0)
                )
                / 3.0
            )
            + record.resource_state.get("stress", 0.0)
        )
        / 2.0
        >= 0.6
    ]
    conflict_cases = [
        record
        for record in normalized
        if record.prediction_error_vector.get("virtual_error", 0.0) > record.prediction_error_vector.get("direct_error", 0.0)
    ]
    conservative_actions = {"rest", "recover", "conserve", "scan", "plan", "query"}
    risky_actions = {"commit", "guess", "retry"}
    inspect_actions = {"scan", "inspect", "query"}
    low_cost_actions = {"rest", "recover", "conserve", "scan"}

    return {
        "uncertainty_confidence_drop_rate": _mean(
            [
                record.observation_evidence.get("uncertainty", 0.0) * (1.0 - record.internal_confidence)
                for record in high_uncertainty
            ]
        ),
        "high_uncertainty_inspect_ratio": _mean([1.0 if record.selected_action in inspect_actions else 0.0 for record in high_uncertainty]),
        "high_expected_error_rejection_rate": _mean([1.0 if record.selected_action not in risky_actions else 0.0 for record in high_error]),
        "post_error_conservative_shift": _mean([1.0 if record.selected_action in conservative_actions else 0.0 for record in high_error]),
        "novel_action_ratio": _mean([1.0 if record.selected_action in inspect_actions else 0.0 for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.5]),
        "choice_repeat_suppression": round(1.0 / max_repeat, 6) if max_repeat else 0.0,
        "dominant_attention_share": _mean([max(record.attention_allocation.values()) for record in normalized if record.attention_allocation]),
        "evidence_aligned_choice_rate": _mean(
            [
                1.0
                if (
                    (record.percept_summary.get("dominant_signal") == "evidence" and record.selected_action == "commit")
                    or (record.percept_summary.get("dominant_signal") == "uncertainty" and record.selected_action in inspect_actions)
                    or (record.percept_summary.get("dominant_signal") == "error" and record.selected_action in {"recover", "rest", "plan"})
                    or (record.percept_summary.get("dominant_signal") == "counterfactual" and record.selected_action in conservative_actions)
                )
                else 0.0
                for record in normalized
            ]
        ),
        "confidence_evidence_slope": _mean(
            [
                record.observation_evidence.get("evidence_strength", 0.0) * record.internal_confidence
                for record in normalized
            ]
        ),
        "high_evidence_commit_rate": _mean([1.0 if record.selected_action == "commit" else 0.0 for record in high_evidence]),
        "mean_update_inverse": round(1.0 - _mean([record.model_update.get("magnitude", 0.0) for record in normalized]), 6),
        "strategy_persistence_after_error": _mean([1.0 - record.model_update.get("strategy_shift", 0.0) for record in high_error]),
        "high_pressure_low_cost_ratio": _mean([1.0 if record.selected_action in low_cost_actions else 0.0 for record in high_pressure]),
        "recovery_trigger_rate": _mean(
            [
                1.0 if record.selected_action in {"rest", "recover", "conserve"} else 0.0
                for record in normalized
                if record.resource_state.get("energy", 0.0) <= 0.35 or record.resource_state.get("time_remaining", 0.0) <= 0.30
            ]
        ),
        "conflict_avoidance_shift": _mean([1.0 if record.selected_action != "commit" else 0.0 for record in conflict_cases]),
        "counterfactual_loss_sensitivity": _mean(
            [
                max(0.0, record.prediction_error_vector.get("virtual_error", 0.0) - record.prediction_error_vector.get("direct_error", 0.0))
                * (1.0 if record.selected_action in conservative_actions else 0.0)
                for record in normalized
            ]
        ),
    }


def audit_decision_log(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, Any]:
    normalized = [
        record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record)
        for record in records
    ]
    required_fields = DecisionLogRecord.schema()["required"]
    missing_counts = {field_name: 0 for field_name in required_fields}
    parameter_snapshot_complete_records = 0
    valid_records = 0
    for record in normalized:
        payload = record.to_dict()
        missing = False
        for field_name in required_fields:
            value = payload.get(field_name)
            if value in (None, "", [], {}):
                missing_counts[field_name] += 1
                missing = True
        snapshot = payload.get("parameter_snapshot", {})
        if isinstance(snapshot, dict) and all(field in snapshot for field in CognitiveStyleParameters.schema()["required"]):
            parameter_snapshot_complete_records += 1
        else:
            missing = True
        if not missing:
            valid_records += 1
    total_records = len(normalized)
    invalid_records = total_records - valid_records
    invalid_rate = round((invalid_records / total_records) if total_records else 0.0, 6)
    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "invalid_rate": invalid_rate,
        "missing_field_counts": missing_counts,
        "parameter_snapshot_complete_records": parameter_snapshot_complete_records,
        "parameter_snapshot_complete_rate": round(
            (parameter_snapshot_complete_records / total_records) if total_records else 0.0,
            6,
        ),
    }


PROFILE_REGISTRY: dict[str, CognitiveStyleParameters] = {
    "high_exploration_low_caution": CognitiveStyleParameters(
        exploration_bias=0.88,
        uncertainty_sensitivity=0.82,
        error_aversion=0.28,
        confidence_gain=0.48,
        update_rigidity=0.32,
        resource_pressure_sensitivity=0.34,
        virtual_prediction_error_gain=0.24,
    ),
    "low_exploration_high_caution": CognitiveStyleParameters(
        exploration_bias=0.22,
        uncertainty_sensitivity=0.35,
        error_aversion=0.90,
        confidence_gain=0.62,
        update_rigidity=0.82,
        resource_pressure_sensitivity=0.86,
        virtual_prediction_error_gain=0.90,
    ),
    "balanced_midline": CognitiveStyleParameters(
        exploration_bias=0.52,
        uncertainty_sensitivity=0.56,
        error_aversion=0.58,
        confidence_gain=0.68,
        update_rigidity=0.60,
        resource_pressure_sensitivity=0.62,
        virtual_prediction_error_gain=0.62,
    ),
}


def classify_profile_from_metrics(metrics: dict[str, float]) -> str:
    if (
        metrics["novel_action_ratio"] >= 0.90
        and metrics["high_pressure_low_cost_ratio"] <= 0.20
    ):
        return "high_exploration_low_caution"
    if (
        metrics["novel_action_ratio"] <= 0.10
        and metrics["high_uncertainty_inspect_ratio"] <= 0.10
    ):
        return "low_exploration_high_caution"
    return "balanced_midline"


def blind_classification_experiment(*, seeds: list[int] | None = None) -> dict[str, Any]:
    active_seeds = seeds or [41, 42, 43, 44]
    samples: list[dict[str, Any]] = []
    confusion_matrix = {
        profile_name: {other_name: 0 for other_name in PROFILE_REGISTRY}
        for profile_name in PROFILE_REGISTRY
    }
    for profile_name, parameters in PROFILE_REGISTRY.items():
        for seed in active_seeds:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=profile_name == "low_exploration_high_caution")
            metrics = dict(trial["observable_metrics"])
            predicted_profile = classify_profile_from_metrics(metrics)
            confusion_matrix[profile_name][predicted_profile] += 1
            samples.append(
                {
                    "seed": seed,
                    "true_profile": profile_name,
                    "predicted_profile": predicted_profile,
                    "selected_actions": trial["summary"]["selected_actions"],
                    "metrics": metrics,
                }
            )
    accuracy = round(sum(1 for sample in samples if sample["true_profile"] == sample["predicted_profile"]) / len(samples), 6)
    per_class = {}
    for profile_name in PROFILE_REGISTRY:
        true_positive = confusion_matrix[profile_name][profile_name]
        predicted_positive = sum(confusion_matrix[other][profile_name] for other in PROFILE_REGISTRY)
        actual_positive = sum(confusion_matrix[profile_name].values())
        per_class[profile_name] = {
            "precision": round(true_positive / predicted_positive, 6) if predicted_positive else 0.0,
            "recall": round(true_positive / actual_positive, 6) if actual_positive else 0.0,
        }
    return {
        "profiles": {name: params.to_dict() for name, params in PROFILE_REGISTRY.items()},
        "seeds": active_seeds,
        "samples": samples,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def parameter_identifiability_probe(*, seed: int = 41) -> dict[str, Any]:
    baseline = run_cognitive_style_trial(CognitiveStyleParameters(), seed=seed)
    matrix = parameter_causality_matrix(seed=seed)
    return {
        "contracts": observable_parameter_contracts(),
        "identifiable": {name: probe["identifiable"] for name, probe in matrix.items()},
        "baseline": baseline["summary"],
        "probes": matrix,
    }


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
