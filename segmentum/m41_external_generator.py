from __future__ import annotations

"""Same-framework synthetic holdout generator retained for sidecar diagnostics.

The file keeps the historical ``external`` module name for compatibility, but
its outputs are repo-owned synthetic holdouts with shared latent semantics.
They are not external human-data validation and do not count as M4.1
acceptance evidence or M4.2 benchmark recovery completion.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random
from statistics import mean
from typing import Any

from .m4_cognitive_style import (
    CognitiveStyleParameters,
    DecisionLogRecord,
    PROFILE_REGISTRY,
    reconstruct_behavior_patterns,
)
from .m41_external_observables import (
    EXTERNAL_MEASUREMENT_MISMATCHES,
    EXTERNAL_OBSERVABLES_IMPLEMENTATION_FAMILY,
    SAME_FRAMEWORK_HOLDOUT_MEASUREMENT_MISMATCHES,
    SAME_FRAMEWORK_HOLDOUT_OBSERVABLES_IMPLEMENTATION_FAMILY,
    compute_same_framework_holdout_observable_metrics,
    metric_values_from_payload,
)


SAME_FRAMEWORK_HOLDOUT_PROFILE_REGISTRY: dict[str, CognitiveStyleParameters] = {
    **PROFILE_REGISTRY,
    "volatile_opportunistic": CognitiveStyleParameters(
        uncertainty_sensitivity=0.73,
        error_aversion=0.38,
        exploration_bias=0.76,
        attention_selectivity=0.44,
        confidence_gain=0.61,
        update_rigidity=0.22,
        resource_pressure_sensitivity=0.49,
        virtual_prediction_error_gain=0.31,
    ),
    "methodical_resource_guarded": CognitiveStyleParameters(
        uncertainty_sensitivity=0.47,
        error_aversion=0.82,
        exploration_bias=0.24,
        attention_selectivity=0.79,
        confidence_gain=0.57,
        update_rigidity=0.74,
        resource_pressure_sensitivity=0.87,
        virtual_prediction_error_gain=0.71,
    ),
}
EXTERNAL_PROFILE_REGISTRY = SAME_FRAMEWORK_HOLDOUT_PROFILE_REGISTRY


_SCENARIO_POOLS: dict[str, list[dict[str, Any]]] = {
    "synthetic_holdout_default": [
        {"name": "field_recon", "actions": ["scan", "inspect", "query", "commit", "recover"]},
        {"name": "market_sweep", "actions": ["scan", "query", "plan", "commit", "retry"]},
        {"name": "ops_triage", "actions": ["inspect", "recover", "conserve", "plan", "commit"]},
        {"name": "watchtower", "actions": ["scan", "query", "conserve", "commit"]},
        {"name": "expedition", "actions": ["scan", "inspect", "query", "recover", "commit", "guess"]},
    ],
    "synthetic_holdout_eval": [
        {"name": "audit_fork", "actions": ["inspect", "query", "plan", "recover", "commit"]},
        {"name": "signal_hunt", "actions": ["scan", "inspect", "query", "commit"]},
        {"name": "scarcity_basin", "actions": ["recover", "conserve", "scan", "plan", "commit"]},
        {"name": "uncertain_corridor", "actions": ["query", "scan", "inspect", "retry", "commit"]},
        {"name": "anomaly_stitch", "actions": ["inspect", "plan", "recover", "query", "commit", "guess"]},
    ],
}

_SCENARIO_FAMILY_ALIASES = {
    "external_default": "synthetic_holdout_default",
    "external_holdout": "synthetic_holdout_eval",
}


@dataclass(frozen=True)
class ExternalState:
    evidence_state: float
    uncertainty_state: float
    pressure_state: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float) -> float:
    return round(float(value), 6)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _softmax(scores: dict[str, float], *, temperature: float, rng: random.Random) -> tuple[str, dict[str, float]]:
    temp = max(0.15, float(temperature))
    scaled = {name: math.exp(score / temp) for name, score in scores.items()}
    total = sum(scaled.values()) or 1.0
    roll = rng.random()
    cumulative = 0.0
    selected = next(iter(scores))
    for name, value in scaled.items():
        cumulative += value / total
        if roll <= cumulative:
            selected = name
            break
    return selected, {name: _round(value / total) for name, value in scaled.items()}


def _canonical_scenario_family(scenario_family: str) -> str:
    return _SCENARIO_FAMILY_ALIASES.get(scenario_family, scenario_family)


def _scenario_family_pool(scenario_family: str) -> list[dict[str, Any]]:
    canonical = _canonical_scenario_family(scenario_family)
    return list(_SCENARIO_POOLS.get(canonical, _SCENARIO_POOLS["synthetic_holdout_default"]))


def _initial_state(parameters: CognitiveStyleParameters, rng: random.Random) -> ExternalState:
    return ExternalState(
        evidence_state=_clamp01(0.35 + parameters.attention_selectivity * 0.20 + rng.random() * 0.15),
        uncertainty_state=_clamp01(0.40 + parameters.uncertainty_sensitivity * 0.18 + rng.random() * 0.16),
        pressure_state=_clamp01(0.28 + parameters.resource_pressure_sensitivity * 0.22 + rng.random() * 0.14),
    )


def _step_state(
    state: ExternalState,
    parameters: CognitiveStyleParameters,
    *,
    rng: random.Random,
    tick: int,
) -> ExternalState:
    markov_anchor = (tick % 5) / 4.0
    evidence_state = _clamp01(
        0.58 * state.evidence_state
        + 0.22 * (1.0 - state.uncertainty_state)
        + 0.10 * parameters.attention_selectivity
        + 0.10 * markov_anchor
        + (rng.random() - 0.5) * 0.12
    )
    uncertainty_state = _clamp01(
        0.52 * state.uncertainty_state
        + 0.18 * (1.0 - evidence_state)
        + 0.16 * parameters.uncertainty_sensitivity
        + 0.14 * (1.0 if tick % 7 in {0, 1} else 0.0)
        + (rng.random() - 0.5) * 0.16
    )
    pressure_state = _clamp01(
        0.60 * state.pressure_state
        + 0.15 * uncertainty_state
        + 0.15 * parameters.resource_pressure_sensitivity
        + 0.10 * (1.0 if tick % 9 in {5, 6, 7} else 0.0)
        + (rng.random() - 0.5) * 0.10
    )
    return ExternalState(
        evidence_state=evidence_state,
        uncertainty_state=uncertainty_state,
        pressure_state=pressure_state,
    )


def _resource_state(parameters: CognitiveStyleParameters, *, tick: int, tick_count: int, state: ExternalState) -> dict[str, float]:
    progress = tick / max(1, tick_count - 1)
    energy = _clamp01(math.exp(-progress * (0.85 + parameters.resource_pressure_sensitivity * 0.55)) - state.pressure_state * 0.18)
    budget = _clamp01(math.exp(-progress * 0.72) - state.uncertainty_state * 0.12)
    time_remaining = _clamp01(1.0 - progress)
    stress = _clamp01(state.pressure_state * 0.62 + state.uncertainty_state * 0.24 + (1.0 - energy) * 0.22)
    return {
        "energy": _round(energy),
        "budget": _round(budget),
        "time_remaining": _round(time_remaining),
        "stress": _round(stress),
    }


def _observation_evidence(parameters: CognitiveStyleParameters, state: ExternalState, *, rng: random.Random) -> dict[str, float]:
    logistic_noise = _logistic((rng.random() - 0.5) * 2.8) - 0.5
    evidence_strength = _clamp01(
        state.evidence_state * (0.82 + parameters.attention_selectivity * 0.24)
        * (0.88 + logistic_noise * 0.20)
    )
    uncertainty = _clamp01(
        state.uncertainty_state * (0.80 + parameters.uncertainty_sensitivity * 0.24)
        * (0.92 + abs(logistic_noise) * 0.18)
    )
    expected_error = _clamp01(
        0.42 * uncertainty
        + 0.24 * state.pressure_state
        + parameters.error_aversion * 0.16
        + (1.0 - evidence_strength) * 0.18
    )
    imagined_risk = _clamp01(
        0.28 * expected_error
        + parameters.virtual_prediction_error_gain * 0.46
        + uncertainty * 0.18
        + max(0.0, 0.45 - evidence_strength) * 0.14
    )
    return {
        "evidence_strength": _round(evidence_strength),
        "uncertainty": _round(uncertainty),
        "expected_error": _round(expected_error),
        "imagined_risk": _round(imagined_risk),
    }


def _attention_allocation(observation_evidence: dict[str, float], parameters: CognitiveStyleParameters) -> dict[str, float]:
    evidence = 0.36 + observation_evidence["evidence_strength"] * (0.42 + parameters.attention_selectivity * 0.18)
    uncertainty = 0.28 + observation_evidence["uncertainty"] * (0.38 + parameters.uncertainty_sensitivity * 0.22)
    total = evidence + uncertainty
    return {
        "evidence": _round(evidence / total),
        "uncertainty": _round(uncertainty / total),
    }


def _prediction_error_vector(observation_evidence: dict[str, float], parameters: CognitiveStyleParameters) -> dict[str, float]:
    direct_error = _clamp01(
        observation_evidence["expected_error"] * (0.74 + observation_evidence["uncertainty"] * 0.18)
    )
    virtual_error = _clamp01(
        direct_error * 0.36
        + observation_evidence["imagined_risk"] * (0.44 + parameters.virtual_prediction_error_gain * 0.44)
        + observation_evidence["uncertainty"] * parameters.virtual_prediction_error_gain * 0.18
        - observation_evidence["evidence_strength"] * 0.08
    )
    signed_total = _clamp01((direct_error * 0.62) + (virtual_error * 0.38))
    return {
        "direct_error": _round(direct_error),
        "virtual_error": _round(virtual_error),
        "signed_total": _round(signed_total),
    }


def _dominant_signal(observation_evidence: dict[str, float], resource_state: dict[str, float]) -> str:
    if observation_evidence["imagined_risk"] > observation_evidence["expected_error"] and observation_evidence["imagined_risk"] >= 0.55:
        return "counterfactual"
    if resource_state["stress"] >= 0.70:
        return "error"
    if observation_evidence["uncertainty"] >= 0.62 and observation_evidence["uncertainty"] > observation_evidence["evidence_strength"]:
        return "uncertainty"
    return "evidence"


def _action_scores(
    *,
    actions: list[str],
    parameters: CognitiveStyleParameters,
    observation_evidence: dict[str, float],
    resource_state: dict[str, float],
) -> dict[str, float]:
    evidence = observation_evidence["evidence_strength"]
    uncertainty = observation_evidence["uncertainty"]
    error = observation_evidence["expected_error"]
    imagined = observation_evidence["imagined_risk"]
    pressure = _clamp01((1.0 - resource_state["energy"] + resource_state["stress"]) / 2.0)

    scores: dict[str, float] = {}
    for action in actions:
        score = 0.0
        if action in {"scan", "inspect", "query"}:
            score += 0.16 + uncertainty * (0.34 + parameters.exploration_bias * 0.32)
            score += (1.0 - evidence) * 0.18
            score -= pressure * 0.08
            score -= error * (1.0 - parameters.error_aversion) * 0.20
        if action == "plan":
            score += parameters.attention_selectivity * 0.18 + uncertainty * 0.14 + imagined * 0.08
            score -= pressure * 0.05
        if action == "commit":
            score += evidence * (0.38 + parameters.confidence_gain * 0.30)
            score += (1.0 - parameters.error_aversion) * 0.12
            score += (1.0 - parameters.resource_pressure_sensitivity) * 0.06
            score -= uncertainty * (0.18 + parameters.uncertainty_sensitivity * 0.24)
            score -= error * (0.18 + parameters.error_aversion * 0.30)
            score -= imagined * parameters.virtual_prediction_error_gain * 0.28
        if action in {"recover", "conserve", "rest"}:
            score += pressure * (0.18 + parameters.resource_pressure_sensitivity * 0.46)
            score += error * (parameters.error_aversion * 0.34)
            score += imagined * (0.04 + parameters.virtual_prediction_error_gain * 0.18)
            score -= parameters.exploration_bias * 0.08
        if action == "retry":
            score += evidence * 0.18
            score -= error * (0.10 + parameters.error_aversion * 0.14)
            score += (1.0 - parameters.update_rigidity) * 0.18
            score += (1.0 - parameters.error_aversion) * 0.14
            score += parameters.exploration_bias * 0.08
        if action == "guess":
            score += parameters.exploration_bias * 0.22
            score += (1.0 - parameters.error_aversion) * 0.18
            score -= error * 0.12
            score -= uncertainty * 0.10
        if action == "commit" and evidence < 0.45:
            score -= 0.12 + parameters.error_aversion * 0.08
        if action in {"recover", "conserve"} and pressure >= 0.58:
            score += 0.10 + parameters.resource_pressure_sensitivity * 0.16
        score += (parameters.attention_selectivity - 0.5) * 0.06
        scores[action] = score
    return scores


def _candidate_payloads(actions: list[str], probabilities: dict[str, float], observation_evidence: dict[str, float], prediction_error_vector: dict[str, float]) -> list[dict[str, Any]]:
    payloads = []
    for action in actions:
        probability = probabilities[action]
        payloads.append(
            {
                "action": {"name": action},
                "total_score": _round(math.log(max(probability, 1e-6))),
                "expected_value": _round(_clamp01(probability + observation_evidence["evidence_strength"] * 0.28)),
                "expected_confidence": _round(_clamp01(probability + observation_evidence["evidence_strength"] * 0.18 - observation_evidence["uncertainty"] * 0.10)),
                "expected_prediction_error": prediction_error_vector["signed_total"],
                "resource_cost": _round(
                    0.08
                    + (0.14 if action == "commit" else 0.10 if action in {"retry", "guess", "plan"} else 0.05)
                ),
            }
        )
    return payloads


def run_same_framework_holdout_trial(
    parameters: CognitiveStyleParameters,
    seed: int,
    tick_count: int = 50,
    scenario_family: str = "synthetic_holdout_default",
) -> dict[str, Any]:
    canonical_family = _canonical_scenario_family(scenario_family)
    rng = random.Random(f"same_framework_holdout_v1:{canonical_family}:{seed}")
    pool = _scenario_family_pool(canonical_family)
    state = _initial_state(parameters, rng)
    logs: list[dict[str, Any]] = []
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)

    for tick in range(tick_count):
        scenario = pool[tick % len(pool)]
        state = _step_state(state, parameters, rng=rng, tick=tick)
        observation = _observation_evidence(parameters, state, rng=rng)
        resources = _resource_state(parameters, tick=tick, tick_count=tick_count, state=state)
        prediction_error = _prediction_error_vector(observation, parameters)
        attention = _attention_allocation(observation, parameters)
        scores = _action_scores(actions=list(scenario["actions"]), parameters=parameters, observation_evidence=observation, resource_state=resources)
        temperature = _clamp01(0.65 + parameters.exploration_bias * 0.45 + observation["uncertainty"] * 0.25)
        selected_action, probabilities = _softmax(scores, temperature=temperature, rng=rng)

        dominant_signal = _dominant_signal(observation, resources)
        percept_summary = {
            "dominant_signal": dominant_signal,
            "evidence_band": "high" if observation["evidence_strength"] >= 0.70 else "medium" if observation["evidence_strength"] >= 0.45 else "low",
            "uncertainty_band": "high" if observation["uncertainty"] >= 0.60 else "medium" if observation["uncertainty"] >= 0.35 else "low",
            "pressure_band": "high" if resources["stress"] >= 0.70 else "medium" if resources["stress"] >= 0.45 else "low",
        }
        confidence = _clamp01(
            0.22
            + observation["evidence_strength"] * (0.36 + parameters.confidence_gain * 0.34)
            - observation["uncertainty"] * (0.12 + parameters.uncertainty_sensitivity * 0.18)
            - prediction_error["signed_total"] * 0.10
        )
        update_magnitude = _clamp01(
            prediction_error["signed_total"] * (0.62 - parameters.update_rigidity * 0.34) + observation["uncertainty"] * 0.10
        )
        reward = _clamp01(
            0.55
            + observation["evidence_strength"] * 0.25
            - prediction_error["signed_total"] * 0.28
            - (0.10 if selected_action == "guess" else 0.0)
        )
        model_update = {
            "magnitude": _round(update_magnitude),
            "strategy_shift": _round(_clamp01(update_magnitude * (1.10 - parameters.update_rigidity * 0.65))),
            "confidence_delta": _round(reward - confidence),
        }
        payload = {
            "schema_version": "m4.decision_log.v3",
            "tick": tick + 1,
            "timestamp": (start + timedelta(minutes=tick * 3 + seed % 5)).isoformat(timespec="seconds"),
            "seed": seed,
            "task_context": {
                "phase": scenario["name"],
                "subject_id": f"same-framework-subject-{seed}",
                "session_id": f"same-framework-{canonical_family}-{seed}",
                "source_name": "same_framework_synth_lab",
                "generator_id": "same_framework_holdout_v1",
            },
            "percept_summary": percept_summary,
            "observation_evidence": observation,
            "prediction_error_vector": prediction_error,
            "attention_allocation": attention,
            "candidate_actions": _candidate_payloads(list(scenario["actions"]), probabilities, observation, prediction_error),
            "parameter_snapshot": parameters.to_dict(),
            "resource_state": resources,
            "internal_confidence": _round(confidence),
            "selected_action": selected_action,
            "result_feedback": {
                "observed_outcome": f"{scenario['name']}::{selected_action}",
                "reward": _round(reward),
                "counterfactual_warning": prediction_error["virtual_error"] > prediction_error["direct_error"],
            },
            "model_update": model_update,
            "prediction_error": prediction_error["signed_total"],
            "update_magnitude": model_update["magnitude"],
        }
        logs.append(DecisionLogRecord.from_dict(payload).to_dict())

    observable_metrics = compute_same_framework_holdout_observable_metrics(logs)
    return {
        "analysis_type": "same_framework_synthetic_holdout_trial",
        "benchmark_scope": "same-framework synthetic holdout generator for sidecar diagnostics",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "generator_family": "same_framework_synthetic_holdout",
        "validation_type": "synthetic_holdout_same_framework",
        "not_acceptance_evidence": True,
        "logs": logs,
        "ground_truth_parameters": parameters.to_dict(),
        "generator_id": "same_framework_holdout_v1",
        "scenario_family": canonical_family,
        "requested_scenario_family": scenario_family,
        "observable_metrics": observable_metrics,
        "observable_metric_values": metric_values_from_payload(observable_metrics),
        "observables_provenance": {
            "implementation_family": SAME_FRAMEWORK_HOLDOUT_OBSERVABLES_IMPLEMENTATION_FAMILY,
            "measurement_mismatch": list(SAME_FRAMEWORK_HOLDOUT_MEASUREMENT_MISMATCHES),
            "shared_metric_names": sorted(observable_metrics),
        },
        "patterns": reconstruct_behavior_patterns(logs),
        "summary": {
            "tick_count": tick_count,
            "selected_actions": [row["selected_action"] for row in logs],
            "mean_confidence": _round(mean(float(row["internal_confidence"]) for row in logs)),
            "scenario_families": sorted({row["task_context"]["phase"] for row in logs}),
        },
        "validation_limits": [
            "generator is owned by the repository and shares the same latent style family as the training-side interfaces",
            "measurement mismatch is synthetic and intentionally controlled, not an independent external benchmark bundle",
            "results are sidecar diagnostics only and do not count as M4.1 acceptance or M4.2 benchmark recovery evidence",
        ],
    }


def run_external_trial(
    parameters: CognitiveStyleParameters,
    seed: int,
    tick_count: int = 50,
    scenario_family: str = "external_default",
) -> dict[str, Any]:
    return run_same_framework_holdout_trial(
        parameters,
        seed=seed,
        tick_count=tick_count,
        scenario_family=scenario_family,
    )
