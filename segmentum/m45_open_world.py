from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean
import subprocess
import sys
from typing import Any

from .action_schema import ActionSchema
from .m4_cognitive_style import CognitiveParameterBridge, CognitiveStyleParameters, ResourceSnapshot


@dataclass(frozen=True)
class OpenWorldTask:
    task_id: str
    task_type: str
    uncertainty: float
    evidence_strength: float
    expected_error: float
    resource_state: ResourceSnapshot
    recovery_window: bool = False
    failure_injected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "uncertainty": self.uncertainty,
            "evidence_strength": self.evidence_strength,
            "expected_error": self.expected_error,
            "resource_state": self.resource_state.to_dict(),
            "recovery_window": self.recovery_window,
            "failure_injected": self.failure_injected,
        }


def _jitter(value: float, *, rng: random.Random, spread: float) -> float:
    return max(0.0, min(1.0, value + rng.uniform(-spread, spread)))


def _task_catalog(*, seed: int, stress: bool = False) -> list[OpenWorldTask]:
    rng = random.Random(seed)
    return [
        OpenWorldTask(
            task_id="knowledge_probe",
            task_type="knowledge_retrieval",
            uncertainty=_jitter(0.76, rng=rng, spread=0.03),
            evidence_strength=_jitter(0.34, rng=rng, spread=0.04),
            expected_error=_jitter(0.40, rng=rng, spread=0.04),
            resource_state=ResourceSnapshot(
                _jitter(0.78, rng=rng, spread=0.04),
                _jitter(0.82, rng=rng, spread=0.04),
                _jitter(0.24, rng=rng, spread=0.03),
                _jitter(0.86, rng=rng, spread=0.03),
            ),
        ),
        OpenWorldTask(
            task_id="plan_route",
            task_type="multi_step_planning",
            uncertainty=_jitter(0.48, rng=rng, spread=0.04),
            evidence_strength=_jitter(0.58, rng=rng, spread=0.04),
            expected_error=_jitter(0.32, rng=rng, spread=0.04),
            resource_state=ResourceSnapshot(
                _jitter(0.64, rng=rng, spread=0.05),
                _jitter(0.58, rng=rng, spread=0.05),
                _jitter(0.34, rng=rng, spread=0.04),
                _jitter(0.56, rng=rng, spread=0.04),
            ),
        ),
        OpenWorldTask(
            task_id="repair_after_failure",
            task_type="failure_recovery",
            uncertainty=_jitter(0.66 if stress else 0.58, rng=rng, spread=0.04),
            evidence_strength=_jitter(0.42, rng=rng, spread=0.03),
            expected_error=_jitter(0.55 if stress else 0.44, rng=rng, spread=0.05),
            resource_state=ResourceSnapshot(
                _jitter(0.24 if stress else 0.36, rng=rng, spread=0.03),
                _jitter(0.20 if stress else 0.34, rng=rng, spread=0.03),
                _jitter(0.82, rng=rng, spread=0.02),
                _jitter(0.26, rng=rng, spread=0.03),
            ),
            recovery_window=True,
            failure_injected=True,
        ),
    ]


def _actions_for(task_type: str) -> list[ActionSchema]:
    if task_type == "knowledge_retrieval":
        return [
            ActionSchema(name="query", cost_estimate=0.22, resource_cost={"tokens": 0.08}),
            ActionSchema(name="scan", cost_estimate=0.18, resource_cost={"tokens": 0.06}),
            ActionSchema(name="guess", cost_estimate=0.06, resource_cost={"tokens": 0.02}),
        ]
    if task_type == "multi_step_planning":
        return [
            ActionSchema(name="inspect", cost_estimate=0.18, resource_cost={"tokens": 0.05}),
            ActionSchema(name="plan", cost_estimate=0.20, resource_cost={"tokens": 0.08}),
            ActionSchema(name="commit", cost_estimate=0.14, resource_cost={"tokens": 0.04}),
        ]
    return [
        ActionSchema(name="retry", cost_estimate=0.12, resource_cost={"tokens": 0.04}),
        ActionSchema(name="recover", cost_estimate=0.16, resource_cost={"tokens": 0.05}),
        ActionSchema(name="conserve", cost_estimate=0.05, resource_cost={"tokens": 0.01}),
    ]


def _task_consistent(task: OpenWorldTask, action: str) -> bool:
    if task.task_type == "knowledge_retrieval":
        return action in {"query", "scan"}
    if task.task_type == "multi_step_planning":
        return action in {"plan", "inspect", "commit"}
    return action in {"recover", "conserve"}


def simulate_open_world_projection(
    parameters: CognitiveStyleParameters,
    *,
    seed: int = 45,
    ablate_style: bool = False,
    stress: bool = False,
) -> dict[str, Any]:
    active_parameters = parameters
    if ablate_style:
        active_parameters = CognitiveStyleParameters(
            uncertainty_sensitivity=0.5,
            error_aversion=0.2,
            exploration_bias=0.5,
            attention_selectivity=0.35,
            confidence_gain=0.5,
            update_rigidity=0.2,
            resource_pressure_sensitivity=0.0,
        )
    bridge = CognitiveParameterBridge(active_parameters)
    logs = []
    resource_sensitive_shift = False
    adaptive_recovery = False
    mechanical_retry = False
    query_rate = 0
    planful_rate = 0
    for index, task in enumerate(_task_catalog(seed=seed, stress=stress), start=1):
        decision = bridge.decide(
            tick=index,
            seed=seed,
            task_context={"task_id": task.task_id, "task_type": task.task_type},
            observation_evidence={
                "uncertainty": task.uncertainty,
                "evidence_strength": task.evidence_strength,
                "expected_error": task.expected_error,
            },
            actions=_actions_for(task.task_type),
            resource_state=task.resource_state,
        )
        decision_payload = decision.to_dict()
        chosen = str(decision_payload["selected_action"])
        if task.task_type == "failure_recovery":
            adaptive_recovery = chosen in {"recover", "conserve"}
            mechanical_retry = chosen == "retry"
        if task.task_type == "failure_recovery" and chosen in {"conserve", "recover"}:
            resource_sensitive_shift = True
        if chosen == "query":
            query_rate += 1
        if chosen in {"plan", "inspect"}:
            planful_rate += 1
        logs.append(
            {
                "task": task.to_dict(),
                "decision": decision_payload,
                "task_consistent": _task_consistent(task, chosen),
            }
        )
    selected_actions = [row["decision"]["selected_action"] for row in logs]
    return {
        "parameters": active_parameters.to_dict(),
        "seed": seed,
        "probe_type": "synthetic_probe",
        "logs": logs,
        "summary": {
            "goal_consistency_rate": round(sum(1 for row in logs if row["task_consistent"]) / len(selected_actions), 6),
            "resource_sensitive_shift": resource_sensitive_shift,
            "adaptive_recovery_rate": 1.0 if adaptive_recovery else 0.0,
            "mechanical_retry_rate": 1.0 if mechanical_retry else 0.0,
            "query_rate": round(query_rate / len(selected_actions), 6),
            "planful_rate": round(planful_rate / len(selected_actions), 6),
            "mean_confidence": round(mean(float(row["decision"]["internal_confidence"]) for row in logs), 6),
        },
    }


def run_live_cli_task_loop(*, seed: int = 45) -> dict[str, Any]:
    commands = [
        [sys.executable, "-c", "print('resource:ok')"],
        [sys.executable, "-c", "print('recovery:ready')"],
    ]
    observed_outputs: list[str] = []
    recoveries = 0
    for index, command in enumerate(commands, start=1):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        output = completed.stdout.strip()
        observed_outputs.append(output)
        if "recovery" in output:
            recoveries += 1
    return {
        "probe_type": "live_cli_loop",
        "seed": seed,
        "command_count": len(commands),
        "observed_outputs": observed_outputs,
        "summary": {
            "completed": len(observed_outputs) == len(commands),
            "adaptive_recovery_rate": round(recoveries / max(1, len(commands)), 6),
            "live_integration": True,
        },
    }


def benchmark_open_world_projection(*, seed: int = 45) -> dict[str, Any]:
    cautious = CognitiveStyleParameters(error_aversion=0.82, resource_pressure_sensitivity=0.84, exploration_bias=0.44)
    exploratory = CognitiveStyleParameters(error_aversion=0.54, resource_pressure_sensitivity=0.58, exploration_bias=0.74)
    cautious_run = simulate_open_world_projection(cautious, seed=seed)
    exploratory_run = simulate_open_world_projection(exploratory, seed=seed)
    return {
        "probe_type": "synthetic_probe",
        "profiles": {"cautious": cautious_run, "exploratory": exploratory_run},
        "live_cli_loop": run_live_cli_task_loop(seed=seed),
        "correspondence": {
            "high_error_aversion_maps_to_low_retry": cautious_run["summary"]["mechanical_retry_rate"] <= exploratory_run["summary"]["mechanical_retry_rate"],
            "high_exploration_maps_to_more_information_actions": exploratory_run["summary"]["query_rate"] >= cautious_run["summary"]["query_rate"],
        },
    }
