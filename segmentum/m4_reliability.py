from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReliabilityAssessment:
    deployment_readiness: str
    claim_envelope: str
    warning_count: int
    warnings: list[str]
    strengths: list[str]
    blocking_gaps: list[str]
    metrics_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_behavior_fit_reliability(
    *,
    benchmark_name: str,
    trial_count: int,
    subject_count: int,
    bootstrap_lower: float,
    subject_floor: float,
    calibration_ceiling: float,
    synthetic_slice: bool,
) -> ReliabilityAssessment:
    warnings: list[str] = []
    strengths: list[str] = []
    blocking_gaps: list[str] = []

    if bootstrap_lower > 0.0:
        strengths.append("Held-out bootstrap margin is positive.")
    else:
        blocking_gaps.append("Held-out bootstrap margin is not robustly positive.")

    if calibration_ceiling < 0.35:
        strengths.append("Worst-subject calibration remains bounded.")
    else:
        warnings.append("Worst-subject calibration is still loose.")

    if subject_floor > -0.30:
        strengths.append("Per-subject held-out floor stays above collapse threshold.")
    else:
        warnings.append("At least one subject-level held-out score is weak.")

    if trial_count < 50:
        blocking_gaps.append(f"{benchmark_name} trial count is too small for deployment-grade fit claims.")
    if subject_count < 10:
        blocking_gaps.append(f"{benchmark_name} subject coverage is too small for production calibration trust.")
    if synthetic_slice:
        blocking_gaps.append(f"{benchmark_name} evidence comes from a repository-curated slice rather than a production benchmark feed.")

    deployment_readiness = "READY" if not blocking_gaps else "NOT_READY"
    claim_envelope = "bounded_product_signal" if not blocking_gaps else "prototype_only"
    return ReliabilityAssessment(
        deployment_readiness=deployment_readiness,
        claim_envelope=claim_envelope,
        warning_count=len(warnings) + len(blocking_gaps),
        warnings=warnings,
        strengths=strengths,
        blocking_gaps=blocking_gaps,
        metrics_snapshot={
            "trial_count": trial_count,
            "subject_count": subject_count,
            "bootstrap_lower": round(float(bootstrap_lower), 6),
            "subject_floor": round(float(subject_floor), 6),
            "calibration_ceiling": round(float(calibration_ceiling), 6),
        },
    )


def assess_cross_task_reliability(
    *,
    confidence_trial_count: int,
    confidence_subject_count: int,
    igt_trial_count: int,
    igt_subject_count: int,
    shared_parameter_count: int,
    parameter_distance_mean: float,
    policy_alignment_rate: float,
    synthetic_slice: bool,
) -> ReliabilityAssessment:
    warnings: list[str] = []
    strengths: list[str] = []
    blocking_gaps: list[str] = []

    if shared_parameter_count >= 4 and parameter_distance_mean <= 0.10:
        strengths.append("A stable shared-parameter core is present across tasks.")
    else:
        blocking_gaps.append("Cross-task shared core is not yet stable enough.")

    if policy_alignment_rate >= 0.50:
        strengths.append("IGT policy alignment clears the minimum behavioral bar.")
    else:
        warnings.append("IGT policy alignment remains weak.")

    if confidence_trial_count < 50 or igt_trial_count < 50:
        blocking_gaps.append("Cross-task evidence is still based on undersized benchmark slices.")
    if confidence_subject_count < 10 or igt_subject_count < 10:
        blocking_gaps.append("Cross-task subject coverage is too small for production generalization claims.")
    if synthetic_slice:
        blocking_gaps.append("Cross-task validation still relies on repository-frozen slices.")

    deployment_readiness = "READY" if not blocking_gaps else "NOT_READY"
    claim_envelope = "cross_task_signal" if not blocking_gaps else "prototype_only"
    return ReliabilityAssessment(
        deployment_readiness=deployment_readiness,
        claim_envelope=claim_envelope,
        warning_count=len(warnings) + len(blocking_gaps),
        warnings=warnings,
        strengths=strengths,
        blocking_gaps=blocking_gaps,
        metrics_snapshot={
            "confidence_trial_count": confidence_trial_count,
            "confidence_subject_count": confidence_subject_count,
            "igt_trial_count": igt_trial_count,
            "igt_subject_count": igt_subject_count,
            "shared_parameter_count": shared_parameter_count,
            "parameter_distance_mean": round(float(parameter_distance_mean), 6),
            "policy_alignment_rate": round(float(policy_alignment_rate), 6),
        },
    )


def assess_synthetic_projection_reliability(
    *,
    milestone_name: str,
    goal_consistency_rate: float,
    adaptive_recovery_rate: float,
    synthetic_environment: bool,
    live_integration: bool,
) -> ReliabilityAssessment:
    warnings: list[str] = []
    strengths: list[str] = []
    blocking_gaps: list[str] = []

    if goal_consistency_rate >= 2.0 / 3.0:
        strengths.append("Behavior remains task-consistent under the synthetic probe.")
    else:
        warnings.append("Task consistency is still too low.")

    if adaptive_recovery_rate >= 1.0:
        strengths.append("Failure recovery remains available under stress.")
    else:
        warnings.append("Recovery behavior is not consistently retained.")

    if synthetic_environment:
        blocking_gaps.append(f"{milestone_name} still operates on a synthetic environment scaffold.")
    if not live_integration:
        blocking_gaps.append(f"{milestone_name} has no live tool or environment integration yet.")

    deployment_readiness = "READY" if not blocking_gaps else "NOT_READY"
    claim_envelope = "runtime_projection" if not blocking_gaps else "synthetic_projection_only"
    return ReliabilityAssessment(
        deployment_readiness=deployment_readiness,
        claim_envelope=claim_envelope,
        warning_count=len(warnings) + len(blocking_gaps),
        warnings=warnings,
        strengths=strengths,
        blocking_gaps=blocking_gaps,
        metrics_snapshot={
            "goal_consistency_rate": round(float(goal_consistency_rate), 6),
            "adaptive_recovery_rate": round(float(adaptive_recovery_rate), 6),
            "synthetic_environment": synthetic_environment,
            "live_integration": live_integration,
        },
    )
