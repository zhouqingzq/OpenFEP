from __future__ import annotations

import json
import re
import random
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.counterfactual import (
    DEFAULT_REVIEW_FAMILIES,
    CounterfactualInsight,
    InsightAbsorber,
    compute_family_coverage,
)
from segmentum.action_schema import action_name
from segmentum.agent import SegmentAgent
from segmentum.preferences import PreferenceModel
from segmentum.world_model import GenerativeWorldModel

FOLLOWUP_METRICS_PATH = ROOT / "reports" / "m2_followup_metrics.json"
LEGACY_METRICS_PATH = ROOT / "reports" / "m2_metrics.json"
READINESS_METRICS_PATH = ROOT / "reports" / "m3_readiness_metrics.json"
READINESS_REPORT_PATH = ROOT / "reports" / "m3_readiness_repair_report.md"
GENERATOR_PATH = "evals/m3_readiness_evaluation.py"
READINESS_TEST_TARGETS = [
    "tests/test_m3_readiness.py",
    "tests/test_memory.py",
    "tests/test_counterfactual_artifact.py",
    "tests/test_m23_ultimate_consolidation_loop.py",
]

STRICT_THRESHOLDS = {
    "ICI": 0.80,
    "EAA": 0.80,
    "MUR": 0.55,
    "PSSR": 0.30,
    "CAQ": 0.55,
    "VCUS": 0.80,
}


def _store_probe_episode(
    agent: SegmentAgent,
    *,
    cycle: int,
    action: str,
    observation: dict[str, float],
    prediction: dict[str, float],
    outcome: dict[str, float],
    body_state: dict[str, float],
) -> None:
    errors = {
        key: float(observation.get(key, 0.0)) - float(prediction.get(key, 0.0))
        for key in observation
    }
    decision = agent.long_term_memory.maybe_store_episode(
        cycle=cycle,
        observation=observation,
        prediction=prediction,
        errors=errors,
        action=action,
        outcome=outcome,
        body_state=body_state,
    )
    if not decision.episode_created and not decision.merged_into_episode_id:
        agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=observation,
            prediction=prediction,
            errors=errors,
            action=action,
            outcome=outcome,
            body_state=body_state,
        )


@dataclass(frozen=True)
class ReadinessStatus:
    status: str
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "reasons": list(self.reasons)}


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_entry(
    *,
    name: str,
    value: float | None,
    threshold: float | None,
    verification_status: str,
    evidence_origin: str,
    strict_metric: bool,
    gating_metric: bool,
    revalidated_this_round: bool,
    inherited_from_followup: bool,
    legacy_metric: bool = False,
    non_gating: bool = False,
    claimed_not_replayed: bool = False,
) -> dict[str, object]:
    passes = None
    if value is not None and threshold is not None:
        passes = float(value) >= float(threshold)
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passes_threshold": passes,
        "verification_status": verification_status,
        "evidence_origin": evidence_origin,
        "revalidated_this_round": revalidated_this_round,
        "inherited_from_followup": inherited_from_followup,
        "claimed_not_replayed": claimed_not_replayed,
        "strict_metric": strict_metric,
        "gating_metric": gating_metric,
        "legacy_metric": legacy_metric,
        "non_gating": non_gating,
    }


def derive_strict_metrics(followup_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    followup_metrics = dict(followup_payload.get("new_metrics", {}))
    strict: dict[str, dict[str, object]] = {}
    for name, threshold in STRICT_THRESHOLDS.items():
        strict[name] = metric_entry(
            name=name,
            value=followup_metrics.get(name),
            threshold=threshold,
            verification_status="INHERITED_STRICT_BASELINE",
            evidence_origin="reports/m2_followup_metrics.json:new_metrics",
            strict_metric=True,
            gating_metric=True,
            revalidated_this_round=False,
            inherited_from_followup=True,
        )
    return strict


def derive_legacy_metrics(legacy_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    legacy_metrics = dict(legacy_payload.get("metrics", {}))
    legacy_thresholds = dict(legacy_payload.get("thresholds", {}))
    derived: dict[str, dict[str, object]] = {}
    for name, value in legacy_metrics.items():
        derived[name] = metric_entry(
            name=name,
            value=value,
            threshold=legacy_thresholds.get(name),
            verification_status="LEGACY_FOR_COMPARISON_ONLY",
            evidence_origin="reports/m2_metrics.json:metrics",
            strict_metric=False,
            gating_metric=False,
            revalidated_this_round=False,
            inherited_from_followup=False,
            legacy_metric=True,
            non_gating=True,
            claimed_not_replayed=True,
        )
    return derived


def derive_claimed_test_evidence(previous_readiness: dict[str, object]) -> dict[str, object]:
    previous_tests = dict(previous_readiness.get("tests", {}))
    if not previous_tests:
        return {
            "verification_status": "NOT_PRESENT",
            "evidence_origin": "missing_previous_readiness_artifact",
            "revalidated_this_round": False,
            "claimed_total_passed": None,
            "claimed_new_readiness_tests": None,
        }
    return {
        "verification_status": "CLAIMED_BUT_NOT_REVALIDATED",
        "evidence_origin": "prior_readiness_artifact",
        "revalidated_this_round": False,
        "claimed_total_passed": previous_tests.get("total_passed"),
        "claimed_new_readiness_tests": previous_tests.get("new_m3_readiness_tests"),
        "regression_tests_passing": previous_tests.get("regression_tests_passing"),
    }


def execute_readiness_test_suite() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", *READINESS_TEST_TARGETS, "-q"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = "\n".join(
        part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
    )
    passed_match = re.search(r"(\d+)\s+passed", combined_output)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    failed_match = re.search(r"(\d+)\s+failed", combined_output)
    failed_count = int(failed_match.group(1)) if failed_match else 0
    if completed.returncode == 0 and passed_count > 0:
        verification_status = "REVALIDATED_THIS_ROUND"
    else:
        verification_status = "FAILED_THIS_ROUND"
    return {
        "verification_status": verification_status,
        "evidence_origin": f"{GENERATOR_PATH}:pytest",
        "revalidated_this_round": completed.returncode == 0,
        "command": command,
        "target_count": len(READINESS_TEST_TARGETS),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "output_summary": combined_output.splitlines()[-1] if combined_output else "",
    }


def derive_lifecycle_evidence(followup_payload: dict[str, object]) -> dict[str, object]:
    sleep_summary = (
        dict(
            dict(followup_payload.get("per_scenario_breakdown", {}))
            .get("sleep_reduction", {})
            .get("sleep_summary", {})
        )
    )
    compression_removed = int(sleep_summary.get("compression_removed", 0))
    archived = int(sleep_summary.get("episodes_archived", 0))
    pruned = int(sleep_summary.get("episodes_deleted", 0))
    lifecycle_activity = any(value > 0 for value in (compression_removed, archived, pruned))
    return {
        "verification_status": "INHERITED_STRICT_BASELINE",
        "evidence_origin": "reports/m2_followup_metrics.json:per_scenario_breakdown.sleep_reduction.sleep_summary",
        "revalidated_this_round": False,
        "lifecycle_activity_observed": lifecycle_activity,
        "compression_specifically_verified": compression_removed > 0,
        "compression_removed_count": compression_removed,
        "archived_count": archived,
        "pruned_count": pruned,
        "compressed_cluster_count": None,
        "compressed_cluster_count_status": "CLAIMED_NOT_REPLAYED",
    }


def evaluate_runtime_lifecycle_evidence() -> dict[str, object]:
    compression_agent = SegmentAgent(rng=random.Random(29))
    compression_agent.long_term_memory.sleep_minimum_support = 3
    compression_observation = {
        "food": 0.38,
        "danger": 0.58,
        "novelty": 0.22,
        "shelter": 0.18,
        "temperature": 0.46,
        "social": 0.18,
    }
    compression_prediction = {
        "food": 0.72,
        "danger": 0.18,
        "novelty": 0.42,
        "shelter": 0.42,
        "temperature": 0.50,
        "social": 0.30,
    }
    compression_outcome = {
        "energy_delta": -0.08,
        "stress_delta": 0.24,
        "fatigue_delta": 0.16,
        "temperature_delta": 0.02,
        "free_energy_drop": -0.42,
    }
    compression_body_state = {
        "energy": 0.18,
        "stress": 0.82,
        "fatigue": 0.32,
        "temperature": 0.46,
    }
    for cycle in range(1, 6):
        errors = {
            key: float(compression_observation.get(key, 0.0))
            - float(compression_prediction.get(key, 0.0))
            for key in compression_observation
        }
        compression_agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=compression_observation,
            prediction=compression_prediction,
            errors=errors,
            action="forage",
            outcome=compression_outcome,
            body_state=compression_body_state,
        )
    compression_summary = compression_agent.sleep()
    compressed_cluster_count = sum(
        1
        for payload in compression_agent.long_term_memory.episodes
        if int(payload.get("compressed_count", 1)) > 1
    )

    prune_agent = SegmentAgent(rng=random.Random(23))
    prune_agent.long_term_memory.minimum_support = 2
    prune_agent.long_term_memory.surprise_threshold = 0.20
    prune_observation = {
        "food": 0.85,
        "danger": 0.10,
        "novelty": 0.55,
        "shelter": 0.75,
        "temperature": 0.50,
        "social": 0.40,
    }
    prune_prediction = {
        "food": 0.84,
        "danger": 0.11,
        "novelty": 0.52,
        "shelter": 0.74,
        "temperature": 0.50,
        "social": 0.39,
    }
    prune_outcome = {
        "energy_delta": 0.10,
        "stress_delta": -0.02,
        "free_energy_drop": 0.15,
    }
    prune_body_state = {
        "energy": 0.88,
        "stress": 0.12,
        "fatigue": 0.16,
        "temperature": 0.50,
    }
    for cycle in range(1, 5):
        _store_probe_episode(
            prune_agent,
            cycle=cycle,
            action="scan",
            observation=prune_observation,
            prediction=prune_prediction,
            outcome=prune_outcome,
            body_state=prune_body_state,
        )
    prune_summary = prune_agent.sleep()

    archive_agent = SegmentAgent(rng=random.Random(41))
    archive_agent.cycle = 50
    archive_agent.long_term_memory.max_active_age = 5
    archive_observation = {
        "food": 0.20,
        "danger": 0.62,
        "novelty": 0.24,
        "shelter": 0.18,
        "temperature": 0.48,
        "social": 0.18,
    }
    archive_prediction = {
        "food": 0.68,
        "danger": 0.22,
        "novelty": 0.40,
        "shelter": 0.34,
        "temperature": 0.50,
        "social": 0.24,
    }
    archive_outcome = {
        "energy_delta": -0.04,
        "stress_delta": 0.12,
        "fatigue_delta": 0.08,
        "temperature_delta": 0.01,
        "free_energy_drop": -0.14,
    }
    archive_body_state = {
        "energy": 0.32,
        "stress": 0.60,
        "fatigue": 0.36,
        "temperature": 0.48,
    }
    _store_probe_episode(
        archive_agent,
        cycle=1,
        action="forage",
        observation=archive_observation,
        prediction=archive_prediction,
        outcome=archive_outcome,
        body_state=archive_body_state,
    )
    archive_agent.long_term_memory.assign_clusters()
    archive_payload = archive_agent.long_term_memory.episodes[0]
    archive_cluster = int(archive_payload["cluster_id"])
    archive_agent.world_model.set_outcome_distribution(
        archive_cluster,
        action_name(archive_payload.get("action_taken", archive_payload.get("action", ""))),
        {
            str(archive_payload.get("predicted_outcome", "neutral")): 0.20,
            "neutral": 0.80,
        },
    )
    archived_count, deleted_count = archive_agent._surprise_based_forgetting(
        [archive_payload]
    )

    compression_removed_count = int(compression_summary.compression_removed)
    archived_count_total = int(compression_summary.episodes_archived) + int(archived_count)
    pruned_count_total = int(compression_summary.episodes_deleted) + int(prune_summary.episodes_deleted) + int(deleted_count)
    lifecycle_activity = any(
        value > 0
        for value in (
            compression_removed_count,
            archived_count_total,
            pruned_count_total,
        )
    )
    compression_verified = compression_removed_count > 0 and compressed_cluster_count > 0
    return {
        "verification_status": "REVALIDATED_THIS_ROUND",
        "evidence_origin": f"{GENERATOR_PATH}:runtime_lifecycle_probe",
        "revalidated_this_round": True,
        "lifecycle_activity_observed": lifecycle_activity,
        "compression_specifically_verified": compression_verified,
        "compression_removed_count": compression_removed_count,
        "archived_count": archived_count_total,
        "pruned_count": pruned_count_total,
        "compressed_cluster_count": compressed_cluster_count,
        "compressed_cluster_count_status": "REVALIDATED_THIS_ROUND",
        "probe_results": [
            {
                "scenario_label": "compression",
                "compression_removed": int(compression_summary.compression_removed),
                "compressed_cluster_count": compressed_cluster_count,
                "episodes_archived": int(compression_summary.episodes_archived),
                "episodes_deleted": int(compression_summary.episodes_deleted),
            },
            {
                "scenario_label": "prune_predictable",
                "compression_removed": int(prune_summary.compression_removed),
                "compressed_cluster_count": 0,
                "episodes_archived": int(prune_summary.episodes_archived),
                "episodes_deleted": int(prune_summary.episodes_deleted),
            },
            {
                "scenario_label": "archive_old_unexplained",
                "compression_removed": 0,
                "compressed_cluster_count": 0,
                "episodes_archived": int(archived_count),
                "episodes_deleted": int(deleted_count),
            },
        ],
    }


def evaluate_runtime_family_coverage() -> dict[str, object]:
    scenarios = [
        {
            "label": "danger_avoidance",
            "observation": {
                "food": 0.20,
                "danger": 0.72,
                "novelty": 0.20,
                "shelter": 0.12,
                "temperature": 0.48,
                "social": 0.18,
            },
            "body_state": {
                "energy": 0.30,
                "stress": 0.60,
                "fatigue": 0.30,
                "temperature": 0.48,
            },
            "original_action": "forage",
            "counterfactual_action": "hide",
        },
        {
            "label": "resource_risk",
            "observation": {
                "food": 0.12,
                "danger": 0.22,
                "novelty": 0.30,
                "shelter": 0.20,
                "temperature": 0.50,
                "social": 0.20,
            },
            "body_state": {
                "energy": 0.18,
                "stress": 0.35,
                "fatigue": 0.28,
                "temperature": 0.50,
            },
            "original_action": "hide",
            "counterfactual_action": "forage",
        },
        {
            "label": "retreat_vs_explore",
            "observation": {
                "food": 0.50,
                "danger": 0.32,
                "novelty": 0.72,
                "shelter": 0.18,
                "temperature": 0.49,
                "social": 0.18,
            },
            "body_state": {
                "energy": 0.28,
                "stress": 0.65,
                "fatigue": 0.36,
                "temperature": 0.49,
            },
            "original_action": "scan",
            "counterfactual_action": "rest",
        },
        {
            "label": "integrity_preservation",
            "observation": {
                "food": 0.35,
                "danger": 0.28,
                "novelty": 0.22,
                "shelter": 0.30,
                "temperature": 0.24,
                "social": 0.18,
            },
            "body_state": {
                "energy": 0.24,
                "stress": 0.72,
                "fatigue": 0.52,
                "temperature": 0.24,
            },
            "original_action": "seek_contact",
            "counterfactual_action": "thermoregulate",
        },
    ]
    all_logs: list[dict[str, object]] = []
    scenario_results: list[dict[str, object]] = []
    preference_model = PreferenceModel()
    for index, scenario in enumerate(scenarios, start=1):
        world_model = GenerativeWorldModel()
        absorber = InsightAbsorber()
        first = CounterfactualInsight(
            source_episode_cycle=index,
            original_action=scenario["original_action"],
            counterfactual_action=scenario["counterfactual_action"],
            original_efe=5.0,
            counterfactual_efe=1.0,
            efe_delta=-4.0,
            confidence=0.66,
            state_context={
                "observation": dict(scenario["observation"]),
                "body_state": dict(scenario["body_state"]),
            },
            cluster_id=0,
            timestamp=10,
        )
        second = CounterfactualInsight(
            source_episode_cycle=index,
            original_action=scenario["original_action"],
            counterfactual_action=scenario["counterfactual_action"],
            original_efe=5.1,
            counterfactual_efe=1.0,
            efe_delta=-4.1,
            confidence=0.68,
            state_context={
                "observation": dict(scenario["observation"]),
                "body_state": dict(scenario["body_state"]),
            },
            cluster_id=0,
            timestamp=11,
        )
        absorber.absorb([first], world_model, preference_model=preference_model)
        absorber.absorb([second], world_model, preference_model=preference_model)
        coverage = compute_family_coverage(absorber.log)
        review_entry = next(
            (entry for entry in absorber.log if entry.get("type") == "candidate_review"),
            {},
        )
        scenario_results.append(
            {
                "scenario_label": scenario["label"],
                "review_family": review_entry.get("review_family"),
                "graduated": any(entry.get("type") == "absorption" for entry in absorber.log),
                "average_benefit": review_entry.get("average_benefit"),
                "pass_rate": review_entry.get("pass_rate"),
                "coverage": coverage,
            }
        )
        all_logs.extend(absorber.log)
    coverage = compute_family_coverage(all_logs)
    runtime_validated_family_count = int(coverage["families_graduated"])
    if runtime_validated_family_count >= 2:
        status = "RUNTIME_DIVERSITY_VALIDATED"
    elif runtime_validated_family_count >= 1:
        status = "LIMITED_RUNTIME_VALIDATED"
    else:
        status = "FRAMEWORK_ONLY"
    return {
        "verification_status": (
            "REVALIDATED_THIS_ROUND"
            if runtime_validated_family_count >= 2
            else "FRAMEWORK_IMPLEMENTED_RUNTIME_NOT_REVALIDATED"
        ),
        "evidence_origin": f"{GENERATOR_PATH}:runtime_family_probe",
        "revalidated_this_round": True,
        "family_schema_count": len(DEFAULT_REVIEW_FAMILIES),
        "runtime_validated_family_count": runtime_validated_family_count,
        "family_coverage_status": status,
        "family_probe_results": scenario_results,
        "coverage_summary": coverage,
    }


def derive_controlled_ready_status(
    strict_metrics: dict[str, dict[str, object]],
    family_coverage: dict[str, object],
    test_evidence: dict[str, object],
) -> ReadinessStatus:
    reasons: list[str] = []
    for name, entry in strict_metrics.items():
        if not bool(entry.get("passes_threshold")):
            reasons.append(f"strict gating metric {name} is below threshold")
        if not bool(entry.get("strict_metric")) or not bool(entry.get("gating_metric")):
            reasons.append(f"strict gating metric {name} has invalid gating metadata")
        if bool(entry.get("legacy_metric")):
            reasons.append(f"strict gating metric {name} is incorrectly marked legacy")
    if family_coverage.get("family_coverage_status") != "RUNTIME_DIVERSITY_VALIDATED":
        reasons.append("runtime review family diversity has not been broadly validated")
    if test_evidence.get("verification_status") != "REVALIDATED_THIS_ROUND":
        reasons.append("full readiness pass-count claim is not revalidated this round")

    if any("below threshold" in reason or "invalid gating metadata" in reason for reason in reasons):
        return ReadinessStatus("NOT_VERIFIED", reasons)
    if reasons:
        return ReadinessStatus("CONTROLLED_READY_CANDIDATE", reasons)
    return ReadinessStatus("CONTROLLED_READY_VERIFIED", [])


def derive_open_ready_status(
    controlled_ready_status: ReadinessStatus,
    family_coverage: dict[str, object],
    lifecycle_evidence: dict[str, object],
) -> ReadinessStatus:
    reasons: list[str] = []
    if controlled_ready_status.status != "CONTROLLED_READY_VERIFIED":
        reasons.append("controlled readiness is not verified in the current round")
    if lifecycle_evidence.get("verification_status") != "REVALIDATED_THIS_ROUND":
        reasons.append("lifecycle evidence is not revalidated this round")
    if not lifecycle_evidence.get("lifecycle_activity_observed"):
        reasons.append("current strict evidence does not show lifecycle activity in this readiness round")
    if not lifecycle_evidence.get("compression_specifically_verified"):
        reasons.append("compression has not been specifically verified in the current readiness round")
    if lifecycle_evidence.get("compressed_cluster_count") in (None, 0):
        reasons.append("compressed cluster count is not currently validated")
    if family_coverage.get("runtime_validated_family_count", 0) < 2:
        reasons.append("runtime family validation breadth is insufficient for open-readiness claims")
    if reasons:
        return ReadinessStatus("NOT_VERIFIED", reasons)
    return ReadinessStatus("OPEN_READY_VERIFIED", [])


def derive_final_recommendation(
    controlled_ready_status: ReadinessStatus,
    open_ready_status: ReadinessStatus,
    family_coverage: dict[str, object],
    test_evidence: dict[str, object],
) -> dict[str, object]:
    if open_ready_status.status == "OPEN_READY_VERIFIED":
        return {
            "status": "OPEN_READY_VERIFIED",
            "rationale": "Controlled readiness is verified and lifecycle/compression runtime evidence is revalidated this round.",
            "why_more_conservative": "",
        }
    if controlled_ready_status.status == "CONTROLLED_READY_VERIFIED":
        return {
            "status": "CONTROLLED_READY_VERIFIED",
            "rationale": "Strict gating metrics pass and readiness evidence is fully revalidated.",
            "why_more_conservative": "",
        }
    reasons = list(controlled_ready_status.reasons)
    if family_coverage.get("runtime_validated_family_count", 0) == 0:
        reasons.append("review family framework exists, but runtime-validated family coverage is still effectively unverified")
    if test_evidence.get("verification_status") == "CLAIMED_BUT_NOT_REVALIDATED":
        reasons.append("previous full-suite pass counts are carried forward as claims, not current-round facts")
    return {
        "status": "RECOMMEND_M3_WITH_CAUTION",
        "rationale": "Strict M2 follow-up metrics remain strong, but readiness evidence is incomplete for a verified CONTROLLED_READY upgrade.",
        "why_more_conservative": "; ".join(dict.fromkeys(reasons)),
    }


def build_readiness_payload(
    *,
    followup_payload: dict[str, object],
    legacy_payload: dict[str, object],
    previous_readiness: dict[str, object] | None = None,
    test_evidence: dict[str, object] | None = None,
    family_coverage: dict[str, object] | None = None,
    lifecycle_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    strict_metrics = derive_strict_metrics(followup_payload)
    legacy_metrics = derive_legacy_metrics(legacy_payload)
    test_evidence = test_evidence or derive_claimed_test_evidence(previous_readiness or {})
    lifecycle_evidence = lifecycle_evidence or derive_lifecycle_evidence(followup_payload)
    family_coverage = family_coverage or evaluate_runtime_family_coverage()
    controlled_ready_status = derive_controlled_ready_status(
        strict_metrics,
        family_coverage,
        test_evidence,
    )
    open_ready_status = derive_open_ready_status(
        controlled_ready_status,
        family_coverage,
        lifecycle_evidence,
    )
    final_recommendation = derive_final_recommendation(
        controlled_ready_status,
        open_ready_status,
        family_coverage,
        test_evidence,
    )
    return {
        "generated_at": str(date.today()),
        "generator_path": GENERATOR_PATH,
        "strict_metrics": strict_metrics,
        "legacy_metrics": legacy_metrics,
        "gating_metrics": list(STRICT_THRESHOLDS),
        "non_gating_metrics": [f"legacy::{name}" for name in sorted(legacy_metrics)],
        "verification_status": {
            "strict_baseline_source": "reports/m2_followup_metrics.json",
            "legacy_comparison_source": "reports/m2_metrics.json",
            "readiness_generator_rebuilt_this_round": True,
        },
        "evidence_origin": {
            "strict_metrics": "reports/m2_followup_metrics.json:new_metrics",
            "legacy_metrics": "reports/m2_metrics.json:metrics",
            "lifecycle": lifecycle_evidence["evidence_origin"],
            "family_coverage": family_coverage["evidence_origin"],
            "tests": test_evidence["evidence_origin"],
        },
        "revalidated_this_round": {
            "strict_metrics": False,
            "legacy_metrics": False,
            "lifecycle": lifecycle_evidence["revalidated_this_round"],
            "family_schema": family_coverage["revalidated_this_round"],
            "tests": test_evidence["revalidated_this_round"],
        },
        "inherited_from_followup": {
            name: entry["inherited_from_followup"] for name, entry in strict_metrics.items()
        },
        "claimed_not_revalidated": {
            "tests": (
                test_evidence
                if test_evidence.get("verification_status") != "REVALIDATED_THIS_ROUND"
                else None
            ),
            "compressed_cluster_count": lifecycle_evidence["compressed_cluster_count_status"] == "CLAIMED_NOT_REPLAYED",
        },
        "current_round_test_evidence": test_evidence,
        "lifecycle_evidence": lifecycle_evidence,
        "family_coverage": family_coverage,
        "controlled_ready_status": controlled_ready_status.to_dict(),
        "open_ready_status": open_ready_status.to_dict(),
        "final_recommendation": final_recommendation,
    }


def build_readiness_report(payload: dict[str, object]) -> str:
    strict_metrics = payload["strict_metrics"]
    legacy_metrics = payload["legacy_metrics"]
    lifecycle = payload["lifecycle_evidence"]
    family = payload["family_coverage"]
    controlled = payload["controlled_ready_status"]
    final_recommendation = payload["final_recommendation"]
    current_tests = payload["current_round_test_evidence"]
    claimed_tests = payload["claimed_not_revalidated"]["tests"]
    next_evidence_lines = [
        "- Keep runtime lifecycle probes in the readiness generator so OPEN readiness remains reproducible from current-round evidence.",
    ]
    if payload["open_ready_status"]["status"] != "OPEN_READY_VERIFIED":
        next_evidence_lines.insert(
            0,
            "- Produce current-round lifecycle activity evidence if open-readiness claims are needed beyond controlled readiness.",
        )
    lines = [
        "# M3 Readiness Repair Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Generator: `{payload['generator_path']}`",
        "",
        "## 1. Audit Problem Summary",
        "",
        "- Strict follow-up metrics are now the default readiness baseline; legacy M2 metrics are retained only for comparison.",
        "- Readiness conclusions are downgraded when evidence is inherited, claimed, or not replayed in the current round.",
        "- Compression claims are separated from generic lifecycle activity.",
        "- Review-family schema existence is separated from runtime-validated family coverage.",
        "",
        "## 2. Strict vs Legacy Metric Inheritance",
        "",
        "Strict metrics",
    ]
    for name, entry in strict_metrics.items():
        lines.append(
            f"- `{name}` = {entry['value']:.4f} "
            f"(status={entry['verification_status']}, origin={entry['evidence_origin']}, "
            f"gating={entry['gating_metric']})"
        )
    lines.extend(["", "Legacy metrics (non-gating, comparison only)"])
    for name, entry in legacy_metrics.items():
        lines.append(
            f"- `{name}` = {entry['value']:.4f} "
            f"(status={entry['verification_status']}, non_gating={entry['non_gating']})"
        )
    lines.extend([
        "",
        "## 3. Evidence Status And Downgrade Rules",
        "",
        "- `INHERITED_STRICT_BASELINE`: carried from the stricter M2 follow-up run and allowed for readiness gating.",
        "- `LEGACY_FOR_COMPARISON_ONLY`: retained only as historical comparison and never used for gating.",
        "- `CLAIMED_BUT_NOT_REVALIDATED`: visible in the artifact, but not upgraded to verified fact in this round.",
        "- Automatic downgrade triggers: strict/legacy mismatch in gating, missing current-round test evidence, and insufficient runtime family validation breadth.",
        "",
        "## 4. Compression vs Lifecycle Evidence",
        "",
        f"- `lifecycle_verification_status`: {lifecycle['verification_status']}",
        f"- `lifecycle_activity_observed`: {str(bool(lifecycle['lifecycle_activity_observed'])).lower()}",
        f"- `compression_specifically_verified`: {str(bool(lifecycle['compression_specifically_verified'])).lower()}",
        f"- `compression_removed_count`: {lifecycle['compression_removed_count']}",
        f"- `archived_count`: {lifecycle['archived_count']}",
        f"- `pruned_count`: {lifecycle['pruned_count']}",
        f"- `compressed_cluster_count`: {lifecycle['compressed_cluster_count']}",
        f"- `probe_results_recorded`: {len(lifecycle.get('probe_results', []))}",
        "- Report rule: memory reduction is not treated as compression verification unless `compression_removed_count > 0` and `compressed_cluster_count > 0`.",
        "",
        "## 5. Family Coverage Boundary",
        "",
        f"- `family_schema_count`: {family['family_schema_count']}",
        f"- `runtime_validated_family_count`: {family['runtime_validated_family_count']}",
        f"- `family_coverage_status`: {family['family_coverage_status']}",
        "- Report rule: schema implemented != runtime coverage verified.",
        "",
        "## 6. Current-Round Test Evidence",
        "",
        f"- Test evidence status: {current_tests.get('verification_status')}",
        f"- Current-round passed count: {current_tests.get('passed_count')}",
        f"- Current-round failed count: {current_tests.get('failed_count')}",
        f"- Current-round summary: {current_tests.get('output_summary')}",
        f"- Carried-forward unverified test claim present: {str(claimed_tests is not None).lower()}",
        "",
        "## 7. Final Readiness Conclusion",
        "",
        f"- `controlled_ready_status`: {controlled['status']}",
        f"- `open_ready_status`: {payload['open_ready_status']['status']}",
        f"- `final_recommendation`: {final_recommendation['status']}",
        f"- Rationale: {final_recommendation['rationale']}",
        f"- Why this is more conservative: {final_recommendation['why_more_conservative'] or 'n/a'}",
        "",
        "## 8. Next Evidence Needed",
        "",
        *next_evidence_lines,
    ])
    return "\n".join(lines) + "\n"


def write_readiness_outputs(
    *,
    metrics_path: Path = READINESS_METRICS_PATH,
    report_path: Path = READINESS_REPORT_PATH,
    test_evidence: dict[str, object] | None = None,
    family_coverage: dict[str, object] | None = None,
    lifecycle_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    previous_readiness = load_json(metrics_path)
    current_round_test_evidence = test_evidence or execute_readiness_test_suite()
    current_round_lifecycle_evidence = lifecycle_evidence or evaluate_runtime_lifecycle_evidence()
    payload = build_readiness_payload(
        followup_payload=load_json(FOLLOWUP_METRICS_PATH),
        legacy_payload=load_json(LEGACY_METRICS_PATH),
        previous_readiness=previous_readiness,
        test_evidence=current_round_test_evidence,
        family_coverage=family_coverage,
        lifecycle_evidence=current_round_lifecycle_evidence,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    report_path.write_text(build_readiness_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    payload = write_readiness_outputs()
    print(json.dumps(payload["final_recommendation"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
