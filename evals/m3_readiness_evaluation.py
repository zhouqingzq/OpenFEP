from __future__ import annotations

import json
import re
import random
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.counterfactual import DEFAULT_REVIEW_FAMILIES
from segmentum.counterfactual import compute_family_coverage, run_counterfactual_phase
from segmentum.action_schema import action_name
from segmentum.agent import SegmentAgent
from evals.m2_followup_repair import FOLLOWUP_SEED_SET, run_followup_evaluation

FOLLOWUP_METRICS_PATH = ROOT / "reports" / "m2_followup_metrics.json"
LEGACY_METRICS_PATH = ROOT / "reports" / "m2_metrics.json"
READINESS_METRICS_PATH = ROOT / "reports" / "m3_readiness_metrics.json"
READINESS_REPORT_PATH = ROOT / "reports" / "m3_readiness_repair_report.md"
PRE_M3_READINESS_PATH = ROOT / "artifacts" / "pre_m3_readiness_report.json"
PRE_M3_REGRESSION_PATH = ROOT / "artifacts" / "pre_m3_regression_summary.json"
GENERATOR_PATH = "evals/m3_readiness_evaluation.py"
READINESS_TEST_TARGETS = [
    "tests/test_m3_readiness.py",
    "tests/test_memory.py",
    "tests/test_counterfactual_artifact.py",
    "tests/test_m23_ultimate_consolidation_loop.py",
]
HISTORICAL_REGRESSION_CHECKS = [
    "soak_regression_passed",
    "snapshot_compatibility_passed",
    "runtime_lifecycle_passed",
    "runtime_family_coverage_passed",
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


def _generated_this_round(timestamp: object) -> bool:
    if not isinstance(timestamp, str) or not timestamp:
        return False
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()


def _artifact_path(payload: dict[str, object]) -> str | None:
    path = payload.get("_artifact_path")
    if not isinstance(path, str):
        return None
    return path


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
    current_round_replay = bool(followup_payload.get("_current_round_replay"))
    evidence_origin = str(
        followup_payload.get("_evidence_origin", "reports/m2_followup_metrics.json:new_metrics")
    )
    verification_status = (
        "REVALIDATED_THIS_ROUND" if current_round_replay else "INHERITED_STRICT_BASELINE"
    )
    strict: dict[str, dict[str, object]] = {}
    for name, threshold in STRICT_THRESHOLDS.items():
        strict[name] = metric_entry(
            name=name,
            value=followup_metrics.get(name),
            threshold=threshold,
            verification_status=verification_status,
            evidence_origin=evidence_origin,
            strict_metric=True,
            gating_metric=True,
            revalidated_this_round=current_round_replay,
            inherited_from_followup=not current_round_replay,
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
        "suite_type": "readiness_targets",
        "suite_scope": "targeted_readiness_tests_only",
        "verification_status": verification_status,
        "evidence_origin": f"{GENERATOR_PATH}:pytest",
        "revalidated_this_round": completed.returncode == 0,
        "command": command,
        "targets": list(READINESS_TEST_TARGETS),
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
    prune_agent.long_term_memory.compression_similarity_threshold = 1.1
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
    prune_agent.long_term_memory.assign_clusters()
    prune_replay_batch = list(prune_agent.long_term_memory.episodes)
    for payload in prune_replay_batch:
        cluster_id = payload.get("cluster_id")
        if isinstance(cluster_id, int):
            prune_agent.world_model.set_outcome_distribution(
                cluster_id,
                action_name(payload.get("action_taken", payload.get("action", ""))),
                {str(payload.get("predicted_outcome", "neutral")): 1.0},
            )
    _prune_archived, prune_deleted = prune_agent._surprise_based_forgetting(prune_replay_batch)

    archive_agent = SegmentAgent(rng=random.Random(41))
    archive_agent.cycle = 50
    archive_agent.long_term_memory.max_active_age = 5
    archive_agent.long_term_memory.minimum_active_episodes = 0
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
    pruned_count_total = int(compression_summary.episodes_deleted) + int(prune_deleted) + int(deleted_count)
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
                "compression_removed": 0,
                "compressed_cluster_count": 0,
                "episodes_archived": 0,
                "episodes_deleted": int(prune_deleted),
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
            "original_action": "forage",
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
            "outcome": {
                "energy_delta": -0.10,
                "stress_delta": 0.28,
                "fatigue_delta": 0.18,
                "temperature_delta": 0.02,
                "free_energy_drop": -0.45,
            },
        },
        {
            "label": "retreat_vs_explore",
            "original_action": "scan",
            "observation": {
                "food": 0.45,
                "danger": 0.40,
                "novelty": 0.90,
                "shelter": 0.05,
                "temperature": 0.49,
                "social": 0.10,
            },
            "body_state": {
                "energy": 0.15,
                "stress": 0.80,
                "fatigue": 0.50,
                "temperature": 0.49,
            },
            "outcome": {
                "energy_delta": -0.18,
                "stress_delta": 0.18,
                "fatigue_delta": 0.12,
                "temperature_delta": 0.0,
                "free_energy_drop": -0.42,
            },
        },
        {
            "label": "integrity_preservation",
            "original_action": "seek_contact",
            "observation": {
                "food": 0.40,
                "danger": 0.20,
                "novelty": 0.25,
                "shelter": 0.15,
                "temperature": 0.10,
                "social": 0.05,
            },
            "body_state": {
                "energy": 0.20,
                "stress": 0.85,
                "fatigue": 0.60,
                "temperature": 0.10,
            },
            "outcome": {
                "energy_delta": -0.14,
                "stress_delta": 0.20,
                "fatigue_delta": 0.14,
                "temperature_delta": -0.05,
                "free_energy_drop": -0.50,
            },
        },
        {
            "label": "resource_risk",
            "original_action": "scan",
            "observation": {
                "food": 0.00,
                "danger": 0.15,
                "novelty": 0.25,
                "shelter": 0.10,
                "temperature": 0.50,
                "social": 0.12,
            },
            "body_state": {
                "energy": 0.08,
                "stress": 0.55,
                "fatigue": 0.45,
                "temperature": 0.50,
            },
            "outcome": {
                "energy_delta": -0.15,
                "stress_delta": 0.15,
                "fatigue_delta": 0.12,
                "temperature_delta": 0.0,
                "free_energy_drop": -0.40,
            },
        },
    ]
    scenario_results: list[dict[str, object]] = []
    all_logs: list[dict[str, object]] = []
    for scenario in scenarios:
        agent = SegmentAgent(rng=random.Random(97))
        agent.energy = 0.95
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        observation = dict(scenario["observation"])
        prediction = {
            key: min(
                1.0,
                max(0.0, value + (0.25 if key in {"food", "shelter", "social"} else -0.25)),
            )
            for key, value in observation.items()
        }
        errors = {key: observation[key] - prediction[key] for key in observation}
        for cycle in range(1, 7):
            decision = agent.long_term_memory.maybe_store_episode(
                cycle=cycle,
                observation=observation,
                prediction=prediction,
                errors=errors,
                action=str(scenario["original_action"]),
                outcome=dict(scenario["outcome"]),
                body_state=dict(scenario["body_state"]),
            )
            if not decision.episode_created and decision.support_delta == 0:
                agent.long_term_memory.store_episode(
                    cycle=cycle,
                    observation=observation,
                    prediction=prediction,
                    errors=errors,
                    action=str(scenario["original_action"]),
                    outcome=dict(scenario["outcome"]),
                    body_state=dict(scenario["body_state"]),
                )
        agent.long_term_memory.assign_clusters()
        replay_batch = agent.long_term_memory.replay_during_sleep(rng=agent.rng)
        scenario_logs: list[dict[str, object]] = []
        for _ in range(2):
            _, summary = run_counterfactual_phase(
                agent_energy=agent.energy,
                current_cycle=agent.cycle,
                episodes=replay_batch,
                world_model=agent.world_model,
                preference_model=agent.long_term_memory.preference_model,
                action_registry=agent.action_registry,
                rng=agent.rng,
                surprise_threshold=0.10,
                max_depth=1,
                energy_budget=0.50,
            )
            scenario_logs.extend(summary.counterfactual_log)
        coverage = compute_family_coverage(scenario_logs)
        review_entries = [
            entry for entry in scenario_logs if entry.get("type") == "candidate_review"
        ]
        absorption_entries = [
            entry for entry in scenario_logs if entry.get("type") == "absorption"
        ]
        scenario_results.append(
            {
                "scenario_label": scenario["label"],
                "probe_type": "runtime_counterfactual_replay",
                "runtime_event_captured": bool(review_entries or absorption_entries),
                "review_families_seen": sorted(
                    {
                        str(entry.get("review_family"))
                        for entry in (*review_entries, *absorption_entries)
                        if entry.get("review_family")
                    }
                ),
                "candidate_review_count": len(review_entries),
                "absorption_count": len(absorption_entries),
                "coverage": coverage,
            }
        )
        all_logs.extend(scenario_logs)
    coverage = compute_family_coverage(all_logs)
    runtime_validated_family_count = int(coverage["families_graduated"])
    missing_graduation_families = [
        family.family_id
        for family in DEFAULT_REVIEW_FAMILIES
        if int(dict(coverage.get("per_family_rates", {})).get(family.family_id, {}).get("graduated", 0)) <= 0
    ]
    fully_graduated = runtime_validated_family_count == len(DEFAULT_REVIEW_FAMILIES)
    if fully_graduated:
        verification_status = "REVALIDATED_THIS_ROUND"
        coverage_status = "RUNTIME_DIVERSITY_VALIDATED"
    elif runtime_validated_family_count >= 1:
        verification_status = "LIMITED_RUNTIME_REPLAY"
        coverage_status = "PARTIAL_RUNTIME_VALIDATED"
    else:
        verification_status = "FAILED_THIS_ROUND"
        coverage_status = "FRAMEWORK_ONLY_NOT_RUNTIME_VALIDATED"
    return {
        "verification_status": verification_status,
        "evidence_origin": f"{GENERATOR_PATH}:runtime_counterfactual_replay",
        "revalidated_this_round": True,
        "evidence_kind": "runtime_replay",
        "family_schema_count": len(DEFAULT_REVIEW_FAMILIES),
        "runtime_validated_family_count": runtime_validated_family_count,
        "fully_graduated": fully_graduated,
        "missing_graduation_families": missing_graduation_families,
        "family_coverage_status": coverage_status,
        "family_probe_results": scenario_results,
        "coverage_summary": coverage,
        "limitations": [
            (
                "Runtime replay is current-round evidence, but the following families still lack real graduation: "
                + ", ".join(missing_graduation_families)
                + "."
            )
        ]
        if not fully_graduated
        else [],
    }


def derive_historical_regression_evidence(regression_payload: dict[str, object] | None) -> dict[str, object]:
    payload = regression_payload or {}
    artifact_path = _artifact_path(payload) or str(PRE_M3_REGRESSION_PATH.relative_to(ROOT))
    acceptance = dict(payload.get("acceptance", {}))
    checks = dict(acceptance.get("checks", {}))
    generated_at = payload.get("generated_at")
    generated_this_round = _generated_this_round(generated_at)
    all_required_checks_present = all(name in checks for name in HISTORICAL_REGRESSION_CHECKS)
    all_passed = all(bool(checks.get(name)) for name in HISTORICAL_REGRESSION_CHECKS)
    if not payload:
        verification_status = "NOT_PRESENT"
    elif generated_this_round and all_required_checks_present and all_passed:
        verification_status = "REVALIDATED_THIS_ROUND"
    elif all_passed:
        verification_status = "INHERITED_NOT_CURRENT_ROUND"
    else:
        verification_status = "FAILED_OR_INCOMPLETE"
    return {
        "suite_type": "historical_regressions",
        "suite_scope": "pre_m3_regression_suite",
        "verification_status": verification_status,
        "evidence_origin": artifact_path,
        "revalidated_this_round": generated_this_round and all_passed,
        "generated_at": generated_at,
        "required_checks": list(HISTORICAL_REGRESSION_CHECKS),
        "checks": checks,
        "passed": all_passed,
        "missing_checks": [name for name in HISTORICAL_REGRESSION_CHECKS if name not in checks],
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
        if not bool(entry.get("revalidated_this_round")):
            reasons.append(f"strict gating metric {name} is inherited rather than current-round verified")
    if family_coverage.get("family_coverage_status") != "RUNTIME_DIVERSITY_VALIDATED":
        reasons.append("runtime review family diversity has not been broadly validated")
    if family_coverage.get("evidence_kind") != "runtime_replay":
        reasons.append("review-family evidence is not derived from current-round runtime replay")
    missing_families = list(family_coverage.get("missing_graduation_families", []))
    if missing_families:
        reasons.append(
            "runtime replay still lacks graduation for: " + ", ".join(missing_families)
        )
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
    if list(family_coverage.get("missing_graduation_families", [])):
        reasons.append("runtime family coverage remains only partially graduated")
    if reasons:
        return ReadinessStatus("NOT_VERIFIED", reasons)
    return ReadinessStatus("OPEN_READY_VERIFIED", [])


def derive_final_recommendation(
    pre_m3_readiness: dict[str, object] | None,
    controlled_ready_status: ReadinessStatus,
    open_ready_status: ReadinessStatus,
    family_coverage: dict[str, object],
    test_evidence: dict[str, object],
    historical_regressions: dict[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    if not pre_m3_readiness:
        blockers.append("pre-M3 readiness artifact is missing")
    elif not bool(pre_m3_readiness.get("passed")):
        blockers.append("pre-M3 readiness gate did not pass in the current artifact")
    if test_evidence.get("verification_status") != "REVALIDATED_THIS_ROUND" or int(test_evidence.get("failed_count", 0)) > 0:
        blockers.append("current-round readiness target tests are not fully passing")
    if historical_regressions.get("verification_status") != "REVALIDATED_THIS_ROUND":
        blockers.append("historical regression evidence is not replayed and passing in the current round")
    if controlled_ready_status.status != "CONTROLLED_READY_VERIFIED":
        blockers.append("controlled readiness is not verified against current-round gating evidence")
    if open_ready_status.status != "OPEN_READY_VERIFIED":
        blockers.append("open readiness is not verified")
    if family_coverage.get("evidence_kind") != "runtime_replay":
        blockers.append("review-family coverage is only a framework/schema probe, not runtime replay evidence")
    if blockers:
        return {
            "status": "NOT_READY_FOR_M3",
            "rationale": "M3 admission stays blocked until all gating evidence is current-round, reproducible, and free of blocking audit gaps.",
            "why_more_conservative": "; ".join(dict.fromkeys(blockers)),
            "blocking_gaps": list(dict.fromkeys(blockers)),
        }
    if pre_m3_readiness:
        return {
            "status": "READY_FOR_M3",
            "rationale": "Pre-M3 gate passed and all readiness gates, runtime evidence, and historical regressions are verified in the current round.",
            "why_more_conservative": "",
            "blocking_gaps": [],
        }
    if open_ready_status.status == "OPEN_READY_VERIFIED":
        return {
            "status": "OPEN_READY_VERIFIED",
            "rationale": "Controlled readiness is verified and lifecycle/compression runtime evidence is revalidated this round.",
            "why_more_conservative": "",
            "blocking_gaps": [],
        }
    if controlled_ready_status.status == "CONTROLLED_READY_VERIFIED":
        return {
            "status": "CONTROLLED_READY_VERIFIED",
            "rationale": "Strict gating metrics pass and readiness evidence is fully revalidated.",
            "why_more_conservative": "",
            "blocking_gaps": [],
        }
    reasons = list(controlled_ready_status.reasons)
    if family_coverage.get("runtime_validated_family_count", 0) == 0:
        reasons.append("review family framework exists, but runtime-validated family coverage is still effectively unverified")
    if test_evidence.get("verification_status") == "CLAIMED_BUT_NOT_REVALIDATED":
        reasons.append("previous full-suite pass counts are carried forward as claims, not current-round facts")
    if historical_regressions.get("verification_status") != "REVALIDATED_THIS_ROUND":
        reasons.append("historical regressions are not replayed and passing in this round")
    return {
        "status": "RECOMMEND_M3_WITH_CAUTION",
        "rationale": "Strict M2 follow-up metrics remain strong, but readiness evidence is incomplete for a verified CONTROLLED_READY upgrade.",
        "why_more_conservative": "; ".join(dict.fromkeys(reasons)),
        "blocking_gaps": [],
    }


def build_readiness_payload(
    *,
    followup_payload: dict[str, object],
    legacy_payload: dict[str, object],
    previous_readiness: dict[str, object] | None = None,
    test_evidence: dict[str, object] | None = None,
    family_coverage: dict[str, object] | None = None,
    lifecycle_evidence: dict[str, object] | None = None,
    pre_m3_readiness: dict[str, object] | None = None,
    historical_regressions: dict[str, object] | None = None,
    strict_seed_set: list[int] | None = None,
) -> dict[str, object]:
    strict_metrics = derive_strict_metrics(followup_payload)
    legacy_metrics = derive_legacy_metrics(legacy_payload)
    test_evidence = test_evidence or derive_claimed_test_evidence(previous_readiness or {})
    lifecycle_evidence = lifecycle_evidence or derive_lifecycle_evidence(followup_payload)
    family_coverage = family_coverage or evaluate_runtime_family_coverage()
    historical_regressions = historical_regressions or derive_historical_regression_evidence({})
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
        pre_m3_readiness,
        controlled_ready_status,
        open_ready_status,
        family_coverage,
        test_evidence,
        historical_regressions,
    )
    strict_metrics_current_round = all(
        bool(entry.get("revalidated_this_round")) for entry in strict_metrics.values()
    )
    strict_metrics_origin = next(iter(strict_metrics.values()))["evidence_origin"]
    tests = {
        "readiness_targets": {
            "suite_type": test_evidence.get("suite_type", "readiness_targets"),
            "suite_scope": test_evidence.get("suite_scope", "targeted_readiness_tests_only"),
            "verification_status": test_evidence.get("verification_status"),
            "revalidated_this_round": bool(test_evidence.get("revalidated_this_round")),
            "targets": list(test_evidence.get("targets", [])),
            "target_count": test_evidence.get("target_count"),
            "passed_count": test_evidence.get("passed_count"),
            "failed_count": test_evidence.get("failed_count"),
            "output_summary": test_evidence.get("output_summary"),
            "command": test_evidence.get("command"),
        },
        "historical_regressions": historical_regressions,
        "coverage_boundary": {
            "readiness_targets_are_full_regression": False,
            "note": "The 4 readiness-target test files are scoped smoke/readiness checks and do not substitute for historical milestone regression replay.",
        },
    }
    gates = {
        "current_round_gating_tests_passed": (
            test_evidence.get("verification_status") == "REVALIDATED_THIS_ROUND"
            and int(test_evidence.get("failed_count", 0)) == 0
        ),
        "strict_metrics_current_round_replayed": strict_metrics_current_round,
        "runtime_family_replay_verified": family_coverage.get("evidence_kind") == "runtime_replay"
        and family_coverage.get("family_coverage_status") == "RUNTIME_DIVERSITY_VALIDATED"
        and bool(family_coverage.get("fully_graduated")),
        "historical_regressions_current_round_replayed": historical_regressions.get("verification_status")
        == "REVALIDATED_THIS_ROUND",
        "controlled_ready_verified": controlled_ready_status.status == "CONTROLLED_READY_VERIFIED",
        "open_ready_verified": open_ready_status.status == "OPEN_READY_VERIFIED",
        "pre_m3_gate_passed": bool(pre_m3_readiness and pre_m3_readiness.get("passed")),
    }
    findings: list[dict[str, object]] = []
    if family_coverage.get("evidence_kind") != "runtime_replay":
        findings.append(
            {
                "severity": "S1",
                "title": "Runtime family coverage is not backed by runtime replay",
                "details": "Current family coverage evidence is a framework/schema probe only, so it cannot satisfy runtime diversity gating.",
                "blocking": True,
            }
        )
    elif list(family_coverage.get("missing_graduation_families", [])):
        findings.append(
            {
                "severity": "S1",
                "title": "Runtime family replay is still missing family graduation",
                "details": (
                    "Current-round runtime replay exists, but these families still lack real graduation: "
                    + ", ".join(list(family_coverage.get("missing_graduation_families", [])))
                    + "."
                ),
                "blocking": True,
            }
        )
    if not gates["strict_metrics_current_round_replayed"]:
        findings.append(
            {
                "severity": "S1",
                "title": "Strict metrics remain inherited",
                "details": "Strict gating metrics still come from inherited follow-up artifacts instead of current-round replay.",
                "blocking": True,
            }
        )
    if historical_regressions.get("verification_status") != "REVALIDATED_THIS_ROUND":
        findings.append(
            {
                "severity": "S1",
                "title": "Historical regressions are not current-round replay evidence",
                "details": "Readiness target tests are distinct from historical regressions and cannot be used as a substitute.",
                "blocking": True,
            }
        )
    residual_risks: list[str] = []
    if not gates["strict_metrics_current_round_replayed"]:
        residual_risks.append(
            "Current strict metrics are inherited from the prior follow-up artifact and remain non-replay evidence for M3 admission."
        )
    if not gates["runtime_family_replay_verified"]:
        missing_families = list(family_coverage.get("missing_graduation_families", []))
        if missing_families:
            residual_risks.append(
                "Runtime family coverage is current-round replay evidence, but these families still have no real graduation: "
                + ", ".join(missing_families)
                + "."
            )
        else:
            residual_risks.append(
                "Runtime family coverage has not yet reached validated breadth across all required families."
            )
    freshness = {
        "strict_metrics": {
            "classification": "current_round_replay" if strict_metrics_current_round else "inherited",
            "gating": True,
            "generated_this_round": strict_metrics_current_round,
            "origin": strict_metrics_origin,
        },
        "legacy_metrics": {
            "classification": "inherited",
            "gating": False,
            "generated_this_round": False,
            "origin": "reports/m2_metrics.json:metrics",
        },
        "readiness_target_tests": {
            "classification": "current_round_replay" if test_evidence.get("revalidated_this_round") else "not_current_round",
            "gating": True,
            "generated_this_round": bool(test_evidence.get("revalidated_this_round")),
            "origin": test_evidence.get("evidence_origin"),
        },
        "historical_regressions": {
            "classification": (
                "current_round_replay"
                if historical_regressions.get("verification_status") == "REVALIDATED_THIS_ROUND"
                else "inherited"
            ),
            "gating": True,
            "generated_this_round": bool(historical_regressions.get("revalidated_this_round")),
            "origin": historical_regressions.get("evidence_origin"),
        },
        "lifecycle_runtime_probe": {
            "classification": (
                "current_round_replay" if lifecycle_evidence.get("revalidated_this_round") else "inherited"
            ),
            "gating": True,
            "generated_this_round": bool(lifecycle_evidence.get("revalidated_this_round")),
            "origin": lifecycle_evidence.get("evidence_origin"),
        },
        "family_coverage": {
            "classification": (
                "current_round_replay"
                if family_coverage.get("evidence_kind") == "runtime_replay"
                else "framework_schema_probe"
            ),
            "gating": True,
            "generated_this_round": bool(family_coverage.get("revalidated_this_round")),
            "origin": family_coverage.get("evidence_origin"),
        },
    }
    return {
        "milestone_id": "Pre-M3",
        "status": "PASS" if final_recommendation["status"] == "READY_FOR_M3" else "BLOCKED",
        "generated_at": str(date.today()),
        "generator_path": GENERATOR_PATH,
        "seed_set": list(dict.fromkeys((strict_seed_set or []) + [17, 23, 29, 41, 44, 91])),
        "artifacts": [
            {"path": str(FOLLOWUP_METRICS_PATH.relative_to(ROOT)), "kind": "strict_metrics", "gating": True},
            {"path": str(LEGACY_METRICS_PATH.relative_to(ROOT)), "kind": "legacy_metrics", "gating": False},
            {"path": str(PRE_M3_READINESS_PATH.relative_to(ROOT)), "kind": "pre_m3_readiness", "gating": True},
            {"path": str(PRE_M3_REGRESSION_PATH.relative_to(ROOT)), "kind": "historical_regressions", "gating": True},
        ],
        "tests": tests,
        "gates": gates,
        "findings": findings,
        "residual_risks": residual_risks,
        "freshness": freshness,
        "recommendation": final_recommendation,
        "strict_metrics": strict_metrics,
        "legacy_metrics": legacy_metrics,
        "gating_metrics": list(STRICT_THRESHOLDS),
        "non_gating_metrics": [f"legacy::{name}" for name in sorted(legacy_metrics)],
        "verification_status": {
            "strict_baseline_source": strict_metrics_origin,
            "legacy_comparison_source": "reports/m2_metrics.json",
            "readiness_generator_rebuilt_this_round": True,
        },
        "evidence_origin": {
            "strict_metrics": strict_metrics_origin,
            "legacy_metrics": "reports/m2_metrics.json:metrics",
            "lifecycle": lifecycle_evidence["evidence_origin"],
            "family_coverage": family_coverage["evidence_origin"],
            "tests": test_evidence["evidence_origin"],
        },
        "revalidated_this_round": {
            "strict_metrics": strict_metrics_current_round,
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
        "historical_regressions": historical_regressions,
        "lifecycle_evidence": lifecycle_evidence,
        "family_coverage": family_coverage,
        "pre_m3_readiness": pre_m3_readiness,
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
    historical_regressions = payload["historical_regressions"]
    claimed_tests = payload["claimed_not_revalidated"]["tests"]
    pre_m3_readiness = payload.get("pre_m3_readiness") or {}
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
        f"- `evidence_kind`: {family.get('evidence_kind')}",
        f"- `fully_graduated`: {str(bool(family.get('fully_graduated'))).lower()}",
        f"- `missing_graduation_families`: {', '.join(family.get('missing_graduation_families', [])) or 'none'}",
        f"- `limitations`: {'; '.join(family.get('limitations', [])) or 'none'}",
        "- Report rule: schema implemented != runtime coverage verified, and framework probes cannot satisfy runtime replay gates.",
        "",
        "## 6. Current-Round Test Evidence",
        "",
        f"- Test evidence status: {current_tests.get('verification_status')}",
        f"- Test suite scope: {current_tests.get('suite_scope')}",
        f"- Readiness targets: {', '.join(current_tests.get('targets', []))}",
        f"- Current-round passed count: {current_tests.get('passed_count')}",
        f"- Current-round failed count: {current_tests.get('failed_count')}",
        f"- Current-round summary: {current_tests.get('output_summary')}",
        f"- Carried-forward unverified test claim present: {str(claimed_tests is not None).lower()}",
        f"- Historical regressions status: {historical_regressions.get('verification_status')}",
        f"- Historical regression checks: {', '.join(historical_regressions.get('required_checks', []))}",
        "- Boundary: readiness targets are partial readiness checks, not a complete historical milestone regression proof.",
        "",
        "## 7. Final Readiness Conclusion",
        "",
        f"- `pre_m3_gate_status`: {pre_m3_readiness.get('status', 'NOT_PROVIDED')}",
        f"- `pre_m3_gate_passed`: {pre_m3_readiness.get('passed')}",
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
    followup_payload: dict[str, object] | None = None,
    test_evidence: dict[str, object] | None = None,
    family_coverage: dict[str, object] | None = None,
    lifecycle_evidence: dict[str, object] | None = None,
    historical_regressions: dict[str, object] | None = None,
) -> dict[str, object]:
    previous_readiness = load_json(metrics_path)
    current_followup_payload = followup_payload or run_followup_evaluation()
    current_followup_payload["_current_round_replay"] = True
    current_followup_payload["_evidence_origin"] = "evals/m2_followup_repair.py:run_followup_evaluation"
    current_round_test_evidence = test_evidence or execute_readiness_test_suite()
    current_round_lifecycle_evidence = lifecycle_evidence or evaluate_runtime_lifecycle_evidence()
    pre_m3_payload = load_json(PRE_M3_READINESS_PATH)
    pre_m3_regression_payload = load_json(PRE_M3_REGRESSION_PATH)
    if pre_m3_regression_payload:
        pre_m3_regression_payload["_artifact_path"] = str(PRE_M3_REGRESSION_PATH.relative_to(ROOT))
    pre_m3_readiness = None
    if pre_m3_payload:
        final = dict(pre_m3_payload.get("recommendation", pre_m3_payload.get("final_recommendation", {})))
        pre_m3_readiness = {
            "status": final.get("status"),
            "passed": bool(final.get("passed")),
            "evidence_origin": str(PRE_M3_READINESS_PATH.relative_to(ROOT)),
            "revalidated_this_round": _generated_this_round(pre_m3_payload.get("generated_at")),
            "seed_set": list(pre_m3_payload.get("seed_set", [])),
        }
    payload = build_readiness_payload(
        followup_payload=current_followup_payload,
        legacy_payload=load_json(LEGACY_METRICS_PATH),
        previous_readiness=previous_readiness,
        test_evidence=current_round_test_evidence,
        family_coverage=family_coverage,
        lifecycle_evidence=current_round_lifecycle_evidence,
        pre_m3_readiness=pre_m3_readiness,
        historical_regressions=historical_regressions or derive_historical_regression_evidence(pre_m3_regression_payload),
        strict_seed_set=list(current_followup_payload.get("seed_set", FOLLOWUP_SEED_SET))
        + list((pre_m3_readiness or {}).get("seed_set", [])),
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
