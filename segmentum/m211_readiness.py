from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evals.m3_readiness_evaluation import (
    evaluate_runtime_family_coverage,
    evaluate_runtime_lifecycle_evidence,
)
from .m210_benchmarks import run_longitudinal_stability, run_personality_validation
from .m29_benchmarks import run_transfer_acceptance_suite
from .runtime import SegmentRuntime


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
SCHEMA_VERSION = "pre_m3_v1"


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def codebase_version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _common_schema(
    *,
    benchmark_id: str,
    seed: int | None,
    cycles: int | None,
    world_id: str | None,
    world_pair: str | None,
    attention_enabled: bool | None,
    profile: str | None,
    metrics: dict[str, object],
    summary: dict[str, object],
    acceptance: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "seed": seed,
        "world_id": world_id,
        "world_pair": world_pair,
        "cycles": cycles,
        "attention_enabled": attention_enabled,
        "profile": profile,
        "metrics": metrics,
        "summary": summary,
        "acceptance": acceptance,
        "generated_at": _generated_at(),
        "codebase_version": codebase_version(),
    }


def build_attention_summary() -> dict[str, object]:
    from scripts.run_m28_attention_benchmark import run_attention_benchmark

    payload = run_attention_benchmark(seed=44, cycles=80)
    conditions = dict(payload["conditions"])
    evaluation = dict(payload["evaluation"])
    return _common_schema(
        benchmark_id="attention",
        seed=int(payload["seed"]),
        cycles=int(payload["cycles"]),
        world_id="simulated_world",
        world_pair=None,
        attention_enabled=True,
        profile=None,
        metrics={
            "conditioned_prediction_error_improvement": evaluation["conditioned_prediction_error_improvement"],
            "survival_ratio": evaluation["survival_ratio"],
            "topk_hit_rate": evaluation["topk_hit_rate"],
        },
        summary={
            "conditions": conditions,
            "acceptance_checks": dict(evaluation["acceptance"]),
        },
        acceptance={
            "passed": all(bool(value) for value in dict(evaluation["acceptance"]).values()),
            "checks": dict(evaluation["acceptance"]),
        },
    )


def build_transfer_summary() -> dict[str, object]:
    payload = run_transfer_acceptance_suite()
    acceptance = dict(payload["acceptance"])
    return _common_schema(
        benchmark_id="transfer",
        seed=None,
        cycles=None,
        world_id=None,
        world_pair="predator_river->foraging_valley,foraging_valley->social_shelter",
        attention_enabled=None,
        profile=None,
        metrics={
            "verified_world_count": acceptance["verified_world_count"],
            "verified_transfer_paths": acceptance["verified_transfer_paths"],
            "transfer_paths_passing": acceptance["transfer_paths_passing"],
        },
        summary={
            "world_rollouts": payload["world_rollouts"],
            "comparison_records": acceptance["comparison_records"],
        },
        acceptance=acceptance,
    )


def build_personality_summary() -> dict[str, object]:
    # Reuse the canonical acceptance protocol already exercised in the dedicated
    # personality validation suite so Pre-M3 readiness is anchored to
    # current-round validated evidence instead of an unratified longer run that
    # can wash out inter-profile differences through over-convergence.
    validation_cycles = 18
    validation_repeats = 3
    validation = run_personality_validation(
        seed=44,
        cycles_per_world=validation_cycles,
        repeats=validation_repeats,
    )
    stability = run_longitudinal_stability(seed=91, cycles_per_world=60, repeats=3)
    acceptance = {
        "validation_passed": bool(validation["acceptance"]["passed"]),
        "stability_passed": bool(stability["acceptance"]["passed"]),
        "passed": bool(validation["acceptance"]["passed"]) and bool(stability["acceptance"]["passed"]),
        "significant_metrics": list(validation["acceptance"]["significant_metrics"]),
        "effect_metrics": list(validation["acceptance"]["effect_metrics"]),
        "passed_profiles": list(stability["acceptance"]["passed_profiles"]),
    }
    return _common_schema(
        benchmark_id="personality",
        seed=44,
        cycles=validation_cycles,
        world_id=None,
        world_pair=None,
        attention_enabled=None,
        profile="multi_profile_protocol",
        metrics={
            "significant_metric_count": len(validation["acceptance"]["significant_metrics"]),
            "effect_metric_count": len(validation["acceptance"]["effect_metrics"]),
            "stability_profiles_passing": stability["acceptance"]["profiles_passing"],
        },
        summary={
            "validation_protocol": {
                "seed": 44,
                "cycles_per_world": validation_cycles,
                "repeats": validation_repeats,
                "protocol_origin": "tests/test_m210_personality_validation.py canonical acceptance configuration",
            },
            "validation_acceptance": dict(validation["acceptance"]),
            "stability_acceptance": dict(stability["acceptance"]),
            "profile_summaries": validation["profile_summaries"],
        },
        acceptance=acceptance,
    )


def run_soak_regression(*, cycles: int = 256, seed: int = 17) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        runtime = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "segment_state.json",
            trace_path=Path(tmp_dir) / "segment_trace.jsonl",
            seed=seed,
            reset=True,
        )
        summary = runtime.run(cycles=cycles, verbose=False)
    checks = {
        "cycles_completed_match": int(summary["cycles_completed"]) == cycles,
        "survived_all_cycles": int(summary["survival_ticks"]) == cycles,
        "unique_actions_gte_3": int(summary["unique_actions"]) >= 3,
        "action_entropy_gte_0_20": float(summary["action_entropy"]) >= 0.20,
        "dominant_action_share_lte_0_92": float(summary["dominant_action_share"]) <= 0.92,
        "max_action_streak_lte_48": int(summary["max_action_streak"]) <= 48,
        "action_switch_count_gte_24": int(summary["action_switch_count"]) >= 24,
    }
    return {
        "seed": seed,
        "cycles": cycles,
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_snapshot_compatibility_payload() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "legacy_state.json"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            seed=17,
            reset=True,
        )
        runtime.run(cycles=2, verbose=False)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["state_version"] = "0.1"
        agent_payload = dict(payload.get("agent", {}))
        for key in (
            "attention_bottleneck",
            "last_attention_trace",
            "last_attention_filtered_observation",
            "decision_history",
            "decision_history_limit",
            "action_history",
            "action_history_limit",
            "drive_history",
            "drive_history_limit",
            "free_energy_history",
            "free_energy_history_limit",
            "narrative_trace",
        ):
            agent_payload.pop(key, None)
        self_model_payload = dict(agent_payload.get("self_model", {}))
        self_model_payload.pop("personality_profile", None)
        self_model_payload.pop("identity_narrative", None)
        self_model_payload.pop("narrative_priors", None)
        agent_payload["self_model"] = self_model_payload
        payload["agent"] = agent_payload
        state_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        restored = SegmentRuntime.load_or_create(state_path=state_path, seed=23)
    checks = {
        "restored_cycle_matches": restored.agent.cycle == 2,
        "attention_defaults_restored": restored.agent.attention_bottleneck.enabled is True,
        "personality_profile_defaulted": restored.agent.self_model.personality_profile is not None,
        "narrative_priors_defaulted": restored.agent.self_model.narrative_priors is not None,
        "identity_narrative_defaulted": restored.agent.self_model.identity_narrative is not None,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


@dataclass(frozen=True)
class ReadinessResult:
    payload: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.payload["final_recommendation"]["passed"])


def build_pre_m3_readiness_report(
    *,
    attention_summary: dict[str, object],
    transfer_summary: dict[str, object],
    personality_summary: dict[str, object],
    soak_regression: dict[str, object],
    snapshot_compatibility: dict[str, object],
    runtime_lifecycle: dict[str, object] | None = None,
    runtime_family_coverage: dict[str, object] | None = None,
) -> ReadinessResult:
    runtime_lifecycle = runtime_lifecycle or evaluate_runtime_lifecycle_evidence()
    runtime_family_coverage = runtime_family_coverage or evaluate_runtime_family_coverage()
    gate_results = {
        "attention_main_loop_established": bool(attention_summary["acceptance"]["passed"]),
        "multi_environment_established": int(transfer_summary["acceptance"]["verified_world_count"]) >= 3,
        "transfer_benchmark_established": bool(transfer_summary["acceptance"]["passed"]),
        "personality_narrative_evidence_established": bool(personality_summary["acceptance"]["passed"]),
        "long_run_soak_regression_passed": bool(soak_regression["passed"]),
        "snapshot_compatibility_passed": bool(snapshot_compatibility["passed"]),
        "runtime_lifecycle_revalidated": runtime_lifecycle.get("verification_status") == "REVALIDATED_THIS_ROUND",
        "runtime_family_coverage_revalidated": runtime_family_coverage.get("family_coverage_status") == "RUNTIME_DIVERSITY_VALIDATED",
    }
    final_passed = all(gate_results.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "pre_m3_readiness",
        "seed": None,
        "world_id": None,
        "world_pair": None,
        "cycles": None,
        "attention_enabled": None,
        "profile": None,
        "metrics": {
            "attention_passed": attention_summary["acceptance"]["passed"],
            "transfer_passed": transfer_summary["acceptance"]["passed"],
            "personality_passed": personality_summary["acceptance"]["passed"],
            "soak_passed": soak_regression["passed"],
            "snapshot_passed": snapshot_compatibility["passed"],
        },
        "summary": {
            "attention_summary_path": "artifacts/pre_m3_attention_summary.json",
            "transfer_summary_path": "artifacts/pre_m3_transfer_summary.json",
            "personality_summary_path": "artifacts/pre_m3_personality_summary.json",
            "soak_regression": soak_regression,
            "snapshot_compatibility": snapshot_compatibility,
            "runtime_lifecycle": runtime_lifecycle,
            "runtime_family_coverage": runtime_family_coverage,
            "gate_results": gate_results,
        },
        "acceptance": {
            "passed": final_passed,
            "gate_results": gate_results,
        },
        "generated_at": _generated_at(),
        "codebase_version": codebase_version(),
        "final_recommendation": {
            "status": "READY_FOR_M3" if final_passed else "NOT_READY_FOR_M3",
            "passed": final_passed,
        },
    }
    return ReadinessResult(payload)


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def run_pre_m3_regression_suite() -> dict[str, object]:
    soak = run_soak_regression()
    snapshot = build_snapshot_compatibility_payload()
    lifecycle = evaluate_runtime_lifecycle_evidence()
    family = evaluate_runtime_family_coverage()
    checks = {
        "soak_regression_passed": bool(soak["passed"]),
        "snapshot_compatibility_passed": bool(snapshot["passed"]),
        "runtime_lifecycle_passed": lifecycle["verification_status"] == "REVALIDATED_THIS_ROUND",
        "runtime_family_coverage_passed": family["family_coverage_status"] == "RUNTIME_DIVERSITY_VALIDATED",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "pre_m3_regressions",
        "seed": 17,
        "world_id": None,
        "world_pair": None,
        "cycles": 256,
        "attention_enabled": None,
        "profile": None,
        "metrics": checks,
        "summary": {
            "soak_regression": soak,
            "snapshot_compatibility": snapshot,
            "runtime_lifecycle": lifecycle,
            "runtime_family_coverage": family,
        },
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "generated_at": _generated_at(),
        "codebase_version": codebase_version(),
    }
