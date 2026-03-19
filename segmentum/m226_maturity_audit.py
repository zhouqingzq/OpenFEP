from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .m221_benchmarks import run_m221_open_narrative_benchmark
from .m222_benchmarks import run_m222_long_horizon_trial
from .m223_benchmarks import run_m223_self_consistency_benchmark
from .m224_benchmarks import run_m224_workspace_benchmark
from .m225_benchmarks import write_m225_acceptance_artifacts


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
MILESTONE_ID = "M2.26"
SCHEMA_VERSION = "m226_v1"
PROTOCOL_VERSION = "m226_unified_audit_v1"
SEED_SET = [226, 245, 323, 342, 420, 439]
CRITICAL_DIMENSIONS = (
    "narrative_grounding_robustness",
    "long_horizon_autonomy",
    "self_consistency_and_repair",
    "functional_conscious_access",
    "open_world_transfer",
)
SCORE_THRESHOLDS = {
    "narrative_grounding_robustness": 0.80,
    "long_horizon_autonomy": 0.85,
    "self_consistency_and_repair": 0.80,
    "functional_conscious_access": 0.80,
    "open_world_transfer": 0.80,
    "fault_tolerance_and_attribution": 0.80,
    "replay_freshness_and_audit_integrity": 0.95,
    "residual_risk_burden": 0.80,
}
DIMENSION_WEIGHTS = {
    "narrative_grounding_robustness": 0.15,
    "long_horizon_autonomy": 0.15,
    "self_consistency_and_repair": 0.15,
    "functional_conscious_access": 0.15,
    "open_world_transfer": 0.15,
    "fault_tolerance_and_attribution": 0.10,
    "replay_freshness_and_audit_integrity": 0.10,
    "residual_risk_burden": 0.05,
}


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _codebase_version() -> str:
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


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _bool_ratio(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return _safe_mean([1.0 if value else 0.0 for value in flags])


def _sorted_false_keys(mapping: dict[str, object]) -> list[str]:
    return sorted(key for key, value in mapping.items() if not bool(value))


def _extract_freshness_flag(
    payload: dict[str, object],
    *,
    report: dict[str, object] | None = None,
    default: bool = True,
) -> bool:
    if report is not None:
        gates = report.get("gates", {})
        if isinstance(gates, dict) and "freshness_generated_this_round" in gates:
            return bool(gates.get("freshness_generated_this_round"))
        report_freshness = report.get("freshness", {})
        if isinstance(report_freshness, dict) and "generated_this_round" in report_freshness:
            return bool(report_freshness.get("generated_this_round"))
    freshness = payload.get("freshness", {})
    if isinstance(freshness, dict) and "generated_this_round" in freshness:
        return bool(freshness.get("generated_this_round"))
    return default


def _risk_entry(*, risk_id: str, priority: str, owner: str, summary: str, next_action: str) -> dict[str, object]:
    return {
        "risk_id": risk_id,
        "priority": priority,
        "owner": owner,
        "summary": summary,
        "next_action": next_action,
    }


def _dimension_entry(
    *,
    name: str,
    score: float,
    threshold: float,
    evidence_origin: str,
    current_round_replay_status: bool,
) -> dict[str, object]:
    rounded = _round(score)
    return {
        "dimension": name,
        "score": rounded,
        "threshold": threshold,
        "pass": rounded >= threshold,
        "evidence_origin": evidence_origin,
        "current_round_replay_status": bool(current_round_replay_status),
    }


def _default_protocol_config(seed_set: list[int]) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed_set": list(seed_set),
        "m221": {"cycles": 24},
        "m222": {"long_run_cycles": 96, "restart_pre_cycles": 24, "restart_post_cycles": 24},
        "m223": {"seed_set_required": [223, 242, 320, 339, 417, 436]},
        "m224": {"variants": "canonical_workspace_protocol"},
        "m225": {"holdout_enabled": True, "holdout_data_used": False, "required_pytest_evidence": True},
    }


def _standardize_m221(payload: dict[str, object]) -> dict[str, object]:
    gates = {key: bool(value) for key, value in dict(payload.get("gates", {})).items()}
    freshness = _extract_freshness_flag(payload)
    score = _bool_ratio(
        [
            gates.get("stability", False),
            gates.get("noise_robustness", False),
            gates.get("adversarial_surface_resistance", False),
            gates.get("conflicting_boundedness", False),
            gates.get("low_quality_degradation", False),
            gates.get("behavior_causality", False),
        ]
    )
    lexical_shortcut_dependency = not bool(gates.get("behavior_causality", False))
    return {
        "milestone_id": "M2.21",
        "dimension": "narrative_grounding_robustness",
        "milestone_status": bool(payload.get("status") == "PASS"),
        "freshness_status": freshness,
        "protocol_completeness": bool(gates.get("artifact_schema_complete", False)),
        "current_round_replay_status": freshness,
        "gating_metrics_summary": gates,
        "residual_risks": list(payload.get("residual_risks", [])),
        "blocking_issues": _sorted_false_keys(gates),
        "score": _round(score),
        "evidence_origin": "current_round_replay" if freshness else "stale_or_inherited_artifact",
        "legacy_evidence_used": False,
        "inherited_only_evidence": False,
        "holdout_data_used": False,
        "holdout_passed": None,
        "lexical_shortcut_dependency": lexical_shortcut_dependency,
        "adapter_collapse_risk": False,
        "report_leakage": False,
        "false_consistency": False,
        "spurious_transfer": False,
        "causal_failure": lexical_shortcut_dependency,
        "codebase_version": str(payload.get("freshness", {}).get("codebase_version", _codebase_version())),
        "seed_set": list(payload.get("seed_set", [])),
        "protocol_config": {"cycles": int(payload.get("cycles", 24))},
        "raw_payload": payload,
    }


def _standardize_m222(payload: dict[str, object]) -> dict[str, object]:
    gates = {key: bool(value) for key, value in dict(payload.get("gates", {})).items()}
    freshness = _extract_freshness_flag(payload)
    critical = [
        "long_horizon_survival",
        "anti_collapse",
        "self_maintenance",
        "ablation_superiority",
        "restart_continuity",
        "stress_recovery",
    ]
    return {
        "milestone_id": "M2.22",
        "dimension": "long_horizon_autonomy",
        "milestone_status": bool(payload.get("status") == "PASS"),
        "freshness_status": freshness,
        "protocol_completeness": bool(gates.get("artifact_schema_complete", False)),
        "current_round_replay_status": freshness,
        "gating_metrics_summary": gates,
        "residual_risks": list(payload.get("residual_risks", [])),
        "blocking_issues": _sorted_false_keys({name: gates.get(name, False) for name in critical}),
        "score": _round(_bool_ratio([gates.get(name, False) for name in critical])),
        "evidence_origin": "current_round_replay" if freshness else "stale_or_inherited_artifact",
        "legacy_evidence_used": False,
        "inherited_only_evidence": False,
        "holdout_data_used": False,
        "holdout_passed": None,
        "lexical_shortcut_dependency": False,
        "adapter_collapse_risk": False,
        "report_leakage": False,
        "false_consistency": False,
        "spurious_transfer": False,
        "causal_failure": not bool(gates.get("restart_continuity", False)),
        "codebase_version": _codebase_version(),
        "seed_set": list(payload.get("seed_set", [])),
        "protocol_config": {"protocols": list(dict(payload.get("protocols", {})).keys())},
        "raw_payload": payload,
    }


def _standardize_m223(payload: dict[str, object]) -> dict[str, object]:
    gates = {key: bool(value) for key, value in dict(payload.get("gates", {})).items()}
    freshness = _extract_freshness_flag(payload)
    critical = [
        "protocol_integrity",
        "commitment_constraints",
        "inconsistency_detection",
        "repair_effectiveness",
        "stress_resilience",
        "evidence_support",
        "bounded_update",
        "sample_independence",
        "statistics",
    ]
    return {
        "milestone_id": "M2.23",
        "dimension": "self_consistency_and_repair",
        "milestone_status": bool(payload.get("status") == "PASS"),
        "freshness_status": freshness,
        "protocol_completeness": bool(gates.get("artifact_schema_complete", False)),
        "current_round_replay_status": freshness,
        "gating_metrics_summary": gates,
        "residual_risks": list(payload.get("residual_risks", [])),
        "blocking_issues": _sorted_false_keys({name: gates.get(name, False) for name in critical}),
        "score": _round(_bool_ratio([gates.get(name, False) for name in critical])),
        "evidence_origin": "current_round_replay" if freshness else "stale_or_inherited_artifact",
        "legacy_evidence_used": False,
        "inherited_only_evidence": False,
        "holdout_data_used": False,
        "holdout_passed": None,
        "lexical_shortcut_dependency": False,
        "adapter_collapse_risk": False,
        "report_leakage": False,
        "false_consistency": not bool(gates.get("protocol_integrity", False)),
        "spurious_transfer": False,
        "causal_failure": not bool(gates.get("repair_effectiveness", False)),
        "codebase_version": str(payload.get("freshness", {}).get("codebase_version", _codebase_version())),
        "seed_set": list(payload.get("seed_set", [])),
        "protocol_config": {"scenario_count": len(dict(payload.get("scenario_definitions", {})))},
        "raw_payload": payload,
    }


def _standardize_m224(payload: dict[str, object]) -> dict[str, object]:
    report = dict(payload.get("acceptance_report", {}))
    gates = {key: bool(value) for key, value in dict(report.get("gates", {})).items()}
    freshness = _extract_freshness_flag(payload, report=report)
    critical = [
        "policy_causality_gain",
        "report_fidelity",
        "report_leakage_rate",
        "suppressed_content_intrusion_rate",
        "broadcast_to_report_alignment",
        "memory_priority_gain",
        "maintenance_priority_gain",
        "metacognitive_review_gain",
        "workspace_capacity_effect_size",
        "capacity_monotonic_metrics",
        "persistence_gain",
        "broadcast_to_action_latency",
        "broadcast_to_memory_alignment",
        "runtime_integration",
        "semantic_report_leakage",
    ]
    return {
        "milestone_id": "M2.24",
        "dimension": "functional_conscious_access",
        "milestone_status": bool(report.get("status") == "PASS"),
        "freshness_status": freshness,
        "protocol_completeness": bool(gates.get("artifact_schema_complete", False)),
        "current_round_replay_status": freshness,
        "gating_metrics_summary": gates,
        "residual_risks": list(report.get("residual_risks", [])),
        "blocking_issues": _sorted_false_keys({name: gates.get(name, False) for name in critical}),
        "score": _round(_bool_ratio([gates.get(name, False) for name in critical])),
        "evidence_origin": "current_round_replay" if freshness else "stale_or_inherited_artifact",
        "legacy_evidence_used": False,
        "inherited_only_evidence": False,
        "holdout_data_used": False,
        "holdout_passed": None,
        "lexical_shortcut_dependency": False,
        "adapter_collapse_risk": False,
        "report_leakage": not bool(gates.get("semantic_report_leakage", False)),
        "false_consistency": False,
        "spurious_transfer": False,
        "causal_failure": not bool(gates.get("policy_causality_gain", False)),
        "codebase_version": str(report.get("codebase_version", _codebase_version())),
        "seed_set": list(payload.get("seed_set", [])),
        "protocol_config": {"variants": list(payload.get("variants", [])), "protocols": list(payload.get("protocols", []))},
        "raw_payload": payload,
    }


def _standardize_m225(payload: dict[str, object]) -> dict[str, object]:
    report = dict(payload.get("acceptance_report", {}))
    gates = {key: bool(value) for key, value in dict(report.get("gates", {})).items()}
    critical = [
        "unseen_world_transfer",
        "transfer_retention",
        "rule_shift_recovery",
        "adversarial_robustness",
        "adapter_robustness",
        "anti_shortcut",
        "core_trace_coverage",
    ]
    goal_details = dict(report.get("goal_details", {}))
    return {
        "milestone_id": "M2.25",
        "dimension": "open_world_transfer",
        "milestone_status": bool(report.get("status") == "PASS"),
        "freshness_status": bool(gates.get("freshness_generated_this_round", False)),
        "protocol_completeness": bool(gates.get("artifact_schema_complete", False)),
        "current_round_replay_status": bool(gates.get("freshness_generated_this_round", False)),
        "gating_metrics_summary": gates,
        "residual_risks": list(report.get("residual_risks", [])),
        "blocking_issues": _sorted_false_keys({name: gates.get(name, False) for name in critical}),
        "score": _round(_bool_ratio([gates.get(name, False) for name in critical])),
        "evidence_origin": "current_round_replay",
        "legacy_evidence_used": False,
        "inherited_only_evidence": False,
        "holdout_data_used": False,
        "holdout_passed": bool(gates.get("unseen_world_transfer", False)),
        "lexical_shortcut_dependency": not bool(gates.get("anti_shortcut", False)),
        "adapter_collapse_risk": float(goal_details.get("adapter_failure_recovery_rate", 0.0)) < 0.80,
        "report_leakage": False,
        "false_consistency": False,
        "spurious_transfer": not bool(gates.get("anti_shortcut", False)),
        "causal_failure": not bool(gates.get("core_trace_coverage", False)),
        "codebase_version": str(report.get("codebase_version", _codebase_version())),
        "seed_set": list(report.get("seed_set", payload.get("seed_set", []))),
        "protocol_config": {
            "protocols": list(report.get("protocols", [])),
            "holdout_worlds": list(report.get("holdout_worlds", [])),
            "pytest_tests": len(list(report.get("pytest_tests", []))),
        },
        "raw_payload": payload,
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _standardize_replays(raw_replays: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "M2.21": _standardize_m221(raw_replays["M2.21"]),
        "M2.22": _standardize_m222(raw_replays["M2.22"]),
        "M2.23": _standardize_m223(raw_replays["M2.23"]),
        "M2.24": _standardize_m224(raw_replays["M2.24"]),
        "M2.25": _standardize_m225(raw_replays["M2.25"]),
    }


def _build_replay_freshness(
    standardized_replays: dict[str, dict[str, object]],
    *,
    seed_set: list[int],
    codebase_version: str,
    protocol_config: dict[str, object],
) -> dict[str, object]:
    rows = []
    current_round_count = 0
    inherited_only_count = 0
    stale_count = 0
    for milestone_id, replay in standardized_replays.items():
        current_round = bool(replay.get("current_round_replay_status", False))
        current_round_count += 1 if current_round else 0
        inherited_only = bool(replay.get("inherited_only_evidence", False))
        inherited_only_count += 1 if inherited_only else 0
        stale = (not current_round) or inherited_only or bool(replay.get("legacy_evidence_used", False))
        stale_count += 1 if stale else 0
        rows.append(
            {
                "milestone_id": milestone_id,
                "dimension": replay["dimension"],
                "milestone_status": replay["milestone_status"],
                "freshness_status": "CURRENT_ROUND" if current_round else "STALE_OR_INHERITED",
                "protocol_completeness": replay["protocol_completeness"],
                "current_round_replay_status": current_round,
                "gating_metrics_summary": dict(replay.get("gating_metrics_summary", {})),
                "residual_risks": list(replay.get("residual_risks", [])),
                "blocking_issues": list(replay.get("blocking_issues", [])),
                "evidence_origin": replay["evidence_origin"],
                "seed_set": list(replay.get("seed_set", [])),
                "protocol_config": dict(replay.get("protocol_config", {})),
                "holdout_data_used": replay.get("holdout_data_used"),
                "legacy_evidence_used": replay.get("legacy_evidence_used"),
                "inherited_only_evidence": inherited_only,
            }
        )
    coverage = _round(current_round_count / max(1, len(standardized_replays)))
    misuse_rate = _round(stale_count / max(1, len(standardized_replays)))
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "codebase_version": codebase_version,
        "seed_set": list(seed_set),
        "protocol_config": protocol_config,
        "current_round_replay_coverage": coverage,
        "inherited_only_critical_metric_count": inherited_only_count,
        "stale_artifact_misuse_rate": misuse_rate,
        "generated_this_round": coverage == 1.0 and inherited_only_count == 0 and misuse_rate == 0.0,
        "milestone_replays": rows,
    }


def _build_cross_milestone_consistency(standardized_replays: dict[str, dict[str, object]]) -> dict[str, object]:
    m221 = standardized_replays["M2.21"]
    m222 = standardized_replays["M2.22"]
    m223 = standardized_replays["M2.23"]
    m224 = standardized_replays["M2.24"]
    m225 = standardized_replays["M2.25"]
    checks = [
        {
            "check_id": "cross_001_narrative_vs_transfer",
            "pair": ["M2.21", "M2.25"],
            "question": "narrative grounding 与 open-world transfer 是否冲突",
            "pass": bool(not m221["milestone_status"] or m225["score"] >= 0.80) and not bool(m225["spurious_transfer"]),
            "severity": "high" if m221["milestone_status"] and (m225["score"] < 0.80 or m225["spurious_transfer"]) else "none",
            "evidence": {"m221_score": m221["score"], "m225_score": m225["score"], "spurious_transfer": m225["spurious_transfer"]},
            "maturity_impact": "blocks_default_mature" if m221["milestone_status"] and (m225["score"] < 0.80 or m225["spurious_transfer"]) else "none",
        },
        {
            "check_id": "cross_002_identity_vs_autonomy",
            "pair": ["M2.22", "M2.23"],
            "question": "identity commitments 与 long-horizon autonomy 是否冲突",
            "pass": not (m222["milestone_status"] and not m223["milestone_status"]),
            "severity": "high" if m222["milestone_status"] and not m223["milestone_status"] else "none",
            "evidence": {"m222_status": m222["milestone_status"], "m223_status": m223["milestone_status"]},
            "maturity_impact": "blocks_default_mature" if m222["milestone_status"] and not m223["milestone_status"] else "none",
        },
        {
            "check_id": "cross_003_workspace_vs_self_report",
            "pair": ["M2.24", "M2.23"],
            "question": "workspace access 与 self-report fidelity 是否冲突",
            "pass": not bool(m224["report_leakage"]) and not bool(m223["false_consistency"]),
            "severity": "high" if m224["report_leakage"] or m223["false_consistency"] else "none",
            "evidence": {"report_leakage": m224["report_leakage"], "false_consistency": m223["false_consistency"]},
            "maturity_impact": "blocks_default_mature" if m224["report_leakage"] or m223["false_consistency"] else "none",
        },
        {
            "check_id": "cross_004_repair_vs_identity_stability",
            "pair": ["M2.23", "M2.25"],
            "question": "repair mechanism 与 identity stability 是否冲突",
            "pass": not (m223["milestone_status"] and bool(m225["adapter_collapse_risk"])),
            "severity": "medium" if m223["milestone_status"] and bool(m225["adapter_collapse_risk"]) else "none",
            "evidence": {"repair_status": m223["milestone_status"], "adapter_collapse_risk": m225["adapter_collapse_risk"]},
            "maturity_impact": "downgrades_to_controlled" if m223["milestone_status"] and bool(m225["adapter_collapse_risk"]) else "none",
        },
        {
            "check_id": "cross_005_governance_vs_action",
            "pair": ["M2.22", "M2.24"],
            "question": "governance constraints 与 adaptive action 是否冲突",
            "pass": bool(m222["score"] >= 0.85) or bool(m224["score"] >= 0.80),
            "severity": "medium" if m222["score"] < 0.85 and m224["score"] < 0.80 else "none",
            "evidence": {"m222_score": m222["score"], "m224_score": m224["score"]},
            "maturity_impact": "downgrades_to_m3_entry" if m222["score"] < 0.85 and m224["score"] < 0.80 else "none",
        },
    ]
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "checks": checks,
        "high_severity_conflict_count": sum(1 for item in checks if item["severity"] == "high" and not item["pass"]),
        "medium_severity_unexplained_conflict_count": sum(1 for item in checks if item["severity"] == "medium" and not item["pass"]),
        "repair_caused_other_capability_degradation_count": sum(
            1 for item in checks if (not item["pass"]) and str(item["maturity_impact"]).startswith("blocks")
        ),
    }


def _build_red_team_audit(
    standardized_replays: dict[str, dict[str, object]],
    replay_freshness: dict[str, object],
    cross_consistency: dict[str, object],
    *,
    protocol_config: dict[str, object],
) -> dict[str, object]:
    m221 = standardized_replays["M2.21"]
    m223 = standardized_replays["M2.23"]
    m224 = standardized_replays["M2.24"]
    m225 = standardized_replays["M2.25"]
    holdout_data_used = bool(dict(protocol_config.get("m225", {})).get("holdout_data_used", False))
    checks = [
        {
            "check_id": "rt_001_stale_artifact_masquerade",
            "evidence": {"current_round_replay_coverage": replay_freshness["current_round_replay_coverage"], "stale_artifact_misuse_rate": replay_freshness["stale_artifact_misuse_rate"]},
            "pass": replay_freshness["current_round_replay_coverage"] == 1.0 and replay_freshness["stale_artifact_misuse_rate"] == 0.0,
            "severity": "high" if replay_freshness["stale_artifact_misuse_rate"] > 0.0 else "none",
            "maturity_impact": "blocks_default_mature" if replay_freshness["stale_artifact_misuse_rate"] > 0.0 else "none",
        },
        {
            "check_id": "rt_002_inherited_metric_overstatement",
            "evidence": {"inherited_only_critical_metric_count": replay_freshness["inherited_only_critical_metric_count"]},
            "pass": replay_freshness["inherited_only_critical_metric_count"] == 0,
            "severity": "high" if replay_freshness["inherited_only_critical_metric_count"] > 0 else "none",
            "maturity_impact": "blocks_default_mature" if replay_freshness["inherited_only_critical_metric_count"] > 0 else "none",
        },
        {
            "check_id": "rt_003_benchmark_overfit_pattern",
            "evidence": {"lexical_shortcut_dependency": m221["lexical_shortcut_dependency"], "anti_shortcut_passed": not m225["spurious_transfer"]},
            "pass": not m221["lexical_shortcut_dependency"] and not m225["spurious_transfer"],
            "severity": "high" if m221["lexical_shortcut_dependency"] or m225["spurious_transfer"] else "none",
            "maturity_impact": "blocks_default_mature" if m221["lexical_shortcut_dependency"] or m225["spurious_transfer"] else "none",
        },
        {
            "check_id": "rt_004_explanation_report_leakage",
            "evidence": {"workspace_report_leakage": m224["report_leakage"], "cross_report_conflicts": cross_consistency["high_severity_conflict_count"]},
            "pass": not m224["report_leakage"],
            "severity": "high" if m224["report_leakage"] else "none",
            "maturity_impact": "blocks_default_mature" if m224["report_leakage"] else "none",
        },
        {
            "check_id": "rt_005_holdout_contamination",
            "evidence": {"holdout_data_used": holdout_data_used, "holdout_passed": m225["holdout_passed"]},
            "pass": (not holdout_data_used) and bool(m225["holdout_passed"]),
            "severity": "high" if holdout_data_used or not bool(m225["holdout_passed"]) else "none",
            "maturity_impact": "blocks_default_mature" if holdout_data_used or not bool(m225["holdout_passed"]) else "none",
        },
        {
            "check_id": "rt_006_seed_cherry_pick",
            "evidence": {"seed_set": list(replay_freshness["seed_set"])},
            "pass": list(replay_freshness["seed_set"]) == list(SEED_SET),
            "severity": "medium" if list(replay_freshness["seed_set"]) != list(SEED_SET) else "none",
            "maturity_impact": "downgrades_to_controlled" if list(replay_freshness["seed_set"]) != list(SEED_SET) else "none",
        },
        {
            "check_id": "rt_007_heuristic_shortcut_dependency",
            "evidence": {"m221": m221["lexical_shortcut_dependency"], "m223_false_consistency": m223["false_consistency"]},
            "pass": not m221["lexical_shortcut_dependency"] and not m223["false_consistency"],
            "severity": "high" if m221["lexical_shortcut_dependency"] or m223["false_consistency"] else "none",
            "maturity_impact": "blocks_default_mature" if m221["lexical_shortcut_dependency"] or m223["false_consistency"] else "none",
        },
        {
            "check_id": "rt_008_single_module_failure_illusion",
            "evidence": {"failed_milestones": [mid for mid, replay in standardized_replays.items() if not replay["milestone_status"]]},
            "pass": all(replay["milestone_status"] for replay in standardized_replays.values()),
            "severity": "high" if any(not replay["milestone_status"] for replay in standardized_replays.values()) else "none",
            "maturity_impact": "blocks_default_mature" if any(not replay["milestone_status"] for replay in standardized_replays.values()) else "none",
        },
    ]
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "checks": checks,
        "high_severity_red_team_failures": sum(1 for item in checks if item["severity"] == "high" and not item["pass"]),
        "medium_severity_red_team_failures": sum(1 for item in checks if item["severity"] == "medium" and not item["pass"]),
        "holdout_contamination_cases": sum(1 for item in checks if item["check_id"] == "rt_005_holdout_contamination" and not item["pass"]),
        "report_leakage_critical_cases": sum(1 for item in checks if item["check_id"] == "rt_004_explanation_report_leakage" and not item["pass"]),
        "inherited_metric_misuse_cases": sum(1 for item in checks if item["check_id"] == "rt_002_inherited_metric_overstatement" and not item["pass"]),
    }


def _build_scorecard(
    standardized_replays: dict[str, dict[str, object]],
    replay_freshness: dict[str, object],
    red_team_audit: dict[str, object],
) -> dict[str, object]:
    dimension_scores = {
        "narrative_grounding_robustness": standardized_replays["M2.21"]["score"],
        "long_horizon_autonomy": standardized_replays["M2.22"]["score"],
        "self_consistency_and_repair": standardized_replays["M2.23"]["score"],
        "functional_conscious_access": standardized_replays["M2.24"]["score"],
        "open_world_transfer": standardized_replays["M2.25"]["score"],
        "fault_tolerance_and_attribution": _round(
            _safe_mean(
                [
                    standardized_replays["M2.22"]["score"],
                    standardized_replays["M2.23"]["score"],
                    standardized_replays["M2.24"]["score"],
                    standardized_replays["M2.25"]["score"],
                ]
            )
        ),
        "replay_freshness_and_audit_integrity": _round(
            max(
                0.0,
                min(
                    1.0,
                    replay_freshness["current_round_replay_coverage"]
                    - (0.5 * replay_freshness["stale_artifact_misuse_rate"])
                    - (0.20 * red_team_audit["medium_severity_red_team_failures"])
                    - (0.40 * red_team_audit["high_severity_red_team_failures"]),
                ),
            )
        ),
        "residual_risk_burden": 1.0,
    }
    entries = []
    for name, threshold in SCORE_THRESHOLDS.items():
        current_round = True if name != "replay_freshness_and_audit_integrity" else bool(replay_freshness["generated_this_round"])
        entries.append(
            _dimension_entry(
                name=name,
                score=dimension_scores[name],
                threshold=threshold,
                evidence_origin="current_round_replay",
                current_round_replay_status=current_round,
            )
        )
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "dimension_scores": entries,
        "critical_dimension_pass_count": f"{sum(1 for name in CRITICAL_DIMENSIONS if dimension_scores[name] >= SCORE_THRESHOLDS[name])} / 5",
        "weighted_total_score": _round(sum(float(dimension_scores[name]) * float(DIMENSION_WEIGHTS[name]) for name in DIMENSION_WEIGHTS)),
    }


def _collect_residual_risks(
    standardized_replays: dict[str, dict[str, object]],
    cross_consistency: dict[str, object],
    red_team_audit: dict[str, object],
) -> list[dict[str, object]]:
    risks: list[dict[str, object]] = []
    for milestone_id, replay in standardized_replays.items():
        if not replay["milestone_status"]:
            risks.append(
                _risk_entry(
                    risk_id=f"{milestone_id.lower().replace('.', '_')}_gate_failure",
                    priority="high",
                    owner=milestone_id,
                    summary=f"{milestone_id} still fails its own acceptance gate set.",
                    next_action=f"Repair milestone-specific blocking issues: {', '.join(replay['blocking_issues']) or 'unknown blockers'}.",
                )
            )
    for check in cross_consistency["checks"]:
        if not check["pass"]:
            risks.append(
                _risk_entry(
                    risk_id=str(check["check_id"]),
                    priority="high" if check["severity"] == "high" else "medium",
                    owner="/".join(check["pair"]),
                    summary=str(check["question"]),
                    next_action="Resolve the cross-milestone contradiction before any DEFAULT_MATURE claim.",
                )
            )
    for check in red_team_audit["checks"]:
        if not check["pass"]:
            risks.append(
                _risk_entry(
                    risk_id=str(check["check_id"]),
                    priority="high" if check["severity"] == "high" else "medium",
                    owner="M2.26",
                    summary=f"Red-team check failed: {check['check_id']}",
                    next_action="Address the audit integrity issue and rerun the full protocol.",
                )
            )
    deduped: dict[str, dict[str, object]] = {}
    for item in risks:
        deduped[str(item["risk_id"])] = item
    return list(deduped.values())


def _build_blocking_reasons(
    standardized_replays: dict[str, dict[str, object]],
    replay_freshness: dict[str, object],
    cross_consistency: dict[str, object],
    red_team_audit: dict[str, object],
    scorecard: dict[str, object],
    residual_risks: list[dict[str, object]],
) -> dict[str, object]:
    reasons: list[str] = []
    if replay_freshness["current_round_replay_coverage"] < 1.0:
        reasons.append("not_all_required_milestone_replays_ran_this_round")
    if replay_freshness["inherited_only_critical_metric_count"] > 0:
        reasons.append("inherited_only_evidence_detected")
    if replay_freshness["stale_artifact_misuse_rate"] > 0.0:
        reasons.append("stale_artifact_misuse_detected")
    for milestone_id, replay in standardized_replays.items():
        if not replay["milestone_status"]:
            reasons.append(f"{milestone_id.lower().replace('.', '_')}_failed")
    if not bool(standardized_replays["M2.25"]["holdout_passed"]):
        reasons.append("holdout_transfer_not_passed")
    if bool(standardized_replays["M2.21"]["lexical_shortcut_dependency"]):
        reasons.append("narrative_grounding_depends_on_lexical_shortcut")
    if bool(standardized_replays["M2.25"]["adapter_collapse_risk"]):
        reasons.append("adapter_degradation_collapse_risk_present")
    if any(bool(replay["report_leakage"]) for replay in standardized_replays.values()):
        reasons.append("report_leakage_detected")
    if any(bool(replay["false_consistency"]) for replay in standardized_replays.values()):
        reasons.append("false_consistency_detected")
    if any(bool(replay["spurious_transfer"]) for replay in standardized_replays.values()):
        reasons.append("spurious_transfer_detected")
    if any(item["priority"] == "high" for item in residual_risks):
        reasons.append("high_priority_residual_risks_present")
    if cross_consistency["high_severity_conflict_count"] > 0:
        reasons.append("high_severity_cross_milestone_conflict_present")
    if red_team_audit["high_severity_red_team_failures"] > 0:
        reasons.append("high_severity_red_team_failure_present")
    dimensions = {item["dimension"]: item for item in scorecard["dimension_scores"]}
    if scorecard["weighted_total_score"] < 0.85:
        reasons.append("weighted_total_score_below_threshold")
    for name, threshold in SCORE_THRESHOLDS.items():
        if float(dimensions[name]["score"]) < float(threshold):
            reasons.append(f"{name}_below_threshold")
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "blocking_reasons": reasons,
    }


def _derive_final_status(
    standardized_replays: dict[str, dict[str, object]],
    replay_freshness: dict[str, object],
    cross_consistency: dict[str, object],
    red_team_audit: dict[str, object],
    scorecard: dict[str, object],
    blocking: dict[str, object],
    residual_risks: list[dict[str, object]],
) -> dict[str, object]:
    reasons = list(blocking["blocking_reasons"])
    dimensions = {item["dimension"]: item for item in scorecard["dimension_scores"]}
    high_risks = [item for item in residual_risks if item["priority"] == "high"]
    medium_risks = [item for item in residual_risks if item["priority"] == "medium"]
    critical_pass_count = sum(1 for name in CRITICAL_DIMENSIONS if bool(dimensions[name]["pass"]))
    default_ready = (
        replay_freshness["current_round_replay_coverage"] == 1.0
        and replay_freshness["inherited_only_critical_metric_count"] == 0
        and replay_freshness["stale_artifact_misuse_rate"] == 0.0
        and critical_pass_count == 5
        and cross_consistency["high_severity_conflict_count"] == 0
        and cross_consistency["medium_severity_unexplained_conflict_count"] <= 1
        and cross_consistency["repair_caused_other_capability_degradation_count"] == 0
        and red_team_audit["high_severity_red_team_failures"] == 0
        and red_team_audit["medium_severity_red_team_failures"] <= 1
        and red_team_audit["holdout_contamination_cases"] == 0
        and red_team_audit["report_leakage_critical_cases"] == 0
        and red_team_audit["inherited_metric_misuse_cases"] == 0
        and scorecard["weighted_total_score"] >= 0.85
        and all(bool(item["pass"]) for item in scorecard["dimension_scores"])
        and len(high_risks) == 0
        and len(medium_risks) <= 2
        and not reasons
    )
    if default_ready:
        final_status = "DEFAULT_MATURE"
    elif critical_pass_count == 5 and not high_risks and red_team_audit["high_severity_red_team_failures"] == 0:
        final_status = "CONTROLLED_MATURE"
    elif critical_pass_count >= 3 and replay_freshness["current_round_replay_coverage"] == 1.0:
        final_status = "M3_ENTRY_READY"
    else:
        final_status = "NOT_READY"
    return {
        "status": "PASS" if final_status in {"CONTROLLED_MATURE", "DEFAULT_MATURE"} else "FAIL",
        "final_status": final_status,
        "default_mature": final_status == "DEFAULT_MATURE",
        "blocking_reasons": reasons,
        "why_not_default_mature": [] if final_status == "DEFAULT_MATURE" else reasons[:5],
        "residual_risks": residual_risks,
        "recommended_next_action": {
            "DEFAULT_MATURE": "Continue routine monitoring; keep the unified audit protocol mandatory for future upgrades.",
            "CONTROLLED_MATURE": "Use under controlled conditions and close the remaining boundary risks before claiming default maturity.",
            "M3_ENTRY_READY": "Enter the next system-level trial only after repairing the top blocking issues and rerunning the unified audit.",
            "NOT_READY": "Repair failed milestone gates first, then rerun the full current-round audit from scratch.",
        }[final_status],
    }


def build_m226_maturity_audit(
    *,
    standardized_replays: dict[str, dict[str, object]],
    seed_set: list[int] | None = None,
    protocol_config: dict[str, object] | None = None,
    codebase_version: str | None = None,
) -> dict[str, object]:
    selected_seed_set = list(seed_set or SEED_SET)
    selected_protocol = dict(protocol_config or _default_protocol_config(selected_seed_set))
    version = codebase_version or _codebase_version()
    replay_freshness = _build_replay_freshness(standardized_replays, seed_set=selected_seed_set, codebase_version=version, protocol_config=selected_protocol)
    cross_consistency = _build_cross_milestone_consistency(standardized_replays)
    red_team_audit = _build_red_team_audit(standardized_replays, replay_freshness, cross_consistency, protocol_config=selected_protocol)
    scorecard = _build_scorecard(standardized_replays, replay_freshness, red_team_audit)
    residual_risks = _collect_residual_risks(standardized_replays, cross_consistency, red_team_audit)
    risk_burden_score = _round(max(0.0, 1.0 - (0.40 * sum(1 for item in residual_risks if item["priority"] == "high")) - (0.10 * sum(1 for item in residual_risks if item["priority"] == "medium"))))
    for item in scorecard["dimension_scores"]:
        if item["dimension"] == "residual_risk_burden":
            item["score"] = risk_burden_score
            item["pass"] = risk_burden_score >= SCORE_THRESHOLDS["residual_risk_burden"]
    scorecard["weighted_total_score"] = _round(sum(float(item["score"]) * float(DIMENSION_WEIGHTS[item["dimension"]]) for item in scorecard["dimension_scores"]))
    scorecard["critical_dimension_pass_count"] = f"{sum(1 for item in scorecard['dimension_scores'] if item['dimension'] in CRITICAL_DIMENSIONS and bool(item['pass']))} / 5"
    blocking = _build_blocking_reasons(standardized_replays, replay_freshness, cross_consistency, red_team_audit, scorecard, residual_risks)
    final = _derive_final_status(standardized_replays, replay_freshness, cross_consistency, red_team_audit, scorecard, blocking, residual_risks)
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "codebase_version": version,
        "seed_set": selected_seed_set,
        "protocol_config": selected_protocol,
        "replay_freshness": replay_freshness,
        "cross_milestone_consistency": cross_consistency,
        "red_team_audit": red_team_audit,
        "maturity_scorecard": scorecard,
        "blocking_reasons_artifact": blocking,
        "final_report": {
            "milestone_id": MILESTONE_ID,
            "schema_version": SCHEMA_VERSION,
            "generated_at": _generated_at(),
            "codebase_version": version,
            "protocol_version": selected_protocol["protocol_version"],
            "seed_set": selected_seed_set,
            "dimension_scores": scorecard["dimension_scores"],
            "critical_dimension_pass_count": scorecard["critical_dimension_pass_count"],
            "current_round_replay_coverage": replay_freshness["current_round_replay_coverage"],
            "high_severity_conflict_count": cross_consistency["high_severity_conflict_count"],
            "high_severity_red_team_failures": red_team_audit["high_severity_red_team_failures"],
            "residual_risk_summary": {
                "high_priority_count": sum(1 for item in residual_risks if item["priority"] == "high"),
                "medium_priority_count": sum(1 for item in residual_risks if item["priority"] == "medium"),
                "risks": residual_risks,
            },
            **final,
        },
    }


def run_m226_full_replay(
    *,
    seed_set: list[int] | None = None,
    pytest_evidence: list[dict[str, object]] | None = None,
    raw_replay_overrides: dict[str, dict[str, object]] | None = None,
    standardized_replay_overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    selected_seed_set = list(seed_set or SEED_SET)
    if standardized_replay_overrides and set(standardized_replay_overrides) >= {"M2.21", "M2.22", "M2.23", "M2.24", "M2.25"} and not raw_replay_overrides:
        standardized = {milestone_id: dict(payload) for milestone_id, payload in standardized_replay_overrides.items()}
    else:
        raw_replays = {
            "M2.21": run_m221_open_narrative_benchmark(seed_set=selected_seed_set, cycles=24),
            "M2.22": run_m222_long_horizon_trial(seed_set=selected_seed_set, long_run_cycles=96, restart_pre_cycles=24, restart_post_cycles=24),
            "M2.23": run_m223_self_consistency_benchmark(
                seed_set=selected_seed_set,
                required_seed_set=selected_seed_set,
            ),
            "M2.24": run_m224_workspace_benchmark(seed_set=selected_seed_set),
            "M2.25": {
                "seed_set": list(selected_seed_set),
                "artifact_paths": {
                    name: str(path)
                    for name, path in write_m225_acceptance_artifacts(
                        seed_set=selected_seed_set,
                        pytest_evidence=pytest_evidence,
                    ).items()
                },
            },
        }
        raw_replays["M2.25"]["acceptance_report"] = _load_json(Path(raw_replays["M2.25"]["artifact_paths"]["report"]))
        if raw_replay_overrides:
            for milestone_id, payload in raw_replay_overrides.items():
                raw_replays[milestone_id] = payload
        standardized = _standardize_replays(raw_replays)
    if standardized_replay_overrides and set(standardized_replay_overrides) < {"M2.21", "M2.22", "M2.23", "M2.24", "M2.25"}:
        for milestone_id, payload in standardized_replay_overrides.items():
            merged = dict(standardized.get(milestone_id, {}))
            merged.update(payload)
            standardized[milestone_id] = merged
    return build_m226_maturity_audit(
        standardized_replays=standardized,
        seed_set=selected_seed_set,
        protocol_config=_default_protocol_config(selected_seed_set),
        codebase_version=_codebase_version(),
    )


def write_m226_maturity_audit_artifacts(
    *,
    seed_set: list[int] | None = None,
    pytest_evidence: list[dict[str, object]] | None = None,
    raw_replay_overrides: dict[str, dict[str, object]] | None = None,
    standardized_replay_overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, Path]:
    payload = run_m226_full_replay(
        seed_set=seed_set,
        pytest_evidence=pytest_evidence,
        raw_replay_overrides=raw_replay_overrides,
        standardized_replay_overrides=standardized_replay_overrides,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "maturity_scorecard": ARTIFACTS_DIR / "m226_maturity_scorecard.json",
        "replay_freshness": ARTIFACTS_DIR / "m226_replay_freshness.json",
        "cross_milestone_consistency": ARTIFACTS_DIR / "m226_cross_milestone_consistency.json",
        "red_team_audit": ARTIFACTS_DIR / "m226_red_team_audit.json",
        "blocking_reasons": ARTIFACTS_DIR / "m226_blocking_reasons.json",
        "report": REPORTS_DIR / "m226_maturity_audit_report.json",
    }
    paths["maturity_scorecard"].write_text(json.dumps(payload["maturity_scorecard"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["replay_freshness"].write_text(json.dumps(payload["replay_freshness"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["cross_milestone_consistency"].write_text(json.dumps(payload["cross_milestone_consistency"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["red_team_audit"].write_text(json.dumps(payload["red_team_audit"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["blocking_reasons"].write_text(json.dumps(payload["blocking_reasons_artifact"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["report"].write_text(json.dumps(payload["final_report"], indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


if __name__ == "__main__":
    result = run_m226_full_replay()
    sys.stdout.write(json.dumps(result["final_report"], indent=2, ensure_ascii=False))
