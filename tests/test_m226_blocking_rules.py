from __future__ import annotations

import json
from pathlib import Path

from segmentum.m226_maturity_audit import SEED_SET, build_m226_maturity_audit
from segmentum.m226_maturity_audit import run_m226_full_replay

from tests._m226_test_utils import base_standardized_replays


def test_inherited_only_evidence_cannot_grant_default_mature() -> None:
    replays = base_standardized_replays()
    replays["M2.23"]["inherited_only_evidence"] = True
    replays["M2.23"]["current_round_replay_status"] = False

    payload = build_m226_maturity_audit(standardized_replays=replays, seed_set=list(SEED_SET), codebase_version="test-sha")
    report = payload["final_report"]

    assert report["default_mature"] is False
    assert "inherited_only_evidence_detected" in report["blocking_reasons"]


def test_holdout_contamination_blocks_maturity() -> None:
    replays = base_standardized_replays()
    protocol_config = {
        "protocol_version": "m226_test_protocol",
        "seed_set": list(SEED_SET),
        "m225": {"holdout_enabled": True, "holdout_data_used": True, "required_pytest_evidence": True},
    }

    payload = build_m226_maturity_audit(
        standardized_replays=replays,
        seed_set=list(SEED_SET),
        codebase_version="test-sha",
        protocol_config=protocol_config,
    )
    report = payload["final_report"]

    assert report["default_mature"] is False
    assert "high_severity_red_team_failure_present" in report["blocking_reasons"]
    assert any(check["check_id"] == "rt_005_holdout_contamination" and not check["pass"] for check in payload["red_team_audit"]["checks"])


def test_m225_write_path_fresh_report_prevents_preview_style_false_failure(monkeypatch, tmp_path: Path) -> None:
    def _m221(*, seed_set: list[int] | None = None, cycles: int = 24) -> dict[str, object]:
        return {
            "status": "PASS",
            "gates": {
                "stability": True,
                "noise_robustness": True,
                "adversarial_surface_resistance": True,
                "conflicting_boundedness": True,
                "low_quality_degradation": True,
                "behavior_causality": True,
                "artifact_schema_complete": True,
            },
            "residual_risks": [],
            "freshness": {"codebase_version": "test-sha"},
            "seed_set": list(seed_set or SEED_SET),
            "cycles": cycles,
        }

    def _m222(*, seed_set: list[int] | None = None, long_run_cycles: int = 96, restart_pre_cycles: int = 24, restart_post_cycles: int = 24) -> dict[str, object]:
        return {
            "status": "PASS",
            "gates": {
                "long_horizon_survival": True,
                "anti_collapse": True,
                "self_maintenance": True,
                "ablation_superiority": True,
                "restart_continuity": True,
                "stress_recovery": True,
                "artifact_schema_complete": True,
            },
            "residual_risks": [],
            "seed_set": list(seed_set or SEED_SET),
            "protocols": {"mixed_stress": {}},
        }

    def _m223(*, seed_set: list[int] | None = None, required_seed_set: list[int] | None = None) -> dict[str, object]:
        return {
            "status": "PASS",
            "gates": {
                "protocol_integrity": True,
                "commitment_constraints": True,
                "inconsistency_detection": True,
                "repair_effectiveness": True,
                "stress_resilience": True,
                "evidence_support": True,
                "bounded_update": True,
                "sample_independence": True,
                "statistics": True,
                "artifact_schema_complete": True,
            },
            "residual_risks": [],
            "freshness": {"codebase_version": "test-sha"},
            "seed_set": list(seed_set or SEED_SET),
            "scenario_definitions": {"s1": {}, "s2": {}, "s3": {}, "s4": {}},
        }

    def _m224(*, seed_set: list[int] | None = None) -> dict[str, object]:
        return {
            "seed_set": list(seed_set or SEED_SET),
            "variants": ["full_workspace"],
            "protocols": ["workspace_protocol"],
            "acceptance_report": {
                "status": "PASS",
                "codebase_version": "test-sha",
                "gates": {
                    "policy_causality_gain": True,
                    "report_fidelity": True,
                    "report_leakage_rate": True,
                    "suppressed_content_intrusion_rate": True,
                    "broadcast_to_report_alignment": True,
                    "memory_priority_gain": True,
                    "maintenance_priority_gain": True,
                    "metacognitive_review_gain": True,
                    "workspace_capacity_effect_size": True,
                    "capacity_monotonic_metrics": True,
                    "persistence_gain": True,
                    "broadcast_to_action_latency": True,
                    "broadcast_to_memory_alignment": True,
                    "runtime_integration": True,
                    "semantic_report_leakage": True,
                    "artifact_schema_complete": True,
                },
                "residual_risks": [],
            },
        }

    def _write_m225(*, seed_set: list[int] | None = None, pytest_evidence: list[dict[str, object]] | None = None) -> dict[str, Path]:
        report_path = tmp_path / "m225_acceptance_report.json"
        report = {
            "status": "PASS",
            "codebase_version": "test-sha",
            "seed_set": list(seed_set or SEED_SET),
            "protocols": ["holdout_transfer_protocol"],
            "holdout_worlds": ["holdout_a"],
            "pytest_tests": list(pytest_evidence or []),
            "goal_details": {"adapter_failure_recovery_rate": 0.91},
            "residual_risks": [],
            "gates": {
                "unseen_world_transfer": True,
                "transfer_retention": True,
                "rule_shift_recovery": True,
                "adversarial_robustness": True,
                "adapter_robustness": True,
                "anti_shortcut": True,
                "core_trace_coverage": True,
                "artifact_schema_complete": True,
                "freshness_generated_this_round": True,
                "pytest_evidence_complete": True,
                "historical_regression_evidence": True,
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            "report": report_path,
            "transfer_graph": tmp_path / "transfer.json",
            "holdout_transfer": tmp_path / "holdout.json",
            "rule_shift_recovery": tmp_path / "rule.json",
            "social_deception": tmp_path / "social.json",
            "adapter_degradation": tmp_path / "adapter.json",
            "identity_preservation": tmp_path / "identity.json",
        }

    monkeypatch.setattr("segmentum.m226_maturity_audit.run_m221_open_narrative_benchmark", _m221)
    monkeypatch.setattr("segmentum.m226_maturity_audit.run_m222_long_horizon_trial", _m222)
    monkeypatch.setattr("segmentum.m226_maturity_audit.run_m223_self_consistency_benchmark", _m223)
    monkeypatch.setattr("segmentum.m226_maturity_audit.run_m224_workspace_benchmark", _m224)
    monkeypatch.setattr("segmentum.m226_maturity_audit.write_m225_acceptance_artifacts", _write_m225)

    payload = run_m226_full_replay(
        seed_set=list(SEED_SET),
        pytest_evidence=[
            {"name": "tests/test_m225_freshness_guards.py", "nodeid": "tests/test_m225_freshness_guards.py", "status": "passed"},
            {"name": "tests/test_self_model.py", "nodeid": "tests/test_self_model.py", "status": "passed"},
            {"name": "tests/test_m2_targeted_repair.py", "nodeid": "tests/test_m2_targeted_repair.py", "status": "passed"},
            {"name": "tests/test_baseline_regressions.py", "nodeid": "tests/test_baseline_regressions.py", "status": "passed"},
        ],
    )

    assert payload["replay_freshness"]["generated_this_round"] is True
    assert payload["cross_milestone_consistency"]["high_severity_conflict_count"] == 0
    assert "m2_25_failed" not in payload["final_report"]["blocking_reasons"]
    assert "stale_artifact_misuse_detected" not in payload["final_report"]["blocking_reasons"]
