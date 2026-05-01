"""Acceptance artifact checks for M5.6 local persona runtime."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m56_acceptance_artifacts import (
    PERFORMANCE_P95_TARGET_MS,
    generate_ablation,
    generate_all,
    generate_performance,
    generate_runtime_trace,
    generate_stress,
)


def test_m56_runtime_trace_schema_and_persistence() -> None:
    trace = generate_runtime_trace()

    assert trace["milestone_id"] == "M5.6"
    assert trace["artifact_type"] == "canonical_runtime_trace"
    assert trace["schema_version"] == 1
    assert trace["creation_paths_checked"]["questionnaire"] is True
    assert trace["creation_paths_checked"]["description"] is True
    assert trace["creation_paths_checked"]["raw_chat_data"] is True
    assert trace["creation_paths_checked"]["raw_chat_cycle"] > 0
    assert len(trace["trace"]["turns"]) == 4
    assert trace["trace"]["persistence"]["cycle_equal"] is True
    assert trace["trace"]["persistence"]["slow_traits_equal"] is True
    assert trace["trace"]["sleep"]["summary_type"] == "dict"


def test_m56_ablation_shows_persona_conditioning_changes_actions() -> None:
    ablation = generate_ablation()

    assert ablation["artifact_type"] == "ablation"
    assert ablation["mechanism"] == "persona_profile_conditioning"
    assert ablation["passed"] is True
    assert ablation["metrics"]["different_action_positions"] > 0
    assert ablation["control"]["action_distribution"] != ablation["treatment"]["action_distribution"]


def test_m56_stress_and_performance_gates() -> None:
    stress = generate_stress()
    performance = generate_performance()

    assert stress["artifact_type"] == "stress_failure_injection"
    assert stress["passed"] is True
    assert stress["checks"]["long_input"]["no_crash"] is True
    assert stress["checks"]["controlled_no_agent_failure"]["raised_runtime_error"] is True
    assert stress["checks"]["blocked_topic_filter"]["blocked"] is True
    assert stress["checks"]["precision_health_filter"]["anomaly_reported"] is True

    assert performance["artifact_type"] == "performance"
    assert performance["passed"] is True
    assert performance["turns"] == 20
    assert performance["latency_ms"]["p95"] < PERFORMANCE_P95_TARGET_MS


def test_m56_generate_all_writes_complete_acceptance_bundle(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    report = generate_all(artifacts_dir, reports_dir)

    assert report["status"] == "PASS"
    assert report["decision"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert not report["findings"]

    waived = set(report["waived_requirements"])
    assert "generic REST/WebSocket persona API" in waived
    assert "POST /persona/{id}/scenario endpoint" in waived

    paths = report["artifacts"]
    for path in paths.values():
        artifact = Path(path)
        if not artifact.is_absolute():
            artifact = tmp_path / artifact
        assert artifact.exists(), path
        json.loads(artifact.read_text(encoding="utf-8"))

    statuses = {gate["id"]: gate["status"] for gate in report["gates"]}
    assert statuses["G1"] == "PASS"
    assert statuses["G2"] == "PASS"
    assert statuses["G3"] == "PASS"
    assert statuses["G4"] == "PASS"
    assert statuses["G5"] == "WAIVED_BY_OWNER"
    assert statuses["G6"] == "WAIVED_BY_OWNER"
