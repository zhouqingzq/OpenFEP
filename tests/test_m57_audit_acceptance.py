"""Strict-audit acceptance checks for M5.7 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.integration_trial import IntegrationTrialConfig, run_integration_trial


STRICT_REQUIRED_REPORT_FIELDS = {
    "milestone_id",
    "status",
    "generated_at",
    "seed_set",
    "artifacts",
    "tests",
    "gates",
    "findings",
    "residual_risks",
    "freshness",
    "recommendation",
}

STRICT_REQUIRED_EVIDENCE = {
    "schema",
    "determinism",
    "causality",
    "ablation",
    "stress",
    "regression",
    "artifact_freshness",
}


def test_m57_report_satisfies_strict_audit_required_fields(tmp_path: Path) -> None:
    report = run_integration_trial(
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        config=IntegrationTrialConfig(
            personas=2,
            turns_per_persona=10,
            simulated_days=2,
            seed=73,
            run_cross_context=False,
        ),
    )

    assert STRICT_REQUIRED_REPORT_FIELDS <= set(report)
    assert report["status"] == "PASS"
    assert report["decision"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert report["findings"] == []
    assert len(report["seed_set"]) > 1
    assert report["freshness"]["generated_current_round"] is True
    assert set(report["freshness"]["generated_artifact_paths"]) == set(
        report["artifacts"].values()
    )


def test_m57_evidence_categories_and_artifacts_are_complete(tmp_path: Path) -> None:
    report = run_integration_trial(
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        config=IntegrationTrialConfig(
            personas=2,
            turns_per_persona=10,
            simulated_days=2,
            seed=91,
            run_cross_context=False,
        ),
    )

    assert STRICT_REQUIRED_EVIDENCE <= set(report["evidence_categories"])
    assert all(
        item["status"] == "PASS" for item in report["evidence_categories"].values()
    )

    artifacts = report["artifacts"]
    assert artifacts["trial_trace"].endswith("m57_trial_trace.json")
    assert artifacts["ablation"].endswith("m57_comparative.json")
    assert artifacts["adversarial"].endswith("m57_adversarial.json")
    assert artifacts["technical_report"].endswith("m57_integration_report.json")
    assert artifacts["human_summary"].endswith("m57_acceptance_summary.md")

    for path_text in set(artifacts.values()):
        path = Path(path_text)
        assert path.exists(), path
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            text = path.read_text(encoding="utf-8")
            assert "# M5.7 Acceptance Summary" in text
            assert "Residual Risks" in text


def test_m57_declares_milestone_and_regression_test_commands(tmp_path: Path) -> None:
    report = run_integration_trial(
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        config=IntegrationTrialConfig(
            personas=1,
            turns_per_persona=8,
            simulated_days=2,
            seed=57,
            run_cross_context=False,
        ),
    )

    tests = report["tests"]
    assert "test_m57_integration_trial.py" in tests["m57_acceptance"]
    assert "test_m57_audit_acceptance.py" in tests["m57_acceptance"]
    for regression in (
        "test_m50_chat_pipeline.py",
        "test_m51_dialogue_channels.py",
        "test_m52_implantation.py",
        "test_m53_dialogue_action.py",
        "test_m55_cross_context.py",
        "test_m56_runtime.py",
        "test_m56_acceptance_artifacts.py",
    ):
        assert regression in tests["m5_regression"]
