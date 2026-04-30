"""M5.7 end-to-end integration trial tests."""

from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.integration_trial import (
    IntegrationTrialConfig,
    run_integration_trial,
)


def test_m57_default_config_matches_scope() -> None:
    config = IntegrationTrialConfig()

    assert config.personas == 5
    assert config.turns_per_persona >= 200
    assert config.simulated_days >= 5
    assert config.min_partners >= 3


def test_m57_trial_writes_complete_bundle(tmp_path: Path) -> None:
    config = IntegrationTrialConfig(
        personas=2,
        turns_per_persona=12,
        simulated_days=3,
        seed=57,
        run_cross_context=False,
    )
    report = run_integration_trial(
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        config=config,
    )

    assert report["milestone_id"] == "M5.7"
    assert report["status"] == "PASS"
    assert report["decision"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert report["summary_metrics"]["qualified_users"] >= 2
    assert report["summary_metrics"]["personas"] == 2
    assert report["summary_metrics"]["turns_per_persona"] == 12
    assert report["summary_metrics"]["total_longitudinal_turns"] == 24
    assert report["summary_metrics"]["adversarial_passed"] is True
    assert report["summary_metrics"]["min_memory_episodic_entries"] > 0

    statuses = {gate["id"]: gate["status"] for gate in report["gates"]}
    assert statuses == {
        "G1": "PASS",
        "G2": "PASS",
        "G3": "PASS",
        "G4": "PASS",
        "G5": "PASS",
        "G6": "PASS",
    }

    for path_text in report["artifacts"].values():
        path = Path(path_text)
        assert path.exists(), path
        json.loads(path.read_text(encoding="utf-8"))

    trace = json.loads(Path(report["artifacts"]["trial_trace"]).read_text(encoding="utf-8"))
    assert trace["chain"] == [
        "raw_chat",
        "m50_pipeline",
        "m52_implantation",
        "m56_runtime",
        "m57_trial",
    ]


def test_m57_longitudinal_persona_schema(tmp_path: Path) -> None:
    config = IntegrationTrialConfig(
        personas=1,
        turns_per_persona=8,
        simulated_days=2,
        seed=91,
        run_cross_context=False,
    )
    report = run_integration_trial(
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        config=config,
    )

    longitudinal = json.loads(
        Path(report["artifacts"]["longitudinal"]).read_text(encoding="utf-8")
    )
    persona = longitudinal["personas"][0]
    assert persona["turns"] == 8
    assert persona["sleep_cycles"] == 2
    assert persona["personality_stability"]["passed"] is True
    assert persona["memory_coherence"]["passed"] is True
    assert persona["sample_transcript"]
    assert persona["action_distribution"]
