import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m11_acceptance_artifact_generation_and_replay():
    subprocess.run(
        [sys.executable, "scripts/generate_m11_acceptance_artifacts.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads((ROOT / "reports/m11_acceptance_report.json").read_text(encoding="utf-8"))
    artifact = json.loads((ROOT / "artifacts/m11_acceptance/acceptance_artifact.json").read_text(encoding="utf-8"))
    assert report["status"] == "ACCEPT"
    assert report["structural_pass"] is True
    assert report["behavioral_pass"] is True
    assert report["phenomenological_pass"] is True
    assert report["replay_check"]["byte_identical"] is True
    for field in (
        "turns",
        "extractor_outputs",
        "user_model_before_after",
        "prediction_ledger",
        "prediction_proposals",
        "reliability_ledger_updates",
        "memory_value_compositions",
        "evidence_cards",
        "reply_policy_effects",
        "quarantined_hypotheses",
        "calibration_audit_report",
    ):
        assert field in artifact


def test_m11_acceptance_fixture_pairs_are_not_self_copies():
    for folder in (ROOT / "fixtures/m11").glob("*/*"):
        input_path = folder / "input_judgments.json"
        expected_path = folder / "expected_state_trace.json"
        rationale_path = folder / "rationale.md"
        assert input_path.exists(), folder
        assert expected_path.exists(), folder
        assert rationale_path.exists(), folder
        assert input_path.read_text(encoding="utf-8") != expected_path.read_text(encoding="utf-8")
