from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import segmentum.m18_audit as m18_audit


def _execution(nodeids: tuple[str, ...], *, passed: bool = True) -> dict[str, object]:
    return {
        "label": "injected",
        "executed": True,
        "passed": passed,
        "returncode": 0 if passed else 1,
        "command": ["python", "-m", "pytest", *nodeids],
        "nodeids": list(nodeids),
        "stdout": "simulated",
        "stderr": "",
        "started_at": "2026-06-04T08:00:00+00:00",
        "completed_at": "2026-06-04T08:00:10+00:00",
    }


def test_m18_acceptance_report_contains_required_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        original_report = m18_audit.M18_REPORT_PATH
        original_summary = m18_audit.M18_SUMMARY_PATH
        original_fixture = m18_audit.M18_HELD_OUT_FIXTURE
        try:
            m18_audit.M18_REPORT_PATH = root / "m18_acceptance_report.json"
            m18_audit.M18_SUMMARY_PATH = root / "m18_acceptance_summary.md"
            m18_audit.M18_HELD_OUT_FIXTURE = root / "m18_held_out_group_chat.json"
            m18_audit.M18_HELD_OUT_FIXTURE.write_text("[]", encoding="utf-8")

            report = m18_audit.write_m18_acceptance_artifacts(
                scenario_execution=_execution(m18_audit.M18_SCENARIO_TESTS),
                regression_execution=_execution(m18_audit.M18_REGRESSION_TESTS),
                execute=False,
            )

            saved = json.loads(m18_audit.M18_REPORT_PATH.read_text(encoding="utf-8"))
            assert saved["milestone_id"] == "M18"
            assert saved["schema_version"] == m18_audit.SCHEMA_VERSION
            assert saved["status"] == "PASS"
            assert saved["readiness_checklist"]["speaker_separation"]["passed"] is True
            assert saved["readiness_checklist"]["end_to_end_group_scenarios"]["passed"] is True
            assert saved["judge_types"]["structured_assertions"]
            assert saved["bounded_operating_envelope"]["active_participants_recent_context"] == "3-5"
            assert Path(m18_audit.M18_SUMMARY_PATH).exists()
            assert report["path_boundary"].startswith("Path B only")
        finally:
            m18_audit.M18_REPORT_PATH = original_report
            m18_audit.M18_SUMMARY_PATH = original_summary
            m18_audit.M18_HELD_OUT_FIXTURE = original_fixture


def test_m18_acceptance_report_fails_when_fixture_or_scenarios_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        original_fixture = m18_audit.M18_HELD_OUT_FIXTURE
        try:
            m18_audit.M18_HELD_OUT_FIXTURE = root / "missing_fixture.json"
            report = m18_audit.build_m18_acceptance_report(
                scenario_execution=_execution(m18_audit.M18_SCENARIO_TESTS, passed=False),
                regression_execution=_execution(m18_audit.M18_REGRESSION_TESTS, passed=True),
                execute=False,
            )
            assert report["status"] == "FAIL"
            assert report["readiness_checklist"]["end_to_end_group_scenarios"]["passed"] is False
        finally:
            m18_audit.M18_HELD_OUT_FIXTURE = original_fixture


def test_m18_suite_execution_record_clears_pytest_addopts(monkeypatch) -> None:
    captured: list[list[str]] = []

    def _fake_run(command, **kwargs):
        captured.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(m18_audit.subprocess, "run", _fake_run)
    record = m18_audit._suite_execution_record(
        label="m18-scenarios",
        nodeids=("tests/test_m18_group_chat_acceptance.py",),
        execute=True,
    )

    assert record["passed"] is True
    assert captured
    assert captured[0][0] == sys.executable
    assert captured[0][1:5] == ["-m", "pytest", "-o", "addopts="]
    assert captured[0][4] == "addopts="
    assert captured[0][-1] == "tests/test_m18_group_chat_acceptance.py"
