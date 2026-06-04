from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import segmentum.group_memory_alpha_audit as alpha_audit


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
        "started_at": "2026-06-05T08:00:00+00:00",
        "completed_at": "2026-06-05T08:00:10+00:00",
    }


def test_group_memory_alpha_report_contains_required_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        original_report = alpha_audit.ALPHA_REPORT_PATH
        original_summary = alpha_audit.ALPHA_SUMMARY_PATH
        original_fixture = alpha_audit.HELD_OUT_FIXTURE
        try:
            alpha_audit.ALPHA_REPORT_PATH = root / "group_memory_alpha_acceptance_report.json"
            alpha_audit.ALPHA_SUMMARY_PATH = root / "group_memory_alpha_acceptance_summary.md"
            alpha_audit.HELD_OUT_FIXTURE = root / "held_out_group_chat.json"
            alpha_audit.HELD_OUT_FIXTURE.write_text("[]", encoding="utf-8")

            report = alpha_audit.write_group_memory_alpha_acceptance_artifacts(
                scenario_execution=_execution(alpha_audit.ALPHA_SCENARIO_TESTS),
                regression_execution=_execution(alpha_audit.ALPHA_REGRESSION_TESTS),
                execute=False,
            )

            saved = json.loads(alpha_audit.ALPHA_REPORT_PATH.read_text(encoding="utf-8"))
            assert saved["milestone_id"] == "group_memory_alpha"
            assert saved["schema_version"] == alpha_audit.SCHEMA_VERSION
            assert saved["status"] == "PASS"
            assert saved["readiness_checklist"]["structured_ingress_traceability"]["passed"] is True
            assert saved["readiness_checklist"]["telegram_bounded_proactive_delivery"]["passed"] is True
            assert saved["judge_types"]["deterministic_replay_assertions"]
            assert saved["target_surfaces"] == ["telegram_dm", "telegram_group", "telegram_topic"]
            assert Path(alpha_audit.ALPHA_SUMMARY_PATH).exists()
            assert report["path_boundary"].startswith("Telegram-first Path B")
        finally:
            alpha_audit.ALPHA_REPORT_PATH = original_report
            alpha_audit.ALPHA_SUMMARY_PATH = original_summary
            alpha_audit.HELD_OUT_FIXTURE = original_fixture


def test_group_memory_alpha_report_fails_when_fixture_or_suites_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        original_fixture = alpha_audit.HELD_OUT_FIXTURE
        try:
            alpha_audit.HELD_OUT_FIXTURE = root / "missing_fixture.json"
            report = alpha_audit.build_group_memory_alpha_report(
                scenario_execution=_execution(alpha_audit.ALPHA_SCENARIO_TESTS, passed=False),
                regression_execution=_execution(alpha_audit.ALPHA_REGRESSION_TESTS, passed=True),
                execute=False,
            )
            assert report["status"] == "FAIL"
            assert report["readiness_checklist"]["deterministic_group_replay_and_settlement"]["passed"] is False
        finally:
            alpha_audit.HELD_OUT_FIXTURE = original_fixture


def test_group_memory_alpha_suite_execution_record_clears_pytest_addopts(monkeypatch) -> None:
    captured: list[list[str]] = []

    def _fake_run(command, **kwargs):
        captured.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(alpha_audit.subprocess, "run", _fake_run)
    record = alpha_audit._suite_execution_record(
        label="group-memory-alpha-scenarios",
        nodeids=("tests/test_m18_group_chat_acceptance.py::test_m18_6_held_out_group_replay_fixture",),
        execute=True,
    )

    assert record["passed"] is True
    assert captured
    assert captured[0][0] == sys.executable
    assert captured[0][1:5] == ["-m", "pytest", "-o", "addopts="]
    assert captured[0][-1] == "tests/test_m18_group_chat_acceptance.py::test_m18_6_held_out_group_replay_fixture"
