from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "group_memory_alpha_audit_v1"

ALPHA_REPORT_PATH = REPORTS_DIR / "group_memory_alpha_acceptance_report.json"
ALPHA_SUMMARY_PATH = REPORTS_DIR / "group_memory_alpha_acceptance_summary.md"
HELD_OUT_FIXTURE = ROOT / "tests" / "fixtures" / "m18_held_out_group_chat.json"

ALPHA_SCENARIO_TESTS: tuple[str, ...] = (
    "tests/test_m18_group_chat_acceptance.py::test_m18_4_memory_rows_stamp_source_audience_scope",
    "tests/test_m18_group_chat_acceptance.py::test_m18_4_scenario_c_cross_user_memory_privacy_boundary",
    "tests/test_m18_group_chat_acceptance.py::test_m18_4_group_common_memory_allows_bounded_cross_user_reuse",
    "tests/test_m18_group_chat_acceptance.py::test_m18_4_soft_boundary_cross_user_recall_is_abstracted",
    "tests/test_m18_group_chat_acceptance.py::test_m18_4_dm_only_fact_is_not_reused_as_group_visible_fact",
    "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_a_turn_taking_correction_continuity_with_restart",
    "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_b_ambiguous_addressee_clarification",
    "tests/test_m18_group_chat_acceptance.py::test_m18_6_held_out_group_replay_fixture",
)

ALPHA_REGRESSION_TESTS: tuple[str, ...] = (
    "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_persists_ingress_evidence_band_in_turn_diagnostics",
    "tests/test_mvp_dialogue_runtime.py::test_group_long_term_memory_write_carries_traceable_owner_fields",
    "tests/test_mvp_dialogue_runtime.py::test_group_thinking_habit_updates_carry_traceable_owner_fields",
    "tests/test_mvp_dialogue_runtime.py::test_group_pacing_feedback_habit_carries_traceable_owner_fields",
    "tests/test_mvp_dialogue_runtime.py::test_group_post_reply_memory_updates_carry_traceable_owner_fields",
    "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_scripted_m17_confirmed_path_writes_settlement_and_calibration",
    "tests/test_mvp_dialogue_runtime.py::test_group_m17_settlement_stays_on_origin_speaker_across_other_group_turns",
    "tests/test_m16_1_runner_loop.py::test_runner_treats_group_no_reply_as_completed_without_error",
    "tests/test_m16_1_delivery_surface.py::test_external_delivery_surface_ready_allows_drain_without_ws",
    "tests/test_telegram_connector.py::test_normalize_private_message_routes_to_dm_session",
    "tests/test_telegram_connector.py::test_normalize_group_topic_message_captures_reply_and_mentions",
    "tests/test_telegram_connector.py::test_connector_ingest_update_sends_runtime_reply_back_to_telegram",
    "tests/test_telegram_connector.py::test_connector_no_reply_does_not_send_telegram_message",
    "tests/test_telegram_connector.py::test_connector_restart_preserves_group_thread_continuity",
    "tests/test_telegram_connector.py::test_connector_idle_drain_no_traceable_outreach_sends_nothing",
    "tests/test_telegram_connector.py::test_connector_idle_drain_sends_traceable_proactive_message_once",
    "tests/test_telegram_connector.py::test_connector_idle_drain_blocks_when_m13_assessor_disallows_delivery",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _artifact_record(path: Path) -> dict[str, object]:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    modified_at = (
        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        if exists
        else None
    )
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "modified_at": modified_at,
    }


def _suite_execution_record(*, label: str, nodeids: tuple[str, ...], execute: bool) -> dict[str, object]:
    if not execute:
        return {
            "label": label,
            "executed": False,
            "passed": False,
            "returncode": None,
            "command": [],
            "nodeids": list(nodeids),
            "stdout": "",
            "stderr": "",
            "started_at": None,
            "completed_at": None,
        }
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", *nodeids]
    started_at = _now_iso()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=180,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired as exc:
        completed_at = _now_iso()
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "label": label,
            "executed": True,
            "passed": False,
            "returncode": None,
            "command": command,
            "nodeids": list(nodeids),
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "started_at": started_at,
            "completed_at": completed_at,
            "timeout_seconds": 180,
        }
    completed_at = _now_iso()
    return {
        "label": label,
        "executed": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "nodeids": list(nodeids),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _checklist_payload(
    *,
    scenarios_ok: bool,
    regressions_ok: bool,
    fixture_exists: bool,
) -> dict[str, dict[str, object]]:
    return {
        "structured_ingress_traceability": {
            "passed": regressions_ok,
            "evidence": [
                "tests/test_telegram_connector.py::test_normalize_private_message_routes_to_dm_session",
                "tests/test_telegram_connector.py::test_normalize_group_topic_message_captures_reply_and_mentions",
                "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_persists_ingress_evidence_band_in_turn_diagnostics",
            ],
        },
        "memory_write_traceability": {
            "passed": scenarios_ok and regressions_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_memory_rows_stamp_source_audience_scope",
                "tests/test_mvp_dialogue_runtime.py::test_group_long_term_memory_write_carries_traceable_owner_fields",
                "tests/test_mvp_dialogue_runtime.py::test_group_thinking_habit_updates_carry_traceable_owner_fields",
                "tests/test_mvp_dialogue_runtime.py::test_group_pacing_feedback_habit_carries_traceable_owner_fields",
                "tests/test_mvp_dialogue_runtime.py::test_group_post_reply_memory_updates_carry_traceable_owner_fields",
            ],
        },
        "group_recall_privacy_policy": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_scenario_c_cross_user_memory_privacy_boundary",
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_group_common_memory_allows_bounded_cross_user_reuse",
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_soft_boundary_cross_user_recall_is_abstracted",
            ],
        },
        "dm_only_fact_not_publicly_reused": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_dm_only_fact_is_not_reused_as_group_visible_fact",
            ],
        },
        "deterministic_group_replay_and_settlement": {
            "passed": scenarios_ok and regressions_ok and fixture_exists,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_a_turn_taking_correction_continuity_with_restart",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_b_ambiguous_addressee_clarification",
                "tests/test_m18_group_chat_acceptance.py::test_m18_6_held_out_group_replay_fixture",
                "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_scripted_m17_confirmed_path_writes_settlement_and_calibration",
                "tests/test_mvp_dialogue_runtime.py::test_group_m17_settlement_stays_on_origin_speaker_across_other_group_turns",
            ],
        },
        "telegram_delivery_paths": {
            "passed": regressions_ok,
            "evidence": [
                "tests/test_telegram_connector.py::test_connector_ingest_update_sends_runtime_reply_back_to_telegram",
                "tests/test_telegram_connector.py::test_connector_no_reply_does_not_send_telegram_message",
                "tests/test_telegram_connector.py::test_connector_restart_preserves_group_thread_continuity",
                "tests/test_m16_1_runner_loop.py::test_runner_treats_group_no_reply_as_completed_without_error",
            ],
        },
        "telegram_bounded_proactive_delivery": {
            "passed": regressions_ok,
            "evidence": [
                "tests/test_m16_1_delivery_surface.py::test_external_delivery_surface_ready_allows_drain_without_ws",
                "tests/test_telegram_connector.py::test_connector_idle_drain_no_traceable_outreach_sends_nothing",
                "tests/test_telegram_connector.py::test_connector_idle_drain_sends_traceable_proactive_message_once",
                "tests/test_telegram_connector.py::test_connector_idle_drain_blocks_when_m13_assessor_disallows_delivery",
            ],
        },
        "blocking_failure_classes": {
            "passed": scenarios_ok and regressions_ok and fixture_exists,
            "evidence": [
                "wrong speaker attribution -> tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_a_turn_taking_correction_continuity_with_restart",
                "wrong addressee selection -> tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_b_ambiguous_addressee_clarification",
                "privacy leak -> tests/test_m18_group_chat_acceptance.py::test_m18_4_scenario_c_cross_user_memory_privacy_boundary",
                "lost thread continuity -> tests/test_telegram_connector.py::test_connector_restart_preserves_group_thread_continuity",
                "replay inconsistency -> tests/test_m18_group_chat_acceptance.py::test_m18_6_held_out_group_replay_fixture",
            ],
        },
    }


def build_group_memory_alpha_report(
    *,
    scenario_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
    execute: bool = True,
) -> dict[str, object]:
    generated_at = _now_iso()
    scenario_record = scenario_execution or _suite_execution_record(
        label="group-memory-alpha-scenarios",
        nodeids=ALPHA_SCENARIO_TESTS,
        execute=execute,
    )
    regression_record = regression_execution or _suite_execution_record(
        label="group-memory-alpha-regressions",
        nodeids=ALPHA_REGRESSION_TESTS,
        execute=execute,
    )
    fixture_exists = HELD_OUT_FIXTURE.exists()
    scenarios_ok = bool(scenario_record.get("passed"))
    regressions_ok = bool(regression_record.get("passed"))
    checklist = _checklist_payload(
        scenarios_ok=scenarios_ok,
        regressions_ok=regressions_ok,
        fixture_exists=fixture_exists,
    )
    all_checklist_ok = all(bool(item.get("passed")) for item in checklist.values())
    status = "PASS" if scenarios_ok and regressions_ok and fixture_exists and all_checklist_ok else "FAIL"
    return {
        "milestone_id": "group_memory_alpha",
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "readiness_checklist": checklist,
        "scenario_execution": scenario_record,
        "regression_execution": regression_record,
        "target_surfaces": ["telegram_dm", "telegram_group", "telegram_topic"],
        "judge_types": {
            "structured_assertions": [
                "ingress evidence trace",
                "memory row owner fields",
                "privacy policy mode",
                "connector delivery/no-delivery behavior",
            ],
            "deterministic_replay_assertions": [
                "held-out Chinese ambiguity replay",
                "group settlement continuity across restart and interleaved turns",
            ],
            "rubric_based_judgments": [],
        },
        "blocking_failure_classes": [
            "wrong speaker attribution",
            "wrong addressee selection",
            "privacy leak",
            "lost thread continuity",
            "replay inconsistency",
        ],
        "review_questions": [
            "what was written to memory",
            "who owned it",
            "who could hear it",
            "why it was recalled",
            "why it was or was not said aloud",
        ],
        "artifacts": {
            "held_out_fixture": _artifact_record(HELD_OUT_FIXTURE),
            "report": _artifact_record(ALPHA_REPORT_PATH),
            "summary": _artifact_record(ALPHA_SUMMARY_PATH),
        },
        "path_boundary": "Telegram-first Path B memory-dynamics group-chat alpha only",
    }


def write_group_memory_alpha_acceptance_artifacts(
    *,
    scenario_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
    execute: bool = True,
) -> dict[str, object]:
    report = build_group_memory_alpha_report(
        scenario_execution=scenario_execution,
        regression_execution=regression_execution,
        execute=execute,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALPHA_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = "\n".join(
        [
            "# Group Memory Alpha Acceptance Summary",
            "",
            f"- status: {report['status']}",
            f"- generated_at: {report['generated_at']}",
            f"- held_out_fixture: {HELD_OUT_FIXTURE}",
            "- checklist:",
            *[
                f"  - {name}: {'PASS' if item['passed'] else 'FAIL'}"
                for name, item in report["readiness_checklist"].items()
            ],
        ]
    )
    ALPHA_SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    report["artifacts"]["report"] = _artifact_record(ALPHA_REPORT_PATH)
    report["artifacts"]["summary"] = _artifact_record(ALPHA_SUMMARY_PATH)
    ALPHA_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
