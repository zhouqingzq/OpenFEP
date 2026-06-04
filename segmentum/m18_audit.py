from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "m18_audit_v1"

M18_REPORT_PATH = REPORTS_DIR / "m18_acceptance_report.json"
M18_SUMMARY_PATH = REPORTS_DIR / "m18_acceptance_summary.md"
M18_HELD_OUT_FIXTURE = ROOT / "tests" / "fixtures" / "m18_held_out_group_chat.json"

M18_SCENARIO_TESTS: tuple[str, ...] = (
    "tests/test_m18_group_chat_acceptance.py",
)

M18_REGRESSION_TESTS: tuple[str, ...] = (
    "tests/test_m16_1_gateway_ws.py::test_bridge_snapshot_preserves_group_turn_metadata",
    "tests/test_m16_1_gateway_ws.py::test_bridge_snapshot_loads_legacy_turn_row_without_claiming_native_group_ownership",
    "tests/test_m16_1_gateway_ws.py::test_ws_client_input_appends_event_without_inline_turn",
    "tests/test_mvp_dialogue_runtime.py::test_chat_request_group_turn_envelope_reaches_mvp_runtime",
    "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_group_participants_alternate_without_collapsing",
    "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_persists_group_turn_binding_and_turn_log",
    "tests/test_mvp_dialogue_runtime.py::test_group_reply_policy_prefers_explicit_reply_to_over_third_party_target_drift",
    "tests/test_mvp_dialogue_runtime.py::test_entity_binding_pronoun_inheritance_survives_group_turn_binding",
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
    command = [sys.executable, "-m", "pytest", "-o", 'addopts=""', *nodeids]
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
        "speaker_separation": {
            "passed": regressions_ok,
            "evidence": [
                "tests/test_mvp_dialogue_runtime.py::test_chat_request_group_turn_envelope_reaches_mvp_runtime",
                "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_group_participants_alternate_without_collapsing",
                "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_persists_group_turn_binding_and_turn_log",
            ],
        },
        "addressing_and_target_resolution": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_b_ambiguous_addressee_clarification",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_reply_to_named_third_party_when_assistant_is_asked_about_them",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_defer_side_thread_when_prior_pending_answer_is_still_active",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_policy_choice_is_deterministic_for_same_structured_input",
                "tests/test_mvp_dialogue_runtime.py::test_group_reply_policy_prefers_explicit_reply_to_over_third_party_target_drift",
                "tests/test_mvp_dialogue_runtime.py::test_entity_binding_pronoun_inheritance_survives_group_turn_binding",
            ],
        },
        "group_transcript_ownership": {
            "passed": regressions_ok,
            "evidence": [
                "tests/test_m16_1_gateway_ws.py::test_bridge_snapshot_preserves_group_turn_metadata",
                "tests/test_m16_1_gateway_ws.py::test_bridge_snapshot_loads_legacy_turn_row_without_claiming_native_group_ownership",
                "tests/test_mvp_dialogue_runtime.py::test_mvp_runtime_persists_group_turn_binding_and_turn_log",
            ],
        },
        "group_memory_and_privacy_boundaries": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_scenario_c_cross_user_memory_privacy_boundary",
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_group_common_memory_allows_bounded_cross_user_reuse",
                "tests/test_m18_group_chat_acceptance.py::test_m18_4_soft_boundary_cross_user_recall_is_abstracted",
            ],
        },
        "multi_party_reply_selection": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_reply_to_whole_group_when_assistant_and_human_are_joint_addressees",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_intentional_no_reply_for_human_side_thread",
            ],
        },
        "cross_turn_social_continuity": {
            "passed": scenarios_ok,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_scenario_a_turn_taking_correction_continuity_with_restart",
                "tests/test_m18_group_chat_acceptance.py::test_m18_5_defer_side_thread_when_prior_pending_answer_is_still_active",
            ],
        },
        "end_to_end_group_scenarios": {
            "passed": scenarios_ok and fixture_exists,
            "evidence": [
                "tests/test_m18_group_chat_acceptance.py",
                str(M18_HELD_OUT_FIXTURE),
            ],
        },
    }


def build_m18_acceptance_report(
    *,
    scenario_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
    execute: bool = True,
) -> dict[str, object]:
    generated_at = _now_iso()
    scenario_record = scenario_execution or _suite_execution_record(
        label="m18-scenarios",
        nodeids=M18_SCENARIO_TESTS,
        execute=execute,
    )
    regression_record = regression_execution or _suite_execution_record(
        label="m18-regressions",
        nodeids=M18_REGRESSION_TESTS,
        execute=execute,
    )
    fixture_exists = M18_HELD_OUT_FIXTURE.exists()
    scenarios_ok = bool(scenario_record.get("passed"))
    regressions_ok = bool(regression_record.get("passed"))
    checklist = _checklist_payload(
        scenarios_ok=scenarios_ok,
        regressions_ok=regressions_ok,
        fixture_exists=fixture_exists,
    )
    all_checklist_ok = all(bool(item.get("passed")) for item in checklist.values())
    readiness_pass = scenarios_ok and regressions_ok and fixture_exists and all_checklist_ok
    status = "PASS" if readiness_pass else "FAIL"
    return {
        "milestone_id": "M18",
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "readiness_checklist": checklist,
        "scenario_execution": scenario_record,
        "regression_execution": regression_record,
        "judge_types": {
            "structured_assertions": [
                "speaker ownership",
                "group reply policy action",
                "privacy decision",
                "thread continuity state",
            ],
            "deterministic_replay_assertions": [
                "held-out replay fixture",
                "restart continuity continuation",
            ],
            "rubric_based_judgments": [],
        },
        "bounded_operating_envelope": {
            "active_participants_recent_context": "3-5",
            "candidate_targets_per_turn": 6,
            "unresolved_group_thread_slots": 8,
            "replay_window": "bounded transcript tail plus durable carry-forward state",
        },
        "blocking_failure_classes": [
            "speaker attribution error",
            "wrong addressee",
            "wrong third-party target",
            "privacy leak",
            "dropped unresolved thread",
            "restart drift",
            "deterministic policy replay mismatch",
            "false pass caused by missing structured input evidence",
        ],
        "artifacts": {
            "held_out_fixture": _artifact_record(M18_HELD_OUT_FIXTURE),
            "report": _artifact_record(M18_REPORT_PATH),
            "summary": _artifact_record(M18_SUMMARY_PATH),
        },
        "path_boundary": "Path B only; no Path A or Streamlit legacy readiness claim",
    }


def write_m18_acceptance_artifacts(
    *,
    scenario_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
    execute: bool = True,
) -> dict[str, object]:
    report = build_m18_acceptance_report(
        scenario_execution=scenario_execution,
        regression_execution=regression_execution,
        execute=execute,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    M18_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = "\n".join(
        [
            "# M18 Acceptance Summary",
            "",
            f"- status: {report['status']}",
            f"- generated_at: {report['generated_at']}",
            f"- held_out_fixture: {M18_HELD_OUT_FIXTURE}",
            "- checklist:",
            *[
                f"  - {name}: {'PASS' if item['passed'] else 'FAIL'}"
                for name, item in report["readiness_checklist"].items()
            ],
        ]
    )
    M18_SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    report["artifacts"]["report"] = _artifact_record(M18_REPORT_PATH)
    report["artifacts"]["summary"] = _artifact_record(M18_SUMMARY_PATH)
    M18_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
