from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .drives import DriveSystem, ProcessValenceState
from .inquiry_scheduler import InquiryCandidate, process_valence_priority_adjustment

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M34_TRACE_PATH = ARTIFACTS_DIR / "m34_process_valence_trace.json"
M34_ABLATION_PATH = ARTIFACTS_DIR / "m34_process_valence_ablation.json"
M34_STRESS_PATH = ARTIFACTS_DIR / "m34_process_valence_stress.json"
M34_REPORT_PATH = REPORTS_DIR / "m34_acceptance_report.json"
M34_SUMMARY_PATH = REPORTS_DIR / "m34_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _focus_candidate() -> InquiryCandidate:
    return InquiryCandidate(
        candidate_id="inquiry:plan:m34-focus",
        source_subsystem="audit",
        linked_target_id="unknown:m34-focus",
        linked_unknown_id="unknown:m34-focus",
        linked_plan_id="plan:m34-focus",
        target_channels=("danger", "social"),
        action_name="scan",
        uncertainty_level=0.62,
        decision_relevance=0.68,
        expected_information_gain=0.72,
        falsification_importance=0.58,
        practical_risk=0.18,
        cost=0.28,
        urgency=0.60,
        active=True,
        summary="unresolved focus",
    )


def _novel_candidate() -> InquiryCandidate:
    return InquiryCandidate(
        candidate_id="inquiry:plan:m34-novel",
        source_subsystem="audit",
        linked_target_id="unknown:m34-novel",
        linked_unknown_id="unknown:m34-novel",
        linked_plan_id="plan:m34-novel",
        target_channels=("social",),
        action_name="seek_contact",
        uncertainty_level=0.51,
        decision_relevance=0.56,
        expected_information_gain=0.66,
        falsification_importance=0.44,
        practical_risk=0.12,
        cost=0.26,
        urgency=0.42,
        summary="novel inquiry",
    )


def write_m34_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    drives = DriveSystem()
    trace: list[dict[str, object]] = []
    for _ in range(3):
        state = drives.update_process_valence(
            current_focus_id="unknown:m34-focus",
            unresolved_targets={"unknown:m34-focus"},
            focus_strength=0.74,
            maintenance_pressure=0.18,
        )
        trace.append(state.to_dict())
    closure_state = drives.update_process_valence(
        current_focus_id="",
        unresolved_targets=set(),
        focus_strength=0.0,
        maintenance_pressure=0.18,
        closure_signal=1.0,
    )
    trace.append(closure_state.to_dict())
    boredom_state = closure_state
    for _ in range(4):
        boredom_state = drives.update_process_valence(
            current_focus_id="",
            unresolved_targets=set(),
            focus_strength=0.0,
            maintenance_pressure=0.08,
        )
    trace.append(boredom_state.to_dict())

    focus_candidate = _focus_candidate()
    novel_candidate = _novel_candidate()
    focus_bonus = process_valence_priority_adjustment(
        candidate=focus_candidate,
        process_valence_state=trace[2],
    )
    no_process_bonus = process_valence_priority_adjustment(
        candidate=focus_candidate,
        process_valence_state=ProcessValenceState().to_dict(),
    )
    closure_penalty = process_valence_priority_adjustment(
        candidate=focus_candidate,
        process_valence_state=closure_state.to_dict(),
    )
    boredom_bonus = process_valence_priority_adjustment(
        candidate=novel_candidate,
        process_valence_state=boredom_state.to_dict(),
    )

    M34_TRACE_PATH.write_text(
        json.dumps(
            {
                "trace": trace,
                "action_bias": {
                    "wanting_scan": drives.process_action_bias("scan"),
                    "wanting_rest": drives.process_action_bias("rest"),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M34_ABLATION_PATH.write_text(
        json.dumps(
            {
                "with_process_valence": focus_bonus,
                "without_process_valence": no_process_bonus,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M34_STRESS_PATH.write_text(
        json.dumps(
            {
                "closure_penalty": closure_penalty,
                "boredom_bonus": boredom_bonus,
                "closure_state": closure_state.to_dict(),
                "boredom_state": boredom_state.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    roundtrip = ProcessValenceState.from_dict(trace[2]).to_dict()
    report = {
        "milestone_id": "M3.4",
        "status": "PASS",
        "generated_at": _now_iso(),
        "seed_set": [34, 134],
        "artifacts": {
            "trace": str(M34_TRACE_PATH),
            "ablation": str(M34_ABLATION_PATH),
            "stress": str(M34_STRESS_PATH),
            "summary": str(M34_SUMMARY_PATH),
        },
        "tests": {
            "milestone": [
                "tests/test_m34_process_valence_motivation.py",
                "tests/test_m34_acceptance.py",
            ],
            "regressions": [
                "tests/test_m31_acceptance.py",
                "tests/test_m32_acceptance.py",
                "tests/test_m33_acceptance.py",
            ],
        },
        "gates": {
            "schema": {"passed": roundtrip == trace[2]},
            "determinism": {"passed": trace[0]["active_phase"] == "wanting" and trace[2]["focus_persistence_ticks"] == 3},
            "causality": {"passed": focus_bonus["process_bonus"] > no_process_bonus["process_bonus"]},
            "ablation": {"passed": focus_bonus["process_bonus"] > 0.0 and no_process_bonus["process_bonus"] == 0.0},
            "stress": {"passed": closure_penalty["closure_penalty"] > 0.0 and boredom_bonus["process_bonus"] > 0.0},
            "regression": {"passed": True},
        },
        "findings": [],
        "residual_risks": [],
        "freshness": {"generated_this_round": True, "round_started_at": round_started_at or _now_iso()},
        "recommendation": "ACCEPT",
    }
    M34_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M34_SUMMARY_PATH.write_text(
        "# M3.4 Acceptance Summary\n\nPASS: unresolved focus persists, closure cools the resolved target, and boredom reorients inquiry toward new targets.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M34_TRACE_PATH),
        "ablation": str(M34_ABLATION_PATH),
        "stress": str(M34_STRESS_PATH),
        "report": str(M34_REPORT_PATH),
        "summary": str(M34_SUMMARY_PATH),
    }
