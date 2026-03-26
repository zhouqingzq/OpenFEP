from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

MILESTONE_ID = "M2.Total"
SCHEMA_VERSION = "m237_m2_total_acceptance_v1"

M237_SCORECARD_PATH = ARTIFACTS_DIR / "m237_m2_total_scorecard.json"
M237_REPORT_PATH = REPORTS_DIR / "m237_m2_total_acceptance_report.json"
M237_SUMMARY_PATH = REPORTS_DIR / "m237_m2_total_acceptance_summary.md"

DEPENDENCY_REPORTS: dict[str, str] = {
    "M2.20": "m220_acceptance_report.json",
    "M2.22": "m222_acceptance_report.json",
    "M2.23": "m223_acceptance_report.json",
    "M2.24": "m224_acceptance_report.json",
    "M2.25": "m225_acceptance_report.json",
    "M2.27": "m227_acceptance_report.json",
    "M2.29": "m229_acceptance_report.json",
    "M2.30": "m230_acceptance_report.json",
    "M2.31": "m231_acceptance_report.json",
    "M2.35": "m235_acceptance_report.json",
    "M2.36": "m236_open_continuity_report.json",
}

PILLARS: tuple[dict[str, object], ...] = (
    {
        "pillar_id": "self_model_initialization",
        "title": "Self-Model Initialization",
        "dependencies": ("M2.20",),
        "target_outcomes": (
            "人格与自我模型可由叙事文本初始化",
        ),
        "next_action": "Keep narrative initialization replay current-round and schema-stable.",
    },
    {
        "pillar_id": "autonomous_homeostasis_and_continuity",
        "title": "Autonomous Homeostasis And Continuity",
        "dependencies": ("M2.22", "M2.36"),
        "target_outcomes": (
            "持续的自维护与生存调度",
            "主体连续性在压力与重启中保持",
        ),
        "next_action": "Repair long-horizon autonomy and restart continuity until both organism trials pass together.",
    },
    {
        "pillar_id": "functional_conscious_access",
        "title": "Functional Conscious Access",
        "dependencies": ("M2.24",),
        "target_outcomes": (
            "workspace/broadcast 真实影响下游模块",
        ),
        "next_action": "Preserve workspace causal evidence as a total-acceptance dependency.",
    },
    {
        "pillar_id": "unified_narrative_and_repair",
        "title": "Unified Narrative And Repair",
        "dependencies": ("M2.23", "M2.27", "M2.31"),
        "target_outcomes": (
            "统一自我叙事",
            "冲突时一致性修复",
        ),
        "next_action": "Hold identity, subject-state, and reconciliation evidence together under one current-round replay set.",
    },
    {
        "pillar_id": "slow_learning_and_bounded_inquiry",
        "title": "Slow Learning And Bounded Inquiry",
        "dependencies": ("M2.29", "M2.30", "M2.35", "M2.36"),
        "target_outcomes": (
            "睡眠与慢变量学习改变长期策略",
            "开放 inquiry 保持有界且连续",
        ),
        "next_action": "Raise verification yield while preserving bounded inquiry and slow-learning stability.",
    },
    {
        "pillar_id": "bounded_transfer_and_identity_preservation",
        "title": "Bounded Transfer And Identity Preservation",
        "dependencies": ("M2.25", "M2.36"),
        "target_outcomes": (
            "跨环境迁移时保留核心身份与偏好",
        ),
        "next_action": "Repair open-world transfer so identity preservation survives beyond the synthetic continuity trial.",
    },
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float) -> float:
    return round(float(value), 6)


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "modified_at": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            if path.exists()
            else None
        ),
    }


def _report_path(milestone_id: str) -> Path:
    return REPORTS_DIR / DEPENDENCY_REPORTS[milestone_id]


def _load_dependency_reports(
    report_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for milestone_id in DEPENDENCY_REPORTS:
        if report_overrides and milestone_id in report_overrides:
            payloads[milestone_id] = dict(report_overrides[milestone_id])
            continue
        path = _report_path(milestone_id)
        if not path.exists():
            payloads[milestone_id] = {"milestone_id": milestone_id, "status": "MISSING", "recommendation": "BLOCK"}
            continue
        payloads[milestone_id] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def _freshness_state(payload: Mapping[str, object], *, path: Path | None = None) -> bool:
    freshness = payload.get("freshness", {})
    if isinstance(freshness, Mapping):
        for key in ("current_round", "generated_this_round", "current_round_replay", "revalidated_this_round"):
            if key in freshness:
                return bool(freshness.get(key))
    generated_at = _parse_iso8601(payload.get("generated_at"))
    if generated_at is not None:
        return generated_at.date() == datetime.now(timezone.utc).date()
    if path is not None and path.exists():
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return modified.date() == datetime.now(timezone.utc).date()
    return False


def _dependency_status(
    milestone_id: str,
    payload: Mapping[str, object],
    *,
    path: Path | None = None,
) -> dict[str, object]:
    status = str(payload.get("status", "MISSING"))
    recommendation = str(payload.get("recommendation", "BLOCK"))
    passed = status == "PASS" and recommendation in {"ACCEPT", "PASS_WITH_RESIDUAL_RISK", ""}
    current_round = _freshness_state(payload, path=path)
    return {
        "milestone_id": milestone_id,
        "status": status,
        "recommendation": recommendation,
        "passed": passed,
        "current_round": current_round,
        "blocking_reason": (
            ""
            if passed and current_round
            else ("stale_evidence" if passed and not current_round else "milestone_not_passing")
        ),
    }


def build_m237_total_acceptance(
    *,
    report_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    reports = _load_dependency_reports(report_overrides)
    dependency_states: dict[str, dict[str, object]] = {}
    for milestone_id, payload in reports.items():
        path = None if report_overrides and milestone_id in report_overrides else _report_path(milestone_id)
        dependency_states[milestone_id] = _dependency_status(milestone_id, payload, path=path)

    pillar_results: list[dict[str, object]] = []
    for pillar in PILLARS:
        dependency_ids = tuple(str(item) for item in pillar["dependencies"])
        dependency_details = [dependency_states[item] for item in dependency_ids]
        passed = all(item["passed"] and item["current_round"] for item in dependency_details)
        blockers = [item["milestone_id"] for item in dependency_details if not (item["passed"] and item["current_round"])]
        pillar_results.append(
            {
                "pillar_id": pillar["pillar_id"],
                "title": pillar["title"],
                "dependencies": dependency_details,
                "target_outcomes": list(pillar["target_outcomes"]),
                "passed": passed,
                "blockers": blockers,
                "next_action": pillar["next_action"],
            }
        )

    passed_pillars = sum(1 for item in pillar_results if item["passed"])
    total_pillars = len(pillar_results) or 1
    blocking_milestones = sorted(
        {
            blocker
            for pillar in pillar_results
            for blocker in pillar["blockers"]
        }
    )
    status = "PASS" if passed_pillars == total_pillars else "BLOCKED"
    recommendation = "ACCEPT" if status == "PASS" else "CONTINUE_M2_REMEDIATION"
    gap_ratio = _round((total_pillars - passed_pillars) / total_pillars)
    residual_risks: list[str] = []
    for milestone_id in blocking_milestones:
        state = dependency_states[milestone_id]
        if state["blocking_reason"] == "stale_evidence":
            residual_risks.append(f"{milestone_id} lacks current-round replay evidence for total M2 acceptance.")
        else:
            residual_risks.append(f"{milestone_id} is not passing and blocks total M2 acceptance.")

    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": status,
        "recommendation": recommendation,
        "pillars": pillar_results,
        "summary": {
            "passed_pillars": passed_pillars,
            "total_pillars": total_pillars,
            "gap_ratio": gap_ratio,
            "distance_to_acceptance": _round(1.0 - (passed_pillars / total_pillars)),
            "blocking_milestones": blocking_milestones,
        },
        "dependency_states": dependency_states,
        "residual_risks": residual_risks,
    }


def write_m237_total_acceptance_artifacts(
    *,
    report_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, str]:
    payload = build_m237_total_acceptance(report_overrides=report_overrides)
    artifact_records = {
        milestone_id: _artifact_record(_report_path(milestone_id))
        for milestone_id in DEPENDENCY_REPORTS
        if not (report_overrides and milestone_id in report_overrides)
    }
    scorecard = {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": payload["generated_at"],
        "pillar_scorecard": [
            {
                "pillar_id": pillar["pillar_id"],
                "passed": pillar["passed"],
                "blockers": list(pillar["blockers"]),
            }
            for pillar in payload["pillars"]
        ],
        "dependency_artifacts": artifact_records,
        "summary": dict(payload["summary"]),
    }
    final_report = {
        **payload,
        "artifacts": {
            "scorecard": str(M237_SCORECARD_PATH),
            "report": str(M237_REPORT_PATH),
            "summary": str(M237_SUMMARY_PATH),
        },
    }
    M237_SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    M237_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    M237_SCORECARD_PATH.write_text(json.dumps(scorecard, indent=2, ensure_ascii=True), encoding="utf-8")
    M237_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    summary_lines = [
        "# M2 Total Acceptance Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Recommendation: {payload['recommendation']}",
        f"- Passed pillars: {payload['summary']['passed_pillars']} / {payload['summary']['total_pillars']}",
        f"- Distance to acceptance: {payload['summary']['distance_to_acceptance']}",
        f"- Blocking milestones: {', '.join(payload['summary']['blocking_milestones']) or 'none'}",
        "",
        "## Pillars",
        "",
    ]
    for pillar in payload["pillars"]:
        summary_lines.append(
            f"- {pillar['title']}: {'PASS' if pillar['passed'] else 'BLOCKED'}"
            + (f" (blockers: {', '.join(pillar['blockers'])})" if pillar["blockers"] else "")
        )
    summary_lines.extend(
        [
            "",
            "## Residual Risks",
            "",
            *[f"- {item}" for item in payload["residual_risks"]],
        ]
    )
    M237_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "scorecard": str(M237_SCORECARD_PATH),
        "report": str(M237_REPORT_PATH),
        "summary": str(M237_SUMMARY_PATH),
    }
