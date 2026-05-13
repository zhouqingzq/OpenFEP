"""Pure view assembly for M12.1 personality reports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams, SECTION_KINDS
from .personality_profile import InsufficientEvidence, PersonalityProfile, ReportStatus
from .plain_language_linter import LinterFinding, lint_report_dict


@dataclass(frozen=True)
class ReportSectionView:
    section_kind: str
    report_section: str
    status: str
    content: dict[str, object] | None
    rendered: str
    confidence_band: str

    def to_dict(self) -> dict[str, object]:
        return {
            "section_kind": self.section_kind,
            "report_section": self.report_section,
            "status": self.status,
            "content": self.content,
            "rendered": self.rendered,
            "confidence_band": self.confidence_band,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReportSectionView":
        content = payload.get("content")
        return cls(
            section_kind=str(payload.get("section_kind", "")),
            report_section=str(payload.get("report_section", "")),
            status=str(payload.get("status", "")),
            content=dict(content) if isinstance(content, Mapping) else None,
            rendered=str(payload.get("rendered", "")),
            confidence_band=str(payload.get("confidence_band", "low")),
        )


@dataclass(frozen=True)
class PersonalityReport:
    report_id: str
    user_id: str
    turn_id: str
    hyperparams_version: str
    report_status: ReportStatus
    sections: tuple[ReportSectionView, ...]
    linter_findings: tuple[LinterFinding, ...]
    trigger_kind: str
    prior_report_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "user_id": self.user_id,
            "turn_id": self.turn_id,
            "hyperparams_version": self.hyperparams_version,
            "report_status": self.report_status,
            "sections": [section.to_dict() for section in self.sections],
            "linter_findings": [finding.to_dict() for finding in self.linter_findings],
            "trigger_kind": self.trigger_kind,
            "prior_report_id": self.prior_report_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalityReport":
        sections = payload.get("sections", [])
        findings = payload.get("linter_findings", [])
        return cls(
            report_id=str(payload.get("report_id", "")),
            user_id=str(payload.get("user_id", "")),
            turn_id=str(payload.get("turn_id", "")),
            hyperparams_version=str(payload.get("hyperparams_version", DEFAULT_HYPERPARAMS.hyperparams_version)),
            report_status=_status(payload.get("report_status")),
            sections=tuple(
                ReportSectionView.from_dict(item)
                for item in sections
                if isinstance(item, Mapping)
            ),
            linter_findings=tuple(
                LinterFinding.from_dict(item)
                for item in findings
                if isinstance(item, Mapping)
            ),
            trigger_kind=str(payload.get("trigger_kind", "")),
            prior_report_id=str(payload.get("prior_report_id", "")),
        )


def assemble_personality_report(
    profile: PersonalityProfile,
    *,
    turn_id: str,
    trigger_kind: str,
    prior_report_id: str = "",
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
    run_linter: bool = True,
) -> PersonalityReport:
    sections = tuple(_section_view(profile, section_kind) for section_kind in SECTION_KINDS)
    status: ReportStatus = "draft"
    step1 = sections[0]
    if step1.status in {"insufficient_evidence", "never_computed"}:
        status = "stale"
    findings: tuple[LinterFinding, ...] = ()
    draft = _report_payload(
        report_id="pending",
        profile=profile,
        turn_id=turn_id,
        trigger_kind=trigger_kind,
        prior_report_id=prior_report_id,
        status=status,
        sections=sections,
        findings=findings,
        hyperparams=hyperparams,
    )
    if run_linter:
        findings = lint_report_dict(draft, hyperparams=hyperparams)
        if findings:
            status = "linter_failed"
        elif status == "draft":
            status = "ready"
    report_id = _deterministic_report_id(
        _report_payload(
            report_id="",
            profile=profile,
            turn_id=turn_id,
            trigger_kind=trigger_kind,
            prior_report_id=prior_report_id,
            status=status,
            sections=sections,
            findings=findings,
            hyperparams=hyperparams,
        )
    )
    return PersonalityReport(
        report_id=report_id,
        user_id=profile.user_id,
        turn_id=turn_id,
        hyperparams_version=hyperparams.hyperparams_version,
        report_status=status,
        sections=sections,
        linter_findings=findings,
        trigger_kind=trigger_kind,
        prior_report_id=prior_report_id,
    )


def ready_report_or_none(report: PersonalityReport) -> PersonalityReport | None:
    return report if report.report_status == "ready" and not report.linter_findings else None


def _section_view(profile: PersonalityProfile, section_kind: str) -> ReportSectionView:
    value = profile.section_for(section_kind)
    if value is None:
        return ReportSectionView(
            section_kind=section_kind,
            report_section=section_kind,
            status="never_computed",
            content=None,
            rendered="insufficient_evidence",
            confidence_band="low",
        )
    content = value.to_dict()
    if isinstance(value, InsufficientEvidence):
        rendered = "insufficient_evidence"
        status = "insufficient_evidence"
    else:
        rendered = _render_section(section_kind, content)
        status = "inferred_hypothesis"
    return ReportSectionView(
        section_kind=section_kind,
        report_section=section_kind,
        status=status,
        content=content,
        rendered=rendered,
        confidence_band=str(content.get("confidence_band", "low")),
    )


def _render_section(section_kind: str, content: Mapping[str, object]) -> str:
    if section_kind == "step_1":
        return str(content.get("summary", "")) or "insufficient_evidence"
    if section_kind == "step_2":
        rows = content.get("evidence_items", [])
        if isinstance(rows, list):
            return " | ".join(str(row.get("content_summary", "")) for row in rows if isinstance(row, Mapping)) or "insufficient_evidence"
    if section_kind == "step_3":
        return " | ".join(str(content.get(key, "")) for key in ("wants", "fears", "default_interpretation") if str(content.get(key, "")))
    if section_kind == "step_4":
        return " | ".join(
            str(row.get("content_summary", ""))
            for row in (content.get("about_self"), content.get("about_others"), content.get("about_world"))
            if isinstance(row, Mapping)
        ) or "insufficient_evidence"
    if section_kind == "step_5":
        return " | ".join(str(content.get(key, "")) for key in ("dominant_emotional_baseline", "threat_response") if str(content.get(key, "")))
    if section_kind == "step_6":
        return " | ".join(str(content.get(key, "")) for key in ("close_relationship_role", "recurring_loop_summary", "conflict_style") if str(content.get(key, "")))
    if section_kind == "step_7":
        stages = content.get("stages", [])
        if isinstance(stages, list):
            return " -> ".join(str(row.get("content_summary", "")) for row in stages if isinstance(row, Mapping)) or "insufficient_evidence"
    if section_kind == "step_8":
        return " | ".join(
            ", ".join(str(item) for item in content.get(key, []) if str(item))
            for key in (
                "stable_parts",
                "fragile_spots",
                "soft_spots",
                "communication_styles_likely_accepted",
                "communication_styles_that_trigger_defenses",
            )
            if isinstance(content.get(key), list) and content.get(key)
        ) or "insufficient_evidence"
    return "insufficient_evidence"


def _report_payload(
    *,
    report_id: str,
    profile: PersonalityProfile,
    turn_id: str,
    trigger_kind: str,
    prior_report_id: str,
    status: ReportStatus,
    sections: tuple[ReportSectionView, ...],
    findings: tuple[LinterFinding, ...],
    hyperparams: M121Hyperparams,
) -> dict[str, object]:
    return {
        "report_id": report_id,
        "user_id": profile.user_id,
        "turn_id": turn_id,
        "hyperparams_version": hyperparams.hyperparams_version,
        "report_status": status,
        "sections": [section.to_dict() for section in sections],
        "linter_findings": [finding.to_dict() for finding in findings],
        "trigger_kind": trigger_kind,
        "prior_report_id": prior_report_id,
    }


def _deterministic_report_id(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "m12_1_report:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _status(value: object) -> ReportStatus:
    text = str(value or "draft")
    if text in {"draft", "linter_failed", "ready", "stale", "superseded"}:
        return text  # type: ignore[return-value]
    return "draft"
