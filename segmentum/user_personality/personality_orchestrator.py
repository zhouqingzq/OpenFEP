"""Deterministic M12.1 eight-step orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams, SECTION_KINDS
from .llm_step_extractors import StepExtractorValidationError, noop_step_extractor, validate_step_output
from .personality_profile import (
    CoreBelief,
    CoreBeliefSet,
    CoreLoop,
    CoreLoopStage,
    DefenseItem,
    EmotionAndDefenses,
    EvidenceExtraction,
    EvidenceItem,
    EvidenceQuoteRef,
    GrowthHints,
    InsufficientEvidence,
    PersonalityProfile,
    PersonalitySummary,
    PredictionSystemAccount,
    RelationshipPatterns,
    RelationshipTarget,
    SectionValue,
    bounded_confidence_band,
)
from .personality_report import PersonalityReport, assemble_personality_report
from .plain_language_linter import LinterFinding

StepExtractor = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class PersonalityRunTrace:
    trigger_kind: str
    snapshots: tuple[dict[str, object], ...]
    step_extractor_outputs: tuple[dict[str, object], ...]
    step_statuses: tuple[str, ...]
    profile_before: dict[str, object]
    profile_after: dict[str, object]
    report: dict[str, object]
    aborted_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "trigger_kind": self.trigger_kind,
            "snapshots": [dict(item) for item in self.snapshots],
            "step_extractor_outputs": [dict(item) for item in self.step_extractor_outputs],
            "step_statuses": list(self.step_statuses),
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
            "report": self.report,
            "aborted_reason": self.aborted_reason,
        }


@dataclass(frozen=True)
class OrchestratorResult:
    profile: PersonalityProfile
    report: PersonalityReport
    trace: PersonalityRunTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.to_dict(),
            "report": self.report.to_dict(),
            "trace": self.trace.to_dict(),
        }


def run_personality_orchestrator(
    profile: PersonalityProfile,
    *,
    turn_id: str,
    trigger_kind: str,
    base_snapshot: Mapping[str, object],
    extractors: Mapping[int, StepExtractor] | None = None,
    prior_report_id: str = "",
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> OrchestratorResult:
    before = profile.to_dict()
    current = profile
    snapshots: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    statuses: list[str] = []
    aborted_reason = ""
    extractor_map = dict(extractors or {})

    for step in range(1, 9):
        section_kind = f"step_{step}"
        snapshot = build_step_snapshot(
            base_snapshot,
            profile=current,
            step=step,
            turn_id=turn_id,
            hyperparams=hyperparams,
        )
        snapshots.append(snapshot)
        raw_output = dict(extractor_map.get(step, lambda _snapshot, s=step: noop_step_extractor(s))(snapshot))
        try:
            validated = validate_step_output(step, raw_output, snapshot=snapshot, hyperparams=hyperparams)
        except StepExtractorValidationError as exc:
            findings = tuple(item for item in getattr(exc, "findings", ()) if isinstance(item, LinterFinding))
            marker = InsufficientEvidence(
                reason=f"linter_or_schema_failed:{type(exc).__name__}",
                last_updated_turn_id=turn_id,
                hyperparams_version=hyperparams.hyperparams_version,
            )
            current = current.with_insufficient(section_kind, marker, keep_prior_section=True)
            report = assemble_personality_report(
                current,
                turn_id=turn_id,
                trigger_kind=trigger_kind,
                prior_report_id=prior_report_id,
                hyperparams=hyperparams,
                run_linter=False,
            )
            report = PersonalityReport(
                report_id=report.report_id,
                user_id=report.user_id,
                turn_id=report.turn_id,
                hyperparams_version=report.hyperparams_version,
                report_status="linter_failed",
                sections=report.sections,
                linter_findings=findings,
                trigger_kind=trigger_kind,
                prior_report_id=prior_report_id,
            )
            outputs.append(raw_output)
            statuses.append("linter_failed")
            aborted_reason = str(exc)
            current = current.with_report_state(turn_id=turn_id, report_status="linter_failed")
            trace = PersonalityRunTrace(
                trigger_kind=trigger_kind,
                snapshots=tuple(snapshots),
                step_extractor_outputs=tuple(outputs),
                step_statuses=tuple(statuses),
                profile_before=before,
                profile_after=current.to_dict(),
                report=report.to_dict(),
                aborted_reason=aborted_reason,
            )
            return OrchestratorResult(profile=current, report=report, trace=trace)

        outputs.append(validated)
        if str(validated.get("status", "")) == "insufficient_evidence":
            marker = _insufficient_from_payload(validated, turn_id=turn_id, hyperparams=hyperparams)
            current = current.with_insufficient(section_kind, marker, keep_prior_section=True)
            statuses.append("insufficient_evidence")
            continue
        section = _section_from_payload(
            step,
            validated,
            profile=current,
            turn_id=turn_id,
            hyperparams=hyperparams,
        )
        current = current.with_section(section_kind, section, turn_id=turn_id)
        statuses.append("updated")

    report = assemble_personality_report(
        current,
        turn_id=turn_id,
        trigger_kind=trigger_kind,
        prior_report_id=prior_report_id,
        hyperparams=hyperparams,
        run_linter=True,
    )
    current = current.with_report_state(turn_id=turn_id, report_status=report.report_status)
    trace = PersonalityRunTrace(
        trigger_kind=trigger_kind,
        snapshots=tuple(snapshots),
        step_extractor_outputs=tuple(outputs),
        step_statuses=tuple(statuses),
        profile_before=before,
        profile_after=current.to_dict(),
        report=report.to_dict(),
        aborted_reason=aborted_reason,
    )
    return OrchestratorResult(profile=current, report=report, trace=trace)


def build_base_snapshot(
    *,
    user_id: str,
    display_name: str,
    turn_id: str,
    current_turn_quotes: Mapping[str, str] | None = None,
    transcript_quote_refs: Sequence[Mapping[str, object]] = (),
    m11_readonly_summary: Mapping[str, object] | None = None,
    m12_readonly_summary: Mapping[str, object] | None = None,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    refs = []
    for row in transcript_quote_refs[: hyperparams.max_transcript_quote_refs]:
        if not isinstance(row, Mapping):
            continue
        refs.append({"turn_id": str(row.get("turn_id", "")), "quote_id": str(row.get("quote_id", ""))})
    allowed = {f"{row['turn_id']}:{row['quote_id']}" for row in refs if row.get("turn_id") and row.get("quote_id")}
    allowed.update(str(row.get("quote_id", "")) for row in refs if row.get("quote_id"))
    if current_turn_quotes:
        for quote_id in current_turn_quotes:
            allowed.add(str(quote_id))
            allowed.add(f"{turn_id}:{quote_id}")
    return {
        "user_id": user_id,
        "display_name": display_name,
        "turn_id": turn_id,
        "current_turn_quotes": dict(current_turn_quotes or {}),
        "transcript_quote_refs": refs,
        "allowed_evidence_quote_refs": sorted(ref for ref in allowed if ref),
        "m11_readonly_summary": _bounded_m11_summary(m11_readonly_summary or {}, hyperparams=hyperparams),
        "m12_readonly_summary": _bounded_m12_summary(m12_readonly_summary or {}, hyperparams=hyperparams),
    }


def build_step_snapshot(
    base_snapshot: Mapping[str, object],
    *,
    profile: PersonalityProfile,
    step: int,
    turn_id: str,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    snapshot = dict(base_snapshot)
    snapshot["step"] = step
    snapshot["turn_id"] = turn_id
    existing = profile.section_for(f"step_{step}")
    snapshot["existing_section"] = existing.to_dict() if existing is not None else None
    snapshot["profile_section_freshness"] = dict(profile.section_freshness)
    snapshot["hyperparams_version"] = hyperparams.hyperparams_version
    return snapshot


def _section_from_payload(
    step: int,
    payload: Mapping[str, object],
    *,
    profile: PersonalityProfile,
    turn_id: str,
    hyperparams: M121Hyperparams,
) -> SectionValue:
    section_kind = f"step_{step}"
    existing = profile.section_for(section_kind)
    existing_band = getattr(existing, "confidence_band", "low")
    refs = _refs(payload.get("evidence_quote_refs"), default_turn_id=turn_id)
    band = bounded_confidence_band(
        str(payload.get("confidence_band", "low")),
        existing_band=str(existing_band),
        evidence_refs=refs,
        hyperparams=hyperparams,
    )
    base = {
        "evidence_refs": refs,
        "confidence_band": band,
        "hyperparams_version": hyperparams.hyperparams_version,
        "last_updated_turn_id": turn_id,
    }
    if step == 1:
        return PersonalitySummary(summary=str(payload.get("summary", ""))[: hyperparams.max_summary_chars], **base)
    if step == 2:
        items = tuple(
            EvidenceItem(
                kind=str(row.get("kind", "")),
                content_summary=str(row.get("content_summary", ""))[: hyperparams.max_summary_chars],
                evidence_refs=_refs(row.get("evidence_quote_refs"), default_turn_id=turn_id)[: hyperparams.max_evidence_refs_per_claim],
                confidence_band=bounded_confidence_band(
                    str(row.get("confidence_band", "low")),
                    existing_band="low",
                    evidence_refs=_refs(row.get("evidence_quote_refs"), default_turn_id=turn_id),
                    hyperparams=hyperparams,
                ),
            )
            for row in _object_list(payload.get("evidence_items"))[: hyperparams.max_evidence_items]
        )
        item_refs = tuple(ref for item in items for ref in item.evidence_refs)
        section_band = bounded_confidence_band(
            max((item.confidence_band for item in items), key=lambda b: {"low": 0, "med": 1, "high": 2}.get(b, 0), default="low"),
            existing_band=str(existing_band),
            evidence_refs=item_refs,
            hyperparams=hyperparams,
        )
        return EvidenceExtraction(evidence_items=items, evidence_refs=item_refs[: hyperparams.max_evidence_refs_per_claim], confidence_band=section_band, hyperparams_version=hyperparams.hyperparams_version, last_updated_turn_id=turn_id)
    if step == 3:
        return PredictionSystemAccount(
            wants=str(payload.get("wants", ""))[: hyperparams.max_summary_chars],
            fears=str(payload.get("fears", ""))[: hyperparams.max_summary_chars],
            hypersensitive_to=_strings(payload.get("hypersensitive_to")),
            ignores=_strings(payload.get("ignores")),
            default_interpretation=str(payload.get("default_interpretation", ""))[: hyperparams.max_summary_chars],
            **base,
        )
    if step == 4:
        beliefs = {
            key: _core_belief(kind, payload.get(key), turn_id=turn_id, hyperparams=hyperparams)
            for key, kind in (
                ("about_self", "about_self"),
                ("about_others", "about_others"),
                ("about_world", "about_world"),
            )
        }
        return CoreBeliefSet(about_self=beliefs["about_self"], about_others=beliefs["about_others"], about_world=beliefs["about_world"], **base)
    if step == 5:
        defenses = tuple(
            DefenseItem(
                defense_kind=str(row.get("defense_kind", "")),
                protects_what=str(row.get("protects_what", ""))[: hyperparams.max_summary_chars],
                short_term_benefit=str(row.get("short_term_benefit", ""))[: hyperparams.max_summary_chars],
                long_term_cost=str(row.get("long_term_cost", ""))[: hyperparams.max_summary_chars],
                evidence_refs=_refs(row.get("evidence_quote_refs"), default_turn_id=turn_id)[: hyperparams.max_evidence_refs_per_claim],
                confidence_band=bounded_confidence_band(
                    str(row.get("confidence_band", "low")),
                    existing_band="low",
                    evidence_refs=_refs(row.get("evidence_quote_refs"), default_turn_id=turn_id),
                    hyperparams=hyperparams,
                ),
            )
            for row in _object_list(payload.get("defenses"))[: hyperparams.max_defenses]
        )
        return EmotionAndDefenses(
            dominant_emotional_baseline=str(payload.get("dominant_emotional_baseline", ""))[: hyperparams.max_summary_chars],
            threat_response=str(payload.get("threat_response", ""))[: hyperparams.max_summary_chars],
            defenses=defenses,
            **base,
        )
    if step == 6:
        return RelationshipPatterns(
            close_relationship_role=str(payload.get("close_relationship_role", ""))[: hyperparams.max_summary_chars],
            recurring_loop_summary=str(payload.get("recurring_loop_summary", ""))[: hyperparams.max_summary_chars],
            conflict_style=str(payload.get("conflict_style", "flee")),
            drawn_to=_relationship_target(payload.get("drawn_to"), turn_id=turn_id, hyperparams=hyperparams),
            clashes_with=_relationship_target(payload.get("clashes_with"), turn_id=turn_id, hyperparams=hyperparams),
            **base,
        )
    if step == 7:
        stages = tuple(
            CoreLoopStage(
                loop_stage=stage,  # type: ignore[arg-type]
                content_summary=str(payload.get(stage, ""))[: hyperparams.max_summary_chars],
                evidence_refs=refs[: hyperparams.max_evidence_refs_per_claim],
            )
            for stage in (
                "trigger_event",
                "interpretation",
                "emotion",
                "action",
                "outcome",
                "belief_reinforcement",
            )
        )
        return CoreLoop(stages=stages, **base)
    if step == 8:
        return GrowthHints(
            stable_parts=_strings(payload.get("stable_parts")),
            fragile_spots=_strings(payload.get("fragile_spots")),
            soft_spots=_strings(payload.get("soft_spots")),
            communication_styles_likely_accepted=_strings(payload.get("communication_styles_likely_accepted")),
            communication_styles_that_trigger_defenses=_strings(payload.get("communication_styles_that_trigger_defenses")),
            **base,
        )
    raise ValueError("unknown step")


def _insufficient_from_payload(
    payload: Mapping[str, object],
    *,
    turn_id: str,
    hyperparams: M121Hyperparams,
) -> InsufficientEvidence:
    return InsufficientEvidence(
        reason=str(payload.get("reason", "insufficient_evidence"))[: hyperparams.max_reason_chars],
        evidence_refs=_refs(payload.get("evidence_quote_refs"), default_turn_id=turn_id),
        hyperparams_version=hyperparams.hyperparams_version,
        last_updated_turn_id=turn_id,
    )


def _core_belief(kind: str, value: object, *, turn_id: str, hyperparams: M121Hyperparams) -> CoreBelief:
    row = dict(value) if isinstance(value, Mapping) else {}
    refs = _refs(row.get("evidence_quote_refs"), default_turn_id=turn_id)
    return CoreBelief(
        core_belief=kind,  # type: ignore[arg-type]
        content_summary=str(row.get("content_summary", ""))[:80],
        evidence_refs=refs[: hyperparams.max_evidence_refs_per_claim],
        confidence_band=bounded_confidence_band(
            str(row.get("confidence_band", "low")),
            existing_band="low",
            evidence_refs=refs,
            hyperparams=hyperparams,
        ),
    )


def _relationship_target(value: object, *, turn_id: str, hyperparams: M121Hyperparams) -> RelationshipTarget:
    row = dict(value) if isinstance(value, Mapping) else {}
    refs = _refs(row.get("evidence_quote_refs"), default_turn_id=turn_id)
    return RelationshipTarget(
        kind=str(row.get("kind", "")),
        why=str(row.get("why", ""))[: hyperparams.max_summary_chars],
        evidence_refs=refs[: hyperparams.max_evidence_refs_per_claim],
        confidence_band=bounded_confidence_band(
            str(row.get("confidence_band", "low")),
            existing_band="low",
            evidence_refs=refs,
            hyperparams=hyperparams,
        ),
    )


def _bounded_m11_summary(payload: Mapping[str, object], *, hyperparams: M121Hyperparams) -> dict[str, object]:
    active = payload.get("active_hypotheses", payload.get("hypotheses", ()))
    rows = []
    if isinstance(active, Sequence) and not isinstance(active, (str, bytes)):
        for row in active[: hyperparams.max_m11_hypothesis_refs]:
            if isinstance(row, Mapping):
                rows.append(
                    {
                        "hypothesis_id": str(row.get("hypothesis_id", "")),
                        "content_summary": str(row.get("content_summary", ""))[: hyperparams.max_summary_chars],
                        "confidence_band": str(row.get("confidence_band", "low")),
                    }
                )
    return {"active_hypotheses": rows, "reliability_summary": dict(payload.get("reliability_summary", {})) if isinstance(payload.get("reliability_summary"), Mapping) else {}}


def _bounded_m12_summary(payload: Mapping[str, object], *, hyperparams: M121Hyperparams) -> dict[str, object]:
    cues = payload.get("continuity_cues", payload.get("recent_continuity_cues", ()))
    rows = []
    if isinstance(cues, Sequence) and not isinstance(cues, (str, bytes)):
        for row in cues[: hyperparams.max_m12_continuity_cues]:
            if isinstance(row, Mapping):
                rows.append(
                    {
                        "cue_id": str(row.get("cue_id", row.get("id", ""))),
                        "cue_kind": str(row.get("cue_kind", "")),
                        "confidence_band": str(row.get("confidence_band", "low")),
                        "supports": str(row.get("supports", "")),
                    }
                )
    return {
        "identity_state": str(payload.get("identity_state", "")),
        "binding_confidence_band": str(payload.get("binding_confidence_band", "")),
        "recent_continuity_cues": rows,
        "strangeness_band": str(payload.get("strangeness_band", "")),
    }


def _refs(value: object, *, default_turn_id: str) -> tuple[EvidenceQuoteRef, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(EvidenceQuoteRef.from_any(item, default_turn_id=default_turn_id) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item)[: DEFAULT_HYPERPARAMS.max_summary_chars] for item in value if str(item).strip())


def _object_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))
