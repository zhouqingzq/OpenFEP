"""M12.2 reciprocal role runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from segmentum.cognitive_events import CognitiveEventBus, make_cognitive_event

from .evidence_cards import (
    ReciprocalEvidenceCard,
    evidence_cards_from_candidates,
    hints_from_candidates,
    prompt_safe_cards,
    reconcile_hints,
)
from .hyperparams import DEFAULT_HYPERPARAMS, M122Hyperparams
from .information_gain import rank_or_no_action
from .reciprocal_model import (
    EvidenceRef,
    InformationGainCandidate,
    ReciprocalClaim,
    ReciprocalClaimGroup,
    ReciprocalRoleModel,
    UncertaintyPoint,
    apply_model_patch,
    mark_group_contradicted,
)
from .safety_linter import apply_safety_linter, findings_to_dict
from .second_order_extractor import (
    SecondOrderExtractorValidationError,
    bound_extractor_snapshot,
    insufficient_output,
    validate_first_order_output,
    validate_second_order_output,
)
from .trigger_policy import TriggerDecision, TriggerPolicyInput, decide_trigger
from .turn_assessment import ReciprocalTurnAssessment, ReplyPolicyHint, assess_turn_light

Extractor = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class M122RuntimeConfig:
    m12_2_reciprocal_role_enabled: bool = False
    persona_kind: str = "legacy"


@dataclass(frozen=True)
class ReciprocalRunRecord:
    turn_id: str
    turn_index: int
    hour_bucket: int
    trigger_kind: str
    patch_non_empty: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "hour_bucket": self.hour_bucket,
            "trigger_kind": self.trigger_kind,
            "patch_non_empty": self.patch_non_empty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReciprocalRunRecord":
        return cls(
            turn_id=str(payload.get("turn_id", "")),
            turn_index=int(payload.get("turn_index", 0) or 0),
            hour_bucket=int(payload.get("hour_bucket", 0) or 0),
            trigger_kind=str(payload.get("trigger_kind", "")),
            patch_non_empty=bool(payload.get("patch_non_empty", False)),
        )


@dataclass(frozen=True)
class M122RuntimeState:
    models_by_user: dict[str, ReciprocalRoleModel] = field(default_factory=dict)
    run_records_by_user: dict[str, tuple[ReciprocalRunRecord, ...]] = field(default_factory=dict)

    @classmethod
    def clean(cls) -> "M122RuntimeState":
        return cls()

    def to_dict(self) -> dict[str, object]:
        return {
            "models_by_user": {user_id: model.to_dict() for user_id, model in sorted(self.models_by_user.items())},
            "run_records_by_user": {
                user_id: [record.to_dict() for record in records]
                for user_id, records in sorted(self.run_records_by_user.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "M122RuntimeState":
        models = payload.get("models_by_user", {})
        records = payload.get("run_records_by_user", {})
        return cls(
            models_by_user={
                str(user_id): ReciprocalRoleModel.from_dict(row)
                for user_id, row in (models.items() if isinstance(models, Mapping) else ())
                if isinstance(row, Mapping)
            },
            run_records_by_user={
                str(user_id): tuple(ReciprocalRunRecord.from_dict(item) for item in rows if isinstance(item, Mapping))
                for user_id, rows in (records.items() if isinstance(records, Mapping) else ())
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
            },
        )


@dataclass(frozen=True)
class RelationshipValueAssessment:
    user_id: str
    active_memories: tuple[dict[str, object], ...]
    persona_about_user: str
    user_about_persona: str
    persona_consistency_pressure_band: str
    user_comfort_pressure_band: str
    predicted_conflict_band: str
    preferred_policy: str = "adapt_to_relationship_value"

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "active_memories": [dict(item) for item in self.active_memories],
            "persona_about_user": self.persona_about_user,
            "user_about_persona": self.user_about_persona,
            "persona_consistency_pressure_band": self.persona_consistency_pressure_band,
            "user_comfort_pressure_band": self.user_comfort_pressure_band,
            "predicted_conflict_band": self.predicted_conflict_band,
            "preferred_policy": self.preferred_policy,
            "relationship_value_constraints": [
                {
                    "summary": str(item.get("summary", "")),
                    "prediction_constraint": str(item.get("prediction_constraint", "")),
                    "priority": str(item.get("priority", "medium")),
                    "confidence": item.get("confidence", 0.0),
                    "source": str(item.get("source", "")),
                }
                for item in self.active_memories
            ],
        }


@dataclass(frozen=True)
class M122TurnResult:
    enabled: bool
    trigger_decision: TriggerDecision
    state_before: dict[str, object]
    state_after: dict[str, object]
    light_turn_assessment: ReciprocalTurnAssessment | None
    evidence_cards: tuple[ReciprocalEvidenceCard, ...]
    prompt_safe_evidence_cards: tuple[dict[str, object], ...]
    reply_policy_hints: tuple[ReplyPolicyHint, ...]
    relationship_value_assessment: RelationshipValueAssessment | None = None
    safety_linter_findings: tuple[Mapping[str, object], ...] = ()
    recorded_extractor_outputs: dict[str, object] = field(default_factory=dict)
    published_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "trigger_decision": self.trigger_decision.to_dict(),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "light_turn_assessment": self.light_turn_assessment.to_dict() if self.light_turn_assessment else None,
            "evidence_cards": [card.to_dict() for card in self.evidence_cards],
            "prompt_safe_evidence_cards": list(self.prompt_safe_evidence_cards),
            "reply_policy_hints": [hint.to_dict() for hint in self.reply_policy_hints],
            "relationship_value_assessment": self.relationship_value_assessment.to_dict() if self.relationship_value_assessment else None,
            "safety_linter_findings": [dict(item) for item in self.safety_linter_findings],
            "recorded_extractor_outputs": _json_safe(self.recorded_extractor_outputs),
            "published_event_ids": list(self.published_event_ids),
        }


def assess_relationship_value_memories(
    *,
    user_id: str,
    memories: Sequence[Mapping[str, object]],
) -> RelationshipValueAssessment | None:
    active = tuple(_prompt_safe_relationship_memory(item) for item in memories)
    active = tuple(item for item in active if item)
    if not active:
        return None
    strongest = _strongest_relationship_band(active)
    top = active[0]
    summary = str(top.get("summary", ""))
    prediction = str(top.get("prediction_constraint", ""))
    return RelationshipValueAssessment(
        user_id=str(user_id or ""),
        active_memories=active[:6],
        persona_about_user=(
            summary
            or "This user is more comfortable when the persona adapts to relationship comfort instead of preserving default style mechanically."
        ),
        user_about_persona=(
            prediction
            or "This user may experience mechanical persona consistency as relationship friction when it conflicts with prior comfort feedback."
        ),
        persona_consistency_pressure_band="medium" if strongest in {"high", "medium"} else "low",
        user_comfort_pressure_band=strongest,
        predicted_conflict_band=strongest,
    )


def relationship_value_hints(
    assessment: RelationshipValueAssessment | None,
    *,
    turn_id: str,
) -> tuple[ReplyPolicyHint, ...]:
    if assessment is None:
        return ()
    hints: list[ReplyPolicyHint] = []
    for idx, memory in enumerate(assessment.active_memories[:3]):
        prediction = str(memory.get("prediction_constraint", "")).strip()
        summary = str(memory.get("summary", "")).strip()
        reason = prediction or summary
        if not reason:
            continue
        hints.append(
            ReplyPolicyHint(
                hint_id=f"hint:{turn_id}:relationship_value:{idx}",
                kind="apply_relationship_value_constraint",
                plain_reason=reason[:260],
                target_axis="user_about_persona",
                priority="high" if assessment.predicted_conflict_band == "high" else "medium",
                source="relationship_value_memory",
                topic_label="relationship_value_context",
            )
        )
    return tuple(hints)


def run_m12_2_tick(
    state: M122RuntimeState,
    *,
    user_id: str,
    turn_id: str,
    turn_index: int,
    hour_bucket: int,
    user_text: str = "",
    current_turn_quotes: Mapping[str, str] | None = None,
    transcript_quote_refs: Sequence[Mapping[str, object]] = (),
    m11_readonly_summary: Mapping[str, object] | None = None,
    m12_readonly_summary: Mapping[str, object] | None = None,
    m121_readonly_summary: Mapping[str, object] | None = None,
    relationship_value_memories: Sequence[Mapping[str, object]] = (),
    extractors: Mapping[str, Extractor] | None = None,
    trigger_input: TriggerPolicyInput | None = None,
    config: M122RuntimeConfig | None = None,
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
    event_bus: CognitiveEventBus | None = None,
    session_id: str = "live",
    persona_id: str = "default",
    cycle: int = 0,
    event_sequence_index: int = 0,
    event_timestamp: str | None = None,
) -> tuple[M122RuntimeState, M122TurnResult]:
    config = config or M122RuntimeConfig()
    before = state.to_dict()
    if not config.m12_2_reciprocal_role_enabled:
        decision = TriggerDecision(False, "no_run", "m12_2_disabled")
        return state, M122TurnResult(False, decision, before, before, None, (), (), ())

    model = state.models_by_user.get(user_id, ReciprocalRoleModel.empty(user_id=user_id, hyperparams=hyperparams))
    bootstrapped = M122RuntimeState(
        models_by_user={**state.models_by_user, user_id: model},
        run_records_by_user=dict(state.run_records_by_user),
    )
    quotes = current_turn_quotes or ({"q_current": user_text} if user_text else {})
    assessment = assess_turn_light(turn_id=turn_id, user_text=user_text, current_turn_quotes=quotes)
    relationship_assessment = assess_relationship_value_memories(
        user_id=user_id,
        memories=relationship_value_memories,
    )
    relationship_hints = relationship_value_hints(relationship_assessment, turn_id=turn_id)
    trigger = trigger_input or _default_trigger_input(
        state,
        model=model,
        user_id=user_id,
        turn_index=turn_index,
        hour_bucket=hour_bucket,
        assessment=assessment,
        hyperparams=hyperparams,
    )
    decision = decide_trigger(trigger, hyperparams=hyperparams)
    durable_candidates: tuple[InformationGainCandidate, ...] = ()
    safety_findings: tuple[Mapping[str, object], ...] = ()
    recorded: dict[str, object] = {}
    next_model = model
    event_ids: tuple[str, ...] = ()
    durable_ran = decision.should_run
    if decision.should_run:
        snapshot = bound_extractor_snapshot(_snapshot(
            user_id=user_id,
            turn_id=turn_id,
            current_turn_quotes=quotes,
            transcript_quote_refs=transcript_quote_refs,
            m11_readonly_summary=m11_readonly_summary,
            m12_readonly_summary=m12_readonly_summary,
            m121_readonly_summary=m121_readonly_summary,
            relationship_value_context=relationship_assessment.to_dict() if relationship_assessment else {},
            model=model,
        ), hyperparams=hyperparams)
        first_raw = _run_extractor(extractors, "first_order", snapshot, default=insufficient_output("first_order"))
        second_raw = _run_extractor(extractors, "second_order", snapshot, default=insufficient_output("second_order"))
        validation_errors: list[dict[str, object]] = []
        first = _validate_or_insufficient("first_order", first_raw, snapshot=snapshot, hyperparams=hyperparams, validation_errors=validation_errors)
        second = _validate_or_insufficient("second_order", second_raw, snapshot=snapshot, hyperparams=hyperparams, validation_errors=validation_errors)
        if validation_errors and not first.get("persona_about_user_claims") and not second.get("user_about_persona_claims"):
            first, second = _fallback_claims_from_assessment(
                first,
                second,
                assessment=assessment,
                turn_id=turn_id,
            )
        recorded = {"first_order": first, "second_order": second, "validation_errors": validation_errors}
        groups = tuple(_group(row, turn_id=turn_id) for row in [*first.get("claim_group_updates", []), *second.get("claim_group_updates", [])] if isinstance(row, Mapping))
        claims = (
            tuple(_claim(row, "persona_about_user", turn_id=turn_id) for row in first.get("persona_about_user_claims", []) if isinstance(row, Mapping))
            + tuple(_claim(row, "user_about_persona", turn_id=turn_id) for row in second.get("user_about_persona_claims", []) if isinstance(row, Mapping))
        )
        points = (
            tuple(_point(row) for row in first.get("unresolved_uncertainty_points", []) if isinstance(row, Mapping))
            + tuple(_point(row) for row in second.get("inferred_user_uncertainties_about_persona", []) if isinstance(row, Mapping))
        )
        raw_candidates = (
            tuple(_candidate(row, source=decision.kind) for row in first.get("high_gain_candidates", []) if isinstance(row, Mapping))
            + tuple(_candidate(row, source=decision.kind) for row in second.get("clarifying_reply_candidates", []) if isinstance(row, Mapping))
        )
        allowed_candidates, findings = apply_safety_linter(raw_candidates)
        safety_findings = tuple(findings_to_dict(findings))
        durable_candidates = rank_or_no_action(tuple(item for item in allowed_candidates if isinstance(item, InformationGainCandidate)))
        patch_base_model = model
        if decision.kind == "contradiction_or_misread":
            affected_group_id = _first_open_second_order_group_id(model)
            if affected_group_id:
                patch_base_model = mark_group_contradicted(
                    model,
                    group_id=affected_group_id,
                    turn_id=turn_id,
                    turn_index=turn_index,
                    hyperparams=hyperparams,
                )
        next_model = apply_model_patch(
            patch_base_model,
            turn_id=turn_id,
            turn_index=turn_index,
            group_updates=groups,
            claims=claims,
            uncertainty_points=points,
            candidates=durable_candidates,
            direct_probe_turn_ids=_direct_probe_turn_ids(model, assessment),
            observed_probe_turn_ids=_observed_probe_turn_ids(assessment),
            decay_cooldown=patch_base_model is model,
            hyperparams=hyperparams,
        )
        if next_model.to_json() != model.to_json() and event_bus is not None:
            event = make_cognitive_event(
                event_type="ReciprocalRoleUpdateEvent",
                turn_id=turn_id,
                cycle=cycle,
                session_id=session_id,
                persona_id=persona_id,
                source="m12_2_reciprocal_role",
                sequence_index=event_sequence_index,
                salience=0.35,
                priority=0.3,
                ttl=1,
                timestamp=event_timestamp,
                payload={
                    "user_id": user_id,
                    "patch_summary": {
                        "claims_added": len(claims),
                        "groups_touched": sorted({claim.group_id for claim in claims} | {group.group_id for group in groups}),
                        "candidates_ranked": [candidate.candidate_id for candidate in durable_candidates],
                    },
                    "source_extractors": sorted(recorded),
                },
            )
            event_bus.publish(event)
            event_ids = (event.event_id,)

    next_state = _state_after_run(
        bootstrapped,
        user_id=user_id,
        model=next_model,
        turn_id=turn_id,
        turn_index=turn_index,
        hour_bucket=hour_bucket,
        trigger_kind=decision.kind,
        patch_non_empty=next_model.to_json() != model.to_json(),
    )
    durable_hints = hints_from_candidates(durable_candidates, source="durable") if durable_ran else ()
    hints = reconcile_hints((*assessment.reply_policy_hints, *relationship_hints), durable_hints, durable_ran=durable_ran)
    cards = evidence_cards_from_candidates(durable_candidates, model=next_model)
    return next_state, M122TurnResult(
        enabled=True,
        trigger_decision=decision,
        state_before=before,
        state_after=next_state.to_dict(),
        light_turn_assessment=assessment,
        evidence_cards=cards,
        prompt_safe_evidence_cards=prompt_safe_cards(cards),
        reply_policy_hints=hints,
        relationship_value_assessment=relationship_assessment,
        safety_linter_findings=safety_findings,
        recorded_extractor_outputs=recorded,
        published_event_ids=event_ids,
    )


def _default_trigger_input(
    state: M122RuntimeState,
    *,
    model: ReciprocalRoleModel,
    user_id: str,
    turn_index: int,
    hour_bucket: int,
    assessment: ReciprocalTurnAssessment,
    hyperparams: M122Hyperparams,
) -> TriggerPolicyInput:
    records = state.run_records_by_user.get(user_id, ())
    turns_since_run = turn_index - max((record.turn_index for record in records), default=-999)
    contradiction_or_misread = (
        assessment.observed_user_probe in {"adversarial", "boundary_test"}
        and any(claim.status != "contradicted" for claim in model.user_about_persona_claims)
    )
    relationship_turning_point = (
        assessment.observed_user_probe == "boundary_test"
        or any(
            token in point.plain_question.casefold()
            for point in assessment.top_uncertainty_points
            for token in ("trust", "boundary", "limit", "consent")
        )
    )
    return TriggerPolicyInput(
        user_id=user_id,
        current_turn_index=turn_index,
        current_hour_bucket=hour_bucket,
        previous_run_turn_indices=tuple(record.turn_index for record in records),
        run_hour_buckets=tuple(record.hour_bucket for record in records),
        has_existing_model=bool(user_id in state.models_by_user or model.last_consolidated_turn_id or model.all_claims()),
        explicit_user_request=assessment.observed_user_probe == "explicit",
        high_second_order_uncertainty=assessment.user_about_persona_uncertainty_band == "high" and assessment.observed_user_probe in {"explicit", "adversarial", "boundary_test"},
        relationship_turning_point=relationship_turning_point,
        contradiction_or_misread=contradiction_or_misread,
        periodic_refresh_due=bool(model.last_consolidated_turn_id and turns_since_run >= max(6, hyperparams.min_turn_window_between_consolidations * 3)),
        evidence_sparse=assessment.insufficient_evidence,
        contradiction_cooldown=model.contradiction_cooldown,
    )


def _state_after_run(
    state: M122RuntimeState,
    *,
    user_id: str,
    model: ReciprocalRoleModel,
    turn_id: str,
    turn_index: int,
    hour_bucket: int,
    trigger_kind: str,
    patch_non_empty: bool,
) -> M122RuntimeState:
    records = list(state.run_records_by_user.get(user_id, ()))
    if trigger_kind != "no_run":
        records.append(ReciprocalRunRecord(turn_id, turn_index, hour_bucket, trigger_kind, patch_non_empty))
    return M122RuntimeState(
        models_by_user={**state.models_by_user, user_id: model},
        run_records_by_user={**state.run_records_by_user, user_id: tuple(records)},
    )


def _snapshot(**kwargs: object) -> dict[str, object]:
    transcript = kwargs.get("transcript_quote_refs", ())
    allowed = []
    if isinstance(transcript, Sequence) and not isinstance(transcript, (str, bytes)):
        for item in transcript:
            if isinstance(item, Mapping):
                allowed.append(f"{item.get('turn_id')}:{item.get('quote_id')}")
    turn_id = str(kwargs.get("turn_id", ""))
    quotes = kwargs.get("current_turn_quotes", {})
    if isinstance(quotes, Mapping):
        allowed.extend(f"{turn_id}:{quote_id}" for quote_id in quotes)
    return {**kwargs, "allowed_evidence_quote_refs": sorted(set(str(item) for item in allowed))}


def _prompt_safe_relationship_memory(item: Mapping[str, object]) -> dict[str, object]:
    summary = str(item.get("summary", "")).strip()
    prediction = str(item.get("prediction_constraint", "")).strip()
    if not summary or not prediction:
        return {}
    priority = str(item.get("priority", "medium")).strip() or "medium"
    if priority not in {"high", "medium", "low"}:
        priority = "medium"
    try:
        confidence = max(0.0, min(float(item.get("confidence", 0.0) or 0.0), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "id": str(item.get("id", "")).strip()[:80],
        "summary": summary[:240],
        "prediction_constraint": prediction[:360],
        "priority": priority,
        "confidence": round(confidence, 6),
        "source": str(item.get("source", "")).strip()[:80],
    }


def _strongest_relationship_band(memories: Sequence[Mapping[str, object]]) -> str:
    strongest = "low"
    for item in memories:
        priority = str(item.get("priority", "medium"))
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if priority == "high" and confidence >= 0.80:
            return "high"
        if priority in {"high", "medium"} and confidence >= 0.60:
            strongest = "medium"
    return strongest


def _run_extractor(
    extractors: Mapping[str, Extractor] | None,
    name: str,
    snapshot: Mapping[str, object],
    *,
    default: Mapping[str, object],
) -> Mapping[str, object]:
    fn = (extractors or {}).get(name)
    if fn is None:
        return dict(default)
    try:
        return fn(snapshot)
    except Exception:
        return dict(default)


def _validate_or_insufficient(
    axis: str,
    raw: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M122Hyperparams,
    validation_errors: list[dict[str, object]],
) -> dict[str, object]:
    try:
        if axis == "first_order":
            return validate_first_order_output(raw, snapshot=snapshot, hyperparams=hyperparams)
        return validate_second_order_output(raw, snapshot=snapshot, hyperparams=hyperparams)
    except SecondOrderExtractorValidationError as exc:
        validation_errors.append(
            {
                "rule": "extractor_validation_failed",
                "axis": axis,
                "message": str(exc)[:240],
            }
        )
        return insufficient_output(axis, reason="extractor_validation_failed")


def _fallback_claims_from_assessment(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *,
    assessment: ReciprocalTurnAssessment,
    turn_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if assessment.observed_user_probe not in {"explicit", "adversarial", "boundary_test"}:
        return dict(first), dict(second)
    refs = _assessment_ref_ids(assessment, turn_id=turn_id)
    if not refs:
        return dict(first), dict(second)
    first_out = dict(first)
    second_out = dict(second)
    first_out["insufficient_evidence"] = False
    second_out["insufficient_evidence"] = False
    first_out["claim_group_updates"] = [
        {
            "group_id": f"group:{turn_id}:persona_about_user",
            "target_axis": "persona_about_user",
            "topic_label": "relationship_read_request",
            "member_claim_ids": [f"claim:{turn_id}:persona_about_user"],
            "status": "open",
        }
    ]
    first_out["persona_about_user_claims"] = [
        {
            "claim_id": f"claim:{turn_id}:persona_about_user",
            "group_id": f"group:{turn_id}:persona_about_user",
            "topic_label": "relationship_read_request",
            "claim_text_internal": "Fallback from explicit reciprocal-role probe after extractor validation failed.",
            "claim_text_plain": "The user may want Hu Tao to read the relationship carefully and answer with evidence, not only with playfulness.",
            "evidence_refs": refs,
            "confidence_band": "low",
            "uncertainty_band": "medium",
            "status": "inferred_hypothesis",
        }
    ]
    second_out["claim_group_updates"] = [
        {
            "group_id": f"group:{turn_id}:user_about_persona",
            "target_axis": "user_about_persona",
            "topic_label": "persona_legibility",
            "member_claim_ids": [f"claim:{turn_id}:user_about_persona"],
            "status": "open",
        }
    ]
    second_out["user_about_persona_claims"] = [
        {
            "claim_id": f"claim:{turn_id}:user_about_persona",
            "group_id": f"group:{turn_id}:user_about_persona",
            "topic_label": "persona_legibility",
            "claim_text_internal": "Fallback from explicit reciprocal-role probe after extractor validation failed.",
            "claim_text_plain": "The user may be checking whether Hu Tao understands them and can stay trustworthy while guessing how they see her.",
            "evidence_refs": refs,
            "confidence_band": "low",
            "uncertainty_band": "high",
            "status": "inferred_hypothesis",
        }
    ]
    return first_out, second_out


def _assessment_ref_ids(assessment: ReciprocalTurnAssessment, *, turn_id: str) -> list[str]:
    refs = [
        ref.ref_id
        for hint in assessment.reply_policy_hints
        for ref in hint.evidence_refs
        if ref.ref_id
    ]
    if not refs:
        refs = [ref.ref_id for point in assessment.top_uncertainty_points for ref in point.evidence_refs if ref.ref_id]
    return list(dict.fromkeys(refs or [f"{turn_id}:q_current"]))


def _refs(rows: object) -> tuple[EvidenceRef, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(EvidenceRef.from_any(item) for item in rows)


def _group(row: Mapping[str, object], *, turn_id: str) -> ReciprocalClaimGroup:
    data = {**row, "created_turn_id": turn_id, "updated_turn_id": turn_id}
    return ReciprocalClaimGroup.from_dict(data)


def _claim(row: Mapping[str, object], axis: str, *, turn_id: str) -> ReciprocalClaim:
    data = {
        **row,
        "target_axis": axis,
        "evidence_refs": _refs(row.get("evidence_refs")),
        "created_turn_id": turn_id,
        "updated_turn_id": turn_id,
    }
    return ReciprocalClaim.from_dict(data)


def _point(row: Mapping[str, object]) -> UncertaintyPoint:
    data = {**row, "evidence_refs": _refs(row.get("evidence_refs"))}
    return UncertaintyPoint.from_dict(data)


def _candidate(row: Mapping[str, object], *, source: str) -> InformationGainCandidate:
    data = {**row, "evidence_refs": _refs(row.get("evidence_refs")), "source": source}
    return InformationGainCandidate.from_dict(data)


def _direct_probe_turn_ids(model: ReciprocalRoleModel, assessment: ReciprocalTurnAssessment) -> tuple[str, ...]:
    if assessment.observed_user_probe not in {"explicit", "adversarial", "boundary_test"}:
        return tuple(model.recent_probe_turn_ids)
    current = tuple(ref.turn_id for hint in assessment.reply_policy_hints for ref in hint.evidence_refs if ref.turn_id)
    return tuple(dict.fromkeys((*model.recent_probe_turn_ids, *current)))


def _observed_probe_turn_ids(assessment: ReciprocalTurnAssessment) -> tuple[str, ...]:
    if assessment.observed_user_probe not in {"explicit", "adversarial", "boundary_test"}:
        return ()
    return tuple(ref.turn_id for hint in assessment.reply_policy_hints for ref in hint.evidence_refs if ref.turn_id)


def _first_open_second_order_group_id(model: ReciprocalRoleModel) -> str:
    second_order_claim_group_ids = {
        claim.group_id
        for claim in model.user_about_persona_claims
        if claim.status != "contradicted"
    }
    for group in model.reciprocal_claim_groups:
        if group.group_id in second_order_claim_group_ids and group.status != "contradicted":
            return group.group_id
    return ""


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return _json_safe(converted)
    return str(value)
