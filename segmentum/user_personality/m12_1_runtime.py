"""Per-tick M12.1 personality runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from segmentum.cognitive_events import CognitiveEventBus, make_cognitive_event

from .evidence_cards import PersonalityEvidenceCard, evidence_cards_from_personality_profile, prompt_safe_cards
from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams
from .personality_orchestrator import OrchestratorResult, StepExtractor, build_base_snapshot, run_personality_orchestrator
from .personality_profile import PersonalityProfile
from .trigger_policy import TriggerDecision, TriggerPolicyInput, decide_trigger


@dataclass(frozen=True)
class M121RuntimeConfig:
    m12_1_personality_enabled: bool = False
    persona_kind: str = "legacy"

    @classmethod
    def for_persona(cls, *, persona_kind: str) -> "M121RuntimeConfig":
        return cls(m12_1_personality_enabled=persona_kind == "new", persona_kind=persona_kind)


@dataclass(frozen=True)
class PersonalityRunRecord:
    turn_id: str
    turn_index: int
    hour_bucket: int
    trigger_kind: str
    report_status: str
    step_1_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "hour_bucket": self.hour_bucket,
            "trigger_kind": self.trigger_kind,
            "report_status": self.report_status,
            "step_1_status": self.step_1_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalityRunRecord":
        return cls(
            turn_id=str(payload.get("turn_id", "")),
            turn_index=int(payload.get("turn_index", 0)),
            hour_bucket=int(payload.get("hour_bucket", 0)),
            trigger_kind=str(payload.get("trigger_kind", "")),
            report_status=str(payload.get("report_status", "")),
            step_1_status=str(payload.get("step_1_status", "")),
        )


@dataclass(frozen=True)
class M121RuntimeState:
    profiles_by_user: dict[str, PersonalityProfile] = field(default_factory=dict)
    latest_reports_by_user: dict[str, dict[str, object]] = field(default_factory=dict)
    run_records_by_user: dict[str, tuple[PersonalityRunRecord, ...]] = field(default_factory=dict)
    consecutive_step1_insufficient_by_user: dict[str, int] = field(default_factory=dict)

    @classmethod
    def clean(cls) -> "M121RuntimeState":
        return cls()

    def to_dict(self) -> dict[str, object]:
        return {
            "profiles_by_user": {user_id: profile.to_dict() for user_id, profile in sorted(self.profiles_by_user.items())},
            "latest_reports_by_user": {user_id: dict(report) for user_id, report in sorted(self.latest_reports_by_user.items())},
            "run_records_by_user": {
                user_id: [record.to_dict() for record in records]
                for user_id, records in sorted(self.run_records_by_user.items())
            },
            "consecutive_step1_insufficient_by_user": dict(sorted(self.consecutive_step1_insufficient_by_user.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "M121RuntimeState":
        profiles = payload.get("profiles_by_user", {})
        latest = payload.get("latest_reports_by_user", {})
        records = payload.get("run_records_by_user", {})
        sparse = payload.get("consecutive_step1_insufficient_by_user", {})
        return cls(
            profiles_by_user={
                str(user_id): PersonalityProfile.from_dict(row)
                for user_id, row in (profiles.items() if isinstance(profiles, Mapping) else ())
                if isinstance(row, Mapping)
            },
            latest_reports_by_user={
                str(user_id): dict(row)
                for user_id, row in (latest.items() if isinstance(latest, Mapping) else ())
                if isinstance(row, Mapping)
            },
            run_records_by_user={
                str(user_id): tuple(
                    PersonalityRunRecord.from_dict(item)
                    for item in rows
                    if isinstance(item, Mapping)
                )
                for user_id, rows in (records.items() if isinstance(records, Mapping) else ())
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
            },
            consecutive_step1_insufficient_by_user={
                str(user_id): int(value)
                for user_id, value in (sparse.items() if isinstance(sparse, Mapping) else ())
            },
        )


@dataclass(frozen=True)
class M121TurnResult:
    enabled: bool
    trigger_decision: TriggerDecision
    state_before: dict[str, object]
    state_after: dict[str, object]
    orchestrator_result: dict[str, object] | None
    evidence_cards: tuple[PersonalityEvidenceCard, ...]
    prompt_safe_evidence_cards: tuple[dict[str, str], ...]
    published_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "trigger_decision": self.trigger_decision.to_dict(),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "orchestrator_result": self.orchestrator_result,
            "evidence_cards": [card.to_dict() for card in self.evidence_cards],
            "prompt_safe_evidence_cards": list(self.prompt_safe_evidence_cards),
            "published_event_ids": list(self.published_event_ids),
        }


def run_m12_1_tick(
    state: M121RuntimeState,
    *,
    user_id: str,
    display_name: str,
    turn_id: str,
    turn_index: int,
    hour_bucket: int,
    current_turn_quotes: Mapping[str, str] | None = None,
    transcript_quote_refs: Sequence[Mapping[str, object]] = (),
    m11_readonly_summary: Mapping[str, object] | None = None,
    m12_readonly_summary: Mapping[str, object] | None = None,
    extractors: Mapping[int, StepExtractor] | None = None,
    trigger_input: TriggerPolicyInput | None = None,
    config: M121RuntimeConfig | None = None,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
    event_bus: CognitiveEventBus | None = None,
    session_id: str = "live",
    persona_id: str = "default",
    cycle: int = 0,
    event_sequence_index: int = 0,
) -> tuple[M121RuntimeState, M121TurnResult]:
    config = config or M121RuntimeConfig()
    before = state.to_dict()
    if not config.m12_1_personality_enabled:
        decision = TriggerDecision(False, "no_run", "m12_1_disabled")
        return state, M121TurnResult(False, decision, before, before, None, (), ())

    profile = state.profiles_by_user.get(
        user_id,
        PersonalityProfile(user_id=user_id, display_name_hint=display_name, hyperparams_version=hyperparams.hyperparams_version),
    )
    profiles = {**state.profiles_by_user, user_id: profile}
    bootstrapped = M121RuntimeState(
        profiles_by_user=profiles,
        latest_reports_by_user=dict(state.latest_reports_by_user),
        run_records_by_user=dict(state.run_records_by_user),
        consecutive_step1_insufficient_by_user=dict(state.consecutive_step1_insufficient_by_user),
    )
    trigger = trigger_input or _default_trigger_input(
        bootstrapped,
        user_id=user_id,
        turn_index=turn_index,
        hour_bucket=hour_bucket,
        m12_readonly_summary=m12_readonly_summary or {},
        hyperparams=hyperparams,
    )
    decision = decide_trigger(trigger, hyperparams=hyperparams)
    if not decision.should_run:
        cards = evidence_cards_from_personality_profile(profile, hyperparams=hyperparams)
        after = bootstrapped.to_dict()
        return bootstrapped, M121TurnResult(
            True,
            decision,
            before,
            after,
            None,
            cards,
            prompt_safe_cards(cards),
        )

    base_snapshot = build_base_snapshot(
        user_id=user_id,
        display_name=display_name,
        turn_id=turn_id,
        current_turn_quotes=current_turn_quotes,
        transcript_quote_refs=transcript_quote_refs,
        m11_readonly_summary=m11_readonly_summary,
        m12_readonly_summary=m12_readonly_summary,
        hyperparams=hyperparams,
    )
    prior_report = bootstrapped.latest_reports_by_user.get(user_id, {})
    prior_report_id = str(prior_report.get("report_id", ""))
    orchestration = run_personality_orchestrator(
        profile,
        turn_id=turn_id,
        trigger_kind=decision.kind,
        base_snapshot=base_snapshot,
        extractors=extractors,
        prior_report_id=prior_report_id,
        hyperparams=hyperparams,
    )
    next_state = _state_after_run(
        bootstrapped,
        user_id=user_id,
        profile=orchestration.profile,
        result=orchestration,
        turn_id=turn_id,
        turn_index=turn_index,
        hour_bucket=hour_bucket,
        trigger_kind=decision.kind,
    )
    cards = evidence_cards_from_personality_profile(
        orchestration.profile,
        report_status=orchestration.report.report_status,
        hyperparams=hyperparams,
    )
    event_ids: tuple[str, ...] = ()
    if orchestration.report.report_status == "ready" and event_bus is not None:
        event = make_cognitive_event(
            event_type="PersonalityProfileUpdateEvent",
            turn_id=turn_id,
            cycle=cycle,
            session_id=session_id,
            persona_id=persona_id,
            source="user_personality",
            sequence_index=event_sequence_index,
            salience=0.4,
            priority=0.3,
            ttl=1,
            payload={
                "user_id": user_id,
                "report_id": orchestration.report.report_id,
                "updated_sections": list(orchestration.profile.section_freshness),
            },
        )
        event_bus.publish(event)
        event_ids = (event.event_id,)
    return next_state, M121TurnResult(
        True,
        decision,
        before,
        next_state.to_dict(),
        orchestration.to_dict(),
        cards,
        prompt_safe_cards(cards),
        event_ids,
    )


def _default_trigger_input(
    state: M121RuntimeState,
    *,
    user_id: str,
    turn_index: int,
    hour_bucket: int,
    m12_readonly_summary: Mapping[str, object],
    hyperparams: M121Hyperparams,
) -> TriggerPolicyInput:
    records = state.run_records_by_user.get(user_id, ())
    last_ready = next((record for record in reversed(records) if record.report_status == "ready"), None)
    run_hours = tuple(record.hour_bucket for record in records)
    new_evidence = int(m12_readonly_summary.get("new_evidence_count", 0) or 0)
    sparse = state.consecutive_step1_insufficient_by_user.get(user_id, 0)
    return TriggerPolicyInput(
        user_id=user_id,
        current_turn_index=turn_index,
        current_hour_bucket=hour_bucket,
        last_successful_report_turn_index=last_ready.turn_index if last_ready else -1,
        run_hour_buckets=run_hours,
        new_evidence_count=new_evidence,
        identity_state_changed=bool(m12_readonly_summary.get("identity_state_changed", False)),
        high_strangeness_turns=int(m12_readonly_summary.get("high_strangeness_turns", 0) or 0),
        session_boundary=bool(m12_readonly_summary.get("session_boundary", False)),
        consecutive_step1_insufficient=sparse,
        new_evidence_since_step1_insufficient=new_evidence >= hyperparams.new_evidence_threshold,
    )


def _state_after_run(
    state: M121RuntimeState,
    *,
    user_id: str,
    profile: PersonalityProfile,
    result: OrchestratorResult,
    turn_id: str,
    turn_index: int,
    hour_bucket: int,
    trigger_kind: str,
) -> M121RuntimeState:
    records = list(state.run_records_by_user.get(user_id, ()))
    step_1_status = result.trace.step_statuses[0] if result.trace.step_statuses else "not_run"
    record = PersonalityRunRecord(
        turn_id=turn_id,
        turn_index=turn_index,
        hour_bucket=hour_bucket,
        trigger_kind=trigger_kind,
        report_status=result.report.report_status,
        step_1_status=step_1_status,
    )
    records.append(record)
    sparse = dict(state.consecutive_step1_insufficient_by_user)
    if step_1_status == "insufficient_evidence":
        sparse[user_id] = sparse.get(user_id, 0) + 1
    elif step_1_status == "updated":
        sparse[user_id] = 0
    return M121RuntimeState(
        profiles_by_user={**state.profiles_by_user, user_id: profile},
        latest_reports_by_user={**state.latest_reports_by_user, user_id: result.report.to_dict()},
        run_records_by_user={**state.run_records_by_user, user_id: tuple(records)},
        consecutive_step1_insufficient_by_user=sparse,
    )
