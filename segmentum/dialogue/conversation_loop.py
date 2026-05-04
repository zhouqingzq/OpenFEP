"""M5.3 scripted conversation driver: observe → decide → generate → outcome tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..cognitive_events import CognitiveEventBus, make_cognitive_event
from ..cognitive_control import CognitiveControlAdapter, MetaControlPolicy
from ..cognitive_paths import (
    cognitive_path_candidates_from_diagnostics,
    cognitive_paths_from_diagnostics,
    path_competition_summary,
    select_cognitive_path_candidate,
)
from ..cognition import CognitiveLoop
from ..memory_anchored import (
    DialogueFactExtractor,
    MemoryCitationGuard,
    MemoryWriteIntent,
    WriteIntentTrace,
    build_memory_repair_instruction,
    build_response_evidence_contract,
)
from ..memory_dynamics import (
    MemoryInterferenceSignal,
    apply_interference_to_evidence_contract,
    consolidate_successful_path_pattern,
    derive_interference_feedback,
    record_failed_path_outcome,
)
from ..m9_bus_integration import (
    build_memory_interference_event_payload,
    build_memory_recall_event_payload,
)
from ..m9_state_patch_runtime import (
    process_subject_patches_for_turn,
    proposals_from_memory_interference,
)
from ..exploration import SelfThoughtProducer
from ..meta_control import MetaControlSignal, derive_meta_control_signal
from ..meta_control_guidance import (
    generate_meta_control_guidance,
    summarize_affective_maintenance,
)
from .generator import ResponseGenerator, RuleBasedGenerator
from .observer import DialogueObserver
from .fep_prompt import build_fep_prompt_capsule
from .outcome import classify_dialogue_outcome, inject_outcome_semantics
from .prediction_bridge import (
    register_dialogue_actions,
    register_dialogue_predictions,
    verify_dialogue_predictions,
)
from .turn_trace import TurnTrace
from .types import TranscriptUtterance

if TYPE_CHECKING:
    from ..agent import SegmentAgent
    from ..tracing import JsonlTraceWriter
    from .turn_trace import ConsciousMarkdownWriter


@dataclass
class ConversationTurn:
    turn_index: int
    speaker: str
    text: str
    action: str | None
    observation: dict[str, float] | None
    strategy: str | None
    diagnostics: Any | None = None
    cognitive_state: Any | None = None
    meta_control_guidance: Any | None = None
    generation_diagnostics: dict[str, object] | None = None
    outcome: str | None = None


def _bounded_diagnostics_summary(diagnostics: Any) -> dict[str, object]:
    if diagnostics is None:
        return {}
    chosen = getattr(diagnostics, "chosen", None)
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    return {
        "chosen_action": str(getattr(chosen, "choice", "")),
        "ranked_option_count": len(ranked),
        "prediction_error": float(getattr(diagnostics, "prediction_error", 0.0)),
        "memory_hit": bool(getattr(diagnostics, "memory_hit", False)),
        "retrieved_episode_ids": [
            str(item) for item in getattr(diagnostics, "retrieved_episode_ids", []) or []
        ][:5],
        "workspace_broadcast_channels": [
            str(item)
            for item in getattr(diagnostics, "workspace_broadcast_channels", []) or []
        ][:8],
        "workspace_suppressed_channels": [
            str(item)
            for item in getattr(diagnostics, "workspace_suppressed_channels", []) or []
        ][:8],
    }


def _candidate_path_summary(diagnostics: Any, *, limit: int = 3) -> list[dict[str, object]]:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    summary: list[dict[str, object]] = []
    for option in ranked[:limit]:
        summary.append(
            {
                "action": str(getattr(option, "choice", "")),
                "policy_score": float(getattr(option, "policy_score", 0.0)),
                "expected_free_energy": float(
                    getattr(option, "expected_free_energy", 0.0)
                ),
                "dominant_component": str(
                    getattr(option, "dominant_component", "")
                ),
            }
        )
    return summary


def _merge_meta_control_signals(
    base: MetaControlSignal,
    control: MetaControlSignal,
) -> MetaControlSignal:
    return MetaControlSignal(
        signal_id="m7-combined-meta-control",
        memory_retrieval_gain_multiplier=min(
            base.memory_retrieval_gain_multiplier,
            control.memory_retrieval_gain_multiplier,
        ),
        retrieval_k_delta=min(base.retrieval_k_delta, control.retrieval_k_delta),
        lambda_energy_multiplier=max(
            base.lambda_energy_multiplier,
            control.lambda_energy_multiplier,
        ),
        lambda_attention_multiplier=max(
            base.lambda_attention_multiplier,
            control.lambda_attention_multiplier,
        ),
        lambda_memory_multiplier=min(
            base.lambda_memory_multiplier,
            control.lambda_memory_multiplier,
        ),
        lambda_control_multiplier=max(
            base.lambda_control_multiplier,
            control.lambda_control_multiplier,
        ),
        beta_efe_multiplier=max(base.beta_efe_multiplier, control.beta_efe_multiplier),
        effective_temperature_delta=round(
            min(
                0.35,
                max(
                    base.effective_temperature_delta,
                    control.effective_temperature_delta,
                ),
            ),
            6,
        ),
        candidate_limit=(
            min(base.candidate_limit, control.candidate_limit)
            if base.candidate_limit is not None and control.candidate_limit is not None
            else base.candidate_limit
            if base.candidate_limit is not None
            else control.candidate_limit
        ),
        reasons=tuple(dict.fromkeys([*base.reasons, *control.reasons])),
    )


def _ensure_memory_store(agent: "SegmentAgent") -> object | None:
    """Get or create the agent's MemoryStore for anchored fact storage."""
    ltm = getattr(agent, "long_term_memory", None)
    if ltm is None:
        return None
    if hasattr(ltm, "ensure_memory_store"):
        ltm.ensure_memory_store()
    return getattr(ltm, "memory_store", None)


def _extract_and_store_dialogue_facts(
    agent: "SegmentAgent",
    text: str,
    turn_id: str,
    utterance_id: str,
    speaker: str = "user",
    *,
    event_bus: CognitiveEventBus | None = None,
    session_id: str = "unknown",
    persona_id: str = "unknown",
) -> None:
    """Extract anchored facts from a dialogue turn and store them.

    When *event_bus* is provided, each extracted fact flows through the
    M8.9 write-intent path:

        DialogueFactExtractionEvent -> MemoryWriteIntent -> commit -> MemoryWriteResultEvent

    When *event_bus* is None, falls back to direct ``store.add_anchored_item()``
    for backward compatibility with callers that do not have a bus.
    """
    store = _ensure_memory_store(agent)
    if store is None:
        return
    extractor = DialogueFactExtractor()
    items = extractor.extract(
        text=text,
        turn_id=turn_id,
        utterance_id=utterance_id,
        speaker=speaker,
        existing_items=list(getattr(store, "anchored_items", [])),
        current_cycle=getattr(agent, 'cycle', 0),
    )
    cycle_no = int(getattr(agent, 'cycle', 0))
    committed_ids: list[str] = []

    if event_bus is not None and hasattr(store, 'commit_write_intent'):
        # M8.9 write-intent path: publish -> intent -> commit -> result
        extraction_event = make_cognitive_event(
            event_type="DialogueFactExtractionEvent",
            turn_id=turn_id,
            cycle=cycle_no,
            session_id=session_id,
            persona_id=persona_id,
            source="dialogue_fact_extractor",
            sequence_index=cycle_no,
            payload={
                "speaker": speaker,
                "turn_id": turn_id,
                "utterance_id": utterance_id,
                "extracted_count": len(items),
                "propositions": [item.proposition for item in items],
            },
            salience=0.6,
            priority=0.6,
        )
        event_bus.publish(extraction_event)

        for item in items:
            trace = WriteIntentTrace(
                source_event_id=extraction_event.event_id,
                source_turn_id=turn_id,
                source_utterance_id=utterance_id,
                source_speaker=speaker,
                extraction_cycle=cycle_no,
            )
            intent = MemoryWriteIntent(
                intent_id=f"mwi_{item.memory_id}",
                item=item,
                trace=trace,
                operation="create",
                reason="dialogue_fact_extraction",
            )
            mid, op = store.commit_write_intent(intent)
            committed_ids.append(mid)

            result_payload: dict[str, object] = {
                "memory_id": mid,
                "operation": op,
                "intent_id": intent.intent_id,
                "proposition": item.proposition,
                "status": item.status,
                "visibility": item.visibility,
            }
            result_event = make_cognitive_event(
                event_type="MemoryWriteResultEvent",
                turn_id=turn_id,
                cycle=cycle_no,
                session_id=session_id,
                persona_id=persona_id,
                source="memory_write_result",
                sequence_index=cycle_no,
                payload=result_payload,
                salience=0.5,
                priority=0.5,
            )
            event_bus.publish(result_event)
    else:
        # Legacy path: direct store writes (backward compatible)
        for item in items:
            store.add_anchored_item(item)

    # M8.5: prune anchored items to prevent unbounded growth
    if hasattr(store, 'prune_anchored'):
        store.prune_anchored(current_cycle=getattr(agent, 'cycle', 0))


def _produce_self_thought_events_for_turn(
    *,
    agent: "SegmentAgent",
    diagnostics: Any,
    channels: dict[str, float],
    outcome_label: str,
    publish_event: Any,
    turn_id: str,
    persona_id: str,
    session_id: str,
) -> None:
    """M10.0: Produce SelfThoughtEvents bounded by cooldown, budget, and triggers."""
    if diagnostics is None:
        return
    previous_state = getattr(agent, "latest_cognitive_state", None)
    cooldown = (
        int(getattr(previous_state.self_agenda, "cooldown", 0))
        if previous_state is not None
        else 0
    )
    budget_spent = (
        float(getattr(previous_state.self_agenda, "budget_remaining", 1.0))
        if previous_state is not None
        else 1.0
    )
    budget_spent = 1.0 - budget_spent  # spent = 1.0 - remaining
    thought_count = (
        int(getattr(previous_state.self_agenda, "self_thought_count", 0))
        if previous_state is not None
        else 0
    )

    # M10.0: prior gap ids for dedupe — use previous active exploration target
    prior_gap_ids: tuple[str, ...] = ()
    if previous_state is not None:
        prev_active = getattr(previous_state.self_agenda, "active_exploration_target", "")
        prev_exploration = getattr(previous_state.self_agenda, "exploration_target", "")
        dedupe_ids = [s for s in (prev_active, prev_exploration) if s]
        prior_gap_ids = tuple(dedupe_ids)

    producer = SelfThoughtProducer()
    prediction_error = float(getattr(diagnostics, "prediction_error", 0.0))
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) >= 2:
        policy_margin = abs(
            float(getattr(ranked[0], "policy_score", 0.0))
            - float(getattr(ranked[1], "policy_score", 0.0))
        )
        efe_margin = abs(
            float(getattr(ranked[0], "expected_free_energy", 0.0))
            - float(getattr(ranked[1], "expected_free_energy", 0.0))
        )
    else:
        policy_margin = 1.0
        efe_margin = 1.0

    memory_conflicts_list = []
    if previous_state is not None:
        memory_conflicts_list = list(previous_state.memory.memory_conflicts)

    negative_outcomes: list[str] = []
    for ep in getattr(agent, "long_term_memory", None) and getattr(
        agent.long_term_memory, "episodes", []
    ) or []:
        if isinstance(ep, dict) and "fail" in str(ep.get("outcome", "")).lower():
            negative_outcomes.append(str(ep.get("outcome", "")))

    identity_tension = float(channels.get("conflict_tension", 0.0))
    commitment_tension = float(channels.get("commitment_tension", 0.0))

    # M10.0: Wire citation audit failures from memory store anchored items
    citation_audit_failures: list[str] = []
    store = _ensure_memory_store(agent)
    if store:
        anchored = list(getattr(store, "anchored_items", []))
        for item in anchored:
            audit = item.get("citation_audit") if isinstance(item, dict) else None
            if isinstance(audit, dict) and audit.get("failed"):
                citation_audit_failures.append(str(audit.get("reason", "citation_failure")))

    # M10.0: Infer unresolved questions from high-ambiguity observation channels
    unresolved_questions: list[str] = []
    if float(channels.get("hidden_intent", 0.0)) >= 0.72:
        unresolved_questions.append("user_intent_ambiguous")
    if float(channels.get("contextual_uncertainty", 0.0)) >= 0.5:
        unresolved_questions.append("context_insufficient")
    if channels.get("missing_context", False) or float(channels.get("missing_context", 0.0)) >= 0.5:
        unresolved_questions.append("missing_context")

    # M10.0: Track open uncertainty duration from previous unresolved gaps
    open_uncertainty_duration = 0
    if previous_state is not None:
        prev_unresolved = list(getattr(previous_state.self_agenda, "unresolved_gaps", []))
        # If we have persistent unresolved gaps from previous turn, count duration
        if prev_unresolved:
            open_uncertainty_duration = 1 + int(
                getattr(previous_state.self_agenda, "self_thought_count", 0)
            )

    triggers = producer.detect_triggers(
        prediction_error=prediction_error,
        policy_margin=policy_margin,
        efe_margin=efe_margin,
        memory_conflicts=memory_conflicts_list,
        citation_audit_failures=citation_audit_failures,
        previous_outcomes=negative_outcomes,
        identity_tension=identity_tension,
        commitment_tension=commitment_tension,
        unresolved_questions=unresolved_questions,
        open_uncertainty_duration=open_uncertainty_duration,
    )

    events = producer.produce(
        turn_id=turn_id,
        cycle=int(getattr(agent, "cycle", 0)),
        session_id=session_id,
        persona_id=persona_id,
        sequence_index=0,
        triggers=triggers,
        thought_count_this_turn=0,
        cooldown_remaining=cooldown,
        budget_spent=budget_spent,
        prior_gap_ids=prior_gap_ids,
    )

    for event in events:
        publish_event(
            event.event_type,
            event.source,
            dict(event.payload),
            salience=event.salience,
            priority=event.priority,
            ttl=event.ttl,
        )


def run_conversation(
    agent: "SegmentAgent",
    interlocutor_turns: list[str],
    *,
    generator: ResponseGenerator | None = None,
    observer: DialogueObserver,
    partner_uid: int = 0,
    session_id: str = "live",
    master_seed: int = 0,
    session_context_extra: dict[str, object] | None = None,
    initial_prior_observation: dict[str, float] | None = None,
    initial_last_action: str | None = None,
    initial_transcript: list[TranscriptUtterance] | None = None,
    cognitive_event_bus: CognitiveEventBus | None = None,
    persona_id: str = "default",
    trace_writer: JsonlTraceWriter | None = None,
    trace_debug: bool = False,
    conscious_writer: ConsciousMarkdownWriter | None = None,
    turn_index_offset: int = 0,
) -> list[ConversationTurn]:
    """Drive a scripted multi-turn dialogue (partner lines only); agent replies each turn."""
    register_dialogue_actions(agent.action_registry)
    gen = generator or RuleBasedGenerator()
    transcript: list[TranscriptUtterance] = list(initial_transcript or [])
    prior_obs: dict[str, float] | None = (
        dict(initial_prior_observation) if initial_prior_observation is not None else None
    )
    last_action: str | None = initial_last_action
    turns_out: list[ConversationTurn] = []
    session_context: dict[str, object] = {"session_id": session_id, "partner_uid": partner_uid}
    if session_context_extra:
        session_context.update(session_context_extra)

    for local_turn_index, partner_text in enumerate(interlocutor_turns):
        turn_index = int(turn_index_offset) + local_turn_index
        turn_id = f"turn_{turn_index:04d}"
        cycle_at_turn_start = int(agent.cycle)
        event_sequence = 0
        turn_events: list[object] = []
        event_bus_for_turn = cognitive_event_bus or CognitiveEventBus()

        def publish_event(
            event_type: str,
            source: str,
            payload: dict[str, object],
            *,
            salience: float = 0.5,
            priority: float = 0.5,
            ttl: int = 1,
        ) -> str:
            nonlocal event_sequence
            event = make_cognitive_event(
                event_type=event_type,
                turn_id=turn_id,
                cycle=cycle_at_turn_start,
                session_id=session_id,
                persona_id=persona_id,
                source=source,
                sequence_index=event_sequence,
                payload=payload,
                salience=salience,
                priority=priority,
                ttl=ttl,
            )
            turn_events.append(event)
            event_bus_for_turn.publish(event)
            event_sequence += 1
            return str(event.event_id)

        obs_obj = observer.observe(
            current_turn=partner_text,
            conversation_history=transcript,
            partner_uid=partner_uid,
            session_context=session_context,
            session_id=session_id,
            turn_index=turn_index,
            speaker_uid=partner_uid,
            timestamp=None,
        )
        channels = dict(obs_obj.channels)

        # ── M8: Extract dialogue facts from user turn ────────────────
        _extract_and_store_dialogue_facts(
            agent=agent,
            text=partner_text,
            turn_id=turn_id,
            utterance_id=f"{turn_id}_user",
            speaker="user",
            event_bus=event_bus_for_turn,
            session_id=session_id,
            persona_id=persona_id,
        )

        publish_event(
            "ObservationEvent",
            "DialogueObserver.observe",
            {
                "channel_count": len(channels),
                "channel_names": sorted(channels),
                "max_channel_value": max(channels.values()) if channels else 0.0,
                "turn_index": turn_index,
            },
            salience=max(channels.values()) if channels else 0.0,
        )
        dialogue_context: dict[str, object] = {
            "partner_uid": partner_uid,
            "session_id": session_id,
            "turn_index": turn_index,
            "current_turn": partner_text,
            "observation": channels,
        }
        outcome_label: str | None = None
        # Outcome tags the *prior* agent turn: integrate_outcome (end of last loop) should have
        # appended an episode whose action_taken is last_action; episodes[-1] is that row.
        if prior_obs is not None and last_action is not None:
            outcome = classify_dialogue_outcome(
                last_action,
                channels,
                dialogue_context,
                previous_observation=prior_obs,
            )
            outcome_label = outcome.value
            publish_event(
                "OutcomeEvent",
                "classify_dialogue_outcome",
                {
                    "outcome": outcome_label,
                    "last_action": last_action,
                    "prior_turn_index": max(0, turn_index - 1),
                },
                salience=0.7 if outcome_label != "neutral" else 0.35,
                priority=0.65,
            )
            if turns_out:
                turns_out[-1].outcome = outcome_label
            if agent.long_term_memory.episodes:
                latest = agent.long_term_memory.episodes[-1]
                if isinstance(latest, dict):
                    inject_outcome_semantics(latest, outcome)
            if turns_out and turns_out[-1].diagnostics is not None:
                patterns_before = list(agent.long_term_memory.reusable_cognitive_paths)
                patterns_after, updated_pattern = consolidate_successful_path_pattern(
                    patterns_before,
                    diagnostics=turns_out[-1].diagnostics,
                    outcome_label=outcome_label,
                    cycle=cycle_at_turn_start,
                )
                if updated_pattern is None:
                    patterns_after, updated_pattern = record_failed_path_outcome(
                        patterns_before,
                        diagnostics=turns_out[-1].diagnostics,
                        outcome_label=outcome_label,
                        cycle=cycle_at_turn_start,
                    )
                agent.long_term_memory.reusable_cognitive_paths = patterns_after
                agent.latest_memory_consolidation = {
                    "updated": updated_pattern is not None,
                    "pattern": dict(updated_pattern or {}),
                    "pattern_count": len(patterns_after),
                    "source": "dialogue_outcome",
                    "outcome_label": outcome_label,
                }

        ctx = {
            "partner_uid": partner_uid,
            "body": partner_text,
            "session_id": session_id,
            "event_type": "dialogue_turn",
            "master_seed": int(master_seed),
        }
        ctx = CognitiveControlAdapter.decision_context(
            ctx,
            getattr(agent, "active_cognitive_control_signal", None),
        )
        result = agent.decision_cycle_from_dict(channels, context=ctx)
        diagnostics = result.get("diagnostics")
        publish_event(
            "MemoryActivationEvent",
            "SegmentAgent.decision_cycle_from_dict",
            {
                "memory_hit": bool(getattr(diagnostics, "memory_hit", False)),
                "retrieved_episode_ids": [
                    str(item)
                    for item in getattr(diagnostics, "retrieved_episode_ids", []) or []
                ][:5],
                "retrieved_memory_count": len(
                    getattr(diagnostics, "retrieved_memories", []) or []
                ),
            },
            salience=0.65 if bool(getattr(diagnostics, "memory_hit", False)) else 0.25,
        )
        publish_event(
            "DecisionEvent",
            "SegmentAgent.decision_cycle_from_dict",
            _bounded_diagnostics_summary(diagnostics),
            salience=0.7,
            priority=0.7,
        )
        publish_event(
            "CandidatePathEvent",
            "SegmentAgent.decision_cycle_from_dict",
            {
                "ranked_options": _candidate_path_summary(diagnostics),
                "ranked_option_count": len(
                    getattr(diagnostics, "ranked_options", []) or []
                ),
            },
            salience=0.6,
        )
        action = (
            str(diagnostics.chosen.choice)
            if diagnostics is not None
            else "ask_question"
        )
        publish_event(
            "PathSelectionEvent",
            "SegmentAgent.decision_cycle_from_dict",
            {
                "selected_action": action,
                "cycle": cycle_at_turn_start,
            },
            salience=0.7,
            priority=0.75,
        )

        store_m9 = _ensure_memory_store(agent)
        anchored_m9 = list(getattr(store_m9, "anchored_items", [])) if store_m9 else []
        recall_bus_payload = build_memory_recall_event_payload(
            cue=partner_text,
            diagnostics=diagnostics,
            anchored_items=anchored_m9,
            turn_id=turn_id,
        )
        publish_event(
            "MemoryRecallEvent",
            "m9_bus_integration",
            recall_bus_payload,
            salience=0.78,
            priority=0.86,
            ttl=2,
        )
        interference_bus_payload = build_memory_interference_event_payload(
            diagnostics=diagnostics,
            last_retrieval_result=getattr(agent, "last_retrieval_result", {}) or {},
        )
        inner_interf_raw = interference_bus_payload.get("interference")
        if not isinstance(inner_interf_raw, dict):
            inner_interf_raw = {}
        interf_detected = (
            isinstance(inner_interf_raw, dict) and bool(inner_interf_raw.get("detected"))
        )
        memory_interference_event_id = publish_event(
            "MemoryInterferenceEvent",
            "m9_bus_integration",
            interference_bus_payload,
            salience=0.82 if interf_detected else 0.38,
            priority=0.88 if interf_detected else 0.55,
            ttl=2,
        )

        # M10.0: Produce SelfThoughtEvents from gap/trigger signals
        _produce_self_thought_events_for_turn(
            agent=agent,
            diagnostics=diagnostics,
            channels=channels,
            outcome_label=outcome_label or "",
            publish_event=publish_event,
            turn_id=turn_id,
            persona_id=persona_id,
            session_id=session_id,
        )

        cognitive_loop_result = CognitiveLoop(event_bus_for_turn).consume_and_update(
            getattr(agent, "latest_cognitive_state", None),
            turn_id=turn_id,
            persona_id=persona_id,
            diagnostics=diagnostics,
            observation=channels,
            previous_outcome=outcome_label or "",
            self_prior_summary=session_context.get("self_prior_summary"),
        )
        cognitive_state = cognitive_loop_result.state
        agent.latest_cognitive_state = cognitive_state
        if isinstance(inner_interf_raw, dict) and inner_interf_raw.get("detected"):
            patch_props = proposals_from_memory_interference(
                interference_payload=inner_interf_raw,
                source_event_id=memory_interference_event_id,
            )
            process_subject_patches_for_turn(
                agent,
                patch_props,
                publish_event,
                cycle=cycle_at_turn_start,
                interference_event_id=memory_interference_event_id,
            )
        path_summary = path_competition_summary(
            cognitive_paths_from_diagnostics(diagnostics)
            if diagnostics is not None
            else []
        )
        prompt_budget = session_context.get("prompt_budget")
        meta_control_guidance = generate_meta_control_guidance(
            cognitive_state,
            diagnostics=diagnostics,
            path_summary=path_summary,
            previous_outcome=outcome_label or "",
            prompt_budget=prompt_budget if isinstance(prompt_budget, dict) else None,
        )
        meta_control_guidance_dict = meta_control_guidance.to_dict()
        meta_control_signal = derive_meta_control_signal(
            state=cognitive_state,
            guidance=meta_control_guidance,
            diagnostics=diagnostics,
        )
        agent.latest_meta_control_signal = meta_control_signal
        cognitive_control_signal = MetaControlPolicy().derive(
            cognitive_state,
            diagnostics,
            path_summary,
            previous_outcome=outcome_label or "",
        )
        agent.latest_cognitive_control_signal = cognitive_control_signal
        agent.active_cognitive_control_signal = cognitive_control_signal
        agent.active_meta_control_signal = _merge_meta_control_signals(
            meta_control_signal,
            CognitiveControlAdapter.to_meta_control_signal(cognitive_control_signal),
        )
        cognitive_control_guidance = CognitiveControlAdapter.compact_prompt_guidance(
            cognitive_control_signal
        )
        controlled_candidates = (
            cognitive_path_candidates_from_diagnostics(
                diagnostics,
                meta_control=cognitive_state.to_dict().get("meta_control", {}),
                cognitive_control=cognitive_control_signal,
            )
            if diagnostics is not None
            else []
        )
        controlled_path_selection = select_cognitive_path_candidate(
            controlled_candidates
        )
        affective_maintenance_summary = summarize_affective_maintenance(
            meta_control_guidance
        )
        prompt_budget_dict = prompt_budget if isinstance(prompt_budget, dict) else None
        included_signals = [
            "observation_channels",
            "decision_diagnostics_summary",
            "cognitive_state",
            "cognitive_paths",
            "path_competition",
            "meta_control_guidance",
            "cognitive_control_guidance",
            "affective_guidance",
        ]
        if session_context.get("self_prior_summary") is not None:
            included_signals.append("self_prior_summary")
        if prompt_budget_dict is not None:
            included_signals.append("prompt_budget")
        for channel in getattr(diagnostics, "workspace_broadcast_channels", []) or []:
            signal = f"workspace:{channel}"
            if signal not in included_signals:
                included_signals.append(signal)
        omitted_signals_raw = session_context.get("omitted_signals")
        if omitted_signals_raw is None and prompt_budget_dict is not None:
            omitted_signals_raw = prompt_budget_dict.get("omitted_signals") or prompt_budget_dict.get("omitted")

        fep_capsule = build_fep_prompt_capsule(
            diagnostics,
            channels,
            previous_outcome=outcome_label or "",
            cognitive_state=cognitive_state,
            self_prior_summary=session_context.get("self_prior_summary"),
            path_summary=path_summary,
            meta_control_guidance=meta_control_guidance_dict,
            cognitive_control_guidance=cognitive_control_guidance,
            affective_state=cognitive_state.affect,
            affective_guidance=affective_maintenance_summary,
            prompt_budget=prompt_budget_dict,
            included_signals=included_signals,
            omitted_signals=omitted_signals_raw if isinstance(omitted_signals_raw, list) else None,
            persona_id=persona_id,
            session_id=session_id,
        ).to_dict()
        publish_event(
            "PromptAssemblyEvent",
            "build_fep_prompt_capsule",
            {
                "selected_action": action,
                "decision_uncertainty": str(
                    fep_capsule.get("decision_uncertainty", "")
                ),
                "prediction_error_label": str(
                    fep_capsule.get("prediction_error_label", "")
                ),
                "previous_outcome": str(fep_capsule.get("previous_outcome", "")),
                "top_alternative_count": len(
                    fep_capsule.get("top_alternatives", [])
                    if isinstance(fep_capsule.get("top_alternatives"), list)
                    else []
                ),
                "meta_control_guidance_flags": [
                    key
                    for key, value in meta_control_guidance_dict.items()
                    if isinstance(value, bool) and value
                ],
                "cognitive_control_guidance": cognitive_control_guidance,
                "affective_maintenance_summary": affective_maintenance_summary,
                "included_signals": included_signals,
                "omitted_signals": fep_capsule.get("omitted_signals") or [],
                "prompt_budget_summary": fep_capsule.get("prompt_budget_summary") or {},
                "redaction_status": {
                    "raw_events_included": False,
                    "full_diagnostics_included": False,
                    "full_prompt_included": False,
                    "full_conscious_markdown_included": False,
                },
            },
            salience=0.55,
        )
        efe_margin = float(fep_capsule.get("efe_margin", 1.0) or 1.0)
        dialogue_context["efe_margin"] = efe_margin
        dialogue_context["fep_prompt_capsule"] = fep_capsule
        dialogue_context["meta_control_guidance"] = meta_control_guidance_dict
        dialogue_context["cognitive_control_guidance"] = cognitive_control_guidance

        # ── M8.9 + M9.0: Evidence contract (unified MemoryEvidence + interference) ──
        store_for_evidence = _ensure_memory_store(agent)
        anchored_now = list(getattr(store_for_evidence, "anchored_items", [])) if store_for_evidence else []
        retrieved_texts: list[str] = []
        retrieval_entries: list[dict[str, object]] = []
        if diagnostics is not None:
            for mem in getattr(diagnostics, "activated_memories", []) or []:
                if hasattr(mem, "content"):
                    retrieved_texts.append(str(mem.content))
                elif isinstance(mem, dict):
                    retrieved_texts.append(str(mem.get("content", "")))
            for mem in getattr(diagnostics, "retrieved_memories", []) or []:
                if isinstance(mem, dict):
                    retrieval_entries.append(dict(mem))
        overdom = bool(interference_bus_payload.get("overdominance"))
        interf_signal = MemoryInterferenceSignal(
            detected=bool(inner_interf_raw.get("detected"))
            if isinstance(inner_interf_raw, dict)
            else False,
            kind=str(inner_interf_raw.get("kind", ""))
            if isinstance(inner_interf_raw, dict)
            else "",
            severity=float(inner_interf_raw.get("severity", 0.0) or 0.0)
            if isinstance(inner_interf_raw, dict)
            else 0.0,
            conflicting_episode_ids=tuple(
                str(x)
                for x in (inner_interf_raw.get("conflicting_episode_ids", []) or [])
            )
            if isinstance(inner_interf_raw, dict)
            else (),
            reasons=tuple(
                str(x) for x in (inner_interf_raw.get("reasons", []) or [])
            )
            if isinstance(inner_interf_raw, dict)
            else (),
        )
        interf_feedback = derive_interference_feedback(
            interf_signal if interf_signal.detected else None,
            overdominance_detected=overdom,
            memory_retrieval_gain=float(cognitive_state.memory.memory_helpfulness),
        )
        interference_ctrl = apply_interference_to_evidence_contract(
            interf_feedback,
            current_caution_level=0.5,
            current_assertiveness=0.5,
            memory_retrieval_gain=float(cognitive_state.memory.memory_helpfulness),
        )
        evidence_contract = build_response_evidence_contract(
            turn_id=turn_id,
            current_turn_text=partner_text,
            anchored_items=anchored_now,
            retrieved_memory_texts=retrieved_texts,
            retrieval_entries=retrieval_entries,
            current_cue=partner_text,
            style_hints=affective_maintenance_summary if isinstance(affective_maintenance_summary, list) else None,
            interference_controls=dict(interference_ctrl),
        )
        dialogue_context["evidence_contract"] = evidence_contract

        personality_state: dict[str, object] = {
            "slow_traits": agent.slow_variable_learner.state.traits.to_dict(),
        }
        policies = getattr(agent.self_model, "preferred_policies", None)
        if policies is not None and hasattr(policies, "to_dict"):
            personality_state["preferred_policies"] = policies.to_dict()
        priors = getattr(agent.self_model, "narrative_priors", None)
        if priors is not None and hasattr(priors, "to_dict"):
            personality_state["narrative_priors"] = priors.to_dict()
        surface_profile = getattr(agent, "dialogue_surface_profile", None)
        if isinstance(surface_profile, dict):
            personality_state["surface_profile"] = dict(surface_profile)
        policy_context = getattr(
            getattr(agent, "policy_evaluator", None),
            "_last_policy_context_by_action",
            {},
        )
        if isinstance(policy_context, dict):
            chosen_policy_context = policy_context.get(action, {})
            if isinstance(chosen_policy_context, dict):
                personality_state["policy_action_selection_context"] = dict(chosen_policy_context)
        reply = gen.generate(
            action,
            dialogue_context,
            personality_state,
            transcript,
            master_seed=master_seed,
            turn_index=turn_index,
        )
        generation_diagnostics = getattr(gen, "last_diagnostics", None)
        if isinstance(generation_diagnostics, dict):
            generation_diagnostics = dict(generation_diagnostics)
        else:
            generation_diagnostics = {}

        # ── M8.5: Memory citation audit after generation ───────────────
        store = _ensure_memory_store(agent)
        anchored_items = list(getattr(store, "anchored_items", [])) if store else []
        audit = MemoryCitationGuard.audit_structured(reply, anchored_items)
        generation_diagnostics["memory_citation_audit"] = audit.to_dict()

        # Conservative retry on blocking violation
        if audit.has_blocking_violation:
            repair_instruction = build_memory_repair_instruction(audit)
            # Inject repair instruction via dialogue_context
            retry_context = dict(dialogue_context)
            retry_context["memory_repair_instruction"] = repair_instruction
            retry_reply = gen.generate(
                action,
                retry_context,
                personality_state,
                transcript,
                master_seed=master_seed,
                turn_index=turn_index,
            )
            second_audit = MemoryCitationGuard.audit_structured(
                retry_reply, anchored_items,
            )
            generation_diagnostics["memory_citation_audit_retry"] = second_audit.to_dict()
            generation_diagnostics["memory_repair_triggered"] = True
            # Use retry result
            reply = retry_reply
            retry_diag = getattr(gen, "last_diagnostics", None)
            if isinstance(retry_diag, dict):
                generation_diagnostics.update({f"retry_{k}": v for k, v in retry_diag.items()})
        else:
            generation_diagnostics["memory_repair_triggered"] = False
            generation_diagnostics["memory_citation_audit_retry"] = None

        generation_diagnostics["fep_prompt_capsule"] = fep_capsule
        generation_diagnostics["selected_action"] = action
        generation_diagnostics["meta_control_guidance"] = meta_control_guidance_dict
        generation_diagnostics["meta_control_signal"] = meta_control_signal.to_dict()
        generation_diagnostics["cognitive_control_signal"] = (
            cognitive_control_signal.to_dict()
        )
        generation_diagnostics["cognitive_path_selection"] = (
            controlled_path_selection.to_dict()
        )
        selected_controlled_path = controlled_path_selection.selected_path
        generation_diagnostics["action_shift_candidate"] = bool(
            selected_controlled_path is not None
            and str(getattr(selected_controlled_path, "proposed_action", ""))
            != action
            and cognitive_control_signal.clarification_bias >= 0.18
            and (
                cognitive_state.gaps.blocking_gaps
                or cognitive_state.gaps.contextual_gaps
            )
            and cognitive_state.candidate_paths.low_margin
        )
        generation_diagnostics["affective_maintenance_summary"] = (
            affective_maintenance_summary
        )
        generation_diagnostics["prompt_capsule_guidance"] = {
            "meta_control_guidance": fep_capsule.get("meta_control_guidance") or {},
            "cognitive_control_guidance": fep_capsule.get("cognitive_control_guidance") or {},
            "affective_guidance": fep_capsule.get("affective_guidance") or {},
            "memory_use_guidance": fep_capsule.get("memory_use_guidance") or {},
            "omitted_signals": fep_capsule.get("omitted_signals") or [],
        }
        publish_event(
            "GenerationEvent",
            "ResponseGenerator.generate",
            {
                "selected_action": action,
                "reply_length": len(reply),
                "diagnostic_keys": sorted(generation_diagnostics),
                "template_id": str(generation_diagnostics.get("template_id", "")),
                "surface_source": str(
                    generation_diagnostics.get("surface_source", "")
                ),
            },
            salience=0.5,
        )
        if isinstance(policy_context, dict):
            chosen_policy_context = policy_context.get(action, {})
            if isinstance(chosen_policy_context, dict):
                generation_diagnostics.update(chosen_policy_context)
        transcript.append(TranscriptUtterance(role="interlocutor", text=partner_text))
        transcript.append(TranscriptUtterance(role="agent", text=reply))

        # ── M8: Extract facts from agent reply for self-consistency ──
        if reply.strip():
            _extract_and_store_dialogue_facts(
                agent=agent,
                text=reply,
                turn_id=turn_id,
                utterance_id=f"{turn_id}_agent",
                speaker="agent",
                event_bus=event_bus_for_turn,
                session_id=session_id,
                persona_id=persona_id,
            )

        strat = None
        if diagnostics is not None:
            desc = diagnostics.chosen.action_descriptor
            params = desc.get("params") if isinstance(desc, dict) else None
            if isinstance(params, dict):
                strat = str(params.get("strategy")) if params.get("strategy") is not None else None

        turns_out.append(
            ConversationTurn(
                turn_index=turn_index,
                speaker="agent",
                text=reply,
                action=action,
                observation=channels,
                strategy=strat,
                diagnostics=diagnostics,
                cognitive_state=cognitive_state,
                meta_control_guidance=meta_control_guidance,
                generation_diagnostics=generation_diagnostics,
                outcome=None,
            )
        )

        episodic_episode_count_before = len(agent.long_term_memory.episodes)
        integrated_outcome = False
        if diagnostics is not None:
            agent.integrate_outcome(
                choice=diagnostics.chosen.choice,
                observed=dict(result.get("observed", channels)),
                prediction=dict(result.get("prediction", {})),
                errors=dict(result.get("errors", {})),
                free_energy_before=float(result.get("free_energy_before", 0.0)),
                free_energy_after=float(result.get("free_energy_after", 0.0)),
            )
            integrated_outcome = True
        episodic_episode_count_after = len(agent.long_term_memory.episodes)

        verify_dialogue_predictions(
            verification_loop=agent.verification_loop,
            ledger=agent.prediction_ledger,
            new_observation=result.get("observed", channels),
            tick=agent.cycle,
        )
        register_dialogue_predictions(
            ledger=agent.prediction_ledger,
            current_observation=result.get("observed", channels),
            tick=agent.cycle,
        )

        if trace_writer is not None or conscious_writer is not None:
            memory_update_signal = {
                "integrated": integrated_outcome,
                "action": action,
                "outcome_label": outcome_label or "neutral",
                "episodic_episode_count_before": episodic_episode_count_before,
                "episodic_episode_count_after": episodic_episode_count_after,
            }
            turn_trace = TurnTrace.from_runtime(
                persona_id=persona_id,
                session_id=session_id,
                turn_id=turn_id,
                turn_index=turn_index,
                cycle=cycle_at_turn_start,
                observation_channels=channels,
                diagnostics=diagnostics,
                fep_prompt_capsule=fep_capsule,
                cognitive_state=cognitive_state,
                meta_control_guidance=meta_control_guidance_dict,
                generation_diagnostics=generation_diagnostics,
                outcome_label=outcome_label or "neutral",
                memory_update_signal=memory_update_signal,
                events=tuple(turn_events),
                debug=trace_debug,
            )
            if trace_writer is not None:
                turn_trace.write_jsonl(trace_writer, debug=trace_debug)
            if conscious_writer is not None:
                conscious_writer.write(turn_trace, debug=trace_debug)

        prior_obs = dict(channels)
        last_action = action
        agent.cycle += 1

    return turns_out
