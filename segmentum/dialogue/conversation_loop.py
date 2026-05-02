"""M5.3 scripted conversation driver: observe → decide → generate → outcome tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..cognitive_events import CognitiveEventBus, make_cognitive_event
from ..cognitive_paths import cognitive_paths_from_diagnostics, path_competition_summary
from ..cognition import CognitiveLoop
from ..memory_dynamics import (
    consolidate_successful_path_pattern,
    record_failed_path_outcome,
)
from ..meta_control import derive_meta_control_signal
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
        ) -> None:
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
        agent.active_meta_control_signal = meta_control_signal
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
        generation_diagnostics["fep_prompt_capsule"] = fep_capsule
        generation_diagnostics["selected_action"] = action
        generation_diagnostics["meta_control_guidance"] = meta_control_guidance_dict
        generation_diagnostics["meta_control_signal"] = meta_control_signal.to_dict()
        generation_diagnostics["affective_maintenance_summary"] = (
            affective_maintenance_summary
        )
        generation_diagnostics["prompt_capsule_guidance"] = {
            "meta_control_guidance": fep_capsule.get("meta_control_guidance") or {},
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
