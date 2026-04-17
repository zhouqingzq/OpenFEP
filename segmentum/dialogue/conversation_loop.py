"""M5.3 scripted conversation driver: observe → decide → generate → outcome tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .generator import ResponseGenerator, RuleBasedGenerator
from .observer import DialogueObserver
from .outcome import classify_dialogue_outcome, inject_outcome_semantics
from .prediction_bridge import (
    register_dialogue_actions,
    register_dialogue_predictions,
    verify_dialogue_predictions,
)
from .types import TranscriptUtterance

if TYPE_CHECKING:
    from ..agent import SegmentAgent


@dataclass
class ConversationTurn:
    turn_index: int
    speaker: str
    text: str
    action: str | None
    observation: dict[str, float] | None
    strategy: str | None
    diagnostics: Any | None = None
    outcome: str | None = None


def run_conversation(
    agent: "SegmentAgent",
    interlocutor_turns: list[str],
    *,
    generator: ResponseGenerator | None = None,
    observer: DialogueObserver,
    partner_uid: int = 0,
    session_id: str = "live",
    master_seed: int = 0,
) -> list[ConversationTurn]:
    """Drive a scripted multi-turn dialogue (partner lines only); agent replies each turn."""
    register_dialogue_actions(agent.action_registry)
    gen = generator or RuleBasedGenerator()
    transcript: list[TranscriptUtterance] = []
    prior_obs: dict[str, float] | None = None
    last_action: str | None = None
    turns_out: list[ConversationTurn] = []
    session_context: dict[str, object] = {"session_id": session_id, "partner_uid": partner_uid}

    for turn_index, partner_text in enumerate(interlocutor_turns):
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
            if turns_out:
                turns_out[-1].outcome = outcome_label
            if agent.long_term_memory.episodes:
                latest = agent.long_term_memory.episodes[-1]
                if isinstance(latest, dict):
                    inject_outcome_semantics(latest, outcome)

        ctx = {
            "partner_uid": partner_uid,
            "body": partner_text,
            "session_id": session_id,
            "event_type": "dialogue_turn",
            "master_seed": int(master_seed),
        }
        result = agent.decision_cycle_from_dict(channels, context=ctx)
        diagnostics = result.get("diagnostics")
        action = (
            str(diagnostics.chosen.choice)
            if diagnostics is not None
            else "ask_question"
        )
        personality_state: dict[str, object] = {
            "slow_traits": agent.slow_variable_learner.state.traits.to_dict(),
        }
        surface_profile = getattr(agent, "dialogue_surface_profile", None)
        if isinstance(surface_profile, dict):
            personality_state["surface_profile"] = dict(surface_profile)
        reply = gen.generate(
            action,
            dialogue_context,
            personality_state,
            transcript,
            master_seed=master_seed,
            turn_index=turn_index,
        )
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
                outcome=None,
            )
        )

        if diagnostics is not None:
            agent.integrate_outcome(
                choice=diagnostics.chosen.choice,
                observed=dict(result.get("observed", channels)),
                prediction=dict(result.get("prediction", {})),
                errors=dict(result.get("errors", {})),
                free_energy_before=float(result.get("free_energy_before", 0.0)),
                free_energy_after=float(result.get("free_energy_after", 0.0)),
            )

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

        prior_obs = dict(channels)
        last_action = action
        agent.cycle += 1

    return turns_out
