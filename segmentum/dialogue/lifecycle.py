from __future__ import annotations

from dataclasses import dataclass

from ..agent import SegmentAgent
from .maturity import (
    PersonalitySnapshot,
    capture_personality_snapshot,
    is_mature,
    maturity_report,
    personality_distance,
)
from .prediction_bridge import (
    register_dialogue_actions,
    register_dialogue_predictions,
    verify_dialogue_predictions,
)
from .world import DialogueWorld


@dataclass(slots=True)
class ImplantationConfig:
    sleep_every_n_sessions: int = 10
    sleep_on_pressure: bool = True
    sleep_pressure_threshold: float = 0.7
    maturity_threshold: float = 0.02
    maturity_window: int = 3
    max_ticks: int | None = None
    snapshot_every_n_sleeps: int = 1


@dataclass(slots=True)
class ImplantationResult:
    user_uid: int
    total_ticks: int
    total_sleep_cycles: int
    matured: bool
    maturity_tick: int | None
    snapshots: list[PersonalitySnapshot]
    final_agent_state: dict[str, object]
    maturity: dict[str, object]


def _sleep_pressure(agent: SegmentAgent) -> float:
    return float(getattr(getattr(agent, "homeostasis", None), "sleep_pressure", 0.0))


def implant_personality(
    agent: SegmentAgent,
    world: DialogueWorld,
    config: ImplantationConfig,
) -> ImplantationResult:
    register_dialogue_actions(agent.action_registry)
    snapshots: list[PersonalitySnapshot] = []
    sleep_count = 0
    sessions_since_sleep = 0
    total_ticks = 0
    maturity_tick: int | None = None
    while not world.exhausted:
        if config.max_ticks is not None and total_ticks >= int(config.max_ticks):
            break
        observation = world.observe()
        context = world.current_turn
        result = agent.decision_cycle_from_dict(observation, context=context)
        diagnostics = result.get("diagnostics")
        if diagnostics is not None:
            agent.integrate_outcome(
                choice=diagnostics.chosen.choice,
                observed=dict(result.get("observed", observation)),
                prediction=dict(result.get("prediction", {})),
                errors=dict(result.get("errors", {})),
                free_energy_before=float(result.get("free_energy_before", 0.0)),
                free_energy_after=float(result.get("free_energy_after", 0.0)),
            )
        verify_dialogue_predictions(
            verification_loop=agent.verification_loop,
            ledger=agent.prediction_ledger,
            new_observation=result.get("observed", observation),
            tick=agent.cycle,
        )
        register_dialogue_predictions(
            ledger=agent.prediction_ledger,
            current_observation=result.get("observed", observation),
            tick=agent.cycle,
        )
        total_ticks += 1
        agent.cycle += 1
        world.advance()
        if world.session_boundary:
            sessions_since_sleep += 1
        should_sleep = sessions_since_sleep >= int(config.sleep_every_n_sessions) or (
            bool(config.sleep_on_pressure)
            and _sleep_pressure(agent) > float(config.sleep_pressure_threshold)
        )
        if not should_sleep:
            continue
        agent.sleep()
        sleep_count += 1
        sessions_since_sleep = 0
        if sleep_count % max(1, int(config.snapshot_every_n_sleeps)) != 0:
            continue
        snapshot = capture_personality_snapshot(agent, sleep_count)
        if snapshots:
            snapshot.maturity_distance = personality_distance(snapshots[-1], snapshot)
        snapshots.append(snapshot)
        if is_mature(snapshots, config):
            maturity_tick = agent.cycle
            break
    matured = maturity_tick is not None
    return ImplantationResult(
        user_uid=world.user_uid,
        total_ticks=total_ticks,
        total_sleep_cycles=sleep_count,
        matured=matured,
        maturity_tick=maturity_tick,
        snapshots=snapshots,
        final_agent_state=agent.to_dict(),
        maturity=maturity_report(
            snapshots,
            threshold=float(config.maturity_threshold),
            window=int(config.maturity_window),
        ),
    )
