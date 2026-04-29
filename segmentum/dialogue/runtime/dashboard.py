from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...agent import SegmentAgent


@dataclass
class DashboardSnapshot:
    tick: int
    big_five: dict[str, float]
    slow_traits: dict[str, float]
    precision_channels: dict[str, float]
    precision_debt: float
    defense_history: list[dict[str, object]]
    memory_stats: dict[str, int]
    body_state: dict[str, float]
    sleep_pressure: float
    cycle: int
    total_interactions: int


class DashboardCollector:
    def __init__(self) -> None:
        self._history: list[DashboardSnapshot] = []

    def snapshot(self, agent: "SegmentAgent") -> DashboardSnapshot:
        pp = agent.self_model.personality_profile
        big_five = {
            "openness": pp.openness,
            "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion,
            "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        slow_traits = agent.slow_variable_learner.state.traits.to_dict()
        precision_channels = dict(agent.precision_manipulator.channel_precisions)
        precision_debt = float(agent.precision_manipulator.precision_debt)

        defense_raw = getattr(
            agent.defense_strategy_selector, "strategy_history", []
        )
        defense_history: list[dict[str, object]] = []
        for entry in defense_raw[-32:]:
            if isinstance(entry, dict):
                defense_history.append(entry)
            else:
                defense_history.append({"entry": str(entry)})

        episodic = (
            agent.memory_store.episodic_count()
            if getattr(agent, "memory_store", None)
            else len(getattr(agent, "long_term_memory", {}).__dict__.get("episodes", []) or [])
        )
        semantic = len(getattr(agent, "semantic_memory", []))
        procedural = len(getattr(agent, "action_history", []))

        memory_stats = {
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
        }
        body_state = {
            "energy": float(agent.energy),
            "stress": float(agent.stress),
            "fatigue": float(agent.fatigue),
        }
        sleep_pressure = float(agent.stress)

        snap = DashboardSnapshot(
            tick=agent.cycle,
            big_five=big_five,
            slow_traits=slow_traits,
            precision_channels=precision_channels,
            precision_debt=precision_debt,
            defense_history=defense_history,
            memory_stats=memory_stats,
            body_state=body_state,
            sleep_pressure=sleep_pressure,
            cycle=agent.cycle,
            total_interactions=procedural,
        )
        self._history.append(snap)
        return snap

    def history(self, last_n: int = 50) -> list[DashboardSnapshot]:
        return self._history[-last_n:]

    def trait_trajectory(self) -> dict[str, list[float]]:
        if not self._history:
            return {}
        keys = list(self._history[0].slow_traits.keys())
        out: dict[str, list[float]] = {k: [] for k in keys}
        for snap in self._history:
            for k in keys:
                out[k].append(snap.slow_traits.get(k, 0.0))
        return out

    def precision_trajectory(self) -> dict[str, list[float]]:
        if not self._history:
            return {}
        keys = list(self._history[0].precision_channels.keys())
        out: dict[str, list[float]] = {k: [] for k in keys}
        for snap in self._history:
            for k in keys:
                out[k].append(snap.precision_channels.get(k, 0.0))
        return out
