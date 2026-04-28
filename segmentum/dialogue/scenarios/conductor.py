"""M5.5 ScenarioConductor: executes scenario battery against a SegmentAgent."""

from __future__ import annotations

import random as _random
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, stdev

from ..actions import DIALOGUE_ACTION_NAMES, DIALOGUE_ACTION_STRATEGY_MAP
from ..channel_registry import DIALOGUE_CHANNEL_NAMES
from ..conversation_loop import ConversationTurn, run_conversation
from ..generator import ResponseGenerator, RuleBasedGenerator
from ..maturity import PersonalitySnapshot, capture_personality_snapshot
from ..observer import DialogueObserver
from ..precision_bounds import ChannelPrecisionBounds
from ..seed_utils import derive_subseed
from .battery import SCENARIO_BATTERY, ScenarioSpec

# Slow trait keys used for deviation tracking.
_SLOW_TRAIT_KEYS: tuple[str, ...] = (
    "caution_bias",
    "threat_sensitivity",
    "trust_stance",
    "exploration_posture",
    "social_approach",
)


@dataclass
class ScenarioResult:
    scenario_id: str
    agent_uid: int
    seed: int
    split_strategy: str
    turns: list[ConversationTurn] = field(default_factory=list)
    action_distribution: dict[str, int] = field(default_factory=dict)
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    channel_means: dict[str, float] = field(default_factory=dict)
    channel_stds: dict[str, float] = field(default_factory=dict)
    precision_trajectory: list[dict[str, float]] = field(default_factory=list)
    pre_snapshot: PersonalitySnapshot | None = None
    post_snapshot: PersonalitySnapshot | None = None
    personality_deviation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "agent_uid": self.agent_uid,
            "seed": self.seed,
            "split_strategy": self.split_strategy,
            "turn_count": len(self.turns),
            "action_distribution": dict(self.action_distribution),
            "strategy_distribution": dict(self.strategy_distribution),
            "channel_means": dict(self.channel_means),
            "channel_stds": dict(self.channel_stds),
            "precision_trajectory": [dict(p) for p in self.precision_trajectory],
            "pre_snapshot": self.pre_snapshot.to_dict() if self.pre_snapshot else None,
            "post_snapshot": self.post_snapshot.to_dict() if self.post_snapshot else None,
            "personality_deviation": dict(self.personality_deviation),
        }


class ScenarioConductor:
    def __init__(
        self,
        generator: ResponseGenerator | None = None,
        observer: DialogueObserver | None = None,
    ) -> None:
        self.generator = generator or RuleBasedGenerator()
        self.observer = observer or DialogueObserver()
        self._precision_bounds = ChannelPrecisionBounds.from_dialogue_channels()

    # ── single scenario ──────────────────────────────────────────────────

    def run_scenario(
        self,
        agent,
        scenario: ScenarioSpec,
        *,
        seed: int = 42,
        split_strategy: str = "random",
    ) -> ScenarioResult:
        scenario_seed = derive_subseed(seed, "scenario", scenario.scenario_id)
        agent_uid = int(getattr(agent, "uid", 0))

        pre_snapshot = capture_personality_snapshot(agent, sleep_cycle=0)
        implant_traits = dict(pre_snapshot.slow_traits)

        session_context_extra: dict[str, object] = dict(scenario.initial_context)
        if scenario.channel_overrides:
            session_context_extra["channel_overrides"] = dict(scenario.channel_overrides)

        turns = run_conversation(
            agent,
            list(scenario.interlocutor_script),
            generator=self.generator,
            observer=self.observer,
            partner_uid=0,
            session_id=f"m55_{scenario.scenario_id}",
            master_seed=scenario_seed,
            session_context_extra=session_context_extra,
        )

        precision_trajectory: list[dict[str, float]] = []
        for turn in turns:
            snap: dict[str, float] = {}
            if turn.observation:
                for ch in DIALOGUE_CHANNEL_NAMES:
                    snap[ch] = float(turn.observation.get(ch, 0.0))
            precision_trajectory.append(snap)

        post_snapshot = capture_personality_snapshot(agent, sleep_cycle=0)

        # action distribution
        action_counter: Counter[str] = Counter()
        strategy_counter: Counter[str] = Counter()
        for turn in turns:
            if turn.action:
                action_counter[turn.action] += 1
            if turn.strategy:
                strategy_counter[turn.strategy] += 1

        # channel means / stds from per-turn observations
        channel_means: dict[str, float] = {}
        channel_stds: dict[str, float] = {}
        for ch in DIALOGUE_CHANNEL_NAMES:
            vals = [float(t.observation.get(ch, 0.0)) for t in turns if t.observation]
            if vals:
                channel_means[ch] = round(mean(vals), 6)
                channel_stds[ch] = round(stdev(vals) if len(vals) >= 2 else 0.0, 6)
            else:
                channel_means[ch] = 0.0
                channel_stds[ch] = 0.0

        # personality deviation per slow trait from implant
        personality_deviation: dict[str, float] = {}
        current_traits = post_snapshot.slow_traits
        for key in _SLOW_TRAIT_KEYS:
            personality_deviation[key] = round(
                abs(float(current_traits.get(key, 0.5)) - float(implant_traits.get(key, 0.5))), 6
            )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            agent_uid=agent_uid,
            seed=seed,
            split_strategy=split_strategy,
            turns=turns,
            action_distribution=dict(action_counter),
            strategy_distribution=dict(strategy_counter),
            channel_means=channel_means,
            channel_stds=channel_stds,
            precision_trajectory=precision_trajectory,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            personality_deviation=personality_deviation,
        )

    # ── full battery ─────────────────────────────────────────────────────

    def run_battery(
        self,
        agent,
        battery: tuple[ScenarioSpec, ...] = SCENARIO_BATTERY,
        *,
        seed: int = 42,
        split_strategy: str = "random",
        fresh_agent_per_scenario: bool = False,
    ) -> list[ScenarioResult]:
        scenarios = list(battery)
        if split_strategy == "random":
            rng = _random.Random(derive_subseed(seed, "split_order"))
            rng.shuffle(scenarios)

        agent_template: dict | None = None
        if fresh_agent_per_scenario:
            agent_template = agent.to_dict()

        results: list[ScenarioResult] = []
        for idx, scenario in enumerate(scenarios):
            active_agent = agent
            if fresh_agent_per_scenario and agent_template is not None:
                clone_seed = derive_subseed(seed, "scenario_clone", scenario.scenario_id)
                active_agent = type(agent).from_dict(
                    agent_template,
                    rng=_random.Random(int(clone_seed)),
                )
            result = self.run_scenario(
                active_agent,
                scenario,
                seed=seed,
                split_strategy=split_strategy,
            )
            results.append(result)
        return results
