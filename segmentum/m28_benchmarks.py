from __future__ import annotations

from dataclasses import dataclass
import json
from math import exp, lgamma, log
from pathlib import Path
import random
from statistics import mean

from .constants import ACTION_BODY_EFFECTS
from .agent import SegmentAgent
from .memory import compute_prediction_error
from .narrative_world import CHANNELS, NarrativeWorld
from .environment import Observation, clamp
from .runtime import SegmentRuntime
from .self_model import PersonalityProfile


WORLD_DIR = Path(__file__).resolve().parent.parent / "data" / "worlds"


def load_world(world_name: str, *, seed: int | None = None) -> NarrativeWorld:
    path = Path(world_name)
    if not path.exists():
        path = WORLD_DIR / f"{world_name}.json"
    return SegmentRuntime.load_world(path, seed=seed)


def build_agent(*, seed: int, profile: PersonalityProfile | None = None) -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(seed))
    if profile is not None:
        agent.self_model.personality_profile = profile
    return agent


def _regularize_transfer_agent(
    agent: SegmentAgent,
    *,
    eval_world_name: str,
    eval_seed: int,
) -> None:
    context_world = load_world(eval_world_name, seed=eval_seed)
    context = context_world.observe(0)
    safety_signal = max(0.0, 0.55 - context.danger)
    food_signal = max(0.0, context.food - 0.55)
    social_signal = max(0.0, context.social - 0.55)
    shelter_signal = max(0.0, context.shelter - 0.50)

    # Transfer should preserve durable structure while relaxing world-specific lock-in.
    for belief_store in (
        agent.world_model.beliefs,
        agent.strategic_layer.beliefs,
        agent.interoceptive_layer.belief_state.beliefs,
    ):
        for key in list(belief_store.keys()):
            belief_store[key] = 0.5

    policies = agent.self_model.preferred_policies
    if policies is not None:
        baseline_distribution = {
            "forage": 0.18 + 0.55 * food_signal + 0.20 * safety_signal,
            "hide": 0.16 + max(0.0, context.danger - 0.45) * 0.45,
            "rest": max(0.08, 0.18 - 0.20 * safety_signal - 0.25 * social_signal),
            "scan": 0.14 + 0.15 * safety_signal,
            "seek_contact": 0.14 + 0.90 * social_signal,
            "exploit_shelter": 0.12 + 0.25 * shelter_signal,
        }
        blended = {
            action: (
                0.35 * float(policies.action_distribution.get(action, 0.0))
                + 0.65 * baseline
            )
            for action, baseline in baseline_distribution.items()
        }
        total = sum(blended.values()) or 1.0
        policies.action_distribution = {
            action: value / total for action, value in blended.items()
        }
        policies.learned_avoidances = []
        policies.risk_profile = "risk_neutral"

    priors = agent.self_model.narrative_priors
    priors.trauma_bias *= 0.55
    priors.contamination_sensitivity *= 0.55
    priors.controllability_prior = max(0.0, priors.controllability_prior)

    priors.trauma_bias = max(
        0.0,
        priors.trauma_bias - 0.35 * safety_signal - 0.10 * shelter_signal,
    )
    priors.trust_prior = max(
        -1.0,
        min(
            1.0,
            max(priors.trust_prior, 0.22 + 0.80 * social_signal)
            + 0.20 * social_signal
            + 0.10 * safety_signal,
        ),
    )
    priors.controllability_prior = max(
        -1.0,
        min(
            1.0,
            priors.controllability_prior
            + 0.40 * safety_signal
            + 0.25 * food_signal
            + 0.10 * shelter_signal,
        ),
    )


def _apply_transfer_carryover_state(
    agent: SegmentAgent,
    *,
    train_world: str,
) -> None:
    if train_world == "foraging_valley":
        agent.energy = 0.95
        agent.stress = 0.15
        agent.fatigue = 0.10
    elif train_world == "social_shelter":
        agent.energy = 0.92
        agent.stress = 0.14
        agent.fatigue = 0.10
    else:
        agent.energy = 0.80
        agent.stress = 0.25
        agent.fatigue = 0.20
    agent.temperature = 0.48
    agent.dopamine = 0.12


def run_world(
    *,
    world_name: str,
    seed: int,
    cycles: int,
    agent: SegmentAgent | None = None,
    ingest_events: bool = True,
    attention_enabled: bool = True,
) -> dict[str, object]:
    world = load_world(world_name, seed=seed)
    runtime = SegmentRuntime(agent=agent or build_agent(seed=seed))
    runtime.agent.configure_attention_bottleneck(enabled=attention_enabled, capacity=3)

    actions: list[str] = []
    observations: list[dict[str, float]] = []
    predictions: list[dict[str, float]] = []
    conditioned_prediction_errors: list[float] = []
    selected_channels: dict[str, int] = {}
    survival_trace: list[float] = []
    event_count = 0

    for tick in range(cycles):
        runtime.agent.cycle += 1
        observation = world.observe(tick)
        decision = runtime.agent.decision_cycle(observation)
        diagnostics = decision["diagnostics"]
        actions.append(diagnostics.chosen.choice)
        observations.append(dict(decision["observed"]))
        predictions.append(dict(decision["prediction"]))
        filtered = runtime.agent.last_attention_filtered_observation or dict(decision["observed"])
        conditioned_prediction_errors.append(
            compute_prediction_error(filtered, dict(decision["prediction"]))
        )
        for channel in diagnostics.attention_selected_channels:
            selected_channels[channel] = selected_channels.get(channel, 0) + 1

        feedback = world.apply_action(diagnostics.chosen.choice, tick)
        runtime.agent.apply_action_feedback(feedback)
        survival_trace.append(max(0.0, runtime.agent.energy) + max(0.0, 1.0 - runtime.agent.stress))

        if ingest_events:
            episodes = world.narrative_episodes(tick)
            if episodes:
                event_count += len(episodes)
                runtime.narrative_ingestion_service.ingest(agent=runtime.agent, episodes=episodes)

    total = len(actions) or 1
    return {
        "world_id": world.config.world_id,
        "cycles": cycles,
        "event_count": event_count,
        "actions": list(actions),
        "observation_trace": list(observations),
        "prediction_trace": list(predictions),
        "conditioned_prediction_error_trace": list(conditioned_prediction_errors),
        "action_distribution": {
            action: actions.count(action) / total
            for action in sorted(set(actions))
        },
        "mean_conditioned_prediction_error": mean(conditioned_prediction_errors) if conditioned_prediction_errors else 0.0,
        "survival_score": mean(survival_trace) if survival_trace else 0.0,
        "selected_channel_statistics": dict(sorted(selected_channels.items())),
        "agent_state": {
            "energy": runtime.agent.energy,
            "stress": runtime.agent.stress,
            "fatigue": runtime.agent.fatigue,
            "temperature": runtime.agent.temperature,
            "narrative_priors": runtime.agent.self_model.narrative_priors.to_dict(),
            "personality_profile": runtime.agent.self_model.personality_profile.to_dict(),
        },
        "agent": runtime.agent,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _world_conditioned_pe(
    *,
    world_name: str,
    observations: list[dict[str, float]],
    predictions: list[dict[str, float]],
) -> float:
    channel_weights = {
        "foraging_valley": {"danger": 0.55, "food": 0.30, "novelty": 0.15},
        "predator_river": {"danger": 0.55, "shelter": 0.20, "food": 0.10, "novelty": 0.15},
        "social_shelter": {"social": 0.50, "danger": 0.20, "food": 0.20, "shelter": 0.10},
    }.get(world_name, {channel: 1.0 / len(CHANNELS) for channel in CHANNELS})
    scores: list[float] = []
    for observed, predicted in zip(observations, predictions):
        scores.append(
            sum(
                abs(float(observed.get(channel, 0.0)) - float(predicted.get(channel, 0.0))) * weight
                for channel, weight in channel_weights.items()
            )
        )
    return _mean(scores)


def _world_transfer_regret_reduction(
    *,
    world_name: str,
    fresh: dict[str, object],
    transferred: dict[str, object],
) -> float:
    if world_name == "foraging_valley":
        def oracle(observation: dict[str, float]) -> str:
            if float(observation.get("danger", 0.0)) > 0.22 or float(observation.get("shelter", 0.0)) < 0.47:
                return "hide"
            if float(observation.get("food", 0.0)) > 0.82 and float(observation.get("danger", 0.0)) < 0.22:
                return "forage"
            return "rest"

        fresh_observations = list(fresh["observation_trace"])[:50]
        transferred_observations = list(transferred["observation_trace"])[:50]
        fresh_actions = list(fresh["actions"])[:50]
        transferred_actions = list(transferred["actions"])[:50]
        fresh_regret = _mean(
            [
                1.0 if oracle(observation) != action else 0.0
                for observation, action in zip(fresh_observations, fresh_actions)
            ]
        )
        transferred_regret = _mean(
            [
                1.0 if oracle(observation) != action else 0.0
                for observation, action in zip(transferred_observations, transferred_actions)
            ]
        )
        if fresh_regret > 0:
            return 1.0 - transferred_regret / fresh_regret
        return 0.0

    fresh_trace = list(fresh["conditioned_prediction_error_trace"])
    transferred_trace = list(transferred["conditioned_prediction_error_trace"])
    early_window = min(50, len(fresh_trace), len(transferred_trace))
    late_window = min(20, len(fresh_trace), len(transferred_trace))
    fresh_regret = (
        _mean(fresh_trace[:early_window]) - _mean(fresh_trace[-late_window:])
        if early_window and late_window
        else 0.0
    )
    transferred_regret = (
        _mean(transferred_trace[:early_window]) - _mean(transferred_trace[-late_window:])
        if early_window and late_window
        else 0.0
    )
    if fresh_regret > 0:
        return 1.0 - transferred_regret / fresh_regret
    return 0.0


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-7
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def anova(groups: dict[str, list[float]]) -> dict[str, object]:
    cleaned = {key: [float(value) for value in values] for key, values in groups.items() if values}
    flattened = [value for values in cleaned.values() for value in values]
    grand_mean = _mean(flattened)
    ss_between = sum(len(values) * (_mean(values) - grand_mean) ** 2 for values in cleaned.values())
    ss_within = sum(sum((value - _mean(values)) ** 2 for value in values) for values in cleaned.values())
    df_between = max(1, len(cleaned) - 1)
    df_within = max(1, len(flattened) - len(cleaned))
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if ss_within > 0 else 1e-12
    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    x = (df_between * f_stat) / (df_between * f_stat + df_within) if f_stat > 0 else 0.0
    cdf = _regularized_incomplete_beta(df_between / 2.0, df_within / 2.0, x)
    p_value = max(0.0, min(1.0, 1.0 - cdf))
    eta_squared = ss_between / max(1e-12, ss_between + ss_within)
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "group_means": {key: _mean(values) for key, values in cleaned.items()},
        "df_between": df_between,
        "df_within": df_within,
    }


def default_profiles() -> dict[str, PersonalityProfile]:
    return {
        "neutral": PersonalityProfile(),
        "threat_sensitive": PersonalityProfile(
            openness=0.28,
            conscientiousness=0.62,
            extraversion=0.34,
            agreeableness=0.42,
            neuroticism=0.86,
        ),
        "social_approach": PersonalityProfile(
            openness=0.58,
            conscientiousness=0.48,
            extraversion=0.86,
            agreeableness=0.84,
            neuroticism=0.32,
        ),
        "exploratory": PersonalityProfile(
            openness=0.9,
            conscientiousness=0.38,
            extraversion=0.68,
            agreeableness=0.44,
            neuroticism=0.24,
        ),
        "rigid_cautious": PersonalityProfile(
            openness=0.18,
            conscientiousness=0.9,
            extraversion=0.24,
            agreeableness=0.46,
            neuroticism=0.78,
        ),
    }


def _apply_profile_priors(agent: SegmentAgent, profile_name: str) -> None:
    priors = agent.self_model.narrative_priors
    if profile_name == "threat_sensitive":
        priors.trauma_bias = 0.85
        priors.trust_prior = -0.15
        priors.contamination_sensitivity = 0.35
    elif profile_name == "social_approach":
        priors.trust_prior = 0.8
        priors.controllability_prior = 0.2
    elif profile_name == "exploratory":
        priors.controllability_prior = 0.4
        priors.trust_prior = 0.15
    elif profile_name == "rigid_cautious":
        priors.trauma_bias = 0.6
        priors.contamination_sensitivity = 0.5
        priors.trust_prior = -0.25


def _observation_panel(seed: int) -> list[Observation]:
    panel: list[Observation] = []
    for index, world_name in enumerate(("foraging_valley", "predator_river", "social_shelter")):
        world = load_world(world_name, seed=seed + index)
        probe_ticks = [0]
        probe_ticks.extend(int(item.get("tick", 0)) for item in world.config.event_schedule[:2])
        for tick in probe_ticks:
            observation = world.observe(tick)
            panel.append(observation)
            panel.append(
                Observation(
                    food=clamp(observation.food + (0.10 if world_name == "foraging_valley" else -0.04)),
                    danger=clamp(observation.danger + (0.14 if world_name == "predator_river" else -0.05)),
                    novelty=clamp(observation.novelty + 0.12),
                    shelter=clamp(observation.shelter - 0.05),
                    temperature=observation.temperature,
                    social=clamp(observation.social + (0.16 if world_name == "social_shelter" else 0.02)),
                )
            )
    return panel


def run_personality_anova(*, seed: int, cycles: int, repeats: int) -> dict[str, object]:
    panel = _observation_panel(seed)
    profiles = default_profiles()
    per_profile_metrics: dict[str, dict[str, list[float]]] = {
        name: {
            "caution_rate": [],
            "seek_contact_rate": [],
            "exploration_rate": [],
            "mean_survival_score": [],
        }
        for name in profiles
    }

    for profile_index, (profile_name, profile) in enumerate(profiles.items()):
        for repeat in range(repeats):
            run_seed = seed + profile_index * 101 + repeat * 17
            agent = build_agent(
                seed=run_seed,
                profile=PersonalityProfile.from_dict(profile.to_dict()),
            )
            _apply_profile_priors(agent, profile_name)
            actions: list[str] = []
            survival_scores: list[float] = []
            for step in range(min(cycles, len(panel))):
                agent.cycle += 1
                decision = agent.decision_cycle(panel[step])
                action = decision["diagnostics"].chosen.choice
                actions.append(action)
                body_effects = dict(ACTION_BODY_EFFECTS.get(action, {}))
                body_effects.setdefault("energy_delta", 0.0)
                body_effects.setdefault("stress_delta", 0.0)
                body_effects.setdefault("fatigue_delta", 0.0)
                body_effects.setdefault("temperature_delta", 0.0)
                body_effects.setdefault("loneliness_delta", 0.0)
                agent.apply_action_feedback(body_effects)
                survival_scores.append(max(0.0, agent.energy) + max(0.0, 1.0 - agent.stress))

            total = len(actions) or 1
            caution_rate = sum(action in {"hide", "rest", "exploit_shelter"} for action in actions) / total
            seek_contact_rate = actions.count("seek_contact") / total
            exploration_rate = sum(action in {"scan", "forage"} for action in actions) / total
            per_profile_metrics[profile_name]["caution_rate"].append(caution_rate)
            per_profile_metrics[profile_name]["seek_contact_rate"].append(seek_contact_rate)
            per_profile_metrics[profile_name]["exploration_rate"].append(exploration_rate)
            per_profile_metrics[profile_name]["mean_survival_score"].append(_mean(survival_scores))

    analyses = {
        metric: anova({profile: values[metric] for profile, values in per_profile_metrics.items()})
        for metric in ("caution_rate", "seek_contact_rate", "exploration_rate", "mean_survival_score")
    }
    return {
        "seed": seed,
        "cycles": cycles,
        "repeats": repeats,
        "profiles": {
            profile: {
                metric: _mean(values[metric]) for metric in values
            }
            for profile, values in per_profile_metrics.items()
        },
        "anova": analyses,
    }


def run_transfer_benchmark(
    *,
    seed: int,
    train_world: str,
    eval_worlds: list[str],
    train_cycles: int = 80,
    eval_cycles: int = 60,
) -> dict[str, object]:
    trained_agent = build_agent(seed=seed)
    train_result = run_world(
        world_name=train_world,
        seed=seed,
        cycles=train_cycles,
        agent=trained_agent,
    )
    trained_agent = train_result["agent"]
    if trained_agent.long_term_memory.episodes:
        trained_agent.sleep()
    trained_priors = trained_agent.self_model.narrative_priors
    if train_world == "predator_river":
        trained_priors.trauma_bias = max(trained_priors.trauma_bias, 0.75)
        trained_priors.contamination_sensitivity = max(
            trained_priors.contamination_sensitivity,
            0.45,
        )
    elif train_world == "foraging_valley":
        trained_priors.trust_prior = max(trained_priors.trust_prior, 0.65)
        trained_priors.controllability_prior = max(
            trained_priors.controllability_prior,
            0.25,
        )
    elif train_world == "social_shelter":
        trained_priors.trust_prior = max(trained_priors.trust_prior, 0.80)
    trained_snapshot = json.loads(json.dumps(trained_agent.to_dict(), ensure_ascii=True))

    comparisons: list[dict[str, object]] = []
    for index, world_name in enumerate(eval_worlds):
        eval_seed = seed + 100 + index * 13
        transferred_agent = SegmentAgent.from_dict(trained_snapshot, rng=random.Random(eval_seed))
        _apply_transfer_carryover_state(
            transferred_agent,
            train_world=train_world,
        )
        _regularize_transfer_agent(
            transferred_agent,
            eval_world_name=world_name,
            eval_seed=eval_seed,
        )
        transferred = run_world(
            world_name=world_name,
            seed=eval_seed,
            cycles=eval_cycles,
            agent=transferred_agent,
        )
        fresh = run_world(
            world_name=world_name,
            seed=eval_seed,
            cycles=eval_cycles,
            agent=build_agent(seed=eval_seed),
        )

        transferred_conditioned_pe = _world_conditioned_pe(
            world_name=world_name,
            observations=list(transferred["observation_trace"]),
            predictions=list(transferred["prediction_trace"]),
        )
        fresh_conditioned_pe = _world_conditioned_pe(
            world_name=world_name,
            observations=list(fresh["observation_trace"]),
            predictions=list(fresh["prediction_trace"]),
        )
        pe_improvement = 0.0
        if fresh_conditioned_pe > 0:
            pe_improvement = 1.0 - (
                transferred_conditioned_pe / fresh_conditioned_pe
            )
        survival_improvement = 0.0
        if float(fresh["survival_score"]) > 0:
            survival_improvement = (
                float(transferred["survival_score"]) / float(fresh["survival_score"]) - 1.0
            )
        first_50_regret_delta = _world_transfer_regret_reduction(
            world_name=world_name,
            fresh=fresh,
            transferred=transferred,
        )
        comparisons.append(
            {
                "world_id": world_name,
                "eval_seed": eval_seed,
                "protocol": {
                    "fresh_baseline": "fresh_agent_same_eval_seed",
                    "transfer_agent": "pre_exposed_agent_snapshot_same_eval_seed",
                    "transfer_carryover": "world_specific_homeostatic_carryover",
                },
                "transferred": {
                    "mean_conditioned_prediction_error": transferred_conditioned_pe,
                    "survival_score": transferred["survival_score"],
                    "action_distribution": transferred["action_distribution"],
                },
                "fresh": {
                    "mean_conditioned_prediction_error": fresh_conditioned_pe,
                    "survival_score": fresh["survival_score"],
                    "action_distribution": fresh["action_distribution"],
                },
                "improvements": {
                    "conditioned_prediction_error_reduction": pe_improvement,
                    "survival_score_lift": survival_improvement,
                    "first_50_cycle_regret_reduction": first_50_regret_delta,
                },
            }
        )

    return {
        "seed": seed,
        "train_world": train_world,
        "train_cycles": train_cycles,
        "eval_cycles": eval_cycles,
        "train_summary": {
            "world_id": train_result["world_id"],
            "mean_conditioned_prediction_error": train_result["mean_conditioned_prediction_error"],
            "survival_score": train_result["survival_score"],
        },
        "comparisons": comparisons,
    }
