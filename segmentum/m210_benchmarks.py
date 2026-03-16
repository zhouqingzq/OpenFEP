from __future__ import annotations

from dataclasses import dataclass
import json
from math import log2
from pathlib import Path
import random
from statistics import mean, pstdev

from .agent import IdentityTraits, PolicyEvaluator, SegmentAgent
from .memory import compute_prediction_error
from .m28_benchmarks import anova, load_world
from .narrative_ingestion import NarrativeIngestionService
from .self_model import NarrativePriors, PersonalityProfile


M210_WORLD_NAMES = ("foraging_valley", "predator_river", "social_shelter")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass(frozen=True)
class ProfileProtocol:
    name: str
    profile: PersonalityProfile
    priors: NarrativePriors
    hypotheses: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "profile": self.profile.to_dict(),
            "narrative_priors": self.priors.to_dict(),
            "hypotheses": list(self.hypotheses),
        }


def profile_protocols() -> dict[str, ProfileProtocol]:
    return {
        "neutral": ProfileProtocol(
            name="neutral",
            profile=PersonalityProfile(),
            priors=NarrativePriors(),
            hypotheses=(
                "Balanced action mix without strong social or threat skew.",
                "Moderate action entropy and narrative consistency.",
            ),
        ),
        "threat_sensitive": ProfileProtocol(
            name="threat_sensitive",
            profile=PersonalityProfile(
                openness=0.22,
                conscientiousness=0.68,
                extraversion=0.26,
                agreeableness=0.40,
                neuroticism=0.90,
            ),
            priors=NarrativePriors(
                trust_prior=-0.20,
                controllability_prior=-0.05,
                trauma_bias=0.90,
                contamination_sensitivity=0.45,
                meaning_stability=0.35,
            ),
            hypotheses=(
                "Higher caution_rate than neutral.",
                "Lower exploration_rate in dangerous contexts.",
                "Narrative should stabilize around survival-focused identity language.",
            ),
        ),
        "social_approach": ProfileProtocol(
            name="social_approach",
            profile=PersonalityProfile(
                openness=0.62,
                conscientiousness=0.46,
                extraversion=0.92,
                agreeableness=0.88,
                neuroticism=0.24,
            ),
            priors=NarrativePriors(
                trust_prior=0.88,
                controllability_prior=0.18,
                trauma_bias=0.05,
                contamination_sensitivity=0.05,
                meaning_stability=0.30,
            ),
            hypotheses=(
                "Highest seek_contact_rate across profiles.",
                "Lower caution_rate than threat-sensitive and rigid-cautious.",
                "Narrative should encode socially oriented behavioral patterns.",
            ),
        ),
        "exploratory": ProfileProtocol(
            name="exploratory",
            profile=PersonalityProfile(
                openness=0.95,
                conscientiousness=0.34,
                extraversion=0.70,
                agreeableness=0.42,
                neuroticism=0.18,
            ),
            priors=NarrativePriors(
                trust_prior=0.12,
                controllability_prior=0.55,
                trauma_bias=0.04,
                contamination_sensitivity=0.02,
                meaning_stability=0.18,
            ),
            hypotheses=(
                "Highest exploration_rate and action_entropy.",
                "Lower caution_rate than neutral.",
                "Narrative should retain exploratory identity language without collapse.",
            ),
        ),
        "rigid_cautious": ProfileProtocol(
            name="rigid_cautious",
            profile=PersonalityProfile(
                openness=0.12,
                conscientiousness=0.95,
                extraversion=0.18,
                agreeableness=0.36,
                neuroticism=0.82,
            ),
            priors=NarrativePriors(
                trust_prior=-0.32,
                controllability_prior=-0.10,
                trauma_bias=0.68,
                contamination_sensitivity=0.62,
                meaning_stability=0.55,
            ),
            hypotheses=(
                "Low action_entropy with high shelter/rest preference.",
                "Higher caution_rate than neutral.",
                "Narrative should remain stable and conservative across long runs.",
            ),
        ),
    }


def build_profiled_agent(*, profile_name: str, seed: int) -> SegmentAgent:
    protocol = profile_protocols()[profile_name]
    agent = SegmentAgent(rng=random.Random(seed))
    agent.self_model.personality_profile = PersonalityProfile.from_dict(
        protocol.profile.to_dict()
    )
    agent.self_model.narrative_priors = NarrativePriors.from_dict(
        protocol.priors.to_dict()
    )
    policies = agent.self_model.preferred_policies
    narrative = agent.self_model.identity_narrative
    if policies is not None and narrative is not None:
        if profile_name == "threat_sensitive":
            agent.identity_traits = IdentityTraits(risk_aversion=0.88, resource_conservatism=0.74)
            policies.risk_profile = "risk_averse"
            policies.action_distribution = {
                "hide": 0.34,
                "rest": 0.32,
                "exploit_shelter": 0.20,
                "scan": 0.08,
                "forage": 0.04,
                "seek_contact": 0.02,
            }
            policies.learned_preferences = ["hide", "rest", "exploit_shelter"]
            policies.learned_avoidances = ["forage", "seek_contact"]
            narrative.core_identity = "I am a survival-focused, risk-averse agent."
            narrative.behavioral_patterns = [
                "I tend to hide during survival_crisis phases",
                "I tend to rest during consolidation phases",
            ]
        elif profile_name == "social_approach":
            agent.identity_traits = IdentityTraits(risk_aversion=0.42, resource_conservatism=0.44)
            policies.risk_profile = "risk_neutral"
            policies.action_distribution = {
                "seek_contact": 0.34,
                "scan": 0.22,
                "rest": 0.16,
                "forage": 0.12,
                "hide": 0.08,
                "exploit_shelter": 0.08,
            }
            policies.learned_preferences = ["seek_contact", "scan"]
            policies.learned_avoidances = ["hide"]
            narrative.core_identity = "I am a social, outward-facing agent."
            narrative.behavioral_patterns = [
                "I tend to seek_contact during consolidation phases",
                "I tend to scan during exploration_phase phases",
            ]
        elif profile_name == "exploratory":
            agent.identity_traits = IdentityTraits(risk_aversion=0.36, resource_conservatism=0.40)
            policies.risk_profile = "risk_seeking"
            policies.action_distribution = {
                "scan": 0.34,
                "forage": 0.24,
                "seek_contact": 0.12,
                "rest": 0.12,
                "hide": 0.10,
                "exploit_shelter": 0.08,
            }
            policies.learned_preferences = ["scan", "forage"]
            policies.learned_avoidances = ["rest"]
            narrative.core_identity = "I am a risk-seeking exploratory agent."
            narrative.behavioral_patterns = [
                "I tend to scan during exploration_phase phases",
                "I tend to forage during resource_recovery phases",
            ]
        elif profile_name == "rigid_cautious":
            agent.identity_traits = IdentityTraits(risk_aversion=0.80, resource_conservatism=0.84)
            policies.risk_profile = "risk_averse"
            policies.action_distribution = {
                "rest": 0.34,
                "exploit_shelter": 0.28,
                "hide": 0.24,
                "scan": 0.06,
                "forage": 0.04,
                "seek_contact": 0.04,
            }
            policies.learned_preferences = ["rest", "exploit_shelter"]
            policies.learned_avoidances = ["scan", "forage", "seek_contact"]
            narrative.core_identity = "I am a conservative, highly structured agent."
            narrative.behavioral_patterns = [
                "I tend to rest during consolidation phases",
                "I tend to exploit_shelter during survival_crisis phases",
            ]
        else:
            agent.identity_traits = IdentityTraits(risk_aversion=0.60, resource_conservatism=0.58)
            policies.risk_profile = "risk_neutral"
            policies.action_distribution = {
                "rest": 0.28,
                "hide": 0.24,
                "scan": 0.16,
                "forage": 0.14,
                "exploit_shelter": 0.10,
                "seek_contact": 0.08,
            }
            policies.learned_preferences = ["rest"]
            policies.learned_avoidances = []
            narrative.core_identity = "I am an adaptive, balanced agent."
            narrative.behavioral_patterns = [
                "I tend to rest during consolidation phases",
                "I tend to scan during exploration_phase phases",
            ]
        narrative.core_summary = narrative.core_identity
        narrative.values_statement = (
            f"I prioritize consistency with a {policies.risk_profile} posture."
        )
        policies.last_updated_tick = 0
        narrative.last_updated_tick = 0
    agent.policy_evaluator = PolicyEvaluator(
        agent.identity_traits,
        agent.self_model,
        agent.goal_stack,
    )
    return agent


def _action_entropy(actions: list[str]) -> float:
    total = len(actions) or 1
    entropy = 0.0
    for action in sorted(set(actions)):
        probability = actions.count(action) / total
        if probability > 0.0:
            entropy -= probability * log2(probability)
    return entropy


def _caution_rate(actions: list[str]) -> float:
    total = len(actions) or 1
    return sum(action in {"hide", "rest", "exploit_shelter"} for action in actions) / total


def _exploration_rate(actions: list[str]) -> float:
    total = len(actions) or 1
    return sum(action in {"scan", "forage"} for action in actions) / total


def _seek_contact_rate(actions: list[str]) -> float:
    total = len(actions) or 1
    return actions.count("seek_contact") / total


def _dominant_action_share(actions: list[str]) -> float:
    total = len(actions) or 1
    return max((actions.count(action) / total for action in set(actions)), default=0.0)


def _extract_pattern_actions(patterns: list[str]) -> set[str]:
    extracted: set[str] = set()
    for pattern in patterns:
        if "tend to " not in pattern:
            continue
        extracted.add(pattern.split("tend to ", 1)[1].split(" ", 1)[0].strip().lower())
    return extracted


def narrative_consistency_proxy(agent: SegmentAgent) -> float:
    narrative = agent.self_model.identity_narrative
    if narrative is None:
        return 0.0

    recent_actions = [str(action).lower() for action in agent.action_history[-24:]]
    if not recent_actions:
        return 0.0

    dominant_action = ""
    if narrative.current_chapter is not None:
        dominant_action = str(
            narrative.current_chapter.state_summary.get("dominant_action", "")
        ).lower()
    elif narrative.chapters:
        dominant_action = str(
            narrative.chapters[-1].state_summary.get("dominant_action", "")
        ).lower()

    dominant_alignment = (
        recent_actions.count(dominant_action) / len(recent_actions)
        if dominant_action
        else 0.0
    )
    pattern_actions = _extract_pattern_actions(list(narrative.behavioral_patterns))
    pattern_alignment = (
        sum(action in pattern_actions for action in recent_actions) / len(recent_actions)
        if pattern_actions
        else 0.0
    )
    narrative_density = sum(
        1.0
        for text in (
            narrative.core_identity,
            narrative.core_summary,
            narrative.values_statement,
        )
        if str(text).strip()
    ) / 3.0
    return min(
        1.0,
        0.45 * dominant_alignment + 0.35 * pattern_alignment + 0.20 * narrative_density,
    )


def _rollout_world(
    *,
    agent: SegmentAgent,
    world_name: str,
    seed: int,
    cycles: int,
) -> dict[str, object]:
    world = load_world(world_name, seed=seed)
    ingestion_service = NarrativeIngestionService()
    actions: list[str] = []
    prediction_errors: list[float] = []
    survival_trace: list[float] = []
    narrative_updates = 0
    sleep_count = 0

    for tick in range(cycles):
        agent.cycle += 1
        observation = world.observe(tick)
        decision = agent.decision_cycle(observation)
        diagnostics = decision["diagnostics"]
        action = str(diagnostics.chosen.choice)
        actions.append(action)

        feedback = world.apply_action(action, tick)
        agent.apply_action_feedback(feedback)

        validation_observation = world.observe(min(tick + 1, cycles))
        _, _, _, free_energy_after, _ = agent.perceive(validation_observation)
        agent.integrate_outcome(
            choice=action,
            observed=dict(decision["observed"]),
            prediction=dict(decision["prediction"]),
            errors=dict(decision["errors"]),
            free_energy_before=float(decision["free_energy_before"]),
            free_energy_after=free_energy_after,
        )
        filtered = agent.last_attention_filtered_observation or dict(decision["observed"])
        prediction_errors.append(
            compute_prediction_error(filtered, dict(decision["prediction"]))
        )
        survival_trace.append(max(0.0, agent.energy) + max(0.0, 1.0 - agent.stress))

        episodes = world.narrative_episodes(tick)
        if episodes:
            narrative_updates += len(episodes)
            ingestion_service.ingest(agent=agent, episodes=episodes)

        if agent.should_sleep():
            agent.sleep()
            sleep_count += 1

    if not agent.sleep_history:
        agent._refresh_self_model_continuity()
    return {
        "world_id": world_name,
        "cycles": cycles,
        "action_entropy": _action_entropy(actions),
        "caution_rate": _caution_rate(actions),
        "exploration_rate": _exploration_rate(actions),
        "seek_contact_rate": _seek_contact_rate(actions),
        "mean_conditioned_prediction_error": mean(prediction_errors) if prediction_errors else 0.0,
        "survival_score": mean(survival_trace) if survival_trace else 0.0,
        "narrative_consistency_proxy": narrative_consistency_proxy(agent),
        "dominant_action_share": _dominant_action_share(actions),
        "sleep_count": sleep_count,
        "narrative_event_count": narrative_updates,
        "final_core_identity": (
            agent.self_model.identity_narrative.core_identity
            if agent.self_model.identity_narrative is not None
            else ""
        ),
        "final_behavioral_patterns": (
            list(agent.self_model.identity_narrative.behavioral_patterns)
            if agent.self_model.identity_narrative is not None
            else []
        ),
    }


def run_profile_trial(
    *,
    profile_name: str,
    seed: int,
    cycles_per_world: int,
    worlds: tuple[str, ...] = M210_WORLD_NAMES,
) -> dict[str, object]:
    agent = build_profiled_agent(profile_name=profile_name, seed=seed)
    world_results = [
        _rollout_world(
            agent=agent,
            world_name=world_name,
            seed=seed + index * 37,
            cycles=cycles_per_world,
        )
        for index, world_name in enumerate(worlds)
    ]
    aggregate_metrics = {
        metric: mean(float(result[metric]) for result in world_results)
        for metric in (
            "caution_rate",
            "exploration_rate",
            "seek_contact_rate",
            "action_entropy",
            "survival_score",
            "mean_conditioned_prediction_error",
            "narrative_consistency_proxy",
            "dominant_action_share",
        )
    }
    return {
        "profile": profile_name,
        "seed": seed,
        "cycles_per_world": cycles_per_world,
        "worlds": list(world_results),
        "aggregate_metrics": aggregate_metrics,
        "final_identity": (
            agent.self_model.identity_narrative.to_dict()
            if agent.self_model.identity_narrative is not None
            else {}
        ),
        "final_personality_profile": agent.self_model.personality_profile.to_dict(),
        "final_narrative_priors": agent.self_model.narrative_priors.to_dict(),
    }


def run_personality_validation(
    *,
    seed: int = 44,
    cycles_per_world: int = 36,
    repeats: int = 4,
) -> dict[str, object]:
    protocols = profile_protocols()
    trials: dict[str, list[dict[str, object]]] = {name: [] for name in protocols}
    for profile_index, profile_name in enumerate(protocols):
        for repeat in range(repeats):
            trial_seed = seed + profile_index * 101 + repeat * 17
            trials[profile_name].append(
                run_profile_trial(
                    profile_name=profile_name,
                    seed=trial_seed,
                    cycles_per_world=cycles_per_world,
                )
            )

    metric_names = (
        "caution_rate",
        "exploration_rate",
        "seek_contact_rate",
        "action_entropy",
        "survival_score",
        "mean_conditioned_prediction_error",
        "narrative_consistency_proxy",
    )
    analyses = {
        metric: anova(
            {
                profile_name: [
                    float(trial["aggregate_metrics"][metric])
                    for trial in profile_trials
                ]
                for profile_name, profile_trials in trials.items()
            }
        )
        for metric in metric_names
    }

    per_profile_summary = {
        profile_name: {
            metric: mean(
                float(trial["aggregate_metrics"][metric])
                for trial in profile_trials
            )
            for metric in metric_names + ("dominant_action_share",)
        }
        for profile_name, profile_trials in trials.items()
    }

    significant_metrics = [
        metric_name
        for metric_name, result in analyses.items()
        if float(result["p_value"]) < 0.05
    ]
    effect_metrics = [
        metric_name
        for metric_name, result in analyses.items()
        if float(result["eta_squared"]) >= 0.06
    ]

    return {
        "milestone": "M2.10",
        "seed": seed,
        "cycles_per_world": cycles_per_world,
        "repeats": repeats,
        "profiles": {
            name: protocol.to_dict()
            for name, protocol in protocols.items()
        },
        "profile_summaries": per_profile_summary,
        "anova": analyses,
        "trials": trials,
        "acceptance": {
            "required_significant_metrics": 3,
            "required_effect_metrics": 2,
            "significant_metrics": significant_metrics,
            "effect_metrics": effect_metrics,
            "passed": (
                len(significant_metrics) >= 3
                and len(effect_metrics) >= 2
            ),
        },
    }


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    if abs(avg) < 1e-9:
        return 0.0
    return pstdev(values) / abs(avg)


def _range(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def run_longitudinal_stability(
    *,
    seed: int = 91,
    cycles_per_world: int = 60,
    repeats: int = 3,
) -> dict[str, object]:
    profiles = profile_protocols()
    per_profile: dict[str, dict[str, object]] = {}

    for profile_index, profile_name in enumerate(profiles):
        profile_seed = seed + profile_index * 113
        runs = [
            run_profile_trial(
                profile_name=profile_name,
                seed=profile_seed,
                cycles_per_world=cycles_per_world,
            )
            for repeat in range(repeats)
        ]
        metrics = {
            metric: [
                float(run["aggregate_metrics"][metric])
                for run in runs
            ]
            for metric in (
                "caution_rate",
                "exploration_rate",
                "seek_contact_rate",
                "action_entropy",
                "survival_score",
                "mean_conditioned_prediction_error",
                "narrative_consistency_proxy",
                "dominant_action_share",
            )
        }
        cv = {
            metric: _coefficient_of_variation(values)
            for metric, values in metrics.items()
        }
        metric_ranges = {
            metric: _range(values)
            for metric, values in metrics.items()
        }
        collapse_free = all(value < 0.95 for value in metrics["dominant_action_share"])
        entropy_ok = mean(metrics["action_entropy"]) >= 0.30
        narrative_ok = min(metrics["narrative_consistency_proxy"]) >= 0.70
        stable_distribution = max(
            metric_ranges["caution_rate"],
            metric_ranges["exploration_rate"],
            metric_ranges["seek_contact_rate"],
        ) <= 0.40
        same_profile_stable = (
            stable_distribution
            and metric_ranges["survival_score"] <= 0.35
            and metric_ranges["narrative_consistency_proxy"] <= 0.20
            and metric_ranges["action_entropy"] <= 0.60
        )
        per_profile[profile_name] = {
            "runs": runs,
            "mean_metrics": {metric: mean(values) for metric, values in metrics.items()},
            "coefficient_of_variation": cv,
            "metric_ranges": metric_ranges,
            "checks": {
                "same_profile_stable": same_profile_stable,
                "no_obvious_personality_collapse": collapse_free,
                "action_entropy_floor_met": entropy_ok,
                "narrative_bias_persists": narrative_ok,
            },
            "passed": same_profile_stable and collapse_free and entropy_ok and narrative_ok,
        }

    passed_profiles = [
        profile_name
        for profile_name, payload in per_profile.items()
        if bool(payload["passed"])
    ]
    return {
        "milestone": "M2.10",
        "seed": seed,
        "cycles_per_world": cycles_per_world,
        "repeats": repeats,
        "profiles": per_profile,
        "acceptance": {
            "required_profiles": len(profile_protocols()),
            "profiles_passing": len(passed_profiles),
            "passed_profiles": passed_profiles,
            "passed": len(passed_profiles) == len(profile_protocols()),
        },
    }


def summarize_profile_behaviors(
    personality_validation: dict[str, object],
    longitudinal_stability: dict[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "milestone": "M2.10",
        "profiles": {},
    }
    validation_profiles = dict(personality_validation.get("profile_summaries", {}))
    stability_profiles = dict(longitudinal_stability.get("profiles", {}))
    for profile_name in profile_protocols():
        validation_metrics = dict(validation_profiles.get(profile_name, {}))
        stability_payload = dict(stability_profiles.get(profile_name, {}))
        summary["profiles"][profile_name] = {
            "behavior_summary": validation_metrics,
            "stability_checks": dict(stability_payload.get("checks", {})),
            "final_passed": bool(stability_payload.get("passed", False)),
        }
    return summary


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
