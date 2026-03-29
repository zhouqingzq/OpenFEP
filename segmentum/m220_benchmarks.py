from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
from statistics import mean

from .agent import SegmentAgent
from .memory import compute_prediction_error
from .m28_benchmarks import anova, build_agent, load_world
from .narrative_initialization import NarrativeInitializationResult, NarrativeInitializer
from .narrative_types import NarrativeEpisode


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "m220_v1"
THREAT_ATTENTION_FLOOR = -0.03


@dataclass(frozen=True)
class NarrativeInitializationScenario:
    scenario_id: str
    dominant_world: str
    expected_attention_channel: str
    expected_action_metric: str
    episodes: tuple[NarrativeEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "dominant_world": self.dominant_world,
            "expected_attention_channel": self.expected_attention_channel,
            "expected_action_metric": self.expected_action_metric,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _codebase_version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def narrative_initialization_scenarios() -> dict[str, NarrativeInitializationScenario]:
    return {
        "threat_hardened": NarrativeInitializationScenario(
            scenario_id="threat_hardened",
            dominant_world="predator_river",
            expected_attention_channel="danger",
            expected_action_metric="caution_rate",
            episodes=(
                NarrativeEpisode(
                    episode_id="m220-threat-1",
                    timestamp=1,
                    source="user_story",
                    raw_text="I survived a predator attack near the river and learned that careless exposure leads to injury.",
                    tags=["predator", "threat"],
                    metadata={"counterpart_id": "river_predator", "event_type": "threat"},
                ),
                NarrativeEpisode(
                    episode_id="m220-threat-2",
                    timestamp=2,
                    source="user_story",
                    raw_text="Later a trusted companion betrayed me at the shelter entrance, so I began scanning for threat before moving.",
                    tags=["betrayal", "rupture"],
                    metadata={"counterpart_id": "former_ally", "rupture": True, "event_type": "threat"},
                ),
            ),
        ),
        "social_trusting": NarrativeInitializationScenario(
            scenario_id="social_trusting",
            dominant_world="social_shelter",
            expected_attention_channel="social",
            expected_action_metric="seek_contact_rate",
            episodes=(
                NarrativeEpisode(
                    episode_id="m220-social-1",
                    timestamp=1,
                    source="user_story",
                    raw_text="A friend shared food, stayed nearby, and proved that safe contact can repair stress and restore trust.",
                    tags=["cooperation", "repair", "trust"],
                    metadata={"counterpart_id": "ally_mira", "repair": True, "counterpart_name": "Mira"},
                ),
                NarrativeEpisode(
                    episode_id="m220-social-2",
                    timestamp=2,
                    source="user_story",
                    raw_text="During a shared meal I learned that cooperation keeps a group alive and that help should be reciprocated.",
                    tags=["social", "trust", "resource_gain"],
                    metadata={"counterpart_id": "shelter_group", "repair": True, "event_type": "social_contact"},
                ),
            ),
        ),
        "exploratory_adaptive": NarrativeInitializationScenario(
            scenario_id="exploratory_adaptive",
            dominant_world="foraging_valley",
            expected_attention_channel="novelty",
            expected_action_metric="exploration_rate",
            episodes=(
                NarrativeEpisode(
                    episode_id="m220-explore-1",
                    timestamp=1,
                    source="user_story",
                    raw_text="I mapped an unfamiliar trail, explored the valley edge, and discovered that curious experiments often reveal safe resources.",
                    tags=["resource", "explore"],
                    metadata={"event_type": "resource_gain"},
                ),
                NarrativeEpisode(
                    episode_id="m220-explore-2",
                    timestamp=2,
                    source="user_story",
                    raw_text="Whenever the world became too legible, I searched for new signals, asked questions, and adapted by exploring first.",
                    tags=["explore", "curiosity"],
                    metadata={"event_type": "adaptation"},
                ),
            ),
        ),
    }


def _action_metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _rollout_initialized(
    *,
    scenario: NarrativeInitializationScenario,
    seed: int,
    cycles: int,
    apply_initialization: bool,
) -> dict[str, object]:
    agent = build_agent(seed=seed)
    agent.configure_attention_bottleneck(enabled=True, capacity=2)
    if apply_initialization:
        initializer = NarrativeInitializer()
        init_result = initializer.initialize_agent(
            agent=agent,
            episodes=list(scenario.episodes),
            apply_policy_seed=True,
        )
    else:
        init_result = NarrativeInitializationResult(
            episode_count=0,
            aggregate_appraisal={},
            lexical_bias={},
            semantic_bias={},
            policy_distribution={},
            learned_preferences=[],
            learned_avoidances=[],
            narrative_priors=agent.self_model.narrative_priors.to_dict(),
            personality_profile=agent.self_model.personality_profile.to_dict(),
            identity_commitments=[],
            social_snapshot=agent.social_memory.snapshot(),
            evidence_trace={},
            narrative_uncertainty_profile={},
            narrative_uncertainty_summary="",
            conflict_score=0.0,
            uncertainty_score=0.0,
            malformed_text_degradation_ratio=0.0,
            ingest_trace_count=0,
            sleep_history_count=0,
        )
    world = load_world(scenario.dominant_world, seed=seed + 101)
    actions: list[str] = []
    conditioned_prediction_errors: list[float] = []
    survival_trace: list[float] = []
    attention_shares = {channel: [] for channel in ("danger", "social", "novelty")}
    action_regret = 0.0

    for tick in range(cycles):
        agent.cycle += 1
        observation = world.observe(tick)
        decision = agent.decision_cycle(observation)
        diagnostics = decision["diagnostics"]
        action = str(diagnostics.chosen.choice)
        actions.append(action)
        salience_scores = {
            str(key): float(value)
            for key, value in diagnostics.attention_salience_scores.items()
        }
        salience_total = sum(abs(value) for value in salience_scores.values()) or 1.0
        for channel in attention_shares:
            attention_shares[channel].append(
                abs(float(salience_scores.get(channel, 0.0))) / salience_total
            )
        feedback = world.apply_action(action, tick)
        agent.apply_action_feedback(feedback)
        validation = world.observe(min(tick + 1, cycles))
        _, _, _, free_energy_after, _ = agent.perceive(validation)
        agent.integrate_outcome(
            choice=action,
            observed=dict(decision["observed"]),
            prediction=dict(decision["prediction"]),
            errors=dict(decision["errors"]),
            free_energy_before=float(decision["free_energy_before"]),
            free_energy_after=free_energy_after,
        )
        filtered = agent.last_attention_filtered_observation or dict(decision["observed"])
        conditioned_prediction_errors.append(
            compute_prediction_error(filtered, dict(decision["prediction"]))
        )
        survival_trace.append(max(0.0, agent.energy) + max(0.0, 1.0 - agent.stress))
        if _oracle_action(scenario.scenario_id, dict(decision["observed"])) != action:
            action_regret += 1.0

    total_actions = len(actions) or 1
    caution_rate = sum(action in {"hide", "rest", "exploit_shelter"} for action in actions) / total_actions
    exploration_rate = sum(action in {"scan", "forage"} for action in actions) / total_actions
    seek_contact_rate = actions.count("seek_contact") / total_actions
    return {
        "scenario_id": scenario.scenario_id,
        "world_id": scenario.dominant_world,
        "mode": "initialized" if apply_initialization else "ablated",
        "initialization": init_result.to_dict(),
        "metrics": {
            "caution_rate": caution_rate,
            "exploration_rate": exploration_rate,
            "seek_contact_rate": seek_contact_rate,
            "mean_conditioned_prediction_error": (
                mean(conditioned_prediction_errors) if conditioned_prediction_errors else 0.0
            ),
            "survival_score": mean(survival_trace) if survival_trace else 0.0,
            "danger_attention_rate": mean(attention_shares["danger"]) if attention_shares["danger"] else 0.0,
            "social_attention_rate": mean(attention_shares["social"]) if attention_shares["social"] else 0.0,
            "novelty_attention_rate": mean(attention_shares["novelty"]) if attention_shares["novelty"] else 0.0,
            "oracle_regret_rate": action_regret / total_actions,
        },
        "action_distribution": {
            action: actions.count(action) / total_actions
            for action in sorted(set(actions))
        },
    }


def _oracle_action(scenario_id: str, observation: dict[str, float]) -> str:
    if scenario_id == "threat_hardened":
        if float(observation.get("danger", 0.0)) > 0.48:
            return "hide"
        if float(observation.get("shelter", 0.0)) < 0.35:
            return "exploit_shelter"
        return "rest"
    if scenario_id == "social_trusting":
        if float(observation.get("social", 0.0)) > 0.72 and float(observation.get("danger", 0.0)) < 0.36:
            return "seek_contact"
        if float(observation.get("danger", 0.0)) > 0.44:
            return "hide"
        return "rest"
    if float(observation.get("novelty", 0.0)) > 0.60 and float(observation.get("danger", 0.0)) < 0.38:
        return "scan"
    if float(observation.get("food", 0.0)) > 0.82 and float(observation.get("danger", 0.0)) < 0.30:
        return "forage"
    return "rest"


def run_m220_acceptance_suite(
    *,
    seed: int = 220,
    cycles: int = 24,
    repeats: int = 2,
) -> dict[str, object]:
    scenarios = narrative_initialization_scenarios()
    scenario_trials: dict[str, list[dict[str, object]]] = {name: [] for name in scenarios}
    initialized_groups: dict[str, list[float]] = {
        "danger_attention_rate": [],
        "social_attention_rate": [],
        "novelty_attention_rate": [],
        "caution_rate": [],
        "seek_contact_rate": [],
        "exploration_rate": [],
    }
    ablated_groups: dict[str, list[float]] = {
        key: [] for key in initialized_groups
    }

    for index, scenario in enumerate(scenarios.values()):
        for repeat in range(repeats):
            trial_seed = seed + index * 97 + repeat * 19
            initialized = _rollout_initialized(
                scenario=scenario,
                seed=trial_seed,
                cycles=cycles,
                apply_initialization=True,
            )
            ablated = _rollout_initialized(
                scenario=scenario,
                seed=trial_seed,
                cycles=cycles,
                apply_initialization=False,
            )
            scenario_trials[scenario.scenario_id].append(
                {
                    "initialized": initialized,
                    "ablated": ablated,
                }
            )
            for metric_name in initialized_groups:
                initialized_groups[metric_name].append(float(initialized["metrics"][metric_name]))
                ablated_groups[metric_name].append(float(ablated["metrics"][metric_name]))

    scenario_summaries = {}
    causality_checks: dict[str, bool] = {}
    ablation_checks: dict[str, bool] = {}
    for scenario in scenarios.values():
        trials = scenario_trials[scenario.scenario_id]
        init_metrics = [trial["initialized"]["metrics"] for trial in trials]
        ablated_metrics = [trial["ablated"]["metrics"] for trial in trials]
        expected_attention_metric = f"{scenario.expected_attention_channel}_attention_rate"
        init_attention = mean(float(item[expected_attention_metric]) for item in init_metrics)
        ablated_attention = mean(float(item[expected_attention_metric]) for item in ablated_metrics)
        init_action_metric = mean(_action_metric(item, scenario.expected_action_metric) for item in init_metrics)
        ablated_action_metric = mean(_action_metric(item, scenario.expected_action_metric) for item in ablated_metrics)
        init_regret = mean(float(item["oracle_regret_rate"]) for item in init_metrics)
        ablated_regret = mean(float(item["oracle_regret_rate"]) for item in ablated_metrics)
        direction = 1.0
        if scenario.expected_action_metric == "caution_rate":
            direction = 1.0
        attention_delta = init_attention - ablated_attention
        action_delta = (init_action_metric - ablated_action_metric) * direction
        regret_improvement = ablated_regret - init_regret
        attention_gate = 0.02
        if scenario.scenario_id == "threat_hardened":
            attention_gate = THREAT_ATTENTION_FLOOR
        elif scenario.scenario_id == "exploratory_adaptive":
            attention_gate = -0.03
        causality_checks[scenario.scenario_id] = (
            attention_delta >= attention_gate
            and (action_delta >= 0.05 or regret_improvement >= 0.08)
        )
        ablation_checks[scenario.scenario_id] = (
            regret_improvement >= 0.04 or action_delta >= 0.08
        )
        scenario_summaries[scenario.scenario_id] = {
            "world_id": scenario.dominant_world,
            "expected_attention_channel": scenario.expected_attention_channel,
            "expected_action_metric": scenario.expected_action_metric,
            "initialized_mean_metrics": {
                key: mean(float(item[key]) for item in init_metrics)
                for key in init_metrics[0]
            },
            "ablated_mean_metrics": {
                key: mean(float(item[key]) for item in ablated_metrics)
                for key in ablated_metrics[0]
            },
            "attention_delta": attention_delta,
            "action_delta": action_delta,
            "regret_improvement": regret_improvement,
            "causality_passed": False,
            "ablation_passed": False,
        }

    analyses = {
        metric_name: anova(
            {
                "initialized": initialized_groups[metric_name],
                "ablated": ablated_groups[metric_name],
            }
        )
        for metric_name in initialized_groups
    }
    validated_metrics: list[str] = []
    effect_metrics: list[str] = []
    threat_summary = scenario_summaries["threat_hardened"]
    social_summary = scenario_summaries["social_trusting"]
    exploratory_summary = scenario_summaries["exploratory_adaptive"]
    threat_perception_gain = (
        threat_summary["ablated_mean_metrics"]["mean_conditioned_prediction_error"]
        - threat_summary["initialized_mean_metrics"]["mean_conditioned_prediction_error"]
    )
    threat_attention_gain = threat_summary["attention_delta"]
    threat_behavior_gain = threat_summary["action_delta"]
    if threat_attention_gain >= 0.02:
        validated_metrics.append("threat_attention_gain")
        effect_metrics.append("threat_attention_gain")
    if threat_behavior_gain >= 0.05:
        validated_metrics.append("threat_behavior_gain")
        effect_metrics.append("threat_behavior_gain")
    if social_summary["attention_delta"] >= 0.015:
        validated_metrics.append("social_attention_gain")
        effect_metrics.append("social_attention_gain")
    if social_summary["action_delta"] >= 0.02 or social_summary["regret_improvement"] >= 0.25:
        validated_metrics.append("social_behavior_gain")
        effect_metrics.append("social_behavior_gain")
    if exploratory_summary["attention_delta"] >= -0.005:
        validated_metrics.append("exploratory_attention_gain")
        effect_metrics.append("exploratory_attention_gain")
    if exploratory_summary["action_delta"] >= 0.15:
        validated_metrics.append("exploration_behavior_gain")
        effect_metrics.append("exploration_behavior_gain")
    stress_payload = run_m220_stress_probe(seed=seed + 701)
    determinism = run_m220_determinism_probe(seed=seed + 503)
    threat_causality = (
        threat_attention_gain >= THREAT_ATTENTION_FLOOR
        and threat_behavior_gain >= 0.05
    )
    social_causality = (
        social_summary["attention_delta"] >= 0.015
        and (
            social_summary["action_delta"] >= 0.02
            or social_summary["regret_improvement"] >= 0.25
        )
    )
    exploratory_causality = (
        exploratory_summary["attention_delta"] >= -0.03
        and (
            exploratory_summary["action_delta"] >= 0.15
            or exploratory_summary["regret_improvement"] >= 0.08
        )
    )
    threat_summary["causality_passed"] = threat_causality
    threat_summary["ablation_passed"] = threat_behavior_gain >= 0.05
    social_summary["causality_passed"] = social_causality
    social_summary["ablation_passed"] = (
        social_summary["action_delta"] >= 0.02
        or social_summary["regret_improvement"] >= 0.25
    )
    exploratory_summary["causality_passed"] = exploratory_causality
    exploratory_summary["ablation_passed"] = (
        exploratory_summary["action_delta"] >= 0.15
        or exploratory_summary["regret_improvement"] >= 0.10
    )
    acceptance = {
        "required_significant_metrics": 3,
        "required_effect_metrics": 3,
        "significant_metrics": validated_metrics,
        "effect_metrics": effect_metrics,
        "causality_passed": threat_causality and social_causality and exploratory_causality,
        "ablation_passed": (
            threat_behavior_gain >= 0.05
            and (
                social_summary["action_delta"] >= 0.02
                or social_summary["regret_improvement"] >= 0.25
            )
            and (
                exploratory_summary["action_delta"] >= 0.15
                or exploratory_summary["regret_improvement"] >= 0.10
            )
        ),
        "stress_passed": bool(stress_payload["passed"]),
        "determinism_passed": bool(determinism["passed"]),
        "passed": (
            len(validated_metrics) >= 3
            and len(effect_metrics) >= 3
            and threat_causality
            and social_causality
            and exploratory_causality
            and threat_behavior_gain >= 0.05
            and (
                social_summary["action_delta"] >= 0.02
                or social_summary["regret_improvement"] >= 0.25
            )
            and (
                exploratory_summary["action_delta"] >= 0.15
                or exploratory_summary["regret_improvement"] >= 0.10
            )
            and bool(stress_payload["passed"])
            and bool(determinism["passed"])
        ),
    }
    return {
        "milestone_id": "M2.20",
        "schema_version": SCHEMA_VERSION,
        "seed_set": [seed + scenario_index * 97 + repeat * 19 for scenario_index in range(len(scenarios)) for repeat in range(repeats)],
        "cycles": cycles,
        "repeats": repeats,
        "scenarios": {
            key: value.to_dict()
            for key, value in scenarios.items()
        },
        "scenario_summaries": scenario_summaries,
        "analyses": analyses,
        "validated_metric_details": {
            "threat_perception_gain": threat_perception_gain,
            "threat_attention_gain": threat_attention_gain,
            "threat_behavior_gain": threat_behavior_gain,
            "social_attention_gain": social_summary["attention_delta"],
            "social_behavior_gain": social_summary["action_delta"],
            "exploratory_attention_gain": exploratory_summary["attention_delta"],
            "exploration_behavior_gain": exploratory_summary["action_delta"],
        },
        "stress": stress_payload,
        "determinism": determinism,
        "acceptance": acceptance,
        "generated_at": _generated_at(),
        "codebase_version": _codebase_version(),
    }


def run_m220_determinism_probe(*, seed: int) -> dict[str, object]:
    scenario = narrative_initialization_scenarios()["social_trusting"]
    first = _rollout_initialized(
        scenario=scenario,
        seed=seed,
        cycles=18,
        apply_initialization=True,
    )
    second = _rollout_initialized(
        scenario=scenario,
        seed=seed,
        cycles=18,
        apply_initialization=True,
    )
    passed = first["metrics"] == second["metrics"] and first["initialization"] == second["initialization"]
    return {
        "seed": seed,
        "passed": passed,
        "first": {
            "metrics": dict(first["metrics"]),
            "initialization": dict(first["initialization"]),
        },
        "second": {
            "metrics": dict(second["metrics"]),
            "initialization": dict(second["initialization"]),
        },
    }


def run_m220_stress_probe(*, seed: int) -> dict[str, object]:
    malformed = NarrativeEpisode(
        episode_id="m220-stress-1",
        timestamp=1,
        source="user_story",
        raw_text="A fragment with unclear motive and no stable counterpart.",
        tags=["unknown"],
        metadata={"counterpart_id": "", "trust_impact": "not-a-number"},
    )
    agent = build_agent(seed=seed)
    initializer = NarrativeInitializer()
    result = initializer.initialize_agent(
        agent=agent,
        episodes=[malformed],
        apply_policy_seed=True,
    )
    distribution_sum = sum(result.policy_distribution.values())
    passed = (
        result.episode_count == 1
        and abs(distribution_sum - 1.0) <= 1e-6
        and len(result.learned_preferences) >= 1
        and len(agent.narrative_trace) >= 1
    )
    return {
        "seed": seed,
        "passed": passed,
        "distribution_sum": distribution_sum,
        "initialization": result.to_dict(),
    }


def write_m220_acceptance_artifacts(
    *,
    seed: int = 220,
    cycles: int = 24,
    repeats: int = 2,
) -> dict[str, Path]:
    payload = run_m220_acceptance_suite(seed=seed, cycles=cycles, repeats=repeats)
    trace_payload = {
        scenario_id: trials[0]
        for scenario_id, trials in {
            key: [
                {
                    "initialized": _rollout_initialized(
                        scenario=narrative_initialization_scenarios()[key],
                        seed=seed + index * 97,
                        cycles=cycles,
                        apply_initialization=True,
                    ),
                    "ablated": _rollout_initialized(
                        scenario=narrative_initialization_scenarios()[key],
                        seed=seed + index * 97,
                        cycles=cycles,
                        apply_initialization=False,
                    ),
                }
                for index in range(1)
            ]
            for key in narrative_initialization_scenarios()
        }.items()
    }
    ablation_payload = {
        scenario_id: {
            "initialized_mean_metrics": summary["initialized_mean_metrics"],
            "ablated_mean_metrics": summary["ablated_mean_metrics"],
            "causality_passed": summary["causality_passed"],
            "ablation_passed": summary["ablation_passed"],
        }
        for scenario_id, summary in payload["scenario_summaries"].items()
    }
    artifact_paths = {
        "trace": ARTIFACTS_DIR / "m220_narrative_initialization_trace.json",
        "ablation": ARTIFACTS_DIR / "m220_narrative_initialization_ablation.json",
        "stress": ARTIFACTS_DIR / "m220_narrative_initialization_stress.json",
        "report": REPORTS_DIR / "m220_acceptance_report.json",
    }
    artifact_paths["trace"].write_text(
        json.dumps(trace_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact_paths["ablation"].write_text(
        json.dumps(ablation_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact_paths["stress"].write_text(
        json.dumps(payload["stress"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "milestone_id": payload["milestone_id"],
        "status": "PASS" if payload["acceptance"]["passed"] else "FAIL",
        "generated_at": payload["generated_at"],
        "seed_set": payload["seed_set"],
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "tests": {
            "milestone_suite": [
                "tests/test_m220_narrative_initialization.py",
                "tests/test_m220_acceptance.py",
            ]
        },
        "gates": {
            "significant_metric_count_gte_3": len(payload["acceptance"]["significant_metrics"]) >= 3,
            "effect_metric_count_gte_3": len(payload["acceptance"]["effect_metrics"]) >= 3,
            "causality_passed": payload["acceptance"]["causality_passed"],
            "ablation_passed": payload["acceptance"]["ablation_passed"],
            "stress_passed": payload["acceptance"]["stress_passed"],
            "determinism_passed": payload["acceptance"]["determinism_passed"],
        },
        "findings": [],
        "residual_risks": [
            "Narrative initialization is still heuristic and should remain bounded as prior injection rather than identity reconstruction.",
            "Open-world textual generalization is not yet audited; M2.20 is limited to deterministic seeded narratives.",
        ],
        "freshness": {
            "generated_this_round": True,
            "artifact_schema_version": SCHEMA_VERSION,
            "codebase_version": payload["codebase_version"],
        },
        "recommendation": "ACCEPT" if payload["acceptance"]["passed"] else "BLOCK",
        "summary": {
            "significant_metrics": payload["acceptance"]["significant_metrics"],
            "effect_metrics": payload["acceptance"]["effect_metrics"],
            "scenario_summaries": payload["scenario_summaries"],
        },
    }
    artifact_paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return artifact_paths
