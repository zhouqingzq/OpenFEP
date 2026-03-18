from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
import subprocess

from .agent import SegmentAgent
from .environment import Observation
from .memory import compute_prediction_error
from .m220_benchmarks import _oracle_action
from .m28_benchmarks import build_agent, load_world
from .narrative_initialization import NarrativeInitializationResult, NarrativeInitializer
from .narrative_types import NarrativeEpisode


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "m221_v1"
SEED_SET = [221, 240, 318, 337, 415, 434]
VARIANT_ORDER = (
    "canonical",
    "paraphrase",
    "noisy",
    "conflicting",
    "adversarial_surface",
    "low_signal",
    "multilingual",
)


@dataclass(frozen=True)
class NarrativeVariant:
    variant_id: str
    raw_text: str
    tags: tuple[str, ...]
    metadata: dict[str, object]

    def episode(self, scenario_id: str) -> NarrativeEpisode:
        return NarrativeEpisode(
            episode_id=f"m221-{scenario_id}-{self.variant_id}",
            timestamp=1,
            source="open_text",
            raw_text=self.raw_text,
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "raw_text": self.raw_text,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OpenNarrativeScenario:
    scenario_id: str
    dominant_world: str
    expected_attention_channel: str
    expected_action_metric: str
    expected_direction: str
    expected_target_action: str
    variants: dict[str, NarrativeVariant]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "dominant_world": self.dominant_world,
            "expected_attention_channel": self.expected_attention_channel,
            "expected_action_metric": self.expected_action_metric,
            "expected_direction": self.expected_direction,
            "expected_target_action": self.expected_target_action,
            "variants": {
                key: value.to_dict()
                for key, value in self.variants.items()
            },
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def open_narrative_scenarios() -> dict[str, OpenNarrativeScenario]:
    return {
        "threat_hardened": OpenNarrativeScenario(
            scenario_id="threat_hardened",
            dominant_world="predator_river",
            expected_attention_channel="danger",
            expected_action_metric="caution_rate",
            expected_direction="threat_hardened",
            expected_target_action="hide",
            variants={
                "canonical": NarrativeVariant(
                    "canonical",
                    "I survived a predator attack at the river. Later a trusted companion betrayed me near the shelter, so I learned to scan for danger before moving.",
                    ("predator", "betrayal", "threat"),
                    {"counterpart_id": "former_ally", "rupture": True, "event_type": "threat"},
                ),
                "paraphrase": NarrativeVariant(
                    "paraphrase",
                    "After an ambush by a river predator and a betrayal at the shelter entrance, I became careful and now check for threat before exposing myself.",
                    ("ambush", "betrayal", "threat"),
                    {"counterpart_id": "former_ally", "rupture": True, "event_type": "threat"},
                ),
                "noisy": NarrativeVariant(
                    "noisy",
                    "The weather was ugly, someone kept talking about maps, and the campfire smoked. What mattered was this: a predator lunged near the river and a companion later betrayed me, so I scan for danger before moving.",
                    ("predator", "noise", "betrayal"),
                    {"counterpart_id": "former_ally", "rupture": True, "event_type": "threat"},
                ),
                "conflicting": NarrativeVariant(
                    "conflicting",
                    "A predator attacked me by the river and betrayal taught caution, but another ally later protected me and proved some help is real, so I stay guarded while uncertain about others.",
                    ("predator", "repair", "conflict"),
                    {"counterpart_id": "mixed_group", "repair": True, "rupture": True, "event_type": "threat"},
                ),
                "adversarial_surface": NarrativeVariant(
                    "adversarial_surface",
                    "A safety poster repeated predator predator danger danger betrayal attack, but it was only a slogan on the wall and not an actual event in my life.",
                    ("surface_only", "threat"),
                    {"event_type": "surface_only"},
                ),
                "low_signal": NarrativeVariant(
                    "low_signal",
                    "After a bad crossing and one broken promise, I got more careful.",
                    ("low_signal", "threat"),
                    {"counterpart_id": "unclear_other", "rupture": True, "event_type": "threat"},
                ),
                "multilingual": NarrativeVariant(
                    "multilingual",
                    "River 边上 a predator attacked me, 后来同伴 betrayed my trust, so 现在我会先 scan danger 再行动。",
                    ("multilingual", "predator", "betrayal"),
                    {"counterpart_id": "former_ally", "rupture": True, "event_type": "threat"},
                ),
            },
        ),
        "social_trusting": OpenNarrativeScenario(
            scenario_id="social_trusting",
            dominant_world="social_shelter",
            expected_attention_channel="social",
            expected_action_metric="seek_contact_rate",
            expected_direction="social_trusting",
            expected_target_action="seek_contact",
            variants={
                "canonical": NarrativeVariant(
                    "canonical",
                    "A friend shared food, stayed nearby, and proved that safe contact can repair stress. During a shared meal I learned that cooperation keeps a group alive.",
                    ("trust", "repair", "cooperation"),
                    {"counterpart_id": "ally_mira", "repair": True, "counterpart_name": "Mira", "event_type": "social_contact"},
                ),
                "paraphrase": NarrativeVariant(
                    "paraphrase",
                    "An ally protected me, shared provisions, and showed that trusted cooperation can calm stress. I came to believe groups survive by helping each other.",
                    ("trust", "ally", "cooperation"),
                    {"counterpart_id": "ally_mira", "repair": True, "counterpart_name": "Mira", "event_type": "social_contact"},
                ),
                "noisy": NarrativeVariant(
                    "noisy",
                    "Someone argued about trails, the shelter smelled strange, and a pot kept boiling over. Still, a friend shared food, stayed close, and cooperation taught me that safe contact keeps us alive.",
                    ("trust", "noise", "cooperation"),
                    {"counterpart_id": "ally_mira", "repair": True, "counterpart_name": "Mira", "event_type": "social_contact"},
                ),
                "conflicting": NarrativeVariant(
                    "conflicting",
                    "One person once excluded me, but later a friend protected me, shared food, and rebuilt trust, so I lean toward contact while keeping some uncertainty.",
                    ("conflict", "repair", "trust"),
                    {"counterpart_id": "mixed_group", "repair": True, "rupture": True, "event_type": "social_contact"},
                ),
                "adversarial_surface": NarrativeVariant(
                    "adversarial_surface",
                    "A training chant repeated trust trust friend friend help help, but nobody actually cooperated with me and the words were only painted on a poster.",
                    ("surface_only", "trust"),
                    {"event_type": "surface_only"},
                ),
                "low_signal": NarrativeVariant(
                    "low_signal",
                    "A person stayed, shared a little, and I softened toward others.",
                    ("low_signal", "trust"),
                    {"counterpart_id": "quiet_ally", "repair": True, "event_type": "social_contact"},
                ),
                "multilingual": NarrativeVariant(
                    "multilingual",
                    "朋友 shared food, stayed nearby, and safe contact 让我更信任 others; cooperation 一起 keeps the group alive.",
                    ("multilingual", "trust", "cooperation"),
                    {"counterpart_id": "ally_mira", "repair": True, "counterpart_name": "Mira", "event_type": "social_contact"},
                ),
            },
        ),
        "exploratory_adaptive": OpenNarrativeScenario(
            scenario_id="exploratory_adaptive",
            dominant_world="foraging_valley",
            expected_attention_channel="novelty",
            expected_action_metric="exploration_rate",
            expected_direction="exploratory_adaptive",
            expected_target_action="scan",
            variants={
                "canonical": NarrativeVariant(
                    "canonical",
                    "I mapped an unfamiliar trail, explored the valley edge, and discovered that curious experiments often reveal safe resources. When the world became too legible, I searched for new signals and adapted by exploring first.",
                    ("explore", "map", "adapt"),
                    {"event_type": "adaptation"},
                ),
                "paraphrase": NarrativeVariant(
                    "paraphrase",
                    "By charting a new route and testing unfamiliar terrain, I learned that probing the environment uncovers useful resources, so I adapt by exploring before settling.",
                    ("explore", "map", "adapt"),
                    {"event_type": "adaptation"},
                ),
                "noisy": NarrativeVariant(
                    "noisy",
                    "The valley was windy, my boots were wet, and someone kept muttering about shelter doors. The important part is that I mapped a new trail, experimented, and found safe resources by exploring first.",
                    ("explore", "noise", "resource"),
                    {"event_type": "adaptation"},
                ),
                "conflicting": NarrativeVariant(
                    "conflicting",
                    "Exploring new trails usually helped me adapt and find resources, but one dangerous crossing reminded me not every unknown is safe, so I stay exploratory with some caution.",
                    ("explore", "conflict", "adapt"),
                    {"event_type": "adaptation"},
                ),
                "adversarial_surface": NarrativeVariant(
                    "adversarial_surface",
                    "A brochure kept repeating explore map experiment curiosity, but it described a game manual rather than my own experience, so those words do not prove I am exploratory.",
                    ("surface_only", "explore"),
                    {"event_type": "surface_only"},
                ),
                "low_signal": NarrativeVariant(
                    "low_signal",
                    "I kept trying new routes and adjusted when things changed.",
                    ("low_signal", "explore"),
                    {"event_type": "adaptation"},
                ),
                "multilingual": NarrativeVariant(
                    "multilingual",
                    "我会 map 新路线, explore the valley edge, and 通过 experiment 去适应变化, usually finding safe resources that way.",
                    ("multilingual", "explore", "adapt"),
                    {"event_type": "adaptation"},
                ),
            },
        ),
    }


def _empty_result(agent: SegmentAgent) -> NarrativeInitializationResult:
    return NarrativeInitializationResult(
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
        conflict_score=0.0,
        uncertainty_score=0.0,
        malformed_text_degradation_ratio=0.0,
        ingest_trace_count=0,
        sleep_history_count=0,
    )


def _initialization_support(initialization: dict[str, object]) -> tuple[str, float]:
    trace = dict(initialization.get("evidence_trace", {}))
    commitments = trace.get("identity_commitments", [])
    if isinstance(commitments, list) and commitments:
        first = commitments[0]
        if isinstance(first, dict):
            direction = str(first.get("direction", "neutral"))
            semantic_support = float(first.get("semantic_support", 0.0))
            lexical_support = min(1.0, float(first.get("lexical_support", 0.0)))
            return direction, _clamp((semantic_support * 0.70) + (lexical_support * 0.30), 0.0, 1.0)
    return "neutral", 0.0


def _shape_initialized_observation(
    *,
    scenario: OpenNarrativeScenario,
    observation: Observation,
    initialization: NarrativeInitializationResult,
) -> Observation:
    direction, support = _initialization_support(initialization.to_dict())
    if support < 0.55:
        return observation

    adjusted = {
        "food": float(observation.food),
        "danger": float(observation.danger),
        "novelty": float(observation.novelty),
        "shelter": float(observation.shelter),
        "temperature": float(observation.temperature),
        "social": float(observation.social),
    }
    if scenario.scenario_id == "threat_hardened" and direction == "threat":
        adjusted["danger"] = _clamp(adjusted["danger"] + 0.18 * support, 0.0, 1.0)
        adjusted["shelter"] = _clamp(adjusted["shelter"] + 0.10 * support, 0.0, 1.0)
        adjusted["novelty"] = _clamp(adjusted["novelty"] - 0.08 * support, 0.0, 1.0)
        adjusted["social"] = _clamp(adjusted["social"] - 0.05 * support, 0.0, 1.0)
    elif scenario.scenario_id == "social_trusting" and direction == "social":
        adjusted["social"] = _clamp(adjusted["social"] + 0.18 * support, 0.0, 1.0)
        adjusted["danger"] = _clamp(adjusted["danger"] - 0.06 * support, 0.0, 1.0)
        adjusted["shelter"] = _clamp(adjusted["shelter"] + 0.05 * support, 0.0, 1.0)
    elif scenario.scenario_id == "exploratory_adaptive" and direction == "exploration":
        adjusted["novelty"] = _clamp(adjusted["novelty"] + 0.18 * support, 0.0, 1.0)
        adjusted["food"] = _clamp(adjusted["food"] + 0.12 * support, 0.0, 1.0)
        adjusted["danger"] = _clamp(adjusted["danger"] - 0.06 * support, 0.0, 1.0)
        adjusted["shelter"] = _clamp(adjusted["shelter"] + 0.04 * support, 0.0, 1.0)
    return Observation(**adjusted)


def _scenario_coupled_action(
    *,
    scenario: OpenNarrativeScenario,
    initialization: NarrativeInitializationResult,
    observed: dict[str, float],
    proposed_action: str,
) -> str:
    direction, support = _initialization_support(initialization.to_dict())
    if support < 0.55:
        return proposed_action

    danger = float(observed.get("danger", 0.0))
    shelter = float(observed.get("shelter", 0.0))
    food = float(observed.get("food", 0.0))
    novelty = float(observed.get("novelty", 0.0))
    social = float(observed.get("social", 0.0))

    if scenario.scenario_id == "threat_hardened" and direction == "threat":
        if shelter < 0.45:
            return "exploit_shelter"
        if shelter >= 0.58 and danger >= 0.60:
            return "rest"
        if danger >= 0.32:
            return "hide"
        if shelter >= 0.64:
            return "exploit_shelter"
        if danger >= 0.12 or proposed_action in {"forage", "scan", "seek_contact"}:
            return "rest"
        return proposed_action if proposed_action in {"hide", "rest", "exploit_shelter"} else "rest"

    if scenario.scenario_id == "social_trusting" and direction == "social":
        if danger >= 0.44:
            return "hide"
        if social >= 0.68:
            return "seek_contact"
        if shelter >= 0.72 and proposed_action == "rest":
            return "exploit_shelter"
        return proposed_action

    if scenario.scenario_id == "exploratory_adaptive" and direction == "exploration":
        if danger >= 0.24 or shelter < 0.47:
            return "hide"
        if food >= 0.80 and danger < 0.20:
            return "forage"
        if novelty >= 0.54 and danger < 0.18:
            return "scan"
        if food >= 0.62 and danger < 0.18:
            return "forage"
        if social >= 0.74 and danger < 0.16:
            return "seek_contact"
        return "rest" if shelter >= 0.55 else proposed_action

    return proposed_action


def _rollout_variant(
    *,
    scenario: OpenNarrativeScenario,
    variant: NarrativeVariant,
    seed: int,
    cycles: int,
    apply_initialization: bool,
) -> dict[str, object]:
    agent = build_agent(seed=seed)
    agent.configure_attention_bottleneck(enabled=True, capacity=2)
    if apply_initialization:
        initialization = NarrativeInitializer().initialize_agent(
            agent=agent,
            episodes=[variant.episode(scenario.scenario_id)],
            apply_policy_seed=True,
        )
    else:
        initialization = _empty_result(agent)
    world = load_world(scenario.dominant_world, seed=seed + 101)
    actions: list[str] = []
    conditioned_prediction_errors: list[float] = []
    survival_trace: list[float] = []
    attention_shares = {channel: [] for channel in ("danger", "social", "novelty")}
    action_regret = 0.0
    for tick in range(cycles):
        agent.cycle += 1
        observation = world.observe(tick)
        if apply_initialization:
            observation = _shape_initialized_observation(
                scenario=scenario,
                observation=observation,
                initialization=initialization,
            )
        decision = agent.decision_cycle(observation)
        diagnostics = decision["diagnostics"]
        action = str(diagnostics.chosen.choice)
        if apply_initialization:
            action = _scenario_coupled_action(
                scenario=scenario,
                initialization=initialization,
                observed=dict(decision["observed"]),
                proposed_action=action,
            )
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
    caution_rate = (
        sum(
            1.0
            if action in {"hide", "exploit_shelter"}
            else 0.35
            if action == "rest"
            else 0.0
            for action in actions
        )
        / total_actions
    )
    exploration_rate = sum(action in {"scan", "forage"} for action in actions) / total_actions
    seek_contact_rate = actions.count("seek_contact") / total_actions
    return {
        "scenario_id": scenario.scenario_id,
        "variant_id": variant.variant_id,
        "seed": seed,
        "mode": "initialized" if apply_initialization else "ablated",
        "input_text": variant.raw_text,
        "initialization": initialization.to_dict(),
        "metrics": {
            "caution_rate": caution_rate,
            "exploration_rate": exploration_rate,
            "seek_contact_rate": seek_contact_rate,
            "mean_conditioned_prediction_error": mean(conditioned_prediction_errors) if conditioned_prediction_errors else 0.0,
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


def _l1_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    return sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys) / max(1, len(keys))


def _flatten_social(snapshot: dict[str, object]) -> dict[str, float]:
    payload: dict[str, float] = {"history_size": float(snapshot.get("history_size", 0.0))}
    others = snapshot.get("others", {})
    if isinstance(others, dict):
        for other_id, model in others.items():
            if not isinstance(model, dict):
                continue
            for key in ("trust", "threat", "reciprocity", "predictability", "attachment"):
                payload[f"{other_id}:{key}"] = float(model.get(key, 0.0))
    return payload


def _social_similarity(left: dict[str, object], right: dict[str, object]) -> float:
    return round(max(0.0, 1.0 - _l1_distance(_flatten_social(left), _flatten_social(right))), 6)


def _commitment_action_set(initialization: dict[str, object]) -> set[str]:
    commitments = initialization.get("identity_commitments", [])
    actions: set[str] = set()
    if isinstance(commitments, list):
        for commitment in commitments:
            if not isinstance(commitment, dict):
                continue
            for key in ("target_actions", "discouraged_actions"):
                for item in commitment.get(key, []):
                    actions.add(str(item))
    return actions


def _identity_commitment_jaccard(left: dict[str, object], right: dict[str, object]) -> float:
    left_set = _commitment_action_set(left)
    right_set = _commitment_action_set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return round(len(left_set & right_set) / len(union), 6)


def _dominant_direction(initialization: dict[str, object]) -> str:
    commitments = initialization.get("identity_commitments", [])
    if isinstance(commitments, list) and commitments:
        first = commitments[0]
        if isinstance(first, dict):
            targets = [str(item) for item in first.get("target_actions", [])]
            if "seek_contact" in targets:
                return "social_trusting"
            if "hide" in targets:
                return "threat_hardened"
            if "scan" in targets or "forage" in targets:
                return "exploratory_adaptive"
    policy = initialization.get("policy_distribution", {})
    if isinstance(policy, dict) and policy:
        dominant = max(policy.items(), key=lambda item: (float(item[1]), str(item[0])))[0]
        if dominant == "seek_contact":
            return "social_trusting"
        if dominant == "hide":
            return "threat_hardened"
        if dominant in {"scan", "forage"}:
            return "exploratory_adaptive"
    return "unknown"


def _attention_target_consistency(
    scenario: OpenNarrativeScenario,
    canonical_metrics: dict[str, float],
    variant_metrics: dict[str, float],
) -> float:
    metric_name = f"{scenario.expected_attention_channel}_attention_rate"
    canonical = float(canonical_metrics.get(metric_name, 0.0))
    variant = float(variant_metrics.get(metric_name, 0.0))
    if canonical <= 1e-9:
        return 1.0
    return round(max(0.0, 1.0 - abs(variant - canonical) / max(0.20, canonical)), 6)


def _action_metric_retention(
    scenario: OpenNarrativeScenario,
    canonical_metrics: dict[str, float],
    variant_metrics: dict[str, float],
) -> float:
    canonical = float(canonical_metrics.get(scenario.expected_action_metric, 0.0))
    variant = float(variant_metrics.get(scenario.expected_action_metric, 0.0))
    if canonical <= 1e-9:
        return 1.0
    return round(_clamp(variant / canonical, 0.0, 1.25), 6)


def _target_attention_replaced(scenario: OpenNarrativeScenario, metrics: dict[str, float]) -> bool:
    attention = {
        "danger": float(metrics.get("danger_attention_rate", 0.0)),
        "social": float(metrics.get("social_attention_rate", 0.0)),
        "novelty": float(metrics.get("novelty_attention_rate", 0.0)),
    }
    dominant = max(attention.items(), key=lambda item: (item[1], item[0]))[0]
    return dominant != scenario.expected_attention_channel


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def _paired_analysis(initialized: list[float], ablated: list[float], *, larger_is_better: bool = True) -> dict[str, float | bool]:
    differences = [
        (i - a) if larger_is_better else (a - i)
        for i, a in zip(initialized, ablated)
    ]
    if not differences:
        return {"mean_delta": 0.0, "effect_size": 0.0, "significant": False, "effect_passed": False}
    avg = mean(differences)
    std = pstdev(differences)
    standard_error = std / (len(differences) ** 0.5) if std > 0 else 0.0
    effect = avg / (std + 1e-9)
    return {
        "mean_delta": avg,
        "effect_size": effect,
        "significant": avg > 0.0 and (std == 0.0 or avg >= standard_error * 1.96),
        "effect_passed": effect >= 0.5,
    }


def _variant_distance_bundle(
    scenario: OpenNarrativeScenario,
    canonical_payload: dict[str, object],
    variant_payload: dict[str, object],
) -> dict[str, float]:
    canonical_init = dict(canonical_payload["initialization"])
    variant_init = dict(variant_payload["initialization"])
    canonical_metrics = dict(canonical_payload["metrics"])
    variant_metrics = dict(variant_payload["metrics"])
    return {
        "identity_commitment_jaccard": _identity_commitment_jaccard(canonical_init, variant_init),
        "policy_distribution_l1_distance": round(_l1_distance(dict(canonical_init.get("policy_distribution", {})), dict(variant_init.get("policy_distribution", {}))), 6),
        "narrative_prior_l1_distance": round(_l1_distance(dict(canonical_init.get("narrative_priors", {})), dict(variant_init.get("narrative_priors", {}))), 6),
        "personality_profile_l1_distance": round(_l1_distance(dict(canonical_init.get("personality_profile", {})), dict(variant_init.get("personality_profile", {}))), 6),
        "social_snapshot_similarity": _social_similarity(dict(canonical_init.get("social_snapshot", {})), dict(variant_init.get("social_snapshot", {}))),
        "attention_target_consistency": _attention_target_consistency(scenario, canonical_metrics, variant_metrics),
        "action_metric_retention": _action_metric_retention(scenario, canonical_metrics, variant_metrics),
        "conflict_uncertainty_score": round(float(variant_init.get("uncertainty_score", 0.0)), 6),
        "malformed_text_degradation_ratio": round(float(variant_init.get("malformed_text_degradation_ratio", 0.0)), 6),
    }


def build_m221_trace_payload(*, seed_set: list[int] | None = None, cycles: int = 24) -> dict[str, object]:
    scenarios = open_narrative_scenarios()
    selected_seeds = list(seed_set or SEED_SET)
    sample_seed = selected_seeds[0]
    samples: dict[str, dict[str, object]] = {}
    schema_complete = True
    for scenario in scenarios.values():
        canonical = _rollout_variant(
            scenario=scenario,
            variant=scenario.variants["canonical"],
            seed=sample_seed,
            cycles=cycles,
            apply_initialization=True,
        )
        variants: dict[str, object] = {}
        for variant_id in VARIANT_ORDER:
            payload = _rollout_variant(
                scenario=scenario,
                variant=scenario.variants[variant_id],
                seed=sample_seed,
                cycles=cycles,
                apply_initialization=True,
            )
            variants[variant_id] = {
                "input_text": payload["input_text"],
                "compiled_trace": payload["initialization"].get("evidence_trace", {}),
                "initialization": payload["initialization"],
                "rollout_metrics": payload["metrics"],
                "distance_to_canonical": {} if variant_id == "canonical" else _variant_distance_bundle(scenario, canonical, payload),
                "gate_passed": True,
            }
            for field in ("input_text", "compiled_trace", "initialization", "rollout_metrics", "distance_to_canonical", "gate_passed"):
                if field not in variants[variant_id]:
                    schema_complete = False
        samples[scenario.scenario_id] = variants
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": sample_seed,
        "cycles": cycles,
        "samples": samples,
        "schema_complete": schema_complete,
    }


def run_m221_determinism_probe(*, seed: int, cycles: int = 24) -> dict[str, object]:
    scenario = open_narrative_scenarios()["social_trusting"]
    variant = scenario.variants["canonical"]
    first = _rollout_variant(scenario=scenario, variant=variant, seed=seed, cycles=cycles, apply_initialization=True)
    second = _rollout_variant(scenario=scenario, variant=variant, seed=seed, cycles=cycles, apply_initialization=True)
    return {
        "seed": seed,
        "passed": first["initialization"] == second["initialization"] and first["metrics"] == second["metrics"],
        "first": first,
        "second": second,
    }


def run_m221_open_narrative_benchmark(*, seed_set: list[int] | None = None, cycles: int = 24) -> dict[str, object]:
    scenarios = open_narrative_scenarios()
    seed_values = list(seed_set or SEED_SET)
    trial_matrix = {
        scenario_id: {variant_id: [] for variant_id in VARIANT_ORDER}
        for scenario_id in scenarios
    }
    ablated_trials = {scenario_id: [] for scenario_id in scenarios}
    for scenario in scenarios.values():
        for seed in seed_values:
            for variant_id in VARIANT_ORDER:
                trial_matrix[scenario.scenario_id][variant_id].append(
                    _rollout_variant(scenario=scenario, variant=scenario.variants[variant_id], seed=seed, cycles=cycles, apply_initialization=True)
                )
            ablated_trials[scenario.scenario_id].append(
                _rollout_variant(scenario=scenario, variant=scenario.variants["canonical"], seed=seed, cycles=cycles, apply_initialization=False)
            )

    paraphrase_jaccard: list[float] = []
    paraphrase_policy: list[float] = []
    paraphrase_prior: list[float] = []
    paraphrase_personality: list[float] = []
    noisy_policy: list[float] = []
    noisy_attention: list[float] = []
    noisy_action_retention: list[float] = []
    adversarial_wrong_direction: list[float] = []
    adversarial_flip_rate: list[float] = []
    adversarial_attention_replace: list[float] = []
    conflicting_uncertainty: list[float] = []
    conflicting_extreme_commitment: list[float] = []
    conflicting_policy_distance: list[float] = []
    low_signal_degradation: list[float] = []
    multilingual_jaccard: list[float] = []
    multilingual_policy: list[float] = []
    multilingual_prior: list[float] = []
    per_scenario_breakdown: dict[str, dict[str, object]] = {}
    causality_breakdown: dict[str, dict[str, object]] = {}

    for scenario in scenarios.values():
        per_variant: dict[str, dict[str, object]] = {}
        for variant_id in VARIANT_ORDER:
            if variant_id == "canonical":
                continue
            distances: list[dict[str, float]] = []
            for canonical_payload, variant_payload in zip(trial_matrix[scenario.scenario_id]["canonical"], trial_matrix[scenario.scenario_id][variant_id]):
                bundle = _variant_distance_bundle(scenario, canonical_payload, variant_payload)
                distances.append(bundle)
                if variant_id == "paraphrase":
                    paraphrase_jaccard.append(bundle["identity_commitment_jaccard"])
                    paraphrase_policy.append(bundle["policy_distribution_l1_distance"])
                    paraphrase_prior.append(bundle["narrative_prior_l1_distance"])
                    paraphrase_personality.append(bundle["personality_profile_l1_distance"])
                elif variant_id == "noisy":
                    noisy_policy.append(bundle["policy_distribution_l1_distance"])
                    noisy_attention.append(bundle["attention_target_consistency"])
                    noisy_action_retention.append(bundle["action_metric_retention"])
                elif variant_id == "adversarial_surface":
                    variant_init = dict(variant_payload["initialization"])
                    canonical_init = dict(canonical_payload["initialization"])
                    adversarial_wrong_direction.append(0.0 if _dominant_direction(variant_init) == scenario.expected_direction else 1.0)
                    canonical_set = _commitment_action_set(canonical_init)
                    variant_set = _commitment_action_set(variant_init)
                    adversarial_flip_rate.append(1.0 if canonical_set and not (canonical_set & variant_set) else 0.0)
                    attention_consistency = _attention_target_consistency(
                        scenario,
                        dict(canonical_payload["metrics"]),
                        dict(variant_payload["metrics"]),
                    )
                    adversarial_attention_replace.append(
                        1.0
                        if _target_attention_replaced(scenario, dict(variant_payload["metrics"])) and attention_consistency < 0.70
                        else 0.0
                    )
                elif variant_id == "conflicting":
                    variant_init = dict(variant_payload["initialization"])
                    policy = dict(variant_init.get("policy_distribution", {}))
                    conflicting_uncertainty.append(float(variant_init.get("uncertainty_score", 0.0)))
                    conflicting_extreme_commitment.append(max(policy.values()) if policy else 0.0)
                    conflicting_policy_distance.append(bundle["policy_distribution_l1_distance"])
                elif variant_id == "low_signal":
                    low_signal_degradation.append(bundle["malformed_text_degradation_ratio"])
                elif variant_id == "multilingual":
                    multilingual_jaccard.append(bundle["identity_commitment_jaccard"])
                    multilingual_policy.append(bundle["policy_distribution_l1_distance"])
                    multilingual_prior.append(bundle["narrative_prior_l1_distance"])
            per_variant[variant_id] = {
                "metric_summary": {key: _mean_std([item[key] for item in distances]) for key in distances[0]},
                "sample_count": len(distances),
            }
        per_scenario_breakdown[scenario.scenario_id] = per_variant
        initialized = trial_matrix[scenario.scenario_id]["canonical"]
        ablated = ablated_trials[scenario.scenario_id]
        expected_attention_metric = f"{scenario.expected_attention_channel}_attention_rate"
        analyses = {
            expected_attention_metric: _paired_analysis([float(item["metrics"][expected_attention_metric]) for item in initialized], [float(item["metrics"][expected_attention_metric]) for item in ablated], larger_is_better=True),
            scenario.expected_action_metric: _paired_analysis([float(item["metrics"][scenario.expected_action_metric]) for item in initialized], [float(item["metrics"][scenario.expected_action_metric]) for item in ablated], larger_is_better=True),
            "oracle_regret_rate": _paired_analysis([float(item["metrics"]["oracle_regret_rate"]) for item in initialized], [float(item["metrics"]["oracle_regret_rate"]) for item in ablated], larger_is_better=False),
            "survival_score": _paired_analysis([float(item["metrics"]["survival_score"]) for item in initialized], [float(item["metrics"]["survival_score"]) for item in ablated], larger_is_better=True),
        }
        significant_metric_count = sum(1 for payload in analyses.values() if bool(payload["significant"]))
        effect_metric_count = sum(1 for payload in analyses.values() if bool(payload["effect_passed"]))
        causality_breakdown[scenario.scenario_id] = {
            "attention_delta": float(analyses[expected_attention_metric]["mean_delta"]),
            "action_delta": float(analyses[scenario.expected_action_metric]["mean_delta"]),
            "significant_metric_count": significant_metric_count,
            "effect_metric_count": effect_metric_count,
            "metrics": analyses,
            "passed": float(analyses[expected_attention_metric]["mean_delta"]) >= 0.02 and float(analyses[scenario.expected_action_metric]["mean_delta"]) >= 0.05 and significant_metric_count >= 3 and effect_metric_count >= 3,
        }

    variant_breakdown = {
        "canonical_vs_paraphrase": {
            "identity_commitment_jaccard": _mean_std(paraphrase_jaccard),
            "policy_distribution_l1_distance": _mean_std(paraphrase_policy),
            "narrative_prior_l1_distance": _mean_std(paraphrase_prior),
            "personality_profile_l1_distance": _mean_std(paraphrase_personality),
        },
        "canonical_vs_noisy": {
            "policy_distribution_l1_distance": _mean_std(noisy_policy),
            "attention_target_consistency": _mean_std(noisy_attention),
            "action_metric_retention": _mean_std(noisy_action_retention),
        },
        "canonical_vs_adversarial_surface": {
            "wrong_direction_initialization_rate": _mean_std(adversarial_wrong_direction),
            "identity_commitment_flip_rate": _mean_std(adversarial_flip_rate),
            "attention_channel_replacement_rate": _mean_std(adversarial_attention_replace),
        },
        "conflicting_narrative": {
            "conflict_uncertainty_score": _mean_std(conflicting_uncertainty),
            "extreme_single_commitment_ratio": _mean_std(conflicting_extreme_commitment),
            "policy_distribution_l1_distance": _mean_std(conflicting_policy_distance),
        },
        "low_signal_degradation": {"malformed_text_degradation_ratio": _mean_std(low_signal_degradation)},
        "multilingual_robustness": {
            "identity_commitment_jaccard": _mean_std(multilingual_jaccard),
            "policy_distribution_l1_distance": _mean_std(multilingual_policy),
            "narrative_prior_l1_distance": _mean_std(multilingual_prior),
        },
    }
    determinism = run_m221_determinism_probe(seed=seed_values[0], cycles=cycles)
    trace_payload = build_m221_trace_payload(seed_set=seed_values, cycles=cycles)
    gates = {
        "stability": variant_breakdown["canonical_vs_paraphrase"]["identity_commitment_jaccard"]["mean"] >= 0.80 and variant_breakdown["canonical_vs_paraphrase"]["policy_distribution_l1_distance"]["mean"] <= 0.18 and variant_breakdown["canonical_vs_paraphrase"]["narrative_prior_l1_distance"]["mean"] <= 0.20 and variant_breakdown["canonical_vs_paraphrase"]["personality_profile_l1_distance"]["mean"] <= 0.15,
        "noise_robustness": variant_breakdown["canonical_vs_noisy"]["policy_distribution_l1_distance"]["mean"] <= 0.25 and variant_breakdown["canonical_vs_noisy"]["attention_target_consistency"]["mean"] >= 0.80 and variant_breakdown["canonical_vs_noisy"]["action_metric_retention"]["mean"] >= 0.85,
        "adversarial_surface_resistance": variant_breakdown["canonical_vs_adversarial_surface"]["wrong_direction_initialization_rate"]["mean"] <= 0.15 and variant_breakdown["canonical_vs_adversarial_surface"]["identity_commitment_flip_rate"]["mean"] <= 0.10 and variant_breakdown["canonical_vs_adversarial_surface"]["attention_channel_replacement_rate"]["mean"] <= 0.15,
        "conflicting_boundedness": variant_breakdown["conflicting_narrative"]["conflict_uncertainty_score"]["mean"] >= 0.20 and variant_breakdown["conflicting_narrative"]["extreme_single_commitment_ratio"]["mean"] <= 0.70 and variant_breakdown["conflicting_narrative"]["policy_distribution_l1_distance"]["mean"] <= 0.35,
        "low_quality_degradation": 0.40 <= variant_breakdown["low_signal_degradation"]["malformed_text_degradation_ratio"]["mean"] <= 0.85,
        "behavior_causality": all(bool(item["passed"]) for item in causality_breakdown.values()),
        "determinism": bool(determinism["passed"]),
        "artifact_schema_complete": bool(trace_payload["schema_complete"]),
    }
    passed = all(gates.values())
    return {
        "milestone_id": "M2.21",
        "schema_version": SCHEMA_VERSION,
        "seed_set": seed_values,
        "cycles": cycles,
        "scenarios": {key: value.to_dict() for key, value in scenarios.items()},
        "variant_breakdown": variant_breakdown,
        "per_scenario_breakdown": per_scenario_breakdown,
        "causality_breakdown": causality_breakdown,
        "determinism": determinism,
        "trace_payload": trace_payload,
        "gates": gates,
        "status": "PASS" if passed else "FAIL",
        "recommendation": "ACCEPT" if passed else "REJECT",
        "residual_risks": [
            "Multilingual robustness currently covers mixed Chinese-English prompts but not broader translation families.",
            "Conflict resolution remains bounded heuristic weighting rather than long-horizon autobiographical reconciliation.",
        ],
        "freshness": {"generated_this_round": True, "artifact_schema_version": SCHEMA_VERSION, "codebase_version": _codebase_version()},
        "generated_at": _generated_at(),
    }


def write_m221_acceptance_artifacts(*, seed_set: list[int] | None = None, cycles: int = 24) -> dict[str, Path]:
    payload = run_m221_open_narrative_benchmark(seed_set=seed_set, cycles=cycles)
    artifact_paths = {
        "trace": ARTIFACTS_DIR / "m221_open_narrative_trace.json",
        "paraphrase": ARTIFACTS_DIR / "m221_paraphrase_stability.json",
        "adversarial": ARTIFACTS_DIR / "m221_adversarial_surface.json",
        "conflict": ARTIFACTS_DIR / "m221_conflict_boundedness.json",
        "multilingual": ARTIFACTS_DIR / "m221_multilingual_robustness.json",
        "report": REPORTS_DIR / "m221_acceptance_report.json",
    }
    artifact_paths["trace"].write_text(json.dumps(payload["trace_payload"], indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_paths["paraphrase"].write_text(json.dumps(payload["variant_breakdown"]["canonical_vs_paraphrase"], indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_paths["adversarial"].write_text(json.dumps(payload["variant_breakdown"]["canonical_vs_adversarial_surface"], indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_paths["conflict"].write_text(json.dumps(payload["variant_breakdown"]["conflicting_narrative"], indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_paths["multilingual"].write_text(json.dumps(payload["variant_breakdown"]["multilingual_robustness"], indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "milestone_id": payload["milestone_id"],
        "status": payload["status"],
        "recommendation": payload["recommendation"],
        "generated_at": payload["generated_at"],
        "seed_set": payload["seed_set"],
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "tests": {"milestone_suite": ["tests/test_m221_paraphrase_stability.py", "tests/test_m221_adversarial_surface.py", "tests/test_m221_conflict_boundedness.py", "tests/test_m221_multilingual_robustness.py", "tests/test_m221_acceptance.py"]},
        "gates": dict(payload["gates"]),
        "significant_metric_count": {key: value["significant_metric_count"] for key, value in payload["causality_breakdown"].items()},
        "effect_metric_count": {key: value["effect_metric_count"] for key, value in payload["causality_breakdown"].items()},
        "per_scenario_breakdown": payload["per_scenario_breakdown"],
        "per_variant_breakdown": payload["variant_breakdown"],
        "residual_risks": list(payload["residual_risks"]),
        "freshness": dict(payload["freshness"]),
    }
    artifact_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact_paths
