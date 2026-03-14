from __future__ import annotations

import random

from evals.m2_followup_repair import (
    evaluate_episode_gating,
    evaluate_mixed_attribution,
    evaluate_value_stability,
    EvidenceWriter,
)
from segmentum.agent import SegmentAgent
from segmentum.counterfactual import CounterfactualInsight, InsightAbsorber
from segmentum.environment import Observation
from segmentum.self_model import IdentityNarrative, PreferredPolicies, build_default_self_model
from segmentum.world_model import GenerativeWorldModel


def _dangerous_observation() -> dict[str, float]:
    return {
        "food": 0.12,
        "danger": 0.82,
        "novelty": 0.22,
        "shelter": 0.10,
        "temperature": 0.46,
        "social": 0.18,
    }


def _dangerous_prediction() -> dict[str, float]:
    return {
        "food": 0.60,
        "danger": 0.20,
        "novelty": 0.40,
        "shelter": 0.40,
        "temperature": 0.50,
        "social": 0.30,
    }


def _dangerous_errors() -> dict[str, float]:
    obs = _dangerous_observation()
    pred = _dangerous_prediction()
    return {key: obs[key] - pred[key] for key in obs}


def test_trivial_events_are_suppressed_from_episodic_store() -> None:
    memory = SegmentAgent(rng=random.Random(1)).long_term_memory
    decision = memory.maybe_store_episode(
        cycle=1,
        observation={"food": 0.50, "danger": 0.10, "novelty": 0.30, "shelter": 0.45, "temperature": 0.50, "social": 0.25},
        prediction={"food": 0.49, "danger": 0.11, "novelty": 0.29, "shelter": 0.44, "temperature": 0.50, "social": 0.24},
        errors={"food": 0.01, "danger": -0.01, "novelty": 0.01, "shelter": 0.01, "temperature": 0.0, "social": 0.01},
        action="rest",
        outcome={"energy_delta": 0.0, "stress_delta": 0.0, "fatigue_delta": 0.0, "temperature_delta": 0.0, "free_energy_drop": 0.01},
        body_state={"energy": 0.8, "stress": 0.1, "fatigue": 0.1, "temperature": 0.5},
    )
    assert not decision.episode_created
    assert len(memory.episodes) == 0


def test_high_surprise_high_value_events_are_preserved() -> None:
    memory = SegmentAgent(rng=random.Random(2)).long_term_memory
    decision = memory.maybe_store_episode(
        cycle=1,
        observation=_dangerous_observation(),
        prediction=_dangerous_prediction(),
        errors=_dangerous_errors(),
        action="forage",
        outcome={"energy_delta": -0.10, "stress_delta": 0.28, "fatigue_delta": 0.18, "temperature_delta": 0.02, "free_energy_drop": -0.45},
        body_state={"energy": 0.18, "stress": 0.8, "fatigue": 0.3, "temperature": 0.46},
    )
    assert decision.episode_created
    assert len(memory.episodes) == 1
    assert memory.episodes[0]["identity_critical"] is True


def test_near_duplicate_episodes_merge_support_instead_of_flattening_store() -> None:
    memory = SegmentAgent(rng=random.Random(3)).long_term_memory
    kwargs = {
        "observation": _dangerous_observation(),
        "prediction": _dangerous_prediction(),
        "errors": _dangerous_errors(),
        "action": "forage",
        "outcome": {"energy_delta": -0.10, "stress_delta": 0.28, "fatigue_delta": 0.18, "temperature_delta": 0.02, "free_energy_drop": -0.45},
        "body_state": {"energy": 0.18, "stress": 0.8, "fatigue": 0.3, "temperature": 0.46},
    }
    first = memory.maybe_store_episode(cycle=1, **kwargs)
    second = memory.maybe_store_episode(cycle=2, **kwargs)
    assert first.episode_created
    assert not second.episode_created
    assert second.support_delta == 1
    assert len(memory.episodes) == 1
    assert memory.episodes[0]["support_count"] >= 2


def test_narrative_claims_include_support_and_contradiction_evidence() -> None:
    model = build_default_self_model()
    model.preferred_policies = PreferredPolicies(
        dominant_strategy="expected_free_energy",
        action_distribution={"forage": 0.7, "hide": 0.3},
        risk_profile="risk_averse",
        last_updated_tick=5,
    )
    model.identity_narrative = IdentityNarrative(core_identity="I am generally cautious under pressure.")
    result = model.evaluate_narrative_contradictions(
        episodic_memory=[
            {"episode_id": "ep-1", "timestamp": 1, "action_taken": "forage", "predicted_outcome": "survival_threat", "risk": 3.0},
            {"episode_id": "ep-2", "timestamp": 2, "action_taken": "hide", "predicted_outcome": "neutral", "risk": 0.3},
        ],
        decision_history=[{"tick": 1, "action": "forage", "risk": 2.0}],
        current_tick=6,
    )
    claims = result["claims"]
    assert any(claim["supported_by"] for claim in claims)
    assert any(claim["contradicted_by"] for claim in claims)


def test_contradiction_checker_catches_trait_value_and_capability_mismatch() -> None:
    model = build_default_self_model()
    model.preferred_policies = PreferredPolicies(
        dominant_strategy="expected_free_energy",
        action_distribution={"forage": 0.8},
        risk_profile="risk_averse",
        last_updated_tick=5,
    )
    model.identity_narrative = IdentityNarrative(core_identity="I am generally cautious under pressure.")
    result = model.evaluate_narrative_contradictions(
        episodic_memory=[
            {"episode_id": "ep-1", "timestamp": 1, "action_taken": "forage", "predicted_outcome": "survival_threat", "risk": 3.4},
            {"episode_id": "ep-2", "timestamp": 2, "action_taken": "forage", "predicted_outcome": "integrity_loss", "risk": 2.5},
        ],
        decision_history=[{"tick": 1, "action": "forage", "risk": 2.2}],
        current_tick=6,
    )
    by_type = {claim["claim_type"]: claim for claim in result["claims"]}
    assert by_type["trait"]["contradict_count"] > 0
    assert by_type["value"]["contradict_count"] > 0
    assert by_type["capability"]["contradict_count"] > 0


def test_counterfactual_adoption_honors_cooling_gate() -> None:
    world_model = GenerativeWorldModel()
    absorber = InsightAbsorber()
    insight_a = CounterfactualInsight(
        source_episode_cycle=1,
        original_action="forage",
        counterfactual_action="hide",
        original_efe=5.0,
        counterfactual_efe=1.0,
        efe_delta=-4.0,
        confidence=0.66,
        state_context={
            "observation": _dangerous_observation(),
            "body_state": {"energy": 0.2, "stress": 0.8, "fatigue": 0.3, "temperature": 0.46},
        },
        cluster_id=0,
        timestamp=10,
    )
    insight_b = CounterfactualInsight(
        source_episode_cycle=2,
        original_action="forage",
        counterfactual_action="hide",
        original_efe=5.1,
        counterfactual_efe=1.1,
        efe_delta=-4.0,
        confidence=0.68,
        state_context={
            "observation": {"food": 0.13, "danger": 0.8, "novelty": 0.2, "shelter": 0.11, "temperature": 0.46, "social": 0.17},
            "body_state": {"energy": 0.19, "stress": 0.79, "fatigue": 0.28, "temperature": 0.46},
        },
        cluster_id=0,
        timestamp=11,
    )
    first = absorber.absorb([insight_a], world_model)
    second = absorber.absorb([insight_b], world_model)
    assert first == 0
    assert second == 1
    assert any(entry.get("type") == "buffered_candidate" for entry in absorber.log)
    assert any(entry.get("type") == "candidate_review" and entry.get("passed") for entry in absorber.log)
    assert world_model.get_policy_bias(0, "hide") > 0.0
    assert world_model.get_preference_penalty(0, "forage") < 0.0


def test_mixed_attribution_keeps_secondary_origin_and_causal_chain() -> None:
    result = evaluate_mixed_attribution(EvidenceWriter())
    assert result["secondary_accuracy"] > 0.0
    assert all(len(case["causal_chain"]) >= 3 for case in result["cases"])


def test_episode_gating_covers_multiple_memory_families_and_lifecycle_events() -> None:
    result = evaluate_episode_gating(EvidenceWriter())
    assert result["high_event_created"]
    assert result["resource_event_created"]
    assert result["social_event_created"]
    assert not result["trivial_event_created"]
    assert result["duplicate_merged"]
    assert result["family_coverage_count"] >= 3
    assert "created" in result["lifecycle_event_counts"]
    assert "support_merged" in result["lifecycle_event_counts"]


def test_retrieval_influence_and_benefit_are_distinguishable() -> None:
    from evals.m2_followup_repair import evaluate_memory_utility

    result = evaluate_memory_utility(EvidenceWriter())
    assert result["retrieval_influence_rate"] >= result["retrieval_benefit_rate"]


def test_value_hierarchy_remains_stable_under_stronger_conflicts() -> None:
    result = evaluate_value_stability(EvidenceWriter())
    assert result["VCUS"] >= 0.5
    assert result["value_flip_rate"] <= 0.5


def test_counterfactual_graduation_produces_perturbed_benefit_signal() -> None:
    from evals.m2_followup_repair import evaluate_counterfactual_generalization

    result = evaluate_counterfactual_generalization(EvidenceWriter())
    assert result["absorbed_count"] >= 1
    assert result["influence_rate"] >= 0.5
    assert result["generalization_rate"] >= 0.5
    assert any(detail["beneficial"] for detail in result["details"])
