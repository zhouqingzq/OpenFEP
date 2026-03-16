from __future__ import annotations

from evals.m210_validation_evaluation import build_m210_audit_payload
from segmentum.m210_benchmarks import (
    run_longitudinal_stability,
    run_personality_validation,
    run_profile_trial,
    summarize_profile_behaviors,
)


def test_same_profile_same_seed_is_stable() -> None:
    first = run_profile_trial(profile_name="threat_sensitive", seed=101, cycles_per_world=20)
    second = run_profile_trial(profile_name="threat_sensitive", seed=101, cycles_per_world=20)
    assert first["aggregate_metrics"] == second["aggregate_metrics"]
    assert first["final_identity"] == second["final_identity"]


def test_longitudinal_stability_rejects_obvious_collapse() -> None:
    result = run_longitudinal_stability(seed=91, cycles_per_world=24, repeats=2)
    assert result["acceptance"]["profiles_passing"] >= 4
    assert "social_approach" in result["profiles"]
    for payload in result["profiles"].values():
        checks = payload["checks"]
        assert checks["action_entropy_floor_met"] is True
        assert checks["narrative_bias_persists"] is True


def test_strict_audit_accepts_current_round_evidence() -> None:
    personality = run_personality_validation(seed=44, cycles_per_world=18, repeats=3)
    stability = run_longitudinal_stability(seed=91, cycles_per_world=24, repeats=2)
    summary = summarize_profile_behaviors(personality, stability)
    audit = build_m210_audit_payload(
        personality_validation=personality,
        longitudinal_stability=stability,
        profile_summary=summary,
    )
    assert audit["gate_results"]["statistical_support"] is True
    assert audit["gate_results"]["artifact_freshness"] is True
    assert audit["final_recommendation"]["passed"] is True
