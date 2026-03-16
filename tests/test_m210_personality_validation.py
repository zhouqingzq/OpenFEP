from __future__ import annotations

from pathlib import Path

from scripts.generate_m210_acceptance_artifacts import generate_artifacts
from segmentum.m210_benchmarks import (
    profile_protocols,
    run_personality_validation,
    run_profile_trial,
)
from segmentum.m28_benchmarks import anova


def test_profile_protocol_factory_is_deterministic() -> None:
    protocols_a = {
        name: protocol.to_dict()
        for name, protocol in profile_protocols().items()
    }
    protocols_b = {
        name: protocol.to_dict()
        for name, protocol in profile_protocols().items()
    }
    assert protocols_a == protocols_b
    assert len(protocols_a) == 5


def test_profile_trial_metrics_are_deterministic_for_same_seed() -> None:
    first = run_profile_trial(profile_name="social_approach", seed=77, cycles_per_world=14)
    second = run_profile_trial(profile_name="social_approach", seed=77, cycles_per_world=14)
    assert first["aggregate_metrics"] == second["aggregate_metrics"]
    assert first["final_personality_profile"] == second["final_personality_profile"]


def test_anova_helper_matches_expected_group_separation() -> None:
    result = anova(
        {
            "a": [0.10, 0.12, 0.11],
            "b": [0.52, 0.55, 0.57],
            "c": [0.89, 0.91, 0.93],
        }
    )
    assert result["f_statistic"] > 100.0
    assert result["p_value"] < 0.001
    assert result["eta_squared"] > 0.95


def test_m210_personality_validation_detects_group_differences() -> None:
    result = run_personality_validation(seed=44, cycles_per_world=18, repeats=3)
    acceptance = result["acceptance"]
    assert acceptance["passed"] is True
    assert len(acceptance["significant_metrics"]) >= 3
    assert len(acceptance["effect_metrics"]) >= 2


def test_generate_m210_acceptance_artifacts_writes_required_outputs() -> None:
    written = generate_artifacts(
        validation_cycles_per_world=18,
        stability_cycles_per_world=24,
        validation_repeats=3,
        stability_repeats=2,
    )
    required_keys = {
        "m210_personality_anova",
        "m210_longitudinal_stability",
        "m210_profile_behavior_summary",
        "m210_audit_summary",
        "m210_audit_report",
    }
    assert required_keys == set(written)
    for path in written.values():
        assert Path(path).exists()
