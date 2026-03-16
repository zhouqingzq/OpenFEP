from __future__ import annotations

from segmentum.m28_benchmarks import run_personality_anova


def test_personality_anova_shows_statistically_meaningful_differences() -> None:
    result = run_personality_anova(seed=42, cycles=36, repeats=2)
    passing = 0
    for metric_name, analysis in result["anova"].items():
        if analysis["p_value"] < 0.05 and analysis["eta_squared"] >= 0.06:
            passing += 1
    assert passing >= 3
