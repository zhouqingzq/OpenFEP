"""M5.5 cross-context stability: scenario battery, conductor, analysis, and intent probe."""

from __future__ import annotations

from .analysis import (
    CrossContextReport,
    adaptation_envelope,
    analyze_cross_context,
    behavioral_adaptation,
    compare_adaptation_to_real,
    personality_consistency_score,
    state_distance_decomposition,
    within_vs_between_retrieval,
)
from .battery import SCENARIO_BATTERY, ScenarioSpec, get_scenario
from .conductor import ScenarioConductor, ScenarioResult
from .intent_probe import probe_intent_precision

__all__ = [
    "ScenarioSpec",
    "ScenarioResult",
    "ScenarioConductor",
    "SCENARIO_BATTERY",
    "get_scenario",
    "CrossContextReport",
    "analyze_cross_context",
    "personality_consistency_score",
    "adaptation_envelope",
    "behavioral_adaptation",
    "compare_adaptation_to_real",
    "state_distance_decomposition",
    "within_vs_between_retrieval",
    "probe_intent_precision",
]
