from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from statistics import mean
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from ..agent import SegmentAgent
    from .lifecycle import ImplantationConfig

from .utils import clamp as _clamp


# ── Big Five → SlowTraitState ──────────────────────────────────────────────

# Default linear coefficients mapping OCEAN (0–1) to the five SlowTraitState dims.
# Each tuple: (O_weight, C_weight, E_weight, A_weight, N_weight, intercept)
# The intercept centers the trait at 0.5 when all Big Five dims are 0.5 (neutral).
_BIG_FIVE_COEFFICIENTS: dict[str, tuple[float, float, float, float, float, float]] = {
    #                  O      C      E      A      N     intercept
    "caution_bias":       ( 0.00, 0.15, -0.30, 0.00, 0.50, 0.325),
    "threat_sensitivity": ( 0.00, 0.00, -0.15, -0.25, 0.60, 0.400),
    "trust_stance":       ( 0.10, 0.00, 0.15, 0.45, -0.35, 0.325),
    "exploration_posture":( 0.60, -0.10, 0.25, 0.00, 0.00, 0.125),
    "social_approach":    ( 0.20, 0.00, 0.60, 0.15, -0.10, 0.075),
}


def big_five_to_slow_traits(
    openness: float = 0.5,
    conscientiousness: float = 0.5,
    extraversion: float = 0.5,
    agreeableness: float = 0.5,
    neuroticism: float = 0.5,
    *,
    coefficients: dict[str, tuple[float, float, float, float, float, float]] | None = None,
) -> dict[str, float]:
    """Map Big Five (OCEAN) scores to SlowTraitState dimensions.

    All Big Five inputs are expected in 0–1 range (normalized).  The default
    coefficients encode reasonable personality-psychology priors:

    * **caution_bias** ← Neuroticism (+) + low Extraversion (+)
    * **threat_sensitivity** ← Neuroticism (++) + low Agreeableness (+)
    * **trust_stance** ← Agreeableness (++) + low Neuroticism (+)
    * **exploration_posture** ← Openness (++) + Extraversion (+)
    * **social_approach** ← Extraversion (++) + Openness (+)

    Pass a custom ``coefficients`` dict to override the mapping.
    """
    coeffs = coefficients if coefficients is not None else _BIG_FIVE_COEFFICIENTS
    big5: dict[str, float] = {
        "O": float(openness),
        "C": float(conscientiousness),
        "E": float(extraversion),
        "A": float(agreeableness),
        "N": float(neuroticism),
    }
    out: dict[str, float] = {}
    for trait_key, (wo, wc, we, wa, wn, intercept) in coeffs.items():
        raw = (
            wo * big5["O"]
            + wc * big5["C"]
            + we * big5["E"]
            + wa * big5["A"]
            + wn * big5["N"]
            + intercept
        )
        out[trait_key] = round(_clamp(raw, 0.05, 0.95), 6)
    return out


@dataclass(slots=True)
class PersonalitySnapshot:
    sleep_cycle: int
    tick: int
    slow_traits: dict[str, float]
    narrative_priors: dict[str, float]
    precision_debt: dict[str, float]
    defense_distribution: dict[str, int]
    memory_stats: dict[str, int]
    maturity_distance: float = 0.0
    #: Dialogue-bridge prediction ledger outcomes since previous personality snapshot (monitoring).
    prediction_verification: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(snapshot: PersonalitySnapshot) -> list[float]:
    ordered: list[float] = []
    for bucket in (
        snapshot.slow_traits,
        snapshot.narrative_priors,
        snapshot.precision_debt,
    ):
        ordered.extend(float(bucket[key]) for key in sorted(bucket))
    ordered.extend(float(snapshot.memory_stats.get(key, 0)) for key in sorted(snapshot.memory_stats))
    ordered.extend(float(snapshot.defense_distribution.get(key, 0)) for key in sorted(snapshot.defense_distribution))
    return ordered


def personality_distance(a: PersonalitySnapshot, b: PersonalitySnapshot) -> float:
    left = _vector(a)
    right = _vector(b)
    if not left or not right:
        return 0.0
    numerator = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, numerator / (norm_left * norm_right)))
    return round(max(0.0, 1.0 - cosine), 6)


def personality_trait_distance(a: PersonalitySnapshot, b: PersonalitySnapshot) -> float:
    """Cosine distance using ONLY slow_traits (deconfounded from memory growth).

    Unlike personality_distance() which mixes slow_traits, narrative_priors,
    precision_debt, memory_stats, and defense_distribution, this isolates the
    true personality signal from slowly-changing trait dimensions.
    """
    keys = sorted(set(a.slow_traits) & set(b.slow_traits))
    if not keys:
        return 0.0
    left = [float(a.slow_traits[k]) for k in keys]
    right = [float(b.slow_traits[k]) for k in keys]
    numerator = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return 0.0
    cosine_trait = max(-1.0, min(1.0, numerator / (norm_left * norm_right)))
    return round(max(0.0, 1.0 - cosine_trait), 6)


def capture_personality_snapshot(
    agent: "SegmentAgent",
    sleep_cycle: int,
    *,
    prediction_verification: Mapping[str, object] | None = None,
) -> PersonalitySnapshot:
    precision_report = agent.precision_manipulator.to_dict()
    debts = precision_report.get("channel_debts", {})
    defense_history = precision_report.get("strategy_history", [])
    defense_distribution: dict[str, int] = {}
    for item in defense_history[-128:]:
        strategy = str(item.get("strategy", ""))
        if not strategy:
            continue
        defense_distribution[strategy] = defense_distribution.get(strategy, 0) + 1
    memory_stats = {
        "episodic": len(agent.long_term_memory.episodes),
        "semantic": len(agent.long_term_memory.semantic_schemas),
        "procedural": len(agent.action_history),
    }
    pv = dict(prediction_verification) if prediction_verification else {}
    return PersonalitySnapshot(
        sleep_cycle=int(sleep_cycle),
        tick=int(agent.cycle),
        slow_traits=agent.slow_variable_learner.state.traits.to_dict(),
        narrative_priors=agent.self_model.narrative_priors.to_dict(),
        precision_debt={str(k): float(v) for k, v in dict(debts).items()},
        defense_distribution=defense_distribution,
        memory_stats=memory_stats,
        maturity_distance=0.0,
        prediction_verification=pv,
    )


def is_mature(
    snapshots: list[PersonalitySnapshot],
    config: "ImplantationConfig",
) -> bool:
    if len(snapshots) < max(2, int(config.maturity_window)):
        return False
    distances = [item.maturity_distance for item in snapshots[-int(config.maturity_window) :]]
    return all(float(distance) < float(config.maturity_threshold) for distance in distances)


def maturity_report(
    snapshots: list[PersonalitySnapshot],
    *,
    threshold: float = 0.02,
    window: int = 3,
) -> dict[str, object]:
    if not snapshots:
        return {"snapshots": 0, "matured": False}
    distances = [item.maturity_distance for item in snapshots]
    required_window = max(2, int(window))
    mature_index = None
    if len(distances) >= required_window:
        for idx in range(required_window - 1, len(distances)):
            candidate = distances[idx - required_window + 1 : idx + 1]
            if all(float(value) < float(threshold) for value in candidate):
                mature_index = idx
                break
    return {
        "snapshots": len(snapshots),
        "mean_distance": round(mean(distances), 6),
        "max_distance": round(max(distances), 6),
        "min_distance": round(min(distances), 6),
        "matured": mature_index is not None,
        "maturity_snapshot_index": mature_index,
        "threshold": round(float(threshold), 6),
        "window": required_window,
        "distance_trace": [round(float(item), 6) for item in distances],
        "final_slow_traits": snapshots[-1].slow_traits,
        "final_narrative_priors": snapshots[-1].narrative_priors,
    }
