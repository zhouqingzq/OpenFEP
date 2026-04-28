"""M5.5 cross-context analysis: consistency scores, adaptation envelopes, diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev

from ..maturity import personality_distance, personality_trait_distance
from .conductor import ScenarioResult

_SLOW_TRAIT_KEYS: tuple[str, ...] = (
    "caution_bias",
    "threat_sensitivity",
    "trust_stance",
    "exploration_posture",
    "social_approach",
)


@dataclass
class CrossContextReport:
    agent_uid: int
    per_scenario: dict[str, ScenarioResult]
    personality_consistency: float
    adaptation_envelope: dict[str, float]
    real_adaptation_envelope: dict[str, float] | None = None
    adaptation_l2_to_real: float | None = None
    anomalies: list[dict[str, object]] = field(default_factory=list)
    conclusion: str = ""


# ── core metrics ─────────────────────────────────────────────────────────


def personality_consistency_score(results: list[ScenarioResult]) -> float:
    """Mean pairwise cosine similarity of post_snapshots using deconfounded slow_traits.

    Uses personality_trait_distance (slow_traits only) to avoid confounding
    from growing episodic memory counts that inflate similarity scores.
    """
    snapshots = [r.post_snapshot for r in results if r.post_snapshot is not None]
    n = len(snapshots)
    if n < 2:
        return 1.0
    similarities: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = personality_trait_distance(snapshots[i], snapshots[j])
            similarities.append(1.0 - dist)
    return round(mean(similarities), 6) if similarities else 1.0


def adaptation_envelope(results: list[ScenarioResult]) -> dict[str, float]:
    """Per-SlowTrait standard deviation across scenarios.

    A healthy agent adapts somewhat to context (std > 0.01).
    Zero variance = rigidity; very large variance = instability.
    """
    envelope: dict[str, float] = {}
    for key in _SLOW_TRAIT_KEYS:
        vals: list[float] = []
        for r in results:
            if r.post_snapshot is not None and key in r.post_snapshot.slow_traits:
                vals.append(float(r.post_snapshot.slow_traits[key]))
        envelope[key] = round(stdev(vals), 6) if len(vals) >= 2 else 0.0
    return envelope


def behavioral_adaptation(results: list[ScenarioResult]) -> dict[str, float]:
    """Per-scenario behavioral variation: action distribution entropy and strategy entropy.

    Returns a dict with:
    - action_variation: mean Jensen-Shannon distance between scenario action distributions
    - strategy_variation: mean JS distance between scenario strategy distributions
    - nonzero_action_dims: count of actions with std > 0.01 across scenarios
    - nonzero_strategy_dims: count of strategies with std > 0.01 across scenarios

    A healthy agent shows behavioral adaptation across contexts (nonzero > 0)
    while maintaining personality consistency.
    """
    from ..actions import DIALOGUE_ACTION_NAMES, DIALOGUE_ACTION_STRATEGY_MAP

    n = len(results)
    if n < 2:
        return {"action_variation": 0.0, "strategy_variation": 0.0, "nonzero_action_dims": 0, "nonzero_strategy_dims": 0}

    # Action distribution per scenario (normalized)
    action_vecs: list[dict[str, float]] = []
    for r in results:
        total = sum(r.action_distribution.values()) or 1
        action_vecs.append({a: r.action_distribution.get(a, 0) / total for a in DIALOGUE_ACTION_NAMES})

    # Strategy distribution per scenario (normalized)
    strategies = sorted(set(DIALOGUE_ACTION_STRATEGY_MAP.values()))
    strategy_vecs: list[dict[str, float]] = []
    for r in results:
        total = sum(r.strategy_distribution.values()) or 1
        strategy_vecs.append({s: r.strategy_distribution.get(s, 0) / total for s in strategies})

    # Pairwise JS distances
    action_js: list[float] = []
    strategy_js: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            action_js.append(_js_distance(action_vecs[i], action_vecs[j]))
            strategy_js.append(_js_distance(strategy_vecs[i], strategy_vecs[j]))

    # Per-action std across scenarios
    nonzero_actions = 0
    for a in DIALOGUE_ACTION_NAMES:
        vals = [v[a] for v in action_vecs]
        if stdev(vals) > 0.01 if len(vals) >= 2 else False:
            nonzero_actions += 1

    nonzero_strategies = 0
    for s in strategies:
        vals = [v[s] for v in strategy_vecs]
        if stdev(vals) > 0.01 if len(vals) >= 2 else False:
            nonzero_strategies += 1

    return {
        "action_variation": round(mean(action_js), 6) if action_js else 0.0,
        "strategy_variation": round(mean(strategy_js), 6) if strategy_js else 0.0,
        "nonzero_action_dims": nonzero_actions,
        "nonzero_strategy_dims": nonzero_strategies,
    }


def _js_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon distance between two discrete distributions."""
    keys = sorted(set(p) | set(q))
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2.0 for k in keys}
    kl_pm = sum(p.get(k, 0.0) * math.log((p.get(k, 0.0) + 1e-12) / (m[k] + 1e-12)) for k in keys if p.get(k, 0.0) > 0)
    kl_qm = sum(q.get(k, 0.0) * math.log((q.get(k, 0.0) + 1e-12) / (m[k] + 1e-12)) for k in keys if q.get(k, 0.0) > 0)
    js = (kl_pm + kl_qm) / 2.0
    return round(math.sqrt(max(0.0, js)), 6)


def compare_adaptation_to_real(
    agent_envelope: dict[str, float],
    real_envelope: dict[str, float],
) -> float:
    """L2 distance between agent and real-human adaptation amplitudes."""
    keys = set(agent_envelope) & set(real_envelope)
    if not keys:
        return float("inf")
    squared = sum((agent_envelope[k] - real_envelope[k]) ** 2 for k in keys)
    return round(squared**0.5, 6)


# ── state-distance decomposition ──────────────────────────────────────────


def state_distance_decomposition(results: list[ScenarioResult]) -> dict[str, object]:
    """Decompose state variance into within-scenario and between-scenario components."""
    # Between-scenario: take the mean per-scenario state vector (slow_traits)
    per_scenario_vecs: dict[str, list[float]] = {}
    for r in results:
        if r.post_snapshot is None:
            continue
        vec = [_safe_trait(r.post_snapshot.slow_traits, k) for k in _SLOW_TRAIT_KEYS]
        per_scenario_vecs[r.scenario_id] = vec

    n_scenarios = len(per_scenario_vecs)
    if n_scenarios < 2:
        return {
            "between_scenario_variance": 0.0,
            "within_scenario_variance": 0.0,
            "total_variance": 0.0,
            "between_ratio": 0.0,
        }

    dim = len(_SLOW_TRAIT_KEYS)
    # grand mean across all scenario means
    scenario_means: list[list[float]] = list(per_scenario_vecs.values())
    grand_mean = [mean(col) for col in zip(*scenario_means)]

    # Between-scenario: variance of scenario means around grand mean
    between_var = 0.0
    for svec in scenario_means:
        between_var += sum((svec[d] - grand_mean[d]) ** 2 for d in range(dim))
    between_var /= n_scenarios

    # Within-scenario: channel_stds averaged (proxy for within-scenario variance)
    within_vars: list[float] = []
    for r in results:
        ch_vars = [v**2 for v in r.channel_stds.values()]
        within_vars.append(mean(ch_vars) if ch_vars else 0.0)

    within_var = mean(within_vars) if within_vars else 0.0
    total_var = between_var + within_var
    between_ratio = between_var / total_var if total_var > 0 else 0.0

    return {
        "between_scenario_variance": round(between_var, 6),
        "within_scenario_variance": round(within_var, 6),
        "total_variance": round(total_var, 6),
        "between_ratio": round(between_ratio, 6),
    }


# ── within vs between person retrieval ───────────────────────────────────


def within_vs_between_retrieval(
    own_results: list[ScenarioResult],
    other_results: list[ScenarioResult],
) -> dict[str, object]:
    """Compare within-person vs between-person distances.

    Within-person: mean personality_distance across own scenario pairs.
    Between-person: mean personality_distance between own and other's snapshots.
    A good implant has within < between.
    """
    own_snapshots = [r.post_snapshot for r in own_results if r.post_snapshot is not None]
    other_snapshots = [r.post_snapshot for r in other_results if r.post_snapshot is not None]

    within_dists: list[float] = []
    for i in range(len(own_snapshots)):
        for j in range(i + 1, len(own_snapshots)):
            within_dists.append(personality_distance(own_snapshots[i], own_snapshots[j]))

    between_dists: list[float] = []
    for own in own_snapshots:
        for other in other_snapshots:
            between_dists.append(personality_distance(own, other))

    within_mean = mean(within_dists) if within_dists else 0.0
    between_mean = mean(between_dists) if between_dists else 0.0

    return {
        "within_person_mean_distance": round(within_mean, 6),
        "between_person_mean_distance": round(between_mean, 6),
        "retrieval_ratio": round(within_mean / between_mean, 6) if between_mean > 0 else 0.0,
        "retrieval_ok": within_mean < between_mean,
    }


# ── main analysis entry point ────────────────────────────────────────────


def analyze_cross_context(
    results: list[ScenarioResult],
    *,
    real_partner_variance: dict[str, float] | None = None,
    implant_traits: dict[str, float] | None = None,
) -> CrossContextReport:
    agent_uid = results[0].agent_uid if results else 0
    per_scenario = {r.scenario_id: r for r in results}

    consistency = personality_consistency_score(results)
    envelope = adaptation_envelope(results)

    real_env: dict[str, float] | None = None
    l2_distance: float | None = None
    if real_partner_variance is not None:
        real_env = dict(real_partner_variance)
        l2_distance = compare_adaptation_to_real(envelope, real_env)

    anomalies: list[dict[str, object]] = []
    for r in results:
        for key, dev in r.personality_deviation.items():
            if dev > 0.3:
                anomalies.append({
                    "scenario_id": r.scenario_id,
                    "trait": key,
                    "deviation": dev,
                    "severity": "crash" if dev > 0.5 else "warning",
                })

    # Build conclusion
    conclusions: list[str] = []
    if consistency >= 0.80:
        conclusions.append(f"人格一致性良好（{consistency:.3f} >= 0.80）")
    else:
        conclusions.append(f"人格一致性不足（{consistency:.3f} < 0.80）")

    nonzero_dims = sum(1 for v in envelope.values() if v > 0.01)
    if nonzero_dims >= 3:
        conclusions.append(f"适应性正常（{nonzero_dims}/5 维度标准差 > 0.01）")
    else:
        conclusions.append(f"适应性偏刚性（仅 {nonzero_dims}/5 维度标准差 > 0.01）")

    crash_count = len([a for a in anomalies if a.get("severity") == "crash"])
    if crash_count == 0:
        conclusions.append("无场景崩溃")
    else:
        conclusions.append(f"{crash_count} 个场景出现崩溃（偏离 > 0.5）")

    if l2_distance is not None:
        if l2_distance < 0.15:
            conclusions.append(f"适应幅度与真实用户接近（L2 = {l2_distance:.3f}）")
        else:
            conclusions.append(f"适应幅度与真实用户有偏差（L2 = {l2_distance:.3f}）")

    return CrossContextReport(
        agent_uid=agent_uid,
        per_scenario=per_scenario,
        personality_consistency=consistency,
        adaptation_envelope=envelope,
        real_adaptation_envelope=real_env,
        adaptation_l2_to_real=l2_distance,
        anomalies=anomalies,
        conclusion="；".join(conclusions) + "。",
    )


def _safe_trait(traits: dict[str, float], key: str) -> float:
    return float(traits.get(key, 0.5))
