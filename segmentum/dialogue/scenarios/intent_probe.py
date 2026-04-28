"""M5.5 hidden_intent precision probe for Scenario 6 (ambiguous intent)."""

from __future__ import annotations

from ..channel_registry import DIALOGUE_CHANNELS, ObservabilityTier, get_channel_spec
from ..precision_bounds import ChannelPrecisionBounds
from .conductor import ScenarioResult

# Tier 3 channel precision bounds (hidden_intent, relationship_depth)
_TIER3_FLOOR = 0.05
_TIER3_CEILING = 0.20


def probe_intent_precision(
    agent,
    scenario_results: list[ScenarioResult],
) -> dict[str, object]:
    """Analyze hidden_intent channel precision behavior in ambiguous-intent scenarios.

    Returns a dictionary with:
    - precision_trajectory: per-turn hidden_intent channel values
    - anomaly_type: "normal" / "paranoid" / "naive"
    - defense_actions: set of escape/dodge strategies used
    - clinical_mapping: human-readable personality characterization
    - tier_compliance: whether precision stays within Tier 3 bounds
    """
    # Find scenario 6 result
    s6 = None
    for r in scenario_results:
        if r.scenario_id == "ambiguous_intent":
            s6 = r
            break

    if s6 is None:
        return {
            "error": "ambiguous_intent scenario not found in results",
            "precision_trajectory": [],
            "anomaly_type": "unknown",
        }

    # Extract hidden_intent trajectory
    hi_trajectory: list[float] = []
    for snap in s6.precision_trajectory:
        hi_trajectory.append(float(snap.get("hidden_intent", 0.10)))

    mean_hi = sum(hi_trajectory) / len(hi_trajectory) if hi_trajectory else 0.10

    # Classify anomaly type
    bounds = ChannelPrecisionBounds.from_dialogue_channels()
    hi_spec = get_channel_spec("hidden_intent")

    if mean_hi > _TIER3_CEILING:
        anomaly_type = "paranoid"
    elif mean_hi < _TIER3_FLOOR:
        anomaly_type = "naive"
    else:
        anomaly_type = "normal"

    # Defense / escape actions used
    defense_actions: set[str] = set()
    for turn in s6.turns:
        if turn.strategy == "escape":
            defense_actions.add(turn.action or "unknown")

    # Tier compliance
    tier_compliant = all(_TIER3_FLOOR <= v <= _TIER3_CEILING for v in hi_trajectory)
    tier_violations = sum(1 for v in hi_trajectory if v < _TIER3_FLOOR or v > _TIER3_CEILING)

    # Clinical mapping
    if anomaly_type == "paranoid":
        clinical = (
            "偏执漂移：hidden_intent 精度高于 Tier 3 上限（> 0.20），"
            "对表面友好的发言过度解读为暗含敌意或目的性。"
            "行为表现：频繁使用 deflect、minimal_response、反问对方意图。"
        )
    elif anomaly_type == "naive":
        clinical = (
            "天真漂移：hidden_intent 精度接近 0，"
            "完全不注意潜在风险，毫无保留地分享信息。"
            "行为表现：在意图模糊场景中仍采取大量 share_opinion、elaborate。"
        )
    else:
        clinical = (
            "正常精度校准：hidden_intent 精度在 Tier 3 范围（0.05-0.20）内，"
            "对话回应正常，既不过度解读也不完全忽略潜在意图。"
        )

    # Per-turn anomaly labels
    anomaly_report = bounds.anomaly_report({"hidden_intent": mean_hi})

    return {
        "scenario_id": "ambiguous_intent",
        "precision_trajectory": [round(v, 6) for v in hi_trajectory],
        "mean_hidden_intent_precision": round(mean_hi, 6),
        "anomaly_type": anomaly_type,
        "anomaly_label": anomaly_report.get("hidden_intent", "none"),
        "defense_actions": sorted(defense_actions),
        "defense_actions_count": len(defense_actions),
        "clinical_mapping": clinical,
        "tier_compliance": tier_compliant,
        "tier_violations": tier_violations,
        "tier_violation_ratio": round(tier_violations / len(hi_trajectory), 6) if hi_trajectory else 0.0,
        "tier_floor": _TIER3_FLOOR,
        "tier_ceiling": _TIER3_CEILING,
        "tier": "LOW (Tier 3)",
    }
