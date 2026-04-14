"""M5.3 patch C: dialogue pattern pressures on CognitiveStyleParameters with drift budget."""

from __future__ import annotations

from typing import Any, Mapping

from ..m4_cognitive_style import PARAMETER_REFERENCE, CognitiveStyleParameters

PER_PARAMETER_CAP = 0.025
MAX_TOTAL_L1 = 0.10
NET_DRIFT_CAP = 0.12

_PATTERN_WEIGHTS: dict[str, dict[str, float]] = {
    "conflict_rupture_pattern": {
        "error_aversion": 0.02,
        "resource_pressure_sensitivity": 0.01,
        "attention_selectivity": 0.01,
    },
    "conflict_repair_pattern": {
        "error_aversion": -0.01,
        "update_rigidity": -0.01,
        "confidence_gain": 0.01,
    },
    "opinion_rejection_pattern": {
        "error_aversion": 0.02,
        "exploration_bias": -0.01,
        "confidence_gain": -0.01,
    },
    "question_reward_pattern": {
        "exploration_bias": 0.02,
        "uncertainty_sensitivity": -0.01,
    },
    "withdrawal_pattern": {
        "resource_pressure_sensitivity": 0.02,
        "exploration_bias": -0.02,
        "virtual_prediction_error_gain": -0.01,
    },
    "trusting_environment_pattern": {
        "uncertainty_sensitivity": -0.01,
        "confidence_gain": 0.01,
        "attention_selectivity": -0.01,
    },
    "suspicious_environment_pattern": {
        "uncertainty_sensitivity": 0.02,
        "attention_selectivity": 0.02,
        "virtual_prediction_error_gain": 0.01,
    },
}


def detect_dialogue_patterns(
    replay_batch: list[dict[str, object]],
    decision_history: list[dict[str, object]],
) -> dict[str, float]:
    """Lightweight heuristics over replay + decisions; intensities in [0,1]."""
    patterns: dict[str, float] = {}
    threat_like = 0
    repair_like = 0
    opinion_reject = 0
    for payload in replay_batch:
        if not isinstance(payload, dict):
            continue
        sem = str(payload.get("dialogue_outcome_semantic", ""))
        if sem in {"social_threat", "identity_threat"}:
            threat_like += 1
        if str(payload.get("predicted_outcome", "")) == "dialogue_reward":
            repair_like += 1
        if sem == "identity_threat" and str(payload.get("action_taken", "")) == "share_opinion":
            opinion_reject += 1
    if threat_like >= 2 and repair_like == 0:
        patterns["conflict_rupture_pattern"] = min(1.0, threat_like / 5.0)
    if threat_like >= 1 and repair_like >= 1:
        patterns["conflict_repair_pattern"] = min(1.0, (threat_like + repair_like) / 6.0)
    if opinion_reject >= 1:
        patterns["opinion_rejection_pattern"] = min(1.0, opinion_reject / 3.0)

    q_reward = 0
    low_intent_affirm = 0
    suspicious_hits = 0
    for payload in replay_batch:
        if not isinstance(payload, dict):
            continue
        if (
            str(payload.get("dialogue_outcome_semantic", "")) == "epistemic_gain"
            and str(payload.get("action_taken", "")) == "ask_question"
        ):
            q_reward += 1
        obs = payload.get("observation")
        hi = 0.5
        if isinstance(obs, dict):
            try:
                hi = float(obs.get("hidden_intent", 0.5))
            except (TypeError, ValueError):
                hi = 0.5
        sem = str(payload.get("dialogue_outcome_semantic", ""))
        if hi <= 0.35 and sem in {"social_reward", "identity_affirm"}:
            low_intent_affirm += 1
        if hi >= 0.65 and sem in {"social_threat", "identity_threat"}:
            suspicious_hits += 1
    if q_reward >= 1:
        patterns["question_reward_pattern"] = min(1.0, q_reward / 3.0)
    if low_intent_affirm >= 2:
        patterns["trusting_environment_pattern"] = min(1.0, low_intent_affirm / 5.0)
    if suspicious_hits >= 2:
        patterns["suspicious_environment_pattern"] = min(1.0, suspicious_hits / 4.0)

    streak = 0
    max_streak = 0
    for row in decision_history[-48:]:
        if str(row.get("action", "")) == "minimal_response":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    if max_streak >= 3:
        patterns["withdrawal_pattern"] = min(1.0, max_streak / 6.0)

    if not patterns and decision_history:
        for row in decision_history[-32:]:
            act = str(row.get("action", ""))
            if act in {"disagree", "deflect", "disengage"}:
                threat_like += 0.2
        if threat_like >= 1.0:
            patterns["conflict_rupture_pattern"] = min(1.0, threat_like / 4.0)
    return patterns


def dialogue_patterns_to_cognitive_pressure(patterns: Mapping[str, float]) -> dict[str, float]:
    raw: dict[str, float] = {name: 0.0 for name in PARAMETER_REFERENCE}
    for pname, intensity in patterns.items():
        weights = _PATTERN_WEIGHTS.get(pname)
        if not weights:
            continue
        for dim, w in weights.items():
            if dim in raw:
                raw[dim] += float(w) * float(intensity)
    return raw


def apply_cognitive_style_drift_budget(
    raw_deltas: Mapping[str, float],
    baseline: Mapping[str, float],
    current: Mapping[str, float],
) -> dict[str, float]:
    """Per M5.3: per-parameter cap, L1 total cap, optional net drift vs baseline."""
    clipped: dict[str, float] = {}
    for key, delta in raw_deltas.items():
        if key not in PARAMETER_REFERENCE:
            continue
        d = max(-PER_PARAMETER_CAP, min(PER_PARAMETER_CAP, float(delta)))
        base = float(baseline.get(key, current.get(key, 0.5)))
        cur = float(current.get(key, base))
        max_up = NET_DRIFT_CAP - (cur - base)
        max_down = NET_DRIFT_CAP - (base - cur)
        if d > 0:
            d = min(d, max(0.0, max_up))
        else:
            d = max(d, -max(0.0, max_down))
        clipped[key] = d
    l1 = sum(abs(v) for v in clipped.values())
    if l1 > MAX_TOTAL_L1 and l1 > 0:
        scale = MAX_TOTAL_L1 / l1
        clipped = {k: v * scale for k, v in clipped.items()}
    return clipped


def merge_cognitive_style(
    style: CognitiveStyleParameters,
    deltas: Mapping[str, float],
) -> CognitiveStyleParameters:
    data = style.to_dict()
    for key, delta in deltas.items():
        if key not in PARAMETER_REFERENCE:
            continue
        data[key] = float(data.get(key, 0.5)) + float(delta)
    return CognitiveStyleParameters.from_dict(data)
