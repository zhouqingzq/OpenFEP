"""M5.3 dialogue outcome classification (pseudo-causal pairing for learning)."""

from __future__ import annotations

from enum import Enum
from typing import Mapping, MutableMapping

from .actions import DIALOGUE_ACTION_STRATEGY_MAP, is_dialogue_action


class DialogueOutcomeType(str, Enum):
    SOCIAL_REWARD = "social_reward"
    SOCIAL_THREAT = "social_threat"
    EPISTEMIC_GAIN = "epistemic_gain"
    EPISTEMIC_LOSS = "epistemic_loss"
    IDENTITY_AFFIRM = "identity_affirm"
    IDENTITY_THREAT = "identity_threat"
    NEUTRAL = "neutral"


def _delta(
    prev: Mapping[str, float] | None,
    cur: Mapping[str, float],
    key: str,
) -> float:
    if prev is None:
        return 0.0
    return float(cur.get(key, 0.0)) - float(prev.get(key, 0.0))


def classify_dialogue_outcome(
    action: str,
    next_observation: Mapping[str, float],
    dialogue_context: Mapping[str, object],
    *,
    previous_observation: Mapping[str, float] | None = None,
) -> DialogueOutcomeType:
    """Label (action, next_observation) pair; not a claim of structural causation."""
    del dialogue_context
    ct = float(next_observation.get("conflict_tension", 0.0))
    et = float(next_observation.get("emotional_tone", 0.5))
    sem = float(next_observation.get("semantic_content", 0.5))
    rel = float(next_observation.get("relationship_depth", 0.5))
    hid = float(next_observation.get("hidden_intent", 0.5))
    d_ct = _delta(previous_observation, next_observation, "conflict_tension")
    d_rel = _delta(previous_observation, next_observation, "relationship_depth")

    if is_dialogue_action(action):
        strat = DIALOGUE_ACTION_STRATEGY_MAP.get(action, "explore")
    else:
        strat = ""

    if action == "share_opinion" and ct > 0.52:
        return DialogueOutcomeType.IDENTITY_THREAT
    if action == "share_opinion" and et > 0.62 and ct < 0.45:
        return DialogueOutcomeType.SOCIAL_REWARD
    if action == "agree" and et > 0.68 and rel > 0.58 and ct < 0.4:
        return DialogueOutcomeType.IDENTITY_AFFIRM
    if action == "ask_question" and sem > 0.58:
        return DialogueOutcomeType.EPISTEMIC_GAIN
    if action == "disagree" and (ct > 0.68 or d_ct > 0.08):
        return DialogueOutcomeType.SOCIAL_THREAT
    if action == "empathize" and d_rel > 0.04 and ct < 0.55:
        return DialogueOutcomeType.SOCIAL_REWARD
    if strat == "escape" and previous_observation is not None and d_ct > 0.06:
        return DialogueOutcomeType.SOCIAL_THREAT
    if strat == "exploit" and et > 0.62 and ct < 0.42:
        return DialogueOutcomeType.SOCIAL_REWARD
    if hid > 0.72 and ct > 0.5:
        return DialogueOutcomeType.EPISTEMIC_LOSS
    if previous_observation is None:
        if ct > 0.75 or hid > 0.78:
            return DialogueOutcomeType.SOCIAL_THREAT
        if et > 0.65 and rel > 0.55:
            return DialogueOutcomeType.SOCIAL_REWARD
        return DialogueOutcomeType.NEUTRAL
    if ct > 0.62 and d_ct <= 0.0:
        return DialogueOutcomeType.SOCIAL_THREAT
    if et > 0.58 and ct < 0.48:
        return DialogueOutcomeType.SOCIAL_REWARD
    return DialogueOutcomeType.NEUTRAL


def inject_outcome_semantics(
    episode_like: MutableMapping[str, object],
    outcome: DialogueOutcomeType,
) -> None:
    """Attach dialogue outcome labels for sleep aggregation (patch A bridge)."""
    episode_like["dialogue_outcome_semantic"] = outcome.value
    if outcome in (DialogueOutcomeType.SOCIAL_THREAT, DialogueOutcomeType.IDENTITY_THREAT):
        episode_like["predicted_outcome"] = "dialogue_threat"
    elif outcome in (DialogueOutcomeType.SOCIAL_REWARD, DialogueOutcomeType.IDENTITY_AFFIRM):
        episode_like["predicted_outcome"] = "dialogue_reward"
    elif outcome == DialogueOutcomeType.EPISTEMIC_GAIN:
        episode_like["predicted_outcome"] = "dialogue_epistemic_gain"
    elif outcome == DialogueOutcomeType.EPISTEMIC_LOSS:
        episode_like["predicted_outcome"] = "dialogue_epistemic_loss"
    else:
        episode_like.setdefault("predicted_outcome", "neutral")

    def _bump(key: str, delta: float) -> None:
        try:
            cur = float(episode_like.get(key, 0.0))
        except (TypeError, ValueError):
            cur = 0.0
        episode_like[key] = round(max(0.0, min(1.0, cur + delta)), 6)

    if outcome in (DialogueOutcomeType.SOCIAL_THREAT, DialogueOutcomeType.IDENTITY_THREAT):
        _bump("relevance_threat", 0.08)
    elif outcome in (DialogueOutcomeType.SOCIAL_REWARD, DialogueOutcomeType.IDENTITY_AFFIRM):
        _bump("relevance_social", 0.06)
    elif outcome == DialogueOutcomeType.EPISTEMIC_GAIN:
        _bump("relevance_reward", 0.05)
    elif outcome == DialogueOutcomeType.EPISTEMIC_LOSS:
        _bump("relevance_threat", 0.04)
        try:
            v = float(episode_like.get("valence", 0.0))
        except (TypeError, ValueError):
            v = 0.0
        episode_like["valence"] = round(max(-1.0, min(1.0, v - 0.05)), 6)
