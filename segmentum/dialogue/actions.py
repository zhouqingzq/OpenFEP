"""M5.3 dialogue action definitions: schemas, costs, strategy map, imagined channel deltas."""

from __future__ import annotations

from typing import Final, Mapping

from ..action_schema import ActionSchema

# M5.1 six-channel keys: if observation is a superset of these, treat as dialogue modality.
DIALOGUE_CHANNEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "semantic_content",
        "topic_novelty",
        "emotional_tone",
        "conflict_tension",
        "relationship_depth",
        "hidden_intent",
    }
)

DIALOGUE_ACTION_NAMES: Final[tuple[str, ...]] = (
    "ask_question",
    "introduce_topic",
    "share_opinion",
    "elaborate",
    "agree",
    "empathize",
    "joke",
    "disagree",
    "deflect",
    "minimal_response",
    "disengage",
)

DIALOGUE_ACTION_STRATEGY_MAP: Final[dict[str, str]] = {
    "ask_question": "explore",
    "introduce_topic": "explore",
    "share_opinion": "explore",
    "elaborate": "exploit",
    "agree": "exploit",
    "empathize": "exploit",
    "joke": "exploit",
    "disagree": "escape",
    "deflect": "escape",
    "minimal_response": "escape",
    "disengage": "escape",
}

# energy / social_risk (M5.3 prompt); cost_estimate uses energy, resource_cost carries social_risk
DIALOGUE_ACTION_COSTS: Final[dict[str, dict[str, float]]] = {
    "ask_question": {"energy": 0.03, "social_risk": 0.02},
    "introduce_topic": {"energy": 0.05, "social_risk": 0.04},
    "share_opinion": {"energy": 0.04, "social_risk": 0.06},
    "elaborate": {"energy": 0.03, "social_risk": 0.01},
    "agree": {"energy": 0.01, "social_risk": 0.00},
    "empathize": {"energy": 0.04, "social_risk": 0.01},
    "joke": {"energy": 0.03, "social_risk": 0.05},
    "disagree": {"energy": 0.05, "social_risk": 0.08},
    "deflect": {"energy": 0.02, "social_risk": 0.03},
    "minimal_response": {"energy": 0.01, "social_risk": 0.02},
    "disengage": {"energy": 0.00, "social_risk": 0.04},
}

# Small bounded deltas on M5.1 channel predictions for world_model.imagine_action
DIALOGUE_IMAGINED_EFFECTS: Final[dict[str, dict[str, float]]] = {
    "ask_question": {
        "topic_novelty": 0.03,
        "semantic_content": 0.02,
        "hidden_intent": 0.01,
    },
    "introduce_topic": {
        "topic_novelty": 0.04,
        "semantic_content": 0.02,
        "relationship_depth": 0.01,
    },
    "share_opinion": {
        "semantic_content": 0.03,
        "conflict_tension": 0.02,
        "relationship_depth": 0.02,
    },
    "elaborate": {
        "semantic_content": 0.03,
        "topic_novelty": -0.01,
        "relationship_depth": 0.02,
    },
    "agree": {
        "emotional_tone": 0.03,
        "conflict_tension": -0.03,
        "relationship_depth": 0.02,
    },
    "empathize": {
        "emotional_tone": 0.03,
        "relationship_depth": 0.03,
        "conflict_tension": -0.02,
    },
    "joke": {
        "emotional_tone": 0.02,
        "conflict_tension": -0.02,
        "hidden_intent": -0.01,
    },
    "disagree": {
        "conflict_tension": 0.04,
        "emotional_tone": -0.02,
        "relationship_depth": -0.01,
    },
    "deflect": {
        "conflict_tension": -0.02,
        "topic_novelty": 0.02,
        "hidden_intent": 0.01,
    },
    "minimal_response": {
        "semantic_content": -0.02,
        "relationship_depth": -0.01,
        "emotional_tone": -0.01,
    },
    "disengage": {
        "relationship_depth": -0.03,
        "semantic_content": -0.02,
        "conflict_tension": -0.02,
    },
}


def is_dialogue_action(name: str) -> bool:
    return name in DIALOGUE_ACTION_NAMES


def is_dialogue_channel_observation(observed: Mapping[str, float]) -> bool:
    """True when observation carries the full M5.1 dialogue channel vector."""
    if not observed:
        return False
    keys = frozenset(str(k) for k in observed.keys())
    return DIALOGUE_CHANNEL_KEYS <= keys


def _schema_for(name: str) -> ActionSchema:
    costs = DIALOGUE_ACTION_COSTS[name]
    energy = float(costs["energy"])
    social = float(costs["social_risk"])
    return ActionSchema(
        name=name,
        cost_estimate=energy,
        resource_cost={"energy": energy, "social_risk": social},
        params={"strategy": DIALOGUE_ACTION_STRATEGY_MAP.get(name, "explore")},
        reversible=True,
    )


DIALOGUE_ACTIONS: tuple[ActionSchema, ...] = tuple(_schema_for(n) for n in DIALOGUE_ACTION_NAMES)
