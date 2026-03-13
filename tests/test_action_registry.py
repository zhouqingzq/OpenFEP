from __future__ import annotations

import random

from segmentum.action_registry import ActionRegistry
from segmentum.action_schema import ActionSchema
from segmentum.counterfactual import CounterfactualEngine, ForwardGenerativeModel
from segmentum.preferences import PreferenceModel
from segmentum.world_model import GenerativeWorldModel


def _episode(action: str = "forage") -> dict[str, object]:
    return {
        "cycle": 1,
        "timestamp": 1,
        "cluster_id": 0,
        "action_taken": ActionSchema(name=action),
        "observation": {
            "food": 0.1,
            "danger": 0.8,
            "novelty": 0.2,
            "shelter": 0.2,
            "temperature": 0.5,
            "social": 0.2,
        },
        "body_state": {
            "energy": 0.5,
            "stress": 0.4,
            "fatigue": 0.2,
            "temperature": 0.48,
        },
        "outcome": {
            "energy_delta": -0.1,
            "stress_delta": 0.2,
            "fatigue_delta": 0.1,
            "temperature_delta": 0.0,
            "free_energy_drop": -0.3,
        },
        "predicted_outcome": "survival_threat",
        "total_surprise": 1.2,
        "risk": 4.0,
    }


def test_registry_register_and_retrieve() -> None:
    registry = ActionRegistry()
    registry.register(ActionSchema(name="forage"), 0.05)
    registry.register(ActionSchema(name="hide"), 0.02)
    registry.register(ActionSchema(name="rest"), 0.01)
    assert {action.name for action in registry.get_all()} == {"forage", "hide", "rest"}
    assert {action.name for action in registry.get_alternatives(ActionSchema(name="forage"))} == {"hide", "rest"}


def test_registry_serialization_roundtrip() -> None:
    registry = ActionRegistry()
    registry.register(ActionSchema(name="tool_a", params={"mode": "safe"}), 0.11)
    restored = ActionRegistry.from_dict(registry.to_dict())
    assert restored.contains("tool_a")
    assert restored.get("tool_a") == ActionSchema(name="tool_a", params={"mode": "safe"}, cost_estimate=0.11)


def test_counterfactual_uses_registry() -> None:
    registry = ActionRegistry()
    registry.register(ActionSchema(name="forage"), 0.05)
    registry.register(ActionSchema(name="hide"), 0.02)
    registry.register(ActionSchema(name="turbo_forage", params={"boost": 2.0}), 0.10)
    world_model = GenerativeWorldModel()
    preference_model = PreferenceModel()
    forward_model = ForwardGenerativeModel(
        world_model=world_model,
        preference_model=preference_model,
        known_episodes=[_episode()],
        action_registry=registry,
    )
    forward_model.register_effects(
        "turbo_forage",
        {
            "energy_delta": 0.25,
            "stress_delta": -0.05,
            "fatigue_delta": 0.02,
            "temperature_delta": 0.0,
        },
    )
    engine = CounterfactualEngine(
        forward_model=forward_model,
        preference_model=preference_model,
        action_registry=registry,
        max_branches=5,
        energy_budget=1.0,
    )
    insights, _ = engine.run(
        episodes=[_episode()],
        current_cycle=2,
        agent_energy=0.8,
        rng=random.Random(7),
    )
    alt_names = {insight.counterfactual_action.name for insight in insights}
    assert "turbo_forage" in alt_names


def test_forward_model_unknown_action_conservative() -> None:
    model = ForwardGenerativeModel(
        world_model=GenerativeWorldModel(),
        preference_model=PreferenceModel(),
        known_episodes=[],
    )
    result = model.simulate(
        state={"energy": 0.5, "stress": 0.3},
        action=ActionSchema(name="unknown_tool", cost_estimate=0.15),
    )
    assert result["energy"] < 0.5
    assert result["stress"] > 0.3


def test_forward_model_register_learned_effects() -> None:
    model = ForwardGenerativeModel(
        world_model=GenerativeWorldModel(),
        preference_model=PreferenceModel(),
        known_episodes=[],
    )
    model.register_effects(
        "new_tool",
        {
            "energy_delta": 0.2,
            "stress_delta": -0.1,
            "fatigue_delta": 0.0,
            "temperature_delta": 0.0,
        },
    )
    result = model.simulate(
        state={"energy": 0.5, "stress": 0.3},
        action=ActionSchema(name="new_tool"),
    )
    assert result["energy"] > 0.5
    assert result["stress"] < 0.3
