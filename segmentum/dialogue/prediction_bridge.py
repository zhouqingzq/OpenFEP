from __future__ import annotations

from typing import Mapping

from ..action_registry import ActionRegistry
from ..action_schema import ActionSchema
from ..prediction_ledger import PredictionHypothesis, PredictionLedger
from ..verification import VerificationLoop


DIALOGUE_PREDICTION_TYPES: dict[str, dict[str, object]] = {
    "topic_continuity": {
        "target_channels": ("semantic_content", "topic_novelty"),
        "description": "Dialogue topic continuity remains stable.",
        "confidence": 0.7,
    },
    "emotional_trajectory": {
        "target_channels": ("emotional_tone",),
        "description": "Emotional tone remains within a bounded drift.",
        "confidence": 0.4,
    },
    "conflict_trajectory": {
        "target_channels": ("conflict_tension",),
        "description": "Conflict tension remains within a bounded drift.",
        "confidence": 0.4,
    },
    "intent_stability": {
        "target_channels": ("hidden_intent",),
        "description": "Hidden intent remains stable across adjacent turns.",
        "confidence": 0.15,
    },
}


DIALOGUE_ACTIONS: tuple[ActionSchema, ...] = (
    ActionSchema(name="listen_reflect", cost_estimate=0.08),
    ActionSchema(name="ask_clarification", cost_estimate=0.10),
    ActionSchema(name="share_perspective", cost_estimate=0.12),
    ActionSchema(name="deescalate_tension", cost_estimate=0.14),
)


def _expected_state(
    observation: Mapping[str, float],
    target_channels: tuple[str, ...],
) -> dict[str, float]:
    return {channel: float(observation.get(channel, 0.5)) for channel in target_channels}


def register_dialogue_predictions(
    ledger: PredictionLedger,
    current_observation: Mapping[str, float],
    tick: int,
) -> list[str]:
    created: list[str] = []
    for prediction_type, config in DIALOGUE_PREDICTION_TYPES.items():
        target_channels = tuple(config["target_channels"])
        prediction_id = f"dialogue:{prediction_type}:{tick}"
        hypothesis = PredictionHypothesis(
            prediction_id=prediction_id,
            prediction_type=prediction_type,
            last_updated_tick=int(tick),
            target_channels=target_channels,
            expected_state=_expected_state(current_observation, target_channels),
            confidence=float(config.get("confidence", 0.3)),
            expected_horizon=1,
            created_tick=int(tick),
            source_module="dialogue_bridge",
            maintenance_context="scan",
        )
        ledger.predictions.append(hypothesis)
        created.append(prediction_id)
    ledger.last_tick = max(int(ledger.last_tick), int(tick))
    return created


def verify_dialogue_predictions(
    verification_loop: VerificationLoop,
    ledger: PredictionLedger,
    new_observation: Mapping[str, float],
    tick: int,
) -> dict[str, str]:
    update = ledger.verify_predictions(tick=int(tick), observation=new_observation)
    verification_loop.process_observation(
        tick=int(tick),
        observation=new_observation,
        ledger=ledger,
        source="dialogue_turn",
    )
    outcomes: dict[str, str] = {}
    for prediction_id in update.verified_predictions:
        outcomes[prediction_id] = "confirmed"
    for prediction_id in update.falsified_predictions:
        outcomes[prediction_id] = "falsified"
    return outcomes


def register_dialogue_actions(registry: ActionRegistry) -> None:
    for schema in DIALOGUE_ACTIONS:
        registry.register(schema)
