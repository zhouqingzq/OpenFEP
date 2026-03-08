from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ACTION_IMAGINED_EFFECTS
from .environment import clamp
from .predictive_coding import LayerBeliefUpdate, SensorimotorLayer, default_beliefs


@dataclass
class GenerativeWorldModel:
    """Mid-level beliefs that generate top-down predictions."""

    sensorimotor_layer: SensorimotorLayer = field(default_factory=SensorimotorLayer)
    learning_rate: float = 0.40

    @property
    def beliefs(self) -> dict[str, float]:
        return self.sensorimotor_layer.belief_state.beliefs

    def predict(
        self,
        priors: dict[str, float],
        memory_context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Generate predictions, optionally modulated by retrieved memory."""
        prediction = self.sensorimotor_layer.predict(priors)
        if memory_context:
            prediction = {
                key: clamp((prediction[key] * 0.80) + (memory_context[key] * 0.20))
                if key in memory_context
                else prediction[key]
                for key in prediction
            }
        return prediction

    def update_from_error(self, errors: dict[str, float]) -> None:
        self.sensorimotor_layer.absorb_error_signal(
            errors,
            strength=self.learning_rate,
        )

    def assimilate(
        self,
        lower_layer_signal: dict[str, float],
        top_down_prediction: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.sensorimotor_layer.assimilate(
            lower_layer_signal,
            top_down_prediction,
            predicted_state=predicted_state,
        )

    def imagine_action(self, action: str, prediction: dict[str, float]) -> dict[str, float]:
        imagined = {}
        for key, value in prediction.items():
            delta = ACTION_IMAGINED_EFFECTS.get(action, {}).get(key, 0.0)
            imagined[key] = clamp(value + delta)
        return imagined

    def to_dict(self) -> dict:
        return {
            "beliefs": dict(self.beliefs),
            "sensorimotor_layer": self.sensorimotor_layer.to_dict(),
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> GenerativeWorldModel:
        if not payload:
            return cls()

        sensorimotor_payload = payload.get("sensorimotor_layer")
        model = cls(
            sensorimotor_layer=SensorimotorLayer.from_dict(sensorimotor_payload),
            learning_rate=float(payload.get("learning_rate", 0.40)),
        )
        if not sensorimotor_payload:
            model.sensorimotor_layer.belief_state.beliefs = (
                dict(payload.get("beliefs", {})) or default_beliefs()
            )
        return model