from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ACTION_IMAGINED_EFFECTS
from .environment import clamp


@dataclass
class GenerativeWorldModel:
    """Mid-level beliefs that generate top-down predictions."""

    beliefs: dict[str, float] = field(
        default_factory=lambda: {
            "food": 0.50,
            "danger": 0.30,
            "novelty": 0.50,
            "shelter": 0.40,
            "temperature": 0.50,
            "social": 0.30,
        }
    )
    learning_rate: float = 0.40

    def predict(
        self,
        priors: dict[str, float],
        memory_context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Generate predictions, optionally modulated by retrieved memory."""
        prediction = {}
        for key, prior in priors.items():
            belief = self.beliefs[key]
            base = (belief * 0.60) + (prior * 0.40)
            if memory_context and key in memory_context:
                base = (base * 0.80) + (memory_context[key] * 0.20)
            prediction[key] = clamp(base)
        return prediction

    def update_from_error(self, errors: dict[str, float]) -> None:
        for key, error in errors.items():
            if key in self.beliefs:
                self.beliefs[key] = clamp(self.beliefs[key] + self.learning_rate * error)

    def imagine_action(self, action: str, prediction: dict[str, float]) -> dict[str, float]:
        imagined = {}
        for key, value in prediction.items():
            delta = ACTION_IMAGINED_EFFECTS.get(action, {}).get(key, 0.0)
            imagined[key] = clamp(value + delta)
        return imagined

    def to_dict(self) -> dict:
        return {
            "beliefs": dict(self.beliefs),
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> GenerativeWorldModel:
        if not payload:
            return cls()

        return cls(
            beliefs=dict(payload.get("beliefs", {})) or cls().beliefs,
            learning_rate=float(payload.get("learning_rate", 0.40)),
        )