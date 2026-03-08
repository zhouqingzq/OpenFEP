from __future__ import annotations

from dataclasses import dataclass, field

from .environment import clamp
from .predictive_coding import LayerBeliefUpdate, StrategicBeliefState
from .types import Drive


@dataclass
class DriveSystem:
    """Manages competing drives that create internal pressure."""

    drives: list[Drive] = field(
        default_factory=lambda: [
            Drive("hunger", 0.0, 1.3, "food"),
            Drive("safety", 0.0, 1.5, "danger"),
            Drive("exploration", 0.0, 0.8, "novelty"),
            Drive("comfort", 0.0, 1.0, "shelter"),
            Drive("thermal", 0.0, 0.9, "temperature"),
            Drive("social", 0.0, 0.7, "social"),
        ]
    )

    def update_urgencies(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        social_isolation: float,
        novelty_deficit: float,
    ) -> None:
        """Update drive urgencies based on body state and deficits."""
        self.drives[0].urgency = clamp(1.0 - energy)
        self.drives[1].urgency = clamp(stress * 1.2)
        self.drives[2].urgency = clamp(novelty_deficit)
        self.drives[3].urgency = clamp(stress * 0.8 + (1.0 - energy) * 0.3)
        self.drives[4].urgency = clamp(abs(temperature - 0.5) * 2.0)
        self.drives[5].urgency = clamp(social_isolation)

    def compute_prior_modulation(self, base_prior: float, modality: str) -> float:
        """Modulate a prior based on competing drives."""
        total_pressure = 0.0
        for drive in self.drives:
            if drive.target_modality == modality:
                total_pressure += drive.urgency * drive.weight

        modulation = clamp(total_pressure * 0.15)
        if modality in ["danger", "temperature"]:
            return clamp(base_prior - modulation)
        return clamp(base_prior + modulation)


@dataclass
class StrategicLayer:
    """Global priors that define what survival means right now."""

    energy_floor: float = 0.50
    danger_ceiling: float = 0.20
    novelty_floor: float = 0.30
    shelter_floor: float = 0.40
    temperature_ideal: float = 0.50
    social_floor: float = 0.25
    belief_state: StrategicBeliefState = field(default_factory=StrategicBeliefState)

    def priors(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        dopamine: float,
        drive_system: DriveSystem,
    ) -> dict[str, float]:
        base = {
            "food": clamp(1.10 - energy),
            "danger": clamp(self.danger_ceiling + stress * 0.60),
            "novelty": clamp(self.novelty_floor + dopamine * 0.10),
            "shelter": clamp(self.shelter_floor + stress * 0.30),
            "temperature": clamp(
                self.temperature_ideal + abs(temperature - self.temperature_ideal) * 0.5
            ),
            "social": clamp(self.social_floor + (1.0 - dopamine) * 0.15),
        }

        return {
            key: drive_system.compute_prior_modulation(value, key)
            for key, value in base.items()
        }

    def dispatch_prediction(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        dopamine: float,
        drive_system: DriveSystem,
    ) -> tuple[dict[str, float], dict[str, float]]:
        base_priors = self.priors(
            energy,
            stress,
            fatigue,
            temperature,
            dopamine,
            drive_system,
        )
        prediction = self.belief_state.predict(base_priors)
        return base_priors, prediction

    def assimilate(
        self,
        lower_layer_signal: dict[str, float],
        base_priors: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.belief_state.posterior_update(
            lower_layer_signal,
            base_priors,
            predicted_state=predicted_state,
        )

    @property
    def beliefs(self) -> dict[str, float]:
        return self.belief_state.beliefs

    def absorb_error_signal(
        self,
        errors: dict[str, float],
        strength: float = 1.0,
    ) -> None:
        self.belief_state.absorb_error_signal(errors, strength=strength)

    def to_dict(self) -> dict[str, object]:
        return {
            "energy_floor": self.energy_floor,
            "danger_ceiling": self.danger_ceiling,
            "novelty_floor": self.novelty_floor,
            "shelter_floor": self.shelter_floor,
            "temperature_ideal": self.temperature_ideal,
            "social_floor": self.social_floor,
            "belief_state": self.belief_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> StrategicLayer:
        layer = cls()
        if not payload:
            return layer
        layer.energy_floor = float(payload.get("energy_floor", layer.energy_floor))
        layer.danger_ceiling = float(payload.get("danger_ceiling", layer.danger_ceiling))
        layer.novelty_floor = float(payload.get("novelty_floor", layer.novelty_floor))
        layer.shelter_floor = float(payload.get("shelter_floor", layer.shelter_floor))
        layer.temperature_ideal = float(
            payload.get("temperature_ideal", layer.temperature_ideal)
        )
        layer.social_floor = float(payload.get("social_floor", layer.social_floor))
        layer.belief_state = StrategicBeliefState.from_dict(payload.get("belief_state"))
        return layer
