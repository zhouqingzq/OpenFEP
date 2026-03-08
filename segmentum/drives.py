from __future__ import annotations

from dataclasses import dataclass, field

from .environment import clamp
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
