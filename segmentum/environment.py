from __future__ import annotations

from dataclasses import dataclass
import random


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class Observation:
    food: float
    danger: float
    novelty: float
    shelter: float
    temperature: float
    social: float


class SimulatedWorld:
    """Small hostile world for the first Segment prototype."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.food_density = 0.55
        self.threat_density = 0.35
        self.novelty_density = 0.60
        self.shelter_density = 0.45
        self.temperature = 0.48
        self.social_density = 0.22
        self.tick = 0

    def observe(self) -> Observation:
        noise = lambda spread: self.rng.uniform(-spread, spread)
        return Observation(
            food=clamp(self.food_density + noise(0.08)),
            danger=clamp(self.threat_density + noise(0.08)),
            novelty=clamp(self.novelty_density + noise(0.08)),
            shelter=clamp(self.shelter_density + noise(0.05)),
            temperature=clamp(self.temperature + noise(0.06)),
            social=clamp(self.social_density + noise(0.09)),
        )

    def drift(self) -> None:
        self.tick += 1
        phase = (self.tick % 12) / 12.0
        self.food_density = clamp(self.food_density - 0.03 + self.rng.uniform(-0.04, 0.04))
        self.threat_density = clamp(self.threat_density + (0.02 if phase > 0.55 else -0.01) + self.rng.uniform(-0.05, 0.05))
        self.novelty_density = clamp(self.novelty_density - 0.02 + self.rng.uniform(-0.03, 0.05))
        self.shelter_density = clamp(self.shelter_density + self.rng.uniform(-0.03, 0.03))
        self.temperature = clamp(0.50 + (phase - 0.5) * 0.40 + self.rng.uniform(-0.06, 0.06))
        self.social_density = clamp(self.social_density + self.rng.uniform(-0.05, 0.05))

    def apply_action(self, action: str) -> dict[str, float]:
        """Mutate the world and return direct physiological consequences."""
        direct = {
            "energy_delta": 0.0,
            "stress_delta": 0.0,
            "fatigue_delta": 0.0,
            "temperature_delta": 0.0,
            "loneliness_delta": 0.0,
        }

        if action == "forage":
            gain = 0.20 + self.rng.uniform(-0.05, 0.08)
            self.food_density = clamp(self.food_density - 0.18)
            self.novelty_density = clamp(self.novelty_density + 0.06)
            self.threat_density = clamp(self.threat_density + 0.08)
            direct["energy_delta"] += gain
            direct["stress_delta"] += 0.08
            direct["fatigue_delta"] += 0.08
        elif action == "hide":
            self.threat_density = clamp(self.threat_density - 0.20)
            self.shelter_density = clamp(self.shelter_density + 0.04)
            self.novelty_density = clamp(self.novelty_density - 0.05)
            direct["energy_delta"] -= 0.03
            direct["stress_delta"] -= 0.14
            direct["fatigue_delta"] += 0.02
            direct["temperature_delta"] += (0.50 - self.temperature) * 0.20
        elif action == "scan":
            self.novelty_density = clamp(self.novelty_density + 0.16)
            self.threat_density = clamp(self.threat_density + 0.03)
            direct["energy_delta"] -= 0.04
            direct["stress_delta"] += 0.02
            direct["fatigue_delta"] += 0.05
        elif action == "exploit_shelter":
            self.shelter_density = clamp(self.shelter_density + 0.18)
            self.threat_density = clamp(self.threat_density - 0.10)
            self.food_density = clamp(self.food_density - 0.04)
            direct["energy_delta"] -= 0.05
            direct["stress_delta"] -= 0.08
            direct["fatigue_delta"] += 0.03
            direct["temperature_delta"] += (0.50 - self.temperature) * 0.25
        elif action == "rest":
            direct["energy_delta"] -= 0.01
            direct["stress_delta"] -= 0.03
            direct["fatigue_delta"] -= 0.10
        elif action == "seek_contact":
            self.social_density = clamp(self.social_density + 0.20)
            self.threat_density = clamp(self.threat_density + 0.05)
            self.novelty_density = clamp(self.novelty_density + 0.04)
            direct["energy_delta"] -= 0.03
            direct["stress_delta"] -= 0.06
            direct["loneliness_delta"] -= 0.24
            direct["fatigue_delta"] += 0.03
        elif action == "thermoregulate":
            midpoint_shift = 0.50 - self.temperature
            self.temperature = clamp(self.temperature + midpoint_shift * 0.65)
            self.shelter_density = clamp(self.shelter_density - 0.03)
            direct["energy_delta"] -= 0.06
            direct["stress_delta"] -= 0.05
            direct["temperature_delta"] += midpoint_shift * 0.60
            direct["fatigue_delta"] += 0.02

        self.drift()
        return direct

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "food_density": self.food_density,
            "threat_density": self.threat_density,
            "novelty_density": self.novelty_density,
            "shelter_density": self.shelter_density,
            "temperature": self.temperature,
            "social_density": self.social_density,
            "tick": self.tick,
            "rng_state": repr(self.rng.getstate()),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> SimulatedWorld:
        if not payload:
            return cls()

        world = cls(seed=int(payload.get("seed", 7)))
        world.food_density = float(payload.get("food_density", world.food_density))
        world.threat_density = float(payload.get("threat_density", world.threat_density))
        world.novelty_density = float(payload.get("novelty_density", world.novelty_density))
        world.shelter_density = float(payload.get("shelter_density", world.shelter_density))
        world.temperature = float(payload.get("temperature", world.temperature))
        world.social_density = float(payload.get("social_density", world.social_density))
        world.tick = int(payload.get("tick", world.tick))

        rng_state = payload.get("rng_state")
        if isinstance(rng_state, str):
            import ast

            world.rng.setstate(ast.literal_eval(rng_state))
        return world
