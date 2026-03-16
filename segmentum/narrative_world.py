from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping

from .environment import Observation, clamp
from .narrative_types import NarrativeEpisode


CHANNELS = ("food", "danger", "novelty", "shelter", "temperature", "social")


def _coerce_float_dict(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


@dataclass(slots=True)
class WorldEvent:
    tick: int
    world_id: str
    event_type: str
    observation_delta: dict[str, float]
    narrative_text: str
    tags: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": int(self.tick),
            "world_id": self.world_id,
            "event_type": self.event_type,
            "observation_delta": dict(self.observation_delta),
            "narrative_text": self.narrative_text,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WorldEvent":
        return cls(
            tick=int(payload.get("tick", 0)),
            world_id=str(payload.get("world_id", "")),
            event_type=str(payload.get("event_type", "world_shift")),
            observation_delta=_coerce_float_dict(payload.get("observation_delta")),
            narrative_text=str(payload.get("narrative_text", "")),
            tags=[str(item) for item in payload.get("tags", []) if isinstance(item, (str, int, float))],
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass(slots=True)
class NarrativeWorldConfig:
    world_id: str
    seed: int
    baseline_observation: dict[str, float]
    drift_profile: dict[str, float]
    event_schedule: list[dict[str, object]]
    resource_rules: dict[str, float]
    hazard_rules: dict[str, float]
    social_rules: dict[str, float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NarrativeWorldConfig":
        return cls(
            world_id=str(payload.get("world_id", "world")),
            seed=int(payload.get("seed", 0)),
            baseline_observation=_coerce_float_dict(payload.get("baseline_observation")),
            drift_profile=_coerce_float_dict(payload.get("drift_profile")),
            event_schedule=[
                dict(item) for item in payload.get("event_schedule", []) if isinstance(item, dict)
            ],
            resource_rules=_coerce_float_dict(payload.get("resource_rules")),
            hazard_rules=_coerce_float_dict(payload.get("hazard_rules")),
            social_rules=_coerce_float_dict(payload.get("social_rules")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "world_id": self.world_id,
            "seed": self.seed,
            "baseline_observation": dict(self.baseline_observation),
            "drift_profile": dict(self.drift_profile),
            "event_schedule": [dict(item) for item in self.event_schedule],
            "resource_rules": dict(self.resource_rules),
            "hazard_rules": dict(self.hazard_rules),
            "social_rules": dict(self.social_rules),
        }


class NarrativeWorld:
    def __init__(self, config: NarrativeWorldConfig, *, rng_seed: int | None = None) -> None:
        self.config = config
        self.rng_seed = int(config.seed if rng_seed is None else rng_seed)
        self.rng = random.Random(self.rng_seed)
        self.tick = 0
        self.last_tick = -1
        self.last_observation = Observation(**{
            channel: float(config.baseline_observation.get(channel, 0.5))
            for channel in CHANNELS
        })
        self._applied_feedback: dict[str, dict[str, float]] = {}
        self._episode_cache: dict[int, list[NarrativeEpisode]] = {}
        self._event_history: list[WorldEvent] = []

    def _drifted_baseline(self, tick: int) -> dict[str, float]:
        baseline = {
            channel: float(self.config.baseline_observation.get(channel, 0.5))
            for channel in CHANNELS
        }
        for index, channel in enumerate(CHANNELS):
            amplitude = float(self.config.drift_profile.get(channel, 0.0))
            phase = (tick + index + self.rng_seed % 11) / 6.0
            baseline[channel] = clamp(baseline[channel] + math.sin(phase) * amplitude)
        if tick in self._applied_feedback:
            for channel, delta in self._applied_feedback[tick].items():
                baseline[channel] = clamp(baseline.get(channel, 0.5) + delta)
        return baseline

    def pending_events(self, tick: int) -> list[WorldEvent]:
        events: list[WorldEvent] = []
        for item in self.config.event_schedule:
            if int(item.get("tick", -1)) != tick:
                continue
            events.append(
                WorldEvent(
                    tick=tick,
                    world_id=self.config.world_id,
                    event_type=str(item.get("event_type", "world_shift")),
                    observation_delta=_coerce_float_dict(item.get("observation_delta")),
                    narrative_text=str(item.get("narrative_text", "")),
                    tags=[str(tag) for tag in item.get("tags", [])],
                    metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
                )
            )
        events.sort(key=lambda event: (event.tick, event.event_type, event.narrative_text))
        return events

    def observe(self, tick: int) -> Observation:
        state = self._drifted_baseline(tick)
        episodes: list[NarrativeEpisode] = []
        for event in self.pending_events(tick):
            for channel, delta in event.observation_delta.items():
                state[channel] = clamp(state.get(channel, 0.5) + delta)
            episodes.append(
                NarrativeEpisode(
                    episode_id=f"{self.config.world_id}:{tick}:{event.event_type}",
                    timestamp=tick,
                    source=self.config.world_id,
                    raw_text=event.narrative_text,
                    tags=list(event.tags),
                    metadata={
                        "world_id": self.config.world_id,
                        "event_type": event.event_type,
                        **dict(event.metadata),
                    },
                )
            )
            self._event_history.append(event)
        self._episode_cache[tick] = episodes
        self.last_tick = tick
        self.tick = tick
        self.last_observation = Observation(**{channel: float(state[channel]) for channel in CHANNELS})
        return self.last_observation

    def apply_action(self, action: str, tick: int) -> dict[str, float]:
        action_key = str(action)
        resource = self.config.resource_rules
        hazard = self.config.hazard_rules
        social = self.config.social_rules
        direct = {
            "energy_delta": 0.0,
            "stress_delta": 0.0,
            "fatigue_delta": 0.0,
            "temperature_delta": 0.0,
            "loneliness_delta": 0.0,
        }
        obs_delta: dict[str, float] = {}

        if action_key == "forage":
            direct["energy_delta"] += resource.get("forage_energy_gain", 0.12)
            direct["stress_delta"] += hazard.get("forage_stress_cost", 0.06)
            direct["fatigue_delta"] += 0.05
            obs_delta["food"] = -resource.get("food_depletion", 0.08)
            obs_delta["danger"] = hazard.get("forage_danger_cost", 0.05)
            obs_delta["novelty"] = 0.04
        elif action_key == "hide":
            direct["energy_delta"] -= hazard.get("hide_energy_cost", 0.03)
            direct["stress_delta"] -= hazard.get("hide_stress_relief", 0.10)
            direct["fatigue_delta"] += hazard.get("hide_fatigue_cost", 0.02)
            obs_delta["danger"] = -hazard.get("hide_danger_relief", 0.12)
            obs_delta["shelter"] = 0.03
        elif action_key == "scan":
            direct["energy_delta"] -= 0.04
            direct["stress_delta"] += 0.01
            obs_delta["novelty"] = 0.12
            obs_delta["danger"] = hazard.get("scan_danger_reveal", 0.03)
        elif action_key == "seek_contact":
            direct["energy_delta"] -= 0.03
            direct["stress_delta"] -= social.get("contact_stress_relief", 0.05)
            direct["loneliness_delta"] -= social.get("contact_loneliness_relief", 0.20)
            obs_delta["social"] = social.get("contact_social_gain", 0.14)
        elif action_key == "exploit_shelter":
            direct["stress_delta"] -= hazard.get("shelter_stress_relief", 0.07)
            obs_delta["shelter"] = 0.10
            obs_delta["danger"] = -hazard.get("shelter_danger_relief", 0.08)
        elif action_key == "rest":
            direct["fatigue_delta"] -= 0.08
            direct["stress_delta"] -= 0.03
        elif action_key == "thermoregulate":
            midpoint_shift = 0.50 - self.last_observation.temperature
            direct["temperature_delta"] += midpoint_shift * 0.5
            direct["energy_delta"] -= 0.05

        if obs_delta:
            self._applied_feedback.setdefault(tick + 1, {})
            for channel, delta in obs_delta.items():
                self._applied_feedback[tick + 1][channel] = (
                    self._applied_feedback[tick + 1].get(channel, 0.0) + float(delta)
                )
        return direct

    def narrative_episodes(self, tick: int) -> list[NarrativeEpisode]:
        return [NarrativeEpisode.from_dict(item.to_dict()) for item in self._episode_cache.get(tick, [])]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "rng_seed": self.rng_seed,
            "tick": self.tick,
            "last_tick": self.last_tick,
            "last_observation": {
                channel: float(getattr(self.last_observation, channel))
                for channel in CHANNELS
            },
            "applied_feedback": {
                str(key): dict(value) for key, value in self._applied_feedback.items()
            },
            "event_history": [event.to_dict() for event in self._event_history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NarrativeWorld":
        config = NarrativeWorldConfig.from_dict(dict(payload.get("config", {})))
        world = cls(config, rng_seed=int(payload.get("rng_seed", config.seed)))
        world.tick = int(payload.get("tick", 0))
        world.last_tick = int(payload.get("last_tick", -1))
        last_observation = _coerce_float_dict(payload.get("last_observation"))
        if last_observation:
            world.last_observation = Observation(**{
                channel: float(last_observation.get(channel, 0.5))
                for channel in CHANNELS
            })
        world._applied_feedback = {
            int(key): _coerce_float_dict(value)
            for key, value in dict(payload.get("applied_feedback", {})).items()
            if isinstance(value, dict)
        }
        world._event_history = [
            WorldEvent.from_dict(item)
            for item in payload.get("event_history", [])
            if isinstance(item, dict)
        ]
        return world
