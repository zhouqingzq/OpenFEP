from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Drive:
    """A competing drive that pushes the agent toward specific goals."""

    name: str
    urgency: float
    weight: float
    target_modality: str


@dataclass
class MemoryEpisode:
    cycle: int
    choice: str
    free_energy_before: float
    free_energy_after: float
    dopamine_gain: float
    observation: dict[str, float]
    prediction: dict[str, float]
    errors: dict[str, float]
    body_state: dict[str, float]


@dataclass
class SleepSummary:
    average_free_energy_drop: float
    preferred_action: str
    stable_beliefs: dict[str, float]
    dream_replay_count: int
    memory_consolidations: int


@dataclass
class DreamReplay:
    """Represents a dream replay during sleep."""

    episode_index: int
    replayed_action: str
    imagined_outcome: dict[str, float]
    learning_signal: float


@dataclass
class InterventionScore:
    choice: str
    expected_free_energy: float
    cost: float


@dataclass
class DecisionDiagnostics:
    chosen: InterventionScore
    ranked_options: list[InterventionScore]
