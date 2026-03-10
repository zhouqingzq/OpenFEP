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
    policy_score: float
    expected_free_energy: float
    predicted_error: float
    action_ambiguity: float
    risk: float
    preferred_probability: float
    memory_bias: float
    pattern_bias: float
    identity_bias: float
    value_score: float
    predicted_outcome: str
    predicted_effects: dict[str, float]
    dominant_component: str
    cost: float

    def policy_components(self) -> dict[str, float]:
        return {
            "negative_expected_free_energy": -self.expected_free_energy,
            "memory_bias": self.memory_bias,
            "pattern_bias": self.pattern_bias,
            "identity_bias": self.identity_bias,
        }


@dataclass
class DecisionDiagnostics:
    chosen: InterventionScore
    ranked_options: list[InterventionScore]
    prediction_error: float
    retrieved_memories: list[dict[str, object]]
    policy_scores: dict[str, float]
    explanation: str
