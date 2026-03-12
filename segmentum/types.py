from __future__ import annotations

from dataclasses import dataclass, field


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
    sleep_cycle_id: int = 0
    episodes_sampled: int = 0
    clusters_created: int = 0
    patterns_found: int = 0
    world_model_updates: int = 0
    policy_bias_updates: int = 0
    epistemic_bonus_updates: int = 0
    episodes_archived: int = 0
    episodes_deleted: int = 0
    memory_compressed: int = 0
    prediction_error_before: float = 0.0
    prediction_error_after: float = 0.0
    rules_extracted: int = 0
    threat_updates: int = 0
    preference_updates: int = 0
    semantic_entries_written: int = 0
    compression_removed: int = 0
    llm_used: bool = False
    rule_ids: list[str] = field(default_factory=list)
    counterfactual_episodes_evaluated: int = 0
    counterfactual_insights_generated: int = 0
    counterfactual_insights_absorbed: int = 0
    counterfactual_energy_spent: float = 0.0
    # M2.4: structured log of counterfactual reasoning (absorption/rejection entries).
    counterfactual_log: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class SleepRule:
    rule_id: str
    type: str
    cluster: int
    action: str
    observed_outcome: str
    confidence: float
    support: int
    average_surprise: float
    average_prediction_error: float
    timestamp: int
    narrative_insight: str = ""


@dataclass(frozen=True)
class SemanticMemoryEntry:
    rule_id: str
    rule_type: str
    cluster: int
    action: str
    confidence: float
    timestamp: int
    observed_outcome: str = "neutral"
    support: int = 1


@dataclass(frozen=True)
class ModelUpdate:
    update_type: str
    cluster: int
    action: str
    delta: float
    target: str
    rule_id: str


@dataclass(frozen=True)
class SleepConsolidationResult:
    rules: list[SleepRule]
    semantic_memory_entries: list[SemanticMemoryEntry]
    model_updates: list[ModelUpdate]
    llm_used: bool = False
    rules_before_llm: list[SleepRule] = field(default_factory=list)


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
    policy_bias: float
    epistemic_bonus: float
    identity_bias: float
    goal_alignment: float
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
            "policy_bias": self.policy_bias,
            "epistemic_bonus": self.epistemic_bonus,
            "identity_bias": self.identity_bias,
            "goal_alignment": self.goal_alignment,
        }


@dataclass
class DecisionDiagnostics:
    chosen: InterventionScore
    ranked_options: list[InterventionScore]
    prediction_error: float
    retrieved_memories: list[dict[str, object]]
    policy_scores: dict[str, float]
    explanation: str
    active_goal: str = ""
    goal_context: dict[str, object] = field(default_factory=dict)
    structured_explanation: dict[str, object] = field(default_factory=dict)
