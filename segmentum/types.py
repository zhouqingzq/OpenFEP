from __future__ import annotations

from dataclasses import dataclass, field

from .action_schema import ActionSchema, action_name

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
    consolidation_metrics: ConsolidationMetrics | None = None
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
class SequenceStep:
    """Single event constraint inside a learned sequence rule."""

    action_name: str
    outcome: str
    context_predicates: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_name": self.action_name,
            "outcome": self.outcome,
        }
        if self.context_predicates:
            payload["context_predicates"] = dict(self.context_predicates)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SequenceStep":
        predicates = payload.get("context_predicates")
        return cls(
            action_name=str(payload.get("action_name", "")),
            outcome=str(payload.get("outcome", "neutral")),
            context_predicates=(
                {str(key): str(value) for key, value in predicates.items()}
                if isinstance(predicates, dict)
                else None
            ),
        )


@dataclass(frozen=True)
class SequenceCondition:
    """History constraints required for a sequence-pattern rule to trigger."""

    steps: list[SequenceStep]
    window_ticks: int
    min_occurrences: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "window_ticks": self.window_ticks,
            "min_occurrences": self.min_occurrences,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SequenceCondition":
        raw_steps = payload.get("steps", [])
        return cls(
            steps=[
                SequenceStep.from_dict(step)
                for step in raw_steps
                if isinstance(step, dict)
            ],
            window_ticks=int(payload.get("window_ticks", 0)),
            min_occurrences=max(1, int(payload.get("min_occurrences", 1))),
        )


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
    action_descriptor: dict[str, object] | None = None
    sequence_condition: SequenceCondition | None = None

    @property
    def rule_type(self) -> str:
        return self.type

    @property
    def action_schema(self) -> ActionSchema:
        if self.action_descriptor is not None:
            return ActionSchema.from_dict(self.action_descriptor)
        if isinstance(self.action, ActionSchema):
            return self.action
        return ActionSchema(name=action_name(self.action))

    def to_dict(self) -> dict[str, object]:
        payload = {
            "rule_id": self.rule_id,
            "type": self.type,
            "cluster": self.cluster,
            "action": self.action,
            "observed_outcome": self.observed_outcome,
            "confidence": self.confidence,
            "support": self.support,
            "average_surprise": self.average_surprise,
            "average_prediction_error": self.average_prediction_error,
            "timestamp": self.timestamp,
        }
        if self.narrative_insight:
            payload["narrative_insight"] = self.narrative_insight
        if self.action_descriptor is not None:
            payload["action_descriptor"] = dict(self.action_descriptor)
        if self.sequence_condition is not None:
            payload["sequence_condition"] = self.sequence_condition.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SleepRule":
        sequence_payload = payload.get("sequence_condition")
        return cls(
            rule_id=str(payload.get("rule_id", "")),
            type=str(payload.get("type", payload.get("rule_type", "risk_pattern"))),
            cluster=int(payload.get("cluster", 0)),
            action=action_name(payload.get("action", "")),
            observed_outcome=str(payload.get("observed_outcome", "neutral")),
            confidence=float(payload.get("confidence", 0.0)),
            support=max(1, int(payload.get("support", 1))),
            average_surprise=float(payload.get("average_surprise", 0.0)),
            average_prediction_error=float(payload.get("average_prediction_error", 0.0)),
            timestamp=int(payload.get("timestamp", 0)),
            narrative_insight=str(payload.get("narrative_insight", "")),
            action_descriptor=(
                dict(payload.get("action_descriptor"))
                if isinstance(payload.get("action_descriptor"), dict)
                else None
            ),
            sequence_condition=(
                SequenceCondition.from_dict(sequence_payload)
                if isinstance(sequence_payload, dict)
                else None
            ),
        )


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
class ClusterPE:
    """Prediction error breakdown for a single cluster during sleep."""

    cluster_id: int
    action: str
    pe_before: float
    pe_after: float
    episode_count: int
    has_rule: bool

    @property
    def delta(self) -> float:
        return self.pe_after - self.pe_before

    @property
    def reduction_ratio(self) -> float:
        if self.pe_before <= 0.0:
            return 0.0
        return 1.0 - self.pe_after / self.pe_before


@dataclass
class ConsolidationMetrics:
    """Conditioned PE metrics that isolate learning signal from drift noise.

    Instead of comparing a raw global mean PE before/after sleep, this
    breaks the measurement into per-cluster-action slices, applies a
    novelty baseline correction, and tracks a windowed history so that
    long-run convergence can be assessed.
    """

    # --- per-cluster conditioned PE ---
    cluster_pe: list[ClusterPE] = field(default_factory=list)

    # --- aggregated conditioned PE (only clusters that have rules) ---
    conditioned_pe_before: float = 0.0
    conditioned_pe_after: float = 0.0

    # --- novelty-baseline-normalised PE ---
    novelty_baseline: float = 0.0
    normalised_pe_before: float = 0.0
    normalised_pe_after: float = 0.0

    # --- windowed rolling average (last N sleep cycles) ---
    windowed_pe_history: list[float] = field(default_factory=list)
    windowed_pe_mean: float = 0.0

    # --- raw (kept for backward compatibility) ---
    raw_pe_before: float = 0.0
    raw_pe_after: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_pe": [
                {
                    "cluster_id": c.cluster_id,
                    "action": c.action,
                    "pe_before": round(c.pe_before, 6),
                    "pe_after": round(c.pe_after, 6),
                    "delta": round(c.delta, 6),
                    "reduction_ratio": round(c.reduction_ratio, 6),
                    "episode_count": c.episode_count,
                    "has_rule": c.has_rule,
                }
                for c in self.cluster_pe
            ],
            "conditioned_pe_before": round(self.conditioned_pe_before, 6),
            "conditioned_pe_after": round(self.conditioned_pe_after, 6),
            "novelty_baseline": round(self.novelty_baseline, 6),
            "normalised_pe_before": round(self.normalised_pe_before, 6),
            "normalised_pe_after": round(self.normalised_pe_after, 6),
            "windowed_pe_history": [round(v, 6) for v in self.windowed_pe_history],
            "windowed_pe_mean": round(self.windowed_pe_mean, 6),
            "raw_pe_before": round(self.raw_pe_before, 6),
            "raw_pe_after": round(self.raw_pe_after, 6),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "ConsolidationMetrics":
        if not payload:
            return cls()
        cluster_pe_raw = payload.get("cluster_pe", [])
        cluster_pe = [
            ClusterPE(
                cluster_id=int(c.get("cluster_id", 0)),
                action=str(c.get("action", "")),
                pe_before=float(c.get("pe_before", 0.0)),
                pe_after=float(c.get("pe_after", 0.0)),
                episode_count=int(c.get("episode_count", 0)),
                has_rule=bool(c.get("has_rule", False)),
            )
            for c in cluster_pe_raw
            if isinstance(c, dict)
        ]
        return cls(
            cluster_pe=cluster_pe,
            conditioned_pe_before=float(payload.get("conditioned_pe_before", 0.0)),
            conditioned_pe_after=float(payload.get("conditioned_pe_after", 0.0)),
            novelty_baseline=float(payload.get("novelty_baseline", 0.0)),
            normalised_pe_before=float(payload.get("normalised_pe_before", 0.0)),
            normalised_pe_after=float(payload.get("normalised_pe_after", 0.0)),
            windowed_pe_history=[
                float(v) for v in payload.get("windowed_pe_history", [])
            ],
            windowed_pe_mean=float(payload.get("windowed_pe_mean", 0.0)),
            raw_pe_before=float(payload.get("raw_pe_before", 0.0)),
            raw_pe_after=float(payload.get("raw_pe_after", 0.0)),
        )


@dataclass
class InterventionScore:
    choice: str
    action_descriptor: dict[str, object]
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
    workspace_bias: float
    social_bias: float
    commitment_bias: float
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
            "workspace_bias": self.workspace_bias,
            "social_bias": self.social_bias,
            "commitment_bias": self.commitment_bias,
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
    memory_hit: bool = False
    retrieved_episode_ids: list[str] = field(default_factory=list)
    memory_context_summary: str = ""
    prediction_before_memory: dict[str, float] = field(default_factory=dict)
    prediction_after_memory: dict[str, float] = field(default_factory=dict)
    prediction_delta: dict[str, float] = field(default_factory=dict)
    attention_selected_channels: list[str] = field(default_factory=list)
    attention_dropped_channels: list[str] = field(default_factory=list)
    attention_salience_scores: dict[str, float] = field(default_factory=dict)
    workspace_broadcast_channels: list[str] = field(default_factory=list)
    workspace_suppressed_channels: list[str] = field(default_factory=list)
    workspace_broadcast_intensity: float = 0.0
    current_commitments: list[str] = field(default_factory=list)
    commitment_focus: list[str] = field(default_factory=list)
    violated_commitments: list[str] = field(default_factory=list)
    identity_tension: float = 0.0
    identity_repair_policy: str = ""
    social_focus: list[str] = field(default_factory=list)
    social_alerts: list[str] = field(default_factory=list)
    social_snapshot: dict[str, object] = field(default_factory=dict)
