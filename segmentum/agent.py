from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from statistics import mean, pvariance
from typing import Callable

from .action_schema import ActionSchema, action_name, ensure_action_schema
from .action_registry import ActionRegistry, build_default_action_registry
from .attention import AttentionBottleneck, AttentionTrace
from .constants import ACTION_BODY_EFFECTS, ACTION_COSTS
from .drives import DriveSystem, ProcessValenceState, StrategicLayer
from .environment import Observation, clamp
from .memory import (
    AutobiographicalMemory,
    LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
    LongTermMemory,
    MemoryDecision,
    _flatten_state_snapshot,
    compute_prediction_error,
    compute_total_surprise,
    suppress_legacy_memory_warnings,
)
from .memory_retrieval import RetrievalQuery
from .memory_state import AgentStateVector, MemoryAwareAgentMixin
from .m4_cognitive_style import CognitiveStyleParameters
from .memory_encoding import SalienceConfig
from .narrative_types import EmbodiedNarrativeEpisode
from .predictive_coding import (
    HierarchicalInference,
    InteroceptiveLayer,
    PredictiveCodingHyperparameters,
    apply_schema_conditioned_prediction,
    compose_upstream_observation,
    default_predictive_coding_hyperparameters,
)
from .counterfactual import CounterfactualInsight, CounterfactualLearning, run_counterfactual_phase
from .preferences import Goal, GoalStack
from .narrative_compiler import NarrativeCompiler
from .narrative_uncertainty import UncertaintyDecompositionResult
from .narrative_experiment import ExperimentDesignResult, NarrativeExperimentDesigner
from .inquiry_scheduler import (
    InquiryBudgetScheduler,
    apply_scheduler_to_experiment_design,
    semantic_uncertainty_priority_bonus,
)
from .social_model import SocialMemory
from .subject_state import SubjectState, derive_subject_state, subject_action_bias, subject_memory_threshold_delta
from .prediction_ledger import PredictionLedger
from .reconciliation import ReconciliationEngine
from .slow_learning import SlowVariableLearner
from .verification import VerificationLoop
from .verification import semantic_priority_adjustment
from .self_model import (
    CapabilityModel,
    NarrativePriors,
    PersonalitySignal,
    SelfModel,
    build_default_self_model,
)
from .sleep_consolidator import SleepConsolidation, SleepConsolidator
from .types import (
    ClusterPE,
    ConsolidationMetrics,
    DecisionDiagnostics,
    DreamReplay,
    InterventionScore,
    MemoryEpisode,
    SemanticMemoryEntry,
    SleepConsolidationResult,
    SleepRule,
    SleepSummary,
)
from .world_model import GenerativeWorldModel
from .precision_manipulation import PrecisionManipulator
from .defense_strategy import DefenseStrategySelector
from .metacognitive import MetaCognitiveLayer
from .workspace import GlobalWorkspace, GlobalWorkspaceState


def observation_dict(observation: Observation) -> dict[str, float]:
    return asdict(observation)


def _deserialize_sleep_summary(payload: dict) -> SleepSummary:
    """Reconstruct a SleepSummary from a dict, handling ConsolidationMetrics."""
    data = dict(payload)
    cm_raw = data.pop("consolidation_metrics", None)
    cm = ConsolidationMetrics.from_dict(cm_raw) if isinstance(cm_raw, dict) else None
    return SleepSummary(**data, consolidation_metrics=cm)


@dataclass(frozen=True)
class IdentityTraits:
    risk_aversion: float = 0.65
    resource_conservatism: float = 0.55


class PolicyEvaluator:
    """Score candidate actions with explicit policy components."""

    def __init__(
        self,
        identity_traits: IdentityTraits,
        self_model: SelfModel,
        goal_stack: GoalStack,
        slow_variable_learner: SlowVariableLearner,
    ) -> None:
        self.identity_traits = identity_traits
        self.self_model = self_model
        self.goal_stack = goal_stack
        self.slow_variable_learner = slow_variable_learner

    def identity_bias(
        self,
        *,
        action: str,
        projected_state: dict[str, float],
        predicted_outcome: dict[str, float],
        cost: float,
    ) -> float:
        danger = projected_state.get("danger", 0.0)
        food = projected_state.get("food", 0.0)
        shelter = projected_state.get("shelter", 0.0)
        social = projected_state.get("social", 0.0)
        energy_delta = predicted_outcome.get("energy_delta", 0.0) - cost
        stress_delta = predicted_outcome.get("stress_delta", 0.0)
        fatigue_delta = predicted_outcome.get("fatigue_delta", 0.0)
        thermal_offset = abs(projected_state.get("temperature", 0.5) - 0.5)

        risk_bias = self.identity_traits.risk_aversion * (
            (0.45 - danger) * 0.80
            + shelter * 0.15
            - max(0.0, stress_delta) * 0.35
        )
        resource_bias = self.identity_traits.resource_conservatism * (
            energy_delta * 1.20
            - max(0.0, fatigue_delta) * 0.25
            - thermal_offset * 0.20
        )

        policy_memory_bias = 0.0
        policies = self.self_model.preferred_policies
        if policies is not None:
            if action in policies.learned_preferences:
                policy_memory_bias += 0.25
            if action in policies.learned_avoidances:
                policy_memory_bias -= 0.35
            frequency = float(policies.action_distribution.get(action, 0.0))
            policy_memory_bias += (frequency - 0.20) * 0.30

        threat_bias = self._threat_awareness_bias(action, cost)
        narrative_bias = self._narrative_bias(action)
        narrative_priors = self.self_model.narrative_priors
        prior_bias = 0.0
        if action in ("hide", "rest", "exploit_shelter"):
            prior_bias += max(0.0, narrative_priors.trauma_bias) * danger * 0.35
            if narrative_priors.controllability_prior > 0.0 and danger < 0.45:
                prior_bias -= narrative_priors.controllability_prior * (
                    0.20 + max(0.0, food - 0.50) * 0.30
                )
            if narrative_priors.trust_prior > 0.0 and social > 0.55:
                prior_bias -= narrative_priors.trust_prior * (
                    0.12 + max(0.0, social - 0.55) * 0.35
                )
        if action == "forage":
            prior_bias -= max(0.0, narrative_priors.trauma_bias) * danger * 0.45
            prior_bias -= max(0.0, narrative_priors.contamination_sensitivity) * 0.30
            if narrative_priors.controllability_prior > 0.0 and danger < 0.45:
                prior_bias += narrative_priors.controllability_prior * (
                    0.18
                    + max(0.0, food - 0.55) * 0.35
                    + max(0.0, shelter - 0.50) * 0.10
                )
        if action == "seek_contact":
            prior_bias += narrative_priors.trust_prior * (
                0.40
                + max(0.0, social - 0.50) * 0.70
                + max(0.0, 0.45 - danger) * 0.25
            )
        if action == "scan":
            prior_bias += max(0.0, -narrative_priors.controllability_prior) * 0.20
            if narrative_priors.controllability_prior > 0.0 and danger < 0.50:
                prior_bias += narrative_priors.controllability_prior * (
                    0.12 + max(0.0, 0.50 - danger) * 0.30
                )

        # M2.6: Personality-driven policy bias
        personality_bias = self.self_model.personality_profile.policy_bias(action, danger)
        slow_learning_bias = self.slow_variable_learner.action_bias(action)

        return max(
            -1.0,
            min(
                1.0,
                risk_bias
                + resource_bias
                + policy_memory_bias
                + threat_bias
                + narrative_bias
                + prior_bias
                + personality_bias
                + slow_learning_bias
            ),
        )

    def _threat_awareness_bias(self, action: str, cost: float) -> float:
        """Couple SelfModel.threat_model to action scoring."""
        tm = self.self_model.threat_model
        rs = self.self_model.resource_state
        bs = self.self_model.body_schema

        token_prox = (
            tm.token_exhaustion_threshold / max(1, rs.tokens_remaining)
            if tm.token_exhaustion_threshold > 0
            else 0.0
        )
        memory_prox = (
            tm.memory_overflow_threshold / max(1.0, rs.memory_free)
            if tm.memory_overflow_threshold > 0
            else 0.0
        )
        sensitivity = max(token_prox, memory_prox)

        if sensitivity <= 0.3:
            return 0.0

        # Penalise costly actions proportional to threat sensitivity.
        penalty = -sensitivity * cost * 4.0
        # Bonus for the cheapest action (rest) when near resource limits.
        if action == "rest":
            penalty += sensitivity * 0.35
        return penalty

    def _narrative_bias(self, action: str) -> float:
        """Couple IdentityNarrative to action scoring."""
        narrative = self.self_model.identity_narrative
        if narrative is None:
            return 0.0

        bias = 0.0
        action_lower = action.lower()
        for pattern in narrative.behavioral_patterns:
            if action_lower in pattern.lower():
                bias += 0.25
        if narrative.core_identity and action_lower in narrative.core_identity.lower():
            bias += 0.20

        # Risk-profile coupling: a risk-seeking identity penalises
        # overly cautious actions and favours active ones.
        if narrative.core_identity and "risk-seeking" in narrative.core_identity.lower():
            if action in ("hide", "rest"):
                bias -= 0.25
            elif action in ("forage", "scan", "exploit_shelter"):
                bias += 0.15

        return max(-0.6, min(0.6, bias))

    def commitment_assessment(
        self,
        *,
        action: str,
        projected_state: dict[str, float],
    ) -> dict[str, object]:
        return self.self_model.assess_action_commitments(
            action=action,
            projected_state=projected_state,
        )

    def dominant_component(
        self,
        *,
        expected_free_energy: float,
        memory_bias: float,
        pattern_bias: float,
        policy_bias: float,
        epistemic_bonus: float,
        workspace_bias: float,
        social_bias: float,
        commitment_bias: float,
        identity_bias: float,
        ledger_bias: float,
        subject_bias: float,
        goal_alignment: float,
        reconciliation_bias: float = 0.0,
        verification_bias: float = 0.0,
        experiment_bias: float = 0.0,
        inquiry_scheduler_bias: float = 0.0,
    ) -> str:
        components = [
            ("expected_free_energy", abs(expected_free_energy)),
            ("memory_bias", abs(memory_bias)),
            ("pattern_bias", abs(pattern_bias)),
            ("policy_bias", abs(policy_bias)),
            ("epistemic_bonus", abs(epistemic_bonus)),
            ("workspace_bias", abs(workspace_bias)),
            ("social_bias", abs(social_bias)),
            ("commitment_bias", abs(commitment_bias)),
            ("identity_bias", abs(identity_bias)),
            ("ledger_bias", abs(ledger_bias)),
            ("subject_bias", abs(subject_bias)),
            ("goal_alignment", abs(goal_alignment)),
            ("reconciliation_bias", abs(reconciliation_bias)),
            ("verification_bias", abs(verification_bias)),
            ("experiment_bias", abs(experiment_bias)),
            ("inquiry_scheduler_bias", abs(inquiry_scheduler_bias)),
        ]
        components.sort(key=lambda item: (-item[1], item[0]))
        return components[0][0]

    def explain_structured(
        self,
        diagnostics: DecisionDiagnostics,
        action: str | None = None,
    ) -> dict[str, object]:
        if action is None:
            chosen = diagnostics.chosen
        else:
            chosen = next(
                (
                    option
                    for option in diagnostics.ranked_options
                    if option.choice == action
                ),
                diagnostics.chosen,
            )
        alternative = next(
            (
                option
                for option in diagnostics.ranked_options
                if option.choice != chosen.choice
            ),
            None,
        )
        if chosen.dominant_component == "memory_bias":
            reason = (
                f"memory_bias ({chosen.memory_bias:.3f}) was the strongest term, "
                "so similar episodes outweighed more speculative alternatives."
            )
        elif chosen.dominant_component == "pattern_bias":
            reason = (
                f"pattern_bias ({chosen.pattern_bias:.3f}) dominated, "
                "reflecting recurring episodic patterns for this action."
            )
        elif chosen.dominant_component == "policy_bias":
            reason = (
                f"policy_bias ({chosen.policy_bias:.3f}) dominated, "
                "reflecting a sleep-consolidated bias learned from repeated state-action outcomes."
            )
        elif chosen.dominant_component == "epistemic_bonus":
            reason = (
                f"epistemic_bonus ({chosen.epistemic_bonus:.3f}) dominated, "
                "so unresolved ambiguity in this state encouraged information-seeking exploration."
            )
        elif chosen.dominant_component == "workspace_bias":
            reason = (
                f"workspace_bias ({chosen.workspace_bias:.3f}) dominated, "
                "so globally broadcast contents overruled weaker local preferences."
            )
        elif chosen.dominant_component == "social_bias":
            reason = (
                f"social_bias ({chosen.social_bias:.3f}) dominated, "
                "so learned expectations about others constrained the local policy."
            )
        elif chosen.dominant_component == "commitment_bias":
            reason = (
                f"commitment_bias ({chosen.commitment_bias:.3f}) dominated, "
                "so identity commitments constrained the policy beyond short-term convenience."
            )
        elif chosen.dominant_component == "identity_bias":
            reason = (
                f"identity_bias ({chosen.identity_bias:.3f}) dominated, "
                "which matches my learned preferences and autobiographical style."
            )
        elif chosen.dominant_component == "ledger_bias":
            reason = (
                f"ledger_bias ({chosen.ledger_bias:.3f}) dominated, "
                "so unresolved predictions and discrepancy burdens constrained the policy."
            )
        elif chosen.dominant_component == "subject_bias":
            reason = (
                f"subject_bias ({chosen.subject_bias:.3f}) dominated, "
                "so my current unified subject state overruled weaker local advantages."
            )
        elif chosen.dominant_component == "goal_alignment":
            reason = (
                f"goal_alignment ({chosen.goal_alignment:.3f}) dominated, "
                f"so the action best served my active goal {diagnostics.active_goal}."
            )
        elif chosen.dominant_component == "verification_bias":
            reason = (
                f"verification_bias ({chosen.verification_bias:.3f}) dominated, "
                "so the policy prioritized gathering evidence for an active prediction."
            )
        elif chosen.dominant_component == "experiment_bias":
            reason = (
                f"experiment_bias ({chosen.experiment_bias:.3f}) dominated, "
                "so narrative hypothesis testing became the main policy pressure."
            )
        elif chosen.dominant_component == "inquiry_scheduler_bias":
            reason = (
                f"inquiry_scheduler_bias ({chosen.inquiry_scheduler_bias:.3f}) dominated, "
                "so bounded inquiry scheduling overruled weaker curiosity pressures."
            )
        elif chosen.dominant_component == "reconciliation_bias":
            reason = (
                f"reconciliation_bias ({chosen.reconciliation_bias:.3f}) dominated, "
                "so a long-running unresolved conflict constrained the local policy."
            )
        else:
            reason = (
                f"expected free energy ({chosen.expected_free_energy:.3f}) stayed lowest, "
                "so risk, ambiguity, and predicted error were minimized."
            )

        if chosen.preferred_probability >= 0.50:
            probability_band = "high"
        elif chosen.preferred_probability >= 0.10:
            probability_band = "moderate"
        else:
            probability_band = "low"
        if chosen.risk <= 1.0:
            risk_band = "low"
        elif chosen.risk <= 3.0:
            risk_band = "moderate"
        else:
            risk_band = "high"

        comparison = ""
        if alternative is not None:
            if chosen.expected_free_energy <= alternative.expected_free_energy:
                comparison = (
                    f" The overall expected free energy was lower than {alternative.choice} "
                    f"({chosen.expected_free_energy:.3f} vs {alternative.expected_free_energy:.3f}), "
                    f"and {chosen.choice} scored {chosen.policy_score:.3f} versus "
                    f"{alternative.choice} at {alternative.policy_score:.3f}."
                )
            else:
                comparison = (
                    f" Even though {alternative.choice} had lower expected free energy "
                    f"({alternative.expected_free_energy:.3f} vs {chosen.expected_free_energy:.3f}), "
                    f"{chosen.choice} still achieved the strongest final policy score "
                    f"({chosen.policy_score:.3f} vs {alternative.policy_score:.3f}) "
                    f"after memory, pattern, identity, and goal terms were included."
                )

        policies = self.self_model.preferred_policies
        narrative = self.self_model.identity_narrative
        historical_frequency = 0.0
        dominant_strategy = "expected_free_energy"
        typical_actions: set[str] = set()
        if narrative is not None:
            for pattern in narrative.behavioral_patterns:
                if "tend to " not in pattern:
                    continue
                typical_actions.add(pattern.split("tend to ", 1)[1].split(" (", 1)[0].lower())
        if policies is not None:
            historical_frequency = float(policies.action_distribution.get(chosen.choice, 0.0))
            dominant_strategy = policies.dominant_strategy or dominant_strategy
        learned_preference = bool(policies is not None and chosen.choice in policies.learned_preferences)
        learned_avoidance = bool(policies is not None and chosen.choice in policies.learned_avoidances)
        identity_consistency = (
            chosen.choice.lower() in typical_actions
            or learned_preference
            or historical_frequency >= 0.20
        ) and not learned_avoidance
        matches_dominant_strategy = chosen.dominant_component == dominant_strategy

        if policies is None:
            consistency_statement = "This is an early decision, so no stable pattern has been consolidated yet."
        elif identity_consistency and matches_dominant_strategy:
            consistency_statement = (
                "This choice is consistent with my established pattern: "
                f"I {dominant_strategy} and historically choose {chosen.choice} "
                f"{historical_frequency:.0%} of the time."
            )
        else:
            consistency_statement = (
                "This choice deviates from my usual pattern "
                f"(dominant strategy: {dominant_strategy}). "
                f"Reason for deviation: {chosen.dominant_component}."
            )

        consistency_info = {
            "dominant_strategy": dominant_strategy,
            "matches_dominant_strategy": matches_dominant_strategy,
            "historical_action_frequency": historical_frequency,
            "identity_consistency": identity_consistency,
            "consistency_statement": consistency_statement,
        }

        bias_references = [
            f"memory_bias={chosen.memory_bias:.3f}",
            f"pattern_bias={chosen.pattern_bias:.3f}",
            f"policy_bias={chosen.policy_bias:.3f}",
            f"workspace_bias={chosen.workspace_bias:.3f}",
            f"social_bias={chosen.social_bias:.3f}",
            f"commitment_bias={chosen.commitment_bias:.3f}",
            f"ledger_bias={chosen.ledger_bias:.3f}",
            f"subject_bias={chosen.subject_bias:.3f}",
            f"reconciliation_bias={chosen.reconciliation_bias:.3f}",
            f"verification_bias={chosen.verification_bias:.3f}",
            f"experiment_bias={chosen.experiment_bias:.3f}",
            f"inquiry_scheduler_bias={chosen.inquiry_scheduler_bias:.3f}",
        ]
        bias_sentence = " Supporting continuity terms: " + ", ".join(bias_references) + "."
        workspace_channels = ", ".join(diagnostics.workspace_broadcast_channels)
        workspace_sentence = ""
        if workspace_channels:
            workspace_sentence = (
                f" My globally accessible focus was constrained to: {workspace_channels}. "
            )
        commitment_sentence = ""
        if diagnostics.commitment_focus:
            commitment_sentence = (
                " Active commitments shaping this choice: "
                + ", ".join(diagnostics.commitment_focus)
                + ". "
            )
        if diagnostics.violated_commitments:
            commitment_sentence += (
                "This action still carries identity tension against: "
                + ", ".join(diagnostics.violated_commitments)
                + ". "
            )
        social_sentence = ""
        if diagnostics.social_focus:
            social_sentence = (
                " Social models currently in focus: "
                + ", ".join(diagnostics.social_focus)
                + ". "
            )
        if diagnostics.social_alerts:
            social_sentence += (
                "Active social alerts: " + ", ".join(diagnostics.social_alerts) + ". "
            )
        subject_sentence = ""
        if diagnostics.subject_state_summary:
            subject_sentence = diagnostics.subject_state_summary + " "
        ledger_sentence = ""
        if diagnostics.ledger_summary:
            ledger_sentence = diagnostics.ledger_summary + " "
        experiment_sentence = ""
        if diagnostics.experiment_summary:
            experiment_sentence = "Experiment design: " + diagnostics.experiment_summary + " "
        inquiry_scheduler_sentence = ""
        if diagnostics.inquiry_scheduler_summary:
            inquiry_scheduler_sentence = (
                "Inquiry scheduler: " + diagnostics.inquiry_scheduler_summary + " "
            )
        explanation_text = (
            f"I chose {chosen.choice}. "
            f"This action predicted outcome '{chosen.predicted_outcome}'. "
            f"According to my preference model this outcome has {probability_band} "
            f"preferred probability ({chosen.preferred_probability:.2f}), "
            f"resulting in {risk_band} risk ({chosen.risk:.3f}). "
            f"My active goal is {diagnostics.active_goal}, so I favored actions aligned with it. "
            f"{reason}{comparison}"
            f"{workspace_sentence}"
            f"{commitment_sentence}"
            f"{social_sentence}"
            f"{ledger_sentence}"
            f"{experiment_sentence}"
            f"{inquiry_scheduler_sentence}"
            f"{subject_sentence}"
            f"{consistency_statement}{bias_sentence} "
            f"This aligns with my resource_conservatism="
            f"{self.identity_traits.resource_conservatism:.2f} and "
            f"risk_aversion={self.identity_traits.risk_aversion:.2f}."
        )
        return {
            "action": chosen.choice,
            "active_goal": diagnostics.active_goal,
            "goal_alignment": chosen.goal_alignment,
            "dominant_component": chosen.dominant_component,
            "historical_action_frequency": historical_frequency,
            "identity_consistency": identity_consistency,
            "consistency": consistency_info,
            "reason": reason,
            "comparison": comparison.strip(),
            "workspace_focus": list(diagnostics.workspace_broadcast_channels),
            "workspace_intensity": diagnostics.workspace_broadcast_intensity,
            "current_commitments": list(diagnostics.current_commitments),
            "relevant_commitments": list(diagnostics.relevant_commitments),
            "commitment_focus": list(diagnostics.commitment_focus),
            "violated_commitments": list(diagnostics.violated_commitments),
            "commitment_compatibility_score": diagnostics.commitment_compatibility_score,
            "self_inconsistency_error": diagnostics.self_inconsistency_error,
            "conflict_type": diagnostics.conflict_type,
            "severity_level": diagnostics.severity_level,
            "consistency_classification": diagnostics.consistency_classification,
            "behavioral_classification": diagnostics.behavioral_classification,
            "repair_triggered": diagnostics.repair_triggered,
            "repair_policy": diagnostics.repair_policy,
            "repair_result": dict(diagnostics.repair_result),
            "identity_tension": diagnostics.identity_tension,
            "identity_repair_policy": diagnostics.identity_repair_policy,
            "social_focus": list(diagnostics.social_focus),
            "social_alerts": list(diagnostics.social_alerts),
            "social_snapshot": dict(diagnostics.social_snapshot),
            "ledger_summary": diagnostics.ledger_summary,
            "prediction_ledger": dict(diagnostics.ledger_payload),
            "subject_state_summary": diagnostics.subject_state_summary,
            "subject_status_flags": dict(diagnostics.subject_status_flags),
            "subject_priority_stack": list(diagnostics.subject_priority_stack),
            "inquiry_scheduler_summary": diagnostics.inquiry_scheduler_summary,
            "inquiry_scheduler": dict(diagnostics.inquiry_scheduler_payload),
            "text": explanation_text,
        }

    def explain(
        self,
        diagnostics: DecisionDiagnostics,
        action: str | None = None,
    ) -> str:
        return str(self.explain_structured(diagnostics, action=action)["text"])

class DecisionLoop:
    """Explicit M2 decision loop surface for structured phase tracking."""

    phase_names = (
        "perception",
        "prediction",
        "memory_retrieval",
        "goal_alignment_evaluation",
        "policy_evaluation",
        "action_selection",
    )

    def describe(self) -> list[str]:
        return list(self.phase_names)


class SegmentAgent(MemoryAwareAgentMixin):
    """A survival-first digital segment with drives, long-term memory, and dream replay."""

    def __init__(
        self,
        rng: random.Random | None = None,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        sleep_llm_extractor: Callable[[list[SleepRule], list[dict[str, object]]], list[SleepRule]]
        | None = None,
        action_registry: ActionRegistry | None = None,
        memory_backend: str = "memory_store",
        memory_cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
        memory_cycle_interval: int = 5,
        memory_salience_config: SalienceConfig | None = None,
        memory_enabled: bool = True,
    ) -> None:
        self.rng = rng or random.Random()
        self.energy = 0.80
        self.stress = 0.25
        self.fatigue = 0.20
        self.temperature = 0.48
        self.dopamine = 0.12
        self.cycle = 0

        self.drive_system = DriveSystem()
        self.strategic_layer = StrategicLayer()
        self.interoceptive_layer = InteroceptiveLayer()
        self.world_model = GenerativeWorldModel()
        self.action_registry = action_registry or build_default_action_registry()
        self.long_term_memory = AutobiographicalMemory()
        self.autobiographical_memory = self.long_term_memory
        self.long_term_memory.memory_backend = self.long_term_memory._normalize_memory_backend(memory_backend)
        self.identity_traits = IdentityTraits()
        self.self_model = build_default_self_model()
        self.self_model.capability_model = CapabilityModel(
            action_schemas=tuple(self.action_registry.get_all()),
            api_limits=self.self_model.capability_model.api_limits,
        )
        self.goal_stack = GoalStack()
        self.goal_stack.log_sink = self.self_model.log_sink
        self.slow_variable_learner = SlowVariableLearner()
        self.policy_evaluator = PolicyEvaluator(
            self.identity_traits,
            self.self_model,
            self.goal_stack,
            self.slow_variable_learner,
        )
        self.decision_loop = DecisionLoop()
        self.counterfactual_learning = CounterfactualLearning()
        self.sleep_llm_extractor = sleep_llm_extractor

        self.episodes: list[MemoryEpisode] = []
        self.semantic_memory: list[SemanticMemoryEntry] = []
        self.sleep_history: list[SleepSummary] = []
        self.action_history: list[str] = []
        self.action_history_limit = 256
        self.decision_history: list[dict[str, object]] = []
        self.decision_history_limit = 512
        self.drive_history: list[dict[str, float]] = []
        self.drive_history_limit = 128
        self.free_energy_history: list[float] = []
        self.free_energy_history_limit = 64
        self.narrative_trace: list[dict[str, object]] = []
        self.last_body_state_snapshot = {
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
        }
        self.last_decision_diagnostics: DecisionDiagnostics | None = None
        self.last_decision_choice: str = ""
        self.last_decision_observation: dict[str, float] = {}
        self.last_memory_context: dict[str, object] = {}
        self.memory_enabled: bool = memory_enabled
        self._sleeping = False
        self.counterfactual_insights: list[CounterfactualInsight] = []
        self.init_memory_awareness(
            memory_store=self.long_term_memory.ensure_memory_store(),
            state_vector=AgentStateVector.from_dict(dict(self.long_term_memory.agent_state_vector)),
            cognitive_style=memory_cognitive_style or dict(self.long_term_memory.memory_cognitive_style),
            memory_cycle_interval=memory_cycle_interval or self.long_term_memory.memory_cycle_interval,
            salience_config=memory_salience_config,
        )
        self.sync_memory_awareness_to_long_term_memory()

        # M2.7: Precision manipulation, defense strategy, and metacognitive layer
        self.precision_manipulator = PrecisionManipulator()
        self.defense_strategy_selector = DefenseStrategySelector(self.precision_manipulator)
        self.metacognitive_layer = MetaCognitiveLayer()
        self.attention_bottleneck = AttentionBottleneck()
        self.last_attention_trace: AttentionTrace | None = None
        self.last_attention_filtered_observation: dict[str, float] = {}
        self.global_workspace = GlobalWorkspace()
        self.last_workspace_state: GlobalWorkspaceState | None = None
        self.social_memory = SocialMemory()
        self.identity_tension_history: list[dict[str, object]] = []
        self.prediction_ledger = PredictionLedger()
        self.reconciliation_engine = ReconciliationEngine()
        self.verification_loop = VerificationLoop()
        self.subject_state = SubjectState()
        self.latest_narrative_uncertainty = UncertaintyDecompositionResult()
        self.narrative_uncertainty_history: list[dict[str, object]] = []
        self.narrative_experiment_designer = NarrativeExperimentDesigner()
        self.latest_narrative_experiment = ExperimentDesignResult()
        self.narrative_experiment_history: list[dict[str, object]] = []
        self.inquiry_budget_scheduler = InquiryBudgetScheduler()

        self.base_metabolic_rate = 0.015
        self.fatigue_accumulation_rate = 0.08
        self.configure_predictive_coding(
            predictive_hyperparameters or default_predictive_coding_hyperparameters(),
            reset_precisions=True,
        )
        self._sync_self_model_body_schema()

    def sync_memory_awareness_to_long_term_memory(self) -> None:
        self.memory_store = self.long_term_memory.memory_store
        self.long_term_memory.agent_state_vector = self.agent_state_vector.to_dict()
        self.long_term_memory.memory_cognitive_style = self.memory_cognitive_style.to_dict()
        self.long_term_memory.memory_cycle_interval = self.memory_cycle_interval
        self.long_term_memory.memory_backend = self._active_memory_backend()

    def _active_memory_backend(self) -> str:
        return self.long_term_memory._normalize_memory_backend(self.long_term_memory.memory_backend)

    def _decision_retrieval_query(
        self,
        observed: dict[str, float],
        baseline_prediction: dict[str, float],
        baseline_errors: dict[str, float],
    ) -> RetrievalQuery:
        semantic_tags: list[str] = []
        context_tags = [
            key
            for key, value in {**observed, **self._current_body_state()}.items()
            if abs(float(value)) >= 0.15
        ][:8]
        content_keywords = [
            key
            for key, _ in sorted(
                {**observed, **baseline_errors}.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            if key
        ][:8]
        return RetrievalQuery(
            semantic_tags=semantic_tags,
            context_tags=context_tags,
            content_keywords=content_keywords,
            state_vector=self.long_term_memory._build_embedding(
                _flatten_state_snapshot(
                    {
                        "observation": dict(observed),
                        "prediction": dict(baseline_prediction),
                        "errors": dict(baseline_errors),
                        "body_state": self._current_body_state(),
                    }
                )
            ),
            reference_cycle=self.cycle,
        )

    def _retrieve_decision_memories(
        self,
        *,
        observed: dict[str, float],
        baseline_prediction: dict[str, float],
        baseline_errors: dict[str, float],
        current_state_snapshot: dict[str, object],
        k: int,
    ) -> list[dict[str, object]]:
        if not self.memory_enabled:
            self.long_term_memory.last_retrieval_result = {
                "memory_backend": self._active_memory_backend(),
                "memory_enabled": False,
                "candidates": [],
                "recall_hypothesis": {},
                "reconstruction_trace": {},
            }
            return []
        if self._active_memory_backend() != "memory_store":
            with suppress_legacy_memory_warnings():
                return self.long_term_memory.retrieve_similar_memories(
                    current_state_snapshot,
                    k=k,
                )
        self.sync_memory_awareness_to_long_term_memory()
        query = self._decision_retrieval_query(observed, baseline_prediction, baseline_errors)
        result = self.retrieve_for_decision(
            query,
            self.cycle,
            current_mood=self.agent_state_vector.recent_mood_baseline,
            k=k,
        )
        projected_by_id = {
            str(payload.get("episode_id", "")): payload
            for payload in self.memory_store.to_legacy_episodes()
        }
        similar_memories: list[dict[str, object]] = []
        for candidate in result.candidates[:k]:
            payload = dict(projected_by_id.get(candidate.entry_id, {}))
            payload.setdefault("episode_id", candidate.entry_id)
            payload.setdefault("content", candidate.entry.content)
            payload["similarity"] = float(candidate.retrieval_score)
            payload["vector_similarity"] = float(candidate.score_breakdown.get("context_overlap", 0.0))
            payload["schema_overlap"] = float(candidate.score_breakdown.get("tag_overlap", 0.0))
            payload["memory_class"] = candidate.memory_class
            payload["source_type"] = candidate.source_type
            payload["validation_status"] = candidate.validation_status
            payload["retrieval_score_breakdown"] = dict(candidate.score_breakdown)
            similar_memories.append(payload)
        self.long_term_memory.last_retrieval_result = {
            "memory_backend": "memory_store",
            **result.to_dict(),
        }
        return similar_memories

    def configure_attention_bottleneck(
        self,
        *,
        enabled: bool,
        capacity: int = 3,
        novelty_weight: float = 0.2,
        threat_weight: float = 0.35,
        surprise_weight: float = 0.45,
    ) -> None:
        self.attention_bottleneck = AttentionBottleneck(
            enabled=enabled,
            capacity=capacity,
            novelty_weight=novelty_weight,
            threat_weight=threat_weight,
            surprise_weight=surprise_weight,
        )

    def attention_state(self) -> dict[str, object]:
        state = self.attention_bottleneck.to_dict()
        if self.last_attention_trace is not None:
            state["last_trace"] = self.last_attention_trace.to_dict()
        if self.last_attention_filtered_observation:
            state["last_filtered_observation"] = dict(self.last_attention_filtered_observation)
        return state

    def _refresh_inquiry_budget(self) -> None:
        self._refresh_process_valence()
        state = self.inquiry_budget_scheduler.schedule(
            tick=self.cycle,
            narrative_uncertainty=self.latest_narrative_uncertainty,
            experiment_design=self.latest_narrative_experiment,
            prediction_ledger=self.prediction_ledger,
            verification_loop=self.verification_loop,
            subject_state=self.subject_state,
            reconciliation_engine=self.reconciliation_engine,
            process_valence_state=self.drive_system.process_valence,
        )
        self.latest_narrative_experiment = apply_scheduler_to_experiment_design(
            self.latest_narrative_experiment,
            state,
        )

    def _refresh_process_valence(self) -> None:
        unresolved_targets: set[str] = set()
        focus_id = ""
        focus_strength = 0.0
        if getattr(self.latest_narrative_experiment, "plans", ()):
            top_plan = None
            for plan in self.latest_narrative_experiment.plans:
                plan_status = str(getattr(plan, "status", ""))
                if plan_status in {"active_experiment", "queued_experiment", "deferred_for_budget"}:
                    top_plan = plan
                    break
            if top_plan is not None:
                focus_id = str(
                    getattr(top_plan, "target_unknown_id", "")
                    or getattr(top_plan, "plan_id", "")
                )
                focus_strength = max(
                    focus_strength,
                    float(getattr(top_plan, "informative_value", 0.0)),
                    float(getattr(top_plan, "score", 0.0)),
                )
        for item in getattr(self.latest_narrative_uncertainty, "unknowns", ()):
            if not getattr(item, "action_relevant", False):
                continue
            unknown_id = str(getattr(item, "unknown_id", ""))
            if unknown_id:
                unresolved_targets.add(unknown_id)
            if not focus_id and unknown_id:
                focus_id = unknown_id
                focus_strength = max(
                    focus_strength,
                    float(getattr(item, "uncertainty_level", 0.0)),
                    float(getattr(getattr(item, "decision_relevance", None), "total_score", 0.0)),
                )
        for item in getattr(self.subject_state, "active_inquiries", ()):
            target_unknown_id = str(getattr(item, "target_unknown_id", ""))
            if target_unknown_id:
                unresolved_targets.add(target_unknown_id)
                if not focus_id:
                    focus_id = target_unknown_id
                    focus_strength = max(focus_strength, float(getattr(item, "salience", 0.0)))
        for tension in getattr(self.subject_state, "unresolved_tensions", ()):
            label = str(getattr(tension, "label", ""))
            if label:
                unresolved_targets.add(label)
                if not focus_id:
                    focus_id = label
                    focus_strength = max(
                        focus_strength,
                        float(getattr(tension, "intensity", 0.0)),
                    )
        self.drive_system.update_process_valence(
            current_focus_id=focus_id,
            unresolved_targets=unresolved_targets,
            focus_strength=focus_strength,
            maintenance_pressure=float(getattr(self.subject_state, "maintenance_pressure", 0.0)),
            closure_signal=min(1.0, float(len(unresolved_targets) == 0)),
        )

    def _is_known_task(self, action: str) -> bool:
        recent_actions = self.action_history[-12:]
        repeated = recent_actions.count(action) >= 2
        preferred = bool(
            self.self_model.preferred_policies is not None
            and action in self.self_model.preferred_policies.learned_preferences
        )
        return repeated or preferred

    def _action_compute_spend(self, action: str) -> float:
        mapping = {
            "rest": 0.18,
            "hide": 0.24,
            "exploit_shelter": 0.26,
            "thermoregulate": 0.28,
            "forage": 0.48,
            "scan": 0.66,
            "seek_contact": 0.62,
        }
        return float(mapping.get(action, 0.40))

    def _record_effort_allocation(self, action: str, observation: dict[str, float]) -> None:
        uncertainty_load = max(
            float(observation.get("novelty", 0.0)),
            float(observation.get("danger", 0.0)) * 0.65,
            max(
                (
                    float(getattr(item, "uncertainty_level", 0.0))
                    for item in getattr(self.latest_narrative_uncertainty, "unknowns", ())
                ),
                default=0.0,
            ),
        )
        compression_pressure = max(
            0.0,
            min(
                1.0,
                1.0 - float(self.self_model.resource_state.memory_free) / 1024.0,
            ),
        )
        self.slow_variable_learner.record_effort_allocation(
            tick=self.cycle,
            action=action,
            known_task=self._is_known_task(action),
            compute_spend=self._action_compute_spend(action),
            uncertainty_load=uncertainty_load,
            compression_pressure=compression_pressure,
            process_pull=float(self.drive_system.process_valence.process_reward),
        )

    def configure_global_workspace(
        self,
        *,
        enabled: bool,
        capacity: int = 2,
        action_bias_gain: float = 0.35,
        memory_gate_gain: float = 0.08,
        persistence_ticks: int = 2,
        carry_over_decay: float = 0.82,
        carry_over_min_salience: float = 0.12,
        report_carry_over: bool = True,
    ) -> None:
        self.global_workspace = GlobalWorkspace(
            enabled=enabled,
            capacity=capacity,
            action_bias_gain=action_bias_gain,
            memory_gate_gain=memory_gate_gain,
            persistence_ticks=persistence_ticks,
            carry_over_decay=carry_over_decay,
            carry_over_min_salience=carry_over_min_salience,
            report_carry_over=report_carry_over,
        )

    def workspace_state(self) -> dict[str, object]:
        return self.global_workspace.to_dict()

    def _record_identity_tension(
        self,
        *,
        chosen_action: str,
        commitment_focus: list[str],
        violated_commitments: list[str],
        identity_tension: float,
        repair_policy: str,
    ) -> None:
        if not commitment_focus and not violated_commitments:
            return
        record = {
            "tick": self.cycle,
            "action": chosen_action,
            "commitment_focus": list(commitment_focus),
            "violated_commitments": list(violated_commitments),
            "identity_tension": float(identity_tension),
            "repair_policy": repair_policy,
        }
        self.identity_tension_history.append(record)
        self.identity_tension_history = self.identity_tension_history[-64:]
        self.metacognitive_layer.meta_beliefs["identity_commitment_state"] = {
            "last_tick": self.cycle,
            "focus_count": len(commitment_focus),
            "violation_count": len(violated_commitments),
            "identity_tension": float(identity_tension),
            "repair_policy": repair_policy,
        }

    def _select_repaired_option(
        self,
        *,
        ranked_options: list[InterventionScore],
        commitment_assessments: dict[str, dict[str, object]],
    ) -> tuple[InterventionScore | None, dict[str, object]]:
        if not ranked_options:
            return None, {}
        chosen = ranked_options[0]
        assessment = commitment_assessments.get(chosen.choice, {})
        review = self.metacognitive_layer.review_self_consistency(
            assessment,
            workspace_state=self.last_workspace_state,
        )
        if not review.review_required:
            return None, {
                "success": False,
                "policy": review.recommended_policy,
                "reason": "review_not_required",
            }

        candidates = sorted(
            ranked_options,
            key=lambda option: (
                float(commitment_assessments.get(option.choice, {}).get("compatibility_score", 0.5))
                + (review.rebias_strength * 0.25 if not commitment_assessments.get(option.choice, {}).get("violations") else 0.0),
                option.policy_score,
                option.choice,
            ),
            reverse=True,
        )
        repaired = candidates[0]
        repaired_assessment = commitment_assessments.get(repaired.choice, {})
        success = (
            repaired.choice != chosen.choice
            and float(repaired_assessment.get("compatibility_score", 0.0))
            > float(assessment.get("compatibility_score", 0.0))
            and not repaired_assessment.get("violations")
        )
        result = {
            "success": success,
            "policy": review.recommended_policy,
            "review_notes": review.notes,
            "pause_strength": review.pause_strength,
            "rebias_strength": review.rebias_strength,
            "target_action": chosen.choice,
            "repaired_action": repaired.choice if success else chosen.choice,
        }
        return (repaired if success else None), result

    def _mark_last_episode_identity_critical(
        self,
        *,
        reason: str,
        commitment_ids: list[str],
    ) -> None:
        if not self.long_term_memory.episodes:
            return
        payload = self.long_term_memory.episodes[-1]
        payload["identity_critical"] = True
        payload["identity_commitment_reason"] = reason
        payload["identity_commitment_ids"] = list(commitment_ids)
        lifecycle_stage = str(
            payload.get("lifecycle_stage", LIFECYCLE_PROTECTED_IDENTITY_CRITICAL)
        )
        if lifecycle_stage != LIFECYCLE_PROTECTED_IDENTITY_CRITICAL:
            payload["lifecycle_stage"] = LIFECYCLE_PROTECTED_IDENTITY_CRITICAL
        self.long_term_memory._synchronize_continuity_metadata(payload)

    def _update_social_memory_from_embodied_episode(
        self,
        embodied_episode: EmbodiedNarrativeEpisode,
    ) -> dict[str, object]:
        provenance = dict(embodied_episode.provenance)
        compiled_event = dict(provenance.get("compiled_event", {}))
        metadata = dict(provenance.get("episode_metadata", {}))
        counterpart_id = str(
            metadata.get(
                "counterpart_id",
                compiled_event.get("annotations", {}).get("counterpart_id", ""),
            )
        )
        if not counterpart_id:
            raw_actor = metadata.get("counterpart_name", "")
            counterpart_id = str(raw_actor).strip().lower().replace(" ", "_")
        if not counterpart_id:
            return {"updated": False, "snapshot": self.social_memory.snapshot()}
        model = self.social_memory.observe_counterpart(
            other_id=counterpart_id,
            tick=embodied_episode.timestamp,
            appraisal=embodied_episode.appraisal,
            metadata=metadata,
            tags=list(embodied_episode.narrative_tags),
            event_type=str(compiled_event.get("event_type", metadata.get("event_type", ""))),
        )
        return {
            "updated": True,
            "counterpart_id": counterpart_id,
            "model": model.to_dict(),
            "snapshot": self.social_memory.snapshot(),
        }

    def ingest_world_events(
        self,
        episodes: list[EmbodiedNarrativeEpisode],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        from .narrative_ingestion import NarrativeIngestionService
        from .narrative_types import NarrativeEpisode

        service = NarrativeIngestionService()
        raw_episodes: list[NarrativeEpisode] = []
        for episode in episodes:
            if isinstance(episode, NarrativeEpisode):
                raw_episodes.append(episode)
                continue
            if isinstance(episode, EmbodiedNarrativeEpisode):
                raw_episodes.append(
                    NarrativeEpisode(
                        episode_id=episode.episode_id,
                        timestamp=episode.timestamp,
                        source=str(episode.provenance.get("world_id", "world")),
                        raw_text=str(episode.provenance.get("raw_text", episode.predicted_outcome)),
                        tags=list(episode.narrative_tags),
                        metadata=dict(episode.provenance),
                    )
                )
        if raw_episodes:
            results = service.ingest(agent=self, episodes=raw_episodes)
        return results

    def _available_action_schemas(self) -> list[ActionSchema]:
        actions = self.action_registry.get_all()
        if actions:
            return actions
        return [
            ActionSchema(name=name, cost_estimate=float(cost))
            for name, cost in ACTION_COSTS.items()
        ]

    def _action_schema_for_name(self, action: str) -> ActionSchema:
        schema = self.action_registry.get(action)
        if schema is not None:
            return schema
        return ActionSchema(
            name=action,
            cost_estimate=float(ACTION_COSTS.get(action, 0.05)),
        )

    def _action_cost(self, action: str) -> float:
        schema = self._action_schema_for_name(action)
        if schema.cost_estimate:
            return float(schema.cost_estimate)
        return float(ACTION_COSTS.get(action, 0.05))

    @property
    def _personality_drive_modulation(self) -> dict[str, float]:
        return self.self_model.personality_profile.drive_modulation()

    @property
    def _personality_strategic_modulation(self) -> dict[str, float]:
        return self.self_model.personality_profile.strategic_modulation()

    def _sync_self_model_body_schema(self) -> None:
        self.self_model.body_schema = self.self_model.body_schema.__class__(
            energy=self.energy,
            token_budget=self.self_model.body_schema.token_budget,
            memory_usage=float(len(self.long_term_memory.episodes)),
            compute_load=self.self_model.body_schema.compute_load,
        )

    def _current_state_snapshot(
        self,
        observed: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
    ) -> dict[str, object]:
        return {
            "observation": observed,
            "prediction": prediction,
            "errors": errors,
            "body_state": {
                "cycle": float(self.cycle),
                "energy": self.energy,
                "stress": self.stress,
                "fatigue": self.fatigue,
                "temperature": self.temperature,
                "dopamine": self.dopamine,
            },
            "free_energy_history": list(self.free_energy_history),
            "subject_state": self.subject_state.to_dict(),
        }

    def _current_body_state(self) -> dict[str, float]:
        return {
            "cycle": float(self.cycle),
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
            "dopamine": self.dopamine,
        }

    def _zero_memory_context(
        self,
        *,
        observed: dict[str, float],
        baseline_prediction: dict[str, float],
        errors: dict[str, float],
        summary: str,
    ) -> dict[str, object]:
        channels = sorted(set(observed) | set(baseline_prediction))
        zero_projection = {key: 0.0 for key in channels}
        zero_delta = {key: 0.0 for key in channels}
        return {
            "memory_hit": False,
            "retrieved_episode_ids": [],
            "summary": summary,
            "state_projection": zero_projection,
            "state_delta": zero_delta,
            "aggregate": {
                "dominant_outcome": "none",
                "risk": 0.0,
                "preferred_probability": 0.0,
                "expected_surprise": 0.0,
                "chronic_threat_bias": 0.0,
                "protected_anchor_bias": 0.0,
                "outcome_distribution": {},
            },
            "actions": {},
            "prediction_blend": 0.0,
            "delta_gain": 0.0,
            "body_state": self._current_body_state(),
            "observation": dict(observed),
            "prediction_error": compute_prediction_error(observed, baseline_prediction),
            "errors": dict(errors),
            "sensitive_channels": [],
            "attention_biases": {},
        }

    def _build_memory_context(
        self,
        *,
        observed: dict[str, float],
        baseline_prediction: dict[str, float],
        errors: dict[str, float],
        similar_memories: list[dict[str, object]],
    ) -> dict[str, object]:
        if not self.memory_enabled:
            return self._zero_memory_context(
                observed=observed,
                baseline_prediction=baseline_prediction,
                errors=errors,
                summary="episodic memory influence suppressed",
            )
        if not similar_memories:
            return self._zero_memory_context(
                observed=observed,
                baseline_prediction=baseline_prediction,
                errors=errors,
                summary="no episodic memory influence",
            )

        weighted_total = sum(
            max(1e-9, float(payload.get("similarity", payload.get("vector_similarity", 0.0))))
            for payload in similar_memories
        )
        aggregate_projection: dict[str, float] = {}
        aggregate_delta: dict[str, float] = {}
        action_rollups: dict[str, dict[str, object]] = {}
        outcome_totals: dict[str, float] = {}
        retrieved_episode_ids: list[str] = []
        protected_anchor_weight = 0.0
        threat_trace_weight = 0.0
        sensitive_channel_totals: dict[str, float] = {}

        for payload in similar_memories:
            weight = max(
                1e-9,
                float(payload.get("similarity", payload.get("vector_similarity", 0.0))),
            )
            observation = {
                str(key): float(value)
                for key, value in dict(payload.get("observation", {})).items()
                if isinstance(value, (int, float))
            }
            action_key = action_name(
                payload.get("action_taken", payload.get("action", "unknown"))
            )
            predicted_effects = {
                str(key): float(value)
                for key, value in dict(payload.get("outcome_state", payload.get("outcome", {}))).items()
                if isinstance(value, (int, float))
            }
            predicted_outcome = str(payload.get("predicted_outcome", "neutral"))
            outcome_totals[predicted_outcome] = outcome_totals.get(predicted_outcome, 0.0) + weight
            episode_id = str(payload.get("episode_id", ""))
            if episode_id:
                retrieved_episode_ids.append(episode_id)
            continuity_tags = {
                str(item) for item in payload.get("continuity_tags", []) if str(item)
            }
            if bool(payload.get("restart_protected", False)) or "structural_trace" in continuity_tags:
                protected_anchor_weight += weight
            if (
                float(payload.get("threat_significance", 0.0)) >= 0.20
                or predicted_outcome in {"survival_threat", "integrity_loss"}
                or str(payload.get("episode_family", "")) == "hazard_response"
            ):
                threat_trace_weight += weight

            rollup = action_rollups.setdefault(
                action_key,
                {
                    "weight": 0.0,
                    "risk": 0.0,
                    "preferred_probability": 0.0,
                    "expected_surprise": 0.0,
                    "observation_projection": {},
                    "predicted_effects": {},
                    "outcome_distribution": {},
                    "action_descriptor": self._action_schema_for_name(action_key).to_dict(),
                },
            )
            rollup["weight"] = float(rollup["weight"]) + weight
            rollup["risk"] = float(rollup["risk"]) + float(payload.get("risk", 0.0)) * weight
            rollup["preferred_probability"] = float(rollup["preferred_probability"]) + float(
                payload.get("preferred_probability", 0.0)
            ) * weight
            rollup["expected_surprise"] = float(rollup["expected_surprise"]) + float(
                payload.get("total_surprise", payload.get("weighted_surprise", 0.0))
            ) * weight

            obs_projection = dict(rollup["observation_projection"])
            for key, value in observation.items():
                aggregate_projection[key] = aggregate_projection.get(key, 0.0) + (value * weight)
                aggregate_delta[key] = aggregate_delta.get(key, 0.0) + (
                    (value - baseline_prediction.get(key, 0.0)) * weight
                )
                obs_projection[key] = obs_projection.get(key, 0.0) + (value * weight)
                if abs(value - baseline_prediction.get(key, 0.0)) >= 0.12 or value >= 0.70:
                    sensitive_channel_totals[key] = sensitive_channel_totals.get(key, 0.0) + weight
            rollup["observation_projection"] = obs_projection

            effect_projection = dict(rollup["predicted_effects"])
            for key, value in predicted_effects.items():
                effect_projection[key] = effect_projection.get(key, 0.0) + (value * weight)
            rollup["predicted_effects"] = effect_projection

            outcome_distribution = dict(rollup["outcome_distribution"])
            outcome_distribution[predicted_outcome] = outcome_distribution.get(predicted_outcome, 0.0) + weight
            rollup["outcome_distribution"] = outcome_distribution

        aggregate_risk = 0.0
        aggregate_probability = 0.0
        aggregate_surprise = 0.0
        for action_key, rollup in action_rollups.items():
            weight = max(1e-9, float(rollup["weight"]))
            rollup["risk"] = float(rollup["risk"]) / weight
            rollup["preferred_probability"] = float(rollup["preferred_probability"]) / weight
            rollup["expected_surprise"] = float(rollup["expected_surprise"]) / weight
            rollup["observation_projection"] = {
                key: float(value) / weight
                for key, value in dict(rollup["observation_projection"]).items()
            }
            total_outcomes = sum(
                float(value) for value in dict(rollup["outcome_distribution"]).values()
            ) or 1.0
            rollup["outcome_distribution"] = {
                key: float(value) / total_outcomes
                for key, value in dict(rollup["outcome_distribution"]).items()
            }
            rollup["predicted_effects"] = {
                key: float(value) / weight
                for key, value in dict(rollup["predicted_effects"]).items()
            }
            aggregate_risk += float(rollup["risk"]) * (weight / weighted_total)
            aggregate_probability += float(rollup["preferred_probability"]) * (weight / weighted_total)
            aggregate_surprise += float(rollup["expected_surprise"]) * (weight / weighted_total)
            action_rollups[action_key] = rollup

        state_projection = {
            key: value / weighted_total for key, value in aggregate_projection.items()
        }
        state_delta = {
            key: value / weighted_total for key, value in aggregate_delta.items()
        }
        chronic_threat_bias = threat_trace_weight / weighted_total
        protected_anchor_bias = protected_anchor_weight / weighted_total
        if chronic_threat_bias > 0.0:
            state_projection["danger"] = max(
                float(state_projection.get("danger", baseline_prediction.get("danger", 0.0))),
                min(1.0, float(state_projection.get("danger", 0.0)) + (0.08 * chronic_threat_bias)),
            )
            state_delta["danger"] = float(state_delta.get("danger", 0.0)) + (0.12 * chronic_threat_bias)
        sensitive_channels = [
            key
            for key, value in sorted(
                sensitive_channel_totals.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if value > 0.0
        ][:3]
        attention_biases = {
            key: min(
                0.30,
                (float(sensitive_channel_totals.get(key, 0.0)) / weighted_total) * 0.20
                + (0.12 if key == "danger" and chronic_threat_bias > 0.0 else 0.0)
                + (0.08 if key in sensitive_channels and protected_anchor_bias > 0.0 else 0.0),
            )
            for key in sensitive_channels
        }
        dominant_outcome = sorted(
            outcome_totals.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        memory_context = {
            "memory_hit": True,
            "retrieved_episode_ids": retrieved_episode_ids,
            "summary": (
                f"{len(similar_memories)} episodic match(es), dominant outcome={dominant_outcome}, "
                f"risk={aggregate_risk:.3f}, expected_surprise={aggregate_surprise:.3f}, "
                f"chronic_threat={chronic_threat_bias:.3f}, protected_anchor={protected_anchor_bias:.3f}"
            ),
            "state_projection": state_projection,
            "state_delta": state_delta,
            "aggregate": {
                "dominant_outcome": dominant_outcome,
                "risk": aggregate_risk,
                "preferred_probability": aggregate_probability,
                "expected_surprise": aggregate_surprise,
                "chronic_threat_bias": chronic_threat_bias,
                "protected_anchor_bias": protected_anchor_bias,
                "outcome_distribution": {
                    key: value / weighted_total for key, value in outcome_totals.items()
                },
            },
            "actions": action_rollups,
            "prediction_blend": min(
                0.55,
                0.12
                + (0.08 * len(similar_memories))
                + (0.12 * chronic_threat_bias)
                + (0.08 * protected_anchor_bias),
            ),
            "delta_gain": min(
                0.60,
                0.18
                + (0.05 * len(similar_memories))
                + (0.15 * chronic_threat_bias)
                + (0.08 * protected_anchor_bias),
            ),
            "body_state": self._current_body_state(),
            "observation": dict(observed),
            "prediction_error": compute_prediction_error(observed, baseline_prediction),
            "errors": dict(errors),
            "sensitive_channels": sensitive_channels,
            "attention_biases": attention_biases,
        }
        return memory_context

    @property
    def current_tick(self) -> int:
        return self.cycle

    def _classify_error_source(self, modality: str, error_value: float) -> str:
        interoceptive_modalities = {"energy", "stress", "fatigue", "temperature"}
        if modality in interoceptive_modalities:
            body_val = {
                "energy": self.energy,
                "stress": self.stress,
                "fatigue": self.fatigue,
                "temperature": self.temperature,
            }.get(modality, 0.5)
            thresholds = self.self_model.threat_profile.get(modality, {}) or {}
            critical_low = float(thresholds.get("critical_low", 0.1))
            critical_high = float(thresholds.get("critical_high", 0.9))
            if body_val < critical_low or body_val > critical_high:
                return "self"
            return "ambiguous"
        return "world"

    def prediction_error_trace(
        self,
        observation: dict[str, float],
        prediction: dict[str, float],
    ) -> dict[str, dict[str, float | str]]:
        keys = sorted(set(observation) | set(prediction))
        traced: dict[str, dict[str, float | str]] = {}
        for modality in keys:
            value = abs(float(observation.get(modality, 0.0)) - float(prediction.get(modality, 0.0)))
            traced[modality] = {
                "value": value,
                "error_source": self._classify_error_source(modality, value),
            }
        return traced

    def _record_decision_history(
        self,
        diagnostics: DecisionDiagnostics,
        observed: dict[str, float],
    ) -> None:
        record = {
            "tick": self.cycle,
            "action": diagnostics.chosen.choice,
            "dominant_component": diagnostics.chosen.dominant_component,
            "risk": diagnostics.chosen.risk,
            "active_goal": diagnostics.active_goal,
            "goal_alignment": diagnostics.chosen.goal_alignment,
            "preferred_probability": diagnostics.chosen.preferred_probability,
            "policy_score": diagnostics.chosen.policy_score,
            "relevant_commitments": list(diagnostics.relevant_commitments),
            "commitment_compatibility_score": diagnostics.commitment_compatibility_score,
            "self_inconsistency_error": diagnostics.self_inconsistency_error,
            "conflict_type": diagnostics.conflict_type,
            "severity_level": diagnostics.severity_level,
            "consistency_classification": diagnostics.consistency_classification,
            "behavioral_classification": diagnostics.behavioral_classification,
            "repair_triggered": diagnostics.repair_triggered,
            "repair_policy": diagnostics.repair_policy,
            "commitment_focus": list(diagnostics.commitment_focus),
            "violated_commitments": list(diagnostics.violated_commitments),
            "identity_tension": diagnostics.identity_tension,
            "social_focus": list(diagnostics.social_focus),
            "social_alerts": list(diagnostics.social_alerts),
            "ledger_summary": diagnostics.ledger_summary,
            "inquiry_scheduler_summary": diagnostics.inquiry_scheduler_summary,
            "subject_state_summary": diagnostics.subject_state_summary,
            "subject_status_flags": dict(diagnostics.subject_status_flags),
            "process_valence": self.drive_system.process_valence.to_dict(),
            "cognitive_style": self.slow_variable_learner.style_snapshot(),
        }
        self.decision_history.append(record)
        if len(self.decision_history) > self.decision_history_limit:
            self.decision_history = self.decision_history[-self.decision_history_limit :]
        self._record_effort_allocation(record["action"], observed)

        drive_snapshot = {
            drive.name: float(drive.urgency) for drive in self.drive_system.drives
        }
        drive_snapshot.update(
            {
                "process_tension": float(self.drive_system.process_valence.unresolved_tension),
                "closure_satisfaction": float(self.drive_system.process_valence.closure_satisfaction),
                "boredom_pressure": float(self.drive_system.process_valence.boredom_pressure),
            }
        )
        self.drive_history.append(drive_snapshot)
        if len(self.drive_history) > self.drive_history_limit:
            self.drive_history = self.drive_history[-self.drive_history_limit :]

    def _refresh_self_model_continuity(
        self,
        sleep_summary: SleepSummary | None = None,
        weight_adjustments: list[object] | None = None,
    ) -> None:
        last_tick = 0
        if self.self_model.preferred_policies is not None:
            last_tick = self.self_model.preferred_policies.last_updated_tick
        recent_history = [
            entry for entry in self.decision_history if int(entry.get("tick", 0)) > last_tick
        ]
        if not recent_history:
            recent_history = list(self.decision_history)
        self.self_model.update_preferred_policies(
            recent_history,
            counterfactual_insights=self.counterfactual_insights,
            drive_history=self.drive_history[-32:],
            current_tick=self.cycle,
        )
        self.self_model.update_identity_narrative(
            episodic_memory=list(self.long_term_memory.episodes),
            preference_labels=self.long_term_memory.preference_model.legacy_value_hierarchy_dict(),
            current_tick=self.cycle,
            decision_history=list(self.decision_history),
            sleep_metrics=asdict(sleep_summary) if sleep_summary is not None else {},
            conflict_history=list(self.goal_stack.conflict_history),
            weight_adjustments=(
                list(weight_adjustments)
                if weight_adjustments is not None
                else list(self.goal_stack.weight_adjustments)
            ),
            chapter_signal=self.goal_stack.consume_chapter_signal(),
            slow_learning_state=self.slow_variable_learner.state,
            slow_learning_explanations=self.slow_variable_learner.recent_explanations(),
        )

    def explain_decision_details(self, action: str | None = None) -> dict[str, object]:
        if self.last_decision_diagnostics is None:
            return {"text": "No decision has been evaluated yet."}
        details = self.policy_evaluator.explain_structured(
            self.last_decision_diagnostics,
            action=action,
        )
        details["prediction_ledger"] = self.prediction_ledger.explanation_payload()
        details["reconciliation"] = self.reconciliation_engine.explanation_payload()
        details["subject_state"] = self.subject_state.explanation_payload()
        details["narrative_uncertainty"] = self.latest_narrative_uncertainty.explanation_payload()
        details["narrative_experiment"] = self.latest_narrative_experiment.explanation_payload()
        details["inquiry_scheduler"] = self.inquiry_budget_scheduler.state.explanation_payload()
        details["slow_learning"] = self.slow_variable_learner.explanation_payload()
        details["process_valence"] = self.drive_system.process_valence.to_dict()
        return details

    def should_sleep(self) -> bool:
        """Decide if the agent needs to sleep."""
        return (
            self.energy < 0.30
            or self.fatigue > 0.75
            or len(self.episodes) >= 10
            or self.long_term_memory.should_sleep(self.cycle)
        )

    def perceive(
        self,
        observation: Observation,
        *,
        apply_attention: bool = True,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        float,
        HierarchicalInference,
    ]:
        """Perceive the world, generate predictions, compute errors."""
        # Update drive urgencies
        novelty_deficit = 1.0 - observation.novelty
        social_isolation = 1.0 - observation.social
        self.drive_system.update_urgencies(
            self.energy,
            self.stress,
            self.fatigue,
            self.temperature,
            social_isolation,
            novelty_deficit,
            personality_modulation=self._personality_drive_modulation,
        )

        observed = observation_dict(observation)
        (
            strategic_prior,
            strategic_prediction,
            sensorimotor_prediction,
            interoceptive_prediction,
        ) = self._top_down_pass(observed)
        raw_errors = {
            key: observed.get(key, 0.0) - interoceptive_prediction.get(key, 0.0)
            for key in sorted(set(observed) | set(interoceptive_prediction))
        }
        self.last_attention_trace = None
        self.last_attention_filtered_observation = dict(observed)
        filtered_observed = dict(observed)
        if apply_attention and self.attention_bottleneck.enabled:
            self.last_attention_trace = self.attention_bottleneck.allocate(
                observation=observed,
                prediction=interoceptive_prediction,
                errors=raw_errors,
                narrative_priors=self.self_model.narrative_priors.to_dict(),
                tick=self.cycle,
                memory_context=self.last_memory_context,
            )
            filtered_observed = self.attention_bottleneck.filter_observation(
                observed,
                self.last_attention_trace.allocation,
                prediction=interoceptive_prediction,
            )
            self.last_attention_filtered_observation = dict(filtered_observed)
        hierarchy = self._bottom_up_pass(
            filtered_observed,
            strategic_prior,
            strategic_prediction,
            sensorimotor_prediction,
            interoceptive_prediction,
        )

        errors = dict(hierarchy.interoceptive_update.raw_error)
        free_energy = self.compute_free_energy(
            errors,
            hierarchy.interoceptive_update.error_precision,
        )
        return observed, interoceptive_prediction, errors, free_energy, hierarchy

    def _top_down_pass(
        self,
        observed: dict[str, float],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        strategic_prior, strategic_prediction = self.strategic_layer.dispatch_prediction(
            self.energy,
            self.stress,
            self.fatigue,
            self.temperature,
            self.dopamine,
            self.drive_system,
            personality_modulation=self._personality_strategic_modulation,
        )

        baseline_prediction = self.world_model.predict(strategic_prediction)
        baseline_errors = {
            key: observed.get(key, 0.0) - baseline_prediction.get(key, 0.0)
            for key in sorted(set(observed) | set(baseline_prediction))
        }
        current_state_snapshot = {
            "observation": dict(observed),
            "prediction": dict(baseline_prediction),
            "errors": baseline_errors,
            "body_state": self._current_body_state(),
        }
        similar_memories = self._retrieve_decision_memories(
            observed=observed,
            baseline_prediction=baseline_prediction,
            baseline_errors=baseline_errors,
            current_state_snapshot=current_state_snapshot,
            k=3,
        )
        memory_context = self._build_memory_context(
            observed=observed,
            baseline_prediction=baseline_prediction,
            errors=baseline_errors,
            similar_memories=similar_memories,
        )
        sensorimotor_prediction = self.world_model.predict(
            strategic_prediction,
            memory_context=memory_context,
        )
        self.last_memory_context = {
            **memory_context,
            "prediction_before_memory": dict(
                self.world_model.last_prediction_details.get(
                    "prediction_before_memory",
                    baseline_prediction,
                )
            ),
            "prediction_after_memory": dict(
                self.world_model.last_prediction_details.get(
                    "prediction_after_memory",
                    sensorimotor_prediction,
                )
            ),
            "prediction_delta": dict(
                self.world_model.last_prediction_details.get("prediction_delta", {})
            ),
            "retrieved_memories": similar_memories,
            "memory_backend": self._active_memory_backend(),
            "recall_hypothesis": dict(self.long_term_memory.last_retrieval_result.get("recall_hypothesis", {}) or {}),
            "reconstruction_trace": dict(
                self.long_term_memory.last_retrieval_result.get("reconstruction_trace", {}) or {}
            ),
            "memory_enabled": self.memory_enabled,
            "memory_bias": 0.0,
            "pattern_bias": 0.0,
        }
        # Restore decision-loop bias fields if they were set in this tick's
        # _evaluate_options (validation perceive() re-invokes _top_down_pass
        # after the decision, which would otherwise wipe them).
        _stashed = getattr(self, "_decision_bias_stash", None)
        if _stashed is not None:
            self.last_memory_context["memory_bias"] = _stashed["memory_bias"]
            self.last_memory_context["pattern_bias"] = _stashed["pattern_bias"]
        interoceptive_prediction = self.interoceptive_layer.predict(sensorimotor_prediction)
        return (
            strategic_prior,
            strategic_prediction,
            sensorimotor_prediction,
            interoceptive_prediction,
        )

    def _bottom_up_pass(
        self,
        observed: dict[str, float],
        strategic_prior: dict[str, float],
        strategic_prediction: dict[str, float],
        sensorimotor_prediction: dict[str, float],
        interoceptive_prediction: dict[str, float],
    ) -> HierarchicalInference:
        interoceptive_update = self.interoceptive_layer.assimilate(
            observed,
            sensorimotor_prediction,
            predicted_state=interoceptive_prediction,
        )
        sensorimotor_signal = compose_upstream_observation(
            sensorimotor_prediction,
            interoceptive_update.propagated_error,
        )
        sensorimotor_update = self.world_model.assimilate(
            sensorimotor_signal,
            strategic_prediction,
            predicted_state=sensorimotor_prediction,
        )
        strategic_signal = compose_upstream_observation(
            strategic_prediction,
            sensorimotor_update.propagated_error,
        )
        strategic_update = self.strategic_layer.assimilate(
            strategic_signal,
            strategic_prior,
            predicted_state=strategic_prediction,
        )
        return HierarchicalInference(
            observation=observed,
            strategic_prior=strategic_prior,
            strategic_prediction=strategic_prediction,
            sensorimotor_prediction=sensorimotor_prediction,
            interoceptive_prediction=interoceptive_prediction,
            sensorimotor_observation=sensorimotor_signal,
            strategic_observation=strategic_signal,
            interoceptive_update=interoceptive_update,
            sensorimotor_update=sensorimotor_update,
            strategic_update=strategic_update,
        )

    def compute_free_energy(
        self,
        errors: dict[str, float],
        precisions: dict[str, float] | None = None,
    ) -> float:
        """Compute free energy from prediction errors and internal pressure."""
        def pe(key: str) -> float:
            magnitude = abs(errors.get(key, 0.0))
            if precisions and key in precisions:
                magnitude *= precisions[key]
            return magnitude

        weighted = (
            pe("food") * 1.30
            + pe("danger") * 1.60
            + pe("novelty") * 0.80
            + pe("shelter") * 1.00
            + pe("temperature") * 0.90
            + pe("social") * 0.70
        )
        # Internal complexity from body state
        energy_pressure = max(0.0, 0.45 - self.energy) * 0.25
        stress_pressure = self.stress * 0.28
        fatigue_pressure = self.fatigue * 0.20
        thermal_pressure = abs(self.temperature - 0.5) * 0.15
        complexity = energy_pressure + stress_pressure + fatigue_pressure + thermal_pressure
        return weighted + complexity

    def evaluate_internal_update(
        self,
        priors: dict[str, float],
        errors: dict[str, float],
    ) -> tuple[float, float]:
        """Evaluate the cost/benefit of updating internal model."""
        imagined_beliefs = dict(self.world_model.beliefs)
        for key, error in errors.items():
            if key in imagined_beliefs:
                imagined_beliefs[key] = clamp(
                    imagined_beliefs[key] + self.world_model.learning_rate * error
                )
        imagined_prediction = {
            key: clamp((imagined_beliefs[key] * 0.60) + (priors[key] * 0.40))
            for key in priors
        }
        residual_errors = {
            key: (self.world_model.predict(priors)[key] + errors[key]) - imagined_prediction[key]
            for key in priors
        }
        expected_fe = self.compute_free_energy(residual_errors) + 0.15
        return expected_fe, 0.13

    def _matching_semantic_rules(
        self,
        cluster_id: int | None,
        action: str,
    ) -> list[SemanticMemoryEntry]:
        if cluster_id is None:
            return []
        return [
            entry
            for entry in self.semantic_memory
            if entry.cluster == cluster_id and entry.action == action
        ]

    def _predict_with_slow_weights(
        self,
        *,
        cluster_id: int | None,
        action: str,
        projected_snapshot: dict[str, object],
        predicted_effects: dict[str, float],
    ) -> tuple[str, float, float, float]:
        preference = self.long_term_memory.preference_model.evaluate_state(
            {
                **projected_snapshot,
                "predicted_outcome": predicted_effects,
            }
        )
        predicted_outcome = preference.outcome
        preferred_probability = preference.preferred_probability
        risk = preference.risk
        value_score = preference.value_score

        if cluster_id is None:
            return predicted_outcome, preferred_probability, risk, value_score

        outcome_distribution = self.world_model.outcome_distribution(cluster_id, action)
        transition_distribution = self.world_model.transition_distribution(cluster_id, action)
        threat_prior = self.world_model.get_threat_prior(cluster_id)
        preference_penalty = self.world_model.get_preference_penalty(cluster_id, action)
        semantic_rules = self._matching_semantic_rules(cluster_id, action)
        semantic_penalty = sum(
            entry.confidence
            for entry in semantic_rules
            if entry.rule_type == "risk_pattern"
        )
        semantic_bonus = sum(
            entry.confidence
            for entry in semantic_rules
            if entry.rule_type != "risk_pattern"
        )

        transition_threat = 0.0
        for next_cluster, probability in transition_distribution.items():
            try:
                transition_threat += float(probability) * self.world_model.get_threat_prior(
                    int(next_cluster)
                )
            except ValueError:
                continue

        if outcome_distribution:
            predicted_outcome = sorted(
                outcome_distribution.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
            preferred_probability = max(
                1e-12,
                float(outcome_distribution.get(predicted_outcome, 0.0)),
            )
            risk = self.long_term_memory.preference_model.risk(predicted_outcome)
            value_score = self.long_term_memory.preference_model.normalized_score(
                predicted_outcome
            )

        risk = max(
            0.0,
            risk
            + (threat_prior * 4.0)
            + (transition_threat * 3.0)
            + max(0.0, -preference_penalty) * 1.5
            + semantic_penalty * 0.8
            - max(0.0, preference_penalty) * 0.5
            - semantic_bonus * 0.3,
        )
        preferred_probability = max(
            1e-12,
            min(
                1.0,
                preferred_probability
                * (1.0 - min(0.85, threat_prior * 0.6 + semantic_penalty * 0.15))
                + max(0.0, preference_penalty) * 0.05
                + semantic_bonus * 0.02,
            ),
        )
        value_score = max(-1.0, min(1.0, value_score + preference_penalty - semantic_penalty * 0.2))
        return predicted_outcome, preferred_probability, risk, value_score

    def _apply_sleep_consolidation(
        self,
        consolidation: SleepConsolidationResult,
    ) -> tuple[int, int, int]:
        new_entries = 0
        seen_rule_ids = {entry.rule_id for entry in self.semantic_memory}
        for entry in consolidation.semantic_memory_entries:
            if entry.rule_id in seen_rule_ids:
                continue
            self.semantic_memory.append(entry)
            seen_rule_ids.add(entry.rule_id)
            new_entries += 1

        threat_updates = 0
        preference_updates = 0
        for update in consolidation.model_updates:
            if update.update_type == "threat_prior":
                self.world_model.adjust_threat_prior(update.cluster, update.delta)
                action_label = action_name(update.action)
                matched_rule = next(
                    (
                        rule
                        for rule in consolidation.rules
                        if rule.rule_id == update.rule_id
                    ),
                    None,
                )
                outcome_label = (
                    matched_rule.observed_outcome
                    if matched_rule is not None
                    else "neutral"
                )
                self.self_model.threat_profile.add_learned_threat(
                    pattern=(
                        f"SEQUENCE: {action_label}->{outcome_label} "
                        f"x{matched_rule.sequence_condition.min_occurrences} "
                        f"in {matched_rule.sequence_condition.window_ticks}t"
                        if (
                            matched_rule is not None
                            and matched_rule.rule_type == "sequence_pattern"
                            and matched_rule.sequence_condition is not None
                        )
                        else f"{action_label} in cluster {update.cluster} -> {outcome_label}"
                    ),
                    risk_level=max(0.0, update.delta),
                    tick=self.current_tick,
                    source="world",
                )
                threat_updates += 1
                continue
            if update.update_type == "preference_penalty":
                self.world_model.adjust_preference_penalty(
                    update.cluster,
                    update.action,
                    update.delta,
                )
                preference_updates += 1
        return new_entries, threat_updates, preference_updates

    def _apply_narrative_sleep_updates(
        self,
        replay_batch: list[dict[str, object]],
    ) -> dict[str, float]:
        appraisals = [
            dict(payload.get("appraisal", {}))
            for payload in replay_batch
            if isinstance(payload.get("appraisal"), dict)
        ]
        if not appraisals:
            return {}

        def avg(name: str) -> float:
            return mean(float(appraisal.get(name, 0.0)) for appraisal in appraisals)

        priors = self.self_model.narrative_priors
        priors.trust_prior = max(
            -1.0,
            min(1.0, priors.trust_prior * 0.75 + avg("trust_impact") * 0.25),
        )
        priors.controllability_prior = max(
            -1.0,
            min(1.0, priors.controllability_prior * 0.75 + avg("controllability") * 0.25),
        )
        priors.trauma_bias = max(
            0.0,
            min(
                1.0,
                priors.trauma_bias * 0.70
                + (avg("physical_threat") + avg("loss")) * 0.20,
            ),
        )
        priors.contamination_sensitivity = max(
            0.0,
            min(
                1.0,
                priors.contamination_sensitivity * 0.70 + avg("contamination") * 0.30,
            ),
        )
        priors.meaning_stability = max(
            -1.0,
            min(1.0, priors.meaning_stability * 0.75 - avg("meaning_violation") * 0.25),
        )

        # M2.6: Update personality profile from accumulated appraisals
        compiler = NarrativeCompiler()
        from .narrative_types import AppraisalVector
        aggregate_signal = compiler.extract_personality_signal(
            AppraisalVector(
                physical_threat=avg("physical_threat"),
                social_threat=avg("social_threat"),
                uncertainty=avg("uncertainty"),
                controllability=avg("controllability"),
                novelty=avg("novelty"),
                loss=avg("loss"),
                moral_salience=avg("moral_salience"),
                contamination=avg("contamination"),
                attachment_signal=avg("attachment_signal"),
                trust_impact=avg("trust_impact"),
                self_efficacy_impact=avg("self_efficacy_impact"),
                meaning_violation=avg("meaning_violation"),
            )
        )
        personality_deltas = self.self_model.personality_profile.absorb_signal(
            aggregate_signal,
            tick=self.cycle,
        )

        result = priors.to_dict()
        result["personality_deltas"] = personality_deltas
        result["personality_signal"] = aggregate_signal.to_dict()
        return result

    def _project_action(
        self,
        *,
        action: str,
        observed: dict[str, float],
        prediction: dict[str, float],
        priors: dict[str, float],
        free_energy_before: float,
        current_cluster_id: int | None,
        active_goal: Goal | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        action_key = action_name(action)
        action_schema = self._action_schema_for_name(action_key)
        cost = self._action_cost(action_key)
        imagined = self.world_model.imagine_action(action_key, prediction)
        body_effects = ACTION_BODY_EFFECTS.get(action_key, {})
        imagined_energy = clamp(
            self.energy - cost - self.base_metabolic_rate
            + body_effects.get("energy_delta", 0.0)
        )
        imagined_stress = clamp(self.stress + body_effects.get("stress_delta", 0.0))
        imagined_fatigue = clamp(
            self.fatigue + self.fatigue_accumulation_rate
            + body_effects.get("fatigue_delta", 0.0)
        )
        imagined_temp = clamp(
            self.temperature + body_effects.get("temperature_delta", 0.0)
        )
        next_priors = self.strategic_layer.priors(
            imagined_energy,
            imagined_stress,
            imagined_fatigue,
            imagined_temp,
            self.dopamine,
            self.drive_system,
            personality_modulation=self._personality_strategic_modulation,
        )
        residual_errors = {key: next_priors[key] - imagined[key] for key in priors}
        predicted_error = compute_prediction_error(next_priors, imagined)
        action_ambiguity = cost + mean(abs(value) for value in residual_errors.values()) * 0.35
        predicted_effects = {
            "energy_delta": imagined_energy - self.energy,
            "stress_delta": imagined_stress - self.stress,
            "fatigue_delta": imagined_fatigue - self.fatigue,
            "temperature_delta": imagined_temp - self.temperature,
            "free_energy_drop": free_energy_before - (predicted_error + action_ambiguity),
        }
        projected_snapshot = {
            "observation": imagined,
            "prediction": next_priors,
            "errors": residual_errors,
            "body_state": {
                "energy": imagined_energy,
                "stress": imagined_stress,
                "fatigue": imagined_fatigue,
                "temperature": imagined_temp,
                "dopamine": self.dopamine,
            },
        }
        (
            predicted_outcome,
            preferred_probability,
            risk,
            value_score,
        ) = self._predict_with_slow_weights(
            cluster_id=current_cluster_id,
            action=action_key,
            projected_snapshot=projected_snapshot,
            predicted_effects=predicted_effects,
        )
        memory_refinement = self.world_model.refine_action_prediction(
            action=action_schema,
            projected_snapshot=projected_snapshot,
            predicted_effects=predicted_effects,
            predicted_outcome=predicted_outcome,
            preferred_probability=preferred_probability,
            risk=risk,
            predicted_error=predicted_error,
            memory_context=memory_context,
        )
        projected_snapshot = dict(memory_refinement["projected_snapshot"])
        predicted_effects = dict(memory_refinement["predicted_effects"])
        predicted_outcome = str(memory_refinement["predicted_outcome"])
        preferred_probability = float(memory_refinement["preferred_probability"])
        risk = float(memory_refinement["risk"])
        predicted_error = float(memory_refinement["expected_surprise"])
        expected_free_energy = self.long_term_memory.preference_model.expected_free_energy(
            outcome=predicted_outcome,
            predicted_error=predicted_error,
            action_ambiguity=action_ambiguity,
            goal=active_goal,
            baseline_risk=risk,
        )
        return {
            "predicted_state": imagined,
            "predicted_error": predicted_error,
            "action_ambiguity": action_ambiguity,
            "risk": risk,
            "preferred_probability": preferred_probability,
            "expected_free_energy": expected_free_energy,
            "predicted_outcome": predicted_outcome,
            "predicted_effects": predicted_effects,
            "value_score": value_score,
            "cost": cost,
            "action_schema": action_schema,
            "observation_distance": compute_prediction_error(
                observed,
                projected_snapshot["observation"],
            ),
            "applied_memory_context": bool(memory_refinement["applied_memory"]),
        }

    def evaluate_action_options(
        self,
        observed: dict[str, float],
        prediction: dict[str, float],
        priors: dict[str, float],
        free_energy_before: float,
        current_cluster_id: int | None,
        active_goal: Goal | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> dict[str, dict[str, object]]:
        """Project candidate actions into explicit policy components."""
        allowed = self.self_model.capability_model.available_actions
        available_actions = self._available_action_schemas()
        candidate_actions = (
            [action.name for action in available_actions if action.name in allowed]
            if allowed
            else [action.name for action in available_actions]
        )
        if not candidate_actions:
            candidate_actions = [action.name for action in available_actions]
        options: dict[str, dict[str, object]] = {}
        for action in candidate_actions:
            options[action] = self._project_action(
                action=action,
                observed=observed,
                prediction=prediction,
                priors=priors,
                free_energy_before=free_energy_before,
                current_cluster_id=current_cluster_id,
                active_goal=active_goal,
                memory_context=memory_context,
            )
        return options

    def decision_cycle(
        self,
        observation: Observation,
    ) -> dict[str, object]:
        if self._sleeping:
            raise RuntimeError(
                "Action space is frozen: agent is in sleep consolidation phase. "
                "External interaction is not permitted until sleep completes."
            )
        # Clear decision-bias stash so stale values from previous ticks
        # don't leak. The stash is set in _evaluate_options and restored
        # in _top_down_pass to survive the validation perceive() call.
        self._decision_bias_stash = None
        observed, prediction, errors, free_energy_before, hierarchy = self.perceive(
            observation
        )
        prediction_error = compute_prediction_error(observed, prediction)
        self.free_energy_history.append(free_energy_before)
        if len(self.free_energy_history) > self.free_energy_history_limit:
            self.free_energy_history = self.free_energy_history[-self.free_energy_history_limit :]
        current_state_snapshot = self._current_state_snapshot(observed, prediction, errors)
        goal_context = self.goal_stack.get_goal_context_for_decision(
            current_state_snapshot,
            tick=self.cycle,
        )
        active_goal = Goal[str(goal_context["active_goal"])]
        self.latest_narrative_experiment = self.narrative_experiment_designer.design(
            tick=self.cycle,
            uncertainty=self.latest_narrative_uncertainty,
            action_registry=self.action_registry,
            active_goal=str(active_goal),
            subject_state=self.subject_state,
            previous_result=self.latest_narrative_experiment,
            verification_loop=self.verification_loop,
        )
        self._refresh_inquiry_budget()
        if self.latest_narrative_experiment.plans:
            self.narrative_experiment_history.append(self.latest_narrative_experiment.to_dict())
            self.narrative_experiment_history = self.narrative_experiment_history[-64:]
        priors = self.strategic_layer.priors(
            self.energy,
            self.stress,
            self.fatigue,
            self.temperature,
            self.dopamine,
            self.drive_system,
            personality_modulation=self._personality_strategic_modulation,
        )
        similar_memories = list(self.last_memory_context.get("retrieved_memories", []))
        retrieved_episode_ids = [
            str(item.get("episode_id", ""))
            for item in similar_memories
            if item.get("episode_id")
        ]
        current_cluster_id = self.long_term_memory.infer_cluster_id(current_state_snapshot)
        ledger_verification = self.prediction_ledger.verify_predictions(
            tick=self.cycle,
            observation=observed,
        )
        verification_refresh = self.verification_loop.refresh_targets(
            tick=self.cycle,
            ledger=self.prediction_ledger,
            subject_state=self.subject_state,
            narrative_uncertainty=self.latest_narrative_uncertainty,
            experiment_design=self.latest_narrative_experiment,
            inquiry_state=self.inquiry_budget_scheduler.state,
        )
        self.last_workspace_state = self.global_workspace.broadcast(
            tick=self.cycle,
            observation=observed,
            prediction=prediction,
            errors=errors,
            attention_trace=self.last_attention_trace,
            ledger_focus={
                **self.inquiry_budget_scheduler.state.workspace_focus(),
                **self.latest_narrative_experiment.workspace_focus(),
                **self.prediction_ledger.workspace_focus(),
                **self.reconciliation_engine.workspace_focus(),
                **self.verification_loop.workspace_focus(),
            },
        )
        self.subject_state = derive_subject_state(
            self,
            previous_state=self.subject_state,
        )
        action_options = self.evaluate_action_options(
            observed,
            prediction,
            priors,
            free_energy_before,
            current_cluster_id,
            active_goal=active_goal,
            memory_context=self.last_memory_context,
        )
        ranked_options: list[InterventionScore] = []
        commitment_assessments: dict[str, dict[str, object]] = {}
        social_assessments: dict[str, dict[str, object]] = {}
        for action, option in action_options.items():
            predicted_state = dict(option["predicted_state"])
            predicted_effects = dict(option["predicted_effects"])
            expected_fe = float(option["expected_free_energy"])
            if self.memory_enabled:
                with suppress_legacy_memory_warnings():
                    memory_bias = self.long_term_memory.memory_bias(action, similar_memories)
                    pattern_bias = self.long_term_memory.pattern_bias(
                        action,
                        action_history=self.action_history,
                    )
            else:
                memory_bias = 0.0
                pattern_bias = 0.0
            policy_bias = self.world_model.get_policy_bias(current_cluster_id, action)
            epistemic_bonus = self.world_model.get_epistemic_bonus(
                current_cluster_id,
                action,
            )
            workspace_bias = self.global_workspace.action_bias(
                action,
                self.last_workspace_state,
            )
            social_assessment = self.social_memory.policy_assessment(
                action=action,
                observation=observed,
            )
            social_bias = float(social_assessment["bias"])
            social_assessments[action] = social_assessment
            commitment_assessment = self.policy_evaluator.commitment_assessment(
                action=action,
                projected_state=predicted_state,
            )
            commitment_bias = float(commitment_assessment["bias"])
            commitment_assessments[action] = commitment_assessment
            identity_bias = self.policy_evaluator.identity_bias(
                action=action,
                projected_state=predicted_state,
                predicted_outcome=predicted_effects,
                cost=float(option["cost"]),
            )
            ledger_bias = self.prediction_ledger.prediction_action_bias(action)
            reconciliation_bias = self.reconciliation_engine.action_bias(action)
            verification_bias = self.verification_loop.action_bias(action)
            experiment_bias = self.latest_narrative_experiment.action_bias(action)
            inquiry_scheduler_bias = self.inquiry_budget_scheduler.state.action_bias(action)
            subject_bias = subject_action_bias(self.subject_state, action)
            subject_bias += self.drive_system.process_action_bias(action)
            subject_bias += self.slow_variable_learner.cognitive_style_bias(
                action=action,
                uncertainty_level=max(
                    float(observed.get("novelty", 0.0)),
                    float(observed.get("danger", 0.0)) * 0.55,
                ),
                known_task=self._is_known_task(action),
                process_tension=float(self.drive_system.process_valence.unresolved_tension),
            )
            goal_alignment = self.goal_stack.goal_alignment_score(
                goal=active_goal,
                action=action,
                projected_state=predicted_state,
                predicted_effects=predicted_effects,
                current_state=current_state_snapshot,
            )
            regression_penalty = self._action_regression_penalty(action)
            continuity_bonus = self._repeated_observation_action_bonus(action, observed)
            policy_score = (
                -expected_fe
                + memory_bias
                + pattern_bias
                + policy_bias
                + epistemic_bonus
                + workspace_bias
                + social_bias
                + commitment_bias
                + identity_bias
                + ledger_bias
                + reconciliation_bias
                + subject_bias
                + goal_alignment
                + verification_bias
                + experiment_bias
                + inquiry_scheduler_bias
                + continuity_bonus
                - regression_penalty
            )
            dominant_component = self.policy_evaluator.dominant_component(
                expected_free_energy=expected_fe,
                memory_bias=memory_bias,
                pattern_bias=pattern_bias,
                policy_bias=policy_bias,
                epistemic_bonus=epistemic_bonus,
                workspace_bias=workspace_bias,
                social_bias=social_bias,
                commitment_bias=commitment_bias,
                identity_bias=identity_bias,
                ledger_bias=ledger_bias,
                subject_bias=subject_bias,
                goal_alignment=goal_alignment,
                reconciliation_bias=reconciliation_bias,
                verification_bias=verification_bias,
                experiment_bias=experiment_bias,
                inquiry_scheduler_bias=inquiry_scheduler_bias,
            )
            ranked_options.append(
                InterventionScore(
                    choice=action,
                    action_descriptor=dict(
                        option["action_schema"].to_dict()
                        if isinstance(option.get("action_schema"), ActionSchema)
                        else self._action_schema_for_name(action).to_dict()
                    ),
                    policy_score=policy_score,
                    expected_free_energy=expected_fe,
                    predicted_error=float(option["predicted_error"]),
                    action_ambiguity=float(option["action_ambiguity"]),
                    risk=float(option["risk"]),
                    preferred_probability=float(option["preferred_probability"]),
                    memory_bias=memory_bias,
                    pattern_bias=pattern_bias,
                    policy_bias=policy_bias,
                    epistemic_bonus=epistemic_bonus,
                    workspace_bias=workspace_bias,
                    social_bias=social_bias,
                    commitment_bias=commitment_bias,
                    identity_bias=identity_bias,
                    ledger_bias=ledger_bias,
                    subject_bias=subject_bias,
                    goal_alignment=goal_alignment,
                    reconciliation_bias=reconciliation_bias,
                    verification_bias=verification_bias,
                    experiment_bias=experiment_bias,
                    inquiry_scheduler_bias=inquiry_scheduler_bias,
                    value_score=float(option["value_score"]),
                    predicted_outcome=str(option["predicted_outcome"]),
                    predicted_effects=predicted_effects,
                    dominant_component=dominant_component,
                    cost=float(option["cost"]),
                    commitment_compatibility_score=float(
                        commitment_assessment.get("compatibility_score", 0.5)
                    ),
                    relevant_commitments=list(
                        commitment_assessment.get("relevant_commitments", [])
                    ),
                    commitment_violations=list(
                        commitment_assessment.get("violations", [])
                    ),
                )
            )
        # Value-consistency override: when danger is extreme, the survival
        # value hierarchy (survival > resource_gain) prevents forage from
        # winning purely on EFE advantage.  The override equalises forage
        # with the best safe alternative so that additive bias terms (memory,
        # policy, identity) remain the actual tie-breakers.
        obs_danger = observed.get("danger", 0.0)
        if obs_danger > 0.80:
            best_safe = max(
                (o.policy_score for o in ranked_options if o.choice != "forage"),
                default=None,
            )
            if best_safe is not None:
                for option in ranked_options:
                    if option.choice == "forage" and option.policy_score > best_safe:
                        option.policy_score = best_safe - 0.10

        ranked_options.sort(
            key=lambda option: (
                option.policy_score,
                -option.expected_free_energy,
                option.choice,
            ),
            reverse=True,
        )
        chosen_option = ranked_options[0]
        chosen_assessment = commitment_assessments.get(
            chosen_option.choice,
            {
                "active_commitments": [],
                "relevant_commitments": [],
                "focus": [],
                "violations": [],
                "compatibility_score": 0.5,
                "self_inconsistency_error": 0.0,
                "conflict_type": "none",
                "severity_level": "none",
                "consistency_classification": "aligned",
                "behavioral_classification": "aligned",
                "repair_triggered": False,
                "tension": 0.0,
                "repair_policy": "",
                "repair_result": {},
            },
        )
        self.self_model.register_self_inconsistency(
            tick=self.cycle,
            action=chosen_option.choice,
            assessment=chosen_assessment,
        )
        repair_result = dict(chosen_assessment.get("repair_result", {}))
        if bool(chosen_assessment.get("repair_triggered", False)):
            repaired_option, repair_result = self._select_repaired_option(
                ranked_options=ranked_options,
                commitment_assessments=commitment_assessments,
            )
            if repaired_option is not None:
                chosen_option = repaired_option
                chosen_assessment = commitment_assessments.get(chosen_option.choice, chosen_assessment)
                ranked_options = [
                    chosen_option,
                    *[option for option in ranked_options if option.choice != chosen_option.choice],
                ]
            self.self_model.record_repair_outcome(
                tick=self.cycle,
                policy=str(repair_result.get("policy", chosen_assessment.get("repair_policy", ""))),
                success=bool(repair_result.get("success", False)),
                target_action=str(repair_result.get("target_action", ranked_options[0].choice)),
                repaired_action=str(repair_result.get("repaired_action", chosen_option.choice)),
                pre_alignment=float(
                    commitment_assessments.get(
                        str(repair_result.get("target_action", ranked_options[0].choice)),
                        chosen_assessment,
                    ).get("compatibility_score", 0.5)
                ),
                post_alignment=float(chosen_assessment.get("compatibility_score", 0.5)),
                bounded_update_applied="bounded_commitment_update" in str(
                    repair_result.get("policy", chosen_assessment.get("repair_policy", ""))
                ),
                social_repair_required="social_repair" in str(
                    repair_result.get("policy", chosen_assessment.get("repair_policy", ""))
                ),
            )
        chosen_social = social_assessments.get(
            chosen_option.choice,
            {"focus": [], "alerts": [], "snapshot": self.social_memory.snapshot()},
        )
        diagnostics = DecisionDiagnostics(
            chosen=chosen_option,
            ranked_options=ranked_options,
            prediction_error=prediction_error,
            retrieved_memories=similar_memories,
            policy_scores={
                option.choice: option.policy_score for option in ranked_options
            },
            explanation="",
            active_goal=str(goal_context["active_goal"]),
            goal_context=goal_context,
            memory_hit=bool(similar_memories),
            retrieved_episode_ids=retrieved_episode_ids,
            memory_context_summary=str(self.last_memory_context.get("summary", "")),
            prediction_before_memory={
                str(key): float(value)
                for key, value in dict(
                    self.last_memory_context.get("prediction_before_memory", {})
                ).items()
                if isinstance(value, (int, float))
            },
            prediction_after_memory={
                str(key): float(value)
                for key, value in dict(
                    self.last_memory_context.get("prediction_after_memory", prediction)
                ).items()
                if isinstance(value, (int, float))
            },
            prediction_delta={
                str(key): float(value)
                for key, value in dict(
                    self.last_memory_context.get("prediction_delta", {})
                ).items()
                if isinstance(value, (int, float))
            },
            attention_selected_channels=list(
                self.last_attention_trace.allocation.selected_channels
                if self.last_attention_trace is not None
                else []
            ),
            attention_dropped_channels=list(
                self.last_attention_trace.allocation.dropped_channels
                if self.last_attention_trace is not None
                else []
            ),
            attention_salience_scores=dict(
                self.last_attention_trace.salience_scores
                if self.last_attention_trace is not None
                else {}
            ),
            workspace_latent_channels=[
                content.channel
                for content in (
                    self.last_workspace_state.latent_candidates
                    if self.last_workspace_state is not None
                    else ()
                )
            ],
            workspace_attended_channels=[
                content.channel
                for content in (
                    self.last_workspace_state.attended_candidates
                    if self.last_workspace_state is not None
                    else ()
                )
            ],
            workspace_broadcast_channels=self.global_workspace.report_focus(
                self.last_workspace_state
            ),
            workspace_suppressed_channels=list(
                self.last_workspace_state.suppressed_channels
                if self.last_workspace_state is not None
                else []
            ),
            workspace_carry_over_channels=[
                content.channel
                for content in (
                    self.last_workspace_state.carry_over_contents
                    if self.last_workspace_state is not None
                    else ()
                )
            ],
            workspace_broadcast_intensity=(
                self.last_workspace_state.broadcast_intensity
                if self.last_workspace_state is not None
                else 0.0
            ),
            workspace_persistence_horizon=(
                self.last_workspace_state.persistence_horizon
                if self.last_workspace_state is not None
                else 0
            ),
            conscious_report_channels=self.global_workspace.conscious_report_payload(
                self.last_workspace_state
            )["accessible_channels"],
            current_commitments=list(chosen_assessment["active_commitments"]),
            relevant_commitments=list(chosen_assessment.get("relevant_commitments", [])),
            commitment_focus=list(chosen_assessment["focus"]),
            violated_commitments=list(chosen_assessment["violations"]),
            commitment_compatibility_score=float(chosen_assessment.get("compatibility_score", 0.5)),
            self_inconsistency_error=float(chosen_assessment.get("self_inconsistency_error", 0.0)),
            conflict_type=str(chosen_assessment.get("conflict_type", "none")),
            severity_level=str(chosen_assessment.get("severity_level", "none")),
            consistency_classification=str(
                chosen_assessment.get("consistency_classification", "aligned")
            ),
            behavioral_classification=str(
                chosen_assessment.get("behavioral_classification", "aligned")
            ),
            repair_triggered=bool(chosen_assessment.get("repair_triggered", False)),
            repair_policy=str(chosen_assessment.get("repair_policy", "")),
            repair_result=repair_result,
            identity_tension=float(chosen_assessment["tension"]),
            identity_repair_policy=str(chosen_assessment["repair_policy"]),
            social_focus=list(chosen_social["focus"]),
            social_alerts=list(chosen_social["alerts"]),
            social_snapshot=dict(chosen_social["snapshot"]),
            ledger_summary="",
            ledger_payload=ledger_verification.to_dict(),
            experiment_summary="",
            experiment_payload={},
        )
        self.subject_state = derive_subject_state(
            self,
            diagnostics=diagnostics,
            previous_state=self.subject_state,
        )
        ledger_seed = self.prediction_ledger.seed_predictions(
            tick=self.cycle,
            diagnostics=diagnostics,
            prediction=prediction,
            subject_state=self.subject_state,
            narrative_uncertainty=self.latest_narrative_uncertainty,
            experiment_design=self.latest_narrative_experiment,
        )
        self._refresh_inquiry_budget()
        verification_seed = self.verification_loop.refresh_targets(
            tick=self.cycle,
            ledger=self.prediction_ledger,
            diagnostics=diagnostics,
            subject_state=self.subject_state,
            narrative_uncertainty=self.latest_narrative_uncertainty,
            experiment_design=self.latest_narrative_experiment,
            inquiry_state=self.inquiry_budget_scheduler.state,
            workspace_channels=tuple(
                content.channel for content in self.last_workspace_state.broadcast_contents
            )
            if self.last_workspace_state is not None
            else (),
        )
        experiment_payload = self.latest_narrative_experiment.explanation_payload(
            chosen_action=diagnostics.chosen.choice
        )
        ledger_payload = self.prediction_ledger.explanation_payload()
        verification_payload = self.verification_loop.explanation_payload(
            chosen_action=diagnostics.chosen.choice
        )
        diagnostics.ledger_summary = str(ledger_payload["summary"])
        diagnostics.ledger_payload = {
            **ledger_verification.to_dict(),
            "seed_update": ledger_seed.to_dict(),
            **ledger_payload,
        }
        diagnostics.verification_summary = str(verification_payload["summary"])
        diagnostics.verification_payload = {
            "refresh_update": verification_refresh.to_dict(),
            "seed_update": verification_seed.to_dict(),
            **verification_payload,
        }
        diagnostics.experiment_summary = str(experiment_payload["summary"])
        diagnostics.experiment_payload = experiment_payload
        diagnostics.inquiry_scheduler_summary = str(
            self.inquiry_budget_scheduler.state.explanation_payload()["summary"]
        )
        diagnostics.inquiry_scheduler_payload = (
            self.inquiry_budget_scheduler.state.explanation_payload()
        )
        diagnostics.reconciliation_payload = self.reconciliation_engine.explanation_payload()
        diagnostics.reconciliation_summary = str(diagnostics.reconciliation_payload["summary"])
        diagnostics.subject_state_summary = self.subject_state.summary_text()
        diagnostics.subject_status_flags = dict(self.subject_state.status_flags)
        diagnostics.subject_priority_stack = [
            item.to_dict() for item in self.subject_state.subject_priority_stack[:4]
        ]
        diagnostics.structured_explanation = self.policy_evaluator.explain_structured(diagnostics)
        diagnostics.structured_explanation["reconciliation"] = diagnostics.reconciliation_payload
        diagnostics.structured_explanation["verification"] = diagnostics.verification_payload
        diagnostics.structured_explanation["experiment_design"] = diagnostics.experiment_payload
        diagnostics.structured_explanation["inquiry_scheduler"] = diagnostics.inquiry_scheduler_payload
        diagnostics.structured_explanation["subject_state"] = self.subject_state.explanation_payload()
        diagnostics.structured_explanation["narrative_uncertainty"] = (
            self.latest_narrative_uncertainty.explanation_payload()
        )
        workspace_review = self.metacognitive_layer.review_self_consistency(
            chosen_assessment,
            workspace_state=self.last_workspace_state,
        )
        diagnostics.structured_explanation["workspace_metacognitive_review"] = {
            "review_required": bool(workspace_review.review_required),
            "pause_strength": float(workspace_review.pause_strength),
            "rebias_strength": float(workspace_review.rebias_strength),
            "recommended_policy": str(workspace_review.recommended_policy),
            "notes": str(workspace_review.notes),
            "workspace_conflict_channels": list(
                self.metacognitive_layer.meta_beliefs.get("last_self_consistency_review", {}).get(
                    "workspace_conflict_channels",
                    [],
                )
            ),
        }
        verification_motive = str(verification_payload.get("verification_motive", ""))
        diagnostics.explanation = str(diagnostics.structured_explanation["text"])
        if verification_motive:
            diagnostics.explanation += " " + verification_motive
        self.last_decision_diagnostics = diagnostics
        self.last_decision_choice = diagnostics.chosen.choice
        self.last_decision_observation = {str(key): float(value) for key, value in observed.items()}
        self.last_memory_context["memory_enabled"] = self.memory_enabled
        self.last_memory_context["memory_bias"] = float(diagnostics.chosen.memory_bias)
        self.last_memory_context["pattern_bias"] = float(diagnostics.chosen.pattern_bias)
        # Stash bias fields so they survive validation perceive()'s _top_down_pass
        self._decision_bias_stash = {
            "memory_bias": self.last_memory_context["memory_bias"],
            "pattern_bias": self.last_memory_context["pattern_bias"],
        }
        self._record_identity_tension(
            chosen_action=diagnostics.chosen.choice,
            commitment_focus=diagnostics.commitment_focus,
            violated_commitments=diagnostics.violated_commitments,
            identity_tension=diagnostics.identity_tension,
            repair_policy=diagnostics.identity_repair_policy,
        )
        self.goal_stack.note_action_choice(self.cycle, diagnostics.chosen.choice)
        self._record_decision_history(diagnostics, observed)
        return {
            "observed": observed,
            "prediction": prediction,
            "errors": errors,
            "free_energy_before": free_energy_before,
            "hierarchy": hierarchy,
            "diagnostics": diagnostics,
        }

    def conscious_report(self) -> dict[str, object]:
        if self.last_decision_diagnostics is None:
            return {
                "text": (
                    "No consciously accessible contents are available yet. "
                    + "Subject state: "
                    + self.subject_state.summary_text()
                    + " Verification: "
                    + self.verification_loop.explanation_payload()["summary"]
                    + " Experiment design: "
                    + self.latest_narrative_experiment.summary
                    + " Narrative uncertainty: "
                    + self.latest_narrative_uncertainty.summary
                    + " Slow learning: "
                    + self.slow_variable_learner.state.last_summary
                ),
                "channels": [],
                "carry_over_channels": [],
                "suppressed_channels": [],
                "leakage_free": True,
                "subject_state": self.subject_state.explanation_payload(),
                "verification": self.verification_loop.explanation_payload(),
                "narrative_experiment": self.latest_narrative_experiment.explanation_payload(),
                "narrative_uncertainty": self.latest_narrative_uncertainty.explanation_payload(),
                "slow_learning": self.slow_variable_learner.explanation_payload(),
            }
        report_payload = self.global_workspace.conscious_report_payload(self.last_workspace_state)
        channels = list(report_payload["accessible_channels"])
        carry_over_channels = list(report_payload["carry_over_contents"])
        if channels:
            text = "Consciously accessible now: " + ", ".join(channels) + "."
        else:
            text = "No contents currently reached global access."
        if carry_over_channels:
            text += " Carry-over still accessible: " + ", ".join(carry_over_channels) + "."
        text += " Subject state: " + self.subject_state.summary_text()
        text += " Verification: " + self.verification_loop.explanation_payload()["summary"]
        if self.latest_narrative_experiment.summary:
            text += " Experiment design: " + self.latest_narrative_experiment.summary
        if self.latest_narrative_uncertainty.summary:
            text += " Narrative uncertainty: " + self.latest_narrative_uncertainty.summary
        if self.slow_variable_learner.state.last_summary:
            text += " Slow learning: " + self.slow_variable_learner.state.last_summary
        return {
            "text": text,
            "channels": channels,
            "carry_over_channels": carry_over_channels,
            "suppressed_channels": list(report_payload["suppressed_channels"]),
            "leakage_free": bool(report_payload["leakage_checked"]),
            "subject_state": self.subject_state.explanation_payload(),
            "verification": self.verification_loop.explanation_payload(),
            "narrative_experiment": self.latest_narrative_experiment.explanation_payload(),
            "narrative_uncertainty": self.latest_narrative_uncertainty.explanation_payload(),
            "slow_learning": self.slow_variable_learner.explanation_payload(),
        }

    def choose_intervention(
        self,
        prediction: dict[str, float],
        errors: dict[str, float],
    ) -> DecisionDiagnostics:
        """Backward-compatible chooser for older callers."""
        prediction_error = compute_prediction_error(prediction, prediction)
        neutral_preference = self.long_term_memory.preference_model.evaluate_state(
            {
                "observation": dict(prediction),
                "prediction": dict(prediction),
                "errors": dict(errors),
                "body_state": {
                    "energy": self.energy,
                    "stress": self.stress,
                    "fatigue": self.fatigue,
                    "temperature": self.temperature,
                    "dopamine": self.dopamine,
                },
                "predicted_outcome": {},
            }
        )
        expected_free_energy = neutral_preference.risk + prediction_error
        diagnostics = DecisionDiagnostics(
            chosen=InterventionScore(
                choice="rest",
                action_descriptor=self._action_schema_for_name("rest").to_dict(),
                policy_score=-expected_free_energy,
                expected_free_energy=expected_free_energy,
                predicted_error=prediction_error,
                action_ambiguity=0.0,
                risk=neutral_preference.risk,
                preferred_probability=neutral_preference.preferred_probability,
                memory_bias=0.0,
                pattern_bias=0.0,
                policy_bias=0.0,
                epistemic_bonus=0.0,
                workspace_bias=0.0,
                social_bias=0.0,
                commitment_bias=0.0,
                identity_bias=0.0,
                ledger_bias=0.0,
                subject_bias=0.0,
                goal_alignment=0.0,
                value_score=neutral_preference.value_score,
                predicted_outcome=neutral_preference.outcome,
                predicted_effects={},
                dominant_component="expected_free_energy",
                cost=0.0,
                commitment_compatibility_score=0.5,
                relevant_commitments=[],
                commitment_violations=[],
            ),
            ranked_options=[],
            prediction_error=prediction_error,
            retrieved_memories=[],
            policy_scores={},
            explanation="I chose rest because no richer decision context was available.",
            active_goal=self.goal_stack.active_goal.name,
            goal_context=self.goal_stack.get_goal_context_for_decision({"body_state": {"energy": self.energy}}),
            prediction_before_memory=dict(prediction),
            prediction_after_memory=dict(prediction),
            prediction_delta={key: 0.0 for key in prediction},
            workspace_broadcast_channels=[],
            workspace_suppressed_channels=[],
            workspace_broadcast_intensity=0.0,
            current_commitments=[],
            relevant_commitments=[],
            commitment_focus=[],
            violated_commitments=[],
            commitment_compatibility_score=0.5,
            self_inconsistency_error=0.0,
            conflict_type="none",
            severity_level="none",
            consistency_classification="aligned",
            behavioral_classification="aligned",
            repair_triggered=False,
            repair_policy="",
            repair_result={},
            identity_tension=0.0,
            identity_repair_policy="",
            social_focus=[],
            social_alerts=[],
            social_snapshot=self.social_memory.snapshot(),
            ledger_summary=self.prediction_ledger.explanation_payload()["summary"],
            ledger_payload=self.prediction_ledger.explanation_payload(),
            subject_state_summary=self.subject_state.summary_text(),
            subject_status_flags=dict(self.subject_state.status_flags),
            subject_priority_stack=[item.to_dict() for item in self.subject_state.subject_priority_stack[:4]],
        )
        self.last_decision_diagnostics = diagnostics
        return diagnostics

    def _action_regression_penalty(self, action: str) -> float:
        recent = self.action_history[-24:]
        if len(recent) < 4:
            return 0.0

        repeat_ratio = recent.count(action) / len(recent)
        streak = 0
        for previous in reversed(self.action_history):
            if previous != action:
                break
            streak += 1

        penalty = 0.0
        if repeat_ratio > 0.50:
            penalty += (repeat_ratio - 0.50) * 0.80
        if streak > 2:
            penalty += (streak - 2) * 0.18
        if streak > 5:
            penalty += 0.35 + ((streak - 5) * 0.22)
        if streak > 8:
            penalty += 1.50 + ((streak - 8) * 0.55)
        if streak > 12:
            penalty += 2.50 + ((streak - 12) * 0.80)
        if action == "rest" and streak > 6:
            penalty += 2.00 + ((streak - 6) * 0.50)
        if action == "internal_update" and repeat_ratio > 0.35:
            penalty += 0.08 + (repeat_ratio - 0.35) * 0.35
        return penalty

    def _repeated_observation_action_bonus(
        self,
        action: str,
        observed: dict[str, float],
    ) -> float:
        if not self.last_decision_observation:
            return 0.0
        previous_action = self.action_history[-1] if self.action_history else self.last_decision_choice
        if action != previous_action:
            return 0.0
        previous = self.last_decision_observation
        keys = sorted(set(previous) | set(observed))
        if not keys:
            return 0.0
        deltas = [abs(float(observed.get(key, 0.0)) - float(previous.get(key, 0.0))) for key in keys]
        mean_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        if max_delta > 0.05 or mean_delta > 0.02:
            return 0.0
        similarity = max(0.0, 1.0 - (mean_delta / 0.02))
        return 0.22 * similarity

    def apply_internal_update(self, errors: dict[str, float]) -> None:
        """Apply internal model update (high metabolic cost)."""
        self.interoceptive_layer.belief_state.absorb_error_signal(
            errors,
            strength=self.world_model.learning_rate * 0.60,
        )
        self.world_model.update_from_error(errors)
        self.strategic_layer.absorb_error_signal(
            errors,
            strength=self.world_model.learning_rate * 0.35,
        )
        self.energy = clamp(self.energy - 0.13 - self.base_metabolic_rate)
        self.fatigue = clamp(self.fatigue + self.fatigue_accumulation_rate * 0.5)
        self.stress = clamp(self.stress - 0.03)

    def apply_action_feedback(self, direct_feedback: dict[str, float]) -> None:
        """Apply feedback from world action."""
        self.energy = clamp(
            self.energy
            - self.base_metabolic_rate
            + direct_feedback.get("energy_delta", 0.0)
        )
        self.stress = clamp(self.stress + direct_feedback.get("stress_delta", 0.0))
        self.fatigue = clamp(
            self.fatigue
            + self.fatigue_accumulation_rate
            + direct_feedback.get("fatigue_delta", 0.0)
        )
        self.temperature = clamp(
            self.temperature + direct_feedback.get("temperature_delta", 0.0)
        )

    def explain_decision(self, action: str | None = None) -> str:
        if self.last_decision_diagnostics is None:
            return "No decision has been evaluated yet."
        return self.policy_evaluator.explain(self.last_decision_diagnostics, action=action)

    def integrate_outcome(
        self,
        choice: str | ActionSchema,
        observed: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        free_energy_before: float,
        free_energy_after: float,
    ) -> MemoryDecision:
        """Integrate the outcome and store in memory."""
        fe_delta = free_energy_before - free_energy_after
        reward_signal = max(0.0, fe_delta)
        self.dopamine = clamp((self.dopamine * 0.72) + reward_signal * 0.50)

        body_state = {
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
        }

        previous_body_state = dict(self.last_body_state_snapshot)
        outcome = {
            "energy_delta": body_state["energy"] - previous_body_state["energy"],
            "stress_delta": body_state["stress"] - previous_body_state["stress"],
            "free_energy_drop": fe_delta,
        }
        action = ensure_action_schema(choice)
        original_surprise_threshold = self.long_term_memory.surprise_threshold
        self.long_term_memory.surprise_threshold = max(
            0.05,
            original_surprise_threshold
            + self.global_workspace.memory_threshold_delta(self.last_workspace_state),
            + self.prediction_ledger.memory_threshold_delta(),
            + self.reconciliation_engine.memory_threshold_delta(),
            + self.verification_loop.memory_threshold_delta(),
            + subject_memory_threshold_delta(self.subject_state),
            + self.slow_variable_learner.memory_threshold_delta(),
        )
        try:
            memory_decision = self.long_term_memory.maybe_store_episode(
                self.cycle,
                observed,
                prediction,
                errors,
                action,
                outcome,
                body_state=body_state,
            )
        finally:
            self.long_term_memory.surprise_threshold = original_surprise_threshold
        self.sync_memory_awareness_to_long_term_memory()
        if not memory_decision.episode_created:
            recent_episode_actions = {
                action_name(payload.get("action_taken", payload.get("action", "")))
                for payload in self.long_term_memory.episodes[-1:]
            }
            last_episode_cycle = 0.0
            if self.long_term_memory.episodes:
                last_episode_cycle = float(
                    self.long_term_memory.episodes[-1].get(
                        "cycle",
                        self.long_term_memory.episodes[-1].get("timestamp", 0),
                    )
                )
            if action.name not in recent_episode_actions or (self.cycle - last_episode_cycle) >= 2:
                payload = self.long_term_memory.store_episode(
                    self.cycle,
                    observed,
                    prediction,
                    errors,
                    action,
                    outcome,
                    body_state=body_state,
                )
                memory_decision = MemoryDecision(
                    value_score=float(payload.get("value_score", memory_decision.value_score)),
                    prediction_error=float(payload.get("prediction_error", memory_decision.prediction_error)),
                    total_surprise=float(payload.get("total_surprise", memory_decision.total_surprise)),
                    episode_created=True,
                    predicted_outcome=str(payload.get("predicted_outcome", memory_decision.predicted_outcome)),
                    preferred_probability=float(payload.get("preferred_probability", memory_decision.preferred_probability)),
                    risk=float(payload.get("risk", memory_decision.risk)),
                    preference_log_value=float(payload.get("preference_log_value", memory_decision.preference_log_value)),
                    episode_score=float(payload.get("episode_score", memory_decision.episode_score)),
                    value_relevance=float(payload.get("value_relevance", memory_decision.value_relevance)),
                    policy_delta=float(payload.get("policy_delta", memory_decision.policy_delta)),
                    threat_significance=float(payload.get("threat_significance", memory_decision.threat_significance)),
                    redundancy_penalty=float(payload.get("redundancy_penalty", memory_decision.redundancy_penalty)),
                    episode_id=str(payload.get("episode_id", "")) or None,
                    gating_reasons=tuple(list(memory_decision.gating_reasons) + ["action_novelty_trace"]),
                )
                self.sync_memory_awareness_to_long_term_memory()
        if memory_decision.episode_created:
            self.episodes.append(
                MemoryEpisode(
                    cycle=self.cycle,
                    choice=action.name,
                    free_energy_before=free_energy_before,
                    free_energy_after=free_energy_after,
                    dopamine_gain=reward_signal,
                    observation=observed,
                    prediction=prediction,
                    errors=errors,
                    body_state=body_state,
                )
            )
            if self.last_decision_diagnostics is not None and (
                self.last_decision_diagnostics.commitment_focus
                or self.last_decision_diagnostics.violated_commitments
            ):
                self._mark_last_episode_identity_critical(
                    reason=(
                        "identity_commitment_violation"
                        if self.last_decision_diagnostics.violated_commitments
                        else "identity_commitment_reaffirmed"
                    ),
                    commitment_ids=(
                        list(self.last_decision_diagnostics.violated_commitments)
                        or list(self.last_decision_diagnostics.commitment_focus)
                    ),
                )
        self.action_history.append(action.name)
        self.action_history = self.action_history[-self.action_history_limit :]
        self.last_body_state_snapshot = dict(body_state)
        self.goal_stack.backfill_conflict_outcome(self.cycle, memory_decision.total_surprise)
        return memory_decision

    def ingest_narrative_episode(
        self,
        embodied_episode: EmbodiedNarrativeEpisode,
    ) -> dict[str, object]:
        uncertainty = UncertaintyDecompositionResult.from_dict(
            embodied_episode.uncertainty_decomposition
            if isinstance(embodied_episode.uncertainty_decomposition, dict)
            else None
        )
        self.latest_narrative_uncertainty = uncertainty
        if uncertainty.episode_id:
            self.narrative_uncertainty_history.append(uncertainty.to_dict())
            self.narrative_uncertainty_history = self.narrative_uncertainty_history[-64:]
        self.latest_narrative_experiment = self.narrative_experiment_designer.design(
            tick=embodied_episode.timestamp,
            uncertainty=uncertainty,
            action_registry=self.action_registry,
            active_goal=getattr(self.goal_stack.active_goal, "name", ""),
            subject_state=self.subject_state,
            previous_result=self.latest_narrative_experiment,
            verification_loop=self.verification_loop,
        )
        self._refresh_inquiry_budget()
        if self.latest_narrative_experiment.plans:
            self.narrative_experiment_history.append(self.latest_narrative_experiment.to_dict())
            self.narrative_experiment_history = self.narrative_experiment_history[-64:]
        social_update = self._update_social_memory_from_embodied_episode(embodied_episode)
        observation = dict(embodied_episode.observation)
        prediction, semantic_prediction = apply_schema_conditioned_prediction(
            dict(self.world_model.beliefs),
            semantic_schemas=self.long_term_memory.semantic_schemas,
            semantic_grounding=embodied_episode.semantic_grounding,
        )
        errors = {
            key: observation.get(key, 0.0) - prediction.get(key, 0.0)
            for key in sorted(set(observation) | set(prediction))
        }
        body_state = {
            "energy": float(embodied_episode.body_state.get("energy", self.energy)),
            "stress": float(embodied_episode.body_state.get("stress", self.stress)),
            "fatigue": float(embodied_episode.body_state.get("fatigue", self.fatigue)),
            "temperature": float(embodied_episode.body_state.get("temperature", self.temperature)),
        }
        outcome = self._narrative_outcome_effects(embodied_episode)
        original_surprise_threshold = self.long_term_memory.surprise_threshold
        self.long_term_memory.surprise_threshold = max(
            0.05,
            original_surprise_threshold
            + self.global_workspace.memory_threshold_delta(self.last_workspace_state),
            + self.prediction_ledger.memory_threshold_delta(),
            + self.reconciliation_engine.memory_threshold_delta(),
            + self.verification_loop.memory_threshold_delta(),
            + self.slow_variable_learner.memory_threshold_delta(),
        )
        try:
            memory_decision = self.long_term_memory.maybe_store_episode(
                embodied_episode.timestamp,
                observation,
                prediction,
                errors,
                "observe_world",
                outcome,
                body_state=body_state,
            )
        finally:
            self.long_term_memory.surprise_threshold = original_surprise_threshold
        self.sync_memory_awareness_to_long_term_memory()
        target_payload = None
        if memory_decision.episode_created and self.long_term_memory.episodes:
            target_payload = self.long_term_memory.episodes[-1]
        elif memory_decision.merged_into_episode_id is not None:
            target_payload = next(
                (
                    payload
                    for payload in self.long_term_memory.episodes
                    if payload.get("episode_id") == memory_decision.merged_into_episode_id
                ),
                None,
            )
        if target_payload is not None:
            target_payload["appraisal"] = dict(embodied_episode.appraisal)
            target_payload["narrative_tags"] = list(embodied_episode.narrative_tags)
            target_payload["compiler_confidence"] = float(embodied_episode.compiler_confidence)
            target_payload["semantic_grounding"] = dict(embodied_episode.semantic_grounding)
            target_payload["source_episode_id"] = str(
                embodied_episode.provenance.get("source_episode_id", embodied_episode.episode_id)
            )
            target_payload["source_type"] = str(
                embodied_episode.provenance.get("source_type", "narrative")
            )
            target_payload["narrative_provenance"] = dict(embodied_episode.provenance)
            target_payload["uncertainty_decomposition"] = uncertainty.to_dict()
            target_payload["experiment_design"] = self.latest_narrative_experiment.to_dict()
            if social_update.get("updated"):
                target_payload["counterpart_id"] = str(social_update.get("counterpart_id", ""))
                target_payload["social_snapshot"] = dict(social_update.get("snapshot", {}))
        trace_payload = {
            "source_episode_id": embodied_episode.provenance.get(
                "source_episode_id",
                embodied_episode.episode_id,
            ),
            "compiled_event": dict(
                embodied_episode.provenance.get("compiled_event", {})
            ),
            "appraisal": dict(embodied_episode.appraisal),
            "compatibility_observation": dict(observation),
            "prediction_before_ingestion": dict(prediction),
            "semantic_prediction": dict(semantic_prediction),
            "prediction_error": memory_decision.prediction_error,
            "total_surprise": memory_decision.total_surprise,
            "value_score": memory_decision.value_score,
            "predicted_outcome": memory_decision.predicted_outcome,
            "episode_created": memory_decision.episode_created,
            "merged_into_episode_id": memory_decision.merged_into_episode_id,
            "compiler_confidence": embodied_episode.compiler_confidence,
            "narrative_tags": list(embodied_episode.narrative_tags),
            "semantic_grounding": dict(embodied_episode.semantic_grounding),
            "narrative_uncertainty": uncertainty.to_dict(),
            "narrative_experiment": self.latest_narrative_experiment.to_dict(),
            "social_update": social_update,
            "semantic_schemas": list(self.long_term_memory.semantic_schemas[-6:]),
            "semantic_priority_bonus": semantic_uncertainty_priority_bonus(
                semantic_grounding=embodied_episode.semantic_grounding,
                semantic_schemas=self.long_term_memory.semantic_schemas,
            ),
            "verification_semantic_priority": semantic_priority_adjustment(
                prediction_id=f"narrative:{embodied_episode.episode_id}",
                semantic_grounding=embodied_episode.semantic_grounding,
                semantic_schemas=self.long_term_memory.semantic_schemas,
            ),
        }
        self.narrative_trace.append(trace_payload)
        self.narrative_trace = self.narrative_trace[-128:]
        return trace_payload

    def _narrative_outcome_effects(
        self,
        embodied_episode: EmbodiedNarrativeEpisode,
    ) -> dict[str, float]:
        appraisal = dict(embodied_episode.appraisal)
        if embodied_episode.predicted_outcome == "resource_gain":
            return {
                "free_energy_drop": 0.16 + max(0.0, appraisal.get("self_efficacy_impact", 0.0)) * 0.08,
                "energy_delta": 0.12,
                "stress_delta": -0.06,
                "fatigue_delta": 0.02,
                "temperature_delta": 0.0,
            }
        if embodied_episode.predicted_outcome == "survival_threat":
            return {
                "free_energy_drop": -0.42 - appraisal.get("physical_threat", 0.0) * 0.10,
                "energy_delta": -0.10,
                "stress_delta": 0.28 + appraisal.get("uncertainty", 0.0) * 0.10,
                "fatigue_delta": 0.12,
                "temperature_delta": 0.0,
            }
        return {
            "free_energy_drop": -0.10 - appraisal.get("loss", 0.0) * 0.06,
            "energy_delta": -0.03,
            "stress_delta": 0.16 + appraisal.get("uncertainty", 0.0) * 0.05,
            "fatigue_delta": 0.05,
            "temperature_delta": 0.0,
        }

    def _replay_prediction_error(
        self,
        episodes: list[dict[str, object]],
    ) -> float:
        if not episodes:
            return 0.0
        return mean(
            compute_prediction_error(
                dict(payload.get("observation", {})),
                self.world_model.beliefs,
            )
            for payload in episodes
        )

    def estimate_action_prediction_error(
        self,
        observation: Observation,
        action: str,
        *,
        include_slow_weights: bool = True,
    ) -> tuple[float, int | None]:
        observed = observation_dict(observation)
        drive_urgencies = {
            drive.name: drive.urgency for drive in self.drive_system.drives
        }
        try:
            novelty_deficit = 1.0 - observation.novelty
            social_isolation = 1.0 - observation.social
            self.drive_system.update_urgencies(
                self.energy,
                self.stress,
                self.fatigue,
                self.temperature,
                social_isolation,
                novelty_deficit,
                personality_modulation=self._personality_drive_modulation,
            )
            priors = self.strategic_layer.priors(
                self.energy,
                self.stress,
                self.fatigue,
                self.temperature,
                self.dopamine,
                self.drive_system,
                personality_modulation=self._personality_strategic_modulation,
            )
            prediction = self.world_model.predict(priors)
            errors = {
                key: observed.get(key, 0.0) - prediction.get(key, 0.0)
                for key in sorted(set(observed) | set(prediction))
            }
            free_energy_before = self.compute_free_energy(errors)
            current_state_snapshot = {
                "observation": observed,
                "prediction": prediction,
                "errors": errors,
                "body_state": {
                    "energy": self.energy,
                    "stress": self.stress,
                    "fatigue": self.fatigue,
                    "temperature": self.temperature,
                    "dopamine": self.dopamine,
                },
            }
            current_cluster_id = self.long_term_memory.infer_cluster_id(
                current_state_snapshot
            )
            active_goal = self.goal_stack.evaluate_priority(current_state_snapshot, record_conflict=False)
            projected_action = self._project_action(
                action=action,
                observed=observed,
                prediction=prediction,
                priors=priors,
                free_energy_before=free_energy_before,
                current_cluster_id=current_cluster_id,
                active_goal=active_goal,
            )
            prediction_error = float(projected_action["observation_distance"])
            if include_slow_weights:
                prediction_error = (
                    prediction_error
                    + max(0.0, 1.0 - float(projected_action["preferred_probability"]))
                ) / 2.0
            return (prediction_error, current_cluster_id)
        finally:
            for drive in self.drive_system.drives:
                if drive.name in drive_urgencies:
                    drive.urgency = drive_urgencies[drive.name]

    def _replay_action_prediction_error(
        self,
        episodes: list[dict[str, object]],
    ) -> float:
        if not episodes:
            return 0.0

        body_snapshot = {
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
        }
        replay_errors: list[float] = []
        try:
            for payload in episodes:
                observation_payload = payload.get("observation")
                action = action_name(payload.get("action_taken", payload.get("action", "")))
                if not isinstance(observation_payload, dict) or not action:
                    continue
                body_state = payload.get("body_state")
                if isinstance(body_state, dict):
                    self.energy = float(body_state.get("energy", self.energy))
                    self.stress = float(body_state.get("stress", self.stress))
                    self.fatigue = float(body_state.get("fatigue", self.fatigue))
                    self.temperature = float(
                        body_state.get("temperature", self.temperature)
                    )
                try:
                    prediction_error, _ = self.estimate_action_prediction_error(
                        Observation(**observation_payload),
                        action,
                    )
                except TypeError:
                    continue
                replay_errors.append(prediction_error)
        finally:
            self.energy = body_snapshot["energy"]
            self.stress = body_snapshot["stress"]
            self.fatigue = body_snapshot["fatigue"]
            self.temperature = body_snapshot["temperature"]
        if not replay_errors:
            return 0.0
        return mean(replay_errors)

    def _grouped_replay_pe(
        self,
        replay_batch: list[dict[str, object]],
    ) -> dict[tuple[int, str], float]:
        """Compute per-(cluster, action) mean PE using current world model."""
        groups: dict[tuple[int, str], list[dict[str, object]]] = {}
        for payload in replay_batch:
            cluster_id = payload.get("cluster_id")
            act = action_name(payload.get("action_taken", payload.get("action", "")))
            if not isinstance(cluster_id, int) or not act:
                continue
            groups.setdefault((cluster_id, act), []).append(payload)
        return {
            key: self._replay_action_prediction_error(payloads)
            for key, payloads in sorted(groups.items())
        }

    def _conditioned_consolidation_metrics(
        self,
        replay_batch: list[dict[str, object]],
        pe_before_raw: float,
        pe_after_raw: float,
        pe_before_grouped: dict[tuple[int, str], float],
        pe_after_grouped: dict[tuple[int, str], float],
        rule_ids: list[str],
    ) -> ConsolidationMetrics:
        """Compute conditioned PE metrics that isolate learning signal from drift.

        Instead of a single global mean PE, this method:
        1. Groups episodes by (cluster, action) and computes per-group PE
           before/after sleep, so environment drift in unrelated clusters
           doesn't mask learning in the rule-affected cluster.
        2. Marks which groups have associated rules, then reports the
           aggregated conditioned PE only for groups with rules.
        3. Estimates a novelty baseline from low-surprise episodes and
           normalises PE against it to remove background drift.
        4. Maintains a windowed PE history across sleep cycles for
           long-run convergence tracking.
        """
        def _weighted_mean(
            rows: list[ClusterPE],
            selector: str,
        ) -> float:
            total_weight = sum(max(0, row.episode_count) for row in rows)
            if total_weight <= 0:
                return 0.0
            return sum(
                getattr(row, selector) * max(0, row.episode_count)
                for row in rows
            ) / total_weight

        # --- group episodes by (cluster, action) ---
        groups: dict[tuple[int, str], list[dict[str, object]]] = {}
        for payload in replay_batch:
            cluster_id = payload.get("cluster_id")
            act = action_name(payload.get("action_taken", payload.get("action", "")))
            if not isinstance(cluster_id, int) or not act:
                continue
            groups.setdefault((cluster_id, act), []).append(payload)

        # Determine which (cluster, action) pairs have rules
        rule_clusters: set[tuple[int, str]] = set()
        for rid in rule_ids:
            # rule_id format: "sleep-{cycle_id}-{cluster}-{action}-{outcome}"
            parts = rid.split("-")
            if len(parts) >= 4:
                try:
                    rc = int(parts[2])
                    ra = parts[3]
                    rule_clusters.add((rc, ra))
                except (ValueError, IndexError):
                    pass
        for entry in self.semantic_memory:
            rule_clusters.add((entry.cluster, entry.action))

        # --- per-cluster PE before/after ---
        cluster_pe_list: list[ClusterPE] = []
        for (cid, act), payloads in sorted(groups.items()):
            has_rule = (cid, act) in rule_clusters
            cluster_pe_list.append(ClusterPE(
                cluster_id=cid,
                action=act,
                pe_before=pe_before_grouped.get((cid, act), 0.0),
                pe_after=pe_after_grouped.get((cid, act), 0.0),
                episode_count=len(payloads),
                has_rule=has_rule,
            ))

        # --- novelty baseline: mean raw PE from episodes NOT in rule-targeted
        # clusters, representing background drift + model imprecision ---
        non_ruled = [c for c in cluster_pe_list if not c.has_rule]
        if non_ruled:
            novelty_baseline = _weighted_mean(non_ruled, "pe_before")
        else:
            # Fallback: use raw per-episode prediction_error from replay batch
            raw_pes = [
                float(p.get("prediction_error", 0.0))
                for p in replay_batch
                if isinstance(p.get("prediction_error"), (int, float))
            ]
            novelty_baseline = mean(raw_pes) if raw_pes else 0.0
        safe_baseline = max(novelty_baseline, 1e-6)
        # Normalise the conditioned metric, not the legacy raw metric, so
        # unrelated-cluster drift does not leak back into the score.
        normalised_before = (
            _weighted_mean([c for c in cluster_pe_list if c.has_rule], "pe_before")
            / safe_baseline
            if cluster_pe_list
            else 0.0
        )
        normalised_after = (
            _weighted_mean([c for c in cluster_pe_list if c.has_rule], "pe_after")
            / safe_baseline
            if cluster_pe_list
            else 0.0
        )

        # --- conditioned PE: only clusters with rules ---
        ruled = [c for c in cluster_pe_list if c.has_rule]
        conditioned_before = _weighted_mean(ruled, "pe_before")
        conditioned_after = _weighted_mean(ruled, "pe_after")

        # --- windowed history (last N conditioned PE values) ---
        window_size = 5
        pe_history: list[float] = []
        for s in self.sleep_history:
            cm = s.consolidation_metrics
            if cm is not None and cm.conditioned_pe_after > 0.0:
                pe_history.append(cm.conditioned_pe_after)
        if conditioned_after > 0.0:
            pe_history.append(conditioned_after)
        pe_history = pe_history[-window_size:]
        windowed_mean = mean(pe_history) if pe_history else 0.0

        return ConsolidationMetrics(
            cluster_pe=cluster_pe_list,
            conditioned_pe_before=conditioned_before,
            conditioned_pe_after=conditioned_after,
            novelty_baseline=novelty_baseline,
            normalised_pe_before=normalised_before,
            normalised_pe_after=normalised_after,
            windowed_pe_history=pe_history,
            windowed_pe_mean=windowed_mean,
            raw_pe_before=pe_before_raw,
            raw_pe_after=pe_after_raw,
        )

    def _update_transition_model_from_replay(
        self,
        transitions: list[dict[str, object]],
    ) -> int:
        updates = 0
        for transition in transitions:
            cluster_id = transition.get("state_cluster")
            action = transition.get("action")
            next_cluster = transition.get("next_cluster")
            if not isinstance(cluster_id, int) or not isinstance(next_cluster, int):
                continue
            if not isinstance(action, str) or not action:
                continue
            if self.world_model.update_transition_count(cluster_id, action, next_cluster):
                updates += 1
        return updates

    def _mine_sleep_patterns(
        self,
        replay_batch: list[dict[str, object]],
    ) -> tuple[int, int, int, int]:
        grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
        for payload in replay_batch:
            cluster_id = payload.get("cluster_id")
            action = action_name(payload.get("action_taken", payload.get("action", "")))
            if not isinstance(cluster_id, int) or not action:
                continue
            grouped.setdefault((cluster_id, action), []).append(payload)

        patterns_found = 0
        world_model_updates = 0
        policy_bias_updates = 0
        epistemic_bonus_updates = 0
        for (cluster_id, action), payloads in sorted(grouped.items()):
            if len(payloads) < self.long_term_memory.minimum_support:
                continue
            counts: dict[str, float] = {}
            for payload in payloads:
                outcome = str(payload.get("predicted_outcome", "neutral"))
                counts[outcome] = counts.get(outcome, 0.0) + 1.0
            total = sum(counts.values())
            if total <= 0.0:
                continue
            empirical = {
                outcome: count / total for outcome, count in sorted(counts.items())
            }
            baseline = self.world_model.outcome_distribution(cluster_id, action)
            patterns_found += 1
            if (
                not baseline
                or self.world_model._kl_divergence(empirical, baseline)
                > self.world_model.kl_divergence_threshold
            ):
                self.world_model.set_outcome_distribution(cluster_id, action, empirical)
                world_model_updates += 1

            high_risk_probability = mean(
                1.0 if float(payload.get("risk", 0.0)) >= 3.0 else 0.0
                for payload in payloads
            )
            outcome_variance = (
                pvariance(float(payload.get("value_score", 0.0)) for payload in payloads)
                if len(payloads) > 1
                else 0.0
            )
            mean_risk = mean(float(payload.get("risk", 0.0)) for payload in payloads)
            if outcome_variance >= 0.15:
                self.world_model.adjust_epistemic_bonus(
                    cluster_id,
                    action,
                    delta=min(
                        0.40,
                        0.08
                        * (len(payloads) / self.long_term_memory.minimum_support)
                        * (1.0 + outcome_variance),
                    ),
                )
                epistemic_bonus_updates += 1
                continue

            if high_risk_probability >= 0.60:
                self.world_model.adjust_policy_bias(
                    cluster_id,
                    action,
                    delta=-min(
                        0.45,
                        0.08
                        * high_risk_probability
                        * (mean_risk / 3.0)
                        * (len(payloads) / self.long_term_memory.minimum_support),
                    ),
                )
                policy_bias_updates += 1
        return (
            patterns_found,
            world_model_updates,
            policy_bias_updates,
            epistemic_bonus_updates,
        )

    def _predicted_outcome_probability(self, payload: dict[str, object]) -> float:
        cluster_id = payload.get("cluster_id")
        action = action_name(payload.get("action_taken", payload.get("action", "")))
        outcome = str(payload.get("predicted_outcome", "neutral"))
        if not isinstance(cluster_id, int) or not action:
            return 0.0
        return float(
            self.world_model.outcome_distribution(cluster_id, action).get(outcome, 0.0)
        )

    def _surprise_based_forgetting(
        self,
        replay_batch: list[dict[str, object]],
    ) -> tuple[int, int]:
        if not replay_batch:
            return 0, 0

        episodes_deleted = 0
        episodes_archived = 0
        for payload in list(replay_batch):
            predicted_probability = self._predicted_outcome_probability(payload)
            if predicted_probability <= 0.0:
                continue

            prediction_error = 1.0 - predicted_probability
            risk = -self.long_term_memory.preference_model.log_preferred_probability(
                str(payload.get("predicted_outcome", "neutral"))
            )
            total_surprise = compute_total_surprise(prediction_error, risk)
            payload["dream_prediction_error"] = prediction_error
            payload["dream_total_surprise"] = total_surprise

            if self.long_term_memory.should_preserve_restart_continuity(
                payload,
                current_cycle=self.cycle,
            ):
                continue

            if total_surprise < self.long_term_memory.surprise_threshold:
                if len(self.long_term_memory.episodes) <= self.long_term_memory.minimum_active_episodes:
                    continue
                # Never delete identity-critical or protected episodes — they
                # represent core learned dangers even when fully predicted.
                lifecycle = str(payload.get("lifecycle_stage", ""))
                if lifecycle in (
                    "protected_identity_critical_episode",
                    LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
                ):
                    continue
                if str(payload.get("predicted_outcome", "neutral")) in {
                    "survival_threat",
                    "integrity_loss",
                }:
                    continue
                if self.long_term_memory.delete_episode(payload):
                    episodes_deleted += 1
                continue

            episode_age = self.cycle - int(payload.get("timestamp", payload.get("cycle", 0)))
            if episode_age > self.long_term_memory.max_active_age:
                if len(self.long_term_memory.episodes) <= self.long_term_memory.minimum_active_episodes:
                    continue
                if self.long_term_memory.archive_episode(
                    payload,
                    archive_cycle=self.cycle,
                    reason="age_out_unexplained",
                ):
                    episodes_archived += 1
        return episodes_archived, episodes_deleted

    def dream_replay(self, episodes: list[MemoryEpisode]) -> list[DreamReplay]:
        """Replay and learn from past episodes during sleep."""
        if not episodes:
            return []
        
        replays: list[DreamReplay] = []
        # Replay 2-4 random episodes
        num_replays = min(self.rng.randint(2, 4), len(episodes))
        selected = self.rng.sample(episodes, num_replays)
        
        for ep in selected:
            # Simulate alternative outcomes
            imagined_outcome = {}
            for key in ep.observation:
                # Dream outcome is a blend of actual and alternative
                actual = ep.observation[key]
                alternative = ep.prediction[key] + self.rng.uniform(-0.1, 0.1)
                imagined_outcome[key] = clamp((actual * 0.6) + (alternative * 0.4))
            
            # Compute learning signal (how much better/worse this would have been)
            imagined_errors = {
                key: imagined_outcome[key] - ep.prediction[key]
                for key in ep.observation
            }
            imagined_fe = self.compute_free_energy(imagined_errors)
            learning_signal = ep.free_energy_before - imagined_fe
            
            replays.append(
                DreamReplay(
                    episode_index=ep.cycle,
                    replayed_action=ep.choice,
                    imagined_outcome=imagined_outcome,
                    learning_signal=learning_signal,
                )
            )
            
            # Slight belief update from dream
            if learning_signal > 0.05:
                self._nudge_hierarchy(imagined_errors, scale=0.05)
        
        return replays

    def sleep(self) -> SleepSummary:
        """Sleep: consolidate memory, replay dreams, restore body.

        The action space is frozen for the duration of sleep.  Any attempt to
        call ``decision_cycle`` while ``_sleeping`` is ``True`` will raise.
        """
        self._sleeping = True
        try:
            return self._sleep_inner()
        finally:
            self._sleeping = False

    def _sleep_inner(self) -> SleepSummary:
        sleep_cycle_id = len(self.sleep_history) + 1
        recent = self.episodes[-10:] if len(self.episodes) >= 10 else self.episodes
        if not recent and not self.long_term_memory.episodes:
            summary = SleepSummary(
                0.0,
                "rest",
                dict(self.world_model.beliefs),
                0,
                0,
                sleep_cycle_id=sleep_cycle_id,
            )
            ledger_sleep = self.prediction_ledger.sleep_review(tick=self.cycle)
            reconciliation_sleep = self.reconciliation_engine.sleep_review(
                tick=self.cycle,
                sleep_cycle_id=sleep_cycle_id,
                continuity_score=self.subject_state.continuity_score,
                verification_loop=self.verification_loop,
                narrative=self.self_model.identity_narrative,
            )
            self.sleep_history.append(summary)
            self.narrative_trace.append(
                {
                    "sleep_cycle_id": sleep_cycle_id,
                    "prediction_ledger_sleep": ledger_sleep,
                    "reconciliation_sleep": reconciliation_sleep,
                    "rule_ids": [],
                    "slow_learning": self.slow_variable_learner.explanation_payload(),
                }
            )
            self.narrative_trace = self.narrative_trace[-128:]
            return summary

        replay_batch = self.long_term_memory.replay_during_sleep(rng=self.rng)
        clusters_created = self.long_term_memory.assign_clusters()
        prediction_error_before = self._replay_action_prediction_error(replay_batch)
        pe_before_grouped = self._grouped_replay_pe(replay_batch)

        # Dream replay
        dreams = self.dream_replay(recent)

        # Compute statistics
        gains = [ep.free_energy_before - ep.free_energy_after for ep in recent]
        action_scores: dict[str, list[float]] = {}
        for episode in recent:
            action_scores.setdefault(episode.choice, []).append(
                episode.free_energy_before - episode.free_energy_after
            )
        preferred_action = (
            max(action_scores.items(), key=lambda item: mean(item[1]))[0]
            if action_scores
            else "rest"
        )

        # Belief consolidation from replayed experience.
        averaged_errors: dict[str, float] = {}
        if replay_batch:
            replay_errors = [
                dict(payload.get("errors", {}))
                for payload in replay_batch
                if isinstance(payload.get("errors"), dict)
            ]
            if replay_errors:
                averaged_errors = {
                    key: mean(error.get(key, 0.0) for error in replay_errors)
                    for key in self.world_model.beliefs
                }
                self._nudge_hierarchy(averaged_errors, scale=0.12)
        elif recent:
            averaged_errors = {
                key: mean(episode.errors[key] for episode in recent)
                for key in self.world_model.beliefs
                if key in recent[0].errors
            }
            self._nudge_hierarchy(averaged_errors, scale=0.15)

        transition_dataset = self.long_term_memory.reconstruct_transitions(replay_batch)
        transition_updates = self._update_transition_model_from_replay(transition_dataset)
        (
            patterns_found,
            world_pattern_updates,
            policy_bias_updates,
            epistemic_bonus_updates,
        ) = self._mine_sleep_patterns(replay_batch)
        consolidator = SleepConsolidator(
            surprise_threshold=self.long_term_memory.surprise_threshold,
            minimum_support=self.long_term_memory.sleep_minimum_support,
            llm_extractor=self.sleep_llm_extractor,
        )
        consolidation = consolidator.consolidate(
            sleep_cycle_id=sleep_cycle_id,
            current_cycle=self.cycle,
            episodes=replay_batch,
            transition_statistics=self.world_model.transition_model,
            outcome_distributions=self.world_model.outcome_model,
        )
        semantic_entries_written, threat_updates, preference_updates = self._apply_sleep_consolidation(
            consolidation
        )
        narrative_prior_updates = self._apply_narrative_sleep_updates(replay_batch)

        # M2.7: Sync personality to precision/defense subsystems
        pp = self.self_model.personality_profile
        self.precision_manipulator.update_personality(
            neuroticism=pp.neuroticism,
            openness=pp.openness,
            extraversion=pp.extraversion,
            agreeableness=pp.agreeableness,
            conscientiousness=pp.conscientiousness,
            trust_prior=self.self_model.narrative_priors.trust_prior,
        )
        self.defense_strategy_selector.update_personality(
            neuroticism=pp.neuroticism,
            openness=pp.openness,
            extraversion=pp.extraversion,
            conscientiousness=pp.conscientiousness,
            agreeableness=pp.agreeableness,
        )
        self.precision_manipulator.decay_precision_debt()

        # M2.7: Run metacognitive observation on accumulated records
        if self.metacognitive_layer.enabled:
            manip_records = self.precision_manipulator.manipulation_history[-5:]
            strat_records = self.defense_strategy_selector.strategy_history[-5:]
            self.metacognitive_layer.observe_cycle(manip_records, strat_records)

        prediction_error_after = self._replay_action_prediction_error(replay_batch)
        pe_after_grouped = self._grouped_replay_pe(replay_batch)
        consolidation_metrics = self._conditioned_consolidation_metrics(
            replay_batch,
            pe_before_raw=prediction_error_before,
            pe_after_raw=prediction_error_after,
            pe_before_grouped=pe_before_grouped,
            pe_after_grouped=pe_after_grouped,
            rule_ids=[rule.rule_id for rule in consolidation.rules],
        )

        # Counterfactual phase: replay high-surprise episodes with alternative
        # actions, compute EFE, and absorb policy insights.
        cf_insights, cf_summary = run_counterfactual_phase(
            agent_energy=self.energy,
            current_cycle=self.cycle,
            episodes=replay_batch,
            world_model=self.world_model,
            preference_model=self.long_term_memory.preference_model,
            action_registry=self.action_registry,
            rng=self.rng,
            surprise_threshold=self.long_term_memory.surprise_threshold,
        )
        self.energy = clamp(self.energy - cf_summary.energy_spent)
        self.counterfactual_insights.extend(cf_insights)

        episodes_archived, episodes_deleted = self._surprise_based_forgetting(replay_batch)
        try:
            compression_removed = self.long_term_memory.compress_episodes(current_cycle=self.cycle)
        except TypeError as exc:
            if "current_cycle" not in str(exc):
                raise
            # Keep the sleep pipeline compatible with older zero-arg compression hooks.
            compression_removed = self.long_term_memory.compress_episodes()

        # Body restoration
        self.energy = clamp(self.energy + 0.31)
        self.stress = clamp(self.stress - 0.23)
        self.fatigue = clamp(self.fatigue - 0.39)
        self.temperature = clamp(self.temperature + (0.5 - self.temperature) * 0.3)
        self.dopamine = clamp(
            self.dopamine + max(0.0, mean(gains) if gains else 0.0) * 0.25
        )

        conflict_adjustments = self.goal_stack.review_conflicts(self.cycle)
        slow_learning_audit = self.slow_variable_learner.apply_sleep_cycle(
            sleep_cycle_id=sleep_cycle_id,
            tick=self.cycle,
            replay_batch=replay_batch,
            decision_history=list(self.decision_history),
            prediction_ledger=self.prediction_ledger,
            verification_loop=self.verification_loop,
            social_memory=self.social_memory,
            identity_tension_history=list(self.identity_tension_history),
            self_model=self.self_model,
            body_state={
                "energy": self.energy,
                "stress": self.stress,
                "fatigue": self.fatigue,
                "temperature": self.temperature,
            },
        )
        slow_learning_updates = len(
            [
                item
                for item in slow_learning_audit.updates
                if item.status in {"accepted", "clipped"} and abs(item.delta) > 1e-9
            ]
        )
        slow_learning_rejections = len(slow_learning_audit.updates) - slow_learning_updates

        summary = SleepSummary(
            average_free_energy_drop=mean(gains) if gains else 0.0,
            preferred_action=preferred_action,
            stable_beliefs=dict(self.world_model.beliefs),
            dream_replay_count=len(dreams),
            memory_consolidations=len(averaged_errors),
            sleep_cycle_id=sleep_cycle_id,
            episodes_sampled=len(replay_batch),
            clusters_created=clusters_created,
            patterns_found=patterns_found,
            world_model_updates=transition_updates + world_pattern_updates,
            policy_bias_updates=policy_bias_updates,
            epistemic_bonus_updates=epistemic_bonus_updates,
            episodes_archived=episodes_archived,
            episodes_deleted=episodes_deleted,
            memory_compressed=episodes_archived + episodes_deleted + compression_removed,
            prediction_error_before=prediction_error_before,
            prediction_error_after=prediction_error_after,
            consolidation_metrics=consolidation_metrics,
            rules_extracted=len(consolidation.rules),
            threat_updates=threat_updates,
            preference_updates=preference_updates,
            semantic_entries_written=semantic_entries_written,
            compression_removed=compression_removed,
            llm_used=consolidation.llm_used,
            rule_ids=[rule.rule_id for rule in consolidation.rules],
            counterfactual_episodes_evaluated=cf_summary.episodes_evaluated,
            counterfactual_insights_generated=cf_summary.insights_generated,
            counterfactual_insights_absorbed=cf_summary.insights_absorbed,
            counterfactual_energy_spent=cf_summary.energy_spent,
            counterfactual_log=cf_summary.counterfactual_log,
            slow_learning_updates=slow_learning_updates,
            slow_learning_rejections=slow_learning_rejections,
            slow_learning_summary=slow_learning_audit.summary,
        )
        ledger_sleep = self.prediction_ledger.sleep_review(tick=self.cycle)
        verification_sleep = self.verification_loop.process_observation(
            tick=self.cycle,
            observation={},
            ledger=self.prediction_ledger,
            source="sleep_review",
            subject_state=self.subject_state,
        )
        reconciliation_sleep = self.reconciliation_engine.sleep_review(
            tick=self.cycle,
            sleep_cycle_id=sleep_cycle_id,
            continuity_score=self.subject_state.continuity_score,
            verification_loop=self.verification_loop,
            narrative=self.self_model.identity_narrative,
        )
        self.sleep_history.append(summary)
        self.narrative_trace.append(
            {
                "sleep_cycle_id": sleep_cycle_id,
                "prediction_ledger_sleep": ledger_sleep,
                "verification_sleep": verification_sleep.to_dict(),
                "reconciliation_sleep": reconciliation_sleep,
                "rule_ids": [rule.rule_id for rule in consolidation.rules],
                "semantic_schemas": list(self.long_term_memory.semantic_schemas),
                "semantic_schema_update": dict(self.long_term_memory.latest_schema_update),
                "slow_learning": slow_learning_audit.to_dict(),
            }
        )
        self.narrative_trace = self.narrative_trace[-128:]
        if narrative_prior_updates:
            self.narrative_trace[-1]["narrative_prior_updates"] = narrative_prior_updates
        self._refresh_self_model_continuity(summary, weight_adjustments=conflict_adjustments)
        self._sync_self_model_body_schema()
        # Keep only last 3 episodes in working memory
        self.episodes = self.episodes[-3:]
        return summary

    def to_dict(self) -> dict:
        self.sync_memory_awareness_to_long_term_memory()
        self._sync_self_model_body_schema()
        return {
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
            "dopamine": self.dopamine,
            "cycle": self.cycle,
            "base_metabolic_rate": self.base_metabolic_rate,
            "fatigue_accumulation_rate": self.fatigue_accumulation_rate,
            "drive_urgencies": {
                drive.name: drive.urgency for drive in self.drive_system.drives
            },
            "process_valence": self.drive_system.process_valence.to_dict(),
            "world_model": self.world_model.to_dict(),
            "action_registry": self.action_registry.to_dict(),
            "interoceptive_layer": self.interoceptive_layer.to_dict(),
            "strategic_layer": self.strategic_layer.to_dict(),
            "long_term_memory": self.long_term_memory.to_dict(),
            "episodes": [asdict(episode) for episode in self.episodes],
            "semantic_memory": [asdict(entry) for entry in self.semantic_memory],
            "sleep_history": [asdict(summary) for summary in self.sleep_history],
            "counterfactual_insights": [
                insight.to_dict() for insight in self.counterfactual_insights
            ],
            "action_history": list(self.action_history),
            "action_history_limit": self.action_history_limit,
            "decision_history": list(self.decision_history),
            "decision_history_limit": self.decision_history_limit,
            "drive_history": list(self.drive_history),
            "drive_history_limit": self.drive_history_limit,
            "free_energy_history": list(self.free_energy_history),
            "free_energy_history_limit": self.free_energy_history_limit,
            "narrative_trace": list(self.narrative_trace),
            "goal_stack": self.goal_stack.to_dict(),
            "self_model": self.self_model.to_dict(),
            "identity_traits": asdict(self.identity_traits),
            "last_body_state_snapshot": dict(self.last_body_state_snapshot),
            "last_decision_choice": self.last_decision_choice,
            "last_decision_observation": dict(self.last_decision_observation),
            "agent_state_vector": self.agent_state_vector.to_dict(),
            "memory_cognitive_style": self.memory_cognitive_style.to_dict(),
            "memory_cycle_interval": self.memory_cycle_interval,
            "memory_backend": self._active_memory_backend(),
            "memory_enabled": self.memory_enabled,
            "predictive_coding_hyperparameters": self.predictive_coding_hyperparameters().to_dict(),
            "attention_bottleneck": self.attention_bottleneck.to_dict(),
            "last_attention_trace": (
                self.last_attention_trace.to_dict()
                if self.last_attention_trace is not None
                else None
            ),
            "last_attention_filtered_observation": dict(self.last_attention_filtered_observation),
            "global_workspace": self.global_workspace.to_dict(),
            "last_workspace_state": (
                self.last_workspace_state.to_dict()
                if self.last_workspace_state is not None
                else None
            ),
            "social_memory": self.social_memory.to_dict(),
            "identity_tension_history": list(self.identity_tension_history),
            "prediction_ledger": self.prediction_ledger.to_dict(),
            "reconciliation_engine": self.reconciliation_engine.to_dict(),
            "verification_loop": self.verification_loop.to_dict(),
            "subject_state": self.subject_state.to_dict(),
            "latest_narrative_uncertainty": self.latest_narrative_uncertainty.to_dict(),
            "narrative_uncertainty_history": list(self.narrative_uncertainty_history),
            "latest_narrative_experiment": self.latest_narrative_experiment.to_dict(),
            "narrative_experiment_history": list(self.narrative_experiment_history),
            "inquiry_budget_scheduler": self.inquiry_budget_scheduler.to_dict(),
            "slow_variable_learner": self.slow_variable_learner.to_dict(),
            # M2.7
            "precision_manipulator": self.precision_manipulator.to_dict(),
            "defense_strategy_selector": self.defense_strategy_selector.to_dict(),
            "metacognitive_layer": self.metacognitive_layer.to_dict(),
            "rng_state": repr(self.rng.getstate()),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict | None,
        rng: random.Random | None = None,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        reset_predictive_precisions: bool = False,
        state_version: str | None = None,
    ) -> SegmentAgent:
        agent = cls(
            rng=rng,
            memory_backend=str(payload.get("memory_backend", "memory_store")) if isinstance(payload, dict) else "memory_store",
            memory_cognitive_style=dict(payload.get("memory_cognitive_style", {}))
            if isinstance(payload, dict) and isinstance(payload.get("memory_cognitive_style"), dict)
            else None,
            memory_cycle_interval=int(payload.get("memory_cycle_interval", 5))
            if isinstance(payload, dict)
            else 5,
        )
        if not payload:
            if predictive_hyperparameters is not None:
                agent.configure_predictive_coding(
                    predictive_hyperparameters,
                    reset_precisions=True,
                )
            return agent

        agent.energy = float(payload.get("energy", agent.energy))
        agent.stress = float(payload.get("stress", agent.stress))
        agent.fatigue = float(payload.get("fatigue", agent.fatigue))
        agent.temperature = float(payload.get("temperature", agent.temperature))
        agent.dopamine = float(payload.get("dopamine", agent.dopamine))
        agent.cycle = int(payload.get("cycle", agent.cycle))
        agent.base_metabolic_rate = float(
            payload.get("base_metabolic_rate", agent.base_metabolic_rate)
        )
        agent.fatigue_accumulation_rate = float(
            payload.get("fatigue_accumulation_rate", agent.fatigue_accumulation_rate)
        )
        agent.strategic_layer = StrategicLayer.from_dict(payload.get("strategic_layer"))
        agent.interoceptive_layer = InteroceptiveLayer.from_dict(
            payload.get("interoceptive_layer")
        )
        agent.world_model = GenerativeWorldModel.from_dict(payload.get("world_model"))
        action_registry_payload = payload.get("action_registry", {})
        if isinstance(action_registry_payload, dict) and action_registry_payload:
            agent.action_registry = ActionRegistry.from_dict(action_registry_payload)
        else:
            agent.action_registry = build_default_action_registry()
        agent.long_term_memory = AutobiographicalMemory.from_dict(
            payload.get("long_term_memory"),
            state_version=state_version,
        )
        agent.autobiographical_memory = agent.long_term_memory
        if isinstance(payload.get("agent_state_vector"), dict):
            agent.long_term_memory.agent_state_vector = dict(payload.get("agent_state_vector", {}))
        if isinstance(payload.get("memory_cognitive_style"), dict):
            agent.long_term_memory.memory_cognitive_style = dict(payload.get("memory_cognitive_style", {}))
        if "memory_cycle_interval" in payload:
            agent.long_term_memory.memory_cycle_interval = int(payload.get("memory_cycle_interval", 5))
        agent.long_term_memory.memory_backend = str(payload.get("memory_backend", agent.long_term_memory.memory_backend))
        agent.memory_enabled = bool(payload.get("memory_enabled", True))
        agent.episodes = [
            MemoryEpisode(**episode) for episode in payload.get("episodes", [])
        ]
        semantic_memory_payload = list(payload.get("semantic_memory", []))
        sleep_history_payload = list(payload.get("sleep_history", []))
        if semantic_memory_payload and not sleep_history_payload:
            if isinstance(semantic_memory_payload[0], dict) and "average_free_energy_drop" in semantic_memory_payload[0]:
                sleep_history_payload = semantic_memory_payload
                semantic_memory_payload = []
        agent.semantic_memory = [
            SemanticMemoryEntry(**entry)
            for entry in semantic_memory_payload
            if isinstance(entry, dict)
        ]
        agent.sleep_history = [
            _deserialize_sleep_summary(summary)
            for summary in sleep_history_payload
            if isinstance(summary, dict)
        ]
        agent.counterfactual_insights = [
            CounterfactualInsight.from_dict(entry)
            for entry in payload.get("counterfactual_insights", [])
            if isinstance(entry, dict)
        ]
        agent.action_history = [
            str(choice) for choice in payload.get("action_history", [])
        ]
        agent.action_history_limit = int(
            payload.get("action_history_limit", agent.action_history_limit)
        )
        agent.attention_bottleneck = AttentionBottleneck.from_dict(
            payload.get("attention_bottleneck")
        )
        agent.last_attention_trace = AttentionTrace.from_dict(
            payload.get("last_attention_trace")
        )
        agent.last_attention_filtered_observation = {
            str(key): float(value)
            for key, value in dict(
                payload.get("last_attention_filtered_observation", {})
            ).items()
            if isinstance(value, (int, float))
        }
        agent.last_decision_observation = {
            str(key): float(value)
            for key, value in dict(
                payload.get("last_decision_observation", {})
            ).items()
            if isinstance(value, (int, float))
        }
        agent.last_decision_choice = str(payload.get("last_decision_choice", ""))
        agent.global_workspace = GlobalWorkspace.from_dict(
            payload.get("global_workspace")
            if isinstance(payload.get("global_workspace"), dict)
            else None
        )
        agent.last_workspace_state = GlobalWorkspaceState.from_dict(
            payload.get("last_workspace_state")
            if isinstance(payload.get("last_workspace_state"), dict)
            else None
        )
        agent.social_memory = SocialMemory.from_dict(
            payload.get("social_memory")
            if isinstance(payload.get("social_memory"), dict)
            else None
        )
        agent.prediction_ledger = PredictionLedger.from_dict(
            payload.get("prediction_ledger")
            if isinstance(payload.get("prediction_ledger"), dict)
            else None
        )
        agent.reconciliation_engine = ReconciliationEngine.from_dict(
            payload.get("reconciliation_engine")
            if isinstance(payload.get("reconciliation_engine"), dict)
            else None
        )
        agent.verification_loop = VerificationLoop.from_dict(
            payload.get("verification_loop")
            if isinstance(payload.get("verification_loop"), dict)
            else None
        )
        agent.subject_state = SubjectState.from_dict(
            payload.get("subject_state")
            if isinstance(payload.get("subject_state"), dict)
            else None
        )
        agent.latest_narrative_uncertainty = UncertaintyDecompositionResult.from_dict(
            payload.get("latest_narrative_uncertainty")
            if isinstance(payload.get("latest_narrative_uncertainty"), dict)
            else None
        )
        agent.narrative_uncertainty_history = [
            dict(entry)
            for entry in payload.get("narrative_uncertainty_history", [])
            if isinstance(entry, dict)
        ]
        agent.latest_narrative_experiment = ExperimentDesignResult.from_dict(
            payload.get("latest_narrative_experiment")
            if isinstance(payload.get("latest_narrative_experiment"), dict)
            else None
        )
        agent.narrative_experiment_history = [
            dict(entry)
            for entry in payload.get("narrative_experiment_history", [])
            if isinstance(entry, dict)
        ]
        agent.inquiry_budget_scheduler = InquiryBudgetScheduler.from_dict(
            payload.get("inquiry_budget_scheduler")
            if isinstance(payload.get("inquiry_budget_scheduler"), dict)
            else None
        )
        agent.slow_variable_learner = SlowVariableLearner.from_dict(
            payload.get("slow_variable_learner")
            if isinstance(payload.get("slow_variable_learner"), dict)
            else None
        )
        agent.identity_tension_history = [
            dict(entry)
            for entry in payload.get("identity_tension_history", [])
            if isinstance(entry, dict)
        ]
        agent.decision_history = [
            dict(entry)
            for entry in payload.get("decision_history", [])
            if isinstance(entry, dict)
        ]
        agent.decision_history_limit = int(
            payload.get("decision_history_limit", agent.decision_history_limit)
        )
        agent.drive_history = [
            {str(key): float(value) for key, value in entry.items()}
            for entry in payload.get("drive_history", [])
            if isinstance(entry, dict)
        ]
        agent.drive_history_limit = int(
            payload.get("drive_history_limit", agent.drive_history_limit)
        )
        agent.free_energy_history = [
            float(value) for value in payload.get("free_energy_history", [])
        ]
        agent.free_energy_history_limit = int(
            payload.get("free_energy_history_limit", agent.free_energy_history_limit)
        )
        agent.narrative_trace = [
            dict(entry)
            for entry in payload.get("narrative_trace", [])
            if isinstance(entry, dict)
        ]
        identity_traits = payload.get("identity_traits")
        if isinstance(identity_traits, dict):
            agent.identity_traits = IdentityTraits(
                risk_aversion=float(
                    identity_traits.get(
                        "risk_aversion",
                        agent.identity_traits.risk_aversion,
                    )
                ),
                resource_conservatism=float(
                    identity_traits.get(
                        "resource_conservatism",
                        agent.identity_traits.resource_conservatism,
                    )
                ),
            )
        agent.goal_stack = GoalStack.from_dict(payload.get("goal_stack"))
        agent.self_model = SelfModel.from_dict(payload.get("self_model"))
        agent.goal_stack.log_sink = agent.self_model.log_sink
        if not agent.self_model.capability_model.available_actions:
            agent.self_model.capability_model = CapabilityModel(
                action_schemas=tuple(agent.action_registry.get_all()),
                api_limits=agent.self_model.capability_model.api_limits,
            )
        else:
            agent.self_model.capability_model = CapabilityModel(
                action_schemas=tuple(agent.action_registry.get_all()),
                api_limits=agent.self_model.capability_model.api_limits,
            )
        agent.policy_evaluator = PolicyEvaluator(
            agent.identity_traits,
            agent.self_model,
            agent.goal_stack,
            agent.slow_variable_learner,
        )
        agent.decision_loop = DecisionLoop()
        agent.counterfactual_learning = CounterfactualLearning()
        last_body_state_snapshot = payload.get("last_body_state_snapshot")
        if isinstance(last_body_state_snapshot, dict):
            agent.last_body_state_snapshot = {
                "energy": float(last_body_state_snapshot.get("energy", agent.energy)),
                "stress": float(last_body_state_snapshot.get("stress", agent.stress)),
                "fatigue": float(last_body_state_snapshot.get("fatigue", agent.fatigue)),
                "temperature": float(
                    last_body_state_snapshot.get("temperature", agent.temperature)
                ),
            }
        else:
            agent.last_body_state_snapshot = {
                "energy": agent.energy,
                "stress": agent.stress,
                "fatigue": agent.fatigue,
                "temperature": agent.temperature,
            }

        # M2.7: Restore precision manipulation, defense strategy, metacognitive layer
        agent.precision_manipulator = PrecisionManipulator.from_dict(
            payload.get("precision_manipulator")
        )
        agent.defense_strategy_selector = DefenseStrategySelector.from_dict(
            payload.get("defense_strategy_selector"),
            agent.precision_manipulator,
        )
        agent.metacognitive_layer = MetaCognitiveLayer.from_dict(
            payload.get("metacognitive_layer")
        )

        drive_urgencies = payload.get("drive_urgencies", {})
        if isinstance(drive_urgencies, dict):
            for drive in agent.drive_system.drives:
                if drive.name in drive_urgencies:
                    drive.urgency = float(drive_urgencies[drive.name])
        agent.drive_system.process_valence = ProcessValenceState.from_dict(
            payload.get("process_valence")
            if isinstance(payload.get("process_valence"), dict)
            else None
        )

        if predictive_hyperparameters is None:
            predictive_hyperparameters = PredictiveCodingHyperparameters.from_dict(
                payload.get("predictive_coding_hyperparameters"),
                default=agent.predictive_coding_hyperparameters(),
            )
            reset_predictive_precisions = False
        agent.configure_predictive_coding(
            predictive_hyperparameters,
            reset_precisions=reset_predictive_precisions,
        )
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, str):
            import ast

            agent.rng.setstate(ast.literal_eval(rng_state))
        agent.init_memory_awareness(
            memory_store=agent.long_term_memory.ensure_memory_store(),
            state_vector=AgentStateVector.from_dict(dict(agent.long_term_memory.agent_state_vector)),
            cognitive_style=dict(agent.long_term_memory.memory_cognitive_style),
            memory_cycle_interval=agent.long_term_memory.memory_cycle_interval,
        )
        agent.sync_memory_awareness_to_long_term_memory()
        agent._sync_self_model_body_schema()
        if not agent.subject_state.core_identity_summary and not agent.subject_state.subject_priority_stack:
            agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)

        return agent

    def _nudge_hierarchy(self, errors: dict[str, float], scale: float) -> None:
        self.interoceptive_layer.belief_state.absorb_error_signal(
            errors,
            strength=scale * 0.70,
        )
        self.world_model.update_from_error(
            {key: value * scale for key, value in errors.items()}
        )
        self.strategic_layer.absorb_error_signal(
            errors,
            strength=scale * 0.45,
        )

    def configure_predictive_coding(
        self,
        hyperparameters: PredictiveCodingHyperparameters,
        *,
        reset_precisions: bool,
    ) -> None:
        self.interoceptive_layer.belief_state.apply_hyperparameters(
            hyperparameters.interoceptive,
            reset_precisions=reset_precisions,
        )
        self.world_model.sensorimotor_layer.belief_state.apply_hyperparameters(
            hyperparameters.sensorimotor,
            reset_precisions=reset_precisions,
        )
        self.strategic_layer.belief_state.apply_hyperparameters(
            hyperparameters.strategic,
            reset_precisions=reset_precisions,
        )

    def predictive_coding_hyperparameters(self) -> PredictiveCodingHyperparameters:
        return PredictiveCodingHyperparameters(
            interoceptive=self.interoceptive_layer.belief_state.hyperparameters(),
            sensorimotor=self.world_model.sensorimotor_layer.belief_state.hyperparameters(),
            strategic=self.strategic_layer.belief_state.hyperparameters(),
        )
