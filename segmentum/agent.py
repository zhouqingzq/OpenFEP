from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from statistics import mean, pvariance
from typing import Callable

from .constants import ACTION_BODY_EFFECTS, ACTION_COSTS
from .drives import DriveSystem, StrategicLayer
from .environment import Observation, clamp
from .memory import (
    LongTermMemory,
    MemoryDecision,
    compute_prediction_error,
    compute_total_surprise,
)
from .predictive_coding import (
    HierarchicalInference,
    InteroceptiveLayer,
    PredictiveCodingHyperparameters,
    compose_upstream_observation,
    default_predictive_coding_hyperparameters,
)
from .sleep_consolidator import SleepConsolidator
from .types import (
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


def observation_dict(observation: Observation) -> dict[str, float]:
    return asdict(observation)


@dataclass(frozen=True)
class IdentityTraits:
    risk_aversion: float = 0.65
    resource_conservatism: float = 0.55


class PolicyEvaluator:
    """Score candidate actions with explicit policy components."""

    def __init__(self, identity_traits: IdentityTraits) -> None:
        self.identity_traits = identity_traits

    def identity_bias(
        self,
        *,
        projected_state: dict[str, float],
        predicted_outcome: dict[str, float],
        cost: float,
    ) -> float:
        danger = projected_state.get("danger", 0.0)
        shelter = projected_state.get("shelter", 0.0)
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
        return max(-1.0, min(1.0, risk_bias + resource_bias))

    def dominant_component(
        self,
        *,
        expected_free_energy: float,
        memory_bias: float,
        pattern_bias: float,
        policy_bias: float,
        epistemic_bonus: float,
        identity_bias: float,
    ) -> str:
        components = [
            ("expected_free_energy", abs(expected_free_energy)),
            ("memory_bias", abs(memory_bias)),
            ("pattern_bias", abs(pattern_bias)),
            ("policy_bias", abs(policy_bias)),
            ("epistemic_bonus", abs(epistemic_bonus)),
            ("identity_bias", abs(identity_bias)),
        ]
        components.sort(key=lambda item: (-item[1], item[0]))
        return components[0][0]

    def explain(
        self,
        diagnostics: DecisionDiagnostics,
        action: str | None = None,
    ) -> str:
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
        elif chosen.dominant_component == "identity_bias":
            reason = (
                f"identity_bias ({chosen.identity_bias:.3f}) dominated, "
                "which matches my risk-averse and resource-conserving traits."
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
                    f"after memory, pattern, and identity terms were included."
                )
        return (
            f"I chose {chosen.choice}. "
            f"This action predicted outcome '{chosen.predicted_outcome}'. "
            f"According to my preference model this outcome has {probability_band} "
            f"preferred probability ({chosen.preferred_probability:.2f}), "
            f"resulting in {risk_band} risk ({chosen.risk:.3f}). "
            f"{reason}{comparison} "
            f"This aligns with my resource_conservatism="
            f"{self.identity_traits.resource_conservatism:.2f} and "
            f"risk_aversion={self.identity_traits.risk_aversion:.2f}."
        )


class SegmentAgent:
    """A survival-first digital segment with drives, long-term memory, and dream replay."""

    def __init__(
        self,
        rng: random.Random | None = None,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        sleep_llm_extractor: Callable[[list[SleepRule], list[dict[str, object]]], list[SleepRule]]
        | None = None,
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
        self.long_term_memory = LongTermMemory()
        self.identity_traits = IdentityTraits()
        self.policy_evaluator = PolicyEvaluator(self.identity_traits)
        self.sleep_llm_extractor = sleep_llm_extractor

        self.episodes: list[MemoryEpisode] = []
        self.semantic_memory: list[SemanticMemoryEntry] = []
        self.sleep_history: list[SleepSummary] = []
        self.action_history: list[str] = []
        self.action_history_limit = 32
        self.last_body_state_snapshot = {
            "energy": self.energy,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "temperature": self.temperature,
        }
        self.last_decision_diagnostics: DecisionDiagnostics | None = None
        self._sleeping = False

        self.base_metabolic_rate = 0.015
        self.fatigue_accumulation_rate = 0.08
        self.configure_predictive_coding(
            predictive_hyperparameters or default_predictive_coding_hyperparameters(),
            reset_precisions=True,
        )

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
        )

        observed = observation_dict(observation)
        (
            strategic_prior,
            strategic_prediction,
            sensorimotor_prediction,
            interoceptive_prediction,
        ) = self._top_down_pass()
        hierarchy = self._bottom_up_pass(
            observed,
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
        )

        sensorimotor_prediction = self.world_model.predict(strategic_prediction)
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

    def _project_action(
        self,
        *,
        action: str,
        observed: dict[str, float],
        prediction: dict[str, float],
        priors: dict[str, float],
        free_energy_before: float,
        current_cluster_id: int | None,
    ) -> dict[str, object]:
        cost = ACTION_COSTS[action]
        imagined = self.world_model.imagine_action(action, prediction)
        imagined_energy = clamp(
            self.energy - cost - self.base_metabolic_rate
            + ACTION_BODY_EFFECTS[action]["energy_delta"]
        )
        imagined_stress = clamp(self.stress + ACTION_BODY_EFFECTS[action]["stress_delta"])
        imagined_fatigue = clamp(
            self.fatigue + self.fatigue_accumulation_rate
            + ACTION_BODY_EFFECTS[action]["fatigue_delta"]
        )
        imagined_temp = clamp(
            self.temperature + ACTION_BODY_EFFECTS[action]["temperature_delta"]
        )
        next_priors = self.strategic_layer.priors(
            imagined_energy,
            imagined_stress,
            imagined_fatigue,
            imagined_temp,
            self.dopamine,
            self.drive_system,
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
            action=action,
            projected_snapshot=projected_snapshot,
            predicted_effects=predicted_effects,
        )
        return {
            "predicted_state": imagined,
            "predicted_error": predicted_error,
            "action_ambiguity": action_ambiguity,
            "risk": risk,
            "preferred_probability": preferred_probability,
            "expected_free_energy": risk + predicted_error + action_ambiguity,
            "predicted_outcome": predicted_outcome,
            "predicted_effects": predicted_effects,
            "value_score": value_score,
            "cost": cost,
            "observation_distance": compute_prediction_error(observed, imagined),
        }

    def evaluate_action_options(
        self,
        observed: dict[str, float],
        prediction: dict[str, float],
        priors: dict[str, float],
        free_energy_before: float,
        current_cluster_id: int | None,
    ) -> dict[str, dict[str, object]]:
        """Project candidate actions into explicit policy components."""
        options: dict[str, dict[str, object]] = {}
        for action in ACTION_COSTS:
            options[action] = self._project_action(
                action=action,
                observed=observed,
                prediction=prediction,
                priors=priors,
                free_energy_before=free_energy_before,
                current_cluster_id=current_cluster_id,
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
        observed, prediction, errors, free_energy_before, hierarchy = self.perceive(
            observation
        )
        prediction_error = compute_prediction_error(observed, prediction)
        current_state_snapshot = {
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
        }
        priors = self.strategic_layer.priors(
            self.energy,
            self.stress,
            self.fatigue,
            self.temperature,
            self.dopamine,
            self.drive_system,
        )
        similar_memories = self.long_term_memory.retrieve_similar_memories(
            current_state_snapshot,
            k=3,
        )
        current_cluster_id = self.long_term_memory.infer_cluster_id(current_state_snapshot)
        action_options = self.evaluate_action_options(
            observed,
            prediction,
            priors,
            free_energy_before,
            current_cluster_id,
        )
        ranked_options: list[InterventionScore] = []
        for action, option in action_options.items():
            predicted_state = dict(option["predicted_state"])
            predicted_effects = dict(option["predicted_effects"])
            expected_fe = float(option["expected_free_energy"])
            memory_bias = self.long_term_memory.memory_bias(action, similar_memories)
            pattern_bias = self.long_term_memory.pattern_bias(
                action,
                action_history=self.action_history,
            )
            policy_bias = self.world_model.get_policy_bias(current_cluster_id, action)
            epistemic_bonus = self.world_model.get_epistemic_bonus(
                current_cluster_id,
                action,
            )
            identity_bias = self.policy_evaluator.identity_bias(
                projected_state=predicted_state,
                predicted_outcome=predicted_effects,
                cost=float(option["cost"]),
            )
            policy_score = (
                -expected_fe
                + memory_bias
                + pattern_bias
                + policy_bias
                + epistemic_bonus
                + identity_bias
            )
            dominant_component = self.policy_evaluator.dominant_component(
                expected_free_energy=expected_fe,
                memory_bias=memory_bias,
                pattern_bias=pattern_bias,
                policy_bias=policy_bias,
                epistemic_bonus=epistemic_bonus,
                identity_bias=identity_bias,
            )
            ranked_options.append(
                InterventionScore(
                    choice=action,
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
                    identity_bias=identity_bias,
                    value_score=float(option["value_score"]),
                    predicted_outcome=str(option["predicted_outcome"]),
                    predicted_effects=predicted_effects,
                    dominant_component=dominant_component,
                    cost=float(option["cost"]),
                )
            )
        ranked_options.sort(
            key=lambda option: (
                option.policy_score,
                -option.expected_free_energy,
                option.choice,
            ),
            reverse=True,
        )
        diagnostics = DecisionDiagnostics(
            chosen=ranked_options[0],
            ranked_options=ranked_options,
            prediction_error=prediction_error,
            retrieved_memories=similar_memories,
            policy_scores={
                option.choice: option.policy_score for option in ranked_options
            },
            explanation="",
        )
        diagnostics.explanation = self.policy_evaluator.explain(diagnostics)
        self.last_decision_diagnostics = diagnostics
        return {
            "observed": observed,
            "prediction": prediction,
            "errors": errors,
            "free_energy_before": free_energy_before,
            "hierarchy": hierarchy,
            "diagnostics": diagnostics,
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
                identity_bias=0.0,
                value_score=neutral_preference.value_score,
                predicted_outcome=neutral_preference.outcome,
                predicted_effects={},
                dominant_component="expected_free_energy",
                cost=0.0,
            ),
            ranked_options=[],
            prediction_error=prediction_error,
            retrieved_memories=[],
            policy_scores={},
            explanation="I chose rest because no richer decision context was available.",
        )
        self.last_decision_diagnostics = diagnostics
        return diagnostics

    def _action_regression_penalty(self, action: str) -> float:
        recent = self.action_history[-12:]
        if len(recent) < 4:
            return 0.0

        repeat_ratio = recent.count(action) / len(recent)
        streak = 0
        for previous in reversed(recent):
            if previous != action:
                break
            streak += 1

        penalty = 0.0
        if repeat_ratio > 0.50:
            penalty += (repeat_ratio - 0.50) * 0.45
        if streak > 3:
            penalty += (streak - 3) * 0.12
        if action == "internal_update" and repeat_ratio > 0.35:
            penalty += 0.08 + (repeat_ratio - 0.35) * 0.35
        return penalty

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
        choice: str,
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
        memory_decision = self.long_term_memory.maybe_store_episode(
            self.cycle,
            observed,
            prediction,
            errors,
            choice,
            outcome,
            body_state=body_state,
        )
        if memory_decision.episode_created:
            self.episodes.append(
                MemoryEpisode(
                    cycle=self.cycle,
                    choice=choice,
                    free_energy_before=free_energy_before,
                    free_energy_after=free_energy_after,
                    dopamine_gain=reward_signal,
                    observation=observed,
                    prediction=prediction,
                    errors=errors,
                    body_state=body_state,
                )
            )
        self.action_history.append(choice)
        self.action_history = self.action_history[-self.action_history_limit :]
        self.last_body_state_snapshot = dict(body_state)
        return memory_decision

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
            )
            priors = self.strategic_layer.priors(
                self.energy,
                self.stress,
                self.fatigue,
                self.temperature,
                self.dopamine,
                self.drive_system,
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
            projected_action = self._project_action(
                action=action,
                observed=observed,
                prediction=prediction,
                priors=priors,
                free_energy_before=free_energy_before,
                current_cluster_id=current_cluster_id,
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
                action = str(payload.get("action_taken", payload.get("action", "")))
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
            action = str(payload.get("action_taken", payload.get("action", "")))
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
        action = str(payload.get("action_taken", payload.get("action", "")))
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

            if total_surprise < self.long_term_memory.surprise_threshold:
                if self.long_term_memory.delete_episode(payload):
                    episodes_deleted += 1
                continue

            episode_age = self.cycle - int(payload.get("timestamp", payload.get("cycle", 0)))
            if episode_age > self.long_term_memory.max_active_age:
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
            self.sleep_history.append(summary)
            return summary

        replay_batch = self.long_term_memory.prioritized_replay_sample(rng=self.rng)
        clusters_created = self.long_term_memory.assign_clusters()
        prediction_error_before = self._replay_action_prediction_error(replay_batch)

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
        prediction_error_after = self._replay_action_prediction_error(replay_batch)
        episodes_archived, episodes_deleted = self._surprise_based_forgetting(replay_batch)
        compression_removed = self.long_term_memory.compress_episodes()

        # Body restoration
        self.energy = clamp(self.energy + 0.28)
        self.stress = clamp(self.stress - 0.20)
        self.fatigue = clamp(self.fatigue - 0.35)
        self.temperature = clamp(self.temperature + (0.5 - self.temperature) * 0.3)
        self.dopamine = clamp(
            self.dopamine + max(0.0, mean(gains) if gains else 0.0) * 0.25
        )

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
            rules_extracted=len(consolidation.rules),
            threat_updates=threat_updates,
            preference_updates=preference_updates,
            semantic_entries_written=semantic_entries_written,
            compression_removed=compression_removed,
            llm_used=consolidation.llm_used,
            rule_ids=[rule.rule_id for rule in consolidation.rules],
        )
        self.sleep_history.append(summary)
        # Keep only last 3 episodes in working memory
        self.episodes = self.episodes[-3:]
        return summary

    def to_dict(self) -> dict:
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
            "world_model": self.world_model.to_dict(),
            "interoceptive_layer": self.interoceptive_layer.to_dict(),
            "strategic_layer": self.strategic_layer.to_dict(),
            "long_term_memory": self.long_term_memory.to_dict(),
            "episodes": [asdict(episode) for episode in self.episodes],
            "semantic_memory": [asdict(entry) for entry in self.semantic_memory],
            "sleep_history": [asdict(summary) for summary in self.sleep_history],
            "action_history": list(self.action_history),
            "action_history_limit": self.action_history_limit,
            "identity_traits": asdict(self.identity_traits),
            "last_body_state_snapshot": dict(self.last_body_state_snapshot),
            "predictive_coding_hyperparameters": self.predictive_coding_hyperparameters().to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict | None,
        rng: random.Random | None = None,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        reset_predictive_precisions: bool = False,
    ) -> SegmentAgent:
        agent = cls(rng=rng)
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
        agent.long_term_memory = LongTermMemory.from_dict(
            payload.get("long_term_memory")
        )
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
            SleepSummary(**summary)
            for summary in sleep_history_payload
            if isinstance(summary, dict)
        ]
        agent.action_history = [
            str(choice) for choice in payload.get("action_history", [])
        ]
        agent.action_history_limit = int(
            payload.get("action_history_limit", agent.action_history_limit)
        )
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
            agent.policy_evaluator = PolicyEvaluator(agent.identity_traits)
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

        drive_urgencies = payload.get("drive_urgencies", {})
        if isinstance(drive_urgencies, dict):
            for drive in agent.drive_system.drives:
                if drive.name in drive_urgencies:
                    drive.urgency = float(drive_urgencies[drive.name])

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