from __future__ import annotations

from dataclasses import dataclass, field
from math import log

from .action_schema import ActionSchema, action_name, ensure_action_schema
from .constants import ACTION_IMAGINED_EFFECTS
from .dialogue.actions import DIALOGUE_IMAGINED_EFFECTS, is_dialogue_action
from .environment import clamp
from .preferences import canonicalize_outcome_label
from .predictive_coding import LayerBeliefUpdate, SensorimotorLayer, default_beliefs


@dataclass
class GenerativeWorldModel:
    """Mid-level beliefs that generate top-down predictions."""

    sensorimotor_layer: SensorimotorLayer = field(default_factory=SensorimotorLayer)
    learning_rate: float = 0.40
    transition_counts: dict[str, dict[str, float]] = field(default_factory=dict)
    transition_model: dict[str, dict[str, float]] = field(default_factory=dict)
    outcome_model: dict[str, dict[str, float]] = field(default_factory=dict)
    policy_biases: dict[str, dict[str, float]] = field(default_factory=dict)
    counterfactual_biases: dict[str, float] = field(default_factory=dict)
    counterfactual_candidate_buffer: dict[str, dict[str, float]] = field(default_factory=dict)
    epistemic_uncertainty_bonuses: dict[str, dict[str, float]] = field(default_factory=dict)
    threat_priors: dict[str, float] = field(default_factory=dict)
    preference_penalties: dict[str, dict[str, float]] = field(default_factory=dict)
    kl_divergence_threshold: float = 0.25
    last_prediction_details: dict[str, object] = field(default_factory=dict)

    @property
    def beliefs(self) -> dict[str, float]:
        return self.sensorimotor_layer.belief_state.beliefs

    def predict(
        self,
        priors: dict[str, float],
        memory_context: dict[str, object] | None = None,
        goal_prior: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """Generate predictions, optionally modulated by retrieved memory."""
        baseline = self.sensorimotor_layer.predict(priors)
        goal_prior_payload = goal_prior
        if goal_prior_payload is None and isinstance(memory_context, dict):
            candidate = memory_context.get("goal_prior")
            if isinstance(candidate, dict):
                goal_prior_payload = candidate
        if not memory_context and not goal_prior_payload:
            self.last_prediction_details = {
                "memory_hit": False,
                "goal_prior_applied": False,
                "prediction_before_memory": dict(baseline),
                "prediction_after_memory": dict(baseline),
                "prediction_delta": {key: 0.0 for key in baseline},
                "memory_context_summary": "",
            }
            return baseline
        if goal_prior_payload:
            modality_shifts = goal_prior_payload.get("modality_shifts", {})
            if isinstance(modality_shifts, dict):
                for key, shift in modality_shifts.items():
                    if key in baseline:
                        baseline[key] = clamp(baseline[key] + float(shift))
        if not memory_context:
            memory_context = {}

        state_projection = memory_context.get("state_projection", {})
        state_delta = memory_context.get("state_delta", {})
        prior_projection = memory_context.get("prior_projection", state_projection)
        prior_delta = memory_context.get("prior_delta", state_delta)
        residual_prior = memory_context.get("residual_prior", {})
        blend = float(memory_context.get("prediction_blend", 0.20))
        delta_gain = float(memory_context.get("delta_gain", 0.35))
        residual_gain = float(memory_context.get("residual_gain", 0.0))
        if not isinstance(state_projection, dict):
            state_projection = {}
        if not isinstance(state_delta, dict):
            state_delta = {}
        if not isinstance(prior_projection, dict):
            prior_projection = state_projection
        if not isinstance(prior_delta, dict):
            prior_delta = state_delta
        if not isinstance(residual_prior, dict):
            residual_prior = {}

        adjusted = {}
        for key, value in baseline.items():
            projection = float(prior_projection.get(key, state_projection.get(key, value)))
            delta = float(prior_delta.get(key, state_delta.get(key, 0.0)))
            residual_target = float(residual_prior.get(key, projection))
            residual_offset = residual_target - projection if key in residual_prior else 0.0
            adjusted[key] = clamp(
                (value * (1.0 - blend))
                + (projection * blend)
                + (delta * delta_gain)
                + (residual_offset * residual_gain)
            )

        delta = {
            key: adjusted[key] - baseline[key]
            for key in baseline
        }
        self.last_prediction_details = {
            "memory_hit": bool(memory_context.get("memory_hit")),
            "goal_prior_applied": bool(goal_prior_payload),
            "prediction_before_memory": dict(baseline),
            "prediction_after_memory": dict(adjusted),
            "prediction_delta": delta,
            "memory_context_summary": str(memory_context.get("summary", "")),
            "prior_projection": {
                key: float(prior_projection.get(key, state_projection.get(key, baseline[key])))
                for key in baseline
            },
            "prior_delta": {
                key: float(prior_delta.get(key, state_delta.get(key, 0.0)))
                for key in baseline
            },
            "residual_prior": {
                key: float(residual_prior.get(key, 0.0))
                for key in baseline
                if key in residual_prior
            },
        }
        if goal_prior_payload:
            self.last_prediction_details["goal_prior"] = dict(goal_prior_payload)
        return adjusted

    def refine_action_prediction(
        self,
        *,
        action: str | ActionSchema,
        projected_snapshot: dict[str, object],
        predicted_effects: dict[str, float],
        predicted_outcome: str,
        preferred_probability: float,
        risk: float,
        predicted_error: float,
        memory_context: dict[str, object] | None,
    ) -> dict[str, object]:
        action_key = action_name(action)
        if not memory_context:
            return {
                "projected_snapshot": projected_snapshot,
                "predicted_effects": predicted_effects,
                "predicted_outcome": predicted_outcome,
                "preferred_probability": preferred_probability,
                "risk": risk,
                "expected_surprise": predicted_error,
                "applied_memory": False,
                "action_descriptor": ensure_action_schema(action).to_dict(),
            }

        action_memory = memory_context.get("actions", {})
        if not isinstance(action_memory, dict):
            action_memory = {}
        action_context = action_memory.get(action_key, {})
        if not isinstance(action_context, dict):
            action_context = {}
        goal_prior_payload = memory_context.get("goal_prior", {}) if isinstance(memory_context, dict) else {}
        if not isinstance(goal_prior_payload, dict):
            goal_prior_payload = {}
        if not action_context and not goal_prior_payload:
            # Narrative/event memories can still help with state prediction, but
            # they are not valid action-specific evidence. Falling back to the
            # aggregate memory payload here leaks risk/surprise from unrelated
            # actions such as `observe_world` into every candidate action.
            return {
                "projected_snapshot": projected_snapshot,
                "predicted_effects": predicted_effects,
                "predicted_outcome": predicted_outcome,
                "preferred_probability": preferred_probability,
                "risk": risk,
                "expected_surprise": predicted_error,
                "applied_memory": False,
                "applied_goal_prior": False,
                "goal_prior": {},
                "action_descriptor": ensure_action_schema(action).to_dict(),
            }

        predicted_effects = dict(predicted_effects)
        projected_snapshot = {
            "observation": dict(projected_snapshot.get("observation", {})),
            "prediction": dict(projected_snapshot.get("prediction", {})),
            "errors": dict(projected_snapshot.get("errors", {})),
            "body_state": dict(projected_snapshot.get("body_state", {})),
        }
        blended_observation = action_context.get("observation_projection", {})
        if isinstance(blended_observation, dict):
            for key, value in blended_observation.items():
                if key in projected_snapshot["observation"]:
                    projected_snapshot["observation"][key] = clamp(
                        (projected_snapshot["observation"][key] * 0.75)
                        + (float(value) * 0.25)
                    )

        blended_effects = action_context.get("predicted_effects", {})
        if isinstance(blended_effects, dict):
            for key, value in blended_effects.items():
                predicted_effects[key] = (
                    predicted_effects.get(key, 0.0) * 0.70
                    + float(value) * 0.30
                )

        outcome_distribution = action_context.get("outcome_distribution", {})
        outcome_distribution_applied = False
        if isinstance(outcome_distribution, dict) and outcome_distribution:
            outcome_distribution_applied = True
            predicted_outcome = canonicalize_outcome_label(
                sorted(
                outcome_distribution.items(),
                key=lambda item: (-float(item[1]), item[0]),
                )[0][0],
                risk=float(risk),
                free_energy_delta=float(predicted_effects.get("free_energy_delta", 0.0) or 0.0),
            )

        field_adjustment = action_context.get("field_adjustment", {})
        applied_field = False
        if isinstance(field_adjustment, dict) and field_adjustment:
            applied_field = True
            for channel in field_adjustment.get("dominant_channels", []):
                if channel in projected_snapshot["observation"]:
                    gradient = float(field_adjustment.get("risk_adjustment", 0.0))
                    projected_snapshot["observation"][channel] = clamp(
                        projected_snapshot["observation"][channel] - (gradient * 0.10)
                    )
            predicted_effects["field_gradient_magnitude"] = float(
                field_adjustment.get("gradient_magnitude", 0.0)
            )
            predicted_effects["field_basin_strength"] = float(
                field_adjustment.get("basin_strength", 0.0)
            )
            predicted_effects["field_ridge_strength"] = float(
                field_adjustment.get("ridge_strength", 0.0)
            )

        applied_goal_prior = False
        if goal_prior_payload:
            modality_shifts = goal_prior_payload.get("modality_shifts", {})
            if isinstance(modality_shifts, dict) and modality_shifts:
                applied_goal_prior = True
                for channel, shift in modality_shifts.items():
                    numeric_shift = float(shift)
                    if channel in projected_snapshot["observation"]:
                        projected_snapshot["observation"][channel] = clamp(
                            projected_snapshot["observation"][channel] + numeric_shift
                        )
                    if channel in projected_snapshot["prediction"]:
                        projected_snapshot["prediction"][channel] = clamp(
                            projected_snapshot["prediction"][channel] + numeric_shift
                        )
                danger_shift = float(modality_shifts.get("danger", 0.0))
                social_shift = float(modality_shifts.get("social", 0.0))
                novelty_shift = float(modality_shifts.get("novelty", 0.0))
                predicted_effects["goal_prior_danger_shift"] = round(danger_shift, 6)
                predicted_effects["goal_prior_social_shift"] = round(social_shift, 6)
                predicted_effects["goal_prior_novelty_shift"] = round(novelty_shift, 6)

        preferred_probability = max(
            1e-12,
            min(
                1.0,
                (preferred_probability * 0.70)
                + (float(action_context.get("preferred_probability", preferred_probability)) * 0.30),
            ),
        )
        # If retrieved memory rewrites the predicted outcome label, we must
        # recalibrate risk against the rewritten action belief. Otherwise a
        # stale catastrophic base label (for example `survival_threat`) can
        # leak forward even after memory has shifted the action toward a
        # positive outcome distribution, which blows up EFE and masks any
        # genuine field advantage.
        if outcome_distribution_applied:
            risk = min(
                max(0.0, float(risk)),
                max(0.0, -log(max(1e-12, preferred_probability))),
            )
        risk = max(
            0.0,
            (risk * 0.65) + (float(action_context.get("risk", risk)) * 0.35),
        )
        if applied_goal_prior:
            modality_shifts = goal_prior_payload.get("modality_shifts", {})
            risk = max(
                0.0,
                risk
                + max(0.0, float(modality_shifts.get("danger", 0.0))) * 0.30
                - max(0.0, float(modality_shifts.get("shelter", 0.0))) * 0.20
            )
            preferred_probability = max(
                1e-12,
                min(
                    1.0,
                    preferred_probability
                    + max(0.0, float(modality_shifts.get("social", 0.0))) * 0.08
                    + max(0.0, float(modality_shifts.get("food", 0.0))) * 0.05,
                ),
            )
        expected_surprise = max(
            0.0,
            (predicted_error * 0.60)
            + (float(action_context.get("expected_surprise", predicted_error)) * 0.40),
        )
        if applied_goal_prior:
            modality_shifts = goal_prior_payload.get("modality_shifts", {})
            expected_surprise = max(
                0.0,
                expected_surprise
                + max(0.0, float(modality_shifts.get("novelty", 0.0))) * 0.12
                - max(0.0, float(modality_shifts.get("shelter", 0.0))) * 0.08
            )
        return {
            "projected_snapshot": projected_snapshot,
            "predicted_effects": predicted_effects,
            "predicted_outcome": predicted_outcome,
            "preferred_probability": preferred_probability,
            "risk": risk,
            "expected_surprise": expected_surprise,
            "field_projected_expected_free_energy": max(
                0.0,
                float(action_context.get("projected_expected_free_energy", 0.0) or 0.0),
            ),
            "field_guidance_strength": max(
                0.0,
                min(0.65, float(action_context.get("field_guidance_strength", 0.0) or 0.0)),
            ),
            "applied_memory": bool(action_context),
            "applied_goal_prior": applied_goal_prior,
            "goal_prior": dict(goal_prior_payload) if goal_prior_payload else {},
            "applied_field": applied_field,
            "field_diagnostics": (
                dict(field_adjustment)
                if isinstance(field_adjustment, dict)
                else {}
            ),
            "action_descriptor": ensure_action_schema(action).to_dict(),
        }

    def update_from_error(self, errors: dict[str, float]) -> None:
        self.sensorimotor_layer.absorb_error_signal(
            errors,
            strength=self.learning_rate,
        )

    def assimilate(
        self,
        lower_layer_signal: dict[str, float],
        top_down_prediction: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.sensorimotor_layer.assimilate(
            lower_layer_signal,
            top_down_prediction,
            predicted_state=predicted_state,
        )

    def imagine_action(self, action: object, prediction: dict[str, float]) -> dict[str, float]:
        action_key = action_name(action)
        if is_dialogue_action(action_key):
            deltas = DIALOGUE_IMAGINED_EFFECTS.get(action_key, {})
        else:
            deltas = ACTION_IMAGINED_EFFECTS.get(action_key, {})
        imagined = {}
        for key, value in prediction.items():
            delta = float(deltas.get(key, 0.0))
            imagined[key] = clamp(float(value) + delta)
        return imagined

    def state_action_key(self, cluster_id: int, action: object) -> str:
        return f"{cluster_id}:{action_name(action)}"

    def transition_distribution(
        self,
        cluster_id: int,
        action: str,
    ) -> dict[str, float]:
        return dict(self.transition_model.get(self.state_action_key(cluster_id, action), {}))

    def outcome_distribution(
        self,
        cluster_id: int,
        action: str,
    ) -> dict[str, float]:
        return dict(self.outcome_model.get(self.state_action_key(cluster_id, action), {}))

    def update_transition_count(
        self,
        cluster_id: int,
        action: str,
        next_cluster_id: int,
    ) -> bool:
        key = self.state_action_key(cluster_id, action)
        baseline = dict(self.transition_model.get(key, {}))
        counts = self.transition_counts.setdefault(key, {})
        next_key = str(next_cluster_id)
        counts[next_key] = counts.get(next_key, 0.0) + 1.0
        total = sum(counts.values())
        if total <= 0.0:
            self.transition_model[key] = {}
            return False
        empirical = {
            state: count / total for state, count in sorted(counts.items())
        }
        if not baseline or self._kl_divergence(empirical, baseline) > self.kl_divergence_threshold:
            self.transition_model[key] = empirical
            return True
        return False

    def set_outcome_distribution(
        self,
        cluster_id: int,
        action: str,
        distribution: dict[str, float],
    ) -> None:
        key = self.state_action_key(cluster_id, action)
        total = sum(max(0.0, value) for value in distribution.values())
        if total <= 0.0:
            self.outcome_model[key] = {}
            return
        self.outcome_model[key] = {
            outcome: max(0.0, value) / total
            for outcome, value in sorted(distribution.items())
        }

    def get_policy_bias(self, cluster_id: int | None, action: object) -> float:
        action_key = action_name(action)
        cluster_bias = 0.0
        if cluster_id is not None:
            cluster_bias = float(self.policy_biases.get(str(cluster_id), {}).get(action_key, 0.0))
        elif self.policy_biases:
            cluster_bias = sum(
                float(cluster_map.get(action_key, 0.0))
                for cluster_map in self.policy_biases.values()
            ) / max(1, len(self.policy_biases))
        global_cf_bias = float(self.counterfactual_biases.get(action_key, 0.0))
        return cluster_bias + global_cf_bias

    def get_epistemic_bonus(self, cluster_id: int | None, action: object) -> float:
        if cluster_id is None:
            return 0.0
        return float(
            self.epistemic_uncertainty_bonuses.get(str(cluster_id), {}).get(action_name(action), 0.0)
        )

    def get_threat_prior(self, cluster_id: int | None) -> float:
        if cluster_id is None:
            return 0.0
        return float(self.threat_priors.get(str(cluster_id), 0.0))

    def get_preference_penalty(self, cluster_id: int | None, action: object) -> float:
        if cluster_id is None:
            return 0.0
        return float(self.preference_penalties.get(str(cluster_id), {}).get(action_name(action), 0.0))

    def adjust_policy_bias(self, cluster_id: int, action: object, delta: float) -> float:
        cluster_key = str(cluster_id)
        biases = self.policy_biases.setdefault(cluster_key, {})
        action_key = action_name(action)
        updated = max(-1.0, min(1.0, float(biases.get(action_key, 0.0)) + delta))
        biases[action_key] = updated
        return updated

    def adjust_epistemic_bonus(self, cluster_id: int, action: object, delta: float) -> float:
        cluster_key = str(cluster_id)
        bonuses = self.epistemic_uncertainty_bonuses.setdefault(cluster_key, {})
        action_key = action_name(action)
        updated = max(0.0, min(1.0, float(bonuses.get(action_key, 0.0)) + delta))
        bonuses[action_key] = updated
        return updated

    def adjust_threat_prior(self, cluster_id: int, delta: float) -> float:
        cluster_key = str(cluster_id)
        updated = max(0.0, min(1.0, float(self.threat_priors.get(cluster_key, 0.0)) + delta))
        self.threat_priors[cluster_key] = updated
        return updated

    def adjust_preference_penalty(self, cluster_id: int, action: object, delta: float) -> float:
        cluster_key = str(cluster_id)
        penalties = self.preference_penalties.setdefault(cluster_key, {})
        action_key = action_name(action)
        updated = max(-2.0, min(1.0, float(penalties.get(action_key, 0.0)) + delta))
        penalties[action_key] = updated
        return updated

    def to_dict(self) -> dict:
        return {
            "beliefs": dict(self.beliefs),
            "sensorimotor_layer": self.sensorimotor_layer.to_dict(),
            "learning_rate": self.learning_rate,
            "transition_counts": {
                key: dict(value) for key, value in self.transition_counts.items()
            },
            "transition_model": {
                key: dict(value) for key, value in self.transition_model.items()
            },
            "outcome_model": {
                key: dict(value) for key, value in self.outcome_model.items()
            },
            "policy_biases": {
                key: dict(value) for key, value in self.policy_biases.items()
            },
            "counterfactual_biases": dict(self.counterfactual_biases),
            "counterfactual_candidate_buffer": {
                key: dict(value) for key, value in self.counterfactual_candidate_buffer.items()
            },
            "epistemic_uncertainty_bonuses": {
                key: dict(value) for key, value in self.epistemic_uncertainty_bonuses.items()
            },
            "threat_priors": dict(self.threat_priors),
            "preference_penalties": {
                key: dict(value) for key, value in self.preference_penalties.items()
            },
            "kl_divergence_threshold": self.kl_divergence_threshold,
            "last_prediction_details": dict(self.last_prediction_details),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> GenerativeWorldModel:
        if not payload:
            return cls()

        sensorimotor_payload = payload.get("sensorimotor_layer")
        model = cls(
            sensorimotor_layer=SensorimotorLayer.from_dict(sensorimotor_payload),
            learning_rate=float(payload.get("learning_rate", 0.40)),
            transition_counts={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("transition_counts", {})).items()
                if isinstance(value, dict)
            },
            transition_model={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("transition_model", {})).items()
                if isinstance(value, dict)
            },
            outcome_model={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("outcome_model", {})).items()
                if isinstance(value, dict)
            },
            policy_biases={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("policy_biases", {})).items()
                if isinstance(value, dict)
            },
            counterfactual_biases={
                str(key): float(value)
                for key, value in dict(payload.get("counterfactual_biases", {})).items()
            },
            counterfactual_candidate_buffer={
                str(key): {
                    str(inner_key): (
                        float(inner_value)
                        if isinstance(inner_value, (int, float))
                        else dict(inner_value)
                        if isinstance(inner_value, dict)
                        else list(inner_value)
                        if isinstance(inner_value, list)
                        else str(inner_value)
                    )
                    for inner_key, inner_value in value.items()
                }
                for key, value in dict(payload.get("counterfactual_candidate_buffer", {})).items()
                if isinstance(value, dict)
            },
            epistemic_uncertainty_bonuses={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("epistemic_uncertainty_bonuses", {})).items()
                if isinstance(value, dict)
            },
            threat_priors={
                str(key): float(value)
                for key, value in dict(payload.get("threat_priors", {})).items()
            },
            preference_penalties={
                str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in value.items()}
                for key, value in dict(payload.get("preference_penalties", {})).items()
                if isinstance(value, dict)
            },
            kl_divergence_threshold=float(payload.get("kl_divergence_threshold", 0.25)),
            last_prediction_details=dict(payload.get("last_prediction_details", {})),
        )
        if not sensorimotor_payload:
            model.sensorimotor_layer.belief_state.beliefs = (
                dict(payload.get("beliefs", {})) or default_beliefs()
            )
        return model

    def _kl_divergence(
        self,
        empirical: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        epsilon = 1e-12
        keys = set(empirical) | set(baseline)
        if not keys:
            return 0.0
        return sum(
            empirical.get(key, 0.0)
            * log(
                max(empirical.get(key, 0.0), epsilon)
                / max(baseline.get(key, epsilon), epsilon)
            )
            for key in keys
            if empirical.get(key, 0.0) > 0.0
        )
