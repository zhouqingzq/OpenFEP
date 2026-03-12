from __future__ import annotations

from dataclasses import dataclass, field
from math import log

from .constants import ACTION_IMAGINED_EFFECTS
from .environment import clamp
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

    @property
    def beliefs(self) -> dict[str, float]:
        return self.sensorimotor_layer.belief_state.beliefs

    def predict(
        self,
        priors: dict[str, float],
        memory_context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Generate predictions, optionally modulated by retrieved memory."""
        prediction = self.sensorimotor_layer.predict(priors)
        if memory_context:
            prediction = {
                key: clamp((prediction[key] * 0.80) + (memory_context[key] * 0.20))
                if key in memory_context
                else prediction[key]
                for key in prediction
            }
        return prediction

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

    def imagine_action(self, action: str, prediction: dict[str, float]) -> dict[str, float]:
        imagined = {}
        for key, value in prediction.items():
            delta = ACTION_IMAGINED_EFFECTS.get(action, {}).get(key, 0.0)
            imagined[key] = clamp(value + delta)
        return imagined

    def state_action_key(self, cluster_id: int, action: str) -> str:
        return f"{cluster_id}:{action}"

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

    def get_policy_bias(self, cluster_id: int | None, action: str) -> float:
        cluster_bias = 0.0
        if cluster_id is not None:
            cluster_bias = float(self.policy_biases.get(str(cluster_id), {}).get(action, 0.0))
        global_cf_bias = float(self.counterfactual_biases.get(action, 0.0))
        return cluster_bias + global_cf_bias

    def get_epistemic_bonus(self, cluster_id: int | None, action: str) -> float:
        if cluster_id is None:
            return 0.0
        return float(
            self.epistemic_uncertainty_bonuses.get(str(cluster_id), {}).get(action, 0.0)
        )

    def get_threat_prior(self, cluster_id: int | None) -> float:
        if cluster_id is None:
            return 0.0
        return float(self.threat_priors.get(str(cluster_id), 0.0))

    def get_preference_penalty(self, cluster_id: int | None, action: str) -> float:
        if cluster_id is None:
            return 0.0
        return float(self.preference_penalties.get(str(cluster_id), {}).get(action, 0.0))

    def adjust_policy_bias(
        self,
        cluster_id: int,
        action: str,
        delta: float,
    ) -> float:
        cluster_key = str(cluster_id)
        biases = self.policy_biases.setdefault(cluster_key, {})
        updated = max(-1.0, min(1.0, float(biases.get(action, 0.0)) + delta))
        biases[action] = updated
        return updated

    def adjust_epistemic_bonus(
        self,
        cluster_id: int,
        action: str,
        delta: float,
    ) -> float:
        cluster_key = str(cluster_id)
        bonuses = self.epistemic_uncertainty_bonuses.setdefault(cluster_key, {})
        updated = max(0.0, min(1.0, float(bonuses.get(action, 0.0)) + delta))
        bonuses[action] = updated
        return updated

    def adjust_threat_prior(self, cluster_id: int, delta: float) -> float:
        cluster_key = str(cluster_id)
        updated = max(0.0, min(1.0, float(self.threat_priors.get(cluster_key, 0.0)) + delta))
        self.threat_priors[cluster_key] = updated
        return updated

    def adjust_preference_penalty(
        self,
        cluster_id: int,
        action: str,
        delta: float,
    ) -> float:
        cluster_key = str(cluster_id)
        penalties = self.preference_penalties.setdefault(cluster_key, {})
        updated = max(-2.0, min(1.0, float(penalties.get(action, 0.0)) + delta))
        penalties[action] = updated
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