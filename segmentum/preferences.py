from __future__ import annotations

from dataclasses import dataclass
from math import copysign, exp, log, log1p


CANONICAL_OUTCOMES = (
    "survival_threat",
    "integrity_loss",
    "resource_loss",
    "neutral",
    "resource_gain",
)

LABEL_ALIASES = {
    "survival": "survival_threat",
    "survival_threat": "survival_threat",
    "integrity": "integrity_loss",
    "integrity_loss": "integrity_loss",
    "resource": "resource_loss",
    "resource_loss": "resource_loss",
    "neutral": "neutral",
    "resource_gain": "resource_gain",
}


def _coerce_float_dict(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


@dataclass(frozen=True)
class PreferenceEvaluation:
    outcome: str
    log_preference: float
    value_score: float
    preferred_probability: float
    log_preferred_probability: float
    risk: float


@dataclass(frozen=True)
class PreferenceModel:
    """Probabilistic C-matrix style preferences over discrete outcomes."""

    survival_threat: float = -1000.0
    integrity_loss: float = -100.0
    resource_loss: float = -10.0
    neutral: float = 0.0
    resource_gain: float = 5.0

    @property
    def outcomes(self) -> tuple[str, ...]:
        return CANONICAL_OUTCOMES

    @property
    def survival(self) -> float:
        return self.survival_threat

    @property
    def integrity(self) -> float:
        return self.integrity_loss

    def _canonical_label(self, label: str) -> str:
        try:
            return LABEL_ALIASES[label]
        except KeyError as exc:
            raise ValueError(f"unknown preference label: {label}") from exc

    @property
    def log_preferences(self) -> tuple[float, ...]:
        return tuple(self.score(label) for label in self.outcomes)

    @property
    def log_probability_distribution(self) -> dict[str, float]:
        preferences = self.log_preferences
        anchor = max(preferences)
        partition = anchor + log(sum(exp(value - anchor) for value in preferences))
        return {
            label: value - partition
            for label, value in zip(self.outcomes, preferences)
        }

    @property
    def probability_distribution(self) -> dict[str, float]:
        return {
            label: exp(log_probability)
            for label, log_probability in self.log_probability_distribution.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "outcomes": list(self.outcomes),
            "log_preferences": {
                label: self.score(label) for label in self.outcomes
            },
            "probabilities": self.probability_distribution,
            "log_probabilities": self.log_probability_distribution,
        }

    def legacy_value_hierarchy_dict(self) -> dict[str, float]:
        return {
            "survival": self.survival_threat,
            "integrity": self.integrity_loss,
            "resource_loss": self.resource_loss,
            "resource": self.resource_loss,
            "neutral": self.neutral,
            "resource_gain": self.resource_gain,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> PreferenceModel:
        if not payload:
            return cls()

        if "log_preferences" in payload:
            raw_preferences = payload.get("log_preferences")
            if isinstance(raw_preferences, dict):
                payload = raw_preferences
            elif isinstance(raw_preferences, list):
                payload = {
                    label: value
                    for label, value in zip(
                        payload.get("outcomes", CANONICAL_OUTCOMES),
                        raw_preferences,
                    )
                }

        canonical: dict[str, float] = {
            "survival_threat": cls().survival_threat,
            "integrity_loss": cls().integrity_loss,
            "resource_loss": cls().resource_loss,
            "neutral": cls().neutral,
            "resource_gain": cls().resource_gain,
        }
        for label, value in payload.items():
            if label in {"outcomes", "probabilities", "log_probabilities"}:
                continue
            try:
                canonical[LABEL_ALIASES[str(label)]] = float(value)
            except (KeyError, TypeError, ValueError):
                continue
        return cls(**canonical)

    def score(self, label: str) -> float:
        return float(getattr(self, self._canonical_label(label)))

    def normalized_score(self, label: str) -> float:
        raw = self.score(label)
        scale = max(abs(value) for value in self.log_preferences) or 1.0
        if raw == 0.0:
            return 0.0
        return copysign(log1p(abs(raw)) / log1p(scale), raw)

    def preferred_probability(self, label: str) -> float:
        return self.probability_distribution[self._canonical_label(label)]

    def log_preferred_probability(self, label: str) -> float:
        return self.log_probability_distribution[self._canonical_label(label)]

    def risk(self, label: str) -> float:
        return -self.log_preferred_probability(label)

    def map_state_to_outcome(self, predicted_state: dict[str, object]) -> str:
        body_state = _coerce_float_dict(predicted_state.get("body_state"))
        observation = _coerce_float_dict(predicted_state.get("observation"))
        predicted_outcome = _coerce_float_dict(
            predicted_state.get("predicted_outcome", predicted_state.get("outcome"))
        )
        energy = body_state.get("energy", 0.5)
        stress = body_state.get("stress", 0.0)
        fatigue = body_state.get("fatigue", 0.0)
        temperature = body_state.get("temperature", 0.5)
        danger = observation.get("danger", 0.0)
        free_energy_drop = float(predicted_outcome.get("free_energy_drop", 0.0))
        energy_delta = float(predicted_outcome.get("energy_delta", 0.0))
        stress_delta = float(predicted_outcome.get("stress_delta", 0.0))
        fatigue_delta = float(predicted_outcome.get("fatigue_delta", 0.0))
        temperature_delta = float(predicted_outcome.get("temperature_delta", 0.0))

        if energy <= 0.15 or danger >= 0.85 or free_energy_drop <= -0.30:
            return "survival_threat"
        if (
            stress >= 0.70
            or fatigue >= 0.80
            or abs(temperature - 0.5) >= 0.20
            or stress_delta >= 0.20
            or fatigue_delta >= 0.20
            or abs(temperature_delta) >= 0.15
        ):
            return "integrity_loss"
        if free_energy_drop > 0.0 or energy_delta > 0.0:
            return "resource_gain"
        if free_energy_drop < 0.0 or energy_delta < 0.0 or stress_delta > 0.0:
            return "resource_loss"
        return "neutral"

    def evaluate_state(self, predicted_state: dict[str, object]) -> PreferenceEvaluation:
        outcome = self.map_state_to_outcome(predicted_state)
        return PreferenceEvaluation(
            outcome=outcome,
            log_preference=self.score(outcome),
            value_score=self.normalized_score(outcome),
            preferred_probability=self.preferred_probability(outcome),
            log_preferred_probability=self.log_preferred_probability(outcome),
            risk=self.risk(outcome),
        )

    def classify(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> str:
        return self.map_state_to_outcome(
            {
                **state_snapshot,
                "predicted_outcome": dict(outcome),
            }
        )

    def evaluate_details(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> tuple[str, float, float]:
        evaluation = self.evaluate_state(
            {
                **state_snapshot,
                "predicted_outcome": dict(outcome),
            }
        )
        return evaluation.outcome, evaluation.log_preference, evaluation.value_score

    def evaluate(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> float:
        return self.evaluate_details(
            state_snapshot=state_snapshot,
            outcome=outcome,
        )[2]


ValueHierarchy = PreferenceModel
