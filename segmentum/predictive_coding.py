from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from .environment import clamp


MODALITIES: tuple[str, ...] = (
    "food",
    "danger",
    "novelty",
    "shelter",
    "temperature",
    "social",
)


def default_beliefs() -> dict[str, float]:
    return {
        "food": 0.50,
        "danger": 0.30,
        "novelty": 0.50,
        "shelter": 0.40,
        "temperature": 0.50,
        "social": 0.30,
    }


def default_precisions(value: float) -> dict[str, float]:
    return {key: value for key in MODALITIES}


@dataclass
class LayerHyperparameters:
    initial_precision: float
    top_down_mix: float
    base_error_precision: float
    error_precision_scale: float
    min_precision: float
    max_precision: float
    digestion_threshold: float
    precision_decay: float

    def to_dict(self) -> dict[str, float]:
        return {
            "initial_precision": self.initial_precision,
            "top_down_mix": self.top_down_mix,
            "base_error_precision": self.base_error_precision,
            "error_precision_scale": self.error_precision_scale,
            "min_precision": self.min_precision,
            "max_precision": self.max_precision,
            "digestion_threshold": self.digestion_threshold,
            "precision_decay": self.precision_decay,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, float] | None,
        default: LayerHyperparameters,
    ) -> LayerHyperparameters:
        if not payload:
            return default
        return cls(
            initial_precision=float(
                payload.get("initial_precision", default.initial_precision)
            ),
            top_down_mix=float(payload.get("top_down_mix", default.top_down_mix)),
            base_error_precision=float(
                payload.get("base_error_precision", default.base_error_precision)
            ),
            error_precision_scale=float(
                payload.get("error_precision_scale", default.error_precision_scale)
            ),
            min_precision=float(payload.get("min_precision", default.min_precision)),
            max_precision=float(payload.get("max_precision", default.max_precision)),
            digestion_threshold=float(
                payload.get("digestion_threshold", default.digestion_threshold)
            ),
            precision_decay=float(
                payload.get("precision_decay", default.precision_decay)
            ),
        )


@dataclass
class PredictiveCodingHyperparameters:
    interoceptive: LayerHyperparameters
    sensorimotor: LayerHyperparameters
    strategic: LayerHyperparameters

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "interoceptive": self.interoceptive.to_dict(),
            "sensorimotor": self.sensorimotor.to_dict(),
            "strategic": self.strategic.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, dict[str, float]] | None,
        default: PredictiveCodingHyperparameters | None = None,
    ) -> PredictiveCodingHyperparameters:
        fallback = default or default_predictive_coding_hyperparameters()
        if not payload:
            return fallback
        return cls(
            interoceptive=LayerHyperparameters.from_dict(
                payload.get("interoceptive"),
                fallback.interoceptive,
            ),
            sensorimotor=LayerHyperparameters.from_dict(
                payload.get("sensorimotor"),
                fallback.sensorimotor,
            ),
            strategic=LayerHyperparameters.from_dict(
                payload.get("strategic"),
                fallback.strategic,
            ),
        )


def default_predictive_coding_hyperparameters() -> PredictiveCodingHyperparameters:
    return PredictiveCodingHyperparameters(
        interoceptive=LayerHyperparameters(
            initial_precision=0.45,
            top_down_mix=0.72,
            base_error_precision=0.80,
            error_precision_scale=1.40,
            min_precision=0.05,
            max_precision=3.00,
            digestion_threshold=0.06,
            precision_decay=0.01,
        ),
        sensorimotor=LayerHyperparameters(
            initial_precision=0.70,
            top_down_mix=0.55,
            base_error_precision=0.65,
            error_precision_scale=1.10,
            min_precision=0.05,
            max_precision=3.00,
            digestion_threshold=0.10,
            precision_decay=0.02,
        ),
        strategic=LayerHyperparameters(
            initial_precision=0.95,
            top_down_mix=0.63,
            base_error_precision=0.55,
            error_precision_scale=0.90,
            min_precision=0.05,
            max_precision=3.00,
            digestion_threshold=0.14,
            precision_decay=0.03,
        ),
    )


def predictive_coding_profile(profile: str) -> PredictiveCodingHyperparameters:
    base = default_predictive_coding_hyperparameters()
    profiles: dict[str, PredictiveCodingHyperparameters] = {
        "balanced": base,
        "high_precision": PredictiveCodingHyperparameters(
            interoceptive=LayerHyperparameters(
                initial_precision=0.60,
                top_down_mix=0.70,
                base_error_precision=1.00,
                error_precision_scale=1.80,
                min_precision=0.08,
                max_precision=3.00,
                digestion_threshold=0.05,
                precision_decay=0.01,
            ),
            sensorimotor=LayerHyperparameters(
                initial_precision=0.90,
                top_down_mix=0.58,
                base_error_precision=0.85,
                error_precision_scale=1.35,
                min_precision=0.08,
                max_precision=3.00,
                digestion_threshold=0.08,
                precision_decay=0.02,
            ),
            strategic=LayerHyperparameters(
                initial_precision=1.10,
                top_down_mix=0.60,
                base_error_precision=0.70,
                error_precision_scale=1.10,
                min_precision=0.08,
                max_precision=3.00,
                digestion_threshold=0.11,
                precision_decay=0.03,
            ),
        ),
        "low_precision": PredictiveCodingHyperparameters(
            interoceptive=LayerHyperparameters(
                initial_precision=0.30,
                top_down_mix=0.74,
                base_error_precision=0.55,
                error_precision_scale=0.90,
                min_precision=0.03,
                max_precision=2.50,
                digestion_threshold=0.09,
                precision_decay=0.01,
            ),
            sensorimotor=LayerHyperparameters(
                initial_precision=0.48,
                top_down_mix=0.55,
                base_error_precision=0.45,
                error_precision_scale=0.80,
                min_precision=0.03,
                max_precision=2.50,
                digestion_threshold=0.13,
                precision_decay=0.02,
            ),
            strategic=LayerHyperparameters(
                initial_precision=0.70,
                top_down_mix=0.66,
                base_error_precision=0.38,
                error_precision_scale=0.65,
                min_precision=0.03,
                max_precision=2.50,
                digestion_threshold=0.18,
                precision_decay=0.03,
            ),
        ),
        "hair_trigger": PredictiveCodingHyperparameters(
            interoceptive=LayerHyperparameters(
                initial_precision=0.45,
                top_down_mix=0.70,
                base_error_precision=1.10,
                error_precision_scale=2.00,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.03,
                precision_decay=0.01,
            ),
            sensorimotor=LayerHyperparameters(
                initial_precision=0.72,
                top_down_mix=0.55,
                base_error_precision=0.90,
                error_precision_scale=1.60,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.05,
                precision_decay=0.02,
            ),
            strategic=LayerHyperparameters(
                initial_precision=0.98,
                top_down_mix=0.63,
                base_error_precision=0.72,
                error_precision_scale=1.20,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.08,
                precision_decay=0.03,
            ),
        ),
        "thick_digestor": PredictiveCodingHyperparameters(
            interoceptive=LayerHyperparameters(
                initial_precision=0.45,
                top_down_mix=0.72,
                base_error_precision=0.72,
                error_precision_scale=1.20,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.12,
                precision_decay=0.01,
            ),
            sensorimotor=LayerHyperparameters(
                initial_precision=0.70,
                top_down_mix=0.55,
                base_error_precision=0.58,
                error_precision_scale=0.95,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.18,
                precision_decay=0.02,
            ),
            strategic=LayerHyperparameters(
                initial_precision=0.95,
                top_down_mix=0.63,
                base_error_precision=0.48,
                error_precision_scale=0.82,
                min_precision=0.05,
                max_precision=3.00,
                digestion_threshold=0.24,
                precision_decay=0.03,
            ),
        ),
    }
    if profile not in profiles:
        raise ValueError(f"unknown predictive coding profile: {profile}")
    return profiles[profile]


def predictive_coding_profile_names() -> list[str]:
    return ["balanced", "high_precision", "low_precision", "hair_trigger", "thick_digestor"]


def compose_upstream_observation(
    baseline_prediction: dict[str, float],
    propagated_error: dict[str, float],
) -> dict[str, float]:
    return {
        key: clamp(baseline_prediction[key] + propagated_error.get(key, 0.0))
        for key in baseline_prediction
    }


@dataclass
class LayerBeliefUpdate:
    layer_name: str
    incoming_observation: dict[str, float]
    top_down_prediction: dict[str, float]
    prediction: dict[str, float]
    belief_before: dict[str, float]
    belief_after: dict[str, float]
    raw_error: dict[str, float]
    error_precision: dict[str, float]
    kalman_gain: dict[str, float]
    weighted_error: dict[str, float]
    residual_error: dict[str, float]
    propagated_error: dict[str, float]
    digestion_threshold: float
    digestion_exceeded: bool

    def mean_abs_raw_error(self) -> float:
        return mean(abs(value) for value in self.raw_error.values()) if self.raw_error else 0.0

    def mean_abs_weighted_error(self) -> float:
        return (
            mean(abs(value) for value in self.weighted_error.values())
            if self.weighted_error
            else 0.0
        )

    def mean_abs_propagated_error(self) -> float:
        return (
            mean(abs(value) for value in self.propagated_error.values())
            if self.propagated_error
            else 0.0
        )


@dataclass
class HierarchicalInference:
    observation: dict[str, float]
    strategic_prior: dict[str, float]
    strategic_prediction: dict[str, float]
    sensorimotor_prediction: dict[str, float]
    interoceptive_prediction: dict[str, float]
    sensorimotor_observation: dict[str, float]
    strategic_observation: dict[str, float]
    interoceptive_update: LayerBeliefUpdate
    sensorimotor_update: LayerBeliefUpdate
    strategic_update: LayerBeliefUpdate


@dataclass
class BayesianBeliefState:
    layer_name: str
    beliefs: dict[str, float] = field(default_factory=default_beliefs)
    initial_precision: float = 0.70
    precisions: dict[str, float] = field(default_factory=dict)
    top_down_mix: float = 0.50
    base_error_precision: float = 0.60
    error_precision_scale: float = 1.00
    min_precision: float = 0.05
    max_precision: float = 3.00
    digestion_threshold: float = 0.10
    precision_decay: float = 0.02

    def __post_init__(self) -> None:
        if not self.precisions:
            self.precisions = default_precisions(self.initial_precision)

    def hyperparameters(self) -> LayerHyperparameters:
        return LayerHyperparameters(
            initial_precision=self.initial_precision,
            top_down_mix=self.top_down_mix,
            base_error_precision=self.base_error_precision,
            error_precision_scale=self.error_precision_scale,
            min_precision=self.min_precision,
            max_precision=self.max_precision,
            digestion_threshold=self.digestion_threshold,
            precision_decay=self.precision_decay,
        )

    def apply_hyperparameters(
        self,
        hyperparameters: LayerHyperparameters,
        *,
        reset_precisions: bool = False,
    ) -> None:
        self.initial_precision = hyperparameters.initial_precision
        self.top_down_mix = hyperparameters.top_down_mix
        self.base_error_precision = hyperparameters.base_error_precision
        self.error_precision_scale = hyperparameters.error_precision_scale
        self.min_precision = hyperparameters.min_precision
        self.max_precision = hyperparameters.max_precision
        self.digestion_threshold = hyperparameters.digestion_threshold
        self.precision_decay = hyperparameters.precision_decay
        if reset_precisions or not self.precisions:
            self.precisions = default_precisions(self.initial_precision)

    def predict(
        self,
        top_down_prediction: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if top_down_prediction is None:
            return dict(self.beliefs)
        prediction: dict[str, float] = {}
        for key, belief in self.beliefs.items():
            target = top_down_prediction.get(key, belief)
            prediction[key] = clamp(
                (belief * (1.0 - self.top_down_mix)) + (target * self.top_down_mix)
            )
        return prediction

    def posterior_update(
        self,
        incoming_observation: dict[str, float],
        top_down_prediction: dict[str, float] | None = None,
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        prediction = (
            dict(predicted_state)
            if predicted_state is not None
            else self.predict(top_down_prediction)
        )
        belief_before = dict(self.beliefs)
        belief_after: dict[str, float] = {}
        raw_error: dict[str, float] = {}
        error_precision: dict[str, float] = {}
        kalman_gain: dict[str, float] = {}
        weighted_error: dict[str, float] = {}
        residual_error: dict[str, float] = {}
        propagated_error: dict[str, float] = {}
        digestion_exceeded = False

        for key, observed in incoming_observation.items():
            prior_precision = self.precisions.get(key, self.base_error_precision)
            innovation = observed - prediction.get(key, belief_before.get(key, 0.5))
            observation_precision = clamp(
                self.base_error_precision + (abs(innovation) * self.error_precision_scale),
                self.min_precision,
                self.max_precision,
            )
            gain = observation_precision / (prior_precision + observation_precision)
            posterior_mean = clamp(belief_before.get(key, 0.5) + (gain * innovation))
            posterior_precision = clamp(
                prior_precision + (gain * observation_precision) - self.precision_decay,
                self.min_precision,
                self.max_precision,
            )
            remaining_error = innovation * (1.0 - gain)
            transmitted_error = (
                remaining_error * observation_precision
                if abs(remaining_error) > self.digestion_threshold
                else 0.0
            )
            if transmitted_error != 0.0:
                digestion_exceeded = True

            raw_error[key] = innovation
            error_precision[key] = observation_precision
            kalman_gain[key] = gain
            weighted_error[key] = innovation * observation_precision
            residual_error[key] = remaining_error
            propagated_error[key] = transmitted_error
            belief_after[key] = posterior_mean
            self.beliefs[key] = posterior_mean
            self.precisions[key] = posterior_precision

        dispatched_prediction = (
            dict(top_down_prediction) if top_down_prediction is not None else dict(prediction)
        )
        return LayerBeliefUpdate(
            layer_name=self.layer_name,
            incoming_observation=dict(incoming_observation),
            top_down_prediction=dispatched_prediction,
            prediction=prediction,
            belief_before=belief_before,
            belief_after=belief_after,
            raw_error=raw_error,
            error_precision=error_precision,
            kalman_gain=kalman_gain,
            weighted_error=weighted_error,
            residual_error=residual_error,
            propagated_error=propagated_error,
            digestion_threshold=self.digestion_threshold,
            digestion_exceeded=digestion_exceeded,
        )

    def absorb_error_signal(
        self,
        errors: dict[str, float],
        strength: float = 1.0,
    ) -> None:
        pseudo_observation = {
            key: clamp(self.beliefs.get(key, 0.5) + (error * strength))
            for key, error in errors.items()
            if key in self.beliefs
        }
        if pseudo_observation:
            self.posterior_update(pseudo_observation)

    def to_dict(self) -> dict[str, object]:
        return {
            "beliefs": dict(self.beliefs),
            "initial_precision": self.initial_precision,
            "precisions": dict(self.precisions),
            "top_down_mix": self.top_down_mix,
            "base_error_precision": self.base_error_precision,
            "error_precision_scale": self.error_precision_scale,
            "min_precision": self.min_precision,
            "max_precision": self.max_precision,
            "digestion_threshold": self.digestion_threshold,
            "precision_decay": self.precision_decay,
        }


@dataclass
class InteroceptiveBeliefState(BayesianBeliefState):
    layer_name: str = "interoceptive"
    beliefs: dict[str, float] = field(default_factory=default_beliefs)
    initial_precision: float = 0.45
    precisions: dict[str, float] = field(default_factory=dict)
    top_down_mix: float = 0.72
    base_error_precision: float = 0.80
    error_precision_scale: float = 1.40
    digestion_threshold: float = 0.06
    precision_decay: float = 0.01

    @classmethod
    def from_dict(cls, payload: dict | None) -> InteroceptiveBeliefState:
        state = cls()
        if not payload:
            return state
        state.beliefs = dict(payload.get("beliefs", state.beliefs))
        state.apply_hyperparameters(
            LayerHyperparameters.from_dict(payload, state.hyperparameters()),
            reset_precisions=True,
        )
        state.precisions = dict(
            payload.get("precisions", default_precisions(state.initial_precision))
        )
        return state


@dataclass
class SensorimotorBeliefState(BayesianBeliefState):
    layer_name: str = "sensorimotor"
    beliefs: dict[str, float] = field(default_factory=default_beliefs)
    initial_precision: float = 0.70
    precisions: dict[str, float] = field(default_factory=dict)
    top_down_mix: float = 0.55
    base_error_precision: float = 0.65
    error_precision_scale: float = 1.10
    digestion_threshold: float = 0.10
    precision_decay: float = 0.02

    @classmethod
    def from_dict(cls, payload: dict | None) -> SensorimotorBeliefState:
        state = cls()
        if not payload:
            return state
        state.beliefs = dict(payload.get("beliefs", state.beliefs))
        state.apply_hyperparameters(
            LayerHyperparameters.from_dict(payload, state.hyperparameters()),
            reset_precisions=True,
        )
        state.precisions = dict(
            payload.get("precisions", default_precisions(state.initial_precision))
        )
        return state


@dataclass
class StrategicBeliefState(BayesianBeliefState):
    layer_name: str = "strategic"
    beliefs: dict[str, float] = field(default_factory=default_beliefs)
    initial_precision: float = 0.95
    precisions: dict[str, float] = field(default_factory=dict)
    top_down_mix: float = 0.63
    base_error_precision: float = 0.55
    error_precision_scale: float = 0.90
    digestion_threshold: float = 0.14
    precision_decay: float = 0.03

    @classmethod
    def from_dict(cls, payload: dict | None) -> StrategicBeliefState:
        state = cls()
        if not payload:
            return state
        state.beliefs = dict(payload.get("beliefs", state.beliefs))
        state.apply_hyperparameters(
            LayerHyperparameters.from_dict(payload, state.hyperparameters()),
            reset_precisions=True,
        )
        state.precisions = dict(
            payload.get("precisions", default_precisions(state.initial_precision))
        )
        return state


@dataclass
class InteroceptiveLayer:
    belief_state: InteroceptiveBeliefState = field(default_factory=InteroceptiveBeliefState)

    def predict(self, top_down_prediction: dict[str, float]) -> dict[str, float]:
        return self.belief_state.predict(top_down_prediction)

    def assimilate(
        self,
        observation: dict[str, float],
        top_down_prediction: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.belief_state.posterior_update(
            observation,
            top_down_prediction,
            predicted_state=predicted_state,
        )

    def to_dict(self) -> dict[str, object]:
        return self.belief_state.to_dict()

    @classmethod
    def from_dict(cls, payload: dict | None) -> InteroceptiveLayer:
        return cls(belief_state=InteroceptiveBeliefState.from_dict(payload))


@dataclass
class SensorimotorLayer:
    belief_state: SensorimotorBeliefState = field(default_factory=SensorimotorBeliefState)

    def predict(self, top_down_prediction: dict[str, float]) -> dict[str, float]:
        return self.belief_state.predict(top_down_prediction)

    def assimilate(
        self,
        lower_layer_signal: dict[str, float],
        top_down_prediction: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.belief_state.posterior_update(
            lower_layer_signal,
            top_down_prediction,
            predicted_state=predicted_state,
        )

    def absorb_error_signal(
        self,
        errors: dict[str, float],
        strength: float = 1.0,
    ) -> None:
        self.belief_state.absorb_error_signal(errors, strength=strength)

    def to_dict(self) -> dict[str, object]:
        return self.belief_state.to_dict()

    @classmethod
    def from_dict(cls, payload: dict | None) -> SensorimotorLayer:
        return cls(belief_state=SensorimotorBeliefState.from_dict(payload))


def apply_schema_conditioned_prediction(
    baseline_prediction: dict[str, float],
    *,
    semantic_schemas: list[dict[str, object]] | None = None,
    semantic_grounding: dict[str, object] | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    """Bias modality predictions using memory-derived semantic schemas."""

    conditioned = dict(baseline_prediction)
    schemas = list(semantic_schemas or ())
    grounding = dict(semantic_grounding or {})
    active_motifs = {str(item) for item in grounding.get("motifs", []) if str(item)}
    applied_schema_ids: list[str] = []
    if not active_motifs:
        return conditioned, {"applied_schema_ids": [], "adjustments": {}, "match_strength": 0.0}

    total_match = 0.0
    adjustments = {key: 0.0 for key in conditioned}
    for schema in schemas:
        motif_signature = {str(item) for item in schema.get("motif_signature", []) if str(item)}
        if not motif_signature:
            continue
        overlap = len(active_motifs & motif_signature) / max(1.0, len(active_motifs | motif_signature))
        if overlap <= 0.0:
            continue
        confidence = float(schema.get("confidence", 0.0))
        strength = overlap * confidence
        total_match += strength
        applied_schema_ids.append(str(schema.get("schema_id", "")))
        direction = str(schema.get("dominant_direction", ""))
        if direction == "threat":
            adjustments["danger"] += 0.22 * strength
            adjustments["shelter"] -= 0.10 * strength
        elif direction == "social":
            adjustments["social"] += 0.22 * strength
            adjustments["danger"] -= 0.08 * strength
        elif direction == "resource":
            adjustments["food"] += 0.18 * strength
        elif direction == "exploration":
            adjustments["novelty"] += 0.20 * strength
            adjustments["danger"] += 0.05 * strength

    for key, delta in adjustments.items():
        conditioned[key] = clamp(conditioned.get(key, 0.5) + delta)
    return conditioned, {
        "applied_schema_ids": [item for item in applied_schema_ids if item],
        "adjustments": {key: round(value, 6) for key, value in adjustments.items() if abs(value) > 1e-9},
        "match_strength": round(total_match, 6),
    }
