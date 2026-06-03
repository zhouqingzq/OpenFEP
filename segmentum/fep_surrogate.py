from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping


STATE_MODALITIES: tuple[str, ...] = (
    "food",
    "danger",
    "novelty",
    "shelter",
    "temperature",
    "social",
)

CHANNEL_WEIGHTS: dict[str, float] = {
    "food": 1.30,
    "danger": 1.60,
    "novelty": 0.80,
    "shelter": 1.00,
    "temperature": 0.90,
    "social": 0.70,
}

BODY_PRESSURE_WEIGHTS: dict[str, float] = {
    "energy": 0.25,
    "stress": 0.28,
    "fatigue": 0.20,
    "temperature": 0.15,
}


def _clamp_unit(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _coerce_state_map(payload: Mapping[str, object] | None) -> dict[str, float]:
    if payload is None:
        return {}
    coerced: dict[str, float] = {}
    for key, value in payload.items():
        try:
            coerced[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def _mean_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(abs(value) for value in values)


@dataclass(frozen=True)
class FreeEnergySurrogate:
    raw_prediction_error: float
    precision_weighted_prediction_error: float
    body_pressure: float
    novelty_signal: float
    free_energy_surrogate: float
    component_errors: dict[str, float] = field(default_factory=dict)
    precision_weighted_errors: dict[str, float] = field(default_factory=dict)
    channel_costs: dict[str, float] = field(default_factory=dict)
    body_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_prediction_error": round(self.raw_prediction_error, 6),
            "precision_weighted_prediction_error": round(
                self.precision_weighted_prediction_error,
                6,
            ),
            "body_pressure": round(self.body_pressure, 6),
            "novelty_signal": round(self.novelty_signal, 6),
            "free_energy_surrogate": round(self.free_energy_surrogate, 6),
            "component_errors": {
                key: round(value, 6)
                for key, value in sorted(self.component_errors.items())
            },
            "precision_weighted_errors": {
                key: round(value, 6)
                for key, value in sorted(self.precision_weighted_errors.items())
            },
            "channel_costs": {
                key: round(value, 6)
                for key, value in sorted(self.channel_costs.items())
            },
            "body_components": {
                key: round(value, 6)
                for key, value in sorted(self.body_components.items())
            },
        }


@dataclass(frozen=True)
class ExpectedFreeEnergySurrogate:
    predicted_error: float
    risk_cost: float
    ambiguity_cost: float
    expected_free_energy_surrogate: float
    predicted_outcome: str = ""
    free_energy_surrogate_after: float | None = None
    free_energy_delta: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "predicted_error": round(self.predicted_error, 6),
            "risk_cost": round(self.risk_cost, 6),
            "ambiguity_cost": round(self.ambiguity_cost, 6),
            "expected_free_energy_surrogate": round(
                self.expected_free_energy_surrogate,
                6,
            ),
            "predicted_outcome": self.predicted_outcome,
        }
        if self.free_energy_surrogate_after is not None:
            payload["free_energy_surrogate_after"] = round(
                self.free_energy_surrogate_after,
                6,
            )
        if self.free_energy_delta is not None:
            payload["free_energy_delta"] = round(self.free_energy_delta, 6)
        return payload


def body_pressure_components(body_state: Mapping[str, object] | None) -> dict[str, float]:
    state = _coerce_state_map(body_state)
    energy_pressure = max(0.0, 0.45 - _clamp_unit(state.get("energy", 0.0))) * BODY_PRESSURE_WEIGHTS["energy"]
    stress_pressure = _clamp_unit(state.get("stress", 0.0)) * BODY_PRESSURE_WEIGHTS["stress"]
    fatigue_pressure = _clamp_unit(state.get("fatigue", 0.0)) * BODY_PRESSURE_WEIGHTS["fatigue"]
    thermal_pressure = abs(_clamp_unit(state.get("temperature", 0.5)) - 0.5) * BODY_PRESSURE_WEIGHTS["temperature"]
    return {
        "energy_pressure": energy_pressure,
        "stress_pressure": stress_pressure,
        "fatigue_pressure": fatigue_pressure,
        "thermal_pressure": thermal_pressure,
    }


def build_free_energy_surrogate(
    *,
    errors: Mapping[str, object] | None,
    body_state: Mapping[str, object] | None = None,
    precisions: Mapping[str, object] | None = None,
) -> FreeEnergySurrogate:
    error_map = _coerce_state_map(errors)
    precision_map = _coerce_state_map(precisions)
    component_errors = {
        key: abs(float(error_map.get(key, 0.0)))
        for key in STATE_MODALITIES
    }
    precision_weighted_errors = {
        key: component_errors[key] * float(precision_map.get(key, 1.0))
        for key in STATE_MODALITIES
    }
    channel_costs = {
        key: precision_weighted_errors[key] * CHANNEL_WEIGHTS[key]
        for key in STATE_MODALITIES
    }
    body_components = body_pressure_components(body_state)
    body_pressure = sum(body_components.values())
    raw_prediction_error = _mean_abs(list(component_errors.values()))
    precision_weighted_prediction_error = _mean_abs(
        list(precision_weighted_errors.values())
    )
    return FreeEnergySurrogate(
        raw_prediction_error=raw_prediction_error,
        precision_weighted_prediction_error=precision_weighted_prediction_error,
        body_pressure=body_pressure,
        novelty_signal=component_errors.get("novelty", 0.0),
        free_energy_surrogate=sum(channel_costs.values()) + body_pressure,
        component_errors=component_errors,
        precision_weighted_errors=precision_weighted_errors,
        channel_costs=channel_costs,
        body_components=body_components,
    )


def build_expected_free_energy_surrogate(
    *,
    predicted_error: float,
    risk_cost: float,
    ambiguity_cost: float,
    predicted_outcome: str = "",
    free_energy_surrogate_after: float | None = None,
    free_energy_before: float | None = None,
) -> ExpectedFreeEnergySurrogate:
    total = max(0.0, float(predicted_error)) + max(0.0, float(risk_cost)) + max(0.0, float(ambiguity_cost))
    free_energy_delta = None
    if free_energy_before is not None and free_energy_surrogate_after is not None:
        free_energy_delta = float(free_energy_before) - float(free_energy_surrogate_after)
    return ExpectedFreeEnergySurrogate(
        predicted_error=max(0.0, float(predicted_error)),
        risk_cost=max(0.0, float(risk_cost)),
        ambiguity_cost=max(0.0, float(ambiguity_cost)),
        expected_free_energy_surrogate=total,
        predicted_outcome=str(predicted_outcome or ""),
        free_energy_surrogate_after=(
            None
            if free_energy_surrogate_after is None
            else max(0.0, float(free_energy_surrogate_after))
        ),
        free_energy_delta=free_energy_delta,
    )
