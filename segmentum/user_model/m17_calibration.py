"""Deterministic M17 prediction-type calibration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


BIN_ORDER = (
    ("0.50-0.60", 0.50, 0.60),
    ("0.60-0.75", 0.60, 0.75),
    ("0.75-0.85", 0.75, 0.85),
    ("0.85-0.90", 0.85, 0.90),
)


def _round(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _bounded(value: Any, *, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(low, min(high, parsed)), 6)


def _bin_label(confidence: float) -> str:
    for label, low, high in BIN_ORDER:
        if confidence < high or label == BIN_ORDER[-1][0]:
            if confidence >= low:
                return label
    return BIN_ORDER[0][0]


@dataclass(frozen=True)
class ConfidenceBinStats:
    count: int = 0
    confirmed: int = 0
    brier_sum: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {"count": self.count, "confirmed": self.confirmed, "brier_sum": _round(self.brier_sum)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConfidenceBinStats":
        return cls(
            count=max(int(payload.get("count", 0) or 0), 0),
            confirmed=max(int(payload.get("confirmed", 0) or 0), 0),
            brier_sum=_round(payload.get("brier_sum", 0.0)),
        )


@dataclass(frozen=True)
class PredictionTypeCalibration:
    prediction_type: str
    precision_ema: float = 0.60
    hit_rate_ema: float = 0.60
    brier_ema: float = 0.25
    sample_count: int = 0
    confirmed_count: int = 0
    violated_count: int = 0
    last_updated_turn: int = 0
    confidence_bins: dict[str, ConfidenceBinStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_type": self.prediction_type,
            "precision_ema": _round(self.precision_ema),
            "hit_rate_ema": _round(self.hit_rate_ema),
            "brier_ema": _round(self.brier_ema),
            "sample_count": self.sample_count,
            "confirmed_count": self.confirmed_count,
            "violated_count": self.violated_count,
            "last_updated_turn": self.last_updated_turn,
            "confidence_bins": {key: value.to_dict() for key, value in self.confidence_bins.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionTypeCalibration":
        raw_bins = payload.get("confidence_bins", {})
        bins = {
            label: ConfidenceBinStats.from_dict(dict(raw_bins.get(label, {})) if isinstance(raw_bins, Mapping) else {})
            for label, *_ in BIN_ORDER
        }
        return cls(
            prediction_type=str(payload.get("prediction_type", "")),
            precision_ema=_bounded(payload.get("precision_ema", 0.60), low=0.35, high=0.90, default=0.60),
            hit_rate_ema=_bounded(payload.get("hit_rate_ema", 0.60), default=0.60),
            brier_ema=_bounded(payload.get("brier_ema", 0.25), default=0.25),
            sample_count=max(int(payload.get("sample_count", 0) or 0), 0),
            confirmed_count=max(int(payload.get("confirmed_count", 0) or 0), 0),
            violated_count=max(int(payload.get("violated_count", 0) or 0), 0),
            last_updated_turn=max(int(payload.get("last_updated_turn", 0) or 0), 0),
            confidence_bins=bins,
        )


@dataclass(frozen=True)
class PredictionCalibrationState:
    by_type: dict[str, PredictionTypeCalibration] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"by_type": {key: value.to_dict() for key, value in self.by_type.items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionCalibrationState":
        raw = payload.get("by_type", {})
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            by_type={
                str(key): PredictionTypeCalibration.from_dict(value)
                for key, value in raw.items()
                if isinstance(value, Mapping)
            }
        )

    def get(self, prediction_type: str) -> PredictionTypeCalibration:
        existing = self.by_type.get(prediction_type)
        if existing is not None:
            return existing
        return PredictionTypeCalibration(
            prediction_type=prediction_type,
            confidence_bins={label: ConfidenceBinStats() for label, *_ in BIN_ORDER},
        )


def update_prediction_calibration(
    state: PredictionCalibrationState,
    *,
    prediction_type: str,
    committed_confidence: float,
    outcome: str,
    brier_score: float | None,
    turn_id: int,
) -> PredictionCalibrationState:
    if outcome not in {"confirmed", "violated"} or brier_score is None:
        return state
    current = state.get(prediction_type)
    hit = 1.0 if outcome == "confirmed" else 0.0
    alpha = 0.20 if current.sample_count < 20 else 0.10
    next_sample = current.sample_count + 1
    next_hit = round((1.0 - alpha) * current.hit_rate_ema + alpha * hit, 6)
    next_brier = round((1.0 - alpha) * current.brier_ema + alpha * float(brier_score), 6)
    next_precision = _bounded(1.0 - next_brier, low=0.35, high=0.90, default=0.60)
    bins = dict(current.confidence_bins)
    label = _bin_label(float(committed_confidence))
    bin_stats = bins.get(label, ConfidenceBinStats())
    bins[label] = ConfidenceBinStats(
        count=bin_stats.count + 1,
        confirmed=bin_stats.confirmed + (1 if outcome == "confirmed" else 0),
        brier_sum=_round(bin_stats.brier_sum + float(brier_score)),
    )
    updated = PredictionTypeCalibration(
        prediction_type=prediction_type,
        precision_ema=next_precision,
        hit_rate_ema=next_hit,
        brier_ema=next_brier,
        sample_count=next_sample,
        confirmed_count=current.confirmed_count + (1 if outcome == "confirmed" else 0),
        violated_count=current.violated_count + (1 if outcome == "violated" else 0),
        last_updated_turn=max(int(turn_id), current.last_updated_turn),
        confidence_bins=bins,
    )
    next_by_type = dict(state.by_type)
    next_by_type[prediction_type] = updated
    return PredictionCalibrationState(by_type=next_by_type)
