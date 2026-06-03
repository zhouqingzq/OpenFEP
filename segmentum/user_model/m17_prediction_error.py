"""Deterministic M17 prediction settlement error helpers."""

from __future__ import annotations

from math import log


PREDICTION_ERROR_EPS = 1e-6
VALID_OUTCOMES = frozenset({"confirmed", "violated", "unclear", "expired"})


def normalize_prediction_outcome(raw: object) -> str:
    text = str(raw or "").strip().casefold()
    if text in {"confirmed", "confirm", "hit", "true"}:
        return "confirmed"
    if text in {"violated", "violate", "miss", "false"}:
        return "violated"
    if text in {"expired", "timeout"}:
        return "expired"
    return "unclear"


def prediction_error_from_outcome(committed_confidence: float, outcome: object) -> float | None:
    normalized = normalize_prediction_outcome(outcome)
    conf = min(0.999999, max(0.0, float(committed_confidence)))
    if normalized == "confirmed":
        return round(-log(max(conf, PREDICTION_ERROR_EPS)), 6)
    if normalized == "violated":
        return round(-log(max(1.0 - conf, PREDICTION_ERROR_EPS)), 6)
    return None


def prediction_brier_from_outcome(committed_confidence: float, outcome: object) -> float | None:
    normalized = normalize_prediction_outcome(outcome)
    conf = min(1.0, max(0.0, float(committed_confidence)))
    if normalized == "confirmed":
        return round((1.0 - conf) ** 2, 6)
    if normalized == "violated":
        return round((0.0 - conf) ** 2, 6)
    return None
