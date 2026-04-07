from __future__ import annotations


def baseline_observation() -> dict[str, float]:
    return {
        "food": 0.40,
        "danger": 0.95,
        "novelty": 0.30,
        "shelter": 0.20,
        "temperature": 0.45,
        "social": 0.25,
    }


def baseline_prediction() -> dict[str, float]:
    return {
        "food": 0.70,
        "danger": 0.10,
        "novelty": 0.45,
        "shelter": 0.50,
        "temperature": 0.50,
        "social": 0.35,
    }


def baseline_errors() -> dict[str, float]:
    return {
        "food": -0.30,
        "danger": 0.85,
        "novelty": -0.15,
        "shelter": -0.30,
        "temperature": -0.05,
        "social": -0.10,
    }


def baseline_body_state() -> dict[str, float]:
    return {
        "energy": 0.10,
        "stress": 0.75,
        "fatigue": 0.25,
        "temperature": 0.45,
    }
