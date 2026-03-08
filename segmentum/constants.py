from __future__ import annotations


ACTION_COSTS = {
    "forage": 0.12,
    "hide": 0.05,
    "scan": 0.06,
    "exploit_shelter": 0.08,
    "rest": 0.03,
    "seek_contact": 0.07,
    "thermoregulate": 0.09,
}

ACTION_BODY_EFFECTS = {
    "forage": {
        "energy_delta": 0.18,
        "stress_delta": 0.10,
        "fatigue_delta": 0.15,
        "temperature_delta": 0.08,
    },
    "hide": {
        "energy_delta": -0.04,
        "stress_delta": -0.15,
        "fatigue_delta": -0.05,
        "temperature_delta": 0.02,
    },
    "scan": {
        "energy_delta": -0.05,
        "stress_delta": 0.03,
        "fatigue_delta": 0.08,
        "temperature_delta": 0.01,
    },
    "exploit_shelter": {
        "energy_delta": -0.06,
        "stress_delta": -0.10,
        "fatigue_delta": -0.08,
        "temperature_delta": -0.04,
    },
    "rest": {
        "energy_delta": -0.02,
        "stress_delta": -0.04,
        "fatigue_delta": -0.12,
        "temperature_delta": -0.02,
    },
    "seek_contact": {
        "energy_delta": -0.04,
        "stress_delta": -0.08,
        "fatigue_delta": 0.05,
        "temperature_delta": 0.03,
    },
    "thermoregulate": {
        "energy_delta": -0.08,
        "stress_delta": -0.06,
        "fatigue_delta": 0.03,
        "temperature_delta": -0.20,
    },
}

ACTION_IMAGINED_EFFECTS = {
    "forage": {
        "food": 0.25,
        "danger": 0.12,
        "novelty": 0.06,
        "shelter": -0.03,
        "temperature": 0.02,
        "social": -0.01,
    },
    "hide": {
        "food": -0.05,
        "danger": -0.22,
        "novelty": -0.04,
        "shelter": 0.08,
        "temperature": 0.01,
        "social": -0.02,
    },
    "scan": {
        "food": 0.03,
        "danger": 0.05,
        "novelty": 0.20,
        "shelter": 0.00,
        "temperature": 0.00,
        "social": 0.01,
    },
    "exploit_shelter": {
        "food": -0.03,
        "danger": -0.14,
        "novelty": -0.04,
        "shelter": 0.20,
        "temperature": -0.05,
        "social": -0.01,
    },
    "rest": {
        "food": -0.02,
        "danger": -0.03,
        "novelty": -0.05,
        "shelter": 0.03,
        "temperature": -0.02,
        "social": 0.00,
    },
    "seek_contact": {
        "food": -0.01,
        "danger": 0.06,
        "novelty": 0.08,
        "shelter": -0.02,
        "temperature": 0.02,
        "social": 0.24,
    },
    "thermoregulate": {
        "food": -0.02,
        "danger": -0.04,
        "novelty": -0.02,
        "shelter": 0.05,
        "temperature": -0.25,
        "social": 0.00,
    },
}
