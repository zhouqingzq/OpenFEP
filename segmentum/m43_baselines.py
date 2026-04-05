from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
import random
from typing import Any

from .m4_benchmarks import (
    BenchmarkTrial,
    IGT_DECK_PROTOCOL,
    IowaTrial,
    evaluate_iowa_predictions,
    evaluate_predictions,
)
from .m4_cognitive_style import logistic


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _normalized_deck_probabilities(counts: Counter[str] | dict[str, int]) -> dict[str, float]:
    total = sum(int(counts.get(deck, 0)) for deck in "ABCD")
    if total <= 0:
        return {deck: 0.25 for deck in "ABCD"}
    return {
        deck: _safe_round(int(counts.get(deck, 0)) / total)
        for deck in "ABCD"
    }


def _sample_deck_from_probabilities(probabilities: dict[str, float], *, rng: random.Random) -> str:
    threshold = rng.random()
    running = 0.0
    chosen_deck = "D"
    for deck in "ABCD":
        running += float(probabilities.get(deck, 0.0))
        if threshold <= running:
            chosen_deck = deck
            break
    return chosen_deck


def _confidence_prediction(
    *,
    trial: BenchmarkTrial,
    predicted_choice: str,
    predicted_confidence: float,
    predicted_probability_right: float,
) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "subject_id": trial.subject_id,
        "session_id": trial.session_id,
        "split": trial.split,
        "human_choice": trial.human_choice,
        "predicted_choice": predicted_choice,
        "predicted_confidence": _clamp01(predicted_confidence),
        "predicted_probability_right": _clamp01(predicted_probability_right),
        "correct": predicted_choice == trial.correct_choice,
        "human_choice_match": predicted_choice == trial.human_choice,
        "human_confidence": _clamp01(trial.human_confidence),
    }


def _confidence_condition_key(trial: BenchmarkTrial) -> tuple[str, float, str]:
    return (str(trial.source_dataset), round(float(trial.stimulus_strength), 3), str(trial.correct_choice))


def _fit_fixed_confidence_from_human_alignment(training_trials: list[BenchmarkTrial]) -> float:
    if not training_trials:
        return 0.5
    aligned = 0
    for trial in training_trials:
        sign_choice = "right" if float(trial.stimulus_strength) >= 0 else "left"
        if sign_choice == trial.human_choice:
            aligned += 1
    return _clamp01(max(0.5, aligned / len(training_trials)))


def _fit_logistic_baseline(training_trials: list[BenchmarkTrial], *, seed: int) -> dict[str, float]:
    weight0 = 0.0
    weight1 = 1.0
    conf0 = 0.5
    conf1 = 0.1
    rng = random.Random(seed)
    ordered = list(training_trials)
    for _ in range(160):
        rng.shuffle(ordered)
        for trial in ordered:
            strength = float(trial.stimulus_strength)
            label = 1.0 if trial.human_choice == "right" else 0.0
            probability = logistic(weight0 + weight1 * strength)
            error = label - probability
            weight0 += 0.08 * error
            weight1 += 0.12 * error * strength

            conf_prediction = _clamp01(conf0 + conf1 * abs(strength))
            conf_error = float(trial.human_confidence) - conf_prediction
            conf0 += 0.03 * conf_error
            conf1 += 0.05 * conf_error * abs(strength)
    return {
        "choice_intercept": _safe_round(weight0),
        "choice_slope": _safe_round(weight1),
        "confidence_intercept": _safe_round(conf0),
        "confidence_slope": _safe_round(conf1),
    }


def fit_confidence_logistic_baseline(training_trials: list[BenchmarkTrial], *, seed: int = 43) -> dict[str, float]:
    return _fit_logistic_baseline(training_trials, seed=seed)


def run_confidence_random_baseline(
    evaluation_trials: list[BenchmarkTrial],
    *,
    seed: int = 43,
) -> dict[str, Any]:
    rng = random.Random(seed)
    predictions = []
    for trial in evaluation_trials:
        probability_right = 0.5
        predicted_choice = "right" if rng.random() >= 0.5 else "left"
        predictions.append(
            _confidence_prediction(
                trial=trial,
                predicted_choice=predicted_choice,
                predicted_confidence=0.5,
                predicted_probability_right=probability_right,
            )
        )
    return {
        "model_label": "random_uniform",
        "trial_count": len(evaluation_trials),
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def run_confidence_stimulus_only_baseline(
    training_trials: list[BenchmarkTrial],
    evaluation_trials: list[BenchmarkTrial],
) -> dict[str, Any]:
    fixed_confidence = _fit_fixed_confidence_from_human_alignment(training_trials)
    predictions = []
    for trial in evaluation_trials:
        predicted_choice = "right" if float(trial.stimulus_strength) >= 0 else "left"
        probability_right = fixed_confidence if predicted_choice == "right" else 1.0 - fixed_confidence
        predictions.append(
            _confidence_prediction(
                trial=trial,
                predicted_choice=predicted_choice,
                predicted_confidence=fixed_confidence,
                predicted_probability_right=probability_right,
            )
        )
    return {
        "model_label": "stimulus_only_fixed_confidence",
        "trial_count": len(evaluation_trials),
        "fixed_confidence": _safe_round(fixed_confidence),
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def run_confidence_statistical_baseline(
    training_trials: list[BenchmarkTrial],
    evaluation_trials: list[BenchmarkTrial],
    *,
    seed: int = 43,
) -> dict[str, Any]:
    coefficients = _fit_logistic_baseline(training_trials, seed=seed)
    predictions = []
    for trial in evaluation_trials:
        probability_right = logistic(coefficients["choice_intercept"] + coefficients["choice_slope"] * float(trial.stimulus_strength))
        confidence = coefficients["confidence_intercept"] + coefficients["confidence_slope"] * abs(float(trial.stimulus_strength))
        predictions.append(
            _confidence_prediction(
                trial=trial,
                predicted_choice="right" if probability_right >= 0.5 else "left",
                predicted_confidence=confidence,
                predicted_probability_right=probability_right,
            )
        )
    return {
        "model_label": "statistical_logistic",
        "trial_count": len(evaluation_trials),
        "coefficients": coefficients,
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def run_confidence_human_match_ceiling(
    training_trials: list[BenchmarkTrial],
    evaluation_trials: list[BenchmarkTrial],
) -> dict[str, Any]:
    condition_stats: dict[tuple[str, float, str], dict[str, Any]] = {}
    global_right_rate = mean(1.0 if trial.human_choice == "right" else 0.0 for trial in training_trials) if training_trials else 0.5
    global_confidence = mean(float(trial.human_confidence) for trial in training_trials) if training_trials else 0.5
    for trial in training_trials:
        key = _confidence_condition_key(trial)
        stats = condition_stats.setdefault(key, {"left": 0, "right": 0, "confidences": []})
        stats[str(trial.human_choice)] += 1
        stats["confidences"].append(float(trial.human_confidence))
    predictions = []
    for trial in evaluation_trials:
        stats = condition_stats.get(_confidence_condition_key(trial))
        if stats is None:
            probability_right = global_right_rate
            confidence = global_confidence
        else:
            total = max(1, int(stats["left"]) + int(stats["right"]))
            probability_right = float(stats["right"]) / total
            confidence = mean(stats["confidences"]) if stats["confidences"] else global_confidence
        predicted_choice = "right" if probability_right >= 0.5 else "left"
        predictions.append(
            _confidence_prediction(
                trial=trial,
                predicted_choice=predicted_choice,
                predicted_confidence=confidence,
                predicted_probability_right=probability_right,
            )
        )
    return {
        "model_label": "human_match_ceiling",
        "trial_count": len(evaluation_trials),
        "condition_count": len(condition_stats),
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def _simulate_igt_policy(
    evaluation_trials: list[IowaTrial],
    *,
    label: str,
    chooser: Any,
) -> dict[str, Any]:
    ordered = sorted(evaluation_trials, key=lambda item: (str(item.subject_id), int(item.trial_index), str(item.trial_id)))
    predictions: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for trial in ordered:
        state = states.setdefault(
            str(trial.subject_id),
            {
                "draw_counts": {deck: 0 for deck in "ABCD"},
                "reward_totals": {deck: 0.0 for deck in "ABCD"},
                "reward_means": {deck: 0.0 for deck in "ABCD"},
                "counts": {deck: 0 for deck in "ABCD"},
                "cumulative_gain": 0,
            },
        )
        chosen_deck, predicted_confidence, expected_value = chooser(trial=trial, state=state)
        spec = IGT_DECK_PROTOCOL[chosen_deck]
        draw_index = int(state["draw_counts"][chosen_deck])
        reward = int(spec["reward"])
        penalty = int(spec["penalties"][draw_index % len(spec["penalties"])])
        net_outcome = reward + penalty
        state["draw_counts"][chosen_deck] += 1
        state["counts"][chosen_deck] += 1
        state["reward_totals"][chosen_deck] += net_outcome
        state["reward_means"][chosen_deck] = state["reward_totals"][chosen_deck] / max(1, state["counts"][chosen_deck])
        state["cumulative_gain"] += net_outcome
        predictions.append(
            {
                "trial_id": trial.trial_id,
                "subject_id": trial.subject_id,
                "split": trial.split,
                "chosen_deck": chosen_deck,
                "actual_deck": trial.deck,
                "expected_value": _safe_round(expected_value),
                "predicted_confidence": _clamp01(predicted_confidence),
                "advantageous_choice": chosen_deck in {"C", "D"},
                "actual_advantageous": bool(trial.advantageous),
                "deck_match": chosen_deck == trial.deck,
                "reward": reward,
                "penalty": penalty,
                "net_outcome": net_outcome,
                "cumulative_gain": state["cumulative_gain"],
            }
        )
    return {
        "model_label": label,
        "trial_count": len(evaluation_trials),
        "predictions": predictions,
        "metrics": evaluate_iowa_predictions(predictions),
    }


def run_igt_random_baseline(
    evaluation_trials: list[IowaTrial],
    *,
    seed: int = 44,
) -> dict[str, Any]:
    rng = random.Random(seed)

    def chooser(*, trial: IowaTrial, state: dict[str, Any]) -> tuple[str, float, float]:
        deck = rng.choice(list("ABCD"))
        return deck, 0.25, 0.0

    return _simulate_igt_policy(evaluation_trials, label="random_uniform", chooser=chooser)


def run_igt_frequency_matching_baseline(
    training_trials: list[IowaTrial],
    evaluation_trials: list[IowaTrial],
    *,
    seed: int = 44,
) -> dict[str, Any]:
    fitted = fit_igt_human_behavior_baseline(training_trials)
    deck_probabilities_by_trial_index = {
        int(index): {deck: float(probability) for deck, probability in probabilities.items()}
        for index, probabilities in fitted["deck_probabilities_by_trial_index"].items()
    }
    global_deck_probabilities = {
        deck: float(probability)
        for deck, probability in fitted["global_deck_probabilities"].items()
    }
    mean_net_by_index_and_deck = dict(fitted["mean_net_by_trial_index_and_deck"])
    global_mean_net_by_deck = dict(fitted["global_mean_net_by_deck"])
    rng = random.Random(seed)

    def chooser(*, trial: IowaTrial, state: dict[str, Any]) -> tuple[str, float, float]:
        index = int(trial.trial_index)
        probabilities = deck_probabilities_by_trial_index.get(index, global_deck_probabilities)
        if not probabilities:
            probabilities = {deck: 0.25 for deck in "ABCD"}
        chosen_deck = _sample_deck_from_probabilities(probabilities, rng=rng)
        confidence = float(probabilities.get(chosen_deck, 0.25))
        expected_value = mean_net_by_index_and_deck.get(
            f"{index}:{chosen_deck}",
            global_mean_net_by_deck.get(chosen_deck, 0.0),
        )
        return chosen_deck, confidence, float(expected_value)

    payload = _simulate_igt_policy(evaluation_trials, label="frequency_matching", chooser=chooser)
    payload["fitted_model"] = fitted
    return payload


def fit_igt_human_behavior_baseline(training_trials: list[IowaTrial]) -> dict[str, Any]:
    by_index: dict[int, Counter[str]] = defaultdict(Counter)
    net_by_index_and_deck: dict[tuple[int, str], list[float]] = defaultdict(list)
    global_counts: Counter[str] = Counter()
    global_nets: dict[str, list[float]] = defaultdict(list)
    for trial in training_trials:
        index = int(trial.trial_index)
        deck = str(trial.deck)
        by_index[index][deck] += 1
        net_by_index_and_deck[(index, deck)].append(float(trial.net_outcome))
        global_counts[deck] += 1
        global_nets[deck].append(float(trial.net_outcome))
    return {
        "deck_counts_by_trial_index": {index: dict(counter) for index, counter in by_index.items()},
        "deck_probabilities_by_trial_index": {
            index: _normalized_deck_probabilities(counter)
            for index, counter in by_index.items()
        },
        "global_deck_counts": dict(global_counts),
        "global_deck_probabilities": _normalized_deck_probabilities(global_counts),
        "mean_net_by_trial_index_and_deck": {
            f"{index}:{deck}": _safe_round(mean(values))
            for (index, deck), values in net_by_index_and_deck.items()
            if values
        },
        "global_mean_net_by_deck": {
            deck: _safe_round(mean(values))
            for deck, values in global_nets.items()
            if values
        },
    }


def run_igt_human_behavior_baseline(
    training_trials: list[IowaTrial],
    evaluation_trials: list[IowaTrial],
) -> dict[str, Any]:
    fitted = fit_igt_human_behavior_baseline(training_trials)
    by_index = {
        int(index): Counter({deck: int(count) for deck, count in counts.items()})
        for index, counts in fitted["deck_counts_by_trial_index"].items()
    }
    global_counts = Counter({deck: int(count) for deck, count in fitted["global_deck_counts"].items()})
    mean_net_by_index_and_deck = dict(fitted["mean_net_by_trial_index_and_deck"])
    global_mean_net_by_deck = dict(fitted["global_mean_net_by_deck"])

    def chooser(*, trial: IowaTrial, state: dict[str, Any]) -> tuple[str, float, float]:
        index = int(trial.trial_index)
        counts = by_index.get(index, global_counts)
        total = max(1, sum(counts.values()))
        if not counts:
            return "C", 0.25, 0.0
        chosen_deck = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        confidence = counts[chosen_deck] / total
        expected_value = mean_net_by_index_and_deck.get(
            f"{index}:{chosen_deck}",
            global_mean_net_by_deck.get(chosen_deck, 0.0),
        )
        return chosen_deck, confidence, float(expected_value)

    payload = _simulate_igt_policy(evaluation_trials, label="human_behavior", chooser=chooser)
    payload["fitted_model"] = fitted
    return payload


__all__ = [
    "fit_confidence_logistic_baseline",
    "fit_igt_human_behavior_baseline",
    "run_confidence_human_match_ceiling",
    "run_confidence_random_baseline",
    "run_confidence_statistical_baseline",
    "run_confidence_stimulus_only_baseline",
    "run_igt_frequency_matching_baseline",
    "run_igt_human_behavior_baseline",
    "run_igt_random_baseline",
]
