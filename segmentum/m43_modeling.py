from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmark_registry import benchmark_status, load_benchmark_bundle, validate_benchmark_bundle
from .m43_baselines import (
    run_confidence_human_match_ceiling,
    run_confidence_random_baseline,
    run_confidence_statistical_baseline,
    run_confidence_stimulus_only_baseline,
    run_igt_frequency_matching_baseline,
    run_igt_human_behavior_baseline,
    run_igt_random_baseline,
)
from .m4_benchmarks import (
    BenchmarkTrial,
    ConfidenceDatabaseAdapter,
    IowaGamblingTaskAdapter,
    IowaTrial,
    _score_action_candidates,
    _safe_round,
    default_acceptance_benchmark_root,
    detect_subject_leakage,
    evaluate_iowa_predictions,
    evaluate_predictions,
    summarize_confidence_predictions,
    summarize_iowa_predictions,
)
from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE


CONFIDENCE_FIT_TRAIN_MAX_TRIALS = 6000
CONFIDENCE_FIT_VALIDATION_MAX_TRIALS = 6000
CONFIDENCE_EVAL_HELDOUT_MAX_TRIALS = 12000
CONFIDENCE_SENSITIVITY_MAX_TRIALS = 2500
CONFIDENCE_MAX_SUBJECTS = 96
PARAMETER_SWEEP_SIGMA = 0.15
SENSITIVITY_STEP_SCHEDULE = (0.10, PARAMETER_SWEEP_SIGMA, 0.30)
SENSITIVITY_NOISE_SEED_OFFSETS = (11, 23, 37)
SENSITIVITY_SIGNAL_TO_NOISE_MIN = 1.25
SENSITIVITY_CONFIDENCE_PROB_SHIFT_MIN = 0.005
SENSITIVITY_CONFIDENCE_RATING_SHIFT_MIN = 0.01
SENSITIVITY_IGT_DECK_FLIP_MIN = 0.01
SENSITIVITY_IGT_ADVANTAGE_FLIP_MIN = 0.01

SENSITIVITY_METRIC_FLOORS = {
    "confidence_heldout_likelihood": 0.005,
    "confidence_brier_score": 0.003,
    "confidence_alignment": 0.010,
    "confidence_ambiguous_confidence": 0.015,
    "confidence_high_evidence_confidence": 0.020,
    "confidence_high_evidence_accuracy": 0.010,
    "confidence_confidence_separation": 0.020,
    "igt_deck_match_rate": 0.010,
    "igt_advantageous_ratio": 0.020,
    "igt_learning_curve_slope": 0.020,
    "igt_early_switch_rate": 0.020,
    "igt_post_loss_switch_rate": 0.030,
    "igt_post_loss_stay_rate": 0.030,
    "igt_loss_streak_escape_rate": 0.010,
    "igt_post_loss_peak_loss_estimate": 10.0,
}

PARAMETER_SENSITIVITY_DIAGNOSTICS = {
    "uncertainty_sensitivity": ("confidence_ambiguous_confidence", "confidence_confidence_separation"),
    "error_aversion": ("igt_post_loss_switch_rate", "igt_advantageous_ratio"),
    "exploration_bias": ("igt_early_switch_rate", "igt_learning_curve_slope", "igt_deck_match_rate"),
    "attention_selectivity": (
        "confidence_high_evidence_accuracy",
        "confidence_confidence_separation",
        "igt_advantageous_ratio",
        "igt_deck_match_rate",
    ),
    "confidence_gain": (
        "confidence_high_evidence_confidence",
        "confidence_brier_score",
        "confidence_confidence_separation",
    ),
    "update_rigidity": ("igt_post_loss_stay_rate", "igt_learning_curve_slope"),
    "resource_pressure_sensitivity": ("confidence_heldout_likelihood", "igt_deck_match_rate"),
    "virtual_prediction_error_gain": (
        "igt_loss_streak_escape_rate",
        "igt_post_loss_peak_loss_estimate",
        "igt_learning_curve_slope",
    ),
}

_CONFIDENCE_CACHE: dict[str, dict[str, Any]] = {}
_IGT_CACHE: dict[str, dict[str, Any]] = {}


def _resolve_benchmark_root(benchmark_root: Path | str | None = None) -> Path | None:
    if benchmark_root is None:
        return default_acceptance_benchmark_root()
    candidate = Path(benchmark_root).resolve()
    required = (
        candidate / "confidence_database" / "manifest.json",
        candidate / "iowa_gambling_task" / "manifest.json",
    )
    return candidate if all(path.exists() for path in required) else None


def _cache_key(benchmark_id: str, benchmark_root: Path | None, allow_smoke_test: bool) -> str:
    if benchmark_root is None:
        return f"{benchmark_id}::smoke::{int(allow_smoke_test)}"
    return f"{benchmark_id}::{benchmark_root.resolve()}::{int(allow_smoke_test)}"


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            rows.append(json.loads(raw_line))
    return rows


def _split_trials(rows: list[Any]) -> dict[str, list[Any]]:
    splits: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        splits[str(row.split)].append(row)
    return {name: list(payload) for name, payload in splits.items()}


def _load_confidence_data(
    *,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
) -> dict[str, Any]:
    resolved_root = None if allow_smoke_test and benchmark_root is None else _resolve_benchmark_root(benchmark_root)
    key = _cache_key("confidence_database", resolved_root, allow_smoke_test)
    cached = _CONFIDENCE_CACHE.get(key)
    if cached is not None:
        return cached

    if resolved_root is None:
        if not allow_smoke_test:
            raise ValueError("Acceptance-grade Confidence Database fitting requires a real external benchmark root.")
        bundle = load_benchmark_bundle("confidence_database", root=None)
        validation = validate_benchmark_bundle("confidence_database", root=None)
        status = benchmark_status("confidence_database", root=None)
        trials = ConfidenceDatabaseAdapter().load_trials(allow_smoke_test=True)
    else:
        bundle = load_benchmark_bundle("confidence_database", root=resolved_root)
        validation = validate_benchmark_bundle("confidence_database", root=resolved_root)
        status = benchmark_status("confidence_database", root=resolved_root)
        rows = _iter_jsonl(bundle.data_path)
        trials = [
            BenchmarkTrial(
                trial_id=str(row["trial_id"]),
                subject_id=str(row["subject_id"]),
                session_id=str(row.get("session_id") or row["subject_id"]),
                stimulus_strength=float(row["stimulus_strength"]),
                correct_choice=str(row["correct_choice"]),
                human_choice=str(row["human_choice"]),
                human_confidence=float(row["human_confidence"]),
                rt_ms=int(row["rt_ms"]),
                split=str(row["split"]),
                source_dataset=str(row.get("source_dataset", "")),
                source_file=str(row.get("source_file", "")),
            )
            for row in rows
        ]

    payload = {
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "splits": _split_trials(trials),
        "all_trials": list(trials),
        "source_type": bundle.source_type,
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "synthetic_diagnostic",
        "external_validation": False,
    }
    _CONFIDENCE_CACHE[key] = payload
    return payload


def _load_igt_data(
    *,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
) -> dict[str, Any]:
    resolved_root = None if allow_smoke_test and benchmark_root is None else _resolve_benchmark_root(benchmark_root)
    key = _cache_key("iowa_gambling_task", resolved_root, allow_smoke_test)
    cached = _IGT_CACHE.get(key)
    if cached is not None:
        return cached

    if resolved_root is None:
        if not allow_smoke_test:
            raise ValueError("Acceptance-grade IGT fitting requires a real external benchmark root.")
        bundle = load_benchmark_bundle("iowa_gambling_task", root=None)
        validation = validate_benchmark_bundle("iowa_gambling_task", root=None)
        status = benchmark_status("iowa_gambling_task", root=None)
        trials = IowaGamblingTaskAdapter().load_trials(allow_smoke_test=True, protocol_mode="smoke_flexible")
    else:
        bundle = load_benchmark_bundle("iowa_gambling_task", root=resolved_root)
        validation = validate_benchmark_bundle("iowa_gambling_task", root=resolved_root)
        status = benchmark_status("iowa_gambling_task", root=resolved_root)
        rows = _iter_jsonl(bundle.data_path)
        by_subject: dict[str, list[IowaTrial]] = defaultdict(list)
        for row in rows:
            trial = IowaTrial(
                trial_id=str(row["trial_id"]),
                subject_id=str(row["subject_id"]),
                deck=str(row["deck"]),
                reward=int(row["reward"]),
                penalty=int(row["penalty"]),
                net_outcome=int(row["net_outcome"]),
                advantageous=bool(row["advantageous"]),
                trial_index=int(row["trial_index"]),
                split=str(row["split"]),
                source_file=str(row.get("source_file", "")),
            )
            by_subject[trial.subject_id].append(trial)
        trials = []
        for subject_id in sorted(by_subject):
            ordered = sorted(by_subject[subject_id], key=lambda item: item.trial_index)
            trials.extend(ordered[:100])

    payload = {
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "splits": _split_trials(trials),
        "all_trials": list(trials),
        "source_type": bundle.source_type,
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "synthetic_diagnostic",
        "external_validation": False,
    }
    _IGT_CACHE[key] = payload
    return payload


def _balanced_subject_sample(
    trials: list[Any],
    *,
    max_trials: int | None = None,
    max_subjects: int | None = None,
    seed: int = 43,
) -> list[Any]:
    if max_trials is None and max_subjects is None:
        return list(trials)
    by_subject: dict[str, list[Any]] = defaultdict(list)
    for trial in trials:
        by_subject[str(trial.subject_id)].append(trial)
    subject_ids = sorted(by_subject)
    if max_subjects is not None:
        rng = __import__("random").Random(seed)
        shuffled = list(subject_ids)
        rng.shuffle(shuffled)
        subject_ids = sorted(shuffled[: max(1, int(max_subjects))])
    ordered_rows = {
        subject_id: sorted(
            by_subject[subject_id],
            key=lambda item: (int(getattr(item, "trial_index", 0)), str(getattr(item, "trial_id", ""))),
        )
        for subject_id in subject_ids
    }
    if max_trials is None:
        selected = []
        for subject_id in subject_ids:
            selected.extend(ordered_rows[subject_id])
        return selected
    selected: list[Any] = []
    cursor = {subject_id: 0 for subject_id in subject_ids}
    while len(selected) < int(max_trials):
        progressed = False
        for subject_id in subject_ids:
            rows = ordered_rows[subject_id]
            index = cursor[subject_id]
            if index >= len(rows):
                continue
            selected.append(rows[index])
            cursor[subject_id] += 1
            progressed = True
            if len(selected) >= int(max_trials):
                break
        if not progressed:
            break
    return selected


def _set_parameter(parameters: CognitiveStyleParameters, field_name: str, value: float) -> CognitiveStyleParameters:
    return replace(parameters, **{field_name: max(0.0, min(1.0, float(value)))})


def _igt_learning_curve_slope(rows: list[dict[str, Any]], *, field_name: str) -> float:
    if not rows:
        return 0.0
    by_block: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        trial_index = int(row.get("trial_index", 0))
        if trial_index <= 0:
            continue
        by_block[(trial_index - 1) // 20 + 1].append(1.0 if bool(row[field_name]) else 0.0)
    if len(by_block) < 2:
        return 0.0
    xs = [float(block) for block in sorted(by_block)]
    ys = [mean(by_block[int(block)]) for block in xs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return _safe_round(numerator / denominator if denominator else 0.0)


def _igt_deck_frequency(rows: list[dict[str, Any]], *, field_name: str) -> dict[str, float]:
    total = len(rows)
    if total <= 0:
        return {deck: 0.25 for deck in "ABCD"}
    return {
        deck: _safe_round(mean(1.0 if str(row[field_name]) == deck else 0.0 for row in rows))
        for deck in "ABCD"
    }


def _igt_distribution_l1(rows: list[dict[str, Any]], *, left_field: str, right_field: str) -> float:
    left = _igt_deck_frequency(rows, field_name=left_field)
    right = _igt_deck_frequency(rows, field_name=right_field)
    return _safe_round(sum(abs(float(left[deck]) - float(right[deck])) for deck in "ABCD"))


def _igt_subject_match_floor(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    subject_match_rates = [
        mean(1.0 if bool(row["deck_match"]) else 0.0 for row in subject_rows)
        for subject_rows in by_subject.values()
    ]
    return _safe_round(min(subject_match_rates) if subject_match_rates else 0.0)


def _igt_post_loss_switch_rate(
    rows: list[dict[str, Any]],
    *,
    deck_field: str,
    outcome_field: str,
) -> float:
    if not rows:
        return 0.0
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    switches: list[float] = []
    for subject_rows in by_subject.values():
        ordered = sorted(subject_rows, key=lambda item: int(item["trial_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            if float(previous[outcome_field]) >= 0.0:
                continue
            switches.append(1.0 if str(previous[deck_field]) != str(current[deck_field]) else 0.0)
    return _safe_round(mean(switches) if switches else 0.0)


def _igt_phase_alignment_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "deck_distribution_l1": 0.0,
            "phase_deck_distribution_l1": 0.0,
            "late_advantageous_gap": 0.0,
            "phase_advantageous_gap": 0.0,
            "post_loss_switch_gap": 0.0,
            "subject_match_floor": 0.0,
        }
    phase_windows = ((1, 20), (41, 60), (81, 100))
    phase_distribution_gaps: list[float] = []
    phase_advantageous_gaps: list[float] = []
    late_rows: list[dict[str, Any]] = []
    for start, end in phase_windows:
        phase_rows = [row for row in rows if start <= int(row["trial_index"]) <= end]
        if not phase_rows:
            continue
        phase_distribution_gaps.append(
            _igt_distribution_l1(phase_rows, left_field="chosen_deck", right_field="human_deck")
        )
        agent_adv = mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in phase_rows)
        human_adv = mean(1.0 if bool(row["actual_advantageous"]) else 0.0 for row in phase_rows)
        phase_advantageous_gaps.append(abs(agent_adv - human_adv))
        if end == 100:
            late_rows = phase_rows
    late_advantageous_gap = 0.0
    if late_rows:
        late_agent_adv = mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in late_rows)
        late_human_adv = mean(1.0 if bool(row["actual_advantageous"]) else 0.0 for row in late_rows)
        late_advantageous_gap = abs(late_agent_adv - late_human_adv)
    agent_post_loss = _igt_post_loss_switch_rate(rows, deck_field="chosen_deck", outcome_field="net_outcome")
    human_post_loss = _igt_post_loss_switch_rate(rows, deck_field="human_deck", outcome_field="human_net_outcome")
    return {
        "deck_distribution_l1": _igt_distribution_l1(rows, left_field="chosen_deck", right_field="human_deck"),
        "phase_deck_distribution_l1": _safe_round(mean(phase_distribution_gaps) if phase_distribution_gaps else 0.0),
        "late_advantageous_gap": _safe_round(late_advantageous_gap),
        "phase_advantageous_gap": _safe_round(mean(phase_advantageous_gaps) if phase_advantageous_gaps else 0.0),
        "post_loss_switch_gap": _safe_round(abs(agent_post_loss - human_post_loss)),
        "subject_match_floor": _igt_subject_match_floor(rows),
    }


def _simulate_confidence_trials(
    trials: list[BenchmarkTrial],
    parameters: CognitiveStyleParameters,
    *,
    seed: int,
    include_predictions: bool = True,
) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    rng_mod = __import__("random")
    for global_trial_index, trial in enumerate(trials, start=1):
        subject_id = str(trial.subject_id)
        state = states.setdefault(subject_id, adapter.initial_state(subject_id=subject_id, parameters=parameters))
        state.setdefault("_run_seed", seed)
        observation = adapter.observation_from_trial(trial, state=state, parameters=parameters, trial_index=global_trial_index)
        decision = _score_action_candidates(
            observation=observation,
            candidates=adapter.action_space(trial, observation=observation, state=state, parameters=parameters),
            parameters=parameters,
            rng=rng_mod.Random(seed + global_trial_index),
            state=state,
        )
        state["_last_decision"] = dict(decision)
        transition = adapter.apply_action(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            state=state,
            parameters=parameters,
            observation=observation,
            global_trial_index=global_trial_index,
        )
        record = adapter.export_trial_record(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            observation=observation,
            transition=transition,
            state=state,
            global_trial_index=global_trial_index,
        )
        records.append(record)
        predictions.append(adapter.legacy_prediction(record))
    return {
        "trial_count": len(trials),
        "metrics": evaluate_predictions(predictions),
        "subject_summary": summarize_confidence_predictions(predictions),
        "trial_trace": records,
        "predictions": predictions if include_predictions else [],
    }


def _simulate_igt_trials(
    trials: list[IowaTrial],
    parameters: CognitiveStyleParameters,
    *,
    seed: int,
    include_predictions: bool = True,
) -> dict[str, Any]:
    adapter = IowaGamblingTaskAdapter()
    ordered = sorted(trials, key=lambda item: (str(item.subject_id), int(item.trial_index), str(item.trial_id)))
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    rng_mod = __import__("random")
    for global_trial_index, trial in enumerate(ordered, start=1):
        subject_id = str(trial.subject_id)
        state = states.setdefault(subject_id, adapter.initial_state(subject_id=subject_id, parameters=parameters))
        state.setdefault("_run_seed", seed)
        observation = adapter.observation_from_trial(trial, state=state, parameters=parameters, trial_index=global_trial_index)
        decision = _score_action_candidates(
            observation=observation,
            candidates=adapter.action_space(trial, observation=observation, state=state, parameters=parameters),
            parameters=parameters,
            rng=rng_mod.Random(seed + global_trial_index),
            state=state,
        )
        transition = adapter.apply_action(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            state=state,
            parameters=parameters,
            observation=observation,
            global_trial_index=global_trial_index,
        )
        record = adapter.export_trial_record(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            observation=observation,
            transition=transition,
            state=state,
            global_trial_index=global_trial_index,
        )
        records.append(record)
        predictions.append(adapter.legacy_prediction(record))
    metrics = dict(evaluate_iowa_predictions(predictions))
    metrics["learning_curve_slope"] = _igt_learning_curve_slope(records, field_name="advantageous_choice")
    metrics["human_learning_curve_slope"] = _igt_learning_curve_slope(records, field_name="actual_advantageous")
    metrics["human_advantageous_rate"] = _safe_round(mean(1.0 if bool(row["actual_advantageous"]) else 0.0 for row in records) if records else 0.0)
    metrics.update(_igt_phase_alignment_metrics(records))
    return {
        "trial_count": len(trials),
        "metrics": metrics,
        "subject_summary": summarize_iowa_predictions(predictions),
        "trial_trace": records,
        "predictions": predictions if include_predictions else [],
    }


def _confidence_trace_diagnostics(trace: list[dict[str, Any]]) -> dict[str, float]:
    ambiguous_rows = [row for row in trace if abs(float(row["stimulus_strength"])) < 0.15]
    high_evidence_rows = [row for row in trace if abs(float(row["stimulus_strength"])) >= 0.50]
    ambiguous_confidence = mean(float(row["agent_confidence_rating"]) for row in ambiguous_rows) if ambiguous_rows else 0.0
    high_evidence_confidence = mean(float(row["agent_confidence_rating"]) for row in high_evidence_rows) if high_evidence_rows else 0.0
    high_evidence_accuracy = mean(1.0 if bool(row["correct"]) else 0.0 for row in high_evidence_rows) if high_evidence_rows else 0.0
    return {
        "confidence_ambiguous_confidence": _safe_round(ambiguous_confidence),
        "confidence_high_evidence_confidence": _safe_round(high_evidence_confidence),
        "confidence_high_evidence_accuracy": _safe_round(high_evidence_accuracy),
        "confidence_confidence_separation": _safe_round(high_evidence_confidence - ambiguous_confidence),
    }


def _igt_trace_diagnostics(trace: list[dict[str, Any]]) -> dict[str, float]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    post_loss_peak_loss_estimates: list[float] = []
    for row in trace:
        by_subject[str(row["subject_id"])].append(row)
        if int(row["net_outcome"]) < 0:
            snapshot = dict(row.get("internal_state_snapshot", {}))
            loss_estimates = [float(value) for value in dict(snapshot.get("loss_estimates", {})).values()]
            if loss_estimates:
                post_loss_peak_loss_estimates.append(max(loss_estimates))

    early_switches: list[float] = []
    post_loss_switches: list[float] = []
    post_loss_stays: list[float] = []
    loss_streak_escapes: list[float] = []
    for subject_rows in by_subject.values():
        ordered = sorted(subject_rows, key=lambda item: int(item["trial_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            switched = 1.0 if str(previous["chosen_deck"]) != str(current["chosen_deck"]) else 0.0
            if int(current["trial_index"]) <= 20:
                early_switches.append(switched)
            if int(previous["net_outcome"]) < 0:
                post_loss_switches.append(switched)
                post_loss_stays.append(1.0 - switched)
            prior_snapshot = dict(previous.get("internal_state_snapshot", {}))
            if int(prior_snapshot.get("loss_streak", 0)) >= 1:
                loss_streak_escapes.append(switched)
    return {
        "igt_early_switch_rate": _safe_round(mean(early_switches) if early_switches else 0.0),
        "igt_post_loss_switch_rate": _safe_round(mean(post_loss_switches) if post_loss_switches else 0.0),
        "igt_post_loss_stay_rate": _safe_round(mean(post_loss_stays) if post_loss_stays else 0.0),
        "igt_loss_streak_escape_rate": _safe_round(mean(loss_streak_escapes) if loss_streak_escapes else 0.0),
        "igt_post_loss_peak_loss_estimate": _safe_round(mean(post_loss_peak_loss_estimates) if post_loss_peak_loss_estimates else 0.0),
    }


def _summarize_confidence_shift(reference_payload: dict[str, Any], comparison_payload: dict[str, Any]) -> dict[str, float]:
    reference_metrics = dict(reference_payload["metrics"])
    comparison_metrics = dict(comparison_payload["metrics"])
    reference_trace = list(reference_payload["trial_trace"])
    comparison_trace = list(comparison_payload["trial_trace"])
    metric_deltas = {
        "confidence_heldout_likelihood": _safe_round(abs(float(comparison_metrics["heldout_likelihood"]) - float(reference_metrics["heldout_likelihood"]))),
        "confidence_brier_score": _safe_round(abs(float(comparison_metrics["brier_score"]) - float(reference_metrics["brier_score"]))),
        "confidence_alignment": _safe_round(abs(float(comparison_metrics["confidence_alignment"]) - float(reference_metrics["confidence_alignment"]))),
    }
    reference_diagnostics = _confidence_trace_diagnostics(reference_trace)
    comparison_diagnostics = _confidence_trace_diagnostics(comparison_trace)
    for metric_name, reference_value in reference_diagnostics.items():
        metric_deltas[metric_name] = _safe_round(abs(float(comparison_diagnostics[metric_name]) - float(reference_value)))
    metric_deltas["confidence_mean_abs_probability_shift"] = _safe_round(
        mean(
            abs(float(left["predicted_probability_right"]) - float(right["predicted_probability_right"]))
            for left, right in zip(reference_trace, comparison_trace)
        ) if reference_trace and comparison_trace else 0.0
    )
    metric_deltas["confidence_mean_abs_rating_shift"] = _safe_round(
        mean(
            abs(float(left["agent_confidence_rating"]) - float(right["agent_confidence_rating"]))
            for left, right in zip(reference_trace, comparison_trace)
        ) if reference_trace and comparison_trace else 0.0
    )
    metric_deltas["confidence_choice_flip_rate"] = _safe_round(
        mean(
            1.0 if str(left["agent_choice"]) != str(right["agent_choice"]) else 0.0
            for left, right in zip(reference_trace, comparison_trace)
        ) if reference_trace and comparison_trace else 0.0
    )
    return metric_deltas


def _summarize_igt_shift(reference_payload: dict[str, Any], comparison_payload: dict[str, Any]) -> dict[str, float]:
    reference_metrics = dict(reference_payload["metrics"])
    comparison_metrics = dict(comparison_payload["metrics"])
    reference_trace = list(reference_payload["trial_trace"])
    comparison_trace = list(comparison_payload["trial_trace"])
    metric_deltas = {
        "igt_deck_match_rate": _safe_round(abs(float(comparison_metrics["deck_match_rate"]) - float(reference_metrics["deck_match_rate"]))),
        "igt_advantageous_ratio": _safe_round(abs(float(comparison_metrics["advantageous_choice_rate"]) - float(reference_metrics["advantageous_choice_rate"]))),
        "igt_learning_curve_slope": _safe_round(abs(float(comparison_metrics["learning_curve_slope"]) - float(reference_metrics["learning_curve_slope"]))),
    }
    reference_diagnostics = _igt_trace_diagnostics(reference_trace)
    comparison_diagnostics = _igt_trace_diagnostics(comparison_trace)
    for metric_name, reference_value in reference_diagnostics.items():
        metric_deltas[metric_name] = _safe_round(abs(float(comparison_diagnostics[metric_name]) - float(reference_value)))
    metric_deltas["igt_deck_flip_rate"] = _safe_round(
        mean(
            1.0 if str(left["chosen_deck"]) != str(right["chosen_deck"]) else 0.0
            for left, right in zip(reference_trace, comparison_trace)
        ) if reference_trace and comparison_trace else 0.0
    )
    metric_deltas["igt_advantageous_flip_rate"] = _safe_round(
        mean(
            1.0 if bool(left["advantageous_choice"]) != bool(right["advantageous_choice"]) else 0.0
            for left, right in zip(reference_trace, comparison_trace)
        ) if reference_trace and comparison_trace else 0.0
    )
    return metric_deltas


def _metric_noise_floor(base_payload: dict[str, Any], comparison_payloads: list[dict[str, Any]], *, task: str) -> dict[str, float]:
    summarizer = _summarize_confidence_shift if task == "confidence" else _summarize_igt_shift
    observed: dict[str, list[float]] = defaultdict(list)
    for comparison_payload in comparison_payloads:
        deltas = summarizer(base_payload, comparison_payload)
        for metric_name, value in deltas.items():
            observed[metric_name].append(float(value))
    return {
        metric_name: _safe_round(mean(values) if values else 0.0)
        for metric_name, values in observed.items()
    }


def _behavior_change_ok(metric_name: str, shift_summary: dict[str, float]) -> bool:
    if metric_name.startswith("confidence_"):
        return bool(
            float(shift_summary.get("confidence_mean_abs_probability_shift", 0.0)) >= SENSITIVITY_CONFIDENCE_PROB_SHIFT_MIN
            or float(shift_summary.get("confidence_mean_abs_rating_shift", 0.0)) >= SENSITIVITY_CONFIDENCE_RATING_SHIFT_MIN
            or float(shift_summary.get("confidence_choice_flip_rate", 0.0)) > 0.0
        )
    return bool(
        float(shift_summary.get("igt_deck_flip_rate", 0.0)) >= SENSITIVITY_IGT_DECK_FLIP_MIN
        or float(shift_summary.get("igt_advantageous_flip_rate", 0.0)) >= SENSITIVITY_IGT_ADVANTAGE_FLIP_MIN
    )


def _activity_threshold(metric_name: str, noise_floor: float) -> float:
    practical_floor = float(SENSITIVITY_METRIC_FLOORS.get(metric_name, 0.01))
    return max(practical_floor, noise_floor * SENSITIVITY_SIGNAL_TO_NOISE_MIN)


def _choose_parameter_activity(
    parameter_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant_metrics = PARAMETER_SENSITIVITY_DIAGNOSTICS.get(parameter_name, ())
    best_candidate: dict[str, Any] | None = None
    best_score = -1.0
    for candidate in candidates:
        shift_summary = dict(candidate["shift_summary"])
        noise_floor = dict(candidate["noise_floor"])
        for metric_name in relevant_metrics:
            effect = float(shift_summary.get(metric_name, 0.0))
            metric_noise_floor = float(noise_floor.get(metric_name, 0.0))
            threshold = _activity_threshold(metric_name, metric_noise_floor)
            behavior_ok = _behavior_change_ok(metric_name, shift_summary)
            score = effect / max(threshold, 1e-6) if behavior_ok else 0.0
            if score > best_score + 1e-9:
                best_score = score
                best_candidate = {
                    "active": bool(behavior_ok and effect >= threshold),
                    "winning_metric": metric_name,
                    "winning_task": "confidence" if metric_name.startswith("confidence_") else "igt",
                    "winning_step": float(candidate["step"]),
                    "winning_direction": str(candidate["direction"]),
                    "signal_to_noise": _safe_round(effect / max(metric_noise_floor, 1e-6)),
                    "noise_floor": _safe_round(metric_noise_floor),
                    "practical_threshold": _safe_round(float(SENSITIVITY_METRIC_FLOORS.get(metric_name, 0.01))),
                    "activity_threshold": _safe_round(threshold),
                    "winning_effect": _safe_round(effect),
                    "winning_behavioral_change": {
                        key: _safe_round(float(shift_summary.get(key, 0.0)))
                        for key in (
                            "confidence_mean_abs_probability_shift",
                            "confidence_mean_abs_rating_shift",
                            "confidence_choice_flip_rate",
                            "igt_deck_flip_rate",
                            "igt_advantageous_flip_rate",
                        )
                    },
                    "behavior_change_ok": behavior_ok,
                }
    if best_candidate is None:
        return {
            "active": False,
            "winning_metric": None,
            "winning_task": None,
            "winning_step": None,
            "winning_direction": None,
            "signal_to_noise": 0.0,
            "noise_floor": 0.0,
            "practical_threshold": 0.0,
            "activity_threshold": 0.0,
            "winning_effect": 0.0,
            "winning_behavioral_change": {},
            "behavior_change_ok": False,
        }
    return best_candidate


def _score_confidence_metrics(metrics: dict[str, float]) -> float:
    return (
        float(metrics["heldout_likelihood"]) * 2.2
        + float(metrics["confidence_alignment"]) * 0.8
        + float(metrics["accuracy"]) * 0.4
        - float(metrics["brier_score"]) * 0.6
        - float(metrics["calibration_error"]) * 0.4
    )


def _score_igt_metrics(metrics: dict[str, float]) -> float:
    advantageous_gap = abs(float(metrics["advantageous_choice_rate"]) - float(metrics["human_advantageous_rate"]))
    learning_curve_gap = abs(float(metrics["learning_curve_slope"]) - float(metrics["human_learning_curve_slope"]))
    return (
        float(metrics["deck_match_rate"]) * 4.6
        + float(metrics["policy_alignment_rate"]) * 1.2
        + float(metrics.get("subject_match_floor", 0.0)) * 0.8
        - float(metrics.get("deck_distribution_l1", 0.0)) * 0.95
        - float(metrics.get("phase_deck_distribution_l1", 0.0)) * 1.15
        - advantageous_gap * 1.35
        - float(metrics.get("late_advantageous_gap", advantageous_gap)) * 1.15
        - float(metrics.get("phase_advantageous_gap", advantageous_gap)) * 1.00
        - learning_curve_gap * 0.90
        - float(metrics.get("post_loss_switch_gap", 0.0)) * 0.55
    )


def _mean_metrics(metric_payloads: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for payload in metric_payloads for key in payload})
    return {
        key: _safe_round(mean(float(payload.get(key, 0.0)) for payload in metric_payloads))
        for key in keys
    }


@dataclass(frozen=True)
class CandidateFit:
    parameters: CognitiveStyleParameters
    objective: float
    metrics: dict[str, float]
    label: str


def _candidate_summary(candidate: CandidateFit) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "objective": _safe_round(candidate.objective),
        "parameters": candidate.parameters.to_dict(),
        "metrics": dict(candidate.metrics),
    }


def _coordinate_descent_fit_confidence(
    training_trials: list[BenchmarkTrial],
    validation_trials: list[BenchmarkTrial],
    *,
    seed: int,
) -> dict[str, Any]:
    parameter_names = ["confidence_gain", "uncertainty_sensitivity", "attention_selectivity", "error_aversion"]
    current = CognitiveStyleParameters()
    history: list[CandidateFit] = []
    training_cache: dict[str, tuple[float, dict[str, float]]] = {}

    def evaluate_candidate(parameters: CognitiveStyleParameters, label: str) -> CandidateFit:
        cache_key = json.dumps(parameters.to_dict(), sort_keys=True)
        cached = training_cache.get(cache_key)
        if cached is None:
            metrics = _simulate_confidence_trials(training_trials, parameters, seed=seed, include_predictions=False)["metrics"]
            cached = (_score_confidence_metrics(metrics), metrics)
            training_cache[cache_key] = cached
        objective, metrics = cached
        return CandidateFit(parameters=parameters, objective=objective, metrics=dict(metrics), label=label)

    current_fit = evaluate_candidate(current, "default")
    history.append(current_fit)
    for step in (0.20, 0.12, 0.06):
        improved = True
        while improved:
            improved = False
            for parameter_name in parameter_names:
                best_candidate = current_fit
                for delta in (-step, step):
                    candidate_parameters = _set_parameter(current, parameter_name, getattr(current, parameter_name) + delta)
                    candidate = evaluate_candidate(candidate_parameters, f"{parameter_name}_{'plus' if delta > 0 else 'minus'}_{step:.2f}")
                    history.append(candidate)
                    if candidate.objective > best_candidate.objective + 1e-6:
                        best_candidate = candidate
                if best_candidate.parameters != current:
                    current = best_candidate.parameters
                    current_fit = best_candidate
                    improved = True

    unique_candidates: dict[str, CandidateFit] = {}
    for candidate in history:
        unique_candidates[json.dumps(candidate.parameters.to_dict(), sort_keys=True)] = candidate

    validation_candidates: list[CandidateFit] = []
    for index, candidate in enumerate(unique_candidates.values(), start=1):
        metrics = _simulate_confidence_trials(
            validation_trials,
            candidate.parameters,
            seed=seed + 200 + index,
            include_predictions=False,
        )["metrics"]
        validation_candidates.append(
            CandidateFit(
                parameters=candidate.parameters,
                objective=_score_confidence_metrics(metrics),
                metrics=metrics,
                label=f"validation_{index}",
            )
        )
    selected = max(validation_candidates, key=lambda item: (item.objective, item.label))
    return {
        "search_strategy": "coordinate_descent_top4",
        "search_parameters": list(parameter_names),
        "selected_parameters": selected.parameters.to_dict(),
        "validation_metrics": selected.metrics,
        "validation_objective": _safe_round(selected.objective),
        "history": [_candidate_summary(candidate) for candidate in history[:40]],
        "selected_candidate": _candidate_summary(selected),
        "parameters": selected.parameters,
    }


def _coordinate_descent_fit_igt(
    training_trials: list[IowaTrial],
    validation_trials: list[IowaTrial],
    *,
    seed: int,
) -> dict[str, Any]:
    parameter_names = [
        "attention_selectivity",
        "error_aversion",
        "exploration_bias",
        "update_rigidity",
        "confidence_gain",
        "uncertainty_sensitivity",
        "virtual_prediction_error_gain",
    ]
    current = CognitiveStyleParameters()
    history: list[CandidateFit] = []
    training_cache: dict[str, tuple[float, dict[str, float]]] = {}
    training_seed_offsets = (0, 97)
    validation_seed_offsets = (0, 131)

    def evaluate_candidate(parameters: CognitiveStyleParameters, label: str) -> CandidateFit:
        cache_key = json.dumps(parameters.to_dict(), sort_keys=True)
        cached = training_cache.get(cache_key)
        if cached is None:
            metric_runs = [
                _simulate_igt_trials(
                    training_trials,
                    parameters,
                    seed=seed + offset,
                    include_predictions=False,
                )["metrics"]
                for offset in training_seed_offsets
            ]
            metrics = _mean_metrics(metric_runs)
            cached = (_score_igt_metrics(metrics), metrics)
            training_cache[cache_key] = cached
        objective, metrics = cached
        return CandidateFit(parameters=parameters, objective=objective, metrics=dict(metrics), label=label)

    current_fit = evaluate_candidate(current, "default")
    history.append(current_fit)
    for step in (0.20, 0.12, 0.06):
        improved = True
        while improved:
            improved = False
            for parameter_name in parameter_names:
                best_candidate = current_fit
                for delta in (-step, step):
                    candidate_parameters = _set_parameter(current, parameter_name, getattr(current, parameter_name) + delta)
                    candidate = evaluate_candidate(
                        candidate_parameters,
                        f"{parameter_name}_{'plus' if delta > 0 else 'minus'}_{step:.2f}",
                    )
                    history.append(candidate)
                    if candidate.objective > best_candidate.objective + 1e-6:
                        best_candidate = candidate
                if best_candidate.parameters != current:
                    current = best_candidate.parameters
                    current_fit = best_candidate
                    improved = True

    unique_candidates: dict[str, CandidateFit] = {}
    for candidate in history:
        unique_candidates[json.dumps(candidate.parameters.to_dict(), sort_keys=True)] = candidate

    validation_candidates: list[CandidateFit] = []
    for index, candidate in enumerate(unique_candidates.values(), start=1):
        validation_runs = [
            _simulate_igt_trials(
                validation_trials,
                candidate.parameters,
                seed=seed + 500 + index + offset,
                include_predictions=False,
            )["metrics"]
            for offset in validation_seed_offsets
        ]
        metrics = _mean_metrics(validation_runs)
        validation_candidates.append(
            CandidateFit(
                parameters=candidate.parameters,
                objective=_score_igt_metrics(metrics),
                metrics=metrics,
                label=f"validation_{index}",
            )
        )
    selected = max(validation_candidates, key=lambda item: (item.objective, item.label))
    return {
        "search_strategy": "coordinate_descent_top6",
        "search_parameters": list(parameter_names),
        "selected_parameters": selected.parameters.to_dict(),
        "validation_metrics": selected.metrics,
        "validation_objective": _safe_round(selected.objective),
        "history": [_candidate_summary(candidate) for candidate in history[:40]],
        "selected_candidate": _candidate_summary(selected),
        "parameters": selected.parameters,
    }


def _competitive_parity(agent_metrics: dict[str, float], baseline_metrics: dict[str, float], *, primary_metric: str) -> bool:
    if float(agent_metrics[primary_metric]) >= float(baseline_metrics[primary_metric]):
        return True
    baseline_brier = float(baseline_metrics.get("brier_score", 0.0))
    agent_brier = float(agent_metrics.get("brier_score", baseline_brier))
    if baseline_brier > 0.0 and agent_brier <= baseline_brier * 1.05:
        return True
    return False


def _relative_worse_than(agent_value: float, baseline_value: float, *, tolerance: float = 0.15) -> bool:
    if agent_value >= baseline_value:
        return False
    denom = max(abs(baseline_value), 1e-6)
    return abs(agent_value - baseline_value) / denom > tolerance


def _confidence_failure_modes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = list(payload["predictions"])
    trace = list(payload["trial_trace"])
    bins = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]
    grouped: list[dict[str, Any]] = []
    for low, high in bins:
        rows = [row for row in trace if low <= abs(float(row["stimulus_strength"])) < high]
        if not rows:
            continue
        grouped.append(
            {
                "stimulus_bin": [low, round(min(high, 1.0), 2)],
                "trial_count": len(rows),
                "human_choice_match_rate": _safe_round(mean(1.0 if row["agent_choice"] == row["human_choice"] else 0.0 for row in rows)),
                "mean_calibration_gap": _safe_round(mean(abs(float(row["agent_confidence_rating"]) - float(row["human_confidence"])) for row in rows)),
            }
        )
    worst_examples = sorted(
        predictions,
        key=lambda row: abs(float(row["predicted_confidence"]) - float(row["human_confidence"])),
        reverse=True,
    )[:5]
    return [
        {
            "failure_mode": "stimulus_bin_breakdown",
            "examples": sorted(grouped, key=lambda item: (item["human_choice_match_rate"], -item["mean_calibration_gap"]))[:3],
        },
        {
            "failure_mode": "largest_confidence_disagreements",
            "examples": worst_examples,
        },
    ]


def _igt_failure_modes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(payload["trial_trace"])
    phases = {
        "early_1_20": [row for row in trace if 1 <= int(row["trial_index"]) <= 20],
        "middle_41_60": [row for row in trace if 41 <= int(row["trial_index"]) <= 60],
        "late_81_100": [row for row in trace if 81 <= int(row["trial_index"]) <= 100],
    }
    phase_examples = []
    for phase_name, rows in phases.items():
        if not rows:
            continue
        phase_examples.append(
            {
                "phase": phase_name,
                "trial_count": len(rows),
                "deck_match_rate": _safe_round(mean(1.0 if bool(row["deck_match"]) else 0.0 for row in rows)),
                "advantageous_choice_rate": _safe_round(mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in rows)),
                "human_advantageous_rate": _safe_round(mean(1.0 if bool(row["actual_advantageous"]) else 0.0 for row in rows)),
            }
        )
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace:
        by_subject[str(row["subject_id"])].append(row)
    worst_subjects = sorted(
        (
            {
                "subject_id": subject_id,
                "deck_match_rate": _safe_round(mean(1.0 if bool(row["deck_match"]) else 0.0 for row in rows)),
                "advantageous_choice_rate": _safe_round(mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in rows)),
            }
            for subject_id, rows in by_subject.items()
        ),
        key=lambda item: (item["deck_match_rate"], item["advantageous_choice_rate"], item["subject_id"]),
    )[:5]
    return [
        {"failure_mode": "phase_breakdown", "examples": phase_examples},
        {"failure_mode": "lowest_matching_subjects", "examples": worst_subjects},
    ]


def run_fitted_confidence_agent(
    *,
    seed: int = 43,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    limits = sample_limits or {}
    data = _load_confidence_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    training_trials = _balanced_subject_sample(
        data["splits"].get("train", data["all_trials"]),
        max_trials=limits.get("confidence_train_max_trials", CONFIDENCE_FIT_TRAIN_MAX_TRIALS),
        max_subjects=limits.get("confidence_train_max_subjects", CONFIDENCE_MAX_SUBJECTS),
        seed=seed,
    )
    validation_trials = _balanced_subject_sample(
        data["splits"].get("validation", data["all_trials"]),
        max_trials=limits.get("confidence_validation_max_trials", CONFIDENCE_FIT_VALIDATION_MAX_TRIALS),
        max_subjects=limits.get("confidence_validation_max_subjects", CONFIDENCE_MAX_SUBJECTS),
        seed=seed + 1,
    )
    heldout_trials = _balanced_subject_sample(
        data["splits"].get("heldout", data["all_trials"]),
        max_trials=limits.get("confidence_heldout_max_trials", CONFIDENCE_EVAL_HELDOUT_MAX_TRIALS),
        max_subjects=limits.get("confidence_heldout_max_subjects", CONFIDENCE_MAX_SUBJECTS * 2),
        seed=seed + 2,
    )
    fit = _coordinate_descent_fit_confidence(training_trials, validation_trials, seed=seed)
    agent = _simulate_confidence_trials(heldout_trials, fit["parameters"], seed=seed + 1000)
    baselines = {
        "random": run_confidence_random_baseline(heldout_trials, seed=seed + 2000),
        "stimulus_only": run_confidence_stimulus_only_baseline(training_trials, heldout_trials),
        "statistical_logistic": run_confidence_statistical_baseline(training_trials, heldout_trials, seed=seed + 2001),
        "human_match_ceiling": run_confidence_human_match_ceiling(training_trials, heldout_trials),
    }
    agent_metrics = dict(agent["metrics"])
    ladder = {
        "primary_metric": "heldout_likelihood",
        "lower_baselines_beaten": bool(
            float(agent_metrics["heldout_likelihood"]) > float(baselines["random"]["metrics"]["heldout_likelihood"])
            and float(agent_metrics["heldout_likelihood"]) > float(baselines["stimulus_only"]["metrics"]["heldout_likelihood"])
        ),
        "competitive_baseline_matched": any(
            _competitive_parity(agent_metrics, baselines[name]["metrics"], primary_metric="heldout_likelihood")
            for name in ("statistical_logistic", "human_match_ceiling")
        ),
        "competitive_review_block": all(
            _relative_worse_than(
                float(agent_metrics["heldout_likelihood"]),
                float(baselines[name]["metrics"]["heldout_likelihood"]),
            )
            for name in ("statistical_logistic", "human_match_ceiling")
        ),
    }
    leakage_check = {
        "subject": detect_subject_leakage(training_trials + validation_trials + heldout_trials, key_field="subject_id"),
        "session": detect_subject_leakage(training_trials + validation_trials + heldout_trials, key_field="session_id"),
    }
    return {
        "benchmark_id": "confidence_database",
        "mode": "benchmark_eval" if data["source_type"] == "external_bundle" else "smoke_only",
        "bundle": data["bundle"],
        "validation": data["validation"],
        "benchmark_status": data["benchmark_status"],
        "source_type": data["source_type"],
        "claim_envelope": data["claim_envelope"],
        "external_validation": False,
        "split_unit": "subject_id",
        "training_trial_count": len(training_trials),
        "validation_trial_count": len(validation_trials),
        "trial_count": len(heldout_trials),
        "metrics": {
            **agent_metrics,
            "confidence_correlation": agent_metrics["confidence_alignment"],
            "primary_metric": agent_metrics["heldout_likelihood"],
        },
        "subject_summary": agent["subject_summary"],
        "predictions": list(agent["predictions"]),
        "trial_trace": list(agent["trial_trace"]),
        "fit": {
            "strategy": fit["search_strategy"],
            "search_parameters": fit["search_parameters"],
            "selected_parameters": fit["selected_parameters"],
            "validation_metrics": fit["validation_metrics"],
            "validation_objective": fit["validation_objective"],
            "selected_candidate": fit["selected_candidate"],
            "history": fit["history"],
        },
        "baselines": baselines,
        "baseline_ladder": ladder,
        "leakage_check": leakage_check,
        "failure_modes": _confidence_failure_modes(agent),
    }


def run_fitted_igt_agent(
    *,
    seed: int = 44,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    limits = sample_limits or {}
    data = _load_igt_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    training_trials = _balanced_subject_sample(
        list(data["splits"].get("train", data["all_trials"])),
        max_trials=limits.get("igt_train_max_trials"),
        max_subjects=limits.get("igt_train_max_subjects"),
        seed=seed,
    )
    validation_trials = _balanced_subject_sample(
        list(data["splits"].get("validation", data["all_trials"])),
        max_trials=limits.get("igt_validation_max_trials"),
        max_subjects=limits.get("igt_validation_max_subjects"),
        seed=seed + 1,
    )
    heldout_trials = _balanced_subject_sample(
        list(data["splits"].get("heldout", data["all_trials"])),
        max_trials=limits.get("igt_heldout_max_trials"),
        max_subjects=limits.get("igt_heldout_max_subjects"),
        seed=seed + 2,
    )
    fit = _coordinate_descent_fit_igt(training_trials, validation_trials, seed=seed)
    agent = _simulate_igt_trials(heldout_trials, fit["parameters"], seed=seed + 1000)
    baselines = {
        "random": run_igt_random_baseline(heldout_trials, seed=seed + 2000),
        "frequency_matching": run_igt_frequency_matching_baseline(training_trials, heldout_trials, seed=seed + 2001),
        "human_behavior": run_igt_human_behavior_baseline(training_trials, heldout_trials),
    }
    agent_metrics = dict(agent["metrics"])
    agent_metrics["deck_selection_accuracy"] = agent_metrics["deck_match_rate"]
    agent_metrics["advantageous_deck_ratio"] = agent_metrics["advantageous_choice_rate"]
    agent_metrics["primary_metric"] = agent_metrics["deck_match_rate"]
    ladder = {
        "primary_metric": "deck_match_rate",
        "lower_baselines_beaten": bool(
            float(agent_metrics["deck_match_rate"]) > float(baselines["random"]["metrics"]["deck_match_rate"])
            and float(agent_metrics["deck_match_rate"]) > float(baselines["frequency_matching"]["metrics"]["deck_match_rate"])
        ),
        "competitive_baseline_matched": _competitive_parity(
            agent_metrics,
            baselines["human_behavior"]["metrics"],
            primary_metric="deck_match_rate",
        ),
        "competitive_review_block": _relative_worse_than(
            float(agent_metrics["deck_match_rate"]),
            float(baselines["human_behavior"]["metrics"]["deck_match_rate"]),
        ),
    }
    leakage_check = {"subject": detect_subject_leakage(training_trials + validation_trials + heldout_trials, key_field="subject_id")}
    return {
        "benchmark_id": "iowa_gambling_task",
        "mode": "benchmark_eval" if data["source_type"] == "external_bundle" else "smoke_only",
        "bundle": data["bundle"],
        "validation": data["validation"],
        "benchmark_status": data["benchmark_status"],
        "source_type": data["source_type"],
        "claim_envelope": data["claim_envelope"],
        "external_validation": False,
        "split_unit": "subject_id",
        "protocol_mode": "standard_100",
        "training_trial_count": len(training_trials),
        "validation_trial_count": len(validation_trials),
        "trial_count": len(heldout_trials),
        "metrics": agent_metrics,
        "subject_summary": agent["subject_summary"],
        "predictions": list(agent["predictions"]),
        "trial_trace": list(agent["trial_trace"]),
        "fit": {
            "strategy": fit["search_strategy"],
            "search_parameters": fit["search_parameters"],
            "selected_parameters": fit["selected_parameters"],
            "validation_metrics": fit["validation_metrics"],
            "validation_objective": fit["validation_objective"],
            "selected_candidate": fit["selected_candidate"],
            "history": fit["history"],
        },
        "baselines": baselines,
        "baseline_ladder": ladder,
        "leakage_check": leakage_check,
        "failure_modes": _igt_failure_modes(agent),
    }


def _run_parameter_sensitivity_analysis(
    *,
    seed: int = 45,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
    confidence_anchor_parameters: CognitiveStyleParameters | None = None,
    igt_anchor_parameters: CognitiveStyleParameters | None = None,
) -> dict[str, Any]:
    limits = sample_limits or {}
    confidence_data = _load_confidence_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    igt_data = _load_igt_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    confidence_trials = _balanced_subject_sample(
        confidence_data["splits"].get("heldout", confidence_data["all_trials"]),
        max_trials=limits.get("sensitivity_confidence_max_trials", CONFIDENCE_SENSITIVITY_MAX_TRIALS),
        max_subjects=limits.get("sensitivity_confidence_max_subjects", CONFIDENCE_MAX_SUBJECTS),
        seed=seed,
    )
    igt_trials = list(igt_data["splits"].get("heldout", igt_data["all_trials"]))
    if confidence_anchor_parameters is None:
        confidence_fit_payload = run_fitted_confidence_agent(
            seed=seed - 2,
            benchmark_root=benchmark_root,
            allow_smoke_test=allow_smoke_test,
            sample_limits=sample_limits,
        )
        confidence_anchor_parameters = CognitiveStyleParameters.from_dict(dict(confidence_fit_payload["fit"]["selected_parameters"]))
    if igt_anchor_parameters is None:
        igt_fit_payload = run_fitted_igt_agent(
            seed=seed - 1,
            benchmark_root=benchmark_root,
            allow_smoke_test=allow_smoke_test,
            sample_limits=sample_limits,
        )
        igt_anchor_parameters = CognitiveStyleParameters.from_dict(dict(igt_fit_payload["fit"]["selected_parameters"]))

    confidence_seed = seed + 4100
    igt_seed = seed + 5100
    confidence_cache: dict[str, dict[str, Any]] = {}
    igt_cache: dict[str, dict[str, Any]] = {}

    def run_confidence(parameters: CognitiveStyleParameters, active_seed: int) -> dict[str, Any]:
        cache_key = f"{json.dumps(parameters.to_dict(), sort_keys=True)}::confidence::{active_seed}"
        cached = confidence_cache.get(cache_key)
        if cached is None:
            cached = _simulate_confidence_trials(
                confidence_trials,
                parameters,
                seed=active_seed,
                include_predictions=False,
            )
            confidence_cache[cache_key] = cached
        return cached

    def run_igt(parameters: CognitiveStyleParameters, active_seed: int) -> dict[str, Any]:
        cache_key = f"{json.dumps(parameters.to_dict(), sort_keys=True)}::igt::{active_seed}"
        cached = igt_cache.get(cache_key)
        if cached is None:
            cached = _simulate_igt_trials(
                igt_trials,
                parameters,
                seed=active_seed,
                include_predictions=False,
            )
            igt_cache[cache_key] = cached
        return cached

    baseline_confidence = run_confidence(confidence_anchor_parameters, confidence_seed)
    baseline_igt = run_igt(igt_anchor_parameters, igt_seed)
    confidence_noise_floor = _metric_noise_floor(
        baseline_confidence,
        [run_confidence(confidence_anchor_parameters, confidence_seed + offset) for offset in SENSITIVITY_NOISE_SEED_OFFSETS],
        task="confidence",
    )
    igt_noise_floor = _metric_noise_floor(
        baseline_igt,
        [run_igt(igt_anchor_parameters, igt_seed + offset) for offset in SENSITIVITY_NOISE_SEED_OFFSETS],
        task="igt",
    )

    sensitivity_rows = []
    active_count = 0
    for index, (parameter_name, spec) in enumerate(PARAMETER_REFERENCE.items(), start=1):
        default_value = float(spec["default"])
        step_candidates: list[dict[str, Any]] = []
        step_summaries: list[dict[str, Any]] = []
        for step in SENSITIVITY_STEP_SCHEDULE:
            low_confidence_parameters = _set_parameter(
                confidence_anchor_parameters,
                parameter_name,
                getattr(confidence_anchor_parameters, parameter_name) - step,
            )
            high_confidence_parameters = _set_parameter(
                confidence_anchor_parameters,
                parameter_name,
                getattr(confidence_anchor_parameters, parameter_name) + step,
            )
            low_igt_parameters = _set_parameter(
                igt_anchor_parameters,
                parameter_name,
                getattr(igt_anchor_parameters, parameter_name) - step,
            )
            high_igt_parameters = _set_parameter(
                igt_anchor_parameters,
                parameter_name,
                getattr(igt_anchor_parameters, parameter_name) + step,
            )
            for direction, candidate_confidence_parameters, candidate_igt_parameters in (
                ("low", low_confidence_parameters, low_igt_parameters),
                ("high", high_confidence_parameters, high_igt_parameters),
            ):
                confidence_shift = _summarize_confidence_shift(
                    baseline_confidence,
                    run_confidence(candidate_confidence_parameters, confidence_seed),
                )
                igt_shift = _summarize_igt_shift(
                    baseline_igt,
                    run_igt(candidate_igt_parameters, igt_seed),
                )
                combined_shift = {**confidence_shift, **igt_shift}
                step_candidates.append(
                    {
                        "step": step,
                        "direction": direction,
                        "shift_summary": combined_shift,
                        "noise_floor": {**confidence_noise_floor, **igt_noise_floor},
                        "confidence_value": float(getattr(candidate_confidence_parameters, parameter_name)),
                        "igt_value": float(getattr(candidate_igt_parameters, parameter_name)),
                    }
                )
            decision = _choose_parameter_activity(parameter_name, step_candidates)
            step_summaries.append(
                {
                    "step": _safe_round(step),
                    "active_after_step": bool(decision["active"]),
                    "winning_metric": decision["winning_metric"],
                    "winning_effect": decision["winning_effect"],
                    "activity_threshold": decision["activity_threshold"],
                }
            )
            if bool(decision["active"]):
                break

        decision = _choose_parameter_activity(parameter_name, step_candidates)
        aggregate_effects: dict[str, float] = {}
        for metric_name in (
            "confidence_heldout_likelihood",
            "confidence_brier_score",
            "confidence_alignment",
            "igt_deck_match_rate",
            "igt_advantageous_ratio",
            "igt_learning_curve_slope",
        ):
            aggregate_effects[metric_name] = _safe_round(
                max(float(candidate["shift_summary"].get(metric_name, 0.0)) for candidate in step_candidates) if step_candidates else 0.0
            )
        aggregate_diagnostics = {
            metric_name: {
                "effect": _safe_round(
                    max(float(candidate["shift_summary"].get(metric_name, 0.0)) for candidate in step_candidates) if step_candidates else 0.0
                ),
                "noise_floor": _safe_round(float({**confidence_noise_floor, **igt_noise_floor}.get(metric_name, 0.0))),
                "practical_threshold": _safe_round(float(SENSITIVITY_METRIC_FLOORS.get(metric_name, 0.01))),
                "signal_to_noise": _safe_round(
                    (
                        max(float(candidate["shift_summary"].get(metric_name, 0.0)) for candidate in step_candidates)
                        / max(float({**confidence_noise_floor, **igt_noise_floor}.get(metric_name, 0.0)), 1e-6)
                    ) if step_candidates else 0.0
                ),
            }
            for metric_name in sorted(
                {
                    *PARAMETER_SENSITIVITY_DIAGNOSTICS.get(parameter_name, ()),
                    "confidence_heldout_likelihood",
                    "confidence_brier_score",
                    "confidence_alignment",
                    "igt_deck_match_rate",
                    "igt_advantageous_ratio",
                    "igt_learning_curve_slope",
                }
            )
        }
        active = bool(decision["active"])
        active_count += 1 if active else 0
        chosen_candidate = next(
            (
                candidate for candidate in step_candidates
                if float(candidate["step"]) == float(decision["winning_step"] or 0.0)
                and str(candidate["direction"]) == str(decision["winning_direction"])
            ),
            step_candidates[-1] if step_candidates else None,
        )
        confidence_anchor_value = float(getattr(confidence_anchor_parameters, parameter_name))
        igt_anchor_value = float(getattr(igt_anchor_parameters, parameter_name))
        if chosen_candidate is not None and decision["winning_task"] == "igt":
            low_value = _safe_round(min(igt_anchor_value, float(chosen_candidate["igt_value"])))
            high_value = _safe_round(max(igt_anchor_value, float(chosen_candidate["igt_value"])))
        elif chosen_candidate is not None:
            low_value = _safe_round(min(confidence_anchor_value, float(chosen_candidate["confidence_value"])))
            high_value = _safe_round(max(confidence_anchor_value, float(chosen_candidate["confidence_value"])))
        else:
            low_value = _safe_round(confidence_anchor_value)
            high_value = _safe_round(confidence_anchor_value)
        if active:
            classification_reason = (
                f"{decision['winning_metric']} exceeded both the practical floor ({decision['practical_threshold']}) "
                f"and the seed-noise-adjusted threshold ({decision['activity_threshold']})."
            )
        else:
            classification_reason = (
                f"No relevant metric cleared the activity threshold above seed noise; "
                f"best metric was {decision['winning_metric'] or 'none'} with effect {decision['winning_effect']}."
            )
        sensitivity_rows.append(
            {
                "parameter": parameter_name,
                "default": default_value,
                "sigma": PARAMETER_SWEEP_SIGMA,
                "low_value": low_value,
                "high_value": high_value,
                "active": active,
                "measured_effects": aggregate_effects,
                "confidence_anchor_value": _safe_round(confidence_anchor_value),
                "igt_anchor_value": _safe_round(igt_anchor_value),
                "winning_task": decision["winning_task"],
                "winning_metric": decision["winning_metric"],
                "winning_step": decision["winning_step"],
                "winning_direction": decision["winning_direction"],
                "noise_floor": decision["noise_floor"],
                "signal_to_noise": decision["signal_to_noise"],
                "practical_threshold": decision["practical_threshold"],
                "activity_threshold": decision["activity_threshold"],
                "winning_effect": decision["winning_effect"],
                "winning_behavioral_change": decision["winning_behavioral_change"],
                "behavior_change_ok": decision["behavior_change_ok"],
                "classification_reason": classification_reason,
                "relevant_metrics": list(PARAMETER_SENSITIVITY_DIAGNOSTICS.get(parameter_name, ())),
                "diagnostic_effects": aggregate_diagnostics,
                "evaluated_steps": step_summaries,
            }
        )

    return {
        "benchmark_id": "m43_parameter_sensitivity",
        "source_type": "external_bundle" if confidence_data["source_type"] == "external_bundle" and igt_data["source_type"] == "external_bundle" else "synthetic_protocol",
        "claim_envelope": "benchmark_eval"
        if confidence_data["source_type"] == "external_bundle" and igt_data["source_type"] == "external_bundle"
        else "synthetic_diagnostic",
        "external_validation": False,
        "active_parameter_count": active_count,
        "required_active_parameter_count": 4,
        "parameters": sensitivity_rows,
        "baseline_metrics": {
            "confidence": baseline_confidence["metrics"],
            "igt": baseline_igt["metrics"],
        },
        "baseline_parameters": {
            "confidence": confidence_anchor_parameters.to_dict(),
            "igt": igt_anchor_parameters.to_dict(),
        },
        "baseline_diagnostics": {
            "confidence": _confidence_trace_diagnostics(list(baseline_confidence["trial_trace"])),
            "igt": _igt_trace_diagnostics(list(baseline_igt["trial_trace"])),
        },
        "noise_floor": {
            "confidence": confidence_noise_floor,
            "igt": igt_noise_floor,
        },
        "analysis_protocol": {
            "confidence_split": "heldout",
            "igt_split": "heldout",
            "step_schedule": list(SENSITIVITY_STEP_SCHEDULE),
            "noise_seed_offsets": list(SENSITIVITY_NOISE_SEED_OFFSETS),
            "signal_to_noise_min": SENSITIVITY_SIGNAL_TO_NOISE_MIN,
            "noise_estimator": "mean_abs_seed_delta",
        },
    }


def run_parameter_sensitivity_analysis(
    *,
    seed: int = 45,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    return _run_parameter_sensitivity_analysis(
        seed=seed,
        benchmark_root=benchmark_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )


def _blocked_suite(*, benchmark_root: Path | str | None = None) -> dict[str, Any]:
    return {
        "seed": None,
        "benchmark_root": str(benchmark_root) if benchmark_root else None,
        "blocked": True,
        "acceptance_state": "blocked_missing_external_bundle",
        "confidence": {"mode": "blocked", "status": "blocked"},
        "igt": {"mode": "blocked", "status": "blocked"},
        "parameter_sensitivity": {"mode": "blocked", "status": "blocked"},
        "failure_analysis": {"failure_modes": []},
    }


def run_m43_single_task_suite(
    *,
    seed: int = 43,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    resolved_root = None if allow_smoke_test and benchmark_root is None else _resolve_benchmark_root(benchmark_root)
    if resolved_root is None and not allow_smoke_test:
        return _blocked_suite(benchmark_root=benchmark_root)

    confidence = run_fitted_confidence_agent(
        seed=seed,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )
    igt = run_fitted_igt_agent(
        seed=seed + 1,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )
    sensitivity = _run_parameter_sensitivity_analysis(
        seed=seed + 2,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        confidence_anchor_parameters=CognitiveStyleParameters.from_dict(dict(confidence["fit"]["selected_parameters"])),
        igt_anchor_parameters=CognitiveStyleParameters.from_dict(dict(igt["fit"]["selected_parameters"])),
    )
    return {
        "seed": seed,
        "benchmark_root": str(resolved_root) if resolved_root else None,
        "blocked": False,
        "acceptance_state": "benchmark_eval" if resolved_root else "smoke_only",
        "confidence": confidence,
        "igt": igt,
        "parameter_sensitivity": sensitivity,
        "failure_analysis": {
            "failure_modes": {
                "confidence_database": confidence["failure_modes"],
                "iowa_gambling_task": igt["failure_modes"],
            }
        },
    }


__all__ = [
    "CONFIDENCE_EVAL_HELDOUT_MAX_TRIALS",
    "CONFIDENCE_FIT_TRAIN_MAX_TRIALS",
    "CONFIDENCE_FIT_VALIDATION_MAX_TRIALS",
    "CONFIDENCE_SENSITIVITY_MAX_TRIALS",
    "PARAMETER_SWEEP_SIGMA",
    "run_fitted_confidence_agent",
    "run_fitted_igt_agent",
    "run_m43_single_task_suite",
    "run_parameter_sensitivity_analysis",
]
