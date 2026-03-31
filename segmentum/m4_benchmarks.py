from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from collections import defaultdict
from typing import Any, Protocol

from .benchmark_registry import benchmark_status, load_benchmark_bundle, validate_benchmark_bundle
from .m4_cognitive_style import CognitiveStyleParameters, logistic


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class BenchmarkTrial:
    trial_id: str
    subject_id: str
    session_id: str
    stimulus_strength: float
    correct_choice: str
    human_choice: str
    human_confidence: float
    rt_ms: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPrediction:
    trial_id: str
    subject_id: str
    session_id: str
    split: str
    human_choice: str
    predicted_choice: str
    predicted_confidence: float
    predicted_probability_right: float
    correct: bool
    human_choice_match: bool
    human_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IowaTrial:
    trial_id: str
    subject_id: str
    deck: str
    reward: int
    penalty: int
    net_outcome: int
    advantageous: bool
    trial_index: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IowaPrediction:
    trial_id: str
    subject_id: str
    split: str
    chosen_deck: str
    actual_deck: str
    expected_value: float
    predicted_confidence: float
    advantageous_choice: bool
    actual_advantageous: bool
    deck_match: bool
    reward: int
    penalty: int
    net_outcome: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(Protocol):
    def benchmark_id(self) -> str: ...
    def schema(self) -> dict[str, Any]: ...


def _active_grouping_field(record: dict[str, Any], bundle_grouping_fields: list[str]) -> str:
    for field_name in bundle_grouping_fields or ["session_id", "subject_id"]:
        if str(record.get(field_name, "")).strip():
            return field_name
    return "subject_id"


def _group_key(record: dict[str, Any], grouping_field: str) -> str:
    if grouping_field == "session_id":
        session_id = str(record.get("session_id", "")).strip()
        if session_id:
            return session_id
    return str(record.get("subject_id", "")).strip()


def assign_group_splits(records: list[dict[str, Any]], *, preferred_split_unit: str = "session_id") -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouping_field = preferred_split_unit if preferred_split_unit == "session_id" and str(record.get("session_id", "")).strip() else "subject_id"
        groups[_group_key(record, grouping_field)].append(record)
    ordered_groups = sorted(groups.items(), key=lambda item: (item[0], len(item[1])))
    if len(ordered_groups) < 3:
        raise ValueError("Confidence benchmark needs at least three disjoint subject/session groups for train/validation/heldout.")
    split_names = ["train", "validation", "heldout"]
    assignments: dict[str, str] = {}
    for index, (group_id, _rows) in enumerate(ordered_groups):
        split = split_names[min(index, len(split_names) - 1)]
        if index >= len(split_names):
            split = split_names[index % len(split_names)]
        assignments[group_id] = split
    for split_name in split_names:
        if split_name not in assignments.values():
            raise ValueError("Confidence benchmark split assignment did not produce all required splits.")
    return assignments


def detect_subject_leakage(rows: list[BenchmarkTrial | dict[str, Any]], *, key_field: str) -> dict[str, Any]:
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        payload = row.to_dict() if isinstance(row, BenchmarkTrial) else dict(row)
        key = str(payload.get(key_field, "")).strip()
        if not key:
            continue
        observed[key].add(str(payload["split"]))
    leaking_keys = sorted(key for key, splits in observed.items() if len(splits) > 1)
    return {
        "key_field": key_field,
        "ok": not leaking_keys,
        "leaking_keys": leaking_keys,
    }


def preprocess_confidence_database(*, allow_smoke_test: bool = False, selected_split: str | None = None) -> dict[str, Any]:
    bundle = load_benchmark_bundle("confidence_database")
    validation = validate_benchmark_bundle("confidence_database")
    status = benchmark_status("confidence_database")
    if not validation.ok:
        raise ValueError(f"Confidence benchmark bundle is invalid: {validation.to_dict()}")
    if bundle.external_bundle_preferred and bundle.source_type != "external_bundle" and not allow_smoke_test:
        blocker = status.blockers[0] if status.blockers else "Acceptance-grade evaluation requires an external bundle."
        raise ValueError(f"{blocker} Repo sample remains available only for smoke tests.")
    raw_records = [json.loads(line) for line in Path(bundle.data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    processed: list[BenchmarkTrial] = []
    valid_records: list[dict[str, Any]] = []
    grouping_fields = list(bundle.grouping_fields or ["session_id", "subject_id"])
    active_split_unit = "subject_id"
    precomputed_split_available = False
    for record in raw_records:
        required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
        if not required <= record.keys():
            continue
        record = dict(record)
        record.setdefault("session_id", str(record.get("subject_id", "")))
        if "split" in record and str(record.get("split", "")).strip():
            precomputed_split_available = True
        active_split_unit = _active_grouping_field(record, grouping_fields)
        valid_records.append(record)
    split_assignments = None if precomputed_split_available else assign_group_splits(valid_records, preferred_split_unit=bundle.default_split_unit or active_split_unit)
    for record in valid_records:
        group_id = _group_key(record, "session_id" if active_split_unit == "session_id" else "subject_id")
        split = str(record.get("split", "")).strip() if precomputed_split_available else str(split_assignments[group_id])
        if selected_split is not None and split != selected_split:
            continue
        processed.append(
            BenchmarkTrial(
                trial_id=str(record["trial_id"]),
                subject_id=str(record["subject_id"]),
                session_id=str(record.get("session_id") or record["subject_id"]),
                stimulus_strength=float(record["stimulus_strength"]),
                correct_choice=str(record["correct_choice"]),
                human_choice=str(record["human_choice"]),
                human_confidence=_clamp01(float(record["human_confidence"])),
                rt_ms=int(record["rt_ms"]),
                split=split,
            )
        )
    leakage_check = {
        "split_unit": active_split_unit,
        "subject": detect_subject_leakage(processed, key_field="subject_id"),
        "session": detect_subject_leakage(processed, key_field="session_id"),
        "precomputed_split_available": precomputed_split_available,
        "selected_split": selected_split,
    }
    manifest = dict(bundle.manifest)
    return {
        "manifest": manifest,
        "trials": [trial.to_dict() for trial in processed],
        "skipped_records": len(raw_records) - len(processed),
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "split_unit": active_split_unit,
        "bundle_mode": "external_bundle" if bundle.source_type == "external_bundle" else "repo_smoke_test",
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "smoke_only",
        "leakage_check": leakage_check,
    }


def preprocess_iowa_gambling_task(*, allow_smoke_test: bool = False) -> dict[str, Any]:
    bundle = load_benchmark_bundle("iowa_gambling_task")
    validation = validate_benchmark_bundle("iowa_gambling_task")
    status = benchmark_status("iowa_gambling_task")
    if not validation.ok:
        raise ValueError(f"IGT benchmark bundle is invalid: {validation.to_dict()}")
    if bundle.external_bundle_preferred and bundle.source_type != "external_bundle" and not allow_smoke_test:
        blocker = status.blockers[0] if status.blockers else "Acceptance-grade IGT evaluation requires an external bundle."
        raise ValueError(f"{blocker} Repo sample remains available only for smoke tests.")
    raw_records = [json.loads(line) for line in Path(bundle.data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    processed: list[IowaTrial] = []
    for record in raw_records:
        required = {
            "trial_id",
            "subject_id",
            "deck",
            "reward",
            "penalty",
            "net_outcome",
            "advantageous",
            "trial_index",
            "split",
        }
        if not required <= record.keys():
            continue
        processed.append(
            IowaTrial(
                trial_id=str(record["trial_id"]),
                subject_id=str(record["subject_id"]),
                deck=str(record["deck"]),
                reward=int(record["reward"]),
                penalty=int(record["penalty"]),
                net_outcome=int(record["net_outcome"]),
                advantageous=bool(record["advantageous"]),
                trial_index=int(record["trial_index"]),
                split=str(record["split"]),
            )
        )
    manifest = dict(bundle.manifest)
    return {
        "manifest": manifest,
        "trials": [trial.to_dict() for trial in processed],
        "skipped_records": len(raw_records) - len(processed),
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "bundle_mode": "external_bundle" if bundle.source_type == "external_bundle" else "repo_smoke_test",
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "smoke_only",
    }


class ConfidenceDatabaseAdapter:
    def benchmark_id(self) -> str:
        return "confidence_database"

    def schema(self) -> dict[str, Any]:
        bundle = load_benchmark_bundle(self.benchmark_id())
        return {
            "benchmark_id": self.benchmark_id(),
            "status": bundle.status,
            "source_type": bundle.source_type,
            "source_label": bundle.source_label,
            "grouping_fields": bundle.grouping_fields,
            "default_split_unit": bundle.default_split_unit,
            "external_bundle_preferred": bundle.external_bundle_preferred,
            "smoke_test_only": bundle.smoke_test_only,
            "benchmark_state": bundle.benchmark_state,
            "available_states": bundle.available_states,
            "blockers": bundle.blockers,
            "trial_fields": list(BenchmarkTrial.__dataclass_fields__.keys()),
            "observation_fields": ["stimulus_strength", "ambiguity", "evidence_strength"],
            "action_fields": ["choose_left", "choose_right"],
            "feedback_fields": ["correct", "human_choice_match", "predicted_confidence"],
        }

    def load_trials(self, *, allow_smoke_test: bool = False) -> list[BenchmarkTrial]:
        return [BenchmarkTrial(**payload) for payload in preprocess_confidence_database(allow_smoke_test=allow_smoke_test)["trials"]]

    def choose_action(
        self,
        trial: BenchmarkTrial,
        parameters: CognitiveStyleParameters,
        *,
        seed: int,
        trial_index: int,
    ) -> dict[str, Any]:
        resource_stress = min(0.80, 0.18 + trial_index * 0.05)
        evidence_strength = abs(float(trial.stimulus_strength))
        uncertainty = 1.0 - evidence_strength
        signed_strength = float(trial.stimulus_strength)
        bias = (parameters.exploration_bias - 0.5) * 0.4
        caution = parameters.error_aversion * resource_stress * 0.15
        p_right = logistic(signed_strength * 4.2 + bias - caution)
        predicted_choice = "right" if p_right >= 0.5 else "left"
        confidence = _clamp01(
            0.50
            + evidence_strength * (0.28 + parameters.confidence_gain * 0.28)
            - uncertainty * parameters.uncertainty_sensitivity * 0.15
            - resource_stress * parameters.resource_pressure_sensitivity * 0.08
        )
        return {
            "prediction": {
                "trial_id": trial.trial_id,
                "subject_id": trial.subject_id,
                "session_id": trial.session_id,
                "split": trial.split,
                "human_choice": trial.human_choice,
                "predicted_choice": predicted_choice,
                "predicted_confidence": confidence,
                "predicted_probability_right": round(p_right, 6),
                "correct": predicted_choice == trial.correct_choice,
                "human_choice_match": predicted_choice == trial.human_choice,
                "human_confidence": trial.human_confidence,
            },
            "resource_state": {},
        }


class IowaGamblingTaskAdapter:
    def benchmark_id(self) -> str:
        return "iowa_gambling_task"

    def schema(self) -> dict[str, Any]:
        bundle = load_benchmark_bundle(self.benchmark_id())
        manifest = bundle.manifest
        return {
            "benchmark_id": manifest["benchmark_id"],
            "status": manifest["status"],
            "source_type": bundle.source_type,
            "source_label": bundle.source_label,
            "external_bundle_preferred": bundle.external_bundle_preferred,
            "smoke_test_only": bundle.smoke_test_only,
            "benchmark_state": bundle.benchmark_state,
            "available_states": bundle.available_states,
            "blockers": bundle.blockers,
            "trial_fields": manifest["fields"],
            "observation_fields": ["trial_index", "last_outcome", "loss_streak", "resource_pressure"],
            "action_fields": ["choose_A", "choose_B", "choose_C", "choose_D"],
            "feedback_fields": ["reward", "penalty", "net_outcome", "advantageous_choice"],
        }

    def load_trials(self, *, allow_smoke_test: bool = False) -> list[IowaTrial]:
        return [IowaTrial(**payload) for payload in preprocess_iowa_gambling_task(allow_smoke_test=allow_smoke_test)["trials"]]

    def choose_action(
        self,
        trial: IowaTrial,
        parameters: CognitiveStyleParameters,
        *,
        seed: int,
        trial_index: int,
        deck_history: dict[str, float] | None = None,
        last_outcome: float = 0.0,
        loss_streak: int = 0,
    ) -> dict[str, Any]:
        resource_stress = min(0.86, 0.16 + trial_index * 0.04)
        safe_preference = (
            parameters.error_aversion * 0.40
            + parameters.resource_pressure_sensitivity * resource_stress * 0.30
            + parameters.update_rigidity * 0.15
        )
        exploration_drive = parameters.exploration_bias * max(0.0, 0.55 - trial_index * 0.03)
        advantageous_bias = safe_preference - exploration_drive
        history = deck_history or {}
        normalized_last_outcome = _clamp01((last_outcome + 100.0) / 200.0)
        history_signal = {
            deck: float(history.get(deck, 0.0))
            for deck in ("A", "B", "C", "D")
        }
        deck_values = {
            "A": 0.36 - safe_preference * 0.18 + exploration_drive * 0.18 + history_signal["A"] * 0.20,
            "B": 0.35 - safe_preference * 0.16 + exploration_drive * 0.16 + history_signal["B"] * 0.18,
            "C": 0.48 + advantageous_bias * 0.22 + history_signal["C"] * 0.20,
            "D": 0.50 + advantageous_bias * 0.24 + history_signal["D"] * 0.22,
        }
        if loss_streak >= 2:
            deck_values["A"] -= parameters.error_aversion * 0.10
            deck_values["B"] -= parameters.error_aversion * 0.08
            deck_values["C"] += parameters.error_aversion * 0.06
            deck_values["D"] += parameters.error_aversion * 0.08
        deck_values[trial.deck] += normalized_last_outcome * parameters.attention_selectivity * 0.06
        chosen_deck = max(deck_values, key=lambda item: (deck_values[item], item))
        sorted_scores = sorted(deck_values.values(), reverse=True)
        score_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = _clamp01(
            0.42
            + parameters.confidence_gain * 0.16
            + advantageous_bias * 0.14
            + score_margin * 0.28
            - parameters.uncertainty_sensitivity * max(0.0, 0.30 - trial_index * 0.01)
            - resource_stress * 0.05
        )
        predicted_net = {
            "A": -20.0,
            "B": -12.0,
            "C": 18.0,
            "D": 22.0,
        }[chosen_deck] + (parameters.attention_selectivity - 0.5) * 6.0
        return {
            "prediction": {
                "trial_id": trial.trial_id,
                "subject_id": trial.subject_id,
                "split": trial.split,
                "chosen_deck": chosen_deck,
                "actual_deck": trial.deck,
                "expected_value": round(predicted_net, 6),
                "predicted_confidence": confidence,
                "advantageous_choice": chosen_deck in {"C", "D"},
                "actual_advantageous": trial.advantageous,
                "deck_match": chosen_deck == trial.deck,
                "reward": trial.reward,
                "penalty": trial.penalty,
                "net_outcome": trial.net_outcome,
            },
            "resource_state": {},
        }


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        mean_rank = (index + 1 + next_index) / 2.0
        for rank_index in range(index, next_index):
            ranks[indexed[rank_index][0]] = mean_rank
        index = next_index
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    mean_left = mean(left)
    mean_right = mean(right)
    numerator = sum((item - mean_left) * (other - mean_right) for item, other in zip(left, right))
    denom_left = math.sqrt(sum((item - mean_left) ** 2 for item in left))
    denom_right = math.sqrt(sum((item - mean_right) ** 2 for item in right))
    if denom_left == 0.0 or denom_right == 0.0:
        return 0.0
    return numerator / (denom_left * denom_right)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _auroc(labels: list[int], scores: list[float]) -> float:
    paired = [(float(score), int(label)) for label, score in zip(labels, scores)]
    positive_count = sum(1 for _score, label in paired if label == 1)
    negative_count = len(paired) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.5
    ranked = sorted(enumerate(paired), key=lambda item: item[1][0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][1][0] == ranked[index][1][0]:
            next_index += 1
        mean_rank = (index + 1 + next_index) / 2.0
        for rank_index in range(index, next_index):
            if ranked[rank_index][1][1] == 1:
                rank_sum_positive += mean_rank
        index = next_index
    u_statistic = rank_sum_positive - (positive_count * (positive_count + 1) / 2.0)
    return u_statistic / (positive_count * negative_count)


def evaluate_predictions(predictions: list[BenchmarkPrediction | dict[str, Any]]) -> dict[str, float]:
    if predictions and isinstance(predictions[0], dict):
        rows = [dict(item) for item in predictions]
        correctness = [1.0 if bool(row["correct"]) else 0.0 for row in rows]
        confidences = [float(row["predicted_confidence"]) for row in rows]
        human_confidences = [float(row["human_confidence"]) for row in rows]
        heldout_likelihood_values = [
            math.log(
                max(
                    1e-6,
                    float(row["predicted_probability_right"]) if str(row["human_choice"]) == "right" else 1.0 - float(row["predicted_probability_right"]),
                )
            )
            for row in rows
        ]
        subject_ids = sorted({str(row["subject_id"]) for row in rows})
    else:
        rows = [item if isinstance(item, BenchmarkPrediction) else BenchmarkPrediction(**item) for item in predictions]
        correctness = [1.0 if row.correct else 0.0 for row in rows]
        confidences = [float(row.predicted_confidence) for row in rows]
        human_confidences = [float(row.human_confidence) for row in rows]
        heldout_likelihood_values = []
        for row in rows:
            likelihood = row.predicted_probability_right if row.human_choice == "right" else 1.0 - row.predicted_probability_right
            heldout_likelihood_values.append(math.log(max(1e-6, likelihood)))
        subject_ids = sorted({row.subject_id for row in rows})
    brier = mean((conf - corr) ** 2 for conf, corr in zip(confidences, correctness))
    calibration = mean(abs(conf - corr) for conf, corr in zip(confidences, correctness))
    accuracy = mean(correctness)
    confidence_bias = mean(conf - human for conf, human in zip(confidences, human_confidences))
    auroc2 = _auroc([int(value) for value in correctness], confidences)
    meta_ratio = _clamp01((auroc2 / max(0.5, accuracy)) - 0.5)
    confidence_alignment = _spearman(confidences, human_confidences)
    return {
        "accuracy": round(accuracy, 6),
        "calibration_error": round(calibration, 6),
        "brier_score": round(brier, 6),
        "heldout_likelihood": round(mean(heldout_likelihood_values), 6),
        "confidence_bias": round(confidence_bias, 6),
        "auroc2": round(auroc2, 6),
        "meta_d_prime_ratio": round(meta_ratio, 6),
        "confidence_alignment": round(confidence_alignment, 6),
        "subject_count": float(len(subject_ids)),
    }


def evaluate_iowa_predictions(predictions: list[IowaPrediction | dict[str, Any]]) -> dict[str, float]:
    if predictions and isinstance(predictions[0], dict):
        rows = [dict(item) for item in predictions]
        advantageous = [1.0 if bool(row["advantageous_choice"]) else 0.0 for row in rows]
        confidences = [float(row["predicted_confidence"]) for row in rows]
        outcomes = [float(row["net_outcome"]) for row in rows]
        expected_values = [float(row["expected_value"]) for row in rows]
        actual_advantageous = [1.0 if bool(row["actual_advantageous"]) else 0.0 for row in rows]
        deck_match = [1.0 if bool(row["deck_match"]) else 0.0 for row in rows]
        late_rows = [row for row in rows if str(row["split"]) in {"validation", "heldout"} or float(row["net_outcome"]) != 0]
        policy_alignment_rate = mean(
            1.0 if bool(row["advantageous_choice"]) == bool(row["actual_advantageous"]) else 0.0 for row in rows
        ) if rows else 0.0
        late_advantageous_rate = mean(
            1.0 if bool(row["advantageous_choice"]) else 0.0 for row in late_rows[-max(1, len(late_rows) // 2) :]
        ) if late_rows else 0.0
    else:
        rows = [item if isinstance(item, IowaPrediction) else IowaPrediction(**item) for item in predictions]
        advantageous = [1.0 if row.advantageous_choice else 0.0 for row in rows]
        confidences = [float(row.predicted_confidence) for row in rows]
        outcomes = [float(row.net_outcome) for row in rows]
        expected_values = [float(row.expected_value) for row in rows]
        actual_advantageous = [1.0 if row.actual_advantageous else 0.0 for row in rows]
        deck_match = [1.0 if row.deck_match else 0.0 for row in rows]
        late_rows = [row for row in rows if row.split in {"validation", "heldout"} or row.net_outcome != 0]
        policy_alignment_rate = mean(
            1.0 if row.advantageous_choice == row.actual_advantageous else 0.0 for row in rows
        ) if rows else 0.0
        late_advantageous_rate = mean(
            1.0 if row.advantageous_choice else 0.0 for row in late_rows[-max(1, len(late_rows) // 2) :]
        ) if late_rows else 0.0
    advantageous_rate = mean(advantageous) if advantageous else 0.0
    mean_net = mean(outcomes) if outcomes else 0.0
    reward_fit = -mean((pred - actual) ** 2 for pred, actual in zip(expected_values, outcomes)) if outcomes else 0.0
    confidence_advantage_alignment = _spearman(confidences, actual_advantageous)
    deck_match_rate = mean(deck_match) if deck_match else 0.0
    return {
        "advantageous_choice_rate": round(advantageous_rate, 6),
        "mean_net_outcome": round(mean_net, 6),
        "reward_fit": round(reward_fit, 6),
        "confidence_advantage_alignment": round(confidence_advantage_alignment, 6),
        "deck_match_rate": round(deck_match_rate, 6),
        "policy_alignment_rate": round(policy_alignment_rate, 6),
        "late_advantageous_rate": round(late_advantageous_rate, 6),
    }


def summarize_confidence_predictions(predictions: list[BenchmarkPrediction | dict[str, Any]]) -> dict[str, Any]:
    if predictions and isinstance(predictions[0], dict):
        rows = [dict(item) for item in predictions]
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_subject.setdefault(str(row["subject_id"]), []).append(row)
    else:
        rows = [item if isinstance(item, BenchmarkPrediction) else BenchmarkPrediction(**item) for item in predictions]
        by_subject: dict[str, list[BenchmarkPrediction]] = {}
        for row in rows:
            by_subject.setdefault(row.subject_id, []).append(row)
    subject_metrics = {subject_id: evaluate_predictions(subject_rows) for subject_id, subject_rows in by_subject.items()}
    heldout_values = [float(metrics["heldout_likelihood"]) for metrics in subject_metrics.values()]
    calibration_values = [float(metrics["calibration_error"]) for metrics in subject_metrics.values()]
    return {
        "subjects": subject_metrics,
        "subject_count": len(subject_metrics),
        "heldout_likelihood_floor": round(min(heldout_values), 6) if heldout_values else 0.0,
        "heldout_likelihood_mean": round(mean(heldout_values), 6) if heldout_values else 0.0,
        "calibration_ceiling": round(max(calibration_values), 6) if calibration_values else 0.0,
    }


def summarize_iowa_predictions(predictions: list[IowaPrediction | dict[str, Any]]) -> dict[str, Any]:
    if predictions and isinstance(predictions[0], dict):
        rows = [dict(item) for item in predictions]
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_subject.setdefault(str(row["subject_id"]), []).append(row)
    else:
        rows = [item if isinstance(item, IowaPrediction) else IowaPrediction(**item) for item in predictions]
        by_subject: dict[str, list[IowaPrediction]] = {}
        for row in rows:
            by_subject.setdefault(row.subject_id, []).append(row)
    subject_metrics = {subject_id: evaluate_iowa_predictions(subject_rows) for subject_id, subject_rows in by_subject.items()}
    deck_match_values = [float(metrics["deck_match_rate"]) for metrics in subject_metrics.values()]
    policy_alignment_values = [float(metrics["policy_alignment_rate"]) for metrics in subject_metrics.values()]
    return {
        "subjects": subject_metrics,
        "subject_count": len(subject_metrics),
        "deck_match_floor": round(min(deck_match_values), 6) if deck_match_values else 0.0,
        "policy_alignment_mean": round(mean(policy_alignment_values), 6) if policy_alignment_values else 0.0,
    }


def run_confidence_database_benchmark(
    parameters: CognitiveStyleParameters | None = None,
    *,
    seed: int = 42,
    split: str | None = None,
    malformed_rows: list[dict[str, Any]] | None = None,
    allow_smoke_test: bool = False,
    include_subject_summary: bool = True,
    include_predictions: bool = True,
    max_trials: int | None = None,
) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    preprocess_payload = preprocess_confidence_database(allow_smoke_test=allow_smoke_test, selected_split=split)
    trials = [BenchmarkTrial(**payload) for payload in preprocess_payload["trials"]]
    skipped_malformed = 0
    if malformed_rows:
        required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
        for row in malformed_rows:
            if not required <= row.keys():
                skipped_malformed += 1
    selected_trials = trials if split is not None else list(trials)
    if max_trials is not None:
        selected_trials = selected_trials[: max(0, int(max_trials))]
    active_parameters = parameters or CognitiveStyleParameters()
    predictions: list[dict[str, Any]] = []
    for index, trial in enumerate(selected_trials):
        predictions.append(adapter.choose_action(trial, active_parameters, seed=seed, trial_index=index)["prediction"])
    metrics = evaluate_predictions(predictions)
    subject_summary = summarize_confidence_predictions(predictions) if include_subject_summary else {"subjects": {}, "subject_count": 0}
    return {
        "benchmark_id": adapter.benchmark_id(),
        "parameters": active_parameters.to_dict(),
        "trial_count": len(selected_trials),
        "split": split or "all",
        "bundle": preprocess_payload["bundle"],
        "benchmark_status": preprocess_payload["benchmark_status"],
        "bundle_mode": preprocess_payload["bundle_mode"],
        "claim_envelope": preprocess_payload["claim_envelope"],
        "split_unit": preprocess_payload["split_unit"],
        "leakage_check": preprocess_payload["leakage_check"],
        "metrics": metrics,
        "predictions": predictions if include_predictions else [],
        "subject_summary": subject_summary,
        "resources": [],
        "schema": adapter.schema(),
        "skipped_malformed": skipped_malformed,
    }


def run_iowa_gambling_benchmark(
    parameters: CognitiveStyleParameters | None = None,
    *,
    seed: int = 44,
    split: str | None = None,
    allow_smoke_test: bool = False,
    include_subject_summary: bool = True,
    include_predictions: bool = True,
    max_trials: int | None = None,
) -> dict[str, Any]:
    adapter = IowaGamblingTaskAdapter()
    preprocess_payload = preprocess_iowa_gambling_task(allow_smoke_test=allow_smoke_test)
    trials = [IowaTrial(**payload) for payload in preprocess_payload["trials"]]
    selected_trials = [trial for trial in trials if split is None or trial.split == split]
    if max_trials is not None:
        selected_trials = selected_trials[: max(0, int(max_trials))]
    active_parameters = parameters or CognitiveStyleParameters()
    subject_state: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for index, trial in enumerate(selected_trials):
        state = subject_state.setdefault(
            trial.subject_id,
            {
                "deck_history": {deck: 0.0 for deck in ("A", "B", "C", "D")},
                "last_outcome": 0.0,
                "loss_streak": 0,
            },
        )
        choice_row = adapter.choose_action(
            trial,
            active_parameters,
            seed=seed,
            trial_index=index,
            deck_history=state["deck_history"],
            last_outcome=state["last_outcome"],
            loss_streak=state["loss_streak"],
        )
        predictions.append(choice_row["prediction"])
        learning_rate = 0.22 + active_parameters.confidence_gain * 0.18 - active_parameters.update_rigidity * 0.10
        normalized_outcome = max(-1.0, min(1.0, float(trial.net_outcome) / 100.0))
        prior = float(state["deck_history"][trial.deck])
        state["deck_history"][trial.deck] = round(prior * (1.0 - learning_rate) + normalized_outcome * learning_rate, 6)
        state["last_outcome"] = float(trial.net_outcome)
        state["loss_streak"] = int(state["loss_streak"]) + 1 if trial.net_outcome < 0 else 0
    metrics = evaluate_iowa_predictions(predictions)
    subject_summary = summarize_iowa_predictions(predictions) if include_subject_summary else {"subjects": {}, "subject_count": 0}
    return {
        "benchmark_id": adapter.benchmark_id(),
        "parameters": active_parameters.to_dict(),
        "trial_count": len(selected_trials),
        "split": split or "all",
        "bundle": preprocess_payload["bundle"],
        "benchmark_status": preprocess_payload["benchmark_status"],
        "bundle_mode": preprocess_payload["bundle_mode"],
        "claim_envelope": preprocess_payload["claim_envelope"],
        "metrics": metrics,
        "predictions": predictions if include_predictions else [],
        "subject_summary": subject_summary,
        "resources": [],
        "schema": adapter.schema(),
    }
