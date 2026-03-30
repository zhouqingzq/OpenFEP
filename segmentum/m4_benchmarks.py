from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from .action_schema import ActionSchema
from .m4_cognitive_style import CognitiveStyleParameters, CognitiveParameterBridge, ResourceSnapshot, logistic

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_DATA_PATH = ROOT / "data" / "benchmarks" / "confidence_database" / "confidence_database_sample.jsonl"
CONFIDENCE_MANIFEST_PATH = ROOT / "data" / "benchmarks" / "confidence_database" / "manifest.json"
IGT_MANIFEST_PATH = ROOT / "data" / "benchmarks" / "iowa_gambling_task" / "igt_placeholder_manifest.json"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class BenchmarkTrial:
    trial_id: str
    subject_id: str
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


class BenchmarkAdapter(Protocol):
    def benchmark_id(self) -> str: ...
    def schema(self) -> dict[str, Any]: ...


def preprocess_confidence_database() -> dict[str, Any]:
    raw_records = [json.loads(line) for line in CONFIDENCE_DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    processed: list[BenchmarkTrial] = []
    for index, record in enumerate(raw_records):
        required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
        if not required <= record.keys():
            continue
        split = "train" if index < 6 else "validation" if index < 9 else "heldout"
        processed.append(
            BenchmarkTrial(
                trial_id=str(record["trial_id"]),
                subject_id=str(record["subject_id"]),
                stimulus_strength=float(record["stimulus_strength"]),
                correct_choice=str(record["correct_choice"]),
                human_choice=str(record["human_choice"]),
                human_confidence=_clamp01(float(record["human_confidence"])),
                rt_ms=int(record["rt_ms"]),
                split=split,
            )
        )
    manifest = json.loads(CONFIDENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "trials": [trial.to_dict() for trial in processed],
        "skipped_records": len(raw_records) - len(processed),
    }


class ConfidenceDatabaseAdapter:
    def benchmark_id(self) -> str:
        return "confidence_database"

    def schema(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id(),
            "trial_fields": list(BenchmarkTrial.__dataclass_fields__.keys()),
            "observation_fields": ["stimulus_strength", "ambiguity", "evidence_strength"],
            "action_fields": ["choose_left", "choose_right"],
            "feedback_fields": ["correct", "human_choice_match", "predicted_confidence"],
        }

    def load_trials(self) -> list[BenchmarkTrial]:
        return [BenchmarkTrial(**payload) for payload in preprocess_confidence_database()["trials"]]

    def choose_action(
        self,
        trial: BenchmarkTrial,
        parameters: CognitiveStyleParameters,
        *,
        seed: int,
        trial_index: int,
    ) -> dict[str, Any]:
        resource_state = ResourceSnapshot(
            energy=max(0.25, 0.82 - trial_index * 0.04),
            budget=max(0.30, 0.88 - trial_index * 0.03),
            stress=min(0.80, 0.18 + trial_index * 0.05),
            time_remaining=max(0.22, 0.95 - trial_index * 0.04),
        )
        evidence_strength = abs(float(trial.stimulus_strength))
        uncertainty = 1.0 - evidence_strength
        expected_error = uncertainty * (0.55 + parameters.error_aversion * 0.25)
        bridge = CognitiveParameterBridge(parameters)
        right_action = ActionSchema(name="choose_right", cost_estimate=0.12, resource_cost={"tokens": 0.03})
        left_action = ActionSchema(name="choose_left", cost_estimate=0.12, resource_cost={"tokens": 0.03})
        bridge.decide(
            tick=trial_index + 1,
            seed=seed,
            task_context={"benchmark": self.benchmark_id(), "trial_id": trial.trial_id},
            observation_evidence={
                "evidence_strength": evidence_strength,
                "uncertainty": uncertainty,
                "expected_error": expected_error,
            },
            actions=[left_action, right_action],
            resource_state=resource_state,
        )
        signed_strength = float(trial.stimulus_strength)
        bias = (parameters.exploration_bias - 0.5) * 0.4
        caution = parameters.error_aversion * resource_state.stress * 0.15
        p_right = logistic(signed_strength * 4.2 + bias - caution)
        predicted_choice = "right" if p_right >= 0.5 else "left"
        confidence = _clamp01(
            0.50
            + evidence_strength * (0.28 + parameters.confidence_gain * 0.28)
            - uncertainty * parameters.uncertainty_sensitivity * 0.15
            - resource_state.stress * parameters.resource_pressure_sensitivity * 0.08
        )
        return {
            "prediction": BenchmarkPrediction(
                trial_id=trial.trial_id,
                split=trial.split,
                human_choice=trial.human_choice,
                predicted_choice=predicted_choice,
                predicted_confidence=confidence,
                predicted_probability_right=round(p_right, 6),
                correct=predicted_choice == trial.correct_choice,
                human_choice_match=predicted_choice == trial.human_choice,
                human_confidence=trial.human_confidence,
            ).to_dict(),
            "resource_state": resource_state.to_dict(),
        }


class IowaGamblingTaskAdapter:
    def benchmark_id(self) -> str:
        return "iowa_gambling_task"

    def schema(self) -> dict[str, Any]:
        manifest = json.loads(IGT_MANIFEST_PATH.read_text(encoding="utf-8"))
        return {
            "benchmark_id": manifest["benchmark_id"],
            "status": manifest["status"],
            "trial_fields": manifest["fields"],
        }


def evaluate_predictions(predictions: list[BenchmarkPrediction | dict[str, Any]]) -> dict[str, float]:
    rows = [item if isinstance(item, BenchmarkPrediction) else BenchmarkPrediction(**item) for item in predictions]
    correctness = [1.0 if row.correct else 0.0 for row in rows]
    confidences = [float(row.predicted_confidence) for row in rows]
    human_confidences = [float(row.human_confidence) for row in rows]
    brier = mean((conf - corr) ** 2 for conf, corr in zip(confidences, correctness))
    calibration = mean(abs(conf - corr) for conf, corr in zip(confidences, correctness))
    eps = 1e-6
    heldout_likelihood_values: list[float] = []
    for row in rows:
        likelihood = row.predicted_probability_right if row.human_choice == "right" else 1.0 - row.predicted_probability_right
        heldout_likelihood_values.append(math.log(max(eps, likelihood)))
    accuracy = mean(correctness)
    confidence_bias = mean(conf - human for conf, human in zip(confidences, human_confidences))
    return {
        "accuracy": round(accuracy, 6),
        "calibration_error": round(calibration, 6),
        "brier_score": round(brier, 6),
        "heldout_likelihood": round(mean(heldout_likelihood_values), 6),
        "confidence_bias": round(confidence_bias, 6),
    }


def run_confidence_database_benchmark(
    parameters: CognitiveStyleParameters | None = None,
    *,
    seed: int = 42,
    split: str | None = None,
    malformed_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    trials = adapter.load_trials()
    skipped_malformed = 0
    if malformed_rows:
        required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
        for row in malformed_rows:
            if not required <= row.keys():
                skipped_malformed += 1
    selected_trials = [trial for trial in trials if split is None or trial.split == split]
    active_parameters = parameters or CognitiveStyleParameters()
    run_rows = [adapter.choose_action(trial, active_parameters, seed=seed, trial_index=index) for index, trial in enumerate(selected_trials)]
    predictions = [row["prediction"] for row in run_rows]
    metrics = evaluate_predictions(predictions)
    return {
        "benchmark_id": adapter.benchmark_id(),
        "parameters": active_parameters.to_dict(),
        "trial_count": len(selected_trials),
        "split": split or "all",
        "metrics": metrics,
        "predictions": predictions,
        "resources": [row["resource_state"] for row in run_rows],
        "schema": adapter.schema(),
        "skipped_malformed": skipped_malformed,
    }
