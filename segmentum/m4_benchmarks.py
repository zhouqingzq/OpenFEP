from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from statistics import mean, pvariance
from typing import Any, Protocol

from .benchmark_registry import benchmark_status, load_benchmark_bundle, validate_benchmark_bundle
from .m4_cognitive_style import CognitiveStyleParameters, logistic


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_BENCHMARK_ROOT = ROOT / "external_benchmark_registry"
STANDARD_IGT_TRIAL_COUNT = 100
DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET = "Siedlecka_unpub"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: list[float], default: float = 0.0) -> float:
    return float(mean(values)) if values else float(default)


def default_acceptance_benchmark_root() -> Path | None:
    if (EXTERNAL_BENCHMARK_ROOT / "confidence_database" / "manifest.json").exists() and (
        EXTERNAL_BENCHMARK_ROOT / "iowa_gambling_task" / "manifest.json"
    ).exists():
        return EXTERNAL_BENCHMARK_ROOT
    return None


def _resolve_root(benchmark_root: Path | str | None = None) -> Path | None:
    if benchmark_root is None:
        return None
    return Path(benchmark_root).resolve()


def _active_bundle_root(*, benchmark_root: Path | str | None = None, prefer_external: bool = False) -> Path | None:
    explicit = _resolve_root(benchmark_root)
    if explicit is not None:
        return explicit
    if prefer_external:
        return default_acceptance_benchmark_root()
    return None


def _load_bundle(benchmark_id: str, *, benchmark_root: Path | str | None = None, prefer_external: bool = False):
    root = _active_bundle_root(benchmark_root=benchmark_root, prefer_external=prefer_external)
    return load_benchmark_bundle(benchmark_id, root=root)


def _validate_bundle(benchmark_id: str, *, benchmark_root: Path | str | None = None, prefer_external: bool = False):
    root = _active_bundle_root(benchmark_root=benchmark_root, prefer_external=prefer_external)
    return validate_benchmark_bundle(benchmark_id, root=root)


def _benchmark_status(benchmark_id: str, *, benchmark_root: Path | str | None = None, prefer_external: bool = False):
    root = _active_bundle_root(benchmark_root=benchmark_root, prefer_external=prefer_external)
    return benchmark_status(benchmark_id, root=root)


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            rows.append(json.loads(raw_line))
    return rows


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
    source_dataset: str = ""
    source_file: str = ""

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
    source_file: str = ""

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


@dataclass(frozen=True)
class BanditTrial:
    trial_id: str
    subject_id: str
    session_id: str
    split: str
    trial_index: int
    arm_reward_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(Protocol):
    def benchmark_id(self) -> str: ...
    def task_id(self) -> str: ...
    def schema(self) -> dict[str, Any]: ...
    def load_trials(self, **kwargs: Any) -> list[Any]: ...
    def validate_protocol(self, trials: list[Any], **kwargs: Any) -> dict[str, Any]: ...
    def initial_state(self, *, subject_id: str, parameters: CognitiveStyleParameters) -> dict[str, Any]: ...
    def observation_from_trial(self, trial: Any, *, state: dict[str, Any], parameters: CognitiveStyleParameters, trial_index: int) -> dict[str, Any]: ...
    def action_space(self, trial: Any, *, observation: dict[str, Any], state: dict[str, Any], parameters: CognitiveStyleParameters) -> list[dict[str, Any]]: ...
    def apply_action(
        self,
        trial: Any,
        *,
        chosen_action: str,
        confidence: float,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        observation: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]: ...
    def export_trial_record(
        self,
        trial: Any,
        *,
        chosen_action: str,
        confidence: float,
        observation: dict[str, Any],
        transition: dict[str, Any],
        state: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]: ...
    def legacy_prediction(self, trial_record: dict[str, Any]) -> dict[str, Any]: ...
    def metrics(self, records: list[dict[str, Any]]) -> dict[str, float]: ...
    def subject_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]: ...


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
        assignments[group_id] = split_names[min(index, len(split_names) - 1)] if index < len(split_names) else split_names[index % len(split_names)]
    return assignments


def detect_subject_leakage(rows: list[BenchmarkTrial | dict[str, Any]], *, key_field: str) -> dict[str, Any]:
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        payload = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        key = str(payload.get(key_field, "")).strip()
        if not key:
            continue
        observed[key].add(str(payload.get("split", "")))
    leaking_keys = sorted(key for key, splits in observed.items() if len(splits) > 1)
    return {"key_field": key_field, "ok": not leaking_keys, "leaking_keys": leaking_keys}


def _confidence_export_schema() -> dict[str, Any]:
    return {
        "schema_id": "m42.confidence.human_aligned_export.v1",
        "required": [
            "trial_id",
            "subject_id",
            "session_id",
            "split",
            "stimulus_strength",
            "evidence_strength",
            "agent_choice",
            "agent_confidence_rating",
            "correct_choice",
            "human_choice",
            "human_confidence",
        ],
        "field_types": {
            "trial_id": "str",
            "subject_id": "str",
            "session_id": "str",
            "split": "str",
            "stimulus_strength": "float",
            "evidence_strength": "float",
            "agent_choice": "str",
            "agent_confidence_rating": "float",
            "correct_choice": "str",
            "human_choice": "str",
            "human_confidence": "float",
            "source_dataset": "str",
            "source_file": "str",
        },
    }


def _igt_export_schema() -> dict[str, Any]:
    return {
        "schema_id": "m42.igt.trial_trace.v1",
        "required": [
            "trial_id",
            "subject_id",
            "split",
            "trial_index",
            "chosen_deck",
            "reward",
            "penalty",
            "net_outcome",
            "cumulative_gain",
            "internal_state_snapshot",
        ],
        "field_types": {
            "trial_id": "str",
            "subject_id": "str",
            "split": "str",
            "trial_index": "int",
            "chosen_deck": "str",
            "reward": "int",
            "penalty": "int",
            "net_outcome": "int",
            "cumulative_gain": "int",
            "internal_state_snapshot": "dict",
            "human_deck": "str",
            "human_reward": "int",
            "human_penalty": "int",
            "human_net_outcome": "int",
        },
    }


def _bandit_export_schema() -> dict[str, Any]:
    return {
        "schema_id": "m42.two_armed_bandit.trial_trace.v1",
        "required": [
            "trial_id",
            "subject_id",
            "session_id",
            "split",
            "trial_index",
            "chosen_arm",
            "reward",
            "cumulative_reward",
            "internal_state_snapshot",
        ],
        "field_types": {
            "trial_id": "str",
            "subject_id": "str",
            "session_id": "str",
            "split": "str",
            "trial_index": "int",
            "chosen_arm": "str",
            "reward": "int",
            "cumulative_reward": "int",
            "internal_state_snapshot": "dict",
        },
    }


def validate_trial_export_records(records: list[dict[str, Any]], schema: dict[str, Any], *, expected_length: int) -> dict[str, Any]:
    required = list(schema.get("required", []))
    field_types = dict(schema.get("field_types", {}))
    missing_fields: dict[str, int] = {field_name: 0 for field_name in required}
    wrong_types: dict[str, int] = {field_name: 0 for field_name in field_types}
    for record in records:
        for field_name in required:
            value = record.get(field_name)
            if value in (None, ""):
                missing_fields[field_name] += 1
        for field_name, expected_type in field_types.items():
            value = record.get(field_name)
            if value is None:
                continue
            if expected_type == "str" and not isinstance(value, str):
                wrong_types[field_name] += 1
            elif expected_type == "float" and not isinstance(value, (int, float)):
                wrong_types[field_name] += 1
            elif expected_type == "int" and not isinstance(value, int):
                wrong_types[field_name] += 1
            elif expected_type == "dict" and not isinstance(value, dict):
                wrong_types[field_name] += 1
    return {
        "expected_length": expected_length,
        "observed_length": len(records),
        "length_matches": len(records) == expected_length,
        "missing_fields": missing_fields,
        "wrong_types": wrong_types,
        "ok": len(records) == expected_length and all(count == 0 for count in missing_fields.values()) and all(count == 0 for count in wrong_types.values()),
    }


def preprocess_confidence_database(
    *,
    allow_smoke_test: bool = False,
    selected_split: str | None = None,
    benchmark_root: Path | str | None = None,
    prefer_external: bool = False,
    selected_source_dataset: str | None = None,
    max_trials: int | None = None,
) -> dict[str, Any]:
    bundle = _load_bundle("confidence_database", benchmark_root=benchmark_root, prefer_external=prefer_external)
    validation = _validate_bundle("confidence_database", benchmark_root=benchmark_root, prefer_external=prefer_external)
    status = _benchmark_status("confidence_database", benchmark_root=benchmark_root, prefer_external=prefer_external)
    if not validation.ok:
        raise ValueError(f"Confidence benchmark bundle is invalid: {validation.to_dict()}")
    if bundle.external_bundle_preferred and bundle.source_type != "external_bundle" and not allow_smoke_test:
        blocker = status.blockers[0] if status.blockers else "Acceptance-grade evaluation requires an external bundle."
        raise ValueError(f"{blocker} Repo sample remains available only for smoke tests.")
    raw_records = _iter_jsonl(bundle.data_path)
    processed: list[BenchmarkTrial] = []
    valid_records: list[dict[str, Any]] = []
    grouping_fields = list(bundle.grouping_fields or ["session_id", "subject_id"])
    active_split_unit = "subject_id"
    precomputed_split_available = False
    required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
    for record in raw_records:
        if not required <= record.keys():
            continue
        row = dict(record)
        row.setdefault("session_id", str(row.get("subject_id", "")))
        row.setdefault("source_dataset", str(row.get("source_dataset", "")))
        row.setdefault("source_file", str(row.get("source_file", "")))
        if selected_source_dataset is not None and str(row.get("source_dataset", "")) != selected_source_dataset:
            continue
        if "split" in row and str(row.get("split", "")).strip():
            precomputed_split_available = True
        active_split_unit = _active_grouping_field(row, grouping_fields)
        valid_records.append(row)
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
                source_dataset=str(record.get("source_dataset", "")),
                source_file=str(record.get("source_file", "")),
            )
        )
        if max_trials is not None and len(processed) >= max(0, int(max_trials)):
            break
    leakage_check = {
        "split_unit": active_split_unit,
        "subject": detect_subject_leakage(processed, key_field="subject_id"),
        "session": detect_subject_leakage(processed, key_field="session_id"),
        "precomputed_split_available": precomputed_split_available,
        "selected_split": selected_split,
    }
    return {
        "manifest": dict(bundle.manifest),
        "trials": [trial.to_dict() for trial in processed],
        "skipped_records": len(raw_records) - len(valid_records),
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "split_unit": active_split_unit,
        "bundle_mode": "external_bundle" if bundle.source_type == "external_bundle" else "repo_smoke_test",
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "smoke_only",
        "leakage_check": leakage_check,
        "trial_count": len(processed),
        "subject_count": len({trial.subject_id for trial in processed}),
        "source_dataset": selected_source_dataset,
        "trial_export_schema": _confidence_export_schema(),
    }


def preprocess_iowa_gambling_task(
    *,
    allow_smoke_test: bool = False,
    benchmark_root: Path | str | None = None,
    prefer_external: bool = False,
    selected_subject_id: str | None = None,
    selected_split: str | None = None,
    max_trials: int | None = None,
) -> dict[str, Any]:
    bundle = _load_bundle("iowa_gambling_task", benchmark_root=benchmark_root, prefer_external=prefer_external)
    validation = _validate_bundle("iowa_gambling_task", benchmark_root=benchmark_root, prefer_external=prefer_external)
    status = _benchmark_status("iowa_gambling_task", benchmark_root=benchmark_root, prefer_external=prefer_external)
    if not validation.ok:
        raise ValueError(f"IGT benchmark bundle is invalid: {validation.to_dict()}")
    if bundle.external_bundle_preferred and bundle.source_type != "external_bundle" and not allow_smoke_test:
        blocker = status.blockers[0] if status.blockers else "Acceptance-grade IGT evaluation requires an external bundle."
        raise ValueError(f"{blocker} Repo sample remains available only for smoke tests.")
    raw_records = _iter_jsonl(bundle.data_path)
    processed: list[IowaTrial] = []
    required = {"trial_id", "subject_id", "deck", "reward", "penalty", "net_outcome", "advantageous", "trial_index", "split"}
    for record in raw_records:
        if not required <= record.keys():
            continue
        if selected_subject_id is not None and str(record["subject_id"]) != selected_subject_id:
            continue
        if selected_split is not None and str(record["split"]) != selected_split:
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
                source_file=str(record.get("source_file", "")),
            )
        )
        if max_trials is not None and len(processed) >= max(0, int(max_trials)):
            break
    return {
        "manifest": dict(bundle.manifest),
        "trials": [trial.to_dict() for trial in processed],
        "skipped_records": len(raw_records) - len(processed),
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "benchmark_status": status.to_dict(),
        "bundle_mode": "external_bundle" if bundle.source_type == "external_bundle" else "repo_smoke_test",
        "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "smoke_only",
        "trial_count": len(processed),
        "subject_count": len({trial.subject_id for trial in processed}),
        "trial_export_schema": _igt_export_schema(),
    }


def _score_action_candidates(
    *,
    observation: dict[str, Any],
    candidates: list[dict[str, Any]],
    parameters: CognitiveStyleParameters,
    rng: random.Random,
    state: dict[str, Any],
) -> dict[str, Any]:
    uncertainty = float(observation.get("uncertainty", 0.0))
    resource_pressure = float(observation.get("resource_pressure", 0.0))
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        novelty = float(candidate.get("novelty_bonus", 0.0))
        evidence_alignment = float(candidate.get("evidence_alignment", 0.0))
        expected_value = float(candidate.get("expected_value", 0.0))
        risk = float(candidate.get("risk_penalty", 0.0))
        habit = float(candidate.get("habit_value", 0.0))
        noise_scale = float(candidate.get("noise_scale", 0.015 + max(0.0, parameters.exploration_bias - 0.4) * 0.10))
        score = (
            expected_value * (0.50 + parameters.attention_selectivity * 0.30)
            + evidence_alignment * (0.65 + parameters.confidence_gain * 0.25)
            + habit * (0.20 + parameters.update_rigidity * 0.20)
            + novelty * (0.15 + parameters.exploration_bias * 0.40)
            - risk * (0.25 + parameters.error_aversion * 0.70)
            - uncertainty * (0.15 + parameters.uncertainty_sensitivity * 0.55)
            - resource_pressure * parameters.resource_pressure_sensitivity * 0.40
            + rng.gauss(0.0, noise_scale)
        )
        row = dict(candidate)
        row["score"] = score
        scored.append(row)
    scored.sort(key=lambda item: (-float(item["score"]), str(item["action_id"])))
    best = scored[0]
    margin = float(best["score"]) - float(scored[1]["score"]) if len(scored) > 1 else abs(float(best["score"]))
    evidence_strength = abs(float(observation.get("evidence_strength", 0.0)))
    confidence = _clamp01(
        0.38
        + logistic(margin * 2.2) * (0.24 + parameters.confidence_gain * 0.18)
        + evidence_strength * 0.22
        - uncertainty * parameters.uncertainty_sensitivity * 0.18
    )
    return {
        "chosen_action": str(best["action_id"]),
        "confidence": _safe_round(confidence),
        "scored_candidates": [
            {
                "action_id": str(item["action_id"]),
                "score": _safe_round(float(item["score"])),
                "expected_value": _safe_round(float(item.get("expected_value", 0.0))),
                "risk_penalty": _safe_round(float(item.get("risk_penalty", 0.0))),
                "novelty_bonus": _safe_round(float(item.get("novelty_bonus", 0.0))),
            }
            for item in scored
        ],
    }


class ConfidenceDatabaseAdapter:
    def benchmark_id(self) -> str:
        return "confidence_database"

    def task_id(self) -> str:
        return "confidence_database"

    def schema(self) -> dict[str, Any]:
        bundle = _load_bundle(self.benchmark_id(), prefer_external=False)
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
            "trial_export_schema": _confidence_export_schema(),
            "observation_fields": ["stimulus_strength", "evidence_strength", "uncertainty", "resource_pressure"],
            "action_fields": ["left", "right"],
        }

    def load_trials(
        self,
        *,
        allow_smoke_test: bool = False,
        benchmark_root: Path | str | None = None,
        prefer_external: bool = False,
        selected_split: str | None = None,
        selected_source_dataset: str | None = None,
        max_trials: int | None = None,
    ) -> list[BenchmarkTrial]:
        payload = preprocess_confidence_database(
            allow_smoke_test=allow_smoke_test,
            selected_split=selected_split,
            benchmark_root=benchmark_root,
            prefer_external=prefer_external,
            selected_source_dataset=selected_source_dataset,
            max_trials=max_trials,
        )
        return [BenchmarkTrial(**row) for row in payload["trials"]]

    def choose_action(
        self,
        trial: BenchmarkTrial,
        parameters: CognitiveStyleParameters,
        *,
        seed: int,
        trial_index: int,
    ) -> dict[str, Any]:
        state = self.initial_state(subject_id=trial.subject_id, parameters=parameters)
        state["_run_seed"] = seed
        observation = self.observation_from_trial(trial, state=state, parameters=parameters, trial_index=trial_index)
        decision = _score_action_candidates(
            observation=observation,
            candidates=self.action_space(trial, observation=observation, state=state, parameters=parameters),
            parameters=parameters,
            rng=random.Random(seed + trial_index),
            state=state,
        )
        transition = self.apply_action(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            state=state,
            parameters=parameters,
            observation=observation,
            global_trial_index=trial_index,
        )
        record = self.export_trial_record(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            observation=observation,
            transition=transition,
            state=state,
            global_trial_index=trial_index,
        )
        return {"prediction": self.legacy_prediction(record), "resource_state": {}}

    def validate_protocol(self, trials: list[BenchmarkTrial], **_: Any) -> dict[str, Any]:
        return {
            "protocol_mode": "dataset_aligned",
            "trial_count": len(trials),
            "subject_count": len({trial.subject_id for trial in trials}),
            "ok": bool(trials),
        }

    def initial_state(self, *, subject_id: str, parameters: CognitiveStyleParameters) -> dict[str, Any]:
        return {"subject_id": subject_id, "last_choice": "", "last_confidence": 0.5}

    def observation_from_trial(
        self,
        trial: BenchmarkTrial,
        *,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        trial_index: int,
    ) -> dict[str, Any]:
        evidence_strength = abs(float(trial.stimulus_strength))
        return {
            "stimulus_strength": float(trial.stimulus_strength),
            "evidence_strength": evidence_strength,
            "uncertainty": 1.0 - evidence_strength,
            "resource_pressure": min(0.85, 0.10 + trial_index * 0.0025),
        }

    def action_space(
        self,
        trial: BenchmarkTrial,
        *,
        observation: dict[str, Any],
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
    ) -> list[dict[str, Any]]:
        signed_strength = float(trial.stimulus_strength)
        last_choice = str(state.get("last_choice", ""))
        return [
            {
                "action_id": "left",
                "expected_value": -signed_strength,
                "evidence_alignment": -signed_strength,
                "risk_penalty": max(0.0, signed_strength),
                "habit_value": 0.08 if last_choice == "left" else -0.02,
                "novelty_bonus": 0.02 if last_choice != "left" else 0.0,
                "noise_scale": 0.02 + observation["uncertainty"] * 0.08,
            },
            {
                "action_id": "right",
                "expected_value": signed_strength,
                "evidence_alignment": signed_strength,
                "risk_penalty": max(0.0, -signed_strength),
                "habit_value": 0.08 if last_choice == "right" else -0.02,
                "novelty_bonus": 0.02 if last_choice != "right" else 0.0,
                "noise_scale": 0.02 + observation["uncertainty"] * 0.08,
            },
        ]

    def apply_action(
        self,
        trial: BenchmarkTrial,
        *,
        chosen_action: str,
        confidence: float,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        observation: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        state["last_choice"] = chosen_action
        state["last_confidence"] = confidence
        return {
            "predicted_probability_right": _safe_round(logistic(float(trial.stimulus_strength) * 4.0)),
            "correct": chosen_action == trial.correct_choice,
            "human_choice_match": chosen_action == trial.human_choice,
            "human_confidence": float(trial.human_confidence),
        }

    def export_trial_record(
        self,
        trial: BenchmarkTrial,
        *,
        chosen_action: str,
        confidence: float,
        observation: dict[str, Any],
        transition: dict[str, Any],
        state: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "subject_id": trial.subject_id,
            "session_id": trial.session_id,
            "split": trial.split,
            "stimulus_strength": _safe_round(trial.stimulus_strength),
            "evidence_strength": _safe_round(observation["evidence_strength"]),
            "uncertainty": _safe_round(observation["uncertainty"]),
            "agent_choice": chosen_action,
            "agent_confidence_rating": _safe_round(confidence),
            "correct_choice": trial.correct_choice,
            "human_choice": trial.human_choice,
            "human_confidence": _safe_round(trial.human_confidence),
            "predicted_choice": chosen_action,
            "predicted_confidence": _safe_round(confidence),
            "predicted_probability_right": transition["predicted_probability_right"],
            "correct": bool(transition["correct"]),
            "human_choice_match": bool(transition["human_choice_match"]),
            "rt_ms": int(trial.rt_ms),
            "source_dataset": trial.source_dataset,
            "source_file": trial.source_file,
        }

    def legacy_prediction(self, trial_record: dict[str, Any]) -> dict[str, Any]:
        return {
            "trial_id": trial_record["trial_id"],
            "subject_id": trial_record["subject_id"],
            "session_id": trial_record["session_id"],
            "split": trial_record["split"],
            "human_choice": trial_record["human_choice"],
            "predicted_choice": trial_record["predicted_choice"],
            "predicted_confidence": trial_record["predicted_confidence"],
            "predicted_probability_right": trial_record["predicted_probability_right"],
            "correct": trial_record["correct"],
            "human_choice_match": trial_record["human_choice_match"],
            "human_confidence": trial_record["human_confidence"],
        }

    def metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        return evaluate_predictions(records)

    def subject_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_confidence_predictions(records)


IGT_DECK_PROTOCOL = {
    "A": {"reward": 100, "penalties": [0, -150, 0, -300, 0, -200, 0, -250, 0, -350]},
    "B": {"reward": 100, "penalties": [0, 0, 0, 0, 0, 0, 0, 0, 0, -1250]},
    "C": {"reward": 50, "penalties": [0, -25, 0, -50, 0, -25, 0, -50, 0, -100]},
    "D": {"reward": 50, "penalties": [0, 0, 0, 0, 0, 0, 0, 0, 0, -250]},
}


class IowaGamblingTaskAdapter:
    def benchmark_id(self) -> str:
        return "iowa_gambling_task"

    def task_id(self) -> str:
        return "iowa_gambling_task"

    def schema(self) -> dict[str, Any]:
        bundle = _load_bundle(self.benchmark_id(), prefer_external=False)
        return {
            "benchmark_id": self.benchmark_id(),
            "status": bundle.status,
            "source_type": bundle.source_type,
            "source_label": bundle.source_label,
            "external_bundle_preferred": bundle.external_bundle_preferred,
            "smoke_test_only": bundle.smoke_test_only,
            "benchmark_state": bundle.benchmark_state,
            "available_states": bundle.available_states,
            "blockers": bundle.blockers,
            "trial_fields": list(IowaTrial.__dataclass_fields__.keys()),
            "trial_export_schema": _igt_export_schema(),
            "standard_protocol": {"trial_count": STANDARD_IGT_TRIAL_COUNT, "decks": ["A", "B", "C", "D"]},
        }

    def load_trials(
        self,
        *,
        allow_smoke_test: bool = False,
        benchmark_root: Path | str | None = None,
        prefer_external: bool = False,
        selected_subject_id: str | None = None,
        selected_split: str | None = None,
        protocol_mode: str = "standard_100",
        max_trials: int | None = None,
    ) -> list[IowaTrial]:
        payload = preprocess_iowa_gambling_task(
            allow_smoke_test=allow_smoke_test,
            benchmark_root=benchmark_root,
            prefer_external=prefer_external,
            selected_subject_id=selected_subject_id,
            selected_split=selected_split,
            max_trials=max_trials,
        )
        trials = [IowaTrial(**row) for row in payload["trials"]]
        if protocol_mode == "standard_100":
            by_subject: dict[str, list[IowaTrial]] = defaultdict(list)
            for trial in trials:
                by_subject[trial.subject_id].append(trial)
            selected: list[IowaTrial] = []
            for subject_id in sorted(by_subject):
                subject_trials = sorted(by_subject[subject_id], key=lambda item: item.trial_index)
                if len(subject_trials) < STANDARD_IGT_TRIAL_COUNT:
                    raise ValueError(f"Subject {subject_id} has only {len(subject_trials)} IGT trials; standard 100-trial protocol requires at least 100.")
                selected.extend(subject_trials[:STANDARD_IGT_TRIAL_COUNT])
            return selected
        return trials

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
        state = self.initial_state(subject_id=trial.subject_id, parameters=parameters)
        if deck_history:
            state["value_estimates"].update({deck: float(value) * 100.0 for deck, value in deck_history.items()})
        state["last_outcome"] = int(last_outcome)
        state["loss_streak"] = int(loss_streak)
        state["_run_seed"] = seed
        observation = self.observation_from_trial(trial, state=state, parameters=parameters, trial_index=trial_index)
        decision = _score_action_candidates(
            observation=observation,
            candidates=self.action_space(trial, observation=observation, state=state, parameters=parameters),
            parameters=parameters,
            rng=random.Random(seed + trial_index),
            state=state,
        )
        transition = self.apply_action(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            state=state,
            parameters=parameters,
            observation=observation,
            global_trial_index=trial_index,
        )
        record = self.export_trial_record(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            observation=observation,
            transition=transition,
            state=state,
            global_trial_index=trial_index,
        )
        return {"prediction": self.legacy_prediction(record), "resource_state": {}}

    def validate_protocol(self, trials: list[IowaTrial], **kwargs: Any) -> dict[str, Any]:
        protocol_mode = str(kwargs.get("protocol_mode", "standard_100"))
        by_subject: dict[str, list[IowaTrial]] = defaultdict(list)
        for trial in trials:
            by_subject[trial.subject_id].append(trial)
        if protocol_mode == "standard_100":
            for subject_id, subject_trials in by_subject.items():
                ordered = sorted(subject_trials, key=lambda item: item.trial_index)
                indices = [trial.trial_index for trial in ordered]
                if len(ordered) != STANDARD_IGT_TRIAL_COUNT or indices != list(range(1, STANDARD_IGT_TRIAL_COUNT + 1)):
                    raise ValueError(f"IGT standard protocol validation failed for {subject_id}: expected explicit 100-trial sequence 1..100.")
        return {
            "protocol_mode": protocol_mode,
            "trial_count": len(trials),
            "subject_count": len(by_subject),
            "standard_trial_count": STANDARD_IGT_TRIAL_COUNT if protocol_mode == "standard_100" else None,
            "ok": bool(trials),
        }

    def initial_state(self, *, subject_id: str, parameters: CognitiveStyleParameters) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "deck_history": [],
            "value_estimates": {deck: 0.0 for deck in "ABCD"},
            "deck_draw_counts": {deck: 0 for deck in "ABCD"},
            "choice_counts": {deck: 0 for deck in "ABCD"},
            "last_outcome": 0,
            "loss_streak": 0,
            "cumulative_gain": 0,
            "confidence": 0.5,
        }

    def observation_from_trial(
        self,
        trial: IowaTrial,
        *,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        trial_index: int,
    ) -> dict[str, Any]:
        estimates = list(float(value) for value in state["value_estimates"].values())
        uncertainty = 1.0 - _clamp01((max(estimates) - min(estimates) + 100.0) / 200.0)
        return {
            "trial_index": int(trial.trial_index),
            "last_outcome": int(state["last_outcome"]),
            "loss_streak": int(state["loss_streak"]),
            "uncertainty": _clamp01(uncertainty),
            "resource_pressure": 0.0,
            "evidence_strength": _clamp01((max(estimates) + 50.0) / 150.0),
        }

    def action_space(
        self,
        trial: IowaTrial,
        *,
        observation: dict[str, Any],
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for deck in "ABCD":
            spec = IGT_DECK_PROTOCOL[deck]
            draw_count = int(state["deck_draw_counts"][deck])
            cycle_penalty = int(spec["penalties"][draw_count % len(spec["penalties"])])
            expected_value = float(state["value_estimates"][deck])
            risk_penalty = abs(cycle_penalty) / 1250.0
            novelty_bonus = max(0.0, 1.0 - (state["choice_counts"][deck] / max(1, observation["trial_index"])))
            habit_value = state["value_estimates"][deck] / 100.0
            if deck in {"C", "D"}:
                expected_value += 8.0 + parameters.error_aversion * 6.0
            else:
                expected_value -= parameters.error_aversion * 5.0
            if state["loss_streak"] >= 2 and deck in {"A", "B"}:
                expected_value -= 8.0
            candidates.append(
                {
                    "action_id": deck,
                    "expected_value": expected_value / 100.0,
                    "evidence_alignment": expected_value / 100.0,
                    "risk_penalty": risk_penalty,
                    "habit_value": habit_value,
                    "novelty_bonus": novelty_bonus * 0.18,
                    "noise_scale": 0.05,
                }
            )
        return candidates

    def apply_action(
        self,
        trial: IowaTrial,
        *,
        chosen_action: str,
        confidence: float,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        observation: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        spec = IGT_DECK_PROTOCOL[chosen_action]
        draw_index = int(state["deck_draw_counts"][chosen_action])
        reward = int(spec["reward"])
        penalty = int(spec["penalties"][draw_index % len(spec["penalties"])])
        net_outcome = reward + penalty
        state["deck_draw_counts"][chosen_action] += 1
        state["choice_counts"][chosen_action] += 1
        state["cumulative_gain"] += net_outcome
        state["last_outcome"] = net_outcome
        state["loss_streak"] = int(state["loss_streak"]) + 1 if net_outcome < 0 else 0
        learning_rate = 0.18 + parameters.confidence_gain * 0.16 - parameters.update_rigidity * 0.06
        prior = float(state["value_estimates"][chosen_action])
        state["value_estimates"][chosen_action] = _safe_round(prior + learning_rate * (net_outcome - prior))
        state["deck_history"].append(chosen_action)
        state["deck_history"] = state["deck_history"][-20:]
        state["confidence"] = confidence
        return {
            "reward": reward,
            "penalty": penalty,
            "net_outcome": net_outcome,
            "cumulative_gain": int(state["cumulative_gain"]),
            "deck_match": chosen_action == trial.deck,
            "advantageous_choice": chosen_action in {"C", "D"},
            "actual_advantageous": bool(trial.advantageous),
            "expected_value": _safe_round(float(state["value_estimates"][chosen_action])),
        }

    def export_trial_record(
        self,
        trial: IowaTrial,
        *,
        chosen_action: str,
        confidence: float,
        observation: dict[str, Any],
        transition: dict[str, Any],
        state: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        snapshot = {
            "deck_history": list(state["deck_history"]),
            "value_estimates": {deck: _safe_round(value) for deck, value in state["value_estimates"].items()},
            "last_outcome": int(state["last_outcome"]),
            "loss_streak": int(state["loss_streak"]),
            "confidence": _safe_round(state["confidence"]),
            "deck_draw_counts": dict(state["deck_draw_counts"]),
        }
        return {
            "trial_id": trial.trial_id,
            "subject_id": trial.subject_id,
            "split": trial.split,
            "trial_index": int(trial.trial_index),
            "chosen_deck": chosen_action,
            "reward": int(transition["reward"]),
            "penalty": int(transition["penalty"]),
            "net_outcome": int(transition["net_outcome"]),
            "cumulative_gain": int(transition["cumulative_gain"]),
            "internal_state_snapshot": snapshot,
            "human_deck": trial.deck,
            "human_reward": int(trial.reward),
            "human_penalty": int(trial.penalty),
            "human_net_outcome": int(trial.net_outcome),
            "human_advantageous": bool(trial.advantageous),
            "deck_match": bool(transition["deck_match"]),
            "advantageous_choice": bool(transition["advantageous_choice"]),
            "actual_advantageous": bool(transition["actual_advantageous"]),
            "agent_confidence_rating": _safe_round(confidence),
            "predicted_confidence": _safe_round(confidence),
            "expected_value": _safe_round(transition["expected_value"]),
            "source_file": trial.source_file,
        }

    def legacy_prediction(self, trial_record: dict[str, Any]) -> dict[str, Any]:
        return {
            "trial_id": trial_record["trial_id"],
            "subject_id": trial_record["subject_id"],
            "split": trial_record["split"],
            "chosen_deck": trial_record["chosen_deck"],
            "actual_deck": trial_record["human_deck"],
            "expected_value": trial_record["expected_value"],
            "predicted_confidence": trial_record["predicted_confidence"],
            "advantageous_choice": trial_record["advantageous_choice"],
            "actual_advantageous": trial_record["actual_advantageous"],
            "deck_match": trial_record["deck_match"],
            "reward": trial_record["reward"],
            "penalty": trial_record["penalty"],
            "net_outcome": trial_record["net_outcome"],
        }

    def metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        return evaluate_iowa_predictions(records)

    def subject_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_iowa_predictions(records)


class TwoArmedBanditAdapter:
    def benchmark_id(self) -> str:
        return "two_armed_bandit"

    def task_id(self) -> str:
        return "two_armed_bandit"

    def schema(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id(),
            "status": "acceptance_ready",
            "source_type": "synthetic_protocol",
            "source_label": "two_armed_bandit_acceptance_demo",
            "trial_export_schema": _bandit_export_schema(),
            "action_fields": ["arm_left", "arm_right"],
        }

    def load_trials(self, *, trial_count: int = 50, split: str = "acceptance", **_: Any) -> list[BanditTrial]:
        trials: list[BanditTrial] = []
        for index in range(1, trial_count + 1):
            probabilities = {"arm_left": 0.72, "arm_right": 0.28} if index <= trial_count // 2 else {"arm_left": 0.65, "arm_right": 0.35}
            trials.append(
                BanditTrial(
                    trial_id=f"bandit::{index:03d}",
                    subject_id="bandit_agent",
                    session_id="bandit_acceptance_session",
                    split=split,
                    trial_index=index,
                    arm_reward_probabilities=probabilities,
                )
            )
        return trials

    def validate_protocol(self, trials: list[BanditTrial], **_: Any) -> dict[str, Any]:
        indices = [trial.trial_index for trial in trials]
        return {"protocol_mode": "synthetic_bandit", "trial_count": len(trials), "ok": indices == list(range(1, len(trials) + 1))}

    def initial_state(self, *, subject_id: str, parameters: CognitiveStyleParameters) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "q_values": {"arm_left": 0.0, "arm_right": 0.0},
            "counts": {"arm_left": 0, "arm_right": 0},
            "last_reward": 0,
            "cumulative_reward": 0,
            "choice_history": [],
            "confidence": 0.5,
        }

    def observation_from_trial(self, trial: BanditTrial, *, state: dict[str, Any], parameters: CognitiveStyleParameters, trial_index: int) -> dict[str, Any]:
        q_values = list(float(value) for value in state["q_values"].values())
        return {
            "trial_index": int(trial.trial_index),
            "uncertainty": 1.0 - _clamp01(abs(q_values[0] - q_values[1])),
            "resource_pressure": 0.0,
            "evidence_strength": max(trial.arm_reward_probabilities.values()),
        }

    def action_space(self, trial: BanditTrial, *, observation: dict[str, Any], state: dict[str, Any], parameters: CognitiveStyleParameters) -> list[dict[str, Any]]:
        total_count = max(1, trial.trial_index - 1)
        candidates: list[dict[str, Any]] = []
        for arm in ("arm_left", "arm_right"):
            count = int(state["counts"][arm])
            candidates.append(
                {
                    "action_id": arm,
                    "expected_value": float(state["q_values"][arm]),
                    "evidence_alignment": float(state["q_values"][arm]),
                    "risk_penalty": 1.0 - float(trial.arm_reward_probabilities[arm]),
                    "habit_value": float(state["q_values"][arm]),
                    "novelty_bonus": max(0.0, 1.0 - count / total_count) * 0.30,
                    "noise_scale": 0.08,
                }
            )
        return candidates

    def apply_action(
        self,
        trial: BanditTrial,
        *,
        chosen_action: str,
        confidence: float,
        state: dict[str, Any],
        parameters: CognitiveStyleParameters,
        observation: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        reward_probability = float(trial.arm_reward_probabilities[chosen_action])
        reward_seed = int(state.get("_run_seed", 0)) * 1009 + global_trial_index * 97 + state["counts"][chosen_action] * 13 + (1 if chosen_action == "arm_left" else 2)
        reward = 1 if random.Random(reward_seed).random() < reward_probability else 0
        state["counts"][chosen_action] += 1
        state["last_reward"] = reward
        state["cumulative_reward"] += reward
        state["choice_history"].append(chosen_action)
        state["choice_history"] = state["choice_history"][-20:]
        learning_rate = 0.18 + parameters.confidence_gain * 0.12
        q_value = float(state["q_values"][chosen_action])
        state["q_values"][chosen_action] = _safe_round(q_value + learning_rate * (reward - q_value))
        state["confidence"] = confidence
        return {
            "reward": reward,
            "reward_probability": reward_probability,
            "cumulative_reward": int(state["cumulative_reward"]),
            "optimal_choice": reward_probability == max(trial.arm_reward_probabilities.values()),
        }

    def export_trial_record(
        self,
        trial: BanditTrial,
        *,
        chosen_action: str,
        confidence: float,
        observation: dict[str, Any],
        transition: dict[str, Any],
        state: dict[str, Any],
        global_trial_index: int,
    ) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "subject_id": trial.subject_id,
            "session_id": trial.session_id,
            "split": trial.split,
            "trial_index": int(trial.trial_index),
            "chosen_arm": chosen_action,
            "reward": int(transition["reward"]),
            "reward_probability": _safe_round(transition["reward_probability"]),
            "optimal_choice": bool(transition["optimal_choice"]),
            "cumulative_reward": int(transition["cumulative_reward"]),
            "agent_confidence_rating": _safe_round(confidence),
            "internal_state_snapshot": {
                "q_values": {arm: _safe_round(value) for arm, value in state["q_values"].items()},
                "counts": dict(state["counts"]),
                "last_reward": int(state["last_reward"]),
                "choice_history": list(state["choice_history"]),
                "confidence": _safe_round(state["confidence"]),
            },
        }

    def legacy_prediction(self, trial_record: dict[str, Any]) -> dict[str, Any]:
        return dict(trial_record)

    def metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        optimal = [1.0 if bool(row["optimal_choice"]) else 0.0 for row in records]
        rewards = [float(row["reward"]) for row in records]
        confidences = [float(row["agent_confidence_rating"]) for row in records]
        return {
            "optimal_choice_rate": _safe_round(_mean(optimal)),
            "mean_reward": _safe_round(_mean(rewards)),
            "reward_variance": _safe_round(pvariance(rewards) if len(rewards) > 1 else 0.0),
            "mean_confidence": _safe_round(_mean(confidences)),
        }

    def subject_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"subjects": {"bandit_agent": self.metrics(records)}, "subject_count": 1}


TASK_ADAPTERS: dict[str, BenchmarkAdapter] = {
    "confidence_database": ConfidenceDatabaseAdapter(),
    "iowa_gambling_task": IowaGamblingTaskAdapter(),
    "two_armed_bandit": TwoArmedBanditAdapter(),
}


def register_task_adapter(adapter: BenchmarkAdapter) -> None:
    TASK_ADAPTERS[adapter.task_id()] = adapter


def task_adapter_registry() -> dict[str, BenchmarkAdapter]:
    return dict(TASK_ADAPTERS)


def _run_adapter(
    adapter: BenchmarkAdapter,
    *,
    parameters: CognitiveStyleParameters | None = None,
    seed: int,
    allow_smoke_test: bool = False,
    benchmark_root: Path | str | None = None,
    prefer_external: bool = False,
    include_subject_summary: bool = True,
    include_predictions: bool = True,
    include_trial_trace: bool = True,
    split: str | None = None,
    max_trials: int | None = None,
    protocol_mode: str | None = None,
    selected_subject_id: str | None = None,
    selected_source_dataset: str | None = None,
    trial_count: int | None = None,
) -> dict[str, Any]:
    active_parameters = parameters or CognitiveStyleParameters()
    loader_kwargs: dict[str, Any] = {
        "allow_smoke_test": allow_smoke_test,
        "benchmark_root": benchmark_root,
        "prefer_external": prefer_external,
        "max_trials": max_trials,
    }
    if split is not None:
        loader_kwargs["selected_split"] = split
    if selected_subject_id is not None:
        loader_kwargs["selected_subject_id"] = selected_subject_id
    if selected_source_dataset is not None:
        loader_kwargs["selected_source_dataset"] = selected_source_dataset
    if protocol_mode is not None:
        loader_kwargs["protocol_mode"] = protocol_mode
    if trial_count is not None:
        loader_kwargs["trial_count"] = trial_count
    trials = adapter.load_trials(**loader_kwargs)
    protocol_validation = adapter.validate_protocol(trials, protocol_mode=protocol_mode)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for global_trial_index, trial in enumerate(trials, start=1):
        subject_id = str(getattr(trial, "subject_id", "global"))
        state = states.setdefault(subject_id, adapter.initial_state(subject_id=subject_id, parameters=active_parameters))
        state.setdefault("_run_seed", seed)
        observation = adapter.observation_from_trial(trial, state=state, parameters=active_parameters, trial_index=global_trial_index)
        decision = _score_action_candidates(
            observation=observation,
            candidates=adapter.action_space(trial, observation=observation, state=state, parameters=active_parameters),
            parameters=active_parameters,
            rng=rng,
            state=state,
        )
        transition = adapter.apply_action(
            trial,
            chosen_action=decision["chosen_action"],
            confidence=float(decision["confidence"]),
            state=state,
            parameters=active_parameters,
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
    payload: dict[str, Any] = {
        "benchmark_id": adapter.benchmark_id(),
        "parameters": active_parameters.to_dict(),
        "seed": seed,
        "trial_count": len(trials),
        "split": split or "all",
        "protocol_validation": protocol_validation,
        "metrics": adapter.metrics(records),
        "predictions": predictions if include_predictions else [],
        "trial_trace": records if include_trial_trace else [],
        "subject_summary": adapter.subject_summary(records) if include_subject_summary else {"subjects": {}, "subject_count": 0},
        "schema": adapter.schema(),
        "resources": [],
    }
    if adapter.task_id() in {"confidence_database", "iowa_gambling_task"}:
        bundle = _load_bundle(adapter.benchmark_id(), benchmark_root=benchmark_root, prefer_external=prefer_external)
        status = _benchmark_status(adapter.benchmark_id(), benchmark_root=benchmark_root, prefer_external=prefer_external)
        payload.update(
            {
                "bundle": bundle.to_dict(),
                "benchmark_status": status.to_dict(),
                "bundle_mode": "external_bundle" if bundle.source_type == "external_bundle" else "repo_smoke_test",
                "claim_envelope": "benchmark_eval" if bundle.source_type == "external_bundle" else "smoke_only",
            }
        )
        if adapter.task_id() == "confidence_database":
            preprocess_payload = preprocess_confidence_database(
                allow_smoke_test=allow_smoke_test,
                selected_split=split,
                benchmark_root=benchmark_root,
                prefer_external=prefer_external,
                selected_source_dataset=selected_source_dataset,
                max_trials=max_trials,
            )
            payload["split_unit"] = preprocess_payload["split_unit"]
            payload["leakage_check"] = preprocess_payload["leakage_check"]
            payload["trial_export_validation"] = validate_trial_export_records(records, _confidence_export_schema(), expected_length=len(trials))
        else:
            payload["trial_export_validation"] = validate_trial_export_records(records, _igt_export_schema(), expected_length=len(trials))
    else:
        payload.update(
            {
                "bundle": {"benchmark_id": adapter.benchmark_id(), "source_type": "synthetic_protocol", "source_label": "two_armed_bandit_acceptance_demo"},
                "benchmark_status": {
                    "benchmark_id": adapter.benchmark_id(),
                    "benchmark_state": "acceptance_ready",
                    "acceptance_ready": True,
                    "available_states": ["scaffold_complete", "acceptance_ready"],
                    "blockers": [],
                    "status_notes": [],
                },
                "bundle_mode": "synthetic_protocol",
                "claim_envelope": "acceptance_grade",
                "trial_export_validation": validate_trial_export_records(records, _bandit_export_schema(), expected_length=len(trials)),
            }
        )
    return payload


def run_task_adapter(task_id: str, **kwargs: Any) -> dict[str, Any]:
    return _run_adapter(TASK_ADAPTERS[task_id], **kwargs)


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
    rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in predictions]
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
    return {
        "accuracy": _safe_round(mean(correctness) if rows else 0.0),
        "calibration_error": _safe_round(mean(abs(conf - corr) for conf, corr in zip(confidences, correctness)) if rows else 0.0),
        "brier_score": _safe_round(mean((conf - corr) ** 2 for conf, corr in zip(confidences, correctness)) if rows else 0.0),
        "heldout_likelihood": _safe_round(_mean(heldout_likelihood_values)),
        "confidence_bias": _safe_round(mean(conf - human for conf, human in zip(confidences, human_confidences)) if rows else 0.0),
        "auroc2": _safe_round(_auroc([int(value) for value in correctness], confidences) if rows else 0.5),
        "meta_d_prime_ratio": _safe_round(_clamp01(((_auroc([int(value) for value in correctness], confidences) if rows else 0.5) / max(0.5, mean(correctness) if rows else 0.5)) - 0.5)),
        "confidence_alignment": _safe_round(_spearman(confidences, human_confidences) if rows else 0.0),
        "subject_count": float(len(subject_ids)),
    }


def evaluate_iowa_predictions(predictions: list[IowaPrediction | dict[str, Any]]) -> dict[str, float]:
    rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in predictions]
    advantageous = [1.0 if bool(row["advantageous_choice"]) else 0.0 for row in rows]
    confidences = [float(row.get("predicted_confidence", row.get("agent_confidence_rating", 0.0))) for row in rows]
    outcomes = [float(row["net_outcome"]) for row in rows]
    expected_values = [float(row.get("expected_value", row["net_outcome"])) for row in rows]
    actual_advantageous = [1.0 if bool(row["actual_advantageous"]) else 0.0 for row in rows]
    deck_match = [1.0 if bool(row["deck_match"]) else 0.0 for row in rows]
    late_rows = rows[-max(1, len(rows) // 2) :] if rows else []
    return {
        "advantageous_choice_rate": _safe_round(_mean(advantageous)),
        "mean_net_outcome": _safe_round(_mean(outcomes)),
        "reward_fit": _safe_round(-mean((pred - actual) ** 2 for pred, actual in zip(expected_values, outcomes)) if rows else 0.0),
        "confidence_advantage_alignment": _safe_round(_spearman(confidences, actual_advantageous) if rows else 0.0),
        "deck_match_rate": _safe_round(_mean(deck_match)),
        "policy_alignment_rate": _safe_round(mean(1.0 if bool(row["advantageous_choice"]) == bool(row["actual_advantageous"]) else 0.0 for row in rows) if rows else 0.0),
        "late_advantageous_rate": _safe_round(mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in late_rows) if late_rows else 0.0),
        "final_cumulative_gain": _safe_round(float(rows[-1]["cumulative_gain"]) if rows and "cumulative_gain" in rows[-1] else 0.0),
    }


def summarize_confidence_predictions(predictions: list[BenchmarkPrediction | dict[str, Any]]) -> dict[str, Any]:
    rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in predictions]
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    subject_metrics = {subject_id: evaluate_predictions(subject_rows) for subject_id, subject_rows in by_subject.items()}
    heldout_values = [float(metrics["heldout_likelihood"]) for metrics in subject_metrics.values()]
    calibration_values = [float(metrics["calibration_error"]) for metrics in subject_metrics.values()]
    return {
        "subjects": subject_metrics,
        "subject_count": len(subject_metrics),
        "heldout_likelihood_floor": _safe_round(min(heldout_values) if heldout_values else 0.0),
        "heldout_likelihood_mean": _safe_round(_mean(heldout_values)),
        "calibration_ceiling": _safe_round(max(calibration_values) if calibration_values else 0.0),
    }


def summarize_iowa_predictions(predictions: list[IowaPrediction | dict[str, Any]]) -> dict[str, Any]:
    rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in predictions]
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    subject_metrics = {subject_id: evaluate_iowa_predictions(subject_rows) for subject_id, subject_rows in by_subject.items()}
    deck_match_values = [float(metrics["deck_match_rate"]) for metrics in subject_metrics.values()]
    policy_alignment_values = [float(metrics["policy_alignment_rate"]) for metrics in subject_metrics.values()]
    return {
        "subjects": subject_metrics,
        "subject_count": len(subject_metrics),
        "deck_match_floor": _safe_round(min(deck_match_values) if deck_match_values else 0.0),
        "policy_alignment_mean": _safe_round(_mean(policy_alignment_values)),
    }


def run_confidence_database_benchmark(parameters: CognitiveStyleParameters | None = None, *, seed: int = 42, split: str | None = None, malformed_rows: list[dict[str, Any]] | None = None, allow_smoke_test: bool = False, include_subject_summary: bool = True, include_predictions: bool = True, include_trial_trace: bool = True, max_trials: int | None = None, benchmark_root: Path | str | None = None, prefer_external: bool = False, selected_source_dataset: str | None = None) -> dict[str, Any]:
    payload = _run_adapter(TASK_ADAPTERS["confidence_database"], parameters=parameters, seed=seed, allow_smoke_test=allow_smoke_test, benchmark_root=benchmark_root, prefer_external=prefer_external, include_subject_summary=include_subject_summary, include_predictions=include_predictions, include_trial_trace=include_trial_trace, split=split, max_trials=max_trials, selected_source_dataset=selected_source_dataset)
    skipped_malformed = 0
    if malformed_rows:
        required = {"trial_id", "subject_id", "stimulus_strength", "correct_choice", "human_choice", "human_confidence", "rt_ms"}
        for row in malformed_rows:
            if not required <= row.keys():
                skipped_malformed += 1
    payload["skipped_malformed"] = skipped_malformed
    return payload


def run_iowa_gambling_benchmark(parameters: CognitiveStyleParameters | None = None, *, seed: int = 44, split: str | None = None, allow_smoke_test: bool = False, include_subject_summary: bool = True, include_predictions: bool = True, include_trial_trace: bool = True, max_trials: int | None = None, benchmark_root: Path | str | None = None, prefer_external: bool = False, protocol_mode: str = "standard_100", selected_subject_id: str | None = None) -> dict[str, Any]:
    return _run_adapter(TASK_ADAPTERS["iowa_gambling_task"], parameters=parameters, seed=seed, allow_smoke_test=allow_smoke_test, benchmark_root=benchmark_root, prefer_external=prefer_external, include_subject_summary=include_subject_summary, include_predictions=include_predictions, include_trial_trace=include_trial_trace, split=split, max_trials=max_trials, protocol_mode=protocol_mode, selected_subject_id=selected_subject_id)


def run_two_armed_bandit_benchmark(parameters: CognitiveStyleParameters | None = None, *, seed: int = 46, include_subject_summary: bool = True, include_predictions: bool = True, include_trial_trace: bool = True, trial_count: int = 50) -> dict[str, Any]:
    return _run_adapter(TASK_ADAPTERS["two_armed_bandit"], parameters=parameters, seed=seed, include_subject_summary=include_subject_summary, include_predictions=include_predictions, include_trial_trace=include_trial_trace, trial_count=trial_count)


def _behavior_signature(run_payload: dict[str, Any], *, task_id: str) -> list[Any]:
    trace = list(run_payload.get("trial_trace", []))
    if task_id == "confidence_database":
        return [(row["trial_id"], row["agent_choice"], row["agent_confidence_rating"]) for row in trace]
    if task_id == "iowa_gambling_task":
        return [(row["trial_index"], row["chosen_deck"], row["net_outcome"], row["cumulative_gain"]) for row in trace]
    return [(row["trial_index"], row["chosen_arm"], row["reward"], row["cumulative_reward"]) for row in trace]


def same_seed_triple_replay(task_id: str, *, seed: int, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    runs = [run_task_adapter(task_id, seed=seed, **dict(run_kwargs)) for _ in range(3)]
    signatures = [_behavior_signature(run, task_id=task_id) for run in runs]
    return {"task_id": task_id, "seed": seed, "exact_match": signatures[0] == signatures[1] == signatures[2], "signatures": signatures, "runs": runs}


def compute_behavioral_seed_summaries(task_id: str, *, seeds: list[int], run_kwargs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    signatures: list[list[Any]] = []
    for seed in seeds:
        run = run_task_adapter(task_id, seed=seed, **dict(run_kwargs))
        signatures.append(_behavior_signature(run, task_id=task_id))
        trace = run["trial_trace"]
        if task_id == "iowa_gambling_task":
            rows.append({"seed": seed, "mean_net_outcome": _safe_round(_mean([float(item["net_outcome"]) for item in trace])), "advantageous_choice_rate": _safe_round(_mean([1.0 if bool(item["advantageous_choice"]) else 0.0 for item in trace])), "final_cumulative_gain": _safe_round(float(trace[-1]["cumulative_gain"]) if trace else 0.0)})
        elif task_id == "confidence_database":
            rows.append({"seed": seed, "choice_right_rate": _safe_round(_mean([1.0 if item["agent_choice"] == "right" else 0.0 for item in trace])), "mean_confidence": _safe_round(_mean([float(item["agent_confidence_rating"]) for item in trace])), "accuracy": _safe_round(_mean([1.0 if bool(item["correct"]) else 0.0 for item in trace]))})
        else:
            rows.append({"seed": seed, "optimal_choice_rate": _safe_round(_mean([1.0 if bool(item["optimal_choice"]) else 0.0 for item in trace])), "mean_reward": _safe_round(_mean([float(item["reward"]) for item in trace])), "reward_variance": _safe_round(pvariance([float(item["reward"]) for item in trace]) if len(trace) > 1 else 0.0)})
    unique_signatures = len({json.dumps(signature, ensure_ascii=False) for signature in signatures})
    summary_by_metric: dict[str, dict[str, float]] = {}
    metric_names = [name for name in rows[0].keys() if name != "seed"] if rows else []
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary_by_metric[metric_name] = {"mean": _safe_round(_mean(values)), "variance": _safe_round(pvariance(values) if len(values) > 1 else 0.0)}
    return {"task_id": task_id, "seed_summaries": rows, "behavioral_summary": summary_by_metric, "unique_behavior_sequences": unique_signatures, "different_seeds_differ": unique_signatures > 1}


def bootstrap_seed_summary_ci(seed_summaries: list[dict[str, Any]], *, metric_name: str, bootstrap_seed: int, samples: int = 500) -> dict[str, float]:
    values = [float(row[metric_name]) for row in seed_summaries]
    if not values:
        return {"metric": metric_name, "mean": 0.0, "variance": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = random.Random(bootstrap_seed)
    boot_means: list[float] = []
    for _ in range(samples):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        boot_means.append(_mean(sampled))
    ordered = sorted(boot_means)
    lower_index = max(0, int(samples * 0.025) - 1)
    upper_index = min(samples - 1, int(samples * 0.975))
    return {"metric": metric_name, "mean": _safe_round(_mean(values)), "variance": _safe_round(pvariance(values) if len(values) > 1 else 0.0), "ci_lower": _safe_round(ordered[lower_index]), "ci_upper": _safe_round(ordered[upper_index])}


def evaluate_seed_tolerance_gate(seed_summaries: list[dict[str, Any]], *, metric_name: str, bootstrap_seed: int, lower_bound: float, upper_bound: float, min_variance: float) -> dict[str, Any]:
    ci = bootstrap_seed_summary_ci(seed_summaries, metric_name=metric_name, bootstrap_seed=bootstrap_seed)
    passed = lower_bound <= float(ci["ci_lower"]) <= float(ci["ci_upper"]) <= upper_bound and float(ci["variance"]) >= min_variance
    return {"metric": metric_name, "passed": passed, "ci": ci, "tolerance": {"lower_bound": lower_bound, "upper_bound": upper_bound, "min_variance": min_variance}}


__all__ = [
    "BenchmarkAdapter",
    "BenchmarkPrediction",
    "BenchmarkTrial",
    "BanditTrial",
    "ConfidenceDatabaseAdapter",
    "DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET",
    "EXTERNAL_BENCHMARK_ROOT",
    "IowaGamblingTaskAdapter",
    "IowaPrediction",
    "IowaTrial",
    "STANDARD_IGT_TRIAL_COUNT",
    "TASK_ADAPTERS",
    "TwoArmedBanditAdapter",
    "_auroc",
    "_spearman",
    "assign_group_splits",
    "bootstrap_seed_summary_ci",
    "compute_behavioral_seed_summaries",
    "default_acceptance_benchmark_root",
    "detect_subject_leakage",
    "evaluate_iowa_predictions",
    "evaluate_predictions",
    "evaluate_seed_tolerance_gate",
    "preprocess_confidence_database",
    "preprocess_iowa_gambling_task",
    "register_task_adapter",
    "run_confidence_database_benchmark",
    "run_iowa_gambling_benchmark",
    "run_task_adapter",
    "run_two_armed_bandit_benchmark",
    "same_seed_triple_replay",
    "summarize_confidence_predictions",
    "summarize_iowa_predictions",
    "task_adapter_registry",
    "validate_trial_export_records",
]
