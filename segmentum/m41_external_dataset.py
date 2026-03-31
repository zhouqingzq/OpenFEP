from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, DecisionLogRecord, PROFILE_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DATA_ROOT = ROOT / "data" / "m41_external"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _iter_json_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.strip():
                    rows.append(json.loads(raw_line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    raise ValueError(f"Unsupported external payload format in {path}")


def _derived_attention_allocation(observation_evidence: dict[str, Any]) -> dict[str, float]:
    evidence_strength = _clamp01(observation_evidence.get("evidence_strength", 0.5))
    uncertainty = _clamp01(observation_evidence.get("uncertainty", 0.5))
    expected_error = _clamp01(observation_evidence.get("expected_error", 0.5))
    imagined_risk = _clamp01(observation_evidence.get("imagined_risk", expected_error))
    raw = {
        "evidence": 0.15 + evidence_strength * 0.70,
        "uncertainty": 0.12 + uncertainty * 0.58,
        "error": 0.10 + expected_error * 0.56,
        "counterfactual": 0.08 + imagined_risk * 0.52,
    }
    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 6) for key, value in raw.items()}


def _dominant_signal(attention_allocation: dict[str, float]) -> str:
    if not attention_allocation:
        return "evidence"
    return max(attention_allocation, key=attention_allocation.get)


def _placeholder_candidates(selected_action: str, observation_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    base_confidence = _clamp01(observation_evidence.get("evidence_strength", 0.5) * 0.75 + 0.2)
    base_error = _clamp01(observation_evidence.get("expected_error", 0.5))
    return [
        {
            "action": {"name": selected_action},
            "total_score": round(base_confidence - base_error, 6),
            "expected_value": round(_clamp01(0.55 + observation_evidence.get("evidence_strength", 0.5) * 0.25 - base_error * 0.15), 6),
            "expected_confidence": round(base_confidence, 6),
            "expected_prediction_error": round(base_error, 6),
            "resource_cost": round(0.12 + _clamp01(observation_evidence.get("uncertainty", 0.5)) * 0.08, 6),
        },
        {
            "action": {"name": "commit"},
            "total_score": round(observation_evidence.get("evidence_strength", 0.5) - base_error * 1.1, 6),
            "expected_value": round(_clamp01(0.45 + observation_evidence.get("evidence_strength", 0.5) * 0.30), 6),
            "expected_confidence": round(_clamp01(base_confidence + 0.04), 6),
            "expected_prediction_error": round(base_error, 6),
            "resource_cost": 0.24,
        },
        {
            "action": {"name": "recover"},
            "total_score": round(0.42 - base_error * 0.35, 6),
            "expected_value": round(_clamp01(0.48 - base_error * 0.10), 6),
            "expected_confidence": round(_clamp01(base_confidence - 0.05), 6),
            "expected_prediction_error": round(base_error, 6),
            "resource_cost": 0.08,
        },
    ]


def normalize_external_record(record: dict[str, Any]) -> dict[str, Any]:
    if str(record.get("schema_version", "")).startswith("m4.decision_log"):
        return DecisionLogRecord.from_dict(record).to_dict()
    if str(record.get("schema_version")) != "m41.external.event.v1":
        raise ValueError(f"Unsupported external record schema: {record.get('schema_version')}")

    observation_evidence = dict(record.get("observation_evidence", {}))
    attention_allocation = dict(record.get("attention_allocation", {})) or _derived_attention_allocation(observation_evidence)
    resource_state = dict(record.get("resource_state", {}))
    prediction_error_vector = dict(record.get("prediction_error_vector", {}))
    if not prediction_error_vector:
        expected_error = _clamp01(observation_evidence.get("expected_error", 0.4))
        imagined_risk = _clamp01(observation_evidence.get("imagined_risk", expected_error))
        prediction_error_vector = {
            "direct_error": round(expected_error, 6),
            "virtual_error": round(_clamp01((expected_error + imagined_risk) / 2.0), 6),
            "signed_total": round(_clamp01(expected_error * 0.7 + imagined_risk * 0.3), 6),
        }
    selected_action = str(record.get("selected_action", "scan"))
    percept_summary = dict(record.get("percept_summary", {}))
    if not percept_summary:
        pressure = 1.0 - ((float(resource_state.get("energy", 0.5)) + float(resource_state.get("budget", 0.5)) + float(resource_state.get("time_remaining", 0.5))) / 3.0)
        percept_summary = {
            "dominant_signal": _dominant_signal(attention_allocation),
            "evidence_band": "high" if observation_evidence.get("evidence_strength", 0.0) >= 0.70 else "medium" if observation_evidence.get("evidence_strength", 0.0) >= 0.40 else "low",
            "uncertainty_band": "high" if observation_evidence.get("uncertainty", 0.0) >= 0.60 else "medium" if observation_evidence.get("uncertainty", 0.0) >= 0.30 else "low",
            "pressure_band": "high" if pressure >= 0.60 else "medium" if pressure >= 0.35 else "low",
        }

    normalized = DecisionLogRecord.from_dict(
        {
            "schema_version": "m4.decision_log.v3",
            "tick": int(record.get("tick", 0)),
            "timestamp": record.get("timestamp", "2026-01-01T00:00:00+00:00"),
            "seed": int(record.get("seed", 0)),
            "task_context": {
                "phase": record.get("task_name", "external_task"),
                "subject_id": record.get("subject_id", ""),
                "session_id": record.get("session_id", ""),
                "source_name": record.get("source_name", ""),
            },
            "percept_summary": percept_summary,
            "observation_evidence": observation_evidence,
            "prediction_error_vector": prediction_error_vector,
            "attention_allocation": attention_allocation,
            "candidate_actions": list(record.get("candidate_actions", [])) or _placeholder_candidates(selected_action, observation_evidence),
            "parameter_snapshot": CognitiveStyleParameters().to_dict(),
            "resource_state": resource_state,
            "internal_confidence": float(record.get("internal_confidence", 0.5)),
            "selected_action": selected_action,
            "result_feedback": dict(record.get("result_feedback", {}))
            or {
                "observed_outcome": record.get("observed_outcome", "external_observation"),
                "reward": round(1.0 - prediction_error_vector["signed_total"], 6),
                "counterfactual_warning": prediction_error_vector["virtual_error"] > prediction_error_vector["direct_error"],
            },
            "model_update": dict(record.get("model_update", {}))
            or {
                "magnitude": round(float(record.get("update_magnitude", 0.2)), 6),
                "strategy_shift": round(float(record.get("strategy_shift", 0.16)), 6),
                "confidence_delta": round(float(record.get("confidence_delta", 0.0)), 6),
            },
            "prediction_error": float(record.get("prediction_error", prediction_error_vector["signed_total"])),
            "update_magnitude": float(record.get("update_magnitude", record.get("model_update", {}).get("magnitude", 0.2))),
        }
    )
    return normalized.to_dict()


def load_external_behavior_dataset(root: Path | str | None = None) -> dict[str, Any]:
    dataset_root = Path(root or DEFAULT_EXTERNAL_DATA_ROOT).resolve()
    files = sorted([path for path in dataset_root.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}])
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend([{**payload, "_source_file": path.name} for payload in _iter_json_payloads(path)])

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("source_name", "")), str(row.get("subject_id", "")), str(row.get("session_id", "")))
        grouped[key].append(row)

    sessions = []
    for (source_name, subject_id, session_id), session_rows in sorted(grouped.items()):
        ordered = sorted(session_rows, key=lambda item: (int(item.get("tick", 0)), str(item.get("timestamp", ""))))
        first = ordered[0]
        sessions.append(
            {
                "source_name": source_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "task_name": str(first.get("task_name", "")),
                "split": str(first.get("split", "heldout")),
                "source_file": str(first.get("_source_file", "")),
                "profile_label": first.get("ground_truth_profile"),
                "ground_truth_parameters": dict(first.get("ground_truth_parameters", {}))
                or (PROFILE_REGISTRY[str(first.get("ground_truth_profile"))].to_dict() if str(first.get("ground_truth_profile")) in PROFILE_REGISTRY else None),
                "records": [normalize_external_record(row) for row in ordered],
                "normalization_notes": [
                    "parameter_snapshot filled with default schema placeholder to preserve decision-log compatibility",
                    "candidate_actions may be synthesized when external logs omit latent alternatives",
                ],
            }
        )

    return {
        "schema_version": "m41.external.dataset.v1",
        "dataset_root": str(dataset_root),
        "record_count": len(rows),
        "session_count": len(sessions),
        "sources": sorted({session["source_name"] for session in sessions}),
        "tasks": sorted({session["task_name"] for session in sessions}),
        "sessions": sessions,
    }
