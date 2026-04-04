from __future__ import annotations

"""Synthetic holdout dataset loader retained under a legacy module name.

The `external` terminology in this module is historical. The bundled records
here are same-framework synthetic holdouts used for sidecar diagnostics, not
independent external human benchmark data.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, DecisionLogRecord


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAME_FRAMEWORK_DATA_ROOT = ROOT / "data" / "m41_external"
DEFAULT_EXTERNAL_DATA_ROOT = DEFAULT_SAME_FRAMEWORK_DATA_ROOT


def _iter_json_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    raise ValueError(f"Unsupported external payload format in {path}")


def normalize_same_framework_holdout_record(record: dict[str, Any]) -> dict[str, Any]:
    if str(record.get("schema_version", "")).startswith("m4.decision_log"):
        return DecisionLogRecord.from_dict(record).to_dict()
    if str(record.get("schema_version")) not in {
        "m41.external.event.v1",
        "m41.external.event.v2",
        "m41.synthetic_holdout.event.v1",
        "m41.synthetic_holdout.event.v2",
    }:
        raise ValueError(f"Unsupported same-framework holdout record schema: {record.get('schema_version')}")

    payload = {
        "schema_version": "m4.decision_log.v3",
        "tick": int(record.get("tick", 0)),
        "timestamp": str(record.get("timestamp", "2026-02-01T00:00:00+00:00")),
        "seed": int(record.get("seed", 0)),
        "task_context": {
            "phase": str(record.get("task_name", "synthetic_holdout_task")),
            "subject_id": str(record.get("subject_id", "")),
            "session_id": str(record.get("session_id", "")),
            "source_name": str(record.get("source_name", "")),
            "generator_id": str(record.get("generator_id", "same_framework_holdout_v1")),
        },
        "percept_summary": dict(record.get("percept_summary", {})),
        "observation_evidence": dict(record.get("observation_evidence", {})),
        "prediction_error_vector": dict(record.get("prediction_error_vector", {})),
        "attention_allocation": dict(record.get("attention_allocation", {})),
        "candidate_actions": list(record.get("candidate_actions", [])),
        "parameter_snapshot": {},
        "resource_state": dict(record.get("resource_state", {})),
        "internal_confidence": float(record.get("internal_confidence", 0.0)),
        "selected_action": str(record.get("selected_action", "")),
        "result_feedback": dict(record.get("result_feedback", {})),
        "model_update": dict(record.get("model_update", {})),
        "prediction_error": float(record.get("prediction_error", 0.0)),
        "update_magnitude": float(record.get("update_magnitude", 0.0)),
    }
    return DecisionLogRecord.from_dict(payload).to_dict()


def normalize_external_record(record: dict[str, Any]) -> dict[str, Any]:
    return normalize_same_framework_holdout_record(record)


def inference_path_blindness_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [DecisionLogRecord.from_dict(record).to_dict() for record in records]
    snapshot_values = [record.get("parameter_snapshot") for record in normalized]
    leaks = [
        index
        for index, record in enumerate(normalized)
        if "ground_truth_parameters" in record or "ground_truth_profile" in record
    ]
    empty_snapshots = sum(1 for snapshot in snapshot_values if snapshot == {})
    return {
        "record_count": len(normalized),
        "parameter_snapshot_empty_records": empty_snapshots,
        "parameter_snapshot_empty_rate": round((empty_snapshots / len(normalized)) if normalized else 0.0, 6),
        "ground_truth_visible_record_indexes": leaks,
        "ground_truth_visible": bool(leaks),
        "inference_path_blinded": not leaks and all(snapshot == {} for snapshot in snapshot_values),
    }


def assert_inference_path_blinded(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = inference_path_blindness_evidence(records)
    if not evidence["inference_path_blinded"]:
        raise ValueError(f"Same-framework synthetic holdout inference path is not blinded: {evidence}")
    return evidence


def load_same_framework_behavior_dataset(root: Path | str | None = None) -> dict[str, Any]:
    dataset_root = Path(root or DEFAULT_SAME_FRAMEWORK_DATA_ROOT).resolve()
    files = sorted(path for path in dataset_root.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend([{**payload, "_source_file": path.name} for payload in _iter_json_payloads(path)])

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("source_name", "")), str(row.get("subject_id", "")), str(row.get("session_id", "")))
        grouped[key].append(row)

    sessions: list[dict[str, Any]] = []
    for (source_name, subject_id, session_id), session_rows in sorted(grouped.items()):
        ordered = sorted(session_rows, key=lambda item: (int(item.get("tick", 0)), str(item.get("timestamp", ""))))
        first = ordered[0]
        ground_truth = first.get("ground_truth_parameters")
        sessions.append(
            {
                "source_name": source_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "task_name": str(first.get("task_name", "")),
                "split": str(first.get("split", "heldout")),
                "source_file": str(first.get("_source_file", "")),
                "profile_label": str(first.get("ground_truth_profile", "")),
                "generator_id": str(first.get("generator_id", "same_framework_holdout_v1")),
                "ground_truth_parameters": dict(ground_truth) if isinstance(ground_truth, dict) else None,
                "records": [normalize_same_framework_holdout_record(row) for row in ordered],
                "record_count": len(ordered),
            }
        )

    parameter_defaults = CognitiveStyleParameters().to_dict()
    for session in sessions:
        if session["ground_truth_parameters"] is None:
            session["ground_truth_is_default"] = False
        else:
            session["ground_truth_is_default"] = session["ground_truth_parameters"] == parameter_defaults

    return {
        "schema_version": "m41.synthetic_holdout.dataset.v1",
        "legacy_schema_version": "m41.external.dataset.v1",
        "benchmark_scope": "same-framework synthetic holdout dataset for sidecar diagnostics",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "validation_type": "synthetic_holdout_same_framework",
        "generator_family": "same_framework_synthetic_holdout",
        "dataset_root": str(dataset_root),
        "record_count": len(rows),
        "session_count": len(sessions),
        "sources": sorted({session["source_name"] for session in sessions}),
        "profiles": sorted({session["profile_label"] for session in sessions}),
        "generator_ids": sorted({session["generator_id"] for session in sessions}),
        "sessions": sessions,
    }


def load_external_behavior_dataset(root: Path | str | None = None) -> dict[str, Any]:
    return load_same_framework_behavior_dataset(root=root)
