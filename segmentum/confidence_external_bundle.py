from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfidenceBundleBuildReport:
    source_dir: str
    destination_root: str
    bundle_dir: str
    data_path: str
    manifest_path: str
    included_files: int
    skipped_files: int
    record_count: int
    subject_count: int
    session_count: int
    skipped_reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CHOICE_MAP = {
    "1": "left",
    "2": "right",
    "left": "left",
    "right": "right",
    "old": "left",
    "new": "right",
    "present": "left",
    "absent": "right",
    "yes": "left",
    "no": "right",
}

_RT_COLUMNS = ("RT_decConf", "RT_dec", "RT_conf")
_MAGNITUDE_SPECS: tuple[tuple[str, bool], ...] = (
    ("Orientation", False),
    ("Difference", False),
    ("ContrastOrTilt", False),
    ("Dot_diff", False),
    ("Contrast", False),
    ("Difficulty", True),
    ("Stimulus Peak Velocity (deg/s)", False),
)
_SPLIT_NAMES = ("train", "validation", "heldout")


def _normalize_header_value(value: str | None) -> str:
    return str(value or "").strip()


def _canonical_choice(value: Any) -> str | None:
    text = str(value).strip().strip('"').lower()
    if text in _CHOICE_MAP:
        return _CHOICE_MAP[text]
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric == 1.0:
        return "left"
    if numeric == 2.0:
        return "right"
    return None


def _parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except ValueError:
        return None


def _parse_rt_ms(row: dict[str, Any]) -> int | None:
    for column in _RT_COLUMNS:
        raw = _parse_float(row.get(column))
        if raw is None:
            continue
        return int(round(raw * 1000.0)) if raw <= 20.0 else int(round(raw))
    return None


def _strength_spec(row: dict[str, Any]) -> tuple[str, bool, float] | None:
    for column, invert in _MAGNITUDE_SPECS:
        value = _parse_float(row.get(column))
        if value is None:
            continue
        return column, invert, abs(value)
    return None


def _session_suffix(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("session", "Session", "Task", "Condition", "Day", "Run", "Block"):
        value = _normalize_header_value(row.get(key))
        if value:
            parts.append(f"{key}={value}")
    return "|".join(parts) if parts else "session=default"


def _dataset_name(data_file: Path) -> str:
    return data_file.stem.removeprefix("data_")


def _deterministic_split(subject_id: str) -> str:
    digest = hashlib.sha256(subject_id.encode("utf-8")).digest()
    return _SPLIT_NAMES[int.from_bytes(digest[:2], byteorder="big") % len(_SPLIT_NAMES)]


def build_confidence_external_bundle(
    source_dir: str | Path,
    destination_root: str | Path,
    *,
    benchmark_id: str = "confidence_database",
) -> ConfidenceBundleBuildReport:
    source_path = Path(source_dir).resolve()
    destination_root_path = Path(destination_root).resolve()
    bundle_dir = destination_root_path / benchmark_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_path = bundle_dir / "confidence_database_external.jsonl"
    manifest_path = bundle_dir / "manifest.json"

    included_files = 0
    skipped_files = 0
    skipped_reasons: dict[str, str] = {}
    record_count = 0
    subject_ids: set[str] = set()
    session_ids: set[str] = set()

    with data_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for csv_path in sorted(source_path.glob("data_*.csv")):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    skipped_files += 1
                    skipped_reasons[csv_path.name] = "missing_header"
                    continue
                columns = {_normalize_header_value(name) for name in reader.fieldnames}
                required = {"Subj_idx", "Stimulus", "Response", "Confidence"}
                if not required <= columns:
                    skipped_files += 1
                    skipped_reasons[csv_path.name] = "missing_required_columns"
                    continue
                preview_rows = list(reader)
            valid_rows: list[dict[str, Any]] = []
            confidence_values: list[float] = []
            strength_values: list[float] = []
            dataset_name = _dataset_name(csv_path)
            for row_index, row in enumerate(preview_rows, start=1):
                correct_choice = _canonical_choice(row.get("Stimulus"))
                human_choice = _canonical_choice(row.get("Response"))
                confidence_raw = _parse_float(row.get("Confidence"))
                rt_ms = _parse_rt_ms(row)
                subject_raw = _normalize_header_value(row.get("Subj_idx"))
                strength_spec = _strength_spec(row)
                if not correct_choice or not human_choice or confidence_raw is None or rt_ms is None or not subject_raw or strength_spec is None:
                    continue
                _, _, strength_value = strength_spec
                confidence_values.append(confidence_raw)
                strength_values.append(strength_value)
                valid_rows.append(
                    {
                        "row": row,
                        "row_index": row_index,
                        "correct_choice": correct_choice,
                        "human_choice": human_choice,
                        "confidence_raw": confidence_raw,
                        "rt_ms": rt_ms,
                        "subject_raw": subject_raw,
                        "strength_spec": strength_spec,
                        "dataset_name": dataset_name,
                    }
                )
            if not valid_rows:
                skipped_files += 1
                skipped_reasons[csv_path.name] = "no_rows_match_confidence_bundle_schema"
                continue
            included_files += 1
            confidence_min = min(confidence_values)
            confidence_max = max(confidence_values)
            strength_max = max(max(strength_values), 1e-9)
            for item in valid_rows:
                row = item["row"]
                subject_id = f"{item['dataset_name']}::{item['subject_raw']}"
                session_id = f"{subject_id}::{_session_suffix(row)}"
                _, invert_strength, raw_strength = item["strength_spec"]
                normalized_strength = raw_strength / strength_max
                if invert_strength:
                    normalized_strength = 1.0 - normalized_strength
                normalized_strength = max(0.0, min(1.0, normalized_strength))
                if confidence_max > confidence_min:
                    human_confidence = (item["confidence_raw"] - confidence_min) / (confidence_max - confidence_min)
                else:
                    human_confidence = 0.5
                signed_strength = normalized_strength if item["correct_choice"] == "right" else -normalized_strength
                payload = {
                    "trial_id": f"{item['dataset_name']}::{item['row_index']}",
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "split": _deterministic_split(subject_id),
                    "stimulus_strength": round(signed_strength, 6),
                    "correct_choice": item["correct_choice"],
                    "human_choice": item["human_choice"],
                    "human_confidence": round(max(0.0, min(1.0, human_confidence)), 6),
                    "rt_ms": item["rt_ms"],
                    "source_dataset": item["dataset_name"],
                    "source_file": csv_path.name,
                }
                out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                record_count += 1
                subject_ids.add(subject_id)
                session_ids.add(session_id)

    manifest = {
        "benchmark_id": benchmark_id,
        "version": "1.0.0",
        "status": "external_bundle_imported",
        "benchmark_slice": "confidence_database_external_bundle",
        "source_type": "external_bundle",
        "source_label": source_path.name,
        "grouping_fields": ["session_id", "subject_id"],
        "default_split_unit": "subject_id",
        "external_bundle_preferred": True,
        "acceptance_requires_external_bundle": True,
        "smoke_test_only": False,
        "is_synthetic": False,
        "data_file": data_path.name,
        "description": "External confidence benchmark bundle converted from the local Confidence Database raw directory.",
        "record_count": record_count,
        "fields": [
            "trial_id",
            "subject_id",
            "session_id",
            "split",
            "stimulus_strength",
            "correct_choice",
            "human_choice",
            "human_confidence",
            "rt_ms",
            "source_dataset",
            "source_file",
        ],
        "notes": [
            "Converted from local Confidence Database raw CSV files.",
            "Only files/rows that could be mapped conservatively into the current confidence benchmark schema were included.",
            "Binary stimulus/response labels were canonicalized onto left/right labels for adapter compatibility.",
            "Split assignments were precomputed deterministically from subject_id to keep heldout evaluation subject-disjoint.",
            "Files with incompatible schemas were skipped and listed in conversion_summary.",
        ],
        "conversion_summary": {
            "included_files": included_files,
            "skipped_files": skipped_files,
            "subject_count": len(subject_ids),
            "session_count": len(session_ids),
            "skipped_reasons": skipped_reasons,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ConfidenceBundleBuildReport(
        source_dir=str(source_path),
        destination_root=str(destination_root_path),
        bundle_dir=str(bundle_dir),
        data_path=str(data_path),
        manifest_path=str(manifest_path),
        included_files=included_files,
        skipped_files=skipped_files,
        record_count=record_count,
        subject_count=len(subject_ids),
        session_count=len(session_ids),
        skipped_reasons=skipped_reasons,
    )
