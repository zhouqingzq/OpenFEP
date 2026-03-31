from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IgtBundleBuildReport:
    source_dir: str
    destination_root: str
    bundle_dir: str
    data_path: str
    manifest_path: str
    included_subjects: int
    skipped_subjects: int
    record_count: int
    subject_count: int
    skipped_reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPLIT_NAMES = ("train", "validation", "heldout")


def _deterministic_split(subject_id: str) -> str:
    digest = hashlib.sha256(subject_id.encode("utf-8")).digest()
    return _SPLIT_NAMES[int.from_bytes(digest[:2], byteorder="big") % len(_SPLIT_NAMES)]


def _parse_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return int(round(float(text)))
    except ValueError:
        return None


def build_igt_external_bundle(
    source_dir: str | Path,
    destination_root: str | Path,
    *,
    benchmark_id: str = "iowa_gambling_task",
) -> IgtBundleBuildReport:
    source_path = Path(source_dir).resolve()
    destination_root_path = Path(destination_root).resolve()
    bundle_dir = destination_root_path / benchmark_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_path = bundle_dir / "iowa_gambling_task_external.jsonl"
    manifest_path = bundle_dir / "manifest.json"

    included_subjects = 0
    skipped_subjects = 0
    skipped_reasons: dict[str, str] = {}
    record_count = 0
    subject_ids: set[str] = set()

    with data_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for subject_dir in sorted(path for path in source_path.iterdir() if path.is_dir() and path.name.startswith("s-")):
            csv_path = subject_dir / "IGT.csv"
            if not csv_path.exists():
                skipped_subjects += 1
                skipped_reasons[subject_dir.name] = "missing_igt_csv"
                continue
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    skipped_subjects += 1
                    skipped_reasons[subject_dir.name] = "missing_header"
                    continue
                columns = {str(name).strip() for name in reader.fieldnames}
                required = {"iteration", "decision", "win", "lose"}
                if not required <= columns:
                    skipped_subjects += 1
                    skipped_reasons[subject_dir.name] = "missing_required_columns"
                    continue
                rows = list(reader)
            split = _deterministic_split(subject_dir.name)
            valid_rows = 0
            for row in rows:
                trial_index = _parse_int(row.get("iteration"))
                reward = _parse_int(row.get("win"))
                lose_raw = _parse_int(row.get("lose"))
                decision = str(row.get("decision", "")).strip().upper()
                if trial_index is None or reward is None or lose_raw is None or decision not in {"A", "B", "C", "D"}:
                    continue
                penalty = -abs(lose_raw) if lose_raw else 0
                net_outcome = reward + penalty
                payload = {
                    "trial_id": f"{subject_dir.name}::trial_{trial_index:03d}",
                    "subject_id": subject_dir.name,
                    "deck": decision,
                    "reward": reward,
                    "penalty": penalty,
                    "net_outcome": net_outcome,
                    "advantageous": decision in {"C", "D"},
                    "trial_index": trial_index,
                    "split": split,
                    "source_file": str(csv_path.relative_to(source_path)).replace("\\", "/"),
                }
                out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                record_count += 1
                valid_rows += 1
                subject_ids.add(subject_dir.name)
            if valid_rows == 0:
                skipped_subjects += 1
                skipped_reasons[subject_dir.name] = "no_valid_igt_rows"
            else:
                included_subjects += 1

    manifest = {
        "benchmark_id": benchmark_id,
        "version": "1.0.0",
        "status": "external_bundle_imported",
        "benchmark_slice": "iowa_gambling_task_external_bundle",
        "source_type": "external_bundle",
        "source_label": source_path.name,
        "grouping_fields": ["subject_id"],
        "default_split_unit": "subject_id",
        "external_bundle_preferred": True,
        "acceptance_requires_external_bundle": True,
        "smoke_test_only": False,
        "is_synthetic": False,
        "data_file": data_path.name,
        "description": "External Iowa Gambling Task benchmark bundle converted from local participant IGT.csv files.",
        "record_count": record_count,
        "fields": [
            "trial_id",
            "subject_id",
            "deck",
            "reward",
            "penalty",
            "net_outcome",
            "advantageous",
            "trial_index",
            "split",
            "source_file",
        ],
        "notes": [
            "Converted from local subject-level IGT.csv files.",
            "Penalty values are normalized to signed negative values for benchmark compatibility.",
            "Split assignments were precomputed deterministically from subject_id.",
        ],
        "conversion_summary": {
            "included_subjects": included_subjects,
            "skipped_subjects": skipped_subjects,
            "subject_count": len(subject_ids),
            "skipped_reasons": skipped_reasons,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return IgtBundleBuildReport(
        source_dir=str(source_path),
        destination_root=str(destination_root_path),
        bundle_dir=str(bundle_dir),
        data_path=str(data_path),
        manifest_path=str(manifest_path),
        included_subjects=included_subjects,
        skipped_subjects=skipped_subjects,
        record_count=record_count,
        subject_count=len(subject_ids),
        skipped_reasons=skipped_reasons,
    )
