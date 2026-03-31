from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "data" / "benchmarks"

CONFIDENCE_REQUIRED_FIELDS = {
    "trial_id",
    "subject_id",
    "stimulus_strength",
    "correct_choice",
    "human_choice",
    "human_confidence",
    "rt_ms",
}
IGT_REQUIRED_FIELDS = {
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

BENCHMARK_STATES = {
    "scaffold_complete",
    "smoke_only",
    "acceptance_ready",
    "acceptance_pass",
    "acceptance_fail",
    "blocked_missing_external_bundle",
}


@dataclass(frozen=True)
class BenchmarkBundle:
    benchmark_id: str
    version: str
    manifest_path: str
    data_path: str
    source_type: str
    source_label: str
    benchmark_slice: str
    record_count: int
    status: str
    grouping_fields: list[str]
    default_split_unit: str
    external_bundle_preferred: bool
    smoke_test_only: bool
    is_synthetic: bool
    manifest: dict[str, Any]
    benchmark_state: str
    available_states: list[str]
    blockers: list[str]
    status_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkValidationResult:
    benchmark_id: str
    ok: bool
    manifest_path: str
    data_path: str
    record_count_declared: int
    record_count_observed: int
    missing_fields: list[str]
    malformed_lines: int
    split_unit: str
    grouping_fields: list[str]
    external_bundle_preferred: bool
    smoke_test_only: bool
    is_synthetic: bool
    warnings: list[str]
    benchmark_state: str
    available_states: list[str]
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkStatus:
    benchmark_id: str
    benchmark_state: str
    available_states: list[str]
    blockers: list[str]
    status_notes: list[str]
    acceptance_ready: bool
    scaffold_complete: bool
    smoke_only: bool
    external_bundle_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_root() -> Path:
    override = os.environ.get("SEGMENTUM_BENCHMARK_ROOT", "").strip()
    return Path(override).resolve() if override else DEFAULT_BENCHMARK_ROOT


def _bundle_paths(benchmark_id: str, *, root: Path | None = None) -> tuple[Path, Path]:
    active_root = (root or benchmark_root()).resolve()
    manifest_path = active_root / benchmark_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found for '{benchmark_id}' at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"benchmark_id", "record_count", "data_file", "source_type", "source_label"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Benchmark manifest '{benchmark_id}' is missing fields: {', '.join(missing)}")
    data_path = manifest_path.parent / str(manifest["data_file"])
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark data file not found for '{benchmark_id}' at {data_path}")
    return manifest_path, data_path


def list_benchmark_bundles(*, root: Path | None = None) -> list[BenchmarkBundle]:
    active_root = (root or benchmark_root()).resolve()
    if not active_root.exists():
        return []
    bundles: list[BenchmarkBundle] = []
    for manifest_path in sorted(active_root.glob("*/manifest.json")):
        bundles.append(load_benchmark_bundle(manifest_path.parent.name, root=active_root))
    return bundles


def load_benchmark_bundle(benchmark_id: str, *, root: Path | None = None) -> BenchmarkBundle:
    manifest_path, data_path = _bundle_paths(benchmark_id, root=root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    external_bundle_required = bool(manifest.get("acceptance_requires_external_bundle", manifest.get("external_bundle_preferred", False)))
    smoke_test_only = bool(manifest.get("smoke_test_only", False))
    source_type = str(manifest["source_type"])
    benchmark_slice = str(manifest.get("benchmark_slice", ""))
    available_states = ["scaffold_complete"]
    blockers: list[str] = []
    status_notes: list[str] = []
    if smoke_test_only or source_type != "external_bundle":
        available_states.append("smoke_only")
        if smoke_test_only:
            status_notes.append("Manifest marks the active bundle as smoke-test-only.")
        elif benchmark_slice.startswith("repo_curated"):
            status_notes.append("The active bundle is a repository-curated fixture rather than an imported external bundle.")
    acceptance_ready = source_type == "external_bundle" and not smoke_test_only
    if external_bundle_required and source_type != "external_bundle":
        available_states.append("blocked_missing_external_bundle")
        blockers.append(
            "Acceptance-grade evaluation requires an imported external bundle, but the active bundle is not external."
        )
        status_notes.append(
            f"Import an external bundle into '{(root or benchmark_root()).resolve() / benchmark_id}' or point SEGMENTUM_BENCHMARK_ROOT at a registry containing one."
        )
    if acceptance_ready:
        available_states.append("acceptance_ready")
    benchmark_state = "acceptance_ready" if acceptance_ready else "blocked_missing_external_bundle" if blockers else "smoke_only" if "smoke_only" in available_states else "scaffold_complete"
    return BenchmarkBundle(
        benchmark_id=str(manifest["benchmark_id"]),
        version=str(manifest.get("version", "0.0.0")),
        manifest_path=str(manifest_path),
        data_path=str(data_path),
        source_type=str(manifest["source_type"]),
        source_label=str(manifest["source_label"]),
        benchmark_slice=str(manifest.get("benchmark_slice", "")),
        record_count=int(manifest["record_count"]),
        status=str(manifest.get("status", manifest.get("source_type", "unknown"))),
        grouping_fields=[str(field) for field in manifest.get("grouping_fields", [])],
        default_split_unit=str(manifest.get("default_split_unit", "subject_id")),
        external_bundle_preferred=bool(manifest.get("external_bundle_preferred", False)),
        smoke_test_only=smoke_test_only,
        is_synthetic=bool(manifest.get("is_synthetic", False)),
        manifest=manifest,
        benchmark_state=benchmark_state,
        available_states=available_states,
        blockers=blockers,
        status_notes=status_notes,
    )


def expected_fields_for(benchmark_id: str) -> set[str]:
    if benchmark_id == "confidence_database":
        return set(CONFIDENCE_REQUIRED_FIELDS)
    if benchmark_id == "iowa_gambling_task":
        return set(IGT_REQUIRED_FIELDS)
    return set()


def validate_benchmark_bundle(benchmark_id: str, *, root: Path | None = None) -> BenchmarkValidationResult:
    bundle = load_benchmark_bundle(benchmark_id, root=root)
    expected_fields = expected_fields_for(bundle.benchmark_id)
    lines = Path(bundle.data_path).read_text(encoding="utf-8").splitlines()
    observed_records = 0
    malformed_lines = 0
    missing_fields: set[str] = set()
    warnings: list[str] = []
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        observed_records += 1
        if expected_fields:
            missing_fields.update(sorted(expected_fields - set(payload)))
    if bundle.record_count != observed_records:
        warnings.append("Declared record_count does not match observed record count.")
    manifest_fields = set(bundle.manifest.get("fields", []))
    if expected_fields and manifest_fields and not expected_fields <= manifest_fields:
        warnings.append("Manifest field list does not cover the full expected benchmark schema.")
    if bundle.external_bundle_preferred and bundle.source_type != "external_bundle":
        warnings.append("External bundle is preferred but the active bundle is not external.")
    if bundle.smoke_test_only:
        warnings.append("Active bundle is marked smoke-test-only.")
    ok = malformed_lines == 0 and not missing_fields and bundle.record_count == observed_records
    benchmark_state = bundle.benchmark_state
    blockers = list(bundle.blockers)
    available_states = list(bundle.available_states)
    if not ok:
        benchmark_state = "acceptance_fail"
        if "acceptance_fail" not in available_states:
            available_states.append("acceptance_fail")
        blockers.append(
            f"Bundle validation failed: missing_fields={sorted(missing_fields)}, malformed_lines={malformed_lines}, declared={bundle.record_count}, observed={observed_records}."
        )
    return BenchmarkValidationResult(
        benchmark_id=bundle.benchmark_id,
        ok=ok,
        manifest_path=bundle.manifest_path,
        data_path=bundle.data_path,
        record_count_declared=bundle.record_count,
        record_count_observed=observed_records,
        missing_fields=sorted(missing_fields),
        malformed_lines=malformed_lines,
        split_unit=bundle.default_split_unit,
        grouping_fields=list(bundle.grouping_fields),
        external_bundle_preferred=bundle.external_bundle_preferred,
        smoke_test_only=bundle.smoke_test_only,
        is_synthetic=bundle.is_synthetic,
        warnings=warnings,
        benchmark_state=benchmark_state,
        available_states=available_states,
        blockers=blockers,
    )


def benchmark_status(benchmark_id: str, *, root: Path | None = None) -> BenchmarkStatus:
    bundle = load_benchmark_bundle(benchmark_id, root=root)
    validation = validate_benchmark_bundle(benchmark_id, root=root)
    benchmark_state = validation.benchmark_state
    available_states = list(dict.fromkeys(validation.available_states))
    blockers = [item for item in validation.blockers if str(item).strip()]
    status_notes = list(bundle.status_notes)
    if not validation.ok:
        status_notes.append("Validation must pass before the bundle can be considered acceptance-ready.")
    return BenchmarkStatus(
        benchmark_id=bundle.benchmark_id,
        benchmark_state=benchmark_state,
        available_states=available_states,
        blockers=blockers,
        status_notes=status_notes,
        acceptance_ready=benchmark_state == "acceptance_ready",
        scaffold_complete=True,
        smoke_only="smoke_only" in available_states,
        external_bundle_required=bool(bundle.manifest.get("acceptance_requires_external_bundle", bundle.external_bundle_preferred)),
    )


def _locate_manifest_dir(source: Path) -> Path:
    if source.is_dir():
        candidate = source / "manifest.json"
        if candidate.exists():
            return source
        nested = list(source.glob("*/manifest.json"))
        if len(nested) == 1:
            return nested[0].parent
        raise FileNotFoundError(f"Could not locate a unique benchmark manifest under {source}")
    raise FileNotFoundError(f"Unsupported benchmark source path: {source}")


def _copy_tree(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)


def import_benchmark_bundle(
    source_path: str | Path,
    *,
    benchmark_id: str | None = None,
    destination_root: Path | None = None,
) -> BenchmarkBundle:
    source = Path(source_path).resolve()
    target_root = (destination_root or benchmark_root()).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if source.is_file() and source.suffix.lower() == ".zip":
            temp_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(source) as archive:
                archive.extractall(temp_dir.name)
            bundle_source = _locate_manifest_dir(Path(temp_dir.name))
        else:
            bundle_source = _locate_manifest_dir(source)

        manifest = json.loads((bundle_source / "manifest.json").read_text(encoding="utf-8"))
        resolved_benchmark_id = benchmark_id or str(manifest.get("benchmark_id", bundle_source.name))
        target_dir = target_root / resolved_benchmark_id
        _copy_tree(bundle_source, target_dir)

        copied_manifest_path = target_dir / "manifest.json"
        copied_manifest = json.loads(copied_manifest_path.read_text(encoding="utf-8"))
        copied_manifest["benchmark_id"] = resolved_benchmark_id
        copied_manifest.setdefault("version", "1.0.0")
        copied_manifest.setdefault("status", "external_bundle_imported")
        copied_manifest.setdefault("source_type", "external_bundle")
        copied_manifest.setdefault("source_label", source.name)
        copied_manifest.setdefault("data_file", copied_manifest.get("data_file", "data.jsonl"))
        copied_manifest.setdefault("external_bundle_preferred", True)
        copied_manifest.setdefault("smoke_test_only", False)
        copied_manifest.setdefault("is_synthetic", False)
        copied_manifest.setdefault("grouping_fields", ["session_id", "subject_id"])
        copied_manifest.setdefault("default_split_unit", "session_id")
        copied_manifest_path.write_text(json.dumps(copied_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        validation = validate_benchmark_bundle(resolved_benchmark_id, root=target_root)
        if not validation.ok:
            shutil.rmtree(target_dir)
            raise ValueError(
                f"Imported benchmark bundle '{resolved_benchmark_id}' failed validation: "
                f"missing_fields={validation.missing_fields}, malformed_lines={validation.malformed_lines}, "
                f"declared={validation.record_count_declared}, observed={validation.record_count_observed}"
            )
        return load_benchmark_bundle(resolved_benchmark_id, root=target_root)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
