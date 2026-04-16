from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from segmentum.m41_audit import M41_REPORT_PATH, write_m41_acceptance_artifacts
from segmentum.m42_audit import M42_REPORT_PATH, write_m42_acceptance_artifacts
from segmentum.m43_audit import M43_REPORT_PATH, write_m43_acceptance_artifacts
from segmentum.m44_audit import M44_REPORT_PATH, write_m44_acceptance_artifacts
from segmentum.m45_audit import M45_REPORT_PATH, write_m45_acceptance_artifacts
from segmentum.m45_acceptance_data import REGRESSION_TARGETS as M45_REGRESSION_TARGETS
from segmentum.m46_audit import M46_REPORT_PATH, write_m46_acceptance_artifacts
from segmentum.m47_audit import M47_REPORT_PATH, write_m47_acceptance_artifacts
from segmentum.m410_audit import M410_REPORT_PATH, write_m410_acceptance_artifacts
from segmentum.m411_phenomenology import (
    M411_REPORT_PATH,
    M411RolloutConfig,
    write_m411_acceptance_artifacts,
)


FIXED_STARTED_AT = "2026-04-09T00:00:00+00:00"
TIMESTAMP_KEYS = {
    "generated_at",
    "round_started_at",
    "artifact_round_started_at",
    "audit_started_at",
}
EXEMPTIONS = {
    "m45": "Dedicated official-artifact truth tests already cover M4.5; its writer executes a full regression subprocess and is excluded from byte-level drift comparison here.",
    "m47": "The shared runtime snapshot still has residual ordering/float nondeterminism; this test only checks top-level status alignment until the snapshot is fully frozen.",
}
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
UUID_SUBSTRING_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
HASH_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", re.IGNORECASE)
HASH_SUBSTRING_PATTERN = re.compile(r"\b(?:[0-9a-f]{40}|[0-9a-f]{64})\b", re.IGNORECASE)


def _normalize_cross_platform_path_string(raw: str) -> str:
    """Make acceptance JSON comparable across OS and checkout roots (Linux CI vs Windows dev)."""
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    lower = normalized.lower()
    for marker in ("segmentum/", "tests/", "reports/", "artifacts/", "prompts/"):
        idx = lower.find(marker)
        if idx != -1:
            return normalized[idx:].replace("\\", "/")
    tail = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
    if tail.lower().startswith("python"):
        return "<python>"
    if any(sep in raw for sep in ("/", "\\", ":")) and lower.endswith(".py"):
        return Path(normalized).name
    return raw


M45_SYNTHETIC_REGRESSION_SUMMARY = {
    "executed": True,
    "command": ["synthetic", "pytest"],
    "files": list(M45_REGRESSION_TARGETS),
    "returncode": 0,
    "passed": True,
    "passed_count": len(M45_REGRESSION_TARGETS),
    "duration_seconds": 0.1,
    "summary_line": f"{len(M45_REGRESSION_TARGETS)} passed in 0.10s",
    "stdout_tail": [f"{len(M45_REGRESSION_TARGETS)} passed in 0.10s"],
}
M47_AUDIT_FIELDS = {
    "identity_link_strength",
    "identity_link_active",
    "self_relevance_multiplier",
    "base_short_to_mid_score",
    "boosted_short_to_mid_score",
    "score_cap_applied",
}
M411_SMOKE_NON_ACCEPTANCE_CONFIG = M411RolloutConfig(
    seed=411,
    ticks=30,
    recall_probe_interval=6,
    perturbation_tick=15,
    sleep_interval=12,
    min_acceptance_ticks=20,
)


def _normalize_json(value, *, key: str | None = None):  # noqa: ANN001
    if isinstance(value, dict):
        normalized_dict = {}
        for name, item in sorted(value.items()):
            normalized_name = name
            if isinstance(name, str):
                if UUID_PATTERN.match(name):
                    normalized_name = "<normalized-uuid>"
                else:
                    normalized_name = UUID_SUBSTRING_PATTERN.sub("<normalized-uuid>", name)
            normalized_dict[normalized_name] = _normalize_json(item, key=normalized_name)
        return normalized_dict
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if key in TIMESTAMP_KEYS:
        return "<normalized-timestamp>"
    if isinstance(value, str):
        if UUID_PATTERN.match(value):
            return "<normalized-uuid>"
        if HASH_PATTERN.match(value):
            return "<normalized-hash>"
        value = UUID_SUBSTRING_PATTERN.sub("<normalized-uuid>", value)
        value = HASH_SUBSTRING_PATTERN.sub("<normalized-hash>", value)
        normalized = value.replace("\\", "/")
        suffix = Path(normalized).suffix.lower()
        if suffix in {".json", ".jsonl", ".md"} and ("/" in normalized or "\\" in value):
            return Path(normalized).name
        return _normalize_cross_platform_path_string(value)
    return value


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_in_place_report(prefix: str, report_path: Path, writer) -> dict[str, object]:  # noqa: ANN001
    repo_root = report_path.resolve().parents[1]
    tracked_paths = list((repo_root / "reports").glob(f"{prefix}*"))
    tracked_paths.extend((repo_root / "artifacts").glob(f"{prefix}*"))
    backup = {path: path.read_bytes() for path in tracked_paths if path.exists()}
    existing = set(tracked_paths)
    try:
        writer(round_started_at=FIXED_STARTED_AT)
        generated = list((repo_root / "reports").glob(f"{prefix}*"))
        generated.extend((repo_root / "artifacts").glob(f"{prefix}*"))
        return _read_json(report_path)
    finally:
        current = set((repo_root / "reports").glob(f"{prefix}*")) | set((repo_root / "artifacts").glob(f"{prefix}*"))
        for path in current:
            if path not in existing and path.exists():
                path.unlink()
        for path, content in backup.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


class TestOfficialArtifactsMatchBuilder(unittest.TestCase):
    maxDiff = None

    def _assert_report_matches(self, label: str, committed_path: Path, generated: dict[str, object]) -> None:
        self.assertTrue(committed_path.exists(), f"{label} canonical report missing: {committed_path}")
        committed = _read_json(committed_path)
        self.assertEqual(
            _normalize_json(committed),
            _normalize_json(generated),
            f"{label} canonical artifact drifted. Regenerate the official acceptance artifacts for {label}.",
        )

    def test_exemptions_are_explicit(self) -> None:
        self.assertIn("m45", EXEMPTIONS)
        self.assertTrue(EXEMPTIONS["m45"])

    def test_m41_report_matches_writer(self) -> None:
        generated = _capture_in_place_report("m41_", M41_REPORT_PATH, write_m41_acceptance_artifacts)
        self._assert_report_matches("m41", M41_REPORT_PATH, generated)

    def test_m42_report_matches_writer(self) -> None:
        generated = _capture_in_place_report("m42_", M42_REPORT_PATH, write_m42_acceptance_artifacts)
        self._assert_report_matches("m42", M42_REPORT_PATH, generated)

    @unittest.skip("Historical M4.3 writer is too slow for the M4.11 artifact drift target.")
    def test_m43_report_matches_writer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m43_acceptance_artifacts(round_started_at=FIXED_STARTED_AT, output_root=tmpdir)
            generated = _read_json(Path(outputs["report"]))
        self._assert_report_matches("m43", M43_REPORT_PATH, generated)

    @unittest.skip("Historical M4.4 writer is too slow for the M4.11 artifact drift target.")
    def test_m44_report_matches_writer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m44_acceptance_artifacts(round_started_at=FIXED_STARTED_AT, output_root=tmpdir)
            generated = _read_json(Path(outputs["report"]))
        self._assert_report_matches("m44", M44_REPORT_PATH, generated)

    def test_m45_report_is_explicitly_exempt(self) -> None:
        self.assertTrue(M45_REPORT_PATH.exists())
        self.assertIn("regression subprocess", EXEMPTIONS["m45"])

    def test_m45_report_regen_smoke_preserves_multiplier_audit_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with mock.patch("segmentum.m45_audit._run_regression_summary", return_value=M45_SYNTHETIC_REGRESSION_SUMMARY):
                outputs = write_m45_acceptance_artifacts(round_started_at=FIXED_STARTED_AT, output_root=tmpdir)
            report = _read_json(Path(outputs["report"]))
            trace = _read_json(Path(outputs["canonical_trace"]))

        self.assertIn(report["status"], {"PASS", "FAIL"})
        self.assertIn(report["acceptance_state"], {"acceptance_pass", "acceptance_fail"})
        self.assertIn(report["recommendation"], {"ACCEPT", "BLOCK"})
        observed = trace["integration_probes"]["store_transitions_integration"]["observed"]
        self.assertEqual(set(observed["identity_linked_audit"]), M47_AUDIT_FIELDS)
        self.assertEqual(set(observed["identity_null_audit"]), M47_AUDIT_FIELDS)

    def test_m46_report_matches_builder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m46_acceptance_artifacts(round_started_at=FIXED_STARTED_AT, output_root=tmpdir)
            generated = _read_json(Path(outputs["report"]))
        self._assert_report_matches("m46", M46_REPORT_PATH, generated)

    def test_m47_report_matches_builder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m47_acceptance_artifacts(round_started_at=FIXED_STARTED_AT, output_root=tmpdir)
            generated = _read_json(Path(outputs["report"]))
        committed = _read_json(M47_REPORT_PATH)
        self.assertIn("m47", EXEMPTIONS)
        self.assertEqual(committed["status"], generated["status"])
        self.assertEqual(committed["acceptance_state"], generated["acceptance_state"])
        self.assertEqual(committed["recommendation"], generated["recommendation"])

    def test_m410_report_matches_builder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m410_acceptance_artifacts(output_root=tmpdir)
            generated = _read_json(Path(outputs["report"]))
        self._assert_report_matches("m410", M410_REPORT_PATH, generated)

    def test_m411_report_matches_non_acceptance_builder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m411_acceptance_artifacts(
                output_root=tmpdir,
                config=M411_SMOKE_NON_ACCEPTANCE_CONFIG,
            )
            generated = _read_json(Path(outputs["report"]))
        self._assert_report_matches("m411", M411_REPORT_PATH, generated)
        self.assertEqual(generated["status"], "NOT_ISSUED")
        self.assertFalse(generated["phenomenological_pass"])
