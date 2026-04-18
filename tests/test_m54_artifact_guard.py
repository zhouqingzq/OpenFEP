from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

from segmentum.dialogue.validation.constants import M54_ACCEPTANCE_RULES_VERSION


REQUIRED_SUMMARY_FIELDS = {
    "artifact_rules_version",
    "classifier_evidence_tier",
    "baseline_c_builder",
    "surface_ablation_gate",
    "formal_blockers",
}
CONSISTENT_SUMMARY_FIELDS = (
    "artifact_rules_version",
    "hard_pass",
    "formal_acceptance_eligible",
    "overall_conclusion",
    "classifier_evidence_tier",
    "baseline_c_builder",
    "formal_blockers",
)
BASELINE_C_INPUT_SCOPE = "leave_one_out_population_train_and_profile_data"
HISTORICAL_MISSING = "historical_missing"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_ls_files(*patterns: str) -> list[Path]:
    root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tracked_m54_summary_artifacts() -> list[Path]:
    return _git_ls_files(
        "artifacts/m54_validation*/aggregate_report.json",
        "artifacts/m54_validation*/m54_acceptance.json",
    )


def _tracked_m54_per_user_artifacts() -> list[Path]:
    return _git_ls_files("artifacts/m54_validation*/per_user/*_report.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rules_version(payload: dict[str, object]) -> object:
    rules_version = payload.get("artifact_rules_version")
    if rules_version is None and isinstance(payload.get("acceptance_rules"), dict):
        rules_version = payload["acceptance_rules"].get("version")
    return rules_version


def _artifact_root(path: Path) -> Path:
    return path.parents[1] if path.parent.name == "per_user" else path.parent


def _is_formal_path(path: Path) -> bool:
    return _artifact_root(path).name.startswith("m54_validation_formal")


class TestM54ArtifactGuard(unittest.TestCase):
    def test_tracked_m54_summary_artifacts_use_current_stop_bleed_schema(self) -> None:
        paths = _tracked_m54_summary_artifacts()
        self.assertTrue(paths, "expected tracked M5.4 summary artifacts")
        for path in paths:
            with self.subTest(path=str(path.relative_to(_repo_root()))):
                payload = _load(path)
                self.assertEqual(_rules_version(payload), M54_ACCEPTANCE_RULES_VERSION)
                self.assertTrue(REQUIRED_SUMMARY_FIELDS <= set(payload))
                self.assertIsInstance(payload["surface_ablation_gate"], dict)
                self.assertIsInstance(payload["formal_blockers"], list)
                if _is_formal_path(path):
                    self.assertTrue(payload.get("formal_requested"))
                    for gate_name in ("baseline_c_gate", "surface_ablation_gate", "diagnostic_trace_gate"):
                        gate = payload.get(gate_name)
                        self.assertIsInstance(gate, dict)
                        self.assertTrue(gate.get("formal_requested"), gate_name)

    def test_acceptance_matches_aggregate_report(self) -> None:
        roots = sorted({_artifact_root(path) for path in _tracked_m54_summary_artifacts()})
        for root in roots:
            aggregate_path = root / "aggregate_report.json"
            acceptance_path = root / "m54_acceptance.json"
            if not acceptance_path.exists():
                continue
            with self.subTest(path=str(root.relative_to(_repo_root()))):
                aggregate = _load(aggregate_path)
                acceptance = _load(acceptance_path)
                for field in CONSISTENT_SUMMARY_FIELDS:
                    self.assertEqual(acceptance.get(field), aggregate.get(field), field)

    def test_formal_per_user_artifacts_carry_v4_evidence_fields(self) -> None:
        paths = [path for path in _tracked_m54_per_user_artifacts() if _is_formal_path(path)]
        self.assertTrue(paths, "expected tracked formal M5.4 per-user artifacts")
        for path in paths:
            with self.subTest(path=str(path.relative_to(_repo_root()))):
                payload = _load(path)
                aggregate = payload.get("aggregate")
                self.assertIsInstance(aggregate, dict)
                self.assertEqual(aggregate.get("metric_version"), M54_ACCEPTANCE_RULES_VERSION)
                self.assertEqual(aggregate.get("artifact_rules_version"), M54_ACCEPTANCE_RULES_VERSION)
                self.assertTrue(aggregate.get("formal_requested"))
                required_population = max(0, int(aggregate.get("required_users", 0) or 0) - 1)
                for strategy_name, strategy in (payload.get("per_strategy") or {}).items():
                    if not isinstance(strategy, dict) or strategy.get("skipped"):
                        continue
                    builder = strategy.get("baseline_c_builder")
                    input_scope = strategy.get("baseline_c_input_scope")
                    self.assertIsNotNone(builder, strategy_name)
                    self.assertIsNotNone(input_scope, strategy_name)
                    if builder == "population_average_full_implant":
                        self.assertEqual(input_scope, BASELINE_C_INPUT_SCOPE)
                        self.assertTrue(strategy.get("baseline_c_leave_one_out"), strategy_name)
                        self.assertEqual(
                            int(strategy.get("baseline_c_population_excluded_uid", -1)),
                            int(payload.get("user_uid", -2)),
                        )
                        self.assertGreaterEqual(
                            int(strategy.get("baseline_c_population_user_count", 0) or 0),
                            required_population,
                        )
                    else:
                        self.assertEqual(builder, HISTORICAL_MISSING)
                        self.assertEqual(input_scope, HISTORICAL_MISSING)

    def test_repo_fixture_classifier_artifacts_fail_closed(self) -> None:
        for path in _tracked_m54_summary_artifacts():
            with self.subTest(path=str(path.relative_to(_repo_root()))):
                payload = _load(path)
                classifier_gate = payload.get("classifier_gate")
                self.assertIsInstance(classifier_gate, dict)
                origin = str(classifier_gate.get("dataset_origin", ""))
                tier = str(
                    payload.get(
                        "classifier_evidence_tier",
                        classifier_gate.get("classifier_evidence_tier", "repo_fixture_smoke"),
                    )
                )
                fixture_like = tier == "repo_fixture_smoke" or any(
                    marker in origin.lower()
                    for marker in ("codex_authored", "fixture", "synthetic", "smoke", "toy")
                )
                if fixture_like:
                    self.assertFalse(classifier_gate.get("passed_3class_gate", False))
                    self.assertFalse(payload.get("formal_acceptance_eligible", False))
                    self.assertFalse(payload.get("hard_pass", False))


if __name__ == "__main__":
    unittest.main()
