from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m234_audit import (
    M234_GATES,
    M234_PREPARATION_PATH,
    M234_RATIONALE_PATH,
    M234_REGRESSIONS,
    M234_SPEC_PATH,
    M234_TESTS,
    SCHEMA_VERSION,
    SEED_SET,
    preparation_manifest,
)


class TestM234AuditPreparation(unittest.TestCase):
    def test_manifest_freezes_expected_audit_contract(self) -> None:
        manifest = preparation_manifest()

        self.assertEqual(manifest["milestone_id"], "M2.34")
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(tuple(manifest["seed_set"]), SEED_SET)
        self.assertEqual(tuple(manifest["tests"]["milestone"]), M234_TESTS)
        self.assertEqual(tuple(manifest["tests"]["regressions"]), M234_REGRESSIONS)
        self.assertEqual(tuple(manifest["gates"]), M234_GATES)

    def test_preparation_documents_exist_and_reference_experiment_scope(self) -> None:
        for path in (M234_SPEC_PATH, M234_PREPARATION_PATH, M234_RATIONALE_PATH):
            self.assertTrue(Path(path).exists(), str(path))

        spec_text = Path(M234_SPEC_PATH).read_text(encoding="utf-8")
        prep_text = Path(M234_PREPARATION_PATH).read_text(encoding="utf-8")
        rationale_text = Path(M234_RATIONALE_PATH).read_text(encoding="utf-8")

        self.assertIn("Narrative Hypothesis And Experiment Design", spec_text)
        self.assertIn("bounded parallel", spec_text.lower())
        self.assertIn("tests/test_m234_experiment_design.py", prep_text)
        self.assertIn("strict mode", prep_text.lower())
        self.assertIn("parallel inquiry", rationale_text.lower())


if __name__ == "__main__":
    unittest.main()
