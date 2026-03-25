from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m233_audit import (
    M233_GATES,
    M233_PREPARATION_PATH,
    M233_REGRESSIONS,
    M233_SPEC_PATH,
    M233_TESTS,
    SEED_SET,
    preparation_manifest,
)


class TestM233AuditPreparation(unittest.TestCase):
    def test_manifest_freezes_expected_audit_contract(self) -> None:
        manifest = preparation_manifest()

        self.assertEqual(manifest["milestone_id"], "M2.33")
        self.assertEqual(tuple(manifest["seed_set"]), SEED_SET)
        self.assertEqual(tuple(manifest["tests"]["milestone"]), M233_TESTS)
        self.assertEqual(tuple(manifest["tests"]["regressions"]), M233_REGRESSIONS)
        self.assertEqual(tuple(manifest["gates"]), M233_GATES)

    def test_preparation_documents_exist_and_reference_uncertainty_scope(self) -> None:
        for path in (M233_SPEC_PATH, M233_PREPARATION_PATH):
            self.assertTrue(Path(path).exists(), str(path))

        spec_text = Path(M233_SPEC_PATH).read_text(encoding="utf-8")
        prep_text = Path(M233_PREPARATION_PATH).read_text(encoding="utf-8")

        self.assertIn("Narrative Uncertainty Decomposition", spec_text)
        self.assertIn("latent cause", spec_text.lower())
        self.assertIn("surface cue", prep_text.lower())
        self.assertIn("tests/test_m233_uncertainty_decomposition.py", prep_text)


if __name__ == "__main__":
    unittest.main()
