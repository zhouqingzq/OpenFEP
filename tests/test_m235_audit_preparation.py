from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m235_audit import (
    M235_GATES,
    M235_PREPARATION_PATH,
    M235_REGRESSIONS,
    M235_SPEC_PATH,
    M235_TESTS,
    SCHEMA_VERSION,
    SEED_SET,
    preparation_manifest,
)


class TestM235AuditPreparation(unittest.TestCase):
    def test_manifest_freezes_expected_audit_contract(self) -> None:
        manifest = preparation_manifest()

        self.assertEqual(manifest["milestone_id"], "M2.35")
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(tuple(manifest["seed_set"]), SEED_SET)
        self.assertEqual(tuple(manifest["tests"]["milestone"]), M235_TESTS)
        self.assertEqual(tuple(manifest["tests"]["regressions"]), M235_REGRESSIONS)
        self.assertEqual(tuple(manifest["gates"]), M235_GATES)

    def test_preparation_documents_exist_and_reference_scheduler_scope(self) -> None:
        for path in (M235_SPEC_PATH, M235_PREPARATION_PATH):
            self.assertTrue(Path(path).exists(), str(path))

        spec_text = Path(M235_SPEC_PATH).read_text(encoding="utf-8")
        prep_text = Path(M235_PREPARATION_PATH).read_text(encoding="utf-8")

        self.assertIn("Inquiry Budget Scheduler", spec_text)
        self.assertIn("verification slots", spec_text.lower())
        self.assertIn("tests/test_m235_acceptance.py", prep_text)
        self.assertIn("strict mode", prep_text.lower())


if __name__ == "__main__":
    unittest.main()
