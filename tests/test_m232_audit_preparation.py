from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m232_audit import (
    M232_GATES,
    M232_PREPARATION_PATH,
    M232_REGRESSIONS,
    M232_SPEC_PATH,
    M232_TESTS,
    SEED_SET,
    preparation_manifest,
)


class TestM232AuditPreparation(unittest.TestCase):
    def test_manifest_freezes_expected_audit_contract(self) -> None:
        manifest = preparation_manifest()

        self.assertEqual(manifest["milestone_id"], "M2.32")
        self.assertEqual(tuple(manifest["seed_set"]), SEED_SET)
        self.assertEqual(tuple(manifest["tests"]["milestone"]), M232_TESTS)
        self.assertEqual(tuple(manifest["tests"]["regressions"]), M232_REGRESSIONS)
        self.assertEqual(tuple(manifest["gates"]), M232_GATES)

    def test_preparation_documents_exist_and_reference_claim_level_scope(self) -> None:
        for path in (M232_SPEC_PATH, M232_PREPARATION_PATH):
            self.assertTrue(Path(path).exists(), str(path))

        spec_text = Path(M232_SPEC_PATH).read_text(encoding="utf-8")
        prep_text = Path(M232_PREPARATION_PATH).read_text(encoding="utf-8")

        self.assertIn("Threat Trace Protection", spec_text)
        self.assertIn("protect_episode_ids", spec_text)
        self.assertIn("threat-trace", prep_text.lower())
        self.assertIn("tests/test_m2_targeted_repair.py", prep_text)


if __name__ == "__main__":
    unittest.main()
