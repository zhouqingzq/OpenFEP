from __future__ import annotations

import unittest

from segmentum.m227_audit import build_soak_artifact


class TestM227SubjectStateSoak(unittest.TestCase):
    def test_subject_state_remains_bounded_under_repeated_pressure_and_recovery(self) -> None:
        artifact = build_soak_artifact(seed=227, cycles=72)
        checks = artifact["checks"]
        summary = artifact["summary"]

        self.assertTrue(checks["subject_state_never_empty"])
        self.assertTrue(checks["anchors_recovered_after_pressure"])
        self.assertTrue(checks["continuity_not_collapsed_to_zero"])
        self.assertTrue(checks["phase_changes_bounded"])
        self.assertGreater(summary["non_empty_anchor_cycles"], 0)
        self.assertGreater(summary["final_anchor_count"], 0)


if __name__ == "__main__":
    unittest.main()
