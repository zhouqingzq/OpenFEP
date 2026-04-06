from __future__ import annotations

import unittest

from segmentum.m49_longitudinal import run_longitudinal_style_suite


class TestM49RecoveryRetention(unittest.TestCase):
    def test_recovery_retains_style_signature(self) -> None:
        payload = run_longitudinal_style_suite()
        self.assertTrue(payload["summary"]["recovery_retains_style"])


if __name__ == "__main__":
    unittest.main()
