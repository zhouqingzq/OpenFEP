from __future__ import annotations

import unittest

from segmentum.m46_longitudinal import run_longitudinal_style_suite


class TestM46StyleStability(unittest.TestCase):
    def test_run_is_deterministic(self) -> None:
        first = run_longitudinal_style_suite()
        second = run_longitudinal_style_suite()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
