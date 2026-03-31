from __future__ import annotations

import unittest

from segmentum.m45_open_world import simulate_open_world_projection
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM45FailureRecovery(unittest.TestCase):
    def test_style_ablation_reduces_recovery_quality(self) -> None:
        full = simulate_open_world_projection(CognitiveStyleParameters(), seed=45)
        ablated = simulate_open_world_projection(CognitiveStyleParameters(), seed=45, ablate_style=True)
        self.assertGreater(full["summary"]["adaptive_recovery_rate"], ablated["summary"]["adaptive_recovery_rate"])


if __name__ == "__main__":
    unittest.main()
