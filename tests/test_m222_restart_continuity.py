from __future__ import annotations

import unittest

from segmentum.m222_benchmarks import build_m222_protocols, run_m222_protocol


class TestM222RestartContinuity(unittest.TestCase):
    def test_restart_preserves_required_continuity_fields(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=48, restart_pre_cycles=24, restart_post_cycles=24)["restart_continuity"]
        result = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        for field in (
            "restart_identity_continuity",
            "restart_policy_continuity",
            "restart_memory_integrity",
        ):
            self.assertIn(field, result.metrics)
            self.assertGreaterEqual(float(result.metrics[field]), 0.80)
        preferred = result.restart["preferred_policy_continuity"]
        self.assertGreaterEqual(float(preferred["anchor_restore_score"]), 0.95)
        self.assertGreaterEqual(float(preferred["rebind_consistency"]), 0.70)
        memory = result.restart["long_term_memory_integrity"]
        self.assertGreaterEqual(float(memory["protected_memory_integrity"]), 0.95)
        self.assertGreaterEqual(float(memory["critical_memory_integrity"]), 0.95)


if __name__ == "__main__":
    unittest.main()
