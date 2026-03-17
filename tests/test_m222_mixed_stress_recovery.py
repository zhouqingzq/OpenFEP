from __future__ import annotations

import unittest

from segmentum.m222_benchmarks import build_m222_protocols, run_m222_protocol


class TestM222MixedStressRecovery(unittest.TestCase):
    def test_mixed_stress_avoids_catastrophic_collapse(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=96)["mixed_stress"]
        result = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        self.assertLessEqual(float(result.metrics["catastrophic_failure_rate"]), 0.10)
        self.assertGreaterEqual(float(result.metrics["survival_ratio"]), 0.90)
        self.assertGreaterEqual(float(result.metrics["chronic_debt_recovery_score"]), 0.60)
        self.assertGreaterEqual(float(result.metrics["resource_guard_success_rate"]), 0.80)
        recoveries = [
            item for item in result.stress_log
            if item.get("event_type") not in {"governance_probe", "restart_interruption"}
        ]
        self.assertTrue(recoveries)
        self.assertTrue(all(bool(item.get("recovered")) for item in recoveries))


if __name__ == "__main__":
    unittest.main()
