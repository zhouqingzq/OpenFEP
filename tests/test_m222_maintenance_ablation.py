from __future__ import annotations

import unittest

from segmentum.m222_benchmarks import build_m222_protocols, run_m222_protocol


class TestM222MaintenanceAblation(unittest.TestCase):
    def test_full_system_outperforms_maintenance_ablation(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=96)["mixed_stress"]
        full_result = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        ablated_result = run_m222_protocol(protocol, seed=222, system_variant="weakened_maintenance")
        self.assertGreater(
            float(full_result.metrics["maintenance_completion_rate"]),
            float(ablated_result.metrics["maintenance_completion_rate"]),
        )
        self.assertGreater(
            float(ablated_result.metrics["mean_free_energy"]),
            float(full_result.metrics["mean_free_energy"]),
        )
        self.assertGreater(
            float(full_result.metrics["action_switch_rate"]),
            float(ablated_result.metrics["action_switch_rate"]),
        )


if __name__ == "__main__":
    unittest.main()
