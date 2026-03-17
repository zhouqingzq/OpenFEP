from __future__ import annotations

import unittest

from segmentum.m222_benchmarks import build_m222_protocols, run_m222_long_horizon_trial, run_m222_protocol


class TestM222LongHorizonSurvival(unittest.TestCase):
    def test_same_seed_same_protocol_is_deterministic(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=48)["mixed_stress"]
        first = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        second = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.summary, second.summary)

    def test_long_run_summary_metrics_are_deterministic(self) -> None:
        first = run_m222_long_horizon_trial(seed_set=[222], long_run_cycles=24, restart_pre_cycles=12, restart_post_cycles=12)
        second = run_m222_long_horizon_trial(seed_set=[222], long_run_cycles=24, restart_pre_cycles=12, restart_post_cycles=12)
        self.assertEqual(first["determinism"]["passed"], True)
        self.assertEqual(first["protocol_breakdown"]["mixed_stress"]["metric_summary"], second["protocol_breakdown"]["mixed_stress"]["metric_summary"])

    def test_protected_mode_suppresses_high_risk_actions(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=96)["mixed_stress"]
        result = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        protected_cycles = [
            item
            for item in result.trace_excerpt
            if bool(dict(item.get("homeostasis", {})).get("agenda", {}).get("protected_mode"))
        ]
        self.assertTrue(protected_cycles)
        self.assertTrue(
            any(
                dict(item.get("homeostasis", {})).get("agenda", {}).get("suppressed_actions")
                for item in protected_cycles
            )
        )
        self.assertTrue(
            any(
                str(item.get("choice")) in {"rest", "hide", "exploit_shelter", "thermoregulate"}
                for item in protected_cycles
            )
        )


if __name__ == "__main__":
    unittest.main()
