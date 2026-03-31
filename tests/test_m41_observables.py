from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import CognitiveStyleParameters, observable_metrics_registry, observable_parameter_contracts, run_cognitive_style_trial


class TestM41Observables(unittest.TestCase):
    def test_each_parameter_has_two_indirect_metrics(self) -> None:
        contracts = observable_parameter_contracts()
        self.assertTrue(all(len(contract["observables"]) >= 2 for contract in contracts.values()))

    def test_all_registered_metrics_can_be_computed_from_logs(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        registry = observable_metrics_registry()
        computed = payload["observable_metrics"]
        for metric_name in registry:
            self.assertIn(metric_name, computed)
            self.assertGreaterEqual(computed[metric_name], 0.0)


if __name__ == "__main__":
    unittest.main()
