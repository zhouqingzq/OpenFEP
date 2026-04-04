from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import (
    CognitiveStyleParameters,
    compute_observable_metrics,
    metric_values_from_payload,
    metrics_have_executable_registry,
    observable_metrics_registry,
    observable_parameter_contracts,
    run_cognitive_style_trial,
)


class TestM41Observables(unittest.TestCase):
    def test_each_parameter_has_two_indirect_metrics(self) -> None:
        contracts = observable_parameter_contracts()
        self.assertTrue(all(len(contract["observables"]) >= 2 for contract in contracts.values()))

    def test_registry_entries_have_executable_evaluators(self) -> None:
        registry = observable_metrics_registry()
        self.assertTrue(metrics_have_executable_registry(registry))
        trial = run_cognitive_style_trial(CognitiveStyleParameters())
        for metric_name, spec in registry.items():
            self.assertEqual(metric_name, spec["metric_id"])
            self.assertTrue(callable(spec["evaluator"]))
            metric_payload = spec["evaluator"](trial["logs"])
            self.assertIsInstance(metric_payload, dict)
            self.assertIn("insufficient_data", metric_payload)

    def test_all_registered_metrics_can_be_computed_from_logs(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        registry = observable_metrics_registry()
        computed = payload["observable_metrics"]
        values = metric_values_from_payload(computed)
        for metric_name in registry:
            self.assertIn(metric_name, computed)
            self.assertIn("insufficient_data", computed[metric_name])
            self.assertFalse(computed[metric_name]["insufficient_data"], msg=metric_name)
            self.assertIn(metric_name, values)

    def test_insufficient_data_does_not_forge_metric_values(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        sparse_metrics = compute_observable_metrics(payload["logs"][:2])
        self.assertTrue(any(item["insufficient_data"] for item in sparse_metrics.values()))
        self.assertTrue(any(item["value"] is None for item in sparse_metrics.values()))


if __name__ == "__main__":
    unittest.main()
