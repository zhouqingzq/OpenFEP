from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import (
    CognitiveStyleParameters,
    DecisionLogRecord,
    PARAMETER_REFERENCE,
    observable_parameter_contracts,
    parameter_intervention_sensitivity_matrix,
    parameter_causality_matrix,
    parameter_identifiability_probe,
    parameter_probe_registry,
    run_cognitive_style_trial,
)


class TestM41CognitiveParameters(unittest.TestCase):
    def test_parameters_roundtrip_and_schema_version_are_stable(self) -> None:
        params = CognitiveStyleParameters(exploration_bias=0.81, resource_pressure_sensitivity=0.93)
        restored = CognitiveStyleParameters.from_dict(params.to_dict())
        self.assertEqual(params, restored)
        self.assertEqual(restored.schema_version, "m4.cognitive_style.v2")

    def test_parameter_schema_contains_all_eight_parameters(self) -> None:
        schema = CognitiveStyleParameters.schema()
        for field_name in PARAMETER_REFERENCE:
            self.assertIn(field_name, schema["required"])
            self.assertIn(field_name, schema["properties"])

    def test_decision_log_roundtrip_preserves_required_fields(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        record = DecisionLogRecord.from_dict(payload["logs"][0])
        restored = record.to_dict()
        self.assertEqual(restored["schema_version"], "m4.decision_log.v3")
        self.assertIn("candidate_actions", restored)
        self.assertIn("parameter_snapshot", restored)
        self.assertIn("prediction_error_vector", restored)
        self.assertIn("result_feedback", restored)
        self.assertIn("model_update", restored)
        self.assertEqual(
            set(restored["parameter_snapshot"].keys()),
            set(CognitiveStyleParameters.schema()["required"]),
        )

    def test_parameter_contracts_expose_executable_observables(self) -> None:
        contracts = observable_parameter_contracts()
        self.assertIn("virtual_prediction_error_gain", contracts)
        self.assertTrue(all(len(contract["observables"]) >= 2 for contract in contracts.values()))
        for contract in contracts.values():
            for observable in contract["observables"]:
                self.assertIn("evaluator", observable)
                self.assertIn("min_samples", observable)

    def test_parameter_interventions_have_expected_direction_and_effect_size(self) -> None:
        matrix = parameter_intervention_sensitivity_matrix()
        self.assertEqual(set(matrix.keys()), set(PARAMETER_REFERENCE.keys()))
        registry = parameter_probe_registry()
        for parameter_name, probe in matrix.items():
            with self.subTest(parameter=parameter_name):
                self.assertFalse(probe["insufficient_data"], msg=probe)
                self.assertTrue(probe["identifiable"], msg=probe)
                min_effect = registry[parameter_name]["min_effect"]
                self.assertGreaterEqual(abs(float(probe["delta"])), min_effect)

    def test_legacy_causality_name_is_only_a_compatibility_alias(self) -> None:
        self.assertEqual(parameter_causality_matrix(), parameter_intervention_sensitivity_matrix())

    def test_parameter_identifiability_probe_reports_intervention_sensitivity(self) -> None:
        probe = parameter_identifiability_probe()
        self.assertEqual(probe["analysis_type"], "intervention_sensitivity")
        self.assertTrue(all(probe["identifiable"].values()))


if __name__ == "__main__":
    unittest.main()
