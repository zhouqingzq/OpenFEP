from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import PARAMETER_REFERENCE
from segmentum.m41_falsification import run_parameter_falsification_suite


class TestM41Falsification(unittest.TestCase):
    def test_each_parameter_has_preregistered_falsification_experiment(self) -> None:
        payload = run_parameter_falsification_suite()
        self.assertEqual(payload["analysis_type"], "parameter_falsification_suite")
        self.assertEqual(set(payload["experiments"].keys()), set(PARAMETER_REFERENCE.keys()))
        for parameter_name, experiment in payload["experiments"].items():
            with self.subTest(parameter=parameter_name):
                self.assertEqual(experiment["analysis_type"], "preregistered_falsification")
                self.assertIn("presence_condition", experiment)
                self.assertIn("absence_condition", experiment)
                self.assertIn("alternative_explanation_test", experiment)
                self.assertIn("failure_condition", experiment)
                self.assertTrue(experiment["supported"], msg=experiment)


if __name__ == "__main__":
    unittest.main()
