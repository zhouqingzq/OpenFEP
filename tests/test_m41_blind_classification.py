from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import blind_classification_experiment


class TestM41BlindClassification(unittest.TestCase):
    def test_blind_classification_has_three_profiles_and_hits_threshold(self) -> None:
        experiment = blind_classification_experiment()
        self.assertGreaterEqual(len(experiment["profiles"]), 3)
        self.assertGreaterEqual(experiment["accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()
