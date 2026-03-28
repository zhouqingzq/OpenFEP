from __future__ import annotations

import unittest

from segmentum.slow_learning import SlowVariableLearner


class TestM35EmergentCognitiveStyles(unittest.TestCase):
    def test_selective_explorer_style_emerges_from_effort_history(self) -> None:
        learner = SlowVariableLearner()
        for index in range(4):
            learner.record_effort_allocation(
                tick=index,
                action="rest",
                known_task=True,
                compute_spend=0.20,
                uncertainty_load=0.18,
                compression_pressure=0.68,
                process_pull=0.10,
            )
        for index in range(4):
            learner.record_effort_allocation(
                tick=10 + index,
                action="scan",
                known_task=False,
                compute_spend=0.74,
                uncertainty_load=0.80,
                compression_pressure=0.32,
                process_pull=0.68,
            )
        snapshot = learner.style_snapshot()
        self.assertEqual(snapshot["label"], "efficient_selective_explorer")
        self.assertGreater(snapshot["selective_gap"], 0.12)

    def test_no_lazy_drive_is_introduced(self) -> None:
        learner = SlowVariableLearner()
        payload = learner.to_dict()
        self.assertNotIn("lazy_drive", payload["state"])
        self.assertNotIn("lazy_drive", payload["state"]["style"])


if __name__ == "__main__":
    unittest.main()
