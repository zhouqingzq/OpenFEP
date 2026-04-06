from __future__ import annotations

import unittest

from segmentum.m44_igt_aggregate import compute_igt_aggregate_metrics


def _rows(agent_decks: list[str], human_decks: list[str], agent_outcomes: list[int] | None = None, human_outcomes: list[int] | None = None) -> list[dict[str, object]]:
    agent_losses = agent_outcomes or [50 for _ in agent_decks]
    human_losses = human_outcomes or [50 for _ in human_decks]
    payload = []
    for index, (agent_deck, human_deck, agent_outcome, human_outcome) in enumerate(
        zip(agent_decks, human_decks, agent_losses, human_losses),
        start=1,
    ):
        payload.append(
            {
                "subject_id": "s1",
                "trial_index": index,
                "chosen_deck": agent_deck,
                "human_deck": human_deck,
                "net_outcome": agent_outcome,
                "human_net_outcome": human_outcome,
                "advantageous_choice": agent_deck in {"C", "D"},
                "actual_advantageous": human_deck in {"C", "D"},
            }
        )
    return payload


class TestM44IgtAggregate(unittest.TestCase):
    def test_learning_curve_distance_zero_when_curves_match(self) -> None:
        decks = (["A"] * 20) + (["B"] * 20) + (["C"] * 20) + (["D"] * 20) + (["C"] * 20)
        metrics = compute_igt_aggregate_metrics(_rows(decks, decks))

        self.assertEqual(metrics["learning_curve_distance"], 0.0)
        self.assertEqual(metrics["deck_distribution_l1"], 0.0)
        self.assertEqual(metrics["exploration_exploitation_entropy_gap"], 0.0)
        self.assertEqual(metrics["igt_behavioral_similarity"], 1.0)

    def test_deck_distribution_alignment_zero_when_distributions_match(self) -> None:
        agent = (["A"] * 25) + (["B"] * 25) + (["C"] * 25) + (["D"] * 25)
        human = (["B"] * 25) + (["A"] * 25) + (["D"] * 25) + (["C"] * 25)
        metrics = compute_igt_aggregate_metrics(_rows(agent, human))

        self.assertEqual(metrics["deck_distribution_l1"], 0.0)
        self.assertEqual(metrics["deck_distribution_alignment"], 1.0)

    def test_behavioral_similarity_drops_for_mismatched_sequences(self) -> None:
        agent = ["A"] * 100
        human = ["C" if index % 2 == 0 else "D" for index in range(100)]
        agent_outcomes = [-150 if (index % 2 == 0) else 50 for index in range(100)]
        human_outcomes = [-25 if (index % 2 == 0) else 50 for index in range(100)]
        metrics = compute_igt_aggregate_metrics(_rows(agent, human, agent_outcomes, human_outcomes))

        self.assertGreater(metrics["learning_curve_distance"], 0.0)
        self.assertGreater(metrics["deck_distribution_l1"], 0.0)
        self.assertGreater(metrics["post_loss_switch_gap"], 0.0)
        self.assertLess(metrics["igt_behavioral_similarity"], 0.5)


if __name__ == "__main__":
    unittest.main()
