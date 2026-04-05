from __future__ import annotations

import json
from pathlib import Path
import random
import unittest

from segmentum.m42_audit import M42_IGT_TRACE_PATH, write_m42_acceptance_artifacts
from segmentum.m4_benchmarks import (
    STANDARD_IGT_TRIAL_COUNT,
    IowaGamblingTaskAdapter,
    IowaTrial,
    _score_action_candidates,
    default_acceptance_benchmark_root,
)
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM44IgtAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if default_acceptance_benchmark_root() is None:
            raise unittest.SkipTest("external acceptance bundle required")
        write_m42_acceptance_artifacts()
        cls.payload = json.loads(Path(M42_IGT_TRACE_PATH).read_text(encoding="utf-8"))

    def test_standard_protocol_is_exactly_100_trials(self) -> None:
        self.assertEqual(self.payload["protocol_validation"]["standard_trial_count"], STANDARD_IGT_TRIAL_COUNT)
        self.assertEqual(self.payload["trial_count"], STANDARD_IGT_TRIAL_COUNT)
        self.assertEqual(self.payload["trial_trace"][0]["trial_index"], 1)
        self.assertEqual(self.payload["trial_trace"][-1]["trial_index"], STANDARD_IGT_TRIAL_COUNT)

    def test_trial_records_include_cumulative_gain_and_internal_state(self) -> None:
        row = self.payload["trial_trace"][0]
        self.assertIn("cumulative_gain", row)
        self.assertIn("internal_state_snapshot", row)
        self.assertTrue(row["internal_state_snapshot"])
        for key in ("deck_history", "value_estimates", "last_outcome", "loss_streak", "confidence"):
            self.assertIn(key, row["internal_state_snapshot"])


class TestM44IgtPolicyRegression(unittest.TestCase):
    def _decision(
        self,
        *,
        value_estimates: dict[str, float] | None = None,
        reward_estimates: dict[str, float] | None = None,
        loss_estimates: dict[str, float] | None = None,
        draw_counts: dict[str, int] | None = None,
        loss_counts: dict[str, int] | None = None,
        recent_outcomes: dict[str, list[float]] | None = None,
        deck_history: list[str] | None = None,
        trial_index: int = 20,
        loss_streak: int = 0,
        seed: int = 17,
    ) -> dict[str, object]:
        adapter = IowaGamblingTaskAdapter()
        parameters = CognitiveStyleParameters()
        trial = IowaTrial("demo", "subject", "A", 100, -250, -150, False, trial_index, "heldout", "demo.csv")
        state = adapter.initial_state(subject_id=trial.subject_id, parameters=parameters)
        if value_estimates:
            state["value_estimates"].update(value_estimates)
        if reward_estimates:
            state["reward_estimates"].update(reward_estimates)
        if loss_estimates:
            state["loss_estimates"].update(loss_estimates)
        if draw_counts:
            state["deck_draw_counts"].update(draw_counts)
            state["choice_counts"].update(draw_counts)
        if loss_counts:
            state["loss_counts"].update(loss_counts)
        if recent_outcomes:
            state["recent_outcomes"].update({deck: list(values) for deck, values in recent_outcomes.items()})
        if deck_history:
            state["deck_history"] = list(deck_history)
        state["loss_streak"] = loss_streak
        observation = adapter.observation_from_trial(trial, state=state, parameters=parameters, trial_index=trial_index)
        return _score_action_candidates(
            observation=observation,
            candidates=adapter.action_space(trial, observation=observation, state=state, parameters=parameters),
            parameters=parameters,
            rng=random.Random(seed),
            state=state,
        )

    def test_cold_start_keeps_all_decks_viable_without_cd_collapse(self) -> None:
        decision = self._decision(trial_index=1, seed=45)
        probabilities = {str(key): float(value) for key, value in dict(decision["action_probabilities"]).items()}

        self.assertTrue(all(probability > 0.0 for probability in probabilities.values()))
        self.assertLess(probabilities["C"] + probabilities["D"], 0.70)

    def test_positive_ab_history_outweighs_cd_when_losses_accumulate(self) -> None:
        decision = self._decision(
            value_estimates={"A": 135.0, "B": 110.0, "C": -55.0, "D": -40.0},
            reward_estimates={"A": 100.0, "B": 100.0, "C": 50.0, "D": 50.0},
            loss_estimates={"A": 10.0, "B": 25.0, "C": 150.0, "D": 130.0},
            draw_counts={"A": 6, "B": 5, "C": 5, "D": 5},
            loss_counts={"A": 0, "B": 1, "C": 3, "D": 3},
            recent_outcomes={
                "A": [100.0, 100.0, 100.0],
                "B": [100.0, 100.0, 50.0],
                "C": [-50.0, -100.0, -50.0],
                "D": [-50.0, -50.0, -100.0],
            },
            deck_history=["A", "B", "A", "B"],
            loss_streak=2,
            seed=52,
        )
        probabilities = {str(key): float(value) for key, value in dict(decision["action_probabilities"]).items()}

        self.assertGreater(probabilities["A"] + probabilities["B"], probabilities["C"] + probabilities["D"])

    def test_positive_cd_history_can_flip_preference_toward_advantageous_decks(self) -> None:
        decision = self._decision(
            value_estimates={"A": -80.0, "B": -120.0, "C": 45.0, "D": 55.0},
            reward_estimates={"A": 100.0, "B": 100.0, "C": 50.0, "D": 50.0},
            loss_estimates={"A": 170.0, "B": 220.0, "C": 20.0, "D": 10.0},
            draw_counts={"A": 6, "B": 6, "C": 5, "D": 5},
            loss_counts={"A": 4, "B": 5, "C": 0, "D": 0},
            recent_outcomes={
                "A": [-150.0, -150.0, 100.0],
                "B": [-250.0, -150.0, -150.0],
                "C": [50.0, 50.0, 50.0],
                "D": [50.0, 50.0, 50.0],
            },
            deck_history=["C", "D", "C", "D"],
            loss_streak=1,
            seed=61,
        )
        probabilities = {str(key): float(value) for key, value in dict(decision["action_probabilities"]).items()}

        self.assertGreater(probabilities["C"] + probabilities["D"], probabilities["A"] + probabilities["B"])


if __name__ == "__main__":
    unittest.main()
