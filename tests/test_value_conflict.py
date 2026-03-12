from __future__ import annotations

import unittest

from segmentum.preferences import Goal, GoalStack, ValueConflict
from segmentum.self_model import NarrativeChapter, build_default_self_model


def _agent_state(*, energy: float, free_energy_history: list[float], danger: float = 0.2) -> dict[str, object]:
    return {
        "tick": 0,
        "body_state": {
            "energy": energy,
            "stress": 0.3,
            "fatigue": 0.2,
            "temperature": 0.5,
        },
        "observation": {"danger": danger, "social": 0.2},
        "free_energy_history": free_energy_history,
    }


class TestValueConflict(unittest.TestCase):
    def test_conflict_detection(self) -> None:
        stack = GoalStack()
        state = _agent_state(energy=0.35, free_energy_history=[0.10, 0.12, 0.14, 0.16, 0.18])

        goal = stack.evaluate_priority(state, record_conflict=True)

        self.assertEqual(goal, Goal.RESOURCES)
        self.assertEqual(len(stack.conflict_history), 1)
        conflict = stack.conflict_history[0]
        self.assertTrue(conflict.competing_goals)
        self.assertTrue(conflict.winner)
        self.assertTrue(conflict.resolution_reason)

    def test_no_conflict_when_clear_winner(self) -> None:
        stack = GoalStack()
        state = _agent_state(energy=0.10, free_energy_history=[0.1, 0.1, 0.1, 0.1, 0.1], danger=0.9)

        goal = stack.evaluate_priority(state, record_conflict=True)

        self.assertEqual(goal, Goal.SURVIVAL)
        self.assertEqual(stack.conflict_history, [])

    def test_conflict_outcome_backfill(self) -> None:
        stack = GoalStack()
        state = _agent_state(energy=0.35, free_energy_history=[0.10, 0.12, 0.14, 0.16, 0.18])
        state["tick"] = 12
        stack.evaluate_priority(state, record_conflict=True)
        stack.note_action_choice(12, "forage")
        stack.backfill_conflict_outcome(12, 0.12)

        conflict = stack.conflict_history[0]
        self.assertEqual(conflict.action_chosen, "forage")
        self.assertAlmostEqual(conflict.outcome_surprise, 0.12)

    def test_sleep_conflict_review_adjusts_weights(self) -> None:
        stack = GoalStack()
        base = stack.base_weights[Goal.RESOURCES]
        for tick in range(1, 5):
            state = _agent_state(energy=0.35, free_energy_history=[0.10, 0.12, 0.14, 0.16, 0.18])
            state["tick"] = tick
            stack.evaluate_priority(state, record_conflict=True)
            stack.note_action_choice(tick, "forage")
            stack.backfill_conflict_outcome(tick, 4.5)

        adjustments = stack.review_conflicts(current_tick=10)

        self.assertTrue(adjustments)
        self.assertLess(stack.base_weights[Goal.RESOURCES], base)
        self.assertTrue(any(item.goal == "RESOURCES" and item.direction == "decreased" for item in adjustments))

    def test_conflict_triggers_chapter_split(self) -> None:
        stack = GoalStack(
            base_weights={
                Goal.SURVIVAL: 0.60,
                Goal.INTEGRITY: 0.58,
                Goal.CONTROL: 0.78,
                Goal.RESOURCES: 0.77,
                Goal.SOCIAL: 0.20,
            }
        )
        stack.conflict_history = [
            ValueConflict(
                conflict_id="vc_0001",
                tick=21,
                competing_goals=[("CONTROL", 0.78), ("RESOURCES", 0.77)],
                winner="CONTROL",
                resolution_reason="base goal weights favor CONTROL",
                context={"energy": 0.62, "free_energy_trend": "stable", "threat_level": "low"},
                action_chosen="scan",
                outcome_surprise=6.0,
            )
        ]
        signal_before = stack.consume_chapter_signal()
        self.assertIsNone(signal_before)

        adjustments = stack.review_conflicts(current_tick=30)
        signal = stack.consume_chapter_signal()

        self.assertTrue(adjustments)
        self.assertIsNotNone(signal)
        self.assertIn("Goal priority shifted", signal)

        model = build_default_self_model()
        current = NarrativeChapter(
            chapter_id=1,
            tick_range=(1, 20),
            dominant_theme="exploration_phase",
            state_summary={"dominant_action": "scan", "risk_profile": "risk_seeking"},
        )
        self.assertTrue(
            model.should_start_new_chapter(
                current_chapter=current,
                recent_episodes=[],
                recent_state_summary={"dominant_action": "scan", "risk_profile": "risk_averse"},
                sleep_metrics={},
                chapter_signal=signal,
                current_tick=30,
            )
        )

    def test_conflict_resolution_consistency(self) -> None:
        stack = GoalStack()
        state_one = _agent_state(energy=0.35, free_energy_history=[0.10, 0.12, 0.14, 0.16, 0.18])
        state_one["tick"] = 1
        state_two = _agent_state(energy=0.35, free_energy_history=[0.10, 0.12, 0.14, 0.16, 0.18])
        state_two["tick"] = 2

        winner_one = stack.evaluate_priority(state_one, record_conflict=True)
        winner_two = stack.evaluate_priority(state_two, record_conflict=True)

        self.assertEqual(winner_one, winner_two)
        self.assertEqual(stack.conflict_history[0].winner, stack.conflict_history[1].winner)


if __name__ == "__main__":
    unittest.main()
