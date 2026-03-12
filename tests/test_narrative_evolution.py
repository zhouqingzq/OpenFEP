from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.self_model import NarrativeChapter, build_default_self_model


def _episode(
    tick: int,
    action: str,
    *,
    outcome: str = "neutral",
    surprise: float = 0.8,
    energy: float = 0.7,
    free_energy_drop: float = 0.05,
) -> dict[str, object]:
    return {
        "timestamp": tick,
        "action_taken": action,
        "predicted_outcome": outcome,
        "total_surprise": surprise,
        "risk": 0.4 if outcome == "neutral" else 4.0,
        "body_state": {
            "energy": energy,
            "stress": 0.3,
            "fatigue": 0.2,
            "temperature": 0.5,
        },
        "outcome_state": {
            "free_energy_drop": free_energy_drop,
        },
    }


def _decision(tick: int, action: str, *, risk: float) -> dict[str, object]:
    return {
        "tick": tick,
        "action": action,
        "dominant_component": "goal_alignment",
        "risk": risk,
        "active_goal": "CONTROL",
        "goal_alignment": 0.4,
        "preferred_probability": 0.5,
        "policy_score": 0.2,
    }


def _refresh_phase(
    agent: SegmentAgent,
    *,
    decisions: list[dict[str, object]],
    episodes: list[dict[str, object]],
    tick: int,
    chapter_signal: str | None = None,
) -> None:
    agent.decision_history.extend(decisions)
    agent.long_term_memory.episodes.extend(episodes)
    agent.self_model.update_preferred_policies(
        agent.decision_history,
        current_tick=tick,
    )
    agent.self_model.update_identity_narrative(
        episodic_memory=list(agent.long_term_memory.episodes),
        preference_labels=agent.long_term_memory.preference_model.legacy_value_hierarchy_dict(),
        current_tick=tick,
        decision_history=list(agent.decision_history),
        sleep_metrics={"policy_bias_updates": 1, "threat_updates": 1},
        conflict_history=list(agent.goal_stack.conflict_history),
        weight_adjustments=list(agent.goal_stack.weight_adjustments),
        chapter_signal=chapter_signal,
    )


class TestNarrativeEvolution(unittest.TestCase):
    def test_chapter_creation_on_behavioral_shift(self) -> None:
        agent = SegmentAgent(rng=random.Random(1))
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 101)],
            episodes=[_episode(tick, "scan", outcome="neutral", surprise=0.8, energy=0.78) for tick in range(1, 101, 20)],
            tick=100,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.3) for tick in range(101, 201)],
            episodes=[_episode(150, "hide", outcome="survival_threat", surprise=5.0, energy=0.18, free_energy_drop=-0.4)],
            tick=200,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(201, 241)],
            episodes=[_episode(220, "rest", outcome="neutral", surprise=0.6, energy=0.52)],
            tick=240,
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertGreaterEqual(len(narrative.chapters), 2)
        self.assertIsNotNone(narrative.chapters[1].behavioral_shift)
        self.assertNotEqual(narrative.chapters[1].dominant_theme, narrative.chapters[0].dominant_theme)

    def test_core_identity_stability(self) -> None:
        agent = SegmentAgent(rng=random.Random(2))
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 81)],
            episodes=[_episode(40, "scan", outcome="neutral", surprise=0.9, energy=0.75)],
            tick=80,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(81, 161)],
            episodes=[_episode(120, "hide", outcome="survival_threat", surprise=4.8, energy=0.16, free_energy_drop=-0.5)],
            tick=160,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(161, 241)],
            episodes=[_episode(200, "rest", outcome="neutral", surprise=0.7, energy=0.58)],
            tick=240,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(241, 321)],
            episodes=[_episode(280, "hide", outcome="neutral", surprise=0.5, energy=0.63)],
            tick=320,
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertTrue(
            any(term in narrative.core_identity for term in ("risk-averse", "resource-conservative"))
        )
        self.assertIn("significant shift", narrative.core_summary.lower())

    def test_narrative_survives_restart(self) -> None:
        agent = SegmentAgent(rng=random.Random(3))
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 51)],
            episodes=[_episode(25, "scan", surprise=0.7)],
            tick=50,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(51, 101)],
            episodes=[_episode(75, "hide", outcome="survival_threat", surprise=4.2, energy=0.14, free_energy_drop=-0.35)],
            tick=100,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(101, 151)],
            episodes=[_episode(125, "rest", surprise=0.6, energy=0.6)],
            tick=150,
        )

        before = agent.self_model.identity_narrative.to_dict()
        restored = SegmentAgent.from_dict(json.loads(json.dumps(agent.to_dict())), rng=random.Random(4))
        after = restored.self_model.identity_narrative.to_dict()
        self.assertEqual(before, after)

        _refresh_phase(
            restored,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(151, 181)],
            episodes=[_episode(165, "rest", surprise=0.5, energy=0.65)],
            tick=180,
            chapter_signal="Goal priority shifted: RESOURCES overtook CONTROL due to conflicts at ticks [165]",
        )
        narrative = restored.self_model.identity_narrative
        assert narrative is not None
        total_chapters = len(narrative.chapters) + (1 if narrative.current_chapter is not None else 0)
        self.assertGreaterEqual(total_chapters, 4)

    def test_core_summary_reflects_history(self) -> None:
        danger_agent = SegmentAgent(rng=random.Random(5))
        _refresh_phase(
            danger_agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(1, 51)],
            episodes=[_episode(40, "hide", outcome="survival_threat", surprise=5.5, energy=0.10, free_energy_drop=-0.45)],
            tick=50,
        )

        stable_agent = SegmentAgent(rng=random.Random(6))
        _refresh_phase(
            stable_agent,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(1, 51)],
            episodes=[_episode(40, "rest", outcome="neutral", surprise=0.5, energy=0.72)],
            tick=50,
        )

        self.assertTrue(
            "near-death" in danger_agent.self_model.identity_narrative.core_summary
            or "survival_crisis" in danger_agent.self_model.identity_narrative.core_summary
        )
        self.assertNotIn("near-death", stable_agent.self_model.identity_narrative.core_summary)


if __name__ == "__main__":
    unittest.main()
