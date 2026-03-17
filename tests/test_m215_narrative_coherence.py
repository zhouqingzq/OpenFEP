from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent


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
        "identity_critical": outcome == "survival_threat",
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


class TestM215NarrativeCoherence(unittest.TestCase):
    def test_narrative_emits_evidence_backed_commitments(self) -> None:
        agent = SegmentAgent(rng=random.Random(3))
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 101)],
            episodes=[_episode(40, "scan", outcome="neutral", surprise=0.9, energy=0.75)],
            tick=100,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(101, 201)],
            episodes=[
                _episode(
                    150,
                    "hide",
                    outcome="survival_threat",
                    surprise=5.0,
                    energy=0.15,
                    free_energy_drop=-0.45,
                )
            ],
            tick=200,
            chapter_signal="Goal priority shifted: SURVIVAL overtook EXPLORATION at tick 150",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertTrue(narrative.autobiographical_summary)
        self.assertEqual(narrative.autobiographical_summary, narrative.core_summary)
        self.assertTrue(narrative.trait_self_model)
        self.assertTrue(narrative.chapter_transition_evidence)
        self.assertTrue(narrative.commitments)
        self.assertTrue(
            all(commitment.evidence_ids for commitment in narrative.commitments)
        )
        self.assertTrue(
            all(commitment.source_claim_ids for commitment in narrative.commitments)
        )

    def test_commitments_survive_restart_and_chapter_progression_is_deterministic(self) -> None:
        def build_agent(seed: int) -> SegmentAgent:
            agent = SegmentAgent(rng=random.Random(seed))
            _refresh_phase(
                agent,
                decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 81)],
                episodes=[_episode(40, "scan", outcome="neutral", surprise=0.9, energy=0.74)],
                tick=80,
            )
            _refresh_phase(
                agent,
                decisions=[_decision(tick, "hide", risk=0.2) for tick in range(81, 161)],
                episodes=[
                    _episode(
                        120,
                        "hide",
                        outcome="survival_threat",
                        surprise=4.8,
                        energy=0.16,
                        free_energy_drop=-0.5,
                    )
                ],
                tick=160,
            )
            _refresh_phase(
                agent,
                decisions=[_decision(tick, "rest", risk=0.2) for tick in range(161, 241)],
                episodes=[_episode(200, "rest", outcome="neutral", surprise=0.6, energy=0.6)],
                tick=240,
            )
            return agent

        first = build_agent(5)
        second = build_agent(5)
        first_narrative = first.self_model.identity_narrative
        second_narrative = second.self_model.identity_narrative
        assert first_narrative is not None
        assert second_narrative is not None

        first_sequence = [chapter.dominant_theme for chapter in first_narrative.chapters]
        second_sequence = [chapter.dominant_theme for chapter in second_narrative.chapters]
        self.assertEqual(first_sequence, second_sequence)

        restored = SegmentAgent.from_dict(
            json.loads(json.dumps(first.to_dict())),
            rng=random.Random(8),
        )
        restored_narrative = restored.self_model.identity_narrative
        assert restored_narrative is not None
        self.assertEqual(
            [commitment.to_dict() for commitment in first_narrative.commitments],
            [commitment.to_dict() for commitment in restored_narrative.commitments],
        )
        self.assertEqual(
            first_narrative.chapter_transition_evidence,
            restored_narrative.chapter_transition_evidence,
        )


if __name__ == "__main__":
    unittest.main()
