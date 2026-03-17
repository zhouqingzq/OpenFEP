from __future__ import annotations

import copy
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.memory import LIFECYCLE_PROTECTED_IDENTITY_CRITICAL
from segmentum.self_model import IdentityCommitment, IdentityNarrative


def _exploration_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I am an exploratory agent.",
        commitments=[
            IdentityCommitment(
                commitment_id="commitment-exploration-drive",
                commitment_type="behavioral_style",
                statement="When conditions are stable, reduce uncertainty through exploration.",
                target_actions=["scan"],
                discouraged_actions=["rest"],
                confidence=0.95,
                priority=0.95,
                source_claim_ids=["claim-explore"],
                source_chapter_ids=[1],
                evidence_ids=["ep-explore-001"],
                last_reaffirmed_tick=40,
            )
        ],
    )


def _survival_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I am a risk-averse, survival-focused agent.",
        commitments=[
            IdentityCommitment(
                commitment_id="commitment-survival-priority",
                commitment_type="value_guardrail",
                statement="Protect survival and integrity before opportunistic gain.",
                target_actions=["hide", "rest", "exploit_shelter"],
                discouraged_actions=["forage"],
                confidence=0.95,
                priority=0.95,
                source_claim_ids=["claim-survival"],
                source_chapter_ids=[1],
                evidence_ids=["ep-survival-001"],
                last_reaffirmed_tick=40,
            )
        ],
    )


class TestM215IdentityCommitments(unittest.TestCase):
    def test_commitments_change_action_ranking(self) -> None:
        observation = Observation(
            food=0.25,
            danger=0.05,
            novelty=1.0,
            shelter=0.1,
            temperature=0.5,
            social=0.1,
        )
        baseline = SegmentAgent(rng=random.Random(7))
        with_commitment = SegmentAgent(rng=random.Random(7))
        with_commitment.self_model.identity_narrative = _exploration_narrative()

        baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
        commitment_diag = with_commitment.decision_cycle(observation)["diagnostics"]

        baseline_scores = {
            option.choice: option.policy_score for option in baseline_diag.ranked_options
        }
        commitment_scores = {
            option.choice: option.policy_score
            for option in commitment_diag.ranked_options
        }
        self.assertLess(baseline_scores["scan"], baseline_scores["rest"])
        self.assertGreater(commitment_scores["scan"], commitment_scores["rest"])
        self.assertIn(
            "commitment-exploration-drive",
            commitment_diag.structured_explanation["current_commitments"],
        )

    def test_violation_detection_and_memory_protection(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        agent.self_model.identity_narrative = _survival_narrative()

        assessment = agent.policy_evaluator.commitment_assessment(
            action="forage",
            projected_state={"danger": 0.95, "novelty": 0.1, "shelter": 0.05},
        )
        self.assertLess(float(assessment["bias"]), 0.0)
        self.assertIn("commitment-survival-priority", assessment["violations"])

        decision = agent.decision_cycle(
            Observation(
                food=0.85,
                danger=0.82,
                novelty=0.15,
                shelter=0.2,
                temperature=0.5,
                social=0.1,
            )
        )
        observed = dict(decision["observed"])
        prediction = dict(decision["prediction"])
        observed["danger"] = 1.0
        observed["shelter"] = 0.0
        prediction["danger"] = 0.0
        prediction["shelter"] = 1.0
        errors = {
            key: abs(observed.get(key, 0.0) - prediction.get(key, 0.0))
            for key in observed
        }
        memory_decision = agent.integrate_outcome(
            choice=decision["diagnostics"].chosen.choice,
            observed=observed,
            prediction=prediction,
            errors=errors,
            free_energy_before=float(decision["free_energy_before"]),
            free_energy_after=float(decision["free_energy_before"]) + 0.5,
        )
        self.assertTrue(memory_decision.episode_created)
        protected = copy.deepcopy(agent.long_term_memory.episodes[-1])
        self.assertTrue(bool(protected.get("identity_critical", False)))
        self.assertEqual(
            protected.get("identity_commitment_reason"),
            "identity_commitment_reaffirmed",
        )

        for index in range(3):
            duplicate = copy.deepcopy(protected)
            duplicate["episode_id"] = f"duplicate-{index}"
            duplicate["identity_critical"] = False
            duplicate["identity_commitment_reason"] = ""
            duplicate["identity_commitment_ids"] = []
            duplicate["lifecycle_stage"] = "validated_episode"
            agent.long_term_memory.episodes.append(duplicate)

        removed = agent.long_term_memory.compress_episodes()
        self.assertGreater(removed, 0)
        self.assertTrue(
            any(
                bool(payload.get("identity_critical", False))
                and payload.get("lifecycle_stage")
                == LIFECYCLE_PROTECTED_IDENTITY_CRITICAL
                and "commitment-survival-priority"
                in payload.get("identity_commitment_ids", [])
                for payload in agent.long_term_memory.episodes
            )
        )


if __name__ == "__main__":
    unittest.main()
