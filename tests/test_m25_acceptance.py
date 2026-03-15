from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_ingestion import NarrativeIngestionService
from segmentum.narrative_types import (
    AppraisalVector,
    CompiledNarrativeEvent,
    EmbodiedNarrativeEpisode,
    NarrativeEpisode,
)
from segmentum.self_model import NarrativePriors


RESOURCE_TEXT = "第一天，agent出门找到了一些吃的。"
PREDATOR_TEXT = "第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。"
FATALITY_TEXT = "第三天，agent看到一个人吃了毒蘑菇死去了。"


def _round_trip_agent(agent: SegmentAgent, seed: int = 999) -> SegmentAgent:
    payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
    return SegmentAgent.from_dict(payload, rng=random.Random(seed))


def _ingest_episode(
    agent: SegmentAgent,
    compiler: NarrativeCompiler,
    *,
    episode_id: str,
    timestamp: int,
    raw_text: str,
    tags: list[str],
    source: str = "user_diary",
) -> dict[str, object]:
    embodied = compiler.compile_episode(
        NarrativeEpisode(
            episode_id=episode_id,
            timestamp=timestamp,
            source=source,
            raw_text=raw_text,
            tags=tags,
            metadata={},
        )
    )
    return agent.ingest_narrative_episode(embodied)


class TestM25Surface(unittest.TestCase):
    def test_m25_surface_names_are_importable(self) -> None:
        self.assertTrue(callable(NarrativeCompiler))
        self.assertTrue(callable(NarrativeIngestionService))
        self.assertTrue(callable(NarrativeEpisode))
        self.assertTrue(callable(CompiledNarrativeEvent))
        self.assertTrue(callable(AppraisalVector))
        self.assertTrue(callable(EmbodiedNarrativeEpisode))
        self.assertIsInstance(NarrativePriors(), NarrativePriors)


class TestM25AcceptanceScenarios(unittest.TestCase):
    def test_scenario_a_resource_gain_compiles_and_improves_controllability_prior(self) -> None:
        agent = SegmentAgent(rng=random.Random(101))
        agent.long_term_memory.episode_score_threshold = 0.0
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()

        ingest_result = _ingest_episode(
            agent,
            compiler,
            episode_id="m25-a-1",
            timestamp=1,
            raw_text=RESOURCE_TEXT,
            tags=["resource"],
        )

        self.assertEqual(ingest_result["compiled_event"]["event_type"], "resource_gain")
        self.assertEqual(ingest_result["predicted_outcome"], "resource_gain")
        self.assertGreater(ingest_result["appraisal"]["self_efficacy_impact"], 0.0)
        self.assertGreater(ingest_result["value_score"], 0.0)
        self.assertTrue(ingest_result["episode_created"])

        before = agent.self_model.narrative_priors.to_dict()
        agent.sleep()
        after = agent.self_model.narrative_priors.to_dict()

        self.assertGreater(after["controllability_prior"], before["controllability_prior"])

    def test_scenario_b_predator_near_miss_updates_trace_memory_and_later_caution(self) -> None:
        baseline = SegmentAgent(rng=random.Random(202))
        danger_observation = Observation(
            food=0.16,
            danger=0.84,
            novelty=0.20,
            shelter=0.12,
            temperature=0.48,
            social=0.18,
        )
        baseline_diag = baseline.decision_cycle(danger_observation)["diagnostics"]
        baseline_forage = next(
            option.policy_score
            for option in baseline_diag.ranked_options
            if option.choice == "forage"
        )

        agent = SegmentAgent(rng=random.Random(202))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()

        first_result = _ingest_episode(
            agent,
            compiler,
            episode_id="m25-b-1",
            timestamp=1,
            raw_text=PREDATOR_TEXT,
            tags=["predator"],
        )
        for index in range(2, 5):
            _ingest_episode(
                agent,
                compiler,
                episode_id=f"m25-b-{index}",
                timestamp=index,
                raw_text=PREDATOR_TEXT,
                tags=["predator"],
            )

        self.assertGreater(first_result["prediction_error"], 0.0)
        self.assertGreater(first_result["total_surprise"], 1.0)
        self.assertTrue(first_result["episode_created"])
        self.assertEqual(first_result["compiled_event"]["event_type"], "predator_attack")
        self.assertIn("source_episode_id", first_result)
        self.assertIn("prediction_before_ingestion", first_result)
        self.assertTrue(agent.long_term_memory.episodes)
        stored = agent.long_term_memory.episodes[-1]
        self.assertIn("appraisal", stored)
        self.assertIn("source_episode_id", stored)
        self.assertIn("narrative_provenance", stored)

        before = agent.self_model.narrative_priors.to_dict()
        agent.sleep()
        after = agent.self_model.narrative_priors.to_dict()
        diag = agent.decision_cycle(danger_observation)["diagnostics"]
        forage_score = next(
            option.policy_score
            for option in diag.ranked_options
            if option.choice == "forage"
        )

        self.assertGreater(after["trauma_bias"], before["trauma_bias"])
        self.assertLess(forage_score, baseline_forage)
        self.assertTrue(
            any("narrative_prior_updates" in entry for entry in agent.narrative_trace)
        )

    def test_scenario_c_witnessed_fatality_updates_contamination_prior_and_audit_chain(self) -> None:
        agent = SegmentAgent(rng=random.Random(303))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()

        ingest_result = _ingest_episode(
            agent,
            compiler,
            episode_id="m25-c-1",
            timestamp=1,
            raw_text=FATALITY_TEXT,
            tags=["fatality"],
        )

        self.assertGreater(ingest_result["appraisal"]["contamination"], 0.8)
        self.assertGreater(ingest_result["appraisal"]["uncertainty"], 0.5)
        self.assertGreater(ingest_result["total_surprise"], 0.0)
        self.assertEqual(ingest_result["compiled_event"]["event_type"], "witnessed_death")

        before = agent.self_model.narrative_priors.to_dict()
        agent.sleep()
        after = agent.self_model.narrative_priors.to_dict()

        self.assertGreater(
            after["contamination_sensitivity"],
            before["contamination_sensitivity"],
        )
        self.assertLess(after["meaning_stability"], before["meaning_stability"])
        sleep_trace = next(
            entry for entry in reversed(agent.narrative_trace)
            if "narrative_prior_updates" in entry
        )
        self.assertIn("sleep_cycle_id", sleep_trace)
        self.assertIn("rule_ids", sleep_trace)


class TestM25AcceptanceCriteria(unittest.TestCase):
    def test_ingestion_service_accepts_raw_narrative_without_manual_observation(self) -> None:
        agent = SegmentAgent(rng=random.Random(404))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        service = NarrativeIngestionService()
        results = service.ingest(
            agent=agent,
            episodes=[
                NarrativeEpisode(
                    episode_id="m25-runtime-1",
                    timestamp=1,
                    source="user_diary",
                    raw_text=PREDATOR_TEXT,
                    tags=["predator"],
                    metadata={},
                )
            ],
        )

        self.assertEqual(len(results), 1)
        self.assertIn("compilation", results[0])
        self.assertIn("ingestion", results[0])
        self.assertIsInstance(results[0]["compilation"]["observation"], dict)

    def test_trace_and_priors_survive_serialization_round_trip(self) -> None:
        agent = SegmentAgent(rng=random.Random(505))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()
        for index, raw_text in enumerate((PREDATOR_TEXT, FATALITY_TEXT), start=1):
            _ingest_episode(
                agent,
                compiler,
                episode_id=f"m25-rt-{index}",
                timestamp=index,
                raw_text=raw_text,
                tags=["audit"],
            )
        agent.sleep()

        restored = _round_trip_agent(agent, seed=808)

        self.assertEqual(
            restored.self_model.narrative_priors.to_dict(),
            agent.self_model.narrative_priors.to_dict(),
        )
        self.assertEqual(restored.narrative_trace, agent.narrative_trace)
        self.assertEqual(
            restored.long_term_memory.episodes,
            agent.long_term_memory.episodes,
        )


if __name__ == "__main__":
    unittest.main()
