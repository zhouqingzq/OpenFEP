from __future__ import annotations

import unittest

from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode


class TestNarrativeCompiler(unittest.TestCase):
    def test_resource_gain_maps_to_positive_efficacy_and_stable_observation(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="n-000",
            timestamp=1,
            source="user_diary",
            raw_text="第一天，agent出门找到了一些吃的。",
            tags=["resource"],
            metadata={},
        )

        compiled = compiler.compile_episode(episode)

        self.assertEqual(compiled.predicted_outcome, "resource_gain")
        self.assertGreater(compiled.appraisal["self_efficacy_impact"], 0.0)
        self.assertGreater(compiled.observation["food"], 0.35)
        self.assertLess(compiled.observation["danger"], 0.3)
        self.assertEqual(
            sorted(compiled.provenance["appraisal_dimensions"]),
            sorted(compiled.appraisal.keys()),
        )

    def test_predator_near_miss_compiles_deterministically(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="n-001",
            timestamp=2,
            source="user_diary",
            raw_text="第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。",
            tags=["predator"],
            metadata={},
        )

        compiled_once = compiler.compile_episode(episode)
        compiled_twice = compiler.compile_episode(episode)

        self.assertEqual(compiled_once.to_dict(), compiled_twice.to_dict())
        self.assertGreater(compiled_once.appraisal["physical_threat"], 0.9)
        self.assertLess(compiled_once.appraisal["controllability"], 0.0)
        self.assertGreater(compiled_once.observation["danger"], 0.9)
        self.assertGreater(compiled_once.compiler_confidence, 0.6)

    def test_witnessed_fatality_populates_contamination_path(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="n-002",
            timestamp=3,
            source="user_diary",
            raw_text="第三天，agent看到一个人吃了毒蘑菇死去了。",
            tags=["fatality"],
            metadata={},
        )

        compiled = compiler.compile_episode(episode)

        self.assertGreater(compiled.appraisal["contamination"], 0.8)
        self.assertGreater(compiled.appraisal["meaning_violation"], 0.3)
        self.assertEqual(compiled.predicted_outcome, "integrity_loss")
        for value in compiled.appraisal.values():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
