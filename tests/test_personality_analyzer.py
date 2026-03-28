"""Tests for the PersonalityAnalyzer inverse inference engine."""

from __future__ import annotations

import unittest

from segmentum.analysis_types import (
    ConfidenceRated,
    CorePriors,
    EvidenceItem,
    FeedbackLoop,
    PersonalityAnalysisResult,
)
from segmentum.personality_analyzer import PersonalityAnalyzer


# ---------------------------------------------------------------------------
# Sample materials for testing
# ---------------------------------------------------------------------------

THREAT_MATERIAL = (
    "第二天，agent昨天路过河边，被一只鳄鱼攻击了，受伤了。"
)

SOCIAL_MATERIAL = (
    "A friend helped me when I was lost. They shared food and "
    "stayed nearby until I felt safe. I trusted them completely."
)

EXPLORATION_MATERIAL = (
    "I explored a new trail through the valley, mapping unfamiliar "
    "terrain. The signals were novel and I adapted quickly."
)

EXCLUSION_MATERIAL = (
    "I was excluded from the group. They rejected me and I felt "
    "abandoned and humiliated. Trust was broken."
)

SCHEMA_MATERIALS = [
    "A close friend found me when I was lost, stayed nearby, and shared food until I felt safe again.",
    "Another ally welcomed me back, made room for me, and remained beside me until I calmed down.",
    "When I froze, my companion reassured me, kept watch, and helped me reconnect with the group.",
]

MIXED_MATERIALS = [
    THREAT_MATERIAL,
    SOCIAL_MATERIAL,
    EXPLORATION_MATERIAL,
    EXCLUSION_MATERIAL,
    "I found berries and shared a meal with my group. Safe resources.",
]


class TestPersonalityAnalyzer(unittest.TestCase):
    """Core personality analyzer tests."""

    def setUp(self) -> None:
        self.analyzer = PersonalityAnalyzer()

    # ------------------------------------------------------------------
    # Basic structural tests
    # ------------------------------------------------------------------

    def test_analyze_returns_complete_result(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertIsInstance(result, PersonalityAnalysisResult)
        self.assertTrue(result.summary)
        self.assertTrue(result.one_line_conclusion)
        self.assertGreater(len(result.evidence_list), 0)
        self.assertGreater(result.analysis_confidence, 0.0)

    def test_result_has_big_five(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        for trait in ("openness", "conscientiousness", "extraversion",
                       "agreeableness", "neuroticism"):
            self.assertIn(trait, result.big_five)
            self.assertGreaterEqual(result.big_five[trait], 0.0)
            self.assertLessEqual(result.big_five[trait], 1.0)

    def test_result_has_via_strengths(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertGreater(len(result.via_strengths), 0)

    def test_to_dict_roundtrip(self) -> None:
        result = self.analyzer.analyze([SOCIAL_MATERIAL])
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("summary", d)
        self.assertIn("big_five", d)
        self.assertIn("core_priors", d)
        self.assertIn("cognitive_style", d)
        self.assertIn("evidence_list", d)

    def test_evidence_roundtrip_preserves_grounding_fields(self) -> None:
        result = self.analyzer.analyze([SOCIAL_MATERIAL])
        payload = result.to_dict()["evidence_list"][0]
        restored = EvidenceItem.from_dict(payload)
        self.assertTrue(restored.source_episode_id)
        self.assertTrue(restored.supporting_segments)
        self.assertTrue(restored.semantic_grounding)

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    def test_evidence_count_matches_materials(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertEqual(len(result.evidence_list), len(MIXED_MATERIALS))

    def test_evidence_categories_assigned(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        categories = {e.category for e in result.evidence_list}
        self.assertTrue(len(categories) >= 2, f"Expected diverse categories, got {categories}")

    def test_single_material_works(self) -> None:
        result = self.analyzer.analyze([SOCIAL_MATERIAL])
        self.assertEqual(len(result.evidence_list), 1)
        self.assertTrue(result.summary)

    def test_evidence_items_include_semantic_grounding(self) -> None:
        result = self.analyzer.analyze([SOCIAL_MATERIAL])
        item = result.evidence_list[0]
        self.assertTrue(item.source_episode_id)
        self.assertTrue(item.supporting_segments)
        self.assertTrue(item.semantic_grounding)
        self.assertTrue(item.compiled_event_type)

    def test_repeated_materials_link_to_compressed_schema(self) -> None:
        result = self.analyzer.analyze(SCHEMA_MATERIALS)
        self.assertTrue(any(item.matched_schema_ids for item in result.evidence_list))

    # ------------------------------------------------------------------
    # Core priors inference
    # ------------------------------------------------------------------

    def test_threat_material_lowers_world_safety(self) -> None:
        threat_result = self.analyzer.analyze([THREAT_MATERIAL])
        safe_result = self.analyzer.analyze([SOCIAL_MATERIAL])
        self.assertLess(
            threat_result.core_priors.world_safety.value,
            safe_result.core_priors.world_safety.value,
        )

    def test_social_material_raises_other_reliability(self) -> None:
        social_result = self.analyzer.analyze([SOCIAL_MATERIAL])
        exclusion_result = self.analyzer.analyze([EXCLUSION_MATERIAL])
        self.assertGreater(
            social_result.core_priors.other_reliability.value,
            exclusion_result.core_priors.other_reliability.value,
        )

    def test_core_priors_include_episode_or_schema_evidence(self) -> None:
        result = self.analyzer.analyze(SCHEMA_MATERIALS)
        self.assertTrue(result.core_priors.other_reliability.evidence)
        self.assertTrue(result.core_priors.other_reliability.evidence_details)
        self.assertTrue(
            any("material[" in entry or "schema:" in entry for entry in result.core_priors.other_reliability.evidence)
        )
        self.assertTrue(
            any(detail.get("kind") in {"episode", "schema"} for detail in result.core_priors.other_reliability.evidence_details)
        )

    # ------------------------------------------------------------------
    # Cognitive style
    # ------------------------------------------------------------------

    def test_cognitive_style_fields_present(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        cs = result.cognitive_style
        self.assertIsInstance(cs.abstract_vs_concrete, ConfidenceRated)
        self.assertIsInstance(cs.ambiguity_tolerance, ConfidenceRated)
        self.assertIsInstance(cs.coherence_need, ConfidenceRated)
        self.assertIsInstance(cs.reflective_depth, ConfidenceRated)

    # ------------------------------------------------------------------
    # Defense mechanisms
    # ------------------------------------------------------------------

    def test_defense_mechanisms_populated(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertGreater(len(result.defense_mechanisms.mechanisms), 0)
        for mech in result.defense_mechanisms.mechanisms:
            self.assertTrue(mech.name)
            self.assertTrue(mech.target_error)

    # ------------------------------------------------------------------
    # Feedback loops
    # ------------------------------------------------------------------

    def test_feedback_loops_present(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertGreater(len(result.feedback_loops), 0)
        for loop in result.feedback_loops:
            self.assertIsInstance(loop, FeedbackLoop)
            self.assertTrue(loop.name)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def test_behavioral_predictions_present(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertGreater(len(result.behavioral_predictions), 0)
        for pred in result.behavioral_predictions:
            self.assertTrue(pred.scenario)
            self.assertTrue(pred.predicted_behavior)

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def test_stability_sections_present(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        self.assertGreater(len(result.stable_core), 0)
        self.assertGreater(len(result.fragile_points), 0)

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def test_missing_evidence_flagged_for_single_material(self) -> None:
        result = self.analyzer.analyze([THREAT_MATERIAL])
        self.assertTrue(len(result.missing_evidence) > 0)

    # ------------------------------------------------------------------
    # Social orientation
    # ------------------------------------------------------------------

    def test_social_orientation_has_expected_keys(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        keys = set(result.social_orientation.orientation_weights.keys())
        expected = {"compete", "cooperate", "attach", "avoid", "please", "dominate", "observe"}
        self.assertEqual(keys, expected)

    # ------------------------------------------------------------------
    # Value hierarchy
    # ------------------------------------------------------------------

    def test_value_hierarchy_ranked(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        values = result.value_hierarchy.ranked_values
        self.assertGreater(len(values), 0)
        # Check ordering: scores should be non-increasing
        scores = [cr.value for _, cr in values]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    # ------------------------------------------------------------------
    # Temporal structure
    # ------------------------------------------------------------------

    def test_temporal_weights_sum_to_one(self) -> None:
        result = self.analyzer.analyze(MIXED_MATERIALS)
        ts = result.temporal_structure
        total = (
            ts.past_trauma_weight.value
            + ts.present_pressure_weight.value
            + ts.future_imagination_weight.value
        )
        self.assertAlmostEqual(total, 1.0, places=2)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic_output(self) -> None:
        r1 = self.analyzer.analyze([SOCIAL_MATERIAL])
        r2 = self.analyzer.analyze([SOCIAL_MATERIAL])
        # Big Five should be identical
        for trait in r1.big_five:
            self.assertAlmostEqual(r1.big_five[trait], r2.big_five[trait], places=6)
        # Core priors should be identical
        self.assertEqual(
            r1.core_priors.self_worth.value,
            r2.core_priors.self_worth.value,
        )


class TestConfidenceRated(unittest.TestCase):
    """Tests for the ConfidenceRated data structure."""

    def test_to_dict_roundtrip(self) -> None:
        cr = ConfidenceRated(
            0.75,
            "high",
            ["excerpt1"],
            "test reasoning",
            [{"kind": "episode", "episode_id": "mat-0001"}],
        )
        d = cr.to_dict()
        restored = ConfidenceRated.from_dict(d)
        self.assertEqual(restored.value, 0.75)
        self.assertEqual(restored.confidence, "high")
        self.assertEqual(restored.evidence, ["excerpt1"])
        self.assertEqual(restored.reasoning, "test reasoning")
        self.assertEqual(restored.evidence_details[0]["kind"], "episode")


if __name__ == "__main__":
    unittest.main()
