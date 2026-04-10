from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.memory_model import AnchorStrength, MemoryClass, MemoryEntry
from segmentum.memory_retrieval import RetrievalQuery
from segmentum.self_model import CapabilityModel, NarrativePriors


REAL_PIPELINE_OBSERVATION = {
    "food": 0.65,
    "danger": 0.58,
    "novelty": 0.78,
    "shelter": 0.38,
    "temperature": 0.63,
    "social": 0.42,
}
TARGET_SCAN_PRIOR = {
    "food": 0.34,
    "danger": 0.10,
    "novelty": 0.88,
    "shelter": 0.79,
    "temperature": 0.45,
    "social": 0.46,
}
DONOR_HIDE_PRIOR = {
    "food": 0.33,
    "danger": 0.03,
    "novelty": 0.04,
    "shelter": 0.37,
    "temperature": 0.41,
    "social": 0.52,
}
BODY_STATE = {
    "energy": 0.62,
    "stress": 0.28,
    "fatigue": 0.24,
    "temperature": 0.50,
}


def _observation_tags() -> list[str]:
    return [
        key
        for key, value in REAL_PIPELINE_OBSERVATION.items()
        if abs(float(value)) >= 0.15
    ]


def _expected_semantic_dimensions(anchor_type: str, source_slot_value: str | None) -> set[str]:
    normalized_value = str(source_slot_value or "").strip().lower()
    if anchor_type == "time":
        return {"novelty", "temperature"}
    if anchor_type == "place":
        return {"shelter", "temperature"}
    if anchor_type == "agents":
        return {"social", "danger"}
    if anchor_type == "action":
        if normalized_value == "scan":
            return {"novelty", "danger"}
        if normalized_value in {"hide", "rest", "exploit_shelter"}:
            return {"shelter", "danger"}
        if normalized_value == "forage":
            return {"food", "danger"}
        if normalized_value == "seek_contact":
            return {"social", "danger"}
        if any(token in normalized_value for token in ("contact", "mentor", "checkin", "social")):
            return {"social", "danger"}
    if anchor_type == "outcome":
        if any(token in normalized_value for token in ("resource", "gain", "food", "reward")):
            return {"food"}
        if any(token in normalized_value for token in ("safe", "safety", "avoidance", "escape", "survival", "threat")):
            return {"danger", "shelter"}
        if any(token in normalized_value for token in ("social", "trust", "bond")):
            return {"social"}
        if any(token in normalized_value for token in ("thermal", "temperature", "warm", "cold")):
            return {"temperature"}
    return set()


def _seed_pipeline_memory(
    agent: SegmentAgent,
    *,
    cycle: int,
    action: str,
    predicted_outcome: str,
    prior: dict[str, float],
    salience: float,
    accessibility: float,
    label: str,
    semantic_tags: list[str] | None = None,
    content: str | None = None,
    last_accessed: int | None = None,
    anchor_outcome: str | None = None,
    place: str | None = None,
) -> str:
    observation = dict(REAL_PIPELINE_OBSERVATION)
    errors = {
        key: round(float(observation[key]) - float(prior[key]), 6)
        for key in prior
    }
    payload = agent.long_term_memory.store_episode(
        cycle=cycle,
        observation=observation,
        prediction=dict(prior),
        errors=dict(errors),
        action=action,
        outcome={
            "free_energy_drop": 0.55,
            "energy_delta": 0.12,
            "summary": predicted_outcome,
        },
        body_state=dict(BODY_STATE),
    )
    entry_id = str(payload.get("episode_id", ""))
    entry = agent.long_term_memory.ensure_memory_store().get(entry_id)
    assert entry is not None
    entry.content = content or f"m49 {label} representational recall {action} {predicted_outcome}"
    entry.sync_content_hash()
    entry.salience = salience
    entry.accessibility = accessibility
    entry.trace_strength = max(0.72, float(entry.trace_strength))
    entry.last_accessed = int(last_accessed if last_accessed is not None else (cycle + 4))
    entry.semantic_tags = list(
        semantic_tags or ["m49", label, action, predicted_outcome]
    )
    entry.context_tags = _observation_tags()
    entry.anchor_slots.update(
        {
            "time": f"cycle-{cycle}",
            "place": place or ("ridge_overlook" if action == "scan" else "stone_cover"),
            "agents": "packmate",
            "action": action,
            "outcome": anchor_outcome or predicted_outcome,
        }
    )
    entry.anchor_strengths.update(
        {
            "agents": AnchorStrength.STRONG,
            "action": AnchorStrength.STRONG,
            "outcome": AnchorStrength.STRONG,
        }
    )
    entry.compression_metadata = dict(entry.compression_metadata or {})
    entry.compression_metadata["m49_seed_label"] = label
    entry.compression_metadata["m49_competition_profile"] = label
    return entry_id


def _build_agent(
    *,
    include_donor: bool,
    donor_profile: str = "default_acceptance",
) -> SegmentAgent:
    agent = SegmentAgent(memory_enabled=True)
    agent.self_model.capability_model = CapabilityModel(available_actions=("scan", "hide"))
    agent.self_model.narrative_priors = NarrativePriors()
    seed_ids = {
        "target": _seed_pipeline_memory(
            agent,
            cycle=1,
            action="scan",
            predicted_outcome="resource_gain",
            prior=TARGET_SCAN_PRIOR,
            salience=0.82,
            accessibility=0.82,
            label="target",
            semantic_tags=["m49", "scan", "resource_gain", "target"],
            content="m49 target representational recall scan resource_gain",
            last_accessed=4,
            anchor_outcome="resource_gain",
        )
    }
    if include_donor:
        donor_kwargs: dict[str, object] = {
            "cycle": 2,
            "action": "hide",
            "predicted_outcome": "neutral",
            "prior": DONOR_HIDE_PRIOR,
            "salience": 0.82,
            "accessibility": 0.82,
            "label": "donor_default",
            "semantic_tags": ["m49", "hide", "neutral", "donor"],
            "content": "m49 donor representational recall hide neutral",
            "last_accessed": 5,
            "anchor_outcome": "survival_threat",
            "place": "stone_cover",
        }
        if donor_profile == "experimental_salience":
            donor_kwargs.update(
                {
                    "salience": 0.98,
                    "accessibility": 0.98,
                    "label": "donor_experimental",
                    "predicted_outcome": "neutral",
                    "semantic_tags": ["m49", "hide", "neutral", "donor", "experimental"],
                    "content": "m49 donor experimental representational recall hide neutral",
                    "anchor_outcome": "survival_threat",
                }
            )
        elif donor_profile == "cue_match":
            donor_kwargs.update(
                {
                    "label": "donor_cue_match",
                    "semantic_tags": [
                        "m49",
                        "scan",
                        "hide",
                        "representational",
                        "recall",
                        "neutral",
                    ],
                    "content": "m49 donor cue-match representational recall scan hide neutral",
                    "anchor_outcome": "survival_threat",
                }
            )
        elif donor_profile == "recency":
            donor_kwargs.update(
                {
                    "cycle": 6,
                    "label": "donor_recency",
                    "last_accessed": 13,
                    "semantic_tags": ["m49", "scan", "hide", "representational", "recent"],
                    "content": "m49 donor recency representational recall scan hide neutral",
                    "anchor_outcome": "survival_threat",
                    "place": "stone_cover",
                }
            )
        seed_ids["donor"] = _seed_pipeline_memory(agent, **donor_kwargs)
    agent.long_term_memory.episodes = agent.memory_store.to_legacy_episodes()
    agent.sync_memory_awareness_to_long_term_memory()
    agent._m49_seed_ids = seed_ids
    return agent


def _seed_ids(agent: SegmentAgent) -> dict[str, str]:
    return dict(getattr(agent, "_m49_seed_ids", {}))


def _legacy_target_id(agent: SegmentAgent, key: str) -> str:
    return _seed_ids(agent)[key]


def _memory_entry(agent: SegmentAgent, key: str) -> MemoryEntry:
    entry = agent.memory_store.get(_legacy_target_id(agent, key))
    assert entry is not None
    return entry


def _retrieval_query(*, reference_cycle: int = 0, target_memory_class: MemoryClass | None = None) -> RetrievalQuery:
    return RetrievalQuery(
        semantic_tags=["m49", "scan", "hide"],
        context_tags=_observation_tags(),
        content_keywords=["m49", "representational", "recall", "scan", "hide"],
        reference_cycle=reference_cycle,
        target_memory_class=target_memory_class,
    )


def _configure_experimental_isolation_harness(
    agent: SegmentAgent,
    *,
    representational_prior: bool,
) -> None:
    agent.configure_memory_decision_pathways(
        legacy_policy_bias=False,
        action_rollups=False,
        representational_prior=representational_prior,
        expected_free_energy_only_policy=True,
    )


def _configure_default_path_ablation(
    agent: SegmentAgent,
    *,
    representational_prior: bool,
) -> None:
    agent.configure_memory_decision_pathways(
        representational_prior=representational_prior,
    )


def _decision(
    agent: SegmentAgent,
    *,
    observation: dict[str, float] | None = None,
) -> object:
    payload = observation or REAL_PIPELINE_OBSERVATION
    return agent.decision_cycle(Observation(**payload))["diagnostics"]


def _expected_free_energy_map(diagnostics) -> dict[str, float]:
    return {
        option.choice: float(option.expected_free_energy)
        for option in diagnostics.ranked_options
    }


class TestM49RepresentationRecall(unittest.TestCase):
    def test_artifact_structure_exports_state_vector_injections_and_residual_prior(self) -> None:
        agent = _build_agent(include_donor=True, donor_profile="experimental_salience")

        result = agent.memory_store.retrieve(
            _retrieval_query(),
            current_mood="neutral",
            k=2,
        )

        self.assertIsNotNone(result.recall_hypothesis)
        recall = result.recall_hypothesis
        assert recall is not None
        self.assertEqual(recall.primary_entry_id, _legacy_target_id(agent, "donor"))
        self.assertEqual(recall.auxiliary_entry_ids, [_legacy_target_id(agent, "target")])
        self.assertEqual(recall.representation_mode, "representational")
        self.assertEqual(recall.acceptance_path, "m49")
        self.assertIsNone(recall.legacy_reason)
        self.assertTrue(recall.reconstructed_state_vector)
        self.assertTrue(recall.latent_perturbation)
        self.assertTrue(recall.residual_prior)
        self.assertTrue(recall.protected_anchor_biases)
        self.assertTrue(recall.donor_injections)
        self.assertGreater(
            float(recall.winner_take_most_weights[_legacy_target_id(agent, "donor")]),
            float(recall.winner_take_most_weights[_legacy_target_id(agent, "target")]),
        )
        self.assertIn(
            f"residual:{_legacy_target_id(agent, 'target')}",
            recall.competing_interpretations[0],
        )
        self.assertGreater(
            float(result.reconstruction_trace["competition_snapshot"]["residual_weight"]),
            0.10,
        )

    def test_experimental_isolation_harness_causally_flips_policy(self) -> None:
        control_agent = _build_agent(include_donor=False)
        interference_agent = _build_agent(include_donor=True, donor_profile="experimental_salience")
        _configure_experimental_isolation_harness(control_agent, representational_prior=True)
        _configure_experimental_isolation_harness(interference_agent, representational_prior=True)

        control = _decision(control_agent)
        interference = _decision(interference_agent)
        control_prediction = control_agent.world_model.last_prediction_details
        interference_prediction = interference_agent.world_model.last_prediction_details

        self.assertEqual(control.chosen.choice, "scan")
        self.assertEqual(interference.chosen.choice, "hide")
        self.assertEqual(
            control_agent.last_memory_context["decision_pathways"],
            {
                "legacy_policy_bias": False,
                "action_rollups": False,
                "representational_prior": True,
                "expected_free_energy_only_policy": True,
            },
        )
        self.assertLess(
            float(_expected_free_energy_map(control)["scan"]),
            float(_expected_free_energy_map(control)["hide"]),
        )
        self.assertLess(
            float(_expected_free_energy_map(interference)["hide"]),
            float(_expected_free_energy_map(interference)["scan"]),
        )
        self.assertLess(
            float(interference_prediction["prediction_after_memory"]["novelty"]),
            float(control_prediction["prediction_after_memory"]["novelty"]),
        )
        self.assertLess(
            float(interference_prediction["prediction_after_memory"]["shelter"]),
            float(control_prediction["prediction_after_memory"]["shelter"]),
        )

    def test_experimental_isolation_ablation_removes_interference_flip(self) -> None:
        control_agent = _build_agent(include_donor=False)
        interference_agent = _build_agent(include_donor=True, donor_profile="experimental_salience")
        _configure_experimental_isolation_harness(control_agent, representational_prior=False)
        _configure_experimental_isolation_harness(interference_agent, representational_prior=False)

        control = _decision(control_agent)
        interference = _decision(interference_agent)

        self.assertEqual(control.chosen.choice, "hide")
        self.assertEqual(interference.chosen.choice, "hide")
        self.assertLess(
            float(_expected_free_energy_map(control)["hide"]),
            float(_expected_free_energy_map(control)["scan"]),
        )
        self.assertLess(
            float(_expected_free_energy_map(interference)["hide"]),
            float(_expected_free_energy_map(interference)["scan"]),
        )

    def test_experimental_audit_strings_do_not_change_prediction_or_policy_ranking(self) -> None:
        baseline_agent = _build_agent(include_donor=True, donor_profile="experimental_salience")
        altered_agent = _build_agent(include_donor=True, donor_profile="experimental_salience")
        _configure_experimental_isolation_harness(baseline_agent, representational_prior=True)
        _configure_experimental_isolation_harness(altered_agent, representational_prior=True)

        baseline = _decision(baseline_agent)
        baseline_prediction = baseline_agent.world_model.last_prediction_details
        baseline_efe = _expected_free_energy_map(baseline)

        original_retrieve = altered_agent.retrieve_for_decision

        def patched_retrieve(*args, **kwargs):
            result = original_retrieve(*args, **kwargs)
            assert result.recall_hypothesis is not None
            altered_recall = replace(
                result.recall_hypothesis,
                content="toy recall string that should be audit-only",
                competing_interpretations=["toy competitor string"],
            )
            return replace(result, recall_hypothesis=altered_recall)

        with patch.object(altered_agent, "retrieve_for_decision", side_effect=patched_retrieve):
            altered = _decision(altered_agent)
        altered_prediction = altered_agent.world_model.last_prediction_details
        altered_efe = _expected_free_energy_map(altered)

        self.assertEqual(baseline.chosen.choice, altered.chosen.choice)
        self.assertEqual(baseline_efe.keys(), altered_efe.keys())
        for action in baseline_efe:
            self.assertAlmostEqual(float(baseline_efe[action]), float(altered_efe[action]), places=9)
        for channel, value in baseline_prediction["prediction_after_memory"].items():
            self.assertAlmostEqual(
                float(value),
                float(altered_prediction["prediction_after_memory"][channel]),
                places=9,
            )
        for channel, value in baseline_prediction["prior_projection"].items():
            self.assertAlmostEqual(
                float(value),
                float(altered_prediction["prior_projection"][channel]),
                places=9,
            )

    def test_default_path_donor_vs_no_donor_proves_recall_contribution(self) -> None:
        control_agent = _build_agent(include_donor=False)
        interference_agent = _build_agent(include_donor=True)

        control = _decision(control_agent)
        interference = _decision(interference_agent)

        self.assertEqual(control.chosen.choice, "scan")
        self.assertEqual(interference.chosen.choice, "hide")
        self.assertTrue(control.decision_changed_by_recall)
        self.assertGreater(control.chosen.representational_recall_bias, 0.0)
        self.assertGreater(control.chosen.recall_counterfactual_rank_delta, 0.0)
        self.assertTrue(control.recall_audit["chosen_component_allowed_for_m49"])
        self.assertTrue(interference.recall_audit["chosen_component_allowed_for_m49"])
        self.assertFalse(
            control_agent.last_memory_context["decision_pathways"]["expected_free_energy_only_policy"]
        )
        self.assertTrue(
            control_agent.last_memory_context["decision_pathways"]["action_rollups"]
        )

    def test_default_path_ablation_of_representational_prior_removes_flip(self) -> None:
        control_agent = _build_agent(include_donor=False)
        interference_agent = _build_agent(include_donor=True)
        _configure_default_path_ablation(control_agent, representational_prior=False)
        _configure_default_path_ablation(interference_agent, representational_prior=False)

        control = _decision(control_agent)
        interference = _decision(interference_agent)

        self.assertEqual(control.chosen.choice, "hide")
        self.assertEqual(interference.chosen.choice, "hide")
        self.assertFalse(control.decision_changed_by_recall)
        self.assertEqual(control.chosen.representational_recall_bias, 0.0)
        self.assertEqual(interference.chosen.representational_recall_bias, 0.0)

    def test_default_path_counterfactual_distinguishes_prediction_change_from_decision_change(self) -> None:
        control_agent = _build_agent(include_donor=False)
        interference_agent = _build_agent(include_donor=True)

        control = _decision(control_agent)
        interference = _decision(interference_agent)

        self.assertTrue(
            any(abs(float(value)) > 1e-9 for value in control.prediction_delta.values())
        )
        self.assertTrue(
            any(abs(float(value)) > 1e-9 for value in interference.prediction_delta.values())
        )
        self.assertTrue(control.decision_changed_by_recall)
        self.assertFalse(interference.decision_changed_by_recall)
        self.assertEqual(control.recall_audit["counterfactual_choice"], "hide")
        self.assertEqual(interference.recall_audit["counterfactual_choice"], "hide")

    def test_natural_cue_match_competition_does_not_depend_on_manual_weight_boosts(self) -> None:
        agent = _build_agent(include_donor=True, donor_profile="cue_match")

        result = agent.memory_store.retrieve(
            _retrieval_query(),
            current_mood="neutral",
            k=2,
        )

        donor_id = _legacy_target_id(agent, "donor")
        target_id = _legacy_target_id(agent, "target")
        snapshot = dict(result.reconstruction_trace["competition_snapshot"])
        donor_logits = dict(snapshot["competition_logits"][donor_id])
        target_logits = dict(snapshot["competition_logits"][target_id])

        self.assertEqual(result.recall_hypothesis.primary_entry_id, donor_id)
        self.assertAlmostEqual(float(donor_logits["salience"]), float(target_logits["salience"]), places=6)
        self.assertAlmostEqual(
            float(donor_logits["accessibility"]),
            float(target_logits["accessibility"]),
            places=6,
        )
        self.assertGreaterEqual(float(donor_logits["cue_match"]), float(target_logits["cue_match"]))

    def test_natural_recency_competition_uses_recency_not_manual_weight_boosts(self) -> None:
        agent = _build_agent(include_donor=True, donor_profile="recency")

        result = agent.memory_store.retrieve(
            _retrieval_query(reference_cycle=14),
            current_mood="neutral",
            k=2,
        )

        donor_id = _legacy_target_id(agent, "donor")
        target_id = _legacy_target_id(agent, "target")
        candidate_map = {
            candidate.entry_id: candidate
            for candidate in result.candidates
        }

        self.assertEqual(result.recall_hypothesis.primary_entry_id, donor_id)
        self.assertAlmostEqual(
            float(candidate_map[donor_id].entry.salience),
            float(candidate_map[target_id].entry.salience),
            places=6,
        )
        self.assertAlmostEqual(
            float(candidate_map[donor_id].entry.accessibility),
            float(candidate_map[target_id].entry.accessibility),
            places=6,
        )
        self.assertGreater(
            float(candidate_map[donor_id].score_breakdown["recency"]),
            float(candidate_map[target_id].score_breakdown["recency"]),
        )

    def test_protected_anchor_biases_follow_semantic_mapping_rules(self) -> None:
        agent = _build_agent(include_donor=True)
        result = agent.memory_store.retrieve(
            _retrieval_query(),
            current_mood="neutral",
            k=2,
        )

        recall = result.recall_hypothesis
        assert recall is not None
        self.assertTrue(recall.protected_anchor_biases)
        for record in recall.protected_anchor_biases:
            anchor_type = str(record["anchor_type"])
            source_slot_value = record.get("source_slot_value")
            dimension = str(record["dimension"])
            self.assertIn(
                dimension,
                _expected_semantic_dimensions(anchor_type, source_slot_value),
            )
            self.assertGreater(float(record["magnitude"]), 0.0)
            self.assertIn(anchor_type, str(record["semantic_reason"]))

    def test_donor_injections_match_donor_memory_semantics(self) -> None:
        agent = _build_agent(include_donor=True)
        result = agent.memory_store.retrieve(
            _retrieval_query(),
            current_mood="neutral",
            k=2,
        )

        recall = result.recall_hypothesis
        assert recall is not None
        donor_entry = agent.memory_store.get(recall.auxiliary_entry_ids[0])
        assert donor_entry is not None
        self.assertTrue(recall.donor_injections)
        for injection in recall.donor_injections:
            anchor_type = str(injection["anchor_type"])
            dimension = str(injection["dimension"])
            self.assertIn("semantic_reason", injection)
            self.assertIn("content_source", injection)
            if anchor_type != "state_vector":
                slot_value = donor_entry.anchor_slots.get(anchor_type)
                self.assertIn(
                    dimension,
                    _expected_semantic_dimensions(anchor_type, slot_value),
                )

    def test_procedural_recall_is_marked_legacy_non_acceptance(self) -> None:
        agent = SegmentAgent(memory_enabled=True)
        store = agent.long_term_memory.ensure_memory_store()
        store.add(
            MemoryEntry(
                id="proc-target",
                memory_class=MemoryClass.PROCEDURAL,
                content="m49 procedural target",
                semantic_tags=["m49", "procedure"],
                context_tags=["lab"],
                procedure_steps=["check gauges", "vent pressure"],
                anchor_slots={"action": "check_gauges", "outcome": "safe"},
                anchor_strengths={"action": "strong", "outcome": "strong"},
                created_at=1,
                last_accessed=2,
                salience=0.82,
                accessibility=0.82,
            )
        )
        store.add(
            MemoryEntry(
                id="proc-aux",
                memory_class=MemoryClass.PROCEDURAL,
                content="m49 procedural aux",
                semantic_tags=["m49", "procedure"],
                context_tags=["lab"],
                procedure_steps=["log readings"],
                anchor_slots={"action": "check_gauges", "outcome": "safe"},
                anchor_strengths={"action": "strong", "outcome": "strong"},
                created_at=2,
                last_accessed=3,
                salience=0.80,
                accessibility=0.80,
            )
        )

        result = store.retrieve(
            _retrieval_query(target_memory_class=MemoryClass.PROCEDURAL),
            current_mood="neutral",
            k=2,
        )

        recall = result.recall_hypothesis
        assert recall is not None
        self.assertEqual(recall.representation_mode, "legacy_procedural")
        self.assertEqual(recall.acceptance_path, "legacy_non_acceptance")
        self.assertTrue(recall.legacy_reason)
        self.assertTrue(recall.procedure_step_outline)
        self.assertFalse(recall.reconstructed_state_vector)
        self.assertFalse(recall.latent_perturbation)

    def test_prompt_boundary_documents_mechanism_vs_default_acceptance(self) -> None:
        acceptance_prompt = Path("prompts/m49_acceptance_criteria.md").read_text(encoding="utf-8")
        work_prompt = Path("prompts/m49_work_prompt.md").read_text(encoding="utf-8")

        self.assertIn("mechanism", acceptance_prompt.lower())
        self.assertIn("default", acceptance_prompt.lower())
        self.assertIn("legacy_non_acceptance", work_prompt.lower())
        self.assertIn("procedural", work_prompt.lower())


if __name__ == "__main__":
    unittest.main()
