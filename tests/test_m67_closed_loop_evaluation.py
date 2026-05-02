from __future__ import annotations

import json

from segmentum.dialogue.m67_scenarios import (
    closure_evidence_fields,
    evaluate_m67_scenario,
    m67_scenarios,
    persona_self_consciousness_path,
    run_m67_evaluation,
)


def _scenario_result(tmp_path, scenario_id: str) -> dict[str, object]:
    scenario = next(item for item in m67_scenarios() if item.scenario_id == scenario_id)
    return evaluate_m67_scenario(scenario, trace_dir=tmp_path / "traces")


def _nested(payload: dict[str, object], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        assert isinstance(value, dict), f"{path} stopped before {part}"
        value = value.get(part)
    return value


def test_m67_scenario_fixtures_are_deterministic() -> None:
    first = [scenario.to_dict() for scenario in m67_scenarios()]
    second = [scenario.to_dict() for scenario in m67_scenarios()]

    assert first == second
    assert [item["scenario_id"] for item in first] == [
        "low_margin_ambiguity",
        "high_conflict_dialogue",
        "affective_recovery_after_repair",
        "memory_interference",
        "prompt_overload",
        "outcome_failure",
    ]
    assert all(item["seed"] for item in first)
    assert all(item["expected_delta_fields"] for item in first)


def test_outcome_event_influences_next_turn_state_or_guidance(tmp_path) -> None:
    result = _scenario_result(tmp_path, "outcome_failure")
    enabled = result["enabled"]
    ablated = result["ablated"]
    assert isinstance(enabled, dict) and isinstance(ablated, dict)

    assert enabled["outcome_event_present"] is True
    assert _nested(enabled, "state.gaps.blocking_gaps")
    assert _nested(enabled, "guidance.prefer_repair_strategy") is True
    assert _nested(enabled, "state.meta_control.lambda_control") > _nested(
        ablated,
        "state.meta_control.lambda_control",
    )


def test_outcome_event_influences_next_turn_affective_state(tmp_path) -> None:
    result = _scenario_result(tmp_path, "affective_recovery_after_repair")
    enabled = result["enabled"]
    ablated = result["ablated"]
    assert isinstance(enabled, dict) and isinstance(ablated, dict)

    assert _nested(enabled, "state.affect.warmth") > _nested(
        ablated,
        "state.affect.warmth",
    )
    assert _nested(enabled, "state.affect.social_safety") > _nested(
        ablated,
        "state.affect.social_safety",
    )
    assert _nested(enabled, "prompt_capsule.affective_state_summary")


def test_cognitive_state_influences_next_turn_prompt_capsule(tmp_path) -> None:
    result = _scenario_result(tmp_path, "low_margin_ambiguity")
    enabled = result["enabled"]
    assert isinstance(enabled, dict)

    active_gaps = _nested(enabled, "prompt_capsule.active_gaps")
    assert isinstance(active_gaps, dict)
    assert active_gaps.get("epistemic_gaps") or active_gaps.get("contextual_gaps")
    assert _nested(enabled, "prompt_capsule.meta_control_guidance")


def test_low_margin_ambiguity_increases_clarification_guidance(tmp_path) -> None:
    result = _scenario_result(tmp_path, "low_margin_ambiguity")
    enabled = result["enabled"]
    ablated = result["ablated"]
    assert isinstance(enabled, dict) and isinstance(ablated, dict)

    assert _nested(enabled, "guidance.ask_clarifying_question") is True
    assert _nested(enabled, "guidance.lower_assertiveness") is True
    assert _nested(ablated, "guidance.ask_clarifying_question") is False
    assert "low decision margin" in _nested(enabled, "guidance.trigger_reasons")


def test_high_conflict_increases_repair_and_control_guidance(tmp_path) -> None:
    result = _scenario_result(tmp_path, "high_conflict_dialogue")
    enabled = result["enabled"]
    assert isinstance(enabled, dict)

    assert _nested(enabled, "state.gaps.social_gaps")
    assert _nested(enabled, "guidance.prefer_repair_strategy") is True
    assert _nested(enabled, "guidance.increase_control_gain") is True
    assert _nested(enabled, "guidance.deescalate_affect") is True


def test_affective_recovery_uses_compressed_capsule_summary(tmp_path) -> None:
    result = _scenario_result(tmp_path, "affective_recovery_after_repair")
    enabled = result["enabled"]
    assert isinstance(enabled, dict)

    text = json.dumps(enabled["prompt_capsule"], ensure_ascii=False).lower()
    assert "affective_notes" not in text
    assert "raw affective" not in text
    assert _nested(enabled, "prompt_capsule.affective_state_summary.warmth") == _nested(
        enabled,
        "state.affect.warmth",
    )


def test_prompt_overload_records_omitted_signals(tmp_path) -> None:
    result = _scenario_result(tmp_path, "prompt_overload")
    enabled = result["enabled"]
    ablated = result["ablated"]
    assert isinstance(enabled, dict) and isinstance(ablated, dict)

    assert _nested(enabled, "guidance.compress_context") is True
    assert _nested(enabled, "prompt_capsule.omitted_signals") == [
        "raw_events",
        "full_diagnostics",
        "full_prompt",
        "full_conscious_markdown",
    ]
    assert _nested(ablated, "prompt_capsule.omitted_signals") is None


def test_memory_interference_reduces_memory_reliance(tmp_path) -> None:
    result = _scenario_result(tmp_path, "memory_interference")
    enabled = result["enabled"]
    ablated = result["ablated"]
    assert isinstance(enabled, dict) and isinstance(ablated, dict)

    assert _nested(enabled, "state.memory.memory_conflicts")
    assert _nested(enabled, "guidance.reduce_memory_reliance") is True
    assert _nested(enabled, "prompt_capsule.memory_use_guidance.reduce_memory_reliance") is True
    assert not _nested(ablated, "state.memory.memory_conflicts")


def test_each_scenario_has_expected_enabled_vs_ablated_delta(tmp_path) -> None:
    for scenario in m67_scenarios():
        result = evaluate_m67_scenario(scenario, trace_dir=tmp_path / "traces")
        assert result["passed"], result
        assert result["missing_expected_delta_fields"] == []
        for field in scenario.expected_delta_fields:
            assert field in result["changed_fields"]
        enabled = result["enabled"]
        ablated = result["ablated"]
        assert isinstance(enabled, dict) and isinstance(ablated, dict)
        assert enabled["selected_action"] == ablated["selected_action"]


def test_two_same_display_name_personas_do_not_share_self_consciousness(tmp_path) -> None:
    display_name = "Alex"
    first = persona_self_consciousness_path(tmp_path, "persona-a-stable-id")
    second = persona_self_consciousness_path(tmp_path, "persona-b-stable-id")
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(f"display={display_name}\nfirst", encoding="utf-8")
    second.write_text(f"display={display_name}\nsecond", encoding="utf-8")

    assert first != second
    assert first.read_text(encoding="utf-8") != second.read_text(encoding="utf-8")
    assert first.parent.name != second.parent.name


def test_conscious_markdown_only_change_does_not_count_as_closure() -> None:
    enabled = {
        "state": {},
        "guidance": {},
        "prompt_capsule": {},
        "memory_update_signal": {},
        "conscious_markdown_path": "a/Conscious.md",
    }
    ablated = {
        "state": {},
        "guidance": {},
        "prompt_capsule": {},
        "memory_update_signal": {},
        "conscious_markdown_path": "b/Conscious.md",
    }

    assert closure_evidence_fields(enabled, ablated) == []


def test_m67_artifact_schema_is_stable_and_honest(tmp_path) -> None:
    result = run_m67_evaluation(artifacts_dir=tmp_path)
    artifact_path = tmp_path / "m67_closed_loop_evaluation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert result["all_passed"] is True
    assert artifact["milestone"] == "M6.7"
    assert artifact["current_run_evidence"] is True
    assert artifact["quality_gates"]["uses_ablation"] is True
    assert artifact["quality_gates"]["rejects_trace_only_closure"] is True
    assert artifact["scenario_count"] == 6
    assert artifact["passed_scenario_count"] == 6
    assert len(artifact["scenario_ids"]) == 6
    for row in artifact["scenarios"]:
        assert {
            "scenario_id",
            "enabled",
            "ablated",
            "changed_fields",
            "expected_delta_fields",
            "missing_expected_delta_fields",
            "passed",
        }.issubset(row)
        assert row["passed"] is True
        assert row["changed_fields"]

