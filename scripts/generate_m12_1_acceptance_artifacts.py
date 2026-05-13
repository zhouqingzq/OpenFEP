"""Generate deterministic M12.1 acceptance fixtures and report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from segmentum.user_personality import (
    PersonalityProfile,
    PersonalitySummary,
    TriggerPolicyInput,
    assemble_personality_report,
    decide_trigger,
    evidence_cards_from_personality_profile,
    prompt_safe_cards,
    run_personality_orchestrator,
)
from segmentum.user_personality.personality_orchestrator import build_base_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "m12_1"
ARTIFACT_PATH = ROOT / "artifacts" / "m12_1_acceptance_report.json"


def _extractors(outputs: Mapping[int, Mapping[str, object]]):
    return {idx: (lambda _snapshot, payload=payload: payload) for idx, payload in outputs.items()}


def _base_snapshot(turn_id: str = "t4") -> dict[str, object]:
    return build_base_snapshot(
        user_id="u_accept",
        display_name="Acceptance User",
        turn_id=turn_id,
        current_turn_quotes={"q4": "I want actual checks before accepting the milestone."},
        transcript_quote_refs=[
            {"turn_id": "t1", "quote_id": "q1"},
            {"turn_id": "t2", "quote_id": "q2"},
            {"turn_id": "t3", "quote_id": "q3"},
            {"turn_id": "t4", "quote_id": "q4"},
        ],
        m11_readonly_summary={
            "active_hypotheses": [
                {"hypothesis_id": "h_accept_1", "content_summary": "User values direct inspection.", "confidence_band": "med"},
            ],
            "reliability_summary": {"task_requirements": "med"},
        },
        m12_readonly_summary={
            "identity_state": "corroborated",
            "binding_confidence_band": "high",
            "recent_continuity_cues": [
                {"cue_id": "cue_accept_1", "cue_kind": "style", "confidence_band": "med", "supports": "binds"},
            ],
        },
    )


def _rich_outputs() -> dict[int, dict[str, object]]:
    refs = ["t1:q1", "t2:q2", "t3:q3", "t4:q4"]
    return {
        1: {"summary": "Usually seeks concrete proof before trusting a direction.", "evidence_quote_refs": refs, "confidence_band": "high"},
        2: {
            "evidence_items": [
                {"kind": "pressure_response", "content_summary": "Moves from doubt to trust when checks are visible.", "evidence_quote_refs": refs, "confidence_band": "high"},
                {"kind": "language_habit", "content_summary": "Names weak tests and thin acceptance as problems.", "evidence_quote_refs": ["t2:q2", "t4:q4"], "confidence_band": "med"},
            ]
        },
        3: {
            "wants": "wants claims to be backed by visible work",
            "fears": "fears being handed polished but weak completion",
            "hypersensitive_to": ["thin tests", "keyword-only cues"],
            "ignores": ["decorative completion language"],
            "default_interpretation": "checks whether the claim was earned",
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        4: {
            "about_self": {"content_summary": "I trust what survives checks.", "evidence_quote_refs": refs, "confidence_band": "high"},
            "about_others": {"content_summary": "Others can overstate readiness.", "evidence_quote_refs": refs, "confidence_band": "high"},
            "about_world": {"content_summary": "Strong work is inspectable.", "evidence_quote_refs": refs, "confidence_band": "high"},
        },
        5: {
            "dominant_emotional_baseline": "alert and exacting",
            "threat_response": "asks for direct verification before moving forward",
            "defenses": [
                {
                    "defense_kind": "control",
                    "protects_what": "confidence in the handoff",
                    "short_term_benefit": "exposes weak pieces early",
                    "long_term_cost": "can make acceptance slower",
                    "evidence_quote_refs": refs,
                    "confidence_band": "high",
                }
            ],
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        6: {
            "close_relationship_role": "direct inspector",
            "recurring_loop_summary": "presses for proof, then relaxes when the proof holds",
            "conflict_style": "confront",
            "drawn_to": {"kind": "careful builders", "why": "they make quality visible", "evidence_quote_refs": refs, "confidence_band": "high"},
            "clashes_with": {"kind": "vague finishers", "why": "they make trust costly", "evidence_quote_refs": refs, "confidence_band": "high"},
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        7: {
            "trigger_event": "a milestone is claimed complete",
            "interpretation": "looks for proof that the claim holds",
            "emotion": "becomes alert",
            "action": "asks for targeted review",
            "outcome": "weak spots are either repaired or named",
            "belief_reinforcement": "trust grows when checks are reproducible",
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        8: {
            "stable_parts": ["will keep asking for evidence"],
            "fragile_spots": ["thin tests and vague completion claims"],
            "soft_spots": ["clear proof lowers resistance"],
            "communication_styles_likely_accepted": ["show changed files and targeted tests"],
            "communication_styles_that_trigger_defenses": ["asserting readiness without evidence"],
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
    }


def _profile_with_prior_med() -> PersonalityProfile:
    profile = PersonalityProfile(user_id="u_accept", display_name_hint="Acceptance User")
    prior = PersonalitySummary(
        summary="Usually seeks concrete proof before trusting a direction.",
        evidence_refs=(),
        confidence_band="med",
        last_updated_turn_id="t3",
    )
    return profile.with_section("step_1", prior, turn_id="t3")


def _run_scenario(name: str, outputs: dict[int, dict[str, object]], *, profile: PersonalityProfile | None = None) -> dict[str, object]:
    trigger_input = TriggerPolicyInput(user_id="u_accept", current_turn_index=4, current_hour_bucket=1, explicit_request=True)
    trigger = decide_trigger(trigger_input)
    result = run_personality_orchestrator(
        profile or PersonalityProfile(user_id="u_accept", display_name_hint="Acceptance User"),
        turn_id="t4",
        trigger_kind=trigger.kind,
        base_snapshot=_base_snapshot("t4"),
        extractors=_extractors(outputs),
    )
    cards = evidence_cards_from_personality_profile(result.profile, report_status=result.report.report_status)
    replay = run_personality_orchestrator(
        profile or PersonalityProfile(user_id="u_accept", display_name_hint="Acceptance User"),
        turn_id="t4",
        trigger_kind=trigger.kind,
        base_snapshot=_base_snapshot("t4"),
        extractors=_extractors(outputs),
    )
    deterministic = result.to_dict() == replay.to_dict()
    row = {
        "turns": ["t1", "t2", "t3", "t4"],
        "trigger_decisions": [trigger.to_dict()],
        "step_extractor_outputs": {str(k): v for k, v in sorted(outputs.items())},
        "personality_profiles_before_after": {
            "before": result.trace.profile_before,
            "after": result.profile.to_dict(),
        },
        "personality_reports": [result.report.to_dict()],
        "linter_findings": [finding.to_dict() for finding in result.report.linter_findings],
        "evidence_cards": [card.to_dict() for card in cards],
        "prompt_safe_evidence_cards": list(prompt_safe_cards(cards)),
        "linter_pass_fail_decisions": {
            "report_status": result.report.report_status,
            "ready_channel_returned": assemble_personality_report(result.profile, turn_id="t4", trigger_kind=trigger.kind).report_status == "ready",
        },
        "deterministic_replay": deterministic,
        "trace": result.trace.to_dict(),
    }
    _write_fixture(name, outputs, row)
    return row


def _write_fixture(name: str, outputs: Mapping[int, Mapping[str, object]], row: Mapping[str, object]) -> None:
    fixture_dir = FIXTURE_ROOT / name
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "input_step_outputs.json").write_text(
        json.dumps({str(k): v for k, v in sorted(outputs.items())}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (fixture_dir / "expected_state_trace.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (fixture_dir / "expected_linter_findings.json").write_text(
        json.dumps(row["linter_findings"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (fixture_dir / "rationale.md").write_text(
        f"# {name}\n\nRecorded extractor outputs are replayed through the deterministic M12.1 layer.\n",
        encoding="utf-8",
    )


def main() -> None:
    rich = _rich_outputs()
    sparse = _rich_outputs()
    sparse[8] = {"status": "insufficient_evidence", "reason": "growth hints need more stable evidence", "evidence_quote_refs": ["t4:q4"]}
    jargon = _rich_outputs()
    jargon[1] = {"summary": "This predictive system checks completion.", "evidence_quote_refs": ["t4:q4"], "confidence_band": "low"}
    dsm = _rich_outputs()
    dsm[4] = {
        "about_self": {"content_summary": "I trust what survives checks.", "evidence_quote_refs": ["t1:q1", "t2:q2"], "confidence_band": "med"},
        "about_others": {"content_summary": "Others seem narcissistic personality driven.", "evidence_quote_refs": ["t2:q2"], "confidence_band": "low"},
        "about_world": {"content_summary": "Strong work is inspectable.", "evidence_quote_refs": ["t3:q3"], "confidence_band": "low"},
    }
    moral = _rich_outputs()
    moral[8] = {
        "stable_parts": ["will keep asking for evidence"],
        "fragile_spots": ["thin tests"],
        "soft_spots": ["clear proof helps"],
        "communication_styles_likely_accepted": ["show tests"],
        "communication_styles_that_trigger_defenses": ["you should just trust it"],
        "evidence_quote_refs": ["t1:q1", "t2:q2"],
        "confidence_band": "med",
    }
    ready_profile = _run_scenario("rich_evidence_full_report", rich, profile=_profile_with_prior_med())["personality_profiles_before_after"]["after"]
    roleplay_profile = PersonalityProfile.from_dict(ready_profile)
    roleplay = {idx: {"status": "insufficient_evidence", "reason": "roleplay window is not stable evidence", "evidence_quote_refs": ["t4:q4"]} for idx in range(1, 9)}

    scenarios = {
        "rich_evidence_full_report": _run_scenario("rich_evidence_full_report", rich, profile=_profile_with_prior_med()),
        "sparse_evidence_insufficient_at_step_8": _run_scenario("sparse_evidence_insufficient_at_step_8", sparse, profile=_profile_with_prior_med()),
        "engineering_jargon_caught_by_linter": _run_scenario("engineering_jargon_caught_by_linter", jargon),
        "dsm_label_caught_by_linter": _run_scenario("dsm_label_caught_by_linter", dsm),
        "moral_verdict_caught_by_linter": _run_scenario("moral_verdict_caught_by_linter", moral),
        "roleplay_does_not_destabilise_profile": _run_scenario("roleplay_does_not_destabilise_profile", roleplay, profile=roleplay_profile),
    }
    audit = {
        name: {
            "passed": _scenario_passed(name, row),
            "threshold": _scenario_threshold(name),
            "report_status": row["personality_reports"][0]["report_status"],
            "deterministic_replay": row["deterministic_replay"],
        }
        for name, row in scenarios.items()
    }
    payload = {
        "milestone": "M12.1",
        "turns": ["t1", "t2", "t3", "t4"],
        "scenarios": scenarios,
        "calibration_audit_report": {
            "scenarios": audit,
            "all_passed": all(row["passed"] for row in audit.values()),
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _scenario_threshold(name: str) -> str:
    if name == "rich_evidence_full_report":
        return "ready report, all eight sections non-empty, at least one high confidence section"
    if name == "sparse_evidence_insufficient_at_step_8":
        return "Step 8 records insufficient_evidence without blocking report assembly"
    if "linter" in name or "label" in name or "moral" in name:
        return "report_status linter_failed and no evidence cards emitted"
    return "prior ready profile sections are not replaced by roleplay-only sparse output"


def _scenario_passed(name: str, row: Mapping[str, object]) -> bool:
    report = row["personality_reports"][0]
    cards = row["evidence_cards"]
    if not row["deterministic_replay"]:
        return False
    if name == "rich_evidence_full_report":
        sections = report["sections"]
        return (
            report["report_status"] == "ready"
            and len(sections) == 8
            and all(section["status"] != "never_computed" for section in sections)
            and any(section["confidence_band"] == "high" for section in sections)
        )
    if name == "sparse_evidence_insufficient_at_step_8":
        return report["report_status"] == "ready" and report["sections"][7]["status"] == "insufficient_evidence"
    if name in {"engineering_jargon_caught_by_linter", "dsm_label_caught_by_linter", "moral_verdict_caught_by_linter"}:
        return report["report_status"] == "linter_failed" and cards == []
    if name == "roleplay_does_not_destabilise_profile":
        after = row["personality_profiles_before_after"]["after"]
        before = row["personality_profiles_before_after"]["before"]
        return after["step_1_summary"] == before["step_1_summary"] and "step_1" in after["section_last_insufficient"]
    return False


if __name__ == "__main__":
    main()
