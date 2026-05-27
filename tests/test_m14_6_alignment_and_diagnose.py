from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.diagnose_idle_reflection import (
    VERDICT_CODES,
    summarize_log,
    verdicts_for_session,
)
from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


def _opted_m13() -> dict[str, object]:
    m13 = default_m13_drive_state()
    m13 = set_initiative_user_opt_in(m13, enabled=True)
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    return normalize_m13_drive_state(m13)


class _PlanLLM:
    def __init__(self, plan: dict[str, object]) -> None:
        self.plan = plan

    def complete_json(self, **_: object) -> dict[str, object]:
        return dict(self.plan)


def _base_plan(*, should_outreach: bool) -> dict[str, object]:
    return {
        "mode": "idle_introspection",
        "reflection_focus": None,
        "self_cognition_patch_proposal": {"apply": False},
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {
            "should_outreach": should_outreach,
            "reason": "open_item_followup",
            "suggested_intent": "check in without traceable evidence",
        },
    }


def test_plan_outreach_without_selector_target_emits_mismatch_and_downgrades(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "mismatch"),
        llm=_PlanLLM(_base_plan(should_outreach=True)),  # type: ignore[arg-type]
    )
    state = runtime.store.load()
    state.update(
        {
            "open_items": [],
            "pending_expectations": [],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": _opted_m13(),
        }
    )
    runtime.store.save(state)

    result = runtime.run_idle_introspection_turn(now=NOW, turn_index=9, structural_signals={})

    mismatch = next(event for event in result.audit_events if event["type"] == "IdlePlanStructuralMismatchEvent")
    assert mismatch["mismatch_reason_code"] == "no_eligible_expectation"
    assert mismatch["plan_recommendation_reason"] == "open_item_followup"
    assert result.outreach_recommendation["should_outreach"] is False
    assert result.outreach_recommendation["m14_6_downgraded_by_structural_selector"] is True


def test_plan_reflect_only_records_selector_target_without_auto_outreach(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "reflect_only"),
        llm=_PlanLLM(_base_plan(should_outreach=False)),  # type: ignore[arg-type]
    )
    state = runtime.store.load()
    state.update(
        {
            "open_items": [
                {
                    "id": "oi_due",
                    "status": "open",
                    "title": "benchmark follow-up",
                    "scheduled_intent_id": "intent_benchmark",
                    "due_at_epoch": NOW - 10_000,
                    "expected_window_seconds": 900,
                    "evidence_refs": ["mem_oi"],
                    "confidence": 0.95,
                }
            ],
            "long_term_memory": [{"id": "mem_oi", "content": "benchmark evidence", "salience": 0.8}],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": _opted_m13(),
        }
    )
    runtime.store.save(state)

    result = runtime.run_idle_introspection_turn(now=NOW, turn_index=10, structural_signals={})

    agreement = next(event for event in result.audit_events if event["type"] == "IdlePlanStructuralAgreementEvent")
    assert agreement["reason"] == "reflect_only_preferred"
    assert agreement["selected_target"]["trigger"] == "memory_efe_outreach"
    assert result.outreach_recommendation["should_outreach"] is False


def test_diagnose_summarize_log_streams_full_file_not_tail(tmp_path: Path) -> None:
    log = tmp_path / "conversation_log.jsonl"
    rows = [
        {"event": "m14_4_implicit_idle_audit", "type": "M14ImplicitIdleProactiveCheckEvent"},
        {"event": "m13_proactive_audit", "type": "IdleCognitiveTickEvent", "reject_reason": "generic_self_only_open_item"},
    ]
    rows.extend({"event": "noise", "type": "Noise", "i": i} for i in range(100))
    log.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_log(log)

    assert summary["counts"]["m14_4_implicit_idle_audit"] == 1
    assert summary["counts"]["IdleCognitiveTickEvent"] == 1


def test_diagnose_verdicts_are_closed_enum_codes() -> None:
    state = {
        "m13_drive_state": {
            "initiative": {
                "user_opt_in": True,
                "idle_introspection": {"user_opt_in": True, "idle_threshold_seconds": 90},
                "background_continuity": {"user_opt_in": True, "ticks_today": 0},
            }
        }
    }
    summary = {
        "counts": {"IdleCognitiveTickEvent": 1, "m14_2_audit": 1, "m14_idle_audit": 0},
        "latest_tick": {"reject_reason": "generic_self_only_open_item"},
        "latest_mismatch": {"mismatch_reason_code": "generic_self_only_open_item"},
        "latest_delivery": {},
        "latest_intro_plan": {},
    }

    verdicts = verdicts_for_session(
        state=state,
        log_summary=summary,
        lock_alive=True,
        has_lock=True,
        idle_elapsed=999,
        structural_should_run=True,
    )

    assert verdicts
    assert set(verdicts) <= VERDICT_CODES
    assert "DAEMON_PROCESS_ALIVE_NO_BACKGROUND_TICKS" in verdicts
    assert "IDLE_INTRO_PLAN_SELECTOR_MISMATCH" in verdicts


def test_rejected_delivery_assessment_is_not_counted_as_delivered(tmp_path: Path) -> None:
    log = tmp_path / "conversation_log.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "m13_proactive_audit",
                        "type": "ProactiveDeliveryAssessmentEvent",
                        "at": NOW,
                        "assessment": {"allow_delivery": False, "confidence": 0.9},
                    }
                ),
                json.dumps(
                    {
                        "event": "m13_proactive_audit",
                        "type": "M13ProactiveSuppressionEvent",
                        "at": NOW + 1,
                        "reason_code": "delivery_assessor_reject",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log)

    assert summary["latest_delivery"] == {}
