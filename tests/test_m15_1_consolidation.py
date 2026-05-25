from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_1_background_continuity import set_background_continuity_opt_in
from segmentum.dialogue.runtime.m15_consolidation import ConsolidationOwner, fingerprint_class
from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, aggregate_fe_components, build_episode
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


def _state() -> dict[str, object]:
    return {
        "pending_expectations": [],
        "open_items": [],
        "short_term_memory": [],
        "long_term_memory": [],
        "temporal_state": {"last_turn_index": 2, "last_user_turn_at": NOW - 3600},
        "m13_drive_state": default_m13_drive_state(),
    }


def _ledger_with_episode(root: Path, state: dict[str, object], *, action: str = "answer", delta_bad: bool = False) -> EpisodeLedger:
    ledger = EpisodeLedger(root)
    before = aggregate_fe_components(state)
    after = dict(before)
    after["expectation_prediction_error_proxy"] = 1.0 if delta_bad else 0.0
    ledger.append(
        build_episode(
            at=NOW,
            turn_index=1,
            phase="user_turn",
            state=state,
            action=action,
            action_trigger="user_message",
            evidence_refs=["mem_a"],
            components_before={**before, "expectation_prediction_error_proxy": 1.0},
            components_after=after,
            outcome_summary="confirmed",
        )
    )
    return ledger


def test_run_skipped_when_ledger_empty_emits_deferred(tmp_path: Path) -> None:
    state = _state()
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=1,
        ledger=EpisodeLedger(tmp_path),
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert result.ran is False
    assert result.deferred_reason == "ledger_empty"
    assert result.events[-1]["type"] == "ConsolidationDeferredEvent"


def test_run_skipped_when_user_turn_in_progress(tmp_path: Path) -> None:
    state = _state()
    state["m13_ui_turn_in_progress"] = True
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=1,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert result.ran is False
    assert result.deferred_reason == "user_turn_in_progress"


def test_run_skipped_when_budget_exceeded_emits_deferred(tmp_path: Path) -> None:
    state = _state()
    m13 = state["m13_drive_state"]  # type: ignore[assignment]
    m13["m15_consolidation"] = {  # type: ignore[index]
        "runs_by_day": {str(NOW // 86400): 6},
    }
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=1,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert result.ran is False
    assert result.deferred_reason == "budget_exceeded"


def test_min_run_interval_blocks_back_to_back_runs(tmp_path: Path) -> None:
    state = _state()
    ledger = _ledger_with_episode(tmp_path, state)
    first = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=1,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    second = ConsolidationOwner.maybe_run(
        state,
        now=NOW + 60,
        turn_index=1,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert first.ran is True
    assert second.ran is False
    assert second.deferred_reason == "recently_ran_within_min_interval"


def test_merge_duplicate_expectations_uses_structural_overlap_not_content(tmp_path: Path) -> None:
    state = _state()
    state["pending_expectations"] = [
        {
            "id": "exp_a",
            "status": "pending",
            "content": "alpha completely different",
            "confidence": 0.7,
            "evidence_refs": ["ev1", "ev2"],
            "bound_memory_ids": ["mem1"],
            "created_at": NOW - 10,
        },
        {
            "id": "exp_b",
            "status": "pending",
            "content": "beta no lexical overlap",
            "confidence": 0.9,
            "evidence_refs": ["ev1", "ev2"],
            "bound_memory_ids": ["mem1"],
            "created_at": NOW,
        },
        {
            "id": "exp_c",
            "status": "pending",
            "content": "alpha completely different",
            "confidence": 0.9,
            "evidence_refs": ["ev3"],
            "bound_memory_ids": ["mem2"],
        },
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    rows = {row["id"]: row for row in state["pending_expectations"]}  # type: ignore[index]
    assert result.ran is True
    assert rows["exp_b"]["status"] == "merged_into:exp_a"
    assert rows["exp_a"]["merged_from"] == ["exp_b"]
    assert rows["exp_c"].get("status") == "pending"


def test_merge_does_not_delete_source_rows_only_marks_status(tmp_path: Path) -> None:
    state = _state()
    state["open_items"] = [
        {"id": "oi_a", "status": "open", "confidence": 0.8, "evidence_refs": ["ev1"], "bound_memory_ids": ["mem1"]},
        {"id": "oi_b", "status": "open", "confidence": 0.8, "evidence_refs": ["ev1"], "bound_memory_ids": ["mem1"]},
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert len(state["open_items"]) == 2  # type: ignore[arg-type]
    assert state["open_items"][1]["status"] == "merged_into:oi_a"  # type: ignore[index]


def test_merge_canonical_row_is_older_created_at_even_when_second(tmp_path: Path) -> None:
    state = _state()
    state["pending_expectations"] = [
        {
            "id": "exp_new",
            "status": "pending",
            "confidence": 0.7,
            "evidence_refs": ["ev1"],
            "bound_memory_ids": ["mem1"],
            "created_at": NOW,
        },
        {
            "id": "exp_old",
            "status": "pending",
            "confidence": 0.8,
            "evidence_refs": ["ev1"],
            "bound_memory_ids": ["mem1"],
            "created_at": NOW - 100,
        },
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    rows = {row["id"]: row for row in state["pending_expectations"]}  # type: ignore[index]
    assert rows["exp_new"]["status"] == "merged_into:exp_old"
    assert rows["exp_old"]["merged_from"] == ["exp_new"]
    assert any(op.retained_id == "exp_old" for op in result.op_results)


def test_promote_stm_to_ltm_routes_through_memory_gate(tmp_path: Path) -> None:
    state = _state()
    state["short_term_memory"] = [
        {
            "id": "stm_a",
            "kind": "episode",
            "content": "User repeatedly prefers concise status updates.",
            "confidence": 0.9,
            "salience": 0.8,
            "recall_count_session": 3,
            "recall_scores": [0.6, 0.7, 0.8],
            "created_at": NOW - 100,
            "evidence_refs": ["ev1", "ev2"],
            "memory_gate_decision": {"write_score": 0.8},
            "value_proxy": 0.8,
            "surprise_proxy": 0.7,
        }
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert any(row.get("promoted_from") == "stm_a" for row in state["long_term_memory"])  # type: ignore[union-attr]
    assert any(event.get("type") == "MemoryGateCommitEvent" for event in result.events)


def test_promote_stm_to_ltm_requires_recall_floor(tmp_path: Path) -> None:
    state = _state()
    state["short_term_memory"] = [
        {
            "id": "stm_low",
            "kind": "episode",
            "content": "Barely recalled note.",
            "confidence": 0.9,
            "salience": 0.8,
            "recall_count_session": 3,
            "recall_scores": [0.1, 0.2],
            "created_at": NOW - 100,
            "evidence_refs": ["ev1", "ev2"],
            "memory_gate_decision": {"write_score": 0.8},
        }
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert not state["long_term_memory"]  # type: ignore[truthy-bool]
    assert any(row.op == "promote_stm_to_ltm" and not row.committed for row in result.op_results)


def test_abstract_repeated_paths_requires_improving_delta_and_marks_consumed(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    before = aggregate_fe_components(state)
    after = {**before, "expectation_prediction_error_proxy": 0.0}
    for index in range(4):
        ledger.append(
            build_episode(
                at=NOW + index,
                turn_index=index,
                phase="user_turn",
                state=state,
                action="answer",
                action_trigger="user_message",
                evidence_refs=[f"ev{index}", "ev_shared"],
                components_before={**before, "expectation_prediction_error_proxy": 1.0},
                components_after=after,
                outcome_summary="confirmed",
            )
        )
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW + 1000,
        turn_index=6,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert any(row.get("kind") == "habit" for row in state["long_term_memory"])  # type: ignore[union-attr]
    assert any(row.op == "abstract_path" and row.committed for row in result.op_results)
    records = [
        json.loads(line)
        for line in (tmp_path / "memory_dynamics_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("record_type") == "abstraction_consumed" for row in records)


def test_fingerprint_class_ignores_traction_order(tmp_path: Path) -> None:
    payload_a = {
        "state_fingerprint": "fp_a",
        "state_fingerprint_payload": {
            "boredom_band": "low",
            "reward_band": "medium",
            "behavior_band": "medium",
            "relation_band": "low",
            "memory_efe_should_outreach": False,
            "memory_efe_selected_policy": "wait",
            "open_items_concrete_count": 1,
            "unsettled_pending_settlement_count": 0,
            "top_3_traction_actions": ["answer", "clarify"],
        },
    }
    payload_b = {
        "state_fingerprint": "fp_b",
        "state_fingerprint_payload": {
            **payload_a["state_fingerprint_payload"],
            "top_3_traction_actions": ["clarify", "answer"],
        },
    }
    assert fingerprint_class(payload_a) == fingerprint_class(payload_b)


def test_decay_extension_does_not_touch_open_items_or_pending_expectations(tmp_path: Path) -> None:
    state = _state()
    state["pending_expectations"] = [{"id": "exp", "salience": 0.01, "status": "pending"}]
    state["open_items"] = [{"id": "oi", "salience": 0.01, "status": "open"}]
    state["long_term_memory"] = [
        {"id": "ltm_low", "content": "low", "salience": 0.04, "confidence": 0.5, "created_at": NOW - 86400 * 2}
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    assert state["pending_expectations"][0]["status"] == "pending"  # type: ignore[index]
    assert state["open_items"][0]["status"] == "open"  # type: ignore[index]
    assert state["long_term_memory"][0]["status"] == "archived"  # type: ignore[index]


def test_archive_capped_per_run(tmp_path: Path) -> None:
    state = _state()
    state["long_term_memory"] = [
        {
            "id": f"ltm_low_{idx}",
            "content": "low",
            "salience": 0.04,
            "confidence": 0.5,
            "created_at": NOW - 86400 * 2,
        }
        for idx in range(8)
    ]
    ledger = _ledger_with_episode(tmp_path, state)
    ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    archived = [row for row in state["long_term_memory"] if row.get("status") == "archived"]  # type: ignore[union-attr]
    assert len(archived) == 3


def test_consolidation_run_event_summarizes_ops_without_subjective_wording(tmp_path: Path) -> None:
    state = _state()
    ledger = _ledger_with_episode(tmp_path, state)
    result = ConsolidationOwner.maybe_run(
        state,
        now=NOW,
        turn_index=2,
        ledger=ledger,
        budget={"triggered_by": "idle_cognitive_tick"},
    )
    summary = result.events[-1]
    assert summary["type"] == "ConsolidationRunEvent"
    assert {"ops_attempted", "ops_committed", "ops_rejected", "budget_remaining_runs_today"} <= set(summary)
    blob = json.dumps(result.events, ensure_ascii=False).casefold()
    assert "dream" not in blob
    assert "sleep" not in blob
    assert "felt" not in blob


def test_idle_cognitive_tick_invokes_consolidation(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "runtime"), llm=None)
    state = runtime.store.load()
    state["long_term_memory"] = [
        {"id": "ltm_low", "content": "low", "salience": 0.04, "confidence": 0.5, "created_at": NOW - 86400 * 2}
    ]
    runtime.store.save(state)
    runtime.run_idle_cognitive_tick(turn_index=3, idle_seconds=120.0, now=NOW)
    rows = [
        json.loads(line)
        for line in (runtime.store.root / "conversation_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "m15_consolidation_audit" for row in rows)
    saved = runtime.store.load()
    assert saved["m13_drive_state"]["m15_consolidation"]["last_run_at"] == NOW


class _BackgroundLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": {
                    "topic": "open item",
                    "evidence_refs": ["mem_bg_1"],
                    "reflection_kind": "open_item",
                },
                "self_cognition_patch_proposal": {
                    "apply": False,
                    "summary_delta": "",
                    "new_identity_tensions": [],
                    "new_known_limits": [],
                    "evidence_refs": [],
                    "confidence": 0.0,
                },
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
            }
        return {
            "reply": "ok",
            "reply_action": "answer",
            "llm_thinking_result": {},
            "memory_writes": [],
        }


def test_background_tick_invokes_consolidation_when_introspection_runs(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "bg"), llm=_BackgroundLLM())
    state = runtime.store.load()
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    m13 = set_background_continuity_opt_in(m13, enabled=True, runner_kind="inline")
    state.update(
        {
            "open_items": [
                {
                    "id": "item_bg_1",
                    "title": "follow up",
                    "status": "open",
                    "next_check": "next_user_turn",
                    "evidence_refs": ["mem_bg_1"],
                    "bound_memory_ids": ["mem_bg_1"],
                    "confidence": 0.8,
                }
            ],
            "short_term_memory": [{"id": "mem_bg_1", "content": "traceable follow-up memory"}],
            "temporal_state": {
                "last_turn_at": NOW - 7200,
                "last_user_turn_at": NOW - 7200,
                "last_turn_index": 2,
                "last_reply": "ok",
            },
            "m13_drive_state": m13,
        }
    )
    runtime.store.save(state)
    _ledger_with_episode(runtime.store.root, state)
    result = runtime.run_background_self_tick(runner_kind="inline")
    saved = runtime.store.load()
    assert result.get("ran_introspection") is True
    assert saved["m13_drive_state"]["m15_consolidation"]["last_run_at"] > 0
