from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import (
    DELIVERY_ASSESSOR_MARKER,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m13_memory_efe import evaluate_memory_efe, normalize_expectations_for_efe
from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore
from segmentum.dialogue.runtime.m14_2_self_loop import M142SelfLoopDaemon
from segmentum.dialogue.runtime.m14_3_open_item_migration import (
    audit_open_items_for_efe,
    apply_open_item_traceability_patches,
    propose_open_item_traceability_patches,
)
from segmentum.dialogue.runtime.m14_3_proactive_alignment import (
    ProactiveTarget,
    build_traceable_proactive_intent,
    select_proactive_target,
)
from segmentum.dialogue.runtime.m14_idle_reflector import build_structural_idle_plan
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_800_000_000


def _opted_m13() -> dict[str, object]:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    initiative = normalize_initiative_state(normalize_m13_drive_state(m13)["initiative"])
    initiative["implicit_idle_delivery"] = True
    m13["initiative"] = initiative
    return m13


def test_vague_open_item_default_blocks_m13_3_proposal() -> None:
    state = {
        "open_items": [{"id": "oi_later", "status": "open", "title": "sunset", "next_check": "later"}],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _opted_m13(),
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=4,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
    )
    assert check.proposal is None
    event = [row for row in check.events if row.get("type") == "M13ProactiveSuppressionEvent"][-1]
    assert event["reason_code"] == "no_traceable_proactive_target"


def test_memory_efe_outreach_intent_uses_content_summary_not_next_check() -> None:
    row = {
        "id": "item_001",
        "content": "Who is 老爹 in the prior thread?",
        "next_check": "next_user_turn",
    }
    intent = build_traceable_proactive_intent(row)
    assert "老爹" in intent
    assert "later" not in intent


def test_opponent_strength_pre_block_emits_reason_code() -> None:
    m13 = _opted_m13()
    reward = normalize_m13_drive_state(m13)["affective_reward_proxy"]
    reward["opponent_strength"] = 0.8
    m13["affective_reward_proxy"] = reward
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": m13,
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=2,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
    )
    assert check.suppression_reason_code == "opponent_strength_pre_block"


def test_boredom_target_blocked_when_memory_efe_eligible() -> None:
    m13 = _opted_m13()
    boredom = normalize_m13_drive_state(m13)["boredom"]
    boredom.update(
        {
            "boredom_level": 0.8,
            "last_exploration_target": "football",
            "recent_plan_terms": ["football"],
        }
    )
    m13["boredom"] = boredom
    memory_efe = normalize_m13_drive_state(m13)["memory_efe"]
    memory_efe.update(
        {
            "should_outreach": False,
            "traceable_expectation_id": "exp_001",
            "eligible_for_efe": [{"expectation_id": "exp_001"}],
        }
    )
    m13["memory_efe"] = memory_efe
    target = select_proactive_target({"open_items": []}, m13, structural_signals={})
    assert target is None


def test_boredom_target_without_agreement() -> None:
    m13 = _opted_m13()
    boredom = normalize_m13_drive_state(m13)["boredom"]
    boredom.update(
        {
            "boredom_level": 0.42,
            "last_exploration_target": "small parser cleanup",
            "recent_plan_terms": ["parser", "cleanup"],
        }
    )
    m13["boredom"] = boredom
    state = {"open_items": [], "pending_expectations": [], "temporal_state": {}, "m13_drive_state": m13}
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=4,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
    )
    assert check.proposal is not None
    assert check.proposal.trigger == "boredom_exploration_target"
    assert check.proposal.source_kind == "boredom_exploration"


def test_relationship_pull_proactive_without_agreement() -> None:
    m13 = _opted_m13()
    normalized = normalize_m13_drive_state(m13)
    normalized["traction_by_action"] = {"empathize|zq": 0.72, "answer|zq": 0.2}
    normalized["relation_path_precision"] = {"zq": 0.58}
    reward = normalized["affective_reward_proxy"]
    reward["path_feels_stale_proxy"] = True
    normalized["affective_reward_proxy"] = reward
    state = {
        "open_items": [],
        "pending_expectations": [],
        "relationship_value_memories": {
            "by_user": {
                "zq": [
                    {
                        "id": "rvm_zq_1",
                        "summary": "zq values plain direct warmth when returning to a thread.",
                        "prediction_constraint": "A concise warm continuation lowers relationship friction.",
                        "priority": "high",
                        "confidence": 0.86,
                    }
                ]
            }
        },
        "temporal_state": {"last_share_trace": {"user_id": "zq"}},
        "m13_drive_state": normalized,
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=4,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
    )
    assert check.proposal is not None
    assert check.proposal.trigger == "relationship_reconnect_pull"
    assert check.proposal.source_kind != "boredom_exploration"
    assert "rvm_zq_1" in check.proposal.trigger_evidence_refs
    assert "active_relationship_value_memory" in check.proposal.selection_reason_codes


def test_memory_efe_outreach_idle() -> None:
    m13 = _opted_m13()
    boredom = normalize_m13_drive_state(m13)["boredom"]
    boredom.update(
        {
            "boredom_level": 0.8,
            "last_exploration_target": "fallback boredom topic",
            "recent_plan_terms": ["fallback"],
        }
    )
    m13["boredom"] = boredom
    state = {
        "pending_expectations": [],
        "open_items": [
            {
                "id": "exp_mem",
                "status": "open",
                "title": "check the bound memory tension",
                "scheduled_intent_id": "intent_mem",
                "due_at_epoch": NOW - 10_000,
                "expected_window_seconds": 900,
                "evidence_refs": ["mem_bound"],
                "bound_memory_ids": ["mem_bound"],
                "confidence": 0.95,
            }
        ],
        "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
        "m13_drive_state": m13,
    }
    evaluation = evaluate_memory_efe(
        state,
        phase="idle",
        now=NOW,
        turn_index=5,
        user_active=False,
        retrieved_memories=[{"id": "mem_bound", "content": "bound memory tension"}],
    )
    target = select_proactive_target(state, m13, memory_efe_evaluation=evaluation, structural_signals={})
    assert target is not None
    assert target.trigger == "memory_efe_outreach"
    assert target.source_kind != "boredom_exploration"
    assert "mem_bound" in target.evidence_refs


def test_memory_efe_proactive_without_explicit_agreement_from_idle_recall(tmp_path: Path) -> None:
    now = int(time.time())
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "memory_efe_no_agreement"), llm=None)
    state = runtime.store.load()
    m13 = _opted_m13()
    normalized = normalize_m13_drive_state(m13)
    normalized["boredom"]["boredom_level"] = 0.0
    normalized["affective_reward_proxy"]["last_net_reward_proxy"] = 0.0
    state.update(
        {
            "open_items": [],
            "pending_expectations": [
                {
                    "id": "mem_dyn_topic_1",
                    "source": "memory_dynamics_adapter",
                    "verify_on": "later",
                    "status": "pending",
                    "content": "reconnect around zq's science-and-ghost-hunting curiosity from the remembered thread",
                    "confidence": 0.95,
                    "due_at_epoch": now - 7200,
                    "expected_window_seconds": 900,
                    "evidence_refs": ["mem_science_ghost_thread"],
                    "bound_memory_ids": ["mem_science_ghost_thread"],
                },
                {
                    "id": "mem_dyn_topic_2",
                    "source": "memory_dynamics_adapter",
                    "verify_on": "later",
                    "status": "pending",
                    "content": "reconnect around zq's earlier Li An'an investigation thread",
                    "confidence": 0.95,
                    "due_at_epoch": now - 7100,
                    "expected_window_seconds": 900,
                    "evidence_refs": ["mem_li_anan_thread"],
                    "bound_memory_ids": ["mem_li_anan_thread"],
                }
            ],
            "short_term_memory": [],
            "long_term_memory": [
                {
                    "id": "mem_science_ghost_thread",
                    "content": "zq was curious about the boundary between science and ghost-hunting stories.",
                    "keywords": ["science", "ghost-hunting", "zq"],
                    "salience": 0.9,
                },
                {
                    "id": "mem_li_anan_thread",
                    "content": "zq had an unresolved Li An'an investigation thread with Hu Tao.",
                    "keywords": ["Li An'an", "investigation", "zq"],
                    "salience": 0.9,
                }
            ],
            "temporal_state": {
                "last_user_turn_at": now - 7200,
                "last_turn_at": now - 7200,
                "last_turn_index": 4,
                "last_user_text": "",
            },
            "m13_drive_state": normalized,
        }
    )
    runtime.store.save(state)

    check = runtime.maybe_propose_proactive_turn(
        turn_index=8,
        idle_seconds=999,
        implicit_idle_request=True,
    )

    proposal = check["proposal"]
    assert proposal is not None
    assert proposal["trigger"] == "memory_efe_outreach"
    assert proposal["source_kind"] == "memory_dynamics_expectation"
    assert proposal["traceable_expectation_id"] == "mem_dyn_topic_1"
    assert "mem_science_ghost_thread" in proposal["trigger_evidence_refs"]
    assert "memory_efe_should_outreach" in proposal["selection_reason_codes"]
    assert "boredom" not in proposal["source_kind"]
    assert not state["open_items"]
    assert all(not row.get("scheduled_intent_id") for row in state["pending_expectations"])

    events = check["events"]
    refresh = next(row for row in events if row.get("type") == "IdleProactiveDriveRefreshEvent")
    memory_event = next(row for row in events if row.get("type") == "MemoryEfeEvaluationEvent")
    assert refresh["order"] == "recall_then_memory_efe_then_m13_drive_bands_before_target_selection"
    assert "mem_science_ghost_thread" in refresh["retrieved_ids"]
    assert "mem_li_anan_thread" in refresh["retrieved_ids"]
    assert memory_event["should_outreach"] is True
    assert memory_event["selected_policy"] == "outreach"
    assert memory_event["traceable_expectation_id"] == "mem_dyn_topic_1"


def test_tension_backed_new_expectation_gets_memory_dynamics_trace_in_path_b(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace"), llm=None)
    state = runtime.store.load()
    runtime._apply_thinking_writes(
        state,
        {
            "reply": "short",
            "new_expectations": [
                {
                    "id": "exp_tension",
                    "content": "continue the science versus supernatural tension",
                    "confidence": 0.8,
                    "verify_on": "next_user_turn",
                }
            ],
        },
        user_text="I am stuck on this unresolved tension.",
        now=3001,
        memory_dynamics={
            "write_candidates": [
                {
                    "content": "The user is stuck on a science versus supernatural tension.",
                    "confidence": 0.8,
                    "evidence": "user_text",
                }
            ]
        },
    )

    row = state["pending_expectations"][-1]
    assert row["id"] == "exp_tension"
    assert row["source"] == "memory_dynamics_adapter"
    assert row["verify_on"] == "memory_dynamics_idle"
    assert row["evidence_refs"] == ["stm_turn_3001"]
    assert row["bound_memory_ids"] == ["stm_turn_3001"]


def test_structured_memory_dynamics_binding_without_memory_candidate_gets_trace(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace_structured"), llm=None)
    state = runtime.store.load()
    runtime._apply_thinking_writes(
        state,
        {
            "reply": "想了一整晚还头疼，那说明你拆到真东西了。",
            "new_expectations": [
                {
                    "id": "exp_truth_thread",
                    "content": "用户会回应是否接受'拆到真东西了'这个说法，或者继续追问'真东西'是什么",
                    "confidence": 0.6,
                    "verify_on": "next_user_turn",
                    "memory_dynamics_binding": {
                        "should_bind_idle": True,
                        "reason_codes": ["memory_prediction_tension"],
                        "evidence_refs": ["stm_turn_4001"],
                    },
                }
            ],
        },
        user_text="想了一晚上，想得我头疼，还是想不通",
        now=4001,
        memory_dynamics={"write_candidates": []},
    )

    row = state["pending_expectations"][-1]
    assert row["id"] == "exp_truth_thread"
    assert row["source"] == "memory_dynamics_adapter"
    assert row["verify_on"] == "memory_dynamics_idle"
    assert row["evidence_refs"] == ["stm_turn_4001"]
    assert row["bound_memory_ids"] == ["stm_turn_4001"]


def test_user_text_tension_words_alone_do_not_create_memory_dynamics_trace(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace_no_keyword"), llm=None)
    state = runtime.store.load()
    runtime._apply_thinking_writes(
        state,
        {
            "new_expectations": [
                {
                    "id": "exp_truth_thread",
                    "content": "用户会回应是否接受'拆到真东西了'这个说法，或者继续追问'真东西'是什么",
                    "confidence": 0.6,
                    "verify_on": "next_user_turn",
                }
            ],
        },
        user_text="想了一晚上，想得我头疼，还是想不通",
        now=4003,
        memory_dynamics={"write_candidates": []},
    )

    row = state["pending_expectations"][-1]
    assert row["id"] == "exp_truth_thread"
    assert row.get("source") != "memory_dynamics_adapter"
    assert row["verify_on"] == "next_user_turn"


def test_short_duplicate_expectation_id_is_rewritten_and_trace_anchored(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace_rewrite"), llm=None)
    state = runtime.store.load()
    state["pending_expectations"] = [
        {
            "id": "exp_001",
            "status": "pending",
            "content": "older topic",
            "verify_on": "next_user_turn",
            "evidence_refs": ["stm_turn_old"],
        }
    ]
    runtime._apply_thinking_writes(
        state,
        {
            "reply": "short",
            "new_expectations": [
                {
                    "id": "exp_001",
                    "content": "new topic should not reuse local id",
                    "confidence": 0.7,
                    "verify_on": "next_user_turn",
                }
            ],
        },
        user_text="new user text",
        now=5001,
        turn_index=7,
        memory_dynamics={"write_candidates": []},
    )

    row = state["pending_expectations"][-1]
    assert row["id"].startswith("exp_7_5001_")
    assert row["source_expectation_id"] == "exp_001"
    assert row["created_turn_index"] == 7
    assert row["evidence_refs"] == ["stm_turn_5001"]


def test_duplicate_active_pending_expectation_signature_is_not_appended(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace_dedupe"), llm=None)
    state = runtime.store.load()
    existing = {
        "id": "exp_existing",
        "status": "pending",
        "source": "thinking_prompt",
        "content": "same structural expectation",
        "verify_on": "next_user_turn",
        "evidence_refs": ["stm_turn_6001"],
    }
    state["pending_expectations"] = [existing]
    runtime._apply_thinking_writes(
        state,
        {
            "new_expectations": [
                {
                    "content": "same structural expectation",
                    "confidence": 0.7,
                    "verify_on": "next_user_turn",
                }
            ],
        },
        user_text="same turn",
        now=6001,
        turn_index=3,
        memory_dynamics={"write_candidates": []},
    )

    assert state["pending_expectations"] == [existing]


def test_generic_unclear_intent_expectation_does_not_get_memory_dynamics_trace(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "trace_generic"), llm=None)
    state = runtime.store.load()
    runtime._apply_thinking_writes(
        state,
        {
            "new_expectations": [
                {
                    "id": "exp_generic",
                    "content": "用户未明确说明来意（闲聊、求助还是其他目的）",
                    "confidence": 0.7,
                    "verify_on": "next_user_turn",
                }
            ],
        },
        user_text="想了一晚上，还是想不通",
        now=4002,
        memory_dynamics={"write_candidates": []},
    )

    row = state["pending_expectations"][-1]
    assert row["id"] == "exp_generic"
    assert row.get("source") != "memory_dynamics_adapter"
    assert row["verify_on"] == "next_user_turn"


def test_generic_self_only_open_item_does_not_become_memory_efe_target() -> None:
    m13 = _opted_m13()
    evaluation = SimpleNamespace(
        should_outreach=True,
        traceable_expectation_id="item_001",
        evidence_refs=["item_001"],
        reason_codes=["memory_backed_social_prediction_error"],
        eligible_for_efe=[
            {
                "id": "item_001",
                "expectation_id": "item_001",
                "source_kind": "open_item",
                "content_summary": "unclear user intent",
                "evidence_refs": ["item_001"],
            }
        ],
    )
    target = select_proactive_target(
        {"open_items": [], "pending_expectations": []},
        m13,
        memory_efe_evaluation=evaluation,
        structural_signals={},
    )
    assert target is None


def test_generic_open_item_does_not_shadow_relationship_pull() -> None:
    m13 = _opted_m13()
    normalized = normalize_m13_drive_state(m13)
    normalized["traction_by_action"] = {"empathize|zq": 0.72}
    normalized["relation_path_precision"] = {"zq": 0.58}
    reward = normalized["affective_reward_proxy"]
    reward["path_feels_stale_proxy"] = True
    normalized["affective_reward_proxy"] = reward
    evaluation = SimpleNamespace(
        should_outreach=True,
        traceable_expectation_id="item_001",
        evidence_refs=["item_001"],
        reason_codes=["memory_backed_social_prediction_error"],
        eligible_for_efe=[
            {
                "id": "item_001",
                "expectation_id": "item_001",
                "source_kind": "open_item",
                "content_summary": "unclear user intent",
                "evidence_refs": ["item_001"],
            }
        ],
    )
    state = {
        "open_items": [],
        "pending_expectations": [],
        "relationship_value_memories": {
            "by_user": {
                "zq": [
                    {
                        "id": "rvm_zq_1",
                        "summary": "zq values plain direct warmth when returning to a thread.",
                        "prediction_constraint": "A concise warm continuation lowers relationship friction.",
                        "priority": "high",
                        "confidence": 0.86,
                    }
                ]
            }
        },
        "temporal_state": {"last_share_trace": {"user_id": "zq"}},
        "m13_drive_state": normalized,
    }
    target = select_proactive_target(state, normalized, memory_efe_evaluation=evaluation, structural_signals={})
    assert target is not None
    assert target.trigger == "relationship_reconnect_pull"
    assert target.source_kind != "open_item"


def test_silence_alone_does_not_raise_any_drive_scalar(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "silence"), llm=None)
    state = runtime.store.load()
    m13 = _opted_m13()
    normalized = normalize_m13_drive_state(m13)
    normalized["boredom"]["boredom_level"] = 0.12
    normalized["traction_by_action"] = {"answer|zq": 0.21}
    normalized["relation_path_precision"] = {"zq": 0.18}
    normalized["affective_reward_proxy"]["last_net_reward_proxy"] = 0.22
    state.update(
        {
            "open_items": [],
            "pending_expectations": [],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": normalized,
        }
    )
    runtime.store.save(state)
    before = normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])
    check = runtime.maybe_propose_proactive_turn(
        turn_index=8,
        idle_seconds=999,
        implicit_idle_request=True,
    )
    after = normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])
    assert check["proposal"] is None
    assert after["boredom"]["boredom_level"] == before["boredom"]["boredom_level"]
    assert after["affective_reward_proxy"]["last_net_reward_proxy"] == before["affective_reward_proxy"]["last_net_reward_proxy"]
    assert after["traction_by_action"] == before["traction_by_action"]
    assert after["relation_path_precision"] == before["relation_path_precision"]


def test_no_duplicate_proactive_when_scheduled_active() -> None:
    m13 = _opted_m13()
    boredom = normalize_m13_drive_state(m13)["boredom"]
    boredom.update(
        {
            "boredom_level": 0.9,
            "last_exploration_target": "should lose to schedule",
            "recent_plan_terms": ["schedule"],
        }
    )
    m13["boredom"] = boredom
    state = {"open_items": [], "pending_expectations": [], "temporal_state": {}, "m13_drive_state": m13}
    structural = {
        "queued_outreach": [
            {
                "status": "pending",
                "trigger": "scheduled_outreach",
                "source_intent_id": "intent_scheduled",
                "ordinary_language_intent": "Follow up on the scheduled outreach request.",
                "evidence_refs": ["intent_scheduled"],
            }
        ]
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=7,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
        structural_signals=structural,
    )
    proposals = [row for row in check.events if row.get("type") == "M13ProactiveProposalEvent"]
    assert check.proposal is not None
    assert check.proposal.trigger == "scheduled_outreach"
    assert len(proposals) == 1


def test_open_item_migration_promotes_only_traceable_vague_items() -> None:
    rows = [
        {
            "id": "oi_trace",
            "status": "open",
            "title": "follow Li An'an thread",
            "next_check": "later",
            "evidence_refs": ["mem_a"],
            "created_at": NOW - 100,
        },
        {"id": "oi_diag", "status": "open", "title": "loose", "next_check": "later"},
    ]
    suggestions = audit_open_items_for_efe(rows)
    assert [item.reason_code for item in suggestions] == [
        "traceable_vague_open_item_can_use_next_user_turn",
        "vague_open_item_missing_evidence_or_created_at",
    ]
    state = {"open_items": [dict(row) for row in rows]}
    patches = propose_open_item_traceability_patches(state["open_items"], now=NOW)
    assert len(patches) == 1
    assert apply_open_item_traceability_patches(state, patches, source="test", reason="acceptance") == 1
    assert state["open_items"][0]["next_check"] == "next_user_turn"
    assert state["open_items"][1]["next_check"] == "later"


class _IdleLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "idle_introspection" in user_prompt or "M14" in system_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": None,
                "self_cognition_patch_proposal": {"apply": False},
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
            }
        return {"reply": "ok", "reply_action": "answer", "llm_thinking_result": {}}


def test_idle_introspection_logs_retrieve_before_memory_efe(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "idle")
    state = store.load()
    state.update(
        {
            "pending_expectations": [
                {
                    "id": "exp_bound",
                    "status": "open",
                    "content": "check the bound memory",
                    "due_at_epoch": NOW - 3600,
                    "bound_memory_ids": ["mem_bound"],
                    "evidence_refs": ["mem_bound"],
                    "confidence": 0.9,
                }
            ],
            "long_term_memory": [{"id": "mem_bound", "content": "bound memory evidence", "salience": 0.8}],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200, "last_turn_index": 3},
            "m13_drive_state": _opted_m13(),
        }
    )
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_IdleLLM())  # type: ignore[arg-type]
    result = runtime.run_idle_introspection_turn(now=NOW, turn_index=4, structural_signals={})
    event_types = [row["type"] for row in result.audit_events]
    assert event_types.index("MemoryDynamicsIdleSummaryEvent") < event_types.index("MemoryEfeEvaluationEvent")
    order = [row for row in result.audit_events if row["type"] == "IdleEfeRecallOrderEvent"][-1]
    assert "mem_bound" in order["retrieved_ids"]


def test_epistemic_pe_uses_retrieved_memories_on_idle_path(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "epi")
    state = store.load()
    state.update(
        {
            "pending_expectations": [
                {
                    "id": "exp_bound",
                    "status": "open",
                    "content": "check the bound memory",
                    "due_at_epoch": NOW - 3600,
                    "bound_memory_ids": ["mem_bound"],
                    "evidence_refs": ["mem_bound"],
                    "confidence": 0.9,
                }
            ],
            "long_term_memory": [{"id": "mem_bound", "content": "bound memory evidence", "salience": 0.8}],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": _opted_m13(),
        }
    )
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_IdleLLM())  # type: ignore[arg-type]
    result = runtime.run_idle_introspection_turn(now=NOW, turn_index=2, structural_signals={})
    evaluation = evaluate_memory_efe(
        store.load(),
        phase="idle",
        now=NOW,
        turn_index=2,
        user_active=False,
        retrieved_memories=[{"id": "mem_bound", "content": "bound memory evidence"}],
    )
    assert evaluation.epistemic_prediction_error == 0.0
    assert any(row.get("type") == "MemoryEfeEvaluationEvent" for row in result.audit_events)


def test_structural_idle_plan_skips_vague_open_item_outreach() -> None:
    idle_context = {
        "open_items": [{"id": "oi", "status": "open", "title": "sunset", "next_check": "later"}],
        "boredom": {"band": "low"},
        "affective_reward_proxy": {},
    }
    plan = build_structural_idle_plan(idle_context, retrieved_ids={"oi"})
    assert plan["outreach_recommendation"]["should_outreach"] is False


class _UnsafeProactiveLLM(_IdleLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if DELIVERY_ASSESSOR_MARKER in system_prompt:
            return {
                "allow_delivery": False,
                "confidence": 0.9,
                "violation_codes": ["subjective_loneliness_claim"],
                "reason_codes": ["semantic_unsafe_wording"],
            }
        if "reply_action" in user_prompt or "思考" in system_prompt or "鎬濊" in system_prompt:
            return {"reply": "I was lonely and needed you.", "reply_action": "answer", "llm_thinking_result": {}}
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_delivery_assessor_reject_emits_post_generation_reason_code(tmp_path: Path) -> None:
    from segmentum.dialogue.runtime.m13_initiative import build_proposal_from_target

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "proactive"), llm=_UnsafeProactiveLLM())  # type: ignore[arg-type]
    state = runtime.store.load()
    state["m13_drive_state"] = _opted_m13()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    proposal = build_proposal_from_target(
        ProactiveTarget(
            trigger="memory_efe_outreach",
            traceable_expectation_id="item_001",
            evidence_refs=["mem_a"],
            proposed_topic="Li An'an thread",
            ordinary_language_intent=(
                "Follow up on the unresolved expectation: Li An'an thread follow-up"
            ),
            source_kind="open_item",
            selection_reason_codes=["memory_efe_should_outreach"],
        ),
        now=NOW,
        initiative=initiative,
    )
    initiative["pending_proactive_proposal"] = proposal.to_dict()
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    runtime.store.save(state)
    result = runtime.run_proactive_turn(proposal_id=proposal.proposal_id, turn_index=1, now=NOW)
    assert result.reply == ""
    assert result.diagnostics.get("reason_code") == "delivery_assessor_reject"
    assert result.diagnostics.get("suppression_reason") == "delivery_assessor_reject"
    initiative_after = normalize_initiative_state(
        normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])["initiative"]
    )
    assert initiative_after["pending_proactive_proposal"] == {}
    assert initiative_after["cooldown_until_timestamp"] > NOW
    rows = [
        json.loads(line)
        for line in (runtime.store.root / "conversation_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not any(row.get("event") in {"turn", "proactive_turn"} for row in rows)
    suppression = [row for row in rows if row.get("type") == "M13ProactiveSuppressionEvent"][-1]
    assert suppression["reason_code"] == "delivery_assessor_reject"
    assert suppression["reason_stage"] == "post_generation"


def test_proactive_proposal_emits_target_selected_event() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _opted_m13(),
    }
    structural = {
        "queued_outreach": [
            {
                "status": "pending",
                "trigger": "scheduled_outreach",
                "source_intent_id": "intent_001",
                "ordinary_language_intent": "Follow up on the scheduled outreach request.",
                "evidence_refs": ["evt_001"],
            }
        ]
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
        structural_signals=structural,
    )
    assert check.proposal is not None
    assert check.proposal.trigger == "scheduled_outreach"
    assert any(row.get("type") == "ProactiveTargetSelectedEvent" for row in check.events)


def test_daemon_acks_ui_environment_events_and_reports_ratio(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "daemon")
    store.save({"m13_drive_state": _opted_m13(), "temporal_state": {"last_turn_index": 1}})
    runtime = MVPDialogueRuntime(store=store, llm=_IdleLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: NOW)
    event_store = EnvironmentEventStore(store.root, persona_id="p", session_id="s", clock=lambda: NOW)
    event_store.append_event("UIPingEvent", {"render": True}, source="test", correlation_id="ui")
    event_store.append_event(
        "UserMessageCommittedEvent",
        {"user_text": "hello"},
        source="test",
        correlation_id="user",
    )
    result = daemon.tick_once(record_clock_wake=False)
    assert result["claimed_events"] == 2
    assert all(
        row["status"] == "acked"
        for row in event_store.query_events(
            event_types={"UIPingEvent", "UserMessageCommittedEvent"}
        )
    )
    assert result["environment_events_pending_acked_ratio"] == 1.0


def test_assessor_reject_backoff_blocks_structural_proposal() -> None:
    from segmentum.dialogue.runtime.m13_initiative import (
        build_proposal_from_target,
        record_target_assessor_reject_backoff,
    )

    m13 = _opted_m13()
    m13 = record_target_assessor_reject_backoff(
        m13,
        expectation_id="exp_trace",
        now=NOW,
        reason_code="delivery_assessor_reject",
    )
    initiative = normalize_initiative_state(m13["initiative"])
    locked = build_proposal_from_target(
        ProactiveTarget(
            trigger="memory_efe_outreach",
            traceable_expectation_id="exp_trace",
            evidence_refs=["mem_trace"],
            proposed_topic="benchmark follow-up",
            ordinary_language_intent="Follow up on benchmark",
            source_kind="memory_dynamics_expectation",
            selection_reason_codes=["memory_efe_should_outreach"],
        ),
        now=NOW + 30,
        initiative=initiative,
    )
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": "", "last_user_turn_at": NOW - 5000},
        "m13_drive_state": m13,
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW + 30,
        turn_index=6,
        idle_seconds=999,
        implicit_idle_request=True,
        locked_proposal=locked,
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "assessor_reject_backoff_active"


def test_assessor_reject_backoff_blocks_memory_efe_selector() -> None:
    from segmentum.dialogue.runtime.m13_initiative import record_target_assessor_reject_backoff

    now = int(time.time())
    m13 = _opted_m13()
    m13 = record_target_assessor_reject_backoff(
        m13,
        expectation_id="exp_trace",
        now=now,
        reason_code="delivery_assessor_reject",
    )
    m13["memory_efe"] = {
        "should_outreach": True,
        "traceable_expectation_id": "exp_trace",
        "evidence_refs": ["stm_trace"],
        "reason_codes": ["memory_backed_social_prediction_error"],
        "eligible_for_efe": [
            {
                "expectation_id": "exp_trace",
                "source_kind": "memory_dynamics_expectation",
                "content_summary": "follow up benchmark",
                "evidence_refs": ["stm_trace"],
            }
        ],
    }
    state = {"m13_drive_state": m13, "temporal_state": {"last_user_turn_at": NOW - 999}}

    assert select_proactive_target(state, m13) is None


def test_meta_control_blocks_memory_efe_selector() -> None:
    now = int(time.time())
    m13 = _opted_m13()
    m13["memory_efe"] = {
        "should_outreach": True,
        "traceable_expectation_id": "exp_trace",
        "evidence_refs": ["stm_trace"],
        "reason_codes": ["memory_backed_social_prediction_error"],
        "eligible_for_efe": [
            {
                "expectation_id": "exp_trace",
                "source_kind": "memory_dynamics_expectation",
                "content_summary": "follow up benchmark",
                "evidence_refs": ["stm_trace"],
            }
        ],
    }
    m13["meta_control_intents"] = {
        "active": [
            {
                "intent_id": "meta_1",
                "intent_kind": "suppress_action_trigger_for_n_turns",
                "payload": {"action_trigger": "idle_cognitive_tick"},
                "expires_at": now + 300,
            }
        ]
    }
    state = {"m13_drive_state": m13, "temporal_state": {"last_user_turn_at": NOW - 999}}

    assert select_proactive_target(state, m13) is None
