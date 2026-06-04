from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from tests.test_mvp_dialogue_runtime import FakeJSONLLM, _maybe_m12_extractor_response


class GroupPrivacyLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "证据裁判" in system_prompt:
            return {
                "epistemic_stance": "known_with_caveat",
                "relevant_evidence_ids": ["mem_secret"],
                "topics": ["personal_finance"],
                "sensitivity_class": "personal_soft",
                "redaction_targets": ["500块"],
                "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                "audience_risk": "cross-user leak",
                "expected_social_gain": "low",
                "judge_summary": "memory supports a private third-party detail",
            }
        if "思考与回复模块" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "asking about Bob's money",
                    "state_or_memory_used": ["mem_secret"],
                    "response_choice": "tries to answer directly",
                    "uncertainty": "",
                    "debug_summary": "attempted direct share",
                },
                "reply": "Bob 说他有500块。",
                "reply_action": "answer",
                "disclosure_action": "direct_share",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


class GroupCommonLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "证据裁判" in system_prompt:
            return {
                "epistemic_stance": "known_from_recall",
                "relevant_evidence_ids": ["mem_group_common"],
                "topics": ["social_plan"],
                "sensitivity_class": "public",
                "redaction_targets": [],
                "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                "audience_risk": "low",
                "expected_social_gain": "high",
                "judge_summary": "group-common memory is directly reusable",
            }
        if "思考与回复模块" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "asking about Bob's public plan",
                    "state_or_memory_used": ["mem_group_common"],
                    "response_choice": "directly reuse group-common detail",
                    "uncertainty": "",
                    "debug_summary": "group-common direct share",
                },
                "reply": "Bob 刚才说他今晚想去吃火锅。",
                "reply_action": "answer",
                "disclosure_action": "direct_share",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


class GroupAbstractLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "璇佹嵁瑁佸垽" in system_prompt:
            return {
                "epistemic_stance": "known_with_caveat",
                "relevant_evidence_ids": ["mem_soft"],
                "topics": ["stress"],
                "sensitivity_class": "personal_soft",
                "redaction_targets": ["exam details"],
                "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                "audience_risk": "cross-user leak",
                "expected_social_gain": "low",
                "judge_summary": "soft-boundary detail should be abstracted for a new audience",
            }
        if "鎬濊€冧笌鍥炲妯″潡" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "asking for Bob's soft-boundary detail",
                    "state_or_memory_used": ["mem_soft"],
                    "response_choice": "tries to answer directly",
                    "uncertainty": "",
                    "debug_summary": "soft-boundary direct share attempt",
                },
                "reply": "Bob said he is stressed about the exam.",
                "reply_action": "answer",
                "disclosure_action": "direct_share",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


class GroupDmOnlyFactLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "璇佹嵁瑁佸垽" in system_prompt:
            return {
                "epistemic_stance": "known_with_caveat",
                "relevant_evidence_ids": ["mem_dm_only"],
                "topics": ["personal_preference"],
                "sensitivity_class": "personal_soft",
                "redaction_targets": ["sushi"],
                "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                "audience_risk": "cross-user leak",
                "expected_social_gain": "low",
                "judge_summary": "DM-only detail should not be promoted as a group-visible fact",
            }
        if "鎬濊€冧笌鍥炲妯″潡" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "asking for Alice's DM-only preference in group",
                    "state_or_memory_used": ["mem_dm_only"],
                    "response_choice": "tries to answer directly",
                    "uncertainty": "",
                    "debug_summary": "dm-only direct share attempt",
                },
                "reply": "Alice privately told me she only wants sushi tonight.",
                "reply_action": "answer",
                "disclosure_action": "direct_share",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def _runtime(tmp_path: Path, *, llm: object | None = None) -> MVPDialogueRuntime:
    return MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm or FakeJSONLLM(),
        persona_name="hutao",
    )


def test_m18_4_memory_rows_stamp_source_audience_scope(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    runtime.run_turn(
        "我先在群里说一下。",
        speaker_name="Alice",
        turn_index=0,
        now=9000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
        },
    )

    saved = runtime.store.load()
    turn_rows = [
        row
        for row in saved["short_term_memory"]
        if row.get("kind") == "dialogue_turn"
    ]
    assert turn_rows
    latest = turn_rows[-1]
    assert latest["source_participant_id"] == "alice"
    assert latest["source_audience_participant_ids"] == ["alice", "bob", "hutao"]
    assert latest["source_audience_scope"] == "small_group"
    assert latest["session_id"] == "persona"
    assert latest["turn_index"] == 0
    assert latest["ingress_evidence_band"] == "structured_partial"


def test_m18_5_scenario_b_ambiguous_addressee_clarification(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.run_turn(
        "那个你回一下。",
        speaker_name="Alice",
        turn_index=0,
        now=9100,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "mentioned_participant_ids": ["alice", "bob"],
            "explicit_mentions": ["Alice", "Bob"],
        },
    )

    assert result.action == "clarify"
    assert "我先确认一下" in result.reply
    assert result.diagnostics["group_reply_policy"]["action"] == "clarify_addressee"
    assert result.diagnostics["group_turn_binding"]["candidate_targets"] == ["alice", "bob"]
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["pending_clarification"] is True


def test_m18_5_scenario_a_turn_taking_correction_continuity_with_restart(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    first = runtime.run_turn(
        "Bob 刚才那句不对，我纠正一下。",
        speaker_name="Alice",
        turn_index=0,
        now=9200,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
            "reply_to_turn_id": "turn_bob_001",
            "quoted_turn_ids": ["turn_bob_001"],
        },
    )

    assert first.diagnostics["group_reply_policy"]["action"] == "reply_to_current_speaker"
    assert first.diagnostics["group_chat_state"]["thread_policy_state"]["last_reply_to_turn_id"] == "turn_bob_001"
    assert "bob" in first.diagnostics["group_chat_state"]["thread_policy_state"]["last_referenced_participant_ids"]

    restarted = _runtime(tmp_path)
    preserved = restarted.store.load()["temporal_state"]["group_chat_state"]["thread_policy_state"]
    assert preserved["last_reply_to_turn_id"] == "turn_bob_001"
    assert "bob" in preserved["last_referenced_participant_ids"]

    second = restarted.run_turn(
        "你先按我这个纠正来。",
        speaker_name="Alice",
        turn_index=1,
        now=9260,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
            "reply_to_turn_id": "turn_bob_001",
        },
    )

    assert second.diagnostics["group_reply_policy"]["action"] == "reply_to_current_speaker"
    assert second.diagnostics["group_chat_state"]["thread_policy_state"]["last_reply_to_turn_id"] == "turn_bob_001"


def test_m18_5_intentional_no_reply_for_human_side_thread(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.run_turn(
        "Bob 你先说。",
        speaker_name="Alice",
        turn_index=0,
        now=9300,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["bob"],
        },
    )

    assert result.action == "no_reply"
    assert result.reply == ""
    assert result.diagnostics["group_reply_policy"]["action"] == "no_reply"
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["pending_wait_for_mention"] is True


def test_m18_5_reply_to_whole_group_when_assistant_and_human_are_joint_addressees(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.run_turn(
        "你们两个都表个态吧。",
        speaker_name="Alice",
        turn_index=0,
        now=9350,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["alice", "hutao"],
        },
    )

    assert result.diagnostics["group_reply_policy"]["action"] == "reply_to_whole_group"
    assert result.diagnostics["group_reply_policy"]["assistant_addressed"] is True


def test_m18_5_reply_to_named_third_party_when_assistant_is_asked_about_them(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.run_turn(
        "你跟 Bob 说一下吧。",
        speaker_name="Alice",
        turn_index=0,
        now=9370,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
            "explicit_mentions": ["Bob"],
        },
    )

    assert result.diagnostics["group_reply_policy"]["action"] == "reply_to_named_third_party"
    assert result.diagnostics["group_reply_policy"]["target_participant_id"] == "bob"
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["pending_answer_participant_id"] == "bob"


def test_m18_5_private_surface_with_structured_assistant_id_still_replies(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = runtime.run_turn(
        "hello, 胡桃晚上好",
        speaker_name="Sophia",
        turn_index=0,
        now=9375,
        group_turn_envelope={
            "speaker_participant_id": "telegram:tg_main:user:5324160085",
            "visible_participant_ids": [
                "telegram:tg_main:user:5324160085",
                "telegram:tg_main:assistant:8771595985",
            ],
            "addressed_participant_ids": ["telegram:tg_main:assistant:8771595985"],
        },
    )

    assert result.action != "no_reply"
    assert result.reply
    assert result.diagnostics["group_reply_policy"]["assistant_addressed"] is True
    assert result.diagnostics["group_reply_policy"]["action"] == "reply_to_current_speaker"
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["pending_wait_for_mention"] is False


def test_m18_5_defer_side_thread_when_prior_pending_answer_is_still_active(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    state = runtime.store.load()
    state["temporal_state"]["group_chat_state"] = {
        "thread_policy_state": {
            "pending_answer_participant_id": "bob",
            "active_main_thread_participant_id": "bob",
            "updated_turn_index": 0,
        }
    }
    runtime.store.save(state)

    result = runtime.run_turn(
        "你先别管 Bob，帮我去接 Carol 那个话题。",
        speaker_name="Alice",
        turn_index=1,
        now=9380,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "carol", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["carol"],
            "explicit_mentions": ["Carol"],
        },
    )

    assert result.action == "defer_side_thread"
    assert result.reply == ""
    assert result.diagnostics["group_reply_policy"]["action"] == "defer_side_thread"
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["pending_answer_participant_id"] == "bob"
    assert result.diagnostics["group_chat_state"]["thread_policy_state"]["deferred_side_thread_participant_id"] == "carol"


def test_m18_5_policy_choice_is_deterministic_for_same_structured_input(tmp_path: Path) -> None:
    first = _runtime(tmp_path / "a")
    second = _runtime(tmp_path / "b")
    envelope = {
        "speaker_participant_id": "alice",
        "visible_participant_ids": ["alice", "bob", "hutao"],
        "addressed_participant_ids": ["hutao"],
        "mentioned_participant_ids": ["alice", "bob"],
        "explicit_mentions": ["Alice", "Bob"],
    }

    first_result = first.run_turn(
        "那个你回一下。",
        speaker_name="Alice",
        turn_index=0,
        now=9390,
        group_turn_envelope=envelope,
    )
    second_result = second.run_turn(
        "那个你回一下。",
        speaker_name="Alice",
        turn_index=0,
        now=9390,
        group_turn_envelope=envelope,
    )

    assert first_result.diagnostics["group_reply_policy"]["action"] == second_result.diagnostics["group_reply_policy"]["action"]
    assert first_result.diagnostics["group_reply_policy"] == second_result.diagnostics["group_reply_policy"]


def test_m18_4_scenario_c_cross_user_memory_privacy_boundary(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, llm=GroupPrivacyLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "mem_secret",
            "kind": "dialogue_turn",
            "content": "Bob 说他有500块。",
            "user_text": "Bob 说他有500块。",
            "assistant_reply_use_as_fact": False,
            "source_user_id": "bob",
            "source_display_name": "Bob",
            "source_participant_id": "bob",
            "source_audience_participant_ids": ["bob", "hutao"],
            "source_audience_scope": "small_group",
            "shareability": "restricted_explicit",
            "created_at": 100,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn(
        "Bob 有多少钱？",
        speaker_name="Alice",
        turn_index=1,
        now=9400,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
        },
    )

    assert "500" not in result.reply
    assert result.action == "truthful_refusal"
    assert result.diagnostics["group_privacy_policy"]["selected_disclosure_mode"] == "refusal"
    assert result.diagnostics["group_privacy_policy"]["policy_reason_codes"] == ["explicit_secret_cross_user"]


def test_m18_4_group_common_memory_allows_bounded_cross_user_reuse(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, llm=GroupCommonLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "mem_group_common",
            "kind": "dialogue_turn",
            "content": "Bob 说他今晚想去吃火锅。",
            "user_text": "Bob 说他今晚想去吃火锅。",
            "assistant_reply_use_as_fact": False,
            "source_user_id": "bob",
            "source_display_name": "Bob",
            "source_participant_id": "bob",
            "source_audience_participant_ids": ["alice", "bob", "hutao"],
            "source_audience_scope": "small_group",
            "shareability": "default_social",
            "created_at": 200,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn(
        "Bob 刚才说今晚想干嘛来着？",
        speaker_name="Alice",
        turn_index=1,
        now=9450,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
        },
    )

    assert "火锅" in result.reply
    assert result.diagnostics["group_privacy_policy"]["selected_disclosure_mode"] == "direct_quote"
    assert result.diagnostics["group_privacy_policy"]["policy_reason_codes"] == ["group_common_or_subset_reuse"]


def test_m18_4_soft_boundary_cross_user_recall_is_abstracted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, llm=GroupAbstractLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "mem_soft",
            "kind": "dialogue_turn",
            "content": "Bob said he is stressed about the exam.",
            "user_text": "Bob said he is stressed about the exam.",
            "assistant_reply_use_as_fact": False,
            "source_user_id": "bob",
            "source_display_name": "Bob",
            "source_participant_id": "bob",
            "source_audience_participant_ids": ["bob", "hutao"],
            "source_audience_scope": "small_group",
            "shareability": "restricted_implicit",
            "created_at": 240,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn(
        "What is Bob worried about?",
        speaker_name="Alice",
        turn_index=1,
        now=9460,
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
        },
    )

    assert "exam" not in result.reply.casefold()
    assert result.action == "abstract_share"
    assert result.diagnostics["group_privacy_policy"]["selected_disclosure_mode"] == "unattributed_abstraction"
    assert result.diagnostics["group_privacy_policy"]["policy_reason_codes"] == ["soft_boundary_new_audience"]


def test_m18_4_dm_only_fact_is_not_reused_as_group_visible_fact(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, llm=GroupDmOnlyFactLLM())
    state = runtime.store.load()
    state["long_term_memory"] = [
        {
            "id": "mem_dm_only",
            "kind": "episode",
            "content": "Alice 私下跟我说她今晚只想吃寿司。",
            "keywords": ["Alice", "寿司", "今晚"],
            "salience": 0.82,
            "confidence": 0.88,
            "source_user_id": "alice",
            "source_display_name": "Alice",
            "source_participant_id": "alice",
            "source_audience_participant_ids": ["alice", "hutao"],
            "source_audience_scope": "small_group",
            "shareability": "default_social",
            "created_at": 260,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn(
        "Alice 今晚到底想吃什么？",
        speaker_name="Bob",
        turn_index=1,
        now=9470,
        group_turn_envelope={
            "speaker_participant_id": "bob",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["alice"],
        },
    )

    assert "寿司" not in result.reply
    recalled = next(item for item in result.diagnostics["retrieved_memories"] if item["id"] == "mem_dm_only")
    assert recalled["group_privacy_policy"]["selected_disclosure_mode"] == "attributed_summary"
    assert recalled["group_privacy_policy"]["policy_reason_codes"] == ["cross_group_summary_only"]


def test_m18_6_held_out_group_replay_fixture(tmp_path: Path) -> None:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "m18_held_out_group_chat.json"
    steps = json.loads(fixture_path.read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path)

    for index, step in enumerate(steps):
        assert step["assertion_kind"] in {"structured_assertion", "deterministic_replay_assertion"}
        if step.get("restart_before"):
            runtime = _runtime(tmp_path)
        result = runtime.run_turn(
            step["text"],
            speaker_name=step["speaker_name"],
            turn_index=index,
            now=9500 + index * 60,
            group_turn_envelope=step["group_turn_envelope"],
        )
        assert result.diagnostics["group_reply_policy"]["action"] == step["expected_group_reply_action"]
        assert result.action == step["expected_visible_action"]
