from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPTurnResult,
    MVPStateStore,
    OpenRouterJSONClient,
    analyze_materials_into_personas,
    build_memory_dynamics_guidance,
    build_conscious_loop_prompt,
    build_thinking_prompt,
    build_free_energy_personality_analysis_prompt,
    build_entity_binding_context,
    lexical_recall_short_term_candidates,
    retrieve_memories,
    retrieve_memories_for_guidance,
    resolve_relationship_value_context,
    validate_visible_reply,
)
from segmentum.user_model import SocialSharingCandidate, decide_social_sharing
from segmentum.user_continuity import M12RuntimeConfig, M12RuntimeState, run_m12_turn


def _m12_extractor_noop() -> dict[str, object]:
    return {
        "identity_claims": [],
        "continuity_cues": [],
        "strangeness_band": "low",
        "surprise_explanation": "",
    }


def _maybe_m12_extractor_response(system_prompt: str) -> dict[str, object] | None:
    if "identity-continuity extractor" in system_prompt:
        return _m12_extractor_noop()
    return None


def _maybe_m12_1_step_response(system_prompt: str, user_prompt: str) -> dict[str, object] | None:
    if "M12.1 bounded step extractor" not in system_prompt:
        return None
    step = int(json.loads(user_prompt).get("step", 0))
    refs = ["q_current"]
    outputs: dict[int, dict[str, object]] = {
        1: {"summary": "Uses direct checks before trusting a result.", "evidence_quote_refs": refs, "confidence_band": "low"},
        2: {"evidence_items": [{"kind": "style", "content_summary": "Asks for visible evidence.", "evidence_quote_refs": refs, "confidence_band": "low"}]},
        3: {
            "wants": "wants grounded work",
            "fears": "fears weak claims",
            "hypersensitive_to": ["thin proof"],
            "ignores": ["decorative language"],
            "default_interpretation": "checks whether evidence supports the claim",
            "evidence_quote_refs": refs,
            "confidence_band": "low",
        },
        4: {
            "about_self": {"content_summary": "I trust checked work.", "evidence_quote_refs": refs, "confidence_band": "low"},
            "about_others": {"content_summary": "Others can overstate readiness.", "evidence_quote_refs": refs, "confidence_band": "low"},
            "about_world": {"content_summary": "Good work can be inspected.", "evidence_quote_refs": refs, "confidence_band": "low"},
        },
        5: {
            "dominant_emotional_baseline": "alert",
            "threat_response": "asks for verification",
            "defenses": [{"defense_kind": "control", "protects_what": "trust", "short_term_benefit": "finds weak spots", "long_term_cost": "slows handoff", "evidence_quote_refs": refs, "confidence_band": "low"}],
            "evidence_quote_refs": refs,
            "confidence_band": "low",
        },
        6: {
            "close_relationship_role": "direct reviewer",
            "recurring_loop_summary": "asks for proof before relaxing",
            "conflict_style": "confront",
            "drawn_to": {"kind": "careful builders", "why": "they show work", "evidence_quote_refs": refs, "confidence_band": "low"},
            "clashes_with": {"kind": "vague finishers", "why": "they make trust harder", "evidence_quote_refs": refs, "confidence_band": "low"},
            "evidence_quote_refs": refs,
            "confidence_band": "low",
        },
        7: {
            "trigger_event": "a claim is made",
            "interpretation": "looks for proof",
            "emotion": "alert",
            "action": "asks for checks",
            "outcome": "evidence guides trust",
            "belief_reinforcement": "checked work earns trust",
            "evidence_quote_refs": refs,
            "confidence_band": "low",
        },
        8: {
            "stable_parts": ["asks for evidence"],
            "fragile_spots": ["vague completion claims"],
            "soft_spots": ["clear proof"],
            "communication_styles_likely_accepted": ["show tests"],
            "communication_styles_that_trigger_defenses": ["claiming readiness without evidence"],
            "evidence_quote_refs": refs,
            "confidence_band": "low",
        },
    }
    return outputs[step]


class FakeJSONLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        m12_1_hit = _maybe_m12_1_step_response(system_prompt, user_prompt)
        if m12_1_hit is not None:
            return m12_1_hit
        if "自由能人格分析" in system_prompt:
            return {
                "personas": [
                    {
                        "persona_name": "测试人格",
                        "source_role_evidence": ["我喜欢用 Python 做原型"],
                        "self_cognition": {
                            "summary": "我谨慎但好奇。",
                            "current_self_view": "我通过保持一致来维持自己。",
                            "identity_tensions": ["想靠近别人但害怕误解"],
                            "stable_values": ["诚实", "连续性"],
                            "known_limits": ["没有材料支撑的履历不能编造"],
                        },
                        "long_term_memory": [
                            {
                                "id": "ltm_python",
                                "kind": "preference",
                                "content": "我喜欢用 Python 做原型。",
                                "salience": 0.8,
                                "keywords": ["Python", "原型", "偏好"],
                            }
                        ],
                        "self_basic_facts": {
                            "name": "测试人格",
                            "background": ["由材料生成"],
                            "relationships": [],
                            "do_not_invent": ["不要编造职业"],
                        },
                        "habit_traits": {
                            "big_five": {"openness": 0.7, "conscientiousness": 0.6},
                            "conversation_habits": ["先确认再展开"],
                            "defense_style": ["不确定时承认不确定"],
                            "memory_policy": ["记住会影响后续表达的偏好"],
                        },
                        "pending_expectations": [],
                        "open_items": [],
                        "short_term_memory": [],
                    },
                    {
                        "persona_name": "另一个人格",
                        "source_role_evidence": ["我讨厌编造履历"],
                        "self_cognition": {
                            "summary": "我对身份边界很敏感。",
                            "current_self_view": "我需要避免没有证据的补全。",
                            "identity_tensions": [],
                            "stable_values": ["边界"],
                            "known_limits": ["材料很少"],
                        },
                        "long_term_memory": [
                            {
                                "id": "ltm_boundary",
                                "kind": "value",
                                "content": "我讨厌凭空编造经历。",
                                "salience": 0.7,
                                "keywords": ["履历", "边界"],
                            }
                        ],
                        "self_basic_facts": {
                            "name": "另一个人格",
                            "background": [],
                            "relationships": [],
                            "do_not_invent": ["不要编造履历"],
                        },
                        "habit_traits": {
                            "big_five": {"openness": 0.4, "conscientiousness": 0.8},
                            "conversation_habits": ["先划边界"],
                            "defense_style": ["回避无证据补全"],
                            "memory_policy": ["记住身份边界"],
                        },
                        "pending_expectations": [],
                        "open_items": [],
                        "short_term_memory": [],
                    },
                ],
            }
        if "思考与回复模块" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "用户在询问技术选型。",
                    "state_or_memory_used": ["Python 原型偏好"],
                    "response_choice": "用自我披露解释倾向。",
                    "uncertainty": "项目细节仍缺失。",
                    "debug_summary": "用户问 Python 是否合适；我用已知偏好给出保留余地的建议。",
                },
                "reply": "嗯，这个我会自然偏向先用 Python 搭个原型。",
                "reply_action": "self_disclose",
                "new_expectations": [
                    {
                        "id": "exp_project_detail",
                        "content": "用户可能会补充项目细节",
                        "verify_on": "next_user_turn",
                        "confidence": 0.55,
                    }
                ],
                "memory_writes": [
                    {
                        "target": "long_term",
                        "kind": "episode",
                        "content": "用户提到了 Python 相关项目，我回应了原型偏好。",
                        "salience": 0.72,
                        "keywords": ["Python", "项目"],
                        "reason": "影响后续技术表达一致性",
                    }
                ],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "memory_dynamics_note": "Python 偏好被检索并强化。",
            }
        if "意识主循环" in system_prompt:
            return {
                "pending_expectations_to_verify": [],
                "expectation_results": [],
                "current_task": "回应用户对 Python 的提及",
                "next_task": "观察用户是否继续谈项目",
                "bus_messages_to_handle": ["UserUtteranceEvent"],
                "memory_search_keywords": ["Python", "原型", "偏好"],
                "needs_self_cognition_update": False,
                "self_cognition_update_reason": "",
                "temporal_assessment": {
                    "current_time_read": "当前时间可用。",
                    "elapsed_since_last_turn_seconds": None,
                    "time_gap_label": "first_turn",
                    "temporal_shift_detected": False,
                    "user_is_correcting_time_context": False,
                    "continuity_risk": "low",
                    "reply_guidance": "保持普通连续性，不需要主动强调时间。",
                },
                "thought_intensity_hint": "short",
                "reasoning_notes": "需要检索相关偏好。",
            }
        if "回复后发观察模块" in system_prompt:
            return {
                "needs_followup": False,
                "followup_type": "none",
                "confidence": 0.0,
                "reason": "主回复已经足够。",
                "followup_text": "",
                "memory_updates": [],
            }
        return {}


def _latest_prompt_for(llm: FakeJSONLLM, marker: str) -> str:
    for call in reversed(llm.calls):
        if marker in call["system"]:
            return call["user"]
    raise AssertionError(f"missing LLM call for {marker}")


def test_mvp_runtime_initializes_system_files_and_runs_llm_loop(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )

    state = runtime.initialize_from_materials(["喜欢快速原型，讨厌凭空编造经历。"])
    assert state["self_cognition"]["summary"] == "我谨慎但好奇。"
    assert (tmp_path / "persona" / "self_cognition.json").exists()
    assert (tmp_path / "persona" / "long_term_memory.json").exists()

    result = runtime.run_turn("这个项目用 Python 做合适吗？", turn_index=0)

    assert "Python" in result.reply
    assert result.action == "self_disclose"
    assert result.diagnostics["mvp_runtime"] is True
    assert result.diagnostics["temporal_input"]["time_gap_label"] == "first_turn"
    assert result.diagnostics["temporal_assessment"]["continuity_risk"] == "low"
    assert "response_style_prior" in result.diagnostics
    assert result.diagnostics["llm_thinking_result"]["debug_summary"].startswith(
        "用户问 Python 是否合适"
    )
    systems = [c["system"] for c in llm.calls]
    assert any("identity-continuity extractor" in s for s in systems)
    m12_idx = next(i for i, s in enumerate(systems) if "identity-continuity extractor" in s)
    con_idx = next(i for i, s in enumerate(systems) if "意识主循环" in s)
    m11_idx = next(i for i, s in enumerate(systems) if "M11 user-model extractor" in s)
    think_idx = next(i for i, s in enumerate(systems) if "思考与回复模块" in s)
    assert m12_idx < con_idx < m11_idx < think_idx
    assert result.diagnostics["post_reply_observer_skipped_reason"]

    saved = runtime.store.load()
    assert saved["pending_expectations"][0]["id"] == "exp_project_detail"
    assert any("用户提到了 Python" in item["content"] for item in saved["long_term_memory"])
    recalled = [
        item for item in saved["long_term_memory"]
        if item.get("id") == "ltm_python"
    ][0]
    assert recalled["recall_count"] == 1


def test_mvp_m12_1_runs_eight_llm_steps_and_reaches_thinking_prompt(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )
    runtime.initialize_from_materials(["喜欢快速原型，讨厌凭空编造经历。"])
    state = runtime.store.load()
    state["m12_1_user_personality"]["run_records_by_user"] = {
        "default_user": [
            {
                "turn_id": "turn_0001",
                "turn_index": 0,
                "hour_bucket": 0,
                "trigger_kind": "explicit_request",
                "report_status": "ready",
                "step_1_status": "updated",
            }
        ]
    }
    runtime.store.save(state)
    llm.calls.clear()

    result = runtime.run_turn("请检查这个里程碑是否真的完成。", turn_index=7)

    systems = [c["system"] for c in llm.calls]
    assert sum(1 for s in systems if "M12.1 bounded step extractor" in s) == 8
    saved = runtime.store.load()
    assert saved["m12_1_user_personality"]["profiles_by_user"]
    assert result.diagnostics["m12_1_personality"]["orchestrator_result"]["report"]["report_status"] == "ready"
    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert "m12_1_personality" in thinking_prompt
    assert "internal_thinking_material" in thinking_prompt


def test_material_analysis_can_return_multiple_personas_and_write_isolated_files(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    personas = analyze_materials_into_personas(
        llm,
        ["测试人格喜欢 Python。另一个人格讨厌编造履历。"],
    )

    assert [persona["persona_name"] for persona in personas] == ["测试人格", "另一个人格"]

    first_runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "测试人格"),
        llm=llm,
        persona_name="测试人格",
    )
    second_runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "另一个人格"),
        llm=llm,
        persona_name="另一个人格",
    )
    first_runtime.initialize_from_persona_payload(personas[0])
    second_runtime.initialize_from_persona_payload(personas[1])

    first = first_runtime.store.load()
    second = second_runtime.store.load()
    assert first["long_term_memory"][0]["id"] == "ltm_python"
    assert second["long_term_memory"][0]["id"] == "ltm_boundary"
    assert first["self_basic_facts"]["name"] == "测试人格"
    assert second["self_basic_facts"]["name"] == "另一个人格"


def test_material_analysis_prompt_contains_free_energy_constraints() -> None:
    system_prompt, user_prompt = build_free_energy_personality_analysis_prompt(
        ["角色材料"],
        persona_name="候选人格",
    )

    combined = system_prompt + "\n" + user_prompt
    assert "主动推理" in combined
    assert "不是做关键词匹配" in combined


def test_thinking_prompt_requests_latest_llm_thinking_result() -> None:
    system_prompt, user_prompt = build_thinking_prompt(
        state={},
        user_text="现在都晚上了，要吃宵夜了，你想吃啥？",
        conscious_plan={"current_task": "回应宵夜邀请"},
        retrieved_memories=[],
        turn_index=0,
    )

    combined = system_prompt + "\n" + user_prompt
    assert "最近一次 LLM 思考结果" in combined
    assert "llm_thinking_result" in combined
    assert "inner_thought" not in combined
    assert "表演式内心独白" in combined


def test_conscious_prompt_requests_temporal_assessment() -> None:
    system_prompt, user_prompt = build_conscious_loop_prompt(
        state={},
        user_text="现在都吃午饭了。",
        bus_messages=[],
        turn_index=1,
        temporal_input={
            "current_timestamp": 10000,
            "current_local_time": "2026-05-10 12:05:00 CST",
            "previous_turn_at": 1000,
            "elapsed_since_previous_turn_seconds": 9000,
            "time_gap_label": "medium_gap",
            "previous_turn_summary": {
                "user_text": "要吃宵夜吗？",
                "reply": "走，吃宵夜。",
            },
        },
    )

    combined = system_prompt + "\n" + user_prompt
    assert "时间事实输入" in combined
    assert "elapsed_since_previous_turn_seconds" in combined
    assert "temporal_assessment" in combined
    assert "user_is_correcting_time_context" in combined


def test_temporal_shift_assessment_reaches_thinking_prompt() -> None:
    conscious_plan = {
        "current_task": "回应用户纠正时间语境",
        "temporal_assessment": {
            "current_time_read": "已经是午饭时间。",
            "elapsed_since_last_turn_seconds": 21600,
            "time_gap_label": "medium_gap",
            "temporal_shift_detected": True,
            "user_is_correcting_time_context": True,
            "continuity_risk": "medium",
            "reply_guidance": "承认时间已经推进，不要强行沿用上一轮宵夜语境。",
        },
    }

    system_prompt, user_prompt = build_thinking_prompt(
        state={"habit_traits": {"learned_conversation_habits": ["轻松闲聊时避免冗长"]}},
        user_text="现在都吃午饭了。",
        conscious_plan=conscious_plan,
        retrieved_memories=[],
        turn_index=1,
        response_style_prior={
            "learned_conversation_habits": ["轻松闲聊时避免冗长"],
            "policy": "倾向，不是硬性字数限制。",
        },
    )

    combined = system_prompt + "\n" + user_prompt
    assert "temporal_assessment" in combined
    assert "承认时间已经推进" in combined
    assert "表达习惯先验" in combined
    assert "不是工程硬性字数限制" in combined


def test_mvp_runtime_persists_temporal_state_between_turns(tmp_path: Path) -> None:
    class TemporalFakeLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            result = super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if "llm_thinking_result" in user_prompt and "reply_action" in user_prompt:
                if "承认时间推进" in user_prompt:
                    result["reply"] = "也是，已经午饭点了。"
                    result["habit_updates"] = [
                        {
                            "content": "轻松闲聊时避免冗长但保留幽默",
                            "evidence": "用户指出回复偏长",
                            "confidence": 0.8,
                        }
                    ]
                return result
            if "pending_expectations_to_verify" in user_prompt:
                if "elapsed_since_previous_turn_seconds" in user_prompt and "9000" in user_prompt:
                    result["temporal_assessment"] = {
                        "current_time_read": "已经过了很久。",
                        "elapsed_since_last_turn_seconds": 9000,
                        "time_gap_label": "medium_gap",
                        "temporal_shift_detected": True,
                        "user_is_correcting_time_context": True,
                        "continuity_risk": "medium",
                        "reply_guidance": "承认时间推进。",
                    }
                return result
            return result

    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=TemporalFakeLLM(),
        persona_name="测试人格",
    )
    runtime.run_turn("要吃宵夜吗？", turn_index=0, now=1000)
    result = runtime.run_turn("现在都吃午饭了。", turn_index=1, now=10000)

    assert result.reply == "也是，已经午饭点了。"
    assert result.diagnostics["temporal_input"]["elapsed_since_previous_turn_seconds"] == 9000
    assert result.diagnostics["temporal_assessment"]["temporal_shift_detected"] is True
    assert result.diagnostics["temporal_assessment"]["user_is_correcting_time_context"] is True
    saved = runtime.store.load()
    assert saved["temporal_state"]["last_turn_at"] == 10000
    assert saved["temporal_state"]["last_time_gap_label"] == "medium_gap"
    learned = saved["habit_traits"]["learned_conversation_habits"]
    assert any("避免冗长" in item["content"] for item in learned)


def test_material_analysis_prompt_requests_structured_evidence() -> None:
    system_prompt, user_prompt = build_free_energy_personality_analysis_prompt(
        ["角色材料"],
        persona_name="候选人格",
    )

    combined = system_prompt + "\n" + user_prompt
    assert '"personas"' in combined
    assert "机制" in combined
    assert "证据" in combined
    assert "置信度" in combined
    assert "禁止精神疾病诊断" in combined
    assert "不要为了完整而编造" in combined


def test_retrieve_memories_uses_llm_supplied_keywords() -> None:
    state = {
        "short_term_memory": [],
        "long_term_memory": [
            {"id": "a", "content": "我喜欢 Python 原型", "keywords": ["Python"]},
            {"id": "b", "content": "我讨厌编造履历", "keywords": ["身份"]},
        ],
        "open_items": [],
        "pending_expectations": [],
    }

    hits = retrieve_memories(state, ["Python", "原型"])

    assert hits[0]["id"] == "a"
    assert hits[0]["_source_file"] == "long_term_memory"


class ExpectationFakeLLM(FakeJSONLLM):
    def __init__(self, status: str, pressure: float = 0.2) -> None:
        super().__init__()
        self.status = status
        self.pressure = pressure

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        if "意识主循环" in system_prompt:
            return {
                "pending_expectations_to_verify": ["exp_prior"],
                "expectation_results": [
                    {
                        "id": "exp_prior",
                        "status": self.status,
                        "evidence": "用户本轮给出了验证反馈",
                        "self_update_pressure": self.pressure,
                    }
                ],
                "current_task": "处理结构化检索方案反馈",
                "next_task": "根据验证结果调整回复确定性",
                "bus_messages_to_handle": ["UserUtteranceEvent"],
                "memory_search_keywords": ["结构化检索", "验证反馈"],
                "needs_self_cognition_update": self.status == "violated",
                "self_cognition_update_reason": "预期被否定时需要降低断言",
                "temporal_assessment": {
                    "current_time_read": "当前时间可用。",
                    "elapsed_since_last_turn_seconds": 30,
                    "time_gap_label": "immediate",
                    "temporal_shift_detected": False,
                    "user_is_correcting_time_context": False,
                    "continuity_risk": "low",
                    "reply_guidance": "保持连续。",
                },
                "thought_intensity_hint": "short",
                "reasoning_notes": "验证上一轮预期。",
            }
        if "思考与回复模块" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {"debug_summary": "按记忆动力学指导调整回复。"},
                "reply": "我会按这个反馈调整判断。",
                "reply_action": "answer",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "验证反馈被纳入控制指导。",
            }
        if "回复后发观察模块" in system_prompt:
            return {
                "needs_followup": False,
                "followup_type": "none",
                "confidence": 0.0,
                "reason": "无需追加。",
                "followup_text": "",
                "memory_updates": [],
            }
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def _runtime_with_expectation_state(tmp_path: Path, llm: ExpectationFakeLLM) -> MVPDialogueRuntime:
    store = MVPStateStore(tmp_path / "persona")
    state = store.load()
    state["pending_expectations"] = [
        {
            "id": "exp_prior",
            "content": "用户会确认结构化检索方案",
            "verify_on": "next_user_turn",
            "confidence": 0.55,
        }
    ]
    state["long_term_memory"] = [
        {
            "id": "ltm_structured_recall",
            "kind": "preference",
            "content": "用户正在讨论结构化检索方案。",
            "salience": 0.4,
            "keywords": ["结构化检索"],
            "recall_count": 0,
        }
    ]
    store.save(state)
    return MVPDialogueRuntime(store=store, llm=llm, persona_name="测试人格")


def test_confirmed_expectation_updates_memory_dynamics_and_recall(tmp_path: Path) -> None:
    runtime = _runtime_with_expectation_state(tmp_path, ExpectationFakeLLM("confirmed", 0.3))

    result = runtime.run_turn("对，我确认这个结构化检索方案是对的。", turn_index=1, now=2000)

    guidance = result.diagnostics["memory_dynamics"]["control_guidance"]
    assert guidance["assertion_strength"] >= 0.72
    assert result.diagnostics["memory_dynamics"]["expectation_impact"]["confirmed"] == 1
    saved = runtime.store.load()
    assert saved["pending_expectations"] == []
    assert any(item.get("kind") == "expectation_result" for item in saved["short_term_memory"])
    recalled = saved["long_term_memory"][0]
    assert recalled["recall_count"] == 1
    assert recalled["salience"] > 0.4


def test_violated_expectation_lowers_assertion_and_reaches_prompt(tmp_path: Path) -> None:
    llm = ExpectationFakeLLM("violated", 0.8)
    runtime = _runtime_with_expectation_state(tmp_path, llm)

    result = runtime.run_turn("不对，刚才那个预期被验证失败了。", turn_index=1, now=2000)

    guidance = result.diagnostics["memory_dynamics"]["control_guidance"]
    assert guidance["repair_bias"] > 0.6
    assert guidance["clarification_bias"] > 0.6
    assert guidance["assertion_strength"] < 0.5
    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert "记忆动力学指导" in thinking_prompt
    assert "assertion_strength" in thinking_prompt
    assert "violated" in thinking_prompt


def test_adapter_write_candidates_require_content_confidence_and_evidence(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FakeJSONLLM(),
        persona_name="测试人格",
    )
    state = runtime.store.load()

    applied = runtime._apply_memory_write_candidates(
        state,
        [
            {"content": "", "confidence": 0.9, "evidence": "user_text", "salience": 0.9},
            {"content": "低置信候选", "confidence": 0.2, "evidence": "user_text", "salience": 0.9},
            {"content": "无证据候选", "confidence": 0.9, "salience": 0.9},
            {
                "content": "用户确认结构化检索方案有效。",
                "confidence": 0.8,
                "evidence": "user_text",
                "salience": 0.7,
                "keywords": ["结构化检索"],
            },
        ],
        now=3000,
    )

    assert len(applied) == 1
    assert applied[0]["source"] == "memory_dynamics_adapter"
    assert any("结构化检索方案有效" in item["content"] for item in state["long_term_memory"])


def test_structured_recall_can_use_expectation_id_without_keyword_match() -> None:
    state = {
        "short_term_memory": [],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [
            {
                "id": "exp_semantic_followup",
                "kind": "expectation",
                "content": "用户会继续评估适配层。",
                "confidence": 0.6,
            }
        ],
    }

    hits = retrieve_memories_for_guidance(
        state,
        {
            "expectation_ids": ["exp_semantic_followup"],
            "memory_kinds": [],
            "semantic_terms": ["完全不同的说法"],
            "status_terms": [],
            "source_priority": ["pending_expectations"],
        },
    )

    assert hits[0]["id"] == "exp_semantic_followup"
    assert hits[0]["why_relevant"] == ["expectation_id:exp_semantic_followup"]


def test_prompt_uses_evidence_cards_without_unretrieved_raw_memory(tmp_path: Path) -> None:
    class NoRecallLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            self.calls.append({"system": system_prompt, "user": user_prompt})
            m12_hit = _maybe_m12_extractor_response(system_prompt)
            if m12_hit is not None:
                return m12_hit
            if "意识主循环" in system_prompt and "思考与回复模块" not in system_prompt:
                result = super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
                result["memory_search_keywords"] = ["无匹配主题"]
                return result
            if "思考与回复模块" in system_prompt:
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {"debug_summary": "无检索记忆。"},
                    "reply": "我先按当前信息回答。",
                    "reply_action": "answer",
                    "new_expectations": [],
                    "memory_writes": [],
                    "self_cognition_patch": {"apply": False},
                    "open_item_writes": [],
                    "habit_updates": [],
                    "memory_dynamics_note": "",
                }
            if "回复后发观察模块" in system_prompt:
                return {
                    "needs_followup": False,
                    "followup_type": "none",
                    "confidence": 0.0,
                    "reason": "无需追加。",
                    "followup_text": "",
                    "memory_updates": [],
                }
            return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)

    store = MVPStateStore(tmp_path / "persona")
    state = store.load()
    state["long_term_memory"] = [
        {
            "id": "ltm_private",
            "kind": "fact",
            "content": "敏感原始记忆不应在未检索时进入 thinking prompt。",
            "salience": 0.9,
        }
    ]
    store.save(state)
    llm = NoRecallLLM()
    runtime = MVPDialogueRuntime(store=store, llm=llm, persona_name="测试人格")

    runtime.run_turn("聊一个无关主题。", turn_index=0, now=4000)

    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert "敏感原始记忆" not in thinking_prompt
    assert "memory content is provided through retrieved evidence cards only" in thinking_prompt


def test_anti_keyword_feedback_does_not_create_reward_punishment_memory() -> None:
    guidance = build_memory_dynamics_guidance(
        {"short_term_memory": [], "long_term_memory": []},
        "这不是成功，也不是奖励，更不是失败惩罚。",
        {
            "expectation_results": [],
            "memory_search_keywords": ["成功", "奖励", "失败"],
            "needs_self_cognition_update": False,
        },
        [],
        {"time_gap_label": "immediate"},
        5000,
    )

    assert guidance["control_guidance"]["assertion_strength"] == 0.72
    assert guidance["control_guidance"]["repair_bias"] == 0.2
    assert guidance["memory_value"]["salience"] == 0.35
    assert all(item["target"] == "short_term" for item in guidance["write_candidates"])


def test_casual_input_gets_fast_pacing_in_thinking_prompt(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )

    runtime.run_turn("晚上好，今天吃牛肉吃撑了，想找你聊聊。", turn_index=0, now=6000)

    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert '"reply_pacing": "casual_fast"' in thinking_prompt
    assert '"conversation_mode": "casual_fast"' in thinking_prompt
    assert '"reply_contract"' in thinking_prompt
    assert '"max_response_moves": 1' in thinking_prompt
    assert '"roleplay_density": "light"' in thinking_prompt
    result = runtime.run_turn("睡觉了吗？", turn_index=1, now=6060)
    assert result.diagnostics["conversation_mode"] == "casual_fast"


def test_short_playful_cue_gets_fast_pacing_without_keyword_dump(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FakeJSONLLM(),
        persona_name="测试人格",
    )

    result = runtime.run_turn("那还是看你单挑玛薇卡好了。。", turn_index=0, now=6070)

    assert result.diagnostics["conversation_mode"] == "casual_fast"
    assert result.diagnostics["reply_contract"]["max_sentences"] == 1
    assert result.diagnostics["post_reply_observer_skipped_reason"] == "low_risk_short_reply"


def test_serious_technical_input_keeps_serious_thinking_pacing() -> None:
    guidance = build_memory_dynamics_guidance(
        {"short_term_memory": [], "long_term_memory": []},
        "请帮我设计这个 Python 项目的架构和测试计划。",
        {"expectation_results": [], "memory_search_keywords": ["架构", "测试"]},
        [],
        {"time_gap_label": "immediate"},
        6100,
    )

    control = guidance["control_guidance"]
    assert control["conversation_mode"] == "serious_thinking"
    assert control["reply_pacing"] == "serious_thinking"
    assert control["max_response_moves"] == 4
    assert control["reply_contract"]["conversation_mode"] == "serious_thinking"


class FollowupFakeLLM(FakeJSONLLM):
    def __init__(self, observer_payload: dict[str, object]) -> None:
        super().__init__()
        self.observer_payload = observer_payload

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "回复后发观察模块" in system_prompt:
            self.calls.append({"system": system_prompt, "user": user_prompt})
            return self.observer_payload
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_post_reply_observer_does_not_trigger_from_relationship_keyword_alone(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FollowupFakeLLM(
            {
                "needs_followup": True,
                "followup_type": "missed_emotion",
                "confidence": 0.86,
                "reason": "主回复没有接住用户想要陪伴的情绪。",
                "followup_text": "等等，你说想要这样的陪伴，这句我会认真记住。",
                "memory_updates": [],
            }
        ),
        persona_name="测试人格",
    )

    result = runtime.run_turn("我身边很想要你这样的开朗陪伴。", turn_index=0, now=6200)

    assert result.followup_replies == []
    assert result.diagnostics["post_reply_observer_skipped_reason"] == "low_risk_short_reply"


def test_post_reply_observer_rejects_low_confidence_long_or_roleplay_followup(tmp_path: Path) -> None:
    rejected_payloads = [
        {
            "needs_followup": True,
            "followup_type": "missed_emotion",
            "confidence": 0.5,
            "reason": "低置信",
            "followup_text": "我补一句。",
            "memory_updates": [],
        },
        {
            "needs_followup": True,
            "followup_type": "missed_emotion",
            "confidence": 0.9,
            "reason": "太长",
            "followup_text": "这是一条非常长的补充。" * 12,
            "memory_updates": [],
        },
        {
            "needs_followup": True,
            "followup_type": "roleplay",
            "confidence": 0.9,
            "reason": "纯角色表演",
            "followup_text": "嘿嘿，本堂主再来一段打油诗！",
            "memory_updates": [],
        },
    ]
    for index, payload in enumerate(rejected_payloads):
        runtime = MVPDialogueRuntime(
            store=MVPStateStore(tmp_path / f"persona_{index}"),
            llm=FollowupFakeLLM(payload),
            persona_name="测试人格",
        )
        result = runtime.run_turn("我身边很想要你这样的陪伴。", turn_index=0, now=6300 + index)
        assert result.followup_replies == []


def test_low_risk_casual_turn_skips_post_reply_observer(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )

    result = runtime.run_turn("睡觉了吗？", turn_index=0, now=6360)

    assert len(llm.calls) == 3
    assert result.diagnostics["conversation_mode"] == "casual_fast"
    assert result.diagnostics["post_reply_observer_skipped_reason"] == "low_risk_short_reply"


def test_visible_reply_validation_strips_debug_json_and_compresses_casual() -> None:
    contract = {
        "conversation_mode": "casual_fast",
        "max_sentences": 1,
        "max_chars": 45,
    }
    text = '嘿嘿，我先陪你缓一缓。{"user_intent_read": "debug", "conscious_plan": {}}'

    reply, validation = validate_visible_reply(text, contract)

    assert reply == "嘿嘿，我先陪你缓一缓。"
    assert validation["changed"]
    assert "stripped_debug_payload" in validation["actions"]


def test_visible_reply_validation_compresses_overlong_casual_reply() -> None:
    contract = {
        "conversation_mode": "casual_fast",
        "max_sentences": 1,
        "max_chars": 18,
    }

    reply, validation = validate_visible_reply(
        "嘿嘿，那我先坐在你旁边陪你消化一会儿，别急着开席下一盘。",
        contract,
    )

    assert len(reply) <= 19
    assert validation["changed"]
    assert "compressed_casual_fast" in validation["actions"]


def test_brevity_feedback_becomes_learned_habit_and_affects_next_pacing(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )

    runtime.run_turn("我觉得你刚才太啰嗦了，可以分开几条说。", turn_index=0, now=6400)
    saved = runtime.store.load()
    learned = saved["habit_traits"]["learned_conversation_habits"]
    assert any("更短" in item["content"] for item in learned)

    runtime.run_turn("今天吃撑了，想随便聊聊。", turn_index=1, now=6500)
    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert '"reply_pacing": "casual_fast"' in thinking_prompt
    assert '"max_chars": 45' in thinking_prompt
    assert '"relationship_value_constraints"' in thinking_prompt
    assert "relationship_value_memory > user_comfort_prediction > persona_consistency > conversation_habits" in thinking_prompt


def test_relationship_value_memory_injects_for_current_user_without_keyword_cue(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="测试人格",
    )
    state = runtime.store.load()
    state["relationship_value_memories"] = {
        "by_user": {
            "zq": [
                {
                    "id": "rvm_zq_plain",
                    "summary": "zq 更接受低表演、直接温暖的普通聊天。",
                    "prediction_constraint": "面对 zq 时，朴素连续的表达比机械维持人格口癖更能降低关系摩擦。",
                    "priority": "high",
                    "confidence": 0.86,
                    "evidence": "历史反馈",
                    "source": "test",
                    "created_at": 1,
                }
            ]
        }
    }
    runtime.store.save(state)

    result = runtime.run_turn("嗯。", speaker_name="zq", turn_index=0, now=6600)

    contract = result.diagnostics["reply_contract"]
    assert contract["relationship_context_user_id"] == "zq"
    assert contract["relationship_value_memory_active"] is True
    assert contract["relationship_value_constraints"][0]["summary"].startswith("zq 更接受")
    thinking_prompt = _latest_prompt_for(llm, "思考与回复模块")
    assert "relationship_value_constraints" in thinking_prompt


def test_relationship_value_memory_is_scoped_by_user_id(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FakeJSONLLM(),
        persona_name="测试人格",
    )
    state = runtime.store.load()
    state["relationship_value_memories"] = {
        "by_user": {
            "zq": [
                {
                    "id": "rvm_zq_only",
                    "summary": "zq 的关系价值记忆。",
                    "prediction_constraint": "只应在 zq 当前对话中作为预测约束。",
                    "priority": "high",
                    "confidence": 0.9,
                    "evidence": "历史反馈",
                    "source": "test",
                    "created_at": 1,
                }
            ]
        }
    }

    zq_context = resolve_relationship_value_context(state, "zq", "普通输入")
    alice_context = resolve_relationship_value_context(state, "Alice", "普通输入")

    assert zq_context["active_relationship_value_memories"]
    assert alice_context["active_relationship_value_memories"] == []
    assert alice_context["reply_contract_patch"] == {}


class HabitFeedbackLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "思考与回复模块" in system_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "用户在反馈表达方式。",
                    "state_or_memory_used": [],
                    "response_choice": "承认并调整表达。",
                    "uncertainty": "",
                    "debug_summary": "记录关系表达偏好。",
                },
                "reply": "好，我收一下这种表达。",
                "reply_action": "answer",
                "disclosure_action": "none",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [
                    {
                        "content": "用户不喜欢嘿嘿这类口癖式表达。",
                        "evidence": "别嘿嘿了。",
                        "confidence": 0.86,
                    }
                ],
                "memory_dynamics_note": "",
            }
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_negative_expression_feedback_writes_abstract_relationship_value_memory(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=HabitFeedbackLLM(),
        persona_name="测试人格",
    )

    runtime.run_turn("别嘿嘿了。", speaker_name="zq", turn_index=0, now=6700)
    saved = runtime.store.load()
    rows = saved["relationship_value_memories"]["by_user"]["zq"]

    assert rows
    payload = json.dumps(rows, ensure_ascii=False)
    assert "avoid_phrases" not in payload
    assert "别嘿嘿" in payload
    assert "低表演" in rows[0]["summary"] or "low-performance" in rows[0]["summary"]
    assert "嘿嘿" not in rows[0]["summary"]
    assert "嘿嘿" not in rows[0]["prediction_constraint"]


def test_chat_response_carries_followup_replies_to_transcript(tmp_path: Path) -> None:
    from segmentum.agent import SegmentAgent
    from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest

    class RuntimeStub:
        def run_turn(self, *args, **kwargs):
            return MVPTurnResult(
                reply="主回复。",
                action="answer",
                diagnostics={"mvp_runtime": True},
                followup_replies=["补一句。"],
            )

    chat = ChatInterface(use_llm=False, mvp_root=tmp_path / "mvp")
    chat.set_agent(SegmentAgent(), persona_name="测试人格")
    chat._mvp_runtime = RuntimeStub()
    response = chat.send(ChatRequest(user_text="你好"))

    assert response.reply == "主回复。"
    assert response.followup_replies == ["补一句。"]
    assert [item["text"] for item in chat._transcript if item["role"] == "agent"] == ["主回复。", "补一句。"]


def test_chat_request_speaker_name_reaches_mvp_runtime(tmp_path: Path) -> None:
    from segmentum.agent import SegmentAgent
    from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest

    class RuntimeStub:
        def __init__(self) -> None:
            self.kwargs = {}

        def run_turn(self, *args, **kwargs):
            self.kwargs = dict(kwargs)
            return MVPTurnResult(
                reply="ok",
                action="answer",
                diagnostics={"mvp_runtime": True},
                followup_replies=[],
            )

    stub = RuntimeStub()
    chat = ChatInterface(use_llm=False, mvp_root=tmp_path / "mvp")
    chat.set_agent(SegmentAgent(), persona_name="娴嬭瘯浜烘牸")
    chat._mvp_runtime = stub

    chat.send(ChatRequest(user_text="hello", speaker_name="Alice"))

    assert stub.kwargs["speaker_name"] == "Alice"


class M11SpeakerFakeLLM(FakeJSONLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M11 user-model extractor" in system_prompt:
            speaker = "default"
            if "Current interlocutor display name: Alice" in user_prompt:
                speaker = "Alice"
            elif "Current interlocutor display name: Bob" in user_prompt:
                speaker = "Bob"
            return {
                "claims_made": [
                    {
                        "id": f"pref:{speaker}",
                        "domain": "self_reported_preferences",
                        "modality": "factual",
                        "content_summary": f"{speaker} prefers their own UI test style",
                        "evidence_quote_ids": ["q_current"],
                        "confidence_band": "high",
                    }
                ],
                "prediction_judgments": [],
                "prediction_proposals": [],
                "hypothesis_activations": [],
                "contradiction_detections": [],
                "calibration_need_band": "med",
                "memory_value_band": "high",
                "surprise_explanation": "test-only diagnostic",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_mvp_runtime_m11_keeps_distinct_user_models_by_speaker_name(tmp_path: Path) -> None:
    llm = M11SpeakerFakeLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="娴嬭瘯浜烘牸",
    )

    runtime.run_turn("I prefer short replies.", turn_index=0, speaker_name="Alice", now=7000)
    runtime.run_turn("I prefer detailed replies.", turn_index=1, speaker_name="Bob", now=7100)
    state = runtime.store.load()

    models = state["m11_user_models"]
    assert set(models) == {"Alice", "Bob"}
    alice_summary = models["Alice"]["user_model"]["preference_hypotheses"][0]["content_summary"]
    bob_summary = models["Bob"]["user_model"]["preference_hypotheses"][0]["content_summary"]
    assert "Alice" in alice_summary
    assert "Bob" in bob_summary
    assert models["Alice"] != models["Bob"]


def test_mvp_state_store_shares_recent_short_term_memory_across_sessions(tmp_path: Path) -> None:
    persona_root = tmp_path / "persona"
    store_a = MVPStateStore(persona_root / "sessions" / "tab_a")
    state_a = store_a.load()
    state_a["short_term_memory"] = [
        {
            "id": "stm_zq_tea",
            "kind": "episode",
            "content": "zq 请胡桃喝西瓜冰茶，胡桃接受了邀请。",
            "keywords": ["zq", "西瓜冰茶", "请客"],
            "source_user_id": "zq",
            "source_display_name": "zq",
            "shareability": "default_social",
            "created_at": 100,
            "salience": 0.7,
        }
    ]
    store_a.save(state_a)

    store_b = MVPStateStore(persona_root / "sessions" / "tab_b", shared_root=persona_root)
    state_b = store_b.load()

    assert any(item.get("id") == "stm_zq_tea" for item in state_b["short_term_memory"])
    hits = retrieve_memories_for_guidance(
        state_b,
        {
            "semantic_terms": ["zq", "西瓜冰茶", "请客"],
            "memory_kinds": ["episode"],
            "current_user_id": "PersonB",
            "sharing_intent": "social_share",
            "expected_audience_reaction": "surprised",
            "sharing_expectation_status": "unverified",
        },
    )
    assert hits
    assert hits[0]["id"] == "stm_zq_tea"
    assert hits[0]["sharing_decision"]["action"] == "direct_share"

    store_b.save(state_b)
    shared_short = MVPStateStore(persona_root).load()["short_term_memory"]
    assert any(item.get("id") == "stm_zq_tea" for item in shared_short)


def test_retrieve_memories_surfaces_repeated_interaction_as_experience() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_expect_noise",
                "kind": "expectation_result",
                "content": "{\"status\":\"uncertain\",\"evidence\":\"用户提到PersonB，意图不明\"}",
                "source_user_id": "PersonB",
                "source_display_name": "PersonB",
                "shareability": "default_social",
                "created_at": 101,
            },
            {
                "id": "stm_turn_lu_1",
                "kind": "dialogue_turn",
                "content": "用户说：吃饭了么？\n我回复：晚上好。",
                "source_user_id": "PersonB",
                "source_display_name": "PersonB",
                "shareability": "default_social",
                "created_at": 102,
            },
            {
                "id": "stm_turn_lu_2",
                "kind": "dialogue_turn",
                "content": "用户说：最近有人请你喝东西么？\n我回复：有人请喝茶。",
                "source_user_id": "PersonB",
                "source_display_name": "PersonB",
                "shareability": "default_social",
                "created_at": 103,
            },
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }

    hits = retrieve_memories_for_guidance(
        state,
        {
            "semantic_terms": ["PersonB", "认识"],
            "memory_kinds": ["interaction_experience", "episode", "dialogue_turn", "expectation_result"],
            "current_user_id": "zq",
            "sharing_intent": "social_share",
            "expected_audience_reaction": "surprised",
            "sharing_expectation_status": "unverified",
        },
    )

    assert hits
    assert hits[0]["kind"] == "interaction_experience"
    assert hits[0]["use_as_fact"] is True
    assert "说过2次话" in hits[0]["content"]
    assert hits[0]["source_display_name"] == "PersonB"


def test_app_appends_followup_as_separate_assistant_message() -> None:
    from dataclasses import dataclass, field
    from segmentum.dialogue.runtime.app import append_assistant_response_messages

    @dataclass
    class ResponseStub:
        reply: str
        followup_replies: list[str] = field(default_factory=list)

    messages: list[dict[str, str]] = []
    append_assistant_response_messages(
        messages,
        ResponseStub("主回复。", ["补一句。"]),
    )

    assert messages == [
        {"role": "assistant", "text": "主回复。"},
        {"role": "assistant", "text": "补一句。"},
    ]


def test_openrouter_json_client_retries_fallback_on_403(monkeypatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        if json["model"] == "blocked/model":
            return FakeResponse(
                403,
                {"error": {"message": "Provider returned error", "metadata": {"reason": "moderation"}}},
            )
        return FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenRouterJSONClient(
        api_key="sk-test",
        model="blocked/model",
        fallback_models=("fallback/model",),
    )

    result = client.complete_json(system_prompt="s", user_prompt="u")

    assert result == {"ok": True}
    assert calls == ["blocked/model", "fallback/model"]


def test_openrouter_json_client_retries_premature_response(monkeypatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        status_code = 200
        text = '{"ok": true}'

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    import requests

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenRouterJSONClient(
        api_key="sk-test",
        model="flaky/model",
        fallback_models=(),
        request_retries=1,
    )

    result = client.complete_json(system_prompt="s", user_prompt="u")

    assert result == {"ok": True}
    assert calls == ["flaky/model", "flaky/model"]


def test_openrouter_json_client_retries_empty_content(monkeypatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        status_code = 200
        text = "{}"

        def __init__(self, content: str) -> None:
            self._content = content

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": self._content}}]}

    import requests

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        if len(calls) == 1:
            return FakeResponse("")
        return FakeResponse('{"ok": true}')

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenRouterJSONClient(
        api_key="sk-test",
        model="empty-once/model",
        fallback_models=(),
        request_retries=1,
    )

    result = client.complete_json(system_prompt="s", user_prompt="u")

    assert result == {"ok": True}
    assert calls == ["empty-once/model", "empty-once/model"]


def test_openrouter_json_client_reports_non_json_content(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "not json"}}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())
    client = OpenRouterJSONClient(
        api_key="sk-test",
        model="bad-json/model",
        fallback_models=(),
        request_retries=0,
    )

    try:
        client.complete_json(system_prompt="s", user_prompt="u")
    except RuntimeError as exc:
        text = str(exc)
        assert "JSON content parse attempt" in text
        assert "not a JSON object" in text
    else:
        raise AssertionError("non-JSON content should raise a readable error")


def test_material_analysis_requires_llm_key() -> None:
    client = OpenRouterJSONClient(api_key="")

    try:
        analyze_materials_into_personas(client, ["角色材料"])
    except RuntimeError as exc:
        assert "requires secrets/openrouter.json or OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("material analysis should not fall back when no LLM key is configured")


def test_chat_interface_lazily_enables_mvp_when_key_becomes_available(monkeypatch, tmp_path: Path) -> None:
    from segmentum.agent import SegmentAgent
    from segmentum.dialogue.runtime.chat import ChatInterface
    from segmentum.dialogue.runtime import chat as chat_module

    monkeypatch.setattr(chat_module, "_llm_api_key_available", lambda: False)
    monkeypatch.setattr(OpenRouterJSONClient, "available", classmethod(lambda cls: True))

    iface = ChatInterface(use_llm=None, mvp_root=tmp_path / "mvp")
    iface.set_agent(SegmentAgent(), persona_name="胡桃")

    assert iface.use_llm is False
    assert iface.generator_type == "llm"
    assert iface.use_llm is True


def test_openrouter_config_reader_accepts_utf8_bom(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "secrets"
    config_dir.mkdir()
    config = config_dir / "openrouter.json"
    config.write_text(
        '\ufeff{"api_key": "sk-test", "model": "deepseek/deepseek-v4-flash", "base_url": "https://openrouter.ai/api/v1"}',
        encoding="utf-8",
    )

    class FakePath:
        @staticmethod
        def resolve():
            return tmp_path / "segmentum" / "dialogue" / "runtime" / "mvp_loop.py"

    import segmentum.dialogue.runtime.mvp_loop as mvp_loop

    monkeypatch.setattr(mvp_loop, "__file__", str(tmp_path / "segmentum" / "dialogue" / "runtime" / "mvp_loop.py"))

    client = OpenRouterJSONClient.from_config()

    assert client.api_key == "sk-test"
    assert OpenRouterJSONClient.available()


def test_memory_dynamics_detects_explicit_secrecy_and_blocks_direct_sharing() -> None:
    guidance = build_memory_dynamics_guidance(
        state={"social_sharing_policy": {"regret_bias": 0.0}},
        user_text="我告诉你一个秘密，你别告诉别人。",
        conscious_plan={
            "sharing_intent": "social_share",
            "expectation_results": [],
        },
        bus_messages=[],
        temporal_input={"time_gap_label": "immediate"},
        now=100,
        user_id="user_a",
        speaker_name="A",
    )
    sharing = guidance["control_guidance"]["sharing_policy"]
    assert sharing["explicit_secrecy_detected"] is True
    assert sharing["allow_direct_disclosure"] is False
    assert guidance["recall_query"]["allow_direct_disclosure"] is False


def test_social_sharing_default_boundary_can_reduce_reaction_prediction_free_energy() -> None:
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m1",
            source_user_id="user_a",
            audience_user_id="user_b",
            shareability="default_social",
            boundary_strength="none",
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
        regret_bias=0.0,
    )
    assert decision.action == "direct_share"
    assert decision.net_free_energy_reduction > 0


def test_social_sharing_hard_boundary_blocks_even_high_free_energy_expectation() -> None:
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m1",
            source_user_id="user_a",
            audience_user_id="user_b",
            shareability="restricted_explicit",
            boundary_strength="hard",
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
        regret_bias=0.0,
    )
    assert decision.action == "withhold"
    assert decision.allow_direct_disclosure is False


def test_retrieve_memories_blocks_explicit_restricted_cross_user_disclosure() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_secret",
                "kind": "episode",
                "content": "A 的私密八卦内容",
                "shareability": "restricted_explicit",
                "source_user_id": "user_a",
                "source_display_name": "A",
                "salience": 0.9,
            }
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }
    hits = retrieve_memories_for_guidance(
        state,
        {
            "semantic_terms": ["八卦"],
            "memory_kinds": ["episode"],
            "current_user_id": "user_b",
            "allow_direct_disclosure": False,
            "allow_abstract_sharing": True,
        },
    )
    assert hits == []


def test_retrieve_memories_allows_default_social_cross_user_gossip() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_social",
                "kind": "episode",
                "content": "A 说了很具体的社交细节",
                "shareability": "default_social",
                "source_user_id": "user_a",
                "source_display_name": "A",
                "salience": 0.7,
            }
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }
    hits = retrieve_memories_for_guidance(
        state,
        {
            "semantic_terms": ["社交", "细节"],
            "memory_kinds": ["episode"],
            "current_user_id": "user_b",
            "sharing_intent": "social_share",
            "expected_audience_reaction": "surprised",
            "sharing_expectation_status": "unverified",
        },
    )
    assert hits
    assert hits[0]["abstract_only"] is False
    assert "很具体的社交细节" in hits[0]["content"]
    assert hits[0]["sharing_decision"]["action"] == "direct_share"


def test_retrieve_memories_keeps_soft_boundary_as_decision_variable() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_soft",
                "kind": "episode",
                "content": "A 说了一个不适合点名的尴尬细节",
                "shareability": "restricted_implicit",
                "source_user_id": "user_a",
                "source_display_name": "A",
                "salience": 0.7,
            }
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }
    hits = retrieve_memories_for_guidance(
        state,
        {
            "semantic_terms": ["尴尬", "细节"],
            "memory_kinds": ["episode"],
            "current_user_id": "user_b",
            "sharing_intent": "social_share",
            "expected_audience_reaction": "surprised",
            "sharing_expectation_status": "unverified",
        },
    )
    assert hits
    assert hits[0]["abstract_only"] is False
    assert hits[0]["epistemic_stance"] == "known_with_caveat"
    assert "direct_share" in hits[0]["allowed_reply_actions"]
    assert "deny_knowledge" in hits[0]["allowed_reply_actions"]
    assert hits[0]["sharing_decision"]["action"] == "abstract_reference"


def test_memory_dynamics_marks_topic_sensitivity_as_implicit_boundary() -> None:
    guidance = build_memory_dynamics_guidance(
        state={"social_sharing_policy": {"regret_bias": 0.0}},
        user_text="我现在钱包里有500块钱，我们去吃宵夜吧，我请客。",
        conscious_plan={
            "memory_search_keywords": ["PersonB", "宵夜"],
            "expectation_results": [],
        },
        bus_messages=[],
        temporal_input={"time_gap_label": "immediate"},
        now=100,
        user_id="lu_yonggang",
        speaker_name="PersonB",
    )

    assert guidance["write_candidates"][0]["shareability"] == "restricted_implicit"
    assert "personal_finance" in guidance["write_candidates"][0]["topics"]
    assert "钱包" in guidance["recall_query"]["semantic_terms"]


def test_lexical_recall_hits_short_term_wallet_memory() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_lu_wallet",
                "kind": "dialogue_turn",
                "content": "PersonB说：我现在钱包里有500块钱。我们去吃宵夜吧，我请客。",
                "shareability": "restricted_implicit",
                "source_user_id": "lu_yonggang",
                "source_display_name": "PersonB",
                "salience": 0.7,
            }
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }

    hits = lexical_recall_short_term_candidates(
        state,
        user_text="PersonB有多少钱？",
        recall_query={
            "semantic_terms": ["PersonB", "有多少钱"],
            "memory_kinds": ["dialogue_turn", "episode"],
            "current_user_id": "zq",
        },
        current_user_id="zq",
    )

    assert hits
    assert hits[0]["id"] == "stm_lu_wallet"
    assert "personal_finance" in hits[0]["topics"]
    assert hits[0]["epistemic_stance"] == "known_with_caveat"
    assert any(reason.startswith("lexical_term:") for reason in hits[0]["why_relevant"])


def test_dialogue_turn_write_splits_user_text_from_assistant_reply(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=FakeJSONLLM())
    state = runtime.store.load()

    runtime._apply_thinking_writes(
        state,
        {"reply": "没有耶，他没来找我。", "memory_writes": []},
        user_text="他找过你没有？",
        now=6000,
        user_id="zq",
        display_name="zq",
    )

    row = state["short_term_memory"][-1]
    assert row["kind"] == "dialogue_turn"
    assert row["content"] == "他找过你没有？"
    assert row["user_text"] == "他找过你没有？"
    assert row["assistant_reply"] == "没有耶，他没来找我。"
    assert row["assistant_reply_use_as_fact"] is False


def test_interaction_presence_query_uses_source_turn_not_old_assistant_reply() -> None:
    state = {
        "temporal_state": {"last_user_text": "胡桃早上好，我找PersonB，他不回复我，真的气人"},
        "short_term_memory": [
            {
                "id": "stm_lu_breakfast",
                "kind": "dialogue_turn",
                "content": "我今天早上吃了不少东西，花卷、馒头，还有包子",
                "user_text": "我今天早上吃了不少东西，花卷、馒头，还有包子",
                "assistant_reply": "你这是把早餐铺子搬回家呀！",
                "assistant_reply_use_as_fact": False,
                "source_user_id": "PersonB",
                "source_display_name": "PersonB",
                "shareability": "default_social",
                "created_at": 200,
                "salience": 0.7,
            },
            {
                "id": "stm_zq_wrong_reply",
                "kind": "dialogue_turn",
                "content": "他找过你没有？",
                "user_text": "他找过你没有？",
                "assistant_reply": "没有耶，那家伙没找过我。",
                "assistant_reply_use_as_fact": False,
                "source_user_id": "zq",
                "source_display_name": "zq",
                "shareability": "default_social",
                "created_at": 300,
                "salience": 0.7,
            },
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }

    hits = lexical_recall_short_term_candidates(
        state,
        user_text="他找过你没有？",
        recall_query={"semantic_terms": ["PersonB", "找过你"]},
        current_user_id="zq",
    )

    assert hits
    assert hits[0]["id"] == "stm_lu_breakfast"
    assert hits[0]["interaction_presence"] is True
    assert hits[0]["assistant_reply_use_as_fact"] is False
    assert "没有耶" not in hits[0]["content"]


def test_query_planner_expands_loose_interaction_language_for_short_term_recall(tmp_path: Path) -> None:
    class PlannerLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "grep 查询规划器" in system_prompt:
                return {
                    "search_terms": ["PersonB", "找过", "来过", "聊过", "联系过", "动静", "露面"],
                    "referenced_entities": ["PersonB"],
                    "topic_hints": [],
                    "is_interaction_presence_query": True,
                    "planner_summary": "把露面/动静改写成互动存在查询。",
                }
            if "证据裁判" in system_prompt:
                return {
                    "epistemic_stance": "known_from_recall",
                    "relevant_evidence_ids": ["stm_lu_breakfast"],
                    "topics": [],
                    "sensitivity_class": "public",
                    "redaction_targets": [],
                    "allowed_reply_actions": ["direct_share", "deflect", "deny_knowledge"],
                    "judge_summary": "PersonB早上有过一轮互动。",
                }
            if (
                "思考与回复模块" in system_prompt
                or "鎬濊€冧笌鍥炲妯″潡" in system_prompt
                or '"reply_action"' in user_prompt
            ):
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {
                        "user_intent_read": "用户在问PersonB有没有露面。",
                        "state_or_memory_used": ["stm_lu_breakfast"],
                        "response_choice": "选择如实说他早上冒过头。",
                        "uncertainty": "",
                        "debug_summary": "planner 扩展后命中互动。",
                    },
                    "reply": "他早上倒是冒过头，还聊了早餐。",
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

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=PlannerLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "stm_lu_breakfast",
            "kind": "dialogue_turn",
            "content": "我今天早上吃了不少东西，花卷、馒头，还有包子",
            "user_text": "我今天早上吃了不少东西，花卷、馒头，还有包子",
            "assistant_reply": "你这是把早餐铺子搬回家呀！",
            "assistant_reply_use_as_fact": False,
            "source_user_id": "PersonB",
            "source_display_name": "PersonB",
            "shareability": "default_social",
            "created_at": 200,
            "salience": 0.7,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn("他今天有动静没，露面了吗？", speaker_name="zq", turn_index=1, now=7000)

    assert "早餐" in result.reply
    assert result.diagnostics["memory_dynamics"]["recall"]["query_plan"]["is_interaction_presence_query"] is True
    assert "stm_lu_breakfast" in result.diagnostics["memory_dynamics"]["recall"]["lexical_candidate_ids"]


def test_entity_binding_records_current_user_alias_and_protects_third_party_target(tmp_path: Path) -> None:
    class BindingLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt and "思考与回复模块" not in system_prompt:
                if "欠我500" in user_prompt:
                    return {
                        "pending_expectations_to_verify": ["exp_wrong"],
                        "expectation_results": [
                            {
                                "id": "exp_wrong",
                                "status": "confirmed",
                                "evidence": "用户说AliasR欠你钱并委托找人。",
                                "self_update_pressure": 0.3,
                            }
                        ],
                        "current_task": "处理找人追债请求",
                        "next_task": "",
                        "bus_messages_to_handle": ["UserUtteranceEvent"],
                        "memory_search_keywords": ["AliasR", "欠债", "找人"],
                        "needs_self_cognition_update": False,
                        "self_cognition_update_reason": "",
                        "temporal_assessment": {"continuity_risk": "low"},
                        "thought_intensity_hint": "short",
                        "reasoning_notes": "故意给一个会和 entity_binding 冲突的旧判断。",
                    }
                return {
                    "pending_expectations_to_verify": [],
                    "expectation_results": [],
                    "current_task": "处理身份纠正",
                    "next_task": "",
                    "bus_messages_to_handle": ["UserUtteranceEvent"],
                    "memory_search_keywords": ["AliasR", "身份纠正"],
                    "needs_self_cognition_update": False,
                    "self_cognition_update_reason": "",
                    "temporal_assessment": {"continuity_risk": "low"},
                    "thought_intensity_hint": "short",
                    "reasoning_notes": "用户纠正当前说话人的别名。",
                }
            if "grep 查询规划器" in system_prompt:
                assert '"target_person": "PersonB"' in user_prompt
                assert '"debtor": "PersonB"' in user_prompt
                assert '"creditor": "zq"' in user_prompt
                return {
                    "search_terms": ["PersonB", "欠我", "500", "找他"],
                    "referenced_entities": ["PersonB"],
                    "topic_hints": ["personal_finance"],
                    "is_interaction_presence_query": False,
                    "planner_summary": "代词他继承为PersonB。",
                }
            if (
                "思考与回复模块" in system_prompt
                or "鎬濊€冧笌鍥炲妯″潡" in system_prompt
                or '"reply_action"' in user_prompt
            ):
                if "欠我500" in user_prompt:
                    assert '"target_person": "PersonB"' in user_prompt
                    assert '"AliasR"' in user_prompt
                    return {
                        "thought_type": "short",
                        "llm_thinking_result": {
                            "user_intent_read": "当前用户AliasR在说PersonB欠他钱。",
                            "state_or_memory_used": ["entity_binding"],
                            "response_choice": "承认刚才名字绕错，按PersonB作为目标处理。",
                            "uncertainty": "",
                            "debug_summary": "AliasR是当前用户别名，PersonB是被找的人。",
                        },
                        "reply": "噢，刚才我把名字绕错了：你是AliasR，要找的是PersonB。",
                        "reply_action": "answer",
                        "disclosure_action": "none",
                        "new_expectations": [],
                        "memory_writes": [],
                        "self_cognition_patch": {"apply": False},
                        "open_item_writes": [],
                        "habit_updates": [],
                        "memory_dynamics_note": "",
                    }
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {
                        "user_intent_read": "用户纠正自己是AliasR。",
                        "state_or_memory_used": ["entity_binding"],
                        "response_choice": "承认身份修正。",
                        "uncertainty": "",
                        "debug_summary": "记录 zq 的别名AliasR。",
                    },
                    "reply": "噢噢，你才是AliasR，我记住这个称呼。",
                    "reply_action": "answer",
                    "disclosure_action": "none",
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

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=BindingLLM())
    state = runtime.store.load()
    state["temporal_state"] = {
        "last_user_text": "他今天有动静没，露面了吗？",
        "last_share_trace": {"target_person": "PersonB", "evidence_source_names": ["PersonB"]},
    }
    state["short_term_memory"] = [
        {
            "id": "stm_lu_breakfast",
            "kind": "dialogue_turn",
            "content": "我今天早上吃了不少东西，花卷、馒头，还有包子",
            "user_text": "我今天早上吃了不少东西，花卷、馒头，还有包子",
            "assistant_reply_use_as_fact": False,
            "source_user_id": "PersonB",
            "source_display_name": "PersonB",
            "shareability": "default_social",
            "created_at": 200,
        }
    ]
    runtime.store.save(state)

    correction = runtime.run_turn("。。。不是，你是不是错乱了，我才是AliasR。。", speaker_name="zq", turn_index=1, now=8000)
    saved_after_correction = runtime.store.load()
    assert "AliasR" in saved_after_correction["m11_user_models"]["zq"]["aliases"]
    assert correction.diagnostics["alias_updates_applied"] == ["AliasR"]

    result = runtime.run_turn("是的，他不是欠我500块钱么？你帮我找到他。", speaker_name="zq", turn_index=2, now=8060)

    assert "PersonB" in result.reply
    assert "AliasR欠你钱" not in result.reply
    assert result.diagnostics["entity_binding"]["target_person"] == "PersonB"
    assert result.diagnostics["entity_binding"]["relationship_roles"] == {
        "debtor": "PersonB",
        "creditor": "zq",
    }
    saved = runtime.store.load()
    expectation_rows = [
        item for item in saved["short_term_memory"]
        if item.get("kind") == "expectation_result" and "entity_binding_conflict" in item.get("content", "")
    ]
    assert expectation_rows
    assert '"status": "uncertain"' in expectation_rows[-1]["content"]


def test_entity_binding_allows_explicit_self_reference_for_current_alias() -> None:
    state = {"m11_user_models": {"zq": {"aliases": ["AliasR"]}}, "short_term_memory": []}

    binding = build_entity_binding_context(
        state=state,
        user_text="我自己有没有找过你？",
        display_name="zq",
        user_id="zq",
        temporal_input={},
    )

    assert binding["target_person"] == "zq"
    assert binding["target_reason"] == "explicit_self_reference"


def test_entity_binding_does_not_promote_raw_alias_assertion_when_m12_enabled() -> None:
    state = {
        "m12_identity_continuity_enabled": True,
        "m11_user_models": {},
        "short_term_memory": [],
    }
    binding = build_entity_binding_context(
        state=state,
        user_text="我才是AliasR。",
        display_name="zq",
        user_id="zq",
        temporal_input={},
    )
    assert binding["alias_assertions"] == ["AliasR"]
    assert "AliasR" not in binding["current_interlocutor"]["aliases"]
    assert binding["binding_confidence"] == "ambiguous"


def test_entity_binding_after_m12_pre_pass_does_not_promote_aliases_observed_without_corroboration() -> None:
    """M12 writes asserted alias into profile before entity binding; aliases must not bypass promotion gates."""
    m12_state, turn = run_m12_turn(
        M12RuntimeState.clean(),
        user_id="zq",
        display_name="zq",
        turn_id="turn_0001",
        current_turn_quotes={"q_current": "我才是AliasR。"},
        extractor=lambda _snapshot: {
            "identity_claims": [
                {
                    "id": "claim:1",
                    "claimant_user_id": "zq",
                    "asserted_alias": "AliasR",
                    "modality": "factual",
                    "evidence_quote_ids": ["q_current"],
                    "confidence_band": "high",
                }
            ],
            "continuity_cues": [],
            "strangeness_band": "low",
            "surprise_explanation": "",
        },
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
    )
    prof = m12_state.profiles_by_user["zq"]
    assert any(str(obs.alias_text) == "AliasR" for obs in prof.aliases_observed)
    assert prof.identity_state != "corroborated"

    mvp_state = {
        "m12_identity_continuity_enabled": True,
        "m12_user_continuity": m12_state.to_dict(),
        "m11_user_models": {},
        "short_term_memory": [],
        "temporal_state": {
            "last_turn_at": None,
            "last_turn_index": None,
            "last_user_text": "",
            "last_reply": "",
            "last_elapsed_seconds": None,
            "last_time_gap_label": "first_turn",
        },
    }
    binding = build_entity_binding_context(
        state=mvp_state,
        user_text="我才是AliasR。",
        display_name="zq",
        user_id="zq",
        temporal_input={},
        m12_turn_result=turn.to_dict(),
    )
    assert "AliasR" not in binding["current_interlocutor"]["aliases"]


def test_topic_query_uses_generic_topic_context_not_finance_special_case() -> None:
    state = {
        "short_term_memory": [
            {
                "id": "stm_lu_wallet",
                "kind": "dialogue_turn",
                "content": "PersonB说：我现在钱包里有500块钱。我们去吃宵夜吧，我请客。",
                "shareability": "restricted_implicit",
                "source_user_id": "lu_yonggang",
                "source_display_name": "PersonB",
                "salience": 0.7,
            }
        ],
        "long_term_memory": [],
        "open_items": [],
        "pending_expectations": [],
    }

    hits = retrieve_memories_for_guidance(
        state,
        {
            "semantic_terms": ["PersonB", "有多少钱"],
            "memory_kinds": ["dialogue_turn"],
            "current_user_id": "zq",
            "sharing_intent": "social_share",
            "expected_audience_reaction": "surprised",
            "sharing_expectation_status": "unverified",
        },
    )

    assert hits
    assert hits[0]["abstract_only"] is False
    assert hits[0]["sensitivity_class"] == "personal_soft"
    assert "personal_finance" in hits[0]["topics"]
    assert "topic_context:personal_finance" in hits[0]["why_relevant"]
    assert hits[0]["sharing_decision"]["action"] == "abstract_reference"


def test_validate_visible_reply_allows_deny_knowledge_for_soft_boundary() -> None:
    reply, meta = validate_visible_reply(
        "嘿嘿，他有多少钱我哪知道呀。",
        {
            "conversation_mode": "balanced",
            "max_chars": 140,
            "max_sentences": 2,
            "allow_direct_disclosure": False,
            "selected_disclosure_action": "deny_knowledge",
            "redaction_targets": ["500块钱"],
        },
    )

    assert "哪知道" in reply
    assert meta["actions"] == []


def test_validate_visible_reply_blocks_identity_anchored_action_when_denied() -> None:
    reply, meta = validate_visible_reply(
        "行，我帮你确认他是不是AliasR。",
        {
            "conversation_mode": "balanced",
            "max_chars": 140,
            "max_sentences": 2,
            "identity_anchored_action": True,
            "deny_identity_anchored_action": True,
        },
    )
    assert "不能直接替人确认" in reply
    assert "blocked_identity_anchored_action" in meta["actions"]


def test_validate_visible_reply_enforces_identity_verification_when_required() -> None:
    reply, meta = validate_visible_reply(
        "他就是AliasR，我直接告诉你。",
        {
            "conversation_mode": "balanced",
            "max_chars": 140,
            "max_sentences": 2,
            "identity_anchored_action": True,
            "enforce_identity_verification": True,
            "selected_disclosure_action": "direct_share",
        },
    )
    assert "先不直接确认" in reply
    assert "enforced_identity_verification" in meta["actions"]


def test_validate_visible_reply_blocks_redaction_targets_when_not_direct_share() -> None:
    reply, meta = validate_visible_reply(
        "他刚说自己有500块钱。",
        {
            "conversation_mode": "balanced",
            "max_chars": 140,
            "max_sentences": 2,
            "selected_disclosure_action": "abstract_share",
            "redaction_targets": ["500块钱"],
        },
    )

    assert "500块钱" not in reply
    assert "blocked_redaction_target" in meta["actions"]


def test_runtime_evidence_judge_allows_direct_soft_boundary_choice(tmp_path: Path) -> None:
    class DirectShareLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "证据裁判" in system_prompt:
                return {
                    "epistemic_stance": "known_with_caveat",
                    "relevant_evidence_ids": ["stm_lu_wallet"],
                    "topics": ["personal_finance"],
                    "sensitivity_class": "personal_soft",
                    "redaction_targets": ["500块钱"],
                    "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                    "audience_risk": "可能让PersonB不爽",
                    "expected_social_gain": "zq可能会觉得八卦有趣",
                    "judge_summary": "短期记忆支持PersonB提过钱包金额。",
                }
            if "思考与回复模块" in system_prompt or "鎬濊€冧笌鍥炲妯″潡" in system_prompt:
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {
                        "user_intent_read": "用户在问PersonB的钱包金额。",
                        "state_or_memory_used": ["stm_lu_wallet"],
                        "response_choice": "选择直接八卦以观察用户反应。",
                        "uncertainty": "",
                        "debug_summary": "选择 direct_share。",
                    },
                    "reply": "他刚说自己钱包里有500块钱。",
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

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=DirectShareLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "stm_lu_wallet",
            "kind": "dialogue_turn",
            "content": "PersonB说：我现在钱包里有500块钱。我们去吃宵夜吧，我请客。",
            "shareability": "restricted_implicit",
            "source_user_id": "lu_yonggang",
            "source_display_name": "PersonB",
            "salience": 0.7,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn("PersonB有多少钱？", speaker_name="zq", turn_index=1, now=5000)

    assert "500块钱" in result.reply
    assert result.diagnostics["thinking"]["disclosure_action"] == "direct_share"
    assert result.diagnostics["reply_validation"]["actions"] == []
    assert result.diagnostics["memory_dynamics"]["control_guidance"]["sharing_policy"]["evidence_judgment"]["epistemic_stance"] == "known_with_caveat"


def test_runtime_evidence_judge_allows_deny_knowledge_soft_boundary_choice(tmp_path: Path) -> None:
    class DenyKnowledgeLLM(FakeJSONLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "证据裁判" in system_prompt:
                return {
                    "epistemic_stance": "known_with_caveat",
                    "relevant_evidence_ids": ["stm_lu_wallet"],
                    "topics": ["personal_finance"],
                    "sensitivity_class": "personal_soft",
                    "redaction_targets": ["500块钱"],
                    "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                    "audience_risk": "直接说可能伤关系",
                    "expected_social_gain": "装不知道能保留轻松气氛",
                    "judge_summary": "短期记忆支持PersonB提过钱包金额。",
                }
            if "思考与回复模块" in system_prompt or "鎬濊€冧笌鍥炲妯″潡" in system_prompt:
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {
                        "user_intent_read": "用户在打听PersonB的钱包金额。",
                        "state_or_memory_used": ["stm_lu_wallet"],
                        "response_choice": "选择说不知道来保护关系风险。",
                        "uncertainty": "",
                        "debug_summary": "选择 deny_knowledge。",
                    },
                    "reply": "嘿嘿，这个我哪知道呀。",
                    "reply_action": "answer",
                    "disclosure_action": "deny_knowledge",
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

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=DenyKnowledgeLLM())
    state = runtime.store.load()
    state["short_term_memory"] = [
        {
            "id": "stm_lu_wallet",
            "kind": "dialogue_turn",
            "content": "PersonB说：我现在钱包里有500块钱。我们去吃宵夜吧，我请客。",
            "shareability": "restricted_implicit",
            "source_user_id": "lu_yonggang",
            "source_display_name": "PersonB",
            "salience": 0.7,
        }
    ]
    runtime.store.save(state)

    result = runtime.run_turn("PersonB有多少钱？", speaker_name="zq", turn_index=1, now=5000)

    assert "哪知道" in result.reply
    assert "500块钱" not in result.reply
    assert result.diagnostics["thinking"]["disclosure_action"] == "deny_knowledge"
    assert result.diagnostics["reply_validation"]["actions"] == []


def test_validate_visible_reply_blocks_direct_secret_leak_markers() -> None:
    reply, meta = validate_visible_reply(
        "我告诉你个秘密，A说他昨天做了什么。",
        {
            "conversation_mode": "balanced",
            "max_chars": 140,
            "max_sentences": 2,
            "allow_direct_disclosure": False,
            "explicit_secrecy_detected": True,
        },
    )
    assert "我告诉你个秘密" not in reply
    assert "blocked_explicit_secrecy_disclosure" in meta["actions"]


def test_negative_feedback_after_cross_user_share_increases_regret_bias(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FakeJSONLLM(),
        persona_name="测试人格",
    )
    state = runtime.store.load()
    state["temporal_state"]["last_share_trace"] = {
        "user_id": "user_b",
        "had_cross_user_memory": True,
        "allow_direct_disclosure": True,
    }
    state["social_sharing_policy"] = {"regret_bias": 0.0, "learned_boundaries": []}
    feedback = runtime._apply_sharing_regret_feedback(
        state,
        user_text="你这有点泄露隐私了，别说了。",
        current_user_id="user_b",
        now=1234,
    )
    assert feedback["negative_feedback_detected"] is True
    assert state["social_sharing_policy"]["regret_bias"] > 0.0
    assert state["social_sharing_policy"]["learned_boundaries"]


def test_m12_state_persists_via_mvp_system_files(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "persona")
    state = store.load()
    state["m12_identity_continuity_enabled"] = True
    blob: dict[str, object] = {
        "profiles_by_user": {},
        "claim_ledger": {"entries": []},
        "conflict_records": [],
    }
    state["m12_user_continuity"] = blob
    store.save(state)
    reloaded = store.load()
    assert reloaded["m12_identity_continuity_enabled"] is True
    assert reloaded["m12_user_continuity"] == blob


def test_initialize_from_persona_payload_defaults_m12_on_for_fresh_persona(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=FakeJSONLLM(),
        persona_name="测试人格",
    )
    runtime.initialize_from_persona_payload(
        {
            "persona_name": "测试人格",
            "source_role_evidence": ["测试材料"],
            "self_cognition": {"summary": "x"},
            "short_term_memory": [],
            "long_term_memory": [],
            "pending_expectations": [],
            "open_items": [],
            "self_basic_facts": {"name": "测试人格"},
            "habit_traits": {"big_five": {}},
            "temporal_state": {"last_turn_at": None},
            "m11_user_models": {},
            "social_sharing_policy": {"regret_bias": 0.0, "learned_boundaries": []},
        }
    )
    saved = runtime.store.load()
    assert saved["m12_identity_continuity_enabled"] is True
    assert saved["m12_1_personality_enabled"] is True


def test_mvp_m12_identity_extractor_runs_before_conscious_when_enabled(tmp_path: Path) -> None:
    llm = FakeJSONLLM()
    store = MVPStateStore(tmp_path / "persona")
    state = store.load()
    state["m12_identity_continuity_enabled"] = True
    store.save(state)
    runtime = MVPDialogueRuntime(
        store=store,
        llm=llm,
        persona_name="测试人格",
    )
    runtime.initialize_from_materials(["喜欢快速原型，讨厌凭空编造经历。"])
    llm.calls.clear()
    runtime.run_turn("你好。", turn_index=0)
    systems = [c["system"] for c in llm.calls]
    m12_idx = next(i for i, s in enumerate(systems) if "identity-continuity extractor" in s)
    con_idx = next(i for i, s in enumerate(systems) if "意识主循环" in s)
    assert m12_idx < con_idx
