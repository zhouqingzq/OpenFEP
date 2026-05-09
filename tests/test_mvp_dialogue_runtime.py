from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    OpenRouterJSONClient,
    analyze_materials_into_personas,
    build_free_energy_personality_analysis_prompt,
    retrieve_memories,
)


class FakeJSONLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
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
                "inner_thought": "这会唤起我对 Python 原型的偏好。",
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
                "thought_intensity_hint": "short",
                "reasoning_notes": "需要检索相关偏好。",
            }
        return {}


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
    assert len(llm.calls) == 3
    assert "意识主循环" in llm.calls[1]["system"]
    assert "思考与回复模块" in llm.calls[2]["system"]

    saved = runtime.store.load()
    assert saved["pending_expectations"][0]["id"] == "exp_project_detail"
    assert any("用户提到了 Python" in item["content"] for item in saved["long_term_memory"])
    recalled = [
        item for item in saved["long_term_memory"]
        if item.get("id") == "ltm_python"
    ][0]
    assert recalled["recall_count"] == 1


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
