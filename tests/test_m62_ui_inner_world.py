from __future__ import annotations

from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest
from segmentum.dialogue.signal_extractors import ConflictTensionExtractor


def test_chat_interface_can_write_ui_conscious_trace(tmp_path: Path) -> None:
    chat = ChatInterface(
        use_llm=False,
        persona_name="ui_persona",
        enable_conscious_trace=True,
        conscious_root=tmp_path / "conscious",
        session_id="ui_session",
    )
    chat.set_agent(SegmentAgent(), persona_name="ui_persona")

    chat.send(ChatRequest(user_text="hello"))
    chat.send(ChatRequest(user_text="can you think this through?"))

    markdown_path = chat.conscious_markdown_path()
    trace_path = chat.conscious_trace_path()
    assert markdown_path is not None and markdown_path.exists()
    assert trace_path is not None and trace_path.exists()

    markdown = chat.get_conscious_markdown()
    rows = chat.get_conscious_trace_rows()

    assert "当前观察" in markdown
    assert "候选路径" in markdown
    assert [row["turn_id"] for row in rows] == ["turn_0000", "turn_0001"]
    assert all(row["persona_id"] == "ui_persona" for row in rows)
    assert all(row["session_id"] == "ui_session" for row in rows)


class _HistoryCaptureGenerator:
    def __init__(self) -> None:
        self.last_diagnostics: dict[str, object] = {}
        self.history_lengths: list[int] = []
        self.history_texts: list[list[str]] = []

    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history,
        *,
        master_seed: int,
        turn_index: int,
    ) -> str:
        del action, dialogue_context, personality_state, master_seed, turn_index
        self.history_lengths.append(len(conversation_history))
        self.history_texts.append([str(item.get("text", "")) for item in conversation_history])
        self.last_diagnostics = {"template_id": "capture:0"}
        return "captured"


def test_chat_interface_passes_text_history_between_ui_turns() -> None:
    generator = _HistoryCaptureGenerator()
    chat = ChatInterface(
        generator=generator,
        use_llm=False,
        persona_name="history_persona",
    )
    chat.set_agent(SegmentAgent(), persona_name="history_persona")

    chat.send(ChatRequest(user_text="我原神 60 级"))
    chat.send(ChatRequest(user_text="你呢？"))

    assert generator.history_lengths == [0, 2]
    assert "我原神 60 级" in generator.history_texts[1]
    assert "captured" in generator.history_texts[1]


def test_chinese_anger_and_threat_raise_conflict_tension() -> None:
    extractor = ConflictTensionExtractor()

    assert extractor.extract("你在骗我么？", [], 0, {}) >= 0.6
    assert extractor.extract("气死我了，我想打死你", [], 0, {}) >= 0.8


def test_chat_interface_can_sync_existing_ui_messages() -> None:
    generator = _HistoryCaptureGenerator()
    chat = ChatInterface(
        generator=generator,
        use_llm=False,
        persona_name="sync_persona",
    )
    chat.set_agent(SegmentAgent(), persona_name="sync_persona")
    chat.sync_transcript_from_messages(
        [
            {"role": "user", "text": "我原神 60 级"},
            {"role": "assistant", "text": "厉害啊"},
            {"role": "user", "text": "你呢？"},
        ],
        pending_user_text="你呢？",
    )

    chat.send(ChatRequest(user_text="你呢？"))

    assert generator.history_lengths == [2]
    assert generator.history_texts[0] == ["我原神 60 级", "厉害啊"]
