from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from segmentum.agent import SegmentAgent
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.dialogue.turn_trace import ConsciousMarkdownWriter
from segmentum.dialogue.types import TranscriptUtterance
from segmentum.tracing import JsonlTraceWriter


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


class PromptLeakGenerator(RuleBasedGenerator):
    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history: Sequence[TranscriptUtterance],
        *,
        master_seed: int,
        turn_index: int,
    ) -> str:
        reply = super().generate(
            action,
            dialogue_context,
            personality_state,
            conversation_history,
            master_seed=master_seed,
            turn_index=turn_index,
        )
        self.last_diagnostics.update(
            {
                "system_prompt": "FULL SYSTEM PROMPT SHOULD NOT LEAK",
                "user_message": "SECRET USER PROMPT SHOULD NOT LEAK",
                "conversation_history": ["FULL HISTORY SHOULD NOT LEAK"],
                "api_key": "sk-secret-should-not-leak",
                "llm_tokens_total": 12,
            }
        )
        return reply


def test_run_conversation_writes_one_trace_row_per_turn(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(trace_path)
    lines = ["hello there", "can we look closer?"]

    turns = run_conversation(
        _agent(),
        lines,
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=620,
        partner_uid=9,
        session_id="m62-session",
        persona_id="persona-a",
        trace_writer=writer,
    )
    rows = _read_jsonl(trace_path)

    assert len(turns) == 2
    assert len(rows) == 2
    assert [row["turn_index"] for row in rows] == [0, 1]
    assert all(row["persona_id"] == "persona-a" for row in rows)
    assert all(row["session_id"] == "m62-session" for row in rows)


def test_trace_contains_observation_decision_prompt_generation_and_outcome(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    run_conversation(
        _agent(),
        ["I am unsure what you mean"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=621,
        session_id="keys",
        trace_writer=JsonlTraceWriter(trace_path),
    )
    row = _read_jsonl(trace_path)[0]

    for key in (
        "observation_channels",
        "selected_events",
        "affective_state_summary",
        "retrieved_memory_summary",
        "ranked_options",
        "chosen_action",
        "fep_prompt_capsule",
        "generation_diagnostics",
        "outcome_label",
        "memory_update_signal",
    ):
        assert key in row
    assert row["observation_source"] == "DialogueObserver.observe"
    assert isinstance(row["ranked_options"], list)
    assert isinstance(row["fep_prompt_capsule"], dict)


def test_affective_state_summary_is_compact_channel_derived(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    run_conversation(
        _agent(),
        ["I feel tense and confused"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=622,
        trace_writer=JsonlTraceWriter(trace_path),
    )
    affect = _read_jsonl(trace_path)[0]["affective_state_summary"]

    assert set(affect) == {
        "emotional_tone",
        "tone_label",
        "conflict_tension",
        "conflict_label",
        "hidden_intent",
        "relationship_depth",
        "prediction_error",
    }
    assert "raw_emotional_speculation" not in affect


def test_conscious_markdown_is_written_under_persona_session_scope(
    tmp_path: Path,
) -> None:
    writer = ConsciousMarkdownWriter(tmp_path / "conscious")
    run_conversation(
        _agent(),
        ["请帮我整理一下这个想法"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=623,
        session_id="session-one",
        persona_id="persona-one",
        conscious_writer=writer,
    )

    conscious_path = (
        tmp_path
        / "conscious"
        / "personas"
        / "persona-one"
        / "sessions"
        / "session-one"
        / "Conscious.md"
    )
    trace_path = conscious_path.with_name("conscious_trace.jsonl")

    assert conscious_path.exists()
    assert trace_path.exists()
    assert not (
        tmp_path
        / "conscious"
        / "personas"
        / "persona-two"
        / "sessions"
        / "session-one"
    ).exists()


def test_conscious_markdown_is_chinese_readable_and_trace_derived(
    tmp_path: Path,
) -> None:
    writer = ConsciousMarkdownWriter(tmp_path / "conscious")
    run_conversation(
        _agent(),
        ["FULL USER PROMPT SHOULD NOT LEAK"],
        observer=DialogueObserver(),
        generator=PromptLeakGenerator(),
        master_seed=624,
        session_id="cn",
        persona_id="persona-cn",
        conscious_writer=writer,
    )
    session_dir = tmp_path / "conscious" / "personas" / "persona-cn" / "sessions" / "cn"
    markdown = (session_dir / "Conscious.md").read_text(encoding="utf-8")
    row = _read_jsonl(session_dir / "conscious_trace.jsonl")[0]

    assert "当前观察" in markdown
    assert "注意与工作空间" in markdown
    assert "候选路径" in markdown
    assert "提示引导" in markdown
    assert str(row["chosen_action"]) in markdown
    assert "FULL USER PROMPT SHOULD NOT LEAK" not in markdown
    assert "FULL SYSTEM PROMPT SHOULD NOT LEAK" not in markdown


def test_default_trace_redacts_raw_prompt_text_and_debug_is_off(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    run_conversation(
        _agent(),
        ["SECRET USER PROMPT SHOULD NOT LEAK"],
        observer=DialogueObserver(),
        generator=PromptLeakGenerator(),
        master_seed=625,
        session_id="redact",
        trace_writer=JsonlTraceWriter(trace_path),
    )
    row = _read_jsonl(trace_path)[0]
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)

    assert "debug" not in row
    assert "SECRET USER PROMPT SHOULD NOT LEAK" not in text
    assert "FULL SYSTEM PROMPT SHOULD NOT LEAK" not in text
    assert "sk-secret-should-not-leak" not in text
    assert row["generation_diagnostics"]["system_prompt"] == "[redacted]"
    assert row["generation_diagnostics"]["user_message"] == "[redacted]"


def test_trace_generation_is_deterministic_without_timestamps(tmp_path: Path) -> None:
    lines = ["hello", "thanks"]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    run_conversation(
        _agent(),
        lines,
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=626,
        session_id="det",
        persona_id="persona-det",
        trace_writer=JsonlTraceWriter(first_path),
    )
    run_conversation(
        _agent(),
        lines,
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=626,
        session_id="det",
        persona_id="persona-det",
        trace_writer=JsonlTraceWriter(second_path),
    )

    assert _read_jsonl(first_path) == _read_jsonl(second_path)


def test_no_trace_run_conversation_behavior_is_unchanged() -> None:
    lines = ["hello, can we check this?", "thanks, that makes sense."]
    observer = DialogueObserver()

    without_trace = run_conversation(
        _agent(),
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=627,
        partner_uid=2,
        session_id="side-effect",
    )

    with_trace_path = Path("artifacts") / "_tmp_m62_side_effect.jsonl"
    with_trace_path.unlink(missing_ok=True)
    with_trace = run_conversation(
        _agent(),
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=627,
        partner_uid=2,
        session_id="side-effect",
        trace_writer=JsonlTraceWriter(with_trace_path),
    )
    with_trace_path.unlink(missing_ok=True)

    assert [turn.action for turn in with_trace] == [
        turn.action for turn in without_trace
    ]
    assert [turn.text for turn in with_trace] == [turn.text for turn in without_trace]
