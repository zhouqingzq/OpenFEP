from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    _call_llm_with_stage_profile,
)
from tests.test_mvp_dialogue_runtime import FakeJSONLLM


class DirectedFastLLM(FakeJSONLLM):
    def __init__(self, fast_payload: dict[str, object] | None = None) -> None:
        super().__init__()
        self.fast_payload = fast_payload or {
            "decision": "reply",
            "reply": "Three priorities: scope, dependencies, and rollback.",
            "reply_action": "answer",
            "reason_codes": ["simple_self_contained_request"],
        }

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "directed fast-reply router" in system_prompt:
            self.calls.append({"system": system_prompt, "user": user_prompt})
            return dict(self.fast_payload)
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def _runtime(tmp_path: Path, llm: FakeJSONLLM) -> MVPDialogueRuntime:
    return MVPDialogueRuntime(
        store=MVPStateStore(tmp_path),
        llm=llm,
        persona_name="hutao",
    )


def _assistant_envelope() -> dict[str, object]:
    return {
        "speaker_participant_id": "alice",
        "visible_participant_ids": ["alice", "telegram:tg:assistant:777"],
        "addressed_participant_ids": ["telegram:tg:assistant:777"],
        "mentioned_participant_ids": ["telegram:tg:assistant:777"],
        "assistant_surface_label": "hutao_bot",
    }


def test_structured_other_addressee_is_zero_llm_fast_silence(tmp_path: Path) -> None:
    llm = DirectedFastLLM()
    runtime = _runtime(tmp_path, llm)

    result = runtime.run_turn(
        "@bob @carol you two decide.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "assistant", "bob", "carol"],
            "addressed_participant_ids": ["bob", "carol"],
            "mentioned_participant_ids": ["bob", "carol"],
        },
    )

    assert result.action == "no_reply"
    assert result.reply == ""
    assert llm.calls == []
    assert result.diagnostics["latency_mode"] == "structured_silence"
    assert result.diagnostics["turn_latency_summary"]["blocking_llm_calls"] == 0
    assert any(
        event.get("type") == "StructuredSilenceFastPathEvent"
        for event in result.diagnostics["bus_messages"]
    )
    assert runtime.store.load()["temporal_state"]["last_user_text"]


def test_reply_to_other_is_structured_silence_not_reply(tmp_path: Path) -> None:
    llm = DirectedFastLLM()
    result = _runtime(tmp_path, llm).run_turn(
        "Following up on Bob's message.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "assistant", "bob"],
            "addressed_participant_ids": ["bob"],
            "reply_to_turn_id": "turn_bob_1",
        },
    )

    assert result.action == "no_reply"
    assert result.diagnostics["group_reply_policy"]["reason_codes"] == [
        "explicit_reply_to_other"
    ]
    assert llm.calls == []


def test_directed_fast_reply_uses_exactly_one_llm_call(tmp_path: Path) -> None:
    llm = DirectedFastLLM()
    result = _runtime(tmp_path, llm).run_turn(
        "@hutao_bot list three release risks.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope=_assistant_envelope(),
    )

    assert result.action == "answer"
    assert result.reply
    assert len(llm.calls) == 1
    assert "directed fast-reply router" in llm.calls[0]["system"]
    assert result.diagnostics["latency_mode"] == "directed_fast_reply"
    assert result.diagnostics["turn_latency_summary"]["blocking_llm_calls"] == 1
    assert any(
        event.get("type") == "DirectedFastReplyEvent"
        for event in result.diagnostics["bus_messages"]
    )


def test_directed_fast_reply_visible_validation_strips_debug_json(tmp_path: Path) -> None:
    llm = DirectedFastLLM(
        {
            "decision": "reply",
            "reply": 'Useful answer. {"diagnostics": "hidden"}',
            "reply_action": "answer",
            "reason_codes": [],
        }
    )
    result = _runtime(tmp_path, llm).run_turn(
        "@hutao_bot answer briefly.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope=_assistant_envelope(),
    )

    assert "diagnostics" not in result.reply
    assert "stripped_debug_payload" in result.diagnostics["reply_validation"]["actions"]
    assert len(llm.calls) == 1


def test_directed_fast_reply_escalation_continues_full_path(tmp_path: Path) -> None:
    llm = DirectedFastLLM(
        {
            "decision": "escalate",
            "reply": "",
            "reply_action": "answer",
            "reason_codes": ["needs_memory"],
        }
    )
    result = _runtime(tmp_path, llm).run_turn(
        "@hutao_bot what budget did Alice mention earlier?",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope=_assistant_envelope(),
    )

    assert len(llm.calls) > 1
    assert result.diagnostics["latency_mode"] == "normal"
    assert "directed_fast_reply_escalated" in result.diagnostics["latency_mode_reasons"]
    assert any(
        event.get("type") == "DirectedFastReplyEscalatedEvent"
        for event in result.diagnostics["bus_messages"]
    )


@pytest.mark.parametrize(
    ("fast_payload", "fast_error"),
    [
        ({"decision": "reply", "reply": "", "reply_action": "answer"}, None),
        ({"decision": "unexpected", "reply": "unsafe", "reply_action": "answer"}, None),
        (None, TimeoutError("directed fast reply timed out")),
        (None, RuntimeError("directed fast reply failed")),
    ],
)
def test_directed_fast_reply_failures_escalate_to_full_path(
    tmp_path: Path,
    fast_payload: dict[str, object] | None,
    fast_error: Exception | None,
) -> None:
    class FailureLLM(DirectedFastLLM):
        def complete_json(
            self, *, system_prompt: str, user_prompt: str
        ) -> dict[str, object]:
            if "directed fast-reply router" in system_prompt:
                self.calls.append({"system": system_prompt, "user": user_prompt})
                if fast_error is not None:
                    raise fast_error
                return dict(fast_payload or {})
            return super().complete_json(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

    llm = FailureLLM()
    result = _runtime(tmp_path, llm).run_turn(
        "@hutao_bot answer this request.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope=_assistant_envelope(),
    )

    assert len(llm.calls) > 1
    assert result.diagnostics["latency_mode"] == "normal"
    assert any(
        event.get("type") == "DirectedFastReplyEscalatedEvent"
        for event in result.diagnostics["bus_messages"]
    )


def test_joint_assistant_and_human_addressees_do_not_use_fast_paths(
    tmp_path: Path,
) -> None:
    llm = DirectedFastLLM()
    result = _runtime(tmp_path, llm).run_turn(
        "Both of you give an opinion.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "assistant", "bob"],
            "addressed_participant_ids": ["assistant", "bob"],
            "assistant_surface_label": "hutao_bot",
        },
    )

    assert result.diagnostics["group_reply_policy"]["action"] == "reply_to_whole_group"
    assert result.diagnostics["latency_mode"] not in {
        "structured_silence",
        "directed_fast_reply",
    }
    assert len(llm.calls) > 1


def test_directed_fast_reply_stage_uses_eight_second_zero_retry_profile() -> None:
    class ProfileLLM:
        timeout_seconds = 35.0
        request_retries = 2
        auxiliary_timeout_seconds = 12.0
        auxiliary_request_retries = 1

        def __init__(self) -> None:
            self.observed: tuple[float, int] | None = None

        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            self.observed = (self.timeout_seconds, self.request_retries)
            return {}

    llm = ProfileLLM()
    _call_llm_with_stage_profile(
        llm,  # type: ignore[arg-type]
        stage="directed_fast_reply",
        system_prompt="s",
        user_prompt="u",
    )
    assert llm.observed == (8.0, 0)
    assert (llm.timeout_seconds, llm.request_retries) == (35.0, 2)


def test_fast_path_turn_log_and_latency_log_match_bus(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, DirectedFastLLM())
    result = runtime.run_turn(
        "@bob decide.",
        speaker_name="Alice",
        turn_index=0,
        now=1000,
        ingress_evidence_band="structured_partial",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "assistant", "bob"],
            "addressed_participant_ids": ["bob"],
        },
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "conversation_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    latency = next(row for row in rows if row.get("event") == "turn_latency")
    turn = next(row for row in rows if row.get("event") == "turn")
    assert latency["blocking_llm_calls"] == 0
    assert latency["latency_mode"] == "structured_silence"
    assert turn["reply"] == result.reply == ""
    assert turn["diagnostics"]["bus_messages"] == result.diagnostics["bus_messages"]
