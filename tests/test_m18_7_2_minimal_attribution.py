"""Tests for M18.7.2 — M18.7 minimal-prompt call site.

M18.7.2 owns a dedicated minimal-prompt LLM call site
for addressee / reaction attribution, decoupled from
the conscious loop. The conscious-loop path is broken
at scale (M18.7.1 real-LLM replay: 0/12 non-empty fills
when the M18.7 v2 attrs segment sat at char 2914 /
37.7% of a 7.7-26k-char prompt). The minimal prompt is
~1.5-2.0k chars and the LLM fills only the M18.7 v1
shape. The result is fed to the same
`state["m18_7_attribution_hypotheses"]` surface that the
M20.4 producer and the M18.7.1 calibration runner read.

The tests are split into two layers:

1. **Pure-function tests** — exercise the prompt builder
   and the orchestrator without any LLM involvement.
2. **Integration tests** — exercise `MVPDialogueRuntime.run_turn`
   with a `FakeJSONLLM` subclass that returns controlled M18.7
   fields on the `"M18.7.2 minimal"` stage marker. The
   conscious-loop path is exercised too (it still runs, but
   no longer requests the M18.7 fields — see
   `test_conscious_loop_prompt_no_longer_requests_m18_7_v2_attrs`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from segmentum.dialogue.runtime import m18_7_1_calibration as cal
from segmentum.dialogue.runtime.m18_7_attribution import (
    M18_7_2_MINIMAL_PROMPT_MAX_CHARS,
    M18_7_2_REASON_FIELD_PRESENT,
    M18_7_2_REASON_MINIMAL_DEGRADED,
    M18_7_2_SOURCE_TAG,
    M18_7_ENGINEERING_PROXY_LABEL,
    M18_7_STATE_SURFACE_CAP,
    build_m18_7_2_addressee_hypothesis_admitted_event,
    build_m18_7_2_minimal_degraded_event,
    build_m18_7_2_reaction_attribution_hypothesis_admitted_event,
    build_m18_7_minimal_prompt,
    emit_m18_7_2_attribution_for_turn,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    build_conscious_loop_prompt,
)
from tests.test_mvp_dialogue_runtime import (
    FakeJSONLLM,
    _maybe_m12_extractor_response,
)


# === Pure-function tests: build_m18_7_minimal_prompt shape ===============


def _default_state() -> dict[str, Any]:
    return {
        "self_basic_facts": {
            "persona_name": "胡桃",
            "do_not_invent": ["不要编造职业"],
        },
    }


def _default_bus() -> list[dict[str, Any]]:
    return [
        {
            "type": "UserUtteranceEvent",
            "turn_index": 7,
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
            "reply_to_turn_id": "turn_5",
            "quoted_turn_ids": [],
            "ingress_evidence_band": "strong",
        },
    ]


def _default_group_turn_binding() -> dict[str, Any]:
    return {
        "speaker_participant_id": "alice",
        "addressed_participant_ids": ["hutao"],
        "mentioned_participant_ids": ["bob"],
        "current_speaker_participant_id": "alice",
        "ambiguity_band": "low",
    }


def _default_entity_binding() -> dict[str, Any]:
    return {
        "current_interlocutor": "alice",
        "aliases": {"桃桃": "hutao"},
    }


def test_build_m18_7_minimal_prompt_length_under_2k_chars() -> None:
    """The combined (system + user) prompt is bounded at
    `M18_7_2_MINIMAL_PROMPT_MAX_CHARS` (v1: 2000; v2: 2500) for
    a representative 11-char Chinese user utterance with the
    full structural payload (entity_binding,
    group_turn_binding, prior turn). The test name retains
    `2k` for back-compat with the v1 path; the actual bound
    is the constant.
    """
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="胡桃,看这个",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="take_addressee_branch",
    )
    total = len(system) + len(user)
    assert total <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS, (
        f"prompt too long: {total} > {M18_7_2_MINIMAL_PROMPT_MAX_CHARS}"
    )


def test_build_m18_7_minimal_prompt_includes_required_signals() -> None:
    """The user prompt must include the structural signals the LLM
    needs: turn_index, speaker, m18_5_decision, entity_binding,
    group_turn_binding, user_text, last_user_utterances, and the
    4-key JSON spec.
    """
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="胡桃你看这个想法怎么样？",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="take_addressee_branch",
    )
    # Persona name from self_basic_facts is exposed in the system prompt.
    assert "胡桃" in system
    # Structural fields appear in the user prompt.
    assert "turn_index: 8" in user
    assert "Alice" in user
    assert "m18_5_structural_decision: take_addressee_branch" in user
    assert "entity_binding" in user
    assert "group_turn_binding" in user
    assert "user_text" in user
    assert "last_user_utterances" in user
    # 4-key JSON spec is present.
    assert "addressee_hypothesis" in user
    assert "reaction_attribution_hypothesis" in user
    assert "reasoning_notes" in user
    assert "_m18_7_2_source" in user
    assert '"m18_7_2_minimal"' in user


def test_build_m18_7_minimal_prompt_excludes_m13_m19_conscious_fields() -> None:
    """The minimal prompt is intentionally decoupled from the
    conscious loop. It must NOT serialize M13 drive state,
    M19 self_expectation_state, pending_expectations, or
    open_items — those fields are ~76% of the conscious-loop
    prompt volume but 100% noise for the attribution decision.
    """
    state = _default_state()
    # Inject the conscious-loop-only fields. They must be
    # ignored by the minimal prompt.
    state["m13_drive_state"] = {"boredom_band": "low"}  # 69k chars in real life
    state["self_expectation_state"] = {"ledger": "huge"}  # 49k chars
    state["pending_expectations"] = [{"id": "p1"}]
    state["open_items"] = [{"id": "o1"}]

    _, user = build_m18_7_minimal_prompt(
        state=state,
        user_text="胡桃你看这个想法怎么样？",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="take_addressee_branch",
    )
    # None of the conscious-loop-only fields should appear.
    assert "m13_drive_state" not in user
    assert "self_expectation_state" not in user
    assert "pending_expectations" not in user
    assert "open_items" not in user
    # Nor should the conscious-loop's "Also include the
    # following M18.7 fields" segment markers.
    assert "Also include" not in user
    assert "addressed_to_assistant" not in user.split("addressee_hypothesis")[0]


def test_build_m18_7_minimal_prompt_handles_missing_group_turn_binding() -> None:
    """When `group_turn_binding` is None, the prompt must still
    build and the field must appear in serialized form (empty
    dict, or "null", or "{}" — anything that doesn't crash and
    stays under the size cap).
    """
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="hello",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=_default_entity_binding(),
        group_turn_binding=None,
        m18_5_structural_decision="",
    )
    assert len(system) + len(user) <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS
    assert "group_turn_binding" in user


def test_build_m18_7_minimal_prompt_handles_missing_entity_binding() -> None:
    """When `entity_binding` is None, the prompt must still build."""
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="hello",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=None,
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="",
    )
    assert len(system) + len(user) <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS
    assert "entity_binding" in user


def test_build_m18_7_minimal_prompt_handles_missing_m18_5_decision() -> None:
    """When `m18_5_structural_decision` is empty, the prompt must
    still build and the field shows the "(none)" placeholder.
    """
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="hello",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=8,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="",
    )
    assert "m18_5_structural_decision: (none)" in user
    assert len(system) + len(user) <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS


def test_build_m18_7_minimal_prompt_handles_empty_state() -> None:
    """When `state` is empty, the system prompt falls back to a
    generic identity line ("数字人格系统的群聊归因助手") and the
    user prompt still builds under the size cap.
    """
    system, user = build_m18_7_minimal_prompt(
        state={},
        user_text="hello",
        speaker_name="",
        bus_messages=None,
        turn_index=0,
        entity_binding=None,
        group_turn_binding=None,
        m18_5_structural_decision="",
    )
    # Generic identity line.
    assert "群聊归因助手" in system
    # Default user name fallback.
    assert "default_user" in user
    # No bus messages → empty list, no crash.
    assert "last_user_utterances" in user
    assert len(system) + len(user) <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS


# === Pure-function tests: emit_m18_7_2_attribution_for_turn ==============


def test_emit_m18_7_2_orchestrator_empty_plan_emits_no_events() -> None:
    """An empty plan (no addressee / reaction) produces no
    M18_7_2_* bus events and no state surface entries.
    """
    bus: list = []
    state: dict = {}
    report = emit_m18_7_2_attribution_for_turn(
        bus=bus,
        state=state,
        plan={
            "addressee_hypothesis": {},
            "reaction_attribution_hypothesis": {},
        },
        turn_index=0,
        at="2026-06-08T00:00:00Z",
    )
    assert report["addressee_event_emitted"] is False
    assert report["reaction_event_emitted"] is False
    assert bus == []
    assert state.get("m18_7_attribution_hypotheses", []) == []
    assert report["source"] == M18_7_2_SOURCE_TAG


def test_emit_m18_7_2_orchestrator_filled_plan_stamps_source_field() -> None:
    """A filled plan emits both M18_7_2_* bus events and writes
    state surface entries with `source: "m18_7_2_minimal"` stamped.
    """
    bus: list = []
    state: dict = {}
    report = emit_m18_7_2_attribution_for_turn(
        bus=bus,
        state=state,
        plan={
            "addressee_hypothesis": {
                "participant_id": "alice",
                "addressed_to_assistant": True,
                "confidence": 0.85,
                "rationale": "directly addresses 胡桃",
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "alice",
                "reaction_to_turn_id": "turn_7",
                "reaction_to_participant_id": "hutao",
                "is_about_assistant_claim": False,
                "confidence": 0.7,
                "rationale": "reacts to a prior reply from 胡桃",
            },
        },
        turn_index=8,
        at="2026-06-08T00:00:08Z",
    )
    assert report["addressee_event_emitted"] is True
    assert report["reaction_event_emitted"] is True
    assert report["source"] == M18_7_2_SOURCE_TAG
    # 2 bus events, both M18_7_2_*
    types = [e["type"] for e in bus]
    assert "M18_7_2_AddresseeHypothesisAdmitted" in types
    assert "M18_7_2_ReactionAttributionHypothesisAdmitted" in types
    # 2 state surface entries, both with source stamped
    surface = state["m18_7_attribution_hypotheses"]
    assert len(surface) == 2
    for entry in surface:
        assert entry["source"] == M18_7_2_SOURCE_TAG


def test_emit_m18_7_2_orchestrator_uses_same_commit_id_as_m18_7() -> None:
    """The commit_id is the SHA-1 of (kind, turn_index,
    source_ref). M18.7.2 uses `source_ref = "m18_7_{kind}_{turn_index}"`
    — the same string the conscious-loop path uses — so commit_id
    values are stable across the two paths.
    """
    bus: list = []
    state: dict = {}
    emit_m18_7_2_attribution_for_turn(
        bus=bus,
        state=state,
        plan={
            "addressee_hypothesis": {
                "participant_id": "alice",
                "addressed_to_assistant": True,
                "confidence": 0.85,
            },
        },
        turn_index=42,
        at="2026-06-08T00:00:42Z",
    )
    entry = state["m18_7_attribution_hypotheses"][0]
    assert entry["commit_id"]
    assert entry["turn_index"] == 42
    assert entry["kind"] == "addressee"
    # The bus event carries the same commit_id.
    addr_event = next(
        e for e in bus if e["type"] == "M18_7_2_AddresseeHypothesisAdmitted"
    )
    assert addr_event["commit_id"] == entry["commit_id"]


# === Pure-function tests: bus event shapes ==============================


def test_build_m18_7_2_addressee_hypothesis_admitted_event_shape() -> None:
    """Defensive shape test for the new M18.7.2 bus event envelope."""
    event = build_m18_7_2_addressee_hypothesis_admitted_event(
        turn_index=8,
        entry={
            "commit_id": "abc123",
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.85,
            "alternative_hypothesis_count": 1,
            "evidence_refs": ["turn_7_user_utterance", "participant_alice"],
        },
        at="2026-06-08T00:00:08Z",
        rationale_chars=24,
    )
    assert event["type"] == "M18_7_2_AddresseeHypothesisAdmitted"
    assert event["turn_index"] == 8
    assert event["commit_id"] == "abc123"
    assert event["participant_id"] == "alice"
    assert event["addressed_to_assistant"] is True
    assert event["confidence"] == 0.85
    assert event["alternative_hypothesis_count"] == 1
    assert event["evidence_ref_count"] == 2
    assert event["rationale_chars"] == 24
    assert event["source"] == M18_7_2_SOURCE_TAG
    assert event["reason_codes"] == [M18_7_2_REASON_FIELD_PRESENT]
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL
    assert event["at"] == "2026-06-08T00:00:08Z"


def test_build_m18_7_2_reaction_attribution_hypothesis_admitted_event_shape() -> None:
    """Defensive shape test for the new reaction event envelope."""
    event = build_m18_7_2_reaction_attribution_hypothesis_admitted_event(
        turn_index=8,
        entry={
            "commit_id": "def456",
            "participant_id": "alice",
            "reaction_to_turn_id": "turn_7",
            "reaction_to_participant_id": "hutao",
            "is_about_assistant_claim": False,
            "confidence": 0.7,
            "alternative_attribution_count": 0,
            "evidence_refs": ["turn_7_reply_to_turn_id"],
        },
        at="2026-06-08T00:00:08Z",
    )
    assert event["type"] == "M18_7_2_ReactionAttributionHypothesisAdmitted"
    assert event["turn_index"] == 8
    assert event["commit_id"] == "def456"
    assert event["reaction_to_turn_id"] == "turn_7"
    assert event["reaction_to_participant_id"] == "hutao"
    assert event["is_about_assistant_claim"] is False
    assert event["confidence"] == 0.7
    assert event["alternative_attribution_count"] == 0
    assert event["evidence_ref_count"] == 1
    assert event["source"] == M18_7_2_SOURCE_TAG
    assert event["reason_codes"] == [M18_7_2_REASON_FIELD_PRESENT]
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL


def test_build_m18_7_2_minimal_degraded_event_shape() -> None:
    """Defensive shape test for the degraded fallback event."""
    event = build_m18_7_2_minimal_degraded_event(
        turn_index=8,
        reason="TimeoutError('m18_7_2_minimal LLM call timed out')",
        at="2026-06-08T00:00:08Z",
    )
    assert event["type"] == "M18_7_2_MinimalDegraded"
    assert event["turn_index"] == 8
    assert "TimeoutError" in event["reason"]
    assert event["reason_code"] == M18_7_2_REASON_MINIMAL_DEGRADED
    assert event["source"] == M18_7_2_SOURCE_TAG
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL
    assert event["at"] == "2026-06-08T00:00:08Z"


# === Integration tests: FakeJSONLLM with M18.7.2 stage marker ===========


class _M187_2FakeLLM(FakeJSONLLM):
    """FakeJSONLLM subclass that returns controlled M18.7 v2 attrs
    on the `"M18.7.2 minimal"` stage marker (the substring used by
    `build_m18_7_minimal_prompt`'s system prompt) and falls back to
    the parent class for the conscious loop. The `_responses_by_turn`
    dict lets each test program per-turn payloads.
    """

    def __init__(self) -> None:
        super().__init__()
        self._responses_by_turn: dict[int, dict[str, object]] = {}
        self._force_failure_on_turn: dict[int, Exception] = {}
        self._m18_7_2_call_count = 0

    def complete_json(
        self, *, system_prompt: str, user_prompt: str
    ) -> dict[str, object]:
        # The M18.7.2 system prompt is identifiable by the
        # "群聊归因助手" substring.
        if "群聊归因助手" in system_prompt:
            self._m18_7_2_call_count += 1
            # Extract the turn_index from the user prompt header
            # ("turn_index: <N>") so per-turn programming works.
            turn_index = self._extract_turn_index(user_prompt)
            if turn_index in self._force_failure_on_turn:
                raise self._force_failure_on_turn[turn_index]
            if turn_index in self._responses_by_turn:
                payload = self._responses_by_turn[turn_index]
                return {
                    "addressee_hypothesis": payload.get(
                        "addressee_hypothesis", {}
                    ),
                    "reaction_attribution_hypothesis": payload.get(
                        "reaction_attribution_hypothesis", {}
                    ),
                    "reasoning_notes": payload.get("reasoning_notes", "test"),
                    "_m18_7_2_source": "m18_7_2_minimal",
                }
            # Default: empty M18.7 fields (no LLM fill).
            return {
                "addressee_hypothesis": {},
                "reaction_attribution_hypothesis": {},
                "reasoning_notes": "",
                "_m18_7_2_source": "m18_7_2_minimal",
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

    @staticmethod
    def _extract_turn_index(user_prompt: str) -> int:
        # The minimal prompt format starts with "turn_index: <N>".
        for line in user_prompt.splitlines():
            if line.startswith("turn_index:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    return -1
        return -1


def _runtime(tmp_path: Path) -> MVPDialogueRuntime:
    return MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=_M187_2FakeLLM(),
        persona_name="胡桃",
    )


def _group_envelope(turn_index: int) -> dict[str, Any]:
    """Build a non-empty `group_turn_envelope` so `bounded_group_turn`
    is truthy and the M18.7.2 call site fires.
    """
    return {
        "speaker_participant_id": "alice",
        "visible_participant_ids": ["alice", "bob", "hutao"],
        "addressed_participant_ids": ["hutao"],
        "mentioned_participant_ids": ["bob"],
        "reply_to_turn_id": f"turn_{turn_index - 1}",
        "quoted_turn_ids": [],
        "explicit_mentions": ["胡桃"],
    }


def test_run_turn_calls_m18_7_2_minimal_stage_with_minimal_prompt(
    tmp_path: Path,
) -> None:
    """`run_turn` invokes the `"m18_7_2_minimal"` stage when
    `bounded_group_turn` is truthy. The fake LLM's
    `_m18_7_2_call_count` confirms the call happened.
    """
    llm = _M187_2FakeLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    assert llm._m18_7_2_call_count == 1


def test_run_turn_writes_m18_7_2_fill_to_state_surface(tmp_path: Path) -> None:
    """When the minimal LLM call returns a non-empty
    `addressee_hypothesis`, the in-memory state surface
    `state["m18_7_attribution_hypotheses"]` gets a new entry with
    `source: "m18_7_2_minimal"` stamped. (The surface is in-memory
    only — `MVPStateStore` does not persist it; the M20.4 producer
    and M18.7.1 calibration runner both read the in-memory state
    during the same `run_turn` call.)

    The M18.7.1 calibration harness is used as the assertion
    scaffold because it iterates `run_turn` over a fixture and
    reads the in-memory surface after each turn. The presence
    of an `M18_7_2_AddresseeHypothesisAdmitted` bus event with
    the M18.7.2 source stamp is the surface-write side effect.
    """
    llm = _M187_2FakeLLM()
    llm._responses_by_turn[0] = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.85,
            "rationale": "directly addresses 胡桃",
        },
        "reaction_attribution_hypothesis": {},
    }
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    result = runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    # The orchestrator writes to the in-memory surface AND emits
    # the addressee admitted event. The event is the observable
    # side effect that proves the surface was written.
    addr_events = [
        e for e in result.diagnostics.get("bus_messages", [])
        if e.get("type") == "M18_7_2_AddresseeHypothesisAdmitted"
    ]
    assert len(addr_events) == 1
    event = addr_events[0]
    assert event["source"] == M18_7_2_SOURCE_TAG
    assert event["commit_id"]
    assert event["participant_id"] == "alice"
    assert event["addressed_to_assistant"] is True
    assert event["confidence"] == 0.85


def test_run_turn_m18_7_2_failure_falls_back_to_empty_without_crashing(
    tmp_path: Path,
) -> None:
    """When the M18.7.2 LLM call raises, `run_turn` does NOT
    crash. It falls back to empty `{}` for both fields and emits
    a degraded bus event.
    """
    llm = _M187_2FakeLLM()
    llm._force_failure_on_turn[0] = RuntimeError("simulated LLM timeout")
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    # run_turn must not raise.
    result = runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    # The reply is still produced.
    assert result.reply
    # The state surface is empty (no M18.7 fill).
    state = runtime.store.load()
    assert state.get("m18_7_attribution_hypotheses", []) == []
    # A degraded bus event is emitted.
    bus_types = {e.get("type") for e in result.diagnostics.get("bus_messages", [])}
    assert "M18_7_2_MinimalDegraded" in bus_types


def test_run_turn_m18_7_2_emits_degraded_event_on_failure(
    tmp_path: Path,
) -> None:
    """The degraded bus event carries the failure reason string
    and the M18_7_2_REASON_MINIMAL_DEGRADED reason_code.
    """
    llm = _M187_2FakeLLM()
    llm._force_failure_on_turn[0] = TimeoutError("m18_7_2_minimal timed out")
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    result = runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    degraded_events = [
        e for e in result.diagnostics.get("bus_messages", [])
        if e.get("type") == "M18_7_2_MinimalDegraded"
    ]
    assert len(degraded_events) == 1
    event = degraded_events[0]
    assert event["turn_index"] == 0
    assert "timed out" in event["reason"]
    assert event["reason_code"] == M18_7_2_REASON_MINIMAL_DEGRADED
    assert event["source"] == M18_7_2_SOURCE_TAG


def test_run_turn_no_group_envelope_skips_m18_7_2_call(tmp_path: Path) -> None:
    """When `group_turn_envelope` is None, `bounded_group_turn` is
    empty and the M18.7.2 LLM call is skipped entirely. The state
    surface is empty.
    """
    llm = _M187_2FakeLLM()
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    runtime.run_turn(
        "hello world",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=None,
        now=1000,
    )
    assert llm._m18_7_2_call_count == 0
    state = runtime.store.load()
    assert state.get("m18_7_attribution_hypotheses", []) == []


def test_run_turn_m20_4_producer_sees_m18_7_2_fill(tmp_path: Path) -> None:
    """The M20.4 producer reads
    `state["m18_7_attribution_hypotheses"]` (in-memory) and admits
    `ActiveCommitment` rows. When the M18.7.2 minimal call
    populates the surface with a high-confidence addressee fill,
    the M20.4 producer should observe it and emit
    `AddresseeTargetMatchAdmitted` (or similar M20.4-owned event).
    """
    llm = _M187_2FakeLLM()
    # High-confidence addressee fill that clears the
    # M20.4 threshold (0.4 admit min, 0.85 tie-breaker).
    llm._responses_by_turn[0] = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "rationale": "directly addresses 胡桃",
            "evidence_refs": ["turn_0_user_utterance"],
        },
        "reaction_attribution_hypothesis": {},
    }
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    result = runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    bus_types = {e.get("type") for e in result.diagnostics.get("bus_messages", [])}
    # The M18.7.2 orchestrator emitted its addressee admitted
    # event. The M20.4 producer observed the in-memory surface
    # and emitted the M20.4-owned downstream event.
    assert "M18_7_2_AddresseeHypothesisAdmitted" in bus_types
    # The M20.4 producer is downstream of the M18.7.2 orchestrator;
    # it admits the commitment as an `AddresseeTargetMatchAdmitted`
    # or `ActiveCommitment*` event when confidence is high enough.
    m20_4_emitted = (
        "AddresseeTargetMatchAdmitted" in bus_types
        or any("ActiveCommitment" in t for t in bus_types)
    )
    assert m20_4_emitted, (
        f"expected M20.4 producer to emit AddresseeTargetMatchAdmitted / "
        f"ActiveCommitment* events; bus_types={sorted(bus_types)}"
    )


def test_run_turn_m18_7_2_true_overrides_wrong_clarify_same_turn(
    tmp_path: Path,
) -> None:
    """The live same-turn gate consumes the M18.7.2 minimal result."""
    llm = _M187_2FakeLLM()
    llm._responses_by_turn[0] = {
        "addressee_hypothesis": {
            "participant_id": "assistant",
            "addressed_to_assistant": True,
            "confidence": 0.8,
            "rationale": "the current speaker is asking the assistant",
        },
        "reaction_attribution_hypothesis": {},
    }
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "same_turn_override"),
        llm=llm,
        persona_name="hutao",
    )
    result = runtime.run_turn(
        "Can you reply to that one?",
        turn_index=0,
        speaker_name="Carol",
        group_turn_envelope={
            "speaker_participant_id": "carol",
            "visible_participant_ids": ["carol", "dave", "hutao"],
            "mentioned_participant_ids": ["carol", "dave"],
            "explicit_mentions": ["Carol", "Dave"],
        },
        now=1000,
    )
    bus = result.diagnostics.get("bus_messages", [])
    gate_events = [
        event
        for event in bus
        if event.get("type") == "SameTurnAddresseeHypothesisGateVerdict"
    ]
    assert gate_events
    assert gate_events[-1]["m20_4_1_audit_only"] is False
    assert result.action == "reply_to_current_speaker"
    assert result.reply


def test_run_turn_explicit_other_recipient_never_overrides(
    tmp_path: Path,
) -> None:
    """Structured explicit-recipient evidence blocks a false True."""
    llm = _M187_2FakeLLM()
    llm._responses_by_turn[0] = {
        "addressee_hypothesis": {
            "participant_id": "assistant",
            "addressed_to_assistant": True,
            "confidence": 1.0,
            "rationale": "incorrect test hypothesis",
        },
        "reaction_attribution_hypothesis": {},
    }
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "explicit_other_no_override"),
        llm=llm,
        persona_name="hutao",
    )
    result = runtime.run_turn(
        "Dave, can you reply to that one?",
        turn_index=0,
        speaker_name="Carol",
        group_turn_envelope={
            "speaker_participant_id": "carol",
            "visible_participant_ids": ["carol", "dave", "hutao"],
            "addressed_participant_ids": ["dave"],
            "mentioned_participant_ids": ["dave"],
            "explicit_mentions": ["Dave"],
        },
        now=1000,
    )
    bus = result.diagnostics.get("bus_messages", [])
    gate_events = [
        event
        for event in bus
        if event.get("type") == "SameTurnAddresseeHypothesisGateVerdict"
    ]
    assert not gate_events
    assert result.action != "reply_to_current_speaker"


def test_run_turn_calibration_runner_sees_m18_7_2_fill(tmp_path: Path) -> None:
    """The M18.7.1 calibration runner reads
    `state["m18_7_attribution_hypotheses"]`. After `run_turn`
    populates the surface via the M18.7.2 minimal call, the
    runner sees the fill end-to-end. This is verified by
    running the calibration harness and asserting
    `n_present > 0` for both fields.
    """
    fixture_path = Path("tests/fixtures/m18_7_1_held_out_calibration.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    runtime = _runtime(tmp_path)
    # Program the M18.7.2 minimal fake to return non-empty fills
    # for every fixture turn. The runner iterates the fixture
    # and calls `run_turn` per turn; turn_index is the iteration
    # index (0..len(fixture)-1).
    for idx, step in enumerate(fixture):
        runtime.llm._responses_by_turn[idx] = {
            "addressee_hypothesis": {
                "participant_id": "alice",
                "addressed_to_assistant": bool(
                    step["ground_truth"].get("addressed_to_assistant", False)
                ),
                "confidence": 0.7,
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "alice",
                "reaction_to_turn_id": str(
                    step["ground_truth"].get("reaction_to_turn_id", "") or ""
                ),
                "reaction_to_participant_id": "alice",
                "is_about_assistant_claim": False,
                "confidence": 0.6,
            },
        }
    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
        fixture_name=str(fixture_path),
    )
    # The M18.7.2 minimal fills are visible to the runner.
    assert report.addressee.n_present > 0
    assert report.reaction.n_present > 0


def test_run_turn_no_double_write_when_conscious_loop_runs(
    tmp_path: Path,
) -> None:
    """The conscious loop no longer requests M18.7 v2 attrs. The
    state surface is written by the M18.7.2 minimal call only.
    A single `run_turn` with both addressee and reaction M18.7.2
    fills must produce exactly two `M18_7_2_*` admitted events
    in the bus (not four — the conscious-loop path no longer
    contributes).
    """
    llm = _M187_2FakeLLM()
    llm._responses_by_turn[0] = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.85,
        },
        "reaction_attribution_hypothesis": {
            "participant_id": "alice",
            "reaction_to_turn_id": "",
            "reaction_to_participant_id": "hutao",
            "is_about_assistant_claim": False,
            "confidence": 0.7,
        },
    }
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=llm,
        persona_name="胡桃",
    )
    result = runtime.run_turn(
        "胡桃你看这个想法怎么样？",
        turn_index=0,
        speaker_name="Alice",
        group_turn_envelope=_group_envelope(0),
        now=1000,
    )
    bus_events = result.diagnostics.get("bus_messages", [])
    addressee_events = [
        e for e in bus_events
        if e.get("type") == "M18_7_2_AddresseeHypothesisAdmitted"
    ]
    reaction_events = [
        e for e in bus_events
        if e.get("type") == "M18_7_2_ReactionAttributionHypothesisAdmitted"
    ]
    # Exactly one of each — the M18.7.2 minimal call is the
    # sole source. The conscious-loop path no longer contributes.
    assert len(addressee_events) == 1
    assert len(reaction_events) == 1
    # Both have the M18.7.2 source stamp.
    for event in addressee_events + reaction_events:
        assert event["source"] == M18_7_2_SOURCE_TAG


def test_conscious_loop_prompt_no_longer_requests_m18_7_v2_attrs() -> None:
    """The M18.7 v2 attrs segment has been removed from
    `build_conscious_loop_prompt`. The conscious loop LLM no
    longer sees the `addressee_hypothesis` /
    `reaction_attribution_hypothesis` JSON schema spec.
    """
    state: dict = {
        "self_basic_facts": {
            "persona_name": "胡桃",
        },
        "conversation_log": [],
        "temporal_state": {},
    }
    system, user = build_conscious_loop_prompt(
        state=state,
        user_text="hello",
        speaker_name="Alice",
        bus_messages=[],
        turn_index=0,
        temporal_input={"now": 1000, "last_user_text": ""},
        entity_binding=None,
    )
    # The conscious-loop prompt must NOT include the M18.7
    # v2 attrs segment. The minimal prompt is the sole source.
    user_lower = user.lower()
    assert "addressee_hypothesis" not in user
    assert "reaction_attribution_hypothesis" not in user
    # The marker phrase from the removed segment must be gone.
    assert "Also include the following M18.7 fields" not in user
    # The minimal-prompt identifier must not leak into the
    # conscious-loop prompt either.
    assert "m18_7_2_minimal" not in user_lower
    # The system prompt is unchanged structurally; it doesn't
    # carry the v2 attrs spec.
    assert "m18_7_2_minimal" not in system.lower()


# === State surface rolling window ========================================


def test_state_surface_cap_holds_for_m18_7_2_minimal(tmp_path: Path) -> None:
    """After many M18.7.2 fills, the state surface rolling window
    must stay bounded at `M18_7_STATE_SURFACE_CAP` (8) — the
    same cap the conscious-loop path uses.
    """
    state: dict = {}
    for turn_index in range(12):
        emit_m18_7_2_attribution_for_turn(
            bus=[],
            state=state,
            plan={
                "addressee_hypothesis": {
                    "participant_id": f"alice_{turn_index}",
                    "addressed_to_assistant": True,
                    "confidence": 0.7,
                },
                "reaction_attribution_hypothesis": {},
            },
            turn_index=turn_index,
            at=f"2026-06-08T00:00:{turn_index:02d}Z",
        )
    surface = state["m18_7_attribution_hypotheses"]
    assert len(surface) == M18_7_STATE_SURFACE_CAP
    # The cap drops the tail; the latest entry is the last turn.
    assert surface[-1]["turn_index"] == 11


# === M18.7.2 v2 prompt revision tests (P0-7 follow-up) ====================
# v2 (2026-06-10) — system_prompt revised to lift the LLM's
# `addressed_to_assistant` default-to-False bias. P0-7 5-run
# stability (commit b030fca) showed recall_on_addressed=0.0-0.25
# in 5/5 runs. The v2 prompt adds a strong-signal list, a
# counter-example list, and 3 inline examples. The v1 prompt
# told the LLM "判断两件事" (judge two things) without telling
# it what strong evidence to consult. v2 closes that gap.


def test_v2_prompt_enumerates_addressed_true_strong_signals() -> None:
    """The v2 system_prompt must enumerate the strong signals the
    LLM should consult for `addressed_to_assistant=True`:

    - bot alias in mentioned/addressed_participant_ids
    - entity_binding.current_interlocutor = bot
    - second-person imperative ('can you' / 'could you' / 'do you')
    - 'OK' / '好的' continuation + bot directive
    - implicit directive ('Someone is reading this' style)

    The 5 signals are the v2 design's answer to P0-7's
    `recall_on_addressed=0.25` finding.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="Can you reply?",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    # Strong-signal list (5 items).
    assert "addressed_to_assistant=True" in system
    assert "bot alias" in system or "mentioned/addressed" in system
    assert "entity_binding.current_interlocutor" in system
    assert "can you" in system
    assert "OK" in system and "好的" in system
    assert "隐含" in system or "隐含指令" in system
    # Counter-example list (≥ 1 item).
    assert "addressed_to_assistant=False" in system
    assert "Dave" in system  # counter-example target


def test_v2_prompt_has_three_inline_examples() -> None:
    """The v2 system_prompt must include 3 inline examples of
    `addressed_to_assistant` decisions: one True, one False, one
    True (OK + can you). Examples are generic (no fixture
    content) so the prompt does not leak the held-out fixture.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    assert "'Can you explain that?'" in system
    assert "True" in system
    assert "'Dave, you first.'" in system
    assert "False" in system
    assert "'OK, can you do X?'" in system


def test_v2_prompt_does_not_leak_fixture_text() -> None:
    """The v2 system_prompt examples must be GENERIC. The held-out
    fixture text ('Can you reply to that one', 'OK, can you go
    back to the part about Eve's note', 'Someone from the team
    is reading this', 'Actually, my previous question still
    stands') must NOT appear in the prompt — that would
    constitute GT-leak and invalidate the calibration.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    # The fixture is tests/fixtures/m18_7_1_held_out_calibration.json
    # These phrases are in the fixture GT; verify NONE leak:
    fixture_phrases = [
        "Can you reply to that one",  # turn 0
        "Actually, my previous question",  # turn 1
        "Someone from the team is reading this",  # turn 4
        "Eve's note",  # turn 8
    ]
    for phrase in fixture_phrases:
        assert phrase not in system, (
            f"fixture phrase leaked into v2 prompt: {phrase!r}"
        )


def test_v2_prompt_emphasizes_entity_binding_and_mentioned_ids() -> None:
    """The v2 system_prompt must explicitly tell the LLM to use
    `entity_binding` and `mentioned_participant_ids` as evidence
    (not raw user text / keyword cues). This is the v2 design's
    response to P0-7's `recall_on_addressed` finding: the LLM
    was not told to read these fields, so it defaulted to False.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    # Both fields must be named in the strong-signal list or
    # in the closing evidence line.
    assert "mentioned" in system or "addressed_participant_ids" in system
    assert "entity_binding" in system
    # Closing line must still be present (preserved from v1).
    assert "不要用关键词或正则做判断" in system or "不要用" in system
    assert "5-key JSON" in system or "4-key JSON" in system


def test_v2_prompt_max_chars_bumped_to_2500() -> None:
    """The M18.7.2 v2 MAX bump (2000 → 2500) is the v2 design's
    nominal budget for the addressed-axis strong-signal /
    counter-example list. v1 nominal was 1647; v2 nominal is
    ~2277. The constant lives at
    `M18_7_2_MINIMAL_PROMPT_MAX_CHARS` and is referenced by
    3 tests; v1 docs said 2000, v2 docs say 2500.

    v3 bump (2026-06-11): MAX 2500 → 2600 to fit the
    default-to-True rule + re-engaging signal + 2 more
    inline examples. v1/v2 content preserved (existing
    v2 tests still pass).
    """
    assert M18_7_2_MINIMAL_PROMPT_MAX_CHARS == 2600, (
        "v3 bumped MAX from 2500 to 2600 to fit the v3 "
        "default-to-True rule + re-engaging signal + 2 more "
        "inline examples (v1/v2 content preserved)"
    )
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    total = len(system) + len(user)
    assert total <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS, (
        f"v2 prompt too long: {total} > "
        f"{M18_7_2_MINIMAL_PROMPT_MAX_CHARS}"
    )


# =====================================================================
# M18.7.2 v3 (2026-06-11) — default-to-True nudge tests
#
# v3 adds 3 things to the system_prompt:
#   1. Strong-signal item 6: re-engaging ("still waiting" /
#      "are you there?"). Targeted at deepseek's polite-request
#      pattern where the user follows up without explicit @bot.
#   2. "默认倾向于 True" rule for mixed/ambiguous signals.
#      Rationale: bot 漏报 (recall 0) is worse than bot 误报
#      (precision 0) for the M20.4 producer — admits feed
#      context that the conscious loop can reject downstream,
#      but missed admits erase evidence.
#   3. Two more inline examples ("Still waiting for an answer."
#      → True (re-engaging); "Anyone want to take this?" → False
#      (group-wide)).
#
# v1/v2 content is preserved (28 existing v2 tests still pass).
# The MAX was bumped 2500 → 2600 to fit the additions; the
# v3 test asserts the constant value + the v3 strings are
# present + v1/v2 strings are preserved + no fixture text
# leaks into the prompt.
# =====================================================================


def test_v3_prompt_has_default_to_true_rule() -> None:
    """v3 system_prompt must contain the explicit
    '默认倾向于 True' / 'bot 漏报 > 误报' rule. This is the
    core v3 nudge — it tells the LLM to default to True on
    mixed signals rather than default to False (which is the
    v2 emit pattern that gave 4.8% True rate in the M20.4 v2
    bundle 5-run, commit dce23f0).
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    assert "默认倾向于 True" in system, (
        "v3 must include the default-to-True rule"
    )
    assert "bot 漏报 > 误报" in system, (
        "v3 must include the bot-漏报-代价-更高 rationale"
    )


def test_v3_prompt_has_re_engaging_strong_signal() -> None:
    """v3 strong-signal item 6 is the re-engaging pattern
    ('still waiting' / 'are you there?'). This is the
    deepseek-typical polite-request / follow-up pattern
    that v2 prompt did not enumerate.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    assert "重新接回 bot" in system, (
        "v3 must include the re-engaging strong-signal label"
    )
    assert "still waiting" in system, (
        "v3 must include the 'still waiting' re-engaging example"
    )
    assert "are you there" in system, (
        "v3 must include the 'are you there' re-engaging example"
    )


def test_v3_prompt_has_two_more_inline_examples() -> None:
    """v3 adds 2 inline examples: 'Still waiting for an answer.'
    (True, re-engaging) and 'Anyone want to take this?'
    (False, group-wide). The v2 3 inline examples are
    preserved (asserted in the next test).
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    assert "Still waiting for an answer." in system, (
        "v3 must include the 'Still waiting for an answer.' example"
    )
    assert "Anyone want to take this?" in system, (
        "v3 must include the 'Anyone want to take this?' example"
    )
    assert "re-engaging" in system, (
        "v3 re-engaging example must be labeled as such"
    )
    assert "group-wide" in system, (
        "v3 group-wide example must be labeled as such"
    )


def test_v3_prompt_preserves_v2_content() -> None:
    """v3 must NOT remove any v2 content. The v2 strong-
    signal list (5 items), counter-example list (2 items),
    and 3 inline examples must all still be present. This
    is the "v1 byte-identity preserved at content level" test
    for v3.
    """
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    # v2 strong-signal items 1-5
    assert "@bot" in system
    assert "entity_binding.current_interlocutor" in system
    assert "第二人称祈使句" in system
    assert "'OK' / '好的'" in system
    assert "Someone is reading this" in system
    # v2 counter-examples
    assert "'Dave, you first'" in system
    assert "'大家怎么看'" in system
    # v2 inline examples
    assert "'Can you explain that?'" in system
    assert "'OK, can you do X?'" in system
    # v2 emphatic framing on strong-signal list
    assert "命中任一即倾向 True" in system, (
        "v2 emphatic framing must be preserved in v3"
    )


def test_v3_prompt_does_not_leak_fixture_text() -> None:
    """v3 prompt must not contain any of the 4 fixture GT
    phrases (the v2 leak test, preserved). v3's new
    re-engaging example uses generic phrases ('still waiting'
    / 'are you there' / 'Still waiting for an answer.'), NOT
    the fixture's GT phrases.
    """
    fixture_gt_phrases = [
        "Can you reply to that one",  # turn 0 GT True
        "Actually, my previous question",  # turn 1 GT True
        "Someone from the team is reading this",  # turn 4 GT True
        "Eve's note",  # turn 8 GT True
    ]
    system, _ = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    for phrase in fixture_gt_phrases:
        assert phrase not in system, (
            f"v3 prompt must not leak fixture GT phrase: {phrase!r}"
        )


def test_v3_prompt_length_under_2600() -> None:
    """v3 MAX bump: 2500 → 2600. v3 prompt is ~2534 chars
    (vs v2 ~2277, vs v1 ~1647). Stays under 2600 to keep
    the M18.7.2 minimal-prompt design goal (vs the
    7.7-26k conscious-loop prompt).
    """
    system, user = build_m18_7_minimal_prompt(
        state=_default_state(),
        user_text="x",
        speaker_name="Alice",
        bus_messages=_default_bus(),
        turn_index=0,
        entity_binding=_default_entity_binding(),
        group_turn_binding=_default_group_turn_binding(),
        m18_5_structural_decision="reply",
    )
    total = len(system) + len(user)
    assert total <= M18_7_2_MINIMAL_PROMPT_MAX_CHARS, (
        f"v3 prompt too long: {total} > "
        f"{M18_7_2_MINIMAL_PROMPT_MAX_CHARS}"
    )
    assert total >= 2400, (
        f"v3 prompt should be ~2534 chars (v2 was ~2277); "
        f"got {total}, which is suspicious"
    )
