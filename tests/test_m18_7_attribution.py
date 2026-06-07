"""Tests for M18.7 §1–§5 (conscious-loop v2 attribution fields).

M18.7 freezes the schema for `addressee_hypothesis` and
`reaction_attribution_hypothesis`, the normalize contract,
the bus event envelopes, the bounded state surface, the
commit_id derivation, and the fast-chat skip event. These
tests cover the engineering layer; prompt-template and
integration tests live in `test_m18_7_prompt_template.py`
and `test_m18_7_integration.py`.
"""

from __future__ import annotations

from segmentum.dialogue.runtime.m18_7_attribution import (
    ALLOWED_M18_7_KIND,
    KIND_ADDRESSEE,
    KIND_REACTION,
    M18_7_ENGINEERING_PROXY_LABEL,
    M18_7_STATE_SURFACE_CAP,
    REASON_FIELD_PRESENT,
    REASON_FIELD_SKIPPED_FAST_CHAT,
    build_addressee_hypothesis_admitted_event,
    build_attribution_hypothesis_skipped_event,
    build_reaction_attribution_hypothesis_admitted_event,
    build_state_entry,
    compute_m18_7_commit_id,
    emit_m18_7_attribution_for_turn,
    normalize_addressee_hypothesis,
    normalize_reaction_attribution_hypothesis,
    record_m18_7_attribution_hypotheses,
    should_emit_attribution_hypothesis_skipped,
)


# === Schema and normalize: addressee_hypothesis =========================


def test_addressee_hypothesis_field_defaults_to_empty_when_missing() -> None:
    assert normalize_addressee_hypothesis(None) == {}
    assert normalize_addressee_hypothesis({}) == {}


def test_addressee_hypothesis_field_defaults_to_empty_on_malformed_payload() -> None:
    assert normalize_addressee_hypothesis("not a mapping") == {}
    assert normalize_addressee_hypothesis(42) == {}


def test_addressee_hypothesis_confidence_is_clamped_to_unit_interval() -> None:
    h = normalize_addressee_hypothesis({"confidence": 1.5})
    assert h["confidence"] == 1.0
    h = normalize_addressee_hypothesis({"confidence": -0.3})
    assert h["confidence"] == 0.0
    h = normalize_addressee_hypothesis({"confidence": 0.7})
    assert h["confidence"] == 0.7


def test_addressee_hypothesis_rationale_is_truncated_to_200_chars() -> None:
    h = normalize_addressee_hypothesis({"rationale": "x" * 500})
    assert len(h["rationale"]) == 200


def test_addressee_hypothesis_participant_id_is_validated_against_m18_1() -> None:
    # Valid: non-empty letters / digits / _-.@:
    h = normalize_addressee_hypothesis({"participant_id": "alice_42"})
    assert h["participant_id"] == "alice_42"
    h = normalize_addressee_hypothesis({"participant_id": "bob@host"})
    assert h["participant_id"] == "bob@host"
    # Invalid: empty / non-shape → empty string (silent default)
    h = normalize_addressee_hypothesis({"participant_id": ""})
    assert h["participant_id"] == ""
    h = normalize_addressee_hypothesis({"participant_id": "has spaces"})
    assert h["participant_id"] == ""
    h = normalize_addressee_hypothesis({"participant_id": 12345})
    assert h["participant_id"] == ""


def test_addressee_hypothesis_evidence_refs_are_bounded_handles() -> None:
    h = normalize_addressee_hypothesis(
        {
            "evidence_refs": [
                "turn_42_user_utterance",
                "turn_42_reply_to_turn_id",
                "bus_event_abc-123",
                "participant_alice",
                # Dropped: not a string
                123,
                # Dropped: empty
                "",
                # Dropped: too long
                "x" * 200,
            ]
        }
    )
    assert h["evidence_refs"] == [
        "turn_42_user_utterance",
        "turn_42_reply_to_turn_id",
        "bus_event_abc-123",
        "participant_alice",
    ]


def test_addressee_hypothesis_evidence_refs_reject_raw_text() -> None:
    """Raw user text in evidence_refs is rejected; only bounded
    handles pass.
    """
    h = normalize_addressee_hypothesis(
        {
            "evidence_refs": [
                "the user said 嗯",  # raw text → rejected
                "turn_42",  # valid handle
            ]
        }
    )
    assert h["evidence_refs"] == ["turn_42"]


def test_addressee_hypothesis_alternative_hypotheses_capped_at_2() -> None:
    h = normalize_addressee_hypothesis(
        {
            "alternative_hypotheses": [
                {"addressed_to_assistant": True, "confidence": 0.3},
                {"addressed_to_assistant": False, "confidence": 0.4},
                {"addressed_to_assistant": True, "confidence": 0.5},
            ]
        }
    )
    assert len(h["alternative_hypotheses"]) == 2


def test_addressee_hypothesis_full_shape_normalizes_correctly() -> None:
    raw = {
        "participant_id": "alice",
        "addressed_to_assistant": True,
        "confidence": 0.85,
        "rationale": "Alice is replying to the assistant's prior turn",
        "evidence_refs": ["turn_42_user_utterance", "turn_41_assistant_reply"],
        "alternative_hypotheses": [
            {"addressed_to_assistant": False, "confidence": 0.15,
             "rationale": "alternative", "evidence_refs": []},
        ],
    }
    h = normalize_addressee_hypothesis(raw)
    assert h["participant_id"] == "alice"
    assert h["addressed_to_assistant"] is True
    assert h["confidence"] == 0.85
    assert h["rationale"] == "Alice is replying to the assistant's prior turn"
    assert h["evidence_refs"] == ["turn_42_user_utterance", "turn_41_assistant_reply"]
    assert len(h["alternative_hypotheses"]) == 1


# === Schema and normalize: reaction_attribution_hypothesis ============


def test_reaction_attribution_hypothesis_field_defaults_to_empty_when_missing() -> None:
    assert normalize_reaction_attribution_hypothesis(None) == {}
    assert normalize_reaction_attribution_hypothesis({}) == {}


def test_reaction_attribution_hypothesis_confidence_is_clamped_to_unit_interval() -> None:
    h = normalize_reaction_attribution_hypothesis({"confidence": 2.0})
    assert h["confidence"] == 1.0
    h = normalize_reaction_attribution_hypothesis({"confidence": -1.0})
    assert h["confidence"] == 0.0


def test_reaction_attribution_hypothesis_alternative_attributions_capped_at_2() -> None:
    h = normalize_reaction_attribution_hypothesis(
        {
            "alternative_attributions": [
                {"reaction_to_turn_id": "turn_40", "confidence": 0.3},
                {"reaction_to_turn_id": "turn_41", "confidence": 0.4},
                {"reaction_to_turn_id": "turn_42", "confidence": 0.5},
            ]
        }
    )
    assert len(h["alternative_attributions"]) == 2


def test_reaction_attribution_hypothesis_reaction_to_turn_id_is_shape_validated() -> None:
    h = normalize_reaction_attribution_hypothesis({"reaction_to_turn_id": "turn_42"})
    assert h["reaction_to_turn_id"] == "turn_42"
    h = normalize_reaction_attribution_hypothesis(
        {"reaction_to_turn_id": "turn_42_user_utterance"}
    )
    assert h["reaction_to_turn_id"] == "turn_42_user_utterance"
    # Invalid: not a turn handle
    h = normalize_reaction_attribution_hypothesis({"reaction_to_turn_id": "garbage"})
    assert h["reaction_to_turn_id"] == ""
    h = normalize_reaction_attribution_hypothesis({"reaction_to_turn_id": ""})
    assert h["reaction_to_turn_id"] == ""


def test_reaction_attribution_hypothesis_evidence_refs_reject_raw_text() -> None:
    h = normalize_reaction_attribution_hypothesis(
        {
            "evidence_refs": [
                "the user said 嗯",
                "turn_42_user_utterance",
            ]
        }
    )
    assert h["evidence_refs"] == ["turn_42_user_utterance"]


# === Commit id derivation ==============================================


def test_compute_m18_7_commit_id_is_deterministic() -> None:
    a = compute_m18_7_commit_id(
        kind=KIND_ADDRESSEE, turn_index=42, source_ref="m18_7_addressee_42"
    )
    b = compute_m18_7_commit_id(
        kind=KIND_ADDRESSEE, turn_index=42, source_ref="m18_7_addressee_42"
    )
    assert a == b


def test_compute_m18_7_commit_id_differs_per_kind() -> None:
    addr = compute_m18_7_commit_id(
        kind=KIND_ADDRESSEE, turn_index=42, source_ref="m18_7_x_42"
    )
    react = compute_m18_7_commit_id(
        kind=KIND_REACTION, turn_index=42, source_ref="m18_7_x_42"
    )
    assert addr != react


def test_compute_m18_7_commit_id_differs_per_turn() -> None:
    a = compute_m18_7_commit_id(
        kind=KIND_ADDRESSEE, turn_index=42, source_ref="m18_7_x_42"
    )
    b = compute_m18_7_commit_id(
        kind=KIND_ADDRESSEE, turn_index=43, source_ref="m18_7_x_42"
    )
    assert a != b


def test_compute_m18_7_commit_id_handles_unknown_kind() -> None:
    """Defensive: unknown kind falls back to 'unknown'."""
    a = compute_m18_7_commit_id(
        kind="unknown_kind", turn_index=0, source_ref="src"
    )
    assert a  # non-empty sha1


# === State surface ======================================================


def test_attribution_hypotheses_state_surface_appends_entry() -> None:
    state: dict = {}
    entry = build_state_entry(
        kind=KIND_ADDRESSEE,
        turn_index=0,
        normalized={
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "evidence_refs": ["turn_0_user_utterance"],
            "alternative_hypotheses": [],
        },
    )
    entry["at"] = "2026-06-06T00:00:00Z"
    record_m18_7_attribution_hypotheses(state, entry)
    assert len(state["m18_7_attribution_hypotheses"]) == 1
    assert state["m18_7_attribution_hypotheses"][0]["kind"] == "addressee"
    assert state["m18_7_attribution_hypotheses"][0]["commit_id"]


def test_attribution_hypotheses_state_surface_caps_at_8_entries() -> None:
    state: dict = {}
    for i in range(12):
        entry = build_state_entry(
            kind=KIND_ADDRESSEE,
            turn_index=i,
            normalized={
                "participant_id": "alice",
                "addressed_to_assistant": True,
                "confidence": 0.9,
                "evidence_refs": [],
                "alternative_hypotheses": [],
            },
        )
        entry["at"] = f"2026-06-06T00:00:{i:02d}Z"
        record_m18_7_attribution_hypotheses(state, entry)
    assert len(state["m18_7_attribution_hypotheses"]) == M18_7_STATE_SURFACE_CAP
    assert M18_7_STATE_SURFACE_CAP == 8


def test_attribution_hypotheses_state_surface_overflow_evicts_oldest() -> None:
    state: dict = {}
    for i in range(10):
        entry = build_state_entry(
            kind=KIND_REACTION,
            turn_index=i,
            normalized={
                "participant_id": "bob",
                "reaction_to_turn_id": "turn_0",
                "reaction_to_participant_id": "alice",
                "is_about_assistant_claim": False,
                "confidence": 0.7,
                "evidence_refs": [],
                "alternative_attributions": [],
            },
        )
        entry["at"] = f"2026-06-06T00:00:{i:02d}Z"
        record_m18_7_attribution_hypotheses(state, entry)
    # First two are evicted; last is the latest.
    surface = state["m18_7_attribution_hypotheses"]
    assert surface[0]["turn_index"] == 2
    assert surface[-1]["turn_index"] == 9


def test_attribution_hypotheses_state_surface_ignores_empty_entry() -> None:
    state: dict = {}
    record_m18_7_attribution_hypotheses(state, {})
    record_m18_7_attribution_hypotheses(state, None)  # type: ignore[arg-type]
    assert state.get("m18_7_attribution_hypotheses", []) == []


# === State surface entry shape ==========================================


def test_build_state_entry_addressee_shape() -> None:
    entry = build_state_entry(
        kind=KIND_ADDRESSEE,
        turn_index=42,
        normalized={
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "evidence_refs": ["turn_42_user_utterance"],
            "alternative_hypotheses": [{"addressed_to_assistant": False}],
        },
    )
    assert entry["kind"] == "addressee"
    assert entry["turn_index"] == 42
    assert entry["commit_id"]  # sha1
    assert entry["participant_id"] == "alice"
    assert entry["addressed_to_assistant"] is True
    assert entry["confidence"] == 0.9
    assert entry["alternative_hypothesis_count"] == 1
    assert entry["evidence_refs"] == ["turn_42_user_utterance"]
    assert entry["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL


def test_build_state_entry_reaction_shape() -> None:
    entry = build_state_entry(
        kind=KIND_REACTION,
        turn_index=42,
        normalized={
            "participant_id": "bob",
            "reaction_to_turn_id": "turn_40",
            "reaction_to_participant_id": "alice",
            "is_about_assistant_claim": True,
            "confidence": 0.7,
            "evidence_refs": ["turn_42_user_utterance"],
            "alternative_attributions": [{"reaction_to_turn_id": "turn_41"}],
        },
    )
    assert entry["kind"] == "reaction"
    assert entry["turn_index"] == 42
    assert entry["commit_id"]
    assert entry["participant_id"] == "bob"
    assert entry["reaction_to_turn_id"] == "turn_40"
    assert entry["reaction_to_participant_id"] == "alice"
    assert entry["is_about_assistant_claim"] is True
    assert entry["confidence"] == 0.7
    assert entry["alternative_attribution_count"] == 1


def test_build_state_entry_unknown_kind_returns_empty() -> None:
    entry = build_state_entry(
        kind="unknown_kind",
        turn_index=0,
        normalized={"participant_id": "alice"},
    )
    assert entry == {}


# === Bus events: AddresseeHypothesisAdmitted =============================


def test_addressee_hypothesis_admitted_event_emitted() -> None:
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "rationale": "Alice is replying",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    }
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert report["addressee_event_emitted"] is True
    assert report["reaction_event_emitted"] is False
    assert any(e["type"] == "AddresseeHypothesisAdmitted" for e in bus)


def test_addressee_hypothesis_admitted_event_shape() -> None:
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "evidence_refs": ["turn_0_user_utterance", "turn_0_reply_to_turn_id"],
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    event = next(e for e in bus if e["type"] == "AddresseeHypothesisAdmitted")
    assert event["turn_index"] == 0
    assert event["commit_id"]  # shared with state entry
    assert event["participant_id"] == "alice"
    assert event["addressed_to_assistant"] is True
    assert event["confidence"] == 0.9
    assert event["evidence_ref_count"] == 2
    assert event["reason_codes"] == [REASON_FIELD_PRESENT]
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL
    assert event["at"] == "2026-06-06T00:00:00Z"


def test_addressee_hypothesis_admitted_event_omitted_when_field_empty() -> None:
    bus: list = []
    state: dict = {}
    conscious = {"addressee_hypothesis": {}}  # empty → no event
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert not any(e["type"] == "AddresseeHypothesisAdmitted" for e in bus)
    assert not any(e["type"] == "ReactionAttributionHypothesisAdmitted" for e in bus)


def test_addressee_hypothesis_admitted_event_omits_rationale_text() -> None:
    """M18.7 DECIDED 11: the audit envelope does NOT include
    the rationale text. The LLM is the source; engineering
    audits only the shape (length is recorded as
    `rationale_chars`).
    """
    bus: list = []
    state: dict = {}
    rationale = "some LLM rationale text"  # 24 chars
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "rationale": rationale,
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    event = next(e for e in bus if e["type"] == "AddresseeHypothesisAdmitted")
    assert "rationale" not in event
    # The envelope records `rationale_chars` as length only
    # (NOT the text itself).
    assert event.get("rationale_chars", 0) == len(rationale)
    # Sanity: the rationale text is NOT in the serialized envelope.
    assert rationale not in str(event)


# === Bus events: ReactionAttributionHypothesisAdmitted =================


def test_reaction_attribution_hypothesis_admitted_event_emitted() -> None:
    bus: list = []
    state: dict = {}
    conscious = {
        "reaction_attribution_hypothesis": {
            "participant_id": "bob",
            "reaction_to_turn_id": "turn_0",
            "reaction_to_participant_id": "alice",
            "is_about_assistant_claim": True,
            "confidence": 0.7,
        }
    }
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert report["reaction_event_emitted"] is True
    assert any(e["type"] == "ReactionAttributionHypothesisAdmitted" for e in bus)


def test_reaction_attribution_hypothesis_admitted_event_shape() -> None:
    bus: list = []
    state: dict = {}
    conscious = {
        "reaction_attribution_hypothesis": {
            "participant_id": "bob",
            "reaction_to_turn_id": "turn_0",
            "reaction_to_participant_id": "alice",
            "is_about_assistant_claim": True,
            "confidence": 0.72,
            "evidence_refs": ["turn_0_user_utterance"],
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    event = next(
        e for e in bus if e["type"] == "ReactionAttributionHypothesisAdmitted"
    )
    assert event["turn_index"] == 0
    assert event["commit_id"]
    assert event["participant_id"] == "bob"
    assert event["reaction_to_turn_id"] == "turn_0"
    assert event["reaction_to_participant_id"] == "alice"
    assert event["is_about_assistant_claim"] is True
    assert event["confidence"] == 0.72
    assert event["reason_codes"] == [REASON_FIELD_PRESENT]
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL


def test_reaction_attribution_hypothesis_admitted_event_omitted_when_field_empty() -> None:
    bus: list = []
    state: dict = {}
    conscious = {"reaction_attribution_hypothesis": {}}
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert not any(
        e["type"] == "ReactionAttributionHypothesisAdmitted" for e in bus
    )


def test_both_hypothesis_events_emitted_independently() -> None:
    """A turn that produces both fields emits both events."""
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
        },
        "reaction_attribution_hypothesis": {
            "participant_id": "bob",
            "reaction_to_turn_id": "turn_0",
            "reaction_to_participant_id": "alice",
            "is_about_assistant_claim": False,
            "confidence": 0.6,
        },
    }
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert report["addressee_event_emitted"] is True
    assert report["reaction_event_emitted"] is True
    # Both state surface entries written.
    surface = state["m18_7_attribution_hypotheses"]
    assert len(surface) == 2
    kinds = {e["kind"] for e in surface}
    assert kinds == {"addressee", "reaction"}


# === Bus events: AttributionHypothesisSkipped (fast_chat) ==============


def test_attribution_hypothesis_skipped_event_emitted_on_fast_chat_group_turn() -> None:
    bus: list = []
    state: dict = {}
    group_turn_binding = {
        "current_speaker_participant_id": "alice",
        "addressed_participant_ids": [],
        "ambiguity_band": "low",
    }
    # Both fields empty.
    conscious = {}
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
        latency_mode="fast_chat",
        group_turn_binding=group_turn_binding,
    )
    assert report["skipped_event_emitted"] is True
    event = next(
        e for e in bus if e["type"] == "AttributionHypothesisSkipped"
    )
    assert event["latency_mode"] == "fast_chat"
    assert event["group_turn_binding_present"] is True
    assert event["addressee_hypothesis_present"] is False
    assert event["reaction_attribution_hypothesis_present"] is False
    assert event["reason_code"] == REASON_FIELD_SKIPPED_FAST_CHAT


def test_attribution_hypothesis_skipped_event_omitted_on_full_conscious() -> None:
    """On full-conscious turns, an empty result is a legitimate
    LLM judgment — no skip event is emitted.
    """
    bus: list = []
    state: dict = {}
    group_turn_binding = {"current_speaker_participant_id": "alice"}
    conscious = {}
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
        latency_mode="normal",
        group_turn_binding=group_turn_binding,
    )
    assert report["skipped_event_emitted"] is False
    assert not any(
        e["type"] == "AttributionHypothesisSkipped" for e in bus
    )


def test_attribution_hypothesis_skipped_event_omitted_on_non_group_turn() -> None:
    """On non-group turns (no group_turn_binding), no skip
    event is emitted even in fast_chat.
    """
    bus: list = []
    state: dict = {}
    conscious = {}
    report = emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
        latency_mode="fast_chat",
        group_turn_binding=None,
    )
    assert report["skipped_event_emitted"] is False


def test_attribution_hypothesis_skipped_event_omitted_when_field_present() -> None:
    """If the LLM DID fill the field, no skip event is emitted
    even in fast_chat.
    """
    bus: list = []
    state: dict = {}
    group_turn_binding = {"current_speaker_participant_id": "alice"}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
        latency_mode="fast_chat",
        group_turn_binding=group_turn_binding,
    )
    assert not any(
        e["type"] == "AttributionHypothesisSkipped" for e in bus
    )


def test_should_emit_attribution_hypothesis_skipped_helper() -> None:
    # Fast chat + group + both empty → True
    assert should_emit_attribution_hypothesis_skipped(
        latency_mode="fast_chat",
        group_turn_binding={"current_speaker_participant_id": "alice"},
        addressee_hypothesis_normalized={},
        reaction_attribution_hypothesis_normalized={},
    ) is True
    # Normal mode + group + both empty → False
    assert should_emit_attribution_hypothesis_skipped(
        latency_mode="normal",
        group_turn_binding={"current_speaker_participant_id": "alice"},
        addressee_hypothesis_normalized={},
        reaction_attribution_hypothesis_normalized={},
    ) is False
    # Fast chat + no group → False
    assert should_emit_attribution_hypothesis_skipped(
        latency_mode="fast_chat",
        group_turn_binding=None,
        addressee_hypothesis_normalized={},
        reaction_attribution_hypothesis_normalized={},
    ) is False
    # Fast chat + group + addressee present → False
    assert should_emit_attribution_hypothesis_skipped(
        latency_mode="fast_chat",
        group_turn_binding={"current_speaker_participant_id": "alice"},
        addressee_hypothesis_normalized={"participant_id": "alice"},
        reaction_attribution_hypothesis_normalized={},
    ) is False


# === Privacy parity =====================================================


def test_addressee_hypothesis_obeys_m18_4_disclosure_policy() -> None:
    """M18.4 disclosure forbade the identification → the LLM
    returns `participant_id = ""`. Engineering preserves the
    empty string; the bus event shows `participant_id = ""`.
    """
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "",  # M18.4 forbade
            "addressed_to_assistant": True,
            "confidence": 0.85,
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    event = next(
        e for e in bus if e["type"] == "AddresseeHypothesisAdmitted"
    )
    assert event["participant_id"] == ""


def test_addressee_hypothesis_with_disallowed_disclosure_returns_empty_participant_id() -> None:
    """A non-shape participant_id is rejected and defaults to
    empty (silent "M18.4 disclosure forbade" / "LLM declined").
    """
    h = normalize_addressee_hypothesis(
        {
            "participant_id": "has spaces and !@# chars",
            "addressed_to_assistant": True,
            "confidence": 0.85,
        }
    )
    assert h["participant_id"] == ""


# === Compatibility with M20 (engineering layer) =========================


def test_m18_7_does_not_block_run_turn_when_fields_empty() -> None:
    """Empty M18.7 fields must never block the run_turn."""
    bus: list = []
    state: dict = {}
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan={},
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    # No bus events, no state surface writes.
    assert bus == []
    assert state.get("m18_7_attribution_hypotheses", []) == []


def test_m18_7_does_not_block_run_turn_when_low_confidence() -> None:
    """A field with low confidence is treated as empty (the LLM
    leaves it as `{}` per the M18.7 prompt; engineering does
    not invent a hypothesis).
    """
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.2,  # below 0.4 — LLM should leave empty
        }
    }
    # Engineering still records the entry even at low
    # confidence (the LLM chose to fill the field; engineering
    # does not second-guess). The M20.4 producer filter is the
    # layer that drops low-confidence rows. This test asserts
    # engineering does NOT block the run_turn.
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert any(
        e["type"] == "AddresseeHypothesisAdmitted" for e in bus
    )


def test_m18_7_v2_field_does_not_break_existing_m20_3_v2_field() -> None:
    """The M20.3 v2 attribute `correcting_assistant_identity`
    is preserved alongside the new M18.7 v2 attributes.
    """
    from segmentum.dialogue.runtime.mvp_loop import (
        normalize_conscious_turn_plan,
    )
    plan = normalize_conscious_turn_plan(
        {
            "correcting_assistant_identity": "wrong_persona",
            "addressee_hypothesis": {
                "participant_id": "alice",
                "addressed_to_assistant": True,
                "confidence": 0.9,
            },
        }
    )
    assert plan["correcting_assistant_identity"] == "wrong_persona"
    assert plan["addressee_hypothesis"]["participant_id"] == "alice"


def test_attribution_hypotheses_dont_admit_active_commitment_yet() -> None:
    """M18.7 is a structured-input contract. M18.7 does NOT
    admit ActiveCommitment rows; that is M20.4.
    """
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    # No ActiveCommitment-related events (those are emitted by
    # M20.0 `_admit_active_commitments`).
    types = {e.get("type") for e in bus}
    assert "ActiveCommitmentCreated" not in types
    assert "ActiveCommitmentRejected" not in types


def test_attribution_hypotheses_surface_is_m20_4_consumable() -> None:
    """The state surface shape matches M20.4's expected
    hand-off contract: kind / turn_index / commit_id /
    hypothesis sub-fields / evidence_refs / engineering_proxy_label.
    """
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=[],
        state=state,
        conscious_plan=conscious,
        turn_index=42,
        at="2026-06-06T00:00:00Z",
    )
    surface = state["m18_7_attribution_hypotheses"]
    assert len(surface) == 1
    entry = surface[0]
    # M20.4 hand-off contract fields.
    assert entry["kind"] == "addressee"
    assert entry["turn_index"] == 42
    assert entry["commit_id"]
    assert entry["participant_id"] == "alice"
    assert entry["addressed_to_assistant"] is True
    assert entry["confidence"] == 0.9
    assert "evidence_refs" in entry
    assert "engineering_proxy_label" in entry


# === CLAUDE.md compliance ==============================================


def test_no_engineering_keyword_matching_in_normalize() -> None:
    """The normalize step is pure: it clamps, truncates, and
    validates shape. It does not look at the rationale or
    evidence_refs content. We assert the helper does not import
    any string-matching library.
    """
    import segmentum.dialogue.runtime.m18_7_attribution as m

    src = open(m.__file__, encoding="utf-8").read()
    # No `re.` calls (regex), no `keyword` library use, no `text
    # contains` patterns in the helpers.
    assert "import re" not in src or src.count("import re") == 0
    assert "import keyword" not in src


def test_no_engineering_regex_matching_in_normalize() -> None:
    import segmentum.dialogue.runtime.m18_7_attribution as m

    src = open(m.__file__, encoding="utf-8").read()
    # No `re.search`, `re.match`, `re.compile`, `re.findall` calls
    # in the normalize / bus event / state surface paths.
    for forbidden in ["re.search", "re.match", "re.compile", "re.findall"]:
        assert forbidden not in src, f"forbidden regex call: {forbidden}"


def test_no_user_text_in_persisted_state() -> None:
    """The state surface does not persist user text. Only
    bounded handles and frozen hypothesis sub-fields.
    """
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "rationale": "the user said something sensitive here",
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=[],
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    # The state entry must not contain the rationale text.
    entry = state["m18_7_attribution_hypotheses"][0]
    serialized = str(entry)
    assert "the user said something sensitive" not in serialized


def test_no_user_text_in_bus_event_envelopes() -> None:
    """The bus event envelopes do not include user text or
    rationale text (M18.7 DECIDED 11).
    """
    bus: list = []
    state: dict = {}
    conscious = {
        "addressee_hypothesis": {
            "participant_id": "alice",
            "addressed_to_assistant": True,
            "confidence": 0.9,
            "rationale": "the user said something private here",
        }
    }
    emit_m18_7_attribution_for_turn(
        bus=bus,
        state=state,
        conscious_plan=conscious,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    serialized = str(bus)
    assert "the user said something private" not in serialized


# === Bus event builder shape (defensive) ================================


def test_build_addressee_hypothesis_admitted_event_empty_entry() -> None:
    event = build_addressee_hypothesis_admitted_event(
        turn_index=0,
        entry={},
        at="2026-06-06T00:00:00Z",
    )
    assert event["type"] == "AddresseeHypothesisAdmitted"
    assert event["commit_id"] == ""


def test_build_reaction_attribution_hypothesis_admitted_event_empty_entry() -> None:
    event = build_reaction_attribution_hypothesis_admitted_event(
        turn_index=0,
        entry={},
        at="2026-06-06T00:00:00Z",
    )
    assert event["type"] == "ReactionAttributionHypothesisAdmitted"
    assert event["commit_id"] == ""


def test_build_attribution_hypothesis_skipped_event_shape() -> None:
    event = build_attribution_hypothesis_skipped_event(
        turn_index=42,
        latency_mode="fast_chat",
        group_turn_binding_present=True,
        addressee_hypothesis_present=False,
        reaction_attribution_hypothesis_present=False,
        at="2026-06-06T00:00:42Z",
    )
    assert event["type"] == "AttributionHypothesisSkipped"
    assert event["turn_index"] == 42
    assert event["latency_mode"] == "fast_chat"
    assert event["group_turn_binding_present"] is True
    assert event["addressee_hypothesis_present"] is False
    assert event["reason_code"] == REASON_FIELD_SKIPPED_FAST_CHAT
    assert event["reason_codes"] == [REASON_FIELD_SKIPPED_FAST_CHAT]
    assert event["engineering_proxy_label"] == M18_7_ENGINEERING_PROXY_LABEL
    assert event["at"] == "2026-06-06T00:00:42Z"


# === Constants =========================================================


def test_allowed_m18_7_kind_is_frozen() -> None:
    assert ALLOWED_M18_7_KIND == frozenset({"addressee", "reaction"})
    assert KIND_ADDRESSEE == "addressee"
    assert KIND_REACTION == "reaction"


def test_m18_7_engineering_proxy_label_is_frozen() -> None:
    assert M18_7_ENGINEERING_PROXY_LABEL == "mvp_local_group_attribution"


def test_m18_7_state_surface_cap_is_frozen() -> None:
    assert M18_7_STATE_SURFACE_CAP == 8
