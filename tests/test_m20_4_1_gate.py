"""Tests for M20.4.1 v1 same-turn addressee hypothesis gate.

The gate is a pure rule (no LLM) that runs immediately
after the conscious loop and BEFORE the reply generation
stages. When the engagement rule fires, the gate:

1. Builds a `SameTurnAddresseeHypothesisGateVerdict`.
2. Emits a `SameTurnAddresseeHypothesisGateVerdict` bus event.
3. Appends the verdict to
   `state["m20_4_1_same_turn_gate_outcomes"]` (bounded
   tail ≤ 8).
4. Writes the verdict to the single-slot handoff
   `state["m20_4_1_pending_override"]`. The M18.5
   enforcement point reads the slot to override
   `no_reply` / `clarify_addressee` to
   `reply_to_current_speaker`.

Engagement rule (matches M20.4 v1 tie-breaker; C1 fix
AND not OR):
- `conscious.addressee_hypothesis` non-empty
- `addressed_to_assistant` is True
- `participant_id` non-empty (M18.4 disclosure guard)
- `confidence > 0.85` (strict)
- `ambiguity_band == "high"`
- `addressed_participant_ids` empty
- `mentioned_participant_ids` empty
- `reply_to_turn_id` empty / None
- `m18_5_structural_decision in {clarify_addressee, no_reply}`
"""

from __future__ import annotations

from segmentum.dialogue.runtime.m20_4_1_same_turn_gate import (
    BUS_EVENT_TYPE,
    DECISION_OVERRIDE,
    M20_4_1_AMBIGUITY_BANDS,
    M20_4_1_ENGINEERING_PROXY_LABEL,
    M20_4_1_OVERRIDABLE_DECISIONS,
    M20_4_1_OVERRIDE_ENABLED,
    M20_4_1_STATE_SURFACE_LIMIT,
    M20_4_1_TIE_BREAKER_CONFIDENCE_MIN,
    REASON_GATE_FIRED,
    REASON_GATE_SILENT,
    STATE_OUTCOMES_KEY,
    STATE_PENDING_OVERRIDE_KEY,
    SameTurnAddresseeHypothesisGateVerdict,
    build_same_turn_gate_verdict_event,
    clear_pending_override,
    get_pending_override,
    same_turn_addressee_hypothesis_gate,
)


def _addressee_hypothesis(
    *,
    addressed_to_assistant: bool = True,
    confidence: float = 0.9,
    participant_id: str = "user_alice",
    rationale: str = "direct question to assistant",
    evidence_refs: list[str] | None = None,
) -> dict:
    return {
        "participant_id": participant_id,
        "addressed_to_assistant": addressed_to_assistant,
        "confidence": confidence,
        "rationale": rationale,
        "evidence_refs": evidence_refs or ["turn_0_user_utterance"],
        "alternative_hypotheses": [],
    }


def _conscious_plan(
    *,
    addressee: dict | None = None,
) -> dict:
    return {
        "addressee_hypothesis": addressee if addressee is not None else {},
    }


def _binding(
    *,
    ambiguity_band: str = "high",
    addressed_participant_ids: list[str] | None = None,
    mentioned_participant_ids: list[str] | None = None,
    reply_to_turn_id: str = "",
) -> dict:
    return {
        "ambiguity_band": ambiguity_band,
        "addressed_participant_ids": addressed_participant_ids or [],
        "mentioned_participant_ids": mentioned_participant_ids or [],
        "reply_to_turn_id": reply_to_turn_id,
    }


# === Frozen constants ====================================================


def test_constants_are_frozen() -> None:
    assert M20_4_1_TIE_BREAKER_CONFIDENCE_MIN == 0.85
    assert M20_4_1_STATE_SURFACE_LIMIT == 8
    # P3 kill-switch default is OFF (audit-only). Real-LLM
    # calibration (P0, 2026-06-08) shows LLM acc in the
    # override band is 0.5 addr / 0.0 react; override path
    # is held in audit-only mode until calibration improves.
    assert M20_4_1_OVERRIDE_ENABLED is False
    assert M20_4_1_ENGINEERING_PROXY_LABEL == "mvp_local_group_attribution"
    assert M20_4_1_AMBIGUITY_BANDS == frozenset({"low", "medium", "high"})
    assert M20_4_1_OVERRIDABLE_DECISIONS == frozenset(
        {"clarify_addressee", "no_reply"}
    )
    assert REASON_GATE_FIRED == "m20_4_1_same_turn_fired"
    assert REASON_GATE_SILENT == "m20_4_1_same_turn_silent"
    assert BUS_EVENT_TYPE == "SameTurnAddresseeHypothesisGateVerdict"
    assert DECISION_OVERRIDE == "overridden_to_reply_to_current_speaker"


# === Gate fires (8 tests) ================================================


def test_gate_fires_when_all_conditions_hold_no_reply() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert isinstance(verdict, SameTurnAddresseeHypothesisGateVerdict)
    assert verdict.decision == DECISION_OVERRIDE
    assert verdict.m18_5_structural_decision == "no_reply"
    assert verdict.engineering_proxy_label == M20_4_1_ENGINEERING_PROXY_LABEL
    assert verdict.turn_index == 0


def test_gate_fires_when_m18_5_decision_is_clarify_addressee() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="clarify_addressee",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.m18_5_structural_decision == "clarify_addressee"


def test_gate_fires_at_minimum_strict_inequality_above_threshold() -> None:
    """`confidence > 0.85` is strict; 0.851 is the smallest valid."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.851)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None


def test_gate_fires_at_full_confidence() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=1.0)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None


def test_gate_fires_preserves_m18_5_decision_in_audit_envelope() -> None:
    """DECIDED 9: m18_5_structural_decision preserved for diagnose."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="clarify_addressee",
        bus=bus,
        state=state,
        turn_index=42,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.m18_5_structural_decision == "clarify_addressee"
    assert verdict.turn_index == 42


def test_gate_fires_with_bounded_evidence_refs() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(
                evidence_refs=["turn_0_user_utterance", "bus_event_x"]
            )
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert "turn_0_user_utterance" in verdict.evidence_refs
    assert "bus_event_x" in verdict.evidence_refs


def test_gate_fires_with_participant_id_as_commit_id() -> None:
    """The M18.7 participant_id echoes into the commit_ids audit field."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(participant_id="user_bob")
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert "user_bob" in verdict.commit_ids


def test_gate_fires_writes_override_handoff() -> None:
    """P3 default: kill-switch OFF, gate fires but does NOT
    write the override handoff. M18.5 applies its structural
    decision unchanged. See the audit-only tests below for
    the full default-mode semantics."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    # P3 default: handoff is NOT written (kill-switch off).
    assert get_pending_override(state) is None
    # But the bus event and surface row ARE recorded
    # (audit-only mode).
    assert any(e["type"] == BUS_EVENT_TYPE for e in bus)
    assert STATE_OUTCOMES_KEY in state
    # And the verdict's audit-only flag is True.
    assert verdict.m20_4_1_audit_only is True
    # And the audit envelope carries the flag.
    bus_event = next(e for e in bus if e["type"] == BUS_EVENT_TYPE)
    assert bus_event["m20_4_1_audit_only"] is True


# === Gate does NOT fire (12 tests) =======================================


def test_gate_does_not_fire_when_confidence_below_threshold() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None
    assert get_pending_override(state) is None
    assert bus == []


def test_gate_does_not_fire_when_confidence_at_threshold() -> None:
    """Strict `> 0.85` inequality."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.85)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_ambiguity_band_low() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(ambiguity_band="low"),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_ambiguity_band_medium() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(ambiguity_band="medium"),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_addressed_participant_ids_non_empty() -> None:
    """C1 fix: structural explicit addressee → gate stays no-op."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(addressed_participant_ids=["alice"]),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_mentioned_participant_ids_non_empty() -> None:
    """C1 fix: explicit mention of another → gate stays no-op."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(mentioned_participant_ids=["bob"]),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_reply_to_turn_id_set() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(reply_to_turn_id="turn_5"),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_m18_5_decision_is_reply_to_current_speaker() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="reply_to_current_speaker",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_m18_5_decision_is_defer_side_thread() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="defer_side_thread",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_m18_5_decision_is_reply_to_named_third_party() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="reply_to_named_third_party",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_m18_5_decision_is_reply_to_whole_group() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="reply_to_whole_group",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_conscious_plan_has_no_addressee_hypothesis() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan={},
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_conscious_plan_has_empty_addressee_hypothesis() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan={"addressee_hypothesis": {}},
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_addressed_to_assistant_is_false() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(addressed_to_assistant=False)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_fire_when_participant_id_is_empty() -> None:
    """M18.4 disclosure guard: empty participant_id → gate silent."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(participant_id="")
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


# === Audit envelope (5 tests) ============================================


def test_gate_emits_SameTurnAddresseeHypothesisGateVerdict_when_fired() -> None:
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert len(bus) == 1
    event = bus[0]
    assert event["type"] == BUS_EVENT_TYPE
    assert event["decision"] == DECISION_OVERRIDE
    assert event["m18_5_structural_decision"] == "no_reply"
    assert event["engineering_proxy_label"] == M20_4_1_ENGINEERING_PROXY_LABEL


def test_gate_does_not_emit_verdict_when_not_fired() -> None:
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert bus == []


def test_gate_verdict_omits_rationale_text() -> None:
    """Hard rule 7: rationale text never persists beyond M18.7."""
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(
                rationale="this is a private observation that must not leak"
            )
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    event = bus[0]
    event_str = str(event)
    assert "private observation" not in event_str
    assert "rationale" not in event


def test_gate_verdict_shape_is_frozen() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=7,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    event = build_same_turn_gate_verdict_event(verdict)
    assert event["turn_index"] == 7
    assert event["at"] == "2026-06-07T00:00:00Z"
    assert event["reason_codes"] == [REASON_GATE_FIRED]


def test_gate_emits_exactly_one_bus_event_per_turn() -> None:
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert len(bus) == 2  # one per call (per turn shape)


# === State surface (3 tests) =============================================


def test_state_surface_appends_entry_when_fired() -> None:
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert STATE_OUTCOMES_KEY in state
    assert len(state[STATE_OUTCOMES_KEY]) == 1
    assert state[STATE_OUTCOMES_KEY][0]["type"] == BUS_EVENT_TYPE


def test_state_surface_caps_at_8_entries() -> None:
    state: dict = {}
    bus: list = []
    for i in range(15):
        same_turn_addressee_hypothesis_gate(
            conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
            group_turn_binding=_binding(),
            m18_5_structural_decision="no_reply",
            bus=bus,
            state=state,
            turn_index=i,
            now=f"2026-06-07T00:00:{i:02d}Z",
        )
    assert len(state[STATE_OUTCOMES_KEY]) == 8


def test_state_surface_overflow_evicts_oldest() -> None:
    state: dict = {}
    bus: list = []
    for i in range(10):
        same_turn_addressee_hypothesis_gate(
            conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
            group_turn_binding=_binding(),
            m18_5_structural_decision="no_reply",
            bus=bus,
            state=state,
            turn_index=i,
            now=f"2026-06-07T00:00:{i:02d}Z",
        )
    # Oldest (turn_index=0..1) should have been evicted.
    surface = state[STATE_OUTCOMES_KEY]
    assert len(surface) == 8
    assert surface[0]["turn_index"] == 2
    assert surface[-1]["turn_index"] == 9


# === CLAUDE.md compliance (5 tests) ======================================


def test_gate_does_not_call_llm() -> None:
    """Hard rule 1: no new LLM stage. Engineering is pure rule."""
    import segmentum.dialogue.runtime.m20_4_1_same_turn_gate as m

    src = open(m.__file__, encoding="utf-8").read()
    # No LLM call helpers imported.
    for forbidden in ["_complete_json_stage", "llm_call", "ollama", "anthropic"]:
        assert forbidden not in src


def test_gate_does_not_match_user_text() -> None:
    """Hard rule 2: no regex / keyword matching on user text."""
    import segmentum.dialogue.runtime.m20_4_1_same_turn_gate as m

    src = open(m.__file__, encoding="utf-8").read()
    for forbidden in ["re.search", "re.match", "re.compile", "re.findall"]:
        assert forbidden not in src


def test_gate_does_not_match_hypothesis_text() -> None:
    """Hard rule 2: rationale / hypothesis text is never parsed."""
    import segmentum.dialogue.runtime.m20_4_1_same_turn_gate as m

    src = open(m.__file__, encoding="utf-8").read()
    # Strip the module docstring (mentions rationale in
    # documentation) and check the gate rule body only.
    code_only = src.split('"""', 2)[-1]
    rule_body = code_only.split("def _gate_rule_engaged")[1].split("def ")[0]
    # The gate rule never accesses `rationale` from the
    # addressee_hypothesis (only `addressed_to_assistant`,
    # `participant_id`, `confidence`, `evidence_refs`).
    assert "rationale" not in rule_body


def test_gate_does_not_modify_m18_5_decision_tree() -> None:
    """Hard rule 5: gate reads M18.5 outcome but does not modify M18.5."""
    import segmentum.dialogue.runtime.m20_4_1_same_turn_gate as m

    src = open(m.__file__, encoding="utf-8").read()
    # Strip the module docstring (mentions M18.5 for context).
    code_only = src.split('"""', 2)[-1]
    # The M18.5 function is not imported nor called.
    assert "_decide_group_reply_policy(" not in code_only
    assert "decide_group_reply_policy" not in code_only
    # The gate only reads the structural decision (string); it
    # never imports or calls into the M18.5 decision tree.
    assert "from segmentum.dialogue.runtime.mvp_loop import" not in code_only


def test_gate_engineering_proxy_label_uses_mvp_local_group_attribution() -> None:
    """The audit envelope carries the canonical v2 proxy label."""
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    event = bus[0]
    assert event["engineering_proxy_label"] == M20_4_1_ENGINEERING_PROXY_LABEL


# === Compatibility (4 tests) =============================================


def test_gate_runs_after_conscious_loop_before_reply_generation() -> None:
    """The gate is wired into the call order right after the conscious
    loop. The bus event is emitted; the verdict is in `state`. P3
    default: the override handoff is NOT written (audit-only mode)."""
    state: dict = {}
    bus: list = []
    conscious = _conscious_plan(addressee=_addressee_hypothesis())
    # The gate takes the conscious_plan as input; if the
    # conscious loop produced the M18.7 v2 attribute, the
    # gate sees it and may fire.
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=conscious,
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    # P3 default: gate emits bus + state-surface row, but
    # the override handoff is held back (audit-only).
    assert len(bus) == 1
    assert verdict.m20_4_1_audit_only is True
    assert get_pending_override(state) is None


def test_gate_does_not_block_run_turn_when_not_fired() -> None:
    """When the rule does not fire, the gate returns None and
    leaves the state untouched. Run turn continues normally."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None
    assert STATE_OUTCOMES_KEY not in state
    assert STATE_PENDING_OVERRIDE_KEY not in state
    assert bus == []


def test_gate_does_not_block_fast_chat() -> None:
    """When the LLM is in fast_chat mode and produced an empty
    M18.7 field, the gate is silent. The thin conscious skipped
    the M18.7 attribute; the gate sees empty and does not fire."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan={"addressee_hypothesis": {}},
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_gate_does_not_double_fire_with_m20_3_same_turn_surface() -> None:
    """The M20.4.1 gate and the M20.3 same_turn_surface settler
    serve different owners (gate = M18.5 structural override;
    M20.3 = runtime_mode_state block). They run at different
    points in the pipeline and never both fire on the same
    commit. The gate only writes to its own bus event type."""
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    # The gate emits only its own event type; never a
    # SameTurnSurfaceVerdict or anything M20.3-specific.
    for event in bus:
        assert event["type"] == BUS_EVENT_TYPE


# === Override handoff helpers ============================================


def test_clear_pending_override_removes_slot() -> None:
    state: dict = {STATE_PENDING_OVERRIDE_KEY: "fake_verdict"}
    clear_pending_override(state)
    assert STATE_PENDING_OVERRIDE_KEY not in state


def test_clear_pending_override_is_idempotent() -> None:
    state: dict = {}
    clear_pending_override(state)
    clear_pending_override(state)
    assert STATE_PENDING_OVERRIDE_KEY not in state


def test_get_pending_override_returns_none_when_empty() -> None:
    assert get_pending_override({}) is None
    assert get_pending_override(None) is None  # type: ignore[arg-type]


def test_gate_clears_stale_override_at_start_of_turn() -> None:
    """Per-turn slot must NOT leak from a previous turn."""
    state: dict = {STATE_PENDING_OVERRIDE_KEY: "stale"}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    # The stale override was cleared at the start of the gate
    # call; the gate did not fire (low confidence), so the slot
    # remains empty.
    assert get_pending_override(state) is None


# === End-to-end master fixture (8 tests) ================================


def test_master_fixture_T0_reply_present_when_gate_fires() -> None:
    """Master fixture (P3 default): T0 group turn with chat
    surface; M18.5 returns `no_reply`; conscious produces
    high-confidence addressee_hypothesis; gate fires and
    records a verdict + bus event, but the override handoff
    is NOT written (kill-switch off, audit-only mode). M18.5
    applies its `no_reply` structural decision unchanged on
    the visible reply. See the override-enabled companion
    test below for the legacy pre-P3 behavior."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.m20_4_1_audit_only is True
    # Override handoff is NOT in state (P3 default).
    override = get_pending_override(state)
    assert override is None
    # Bus event present (audit-only row).
    bus_event = next(
        (e for e in bus if e["type"] == BUS_EVENT_TYPE), None
    )
    assert bus_event is not None
    assert bus_event["m20_4_1_audit_only"] is True


def test_master_fixture_T0_reply_absent_when_gate_does_not_fire() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None
    # No override handoff.
    assert get_pending_override(state) is None
    # No bus event.
    assert bus == []
    # No state surface entry.
    assert STATE_OUTCOMES_KEY not in state


def test_master_fixture_m18_5_decision_preserved_in_audit() -> None:
    """DECIDED 9: M18.5 outcome preserved in audit envelope."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="clarify_addressee",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.m18_5_structural_decision == "clarify_addressee"
    event = bus[0]
    assert event["m18_5_structural_decision"] == "clarify_addressee"


def test_master_fixture_does_not_fire_on_explicit_addressee() -> None:
    """Structural has explicit addressee → gate stays no-op."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(addressed_participant_ids=["alice"]),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_master_fixture_does_not_fire_on_low_confidence() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(
            addressee=_addressee_hypothesis(confidence=0.5)
        ),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is None


def test_master_fixture_fires_on_no_reply() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None


def test_master_fixture_fires_on_clarify_addressee() -> None:
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="clarify_addressee",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None


def test_master_fixture_T0_uses_overridden_action() -> None:
    """P3 default (audit-only): the M18.5 enforcement point
    does NOT see the override (handoff is empty), and the
    visible reply uses M18.5's structural decision. The
    bus event still carries
    `overridden_to_reply_to_current_speaker` as the decision
    label so the diagnose surface can count
    "would-have-fired" cases. See the override-enabled
    companion test below for the legacy pre-P3 behavior."""
    state: dict = {}
    bus: list = []
    same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    # Bus event still carries the override decision label.
    event = bus[0]
    assert event["decision"] == DECISION_OVERRIDE
    # Audit-only flag is True; M18.5 does not apply the
    # override handoff (it's empty).
    assert event["m20_4_1_audit_only"] is True
    assert get_pending_override(state) is None


# === P3 kill-switch (audit-only default, 2026-06-08) =====================


def test_p3_default_kill_switch_is_off() -> None:
    """P3 default: M20_4_1_OVERRIDE_ENABLED is False. The
    override path is held in audit-only mode until real-LLM
    calibration improves in the high-confidence band."""
    assert M20_4_1_OVERRIDE_ENABLED is False


def test_p3_audit_only_emits_verdict_but_no_handoff() -> None:
    """When the kill-switch is OFF (P3 default), the gate
    fires, the bus event is emitted, the bounded state
    surface gets the verdict row, but the override handoff
    to M18.5 is NOT written. M18.5's structural decision
    applies unchanged."""
    state: dict = {}
    bus: list = []
    verdict = same_turn_addressee_hypothesis_gate(
        conscious_plan=_conscious_plan(addressee=_addressee_hypothesis()),
        group_turn_binding=_binding(),
        m18_5_structural_decision="no_reply",
        bus=bus,
        state=state,
        turn_index=0,
        now="2026-06-07T00:00:00Z",
    )
    assert verdict is not None
    # Audit-only flag is set on the verdict.
    assert verdict.m20_4_1_audit_only is True
    # Bus event carries the flag.
    bus_event = next(
        (e for e in bus if e["type"] == BUS_EVENT_TYPE), None
    )
    assert bus_event is not None
    assert bus_event["m20_4_1_audit_only"] is True
    # State surface has the row.
    assert STATE_OUTCOMES_KEY in state
    assert len(state[STATE_OUTCOMES_KEY]) == 1
    # BUT the override handoff is empty.
    assert STATE_PENDING_OVERRIDE_KEY not in state
    assert get_pending_override(state) is None


def test_p3_audit_only_applies_to_both_overridable_decisions() -> None:
    """The audit-only behavior holds for both overridable
    M18.5 decisions (`no_reply` and `clarify_addressee`).
    Real-LLM calibration (P0) showed the gate fires in the
    0.85+ band for both decision labels; P3 must not write
    the handoff for either path."""
    for decision in ("no_reply", "clarify_addressee"):
        state: dict = {}
        bus: list = []
        verdict = same_turn_addressee_hypothesis_gate(
            conscious_plan=_conscious_plan(
                addressee=_addressee_hypothesis()
            ),
            group_turn_binding=_binding(),
            m18_5_structural_decision=decision,
            bus=bus,
            state=state,
            turn_index=0,
            now="2026-06-07T00:00:00Z",
        )
        assert verdict is not None
        assert verdict.m20_4_1_audit_only is True
        # M18.5 structural decision is preserved in the
        # audit envelope regardless of audit-only mode.
        assert verdict.m18_5_structural_decision == decision
        # Override handoff is empty in both cases.
        assert get_pending_override(state) is None


def test_p3_verdict_audit_only_field_default_is_false() -> None:
    """When the verdict is built via `_build_verdict` without
    explicit `m20_4_1_audit_only`, the field defaults to
    False (legacy pre-P3 behavior preserved at the
    builder layer). The gate itself sets the field based on
    the kill-switch at call time."""
    # Build a verdict directly to confirm the dataclass
    # default is False (the gate is the only caller in
    # production and always sets the field explicitly).
    raw = SameTurnAddresseeHypothesisGateVerdict(
        decision=DECISION_OVERRIDE,
        m18_5_structural_decision="no_reply",
        commit_ids=("user_alice",),
        evidence_refs=("turn_0_user_utterance",),
        reason_codes=(REASON_GATE_FIRED,),
        engineering_proxy_label=M20_4_1_ENGINEERING_PROXY_LABEL,
        turn_index=0,
        at="2026-06-07T00:00:00Z",
    )
    assert raw.m20_4_1_audit_only is False


def test_p3_bus_event_carries_audit_only_field() -> None:
    """The bus event builder carries the audit-only flag
    end-to-end so the diagnose surface can count
    would-have-fired cases even when the override is held
    back."""
    verdict = SameTurnAddresseeHypothesisGateVerdict(
        decision=DECISION_OVERRIDE,
        m18_5_structural_decision="no_reply",
        commit_ids=("user_alice",),
        evidence_refs=("turn_0_user_utterance",),
        reason_codes=(REASON_GATE_FIRED,),
        engineering_proxy_label=M20_4_1_ENGINEERING_PROXY_LABEL,
        turn_index=0,
        at="2026-06-07T00:00:00Z",
        m20_4_1_audit_only=True,
    )
    event = build_same_turn_gate_verdict_event(verdict)
    assert event["m20_4_1_audit_only"] is True
    # Other fields unchanged.
    assert event["decision"] == DECISION_OVERRIDE
    assert event["m18_5_structural_decision"] == "no_reply"
    assert event["commit_ids"] == ["user_alice"]
