"""Tests for M20.4 v1 §2 producer (M18.7 → M20).

The M20.4 producer reads `state["m18_7_attribution_hypotheses"]`
(M18.7 §5 surface) and admits one `ActiveCommitment` per
matching entry on `group_addressee_graph`. Filters:
- `confidence >= 0.4`
- `participant_id != ""`

Empty surface → silent no-op. Each admit bumps the
`m20_4_attribution_diagnostics` surface.
"""

from __future__ import annotations

from segmentum.dialogue.runtime.active_commitment import (
    OBSERVABLE_V3,
    REASON_CODES_V2,
)
from segmentum.dialogue.runtime.m18_7_attribution import (
    KIND_ADDRESSEE,
    KIND_REACTION,
    build_state_entry,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    M20_4_AGGREGATION_BUNDLE_WEAK,
    M20_4_AGGREGATION_SINGLE_STRONG,
    M20_4_BUNDLE_AGGREGATED_THRESHOLD,
    M20_4_BUNDLE_DECAY_BASE,
    M20_4_BUNDLE_MAX_SINGLE_THRESHOLD,
    M20_4_BUNDLE_MEMORY_CAP,
    M20_4_BUNDLE_UNIQUE_COUNT_MIN,
    M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN,
    _bundle_aggregated_support,
    append_bundle_memory,
    build_addressee_target_match_admitted_event,
    produce_m20_4_attribution_commitments,
)


# === Helpers ==========================================================


def _entry(
    *,
    kind: str,
    turn_index: int,
    participant_id: str = "alice",
    confidence: float = 0.9,
    addressed_to_assistant: bool = True,
    reaction_to_turn_id: str = "",
    reaction_to_participant_id: str = "",
    is_about_assistant_claim: bool = True,
) -> dict:
    """Build a frozen M18.7 state surface entry (hand-off contract)."""
    sub: dict = {
        "participant_id": participant_id,
        "confidence": confidence,
    }
    if kind == KIND_ADDRESSEE:
        sub["addressed_to_assistant"] = addressed_to_assistant
        sub["alternative_hypothesis_count"] = 0
    else:
        sub["reaction_to_turn_id"] = reaction_to_turn_id
        sub["reaction_to_participant_id"] = reaction_to_participant_id
        sub["is_about_assistant_claim"] = is_about_assistant_claim
        sub["alternative_attribution_count"] = 0
    return build_state_entry(
        kind=kind, turn_index=turn_index, normalized=sub
    )


# === Producer admit rule ==============================================


def test_producer_admits_addressee_with_confidence_above_threshold() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        group_turn_binding={"ambiguity_band": "high"},
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.owner_id == "group_addressee_graph"
    assert commitment.observable == "addressee_target_match"
    assert commitment.source_kind == "state"
    assert commitment.layer == "B_per_turn_commitment"
    assert commitment.horizon == "next_turn"
    assert "m20_4_attribution" in commitment.reason_codes


def test_producer_rejects_addressee_with_confidence_below_threshold() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.3)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


def test_producer_rejects_addressee_with_empty_participant_id() -> None:
    """M18.4 disclosure forbade the identification; the LLM
    returned `participant_id = ""`. Engineering drops the
    row silently (no admission).
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, participant_id="")
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


def test_producer_admits_reaction_with_confidence_above_threshold() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.7,
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 1
    assert admitted[0].observable == "reaction_attribution_match"


def test_producer_rejects_reaction_with_confidence_below_threshold() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.1,
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


def test_producer_filters_hypotheses_by_threshold() -> None:
    """Mixed surface: one admits, one rejects per sub-class.

    P0-4 (2026-06-09): the addressee threshold is sub-class-
    dependent. `addressed_to_assistant == False` admits at
    0.4 (the v1 default); `addressed_to_assistant == True`
    admits at 0.7 (P0-4 raised bar; P1 surfaced
    recall_on_addressed = 0.0). This test pins the
    "not addressed" sub-class behavior. The "addressed"
    sub-class has its own dedicated tests (P0-4 series).
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.5,
                addressed_to_assistant=False,  # P0-4: "not addressed" sub-class
            ),
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.3,
                addressed_to_assistant=False,  # below
            ),
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.4,  # at threshold (admit)
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            ),
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.39,  # just below
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            ),
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 2
    observables = {c.observable for c in admitted}
    assert observables == {"addressee_target_match", "reaction_attribution_match"}


def test_producer_threshold_constant() -> None:
    assert M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN == 0.4


def test_producer_does_not_admit_when_state_attribution_surface_empty() -> None:
    state: dict = {}
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


def test_producer_does_not_admit_when_state_surface_malformed() -> None:
    """Non-list M18.7 surface → silent no-op."""
    state: dict = {"m18_7_attribution_hypotheses": "not a list"}
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


def test_producer_does_not_call_llm() -> None:
    """The producer is a pure function; no LLM call."""
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    # The function does not accept an LLM call parameter; the
    # check is structural (no LLM call injection point in the
    # producer).
    import inspect
    sig = inspect.signature(produce_m20_4_attribution_commitments)
    assert "llm_call" not in sig.parameters


# === Observable payload shape =========================================


def test_producer_observable_payload_omits_rationale_text() -> None:
    """Engineering never persists the LLM's rationale; the
    payload is bounded to the frozen hypothesis subset.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    # No rationale field
    assert "rationale" not in payload
    # Hypothesis is the frozen subset
    assert "addressed_to_assistant" in payload["hypothesis"]
    assert "confidence" in payload["hypothesis"]
    # inbound_bounded_excerpt is present and bounded
    assert "inbound_bounded_excerpt" in payload
    assert len(payload["inbound_bounded_excerpt"]) <= 200


def test_producer_observable_payload_includes_bounded_excerpt() -> None:
    long_text = "x" * 500
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt=long_text,
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    assert len(payload["inbound_bounded_excerpt"]) == 200


def test_producer_observable_payload_evidence_refs_shape_validated() -> None:
    """The M18.7 entry's `evidence_refs` are carried over to
    the M20.4 `ActiveCommitment.evidence_refs`. Shape is
    already validated by the M18.7 normalize step.
    """
    entry = _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
    entry["evidence_refs"] = ["turn_0_user_utterance"]
    state: dict = {"m18_7_attribution_hypotheses": [entry]}
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted[0].evidence_refs == ("turn_0_user_utterance",)


def test_producer_observable_payload_includes_hypothesis_commit_id() -> None:
    """M20.4 must use the M18.7 entry's `commit_id` for
    traceable hand-off (M20.4 §2)."""
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    assert "hypothesis_commit_id" in payload
    # The hypothesis_commit_id is a sha1 hex (40 chars).
    assert len(payload["hypothesis_commit_id"]) == 40


def test_producer_observable_payload_includes_group_turn_binding_snapshot() -> None:
    binding = {
        "ambiguity_band": "high",
        "addressed_participant_ids": [],
        "mentioned_participant_ids": [],
        "reply_to_turn_id": "",
    }
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        group_turn_binding=binding,
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    snapshot = payload["group_turn_binding_snapshot"]
    assert snapshot["ambiguity_band"] == "high"
    assert snapshot["addressed_participant_ids"] == []


def test_producer_observable_payload_includes_ambiguity_band() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        group_turn_binding={"ambiguity_band": "high"},
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    assert payload["ambiguity_band"] == "high"


# === Reaction attributer settler excerpt lookup ========================


def test_producer_reaction_looks_up_attributed_turn_in_bus() -> None:
    bus = [
        {
            "type": "UserUtteranceEvent",
            "turn_id": "turn_0",
            "text": "this is a prior turn text",
        }
    ]
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.7,
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=bus,
        current_turn_id=1,
        inbound_excerpt="",
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    assert payload["attributed_turn_id"] == "turn_0"
    assert "this is a prior turn text" in payload["attributed_bounded_excerpt"]


def test_producer_reaction_attributed_turn_not_in_bus() -> None:
    """When the attributed turn is not in the bus (older turn
    evicted), the excerpt is empty. The settler works with
    whatever is available.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.7,
                reaction_to_turn_id="turn_999",
                reaction_to_participant_id="alice",
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="",
        at="2026-06-06T00:00:00Z",
    )
    payload = dict(admitted[0].observable_payload or {})
    assert payload["attributed_turn_id"] == "turn_999"
    assert payload["attributed_bounded_excerpt"] == ""


# === Call order / invariants ==========================================


def test_producer_every_turn_no_op_when_empty() -> None:
    """The producer runs every turn (per M20.4 DECIDED 3) with
    an `if not hypotheses: return []` no-op.
    """
    state: dict = {}
    for _ in range(5):
        admitted = produce_m20_4_attribution_commitments(
            state=state,
            bus=[],
            current_turn_id=0,
            inbound_excerpt="",
            at="2026-06-06T00:00:00Z",
        )
        assert admitted == []


def test_producer_does_not_block_run_turn() -> None:
    """Empty M18.7 surface must never block the run_turn."""
    state: dict = {}
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=0,
        inbound_excerpt="",
        at="2026-06-06T00:00:00Z",
    )
    assert admitted == []


# === Diagnostic surface ==============================================


def test_attribution_diagnostics_records_producer_admit_total() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9),
            _entry(kind=KIND_REACTION, turn_index=0, confidence=0.7,
                   reaction_to_turn_id="turn_0", reaction_to_participant_id="alice"),
        ]
    }
    produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("producer_admit_total") == 2


def test_attribution_diagnostics_records_producer_reject_low_confidence_total() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.1),
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.2),
        ]
    }
    produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("producer_reject_low_confidence_total") == 2


def test_attribution_diagnostics_records_producer_reject_disclosure_total() -> None:
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, participant_id=""),
        ]
    }
    produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("producer_reject_disclosure_total") == 1


# === Observables + reason codes (v3 vocab) ==============================


def test_addressee_target_match_is_in_observable_v3() -> None:
    assert "addressee_target_match" in OBSERVABLE_V3
    assert OBSERVABLE_V3["addressee_target_match"]["settler_hint"] == "llm_judge"


def test_reaction_attribution_match_is_in_observable_v3() -> None:
    assert "reaction_attribution_match" in OBSERVABLE_V3
    assert OBSERVABLE_V3["reaction_attribution_match"]["settler_hint"] == "llm_judge"


def test_m20_4_attribution_reason_code_is_in_v2() -> None:
    assert "m20_4_attribution" in REASON_CODES_V2
    assert "m20_4_attribution_tie_breaker_engaged" in REASON_CODES_V2
    assert "m20_4_attribution_tie_breaker_rejected" in REASON_CODES_V2
    assert "m20_4_addressee_graph_microadjust" in REASON_CODES_V2


# === CLAUDE.md compliance ==============================================


def test_producer_does_not_match_user_text() -> None:
    """The producer is bounded to confidence + participant_id
    + commit_id checks. It does not look at the rationale or
    evidence_refs content.
    """
    import segmentum.dialogue.runtime.m20_4_attribution as m

    src = open(m.__file__, encoding="utf-8").read()
    for forbidden in ["re.search", "re.match", "re.compile", "re.findall"]:
        assert forbidden not in src, f"forbidden regex call: {forbidden}"


def test_no_user_text_in_observable_payload() -> None:
    """Engineering never persists the LLM's rationale; the
    observable_payload is bounded to the frozen hypothesis
    subset. The 'rationale' field of the LLM's M18.7
    hypothesis is not in the v1 frozen subset.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(kind=KIND_ADDRESSEE, turn_index=0, confidence=0.9)
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    payload_str = str(admitted[0].observable_payload)
    assert "rationale" not in payload_str


# === P0-4 (2026-06-09): sub-class admit threshold ====================
# P1 (M18.7.1 v2 + P1 real-LLM calibration) surfaced
# `precision_on_not_addressed = 1.0` and
# `recall_on_addressed = 0.0` on the bqxsmofri held-out
# fixture (12 turns, real OpenRouter deepseek-v4-flash).
# The structural asymmetry drives the M20.4 P0-4 producer
# change: raise the admit threshold for the
# `addressed_to_assistant == True` sub-class from 0.4 to
# 0.7. The "not addressed" sub-class keeps the v1 0.4
# default (LLM is 100% precise there).
# See `reports/m20_4_p0_4_subclass_admit.md`.


def test_p0_4_subclass_threshold_constant_is_frozen() -> None:
    """P0-4 frozen constant. The `addressee_target_match`
    sub-class `addressed_to_assistant == True` admits at
    0.7; the v1 default (`M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN`)
    is preserved at 0.4 for back-compat and for the
    "not addressed" sub-class.
    """
    from segmentum.dialogue.runtime.m20_4_attribution import (
        M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN,
        M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED,
    )
    assert M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN == 0.4
    assert (
        M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED
        == 0.7
    )
    # P0-4: the directed sub-class bar is strictly above
    # the v1 default. The 0.3 gap reflects the LLM's
    # high-band overconfidence drift on "addressed"
    # claims (P1: conf=0.85 bin gap 0.85, conf=0.95
    # bin gap 0.283).
    assert (
        M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED
        > M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN
    )


def test_p0_4_addressee_directed_admits_at_threshold_above_0_7() -> None:
    """P0-4: an `addressee_target_match` with
    `addressed_to_assistant == True` and conf=0.71 admits
    (just above the new 0.7 threshold). This is the
    boundary case the P0-4 raise creates: the 0.5-0.6
    overconfidence band is now rejected; the 0.8-0.9
    and 0.9-1.0 bands still admit.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.71,
                addressed_to_assistant=True,
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 1
    assert admitted[0].observable == "addressee_target_match"


def test_p0_4_addressee_directed_rejects_at_0_4_under_new_rule() -> None:
    """P0-4 regression: under the v1 uniform 0.4 threshold,
    `addressed_to_assistant == True` at conf=0.4 admitted.
    P0-4 raises the bar to 0.7, so the same entry now
    rejects. This is the v1→P0-4 behavior change.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.4,  # would admit under v1
                addressed_to_assistant=True,
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    # P0-4 rejects: 0.4 < 0.7.
    assert admitted == []
    # Diagnostic counter: per-sub-class reject bucket.
    diag = state["m20_4_attribution_diagnostics"]
    assert (
        diag.get(
            "producer_reject_low_confidence_addressee_directed_total"
        )
        == 1
    )


def test_p0_4_addressee_not_directed_admits_at_v1_threshold() -> None:
    """P0-4: `addressed_to_assistant == False` keeps the
    v1 0.4 admit threshold. The LLM is 100% precise on
    "not addressed" claims (P1), so the standard bar is
    appropriate.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.4,
                addressed_to_assistant=False,  # not addressed
            )
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 1
    diag = state["m20_4_attribution_diagnostics"]
    assert (
        diag.get(
            "producer_admit_addressee_not_directed_total"
        )
        == 1
    )


def test_p0_4_reaction_admit_rule_unchanged() -> None:
    """P0-4: reaction admit rule is unchanged (0.4 across
    the board). The reaction joint-axis asymmetry is in
    the LLM's emit decision (50% no-emit rate, P1), not
    in the admit calibration. P0-4 does not change the
    reaction admit threshold.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.4,
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            ),
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.39,  # just below
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            ),
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 1
    assert admitted[0].observable == "reaction_attribution_match"


def test_p0_4_mixed_batch_subclass_split() -> None:
    """P0-4: a mixed surface with both sub-classes admits
    only the ones above the per-sub-class threshold.

    The fixture mirrors the bqxsmofri surfaced distribution:
    - 1 "not addressed" at conf=0.95 (admit; v1 threshold)
    - 1 "addressed" at conf=0.95 (admit; P0-4 threshold 0.7)
    - 1 "addressed" at conf=0.4 (reject under P0-4;
      would have admitted under v1)
    - 1 reaction at conf=0.4 (admit; unchanged)
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.95,
                addressed_to_assistant=False,  # admit
            ),
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.95,
                addressed_to_assistant=True,  # admit
            ),
            _entry(
                kind=KIND_ADDRESSEE,
                turn_index=0,
                confidence=0.4,  # reject under P0-4
                addressed_to_assistant=True,
            ),
            _entry(
                kind=KIND_REACTION,
                turn_index=0,
                confidence=0.4,  # admit (unchanged)
                reaction_to_turn_id="turn_0",
                reaction_to_participant_id="alice",
            ),
        ]
    }
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
        inbound_excerpt="hi",
        at="2026-06-06T00:00:00Z",
    )
    assert len(admitted) == 3
    # Per-sub-class diagnostic counters:
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("producer_admit_total") == 3
    assert (
        diag.get("producer_admit_addressee_directed_total")
        == 1
    )
    assert (
        diag.get("producer_admit_addressee_not_directed_total")
        == 1
    )
    assert diag.get("producer_admit_reaction_total") == 1
    assert (
        diag.get(
            "producer_reject_low_confidence_addressee_directed_total"
        )
        == 1
    )


def test_p0_4_admit_threshold_helper_for_kind_and_subclass() -> None:
    """`_admit_threshold_for` returns the per-sub-class
    threshold. The helper is the single source of truth
    for the producer's admit calibration.
    """
    from segmentum.dialogue.runtime.m20_4_attribution import (
        _admit_threshold_for,
    )
    # addressee + addressed_to_assistant == True: 0.7
    assert (
        _admit_threshold_for(
            kind="addressee", addressed_to_assistant=True
        )
        == 0.7
    )
    # addressee + addressed_to_assistant == False: 0.4
    assert (
        _admit_threshold_for(
            kind="addressee", addressed_to_assistant=False
        )
        == 0.4
    )
    # reaction (any sub-class is None for reaction): 0.4
    assert _admit_threshold_for(kind="reaction") == 0.4
    assert (
        _admit_threshold_for(
            kind="reaction", addressed_to_assistant=None
        )
        == 0.4
    )
    # unknown kind: 0.4 (no-regression fallback)
    assert _admit_threshold_for(kind="unknown_kind") == 0.4
    # non-string kind: 0.4 (defensive)
    assert _admit_threshold_for(kind=None) == 0.4  # type: ignore[arg-type]
    # Whitespace + case insensitive on the kind
    assert (
        _admit_threshold_for(
            kind="  Addressee  ", addressed_to_assistant=True
        )
        == 0.7
    )


# === Bundle aggregation (M20.4 v2, 2026-06-11) ==============
#
# The bundle aggregation path is a second admit
# route in the M20.4 producer. The bundle memory
# is a separate M20.4 owner (cap 24) that mirrors
# addressee-directed M18.7 surface entries. The
# producer dispatches bundle admit when the
# decayed `aggregated_support` is >= 0.85 AND
# `max_single_support` < 0.7 AND
# `unique_emit_count` >= 2.
#
# Tests cover:
#  - T1-T3: append_bundle_memory contract (no-op
#    on bad input, cap at 24).
#  - T4-T5: _bundle_aggregated_support math
#    (empty, decay, dedup by commit_id).
#  - T6-T8: bundle admit fires / rejects per
#    rule (synthetic 12-turn fixtures).
#  - T9: bundle is for the `addressee` kind only
#    (reaction does not trigger).
#  - T10: v1 admission event byte-identity
#    (default `aggregation_kind="single_strong"`).


def test_t1_append_bundle_memory_skips_not_addressed_entry() -> None:
    """T1: append_bundle_memory no-ops on
    `addressed_to_assistant=False` (D6).
    """
    state: dict = {}
    not_addressed = _entry(
        kind=KIND_ADDRESSEE,
        turn_index=0,
        confidence=0.55,
        addressed_to_assistant=False,
    )
    append_bundle_memory(state=state, entry=not_addressed)
    assert state.get("m20_4_bundle_memory", []) == []


def test_t2_append_bundle_memory_skips_empty_commit_id() -> None:
    """T2: append_bundle_memory no-ops on empty
    `commit_id` (M18.7 contract violation).
    """
    state: dict = {}
    entry = _entry(
        kind=KIND_ADDRESSEE,
        turn_index=0,
        confidence=0.55,
        addressed_to_assistant=True,
    )
    entry["commit_id"] = ""  # simulate contract violation
    append_bundle_memory(state=state, entry=entry)
    assert state.get("m20_4_bundle_memory", []) == []


def test_t3_append_bundle_memory_caps_at_24_oldest_evicted() -> None:
    """T3: bundle memory caps at 24; oldest evicted."""
    state: dict = {}
    for i in range(30):
        entry = _entry(
            kind=KIND_ADDRESSEE,
            turn_index=i,
            confidence=0.5,
            addressed_to_assistant=True,
        )
        append_bundle_memory(state=state, entry=entry)
    memory = state.get("m20_4_bundle_memory", [])
    assert len(memory) == M20_4_BUNDLE_MEMORY_CAP
    # The first 6 (i=0..5) were evicted; the kept
    # entries are i=6..29.
    assert memory[0]["turn_index"] == 6
    assert memory[-1]["turn_index"] == 29


def test_t4_bundle_aggregated_support_empty_memory_returns_zero() -> None:
    """T4: empty memory → (0.0, 0.0, 0)."""
    agg, max_single, unique = _bundle_aggregated_support(
        bundle_memory=[],
        current_turn_index=10,
    )
    assert agg == 0.0
    assert max_single == 0.0
    assert unique == 0


def test_t5_bundle_aggregated_support_decay_and_dedup() -> None:
    """T5: decayed sum + dedup by commit_id + max_single.

    3 unique emits at confidence 0.55, turns 0/1/2,
    current_turn=3:
      emit_turn=2: weight = 0.85**1 = 0.85
      emit_turn=1: weight = 0.85**2 = 0.7225
      emit_turn=0: weight = 0.85**3 = 0.614125
      agg ≈ 0.55 * (0.85 + 0.7225 + 0.614125)
        ≈ 0.55 * 2.186625 ≈ 1.2026
      max_single = 0.55
      unique = 3
    """
    memory = [
        {"turn_index": 0, "commit_id": "c0",
         "confidence": 0.55, "participant_id": "alice"},
        {"turn_index": 1, "commit_id": "c1",
         "confidence": 0.55, "participant_id": "alice"},
        {"turn_index": 2, "commit_id": "c2",
         "confidence": 0.55, "participant_id": "alice"},
        # Duplicate commit_id (the redundancy guard):
        {"turn_index": 2, "commit_id": "c2",
         "confidence": 0.55, "participant_id": "alice"},
    ]
    agg, max_single, unique = _bundle_aggregated_support(
        bundle_memory=memory, current_turn_index=3
    )
    expected_weight = sum(
        M20_4_BUNDLE_DECAY_BASE ** (3 - t) for t in (0, 1, 2)
    )
    assert abs(agg - 0.55 * expected_weight) < 1e-9
    assert max_single == 0.55
    assert unique == 3  # the duplicate was deduped


def test_t6_bundle_admit_fires_on_three_weak_addressed_emits() -> None:
    """T6: 3 emits at conf 0.55 (each below 0.7) +
    max_single 0.55 (< 0.7) + unique 3 (>= 2) →
    bundle admit fires.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE, turn_index=0,
                confidence=0.55, addressed_to_assistant=True,
            ),
            _entry(
                kind=KIND_ADDRESSEE, turn_index=1,
                confidence=0.55, addressed_to_assistant=True,
            ),
            _entry(
                kind=KIND_ADDRESSEE, turn_index=2,
                confidence=0.55, addressed_to_assistant=True,
            ),
        ],
    }
    for entry in state["m18_7_attribution_hypotheses"]:
        append_bundle_memory(state=state, entry=entry)
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=3,
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    # The v1 single-strong path admits ZERO (0.55
    # < 0.7). The bundle path admits ONE.
    assert diag.get("producer_admit_single_strong_total", 0) == 0
    assert diag.get("producer_admit_bundle_weak_total", 0) == 1
    assert diag.get("producer_admit_total", 0) == 1
    # The admitted commitment has the
    # `aggregation_kind` stamped on its payload.
    assert len(admitted) == 1
    payload = admitted[0].observable_payload or {}
    assert payload.get("aggregation_kind") == M20_4_AGGREGATION_BUNDLE_WEAK


def test_t7_bundle_admit_rejects_when_max_single_not_below_threshold() -> None:
    """T7: when any single emit >= 0.7, the v1
    single-strong path takes over and the bundle
    path does NOT fire (D7).
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            # Below 0.7: weak
            _entry(
                kind=KIND_ADDRESSEE, turn_index=0,
                confidence=0.55, addressed_to_assistant=True,
            ),
            # Below 0.7: weak
            _entry(
                kind=KIND_ADDRESSEE, turn_index=1,
                confidence=0.55, addressed_to_assistant=True,
            ),
            # >= 0.7: strong (this one is admitted by v1)
            _entry(
                kind=KIND_ADDRESSEE, turn_index=2,
                confidence=0.9, addressed_to_assistant=True,
            ),
        ],
    }
    for entry in state["m18_7_attribution_hypotheses"]:
        append_bundle_memory(state=state, entry=entry)
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=3,
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    # The v1 single-strong path admitted the 0.9
    # emit; the bundle path did NOT fire (max_single
    # is 0.9, >= 0.7).
    assert diag.get("producer_admit_single_strong_total", 0) == 1
    assert diag.get("producer_admit_bundle_weak_total", 0) == 0
    assert diag.get("producer_admit_total", 0) == 1
    # No bundle rejection counter bumps because
    # the bundle dispatch doesn't run when v1
    # already admitted — actually it does run
    # (always runs on the producer), and the
    # max_single gate fires.
    assert diag.get("producer_bundle_reject_total", 0) == 1
    by_gate = diag.get("producer_bundle_reject_by_gate", {})
    assert by_gate.get("max_single_not_below_threshold", 0) == 1


def test_t8_bundle_admit_rejects_when_unique_count_below_min() -> None:
    """T8: only 1 unique emit → bundle rule fails
    the unique_count gate.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            _entry(
                kind=KIND_ADDRESSEE, turn_index=0,
                confidence=0.55, addressed_to_assistant=True,
            ),
        ],
    }
    for entry in state["m18_7_attribution_hypotheses"]:
        append_bundle_memory(state=state, entry=entry)
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=1,
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert diag.get("producer_admit_bundle_weak_total", 0) == 0
    assert diag.get("producer_bundle_reject_total", 0) == 1
    by_gate = diag.get("producer_bundle_reject_by_gate", {})
    # With 1 emit at conf 0.55, current_turn=1,
    # emit_turn=0, decay weight 0.85 → agg ≈
    # 0.4675, which is below the 0.85 threshold.
    # The `aggregated_below_threshold` gate is the
    # binding constraint.
    assert by_gate.get("aggregated_below_threshold", 0) == 1
    assert admitted == []


def test_t9_bundle_does_not_fire_for_reaction_kind() -> None:
    """T9: bundle admit is for the `addressee` kind
    only (D5). Reaction emits do NOT enter the
    bundle memory and do NOT trigger bundle admit.
    """
    state: dict = {
        "m18_7_attribution_hypotheses": [
            # 3 reaction emits at conf 0.55 — must
            # NOT trigger bundle admit
            _entry(
                kind=KIND_REACTION, turn_index=0,
                confidence=0.55, is_about_assistant_claim=True,
            ),
            _entry(
                kind=KIND_REACTION, turn_index=1,
                confidence=0.55, is_about_assistant_claim=True,
            ),
            _entry(
                kind=KIND_REACTION, turn_index=2,
                confidence=0.55, is_about_assistant_claim=True,
            ),
        ],
    }
    for entry in state["m18_7_attribution_hypotheses"]:
        append_bundle_memory(state=state, entry=entry)
    admitted = produce_m20_4_attribution_commitments(
        state=state,
        bus=[],
        current_turn_id=3,
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    # Reaction emits are filtered out at the
    # append_bundle_memory boundary (D5: only
    # `kind == "addressee"` is mirrored). The
    # bundle memory is empty.
    assert state.get("m20_4_bundle_memory", []) == []
    # v1 single-strong path admits 0 (reaction at
    # 0.55 < 0.4 wait no — 0.55 > 0.4 so v1 admits).
    # Actually reaction admit threshold is 0.4
    # (not 0.7 — that's only for addressee-directed).
    # So v1 admits all 3.
    assert diag.get("producer_admit_single_strong_total", 0) == 3
    assert diag.get("producer_admit_bundle_weak_total", 0) == 0
    # All 3 admits are reaction.
    assert len(admitted) == 3


def test_t10_v1_admission_event_default_aggregation_kind_is_single_strong() -> None:
    """T10: v1 byte-identity — the admission event
    builder's default `aggregation_kind` is
    `"single_strong"`, so v1 callers see no
    observable change.
    """
    from segmentum.dialogue.runtime.active_commitment import (
        ActiveCommitment,
    )
    commitment = ActiveCommitment(
        commit_id="abc",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_addressee_0",
        layer="B_per_turn_commitment",
        observable="addressee_target_match",
        observable_payload={},
        target={"m18_7_commit_id": "c0"},
        due_at={"kind": "next_turn"},
        priority=0.9,
        confidence=0.9,
        evidence_refs=(),
        created_turn=0,
        created_at="2026-06-11T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )
    # Default kwarg → `"single_strong"`.
    event = build_addressee_target_match_admitted_event(
        turn_index=0,
        commitment=commitment,
        at="2026-06-11T00:00:00Z",
    )
    assert event["aggregation_kind"] == M20_4_AGGREGATION_SINGLE_STRONG
    # The new field is present (additive) but the
    # rest of the event shape is unchanged.
    assert event["type"] == "AddresseeTargetMatchAdmitted"
    assert event["commit_id"] == "abc"
    # Explicit kwarg overrides the default.
    event_bundle = build_addressee_target_match_admitted_event(
        turn_index=0,
        commitment=commitment,
        at="2026-06-11T00:00:00Z",
        aggregation_kind=M20_4_AGGREGATION_BUNDLE_WEAK,
    )
    assert event_bundle["aggregation_kind"] == M20_4_AGGREGATION_BUNDLE_WEAK


