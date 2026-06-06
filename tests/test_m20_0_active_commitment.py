"""Tests for M20.0 ActiveCommitment meta-contract (schema, registry, observable).

These tests cover the schema-only milestone. M20.0 freezes:

- the `ActiveCommitment` dataclass shape
- `CommitmentRegistry` v1 (10 owners with accepts_layers / accepts_source_kinds)
- `Observable` v1 (11 observables with settler hints)
- `CommitmentPhase` enum
- `reason_codes` v1
- `engineering_proxy_label` v1
- `ActiveCommitmentAdapter` admission path

M20.0 is admission-only. Settler / promotion / revocation tests
are explicitly deferred to M20.1 / M20.2; this file MUST NOT stub
them silently.
"""

from __future__ import annotations

import inspect

from segmentum.dialogue.runtime.active_commitment import (
    ALLOWED_LAYERS,
    ALLOWED_SOURCE_KINDS,
    ActiveCommitment,
    ActiveCommitmentAdapter,
    COMMITMENT_PHASE,
    COMMITMENT_REGISTRY_V1,
    ENGINEERING_PROXY_LABELS_V1,
    OBSERVABLE_V1,
    REASON_CODES_V1,
    build_active_commitment_created_event,
    compute_commit_id,
    record_active_commitment_event,
    update_commitment_registry_diagnostics,
    wrap_self_response_expectation_proposal,
)
from segmentum.dialogue.runtime.mvp_loop import (
    normalize_conscious_turn_plan,
)


def test_active_commitment_schema_is_frozen() -> None:
    fields = set(ActiveCommitment.__dataclass_fields__.keys())
    expected = {
        "commit_id",
        "owner_id",
        "source_kind",
        "source_ref",
        "layer",
        "observable",
        "observable_payload",
        "target",
        "due_at",
        "priority",
        "confidence",
        "evidence_refs",
        "created_turn",
        "created_at",
        "reason_codes",
        "engineering_proxy_label",
    }
    assert expected.issubset(fields)
    assert ActiveCommitment.__dataclass_params__.frozen is True
    # dataclass with frozen=True rejects attribute assignment.
    instance = ActiveCommitment(
        commit_id="x",
        owner_id="policy_state",
        source_kind="policy",
        source_ref="ref1",
        layer="A_long_term_prior",
        observable="repair_bias_band",
        observable_payload={},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref1",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_memory_dynamics",
    )
    try:
        instance.commit_id = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ActiveCommitment is not frozen")


def test_commit_id_is_deterministic_from_inputs() -> None:
    a = compute_commit_id(
        owner_id="mismatch_memory_fast",
        source_ref="self_exp_1",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        created_turn=7,
    )
    b = compute_commit_id(
        owner_id="mismatch_memory_fast",
        source_ref="self_exp_1",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        created_turn=7,
    )
    assert a == b
    assert len(a) == 40  # sha1 hex length
    # different inputs -> different commit_id
    c = compute_commit_id(
        owner_id="mismatch_memory_fast",
        source_ref="self_exp_2",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        created_turn=7,
    )
    assert a != c


def test_commitment_registry_v1_has_ten_owners() -> None:
    assert len(COMMITMENT_REGISTRY_V1) == 10
    expected_owners = {
        "policy_state",
        "m13_drive_state",
        "m15_episode_ledger",
        "mismatch_memory_fast",
        "self_repair_expectation",
        "self_cognition_calibrated_tendencies",
        "user_prediction_ledger",
        "memory_dynamics_control_guidance",
        "outreach_intent_registry",
        "group_addressee_graph",
    }
    assert set(COMMITMENT_REGISTRY_V1.keys()) == expected_owners


def test_commitment_registry_owner_accepts_declared_layer_and_source_kind() -> None:
    # m13_drive_state accepts A_long_term_prior, C_observation
    # and source kinds state, episodic.
    row = COMMITMENT_REGISTRY_V1["m13_drive_state"]
    assert "A_long_term_prior" in row["accepts_layers"]
    assert "C_observation" in row["accepts_layers"]
    assert "B_per_turn_commitment" not in row["accepts_layers"]
    assert "state" in row["accepts_source_kinds"]
    assert "episodic" in row["accepts_source_kinds"]
    assert "policy" not in row["accepts_source_kinds"]


def test_commitment_registry_rejects_unknown_owner() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "unknown_owner_xyz",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": ["ref1"],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=7,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["type"] == "ActiveCommitmentRejected"
    assert rejection["reason_code"] == "unknown_owner"
    assert rejection["turn_index"] == 7


def test_commitment_registry_rejects_invalid_layer_for_owner() -> None:
    # policy_state accepts A_long_term_prior but not B_per_turn_commitment
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "B_per_turn_commitment",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": ["ref1"],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "invalid_layer_for_owner"


def test_commitment_registry_rejects_invalid_source_kind_for_owner() -> None:
    # m15_episode_ledger accepts source_kind "episodic" only
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "m15_episode_ledger",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "B_per_turn_commitment",
        "observable": "behavioral_pull_shift",
        "observable_payload": {"action": "x", "delta": 0.1, "evidence_refs": []},
        "evidence_refs": ["ref1"],
        "reason_codes": ["m13_drive_signal"],
        "engineering_proxy_label": "mvp_local_m15_episode",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "invalid_source_kind_for_owner"


def test_observable_v1_has_eleven_observables() -> None:
    assert len(OBSERVABLE_V1) == 11
    expected_observables = {
        "expectation_outcome_match",
        "prediction_error_band",
        "repair_bias_band",
        "behavioral_pull_shift",
        "mismatch_type_band",
        "pacing_match",
        "identity_voice_match",
        "boundary_handled",
        "initiative_timing_match",
        "silent_then_resolved",
        "traction_delta_band",
    }
    assert set(OBSERVABLE_V1.keys()) == expected_observables


def test_observable_v1_rejects_unknown_observable() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "made_up_observable",
        "observable_payload": {},
        "evidence_refs": ["ref1"],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "unknown_observable"


def test_evidence_refs_must_be_non_empty() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": [],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "empty_evidence_refs"


def test_reason_codes_must_be_bounded() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": ["ref1"],
        "reason_codes": ["nonsense_code"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "unknown_reason_code"


def test_reason_codes_must_be_non_empty() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": ["ref1"],
        "reason_codes": [],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "empty_reason_codes"


def test_engineering_proxy_label_must_be_bounded() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "policy_state",
        "source_kind": "policy",
        "source_ref": "ref1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "x", "band": "low", "value": 0.1},
        "evidence_refs": ["ref1"],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "made_up_label",
        "priority": 0.5,
        "confidence": 0.5,
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert commitment is None
    assert rejection is not None
    assert rejection["reason_code"] == "unknown_engineering_proxy_label"


def test_active_commitment_adapter_emits_created_audit_event() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "mismatch_memory_fast",
        "source_kind": "state",
        "source_ref": "self_exp_1",
        "layer": "B_per_turn_commitment",
        "observable": "expectation_outcome_match",
        "observable_payload": {
            "source_expectation_id": "self_exp_1",
            "target_context": "short_casual_reply",
            "outcome": "violated",
            "evidence_refs": ["turn_3"],
        },
        "target": {"target_context": "short_casual_reply"},
        "due_at": {"kind": "next_turn"},
        "priority": 0.6,
        "confidence": 0.7,
        "evidence_refs": ["turn_3"],
        "reason_codes": ["self_expectation_formation"],
        "engineering_proxy_label": "mvp_local_self_expectation",
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=3,
        created_at="2026-06-06T00:00:00Z",
    )
    assert rejection is None
    assert commitment is not None
    event = build_active_commitment_created_event(commitment)
    assert event["type"] == "ActiveCommitmentCreated"
    assert event["turn_index"] == 3
    assert event["owner_id"] == "mismatch_memory_fast"
    assert event["source_kind"] == "state"
    assert event["observable"] == "expectation_outcome_match"
    assert event["layer"] == "B_per_turn_commitment"
    assert event["priority"] == 0.6
    assert event["confidence"] == 0.7
    assert event["evidence_refs"] == ["turn_3"]
    assert event["reason_codes"] == ["self_expectation_formation"]
    assert event["engineering_proxy_label"] == "mvp_local_self_expectation"
    assert event["commit_id"] == compute_commit_id(
        owner_id="mismatch_memory_fast",
        source_ref="self_exp_1",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        created_turn=3,
    )


def test_active_commitment_adapter_clamps_priority_and_confidence() -> None:
    adapter = ActiveCommitmentAdapter()
    proposal = {
        "owner_id": "mismatch_memory_fast",
        "source_kind": "state",
        "source_ref": "self_exp_x",
        "layer": "B_per_turn_commitment",
        "observable": "expectation_outcome_match",
        "observable_payload": {
            "source_expectation_id": "self_exp_x",
            "target_context": "x",
            "outcome": "",
            "evidence_refs": [],
        },
        "target": {"target_context": "x"},
        "due_at": None,
        "priority": 5.0,
        "confidence": -0.5,
        "evidence_refs": ["ref1"],
        "reason_codes": ["self_expectation_formation"],
        "engineering_proxy_label": "mvp_local_self_expectation",
    }
    commitment, rejection = adapter.admit(
        proposal=proposal,
        turn_index=0,
        created_at="2026-06-06T00:00:00Z",
    )
    assert rejection is None
    assert commitment is not None
    assert commitment.priority == 1.0
    assert commitment.confidence == 0.0


def test_active_commitment_adapter_does_not_settle_or_promote() -> None:
    # M20.0 admission-only: there is no settle, promote, revoke, or expire
    # call on the adapter. Asserting this prevents accidental scope creep.
    adapter = ActiveCommitmentAdapter()
    forbidden_methods = {"settle", "promote", "revoke", "expire", "dispatch"}
    for name in forbidden_methods:
        assert not hasattr(adapter, name), (
            f"ActiveCommitmentAdapter must not have a {name} method in M20.0"
        )


def test_record_active_commitment_event_updates_last_event_pointer() -> None:
    state: dict = {}
    event = {
        "type": "ActiveCommitmentCreated",
        "turn_index": 4,
        "commit_id": "abc",
        "owner_id": "mismatch_memory_fast",
        "source_kind": "state",
        "layer": "B_per_turn_commitment",
        "observable": "expectation_outcome_match",
        "engineering_proxy_label": "mvp_local_self_expectation",
    }
    record_active_commitment_event(state, event)
    audit = state["commitment_registry_audit_tail"]
    assert audit["last_event"] == event
    assert audit["events"] == [event]


def test_commitment_registry_index_diagnostic_exposes_counts() -> None:
    state: dict = {}
    update_commitment_registry_diagnostics(
        state,
        admitted=3,
        rejected_by_reason_code={"unknown_owner": 1, "empty_evidence_refs": 2},
    )
    index = state["commitment_registry_index"]
    assert index["commitment_registry_owner_count"] == len(COMMITMENT_REGISTRY_V1)
    assert index["commitment_registry_observable_count"] == len(OBSERVABLE_V1)
    assert index["active_commitment_created_total"] == 3
    assert index["active_commitment_rejected_total"] == 3
    assert index["active_commitment_rejected_by_reason_code"] == {
        "unknown_owner": 1,
        "empty_evidence_refs": 2,
    }

    # second call accumulates
    update_commitment_registry_diagnostics(
        state,
        admitted=1,
        rejected_by_reason_code={"unknown_observable": 1},
    )
    assert state["commitment_registry_index"]["active_commitment_created_total"] == 4
    assert state["commitment_registry_index"]["active_commitment_rejected_total"] == 4
    assert state["commitment_registry_index"]["active_commitment_rejected_by_reason_code"][
        "unknown_observable"
    ] == 1


def test_m19_0_self_response_expectation_proposal_wraps_to_active_commitment() -> None:
    sre = {
        "proposal_id": "self_exp_42",
        "target_context": "short_casual_reply",
        "expected_outcome": "casual_turn_stays_light_and_short",
        "expected_reply_quality": "light",
        "confidence": 0.8,
        "evidence_refs": ["turn_41_bus"],
        "reason_codes": [],
        "engineering_proxy_label": "",
    }
    wrapped = wrap_self_response_expectation_proposal(
        sre,
        created_turn=42,
    )
    assert wrapped is not None
    assert wrapped["owner_id"] == "mismatch_memory_fast"
    assert wrapped["observable"] == "expectation_outcome_match"
    assert wrapped["source_ref"] == "self_exp_42"
    assert wrapped["layer"] == "B_per_turn_commitment"
    assert wrapped["source_kind"] == "state"
    assert wrapped["observable_payload"]["source_expectation_id"] == "self_exp_42"
    assert wrapped["observable_payload"]["target_context"] == "short_casual_reply"
    assert wrapped["evidence_refs"] == ["turn_41_bus"]
    assert wrapped["reason_codes"] == ["self_expectation_formation"]
    assert wrapped["engineering_proxy_label"] == "mvp_local_self_expectation"
    # M20.0 wraps into a known-valid active_commitment_proposal shape.
    adapter = ActiveCommitmentAdapter()
    commitment, rejection = adapter.admit(
        proposal=wrapped,
        turn_index=42,
        created_at="2026-06-06T00:00:00Z",
    )
    assert rejection is None
    assert commitment is not None
    assert commitment.observable == "expectation_outcome_match"
    assert commitment.owner_id == "mismatch_memory_fast"


def test_m19_0_wrapper_returns_none_for_invalid_proposal() -> None:
    assert wrap_self_response_expectation_proposal({}, created_turn=0) is None
    assert wrap_self_response_expectation_proposal(
        {"proposal_id": "p1"}, created_turn=0
    ) is None
    assert wrap_self_response_expectation_proposal(
        {"target_context": "x"}, created_turn=0
    ) is None
    assert wrap_self_response_expectation_proposal(
        None, created_turn=0  # type: ignore[arg-type]
    ) is None


def test_conscious_plan_accepts_empty_active_commitment_proposals() -> None:
    plan = normalize_conscious_turn_plan({})
    assert plan["active_commitment_proposals"] == []


def test_conscious_plan_accepts_active_commitment_proposals_list() -> None:
    raw = {
        "active_commitment_proposals": [
            {
                "owner_id": "mismatch_memory_fast",
                "source_kind": "state",
                "source_ref": "self_exp_1",
                "layer": "B_per_turn_commitment",
                "observable": "expectation_outcome_match",
            },
            {"owner_id": "garbage"},
        ]
    }
    plan = normalize_conscious_turn_plan(raw)
    assert len(plan["active_commitment_proposals"]) == 2
    assert plan["active_commitment_proposals"][0]["owner_id"] == "mismatch_memory_fast"


def test_conscious_plan_rejects_non_list_active_commitment_proposals() -> None:
    plan = normalize_conscious_turn_plan(
        {"active_commitment_proposals": "not a list"}
    )
    assert plan["active_commitment_proposals"] == []


def test_commitment_phase_enum_is_frozen() -> None:
    assert COMMITMENT_PHASE == frozenset({
        "created",
        "settled",
        "promoted",
        "revoked",
        "expired",
    })


def test_reason_codes_v1_is_frozen() -> None:
    assert "policy_prior" in REASON_CODES_V1
    assert "self_expectation_formation" in REASON_CODES_V1
    assert len(REASON_CODES_V1) == 11


def test_engineering_proxy_labels_v1_is_frozen() -> None:
    assert "mvp_local_self_expectation" in ENGINEERING_PROXY_LABELS_V1
    assert "mvp_local_active_commitment" not in ENGINEERING_PROXY_LABELS_V1
    assert len(ENGINEERING_PROXY_LABELS_V1) == 9


def test_allowed_source_kinds_is_frozen() -> None:
    assert ALLOWED_SOURCE_KINDS == frozenset({"policy", "state", "episodic"})


def test_allowed_layers_is_frozen() -> None:
    assert ALLOWED_LAYERS == frozenset({
        "A_long_term_prior",
        "B_per_turn_commitment",
        "C_observation",
    })


def test_active_commitment_module_does_not_import_settler_paths() -> None:
    # M20.0 is schema-only. The active_commitment module must not define
    # settler, dispatcher, or promotion machinery at code level. We
    # inspect the module AST (not the source text) so that the docstring's
    # explanatory references to M20.1 / M20.2 do not trigger a false
    # positive.
    import ast

    from segmentum.dialogue.runtime import active_commitment
    tree = ast.parse(inspect.getsource(active_commitment))

    forbidden_class_substrings = ("Settler", "Dispatcher", "Promoter", "Revoker")
    forbidden_function_substrings = (
        "settle_",
        "dispatch_",
        "promote_",
        "revoke_",
        "expire_",
    )
    forbidden_module_paths = (
        "active_commitment_settlers",
        "active_commitment_settlement",
    )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for fragment in forbidden_class_substrings:
                assert fragment not in node.name, (
                    f"active_commitment must not define class with name "
                    f"containing {fragment!r} in M20.0 (got {node.name!r})"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for fragment in forbidden_function_substrings:
                assert fragment not in node.name, (
                    f"active_commitment must not define function with name "
                    f"containing {fragment!r} in M20.0 (got {node.name!r})"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for forbidden in forbidden_module_paths:
                assert forbidden not in module, (
                    f"active_commitment must not import from {forbidden!r} "
                    f"in M20.0 (got {module!r})"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_module_paths:
                    assert forbidden not in alias.name, (
                        f"active_commitment must not import {forbidden!r} "
                        f"in M20.0 (got {alias.name!r})"
                    )
