"""M20.0 ActiveCommitment meta-contract: schema, registry, observable, adapter.

This module is the schema-only milestone. It freezes:

- the `ActiveCommitment` dataclass shape
- `CommitmentRegistry` v1 (10 owners with accepts_layers / accepts_source_kinds)
- `Observable` v1 (11 observables with settler hints)
- `CommitmentPhase` enum
- `reason_codes` v1
- `engineering_proxy_label` v1
- `ALLOWED_SOURCE_KINDS`, `ALLOWED_LAYERS` enums
- `ActiveCommitmentAdapter` admission path
- `compute_commit_id` deterministic sha1
- `wrap_self_response_expectation_proposal` M19.0 wrapper
- `record_active_commitment_event` next-turn state read
- `update_commitment_registry_diagnostics` counter helper

M20.0 is admission-only. It does NOT implement:
- settlers (M20.1)
- promotion / revocation / expiration (M20.2)
- actual owner storage writes (M20.1+)
- per-loop settler migration (M20.3 / M20.1.1)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


ALLOWED_SOURCE_KINDS: frozenset[str] = frozenset({"policy", "state", "episodic"})

ALLOWED_LAYERS: frozenset[str] = frozenset({
    "A_long_term_prior",
    "B_per_turn_commitment",
    "C_observation",
})

COMMITMENT_PHASE: frozenset[str] = frozenset({
    "created",
    "settled",
    "promoted",
    "revoked",
    "expired",
})

REASON_CODES_V1: frozenset[str] = frozenset({
    "policy_prior",
    "user_explicit_statement",
    "user_implied_signal",
    "self_expectation_formation",
    "self_repair_bridge",
    "mismatch_observation",
    "memory_dynamics_guidance",
    "m13_drive_signal",
    "outreach_intent",
    "boundary_check",
    "silent_resolution",
})

ENGINEERING_PROXY_LABELS_V1: frozenset[str] = frozenset({
    "mvp_local_prediction_lock",
    "mvp_local_self_expectation",
    "mvp_local_self_repair",
    "mvp_local_mismatch",
    "mvp_local_memory_dynamics",
    "mvp_local_m13_drive",
    "mvp_local_m15_episode",
    "mvp_local_boundary",
    "mvp_local_outreach",
})


COMMITMENT_REGISTRY_V1: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "policy_state": {
        "description": "durable priors and repair priors",
        "storage_hint": (
            "self_cognition.calibrated_tendencies | "
            "self_cognition.repair_priors | "
            "m9_memory_dynamics.control_guidance"
        ),
        "accepts_layers": ["A_long_term_prior", "C_observation"],
        "accepts_source_kinds": ["policy", "episodic"],
    },
    "m13_drive_state": {
        "description": "behavioral pull, traction, path patterns",
        "storage_hint": (
            "m13_drive_state.path_patterns_by_action | "
            "m13_drive_state.traction_by_action"
        ),
        "accepts_layers": ["A_long_term_prior", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
    },
    "m15_episode_ledger": {
        "description": "episodic memory ledger",
        "storage_hint": "m15_episode_ledger",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["episodic"],
    },
    "mismatch_memory_fast": {
        "description": "M19.0 fast-layer mismatch memory",
        "storage_hint": "self_expectation_state.mismatch_memory_fast",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["episodic", "state"],
    },
    "self_repair_expectation": {
        "description": "M19.1 mid-layer repair expectations",
        "storage_hint": "self_repair_expectation_state.expectations_tail",
        "accepts_layers": ["B_per_turn_commitment"],
        "accepts_source_kinds": ["state", "episodic"],
    },
    "self_cognition_calibrated_tendencies": {
        "description": "M19.3 slow-layer calibrated tendencies",
        "storage_hint": (
            "self_cognition.calibrated_tendencies | "
            "self_cognition.repair_priors"
        ),
        "accepts_layers": ["A_long_term_prior"],
        "accepts_source_kinds": ["policy", "episodic"],
    },
    "user_prediction_ledger": {
        "description": "M11/M17 user-side predictions",
        "storage_hint": "UserPredictionLedger.pending | confirmed | violated | uncertain",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
    },
    "memory_dynamics_control_guidance": {
        "description": "M9.0 control guidance floats",
        "storage_hint": "memory_dynamics.control_guidance",
        "accepts_layers": ["A_long_term_prior"],
        "accepts_source_kinds": ["policy", "state"],
    },
    "outreach_intent_registry": {
        "description": "M13.3 / M14.x outreach intents",
        "storage_hint": "outreach_intent_registry",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
    },
    "group_addressee_graph": {
        "description": "M18.2 addressee / target graph",
        "storage_hint": "addressee_graph",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
    },
})


OBSERVABLE_V1: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "expectation_outcome_match": {
        "payload_keys": ("source_expectation_id", "target_context", "outcome", "evidence_refs"),
        "settler_hint": "deterministic",
    },
    "prediction_error_band": {
        "payload_keys": ("prediction_id", "type", "band", "committed_confidence", "value"),
        "settler_hint": "deterministic",
    },
    "repair_bias_band": {
        "payload_keys": ("context", "band", "value"),
        "settler_hint": "deterministic",
    },
    "behavioral_pull_shift": {
        "payload_keys": ("action", "delta", "evidence_refs"),
        "settler_hint": "deterministic",
    },
    "mismatch_type_band": {
        "payload_keys": ("mismatch_type", "target_context", "support_count", "weighted_support"),
        "settler_hint": "deterministic",
    },
    "pacing_match": {
        "payload_keys": ("expected_pacing", "actual_pacing", "evidence_refs"),
        "settler_hint": "llm_judge",
    },
    "identity_voice_match": {
        "payload_keys": ("expected_voice", "actual_voice", "drift_target", "evidence_span", "evidence_refs"),
        "settler_hint": "llm_judge",
    },
    "boundary_handled": {
        "payload_keys": ("boundary_kind", "outcome", "evidence_refs"),
        "settler_hint": "llm_judge",
    },
    "initiative_timing_match": {
        "payload_keys": ("expected_window", "actual_window", "evidence_refs"),
        "settler_hint": "hybrid",
    },
    "silent_then_resolved": {
        "payload_keys": ("silence_turns", "resolution_kind", "evidence_refs"),
        "settler_hint": "deterministic",
    },
    "traction_delta_band": {
        "payload_keys": ("action", "delta", "context"),
        "settler_hint": "deterministic",
    },
})


@dataclass(frozen=True)
class ActiveCommitment:
    commit_id: str
    owner_id: str
    source_kind: str
    source_ref: str
    layer: str
    observable: str
    observable_payload: Mapping[str, Any]
    target: Mapping[str, Any]
    due_at: Mapping[str, Any] | None
    priority: float
    confidence: float
    evidence_refs: tuple[str, ...]
    created_turn: int
    created_at: str
    reason_codes: tuple[str, ...]
    engineering_proxy_label: str


def compute_commit_id(
    *,
    owner_id: str,
    source_ref: str,
    layer: str,
    observable: str,
    created_turn: int,
) -> str:
    canonical = f"{owner_id}|{source_ref}|{layer}|{observable}|{created_turn}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _string_list(value: Any, *, limit: int = 64) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _string(value: Any, *, default: str = "", limit: int = 240) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


class ActiveCommitmentAdapter:
    """Admission path for ActiveCommitment.

    M20.0 is admission-only. The adapter validates the schema, registry,
    observable, and evidence_refs, derives `commit_id` deterministically,
    and emits audit events. It does NOT write to any owner's storage
    bucket (M20.1+ implement settlers that do that). It does NOT settle,
    promote, revoke, or expire.
    """

    ENGINEERING_PROXY_LABEL: str = "mvp_local_active_commitment"

    def admit(
        self,
        *,
        proposal: Mapping[str, Any],
        turn_index: int,
        created_at: str,
    ) -> tuple[ActiveCommitment | None, dict[str, Any] | None]:
        rejection = self._validate_or_reject(
            proposal=proposal,
            turn_index=turn_index,
            created_at=created_at,
        )
        if rejection is not None:
            return None, rejection

        owner_id = str(proposal["owner_id"])
        source_kind = str(proposal["source_kind"])
        source_ref = str(proposal["source_ref"])
        layer = str(proposal["layer"])
        observable = str(proposal["observable"])
        commit_id = compute_commit_id(
            owner_id=owner_id,
            source_ref=source_ref,
            layer=layer,
            observable=observable,
            created_turn=turn_index,
        )
        evidence_refs = tuple(_string_list(proposal.get("evidence_refs"), limit=32))
        reason_codes = tuple(_string_list(proposal.get("reason_codes"), limit=16))
        proxy_label = str(proposal["engineering_proxy_label"])

        observable_payload = proposal.get("observable_payload") or {}
        if not isinstance(observable_payload, Mapping):
            observable_payload = {}

        target = proposal.get("target") or {}
        if not isinstance(target, Mapping):
            target = {}

        due_at = proposal.get("due_at")
        if due_at is not None and not isinstance(due_at, Mapping):
            due_at = None

        commitment = ActiveCommitment(
            commit_id=commit_id,
            owner_id=owner_id,
            source_kind=source_kind,
            source_ref=source_ref,
            layer=layer,
            observable=observable,
            observable_payload=MappingProxyType(dict(observable_payload)),
            target=MappingProxyType(dict(target)),
            due_at=MappingProxyType(dict(due_at)) if due_at else None,
            priority=_bounded_float(proposal.get("priority")),
            confidence=_bounded_float(proposal.get("confidence")),
            evidence_refs=evidence_refs,
            created_turn=turn_index,
            created_at=created_at,
            reason_codes=reason_codes,
            engineering_proxy_label=proxy_label,
        )
        return commitment, None

    def admit_batch(
        self,
        *,
        proposals: list[Mapping[str, Any]],
        turn_index: int,
        created_at: str,
    ) -> tuple[list[ActiveCommitment], list[dict[str, Any]]]:
        admitted: list[ActiveCommitment] = []
        rejected: list[dict[str, Any]] = []
        for proposal in proposals:
            commitment, rejection = self.admit(
                proposal=proposal,
                turn_index=turn_index,
                created_at=created_at,
            )
            if commitment is not None:
                admitted.append(commitment)
            if rejection is not None:
                rejected.append(rejection)
        return admitted, rejected

    def _validate_or_reject(
        self,
        *,
        proposal: Mapping[str, Any],
        turn_index: int,
        created_at: str,
    ) -> dict[str, Any] | None:
        owner_id = _string(proposal.get("owner_id"))
        if owner_id not in COMMITMENT_REGISTRY_V1:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_owner",
                created_at=created_at,
            )

        source_kind = _string(proposal.get("source_kind"))
        if source_kind not in ALLOWED_SOURCE_KINDS:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_source_kind",
                created_at=created_at,
            )

        source_ref = _string(proposal.get("source_ref"))
        if not source_ref:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="empty_source_ref",
                created_at=created_at,
            )

        layer = _string(proposal.get("layer"))
        if layer not in ALLOWED_LAYERS:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_layer",
                created_at=created_at,
            )

        observable = _string(proposal.get("observable"))
        if observable not in OBSERVABLE_V1:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_observable",
                created_at=created_at,
            )

        owner_row = COMMITMENT_REGISTRY_V1[owner_id]
        if layer not in owner_row.get("accepts_layers", []):
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="invalid_layer_for_owner",
                created_at=created_at,
            )
        if source_kind not in owner_row.get("accepts_source_kinds", []):
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="invalid_source_kind_for_owner",
                created_at=created_at,
            )

        evidence_refs = _string_list(proposal.get("evidence_refs"), limit=32)
        if not evidence_refs:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="empty_evidence_refs",
                created_at=created_at,
            )

        reason_codes = _string_list(proposal.get("reason_codes"), limit=16)
        if not reason_codes:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="empty_reason_codes",
                created_at=created_at,
            )
        for rc in reason_codes:
            if rc not in REASON_CODES_V1:
                return self._rejection(
                    proposal=proposal,
                    turn_index=turn_index,
                    reason_code="unknown_reason_code",
                    created_at=created_at,
                )

        proxy_label = _string(proposal.get("engineering_proxy_label"))
        if proxy_label not in ENGINEERING_PROXY_LABELS_V1:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_engineering_proxy_label",
                created_at=created_at,
            )

        return None

    def _rejection(
        self,
        *,
        proposal: Mapping[str, Any],
        turn_index: int,
        reason_code: str,
        created_at: str,
    ) -> dict[str, Any]:
        return {
            "type": "ActiveCommitmentRejected",
            "turn_index": turn_index,
            "proposed_owner_id": _string(proposal.get("owner_id")),
            "proposed_observable": _string(proposal.get("observable")),
            "reason_code": reason_code,
            "engineering_proxy_label": self.ENGINEERING_PROXY_LABEL,
            "at": created_at,
        }


def build_active_commitment_created_event(
    commitment: ActiveCommitment,
) -> dict[str, Any]:
    return {
        "type": "ActiveCommitmentCreated",
        "turn_index": commitment.created_turn,
        "commit_id": commitment.commit_id,
        "owner_id": commitment.owner_id,
        "source_kind": commitment.source_kind,
        "source_ref": commitment.source_ref,
        "layer": commitment.layer,
        "observable": commitment.observable,
        "priority": commitment.priority,
        "confidence": commitment.confidence,
        "evidence_refs": list(commitment.evidence_refs),
        "reason_codes": list(commitment.reason_codes),
        "engineering_proxy_label": commitment.engineering_proxy_label,
        "created_at": commitment.created_at,
    }


def wrap_self_response_expectation_proposal(
    proposal: Mapping[str, Any],
    *,
    created_turn: int,
) -> dict[str, Any] | None:
    """Convert an M19.0 self_response_expectation_proposal into an
    active_commitment_proposal dict suitable for the adapter.

    Returns None if the proposal lacks the required M19.0 fields.
    """
    if not isinstance(proposal, Mapping):
        return None
    proposal_id = _string(proposal.get("proposal_id"), limit=120)
    if not proposal_id:
        return None
    target_context = _string(proposal.get("target_context"), limit=120)
    if not target_context:
        return None

    evidence_refs = _string_list(proposal.get("evidence_refs"), limit=16)
    return {
        "owner_id": "mismatch_memory_fast",
        "source_kind": "state",
        "source_ref": proposal_id,
        "layer": "B_per_turn_commitment",
        "observable": "expectation_outcome_match",
        "observable_payload": {
            "source_expectation_id": proposal_id,
            "target_context": target_context,
            "outcome": "",
            "evidence_refs": evidence_refs,
        },
        "target": {"target_context": target_context},
        "due_at": {"kind": "next_turn"},
        "priority": _bounded_float(proposal.get("confidence"), default=0.5),
        "confidence": _bounded_float(proposal.get("confidence"), default=0.5),
        "evidence_refs": evidence_refs or [proposal_id],
        "reason_codes": ["self_expectation_formation"],
        "engineering_proxy_label": "mvp_local_self_expectation",
    }


def record_active_commitment_event(state: dict, event: dict) -> None:
    if not isinstance(state, dict):
        return
    audit = state.get("commitment_registry_audit_tail")
    if not isinstance(audit, dict):
        audit = {}
    events = audit.get("events")
    if not isinstance(events, list):
        events = []
    events.append(dict(event))
    audit["events"] = events[-40:]
    audit["last_event"] = dict(event)
    state["commitment_registry_audit_tail"] = audit


def update_commitment_registry_diagnostics(
    state: dict,
    *,
    admitted: int = 0,
    rejected_by_reason_code: Mapping[str, int] | None = None,
) -> None:
    if not isinstance(state, dict):
        return
    index = state.get("commitment_registry_index")
    if not isinstance(index, dict):
        index = {}

    index["commitment_registry_owner_count"] = len(COMMITMENT_REGISTRY_V1)
    index["commitment_registry_observable_count"] = len(OBSERVABLE_V1)

    index["active_commitment_created_total"] = int(
        index.get("active_commitment_created_total", 0)
    ) + int(admitted)

    rejected_counts = index.get("active_commitment_rejected_by_reason_code")
    if not isinstance(rejected_counts, dict):
        rejected_counts = {}
    if rejected_by_reason_code:
        for code, count in rejected_by_reason_code.items():
            if not isinstance(code, str) or not code:
                continue
            rejected_counts[code] = int(rejected_counts.get(code, 0)) + int(count)
    index["active_commitment_rejected_by_reason_code"] = dict(rejected_counts)
    index["active_commitment_rejected_total"] = sum(rejected_counts.values())

    state["commitment_registry_index"] = index


__all__ = [
    "ALLOWED_LAYERS",
    "ALLOWED_SOURCE_KINDS",
    "ActiveCommitment",
    "ActiveCommitmentAdapter",
    "COMMITMENT_PHASE",
    "COMMITMENT_REGISTRY_V1",
    "ENGINEERING_PROXY_LABELS_V1",
    "OBSERVABLE_V1",
    "REASON_CODES_V1",
    "build_active_commitment_created_event",
    "compute_commit_id",
    "record_active_commitment_event",
    "update_commitment_registry_diagnostics",
    "wrap_self_response_expectation_proposal",
]
