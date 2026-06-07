"""M20.0 ActiveCommitment meta-contract: schema, registry, observable, adapter.
M20.1 adds: settler protocol, SettledValue / NoSettlement result types,
SettlementScheduler, owner observability writes, and per-observable
outcome / magnitude / reason-code freezes.

M20.0 is admission-only. It does NOT implement:
- settlers (M20.1)
- promotion / revocation / expiration (M20.2)
- actual owner storage writes (M20.1+)
- per-loop settler migration (M20.3 / M20.1.1)

M20.1 owns the observation half (settlers, scheduler, observability).
M20.1 does NOT mutate any long-term state bucket. It only writes
to `state["commitment_owner_observability"][owner_id][commit_id]`,
which is a diagnostic surface that M20.2 reads to drive graded
correction. M20.1 also does NOT implement promotion / microadjust /
revocation / expiration logic.

M20.2 adds: graded correction dispatcher (6 levels, per-owner
`graded_action_set`), and routing stubs that call into the existing
owner write paths. M20.2 ships no real write paths; M20.2.1 wires
the m13_drive_state / self_cognition_calibrated_tendencies scopes.

M20.3 adds (additive v2 vocabulary):
- `runtime_mode_state` owner row + observable (registry v2 bump)
- `outreach_intent_on` / `outreach_intent_off` observables
- `horizon` attribute on `ActiveCommitment` (v2-only, defaults to
  "next_turn" for v1 commitments)
- registry v2 `accepts_policy_correction` flag — let PolicyProducer-
  admitted `runtime_mode_state` commitments actually update the
  persistent mode state through the regular M20.2 dispatch path,
  bypassing M20.2 §2's general "policy -> expire" rule.

M20.3 is a layer above M20.0–M20.2. v1 rows in
`COMMITMENT_REGISTRY_V1` and `OBSERVABLE_V1` are unchanged; v2
additions live in `COMMITMENT_REGISTRY_V2` and `OBSERVABLE_V2`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol


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
    # M20.3 v1 additive label: PolicyProducer-admitted rows.
    "mvp_local_policy_admission",
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
        "graded_action_set": [],
    },
    "m13_drive_state": {
        "description": "behavioral pull, traction, path patterns",
        "storage_hint": (
            "m13_drive_state.path_patterns_by_action | "
            "m13_drive_state.traction_by_action"
        ),
        "accepts_layers": ["A_long_term_prior", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
        "graded_action_set": ["microadjust", "next_turn", "same_turn"],
    },
    "m15_episode_ledger": {
        "description": "episodic memory ledger",
        "storage_hint": "m15_episode_ledger",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["episodic"],
        "graded_action_set": ["microadjust", "next_turn"],
    },
    "mismatch_memory_fast": {
        "description": "M19.0 fast-layer mismatch memory",
        "storage_hint": "self_expectation_state.mismatch_memory_fast",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["episodic", "state"],
        "graded_action_set": ["microadjust", "next_turn", "revoke"],
    },
    "self_repair_expectation": {
        "description": "M19.1 mid-layer repair expectations",
        "storage_hint": "self_repair_expectation_state.expectations_tail",
        "accepts_layers": ["B_per_turn_commitment"],
        "accepts_source_kinds": ["state", "episodic"],
        "graded_action_set": ["next_turn", "same_turn", "slow_promote", "revoke"],
    },
    "self_cognition_calibrated_tendencies": {
        "description": "M19.3 slow-layer calibrated tendencies",
        "storage_hint": (
            "self_cognition.calibrated_tendencies | "
            "self_cognition.repair_priors"
        ),
        "accepts_layers": ["A_long_term_prior"],
        "accepts_source_kinds": ["policy", "episodic"],
        "graded_action_set": ["slow_promote", "revoke"],
    },
    "user_prediction_ledger": {
        "description": "M11/M17 user-side predictions",
        "storage_hint": "UserPredictionLedger.pending | confirmed | violated | uncertain",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
        "graded_action_set": ["microadjust", "next_turn"],
    },
    "memory_dynamics_control_guidance": {
        "description": "M9.0 control guidance floats",
        "storage_hint": "memory_dynamics.control_guidance",
        "accepts_layers": ["A_long_term_prior"],
        "accepts_source_kinds": ["policy", "state"],
        "graded_action_set": ["microadjust", "next_turn", "same_turn"],
    },
    "outreach_intent_registry": {
        "description": "M13.3 / M14.x outreach intents",
        "storage_hint": "outreach_intent_registry",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
        "graded_action_set": ["next_turn", "same_turn", "revoke"],
    },
    "group_addressee_graph": {
        "description": "M18.2 addressee / target graph",
        "storage_hint": "addressee_graph",
        "accepts_layers": ["B_per_turn_commitment", "C_observation"],
        "accepts_source_kinds": ["state", "episodic"],
        "graded_action_set": ["microadjust", "next_turn"],
    },
})


# === M20.2 GradedCorrection types and thresholds ========================

GRADED_CORRECTION_V1: frozenset[str] = frozenset({
    "microadjust",
    "next_turn",
    "same_turn",
    "slow_promote",
    "revoke",
    "expire",
})


# Magnitude -> level table (M20.2 §2). A change is a vocabulary bump.
# Magnitudes are clamped to [0.0, 1.0] before the table is consulted.
#
# N2 (M20.3 follow-up): the threshold boundaries (0.1, 0.3, 0.6,
# 0.85) and the level mapping are design-time constants, not
# calibrated against any empirical replay. They were chosen so
# the v1 table covered the full [0, 1] interval with five
# monotonic levels. A future M20.x milestone should run a
# replay-style calibration (e.g. on the M20.0 acceptance
# fixture) and re-pick the boundaries if needed. Changing this
# table is a M20.2 vocabulary bump.
_MAGNITUDE_LEVEL_TABLE: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.1, "expire"),
    (0.1, 0.3, "microadjust"),
    (0.3, 0.6, "next_turn"),
    (0.6, 0.85, "same_turn"),
    (0.85, 1.00001, "slow_promote"),
)


# Settler reason codes for the dispatcher side. M20.2 = M20.1's
# settler codes ∪ dispatcher codes. Defined after
# `ALL_SETTLEMENT_REASON_CODES_V1` so it can take the union.


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
    # M20.3 v2 attribute. Defaults to "next_turn" so v1 commitments
    # constructed without it continue to work. v2 producers (e.g.
    # PolicyProducer, the post-settled M19.x / M18.x hooks) may set
    # "same_turn_surface" or "natural_context".
    horizon: str = "next_turn"


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

        horizon = _string(proposal.get("horizon"), default="next_turn")
        if horizon not in HORIZON_V1:
            horizon = "next_turn"

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
            horizon=horizon,
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
        # M20.3 bump: consult the v2 registry, which is v1 ∪ new
        # owners. v1 owners pass unchanged; v2-only owners
        # (e.g. `runtime_mode_state`) are accepted.
        if owner_id not in COMMITMENT_REGISTRY_V2:
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
        if observable not in OBSERVABLE_V2:
            return self._rejection(
                proposal=proposal,
                turn_index=turn_index,
                reason_code="unknown_observable",
                created_at=created_at,
            )

        owner_row = COMMITMENT_REGISTRY_V2[owner_id]
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

    index["commitment_registry_owner_count"] = len(COMMITMENT_REGISTRY_V2)
    index["commitment_registry_observable_count"] = len(OBSERVABLE_V2)

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


def _record_long_pending_staleness(
    state: dict,
    *,
    commit_id: str,
    owner_id: str,
    observable: str,
    current_turn: int,
    created_turn: int,
    reason_code: str,
) -> None:
    """Surface long-pending commitment staleness in the diagnostic index.

    M20.3 N1 follow-up: when a commitment has been retried for at
    least `LONG_PENDING_STALENESS_TURNS` turns without settling
    (`settler_unavailable` or `no_eligible_observation`), record
    the staleness in `state["commitment_registry_index"]`. The
    scheduler does NOT change behavior (it still preserves the
    commitment per M20.1 §7); the operator can see the capacity
    drain via the diagnose surface.

    The counter is keyed by `commit_id` so a single commitment that
    is retried many times does not double-count: only the first
    time it crosses the staleness threshold is recorded.
    """
    if not isinstance(state, dict):
        return
    if current_turn - created_turn < LONG_PENDING_STALENESS_TURNS:
        return
    index = state.get("commitment_registry_index")
    if not isinstance(index, dict):
        index = {}
    long_pending = index.get("long_pending_commitments")
    if not isinstance(long_pending, dict):
        long_pending = {}
    if commit_id in long_pending:
        return  # already recorded
    long_pending[commit_id] = {
        "owner_id": owner_id,
        "observable": observable,
        "first_stale_turn": int(current_turn),
        "created_turn": int(created_turn),
        "reason_code": reason_code,
    }
    index["long_pending_commitments"] = long_pending
    index["long_pending_commitment_count"] = len(long_pending)
    state["commitment_registry_index"] = index


# === M20.1 settler protocol core =========================================
# M20.1 freezes the settler half of the unified-commitment loop. The
# scheduler and reference settlers speak through these types. M20.1 does
# not write to any long-term state bucket; it only writes to
# `state["commitment_owner_observability"][owner_id][commit_id]`.

SETTLER_TYPE_V1: frozenset[str] = frozenset({
    "deterministic",
    "llm_judge",
    "hybrid",
    "silent",
})


OUTCOME_V1: frozenset[str] = frozenset({
    "confirmed",
    "violated",
    "uncertain",
    "ambiguous",
})


# Per-observable bounded outcome set. A new outcome is a vocabulary bump.
OUTCOME_BY_OBSERVABLE_V1: Mapping[str, frozenset[str]] = MappingProxyType({
    "expectation_outcome_match": frozenset({"confirmed", "violated", "uncertain"}),
    "prediction_error_band": frozenset({"confirmed", "violated", "uncertain"}),
    "repair_bias_band": frozenset({"confirmed", "violated", "uncertain"}),
    "behavioral_pull_shift": frozenset({"confirmed", "violated", "uncertain"}),
    "mismatch_type_band": frozenset({"confirmed", "violated", "uncertain"}),
    "pacing_match": frozenset({"confirmed", "violated", "ambiguous"}),
    "identity_voice_match": frozenset({"confirmed", "violated", "ambiguous"}),
    "boundary_handled": frozenset({"confirmed", "violated", "ambiguous"}),
    "initiative_timing_match": frozenset({"confirmed", "violated", "uncertain"}),
    "silent_then_resolved": frozenset({"confirmed", "violated"}),
    "traction_delta_band": frozenset({"confirmed", "violated", "uncertain"}),
})


# Per-observable magnitude scale (§3a). All scales are non-zero.
# Binary / categorical observables use 1.0; bounded-delta observables
# use 0.5.
MAGNITUDE_SCALES_V1: Mapping[str, float] = MappingProxyType({
    "expectation_outcome_match": 1.0,
    "prediction_error_band": 1.0,
    "repair_bias_band": 1.0,
    "behavioral_pull_shift": 0.5,
    "mismatch_type_band": 1.0,
    "pacing_match": 1.0,
    "identity_voice_match": 1.0,
    "boundary_handled": 1.0,
    "initiative_timing_match": 1.0,
    "silent_then_resolved": 1.0,
    "traction_delta_band": 0.5,
})


# Settler-side reason codes v1. Additive to M20.0's REASON_CODES_V1.
# A `SettledValue` MUST include at least one of these. A `NoSettlement`
# MUST include at least one of `due_at_passed`, `settler_unavailable`,
# `no_eligible_observation`, or an M20.0 vocabulary code surfaced by
# the settler.
SETTLER_REASON_CODES_V1: frozenset[str] = frozenset({
    "settler_deterministic",
    "settler_llm_judge",
    "settler_hybrid_fallback",
    "settler_silent_carry_forward",
    "magnitude_defaulted",
    "evidence_ref_filtered",
    "due_at_passed",
    "settler_unavailable",
    "no_eligible_observation",
    "settler_hybrid_fallback_exhausted",
    "settler_llm_invalid_response",
    "observation_already_processed",
})


# All reason codes a settler may surface in a `SettledValue` or
# `NoSettlement`. M20.1 = M20.0 reason codes ∪ settler reason codes.
ALL_SETTLEMENT_REASON_CODES_V1: frozenset[str] = (
    REASON_CODES_V1 | SETTLER_REASON_CODES_V1
)


# Dispatcher reason codes (M20.2 §6). The dispatcher may surface any
# of these on a `CorrectionDeferred` or `CorrectionRejected` event.
DISPATCHER_REASON_CODES_V1: frozenset[str] = frozenset({
    "magnitude_below_threshold",
    "policy_source_no_correction",
    "ambiguous_outcome",
    "m19_3_already_promoted",
    "action_set_violation",
    "slow_promote_not_supported",
    "same_turn_not_advisory",
    "owner_state_unavailable",
    "unknown_owner",
})


# All reason codes a graded correction may surface across the unified
# commitment loop. M20.2 = M20.1's settlement codes ∪ dispatcher codes.
ALL_GRADED_CORRECTION_REASON_CODES_V1: frozenset[str] = (
    ALL_SETTLEMENT_REASON_CODES_V1 | DISPATCHER_REASON_CODES_V1
)


# Minimal eligibility window: a commitment whose `due_at` is
# `{"kind": "next_turn"}` is considered past when current_turn >
# created_turn + this constant. Frozen at 1 by M20.1 §3.
_NEXT_TURN_DUE_AT_WINDOW: int = 1


@dataclass(frozen=True)
class SettledValue:
    """Frozen result of a successful settlement attempt (M20.1 §1).

    The promoter (M20.2) reads `magnitude` to choose graded correction
    intensity. M20.1 freezes the field and emits it; the promotion
    semantics belong to M20.2.
    """

    commit_id: str
    outcome: str
    magnitude: float
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    at: str
    turn_index: int
    settler_type: str
    engineering_proxy_label: str


@dataclass(frozen=True)
class NoSettlement:
    """Frozen result when a settler declines to produce a `SettledValue`.

    This is the *only* non-result path in M20.1. The scheduler emits a
    `NoSettlementMade` audit event from this dataclass.
    """

    commit_id: str
    reason_code: str
    settler_type: str
    engineering_proxy_label: str
    at: str
    turn_index: int


class SettlerUnavailable(Exception):
    """Raised when a settler cannot run (e.g. LLM offline, invalid input).

    The scheduler catches this and converts to a `NoSettlement` with
    `reason_code="settler_unavailable"`.
    """


class Settler(Protocol):
    """Structural interface for a settler (M20.1 §5).

    All four settler types (deterministic, llm_judge, hybrid, silent)
    implement this protocol. The caller (scheduler) treats them
    uniformly.
    """

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        ...


# === M20.1 event builders ==============================================


def build_active_commitment_settled_event(
    settled: SettledValue,
) -> dict[str, Any]:
    """Build the `ActiveCommitmentSettled` audit envelope (M20.1 §10)."""
    return {
        "type": "ActiveCommitmentSettled",
        "turn_index": settled.turn_index,
        "commit_id": settled.commit_id,
        "outcome": settled.outcome,
        "magnitude": settled.magnitude,
        "evidence_refs": list(settled.evidence_refs),
        "reason_codes": list(settled.reason_codes),
        "settler_type": settled.settler_type,
        "engineering_proxy_label": settled.engineering_proxy_label,
        "at": settled.at,
    }


def build_no_settlement_made_event(
    no_settlement: NoSettlement,
) -> dict[str, Any]:
    """Build the `NoSettlementMade` audit envelope (M20.1 §6, §10)."""
    return {
        "type": "NoSettlementMade",
        "turn_index": no_settlement.turn_index,
        "commit_id": no_settlement.commit_id,
        "reason_code": no_settlement.reason_code,
        "settler_type": no_settlement.settler_type,
        "engineering_proxy_label": no_settlement.engineering_proxy_label,
        "at": no_settlement.at,
    }


# === M20.1 owner observability writes ==================================


def _ensure_owner_observability_map(
    state: dict,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the per-owner observability map, creating it on first read."""
    if not isinstance(state, dict):
        return {}
    observability = state.get("commitment_owner_observability")
    if not isinstance(observability, dict):
        observability = {}
        state["commitment_owner_observability"] = observability
    return observability


def init_owner_observability_for_commitment(
    state: dict,
    *,
    owner_id: str,
    commitment: ActiveCommitment,
) -> None:
    """Initialize the observability entry for a freshly admitted commitment.

    Called at admission time (M20.0) so the dispatcher (M20.2) can
    read the commitment data after the pending row is removed on
    settlement. The observability map is the only durable record of
    the commitment once the pending list drops it.

    This is an additive write; it does NOT touch any owner storage
    bucket. It only sets:
    - `commitment`: a bounded copy of the fields the dispatcher needs
    - `settled_value`: None (filled by M20.1 settlement)
    - `settlement_attempts`: 0
    - `dispatched`: False (M20.2 sets True after GradedCorrectionRouted)
    - `dispatched_at_turn`: None
    - `dispatched_correction_level`: None
    """
    if not isinstance(state, dict):
        return
    observability = _ensure_owner_observability_map(state)
    owner_row = observability.get(owner_id)
    if not isinstance(owner_row, dict):
        owner_row = {}
    prior = owner_row.get(commitment.commit_id)
    if not isinstance(prior, dict) or not prior.get("commitment"):
        prior = {
            "commitment": _commitment_to_observability_row(commitment),
            "settled_value": None,
            "settlement_attempts": 0,
            "last_attempt_turn_index": None,
            "last_attempt_reason_code": None,
            "dispatched": False,
            "dispatched_at_turn": None,
            "dispatched_correction_level": None,
        }
        owner_row[commitment.commit_id] = prior
    observability[owner_id] = owner_row


def _commitment_to_observability_row(commitment: ActiveCommitment) -> dict[str, Any]:
    """Build a bounded dict copy of the commitment for observability.

    M20.2's dispatcher only needs a small subset of fields. Storing
    the full `ActiveCommitment` would bloat state without value.
    """
    return {
        "commit_id": commitment.commit_id,
        "owner_id": commitment.owner_id,
        "source_kind": commitment.source_kind,
        "source_ref": commitment.source_ref,
        "layer": commitment.layer,
        "observable": commitment.observable,
        "observable_payload": dict(commitment.observable_payload),
        "target": dict(commitment.target),
        "due_at": dict(commitment.due_at) if commitment.due_at else None,
        "priority": commitment.priority,
        "confidence": commitment.confidence,
        "evidence_refs": list(commitment.evidence_refs),
        "reason_codes": list(commitment.reason_codes),
        "engineering_proxy_label": commitment.engineering_proxy_label,
        "created_turn": commitment.created_turn,
        "created_at": commitment.created_at,
        "horizon": commitment.horizon,
    }


def write_owner_observability(
    state: dict,
    *,
    owner_id: str,
    commit_id: str,
    settled_value: SettledValue | None,
    last_attempt_turn_index: int,
    last_attempt_reason_code: str,
) -> None:
    """Write a single (owner, commit) observability entry (M20.1 §9).

    Owners MUST NOT lose their existing fields. The observability map
    is additive. M20.1 only writes to it; M20.2 will read from it.

    If the entry does not yet exist (e.g. settlement before admission
    was wired into observability), this function initializes a minimal
    one without commitment data. M20.2 may skip such entries.
    """
    if not isinstance(state, dict):
        return
    observability = _ensure_owner_observability_map(state)
    owner_row = observability.get(owner_id)
    if not isinstance(owner_row, dict):
        owner_row = {}
    prior = owner_row.get(commit_id)
    if not isinstance(prior, dict):
        prior = {
            "commitment": None,
            "settled_value": None,
            "settlement_attempts": 0,
            "last_attempt_turn_index": None,
            "last_attempt_reason_code": None,
            "dispatched": False,
            "dispatched_at_turn": None,
            "dispatched_correction_level": None,
        }
    prior["settled_value"] = (
        {
            "outcome": settled_value.outcome,
            "magnitude": settled_value.magnitude,
            "settler_type": settled_value.settler_type,
            "evidence_refs": list(settled_value.evidence_refs),
            "reason_codes": list(settled_value.reason_codes),
            "at": settled_value.at,
            "turn_index": settled_value.turn_index,
            "engineering_proxy_label": settled_value.engineering_proxy_label,
        }
        if settled_value is not None
        else None
    )
    prior["settlement_attempts"] = int(prior.get("settlement_attempts", 0)) + 1
    prior["last_attempt_turn_index"] = int(last_attempt_turn_index)
    prior["last_attempt_reason_code"] = str(last_attempt_reason_code)
    owner_row[commit_id] = prior
    observability[owner_id] = owner_row


# === M20.1 pending-commitment bookkeeping ==============================


def record_pending_commitment(
    state: dict,
    commitment: ActiveCommitment,
) -> None:
    """Append an admitted commitment to the pending list (M20.1 §7).

    The scheduler reads this list. M20.1 only appends (admission) and
    removes (settled or `due_at_passed`). M20.2 adds removal paths for
    promotion / revocation / expiration.
    """
    if not isinstance(state, dict):
        return
    pending = state.get("active_commitments_pending")
    if not isinstance(pending, list):
        pending = []
    pending.append(
        {
            "commit_id": commitment.commit_id,
            "owner_id": commitment.owner_id,
            "source_kind": commitment.source_kind,
            "source_ref": commitment.source_ref,
            "layer": commitment.layer,
            "observable": commitment.observable,
            "observable_payload": dict(commitment.observable_payload),
            "target": dict(commitment.target),
            "due_at": dict(commitment.due_at) if commitment.due_at else None,
            "priority": commitment.priority,
            "confidence": commitment.confidence,
            "evidence_refs": list(commitment.evidence_refs),
            "reason_codes": list(commitment.reason_codes),
            "engineering_proxy_label": commitment.engineering_proxy_label,
            "created_turn": commitment.created_turn,
            "created_at": commitment.created_at,
            "horizon": commitment.horizon,
        }
    )
    # Cap pending list to prevent unbounded growth; M20.2 may tighten.
    state["active_commitments_pending"] = pending[-256:]


def remove_pending_commitment(state: dict, commit_id: str) -> None:
    """Remove a commitment from the pending list (M20.1 §7).

    Called after a successful settlement or after a `due_at_passed`
    emission. The scheduler never re-attempts a removed commitment.
    """
    if not isinstance(state, dict):
        return
    pending = state.get("active_commitments_pending")
    if not isinstance(pending, list):
        return
    state["active_commitments_pending"] = [
        row for row in pending
        if isinstance(row, dict) and row.get("commit_id") != commit_id
    ]


def _is_due_at_passed(due_at: Any, created_turn: int, turn_index: int) -> bool:
    """Return True if the commitment's due window has elapsed (M20.1 §7).

    Frozen semantics in M20.1: only `{"kind": "next_turn"}` is recognized.
    All other shapes (or no `due_at`) are treated as open-ended (never
    past). M20.2 may add explicit timestamp and `natural_idle_shadow_eval`
    semantics.
    """
    if not isinstance(due_at, Mapping):
        return False
    kind = str(due_at.get("kind", "") or "")
    if kind == "next_turn":
        return turn_index > created_turn + _NEXT_TURN_DUE_AT_WINDOW
    return False


# === M20.1 magnitude computation ========================================


def compute_magnitude(
    *,
    observable: str,
    observable_payload: Mapping[str, Any],
    committed_value: Any,
    expected_value: Any,
) -> tuple[float, tuple[str, ...]]:
    """Deterministically compute magnitude (M20.1 §3).

    Returns (magnitude, reason_codes). If the observable has no numeric
    value, magnitude defaults to 0.5 and `magnitude_defaulted` is added
    to reason_codes.

    N3 (M20.3 follow-up): the 0.5 default places a non-numeric
    commitment in the `next_turn` band (0.3–0.6) of the magnitude
    level table, which means a non-numeric observation defaults
    to a moderate "next-turn" correction. This is the opposite
    direction from R2 (which expires `ambiguous`/`uncertain`):
    R2 expires at the SETTLE stage when the settler reports
    low-signal outcomes; N3 default-applies at the MAGNITUDE
    stage when the committed/expected value is not numeric.
    Both paths exist and may fire on different commitments. A
    future M20.x milestone may want a smaller default (e.g.
    0.0 → expire) for non-numeric observables, but that is a
    M20.1 vocabulary change. The current default of 0.5 is
    documented so the operator can read it off the diagnostic
    surface (`magnitude_defaulted` in settled_value's
    reason_codes).
    """
    reason_codes: list[str] = []
    if observable not in MAGNITUDE_SCALES_V1:
        reason_codes.append("magnitude_defaulted")
        return 0.5, tuple(reason_codes)
    scale = float(MAGNITUDE_SCALES_V1[observable])
    if scale <= 0.0:
        reason_codes.append("magnitude_defaulted")
        return 0.5, tuple(reason_codes)
    try:
        committed_num = float(committed_value) if committed_value is not None else None
    except (TypeError, ValueError):
        committed_num = None
    try:
        expected_num = float(expected_value) if expected_value is not None else None
    except (TypeError, ValueError):
        expected_num = None
    if committed_num is None or expected_num is None or committed_num != committed_num or expected_num != expected_num:
        reason_codes.append("magnitude_defaulted")
        return 0.5, tuple(reason_codes)
    raw = abs(committed_num - expected_num) / scale
    if raw != raw or raw < 0.0:
        reason_codes.append("magnitude_defaulted")
        return 0.5, tuple(reason_codes)
    clamped = 0.0 if raw < 0.0 else (1.0 if raw > 1.0 else raw)
    return clamped, tuple(reason_codes)


# === M20.1 settlement attempts diagnostics ==============================


def update_settlement_attempts_diagnostics(
    state: dict,
    *,
    settled: int = 0,
    no_settlement: int = 0,
    by_settler_type: Mapping[str, int] | None = None,
    by_observable: Mapping[str, int] | None = None,
    by_reason_code: Mapping[str, int] | None = None,
    magnitudes: tuple[float, ...] | None = None,
) -> None:
    """Accumulate M20.1 settlement diagnostic counters (M20.1 §8)."""
    if not isinstance(state, dict):
        return
    diag = state.get("settlement_attempts_diagnostics")
    if not isinstance(diag, dict):
        diag = {}
    diag["settlement_attempts_total"] = int(diag.get("settlement_attempts_total", 0)) + int(settled) + int(no_settlement)
    diag["settled_total"] = int(diag.get("settled_total", 0)) + int(settled)
    diag["no_settlement_total"] = int(diag.get("no_settlement_total", 0)) + int(no_settlement)

    def _merge(target_key: str, additions: Mapping[str, int] | None) -> None:
        if not additions:
            return
        target = diag.get(target_key)
        if not isinstance(target, dict):
            target = {}
        for key, count in additions.items():
            if not isinstance(key, str) or not key:
                continue
            target[key] = int(target.get(key, 0)) + int(count)
        diag[target_key] = target

    _merge("settlement_attempts_by_settler_type", by_settler_type)
    _merge("settlement_attempts_by_observable", by_observable)
    _merge("no_settlement_by_reason_code", by_reason_code)

    if magnitudes:
        bucket = diag.get("settled_value_magnitude_distribution")
        if not isinstance(bucket, list):
            bucket = []
        for m in magnitudes:
            try:
                v = float(m)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            bucket.append(max(0.0, min(1.0, v)))
        diag["settled_value_magnitude_distribution"] = bucket[-256:]

    state["settlement_attempts_diagnostics"] = diag


# === M20.1 SettlementScheduler =========================================


class SettlementScheduler:
    """Schedules settlement attempts on admitted commitments (M20.1 §7).

    The scheduler is the *only* place that decides *when* to attempt
    settlement. The settler decides *how* to interpret the payload.
    The scheduler does NOT mutate any long-term state bucket; it
    writes only to `commitment_owner_observability` and emits audit
    events.
    """

    def __init__(
        self,
        *,
        settlers_by_observable: Mapping[str, Settler] | None = None,
    ) -> None:
        self._settlers: dict[str, Settler] = (
            dict(settlers_by_observable) if settlers_by_observable else {}
        )

    def register_settler(self, observable: str, settler: Settler) -> None:
        if observable not in OBSERVABLE_V1:
            raise ValueError(f"unknown observable: {observable!r}")
        self._settlers[observable] = settler

    def get_settler(self, observable: str) -> Settler | None:
        return self._settlers.get(observable)

    def attempt_settlements(
        self,
        *,
        state: dict,
        turn_index: int,
        now: str,
        observation_context_provider: Any = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Attempt settlement for any eligible pending commitments.

        Returns (settled_events, no_settlement_events). Each tuple
        contains audit envelopes ready to be appended to the per-turn
        bus. The caller is responsible for also writing them to the
        conversation log and diagnose surface.
        """
        if not isinstance(state, dict):
            return [], []

        pending = state.get("active_commitments_pending")
        if not isinstance(pending, list) or not pending:
            return [], []

        settled_events: list[dict[str, Any]] = []
        no_settlement_events: list[dict[str, Any]] = []

        diag_by_settler: dict[str, int] = {}
        diag_by_observable: dict[str, int] = {}
        diag_by_reason: dict[str, int] = {}
        diag_magnitudes: list[float] = []

        # Snapshot the observability map for "already settled" checks.
        observability = _ensure_owner_observability_map(state)

        for row in pending:
            if not isinstance(row, Mapping):
                continue
            commit_id = str(row.get("commit_id", "") or "")
            owner_id = str(row.get("owner_id", "") or "")
            observable = str(row.get("observable", "") or "")
            if not commit_id or not owner_id or not observable:
                continue
            created_turn = int(row.get("created_turn", 0) or 0)

            # T0+1 minimum eligibility.
            if turn_index < created_turn + 1:
                continue

            # Skip if already settled.
            owner_row = observability.get(owner_id)
            if isinstance(owner_row, dict):
                commit_row = owner_row.get(commit_id)
                if isinstance(commit_row, dict) and commit_row.get("settled_value") is not None:
                    # Already settled: do not re-attempt.
                    diag_by_reason["observation_already_processed"] = (
                        diag_by_reason.get("observation_already_processed", 0) + 1
                    )
                    continue

            # `due_at` past and not yet settled: emit once and drop.
            if _is_due_at_passed(row.get("due_at"), created_turn, turn_index):
                no_settlement = NoSettlement(
                    commit_id=commit_id,
                    reason_code="due_at_passed",
                    settler_type="deterministic",  # scheduler-level, not settler-level
                    engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
                    at=now,
                    turn_index=turn_index,
                )
                event = build_no_settlement_made_event(no_settlement)
                no_settlement_events.append(event)
                write_owner_observability(
                    state,
                    owner_id=owner_id,
                    commit_id=commit_id,
                    settled_value=None,
                    last_attempt_turn_index=turn_index,
                    last_attempt_reason_code="due_at_passed",
                )
                diag_by_reason["due_at_passed"] = (
                    diag_by_reason.get("due_at_passed", 0) + 1
                )
                remove_pending_commitment(state, commit_id)
                continue

            # Route to settler.
            settler = self._settlers.get(observable)
            if settler is None:
                # No settler for this observable in M20.1 (not yet migrated
                # in M20.1.1). Emit NoSettlement with settler_unavailable.
                no_settlement = NoSettlement(
                    commit_id=commit_id,
                    reason_code="settler_unavailable",
                    settler_type="deterministic",
                    engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
                    at=now,
                    turn_index=turn_index,
                )
                event = build_no_settlement_made_event(no_settlement)
                no_settlement_events.append(event)
                write_owner_observability(
                    state,
                    owner_id=owner_id,
                    commit_id=commit_id,
                    settled_value=None,
                    last_attempt_turn_index=turn_index,
                    last_attempt_reason_code="settler_unavailable",
                )
                diag_by_reason["settler_unavailable"] = (
                    diag_by_reason.get("settler_unavailable", 0) + 1
                )
                diag_by_observable[observable] = (
                    diag_by_observable.get(observable, 0) + 1
                )
                # Per M20.1 §7: do not remove on settler_unavailable; the
                # settler may be wired up in a later milestone. The
                # observability entry is recorded but the commitment
                # stays pending.
                #
                # N1 (M20.3 follow-up): a long-pending commitment
                # (e.g. observable without a settler, or
                # `no_eligible_observation` that never resolves) can
                # silently drain the 256-row pending cap. The
                # scheduler does not change behavior here, but it
                # records the staleness in the diagnostic surface so
                # the operator can see the capacity drain.
                _record_long_pending_staleness(
                    state,
                    commit_id=commit_id,
                    owner_id=owner_id,
                    observable=observable,
                    current_turn=turn_index,
                    created_turn=created_turn,
                    reason_code="settler_unavailable",
                )
                continue

            # Build observation context.
            if observation_context_provider is not None:
                ctx = observation_context_provider(turn_index, row)
                if not isinstance(ctx, Mapping):
                    ctx = {}
            else:
                ctx = {}

            # Run the settler.
            try:
                result = settler.settle(
                    commitment=_row_to_active_commitment(row, turn_index=turn_index, now=now),
                    observation_context=ctx,
                )
            except SettlerUnavailable:
                result = NoSettlement(
                    commit_id=commit_id,
                    reason_code="settler_unavailable",
                    settler_type=_settler_type_of(settler),
                    engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
                    at=now,
                    turn_index=turn_index,
                )
            except Exception:  # noqa: BLE001
                # A buggy settler MUST NOT crash the run_turn path. Emit
                # NoSettlement with settler_unavailable and continue.
                result = NoSettlement(
                    commit_id=commit_id,
                    reason_code="settler_unavailable",
                    settler_type=_settler_type_of(settler),
                    engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
                    at=now,
                    turn_index=turn_index,
                )

            if isinstance(result, SettledValue):
                # Validate outcome is in the per-observable set.
                allowed = OUTCOME_BY_OBSERVABLE_V1.get(observable, frozenset())
                if result.outcome not in allowed:
                    result = NoSettlement(
                        commit_id=commit_id,
                        reason_code="settler_llm_invalid_response",
                        settler_type=result.settler_type,
                        engineering_proxy_label=result.engineering_proxy_label,
                        at=now,
                        turn_index=turn_index,
                    )
                else:
                    event = build_active_commitment_settled_event(result)
                    settled_events.append(event)
                    write_owner_observability(
                        state,
                        owner_id=owner_id,
                        commit_id=commit_id,
                        settled_value=result,
                        last_attempt_turn_index=turn_index,
                        last_attempt_reason_code=str(result.reason_codes[0]) if result.reason_codes else "settler_deterministic",
                    )
                    diag_by_settler[result.settler_type] = (
                        diag_by_settler.get(result.settler_type, 0) + 1
                    )
                    diag_by_observable[observable] = (
                        diag_by_observable.get(observable, 0) + 1
                    )
                    diag_magnitudes.append(result.magnitude)
                    remove_pending_commitment(state, commit_id)
                    continue

            if isinstance(result, NoSettlement):
                event = build_no_settlement_made_event(result)
                no_settlement_events.append(event)
                write_owner_observability(
                    state,
                    owner_id=owner_id,
                    commit_id=commit_id,
                    settled_value=None,
                    last_attempt_turn_index=turn_index,
                    last_attempt_reason_code=result.reason_code,
                )
                diag_by_reason[result.reason_code] = (
                    diag_by_reason.get(result.reason_code, 0) + 1
                )
                diag_by_observable[observable] = (
                    diag_by_observable.get(observable, 0) + 1
                )
                # Removal policy (M20.1 §7): drop the commitment only
                # when the failure is terminal for that commitment.
                # `due_at_passed` is terminal. `settler_hybrid_fallback_exhausted`
                # is terminal for the hybrid settler. `settler_unavailable`
                # and `no_eligible_observation` are transient: the
                # settler may be wired up later (M20.1.1) or the
                # observation may arrive on a later turn, so the
                # commitment stays pending.
                #
                # N1 (M20.3 follow-up): record staleness in the
                # diagnostic surface so a long-pending
                # `no_eligible_observation` is observable.
                if result.reason_code in (
                    "no_eligible_observation",
                    "settler_unavailable",
                ):
                    _record_long_pending_staleness(
                        state,
                        commit_id=commit_id,
                        owner_id=owner_id,
                        observable=observable,
                        current_turn=turn_index,
                        created_turn=created_turn,
                        reason_code=result.reason_code,
                    )
                if result.reason_code in (
                    "due_at_passed",
                    "settler_hybrid_fallback_exhausted",
                ):
                    remove_pending_commitment(state, commit_id)
                continue

        update_settlement_attempts_diagnostics(
            state,
            settled=len(settled_events),
            no_settlement=len(no_settlement_events),
            by_settler_type=diag_by_settler,
            by_observable=diag_by_observable,
            by_reason_code=diag_by_reason,
            magnitudes=tuple(diag_magnitudes),
        )
        return settled_events, no_settlement_events


def _row_to_active_commitment(
    row: Mapping[str, Any],
    *,
    turn_index: int,
    now: str,
) -> ActiveCommitment:
    """Reconstruct an `ActiveCommitment` from a pending-list row.

    Used by the scheduler to hand a typed value to the settler. The
    commit_id is preserved (deterministic sha1 from admission).
    """
    payload = row.get("observable_payload")
    if not isinstance(payload, Mapping):
        payload = {}
    target = row.get("target")
    if not isinstance(target, Mapping):
        target = {}
    due_at = row.get("due_at")
    if due_at is not None and not isinstance(due_at, Mapping):
        due_at = None
    return ActiveCommitment(
        commit_id=str(row.get("commit_id", "") or ""),
        owner_id=str(row.get("owner_id", "") or ""),
        source_kind=str(row.get("source_kind", "") or ""),
        source_ref=str(row.get("source_ref", "") or ""),
        layer=str(row.get("layer", "") or ""),
        observable=str(row.get("observable", "") or ""),
        observable_payload=MappingProxyType(dict(payload)),
        target=MappingProxyType(dict(target)),
        due_at=MappingProxyType(dict(due_at)) if due_at else None,
        priority=_bounded_float(row.get("priority")),
        confidence=_bounded_float(row.get("confidence")),
        evidence_refs=tuple(_string_list(row.get("evidence_refs"), limit=32)),
        created_turn=int(row.get("created_turn", 0) or 0),
        created_at=str(row.get("created_at", "") or now),
        reason_codes=tuple(_string_list(row.get("reason_codes"), limit=16)),
        engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
        horizon=str(row.get("horizon", "") or "next_turn"),
    )


def _settler_type_of(settler: Any) -> str:
    """Best-effort inference of a settler's `settler_type` label."""
    cls = getattr(settler, "__class__", None)
    name = getattr(cls, "__name__", "") or ""
    if "Deterministic" in name:
        return "deterministic"
    if "Hybrid" in name:
        return "hybrid"
    if "Silent" in name:
        return "silent"
    if "LLMJudge" in name or "LLM" in name:
        return "llm_judge"
    return "deterministic"


# === M20.2 dispatcher core ==============================================


@dataclass(frozen=True)
class GradedCorrectionDecision:
    """Frozen result of the pure dispatcher (M20.2 §4).

    The dispatcher does NOT mutate state. The router reads this
    decision and dispatches to the existing owner write path.
    """

    commit_id: str
    correction_level: str
    routed_owner_id: str
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    magnitude_before: float | None
    magnitude_after: float | None
    outcome: str
    at: str
    turn_index: int
    engineering_proxy_label: str
    deferred: bool = False
    rejected: bool = False


def _level_from_magnitude(magnitude: float) -> str:
    """Map magnitude to a `GradedCorrection` level via the frozen table."""
    for low, high, level in _MAGNITUDE_LEVEL_TABLE:
        if low <= magnitude < high:
            return level
    return "slow_promote"


def _is_m19_3_already_promoted(
    owner_state_snapshot: Mapping[str, Any] | None,
    source_ref: str,
) -> bool:
    """Return True if M19.3 has already promoted `source_ref`.

    M20.2 only reads. The promotion lock is owned by M19.3.
    """
    if not isinstance(owner_state_snapshot, Mapping):
        return False
    # M19.3 may surface the lock in multiple shapes. The M20.2
    # dispatcher only reads known keys; new keys are a M19.3
    # vocabulary bump.
    promotion_lock = owner_state_snapshot.get("m19_3_promotion_lock")
    if isinstance(promotion_lock, Mapping):
        if source_ref in promotion_lock:
            return True
        promoted_set = promotion_lock.get("promoted")
        if isinstance(promoted_set, list) and source_ref in promoted_set:
            return True
    calibrated = owner_state_snapshot.get("calibrated_tendencies")
    if isinstance(calibrated, list):
        for row in calibrated:
            if (
                isinstance(row, Mapping)
                and str(row.get("source_ref", "") or "") == source_ref
            ):
                return True
    return False


class GradedCorrectionDispatcher:
    """Pure dispatcher (M20.2 §4).

    Maps `(commitment, settled_value, owner_state_snapshot)` to a
    `GradedCorrectionDecision`. Does NOT mutate state, call any LLM,
    or re-interpret `observable_payload`.
    """

    def __init__(
        self,
        *,
        # m19_3 already promoted check is read-only. Default reads from
        # the standard M19.3 surface. Tests may inject a custom check.
        is_m19_3_already_promoted: Any = _is_m19_3_already_promoted,
    ) -> None:
        self._is_m19_3_already_promoted = is_m19_3_already_promoted

    def decide(
        self,
        *,
        commitment: ActiveCommitment,
        settled_value: SettledValue,
        owner_state_snapshot: Mapping[str, Any] | None = None,
        turn_index: int = 0,
        now: str = "",
    ) -> GradedCorrectionDecision:
        """Map (commitment, settled_value, owner_state_snapshot) to a decision."""
        owner_id = commitment.owner_id
        outcome = settled_value.outcome
        magnitude = float(settled_value.magnitude)
        # Clamp magnitude to [0.0, 1.0].
        if magnitude < 0.0:
            magnitude = 0.0
        elif magnitude > 1.0:
            magnitude = 1.0
        source_kind = commitment.source_kind
        source_ref = commitment.source_ref
        evidence_refs = tuple(_string_list(list(settled_value.evidence_refs), limit=32))
        if not evidence_refs:
            evidence_refs = tuple(_string_list(list(commitment.evidence_refs), limit=32))

        common_kwargs: dict[str, Any] = dict(
            commit_id=commitment.commit_id,
            routed_owner_id=owner_id,
            evidence_refs=evidence_refs,
            outcome=outcome,
            at=now,
            turn_index=turn_index,
            engineering_proxy_label=commitment.engineering_proxy_label,
        )

        # Unknown owner → reject. M20.3 bump: consult v2 registry
        # (v1 ∪ new v2 owners), so `runtime_mode_state` is a known
        # owner at dispatch time.
        if owner_id not in COMMITMENT_REGISTRY_V2:
            return GradedCorrectionDecision(
                **common_kwargs,
                correction_level="expire",
                reason_codes=("unknown_owner",),
                magnitude_before=magnitude,
                magnitude_after=magnitude,
                rejected=True,
            )

        owner_row = COMMITMENT_REGISTRY_V2[owner_id]
        action_set = owner_row.get("graded_action_set", [])
        if not isinstance(action_set, list):
            action_set = []

        # source_kind = "policy" → expire (observation-only).
        # M20.3 §3.5 v2 exception: if the owner has
        # `accepts_policy_correction: true` in the v2 registry, the
        # general "policy -> expire" rule is bypassed and the
        # regular magnitude-to-level table applies. M20.2's frozen
        # rule remains "policy -> expire"; the registry column is
        # the parameterization point. The exception only applies
        # to the v2 `runtime_mode_state` owner in v1 scope.
        if source_kind == "policy" and not is_registry_v2_accepts_policy_correction(owner_id):
            return GradedCorrectionDecision(
                **common_kwargs,
                correction_level="expire",
                reason_codes=("policy_source_no_correction",),
                magnitude_before=magnitude,
                magnitude_after=magnitude,
                deferred=True,
            )

        # outcome = "ambiguous" → expire (not enough signal).
        if outcome == "ambiguous":
            return GradedCorrectionDecision(
                **common_kwargs,
                correction_level="expire",
                reason_codes=("ambiguous_outcome",),
                magnitude_before=magnitude,
                magnitude_after=magnitude,
                deferred=True,
            )

        # Map magnitude → base level.
        base_level = _level_from_magnitude(magnitude)

        # outcome = "uncertain" → microadjust only if magnitude >= 0.5
        # else expire.
        if outcome == "uncertain":
            if magnitude < 0.5:
                return GradedCorrectionDecision(
                    **common_kwargs,
                    correction_level="expire",
                    reason_codes=("magnitude_below_threshold",),
                    magnitude_before=magnitude,
                    magnitude_after=magnitude,
                    deferred=True,
                )
            base_level = "microadjust"

        # outcome = "violated" AND magnitude >= 0.85 → revoke.
        if outcome == "violated" and magnitude >= 0.85:
            base_level = "revoke"

        # M19.3 already promoted → downgrade slow_promote to deferred.
        if base_level == "slow_promote":
            already_promoted = bool(
                self._is_m19_3_already_promoted(owner_state_snapshot, source_ref)
            )
            if already_promoted:
                return GradedCorrectionDecision(
                    **common_kwargs,
                    correction_level="expire",
                    reason_codes=("m19_3_already_promoted",),
                    magnitude_before=magnitude,
                    magnitude_after=magnitude,
                    deferred=True,
                )

        # Action-set validation.
        if base_level not in action_set:
            reason_code = (
                "slow_promote_not_supported"
                if base_level == "slow_promote"
                else "action_set_violation"
            )
            return GradedCorrectionDecision(
                **common_kwargs,
                correction_level="expire",
                reason_codes=(reason_code,),
                magnitude_before=magnitude,
                magnitude_after=magnitude,
                rejected=True,
            )

        # Compute magnitude_after as a small bounded delta.
        if base_level == "microadjust":
            delta = 0.05 * magnitude
            magnitude_after = min(1.0, magnitude + delta)
        elif base_level == "next_turn":
            delta = 0.1 * magnitude
            magnitude_after = min(1.0, magnitude + delta)
        elif base_level == "same_turn":
            delta = 0.15 * magnitude
            magnitude_after = min(1.0, magnitude + delta)
        elif base_level == "slow_promote":
            magnitude_after = 1.0
        elif base_level == "revoke":
            magnitude_after = 0.0
        else:
            magnitude_after = magnitude

        return GradedCorrectionDecision(
            **common_kwargs,
            correction_level=base_level,
            reason_codes=("graded_correction_routed",),
            magnitude_before=magnitude,
            magnitude_after=magnitude_after,
        )


# === M20.2 audit event builders =========================================


def build_graded_correction_routed_event(
    decision: GradedCorrectionDecision,
) -> dict[str, Any]:
    return {
        "type": "GradedCorrectionRouted",
        "turn_index": decision.turn_index,
        "commit_id": decision.commit_id,
        "routed_owner_id": decision.routed_owner_id,
        "correction_level": decision.correction_level,
        "outcome": decision.outcome,
        "magnitude_before": decision.magnitude_before,
        "magnitude_after": decision.magnitude_after,
        "evidence_refs": list(decision.evidence_refs),
        "reason_codes": list(decision.reason_codes),
        "engineering_proxy_label": decision.engineering_proxy_label,
        "at": decision.at,
    }


def build_correction_deferred_event(
    decision: GradedCorrectionDecision,
) -> dict[str, Any]:
    return {
        "type": "CorrectionDeferred",
        "turn_index": decision.turn_index,
        "commit_id": decision.commit_id,
        "routed_owner_id": decision.routed_owner_id,
        "reason_code": (
            decision.reason_codes[0] if decision.reason_codes else "magnitude_below_threshold"
        ),
        "engineering_proxy_label": decision.engineering_proxy_label,
        "at": decision.at,
    }


def build_correction_rejected_event(
    decision: GradedCorrectionDecision,
) -> dict[str, Any]:
    return {
        "type": "CorrectionRejected",
        "turn_index": decision.turn_index,
        "commit_id": decision.commit_id,
        "routed_owner_id": decision.routed_owner_id,
        "reason_code": (
            decision.reason_codes[0] if decision.reason_codes else "action_set_violation"
        ),
        "engineering_proxy_label": decision.engineering_proxy_label,
        "at": decision.at,
    }


# === M20.2 graded correction diagnostic counters ========================


def update_graded_correction_diagnostics(
    state: dict,
    *,
    routed: int = 0,
    deferred: int = 0,
    rejected: int = 0,
    by_level: Mapping[str, int] | None = None,
    by_owner_id: Mapping[str, int] | None = None,
    by_outcome: Mapping[str, int] | None = None,
    by_reason_code: Mapping[str, int] | None = None,
    magnitudes_before: tuple[float, ...] | None = None,
    magnitudes_after: tuple[float, ...] | None = None,
    m19_3_shortcut: int = 0,
    same_turn_advisory_violations: int = 0,
) -> None:
    """Accumulate M20.2 dispatcher diagnostic counters (M20.2 §9)."""
    if not isinstance(state, dict):
        return
    diag = state.get("graded_correction_diagnostics")
    if not isinstance(diag, dict):
        diag = {}
    diag["graded_correction_total"] = int(diag.get("graded_correction_total", 0)) + int(routed) + int(deferred) + int(rejected)
    diag["graded_correction_routed_total"] = int(diag.get("graded_correction_routed_total", 0)) + int(routed)
    diag["correction_deferred_total"] = int(diag.get("correction_deferred_total", 0)) + int(deferred)
    diag["correction_rejected_total"] = int(diag.get("correction_rejected_total", 0)) + int(rejected)
    diag["m19_3_already_promoted_shortcut_count"] = (
        int(diag.get("m19_3_already_promoted_shortcut_count", 0)) + int(m19_3_shortcut)
    )
    diag["same_turn_advisory_violations"] = (
        int(diag.get("same_turn_advisory_violations", 0)) + int(same_turn_advisory_violations)
    )

    def _merge(target_key: str, additions: Mapping[str, int] | None) -> None:
        if not additions:
            return
        target = diag.get(target_key)
        if not isinstance(target, dict):
            target = {}
        for key, count in additions.items():
            if not isinstance(key, str) or not key:
                continue
            target[key] = int(target.get(key, 0)) + int(count)
        diag[target_key] = target

    _merge("graded_correction_by_level", by_level)
    _merge("graded_correction_by_owner_id", by_owner_id)
    _merge("graded_correction_by_outcome", by_outcome)
    _merge("correction_by_reason_code", by_reason_code)

    def _append_distribution(key: str, additions: tuple[float, ...] | None) -> None:
        if not additions:
            return
        bucket = diag.get(key)
        if not isinstance(bucket, list):
            bucket = []
        for m in additions:
            try:
                v = float(m)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            bucket.append(max(0.0, min(1.0, v)))
        diag[key] = bucket[-256:]

    _append_distribution("magnitude_before_distribution", magnitudes_before)
    _append_distribution("magnitude_after_distribution", magnitudes_after)
    state["graded_correction_diagnostics"] = diag


__all__ = [
    "ALLOWED_LAYERS",
    "ALLOWED_SOURCE_KINDS",
    "ALL_GRADED_CORRECTION_REASON_CODES_V1",
    "ALL_SETTLEMENT_REASON_CODES_V1",
    "ActiveCommitment",
    "ActiveCommitmentAdapter",
    "COMMITMENT_PHASE",
    "COMMITMENT_REGISTRY_V1",
    "COMMITMENT_REGISTRY_V2",
    "DISPATCHER_REASON_CODES_V1",
    "ENGINEERING_PROXY_LABELS_V1",
    "GRADED_CORRECTION_V1",
    "GradedCorrectionDecision",
    "GradedCorrectionDispatcher",
    "HORIZON_V1",
    "MAGNITUDE_SCALES_V1",
    "NoSettlement",
    "OBSERVABLE_V1",
    "OBSERVABLE_V2",
    "OUTCOME_BY_OBSERVABLE_V1",
    "OUTCOME_BY_OBSERVABLE_V2",
    "OUTCOME_V1",
    "REASON_CODES_V1",
    "SETTLER_REASON_CODES_V1",
    "SETTLER_TYPE_V1",
    "SettledValue",
    "SettlementScheduler",
    "Settler",
    "SettlerUnavailable",
    "build_active_commitment_created_event",
    "build_active_commitment_settled_event",
    "build_correction_deferred_event",
    "build_correction_rejected_event",
    "build_graded_correction_routed_event",
    "build_no_settlement_made_event",
    "compute_commit_id",
    "init_owner_observability_for_commitment",
    "compute_magnitude",
    "record_active_commitment_event",
    "record_pending_commitment",
    "remove_pending_commitment",
    "update_commitment_registry_diagnostics",
    "update_graded_correction_diagnostics",
    "update_settlement_attempts_diagnostics",
    "wrap_self_response_expectation_proposal",
    "write_owner_observability",
]


# === M20.3 v2 vocabulary =================================================
# M20.3 is a layer above M20.0–M20.2. v1 rows in
# `COMMITMENT_REGISTRY_V1` and `OBSERVABLE_V1` are unchanged. v2
# additions live in `COMMITMENT_REGISTRY_V2` and `OBSERVABLE_V2` and
# are derived as v1 ∪ new entries.

# N1 (M20.3 follow-up): a commitment is "long-pending" when it has
# been retried for at least this many turns without a settlement.
# The scheduler does not auto-expire such commitments (M20.1 §7
# preserves them for future M20.1.1 migration), but it surfaces the
# staleness in `state["commitment_registry_index"]` so the operator
# can see the capacity drain. Frozen at 8 turns by M20.3.
LONG_PENDING_STALENESS_TURNS: int = 8

# v2 horizon enum. M20.3 v2 attribute on ActiveCommitment. Default
# is "next_turn" for v1 commitments. PolicyProducer sets
# "same_turn_surface" on `runtime_mode_state` rows.
HORIZON_V1: frozenset[str] = frozenset({
    "same_turn_surface",
    "next_turn",
    "natural_context",
})


# v2 owner row. Added in M20.3; the v2 registry is v1 ∪ this row.
_RUNTIME_MODE_STATE_OWNER: Mapping[str, Any] = MappingProxyType({
    "description": (
        "Writable runtime mode container. Holds the current persona / "
        "surface / roleplay mode, mode_changed_at, mode_owner, and "
        "mode_constraints. Distinct from policy_state (which is "
        "observation-only)."
    ),
    "storage_hint": (
        "runtime_mode_state.{mode, mode_changed_at, mode_owner, mode_constraints}"
    ),
    "accepts_layers": ["A_long_term_prior", "B_per_turn_commitment"],
    "accepts_source_kinds": ["policy", "state"],
    "graded_action_set": ["microadjust", "next_turn", "same_turn", "revoke"],
    # M20.3 v2 fields.
    "accepts_same_turn_block": True,
    "accepts_policy_correction": True,
    "notes": (
        "Accepts same_turn BLOCK (pre-send gate can refuse a reply "
        "that violates the current mode). All other same_turn routes "
        "from M20.2 are advisory only; this owner is the documented "
        "exception. Also accepts policy-source correction (registry "
        "v2 exception table in §3.5), bypassing M20.2 §2's general "
        "'policy -> expire' rule."
    ),
})


COMMITMENT_REGISTRY_V2: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    **dict(COMMITMENT_REGISTRY_V1),
    "runtime_mode_state": dict(_RUNTIME_MODE_STATE_OWNER),
    # M20.3 v2 additive: v1 `outreach_intent_registry` opts into
    # `policy` source_kind so PolicyProducer's /quiet + /resume
    # commands route through. v1 owners remain unchanged.
    "outreach_intent_registry": {
        **dict(COMMITMENT_REGISTRY_V1["outreach_intent_registry"]),
        "accepts_source_kinds": ["state", "episodic", "policy"],
    },
})


# v2 observable additions. `runtime_mode_state` is the new observable
# the pre-send gate can block on. `outreach_intent_on/off` are
# observation-only in v1 (no settler; only "silent" carries forward).
_RUNTIME_MODE_STATE_OBSERVABLE: Mapping[str, Any] = MappingProxyType({
    "payload_keys": (
        "expected_mode",
        "actual_mode",
        "mode_owner",
        "evidence_refs",
    ),
    "settler_hint": "llm_judge",
    "notes": (
        "LLM judge checks whether the reply's persona / surface "
        "matches the currently admitted mode. Uses bounded M19.x "
        "surface_consistency call shape."
    ),
})

_OUTREACH_INTENT_ON_OBSERVABLE: Mapping[str, Any] = MappingProxyType({
    "payload_keys": ("expected_mode", "evidence_refs"),
    "settler_hint": "silent",
    "notes": "Observation-only; no settler v1. silent carries forward.",
})

_OUTREACH_INTENT_OFF_OBSERVABLE: Mapping[str, Any] = MappingProxyType({
    "payload_keys": ("expected_mode", "evidence_refs"),
    "settler_hint": "silent",
    "notes": "Observation-only; no settler v1. silent carries forward.",
})


OBSERVABLE_V2: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    **dict(OBSERVABLE_V1),
    "runtime_mode_state": dict(_RUNTIME_MODE_STATE_OBSERVABLE),
    "outreach_intent_on": dict(_OUTREACH_INTENT_ON_OBSERVABLE),
    "outreach_intent_off": dict(_OUTREACH_INTENT_OFF_OBSERVABLE),
})


# v2 outcome set per observable. v1 outcomes carry forward; v2
# observables get the LLM-judge 3-outcome set (confirmed / violated /
# ambiguous). outreach_intent_on/off are observation-only; the only
# outcome that carries forward is "confirmed" (i.e. observed) when
# the settler_hint is "silent".
OUTCOME_BY_OBSERVABLE_V2: Mapping[str, frozenset[str]] = MappingProxyType({
    **dict(OUTCOME_BY_OBSERVABLE_V1),
    "runtime_mode_state": frozenset({"confirmed", "violated", "ambiguous"}),
    "outreach_intent_on": frozenset({"confirmed", "violated"}),
    "outreach_intent_off": frozenset({"confirmed", "violated"}),
})


def registry_v2_owner_row(owner_id: str) -> Mapping[str, Any] | None:
    """Return the v2 owner row (or None for unknown owners).

    Convenience for the dispatcher v2 exception table and the
    PolicyProducer. Read-only access; callers MUST NOT mutate.
    """
    row = COMMITMENT_REGISTRY_V2.get(owner_id)
    if row is None:
        return None
    return row


def is_registry_v2_accepts_policy_correction(owner_id: str) -> bool:
    """Return True iff the v2 owner row opts into the M20.2 §2
    policy-source-correction exception.

    M20.2 §2 freezes the rule "policy -> expire" for any outcome. The
    M20.3 v2 exception table in §3.5 lets a v2 owner opt out by
    declaring `accepts_policy_correction: true`. The dispatcher reads
    this flag at runtime; the frozen M20.2 rule is unchanged.
    """
    row = registry_v2_owner_row(owner_id)
    if row is None:
        return False
    return bool(row.get("accepts_policy_correction", False))


def is_registry_v2_accepts_same_turn_block(owner_id: str) -> bool:
    """Return True iff the v2 owner row opts into same-turn BLOCK.

    The pre-send gate can only `block` for owners that opt in via
    `accepts_same_turn_block: true`. For all other observables, the
    gate returns `pass` or `advisory_guidance`.
    """
    row = registry_v2_owner_row(owner_id)
    if row is None:
        return False
    return bool(row.get("accepts_same_turn_block", False))
