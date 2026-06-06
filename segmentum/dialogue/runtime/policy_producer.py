"""M20.3 §1 PolicyProducer — non-LLM admission path for policy commitments.

PolicyProducer emits `ActiveCommitment` rows with `source_kind = "policy"`.
The producer is deterministic; it does not call an LLM. It maps the
frozen v1+v2 producer input table (turn_context, runtime_mode_flags,
command_envelope, user_correction_signal) to one or more admitted
commitments and a `PolicyAdmitted` audit envelope per row.

M20.3 closes gap A from the M20.0–M20.2 review: policy commitments
had no producer and no writable owner. PolicyProducer is the producer;
`runtime_mode_state` is the writable owner. M20.2 §3.5's v2 exception
table lets policy-source `runtime_mode_state` commitments actually
update the persistent mode state.

v1 commitments are unaffected. M20.0–M20.2 admission paths and
`ActiveCommitmentAdapter` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    COMMITMENT_REGISTRY_V2,
    ActiveCommitment,
    ActiveCommitmentAdapter,
)


# === Input types =========================================================
# These are intentionally small / structural. The producer is not
# the LLM; it only validates bounded shapes.

# `runtime_mode_flags.surface_intent` (per M18.x group_turn_binding):
#   "bot" | "chat" | "abstain"
_ALLOWED_SURFACE_INTENTS: frozenset[str] = frozenset({"bot", "chat", "abstain"})

# `user_correction_signal.correcting_assistant_identity` (M20.3 §3.1):
#   "" | "wrong_persona" | "wrong_voice" | "right_persona_reaffirm"
_ALLOWED_USER_CORRECTION_SIGNALS: frozenset[str] = frozenset({
    "",
    "wrong_persona",
    "wrong_voice",
    "right_persona_reaffirm",
})

# `envelope.platform_command` for the cases PolicyProducer reacts to:
_ALLOWED_PLATFORM_COMMANDS: frozenset[str] = frozenset({
    "/status",
    "/mode",
    "/persona",
    "/quiet",
    "/resume",
})

# `envelope.bot_command_args[0]` is the mode for `/mode` and `/persona`.
_ALLOWED_RUNTIME_MODES: frozenset[str] = frozenset({
    "bot_system",
    "persona_chat",
    "roleplay",
    "abstain",
})


# === Frozen producer input -> observable -> owner_id -> scope table ====
# v1 frozen. Adding a row is a vocabulary bump; changing semantics
# of an existing row is a M20.3 vocabulary change. The full table is
# also returned via `policy_producer_table_snapshot()` for tests.

# Each row is keyed by `kind` and produces at most one
# `ActiveCommitment` with `source_kind = "policy"`, non-empty
# `evidence_refs`, and `commit_id` derived deterministically from
# `(owner_id, source_ref, layer, observable, created_turn)`.
#
# `scope` is "turn_scoped" or "durable_mutate":
# - turn_scoped: pre-send gate uses `expected_mode` for this turn; the
#   persistent `runtime_mode_state.mode` is NOT mutated after M20.2
#   dispatch.
# - durable_mutate: persistent `runtime_mode_state.mode` IS mutated
#   after M20.2 dispatch with `correction_level` >= `microadjust`.

_TURN_SCOPED = "turn_scoped"
_DURABLE_MUTATE = "durable_mutate"


@dataclass(frozen=True)
class _PolicyProducerRule:
    kind: str
    observable: str
    owner_id: str
    scope: str
    reason_code: str


_POLICY_PRODUCER_RULES: tuple[_PolicyProducerRule, ...] = (
    _PolicyProducerRule(
        kind="command_status",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_TURN_SCOPED,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="command_mode",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_DURABLE_MUTATE,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="command_persona",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_DURABLE_MUTATE,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="surface_intent_bot",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_TURN_SCOPED,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="surface_intent_chat",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_TURN_SCOPED,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="surface_intent_abstain",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_TURN_SCOPED,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="user_correction_wrong_persona",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_DURABLE_MUTATE,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="user_correction_wrong_voice",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_DURABLE_MUTATE,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="user_correction_reaffirm",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_TURN_SCOPED,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="group_mode_ingress_change",
        observable="runtime_mode_state",
        owner_id="runtime_mode_state",
        scope=_DURABLE_MUTATE,
        reason_code="policy_prior",
    ),
    _PolicyProducerRule(
        kind="command_quiet",
        observable="outreach_intent_off",
        owner_id="outreach_intent_registry",
        scope=_TURN_SCOPED,
        reason_code="outreach_intent",
    ),
    _PolicyProducerRule(
        kind="command_resume",
        observable="outreach_intent_on",
        owner_id="outreach_intent_registry",
        scope=_TURN_SCOPED,
        reason_code="outreach_intent",
    ),
)


def policy_producer_table_snapshot() -> tuple[Mapping[str, Any], ...]:
    """Return the frozen producer table as a tuple of dicts (for tests)."""
    return tuple(
        {
            "kind": rule.kind,
            "observable": rule.observable,
            "owner_id": rule.owner_id,
            "scope": rule.scope,
            "reason_code": rule.reason_code,
        }
        for rule in _POLICY_PRODUCER_RULES
    )


def is_policy_owner_accepts_policy_correction(owner_id: str) -> bool:
    """Return True iff `owner_id` opts into the M20.2 §3.5 v2 exception.

    Thin pass-through to the registry v2 helper so callers can reason
    about routing without importing `active_commitment` directly.
    """
    row = COMMITMENT_REGISTRY_V2.get(owner_id)
    if row is None:
        return False
    return bool(row.get("accepts_policy_correction", False))


# === Audit event builder ================================================


def build_policy_admitted_event(
    *,
    commitment: ActiveCommitment,
    turn_index: int,
    at: str,
    rule_kind: str,
) -> dict[str, Any]:
    """Build the `PolicyAdmitted` audit envelope (M20.3 §1.4).

    Emitted ALONGSIDE the M20.0 `ActiveCommitmentCreated` event. The
    two events share the same `commit_id` so they can be cross-
    referenced in the audit tail.
    """
    return {
        "type": "PolicyAdmitted",
        "turn_index": turn_index,
        "commit_id": commitment.commit_id,
        "owner_id": commitment.owner_id,
        "source_ref": commitment.source_ref,
        "observable": commitment.observable,
        "rule_kind": rule_kind,
        "horizon": commitment.horizon,
        "evidence_refs": list(commitment.evidence_refs),
        "reason_codes": list(commitment.reason_codes),
        "engineering_proxy_label": "mvp_local_policy_admission",
        "at": at,
    }


# === Producer input validation helpers ==================================


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _bounded_string_list(value: Any, *, limit: int = 16) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _classify_command(envelope: Mapping[str, Any] | None) -> tuple[str, str]:
    """Return (command, arg0) from an envelope, both empty when not a command."""
    if not isinstance(envelope, Mapping):
        return "", ""
    cmd = _bounded_string(envelope.get("platform_command"), default="", limit=32)
    if cmd not in _ALLOWED_PLATFORM_COMMANDS:
        return "", ""
    args = envelope.get("bot_command_args")
    arg0 = ""
    if isinstance(args, (list, tuple)) and args:
        first = args[0]
        if isinstance(first, str):
            arg0 = first.strip()[:32]
    return cmd, arg0


def _resolve_runtime_mode(
    *,
    arg0: str,
    surface_intent: str,
    signal: str,
    command: str = "",
) -> str:
    """Return the `expected_mode` for a runtime_mode_state commitment.

    Order:
    1. `arg0` from /mode or /persona (when present and valid).
    2. `command == "/status"` (turn-scoped bot_system).
    3. `surface_intent` mapped to a runtime mode.
    4. Correction signal → persona_chat.
    5. Empty (the commitment is admitted without `expected_mode`).
    """
    if arg0 and arg0 in _ALLOWED_RUNTIME_MODES:
        return arg0
    if command == "/status":
        return "bot_system"
    if surface_intent in _ALLOWED_SURFACE_INTENTS:
        if surface_intent == "bot":
            return "bot_system"
        if surface_intent == "abstain":
            return "abstain"
        if surface_intent == "chat":
            return "persona_chat"
    if signal in {"wrong_persona", "wrong_voice", "right_persona_reaffirm"}:
        # Without an explicit target, the correction implies the
        # persona_chat target (the engineer's most common case).
        return "persona_chat"
    return ""


# === PolicyProducer =====================================================


class PolicyProducer:
    """Non-LLM admission path for `source_kind = "policy"` commitments.

    M20.3 §1.1 frozen producer surface. The producer is deterministic:
    the same inputs always produce the same `ActiveCommitment` rows.
    The producer does NOT call an LLM and does NOT touch the conscious
    loop. It is the only legitimate source of `runtime_mode_state`
    commitments.

    `evaluate()` runs at T0 admission. It is reachable from the
    `fast_chat` path: M20.3 §4 closes the runtime invariant gap
    (fast_chat must not skip PolicyProducer).
    """

    ENGINEERING_PROXY_LABEL: str = "mvp_local_policy_admission"

    def __init__(self, adapter: ActiveCommitmentAdapter | None = None) -> None:
        self._adapter = adapter if adapter is not None else ActiveCommitmentAdapter()

    def evaluate(
        self,
        *,
        turn_context: Mapping[str, Any] | None,
        runtime_mode_flags: Mapping[str, Any] | None,
        command_envelope: Mapping[str, Any] | None,
        user_correction_signal: str | None,
    ) -> tuple[list[ActiveCommitment], list[dict[str, Any]]]:
        """Produce 0..N `ActiveCommitment` rows from the frozen table.

        Returns `(admitted, audit_events)`. `audit_events` contains
        one `PolicyAdmitted` envelope per admitted commitment, ready
        to be appended to the per-turn bus. Admitted commitments
        share their `commit_id` with the `PolicyAdmitted` event so
        callers can cross-reference.
        """
        turn_context = turn_context if isinstance(turn_context, Mapping) else {}
        runtime_mode_flags = (
            runtime_mode_flags if isinstance(runtime_mode_flags, Mapping) else {}
        )

        turn_index = int(turn_context.get("turn_index", 0) or 0)
        created_at = str(turn_context.get("at", "") or "")
        if not created_at:
            # Stable fallback for tests that omit `at`. Real callers
            # always provide an ISO 8601 string from `mvp_loop`.
            created_at = "1970-01-01T00:00:00Z"

        command, arg0 = _classify_command(command_envelope)
        surface_intent = _bounded_string(
            runtime_mode_flags.get("surface_intent"),
            default="",
            limit=32,
        )
        if surface_intent not in _ALLOWED_SURFACE_INTENTS:
            surface_intent = ""
        signal = _bounded_string(user_correction_signal, default="", limit=32)
        if signal not in _ALLOWED_USER_CORRECTION_SIGNALS:
            signal = ""
        group_mode_ingress = bool(
            runtime_mode_flags.get("group_mode_ingress_change", False)
        )

        proposals: list[dict[str, Any]] = []
        rule_kinds: list[str] = []

        # 1. /status, /mode, /persona
        if command == "/status":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="command_status",
                    command=command,
                    arg0="",
                    surface_intent=surface_intent,
                    signal="",
                    source_ref=f"command_{command}",
                    evidence_refs=[f"turn_{turn_index}_command_envelope"],
                )
            )
            rule_kinds.append("command_status")
        elif command == "/mode":
            if arg0 in _ALLOWED_RUNTIME_MODES:
                proposals.append(
                    _runtime_mode_state_proposal(
                        rule_kind="command_mode",
                        command=command,
                        arg0=arg0,
                        surface_intent=surface_intent,
                        signal="",
                        source_ref=f"command_{command}_{arg0}",
                        evidence_refs=[f"turn_{turn_index}_command_envelope"],
                    )
                )
                rule_kinds.append("command_mode")
        elif command == "/persona":
            if arg0 in _ALLOWED_RUNTIME_MODES:
                proposals.append(
                    _runtime_mode_state_proposal(
                        rule_kind="command_persona",
                        command=command,
                        arg0=arg0,
                        surface_intent=surface_intent,
                        signal="",
                        source_ref=f"command_{command}_{arg0}",
                        evidence_refs=[f"turn_{turn_index}_command_envelope"],
                    )
                )
                rule_kinds.append("command_persona")

        # 2. /quiet, /resume (outreach_intent observation-only)
        if command == "/quiet":
            proposals.append(
                _outreach_intent_proposal(
                    rule_kind="command_quiet",
                    observable="outreach_intent_off",
                    source_ref="command_quiet",
                    expected_mode="abstain",
                    evidence_refs=[f"turn_{turn_index}_command_envelope"],
                )
            )
            rule_kinds.append("command_quiet")
        elif command == "/resume":
            proposals.append(
                _outreach_intent_proposal(
                    rule_kind="command_resume",
                    observable="outreach_intent_on",
                    source_ref="command_resume",
                    expected_mode="persona_chat",
                    evidence_refs=[f"turn_{turn_index}_command_envelope"],
                )
            )
            rule_kinds.append("command_resume")

        # 3. Surface intent flags (turn-scoped, mode derivation)
        if surface_intent == "bot":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="surface_intent_bot",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal="",
                    source_ref="surface_intent_bot",
                    evidence_refs=[f"turn_{turn_index}_surface_intent"],
                )
            )
            rule_kinds.append("surface_intent_bot")
        elif surface_intent == "chat":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="surface_intent_chat",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal="",
                    source_ref="surface_intent_chat",
                    evidence_refs=[f"turn_{turn_index}_surface_intent"],
                )
            )
            rule_kinds.append("surface_intent_chat")
        elif surface_intent == "abstain":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="surface_intent_abstain",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal="",
                    source_ref="surface_intent_abstain",
                    evidence_refs=[f"turn_{turn_index}_surface_intent"],
                )
            )
            rule_kinds.append("surface_intent_abstain")

        # 4. User correction signal (from conscious-loop JSON v2)
        if signal == "wrong_persona":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="user_correction_wrong_persona",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal=signal,
                    source_ref=f"user_correction_{signal}",
                    evidence_refs=[f"turn_{turn_index}_user_correction"],
                )
            )
            rule_kinds.append("user_correction_wrong_persona")
        elif signal == "wrong_voice":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="user_correction_wrong_voice",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal=signal,
                    source_ref=f"user_correction_{signal}",
                    evidence_refs=[f"turn_{turn_index}_user_correction"],
                )
            )
            rule_kinds.append("user_correction_wrong_voice")
        elif signal == "right_persona_reaffirm":
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="user_correction_reaffirm",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal=signal,
                    source_ref="user_correction_right_persona_reaffirm",
                    evidence_refs=[f"turn_{turn_index}_user_correction"],
                )
            )
            rule_kinds.append("user_correction_reaffirm")

        # 5. group_mode ingress change (durable mutate)
        if group_mode_ingress:
            proposals.append(
                _runtime_mode_state_proposal(
                    rule_kind="group_mode_ingress_change",
                    command="",
                    arg0="",
                    surface_intent=surface_intent,
                    signal="",
                    source_ref="group_mode_ingress_change",
                    evidence_refs=[f"turn_{turn_index}_group_mode_ingress"],
                )
            )
            rule_kinds.append("group_mode_ingress_change")

        if not proposals:
            return [], []

        # Run the existing v1 adapter (so v1 schema validation, v1
        # reason codes, v1 engineering_proxy_label set, and the v1
        # commit_id derivation all apply unchanged).
        admitted, rejected = self._adapter.admit_batch(
            proposals=proposals,
            turn_index=turn_index,
            created_at=created_at,
        )

        # Build PolicyAdmitted audit events in the same order.
        # We rely on the deterministic commit_id derivation to
        # cross-reference; v1 commit_id takes (owner_id, source_ref,
        # layer, observable, created_turn) — all of which are
        # produced by `_runtime_mode_state_proposal` /
        # `_outreach_intent_proposal`. Since the source_ref is
        # unique per rule, ordering is preserved.
        rule_kind_by_ref: dict[str, str] = {}
        for kind, prop in zip(rule_kinds, proposals):
            rule_kind_by_ref[str(prop.get("source_ref", ""))] = kind

        audit_events: list[dict[str, Any]] = []
        for commitment in admitted:
            rule_kind = rule_kind_by_ref.get(commitment.source_ref, "")
            audit_events.append(
                build_policy_admitted_event(
                    commitment=commitment,
                    turn_index=turn_index,
                    at=created_at,
                    rule_kind=rule_kind,
                )
            )

        return admitted, audit_events


# === proposal builders ===================================================
# These return v1-shape proposals that the existing
# `ActiveCommitmentAdapter` will accept. The v1 adapter now consults
# `COMMITMENT_REGISTRY_V2` and `OBSERVABLE_V2` (additive: v1 ∪ v2
# entries), so v2 owners/observables pass unchanged.


def _runtime_mode_state_proposal(
    *,
    rule_kind: str,
    command: str,
    arg0: str,
    surface_intent: str,
    signal: str,
    source_ref: str,
    evidence_refs: list[str],
) -> dict[str, Any]:
    """Build a v2-shape `runtime_mode_state` proposal.

    The proposal uses `layer = "B_per_turn_commitment"` (turn-scoped
    by default) and `horizon = "same_turn_surface"` (the v2
    attribute that lets the pre-send gate pick it up). Durable
    mutations (e.g. /mode) keep the same horizon — the pre-send
    gate acts this turn; the persistent state is updated via the
    M20.2 dispatch path (which uses the v2 exception table for
    `runtime_mode_state`).
    """
    expected_mode = _resolve_runtime_mode(
        arg0=arg0,
        surface_intent=surface_intent,
        signal=signal,
        command=command,
    )
    return {
        "owner_id": "runtime_mode_state",
        "source_kind": "policy",
        "source_ref": source_ref,
        "layer": "B_per_turn_commitment",
        "observable": "runtime_mode_state",
        "observable_payload": {
            "expected_mode": expected_mode,
            "actual_mode": "",
            "mode_owner": signal or command or "policy",
            "evidence_refs": _bounded_string_list(evidence_refs, limit=8),
            "rule_kind": rule_kind,
        },
        "target": {"expected_mode": expected_mode},
        "due_at": {"kind": "next_turn"},
        "priority": 0.6,
        "confidence": 0.7,
        "evidence_refs": _bounded_string_list(evidence_refs, limit=8) or [source_ref],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_policy_admission",
        "horizon": "same_turn_surface",
    }


def _outreach_intent_proposal(
    *,
    rule_kind: str,
    observable: str,
    source_ref: str,
    expected_mode: str,
    evidence_refs: list[str],
) -> dict[str, Any]:
    """Build a v2-shape `outreach_intent_*` proposal (observation-only)."""
    return {
        "owner_id": "outreach_intent_registry",
        "source_kind": "policy",
        "source_ref": source_ref,
        "layer": "B_per_turn_commitment",
        "observable": observable,
        "observable_payload": {
            "expected_mode": expected_mode,
            "evidence_refs": _bounded_string_list(evidence_refs, limit=8),
            "rule_kind": rule_kind,
        },
        "target": {"expected_mode": expected_mode},
        "due_at": {"kind": "next_turn"},
        "priority": 0.5,
        "confidence": 0.6,
        "evidence_refs": _bounded_string_list(evidence_refs, limit=8) or [source_ref],
        "reason_codes": ["outreach_intent"],
        "engineering_proxy_label": "mvp_local_outreach",
        "horizon": "next_turn",
    }


__all__ = [
    "PolicyProducer",
    "build_policy_admitted_event",
    "is_policy_owner_accepts_policy_correction",
    "policy_producer_table_snapshot",
]
