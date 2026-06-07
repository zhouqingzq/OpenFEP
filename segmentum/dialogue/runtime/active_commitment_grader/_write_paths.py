"""M20.2.1 real write path adapters for the unified-commitment loop.

M20.2 ships no-op routing stubs. M20.2.1 wires the stubs to real
owner-state mutations for a v1 scope:

- `m13_drive_state` for `microadjust`, `next_turn`, `same_turn`
- `self_cognition_calibrated_tendencies` for `slow_promote`, `revoke`

All other (level, owner_id) combinations in M20.2 §5 remain
no-op stubs; expanding scope is a future M20.2.x milestone.

Each write function:

- reads the existing owner state via the standard state key
  (e.g. `state["m13_drive_state"]`, `state["self_cognition"]`),
- applies a small bounded mutation,
- emits a per-owner audit event alongside the existing
  `GradedCorrectionRouted` envelope on the bus,
- NEVER mutates the `SettledValue` (immutable from M20.2 perspective),
- NEVER invents a new long-term state bucket,
- NEVER calls an LLM.

The dispatcher hands the write path the originating
`ActiveCommitment` (reconstructed from observability) alongside
the `GradedCorrectionDecision`. The decision is a frozen summary;
the commitment carries the dispatch context (action, user_id,
source_ref, observable_payload) needed to perform the write.

If the owner state is missing or the dispatch context is too thin
(no action / no user_id / no source_ref), the write function is a
no-op for that dispatch decision; the existing
`GradedCorrectionRouted` event still fires so the audit surface is
complete.
"""

from __future__ import annotations

from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
)


# Bounded deltas per M20.2 §4 magnitude_after computation. The dispatcher
# has already computed `magnitude_after` and the routing stub passes
# it through; the write function applies a per-level fraction.
_MICROADJUST_DELTA = 0.05
_NEXT_TURN_DELTA = 0.10
_SAME_TURN_NUDGE_WEIGHT = 0.15


def _bounded(value: Any, *, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return max(0.0, min(1.0, v))


def _string_list(value: Any, *, limit: int = 16) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _m13_state(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the m13_drive_state dict or None if missing / wrong shape."""
    m13 = state.get("m13_drive_state")
    if isinstance(m13, Mapping):
        return dict(m13)
    return None


def _resolve_m13_action_user(
    payload: Mapping[str, Any],
    m13_state: Mapping[str, Any],
) -> tuple[str, str]:
    """Return (action, user_id) for an m13 microadjust/next_turn write.

    Falls back to deriving user_id from the most recent pattern in
    `path_patterns_by_action` so the write can still target a per-user
    traction key. Returns `("", "")` if neither is available.
    """
    action = str(payload.get("action", "") or "").strip()
    user_id = str(payload.get("user_id", "") or "").strip()
    if not user_id and isinstance(m13_state.get("path_patterns_by_action"), list):
        for row in m13_state["path_patterns_by_action"]:
            if isinstance(row, Mapping) and row.get("user_id"):
                user_id = str(row.get("user_id", "")).strip()
                break
    return action, user_id


def _m13_traction_key(action: str, user_id: str) -> str:
    return f"{action}|{user_id}" if user_id else action


# === m13_drive_state write paths =========================================


def apply_m13_microadjust(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`microadjust` for owner `m13_drive_state`.

    Bumps `traction_by_action[action|user_id]` by a small bounded
    delta derived from `magnitude_before`. Emits a per-owner
    `M13TractionMicroadjust` audit event.

    v1 scope: only updates `traction_by_action`. Path patterns,
    topic precision, and other m13 sub-structures are owned by the
    existing post-turn m13 patch path; M20.2.1 does not duplicate
    that logic.
    """
    m13 = _m13_state(state)
    if m13 is None:
        return
    payload = commitment.observable_payload or {}
    action, user_id = _resolve_m13_action_user(payload, m13)
    if not action:
        return

    traction = dict(m13.get("traction_by_action") or {})
    key = _m13_traction_key(action, user_id)
    before = _bounded(traction.get(key), default=0.0)
    delta = _MICROADJUST_DELTA * _bounded(decision.magnitude_before, default=0.0)
    after = min(1.0, before + delta)
    traction[key] = round(after, 6)
    m13["traction_by_action"] = traction
    state["m13_drive_state"] = m13

    bus.append(
        {
            "type": "M13TractionMicroadjust",
            "turn_index": decision.turn_index,
            "commit_id": decision.commit_id,
            "routed_owner_id": decision.routed_owner_id,
            "action": action,
            "user_id": user_id,
            "traction_before": round(before, 6),
            "traction_after": round(after, 6),
            "delta": round(delta, 6),
            "engineering_proxy_label": "mvp_local_m13_drive",
            "at": decision.at,
        }
    )


def apply_m13_next_turn(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`next_turn` for owner `m13_drive_state`.

    Larger traction bump than microadjust, plus appends a
    `pending_settlements` row so the existing post-turn patch path
    can see the decision and re-evaluate on the next turn.
    """
    m13 = _m13_state(state)
    if m13 is None:
        return
    payload = commitment.observable_payload or {}
    action, user_id = _resolve_m13_action_user(payload, m13)
    if not action:
        return

    traction = dict(m13.get("traction_by_action") or {})
    key = _m13_traction_key(action, user_id)
    before = _bounded(traction.get(key), default=0.0)
    delta = _NEXT_TURN_DELTA * _bounded(decision.magnitude_before, default=0.0)
    after = min(1.0, before + delta)
    traction[key] = round(after, 6)
    m13["traction_by_action"] = traction

    pending = [dict(row) for row in m13.get("pending_settlements", []) if isinstance(row, Mapping)]
    pending.append(
        {
            "commit_id": decision.commit_id,
            "action": action,
            "user_id": user_id,
            "traction_after": round(after, 6),
            "outcome": decision.outcome,
            "evidence_refs": _string_list(list(decision.evidence_refs)),
            "applied_at_turn": decision.turn_index,
        }
    )
    m13["pending_settlements"] = pending[-32:]
    state["m13_drive_state"] = m13

    bus.append(
        {
            "type": "M13TractionNextTurn",
            "turn_index": decision.turn_index,
            "commit_id": decision.commit_id,
            "routed_owner_id": decision.routed_owner_id,
            "action": action,
            "user_id": user_id,
            "traction_before": round(before, 6),
            "traction_after": round(after, 6),
            "delta": round(delta, 6),
            "engineering_proxy_label": "mvp_local_m13_drive",
            "at": decision.at,
        }
    )


def apply_m13_same_turn(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`same_turn` for owner `m13_drive_state`.

    Advisory only. M20.2 §7 caps `same_turn` to advisory fields. This
    implementation nudges the priority of the action in
    `recent_action_trace`; it does NOT mutate `traction_by_action`
    (a non-advisory field). The action is `nudge` rather than
    `set` — it adds a `priority_boost` row to the trace, and the
    next m13 evaluation reads the trace to compute the
    `evaluation.top_behavioral_pull_action` candidate margin.
    """
    m13 = _m13_state(state)
    if m13 is None:
        return
    payload = commitment.observable_payload or {}
    action, user_id = _resolve_m13_action_user(payload, m13)
    if not action:
        return

    trace = [dict(row) for row in m13.get("recent_action_trace", []) if isinstance(row, Mapping)]
    weight = _bounded(
        decision.magnitude_after or decision.magnitude_before,
        default=0.0,
    )
    trace.append(
        {
            "kind": "m20_2_1_pull_nudge",
            "action": action,
            "user_id": user_id,
            "priority_boost": round(_SAME_TURN_NUDGE_WEIGHT * weight, 6),
            "commit_id": decision.commit_id,
            "applied_at_turn": decision.turn_index,
        }
    )
    m13["recent_action_trace"] = trace[-32:]
    state["m13_drive_state"] = m13

    bus.append(
        {
            "type": "M13PullNudge",
            "turn_index": decision.turn_index,
            "commit_id": decision.commit_id,
            "routed_owner_id": decision.routed_owner_id,
            "action": action,
            "user_id": user_id,
            "priority_boost": round(_SAME_TURN_NUDGE_WEIGHT * weight, 6),
            "advisory": True,
            "engineering_proxy_label": "mvp_local_m13_drive",
            "at": decision.at,
        }
    )


# === self_cognition_calibrated_tendencies write paths ====================


def _self_cognition_dict(state: dict) -> dict[str, Any] | None:
    sc = state.get("self_cognition")
    if isinstance(sc, Mapping):
        return dict(sc)
    return None


def _self_cognition_tendencies(sc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in sc.get("calibrated_tendencies", []) if isinstance(row, Mapping)]


def _self_cognition_repair_priors(sc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in sc.get("repair_priors", []) if isinstance(row, Mapping)]


def apply_self_cognition_slow_promote(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`slow_promote` for owner `self_cognition_calibrated_tendencies`.

    Appends (or updates) an entry in `self_cognition.calibrated_tendencies`
    keyed by `commitment.source_ref`. Confidence is taken from
    `decision.magnitude_after` (clamped to [0.0, 1.0]). A companion
    `repair_priors` row is also appended for the same source_ref
    when the M19.3 promotion pattern requires it.

    The M19.3 promotion lock is maintained here: a source_ref that
    is already present with `status="active"` in
    `calibrated_tendencies` is treated as "already promoted" and
    the write is skipped (the dispatcher's `m19_3_already_promoted`
    shortcut should normally catch this first; this is a defensive
    double-check at write time).
    """
    sc = _self_cognition_dict(state)
    if sc is None:
        return
    source_ref = str(commitment.source_ref or "").strip()
    if not source_ref:
        return

    tendencies = _self_cognition_tendencies(sc)
    priors = _self_cognition_repair_priors(sc)

    # Defensive lock: if any tendency with this source_ref is already
    # active or recently downgraded, skip the promotion to avoid
    # re-promoting on re-dispatch.
    for row in tendencies:
        if (
            str(row.get("source_mismatch_key", "") or "") == source_ref
            and str(row.get("status", "") or "") == "active"
        ):
            bus.append(
                {
                    "type": "M19_3PromotionAlreadyActive",
                    "turn_index": decision.turn_index,
                    "commit_id": decision.commit_id,
                    "routed_owner_id": decision.routed_owner_id,
                    "source_mismatch_key": source_ref,
                    "tendency_id": str(row.get("id", "") or ""),
                    "engineering_proxy_label": "mvp_local_self_repair",
                    "at": decision.at,
                }
            )
            return

    confidence = _bounded(decision.magnitude_after, default=0.0)
    if confidence <= 0.0:
        return

    target_context = str(
        (commitment.observable_payload or {}).get("target_context", "")
        or (commitment.observable_payload or {}).get("context", "")
    ).strip()[:120]

    new_id = f"cal_tend_{decision.commit_id[:12]}"
    new_row = {
        "id": new_id,
        "target_context": target_context,
        "tendency_summary": (
            f"In {target_context or 'unspecified'}, dispatcher promoted a calibrated tendency."
        ),
        "confidence": round(confidence, 6),
        "source_mismatch_key": source_ref,
        "evidence_refs": _string_list(list(decision.evidence_refs)),
        "status": "active",
        "promoted_at_turn": decision.turn_index,
        "engineering_proxy_label": "mvp_local_self_repair",
    }
    tendencies.append(new_row)
    sc["calibrated_tendencies"] = tendencies[-64:]

    # Companion repair_prior (M19.3 pattern).
    new_prior = {
        "id": f"repair_prior_{decision.commit_id[:12]}",
        "target_context": target_context,
        "preferred_intervention": "reduce_assertion_strength_before_clarify",
        "confidence": round(confidence, 6),
        "source_expectation_id": source_ref,
        "source_mismatch_key": source_ref,
        "settlement_ids": [decision.commit_id],
        "evidence_refs": _string_list(list(decision.evidence_refs)),
        "status": "active",
        "promoted_at_turn": decision.turn_index,
    }
    priors.append(new_prior)
    sc["repair_priors"] = priors[-64:]
    state["self_cognition"] = sc

    bus.append(
        {
            "type": "M19_3TendencyPromoted",
            "turn_index": decision.turn_index,
            "commit_id": decision.commit_id,
            "routed_owner_id": decision.routed_owner_id,
            "tendency_id": new_id,
            "repair_prior_id": new_prior["id"],
            "source_mismatch_key": source_ref,
            "target_context": target_context,
            "confidence": round(confidence, 6),
            "evidence_refs": _string_list(list(decision.evidence_refs)),
            "engineering_proxy_label": "mvp_local_self_repair",
            "at": decision.at,
        }
    )


def apply_self_cognition_revoke(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`revoke` for owner `self_cognition_calibrated_tendencies`.

    Sets `status="revoked"` on every `calibrated_tendencies` and
    `repair_priors` entry keyed by `commitment.source_ref`. Revoked
    entries are kept (not deleted) so the M19.3 audit tail is
    durable. A companion audit event lists the affected row IDs.
    """
    sc = _self_cognition_dict(state)
    if sc is None:
        return
    source_ref = str(commitment.source_ref or "").strip()
    if not source_ref:
        return

    tendencies = _self_cognition_tendencies(sc)
    priors = _self_cognition_repair_priors(sc)

    revoked_tendency_ids: list[str] = []
    for row in tendencies:
        if str(row.get("source_mismatch_key", "") or "") == source_ref:
            if str(row.get("status", "") or "") != "revoked":
                row["status"] = "revoked"
                row["revoked_at_turn"] = decision.turn_index
                revoked_tendency_ids.append(str(row.get("id", "") or ""))

    revoked_prior_ids: list[str] = []
    for row in priors:
        if str(row.get("source_mismatch_key", "") or "") == source_ref:
            if str(row.get("status", "") or "") != "revoked":
                row["status"] = "revoked"
                row["revoked_at_turn"] = decision.turn_index
                revoked_prior_ids.append(str(row.get("id", "") or ""))

    sc["calibrated_tendencies"] = tendencies
    sc["repair_priors"] = priors
    state["self_cognition"] = sc

    bus.append(
        {
            "type": "M19_3TendencyRevoked",
            "turn_index": decision.turn_index,
            "commit_id": decision.commit_id,
            "routed_owner_id": decision.routed_owner_id,
            "source_mismatch_key": source_ref,
            "revoked_tendency_ids": revoked_tendency_ids,
            "revoked_prior_ids": revoked_prior_ids,
            "engineering_proxy_label": "mvp_local_self_repair",
            "at": decision.at,
        }
    )


# === dispatcher: route (level, owner_id) to the right write function ======


def run_m20_2_1_write_path(
    *,
    level: str,
    owner_id: str,
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> bool:
    """Top-level dispatcher for the v1-scope write paths.

    Returns True if the (level, owner_id) pair was handled, False
    if the pair is not in v1 scope (caller should still emit the
    `GradedCorrectionRouted` event so the audit surface is complete).
    """
    if level == "microadjust" and owner_id == "m13_drive_state":
        apply_m13_microadjust(decision, commitment, state, bus)
        return True
    if level == "next_turn" and owner_id == "m13_drive_state":
        apply_m13_next_turn(decision, commitment, state, bus)
        return True
    if level == "same_turn" and owner_id == "m13_drive_state":
        apply_m13_same_turn(decision, commitment, state, bus)
        return True
    if level == "slow_promote" and owner_id == "self_cognition_calibrated_tendencies":
        apply_self_cognition_slow_promote(decision, commitment, state, bus)
        return True
    if level == "revoke" and owner_id == "self_cognition_calibrated_tendencies":
        apply_self_cognition_revoke(decision, commitment, state, bus)
        return True
    # M20.4 — real write path on `group_addressee_graph.microadjust`
    # (was no-op in M20.3). The v1 group_addressee_graph.owner row
    # is in COMMITMENT_REGISTRY_V1 with graded_action_set =
    # ["microadjust", "next_turn"]. M20.4 fills in microadjust.
    if level == "microadjust" and owner_id == "group_addressee_graph":
        apply_group_addressee_graph_microadjust(decision, commitment, state, bus)
        return True
    if level == "revoke" and owner_id == "group_addressee_graph":
        apply_group_addressee_graph_revoke(decision, commitment, state, bus)
        return True
    return False


def apply_group_addressee_graph_microadjust(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`microadjust` for owner `group_addressee_graph` (M20.4).

    Appends an attribution row to `state["addressee_graph"]`
    keyed by the M18.7 commit_id. Emits a
    `GroupAddresseeGraphUpdated` audit event. The
    previously no-op write path (M20.3) is now real.
    """
    from segmentum.dialogue.runtime.m20_4_attribution import (
        write_addressee_graph_microadjust as _m20_4_write,
    )
    at = str(getattr(decision, "at", "") or "")
    event = _m20_4_write(
        state=state,
        decision=decision,
        commitment=commitment,
        at=at,
    )
    if event:
        bus.append(event)


def apply_group_addressee_graph_revoke(
    decision: GradedCorrectionDecision,
    commitment: ActiveCommitment,
    state: dict,
    bus: list,
) -> None:
    """`revoke` for owner `group_addressee_graph` (M20.4)."""
    from segmentum.dialogue.runtime.m20_4_attribution import (
        clear_addressee_graph_row as _m20_4_clear,
    )
    event = _m20_4_clear(state=state, commitment=commitment)
    if event:
        bus.append(event)


__all__ = [
    "apply_m13_microadjust",
    "apply_m13_next_turn",
    "apply_m13_same_turn",
    "apply_self_cognition_slow_promote",
    "apply_self_cognition_revoke",
    "run_m20_2_1_write_path",
]
