"""M20.3 §3 Same-turn surface horizon — pre-send gate and post-send advisory.

The same_turn surface horizon closes M20.0–M20.2 gaps B and C:

- **Gap B**: identity violation in turn T used to be settled in T+1
  (M20.1 §7 enforces T+1 minimum for all settlements). The pre-send
  gate lets the runtime enforce the violation **before** the visible
  reply is committed, on the same turn.
- **Gap C**: `same_turn` corrections were advisory only;
  already-sent wrong-persona text used to stay. The pre-send gate
  can `block` the reply (replace it with a bounded fallback) for
  one specific owner: `runtime_mode_state`.

M20.3 hard-codes the rules:

1. The pre-send gate can `block` only when the failing observable's
   owner is `runtime_mode_state` AND the owner has
   `accepts_same_turn_block = true`. For all other observables, the
   gate returns `pass` or `advisory_guidance`.
2. The post-send advisory cannot `block` (the reply is already
   committed); it writes only to next-turn `control_guidance`.
3. A `commit_id` with `horizon = "same_turn_surface"` may appear in
   at most one `SameTurnSurfaceVerdict` per turn (either pre-send
   or post-send, not both).
4. The pre-send gate runs BEFORE the assistant reply is committed;
   it cannot rewrite an already-committed reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    is_registry_v2_accepts_same_turn_block,
)


# Decision enum for the verdict. Frozen at module level so callers
# can branch deterministically.
_DECISION_PASS = "pass"
_DECISION_ADVISORY = "advisory_guidance"
_DECISION_BLOCK = "block"


# Engineering proxy label used by the audit envelope.
_ENGINEERING_PROXY_LABEL = "mvp_local_same_turn_surface"


@dataclass(frozen=True)
class SameTurnSurfaceVerdict:
    """Frozen result of one pre-send or post-send settler call.

    M20.3 §3.4 audit envelope. The caller emits one envelope per
    `commit_id` per turn; the gate enforces uniqueness via
    `_seen_commit_ids`.
    """

    horizon: str  # "pre_send" | "post_send"
    commit_ids: tuple[str, ...]
    decision: str  # "pass" | "advisory_guidance" | "block"
    owner_id: str
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    guidance: str
    replacement: str
    engineering_proxy_label: str
    turn_index: int
    at: str


def build_same_turn_surface_verdict_event(verdict: SameTurnSurfaceVerdict) -> dict[str, Any]:
    """Build the `SameTurnSurfaceVerdict` audit envelope (M20.3 §3.4)."""
    event: dict[str, Any] = {
        "type": "SameTurnSurfaceVerdict",
        "turn_index": verdict.turn_index,
        "horizon": verdict.horizon,
        "commit_ids": list(verdict.commit_ids),
        "decision": verdict.decision,
        "owner_id": verdict.owner_id,
        "evidence_refs": list(verdict.evidence_refs),
        "reason_codes": list(verdict.reason_codes),
        "engineering_proxy_label": verdict.engineering_proxy_label,
        "at": verdict.at,
    }
    if verdict.guidance:
        event["guidance"] = verdict.guidance
    if verdict.replacement:
        event["replacement"] = verdict.replacement
    return event


# === Settler-facing helpers ==============================================


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _string_list(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


# Mapping from M19.x surface_intent_outcome -> verdict outcome.
# - "consistent" -> "confirmed" -> pre-send "pass"
# - "drifted_intent" / "drifted_self_id" / "drifted_voice" -> "violated"
#   (the pre-send gate may block for runtime_mode_state; for other
#   owners it returns "advisory_guidance")
# - "ambiguous" -> "ambiguous" -> "advisory_guidance" (never block)
_SURFACE_TO_M20: dict[str, str] = {
    "consistent": "confirmed",
    "drifted_intent": "violated",
    "drifted_self_id": "violated",
    "drifted_voice": "violated",
    "ambiguous": "ambiguous",
}


def _resolve_runtime_mode_owner(commitment: ActiveCommitment) -> str:
    """Return the owner_id for a `runtime_mode_state` commitment, or "".

    The pre-send gate cares specifically about `runtime_mode_state`
    owners because that's the only owner with `accepts_same_turn_block`
    in v2. Other owners get the standard `pass` / `advisory_guidance`
    treatment.
    """
    if commitment.observable != "runtime_mode_state":
        return ""
    if commitment.owner_id != "runtime_mode_state":
        return ""
    return commitment.owner_id


def _expected_mode(commitment: ActiveCommitment) -> str:
    payload = dict(commitment.observable_payload or {})
    return _bounded_string(payload.get("expected_mode"), default="", limit=32)


def _actual_mode(draft_reply: str, observation_context: Mapping[str, Any]) -> str:
    """Best-effort derivation of the draft reply's actual mode.

    v1: read the existing `surface_consistency_verification` audit row
    that the conscious loop already produced. The actual_mode is
    `committed_surface_intent` when present, else empty.
    """
    surface = observation_context.get("surface_consistency_verification")
    if not isinstance(surface, Mapping):
        return ""
    return _bounded_string(surface.get("committed_surface_intent"), default="", limit=32)


def _surface_intent_outcome(
    observation_context: Mapping[str, Any],
) -> str:
    """Read the M19.x surface-consistency outcome from observation context."""
    surface = observation_context.get("surface_consistency_verification")
    if not isinstance(surface, Mapping):
        return ""
    return _bounded_string(
        surface.get("surface_intent_outcome"), default="", limit=32
    ).lower()


def _drift_target(
    observation_context: Mapping[str, Any],
) -> str:
    surface = observation_context.get("surface_consistency_verification")
    if not isinstance(surface, Mapping):
        return ""
    return _bounded_string(surface.get("drift_target"), default="", limit=32)


# === SameTurnSurfaceSettler ==============================================


class SameTurnSurfaceSettler:
    """M20.3 §3 pre-send gate and post-send advisory settler.

    The settler is pure: it does not mutate any long-term state
    bucket. It returns a `SameTurnSurfaceVerdict` that the caller
    (mvp_loop.py) routes to either the reply-replacement path
    (pre-send) or the `control_guidance` enrichment (post-send).
    """

    ENGINEERING_PROXY_LABEL: str = _ENGINEERING_PROXY_LABEL

    def __init__(self) -> None:
        # Per-instance dedup: at most one verdict per `commit_id`
        # per turn. State is reset at the start of each turn by
        # `reset_turn_dedup` (called by mvp_loop.run_turn).
        self._seen_commit_ids_pre: set[str] = set()
        self._seen_commit_ids_post: set[str] = set()

    # -- public API --------------------------------------------------------

    def reset_turn_dedup(self) -> None:
        """Clear per-turn dedup state. Called by mvp_loop.run_turn."""
        self._seen_commit_ids_pre = set()
        self._seen_commit_ids_post = set()

    def run_pre_send(
        self,
        draft_reply: str,
        horizon_commitments: list[ActiveCommitment],
        *,
        observation_context: Mapping[str, Any] | None = None,
        turn_index: int = 0,
        at: str = "",
    ) -> SameTurnSurfaceVerdict | None:
        """Pre-send gate (M20.3 §3.2).

        Returns a `SameTurnSurfaceVerdict` if the gate has something
        to act on, or `None` when there is no `horizon =
        "same_turn_surface"` commitment for this turn.

        Block rule: the gate may `block` only when the failing
        observable's owner is `runtime_mode_state` and the v2 owner
        row has `accepts_same_turn_block = true`. For all other
        observables, the gate returns `pass` or `advisory_guidance`.
        """
        observation_context = observation_context or {}
        relevant = [
            c for c in horizon_commitments
            if c.horizon == "same_turn_surface"
            and c.commit_id not in self._seen_commit_ids_pre
            and c.commit_id not in self._seen_commit_ids_post
        ]
        if not relevant:
            return None

        # Aggregate. When multiple commitments are admitted, the
        # strongest signal wins (block > advisory > pass). The
        # owner_id is the one whose commitment the verdict reports;
        # if `runtime_mode_state` is present, it claims the verdict
        # (the only owner that can block).
        owner_id = ""
        any_violated = False
        any_ambiguous = False
        evidence_refs: list[str] = []
        reason_codes: list[str] = []
        guidance_parts: list[str] = []

        for c in relevant:
            owner_id = owner_id or c.owner_id
            evidence_refs.extend(_string_list(list(c.evidence_refs), limit=8))
            reason_codes.append("horizon_same_turn_surface")
            # Derive per-commitment signal.
            if c.observable == "runtime_mode_state":
                expected = _expected_mode(c)
                actual = _actual_mode(draft_reply, observation_context)
                outcome = _SURFACE_TO_M20.get(_surface_intent_outcome(observation_context), "")
                if outcome == "violated" and expected and actual and expected != actual:
                    any_violated = True
                    reason_codes.append("runtime_mode_state_violated")
                    guidance_parts.append(
                        f"draft persona={actual or 'unknown'} but expected {expected}"
                    )
                elif outcome == "ambiguous":
                    any_ambiguous = True
                    reason_codes.append("runtime_mode_state_ambiguous")
                elif not outcome:
                    # No surface audit yet. The gate falls back to
                    # expected vs. (unknown) actual and treats the
                    # absence of an actual as "ambiguous", not
                    # "violated": the LLM self-audit may not have
                    # run on this turn (e.g. fast_chat). Blocking on
                    # a missing audit would be too eager.
                    any_ambiguous = True
                    reason_codes.append("runtime_mode_state_audit_absent")
            elif c.observable == "identity_voice_match":
                # Other horizon owners never block. The verdict is
                # advisory only.
                outcome = _SURFACE_TO_M20.get(_surface_intent_outcome(observation_context), "")
                if outcome == "violated":
                    any_violated = True
                    reason_codes.append("identity_voice_match_violated")
                elif outcome == "ambiguous":
                    any_ambiguous = True
                    reason_codes.append("identity_voice_match_ambiguous")
                else:
                    reason_codes.append("identity_voice_match_pass")
            elif c.observable == "boundary_handled":
                outcome = _SURFACE_TO_M20.get(_surface_intent_outcome(observation_context), "")
                if outcome == "violated":
                    any_violated = True
                    reason_codes.append("boundary_handled_violated")
                else:
                    reason_codes.append("boundary_handled_advisory")
            elif c.observable == "pacing_match":
                outcome = _SURFACE_TO_M20.get(_surface_intent_outcome(observation_context), "")
                if outcome == "violated":
                    any_violated = True
                    reason_codes.append("pacing_match_violated")
                else:
                    reason_codes.append("pacing_match_advisory")
            else:
                reason_codes.append(f"unknown_horizon_observable:{c.observable}")

        # Decide.
        decision = _DECISION_PASS
        replacement = ""
        if any_violated and owner_id == "runtime_mode_state" and is_registry_v2_accepts_same_turn_block(owner_id):
            decision = _DECISION_BLOCK
            replacement = _assistant_identity_repair_fallback(
                expected_mode=_expected_mode(relevant[0]),
            )
            reason_codes.append("pre_send_block_runtime_mode_state")
        elif any_violated:
            decision = _DECISION_ADVISORY
            reason_codes.append("pre_send_advisory_only")
        elif any_ambiguous:
            decision = _DECISION_ADVISORY
            reason_codes.append("pre_send_advisory_ambiguous")
        else:
            reason_codes.append("pre_send_pass")

        # Dedup: mark all `commit_id`s as seen.
        for c in relevant:
            self._seen_commit_ids_pre.add(c.commit_id)

        return SameTurnSurfaceVerdict(
            horizon="pre_send",
            commit_ids=tuple(c.commit_id for c in relevant),
            decision=decision,
            owner_id=owner_id,
            evidence_refs=tuple(_dedup_preserve_order(evidence_refs)),
            reason_codes=tuple(_dedup_preserve_order(reason_codes)),
            guidance="; ".join(g for g in guidance_parts if g)[:240],
            replacement=replacement,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
            turn_index=turn_index,
            at=at,
        )

    def run_post_send(
        self,
        committed_reply: str,
        horizon_commitments: list[ActiveCommitment],
        *,
        observation_context: Mapping[str, Any] | None = None,
        turn_index: int = 0,
        at: str = "",
    ) -> SameTurnSurfaceVerdict | None:
        """Post-send advisory (M20.3 §3.3).

        Returns `None` when there is nothing to act on. The post-send
        path can never `block`: the reply is already committed. It
        only emits `pass` or `advisory_guidance`. The caller writes
        the guidance to next-turn `control_guidance`.
        """
        observation_context = observation_context or {}
        relevant = [
            c for c in horizon_commitments
            if c.horizon == "same_turn_surface"
            and c.commit_id not in self._seen_commit_ids_pre
            and c.commit_id not in self._seen_commit_ids_post
        ]
        if not relevant:
            return None

        # Post-send only handles non-runtime owners (the runtime
        # owner already had its chance pre-send). Filter to those.
        non_runtime = [c for c in relevant if c.owner_id != "runtime_mode_state"]
        if not non_runtime:
            return None

        owner_id = non_runtime[0].owner_id
        evidence_refs: list[str] = []
        reason_codes: list[str] = ["horizon_same_turn_surface"]
        guidance_parts: list[str] = []
        any_violated = False
        any_ambiguous = False

        for c in non_runtime:
            evidence_refs.extend(_string_list(list(c.evidence_refs), limit=8))
            outcome = _SURFACE_TO_M20.get(_surface_intent_outcome(observation_context), "")
            if outcome == "violated":
                any_violated = True
                reason_codes.append(f"{c.observable}_violated")
                guidance_parts.append(
                    f"post_send: {c.observable} violated"
                )
            elif outcome == "ambiguous":
                any_ambiguous = True
                reason_codes.append(f"{c.observable}_ambiguous")
            else:
                reason_codes.append(f"{c.observable}_pass")

        if any_violated:
            decision = _DECISION_ADVISORY
            reason_codes.append("post_send_advisory_violated")
        elif any_ambiguous:
            decision = _DECISION_ADVISORY
            reason_codes.append("post_send_advisory_ambiguous")
        else:
            decision = _DECISION_PASS
            reason_codes.append("post_send_pass")

        for c in non_runtime:
            self._seen_commit_ids_post.add(c.commit_id)

        return SameTurnSurfaceVerdict(
            horizon="post_send",
            commit_ids=tuple(c.commit_id for c in non_runtime),
            decision=decision,
            owner_id=owner_id,
            evidence_refs=tuple(_dedup_preserve_order(evidence_refs)),
            reason_codes=tuple(_dedup_preserve_order(reason_codes)),
            guidance="; ".join(g for g in guidance_parts if g)[:240],
            replacement="",
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
            turn_index=turn_index,
            at=at,
        )


# === module-level helpers ===============================================


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _assistant_identity_repair_fallback(*, expected_mode: str) -> str:
    """Bounded fallback used when the pre-send gate `block`s.

    Mirrors `mvp_loop._assistant_identity_repair_fallback` but is
    kept local to avoid a circular import. The fallback is a short,
    bounded Chinese string; the v1 fallback also covers bot_command
    cases.
    """
    expected = (expected_mode or "").strip().lower()
    if expected == "bot_system":
        return "在线，路由正常，待命中。"
    if expected == "abstain":
        return "这一轮我不接，你自己拿主意。"
    # Default persona_chat / roleplay / unknown -> generic identity
    # reaffirmation. The M19.x surface_consistency_verification
    # audit tells the conscious loop which identity was claimed;
    # the persona's name is not in v1 scope for the gate itself.
    return "刚才那句身份说乱了，我按当前这个身份继续说。"


__all__ = [
    "SameTurnSurfaceSettler",
    "SameTurnSurfaceVerdict",
    "build_same_turn_surface_verdict_event",
]
