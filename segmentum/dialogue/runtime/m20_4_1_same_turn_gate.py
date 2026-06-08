"""M20.4.1 v1: Same-Turn Addressee Hypothesis Gate.

Closes the G1 cross-turn-only gap from M20.4 v1. The
conscious loop runs AFTER `_decide_group_reply_policy` in
`mvp_loop.run_turn` (the existing v1 ordering). M18.5's
structural decision is made before the M18.7 hypothesis is
even available. M20.4 v1 ships a cross-turn feedback row
that flips the M18.5 outcome on T+1+. M20.4.1 adds a
same-turn gate that runs immediately after the conscious
loop and may override the M18.5 outcome in T0.

Frozen v1 design (per `prompts/M20.4.1_Work_Prompt.md`):

- Pure rule (no LLM). Engineering deterministic, in line
  with CLAUDE.md "no keyword / regex" red line.
- Engagement rule (matches M20.4 v1 tie-breaker, the C1
  fix): `confidence > 0.85` AND `ambiguity_band == "high"`
  AND `addressed_participant_ids` empty AND
  `mentioned_participant_ids` empty AND `reply_to_turn_id`
  None AND `m18_5_structural_decision in
  {clarify_addressee, no_reply}`.
- The gate is in-line (a function call), not a M20.1
  scheduler path. The bus event is the audit envelope, not
  an admission. No `ActiveCommitment` rows.
- The M18.5 outcome is preserved in the audit envelope
  (`m18_5_structural_decision` field). The override only
  affects the visible reply; side effects (memory, episode,
  M19.x expectation outcome settlement) read M18.5's
  structural outcome.
- Per DECIDED 7, the gate ships same-turn capability for
  the **addressee** hypothesis only. Reaction attribution
  stays on the M20.4 v1 T+1 path.

Architecture position (M20.4.1 wiring slot 3, immediately
after the conscious loop):

```text
1. _decide_group_reply_policy     <- M18.5 structural (T0)
2. conscious loop (LLM)            <- M18.7 fields produced
3. <NEW> same_turn_addressee_      <- pure rule, may override
       hypothesis_gate              action in T0
4. PolicyProducer (M20.3) post-
   conscious + M20.4 producer
5. ... thinking, memory, recall ...
6. M18.5 enforcement point         <- reads gate override slot
                                      and applies override
7. M20.3 same_turn_surface pre-send gate (runtime_mode_state)
8. reply commit
```

CLAUDE.md compliance:
- No new LLM stage.
- No regex / keyword matching. The gate reads numeric
  confidence, frozen enum, and list emptiness only.
- The M18.5 decision tree is unchanged.
- fast_chat safe (no LLM).
- Rationale / participant_id never persist beyond what
  M18.7 already persists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# === Frozen v1 constants =================================================

# Tie-breaker confidence threshold (matches M20.4 v1, the
# DECIDED 2 / DECIDED 6 rule). Strict inequality.
M20_4_1_TIE_BREAKER_CONFIDENCE_MIN: float = 0.85

# Allowed ambiguity_band values. Aligned with the M18.2 v1
# frozen string and the M20.4 v1 ambiguity bands.
M20_4_1_AMBIGUITY_BANDS: frozenset[str] = frozenset({"low", "medium", "high"})

# M18.5 decisions the gate may override. Other decisions
# (e.g., `reply_to_current_speaker`, `defer_side_thread`)
# leave the M18.5 path unchanged.
M20_4_1_OVERRIDABLE_DECISIONS: frozenset[str] = frozenset(
    {"clarify_addressee", "no_reply"}
)

# === P3 kill-switch (2026-06-08) ==========================================
# Real-LLM calibration on the M18.7.1 held-out fixture
# (see reports/m18_7_2_implementation_summary.md) shows
# LLM accuracy in the override band (conf >= 0.85) is
# 0.5 on addressee and 0.0 on reaction. Strict-inequality
# is not enough: the gate still fires in the high band
# and overrides the M18.5 structural decision visible
# to the user.
#
# P3 holds the override in **audit-only** mode. When
# M20_4_1_OVERRIDE_ENABLED is False (default):
#   - The gate still runs (rule check, verdict build,
#     bounded state-surface append, bus event emit).
#   - The override handoff to M18.5 is NOT written. M18.5
#     applies its structural path unchanged, so the
#     visible reply is what M18.5 decided.
#   - The verdict envelope still records
#     `m20_4_1_audit_only: True` so the production
#     diagnose surface can count "would-have-fired"
#     cases even though no override was applied.
# This is a safety gate, not a removal. Re-enable by
# editing the constant to True (explicit, not env-flag,
# so the override path is never live-by-default).
M20_4_1_OVERRIDE_ENABLED: bool = False

# State surface cap. Frozen at 8 in v1 (rolling window).
M20_4_1_STATE_SURFACE_LIMIT: int = 8

# Engineering proxy label (additive in V2; shared with
# M18.7 / M20.4).
M20_4_1_ENGINEERING_PROXY_LABEL: str = "mvp_local_group_attribution"

# Reason codes (M20.4.1 v1).
REASON_GATE_FIRED: str = "m20_4_1_same_turn_fired"
REASON_GATE_SILENT: str = "m20_4_1_same_turn_silent"

# Override handoff key in `state`. Single slot (per turn),
# read once at the M18.5 enforcement point and cleared
# immediately. Does not leak to T+1.
STATE_PENDING_OVERRIDE_KEY: str = "m20_4_1_pending_override"
STATE_OUTCOMES_KEY: str = "m20_4_1_same_turn_gate_outcomes"

# Bus event type (frozen at v1).
BUS_EVENT_TYPE: str = "SameTurnAddresseeHypothesisGateVerdict"

# Override decision label (frozen at v1).
DECISION_OVERRIDE: str = "overridden_to_reply_to_current_speaker"


# === Verdict dataclass ====================================================


@dataclass(frozen=True)
class SameTurnAddresseeHypothesisGateVerdict:
    """Frozen audit envelope for the M20.4.1 same-turn gate.

    Carries the override decision, the preserved M18.5
    structural outcome, the M18.7 commit_ids that fired
    the gate, and the bounded evidence_refs. The caller
    emits one verdict per `turn_index` when the rule
    fires; not emitted when the rule does not fire.

    `m20_4_1_audit_only` (P3, 2026-06-08): when True,
    the verdict's bus event and state-surface row were
    recorded, but the override handoff to the M18.5
    enforcement point was NOT written. The M18.5
    structural decision applies unchanged on the visible
    reply. Audit-only is the default at the kill-switch
    `M20_4_1_OVERRIDE_ENABLED = False`; when the
    kill-switch is flipped to True, the verdict's
    `m20_4_1_audit_only` is False and the override
    handoff IS written.
    """

    decision: str
    m18_5_structural_decision: str
    commit_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    engineering_proxy_label: str
    turn_index: int
    at: str
    m20_4_1_audit_only: bool = False


# === Bounded helpers ======================================================


def _bounded_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return default
    v = float(value)
    if v != v:  # NaN
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _bounded_evidence_refs(value: Any, *, limit: int = 16) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:120])
        if len(out) >= limit:
            break
    return out


# === State surface helpers ================================================


def _get_outcomes(state: dict) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []
    surface = state.get(STATE_OUTCOMES_KEY)
    if isinstance(surface, list):
        return surface
    return []


def _append_outcome(state: dict, row: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    surface = state.get(STATE_OUTCOMES_KEY)
    if not isinstance(surface, list):
        surface = []
    surface.append(row)
    if len(surface) > M20_4_1_STATE_SURFACE_LIMIT:
        surface = surface[-M20_4_1_STATE_SURFACE_LIMIT:]
    state[STATE_OUTCOMES_KEY] = surface


def _set_pending_override(state: dict, verdict: SameTurnAddresseeHypothesisGateVerdict) -> None:
    if not isinstance(state, dict):
        return
    state[STATE_PENDING_OVERRIDE_KEY] = verdict


def clear_pending_override(state: dict) -> None:
    """Clear the M20.4.1 override handoff slot.

    Called by mvp_loop after the M18.5 enforcement point
    reads (and applies) the override. Idempotent; safe to
    call on a state that does not have the slot.
    """
    if not isinstance(state, dict):
        return
    state.pop(STATE_PENDING_OVERRIDE_KEY, None)


def get_pending_override(state: dict) -> SameTurnAddresseeHypothesisGateVerdict | None:
    """Read the M20.4.1 override handoff slot.

    Returns the verdict if the gate fired on this turn,
    else None. The caller (M18.5 enforcement point) reads
    this once per turn.
    """
    if not isinstance(state, dict):
        return None
    verdict = state.get(STATE_PENDING_OVERRIDE_KEY)
    if isinstance(verdict, SameTurnAddresseeHypothesisGateVerdict):
        return verdict
    return None


# === Bus event builder ====================================================


def build_same_turn_gate_verdict_event(
    verdict: SameTurnAddresseeHypothesisGateVerdict,
) -> dict[str, Any]:
    """Build the `SameTurnAddresseeHypothesisGateVerdict` audit event."""
    return {
        "type": BUS_EVENT_TYPE,
        "turn_index": int(verdict.turn_index),
        "decision": str(verdict.decision),
        "m18_5_structural_decision": str(verdict.m18_5_structural_decision),
        "commit_ids": list(verdict.commit_ids),
        "evidence_refs": list(verdict.evidence_refs),
        "reason_codes": list(verdict.reason_codes),
        "engineering_proxy_label": str(verdict.engineering_proxy_label),
        "at": str(verdict.at),
        "m20_4_1_audit_only": bool(verdict.m20_4_1_audit_only),
    }


# === Verdict builder ======================================================


def _build_verdict(
    *,
    m18_5_structural_decision: str,
    commit_ids: list[str],
    evidence_refs: list[str],
    turn_index: int,
    at: str,
    m20_4_1_audit_only: bool = False,
) -> SameTurnAddresseeHypothesisGateVerdict:
    return SameTurnAddresseeHypothesisGateVerdict(
        decision=DECISION_OVERRIDE,
        m18_5_structural_decision=str(m18_5_structural_decision or ""),
        commit_ids=tuple(commit_ids),
        evidence_refs=tuple(evidence_refs),
        reason_codes=(REASON_GATE_FIRED,),
        engineering_proxy_label=M20_4_1_ENGINEERING_PROXY_LABEL,
        turn_index=int(turn_index),
        at=str(at),
        m20_4_1_audit_only=bool(m20_4_1_audit_only),
    )


# === Gate rule (pure) =====================================================


def _gate_rule_engaged(
    *,
    conscious_addressee: Mapping[str, Any] | None,
    binding: Mapping[str, Any] | None,
    m18_5_structural_decision: str,
) -> bool:
    """Apply the M20.4.1 v1 engagement rule (pure, no side effects).

    Returns True iff every condition holds:

    - `conscious_addressee` is a non-empty mapping
    - `addressed_to_assistant` is True
    - `participant_id` is non-empty (M18.4 disclosure guard)
    - `confidence` is in (0.85, 1.0] (strict `>` 0.85)
    - `ambiguity_band` is "high"
    - `addressed_participant_ids` is empty
    - `mentioned_participant_ids` is empty
    - `reply_to_turn_id` is empty / None
    - `m18_5_structural_decision in
      {clarify_addressee, no_reply}`

    This is the same rule as M20.4 v1's
    `_tie_breaker_engaged` (C1 fix: AND not OR), with the
    addition of `addressed_to_assistant` and
    `participant_id` checks from the M18.7 v2 attribute.
    """
    if not isinstance(conscious_addressee, Mapping) or not conscious_addressee:
        return False
    if not _bounded_bool(conscious_addressee.get("addressed_to_assistant")):
        return False
    participant_id = _bounded_string(
        conscious_addressee.get("participant_id", ""), default=""
    )
    if not participant_id:
        return False  # M18.4 disclosure forbade the identification
    confidence = _bounded_float(conscious_addressee.get("confidence", 0.0))
    if not (confidence > M20_4_1_TIE_BREAKER_CONFIDENCE_MIN):
        return False
    binding = binding or {}
    ambiguity_band = _bounded_string(binding.get("ambiguity_band", ""))
    if ambiguity_band != "high":
        return False
    addressed = list(binding.get("addressed_participant_ids", []) or [])
    if addressed:
        return False
    mentioned = list(binding.get("mentioned_participant_ids", []) or [])
    if mentioned:
        return False
    reply_to = _bounded_string(binding.get("reply_to_turn_id", ""))
    if reply_to:
        return False
    if m18_5_structural_decision not in M20_4_1_OVERRIDABLE_DECISIONS:
        return False
    return True


# === Top-level gate function ==============================================


def same_turn_addressee_hypothesis_gate(
    *,
    conscious_plan: Mapping[str, Any] | None,
    group_turn_binding: Mapping[str, Any] | None,
    m18_5_structural_decision: str,
    bus: list,
    state: dict,
    turn_index: int,
    now: str,
) -> SameTurnAddresseeHypothesisGateVerdict | None:
    """M20.4.1 same-turn gate.

    Runs immediately after the conscious loop, BEFORE the
    reply generation stages. Reads the M18.7
    `addressee_hypothesis` v2 attribute on `conscious_plan`
    and the v1 group_turn_binding snapshot, applies the v1
    engagement rule, and:

    - On fire: builds a verdict, emits the bus event,
      appends to `state["m20_4_1_same_turn_gate_outcomes"]`
      (bounded tail ≤ 8), and writes the verdict to the
      single-slot handoff `state["m20_4_1_pending_override"]`
      for the M18.5 enforcement point.
    - On silence: returns None. M18.5's structural path
      applies unchanged. No bus event. No state surface
      entry. No override.

    Per DECIDED 9, the override only affects the visible
    reply. Side effects (memory, episode, M19.x
    expectation outcome settlement) read M18.5's
    structural outcome; the verdict preserves it in the
    audit envelope.
    """
    conscious = conscious_plan or {}
    conscious_addressee = conscious.get("addressee_hypothesis")
    binding = group_turn_binding or {}

    # Always clear the per-turn override slot first so a
    # previous-turn verdict does not leak into this turn.
    clear_pending_override(state)

    engaged = _gate_rule_engaged(
        conscious_addressee=conscious_addressee
        if isinstance(conscious_addressee, Mapping)
        else None,
        binding=binding,
        m18_5_structural_decision=str(m18_5_structural_decision or ""),
    )
    if not engaged:
        return None

    commit_ids = _bounded_evidence_refs(
        [str((conscious_addressee or {}).get("participant_id", "") or "")],
        limit=8,
    )
    if conscious_addressee and conscious_addressee.get("participant_id"):
        # The M18.7 commit_id is the participant_id echo
        # from the normalized hypothesis (M18.7 contract).
        # Use a stable identifier — engineering records
        # only the participant_id (which is the M18.4-
        # compliant echo), not the rationale.
        commit_ids = [
            str(conscious_addressee.get("participant_id", "") or "").strip()[:64]
        ]
    evidence_refs = _bounded_evidence_refs(
        (conscious_addressee or {}).get("evidence_refs")
        if isinstance(conscious_addressee, Mapping)
        else []
    )

    # P3 kill-switch (2026-06-08): when disabled, the gate
    # still records its verdict and bus event (audit-only
    # mode) but does NOT write the override handoff. M18.5
    # applies its structural decision unchanged on the
    # visible reply. The verdict's `m20_4_1_audit_only`
    # flag carries the diagnostic signal end-to-end so
    # production diagnose can count "would-have-fired"
    # cases even though no override was applied.
    audit_only = not M20_4_1_OVERRIDE_ENABLED
    verdict = _build_verdict(
        m18_5_structural_decision=str(m18_5_structural_decision or ""),
        commit_ids=commit_ids,
        evidence_refs=evidence_refs,
        turn_index=int(turn_index),
        at=str(now),
        m20_4_1_audit_only=audit_only,
    )

    # Append to bounded state surface.
    _append_outcome(state, build_same_turn_gate_verdict_event(verdict))

    # Emit bus event (caller is responsible for any
    # additional audit; we append to the per-turn bus).
    if isinstance(bus, list):
        bus.append(build_same_turn_gate_verdict_event(verdict))

    # Write the override handoff for the M18.5 enforcement
    # point ONLY when the kill-switch is enabled. When
    # disabled, M18.5 sees no override and the visible
    # reply is what M18.5 decided.
    if not audit_only:
        _set_pending_override(state, verdict)

    return verdict


__all__ = [
    "BUS_EVENT_TYPE",
    "DECISION_OVERRIDE",
    "M20_4_1_AMBIGUITY_BANDS",
    "M20_4_1_ENGINEERING_PROXY_LABEL",
    "M20_4_1_OVERRIDABLE_DECISIONS",
    "M20_4_1_OVERRIDE_ENABLED",
    "M20_4_1_STATE_SURFACE_LIMIT",
    "M20_4_1_TIE_BREAKER_CONFIDENCE_MIN",
    "REASON_GATE_FIRED",
    "REASON_GATE_SILENT",
    "STATE_OUTCOMES_KEY",
    "STATE_PENDING_OVERRIDE_KEY",
    "SameTurnAddresseeHypothesisGateVerdict",
    "build_same_turn_gate_verdict_event",
    "clear_pending_override",
    "get_pending_override",
    "same_turn_addressee_hypothesis_gate",
]
