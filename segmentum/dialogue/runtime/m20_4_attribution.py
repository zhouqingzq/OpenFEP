"""M20.4 v1: Attribution Commitment Bridge.

M20.4 reads the M18.7 structured hypothesis surface
(`state["m18_7_attribution_hypotheses"]`) and admits
`ActiveCommitment` rows. The M20.1 scheduler settles them
via LLM-judge settlers; the M20.2 dispatcher routes
graded correction to a *real* write path on
`group_addressee_graph.microadjust` (was no-op in M20.3).

Frozen v1 design (per `prompts/M20.4_Work_Prompt.md`):

- Two v3 observables: `addressee_target_match`,
  `reaction_attribution_match`. Both LLM-judge; outcome set
  `{confirmed, violated, ambiguous}`.
- Producer admit rule: M18.7 field non-empty AND
  `confidence >= 0.4` AND `participant_id != ""`. Every
  admit is on the existing `group_addressee_graph` owner
  (no new owner rows).
- Settlers: `AddresseeTargetMatchLLMJudgeSettler` compares
  the M18.7 hypothesis against the **INBOUND turn** excerpt
  (C2 fix, NOT the assistant's reply).
  `ReactionAttributionMatchLLMJudgeSettler` compares against
  the attributed turn excerpt.
- Cross-turn tie-breaker feedback row: when the dispatcher
  emits `microadjust` + `confirmed` + `ambiguity_band ==
  "high"` + `confidence > 0.85` + structural NOT
  explicit-addressee-of-another + M18.5 decision was
  `clarify_addressee` or `no_reply`, the feedback row
  records `tie_breaker_engaged=True` and
  `patched_decision="reply_to_current_speaker"`. M18.5
  reads the row on subsequent turns.

CLAUDE.md compliance:
- Producer reads state, not user text. Settlers are LLM
  judges; engineering only validates shapes and persists
  state. No regex / keyword matching in engineering.
- Rationale text is NEVER in any persisted surface; the
  audit envelope records `rationale_chars` as length
  only.
- Bounded excerpts (200 chars) are admitted at admission
  time, not re-read at settle time.

M20.4 v1 ships **cross-turn feedback only**. The same-turn
gate is M20.4.1 territory.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
    SettlerUnavailable,
)


# === Frozen v1 constants =================================================

# Producer admit threshold (per M20.4 DECIDED 1 / OPEN→DECIDED 1).
# Frozen at 0.4 in v1; M18.7.1 calibration may revise.
M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN: float = 0.4

# P0-4 (2026-06-09) — sub-class admit threshold for the
# `addressee_target_match` observable when the M18.7 hypothesis
# is `addressed_to_assistant == True` (LLM says the message
# IS directed at the assistant).
#
# M18.7.1 v2 + P1 real-LLM calibration (commit 26d2157,
# `reports/m18_7_1_p1_precision_recall_split.md`):
#   - `precision_on_not_addressed` = 1.0 (LLM is perfect on
#     "not addressed" claims; n=4, all correct including
#     1 noemit counted under v2 fix).
#   - `recall_on_addressed` = 0.0 (LLM misses ALL 4
#     "addressed" cases; 3 fn_addressed_present +
#     1 fn_addressed_noemit).
#   - High-band overconfidence drift: conf=0.85 bin has
#     1 wrong (gap 0.85, the largest single-bin gap);
#     conf=0.95 bin has 1 wrong out of 3 (gap 0.283).
#
# Therefore the LLM is structurally asymmetric: its
# "addressed" claims are unreliable even at high confidence.
# The v1 uniform 0.4 admit threshold admits too many
# false "addressed" claims that the settler then has to
# process (and the LLM judge may incorrectly confirm,
# because the LLM judge's job is "is the hypothesis
# consistent with the inbound turn", not "is the
# hypothesis correct" — the judge's answer depends on
# the LLM's reading of the inbound turn, which can agree
# with the M18.7 hypothesis for the wrong reasons).
#
# P0-4 raises the admit threshold for the "addressed"
# sub-class from 0.4 to 0.7. "Not addressed" claims
# stay at the v1 0.4 (the LLM is 100% precise on these).
# Reaction claims are unchanged (0.4 across the board;
# the joint-axis asymmetry is in the *decision* to emit,
# not in the admit threshold).
#
# The 0.7 value is calibrated against the bqxsmofri
# drift signature: at conf=0.7 the 0.50-0.60 band has
# 1 wrong (gap 0.6), but the 0.80-0.90 and 0.90-1.00
# bands (which are the M20.4 actionable signal) start
# above 0.7. Setting the threshold at 0.7 admits the
# 0.80-0.90 / 0.90-1.00 cases (which the settler still
# judges) while rejecting the 0.50-0.60 case (which is
# the high-band overconfidence drift starting point).
#
# M20.4.1 (same-turn gate, currently in P3 kill-switch
# audit-only mode) and the M20.4 v1 tie-breaker (per-field
# threshold 0.9 / 0.7) are unchanged by P0-4. P0-4 is a
# producer-only change.
M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED: float = 0.7

# P0-5 (2026-06-09) — write-path filter for the
# `addressee_target_match` observable when the M18.7
# hypothesis is `addressed_to_assistant == True`.
#
# P0-4 raises the producer admit bar for the
# "addressed" sub-class to 0.7. P0-5 adds a *second*
# filter at the write path: even after the producer
# admits an "addressed" claim and the settler confirms
# it (the M20.1 scheduler only calls the write path on
# `microadjust + confirmed`), the write path skips
# the persistent `state["addressee_graph"]` write when
# the M18.7 hypothesis confidence is below 0.9.
#
# Why 0.9 (strict `>`):
#   - P1 bqxsmofri: 0.85 conf bin has 1 wrong (gap
#     0.85 — the largest single-bin gap). Rejected by
#     0.9 (the only conf=0.85 case in this run is
#     filtered out).
#   - P1 bqxsmofri: 0.95 conf bin has 1 wrong out of
#     3 (gap 0.283). Admitted by 0.9 (3 cases, 2
#     correct + 1 wrong — 67% accuracy on this bin).
#   - 0.9 matches the M20.4 v1 tie-breaker threshold
#     for the addressee field (per-field 0.9, strict
#     `>`). The write path is at least as strict as
#     the tie-breaker; the threshold values are
#     consistent.
#
# Why a write-path filter (not just the producer):
#   - The producer admit rule filters the *first* layer
#     of unreliable claims (P0-4). The settler is a
#     separate LLM that judges "is the M18.7 hypothesis
#     consistent with the inbound turn" — a different
#     question from "is the hypothesis correct". We do
#     not have direct settler-accuracy data; the
#     conservative assumption is that the settler's
#     agreement with the M18.7 hypothesis is correlated
#     with the M18.7 hypothesis's reliability.
#   - The write path persists data: rows in
#     `state["addressee_graph"]` are read on later
#     turns (M18.5 tie-breaker feedback, future
#     retrieval). Writing an unreliable row corrupts
#     the future. Skipping the write is reversible;
#     the producer admit and settler judgment still
#     run, so the tie-breaker feedback path is
#     unaffected.
#
# M20.4 v1 design intent: the write path is supposed
# to be a memory of attribution events. P0-5 narrows
# this to "memory of *reliable* attribution events"
# (precision 1.0 sub-class always writes; "addressed"
# sub-class writes only at high confidence). The
# "not addressed" path is unchanged (precision 1.0
# per P1, so always writing is correct).
#
# The 0.9 value is **directional, not definitive**.
# A future M18.7.1 stability rerun on settler
# accuracy (P2/P3) may surface a tighter value. M20.4
# owner can revise
# `M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE`
# up or down with a documented decision.
M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE: float = 0.9

# Tie-breaker engagement threshold (per M20.4 DECIDED 2).
# Strict inequality. P0-3 (2026-06-08) splits the v1 single
# threshold (0.85) into per-field thresholds because M18.7.1
# v3 real-LLM calibration surfaced that 0.85 is too lax for
# reaction (engaging on 0/1 correct at 0.85 confidence) and
# addressee (engaging on 2/4 correct at 0.95 confidence).
#
# Per-field values from M18.7.1 v3 candidate_tie_breaker_min:
#   - addressee: 0.9  (calibration ECE 0.225, Brier 0.226)
#   - reaction : 0.7  (calibration ECE 0.258, Brier 0.202)
#
# `_tie_breaker_engaged` dispatches by `kind` (derived from
# `commitment.observable` at the call site). Unknown kinds
# fall back to `_M20_4_TIE_BREAKER_DEFAULT` (the v1 0.85).
M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND: dict[str, float] = {
    "addressee": 0.9,
    "reaction": 0.7,
}
_M20_4_TIE_BREAKER_DEFAULT: float = 0.85

# Backward-compat alias. P0-3 (2026-06-08) splits the v1
# single threshold into per-field thresholds. The alias
# preserves the v1 module-level constant for any consumer
# still reading the single value; new code should use
# `_tie_breaker_min_for(kind)` (per-field dispatch) or
# `M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND` (the dict).
# The alias is the v1 default (0.85) and is NOT used in
# the engagement path; `_tie_breaker_engaged` reads
# `_tie_breaker_min_for` directly.
M20_4_TIE_BREAKER_CONFIDENCE_MIN: float = _M20_4_TIE_BREAKER_DEFAULT


def _tie_breaker_min_for(kind: str) -> float:
    """Per-field tie-breaker confidence threshold.

    Unknown kinds (e.g. a future M20.4.x field) fall back to
    the v1 0.85 default. The fallback is intentional: a new
    field without an explicit per-field threshold is
    conservative (no regression from v1).
    """
    if not isinstance(kind, str):
        return _M20_4_TIE_BREAKER_DEFAULT
    return M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND.get(
        kind.strip().lower(),
        _M20_4_TIE_BREAKER_DEFAULT,
    )


# Map M20.4 commitment observables to per-field `kind` keys.
# Both names are part of the v1 surface; a new observable
# would have to be added here AND in
# `M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND`.
_M20_4_OBSERVABLE_TO_KIND: dict[str, str] = {
    "addressee_target_match": "addressee",
    "reaction_attribution_match": "reaction",
}


def _kind_from_observable(observable: str) -> str:
    """Resolve a per-field `kind` from a commitment's observable.

    Returns "" for unknown observables, which causes
    `_tie_breaker_min_for` to fall back to the v1 0.85
    default. This is intentionally conservative: a new
    observable without a per-field threshold is a no-regression
    change from v1.
    """
    if not isinstance(observable, str):
        return ""
    return _M20_4_OBSERVABLE_TO_KIND.get(observable.strip().lower(), "")


def _admit_threshold_for(
    *,
    kind: str,
    addressed_to_assistant: bool | None = None,
) -> float:
    """Per-sub-class admit confidence threshold (P0-4, 2026-06-09).

    M20.4 v1 uses a single admit threshold
    (`M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN = 0.4`) for both
    `addressee` and `reaction` hypotheses. P0-4 splits the
    `addressee` admit rule by the LLM's
    `addressed_to_assistant` boolean:

    - `addressed_to_assistant == True` (LLM says the
      message IS directed at the assistant): admit at
      `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED`
      (0.7). The LLM is structurally unreliable on this
      sub-class; raising the bar filters out the
      high-band overconfidence drift that P1 surfaced.
    - `addressed_to_assistant == False` (LLM says the
      message is NOT directed at the assistant): admit at
      `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` (0.4, the
      v1 default). The LLM is 100% precise on this
      sub-class; the standard threshold is appropriate.
    - `kind == "reaction"`: admit at the v1 0.4 default.
      The reaction joint-axis asymmetry (50% no-emit
      rate) is in the LLM's emit decision, not in
      admit calibration; P0-4 does not change the
      reaction admit rule.
    - Unknown kind: admit at the v1 0.4 default
      (no-regression fallback).

    The function is a pure dispatcher; it does not
    validate the boolean or read the entry. The
    producer is responsible for passing the right
    `addressed_to_assistant` flag (defaults to None
    for non-addressee kinds).
    """
    if not isinstance(kind, str):
        return M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN
    k = kind.strip().lower()
    if k == "addressee" and addressed_to_assistant is True:
        return M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED
    return M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN


def _should_skip_addressee_directed_write(
    *,
    confidence: float,
) -> bool:
    """P0-5 (2026-06-09): write-path filter for the
    `addressee_target_match` observable with
    `addressed_to_assistant == True`.

    Returns True when the write path should skip the
    persistent `state["addressee_graph"]` write. The
    filter applies ONLY to the "addressed" sub-class
    (`addressed_to_assistant == True`); the caller is
    responsible for dispatching this check to the
    "addressed" sub-class only.

    The 0.9 threshold is strict `>` (consistent with
    the M20.4 v1 tie-breaker style). The function is
    a pure dispatcher; it does not read state or
    validate shapes.
    """
    return (
        not isinstance(confidence, (int, float))
        or bool(confidence != confidence)  # NaN
        or float(confidence)
        <= M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE
    )


# Bounded excerpt cap. Aligned with M19.x surface-consistency
# excerpt and M18.7 rationale cap. Frozen at 200 in v1.
M20_4_BOUNDED_EXCERPT_CHARS: int = 200

# Allowed ambiguity_band values. The frozen M18.2 v1 string.
M20_4_AMBIGUITY_BANDS: frozenset[str] = frozenset({"low", "medium", "high"})

# Allowed outcomes from the M20.1 outcome_v1 set.
_M20_1_OUTCOMES: frozenset[str] = frozenset({"confirmed", "violated", "ambiguous"})

# Engineering proxy label shared with M18.7 (additive in V2).
M20_4_ENGINEERING_PROXY_LABEL: str = "mvp_local_group_attribution"

# Reason codes (M20.4 v1).
REASON_ADMISSION: str = "m20_4_attribution"
REASON_TIE_BREAKER_ENGAGED: str = "m20_4_attribution_tie_breaker_engaged"
REASON_TIE_BREAKER_REJECTED: str = "m20_4_attribution_tie_breaker_rejected"
REASON_GRAPH_MICROADJUST: str = "m20_4_addressee_graph_microadjust"
REASON_GRAPH_REVOKE: str = "m20_4_addressee_graph_revoke"
REASON_GRAPH_NOOP: str = "m20_4_addressee_graph_noop"
# P0-5 (2026-06-09): the write-path filter's skip reason
# code. Emitted as part of the diagnostic counter, NOT as
# a bus event (the write path returns None on skip; no
# `GroupAddresseeGraphUpdated` event is emitted).
REASON_GRAPH_SKIP_ADDRESSEE_DIRECTED_LOW_CONFIDENCE: str = (
    "m20_4_addressee_graph_skip_addressee_directed_low_confidence"
)

# Tie-breaker rejection reasons (for the
# `tie_breaker_rejected_by_reason` diagnostic counter).
TIEREJECT_STRUCTURAL_ADDRESSED: str = "structural_explicit_addressee"
TIEREJECT_EXPLICIT_MENTION_OTHER: str = "explicit_mention_of_other"
TIEREJECT_EXPLICIT_REPLY_TO: str = "explicit_reply_to_set"
TIEREJECT_CONFIDENCE_LOW: str = "confidence_below_threshold"
TIEREJECT_AMBIGUITY_NOT_HIGH: str = "ambiguity_band_not_high"
TIEREJECT_DECISION_NOT_FLIPPABLE: str = "m18_5_decision_not_flippable"
TIEREJECT_OUTCOME_NOT_CONFIRMED: str = "outcome_not_confirmed"
TIEREJECT_LEVEL_NOT_MICROADJUST: str = "level_not_microadjust"


# === Bounded helpers ======================================================


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _bounded_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return default
    v = float(value)
    if v != v:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _bounded_excerpt(value: Any, *, limit: int = M20_4_BOUNDED_EXCERPT_CHARS) -> str:
    """Take the first `limit` chars of the value. Non-strings → ''.

    Admitted at admission time so the settler does NOT
    re-read the source at settle time (avoid persistence-
    driven discrepancy).
    """
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _bounded_evidence_refs(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:120])
        if len(out) >= limit:
            break
    return out


def _is_turn_id_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or len(s) > 120:
        return False
    if not s.startswith("turn_"):
        return False
    rest = s[len("turn_"):]
    parts = rest.split("_", 1)
    if not parts or not parts[0].isdigit():
        return False
    return True


# === Frozen hypothesis subset (per M20.4 §2) ==============================


def _frozen_hypothesis_subset(
    *,
    kind: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen M18.7→M20.4 hypothesis subset.

    The subset strips rationale, participant_id, and
    evidence_refs to a bounded shape that the settler can
    use. Engineering records only the frozen subset, not
    the source field.
    """
    if not isinstance(entry, Mapping):
        return {}
    if kind == "addressee":
        return {
            "addressed_to_assistant": _bounded_bool(
                entry.get("addressed_to_assistant", False)
            ),
            "confidence": _bounded_float(entry.get("confidence", 0.0)),
            "alternative_hypothesis_count": int(
                entry.get("alternative_hypothesis_count", 0) or 0
            ),
        }
    if kind == "reaction":
        return {
            "reaction_to_turn_id": _bounded_string(
                entry.get("reaction_to_turn_id", ""), default=""
            ),
            "reaction_to_participant_id": _bounded_string(
                entry.get("reaction_to_participant_id", ""), default=""
            ),
            "is_about_assistant_claim": _bounded_bool(
                entry.get("is_about_assistant_claim", False)
            ),
            "confidence": _bounded_float(entry.get("confidence", 0.0)),
            "alternative_attribution_count": int(
                entry.get("alternative_attribution_count", 0) or 0
            ),
        }
    return {}


def _frozen_binding_snapshot(
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Snapshot the relevant subset of group_turn_binding.

    The dispatcher v2 exception table and the M18.5
    tie-breaker need the ambiguity_band + structural
    answer (addressed / mentioned / reply_to). The
    snapshot is bounded; engineering does NOT include
    the full binding.
    """
    if not isinstance(binding, Mapping) or not binding:
        return {}
    return {
        "ambiguity_band": _bounded_string(
            binding.get("ambiguity_band", ""), default=""
        ),
        "addressed_participant_ids": list(
            binding.get("addressed_participant_ids", []) or []
        )[:8],
        "mentioned_participant_ids": list(
            binding.get("mentioned_participant_ids", []) or []
        )[:8],
        "reply_to_turn_id": _bounded_string(
            binding.get("reply_to_turn_id", ""), default=""
        ),
    }


# === State surface helpers (M20.4 v1) =====================================


def _state_diag(state: dict) -> dict[str, Any]:
    """Read the M20.4 diagnostic surface, or return a default."""
    if not isinstance(state, dict):
        return {}
    diag = state.get("m20_4_attribution_diagnostics")
    if isinstance(diag, dict):
        return diag
    return {}


def _bump_diag(state: dict, **delta: int) -> None:
    """Increment diagnostic counters on the M20.4 surface."""
    if not isinstance(state, dict):
        return
    diag = state.get("m20_4_attribution_diagnostics")
    if not isinstance(diag, dict):
        diag = {}
    for key, value in delta.items():
        diag[key] = int(diag.get(key, 0)) + int(value)
    state["m20_4_attribution_diagnostics"] = diag


def _bump_tie_breaker_rejected(
    state: dict, reason_code: str
) -> None:
    if not isinstance(state, dict):
        return
    diag = state.get("m20_4_attribution_diagnostics")
    if not isinstance(diag, dict):
        diag = {}
    bucket = diag.get("tie_breaker_rejected_by_reason")
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[reason_code] = int(bucket.get(reason_code, 0)) + 1
    diag["tie_breaker_rejected_by_reason"] = bucket
    state["m20_4_attribution_diagnostics"] = diag


# === Bus event builders =================================================


def build_addressee_target_match_admitted_event(
    *,
    turn_index: int,
    commitment: ActiveCommitment,
    at: str,
) -> dict[str, Any]:
    if not isinstance(commitment, ActiveCommitment):
        return {}
    payload = dict(commitment.observable_payload or {})
    return {
        "type": "AddresseeTargetMatchAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(commitment.commit_id or ""),
        "m18_7_commit_id": str(payload.get("hypothesis_commit_id", "") or ""),
        "owner_id": "group_addressee_graph",
        "hypothesis": payload.get("hypothesis", {}),
        "current_turn_id": payload.get("current_turn_id", ""),
        "inbound_bounded_excerpt": payload.get("inbound_bounded_excerpt", ""),
        "ambiguity_band": payload.get("ambiguity_band", ""),
        "reason_codes": list(commitment.reason_codes or []),
        "engineering_proxy_label": M20_4_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


def build_reaction_attribution_match_admitted_event(
    *,
    turn_index: int,
    commitment: ActiveCommitment,
    at: str,
) -> dict[str, Any]:
    if not isinstance(commitment, ActiveCommitment):
        return {}
    payload = dict(commitment.observable_payload or {})
    return {
        "type": "ReactionAttributionMatchAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(commitment.commit_id or ""),
        "m18_7_commit_id": str(payload.get("hypothesis_commit_id", "") or ""),
        "owner_id": "group_addressee_graph",
        "hypothesis": payload.get("hypothesis", {}),
        "current_turn_id": payload.get("current_turn_id", ""),
        "attributed_turn_id": payload.get("attributed_turn_id", ""),
        "attributed_bounded_excerpt": payload.get(
            "attributed_bounded_excerpt", ""
        ),
        "ambiguity_band": payload.get("ambiguity_band", ""),
        "reason_codes": list(commitment.reason_codes or []),
        "engineering_proxy_label": M20_4_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


# === Producer (M18.7 → M20) ============================================


def _attributed_excerpt_from_bus(
    bus: list,
    *,
    turn_id: str,
    limit: int = M20_4_BOUNDED_EXCERPT_CHARS,
) -> str:
    """Search the bus for a `UserUtteranceEvent` matching `turn_id`.

    The attributed turn's text is bounded to `limit` chars.
    Returns "" when the bus does not have the turn (older
    turns may have been evicted from the bus). The settler
    works with whatever is available.
    """
    if not turn_id:
        return ""
    for event in bus:
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "UserUtteranceEvent":
            continue
        if str(event.get("turn_id", "") or "") == turn_id:
            text = str(event.get("text", "") or "")
            return text.strip()[:limit]
    return ""


def _admit_one(
    *,
    kind: str,
    entry: Mapping[str, Any],
    current_turn_id: int,
    inbound_excerpt: str,
    binding: Mapping[str, Any] | None,
    bus: list,
    at: str,
) -> ActiveCommitment | None:
    """Admit one ActiveCommitment from one M18.7 entry.

    Returns None when the entry fails the admit rule
    (confidence < threshold OR participant_id empty).
    """
    if not isinstance(entry, Mapping):
        return None
    confidence = _bounded_float(entry.get("confidence", 0.0))
    if confidence < M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN:
        return None
    participant_id = str(entry.get("participant_id", "") or "")
    if not participant_id:
        return None  # M18.4 disclosure or LLM declined

    m18_7_commit_id = str(entry.get("commit_id", "") or "")
    if not m18_7_commit_id:
        return None  # M18.7 contract violation; engineering rejects
    if not _is_turn_id_valid(m18_7_commit_id) and m18_7_commit_id:
        # commit_id is not a turn_id, but the M18.7 contract
        # says it's a sha1 of (turn_index, kind, source_ref).
        # The shape validation is loose; we accept any non-
        # empty commit_id. M20.4 only checks non-empty.
        pass

    source_ref = f"m18_7_{kind}_{m18_7_commit_id[:8]}"
    hypothesis = _frozen_hypothesis_subset(kind=kind, entry=entry)
    binding_snapshot = _frozen_binding_snapshot(binding)
    ambiguity_band = (
        binding_snapshot.get("ambiguity_band", "")
        if isinstance(binding_snapshot, Mapping)
        else ""
    )
    evidence_refs = _bounded_evidence_refs(entry.get("evidence_refs"))

    if kind == "addressee":
        observable = "addressee_target_match"
        observable_payload = {
            "hypothesis": hypothesis,
            "hypothesis_commit_id": m18_7_commit_id,
            "current_turn_id": str(current_turn_id),
            "inbound_bounded_excerpt": _bounded_excerpt(inbound_excerpt),
            "group_turn_binding_snapshot": binding_snapshot,
            "ambiguity_band": ambiguity_band,
        }
        layer = "B_per_turn_commitment"
        horizon = "next_turn"
    elif kind == "reaction":
        observable = "reaction_attribution_match"
        attributed_turn_id = str(
            entry.get("reaction_to_turn_id", "") or ""
        )
        observable_payload = {
            "hypothesis": hypothesis,
            "hypothesis_commit_id": m18_7_commit_id,
            "current_turn_id": str(current_turn_id),
            "attributed_turn_id": attributed_turn_id,
            "attributed_bounded_excerpt": _bounded_excerpt(
                _attributed_excerpt_from_bus(
                    bus, turn_id=attributed_turn_id
                )
            ),
            "group_turn_binding_snapshot": binding_snapshot,
            "ambiguity_band": ambiguity_band,
        }
        layer = "B_per_turn_commitment"
        horizon = "next_turn"
    else:
        return None

    commit_id = hashlib.sha1(
        f"{observable}|{source_ref}|{current_turn_id}".encode("utf-8")
    ).hexdigest()
    return ActiveCommitment(
        commit_id=commit_id,
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref=source_ref,
        layer=layer,
        observable=observable,
        observable_payload=observable_payload,
        target={"m18_7_commit_id": m18_7_commit_id},
        due_at={"kind": "next_turn"},
        priority=confidence,
        confidence=confidence,
        evidence_refs=tuple(evidence_refs),
        created_turn=int(current_turn_id),
        created_at=str(at),
        reason_codes=(REASON_ADMISSION,),
        engineering_proxy_label=M20_4_ENGINEERING_PROXY_LABEL,
        horizon=horizon,
    )


def produce_m20_4_attribution_commitments(
    *,
    state: dict,
    bus: list,
    current_turn_id: int,
    inbound_excerpt: str = "",
    group_turn_binding: Mapping[str, Any] | None = None,
    at: str = "",
) -> list[ActiveCommitment]:
    """M20.4 producer: read M18.7 surface, admit per entry.

    Per M20.4 §2: filter on `confidence >= 0.4` AND
    `participant_id != ""`. The producer is a no-op when the
    M18.7 surface is empty. Empty surface is silent; the
    fast_chat skip path is handled in M18.7
    (AttributionHypothesisSkipped event).

    The admitted commitments are returned to the caller.
    The caller is responsible for running them through
    the M20.0 ActiveCommitmentAdapter (admit), the M20.1
    scheduler (settle), and the M20.2 dispatcher
    (dispatch). This helper only builds the proposals.

    Diagnostic counters are bumped on the
    `m20_4_attribution_diagnostics` surface.
    """
    if not isinstance(state, dict):
        return []
    surface = state.get("m18_7_attribution_hypotheses")
    if not isinstance(surface, list) or not surface:
        return []
    admitted: list[ActiveCommitment] = []
    for entry in surface:
        if not isinstance(entry, Mapping):
            continue
        kind = str(entry.get("kind", "") or "")
        if kind not in ("addressee", "reaction"):
            continue
        # Pre-admit filter (mirrored in _admit_one for safety).
        confidence = _bounded_float(entry.get("confidence", 0.0))
        participant_id = str(entry.get("participant_id", "") or "")
        # P0-4 (2026-06-09): per-sub-class admit threshold.
        # For `addressee` with `addressed_to_assistant == True`,
        # the LLM is structurally unreliable (P1:
        # recall_on_addressed = 0.0); raise the bar from
        # 0.4 to 0.7. Other sub-classes / kinds keep the
        # v1 0.4 default.
        addressed_flag: bool | None = None
        if kind == "addressee":
            addressed_flag = bool(
                entry.get("addressed_to_assistant", False)
            )
        threshold = _admit_threshold_for(
            kind=kind, addressed_to_assistant=addressed_flag
        )
        if (
            confidence < threshold
            or not participant_id
        ):
            if confidence < threshold:
                _bump_diag(
                    state, producer_reject_low_confidence_total=1
                )
                # P0-4: per-sub-class reject histogram. The
                # aggregate `producer_reject_low_confidence_total`
                # is preserved for back-compat with the v1
                # diagnostic surface; the per-sub-class
                # buckets are additive and let M20.4
                # diagnose distinguish "addressed" vs
                # "not addressed" reject rates.
                bucket_key = (
                    "producer_reject_low_confidence_"
                    f"{kind}_directed_total"
                    if kind == "addressee" and addressed_flag
                    else (
                        "producer_reject_low_confidence_"
                        f"{kind}_not_directed_total"
                        if kind == "addressee" and not addressed_flag
                        else f"producer_reject_low_confidence_{kind}_total"
                    )
                )
                diag = state.get("m20_4_attribution_diagnostics")
                if not isinstance(diag, dict):
                    diag = {}
                diag[bucket_key] = int(diag.get(bucket_key, 0)) + 1
                state["m20_4_attribution_diagnostics"] = diag
            if not participant_id:
                _bump_diag(
                    state, producer_reject_disclosure_total=1
                )
            continue
        commitment = _admit_one(
            kind=kind,
            entry=entry,
            current_turn_id=current_turn_id,
            inbound_excerpt=inbound_excerpt,
            binding=group_turn_binding,
            bus=bus,
            at=at,
        )
        if commitment is not None:
            admitted.append(commitment)
            _bump_diag(state, producer_admit_total=1)
            # P0-4: per-sub-class admit histogram. Additive
            # over the v1 `producer_admit_total` aggregate.
            admit_bucket_key = (
                "producer_admit_addressee_directed_total"
                if kind == "addressee" and addressed_flag
                else (
                    "producer_admit_addressee_not_directed_total"
                    if kind == "addressee" and not addressed_flag
                    else f"producer_admit_{kind}_total"
                )
            )
            diag = state.get("m20_4_attribution_diagnostics")
            if not isinstance(diag, dict):
                diag = {}
            diag[admit_bucket_key] = int(
                diag.get(admit_bucket_key, 0)
            ) + 1
            state["m20_4_attribution_diagnostics"] = diag
    return admitted


# === Settlers: LLM-judge (mirror M20.1 BoundaryHandledLLMJudgeSettler) ====

LLMCallFn = Callable[[str, str], dict[str, Any]]


_ADDRESSEE_JUDGE_SYSTEM_PROMPT: str = (
    "You are the addressee_target_match judge for the "
    "assistant's reply. Decide whether the M18.7 LLM's "
    "addressed_to_assistant boolean is consistent with the "
    "INBOUND turn excerpt and the group_turn_binding "
    "snapshot. The bounded excerpt is the user's CURRENT "
    "turn (first 200 chars), NOT the assistant's reply. "
    "The structural signals (addressed_participant_ids, "
    "mentioned_participant_ids, reply_to_turn_id) are "
    "informational; the inbound turn text is the primary "
    "evidence. Output JSON only. Do not include any "
    "commentary, debug fields, or markdown.\n"
    "\n"
    "Return a JSON object with these bounded fields:\n"
    '- "outcome": one of "confirmed" | "violated" | "ambiguous"\n'
    '- "rationale_span": a short label (max 80 chars, no quoted text)\n'
    '- "reason": a one-sentence justification (max 200 chars, no quoted text)\n'
    '- "evidence_refs": a list of bounded turn-local ref ids '
    '(e.g., "turn_<n>_user_utterance")\n'
    "\n"
    "Do not interpret the user's text with regex or "
    "keyword cues. Use your own semantic judgment of "
    "the conversation context."
)


_REACTION_JUDGE_SYSTEM_PROMPT: str = (
    "You are the reaction_attribution_match judge for the "
    "assistant's prior claim. Decide whether the M18.7 "
    "reaction_attribution_hypothesis is consistent with "
    "the attributed turn excerpt (first 200 chars) and "
    "the group_turn_binding snapshot. Output JSON only. "
    "Do not include any commentary, debug fields, or "
    "markdown.\n"
    "\n"
    "Return a JSON object with these bounded fields:\n"
    '- "outcome": one of "confirmed" | "violated" | "ambiguous"\n'
    '- "rationale_span": a short label (max 80 chars, no quoted text)\n'
    '- "reason": a one-sentence justification (max 200 chars, no quoted text)\n'
    '- "evidence_refs": a list of bounded turn-local ref ids\n'
    "\n"
    "Do not interpret the user's text with regex or "
    "keyword cues. Use your own semantic judgment."
)


def _build_user_prompt_addressee(
    payload: Mapping[str, Any],
) -> str:
    hypothesis = payload.get("hypothesis", {})
    inbound = payload.get("inbound_bounded_excerpt", "")
    binding = payload.get("group_turn_binding_snapshot", {})
    ambiguity_band = payload.get("ambiguity_band", "")
    return (
        f"hypothesis: {_bounded_string(str(hypothesis), limit=600)}\n"
        f"current_turn_id: {_bounded_string(str(payload.get('current_turn_id', '')), limit=20)}\n"
        f"inbound_bounded_excerpt: <{_bounded_excerpt(inbound)}>\n"
        f"group_turn_binding_snapshot: {_bounded_string(str(binding), limit=600)}\n"
        f"ambiguity_band: {_bounded_string(ambiguity_band, limit=8)}\n"
    )


def _build_user_prompt_reaction(
    payload: Mapping[str, Any],
) -> str:
    hypothesis = payload.get("hypothesis", {})
    attributed = payload.get("attributed_bounded_excerpt", "")
    attributed_turn_id = payload.get("attributed_turn_id", "")
    binding = payload.get("group_turn_binding_snapshot", {})
    ambiguity_band = payload.get("ambiguity_band", "")
    return (
        f"hypothesis: {_bounded_string(str(hypothesis), limit=600)}\n"
        f"current_turn_id: {_bounded_string(str(payload.get('current_turn_id', '')), limit=20)}\n"
        f"attributed_turn_id: {_bounded_string(attributed_turn_id, limit=120)}\n"
        f"attributed_bounded_excerpt: <{_bounded_excerpt(attributed)}>\n"
        f"group_turn_binding_snapshot: {_bounded_string(str(binding), limit=600)}\n"
        f"ambiguity_band: {_bounded_string(ambiguity_band, limit=8)}\n"
    )


def _settle_via_llm(
    *,
    commitment: ActiveCommitment,
    observation_context: Mapping[str, Any],
    llm_call: LLMCallFn | None,
    system_prompt: str,
    user_prompt: str,
    at: str,
) -> SettledValue | NoSettlement:
    """Shared LLM-judge settle helper.

    - Fail closed (`SettlerUnavailable` raised) when `llm_call`
      is not injected; the M20.1 scheduler converts this to a
      `NoSettlement` with `settler_unavailable`.
    - Fail closed (`NoSettlement` with
      `settler_llm_invalid_response`) when the response fails
      schema or the outcome is not in the frozen v1 set.
    """
    if llm_call is None:
        raise SettlerUnavailable(
            f"{commitment.observable} requires an LLM call injection"
        )
    try:
        response = llm_call(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001
        raise SettlerUnavailable(
            f"{commitment.observable} LLM call failed: "
            f"{type(exc).__name__}"
        ) from exc
    if not isinstance(response, Mapping):
        return NoSettlement(
            commit_id=commitment.commit_id,
            reason_code="settler_llm_invalid_response",
            settler_type="llm_judge",
            engineering_proxy_label=commitment.engineering_proxy_label,
            at=at,
            turn_index=int(
                observation_context.get("turn_index", commitment.created_turn)
                or commitment.created_turn
            ),
        )
    outcome = str(response.get("outcome", "") or "").strip().lower()
    if outcome not in _M20_1_OUTCOMES:
        return NoSettlement(
            commit_id=commitment.commit_id,
            reason_code="settler_llm_invalid_response",
            settler_type="llm_judge",
            engineering_proxy_label=commitment.engineering_proxy_label,
            at=at,
            turn_index=int(
                observation_context.get("turn_index", commitment.created_turn)
                or commitment.created_turn
            ),
        )
    # M20.1 magnitude mapping (binary → 1.0; ambiguous → 0.5).
    magnitude = 0.5 if outcome == "ambiguous" else 1.0
    evidence_refs = _bounded_evidence_refs(
        response.get("evidence_refs", []), limit=16
    )
    if not evidence_refs:
        return NoSettlement(
            commit_id=commitment.commit_id,
            reason_code="no_eligible_observation",
            settler_type="llm_judge",
            engineering_proxy_label=commitment.engineering_proxy_label,
            at=at,
            turn_index=int(
                observation_context.get("turn_index", commitment.created_turn)
                or commitment.created_turn
            ),
        )
    return SettledValue(
        commit_id=commitment.commit_id,
        outcome=outcome,
        magnitude=magnitude,
        evidence_refs=tuple(evidence_refs),
        reason_codes=("settler_llm_judge",),
        at=at,
        turn_index=int(
            observation_context.get("turn_index", commitment.created_turn)
            or commitment.created_turn
        ),
        settler_type="llm_judge",
        engineering_proxy_label=M20_4_ENGINEERING_PROXY_LABEL,
    )


class AddresseeTargetMatchLLMJudgeSettler:
    """M20.4 §3 — LLM-judge settler for `addressee_target_match`.

    Compares the M18.7 `addressee_hypothesis` against the
    INBOUND turn excerpt + structural snapshot (C2 fix:
    the bounded excerpt is the INBOUND turn, NOT the
    assistant's reply).
    """

    SETTLER_TYPE: str = "llm_judge"
    ENGINEERING_PROXY_LABEL: str = M20_4_ENGINEERING_PROXY_LABEL

    def __init__(self, llm_call: LLMCallFn | None = None) -> None:
        self._llm_call = llm_call

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        at = str(observation_context.get("now", "") or "")
        user_prompt = _build_user_prompt_addressee(
            commitment.observable_payload or {}
        )
        return _settle_via_llm(
            commitment=commitment,
            observation_context=observation_context,
            llm_call=self._llm_call,
            system_prompt=_ADDRESSEE_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            at=at,
        )


class ReactionAttributionMatchLLMJudgeSettler:
    """M20.4 §3 — LLM-judge settler for `reaction_attribution_match`.

    Compares the M18.7 `reaction_attribution_hypothesis`
    against the attributed turn excerpt + structural
    snapshot.
    """

    SETTLER_TYPE: str = "llm_judge"
    ENGINEERING_PROXY_LABEL: str = M20_4_ENGINEERING_PROXY_LABEL

    def __init__(self, llm_call: LLMCallFn | None = None) -> None:
        self._llm_call = llm_call

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        at = str(observation_context.get("now", "") or "")
        user_prompt = _build_user_prompt_reaction(
            commitment.observable_payload or {}
        )
        return _settle_via_llm(
            commitment=commitment,
            observation_context=observation_context,
            llm_call=self._llm_call,
            system_prompt=_REACTION_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            at=at,
        )


# === Tie-breaker feedback row ============================================


def _tie_breaker_engaged(
    *,
    decision_level: str,
    outcome: str,
    ambiguity_band: str,
    confidence: float,
    addressed_participant_ids: list[str] | tuple[str, ...] | None,
    mentioned_participant_ids: list[str] | tuple[str, ...] | None,
    reply_to_turn_id: str,
    m18_5_structural_decision: str,
    kind: str = "",
) -> tuple[bool, str]:
    """Apply the M20.4 v1 tie-breaker engagement rule (C1 fix).

    P0-3 (2026-06-08) adds a `kind` parameter. The
    engagement threshold is per-field: 0.9 for addressee,
    0.7 for reaction, default 0.85 for unknown kinds (the
    v1 single threshold, retained as a conservative
    fallback). Strict inequality, same as v1.

    Returns (engaged, rejection_reason). When engaged,
    rejection_reason is "". When not engaged, rejection_reason
    is one of the TIEREJECT_* constants.
    """
    if decision_level != "microadjust":
        return False, TIEREJECT_LEVEL_NOT_MICROADJUST
    if outcome != "confirmed":
        return False, TIEREJECT_OUTCOME_NOT_CONFIRMED
    if ambiguity_band != "high":
        return False, TIEREJECT_AMBIGUITY_NOT_HIGH
    if not (
        confidence > _tie_breaker_min_for(kind)
    ):
        return False, TIEREJECT_CONFIDENCE_LOW
    if addressed_participant_ids:
        return False, TIEREJECT_STRUCTURAL_ADDRESSED
    if mentioned_participant_ids:
        return False, TIEREJECT_EXPLICIT_MENTION_OTHER
    if reply_to_turn_id:
        return False, TIEREJECT_EXPLICIT_REPLY_TO
    if m18_5_structural_decision not in ("clarify_addressee", "no_reply"):
        return False, TIEREJECT_DECISION_NOT_FLIPPABLE
    return True, ""


def build_m18_5_attribution_feedback_row(
    *,
    feedback_id: str,
    current_turn_id: int,
    m18_5_structural_decision: str,
    hypothesis: Mapping[str, Any],
    ambiguity_band: str,
    engaged: bool,
    patched_decision: str | None,
    patched_reason: str,
    at: str,
) -> dict[str, Any]:
    return {
        "feedback_id": str(feedback_id),
        "current_turn_id": int(current_turn_id),
        "m18_5_structural_decision": str(m18_5_structural_decision),
        "hypothesis": dict(hypothesis) if isinstance(hypothesis, Mapping) else {},
        "ambiguity_band": str(ambiguity_band),
        "tie_breaker_engaged": bool(engaged),
        "patched_decision": str(patched_decision) if patched_decision else None,
        "patched_reason": _bounded_string(patched_reason, default="", limit=120),
        "at": str(at),
        "engineering_proxy_label": M20_4_ENGINEERING_PROXY_LABEL,
    }


def record_m18_5_attribution_feedback(
    state: dict,
    *,
    feedback_id: str,
    current_turn_id: int,
    m18_5_structural_decision: str,
    hypothesis: Mapping[str, Any],
    ambiguity_band: str,
    engaged: bool,
    patched_decision: str | None,
    patched_reason: str,
    at: str,
) -> dict[str, Any]:
    """Write a feedback row to `state["m18_5_attribution_feedback"]`.

    M18.5 reads this row on subsequent turns as a
    *supplementary* input, strictly after its structural
    decision. M18.5's decision tree is unchanged.
    """
    if not isinstance(state, dict):
        return {}
    surface = state.get("m18_5_attribution_feedback")
    if not isinstance(surface, dict):
        surface = {}
    row = build_m18_5_attribution_feedback_row(
        feedback_id=feedback_id,
        current_turn_id=current_turn_id,
        m18_5_structural_decision=m18_5_structural_decision,
        hypothesis=hypothesis,
        ambiguity_band=ambiguity_band,
        engaged=engaged,
        patched_decision=patched_decision,
        patched_reason=patched_reason,
        at=at,
    )
    surface[feedback_id] = row
    state["m18_5_attribution_feedback"] = surface
    if engaged:
        _bump_diag(state, tie_breaker_engaged_total=1)
    else:
        _bump_diag(state, tie_breaker_rejected_total=1)
        _bump_tie_breaker_rejected(state, patched_reason or "unknown")
    return row


# === State surface writers (M20.4 → addressee_graph write path) =========


def _ensure_addressee_graph(state: dict) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    graph = state.get("addressee_graph")
    if not isinstance(graph, dict):
        graph = {}
    state["addressee_graph"] = graph
    return graph


def write_addressee_graph_microadjust(
    *,
    state: dict,
    decision: Any,
    commitment: ActiveCommitment,
    at: str,
) -> dict[str, Any] | None:
    """`microadjust` for owner `group_addressee_graph`.

    Appends an attribution row to `state["addressee_graph"]`
    keyed by `commitment.target["m18_7_commit_id"]` (or the
    commitment's own `commit_id` as a fallback). The row
    carries:
      - m18_7_commit_id
      - speaker (M18.7 hypothesis `participant_id`)
      - addressed_to_assistant (the M18.7 boolean)
      - confidence (M18.7 clamped)
      - settled_outcome (confirmed / violated / ambiguous)
      - attributed_turn_id (for reaction_attribution_match)
      - evidence_refs (bounded handles)
    Emits a `GroupAddresseeGraphUpdated` audit event.

    P0-5 (2026-06-09) — write-path filter for the
    `addressee_target_match` observable with
    `addressed_to_assistant == True`. When the M18.7
    hypothesis confidence is `<= 0.9` (the bqxsmofri
    high-band overconfidence drift zone), the function
    returns None (no graph write, no audit event) and
    bumps the
    `write_path_skip_addressee_directed_low_confidence_total`
    diagnostic counter. The "not addressed" sub-class
    (precision 1.0) and the `reaction_attribution_match`
    observable are unchanged. The producer admit rule
    (P0-4) and the settler (M20.1) still run, so the
    tie-breaker feedback path is unaffected.

    Returns the audit event dict (the caller is
    responsible for appending it to the per-turn bus),
    or None when the write is skipped.
    """
    if not isinstance(commitment, ActiveCommitment):
        return None
    payload = dict(commitment.observable_payload or {})
    hypothesis = payload.get("hypothesis", {})
    m18_7_commit_id = str(
        payload.get("hypothesis_commit_id", "") or ""
    )
    if not m18_7_commit_id:
        m18_7_commit_id = str(
            commitment.target.get("m18_7_commit_id", "")
            if isinstance(commitment.target, Mapping)
            else ""
        )
    if not m18_7_commit_id:
        m18_7_commit_id = commitment.commit_id
    speaker = ""
    addressed_to_assistant: bool = False
    confidence = 0.0
    attributed_turn_id = ""
    if commitment.observable == "addressee_target_match":
        speaker = str(
            hypothesis.get("addressed_to_assistant", "") or ""
        )  # placeholder; M18.7 entry is on observable_payload elsewhere
        # The frozen hypothesis subset does NOT carry the
        # participant_id; engineering reads it from the
        # M18.7 entry that the producer used. For the write
        # path, we use the source_ref-derived id.
        addressed_to_assistant = bool(
            hypothesis.get("addressed_to_assistant", False)
        )
        confidence = float(hypothesis.get("confidence", 0.0) or 0.0)
        # P0-5: write-path filter for the "addressed"
        # sub-class ONLY. Skips the persistent graph
        # write when the LLM's "addressed" claim is at
        # high confidence but the bqxsmofri drift zone
        # (P1: recall_on_addressed = 0.0). The "not
        # addressed" sub-class (P1 precision 1.0) is
        # NOT touched — its admit threshold is 0.4 (P0-4)
        # and the v1 write path is preserved. The filter
        # is at the write boundary, not the producer
        # admit boundary; the producer admit (P0-4) and
        # the settler still run, so the tie-breaker
        # feedback path is unaffected.
        if (
            addressed_to_assistant
            and _should_skip_addressee_directed_write(
                confidence=confidence
            )
        ):
            _bump_diag(
                state,
                write_path_skip_addressee_directed_low_confidence_total=1,
            )
            return None
    elif commitment.observable == "reaction_attribution_match":
        attributed_turn_id = str(hypothesis.get("reaction_to_turn_id", "") or "")
        addressed_to_assistant = bool(
            hypothesis.get("is_about_assistant_claim", False)
        )
        confidence = float(hypothesis.get("confidence", 0.0) or 0.0)

    settled_outcome = "confirmed"  # microadjust only fires on confirmed
    graph = _ensure_addressee_graph(state)
    row = {
        "m18_7_commit_id": m18_7_commit_id,
        "speaker": speaker,
        "addressed_to_assistant": addressed_to_assistant,
        "confidence": confidence,
        "settled_outcome": settled_outcome,
        "attributed_turn_id": attributed_turn_id,
        "evidence_refs": list(commitment.evidence_refs or []),
    }
    graph[m18_7_commit_id] = row
    # Cap the graph at 256 entries (M20.3-style).
    if len(graph) > 256:
        keys = list(graph.keys())
        for k in keys[: len(keys) - 256]:
            del graph[k]
    return {
        "type": "GroupAddresseeGraphUpdated",
        "turn_index": int(commitment.created_turn or 0),
        "commit_id": str(commitment.commit_id or ""),
        "m18_7_commit_id": m18_7_commit_id,
        "owner_id": "group_addressee_graph",
        "settled_outcome": settled_outcome,
        "reason_codes": [REASON_GRAPH_MICROADJUST],
        "engineering_proxy_label": M20_4_ENGINEERING_PROXY_LABEL,
        "at": str(at),
    }


def clear_addressee_graph_row(
    *,
    state: dict,
    commitment: ActiveCommitment,
) -> dict[str, Any] | None:
    """`revoke` for owner `group_addressee_graph` — clear the row."""
    if not isinstance(commitment, ActiveCommitment):
        return None
    payload = dict(commitment.observable_payload or {})
    m18_7_commit_id = str(
        payload.get("hypothesis_commit_id", "") or ""
    )
    if not m18_7_commit_id:
        m18_7_commit_id = commitment.commit_id
    graph = state.get("addressee_graph") if isinstance(state, dict) else None
    if isinstance(graph, dict) and m18_7_commit_id in graph:
        del graph[m18_7_commit_id]
    return {
        "type": "GroupAddresseeGraphUpdated",
        "turn_index": int(commitment.created_turn or 0),
        "commit_id": str(commitment.commit_id or ""),
        "m18_7_commit_id": m18_7_commit_id,
        "owner_id": "group_addressee_graph",
        "settled_outcome": "revoked",
        "reason_codes": [REASON_GRAPH_REVOKE],
        "engineering_proxy_label": M20_4_ENGINEERING_PROXY_LABEL,
    }


# === Top-level dispatcher helper =========================================


def emit_m20_4_tie_breaker_feedback(
    state: dict,
    *,
    decision: Any,
    commitment: ActiveCommitment,
    settled_value: Any,
    m18_5_structural_decision: str,
    at: str,
) -> dict[str, Any] | None:
    """Write a feedback row to `state["m18_5_attribution_feedback"]`.

    Called by mvp_loop.run_turn after the M20.2 dispatch
    returns a `microadjust` decision for an M20.4-admitted
    commitment. The function checks the engagement rule
    (C1 fix) and writes the row.

    Returns the row dict (caller may also emit a bus event).
    """
    if not isinstance(commitment, ActiveCommitment):
        return None
    if not isinstance(decision, object):
        return None
    decision_level = str(getattr(decision, "correction_level", "") or "")
    outcome = str(getattr(settled_value, "outcome", "") or "")
    payload = dict(commitment.observable_payload or {})
    binding = payload.get("group_turn_binding_snapshot", {})
    if not isinstance(binding, Mapping):
        binding = {}
    ambiguity_band = str(binding.get("ambiguity_band", "") or "")
    addressed = list(binding.get("addressed_participant_ids", []) or [])
    mentioned = list(binding.get("mentioned_participant_ids", []) or [])
    reply_to = str(binding.get("reply_to_turn_id", "") or "")
    hypothesis = payload.get("hypothesis", {})

    engaged, rejection_reason = _tie_breaker_engaged(
        decision_level=decision_level,
        outcome=outcome,
        ambiguity_band=ambiguity_band,
        confidence=float(hypothesis.get("confidence", 0.0) or 0.0),
        addressed_participant_ids=addressed,
        mentioned_participant_ids=mentioned,
        reply_to_turn_id=reply_to,
        m18_5_structural_decision=m18_5_structural_decision,
        # P0-3: dispatch the per-field threshold by commitment
        # observable. addressee_target_match -> 0.9,
        # reaction_attribution_match -> 0.7, unknown -> 0.85
        # default. The kind encoding is fixed in v1.
        kind=_kind_from_observable(str(commitment.observable or "")),
    )

    if engaged:
        patched_decision = "reply_to_current_speaker"
        patched_reason = "tie_breaker_engaged"
    else:
        patched_decision = None
        patched_reason = rejection_reason

    feedback_id = f"fb_{commitment.commit_id[:12]}_{int(commitment.created_turn or 0)}"
    return record_m18_5_attribution_feedback(
        state,
        feedback_id=feedback_id,
        current_turn_id=int(commitment.created_turn or 0),
        m18_5_structural_decision=str(m18_5_structural_decision),
        hypothesis=hypothesis,
        ambiguity_band=ambiguity_band,
        engaged=engaged,
        patched_decision=patched_decision,
        patched_reason=patched_reason,
        at=at,
    )


# === State-surface bump helpers (settler-side) =========================


def record_settler_outcome(state: dict, outcome: str) -> None:
    """Bump the M20.4 settler outcome counter on the diagnostic surface."""
    if not isinstance(state, dict):
        return
    bucket = {
        "confirmed": "settler_confirmed_total",
        "violated": "settler_violated_total",
        "ambiguous": "settler_ambiguous_total",
    }
    key = bucket.get(str(outcome or "").lower())
    if key:
        _bump_diag(state, **{key: 1})


def record_settler_unavailable(state: dict) -> None:
    _bump_diag(state, settler_unavailable_total=1)


__all__ = [
    "AddresseeTargetMatchLLMJudgeSettler",
    "M20_4_AMBIGUITY_BANDS",
    "M20_4_BOUNDED_EXCERPT_CHARS",
    "M20_4_ENGINEERING_PROXY_LABEL",
    "M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN",
    "M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED",
    "M20_4_TIE_BREAKER_CONFIDENCE_MIN",
    "M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND",
    "M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE",
    "REASON_ADMISSION",
    "REASON_GRAPH_MICROADJUST",
    "REASON_GRAPH_NOOP",
    "REASON_GRAPH_REVOKE",
    "REASON_GRAPH_SKIP_ADDRESSEE_DIRECTED_LOW_CONFIDENCE",
    "REASON_TIE_BREAKER_ENGAGED",
    "REASON_TIE_BREAKER_REJECTED",
    "ReactionAttributionMatchLLMJudgeSettler",
    "_admit_threshold_for",
    "_should_skip_addressee_directed_write",
    "build_addressee_target_match_admitted_event",
    "build_m18_5_attribution_feedback_row",
    "build_reaction_attribution_match_admitted_event",
    "clear_addressee_graph_row",
    "emit_m20_4_tie_breaker_feedback",
    "produce_m20_4_attribution_commitments",
    "record_m18_5_attribution_feedback",
    "record_settler_outcome",
    "record_settler_unavailable",
    "write_addressee_graph_microadjust",
]
