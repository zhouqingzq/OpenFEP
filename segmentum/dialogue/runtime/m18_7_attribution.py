"""M18.7 Group Turn Semantic Attribution Contract.

M18.7 owns the conscious-loop JSON v2 attributes
`addressee_hypothesis` and `reaction_attribution_hypothesis`,
the prompt template, the normalize contract, the bus
event envelopes, and the bounded state surface. M18.7 is
a pure-data layer; the LLM is the only legitimate source
of the two fields (per CLAUDE.md "no keyword / regex"
red line). Engineering only validates shapes, clamps
bounded values, persists state, and audits.

M18.7 does NOT admit ActiveCommitment rows; M20.4 reads
the M18.7 state surface and admits commitments. M18.7
also does NOT modify the M18.5 reply policy (DECIDED 2);
the M20.4.1 same-turn gate and the M20.4 v1 cross-turn
feedback row are M20.4 territory.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


# === Frozen v1 enums / constants =========================================

# M18.7 §1 — frozen enum for the two hypothesis `kind`s.
KIND_ADDRESSEE: str = "addressee"
KIND_REACTION: str = "reaction"
ALLOWED_M18_7_KIND: frozenset[str] = frozenset({KIND_ADDRESSEE, KIND_REACTION})

# M18.7 §1 — bounded enum for `addressee_hypothesis.addressed_to_assistant`.
# True = message is directed at the assistant; False = directed at
# another participant or at nobody; missing / "" = empty mapping.
# (M18.7 normalize: the field itself is a Mapping; the boolean
# is inside the mapping. Empty mapping means "no hypothesis".)

# M18.7 §1 — bounded enum for
# `reaction_attribution_hypothesis.is_about_assistant_claim`.
# True = reaction targets the assistant's own claim in the
# attributed turn. False = otherwise. Empty mapping = "no hypothesis".

# M18.7 §3.1 — bounded enum for `user_correction_signal`
# already in M20.3 (`_normalize_correcting_assistant_identity`).
# M18.7 does NOT introduce a new bounded enum for the user
# correction signal; it reuses M20.3's.

# M18.7 DECIDED 5 — bounded rolling window size for the
# state surface. Frozen at 8.
M18_7_STATE_SURFACE_CAP: int = 8

# M18.7 §1 — frozen enums and length caps.
M18_7_RATIONALE_MAX_CHARS: int = 200
M18_7_ALTERNATIVE_CAP: int = 2
M18_7_EVIDENCE_REFS_CAP: int = 32
M18_7_PARTICIPANT_ID_MAX_CHARS: int = 120
M18_7_TURN_ID_MAX_CHARS: int = 120


# Engineering proxy label used by all M18.7 audit envelopes
# and the state surface. Shared with M20.4; declared in
# ENGINEERING_PROXY_LABELS_V2 in M20.4 (additive over v1).
M18_7_ENGINEERING_PROXY_LABEL: str = "mvp_local_group_attribution"

# Reason codes emitted on the bus / state surface.
REASON_FIELD_PRESENT: str = "m18_7_field_present"
REASON_FIELD_SKIPPED_FAST_CHAT: str = "m18_7_field_skipped_fast_chat"


# === M18.7.2 minimal-prompt call site constants ==========================
# M18.7.2 owns a dedicated minimal-prompt LLM call site for
# addressee / reaction attribution, decoupled from the conscious
# loop. The minimal prompt is ~1.5-2.0k chars (vs the 7.7-26k
# conscious-loop prompt) and the LLM fills only the M18.7 v1
# shape, with a `_m18_7_2_source` tag for traceability.

M18_7_2_SOURCE_TAG: str = "m18_7_2_minimal"
# M18.7.2 v2 (2026-06-10): bumped from 2000 to 2500 to
# accommodate the v2 system_prompt revision (strong-signal
# list + counter-example list + 3 inline examples for the
# `addressed_to_assistant` axis). v1 nominal was 1647 chars
# (within 2000); v2 nominal is 2277 chars (within 2500).
# The MAX is still well below the 7.7-26k conscious-loop
# prompt that motivated the M18.7.2 minimal-prompt design.
# Bump 2000 (v1) → 2500 (v2 strong-signal list) → 2600 (v3
# default-to-True rule + re-engaging signal + 2 more examples).
M18_7_2_MINIMAL_PROMPT_MAX_CHARS: int = 2600
M18_7_2_REASON_FIELD_PRESENT: str = "m18_7_2_field_present"
M18_7_2_REASON_MINIMAL_DEGRADED: str = "m18_7_2_minimal_llm_failure"


# === Bounded helpers ======================================================


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    """Return a stripped, length-capped string; non-strings → default."""
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _bounded_bool(value: Any, *, default: bool = False) -> bool:
    """Return a strict bool; non-bools → default (no truthy coercion)."""
    if isinstance(value, bool):
        return value
    return default


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    """Return a float clamped to [0.0, 1.0]; out-of-range or NaN → 0.0."""
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


def _is_evidence_ref_handle(value: str) -> bool:
    """Shape-validate a v1 evidence_refs handle.

    M18.7 §Hard Rules §3: engineering validates the *prefix
    pattern*, not the specific suffix value (M18.7
    DECIDED 4: shape-validated, not enumerated). Valid
    shapes:
    - `turn_<digits>_*` or `turn_<digits>` (turn-local handle)
    - `bus_event_<uuid>` (a specific bus event id)
    - `participant_<id>` (a specific participant id)

    Any non-shape string is rejected. The LLM is the
    source of the value; engineering only validates the
    shape.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or len(s) > 120:
        return False
    if s.startswith("turn_"):
        rest = s[len("turn_"):]
        parts = rest.split("_", 1)
        if not parts or not parts[0].isdigit():
            return False
        return True
    if s.startswith("bus_event_") or s.startswith("participant_"):
        return True
    return False


def _bounded_evidence_refs(value: Any, *, limit: int = M18_7_EVIDENCE_REFS_CAP) -> list[str]:
    """Return a list of bounded handles; non-shape entries dropped; cap at `limit`.

    The list is NOT reordered; the cap drops the tail.
    M18.7 DECIDED 4: shape-validated, not enumerated.
    """
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and _is_evidence_ref_handle(item):
            out.append(item.strip()[:120])
        if len(out) >= limit:
            break
    return out


def _bounded_alternatives(
    value: Any,
    *,
    limit: int = M18_7_ALTERNATIVE_CAP,
) -> list[dict[str, Any]]:
    """Return a list of bounded alternative dicts; cap at `limit`.

    Each alternative is `{"addressed_to_assistant": bool, "confidence": float,
    "rationale": str, "evidence_refs": list[str]}` (or the
    reaction_attribution equivalent). Engineering only validates
    the shape; the LLM is the source of the unclamped values.
    """
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        out.append(dict(item))
        if len(out) >= limit:
            break
    return out


def _is_participant_id_valid(value: Any) -> bool:
    """Shape-validate a v1 M18.1 participant_id.

    Per M18.7 §Hard Rules §3, `participant_id` is the v1
    frozen M18.1 form. Engineering validates the shape (a
    bounded handle), not the specific value. The LLM is
    the source of the value.

    The v1 shape: a non-empty string of bounded length
    consisting of letters, digits, underscores, hyphens,
    and dots. This is intentionally permissive; the M18.6
    acceptance harness verifies the LLM does not invent
    nonsense ids.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or len(s) > M18_7_PARTICIPANT_ID_MAX_CHARS:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.@:")
    return all(ch in allowed for ch in s)


def _is_turn_id_valid(value: Any) -> bool:
    """Shape-validate a v1 turn id (M18.3 frozen form).

    The v1 shape: a bounded handle of the form
    `turn_<digits>_<suffix>` or `turn_<digits>`. Engineering
    only validates the prefix pattern; the suffix is open
    (M18.7 §OPEN→DECIDED 4 — shape-validated, not
    enumerated).
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or len(s) > M18_7_TURN_ID_MAX_CHARS:
        return False
    # `turn_<digits>_*` or `turn_<digits>`; allow optional suffix
    # (one underscore-separated token).
    if not s.startswith("turn_"):
        return False
    rest = s[len("turn_"):]
    parts = rest.split("_", 1)
    if not parts or not parts[0].isdigit():
        return False
    return True


# === Normalize: addressee_hypothesis =====================================


def _normalize_addressee_alternative(alt: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one alternative_hypotheses entry (addressee shape)."""
    return {
        "addressed_to_assistant": _bounded_bool(alt.get("addressed_to_assistant")),
        "confidence": _bounded_float(alt.get("confidence")),
        "rationale": _bounded_string(
            alt.get("rationale"), default="", limit=M18_7_RATIONALE_MAX_CHARS
        ),
        "evidence_refs": _bounded_evidence_refs(alt.get("evidence_refs")),
    }


def normalize_addressee_hypothesis(value: Any) -> dict[str, Any]:
    """Normalize M18.7 `addressee_hypothesis` to the v1 frozen shape.

    M18.7 §1 schema:

    ```text
    {
        "participant_id": str,             # shape-validated; "" = no
        "addressed_to_assistant": bool,
        "confidence": float,                # clamped to [0, 1]
        "rationale": str,                  # ≤ 200 chars
        "evidence_refs": list[str],
        "alternative_hypotheses": list[dict],  # ≤ 2
    }
    ```

    Engineering rules:
    - Empty input (`{}` / `None` / non-mapping) → `{}` (silent
      "no hypothesis"; M18.7 DECIDED 6). The orchestrator
      checks `bool(normalized)` to decide whether to emit
      the bus event and write the state surface.
    - Missing or malformed fields → defaults; rationale
      truncated to 200 chars; evidence_refs shape-validated;
      alternative_hypotheses capped at 2.
    - `participant_id` is shape-validated (non-empty letters /
      digits / `_-.@:` of bounded length) per the M18.7
      prompt's §Hard Rules §3. A non-shape value is rejected
      and the field defaults to `""` (silent "M18.4
      disclosure forbade the identification" or "LLM
      declined to identify").
    """
    if not isinstance(value, Mapping) or not value:
        return {}
    raw_pid = value.get("participant_id")
    if raw_pid == "" or raw_pid is None:
        participant_id = ""
    elif _is_participant_id_valid(raw_pid):
        participant_id = str(raw_pid).strip()
    else:
        # Malformed / un-shape-validated participant_id:
        # default to empty (silent). The LLM is the source;
        # engineering only validates the shape.
        participant_id = ""

    return {
        "participant_id": participant_id,
        "addressed_to_assistant": _bounded_bool(value.get("addressed_to_assistant")),
        "confidence": _bounded_float(value.get("confidence")),
        "rationale": _bounded_string(
            value.get("rationale"), default="", limit=M18_7_RATIONALE_MAX_CHARS
        ),
        "evidence_refs": _bounded_evidence_refs(value.get("evidence_refs")),
        "alternative_hypotheses": [
            _normalize_addressee_alternative(alt)
            for alt in _bounded_alternatives(value.get("alternative_hypotheses"))
        ],
    }


# === Normalize: reaction_attribution_hypothesis =========================


def _normalize_reaction_alternative(alt: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one alternative_attributions entry (reaction shape)."""
    return {
        "reaction_to_turn_id": _bounded_string(alt.get("reaction_to_turn_id"), default=""),
        "reaction_to_participant_id": _bounded_string(
            alt.get("reaction_to_participant_id"), default=""
        ),
        "is_about_assistant_claim": _bounded_bool(alt.get("is_about_assistant_claim")),
        "confidence": _bounded_float(alt.get("confidence")),
        "rationale": _bounded_string(
            alt.get("rationale"), default="", limit=M18_7_RATIONALE_MAX_CHARS
        ),
        "evidence_refs": _bounded_evidence_refs(alt.get("evidence_refs")),
    }


def normalize_reaction_attribution_hypothesis(value: Any) -> dict[str, Any]:
    """Normalize M18.7 `reaction_attribution_hypothesis` to the v1 frozen shape.

    M18.7 §1 schema:

    ```text
    {
        "participant_id": str,
        "reaction_to_turn_id": str,
        "reaction_to_participant_id": str,
        "is_about_assistant_claim": bool,
        "confidence": float,
        "rationale": str,                  # ≤ 200 chars
        "evidence_refs": list[str],
        "alternative_attributions": list[dict],  # ≤ 2
    }
    ```

    Engineering rules mirror `normalize_addressee_hypothesis`.
    Empty input → `{}` (silent "no hypothesis").
    """
    if not isinstance(value, Mapping) or not value:
        return {}
    raw_pid = value.get("participant_id")
    if raw_pid == "" or raw_pid is None:
        participant_id = ""
    elif _is_participant_id_valid(raw_pid):
        participant_id = str(raw_pid).strip()
    else:
        participant_id = ""

    reaction_to_turn_id = _bounded_string(
        value.get("reaction_to_turn_id"), default="", limit=M18_7_TURN_ID_MAX_CHARS
    )
    if reaction_to_turn_id and not _is_turn_id_valid(reaction_to_turn_id):
        # M18.7 §Hard Rules §3 — engineering validates the
        # shape of `reaction_to_turn_id`. A non-shape value
        # is rejected and the field defaults to `""` (silent
        # "LLM declined to attribute" or "M18.7 hand-off
        # contract violation").
        reaction_to_turn_id = ""

    raw_rpid = value.get("reaction_to_participant_id")
    if raw_rpid == "" or raw_rpid is None:
        reaction_to_participant_id = ""
    elif _is_participant_id_valid(raw_rpid):
        reaction_to_participant_id = str(raw_rpid).strip()
    else:
        reaction_to_participant_id = ""

    return {
        "participant_id": participant_id,
        "reaction_to_turn_id": reaction_to_turn_id,
        "reaction_to_participant_id": reaction_to_participant_id,
        "is_about_assistant_claim": _bounded_bool(
            value.get("is_about_assistant_claim")
        ),
        "confidence": _bounded_float(value.get("confidence")),
        "rationale": _bounded_string(
            value.get("rationale"), default="", limit=M18_7_RATIONALE_MAX_CHARS
        ),
        "evidence_refs": _bounded_evidence_refs(value.get("evidence_refs")),
        "alternative_attributions": [
            _normalize_reaction_alternative(alt)
            for alt in _bounded_alternatives(value.get("alternative_attributions"))
        ],
    }


# === Commit id derivation ================================================


def compute_m18_7_commit_id(
    *,
    kind: str,
    turn_index: int,
    source_ref: str,
) -> str:
    """Deterministic sha1 of (kind, turn_index, source_ref).

    M18.7 §5 — the M20.4 hand-off contract. M20.4 reuses
    this commit_id as the `ActiveCommitment.source_ref` to
    keep M18.7 → M20.4 traceable in the audit tail.
    """
    if kind not in ALLOWED_M18_7_KIND:
        kind = "unknown"
    canonical = f"{kind}|{int(turn_index)}|{source_ref}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


# === State surface helpers ===============================================


def _empty_addressee_entry() -> dict[str, Any]:
    return {
        "participant_id": "",
        "addressed_to_assistant": False,
        "confidence": 0.0,
        "alternative_hypothesis_count": 0,
    }


def _empty_reaction_entry() -> dict[str, Any]:
    return {
        "participant_id": "",
        "reaction_to_turn_id": "",
        "reaction_to_participant_id": "",
        "is_about_assistant_claim": False,
        "confidence": 0.0,
        "alternative_attribution_count": 0,
    }


def build_state_entry(
    *,
    kind: str,
    turn_index: int,
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the M18.7 §5 frozen state-entry dict.

    The state surface (`state["m18_7_attribution_hypotheses"]`)
    holds a list of these dicts. M20.4 reads `kind` and
    `commit_id` to admit commitments.
    """
    if kind == KIND_ADDRESSEE:
        source_ref = f"m18_7_addressee_{turn_index}"
        sub = {
            "participant_id": str(normalized.get("participant_id", "") or ""),
            "addressed_to_assistant": bool(
                normalized.get("addressed_to_assistant", False)
            ),
            "confidence": float(normalized.get("confidence", 0.0) or 0.0),
            "alternative_hypothesis_count": len(
                normalized.get("alternative_hypotheses", []) or []
            ),
        }
    elif kind == KIND_REACTION:
        source_ref = f"m18_7_reaction_{turn_index}"
        sub = {
            "participant_id": str(normalized.get("participant_id", "") or ""),
            "reaction_to_turn_id": str(normalized.get("reaction_to_turn_id", "") or ""),
            "reaction_to_participant_id": str(
                normalized.get("reaction_to_participant_id", "") or ""
            ),
            "is_about_assistant_claim": bool(
                normalized.get("is_about_assistant_claim", False)
            ),
            "confidence": float(normalized.get("confidence", 0.0) or 0.0),
            "alternative_attribution_count": len(
                normalized.get("alternative_attributions", []) or []
            ),
        }
    else:
        # Defensive: unknown kind → empty entry, no commit_id
        # emitted on the bus, no state surface write.
        return {}

    commit_id = compute_m18_7_commit_id(
        kind=kind, turn_index=int(turn_index), source_ref=source_ref
    )
    return {
        "kind": kind,
        "turn_index": int(turn_index),
        "commit_id": commit_id,
        **sub,
        "evidence_refs": list(normalized.get("evidence_refs", []) or []),
        "at": str(normalized.get("_at", "") or ""),
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
    }


def record_m18_7_attribution_hypotheses(
    state: dict,
    entry: Mapping[str, Any],
) -> None:
    """Append a frozen state entry to the bounded rolling window.

    M18.7 §5 — the surface is `state["m18_7_attribution_hypotheses"]`,
    a list of at most `M18_7_STATE_SURFACE_CAP = 8` entries. The
    oldest entry is evicted on overflow. Engineering does NOT
    write to this surface directly; this helper is the only
    writer.

    The helper is a no-op when the entry is empty (defensive
    guard against malformed input).
    """
    if not isinstance(state, dict):
        return
    if not isinstance(entry, Mapping) or not entry:
        return
    surface = state.get("m18_7_attribution_hypotheses")
    if not isinstance(surface, list):
        surface = []
    surface.append(dict(entry))
    if len(surface) > M18_7_STATE_SURFACE_CAP:
        surface = surface[-M18_7_STATE_SURFACE_CAP:]
    state["m18_7_attribution_hypotheses"] = surface


# === Bus event builders =================================================


def build_addressee_hypothesis_admitted_event(
    *,
    turn_index: int,
    entry: Mapping[str, Any],
    at: str,
    rationale_chars: int = 0,
) -> dict[str, Any]:
    """Build the `AddresseeHypothesisAdmitted` audit envelope.

    M18.7 §3 — emitted ALONGSIDE the bus when the
    `addressee_hypothesis` is non-empty. The envelope
    shares `commit_id` with the state surface entry
    so diagnose can cross-reference.

    The envelope does NOT include the rationale text
    (M18.7 DECIDED 11). The LLM is the source of the
    rationale; engineering audits only the shape
    (length is recorded as `rationale_chars`).
    """
    if not isinstance(entry, Mapping):
        return {}
    return {
        "type": "AddresseeHypothesisAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(entry.get("commit_id", "") or ""),
        "participant_id": str(entry.get("participant_id", "") or ""),
        "addressed_to_assistant": bool(
            entry.get("addressed_to_assistant", False)
        ),
        "confidence": float(entry.get("confidence", 0.0) or 0.0),
        "alternative_hypothesis_count": int(
            entry.get("alternative_hypothesis_count", 0) or 0
        ),
        "evidence_ref_count": len(entry.get("evidence_refs", []) or []),
        "rationale_chars": int(rationale_chars),
        "reason_codes": [REASON_FIELD_PRESENT],
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


def build_reaction_attribution_hypothesis_admitted_event(
    *,
    turn_index: int,
    entry: Mapping[str, Any],
    at: str,
) -> dict[str, Any]:
    """Build the `ReactionAttributionHypothesisAdmitted` audit envelope.

    M18.7 §3 — emitted ALONGSIDE the bus when the
    `reaction_attribution_hypothesis` is non-empty.
    """
    if not isinstance(entry, Mapping):
        return {}
    return {
        "type": "ReactionAttributionHypothesisAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(entry.get("commit_id", "") or ""),
        "participant_id": str(entry.get("participant_id", "") or ""),
        "reaction_to_turn_id": str(entry.get("reaction_to_turn_id", "") or ""),
        "reaction_to_participant_id": str(
            entry.get("reaction_to_participant_id", "") or ""
        ),
        "is_about_assistant_claim": bool(
            entry.get("is_about_assistant_claim", False)
        ),
        "confidence": float(entry.get("confidence", 0.0) or 0.0),
        "alternative_attribution_count": int(
            entry.get("alternative_attribution_count", 0) or 0
        ),
        "evidence_ref_count": len(entry.get("evidence_refs", []) or []),
        "reason_codes": [REASON_FIELD_PRESENT],
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


def build_attribution_hypothesis_skipped_event(
    *,
    turn_index: int,
    latency_mode: str,
    group_turn_binding_present: bool,
    addressee_hypothesis_present: bool,
    reaction_attribution_hypothesis_present: bool,
    at: str,
) -> dict[str, Any]:
    """Build the `AttributionHypothesisSkipped` audit envelope.

    M18.7 DECIDED 13 — emitted when `latency_mode == "fast_chat"`
    AND `group_turn_binding` is non-empty AND the LLM did not
    fill the M18.7 fields. The envelope lets the diagnose
    surface distinguish "the LLM judged no hypothesis" from
    "the thin-conscious loop skipped the field".

    NOT emitted on full-conscious turns (the LLM is expected
    to fill the fields; an empty result is a legitimate
    judgment). NOT emitted on non-group turns (no
    group_turn_binding → no attribution needed).
    """
    return {
        "type": "AttributionHypothesisSkipped",
        "turn_index": int(turn_index),
        "latency_mode": str(latency_mode or ""),
        "group_turn_binding_present": bool(group_turn_binding_present),
        "addressee_hypothesis_present": bool(addressee_hypothesis_present),
        "reaction_attribution_hypothesis_present": bool(
            reaction_attribution_hypothesis_present
        ),
        "reason_code": REASON_FIELD_SKIPPED_FAST_CHAT,
        "reason_codes": [REASON_FIELD_SKIPPED_FAST_CHAT],
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


# === Top-level emission orchestrator ======================================


def should_emit_attribution_hypothesis_skipped(
    *,
    latency_mode: str,
    group_turn_binding: Mapping[str, Any] | None,
    addressee_hypothesis_normalized: Mapping[str, Any] | None,
    reaction_attribution_hypothesis_normalized: Mapping[str, Any] | None,
) -> bool:
    """Return True when the M18.7 fast-chat skip event should fire.

    Conditions (M18.7 DECIDED 13):
    - `latency_mode == "fast_chat"`
    - `group_turn_binding` is non-empty (a group turn)
    - Both M18.7 fields are empty (no hypothesis filled)

    The function does NOT mutate state. The caller appends
    the bus event and may record a state-surface entry.
    """
    if str(latency_mode or "") != "fast_chat":
        return False
    if not isinstance(group_turn_binding, Mapping) or not group_turn_binding:
        return False
    addressee_present = bool(
        isinstance(addressee_hypothesis_normalized, Mapping)
        and addressee_hypothesis_normalized
    )
    reaction_present = bool(
        isinstance(reaction_attribution_hypothesis_normalized, Mapping)
        and reaction_attribution_hypothesis_normalized
    )
    return not addressee_present and not reaction_present


def emit_m18_7_attribution_for_turn(
    *,
    bus: list,
    state: dict,
    conscious_plan: Mapping[str, Any],
    turn_index: int,
    at: str,
    latency_mode: str = "",
    group_turn_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """M18.7 emission orchestrator — run after the conscious loop.

    M18.7 §3 / §5: emit the bus events for the two
    hypotheses (when non-empty), append the entries to
    the bounded state surface, and emit
    `AttributionHypothesisSkipped` when fast_chat +
    group_turn_binding + empty M18.7 fields.

    Returns a small report dict for diagnose / tests:

    ```text
    {
        "addressee_event_emitted": bool,
        "reaction_event_emitted": bool,
        "skipped_event_emitted": bool,
        "addressee_commit_id": str,
        "reaction_commit_id": str,
    }
    ```
    """
    addressee_normalized = normalize_addressee_hypothesis(
        conscious_plan.get("addressee_hypothesis")
        if isinstance(conscious_plan, Mapping) else None
    )
    reaction_normalized = normalize_reaction_attribution_hypothesis(
        conscious_plan.get("reaction_attribution_hypothesis")
        if isinstance(conscious_plan, Mapping) else None
    )

    report: dict[str, Any] = {
        "addressee_event_emitted": False,
        "reaction_event_emitted": False,
        "skipped_event_emitted": False,
        "addressee_commit_id": "",
        "reaction_commit_id": "",
    }

    # Build state entries (one per non-empty field). The
    # state entries carry `at`; the bus events use the same
    # timestamp. We stamp `_at` on the normalized field
    # BEFORE building the state entry.
    if addressee_normalized:
        # Capture rationale length BEFORE stamping `_at`
        # onto the normalized field (the rationale is not
        # part of the v1 state entry shape, but the bus
        # event records its length as `rationale_chars`).
        rationale_chars = len(
            str(addressee_normalized.get("rationale", "") or "")
        )
        addressee_normalized["_at"] = at
        addressee_entry = build_state_entry(
            kind=KIND_ADDRESSEE,
            turn_index=turn_index,
            normalized=addressee_normalized,
        )
        if addressee_entry:
            record_m18_7_attribution_hypotheses(state, addressee_entry)
            event = build_addressee_hypothesis_admitted_event(
                turn_index=turn_index,
                entry=addressee_entry,
                at=at,
                rationale_chars=rationale_chars,
            )
            if event:
                bus.append(event)
                report["addressee_event_emitted"] = True
                report["addressee_commit_id"] = str(
                    addressee_entry.get("commit_id", "")
                )

    if reaction_normalized:
        reaction_normalized["_at"] = at
        reaction_entry = build_state_entry(
            kind=KIND_REACTION,
            turn_index=turn_index,
            normalized=reaction_normalized,
        )
        if reaction_entry:
            record_m18_7_attribution_hypotheses(state, reaction_entry)
            event = build_reaction_attribution_hypothesis_admitted_event(
                turn_index=turn_index,
                entry=reaction_entry,
                at=at,
            )
            if event:
                bus.append(event)
                report["reaction_event_emitted"] = True
                report["reaction_commit_id"] = str(
                    reaction_entry.get("commit_id", "")
                )

    # Fast-chat skip event: only on fast_chat + group_turn +
    # both fields empty.
    if should_emit_attribution_hypothesis_skipped(
        latency_mode=latency_mode,
        group_turn_binding=group_turn_binding,
        addressee_hypothesis_normalized=addressee_normalized,
        reaction_attribution_hypothesis_normalized=reaction_normalized,
    ):
        event = build_attribution_hypothesis_skipped_event(
            turn_index=turn_index,
            latency_mode=latency_mode,
            group_turn_binding_present=bool(group_turn_binding),
            addressee_hypothesis_present=bool(addressee_normalized),
            reaction_attribution_hypothesis_present=bool(reaction_normalized),
            at=at,
        )
        bus.append(event)
        report["skipped_event_emitted"] = True

    return report


# === M18.7.2 minimal-prompt call site ====================================
# M18.7.2 owns a dedicated minimal-prompt LLM call site for
# addressee / reaction attribution. It is decoupled from the
# conscious loop, so the LLM does not compete with the 60+ other
# conscious-loop fields for instruction-following budget. The
# M18.7.1 real-LLM replay (commits b13f07f / b969d8e) confirmed
# that the conscious-loop path is broken at scale: 0/12 turns
# produce non-empty M18.7 v2 attrs when the segment sits at
# char 2914 (37.7%) of a 7.7-26k prompt. The minimal prompt
# is ~1.5-2.0k chars, focused on the two M18.7 fields, and
# the v1 schema is reused unchanged.


def _extract_recent_user_utterances(
    bus_messages: list[Mapping[str, Any]] | None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return the last `limit` UserUtteranceEvent entries from the bus.

    Used by `build_m18_7_minimal_prompt` to provide the LLM with
    the prior 2-3 inbound turns so it can attribute reactions.
    M18.7.2 NEVER inspects raw user text; only structural fields
    (turn id, addressed_participant_ids, reply_to_turn_id,
    ingress_evidence_band) are forwarded.
    """
    if not isinstance(bus_messages, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for evt in reversed(list(bus_messages)):
        if not isinstance(evt, Mapping):
            continue
        if str(evt.get("type", "") or "") != "UserUtteranceEvent":
            continue
        out.append({
            "turn_index": int(evt.get("turn_index", 0) or 0),
            "addressed_participant_ids": list(
                evt.get("addressed_participant_ids", []) or []
            ),
            "mentioned_participant_ids": list(
                evt.get("mentioned_participant_ids", []) or []
            ),
            "reply_to_turn_id": str(evt.get("reply_to_turn_id", "") or ""),
            "quoted_turn_ids": list(evt.get("quoted_turn_ids", []) or []),
            "ingress_evidence_band": str(
                evt.get("ingress_evidence_band", "") or ""
            ),
        })
        if len(out) >= limit:
            break
    out.reverse()
    return out


def _extract_persona_name(state: Mapping[str, Any]) -> str:
    """Best-effort persona name from `state["self_basic_facts"]`."""
    facts = state.get("self_basic_facts") if isinstance(state, Mapping) else None
    if not isinstance(facts, Mapping):
        return ""
    for key in ("persona_name", "display_name", "name", "character_name"):
        v = facts.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()[:64]
    return ""


def build_m18_7_minimal_prompt(
    *,
    state: Mapping[str, Any],
    user_text: str,
    speaker_name: str,
    bus_messages: list[Mapping[str, Any]] | None,
    turn_index: int,
    entity_binding: Mapping[str, Any] | None,
    group_turn_binding: Mapping[str, Any] | None,
    m18_5_structural_decision: str,
) -> tuple[str, str]:
    """Build the M18.7.2 minimal prompt for addressee / reaction
    attribution, decoupled from the conscious loop.

    Returns `(system_prompt, user_prompt)`. The combined length
    is bounded at `M18_7_2_MINIMAL_PROMPT_MAX_CHARS = 2500`
    (v2 bump from 2000 to accommodate the addressed-axis
    strong-signal / counter-example list). The
    LLM is asked to fill a 4-key JSON:

    ```text
    {
      "addressee_hypothesis": {...} or {},
      "reaction_attribution_hypothesis": {...} or {},
      "reasoning_notes": "<one short sentence>",
      "_m18_7_2_source": "m18_7_2_minimal"   # marker; the LLM
                                             # MUST emit this exact
                                             # string verbatim
    }
    ```

    The M18.7 v1 schema is reused unchanged: the LLM returns the
    same `participant_id` / `addressed_to_assistant` /
    `confidence` / `rationale` / `evidence_refs` /
    `alternative_hypotheses` fields that
    `normalize_addressee_hypothesis` and
    `normalize_reaction_attribution_hypothesis` already accept.
    The LLM is the only legitimate source of these fields per
    CLAUDE.md "no keyword / regex" red line.

    Inputs (small subset of state, ~1.5-2.0k chars total):
    - `state["self_basic_facts"]` (persona name only)
    - `state["conversation_log"]` is NOT used directly; the last
      2-3 `UserUtteranceEvent` entries from the bus are forwarded
      (structural fields only — no raw text)
    - `entity_binding` (current_interlocutor, aliases,
      pronoun_bindings)
    - `group_turn_binding` (current_speaker,
      addressed_participant_ids, mentioned_participant_ids,
      ambiguity_band)
    - `m18_5_structural_decision` (so the LLM knows what M18.5
      decided and can reason about it)

    Conscious-loop coupling (M13 / M19 / pending_expectations /
    open_items / etc.) is intentionally absent. The M18.7.1
    real-LLM replay (commits b13f07f / b969d8e) showed that
    these fields consume ~76% of the conscious-loop prompt
    volume but are 100% noise for the attribution decision.
    """
    persona_name = _extract_persona_name(state)
    if persona_name:
        identity_line = f"你是「{persona_name}」，数字人格系统的意识主体。"
    else:
        identity_line = "你是数字人格系统的群聊归因助手。"

    # M18.7.2 v2 (2026-06-10) — system_prompt revised to
    # lift the LLM's `addressed_to_assistant` default-to-False
    # bias. v1 didn't enumerate strong-signal evidence; v2
    # added a 5-item True list, 2-item False list, 3 examples.
    #
    # M18.7.2 v3 (2026-06-11) — bundle 5-run (dce23f0) showed
    # True emit rate 4.8% (1/21). v3 adds 6th strong-signal
    # (re-engaging), default-to-True rule on mixed signals
    # (bot 漏报 > 误报), 2 more inline examples. v1/v2 content
    # preserved. Stays under
    # `M18_7_2_MINIMAL_PROMPT_MAX_CHARS = 2500`.
    system_prompt = (
        f"{identity_line}\n"
        "当前轮次是群聊里某人的一句话。判断两件事：\n"
        "\n"
        "1. addressee_hypothesis: 这句话是否对你（bot）说的？\n"
        "   addressed_to_assistant=True 的强信号（命中任一即倾向 True）：\n"
        "     - @bot，或 mentioned/addressed_participant_ids 含 bot alias\n"
        "     - entity_binding.current_interlocutor = bot\n"
        "     - 第二人称祈使句：'can you' / 'could you' / 'do you'\n"
        "     - 'OK' / '好的' 等接续语后接 bot 指令\n"
        "     - 隐含指令：'Someone is reading this'（bot 是隐含收件人）\n"
        "     - 重新接回 bot：'still waiting' / 'are you there?'\n"
        "   addressed_to_assistant=False 的反例（必须显式 other-recipient 才发）：\n"
        "     - 对其他人说话：'Dave, you first'（target=Dave）\n"
        "     - 对整个群组：'大家怎么看' / 'anyone?'\n"
        "   信号混合或语义不明时，**默认倾向于 True** — bot 漏报 > 误报。\n"
        "   简例：'Can you explain that?' → True；"
        "'Dave, you first.' → False（target=Dave）；"
        "'OK, can you do X?' → True；"
        "'Still waiting for an answer.' → True（re-engaging）；"
        "'Anyone want to take this?' → False（group-wide）。\n"
        "\n"
        "2. reaction_attribution_hypothesis: 这句话是否对某条之前轮次的反应？\n"
        "   - user_text 是否有 '我之前' / '你刚才' / 'that thing ... raised'\n"
        "   - last_user_utterances 的 reply_to_turn_id / quoted_turn_ids\n"
        "\n"
        "基于 entity_binding / group_turn_binding / last_user_utterances 做语义判断。\n"
        "不要用关键词或正则做判断；语义判断由你做。\n"
        "不要生成回复内容；只输出 JSON。\n"
        "5-key JSON spec (the 4 below + the _m18_7_2_source marker):\n"
        "  addressee_hypothesis, reaction_attribution_hypothesis, "
        "reasoning_notes, _m18_7_2_source.\n"
    )

    user_prompt = (
        f"turn_index: {turn_index}\n"
        f"speaker: {speaker_name or 'default_user'}\n"
        f"m18_5_structural_decision: {m18_5_structural_decision or '(none)'}\n"
        f"\n"
        f"entity_binding:\n"
        f"{json.dumps(dict(entity_binding or {}), ensure_ascii=False, indent=2)}\n"
        f"\n"
        f"group_turn_binding:\n"
        f"{json.dumps(dict(group_turn_binding or {}), ensure_ascii=False, indent=2)}\n"
        f"\n"
        f"user_text:\n"
        f"{user_text or ''}\n"
        f"\n"
        f"last_user_utterances (structural fields only, no raw text):\n"
        f"{json.dumps(_extract_recent_user_utterances(bus_messages), ensure_ascii=False, indent=2)}\n"
        f"\n"
        f"输出 JSON（4 数据键 + 1 _m18_7_2_source 标记键）：\n"
        f"{{\n"
        f'  "addressee_hypothesis": {{participant_id, addressed_to_assistant, '
        f'confidence(0-1), rationale(≤200字), evidence_refs(handles), '
        f"alternative_hypotheses(≤2)}}\n"
        f'    或 {{}}（confidence<0.4 或无明确收件人时省略），\n'
        f'  "reaction_attribution_hypothesis": {{participant_id, '
        f'reaction_to_turn_id, reaction_to_participant_id, '
        f'is_about_assistant_claim, confidence(0-1), rationale(≤200字), '
        f"evidence_refs, alternative_attributions(≤2)}}\n"
        f'    或 {{}}（confidence<0.4 或无明确反应目标时省略），\n'
        f'  "reasoning_notes": "<≤120字>",\n'
        f'  "_m18_7_2_source": "m18_7_2_minimal"\n'
        f"}}\n"
    )

    return system_prompt, user_prompt


# === M18.7.2 bus event builders ==========================================


def build_m18_7_2_addressee_hypothesis_admitted_event(
    *,
    turn_index: int,
    entry: Mapping[str, Any],
    at: str,
    rationale_chars: int = 0,
) -> dict[str, Any]:
    """Build the `M18_7_2_AddresseeHypothesisAdmitted` audit envelope.

    M18.7.2 — emitted by the M18.7.2 minimal-prompt call site
    when the LLM fills a non-empty `addressee_hypothesis`. The
    envelope shares `commit_id` with the state surface entry
    so diagnose can cross-reference, and carries `source:
    "m18_7_2_minimal"` so diagnose can distinguish minimal-path
    fills from any future conscious-loop fills.

    The envelope does NOT include the rationale text (M18.7
    DECIDED 11). Engineering audits only the shape (length is
    recorded as `rationale_chars`).
    """
    if not isinstance(entry, Mapping):
        return {}
    return {
        "type": "M18_7_2_AddresseeHypothesisAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(entry.get("commit_id", "") or ""),
        "participant_id": str(entry.get("participant_id", "") or ""),
        "addressed_to_assistant": bool(
            entry.get("addressed_to_assistant", False)
        ),
        "confidence": float(entry.get("confidence", 0.0) or 0.0),
        "alternative_hypothesis_count": int(
            entry.get("alternative_hypothesis_count", 0) or 0
        ),
        "evidence_ref_count": len(entry.get("evidence_refs", []) or []),
        "rationale_chars": int(rationale_chars),
        "source": M18_7_2_SOURCE_TAG,
        "reason_codes": [M18_7_2_REASON_FIELD_PRESENT],
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


def build_m18_7_2_reaction_attribution_hypothesis_admitted_event(
    *,
    turn_index: int,
    entry: Mapping[str, Any],
    at: str,
) -> dict[str, Any]:
    """Build the `M18_7_2_ReactionAttributionHypothesisAdmitted`
    audit envelope. M18.7.2 — emitted by the minimal-prompt
    call site when the LLM fills a non-empty
    `reaction_attribution_hypothesis`.
    """
    if not isinstance(entry, Mapping):
        return {}
    return {
        "type": "M18_7_2_ReactionAttributionHypothesisAdmitted",
        "turn_index": int(turn_index),
        "commit_id": str(entry.get("commit_id", "") or ""),
        "participant_id": str(entry.get("participant_id", "") or ""),
        "reaction_to_turn_id": str(
            entry.get("reaction_to_turn_id", "") or ""
        ),
        "reaction_to_participant_id": str(
            entry.get("reaction_to_participant_id", "") or ""
        ),
        "is_about_assistant_claim": bool(
            entry.get("is_about_assistant_claim", False)
        ),
        "confidence": float(entry.get("confidence", 0.0) or 0.0),
        "alternative_attribution_count": int(
            entry.get("alternative_attribution_count", 0) or 0
        ),
        "evidence_ref_count": len(entry.get("evidence_refs", []) or []),
        "source": M18_7_2_SOURCE_TAG,
        "reason_codes": [M18_7_2_REASON_FIELD_PRESENT],
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


def build_m18_7_2_minimal_degraded_event(
    *,
    turn_index: int,
    reason: str,
    at: str,
) -> dict[str, Any]:
    """Build the `M18_7_2_MinimalDegraded` audit envelope.

    M18.7.2 — emitted when the minimal-prompt LLM call fails
    (timeout, malformed JSON, missing required keys, etc.) and
    the runtime falls back to empty `{}` for both M18.7 fields.
    The envelope lets diagnose distinguish a graceful degraded
    path from a crash. `run_turn` does NOT crash on M18.7.2
    failures (M12-pre pattern).
    """
    return {
        "type": "M18_7_2_MinimalDegraded",
        "turn_index": int(turn_index),
        "reason": str(reason or ""),
        "reason_code": M18_7_2_REASON_MINIMAL_DEGRADED,
        "source": M18_7_2_SOURCE_TAG,
        "engineering_proxy_label": M18_7_ENGINEERING_PROXY_LABEL,
        "at": at,
    }


# === M18.7.2 emission orchestrator =======================================


def emit_m18_7_2_attribution_for_turn(
    *,
    bus: list,
    state: dict,
    plan: Mapping[str, Any],
    turn_index: int,
    at: str,
) -> dict[str, Any]:
    """M18.7.2 emission orchestrator — runs after the minimal-prompt
    LLM call site. Emits `M18_7_2_*` bus events (when non-empty),
    appends entries to the bounded state surface, and stamps
    `source: "m18_7_2_minimal"` on every entry for traceability.

    The orchestrator reuses `normalize_addressee_hypothesis`,
    `normalize_reaction_attribution_hypothesis`,
    `build_state_entry`, and `record_m18_7_attribution_hypotheses`
    unchanged — the M18.7.1 calibration runner and the M20.4
    producer read the same `state["m18_7_attribution_hypotheses"]`
    surface and "just work" once M18.7.2 populates it. The
    `commit_id` is the same SHA-1 as the conscious-loop path
    would produce (`source_ref = "m18_7_{kind}_{turn_index}"`);
    the `source: "m18_7_2_minimal"` field on the state entry
    is the new distinguishability lever.

    Returns a small report dict for diagnose / tests:

    ```text
    {
        "addressee_event_emitted": bool,
        "reaction_event_emitted": bool,
        "addressee_commit_id": str,
        "reaction_commit_id": str,
        "source": "m18_7_2_minimal",
    }
    ```

    This orchestrator does NOT emit
    `AttributionHypothesisSkipped` — that event is owned by the
    M18.7 fast-chat path and is not relevant to the M18.7.2
    minimal call site.
    """
    addressee_normalized = normalize_addressee_hypothesis(
        plan.get("addressee_hypothesis")
        if isinstance(plan, Mapping) else None
    )
    reaction_normalized = normalize_reaction_attribution_hypothesis(
        plan.get("reaction_attribution_hypothesis")
        if isinstance(plan, Mapping) else None
    )

    report: dict[str, Any] = {
        "addressee_event_emitted": False,
        "reaction_event_emitted": False,
        "addressee_commit_id": "",
        "reaction_commit_id": "",
        "source": M18_7_2_SOURCE_TAG,
    }

    if addressee_normalized:
        rationale_chars = len(
            str(addressee_normalized.get("rationale", "") or "")
        )
        addressee_normalized["_at"] = at
        addressee_entry = build_state_entry(
            kind=KIND_ADDRESSEE,
            turn_index=turn_index,
            normalized=addressee_normalized,
        )
        if addressee_entry:
            addressee_entry["source"] = M18_7_2_SOURCE_TAG
            record_m18_7_attribution_hypotheses(state, addressee_entry)
            event = build_m18_7_2_addressee_hypothesis_admitted_event(
                turn_index=turn_index,
                entry=addressee_entry,
                at=at,
                rationale_chars=rationale_chars,
            )
            if event:
                bus.append(event)
                report["addressee_event_emitted"] = True
                report["addressee_commit_id"] = str(
                    addressee_entry.get("commit_id", "")
                )

    if reaction_normalized:
        reaction_normalized["_at"] = at
        reaction_entry = build_state_entry(
            kind=KIND_REACTION,
            turn_index=turn_index,
            normalized=reaction_normalized,
        )
        if reaction_entry:
            reaction_entry["source"] = M18_7_2_SOURCE_TAG
            record_m18_7_attribution_hypotheses(state, reaction_entry)
            event = build_m18_7_2_reaction_attribution_hypothesis_admitted_event(
                turn_index=turn_index,
                entry=reaction_entry,
                at=at,
            )
            if event:
                bus.append(event)
                report["reaction_event_emitted"] = True
                report["reaction_commit_id"] = str(
                    reaction_entry.get("commit_id", "")
                )

    return report


__all__ = [
    "ALLOWED_M18_7_KIND",
    "KIND_ADDRESSEE",
    "KIND_REACTION",
    "M18_7_2_MINIMAL_PROMPT_MAX_CHARS",
    "M18_7_2_REASON_FIELD_PRESENT",
    "M18_7_2_REASON_MINIMAL_DEGRADED",
    "M18_7_2_SOURCE_TAG",
    "M18_7_ENGINEERING_PROXY_LABEL",
    "M18_7_RATIONALE_MAX_CHARS",
    "M18_7_ALTERNATIVE_CAP",
    "M18_7_EVIDENCE_REFS_CAP",
    "M18_7_PARTICIPANT_ID_MAX_CHARS",
    "M18_7_STATE_SURFACE_CAP",
    "M18_7_TURN_ID_MAX_CHARS",
    "REASON_FIELD_PRESENT",
    "REASON_FIELD_SKIPPED_FAST_CHAT",
    "build_addressee_hypothesis_admitted_event",
    "build_attribution_hypothesis_skipped_event",
    "build_m18_7_2_addressee_hypothesis_admitted_event",
    "build_m18_7_2_minimal_degraded_event",
    "build_m18_7_2_reaction_attribution_hypothesis_admitted_event",
    "build_m18_7_minimal_prompt",
    "build_reaction_attribution_hypothesis_admitted_event",
    "build_state_entry",
    "compute_m18_7_commit_id",
    "emit_m18_7_2_attribution_for_turn",
    "emit_m18_7_attribution_for_turn",
    "normalize_addressee_hypothesis",
    "normalize_reaction_attribution_hypothesis",
    "record_m18_7_attribution_hypotheses",
    "should_emit_attribution_hypothesis_skipped",
]
