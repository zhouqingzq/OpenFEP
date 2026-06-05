"""M19 self-expectation loop for Path B."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_drive import (
    _bounded_float,
    _evict_patterns,
    _mapping,
    _new_id,
    _string_list,
    _traction_key,
    _upsert_pattern,
    normalize_m13_drive_state,
)

M19_ENGINEERING_PROXY_LABEL = "mvp_local_self_expectation"

MAX_EXPECTATIONS_TAIL = 24
MAX_MISMATCHES_TAIL = 24
MAX_MISMATCH_MEMORY = 12
MAX_REPAIR_EXPECTATIONS = 12
MAX_SETTLEMENTS_TAIL = 24
MAX_TRACTION_PROPOSALS = 24
MAX_OBSERVATIONS_TAIL = 24
MAX_TOPK = 3

ALLOWED_TARGET_CONTEXTS = frozenset(
    {
        "short_casual_reply",
        "group_privacy_boundary",
        "user_requests_directness",
        "high_stakes_clarification",
        "initiative_after_silence",
        "repair_after_prior_tension",
    }
)
ALLOWED_REPLY_QUALITIES = frozenset({"light", "direct", "repair", "boundary_safe", "compact"})
ALLOWED_OUTCOME_STATUSES = frozenset({"confirmed", "violated", "uncertain"})
ALLOWED_REVIEW_STATUSES = frozenset({"still_active", "stale", "unsupported", "reinforced"})
ALLOWED_MISMATCH_STATUSES = frozenset({"active", "cooling", "resolved", "revoked"})
ALLOWED_REPAIR_STATUSES = frozenset(
    {"pending", "active", "confirmed", "violated", "uncertain", "expired", "superseded"}
)
ALLOWED_VERIFY_ON = frozenset({"next_similar_turn", "natural_idle_shadow_eval"})
ALLOWED_SETTLEMENT_STATUSES = frozenset({"confirmed", "violated", "uncertain", "expired", "superseded"})
ALLOWED_SLOW_STATUSES = frozenset({"active", "stale", "revoked", "downgraded"})

_TARGET_CONTEXT_TO_MISMATCH_TYPE = {
    "short_casual_reply": "outcome_too_heavy_for_context",
    "group_privacy_boundary": "privacy_boundary_miss",
    "user_requests_directness": "outcome_too_heavy_for_context",
    "high_stakes_clarification": "outcome_too_thin_for_context",
    "initiative_after_silence": "initiative_mistiming",
    "repair_after_prior_tension": "repair_insufficient",
}
_TARGET_CONTEXT_TO_INTERVENTION = {
    "short_casual_reply": "prefer_short_casual_surface_form",
    "group_privacy_boundary": "prefer_semantic_only_group_boundary_repair",
    "user_requests_directness": "reduce_assertion_strength_before_clarify",
    "high_stakes_clarification": "reduce_assertion_strength_before_clarify",
    "initiative_after_silence": "delay_initiative_until_structural_silence_threshold",
    "repair_after_prior_tension": "prefer_repair_before_new_assertion",
}
INDIRECT_MISMATCH_REPAIR_BIAS = 0.35
INDIRECT_MISMATCH_CONFLICT_LEVEL = 0.40
INDIRECT_MISMATCH_PREDICTION_ERROR = 0.42
INDIRECT_EXPECTATION_LOOKBACK_TURNS = 2

_INTERVENTION_TO_ACTION_BIASES = {
    "prefer_short_casual_surface_form": {"answer": 0.05, "clarify": 0.03, "ask_question": -0.02},
    "prefer_semantic_only_group_boundary_repair": {
        "abstract_share": 0.08,
        "clarify": 0.05,
        "self_disclose": -0.05,
        "answer": -0.03,
    },
    "reduce_assertion_strength_before_clarify": {"clarify": 0.08, "answer": -0.03, "disagree": -0.05},
    "delay_initiative_until_structural_silence_threshold": {
        "ask_question": 0.04,
        "answer": 0.02,
        "self_disclose": -0.04,
    },
    "prefer_repair_before_new_assertion": {"clarify": 0.07, "empathize": 0.03, "disagree": -0.05},
}


@dataclass
class SelfExpectationPostTurnResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    slow_patch_proposal: dict[str, Any] | None = None
    traction_proposals: list[dict[str, Any]] = field(default_factory=list)


def default_self_expectation_state() -> dict[str, Any]:
    return {
        "expectations_tail": [],
        "mismatches_tail": [],
        "mismatch_memory_fast": [],
        "active_mismatch_focus_topk": [],
        "last_prediction_error_proxy": 0.0,
        "repair_expectations": [],
        "settlements_tail": [],
        "traction_proposals_tail": [],
        "observations_tail": [],
    }


def normalize_self_expectation_state(raw: Any) -> dict[str, Any]:
    base = default_self_expectation_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    for key in (
        "expectations_tail",
        "mismatches_tail",
        "mismatch_memory_fast",
        "active_mismatch_focus_topk",
        "repair_expectations",
        "settlements_tail",
        "traction_proposals_tail",
        "observations_tail",
    ):
        rows = merged.get(key)
        merged[key] = [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    merged["last_prediction_error_proxy"] = round(
        _bounded_float(merged.get("last_prediction_error_proxy"), default=0.0),
        6,
    )
    return copy.deepcopy(merged)


def ensure_self_expectation_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_self_expectation_state(state.get("self_expectation_state"))
    state["self_expectation_state"] = normalized
    return normalized


def prompt_safe_self_expectation_summary(state_or_self_expectation: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (
        _mapping(state_or_self_expectation).get("self_expectation_state")
        if isinstance(state_or_self_expectation, Mapping) and "self_expectation_state" in state_or_self_expectation
        else state_or_self_expectation
    )
    normalized = normalize_self_expectation_state(raw)
    return {
        "active_mismatch_focus_topk": [
            {
                "mismatch_key": str(item.get("mismatch_key", ""))[:120],
                "mismatch_type": str(item.get("mismatch_type", ""))[:80],
                "target_context": str(item.get("target_context", ""))[:80],
                "weighted_support": round(_bounded_float(item.get("weighted_support"), default=0.0), 4),
                "status": str(item.get("status", ""))[:32],
            }
            for item in normalized.get("active_mismatch_focus_topk", [])[:MAX_TOPK]
        ],
        "expectations_tail": [
            {
                "expectation_id": str(item.get("expectation_id", ""))[:120],
                "target_context": str(item.get("target_context", ""))[:80],
                "expected_outcome": str(item.get("expected_outcome", ""))[:160],
                "expected_reply_quality": str(item.get("expected_reply_quality", ""))[:40],
            }
            for item in normalized.get("expectations_tail", [])[-4:]
        ],
        "mismatches_tail": [
            {
                "mismatch_id": str(item.get("mismatch_id", ""))[:120],
                "mismatch_type": str(item.get("mismatch_type", ""))[:80],
                "target_context": str(item.get("target_context", ""))[:80],
                "severity": round(_bounded_float(item.get("severity"), default=0.0), 4),
            }
            for item in normalized.get("mismatches_tail", [])[-4:]
        ],
        "active_repair_expectations": [
            {
                "expectation_id": str(item.get("expectation_id", ""))[:120],
                "target_context": str(item.get("target_context", ""))[:80],
                "intervention": str(item.get("intervention", ""))[:120],
                "status": str(item.get("status", ""))[:32],
                "verify_on": str(item.get("verify_on", ""))[:40],
            }
            for item in normalized.get("repair_expectations", [])[:4]
            if str(item.get("status", "") or "pending") in {"pending", "active", "uncertain"}
        ],
        "active_repair_priors": [
            {
                "id": str(item.get("id", ""))[:120],
                "target_context": str(item.get("target_context", ""))[:80],
                "preferred_intervention": str(item.get("preferred_intervention", ""))[:120],
                "status": str(item.get("status", ""))[:32],
            }
            for item in (
                _mapping(state_or_self_expectation).get("self_cognition", {}).get("repair_priors", [])
                if isinstance(state_or_self_expectation, Mapping)
                else []
            )[:3]
            if isinstance(item, Mapping)
            and str(item.get("status", "") or "active") in {"active", "downgraded"}
        ],
        "last_prediction_error_proxy": round(
            _bounded_float(normalized.get("last_prediction_error_proxy"), default=0.0),
            6,
        ),
    }


def prompt_safe_state_with_self_expectation_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(state)
    safe["self_expectation_state"] = prompt_safe_self_expectation_summary(state)
    return safe


def _bounded_refs(value: Any) -> list[str]:
    return _string_list(value, limit=8)


def normalize_self_response_expectation_proposals(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        target_context = str(item.get("target_context", "") or "").strip()
        if target_context not in ALLOWED_TARGET_CONTEXTS:
            continue
        refs = _bounded_refs(item.get("evidence_refs"))
        if not refs:
            continue
        expected_reply_quality = str(item.get("expected_reply_quality", "") or "").strip()
        if expected_reply_quality not in ALLOWED_REPLY_QUALITIES:
            expected_reply_quality = "direct"
        proposal_id = str(item.get("proposal_id", "") or "").strip() or _new_id("self_exp")
        rows.append(
            {
                "proposal_id": proposal_id[:120],
                "target_context": target_context,
                "expected_outcome": str(item.get("expected_outcome", "") or "").strip()[:160],
                "expected_reply_quality": expected_reply_quality,
                "confidence": round(_bounded_float(item.get("confidence"), default=0.55), 6),
                "evidence_refs": refs,
                "reason_codes": _string_list(item.get("reason_codes"), limit=8),
                "engineering_proxy_label": str(item.get("engineering_proxy_label", "") or M19_ENGINEERING_PROXY_LABEL)[
                    :120
                ],
            }
        )
        if len(rows) >= 2:
            break
    return rows


def normalize_self_expectation_outcome_results(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw or []):
        if not isinstance(item, Mapping):
            continue
        source_expectation_id = str(item.get("source_expectation_id", "") or "").strip()
        target_context = str(item.get("target_context", "") or "").strip()
        status = str(item.get("status", "") or "").strip()
        refs = _bounded_refs(item.get("evidence_refs"))
        if not source_expectation_id or target_context not in ALLOWED_TARGET_CONTEXTS or status not in ALLOWED_OUTCOME_STATUSES:
            continue
        if not refs:
            refs = [f"turn_slot:self_outcome:{index}"]
        rows.append(
            {
                "result_id": str(item.get("result_id", "") or _new_id("self_outcome"))[:120],
                "source_expectation_id": source_expectation_id[:120],
                "target_context": target_context,
                "status": status,
                "evidence_refs": refs,
                "reason_codes": _string_list(item.get("reason_codes"), limit=8),
                "engineering_proxy_label": str(item.get("engineering_proxy_label", "") or M19_ENGINEERING_PROXY_LABEL)[
                    :120
                ],
            }
        )
    return rows


def normalize_self_expectation_review_proposals(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw or []):
        if not isinstance(item, Mapping):
            continue
        source_expectation_id = str(item.get("source_expectation_id", "") or "").strip()
        target_context = str(item.get("target_context", "") or "").strip()
        review_status = str(item.get("review_status", "") or "").strip()
        refs = _bounded_refs(item.get("evidence_refs"))
        if not source_expectation_id or target_context not in ALLOWED_TARGET_CONTEXTS or review_status not in ALLOWED_REVIEW_STATUSES:
            continue
        if not refs:
            refs = [f"idle_slot:self_review:{index}"]
        rows.append(
            {
                "review_id": str(item.get("review_id", "") or _new_id("self_review"))[:120],
                "source_expectation_id": source_expectation_id[:120],
                "target_context": target_context,
                "review_status": review_status,
                "evidence_refs": refs,
                "reason_codes": _string_list(item.get("reason_codes"), limit=8),
                "engineering_proxy_label": str(item.get("engineering_proxy_label", "") or M19_ENGINEERING_PROXY_LABEL)[
                    :120
                ],
            }
        )
    return rows


def apply_conscious_self_expectation_proposals(
    state: dict[str, Any],
    *,
    conscious_plan: Mapping[str, Any],
    now: int,
    turn_index: int,
) -> list[dict[str, Any]]:
    self_state = ensure_self_expectation_state(state)
    events: list[dict[str, Any]] = []
    expectations = list(self_state.get("expectations_tail", []))
    known_ids = {str(item.get("expectation_id", "")) for item in expectations if item.get("expectation_id")}
    for proposal in normalize_self_response_expectation_proposals(
        conscious_plan.get("self_response_expectation_proposals")
    ):
        expectation_id = str(proposal.get("proposal_id", "") or _new_id("self_exp"))[:120]
        if expectation_id in known_ids:
            continue
        row = {
            "expectation_id": expectation_id,
            "target_context": proposal["target_context"],
            "expected_outcome": proposal["expected_outcome"],
            "expected_reply_quality": proposal["expected_reply_quality"],
            "confidence": proposal["confidence"],
            "turn_index": turn_index,
            "at": now,
            "evidence_refs": list(proposal["evidence_refs"]),
            "reason_codes": list(proposal["reason_codes"]),
            "engineering_proxy_label": proposal["engineering_proxy_label"],
            "status": "active",
        }
        expectations.append(row)
        known_ids.add(expectation_id)
        events.append(
            {
                "type": "SelfResponseExpectationCreatedEvent",
                "turn_index": turn_index,
                "at": now,
                **row,
            }
        )
    self_state["expectations_tail"] = expectations[-MAX_EXPECTATIONS_TAIL:]
    state["self_expectation_state"] = self_state
    return events


def _structural_match_for_context(
    target_context: str,
    conscious_plan: Mapping[str, Any],
    *,
    group_turn_binding: Mapping[str, Any] | None = None,
) -> bool:
    pacing = str(conscious_plan.get("reply_pacing_hint", "") or "").strip()
    temporal = _mapping(conscious_plan.get("temporal_assessment"))
    gap = str(temporal.get("time_gap_label", "") or "").strip()
    binding = _mapping(group_turn_binding)
    audience = binding.get("audience_participant_ids") or binding.get("participant_ids") or []
    multi_party = isinstance(audience, list) and len(audience) > 1
    if target_context == "short_casual_reply":
        return pacing == "casual_fast" or bool(conscious_plan.get("prefers_compact_reply"))
    if target_context == "group_privacy_boundary":
        return multi_party
    if target_context == "high_stakes_clarification":
        return pacing == "serious_thinking"
    if target_context == "user_requests_directness":
        return pacing in {"balanced", "serious_thinking"}
    if target_context == "initiative_after_silence":
        return gap in {"medium_gap", "long_gap"}
    if target_context == "repair_after_prior_tension":
        return gap in {"short_gap", "medium_gap", "long_gap"}
    return False


def infer_matching_target_contexts(
    conscious_plan: Mapping[str, Any],
    *,
    self_state: Mapping[str, Any] | None = None,
    group_turn_binding: Mapping[str, Any] | None = None,
) -> set[str]:
    contexts: set[str] = set()
    for item in normalize_self_response_expectation_proposals(
        conscious_plan.get("self_response_expectation_proposals")
    ):
        contexts.add(str(item.get("target_context", "") or "").strip())
    for item in normalize_self_expectation_outcome_results(
        conscious_plan.get("self_expectation_outcome_results")
    ):
        contexts.add(str(item.get("target_context", "") or "").strip())
    if str(conscious_plan.get("reply_pacing_hint", "") or "").strip() == "casual_fast" or bool(
        conscious_plan.get("prefers_compact_reply")
    ):
        contexts.add("short_casual_reply")
    if str(conscious_plan.get("reply_pacing_hint", "") or "").strip() == "serious_thinking":
        contexts.add("high_stakes_clarification")
    binding = _mapping(group_turn_binding)
    audience = binding.get("audience_participant_ids") or binding.get("participant_ids") or []
    if isinstance(audience, list) and len(audience) > 1:
        contexts.add("group_privacy_boundary")
    normalized_self = normalize_self_expectation_state(self_state)
    for repair in normalized_self.get("repair_expectations", []):
        if not isinstance(repair, Mapping):
            continue
        if str(repair.get("status", "") or "pending") not in {"pending", "active", "uncertain"}:
            continue
        ctx = str(repair.get("target_context", "") or "").strip()
        if ctx in ALLOWED_TARGET_CONTEXTS and _structural_match_for_context(
            ctx,
            conscious_plan,
            group_turn_binding=group_turn_binding,
        ):
            contexts.add(ctx)
    return {ctx for ctx in contexts if ctx in ALLOWED_TARGET_CONTEXTS}


def collect_m19_audit_evidence_ids(state: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    self_state = normalize_self_expectation_state(_mapping(state).get("self_expectation_state"))
    for bucket in (
        "expectations_tail",
        "mismatches_tail",
        "repair_expectations",
        "settlements_tail",
        "traction_proposals_tail",
    ):
        for row in self_state.get(bucket, []) or []:
            if not isinstance(row, Mapping):
                continue
            for key in (
                "expectation_id",
                "mismatch_id",
                "proposal_id",
                "settlement_id",
                "proposal_id",
                "source_expectation_id",
            ):
                value = str(row.get(key, "") or "").strip()
                if value:
                    ids.add(value)
    cognition = _mapping(state.get("self_cognition"))
    for row in cognition.get("calibrated_tendencies", []) or []:
        if isinstance(row, Mapping) and str(row.get("id", "") or "").strip():
            ids.add(str(row.get("id", "")).strip())
    for row in cognition.get("repair_priors", []) or []:
        if isinstance(row, Mapping) and str(row.get("id", "") or "").strip():
            ids.add(str(row.get("id", "")).strip())
    return ids


def intervention_primary_action(intervention: str) -> str:
    biases = _INTERVENTION_TO_ACTION_BIASES.get(intervention, {})
    if not biases:
        return "clarify"
    return max(biases.items(), key=lambda item: item[1])[0]


def _repair_action_biases(intervention: str, *, scale: float = 1.0) -> dict[str, float]:
    return {
        action: round(delta * scale, 6)
        for action, delta in _INTERVENTION_TO_ACTION_BIASES.get(intervention, {}).items()
    }


def _apply_intervention_to_guidance(
    *,
    intervention: str,
    weight: float,
    repair_bias_delta: float,
    conflict_delta: float,
    assertion_cap: float | None,
    action_biases: dict[str, float],
    preferred: list[str],
    discouraged: list[str],
) -> tuple[float, float, float | None]:
    repair_bias_delta = max(repair_bias_delta, min(0.22, (0.08 + weight * 0.14)))
    conflict_delta = max(conflict_delta, min(0.15, (0.05 + weight * 0.10)))
    if intervention == "reduce_assertion_strength_before_clarify":
        assertion_cap = 0.52 if assertion_cap is None else min(assertion_cap, 0.52)
    for action, delta in _repair_action_biases(intervention, scale=weight).items():
        action_biases[action] = round(action_biases.get(action, 0.0) + delta, 6)
        if delta > 0:
            preferred.append(action)
        elif delta < 0:
            discouraged.append(action)
    return repair_bias_delta, conflict_delta, assertion_cap


def build_self_repair_guidance(
    state: Mapping[str, Any],
    *,
    conscious_plan: Mapping[str, Any],
    group_turn_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    self_state = normalize_self_expectation_state(_mapping(state).get("self_expectation_state"))
    cognition = _mapping(state.get("self_cognition"))
    current_contexts = infer_matching_target_contexts(
        conscious_plan,
        self_state=self_state,
        group_turn_binding=group_turn_binding,
    )
    matched_rows = [
        row
        for row in self_state.get("repair_expectations", [])
        if str(row.get("status", "") or "pending") in {"pending", "active", "uncertain"}
        and str(row.get("target_context", "") or "") in current_contexts
    ]
    matched_priors = [
        row
        for row in cognition.get("repair_priors", []) or []
        if isinstance(row, Mapping)
        and str(row.get("status", "") or "active") in {"active", "downgraded"}
        and str(row.get("target_context", "") or "") in current_contexts
    ]
    if not matched_rows and not matched_priors:
        return {
            "repair_bias_delta": 0.0,
            "conflict_level_delta": 0.0,
            "assertion_strength_cap": None,
            "reply_action_biases": {},
            "preferred_reply_actions": [],
            "discouraged_reply_actions": [],
            "summary": {},
        }
    repair_bias_delta = 0.0
    conflict_delta = 0.0
    assertion_cap: float | None = None
    preferred: list[str] = []
    discouraged: list[str] = []
    action_biases: dict[str, float] = {}
    summary_rows: list[dict[str, Any]] = []
    for row in matched_rows[:2]:
        priority = _bounded_float(row.get("priority"), default=0.5)
        intervention = str(row.get("intervention", "") or "")
        repair_bias_delta, conflict_delta, assertion_cap = _apply_intervention_to_guidance(
            intervention=intervention,
            weight=0.55 + priority * 0.35,
            repair_bias_delta=repair_bias_delta,
            conflict_delta=conflict_delta,
            assertion_cap=assertion_cap,
            action_biases=action_biases,
            preferred=preferred,
            discouraged=discouraged,
        )
        summary_rows.append(
            {
                "expectation_id": str(row.get("expectation_id", ""))[:120],
                "target_context": str(row.get("target_context", ""))[:80],
                "intervention": intervention[:120],
                "status": str(row.get("status", ""))[:32],
                "verify_on": str(row.get("verify_on", ""))[:40],
                "source": "repair_expectation",
            }
        )
    for prior in matched_priors[:2]:
        intervention = str(prior.get("preferred_intervention", "") or "")
        confidence = _bounded_float(prior.get("confidence"), default=0.55)
        repair_bias_delta, conflict_delta, assertion_cap = _apply_intervention_to_guidance(
            intervention=intervention,
            weight=0.35 + confidence * 0.25,
            repair_bias_delta=repair_bias_delta,
            conflict_delta=conflict_delta,
            assertion_cap=assertion_cap,
            action_biases=action_biases,
            preferred=preferred,
            discouraged=discouraged,
        )
        summary_rows.append(
            {
                "expectation_id": str(prior.get("id", ""))[:120],
                "target_context": str(prior.get("target_context", ""))[:80],
                "intervention": intervention[:120],
                "status": str(prior.get("status", ""))[:32],
                "verify_on": "slow_repair_prior",
                "source": "repair_prior",
            }
        )
    return {
        "repair_bias_delta": round(repair_bias_delta, 6),
        "conflict_level_delta": round(conflict_delta, 6),
        "assertion_strength_cap": assertion_cap,
        "reply_action_biases": action_biases,
        "preferred_reply_actions": list(dict.fromkeys(preferred))[:3],
        "discouraged_reply_actions": list(dict.fromkeys(discouraged))[:3],
        "summary": {
            "active_expectations": summary_rows,
            "advisory_only": True,
            "ordinary_language_hint": "Prefer the lower-friction repair tendency only in matching contexts.",
            "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
        },
    }


def _find_expectation(
    self_state: Mapping[str, Any],
    *,
    expectation_id: str,
) -> dict[str, Any] | None:
    for row in reversed(list(self_state.get("expectations_tail", []))):
        if str(row.get("expectation_id", "") or "") == expectation_id:
            return dict(row)
    return None


def _make_recurrence_key(target_context: str, mismatch_type: str) -> str:
    return f"{target_context}:{mismatch_type}"[:160]


def _prediction_error_after(
    *,
    reward_prediction_error: float,
    outcome_status: str,
    reduction_target: float,
) -> float:
    current = _bounded_float(abs(reward_prediction_error), default=0.0)
    if outcome_status == "confirmed":
        return round(min(current, max(0.02, reduction_target * 0.8)), 6)
    if outcome_status == "violated":
        return round(min(1.0, max(current, reduction_target + 0.15)), 6)
    return round(min(1.0, max(current, reduction_target)), 6)


def _append_tail(rows: list[dict[str, Any]], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows.append(payload)
    return rows[-limit:]


def _lookup_fast_mismatch(rows: list[dict[str, Any]], *, mismatch_key: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("mismatch_key", "") or "") == mismatch_key:
            return row
    return None


def _apply_mismatch_decay(self_state: dict[str, Any], *, turn_index: int) -> None:
    rows = [dict(item) for item in self_state.get("mismatch_memory_fast", []) if isinstance(item, Mapping)]
    for row in rows:
        last_seen_turn = int(row.get("last_seen_turn", turn_index) or turn_index)
        gap = max(0, turn_index - last_seen_turn)
        if gap <= 0:
            continue
        decay = min(0.25, gap * 0.03)
        row["recent_support"] = round(max(0.0, _bounded_float(row.get("recent_support"), default=0.0) - decay), 6)
        row["weighted_support"] = round(
            max(0.0, _bounded_float(row.get("weighted_support"), default=0.0) - decay * 0.6),
            6,
        )
        if _bounded_float(row.get("weighted_support"), default=0.0) <= 0.08:
            row["status"] = "resolved"
        elif gap >= 3 and str(row.get("status", "") or "active") == "active":
            row["status"] = "cooling"
    self_state["mismatch_memory_fast"] = rows[-MAX_MISMATCH_MEMORY:]


def _recompute_active_focus(self_state: dict[str, Any], *, turn_index: int) -> None:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in self_state.get("mismatch_memory_fast", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "") or "active")
        if status not in {"active", "cooling"}:
            continue
        recency = max(0.0, 1.0 - max(0, turn_index - int(row.get("last_seen_turn", turn_index) or turn_index)) * 0.15)
        score = (
            _bounded_float(row.get("weighted_support"), default=0.0)
            + recency * 0.25
            + _bounded_float(row.get("last_prediction_error_proxy"), default=0.0) * 0.30
            + (0.10 if str(row.get("target_context", "") or "") in ALLOWED_TARGET_CONTEXTS else 0.0)
        )
        ranked.append((round(score, 6), dict(row)))
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("mismatch_key", ""))))
    self_state["active_mismatch_focus_topk"] = [row for _, row in ranked[:MAX_TOPK]]


def _maybe_create_repair_expectation(
    self_state: dict[str, Any],
    *,
    mismatch_row: Mapping[str, Any],
    now: int,
    turn_index: int,
) -> dict[str, Any] | None:
    key = str(mismatch_row.get("mismatch_key", "") or "")
    if not key:
        return None
    existing = [
        row
        for row in self_state.get("repair_expectations", [])
        if str(row.get("source_mismatch_key", "") or "") == key
        and str(row.get("status", "") or "pending") in {"pending", "active", "uncertain"}
    ]
    if existing:
        return None
    if _bounded_float(mismatch_row.get("weighted_support"), default=0.0) < 0.95:
        return None
    if int(mismatch_row.get("support_count", 0) or 0) < 2:
        return None
    target_context = str(mismatch_row.get("target_context", "") or "")
    intervention = _TARGET_CONTEXT_TO_INTERVENTION.get(target_context, "reduce_assertion_strength_before_clarify")
    expectation_id = _new_id("self_repair")
    reduction_target = round(
        min(
            1.0,
            max(0.12, _bounded_float(mismatch_row.get("last_prediction_error_proxy"), default=0.25) * 0.75),
        ),
        6,
    )
    row = {
        "expectation_id": expectation_id,
        "source_mismatch_key": key,
        "target_context": target_context,
        "intervention": intervention,
        "success_criteria": [
            f"{target_context} recurs without the same mismatch pattern",
            "indirect tension drops below the reduction target",
        ],
        "failure_criteria": [
            f"{target_context} recurs with the same recurrence class again",
            "prediction error does not move toward the reduction target",
        ],
        "verify_on": "next_similar_turn",
        "opportunity_window": 4,
        "expires_after_opportunities": 2,
        "priority": round(min(1.0, 0.4 + _bounded_float(mismatch_row.get("weighted_support"), default=0.0) * 0.25), 6),
        "prediction_error_reduction_target": reduction_target,
        "status": "active",
        "created_at": now,
        "created_turn_index": turn_index,
        "evidence_refs": _bounded_refs(mismatch_row.get("evidence_refs_tail")),
        "reason_codes": ["stable_fast_mismatch"],
        "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
        "opportunities_seen": 0,
        "settlement_ids": [],
    }
    self_state["repair_expectations"] = _append_tail(
        [dict(item) for item in self_state.get("repair_expectations", []) if isinstance(item, Mapping)],
        row,
        limit=MAX_REPAIR_EXPECTATIONS,
    )
    return row


def build_shadow_validation(
    repair: Mapping[str, Any],
    *,
    prediction_error_before: float,
    prediction_error_after: float,
    control_guidance: Mapping[str, Any],
) -> dict[str, Any]:
    intervention = str(repair.get("intervention", "") or "").strip()
    control = _mapping(control_guidance)
    repair_bias = _bounded_float(control.get("repair_bias"), default=0.0)
    estimated_delta = round(prediction_error_before - prediction_error_after, 6)
    active_biases = _repair_action_biases(intervention)
    discouraged_actions = [action for action, delta in active_biases.items() if delta < 0]
    alternative_intervention = ""
    for candidate, biases in _INTERVENTION_TO_ACTION_BIASES.items():
        if candidate == intervention:
            continue
        if discouraged_actions and any(biases.get(action, 0.0) > 0 for action in discouraged_actions):
            alternative_intervention = candidate
            break
    return {
        "shadow_id": _new_id("self_shadow"),
        "expectation_id": str(repair.get("expectation_id", "") or "")[:120],
        "intervention": intervention[:120],
        "alternative_intervention": alternative_intervention[:120],
        "preferred_action": intervention_primary_action(intervention),
        "estimated_prediction_error_delta": estimated_delta,
        "repair_bias_at_eval": round(repair_bias, 6),
        "active_intervention_fits": bool(estimated_delta > 0.02 and repair_bias >= 0.12),
        "advisory_only": True,
        "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
    }


def _settlement_status_from_outcome(
    *,
    outcome_status: str | None,
    prediction_after: float,
    reduction_target: float,
) -> str:
    status = str(outcome_status or "").strip()
    if status == "violated":
        return "violated"
    if status == "uncertain":
        return "uncertain"
    if status == "confirmed":
        return "confirmed"
    return "uncertain"


def _has_primary_indirect_signal(
    *,
    control_guidance: Mapping[str, Any],
    reward_prediction_error_proxy: float,
) -> bool:
    control = _mapping(control_guidance)
    repair_bias = _bounded_float(control.get("repair_bias"), default=0.0)
    conflict_level = _bounded_float(control.get("conflict_level"), default=0.0)
    prediction_error = _bounded_float(abs(reward_prediction_error_proxy), default=0.0)
    return (
        repair_bias >= INDIRECT_MISMATCH_REPAIR_BIAS
        or conflict_level >= INDIRECT_MISMATCH_CONFLICT_LEVEL
        or prediction_error >= INDIRECT_MISMATCH_PREDICTION_ERROR
    )


def _record_mismatch_row(
    *,
    source_expectation_id: str,
    target_context: str,
    status: str,
    severity: float,
    confidence: float,
    prediction_after: float,
    evidence_refs: list[str],
    reason_codes: list[str],
    now: int,
    turn_index: int,
    mismatches_tail: list[dict[str, Any]],
    mismatch_memory_fast: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatch_type = _TARGET_CONTEXT_TO_MISMATCH_TYPE.get(target_context, "persona_drift")
    mismatch_key = _make_recurrence_key(target_context, mismatch_type)
    mismatch = {
        "mismatch_id": _new_id("self_mismatch"),
        "source_expectation_id": source_expectation_id,
        "target_context": target_context,
        "mismatch_type": mismatch_type,
        "mismatch_summary": f"{target_context} drifted away from the expected reply shape."[:200],
        "severity": round(min(1.0, severity), 6),
        "confidence": round(confidence, 6),
        "prediction_error_proxy": prediction_after,
        "recurrence_key": mismatch_key,
        "turn_index": turn_index,
        "at": now,
        "evidence_refs": evidence_refs[:8],
        "reason_codes": reason_codes[:8],
        "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
    }
    mismatches_tail[:] = _append_tail(mismatches_tail, mismatch, limit=MAX_MISMATCHES_TAIL)
    fast_row = _lookup_fast_mismatch(mismatch_memory_fast, mismatch_key=mismatch_key)
    if fast_row is None:
        fast_row = {
            "mismatch_key": mismatch_key,
            "mismatch_type": mismatch_type,
            "target_context": target_context,
            "support_count": 0,
            "weighted_support": 0.0,
            "recent_support": 0.0,
            "last_prediction_error_proxy": 0.0,
            "last_seen_at": now,
            "last_seen_turn": turn_index,
            "evidence_refs_tail": [],
            "status": "cooling",
        }
        mismatch_memory_fast.append(fast_row)
    fast_row["support_count"] = int(fast_row.get("support_count", 0) or 0) + 1
    increment = round(
        mismatch["severity"] * mismatch["confidence"] + prediction_after * 0.30 + (0.10 if status == "violated" else 0.04),
        6,
    )
    fast_row["weighted_support"] = round(
        min(4.0, _bounded_float(fast_row.get("weighted_support"), default=0.0) + increment),
        6,
    )
    fast_row["recent_support"] = round(min(2.0, _bounded_float(fast_row.get("recent_support"), default=0.0) + 0.45), 6)
    fast_row["last_prediction_error_proxy"] = prediction_after
    fast_row["last_seen_at"] = now
    fast_row["last_seen_turn"] = turn_index
    fast_row["evidence_refs_tail"] = _string_list(
        [*fast_row.get("evidence_refs_tail", []), *mismatch["evidence_refs"]],
        limit=8,
    )
    if int(fast_row.get("support_count", 0) or 0) >= 2 and _bounded_float(fast_row.get("weighted_support"), default=0.0) >= 0.95:
        fast_row["status"] = "active"
    return mismatch


def _repair_expectation_by_context(
    self_state: Mapping[str, Any],
    *,
    target_context: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in self_state.get("repair_expectations", [])
        if isinstance(row, Mapping)
        and str(row.get("target_context", "") or "") == target_context
        and str(row.get("status", "") or "pending") in {"pending", "active", "uncertain"}
    ]


def _promotion_candidate(
    *,
    self_state: Mapping[str, Any],
    self_cognition: Mapping[str, Any],
    now: int,
    turn_index: int,
) -> dict[str, Any] | None:
    settlements = [dict(item) for item in self_state.get("settlements_tail", []) if isinstance(item, Mapping)]
    if not settlements:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in settlements:
        key = str(row.get("source_mismatch_key", "") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    existing_tendencies = [dict(item) for item in _mapping(self_cognition).get("calibrated_tendencies", []) if isinstance(item, Mapping)]
    existing_priors = [dict(item) for item in _mapping(self_cognition).get("repair_priors", []) if isinstance(item, Mapping)]
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("turn_index", 0) or 0))
        confirmed = [row for row in rows if str(row.get("status", "") or "") == "confirmed"]
        violated = [row for row in rows if str(row.get("status", "") or "") == "violated"]
        latest = rows[-1]
        target_context = str(latest.get("matched_context", "") or "")
        repair_expectation_id = str(latest.get("expectation_id", "") or "")
        if len(confirmed) >= 2:
            intervention = ""
            for repair in reversed(list(self_state.get("repair_expectations", []))):
                if str(repair.get("expectation_id", "") or "") == repair_expectation_id:
                    intervention = str(repair.get("intervention", "") or "")
                    break
            tendency_id = ""
            for row in existing_tendencies:
                if str(row.get("source_mismatch_key", "") or "") == key:
                    tendency_id = str(row.get("id", "") or "")
                    break
            prior_id = ""
            for row in existing_priors:
                if str(row.get("source_mismatch_key", "") or "") == key:
                    prior_id = str(row.get("id", "") or "")
                    break
            confidence = round(
                min(0.92, 0.55 + len(confirmed) * 0.10 + max(0.0, float(latest.get("prediction_error_delta", 0.0) or 0.0)) * 0.2),
                6,
            )
            evidence_refs = _string_list(
                [
                    *latest.get("evidence_refs", []),
                    *[row.get("settlement_id") for row in confirmed[-3:]],
                ],
                limit=8,
            )
            return {
                "apply": True,
                "summary_delta": (
                    f"When {target_context} recurs, a lighter repair strategy reduces prediction error more reliably."
                ),
                "new_identity_tensions": [],
                "new_known_limits": [],
                "calibrated_tendencies": [
                    {
                        "id": tendency_id or _new_id("cal_tend"),
                        "target_context": target_context,
                        "tendency_summary": (
                            f"In {target_context}, the default reply tendency increases prediction error unless repair is preferred."
                        ),
                        "confidence": confidence,
                        "source_mismatch_key": key,
                        "evidence_refs": evidence_refs,
                        "status": "active",
                    }
                ],
                "repair_priors": [
                    {
                        "id": prior_id or _new_id("repair_prior"),
                        "target_context": target_context,
                        "preferred_intervention": intervention or _TARGET_CONTEXT_TO_INTERVENTION.get(
                            target_context,
                            "reduce_assertion_strength_before_clarify",
                        ),
                        "confidence": confidence,
                        "source_expectation_id": repair_expectation_id,
                        "source_mismatch_key": key,
                        "settlement_ids": [str(row.get("settlement_id", "")) for row in confirmed[-4:] if row.get("settlement_id")],
                        "evidence_refs": evidence_refs,
                        "status": "active",
                    }
                ],
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "reason": "m19_self_expectation_promotion",
            }
        active_tendency = next(
            (
                row
                for row in existing_tendencies
                if str(row.get("source_mismatch_key", "") or "") == key
                and str(row.get("status", "") or "") in {"active", "stale"}
            ),
            None,
        )
        active_prior = next(
            (
                row
                for row in existing_priors
                if str(row.get("source_mismatch_key", "") or "") == key
                and str(row.get("status", "") or "") in {"active", "downgraded"}
            ),
            None,
        )
        if active_prior and len(violated) >= 2:
            evidence_refs = _string_list(
                [row.get("settlement_id") for row in violated[-3:] if row.get("settlement_id")],
                limit=8,
            )
            return {
                "apply": True,
                "summary_delta": f"Recent evidence shows the prior calibration for {target_context} should be downgraded.",
                "new_identity_tensions": [],
                "new_known_limits": [],
                "calibrated_tendencies": [
                    {
                        "id": str((active_tendency or {}).get("id", "") or _new_id("cal_tend")),
                        "target_context": target_context,
                        "tendency_summary": str((active_tendency or {}).get("tendency_summary", "") or "")[:240],
                        "confidence": round(
                            max(0.25, _bounded_float((active_tendency or {}).get("confidence"), default=0.6) - 0.18),
                            6,
                        ),
                        "source_mismatch_key": key,
                        "evidence_refs": evidence_refs,
                        "status": "stale",
                    }
                ],
                "repair_priors": [
                    {
                        "id": str(active_prior.get("id", "") or _new_id("repair_prior")),
                        "target_context": target_context,
                        "preferred_intervention": str(active_prior.get("preferred_intervention", "") or "")[:120],
                        "confidence": round(max(0.2, _bounded_float(active_prior.get("confidence"), default=0.6) - 0.20), 6),
                        "source_expectation_id": str(active_prior.get("source_expectation_id", "") or "")[:120],
                        "source_mismatch_key": key,
                        "settlement_ids": [str(row.get("settlement_id", "")) for row in violated[-4:] if row.get("settlement_id")],
                        "evidence_refs": evidence_refs,
                        "status": "downgraded",
                    }
                ],
                "evidence_refs": evidence_refs,
                "confidence": 0.72,
                "reason": "m19_self_expectation_downgrade",
            }
    return None


def apply_self_expectation_post_turn(
    state: dict[str, Any],
    *,
    conscious_plan: Mapping[str, Any],
    control_guidance: Mapping[str, Any],
    reward_prediction_error_proxy: float,
    reward_event_id: str,
    now: int,
    turn_index: int,
    group_turn_binding: Mapping[str, Any] | None = None,
) -> SelfExpectationPostTurnResult:
    result = SelfExpectationPostTurnResult()
    self_state = ensure_self_expectation_state(state)
    previous_prediction_error = _bounded_float(self_state.get("last_prediction_error_proxy"), default=0.0)
    _apply_mismatch_decay(self_state, turn_index=turn_index)
    mismatches_tail = [dict(item) for item in self_state.get("mismatches_tail", []) if isinstance(item, Mapping)]
    mismatch_memory_fast = [dict(item) for item in self_state.get("mismatch_memory_fast", []) if isinstance(item, Mapping)]
    current_contexts = infer_matching_target_contexts(
        conscious_plan,
        self_state=self_state,
        group_turn_binding=group_turn_binding,
    )
    control = _mapping(control_guidance)
    conflict_level = _bounded_float(control.get("conflict_level"), default=0.0)
    repair_bias = _bounded_float(control.get("repair_bias"), default=0.0)

    outcome_results = normalize_self_expectation_outcome_results(
        conscious_plan.get("self_expectation_outcome_results")
    )
    outcome_by_context: dict[str, dict[str, Any]] = {}
    outcome_source_ids: set[str] = set()
    mismatches_this_turn: list[dict[str, Any]] = []
    for outcome in outcome_results:
        source_expectation_id = str(outcome.get("source_expectation_id", "") or "")
        target_context = str(outcome.get("target_context", "") or "")
        expectation = _find_expectation(self_state, expectation_id=source_expectation_id)
        if not expectation:
            continue
        outcome_by_context[target_context] = outcome
        outcome_source_ids.add(source_expectation_id)
        status = str(outcome.get("status", "") or "")
        mismatch_type = _TARGET_CONTEXT_TO_MISMATCH_TYPE.get(target_context, "persona_drift")
        mismatch_key = _make_recurrence_key(target_context, mismatch_type)
        prediction_after = _prediction_error_after(
            reward_prediction_error=reward_prediction_error_proxy,
            outcome_status=status,
            reduction_target=max(0.15, previous_prediction_error or 0.18),
        )
        if status == "confirmed":
            row = _lookup_fast_mismatch(mismatch_memory_fast, mismatch_key=mismatch_key)
            if row is not None:
                row["weighted_support"] = round(max(0.0, _bounded_float(row.get("weighted_support"), default=0.0) - 0.28), 6)
                row["recent_support"] = round(max(0.0, _bounded_float(row.get("recent_support"), default=0.0) - 0.22), 6)
                row["last_prediction_error_proxy"] = prediction_after
                row["status"] = "resolved" if _bounded_float(row.get("weighted_support"), default=0.0) <= 0.10 else "cooling"
            result.events.append(
                {
                    "type": "SelfExpectationOutcomeConfirmedEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "source_expectation_id": source_expectation_id,
                    "target_context": target_context,
                    "evidence_refs": list(dict.fromkeys([*outcome.get("evidence_refs", []), reward_event_id]))[:8],
                    "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
                }
            )
            continue
        severity = round(
            min(1.0, 0.28 + conflict_level * 0.28 + repair_bias * 0.24 + prediction_after * 0.20),
            6,
        )
        mismatch = _record_mismatch_row(
            source_expectation_id=source_expectation_id,
            target_context=target_context,
            status=status,
            severity=severity,
            confidence=round(0.70 if status == "violated" else 0.56, 6),
            prediction_after=prediction_after,
            evidence_refs=list(
                dict.fromkeys([*outcome.get("evidence_refs", []), source_expectation_id, reward_event_id])
            )[:8],
            reason_codes=list(dict.fromkeys([*outcome.get("reason_codes", []), status]))[:8],
            now=now,
            turn_index=turn_index,
            mismatches_tail=mismatches_tail,
            mismatch_memory_fast=mismatch_memory_fast,
        )
        mismatches_this_turn.append(mismatch)
        result.events.append(
            {
                "type": "SelfExpectationMismatchObservedEvent",
                "turn_index": turn_index,
                "at": now,
                **mismatch,
            }
        )

    if _has_primary_indirect_signal(
        control_guidance=control,
        reward_prediction_error_proxy=reward_prediction_error_proxy,
    ):
        for expectation in reversed(list(self_state.get("expectations_tail", []))):
            if not isinstance(expectation, Mapping):
                continue
            source_expectation_id = str(expectation.get("expectation_id", "") or "").strip()
            if not source_expectation_id or source_expectation_id in outcome_source_ids:
                continue
            if str(expectation.get("status", "") or "active") != "active":
                continue
            if turn_index - int(expectation.get("turn_index", turn_index) or turn_index) > INDIRECT_EXPECTATION_LOOKBACK_TURNS:
                continue
            target_context = str(expectation.get("target_context", "") or "").strip()
            if target_context not in ALLOWED_TARGET_CONTEXTS:
                continue
            mismatch_key = _make_recurrence_key(
                target_context,
                _TARGET_CONTEXT_TO_MISMATCH_TYPE.get(target_context, "persona_drift"),
            )
            if any(str(item.get("recurrence_key", "") or "") == mismatch_key for item in mismatches_this_turn):
                continue
            prediction_after = _prediction_error_after(
                reward_prediction_error=reward_prediction_error_proxy,
                outcome_status="violated",
                reduction_target=max(0.15, previous_prediction_error or 0.18),
            )
            mismatch = _record_mismatch_row(
                source_expectation_id=source_expectation_id,
                target_context=target_context,
                status="uncertain",
                severity=round(
                    min(1.0, 0.22 + conflict_level * 0.24 + repair_bias * 0.22 + prediction_after * 0.18),
                    6,
                ),
                confidence=0.58,
                prediction_after=prediction_after,
                evidence_refs=list(
                    dict.fromkeys(
                        [
                            source_expectation_id,
                            reward_event_id,
                            f"turn:{turn_index}:indirect_control",
                        ]
                    )
                )[:8],
                reason_codes=["indirect_control_guidance", "prediction_error_proxy"],
                now=now,
                turn_index=turn_index,
                mismatches_tail=mismatches_tail,
                mismatch_memory_fast=mismatch_memory_fast,
            )
            mismatches_this_turn.append(mismatch)
            result.events.append(
                {
                    "type": "SelfExpectationMismatchObservedEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "indirect_observation": True,
                    **mismatch,
                }
            )

    for repair in [item for item in self_state.get("repair_expectations", []) if isinstance(item, dict)]:
        if int(repair.get("created_turn_index", turn_index) or turn_index) + int(repair.get("opportunity_window", 4) or 4) < turn_index:
            if str(repair.get("status", "") or "") in {"pending", "active", "uncertain"}:
                repair["status"] = "expired"
                settlement = {
                    "settlement_id": _new_id("self_settlement"),
                    "expectation_id": str(repair.get("expectation_id", "") or "")[:120],
                    "source_mismatch_key": str(repair.get("source_mismatch_key", "") or "")[:160],
                    "matched_context": str(repair.get("target_context", "") or "")[:80],
                    "status": "expired",
                    "prediction_error_before": previous_prediction_error,
                    "prediction_error_after": previous_prediction_error,
                    "prediction_error_delta": 0.0,
                    "success_signals": [],
                    "failure_signals": [],
                    "confidence": 0.5,
                    "at": now,
                    "turn_index": turn_index,
                    "evidence_refs": _bounded_refs(repair.get("evidence_refs")),
                    "reason_codes": ["opportunity_window_elapsed"],
                    "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
                }
                self_state["settlements_tail"] = _append_tail(
                    [dict(item) for item in self_state.get("settlements_tail", []) if isinstance(item, Mapping)],
                    settlement,
                    limit=MAX_SETTLEMENTS_TAIL,
                )
                result.events.append({"type": "SelfRepairSettlementEvent", **settlement})

    for target_context in sorted(current_contexts):
        repairs = _repair_expectation_by_context(self_state, target_context=target_context)
        if not repairs:
            continue
        outcome = outcome_by_context.get(target_context)
        for repair in repairs:
            if int(repair.get("last_settlement_turn_index", -1) or -1) == turn_index:
                continue
            max_opportunities = int(repair.get("expires_after_opportunities", 2) or 2)
            if int(repair.get("opportunities_seen", 0) or 0) >= max_opportunities:
                continue
            repair["opportunities_seen"] = int(repair.get("opportunities_seen", 0) or 0) + 1
            prediction_after = _prediction_error_after(
                reward_prediction_error=reward_prediction_error_proxy,
                outcome_status=str(_mapping(outcome).get("status", "") or ""),
                reduction_target=_bounded_float(repair.get("prediction_error_reduction_target"), default=0.2),
            )
            settlement_status = _settlement_status_from_outcome(
                outcome_status=str(_mapping(outcome).get("status", "") or ""),
                prediction_after=prediction_after,
                reduction_target=_bounded_float(repair.get("prediction_error_reduction_target"), default=0.2),
            )
            before = previous_prediction_error or _bounded_float(
                repair.get("prediction_error_reduction_target"),
                default=0.2,
            )
            delta = round(before - prediction_after, 6)
            shadow_validation = build_shadow_validation(
                repair,
                prediction_error_before=before,
                prediction_error_after=prediction_after,
                control_guidance=control,
            )
            settlement = {
                "settlement_id": _new_id("self_settlement"),
                "expectation_id": str(repair.get("expectation_id", "") or "")[:120],
                "source_mismatch_key": str(repair.get("source_mismatch_key", "") or "")[:160],
                "matched_context": target_context,
                "status": settlement_status,
                "prediction_error_before": round(before, 6),
                "prediction_error_after": round(prediction_after, 6),
                "prediction_error_delta": delta,
                "success_signals": (
                    ["prediction_error_reduced", "matching_context_without_repeat_mismatch"]
                    if settlement_status == "confirmed"
                    else []
                ),
                "failure_signals": (
                    ["same_recurrence_class_reappeared", "prediction_error_above_target"]
                    if settlement_status == "violated"
                    else []
                ),
                "shadow_validation": shadow_validation,
                "confidence": round(0.76 if settlement_status == "confirmed" else 0.66 if settlement_status == "violated" else 0.55, 6),
                "at": now,
                "turn_index": turn_index,
                "evidence_refs": _string_list(
                    [
                        *repair.get("evidence_refs", []),
                        *(_mapping(outcome).get("evidence_refs", []) if outcome else []),
                        reward_event_id,
                        shadow_validation.get("shadow_id"),
                    ],
                    limit=8,
                ),
                "reason_codes": _string_list(
                    [
                        *(outcome.get("reason_codes", []) if outcome else []),
                        settlement_status,
                        "shadow_validation_advisory",
                    ],
                    limit=8,
                ),
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            }
            if settlement_status == "uncertain" and shadow_validation.get("active_intervention_fits"):
                settlement["confidence"] = round(min(0.68, _bounded_float(settlement["confidence"], default=0.55) + 0.06), 6)
            result.events.append(
                {
                    "type": "SelfRepairShadowValidationEvent",
                    "turn_index": turn_index,
                    "at": now,
                    **shadow_validation,
                }
            )
            settlements_tail = [dict(item) for item in self_state.get("settlements_tail", []) if isinstance(item, Mapping)]
            self_state["settlements_tail"] = _append_tail(settlements_tail, settlement, limit=MAX_SETTLEMENTS_TAIL)
            repair["settlement_ids"] = _string_list(
                [*repair.get("settlement_ids", []), settlement["settlement_id"]],
                limit=8,
            )
            if settlement_status == "confirmed":
                repair["confirmed_count"] = int(repair.get("confirmed_count", 0) or 0) + 1
                repair["status"] = "confirmed" if int(repair.get("confirmed_count", 0) or 0) >= 2 else "active"
            elif settlement_status == "violated":
                repair["status"] = "violated"
            elif settlement_status == "uncertain":
                repair["status"] = "uncertain"
            repair["last_settlement_turn_index"] = turn_index
            fast_row = _lookup_fast_mismatch(
                mismatch_memory_fast,
                mismatch_key=str(repair.get("source_mismatch_key", "") or ""),
            )
            traction_proposal = {
                "proposal_id": _new_id("m19_traction"),
                "source_mismatch_key": str(repair.get("source_mismatch_key", "") or "")[:160],
                "expectation_id": str(repair.get("expectation_id", "") or "")[:120],
                "target_context": target_context,
                "intervention": str(repair.get("intervention", "") or "")[:120],
                "status": settlement_status,
                "evidence_refs": list(settlement["evidence_refs"]),
                "turn_index": turn_index,
                "at": now,
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            }
            if settlement_status == "confirmed":
                traction_proposal["traction_delta"] = 0.05
                if fast_row is not None:
                    fast_row["weighted_support"] = round(max(0.0, _bounded_float(fast_row.get("weighted_support"), default=0.0) - 0.35), 6)
                    fast_row["recent_support"] = round(max(0.0, _bounded_float(fast_row.get("recent_support"), default=0.0) - 0.20), 6)
                    fast_row["last_prediction_error_proxy"] = prediction_after
                    fast_row["status"] = "resolved" if _bounded_float(fast_row.get("weighted_support"), default=0.0) <= 0.10 else "cooling"
            elif settlement_status == "violated":
                traction_proposal["traction_delta"] = -0.05
                traction_proposal["alternative_pull_delta"] = 0.06
                if fast_row is not None:
                    fast_row["weighted_support"] = round(min(4.0, _bounded_float(fast_row.get("weighted_support"), default=0.0) + 0.30), 6)
                    fast_row["recent_support"] = round(min(2.0, _bounded_float(fast_row.get("recent_support"), default=0.0) + 0.18), 6)
                    fast_row["last_prediction_error_proxy"] = prediction_after
                    fast_row["status"] = "active"
            result.events.append({"type": "SelfRepairSettlementEvent", **settlement})
            self_state["traction_proposals_tail"] = _append_tail(
                [dict(item) for item in self_state.get("traction_proposals_tail", []) if isinstance(item, Mapping)],
                traction_proposal,
                limit=MAX_TRACTION_PROPOSALS,
            )
            result.events.append({"type": "SelfRepairTractionProposalEvent", **traction_proposal})
            result.traction_proposals.append(dict(traction_proposal))

    self_state["mismatches_tail"] = mismatches_tail[-MAX_MISMATCHES_TAIL:]
    self_state["mismatch_memory_fast"] = mismatch_memory_fast[-MAX_MISMATCH_MEMORY:]
    self_state["last_prediction_error_proxy"] = round(
        _bounded_float(abs(reward_prediction_error_proxy), default=0.0),
        6,
    )
    _recompute_active_focus(self_state, turn_index=turn_index)
    for row in list(self_state.get("active_mismatch_focus_topk", [])):
        created = _maybe_create_repair_expectation(
            self_state,
            mismatch_row=row,
            now=now,
            turn_index=turn_index,
        )
        if created is not None:
            result.events.append(
                {
                    "type": "SelfRepairExpectationCreatedEvent",
                    "turn_index": turn_index,
                    "at": now,
                    **created,
                }
            )
    promotion = _promotion_candidate(
        self_state=self_state,
        self_cognition=_mapping(state.get("self_cognition")),
        now=now,
        turn_index=turn_index,
    )
    if promotion:
        result.slow_patch_proposal = promotion
        result.events.append(
            {
                "type": "SelfExpectationSlowPromotionProposalEvent",
                "turn_index": turn_index,
                "at": now,
                "reason": str(promotion.get("reason", "") or "")[:120],
                "evidence_refs": _bounded_refs(promotion.get("evidence_refs")),
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            }
        )
    state["self_expectation_state"] = self_state
    return result


def apply_m19_traction_proposals_to_m13(
    m13_state: dict[str, Any],
    proposals: list[Mapping[str, Any]],
    *,
    user_id: str,
    topic_fingerprint: str,
    turn_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_m13_drive_state(m13_state)
    events: list[dict[str, Any]] = []
    if not proposals:
        return state, events
    patterns = [dict(row) for row in state.get("path_patterns_by_action", []) if isinstance(row, Mapping)]
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        status = str(proposal.get("status", "") or "").strip()
        if status not in {"confirmed", "violated"}:
            continue
        intervention = str(proposal.get("intervention", "") or "").strip()
        primary_action = intervention_primary_action(intervention)
        proposal_id = str(proposal.get("proposal_id", "") or _new_id("m19_traction"))[:120]
        traction_delta = _bounded_float(proposal.get("traction_delta"), default=0.05)
        if status == "violated":
            traction_delta = -abs(traction_delta)
        pattern = _upsert_pattern(
            patterns,
            action=primary_action,
            user_id=user_id,
            topic_fingerprint=topic_fingerprint,
            turn_index=turn_index,
            evidence_id=proposal_id,
        )
        previous_hp = _bounded_float(pattern.get("habit_precision"), default=0.0)
        pattern["habit_precision"] = round(
            max(0.0, min(1.0, previous_hp + traction_delta)),
            6,
        )
        if status == "violated":
            alt_delta = _bounded_float(proposal.get("alternative_pull_delta"), default=0.06)
            biases = _INTERVENTION_TO_ACTION_BIASES.get(intervention, {})
            alt_actions = [action for action, delta in biases.items() if delta < 0]
            for alt_action in alt_actions[:1]:
                alt_pattern = _upsert_pattern(
                    patterns,
                    action=alt_action,
                    user_id=user_id,
                    topic_fingerprint=topic_fingerprint,
                    turn_index=turn_index,
                    evidence_id=proposal_id,
                )
                alt_hp = _bounded_float(alt_pattern.get("habit_precision"), default=0.0)
                alt_pattern["habit_precision"] = round(min(1.0, alt_hp + alt_delta), 6)
        traction = _mapping(state.get("traction_by_action"))
        traction[_traction_key(primary_action, user_id)] = round(
            _bounded_float(pattern.get("habit_precision"), default=0.0),
            6,
        )
        state["traction_by_action"] = traction
        events.append(
            {
                "type": "M13DrivePatchProposal",
                "patch_id": proposal_id,
                "target": "m13_drive_state",
                "operation": "increment" if status == "confirmed" else "adjust",
                "field_path": f"path_patterns_by_action/{primary_action}/{user_id}",
                "previous_summary": f"habit={previous_hp:.3f}",
                "new_summary": f"habit={pattern['habit_precision']:.3f}",
                "source_event_id": proposal_id,
                "reason": f"m19_self_repair_{status}",
                "confidence": 0.72 if status == "confirmed" else 0.66,
                "ttl": 8,
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            }
        )
        events.append(
            {
                "type": "M13DrivePatchCommit",
                "commit_id": _new_id("m13_commit"),
                "patch_id": proposal_id,
                "accepted": True,
                "owner": "M19SelfExpectationAdapter",
                "reason": f"m19_self_repair_{status}",
                "committed_summary": f"m19 traction {status} for {primary_action}",
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            }
        )
    _evict_patterns(patterns)
    state["path_patterns_by_action"] = patterns
    return state, events


def apply_idle_self_expectation_review(
    state: dict[str, Any],
    *,
    review_proposals: Any,
    now: int,
    turn_index: int,
) -> list[dict[str, Any]]:
    self_state = ensure_self_expectation_state(state)
    events: list[dict[str, Any]] = []
    reviews = normalize_self_expectation_review_proposals(review_proposals)
    mismatch_memory_fast = [dict(item) for item in self_state.get("mismatch_memory_fast", []) if isinstance(item, Mapping)]
    observations = [dict(item) for item in self_state.get("observations_tail", []) if isinstance(item, Mapping)]
    for review in reviews:
        target_context = str(review.get("target_context", "") or "")
        mismatch_type = _TARGET_CONTEXT_TO_MISMATCH_TYPE.get(target_context, "persona_drift")
        mismatch_key = _make_recurrence_key(target_context, mismatch_type)
        row = _lookup_fast_mismatch(mismatch_memory_fast, mismatch_key=mismatch_key)
        review_status = str(review.get("review_status", "") or "")
        if row is not None:
            if review_status in {"stale", "unsupported"}:
                row["weighted_support"] = round(max(0.0, _bounded_float(row.get("weighted_support"), default=0.0) - 0.18), 6)
                row["recent_support"] = round(max(0.0, _bounded_float(row.get("recent_support"), default=0.0) - 0.12), 6)
                row["status"] = "revoked" if review_status == "unsupported" else "cooling"
            elif review_status == "reinforced":
                row["weighted_support"] = round(min(4.0, _bounded_float(row.get("weighted_support"), default=0.0) + 0.12), 6)
                row["recent_support"] = round(min(2.0, _bounded_float(row.get("recent_support"), default=0.0) + 0.08), 6)
                row["status"] = "active"
        observations = _append_tail(
            observations,
            {
                "observation_id": _new_id("self_obs"),
                "turn_index": turn_index,
                "at": now,
                "target_context": target_context,
                "review_status": review_status,
                "source_expectation_id": str(review.get("source_expectation_id", "") or "")[:120],
                "evidence_refs": list(review.get("evidence_refs", [])),
                "engineering_proxy_label": M19_ENGINEERING_PROXY_LABEL,
            },
            limit=MAX_OBSERVATIONS_TAIL,
        )
        events.append(
            {
                "type": "SelfExpectationIdleReviewEvent",
                "turn_index": turn_index,
                "at": now,
                **review,
            }
        )
    self_state["mismatch_memory_fast"] = mismatch_memory_fast[-MAX_MISMATCH_MEMORY:]
    self_state["observations_tail"] = observations[-MAX_OBSERVATIONS_TAIL:]
    _recompute_active_focus(self_state, turn_index=turn_index)
    state["self_expectation_state"] = self_state
    return events
