"""M14.7 deterministic Path B memory write gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import uuid

from .m17_bundle_features import (
    aggregate_memory_bundle_support,
    scored_memory_evidence_from_mapping,
)


ENGINEERING_PROXY_LABEL = "mvp_local_memory_gate"
MEMORY_GATE_THRESHOLD_SHORT_TERM = 0.30
MEMORY_GATE_THRESHOLD_LONG_TERM = 0.55
MAX_GATE_COMMITS_PER_SESSION_PER_PROPOSER = 8
M17_BUNDLE_TRIGGER_THRESHOLD = 0.74
M17_SINGLE_TRIGGER_THRESHOLD = 0.60

GATE_WEIGHTS = {
    "surprise": 0.30,
    "value": 0.25,
    "identity": 0.15,
    "evidence": 0.20,
    "confidence": 0.10,
}
M17_POSITIVE_WEIGHTS = {
    "prediction_error_signal": 0.24,
    "confirmation_signal": 0.18,
    "novelty_signal": 0.10,
    "recurrence_signal": 0.12,
    "evidence_factor": 0.16,
    "identity_relevance": 0.10,
    "confidence": 0.10,
}
M17_NEGATIVE_WEIGHTS = {
    "maintenance_cost": 0.08,
    "contradiction_risk": 0.12,
}

VIOLATION_CODES = frozenset(
    {
        "missing_evidence_refs",
        "content_too_short",
        "content_too_long",
        "confidence_below_floor",
        "duplicate_of_recent_episode",
        "gate_score_below_threshold",
        "proposer_session_cap_reached",
    }
)


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _string_list(raw: Any, *, limit: int = 8) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [raw]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text[:120])
            if len(result) >= limit:
                break
    return result


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def memory_intent_fingerprint(intent: "MemoryWriteIntent") -> str:
    payload = "|".join(
        [
            str(intent.proposer or ""),
            str(intent.source or ""),
            str(intent.target or ""),
            str(intent.kind or ""),
            " ".join(str(intent.content or "").strip().casefold().split()),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class MemoryWriteIntent:
    target: str
    kind: str
    content: str
    confidence: float
    evidence_refs: list[str]
    identity_relevance: float = 0.0
    value_proxy: float = 0.0
    surprise_proxy: float = 0.0
    source: str = ""
    proposer: str = ""
    audit_reason: str = ""
    prediction_error_signal: float = 0.0
    confirmation_signal: float = 0.0
    novelty_signal: float = 0.0
    recurrence_signal: float = 0.0
    maintenance_cost: float = 0.0
    contradiction_risk: float = 0.0
    linked_prediction_ids: list[str] = field(default_factory=list)
    bundle_id: str = ""
    member_memory_ids: list[str] = field(default_factory=list)
    member_evidence_refs: list[str] = field(default_factory=list)
    aggregated_support: float = 0.0
    max_single_support: float = 0.0
    synergy_margin: float = 0.0
    bundle_required: bool = False
    unique_memory_count: int = 0
    unique_evidence_ref_count: int = 0
    intent_id: str = field(default_factory=lambda: _new_id("mem_gate_intent"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "target": self.target,
            "kind": self.kind,
            "content": self.content[:240],
            "confidence": round(self.confidence, 6),
            "evidence_refs": list(self.evidence_refs[:8]),
            "identity_relevance": round(self.identity_relevance, 6),
            "value_proxy": round(self.value_proxy, 6),
            "surprise_proxy": round(self.surprise_proxy, 6),
            "source": self.source,
            "proposer": self.proposer,
            "audit_reason": self.audit_reason,
            "prediction_error_signal": round(self.prediction_error_signal, 6),
            "confirmation_signal": round(self.confirmation_signal, 6),
            "novelty_signal": round(self.novelty_signal, 6),
            "recurrence_signal": round(self.recurrence_signal, 6),
            "maintenance_cost": round(self.maintenance_cost, 6),
            "contradiction_risk": round(self.contradiction_risk, 6),
            "linked_prediction_ids": list(self.linked_prediction_ids[:8]),
            "bundle_id": self.bundle_id,
            "member_memory_ids": list(self.member_memory_ids[:8]),
            "member_evidence_refs": list(self.member_evidence_refs[:8]),
            "aggregated_support": round(self.aggregated_support, 6),
            "max_single_support": round(self.max_single_support, 6),
            "synergy_margin": round(self.synergy_margin, 6),
            "bundle_required": self.bundle_required,
            "unique_memory_count": self.unique_memory_count,
            "unique_evidence_ref_count": self.unique_evidence_ref_count,
            "intent_fingerprint": memory_intent_fingerprint(self),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }


@dataclass(frozen=True)
class MemoryGateDecision:
    commit: bool
    write_score: float
    threshold: float
    factors: dict[str, float]
    violation_codes: list[str] = field(default_factory=list)
    policy_profile: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "write_score": round(self.write_score, 6),
            "threshold": round(self.threshold, 6),
            "factors": dict(self.factors),
            "violation_codes": list(self.violation_codes),
            "policy_profile": self.policy_profile,
        }


class MemoryGate:
    def __init__(self, *, short_threshold: float = MEMORY_GATE_THRESHOLD_SHORT_TERM, long_threshold: float = MEMORY_GATE_THRESHOLD_LONG_TERM) -> None:
        self.short_threshold = float(short_threshold)
        self.long_threshold = float(long_threshold)

    def evaluate(
        self,
        intent: MemoryWriteIntent,
        *,
        proposer_commits_this_session: int = 0,
        recent_intent_fingerprints: set[str] | None = None,
        policy_profile: str = "legacy",
    ) -> MemoryGateDecision:
        target = str(intent.target or "short_term")
        threshold = self.long_threshold if target == "long_term" else self.short_threshold
        evidence_factor = min(1.0, len(intent.evidence_refs) / 2.0)
        factors = {
            "surprise_proxy": _bounded_float(intent.surprise_proxy),
            "value_proxy": _bounded_float(intent.value_proxy),
            "identity_relevance": _bounded_float(intent.identity_relevance),
            "evidence_factor": round(evidence_factor, 6),
            "confidence": _bounded_float(intent.confidence),
            "prediction_error_signal": _bounded_float(intent.prediction_error_signal),
            "confirmation_signal": _bounded_float(intent.confirmation_signal),
            "novelty_signal": _bounded_float(intent.novelty_signal),
            "recurrence_signal": _bounded_float(intent.recurrence_signal),
            "maintenance_cost": _bounded_float(intent.maintenance_cost),
            "contradiction_risk": _bounded_float(intent.contradiction_risk),
            "aggregated_support": _bounded_float(intent.aggregated_support),
            "max_single_support": _bounded_float(intent.max_single_support),
            "synergy_margin": _bounded_float(intent.synergy_margin),
        }
        legacy_score = (
            GATE_WEIGHTS["surprise"] * factors["surprise_proxy"]
            + GATE_WEIGHTS["value"] * factors["value_proxy"]
            + GATE_WEIGHTS["identity"] * factors["identity_relevance"]
            + GATE_WEIGHTS["evidence"] * factors["evidence_factor"]
            + GATE_WEIGHTS["confidence"] * factors["confidence"]
        )
        score = legacy_score
        if policy_profile == "m17_observe_only":
            # Preserve legacy write behavior while still exposing M17 sidecar factors.
            score = legacy_score
        elif policy_profile == "m17_blended":
            score = (
                sum(M17_POSITIVE_WEIGHTS[key] * factors[key] for key in M17_POSITIVE_WEIGHTS)
                - sum(M17_NEGATIVE_WEIGHTS[key] * factors[key] for key in M17_NEGATIVE_WEIGHTS)
            )
        violations: list[str] = []
        content_len = len(str(intent.content or "").strip())
        if not intent.evidence_refs:
            violations.append("missing_evidence_refs")
        if content_len < 4:
            violations.append("content_too_short")
        if content_len > 400:
            violations.append("content_too_long")
        if intent.confidence < 0.35:
            violations.append("confidence_below_floor")
        if memory_intent_fingerprint(intent) in (recent_intent_fingerprints or set()):
            violations.append("duplicate_of_recent_episode")
        if proposer_commits_this_session >= MAX_GATE_COMMITS_PER_SESSION_PER_PROPOSER:
            violations.append("proposer_session_cap_reached")
        if score < threshold:
            violations.append("gate_score_below_threshold")
        if (
            intent.bundle_required
            and (
                factors["aggregated_support"] < M17_BUNDLE_TRIGGER_THRESHOLD
                or factors["max_single_support"] >= M17_SINGLE_TRIGGER_THRESHOLD
                or int(intent.unique_memory_count) < 2
                or int(intent.unique_evidence_ref_count) < 2
            )
        ):
            violations.append("bundle_support_contract_failed")
        return MemoryGateDecision(
            commit=not violations,
            write_score=round(score, 6),
            threshold=threshold,
            factors=factors,
            violation_codes=list(dict.fromkeys(violations)),
            policy_profile=policy_profile,
        )


def memory_gate_event(
    *,
    event_type: str,
    intent: MemoryWriteIntent,
    decision: MemoryGateDecision,
    turn_index: int,
    now: int,
    store_target: str = "",
    store_id: str = "",
) -> dict[str, Any]:
    payload = {
        "type": event_type,
        "at": now,
        "turn_index": turn_index,
        "source": intent.source,
        "proposer": intent.proposer,
        "intent_summary": intent.content[:160],
        "intent_fingerprint": memory_intent_fingerprint(intent),
        "write_score": decision.write_score,
        "threshold": decision.threshold,
        "factors": dict(decision.factors),
        "violation_codes": list(decision.violation_codes),
        "policy_profile": decision.policy_profile,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
    if store_target:
        payload["store_target"] = store_target
    if store_id:
        payload["store_id"] = store_id
    return payload


def intent_from_mapping(
    row: Mapping[str, Any],
    *,
    target: str,
    kind: str,
    content: str,
    confidence: float,
    evidence_refs: list[str],
    source: str,
    proposer: str,
    audit_reason: str,
    identity_relevance: float = 0.0,
    value_proxy: float = 0.0,
    surprise_proxy: float = 0.0,
    prediction_error_signal: float = 0.0,
    confirmation_signal: float = 0.0,
    novelty_signal: float = 0.0,
    recurrence_signal: float = 0.0,
    maintenance_cost: float = 0.0,
    contradiction_risk: float = 0.0,
    linked_prediction_ids: list[str] | None = None,
    bundle_id: str = "",
    member_memory_ids: list[str] | None = None,
    member_evidence_refs: list[str] | None = None,
    aggregated_support: float = 0.0,
    max_single_support: float = 0.0,
    synergy_margin: float = 0.0,
    bundle_required: bool = False,
    unique_memory_count: int = 0,
    unique_evidence_ref_count: int = 0,
) -> MemoryWriteIntent:
    del row
    return MemoryWriteIntent(
        target=target if target in {"short_term", "long_term"} else "short_term",
        kind=kind or "episode",
        content=str(content or "").strip()[:400],
        confidence=_bounded_float(confidence),
        evidence_refs=_string_list(evidence_refs, limit=8),
        identity_relevance=_bounded_float(identity_relevance),
        value_proxy=_bounded_float(value_proxy),
        surprise_proxy=_bounded_float(surprise_proxy),
        source=source,
        proposer=proposer,
        audit_reason=audit_reason,
        prediction_error_signal=_bounded_float(prediction_error_signal),
        confirmation_signal=_bounded_float(confirmation_signal),
        novelty_signal=_bounded_float(novelty_signal),
        recurrence_signal=_bounded_float(recurrence_signal),
        maintenance_cost=_bounded_float(maintenance_cost),
        contradiction_risk=_bounded_float(contradiction_risk),
        linked_prediction_ids=_string_list(linked_prediction_ids, limit=8),
        bundle_id=str(bundle_id or "")[:120],
        member_memory_ids=_string_list(member_memory_ids, limit=8),
        member_evidence_refs=_string_list(member_evidence_refs, limit=8),
        aggregated_support=_bounded_float(aggregated_support),
        max_single_support=_bounded_float(max_single_support),
        synergy_margin=_bounded_float(synergy_margin),
        bundle_required=bool(bundle_required),
        unique_memory_count=max(int(unique_memory_count or 0), 0),
        unique_evidence_ref_count=max(int(unique_evidence_ref_count or 0), 0),
    )


def memory_gate_signals_from_prediction_settlement(
    *,
    settlement_outcome: str,
    committed_confidence: float,
    prediction_error: float | None = None,
    confirmation_count: int = 0,
    novelty_signal: float = 0.0,
    recurrence_signal: float = 0.0,
    maintenance_cost: float = 0.0,
    contradiction_risk: float = 0.0,
) -> dict[str, float]:
    outcome = str(settlement_outcome or "").strip().casefold()
    error_signal = 0.0
    confirmation_signal = 0.0
    if outcome == "violated":
        source_error = prediction_error if prediction_error is not None else max(0.0, float(committed_confidence))
        error_signal = _bounded_float(source_error / 2.0)
    elif outcome == "confirmed":
        confirmation_signal = _bounded_float(0.35 + 0.10 * max(0, int(confirmation_count)))
    return {
        "prediction_error_signal": error_signal,
        "confirmation_signal": confirmation_signal,
        "novelty_signal": _bounded_float(novelty_signal),
        "recurrence_signal": _bounded_float(recurrence_signal),
        "maintenance_cost": _bounded_float(maintenance_cost),
        "contradiction_risk": _bounded_float(contradiction_risk),
    }


def aggregate_memory_gate_bundle_support(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    summary = aggregate_memory_bundle_support(
        [scored_memory_evidence_from_mapping(row) for row in rows]
    )
    return {
        "aggregated_support": round(_bounded_float(summary.get("aggregated_support", 0.0)), 6),
        "max_single_support": round(_bounded_float(summary.get("max_single_support", 0.0)), 6),
        "synergy_margin": round(_bounded_float(summary.get("synergy_margin", 0.0)), 6),
        "bundle_required": bool(summary.get("bundle_required", False)),
        "unique_memory_count": int(summary.get("unique_memory_count", 0) or 0),
        "unique_evidence_ref_count": int(summary.get("unique_evidence_ref_count", 0) or 0),
    }
