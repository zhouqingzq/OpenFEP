"""M14.7 deterministic Path B memory write gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import uuid


ENGINEERING_PROXY_LABEL = "mvp_local_memory_gate"
MEMORY_GATE_THRESHOLD_SHORT_TERM = 0.30
MEMORY_GATE_THRESHOLD_LONG_TERM = 0.55
MAX_GATE_COMMITS_PER_SESSION_PER_PROPOSER = 8

GATE_WEIGHTS = {
    "surprise": 0.30,
    "value": 0.25,
    "identity": 0.15,
    "evidence": 0.20,
    "confidence": 0.10,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "write_score": round(self.write_score, 6),
            "threshold": round(self.threshold, 6),
            "factors": dict(self.factors),
            "violation_codes": list(self.violation_codes),
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
        }
        score = (
            GATE_WEIGHTS["surprise"] * factors["surprise_proxy"]
            + GATE_WEIGHTS["value"] * factors["value_proxy"]
            + GATE_WEIGHTS["identity"] * factors["identity_relevance"]
            + GATE_WEIGHTS["evidence"] * factors["evidence_factor"]
            + GATE_WEIGHTS["confidence"] * factors["confidence"]
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
        return MemoryGateDecision(
            commit=not violations,
            write_score=round(score, 6),
            threshold=threshold,
            factors=factors,
            violation_codes=list(dict.fromkeys(violations)),
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
    )
