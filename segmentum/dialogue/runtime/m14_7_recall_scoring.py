"""M14.7 deterministic precision- and recency-weighted recall scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Mapping
import json
import re

from segmentum.dialogue.runtime.m15_3_cleanup_control import cleanup_recall_suppression_reason


MEMORY_EFE_RECALL_FLOOR = 0.2
RECENCY_HALF_LIFE_SECONDS = 14 * 86400


@dataclass(frozen=True)
class ScoredRecallCandidate:
    memory_id: str
    score: float
    lexical_overlap_norm: float
    salience_factor: float
    precision_factor: float
    recency_factor: float
    value_factor: float
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "lexical_overlap_norm": self.lexical_overlap_norm,
            "salience_factor": self.salience_factor,
            "precision_factor": self.precision_factor,
            "recency_factor": self.recency_factor,
            "value_factor": self.value_factor,
            "evidence_refs": list(self.evidence_refs),
        }


def _bounded_float(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _epoch(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", str(text or ""))
        if token.strip()
    }


def _lexical_overlap_norm(candidate: Mapping[str, Any], query: Any) -> float:
    query_terms = _terms(" ".join(str(item) for item in query) if isinstance(query, list) else str(query or ""))
    if not query_terms:
        return 0.5
    candidate_id = str(candidate.get("id", candidate.get("expectation_id", "")) or "").strip().casefold()
    if candidate_id and candidate_id in query_terms:
        return 1.0
    text = json.dumps(dict(candidate), ensure_ascii=False)
    candidate_terms = _terms(text)
    if not candidate_terms:
        return 0.0
    return min(1.0, len(query_terms & candidate_terms) / max(1, len(query_terms)))


def score_recall_candidate(
    candidate: Mapping[str, Any],
    *,
    query: Any,
    now: int,
    retrieved_context: Mapping[str, Any] | None = None,
) -> float:
    return explain_recall_candidate(
        candidate,
        query=query,
        now=now,
        retrieved_context=retrieved_context,
    ).score


def explain_recall_candidate(
    candidate: Mapping[str, Any],
    *,
    query: Any,
    now: int,
    retrieved_context: Mapping[str, Any] | None = None,
) -> ScoredRecallCandidate:
    context = retrieved_context if isinstance(retrieved_context, Mapping) else {}
    phase = str(context.get("phase", "") or "")
    if str(candidate.get("status", "") or "") == "archived":
        return ScoredRecallCandidate(
            memory_id=str(candidate.get("id", candidate.get("expectation_id", "")) or ""),
            score=0.0,
            lexical_overlap_norm=0.0,
            salience_factor=0.0,
            precision_factor=0.0,
            recency_factor=0.0,
            value_factor=0.0,
            evidence_refs=(),
        )
    if cleanup_recall_suppression_reason(candidate, now=now, phase=phase):
        return ScoredRecallCandidate(
            memory_id=str(candidate.get("id", candidate.get("expectation_id", "")) or ""),
            score=0.0,
            lexical_overlap_norm=0.0,
            salience_factor=0.0,
            precision_factor=0.0,
            recency_factor=0.0,
            value_factor=0.0,
            evidence_refs=(),
        )
    lexical = _lexical_overlap_norm(candidate, query)
    salience = _bounded_float(candidate.get("salience"), default=0.5)
    precision = _bounded_float(candidate.get("precision", candidate.get("confidence", 0.5)), default=0.5)
    last = _epoch(candidate.get("last_recall_at") or candidate.get("last_recalled_at") or candidate.get("created_at"))
    if last <= 0 or now <= 0:
        recency = 1.0
    else:
        recency = exp(-max(0, now - last) / float(RECENCY_HALF_LIFE_SECONDS))
    value = _bounded_float(candidate.get("value_proxy", candidate.get("future_prediction_value", 0.5)), default=0.5)
    score = lexical * salience * precision * recency * value
    return ScoredRecallCandidate(
        memory_id=str(candidate.get("id", candidate.get("expectation_id", "")) or ""),
        score=round(max(0.0, min(1.0, score)), 6),
        lexical_overlap_norm=round(lexical, 6),
        salience_factor=round(salience, 6),
        precision_factor=round(precision, 6),
        recency_factor=round(recency, 6),
        value_factor=round(value, 6),
        evidence_refs=tuple(str(value) for value in candidate.get("evidence_refs", []) if str(value or "").strip()),
    )
