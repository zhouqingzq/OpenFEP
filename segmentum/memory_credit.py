from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return ()


@dataclass(frozen=True)
class MemoryCreditSignal:
    linked_prediction_id: str
    linked_memory_ids: tuple[str, ...]
    linked_path_ids: tuple[str, ...]
    outcome: str
    support_score: float
    contradiction_score: float
    prediction_error_delta: float
    free_energy_delta: float
    confidence_weight: float
    source_module: str
    settlement_version: int = 1

    @property
    def application_keys(self) -> tuple[str, ...]:
        return tuple(
            f"{self.linked_prediction_id}:{target_id}:{self.settlement_version}"
            for target_id in [*self.linked_memory_ids, *self.linked_path_ids]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "linked_prediction_id": self.linked_prediction_id,
            "linked_memory_ids": list(self.linked_memory_ids),
            "linked_path_ids": list(self.linked_path_ids),
            "outcome": self.outcome,
            "support_score": round(self.support_score, 6),
            "contradiction_score": round(self.contradiction_score, 6),
            "prediction_error_delta": round(self.prediction_error_delta, 6),
            "free_energy_delta": round(self.free_energy_delta, 6),
            "confidence_weight": round(self.confidence_weight, 6),
            "source_module": self.source_module,
            "settlement_version": int(self.settlement_version),
            "application_keys": list(self.application_keys),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "MemoryCreditSignal":
        if not payload:
            return cls(
                linked_prediction_id="",
                linked_memory_ids=(),
                linked_path_ids=(),
                outcome="unclear",
                support_score=0.0,
                contradiction_score=0.0,
                prediction_error_delta=0.0,
                free_energy_delta=0.0,
                confidence_weight=0.0,
                source_module="",
            )
        return cls(
            linked_prediction_id=str(payload.get("linked_prediction_id", "")),
            linked_memory_ids=_string_tuple(payload.get("linked_memory_ids", ())),
            linked_path_ids=_string_tuple(payload.get("linked_path_ids", ())),
            outcome=str(payload.get("outcome", "unclear")),
            support_score=float(payload.get("support_score", 0.0)),
            contradiction_score=float(payload.get("contradiction_score", 0.0)),
            prediction_error_delta=float(payload.get("prediction_error_delta", 0.0)),
            free_energy_delta=float(payload.get("free_energy_delta", 0.0)),
            confidence_weight=float(payload.get("confidence_weight", 0.0)),
            source_module=str(payload.get("source_module", "")),
            settlement_version=max(1, int(payload.get("settlement_version", 1))),
        )


def build_memory_credit_signal(
    *,
    prediction_id: str,
    semantic_provenance: Mapping[str, object] | None,
    outcome: str,
    support_score: float,
    contradiction_score: float,
    confidence_weight: float,
    source_module: str,
) -> MemoryCreditSignal | None:
    provenance = dict(semantic_provenance or {})
    linked_memory_ids = _string_tuple(
        provenance.get("committed_memory_ids") or provenance.get("linked_memory_ids")
    )
    linked_path_ids = _string_tuple(provenance.get("linked_path_ids"))
    if not linked_memory_ids and not linked_path_ids:
        return None
    bounded_support = _clamp(support_score)
    bounded_contradiction = _clamp(contradiction_score)
    bounded_confidence = _clamp(confidence_weight)
    prediction_error_delta = (
        (bounded_support - bounded_contradiction) * max(0.05, bounded_confidence)
    )
    normalized_outcome = {
        "confirmed": "confirmed",
        "falsified": "violated",
        "contradicted_by_new_evidence": "violated",
        "partially_supported": "partial",
        "inconclusive": "unclear",
        "expired_unverified": "expired",
    }.get(str(outcome), str(outcome or "unclear"))
    return MemoryCreditSignal(
        linked_prediction_id=prediction_id,
        linked_memory_ids=linked_memory_ids,
        linked_path_ids=linked_path_ids,
        outcome=normalized_outcome,
        support_score=bounded_support,
        contradiction_score=bounded_contradiction,
        prediction_error_delta=prediction_error_delta,
        free_energy_delta=prediction_error_delta,
        confidence_weight=bounded_confidence,
        source_module=source_module,
    )
