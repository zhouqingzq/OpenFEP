"""Deterministic M17 bundle support features."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping


MAX_BUNDLE_SIZE = 4
SINGLE_TRIGGER_THRESHOLD = 0.60
BUNDLE_TRIGGER_THRESHOLD = 0.74
MIN_SYNERGY_MARGIN = 0.12
REDUNDANCY_PENALTY_FLOOR = 0.10
ENGINEERING_PROXY_LABEL = "mvp_local_bundle_policy"


def _bounded(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(0.0, min(1.0, parsed)), 6)


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _group_targets(row: Mapping[str, Any], *, kind: str) -> list[str]:
    singular = str(row.get(kind[:-1], "") or "").strip()
    plural = _string_list(row.get(kind), limit=8)
    combined = plural[:]
    if singular and singular not in combined:
        combined.insert(0, singular)
    return combined[:8]


@dataclass(frozen=True)
class ScoredMemoryEvidence:
    memory_id: str
    item_support: float
    evidence_refs: tuple[str, ...] = ()
    prediction_ids: tuple[str, ...] = ()
    expectation_ids: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    contradiction_risk: float = 0.0
    factor_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "item_support": self.item_support,
            "evidence_refs": list(self.evidence_refs),
            "prediction_ids": list(self.prediction_ids),
            "expectation_ids": list(self.expectation_ids),
            "episode_ids": list(self.episode_ids),
            "contradiction_risk": self.contradiction_risk,
            "factor_breakdown": dict(self.factor_breakdown),
        }


@dataclass(frozen=True)
class MemoryEvidenceBundle:
    bundle_id: str
    shared_target_kind: str
    shared_target_id: str
    member_memory_ids: tuple[str, ...]
    member_evidence_refs: tuple[str, ...]
    aggregated_support: float
    max_single_support: float
    synergy_margin: float
    redundancy_penalty: float
    contradiction_penalty: float
    unique_memory_count: int
    unique_evidence_ref_count: int
    bundle_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "shared_target_kind": self.shared_target_kind,
            "shared_target_id": self.shared_target_id,
            "member_memory_ids": list(self.member_memory_ids),
            "member_evidence_refs": list(self.member_evidence_refs),
            "aggregated_support": self.aggregated_support,
            "max_single_support": self.max_single_support,
            "synergy_margin": self.synergy_margin,
            "redundancy_penalty": self.redundancy_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "unique_memory_count": self.unique_memory_count,
            "unique_evidence_ref_count": self.unique_evidence_ref_count,
            "bundle_required": self.bundle_required,
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }


def scored_memory_evidence_from_mapping(row: Mapping[str, Any]) -> ScoredMemoryEvidence:
    return ScoredMemoryEvidence(
        memory_id=str(row.get("memory_id", row.get("id", row.get("expectation_id", ""))) or "").strip(),
        item_support=_bounded(row.get("item_support", row.get("_m17_item_support", row.get("_m14_7_recall_score", 0.0)))),
        evidence_refs=tuple(_string_list(row.get("evidence_refs", row.get("_m17_evidence_refs")), limit=8)),
        prediction_ids=tuple(_group_targets(row, kind="prediction_ids")),
        expectation_ids=tuple(_group_targets(row, kind="expectation_ids")),
        episode_ids=tuple(_group_targets(row, kind="episode_ids")),
        contradiction_risk=_bounded(row.get("contradiction_risk", row.get("_m17_contradiction_risk", 0.0))),
        factor_breakdown={
            key: _bounded(value)
            for key, value in dict(row.get("factor_breakdown", row.get("_m17_factor_breakdown", {})) or {}).items()
        },
    )


def redundancy_penalty(members: Iterable[ScoredMemoryEvidence]) -> float:
    rows = list(members)
    total_refs = [ref for row in rows for ref in row.evidence_refs]
    if len(total_refs) <= 1:
        return 0.0
    duplicate_count = len(total_refs) - len(set(total_refs))
    if duplicate_count <= 0:
        return 0.0
    duplicate_ratio = duplicate_count / float(max(1, len(total_refs)))
    return round(max(REDUNDANCY_PENALTY_FLOOR, duplicate_ratio * 0.40), 6)


def contradiction_penalty(members: Iterable[ScoredMemoryEvidence]) -> float:
    rows = list(members)
    if not rows:
        return 0.0
    max_risk = max(row.contradiction_risk for row in rows)
    mean_risk = sum(row.contradiction_risk for row in rows) / len(rows)
    return round(min(1.0, 0.60 * max_risk + 0.20 * mean_risk), 6)


def aggregate_memory_bundle_support(members: Iterable[ScoredMemoryEvidence]) -> dict[str, float | int | bool]:
    rows = list(members)
    if not rows:
        return {
            "aggregated_support": 0.0,
            "max_single_support": 0.0,
            "synergy_margin": 0.0,
            "redundancy_penalty": 0.0,
            "contradiction_penalty": 0.0,
            "unique_memory_count": 0,
            "unique_evidence_ref_count": 0,
            "bundle_required": False,
        }
    supports = sorted((_bounded(row.item_support) for row in rows), reverse=True)
    aggregate = 1.0
    for support in supports[:MAX_BUNDLE_SIZE]:
        aggregate *= 1.0 - support
    base_support = 1.0 - aggregate
    redundant = redundancy_penalty(rows)
    contradictory = contradiction_penalty(rows)
    aggregated_support = max(0.0, min(1.0, base_support - redundant - contradictory))
    max_single_support = supports[0]
    synergy_margin = max(0.0, aggregated_support - max_single_support)
    unique_memory_count = len({row.memory_id for row in rows if row.memory_id})
    unique_evidence_ref_count = len({ref for row in rows for ref in row.evidence_refs})
    bundle_required = (
        aggregated_support >= BUNDLE_TRIGGER_THRESHOLD
        and max_single_support < SINGLE_TRIGGER_THRESHOLD
        and synergy_margin >= MIN_SYNERGY_MARGIN
        and unique_memory_count >= 2
        and unique_evidence_ref_count >= 2
    )
    return {
        "aggregated_support": round(aggregated_support, 6),
        "max_single_support": round(max_single_support, 6),
        "synergy_margin": round(synergy_margin, 6),
        "redundancy_penalty": round(redundant, 6),
        "contradiction_penalty": round(contradictory, 6),
        "unique_memory_count": unique_memory_count,
        "unique_evidence_ref_count": unique_evidence_ref_count,
        "bundle_required": bundle_required,
    }


def candidate_target_groups(rows: Iterable[ScoredMemoryEvidence]) -> dict[tuple[str, str], list[ScoredMemoryEvidence]]:
    groups: dict[tuple[str, str], list[ScoredMemoryEvidence]] = {}
    for row in rows:
        for kind, targets in (
            ("prediction_id", row.prediction_ids),
            ("expectation_id", row.expectation_ids),
            ("episode_id", row.episode_ids),
        ):
            for target in targets:
                groups.setdefault((kind, target), []).append(row)
    return groups


def best_bundle_for_target(
    *,
    target_kind: str,
    target_id: str,
    rows: list[ScoredMemoryEvidence],
) -> MemoryEvidenceBundle | None:
    unique_rows: list[ScoredMemoryEvidence] = []
    seen_ids: set[str] = set()
    for row in sorted(rows, key=lambda item: item.item_support, reverse=True):
        if not row.memory_id or row.memory_id in seen_ids:
            continue
        seen_ids.add(row.memory_id)
        unique_rows.append(row)
    if len(unique_rows) < 2:
        return None
    best: MemoryEvidenceBundle | None = None
    max_size = min(MAX_BUNDLE_SIZE, len(unique_rows))
    for size in range(2, max_size + 1):
        for members in combinations(unique_rows, size):
            summary = aggregate_memory_bundle_support(members)
            bundle = MemoryEvidenceBundle(
                bundle_id=f"bundle:{target_kind}:{target_id}:{size}:{'-'.join(row.memory_id for row in members[:2])}"[:160],
                shared_target_kind=target_kind,
                shared_target_id=target_id,
                member_memory_ids=tuple(row.memory_id for row in members),
                member_evidence_refs=tuple(
                    dict.fromkeys(ref for row in members for ref in row.evidence_refs)
                )[:8],
                aggregated_support=float(summary["aggregated_support"]),
                max_single_support=float(summary["max_single_support"]),
                synergy_margin=float(summary["synergy_margin"]),
                redundancy_penalty=float(summary["redundancy_penalty"]),
                contradiction_penalty=float(summary["contradiction_penalty"]),
                unique_memory_count=int(summary["unique_memory_count"]),
                unique_evidence_ref_count=int(summary["unique_evidence_ref_count"]),
                bundle_required=bool(summary["bundle_required"]),
            )
            if best is None or (
                bundle.bundle_required,
                bundle.aggregated_support,
                bundle.synergy_margin,
            ) > (
                best.bundle_required,
                best.aggregated_support,
                best.synergy_margin,
            ):
                best = bundle
    return best
