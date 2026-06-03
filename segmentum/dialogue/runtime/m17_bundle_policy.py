"""Deterministic M17 bundle policy owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .m17_bundle_features import (
    BUNDLE_TRIGGER_THRESHOLD,
    ENGINEERING_PROXY_LABEL,
    MemoryEvidenceBundle,
    ScoredMemoryEvidence,
    SINGLE_TRIGGER_THRESHOLD,
    best_bundle_for_target,
    candidate_target_groups,
    scored_memory_evidence_from_mapping,
)


ALLOWED_CONSUMERS = frozenset(
    {
        "reply_policy_bias",
        "memory_consolidation_candidate",
        "memory_revision_candidate",
    }
)


@dataclass(frozen=True)
class BundleDecision:
    commit: bool
    consumer_kind: str
    bundle_id: str
    bundle_required: bool
    aggregated_support: float
    max_single_support: float
    synergy_margin: float
    trigger_threshold: float
    best_single_memory_id: str = ""
    best_single_support: float = 0.0
    best_single_counterfactual_would_trigger: bool = False
    violation_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "consumer_kind": self.consumer_kind,
            "bundle_id": self.bundle_id,
            "bundle_required": self.bundle_required,
            "aggregated_support": self.aggregated_support,
            "max_single_support": self.max_single_support,
            "synergy_margin": self.synergy_margin,
            "trigger_threshold": self.trigger_threshold,
            "best_single_memory_id": self.best_single_memory_id,
            "best_single_support": self.best_single_support,
            "best_single_counterfactual_would_trigger": self.best_single_counterfactual_would_trigger,
            "violation_codes": list(self.violation_codes),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }


@dataclass(frozen=True)
class BundleLinkageDiagnostics:
    retrieval_eligible_count: int
    bundle_linkable_count: int
    unlinked_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "retrieval_eligible_count": self.retrieval_eligible_count,
            "bundle_linkable_count": self.bundle_linkable_count,
            "unlinked_count": self.unlinked_count,
        }


def assemble_memory_evidence_bundles(
    rows: Sequence[Mapping[str, Any] | ScoredMemoryEvidence],
    *,
    allowed_expectation_ids: Sequence[str] | None = None,
) -> tuple[list[MemoryEvidenceBundle], BundleLinkageDiagnostics]:
    evidence_rows: list[ScoredMemoryEvidence] = []
    for row in rows:
        evidence_rows.append(row if isinstance(row, ScoredMemoryEvidence) else scored_memory_evidence_from_mapping(row))
    groups = candidate_target_groups(evidence_rows)
    allowed = {str(item).strip() for item in (allowed_expectation_ids or ()) if str(item).strip()}
    bundles: list[MemoryEvidenceBundle] = []
    linkable_memory_ids: set[str] = set()
    for (target_kind, target_id), members in groups.items():
        if target_kind == "expectation_id" and allowed and target_id not in allowed:
            continue
        bundle = best_bundle_for_target(target_kind=target_kind, target_id=target_id, rows=members)
        if bundle is None:
            continue
        bundles.append(bundle)
        linkable_memory_ids.update(bundle.member_memory_ids)
    bundles.sort(
        key=lambda item: (
            item.bundle_required,
            item.aggregated_support,
            item.synergy_margin,
            -item.redundancy_penalty,
            -item.contradiction_penalty,
        ),
        reverse=True,
    )
    retrieval_ids = {row.memory_id for row in evidence_rows if row.memory_id}
    diagnostics = BundleLinkageDiagnostics(
        retrieval_eligible_count=len(retrieval_ids),
        bundle_linkable_count=len(linkable_memory_ids),
        unlinked_count=max(0, len(retrieval_ids) - len(linkable_memory_ids)),
    )
    return bundles, diagnostics


def evaluate_bundle_decision(bundle: MemoryEvidenceBundle, *, consumer_kind: str) -> BundleDecision:
    violations: list[str] = []
    if consumer_kind not in ALLOWED_CONSUMERS:
        violations.append("unsupported_consumer")
    best_single_counterfactual_would_trigger = bundle.max_single_support >= SINGLE_TRIGGER_THRESHOLD
    if not bundle.bundle_required:
        violations.append("bundle_not_required")
    if best_single_counterfactual_would_trigger:
        violations.append("best_single_would_trigger")
    commit = not violations
    return BundleDecision(
        commit=commit,
        consumer_kind=consumer_kind,
        bundle_id=bundle.bundle_id,
        bundle_required=bundle.bundle_required,
        aggregated_support=bundle.aggregated_support,
        max_single_support=bundle.max_single_support,
        synergy_margin=bundle.synergy_margin,
        trigger_threshold=BUNDLE_TRIGGER_THRESHOLD,
        best_single_memory_id=bundle.member_memory_ids[0] if bundle.member_memory_ids else "",
        best_single_support=bundle.max_single_support,
        best_single_counterfactual_would_trigger=best_single_counterfactual_would_trigger,
        violation_codes=tuple(dict.fromkeys(violations)),
    )


def bundle_decision_event(
    *,
    bundle: MemoryEvidenceBundle,
    decision: BundleDecision,
    turn_index: int,
    now: int,
) -> dict[str, Any]:
    return {
        "type": "BundleDecisionEvent" if decision.commit else "BundleDecisionSuppressedEvent",
        "at": int(now),
        "turn_index": int(turn_index),
        "bundle_id": bundle.bundle_id,
        "member_memory_ids": list(bundle.member_memory_ids),
        "member_evidence_refs": list(bundle.member_evidence_refs),
        "shared_target_kind": bundle.shared_target_kind,
        "shared_target_id": bundle.shared_target_id,
        "aggregated_support": bundle.aggregated_support,
        "max_single_support": bundle.max_single_support,
        "synergy_margin": bundle.synergy_margin,
        "best_single_memory_id": decision.best_single_memory_id,
        "best_single_support": decision.best_single_support,
        "best_single_counterfactual_would_trigger": decision.best_single_counterfactual_would_trigger,
        "consumer_kind": decision.consumer_kind,
        "reason_codes": list(decision.violation_codes if decision.violation_codes else ("bundle_required",)),
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
