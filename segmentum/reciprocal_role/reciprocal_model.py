"""Durable M12.2 reciprocal role model.

The model is a bounded working account of dialogue uncertainty. It is not a
fact ledger about either person; every durable row keeps evidence refs,
confidence, uncertainty, and status.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Literal, Mapping, Sequence

from .hyperparams import (
    BAND_ORDER,
    DEFAULT_HYPERPARAMS,
    M122Hyperparams,
    lower_band,
    normalize_band,
    raise_band,
)
from .plain_language_linter import lint_text

TargetAxis = Literal["persona_about_user", "user_about_persona"]
ClaimStatus = Literal["inferred_hypothesis", "confirmed", "contradicted", "insufficient_evidence"]
GroupStatus = Literal["open", "converging", "resolved", "contradicted"]
PointStatus = Literal["open", "resolved", "unsafe", "stale"]
CandidateKind = Literal["ask_question", "clarify_self", "demonstrate_consistency", "defer", "no_action"]
ConsentRequirement = Literal["none", "soft_check", "explicit_permission"]


@dataclass(frozen=True)
class EvidenceRef:
    ref_id: str
    ref_kind: str = "evidence_quote_ref"

    @property
    def turn_id(self) -> str:
        return self.ref_id.split(":", 1)[0] if ":" in self.ref_id else ""

    @property
    def is_direct_quote(self) -> bool:
        return self.ref_kind == "evidence_quote_ref" and bool(self.ref_id)

    def to_dict(self) -> dict[str, str]:
        return {"ref_id": self.ref_id, "ref_kind": self.ref_kind}

    @classmethod
    def from_any(cls, value: object) -> "EvidenceRef":
        if isinstance(value, Mapping):
            return cls(ref_id=str(value.get("ref_id", value.get("evidence_quote_ref", ""))), ref_kind=str(value.get("ref_kind", "evidence_quote_ref")))
        return cls(ref_id=str(value or ""), ref_kind="evidence_quote_ref")


@dataclass(frozen=True)
class ReciprocalClaim:
    claim_id: str
    group_id: str
    target_axis: TargetAxis
    claim_text_internal: str
    claim_text_plain: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    confidence_band: str = "low"
    uncertainty_band: str = "high"
    status: ClaimStatus = "inferred_hypothesis"
    created_turn_id: str = ""
    updated_turn_id: str = ""
    confidence_adjustment_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "group_id": self.group_id,
            "target_axis": self.target_axis,
            "claim_text_internal": self.claim_text_internal,
            "claim_text_plain": self.claim_text_plain,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "uncertainty_band": self.uncertainty_band,
            "status": self.status,
            "created_turn_id": self.created_turn_id,
            "updated_turn_id": self.updated_turn_id,
            "confidence_adjustment_reasons": list(self.confidence_adjustment_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReciprocalClaim":
        axis = _axis(payload.get("target_axis"))
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            group_id=str(payload.get("group_id", "")),
            target_axis=axis,
            claim_text_internal=str(payload.get("claim_text_internal", "")),
            claim_text_plain=str(payload.get("claim_text_plain", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=normalize_band(payload.get("confidence_band")),
            uncertainty_band=normalize_band(payload.get("uncertainty_band"), default="high"),
            status=_claim_status(payload.get("status")),
            created_turn_id=str(payload.get("created_turn_id", "")),
            updated_turn_id=str(payload.get("updated_turn_id", "")),
            confidence_adjustment_reasons=_strings(payload.get("confidence_adjustment_reasons")),
        )


@dataclass(frozen=True)
class ReciprocalClaimGroup:
    group_id: str
    target_axis: TargetAxis
    topic_label: str
    member_claim_ids: tuple[str, ...] = ()
    status: GroupStatus = "open"
    created_turn_id: str = ""
    updated_turn_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "target_axis": self.target_axis,
            "topic_label": self.topic_label,
            "member_claim_ids": list(self.member_claim_ids),
            "status": self.status,
            "created_turn_id": self.created_turn_id,
            "updated_turn_id": self.updated_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReciprocalClaimGroup":
        return cls(
            group_id=str(payload.get("group_id", "")),
            target_axis=_axis(payload.get("target_axis")),
            topic_label=str(payload.get("topic_label", "")),
            member_claim_ids=_strings(payload.get("member_claim_ids")),
            status=_group_status(payload.get("status")),
            created_turn_id=str(payload.get("created_turn_id", "")),
            updated_turn_id=str(payload.get("updated_turn_id", "")),
        )


@dataclass(frozen=True)
class UncertaintyPoint:
    point_id: str
    target_axis: TargetAxis
    plain_question: str
    why_it_matters_internal: str
    expected_gain_band: str
    risk_band: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    status: PointStatus = "open"

    def to_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "target_axis": self.target_axis,
            "plain_question": self.plain_question,
            "why_it_matters_internal": self.why_it_matters_internal,
            "expected_gain_band": self.expected_gain_band,
            "risk_band": self.risk_band,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UncertaintyPoint":
        return cls(
            point_id=str(payload.get("point_id", "")),
            target_axis=_axis(payload.get("target_axis")),
            plain_question=str(payload.get("plain_question", "")),
            why_it_matters_internal=str(payload.get("why_it_matters_internal", "")),
            expected_gain_band=normalize_band(payload.get("expected_gain_band")),
            risk_band=normalize_band(payload.get("risk_band")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            status=_point_status(payload.get("status")),
        )


@dataclass(frozen=True)
class InformationGainCandidate:
    candidate_id: str
    kind: CandidateKind
    target_axis: TargetAxis
    plain_action: str
    expected_gain_band: str
    risk_band: str
    consent_requirement: ConsentRequirement = "none"
    evidence_refs: tuple[EvidenceRef, ...] = ()
    blocked_by_safety: bool = False
    source: str = "extractor"
    claim_id: str = ""
    topic_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "target_axis": self.target_axis,
            "plain_action": self.plain_action,
            "expected_gain_band": self.expected_gain_band,
            "risk_band": self.risk_band,
            "consent_requirement": self.consent_requirement,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "blocked_by_safety": self.blocked_by_safety,
            "source": self.source,
            "claim_id": self.claim_id,
            "topic_label": self.topic_label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "InformationGainCandidate":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            kind=_candidate_kind(payload.get("kind")),
            target_axis=_axis(payload.get("target_axis")),
            plain_action=str(payload.get("plain_action", "")),
            expected_gain_band=normalize_band(payload.get("expected_gain_band")),
            risk_band=normalize_band(payload.get("risk_band")),
            consent_requirement=_consent(payload.get("consent_requirement")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            blocked_by_safety=bool(payload.get("blocked_by_safety", False)),
            source=str(payload.get("source", "extractor")),
            claim_id=str(payload.get("claim_id", "")),
            topic_label=str(payload.get("topic_label", "")),
        )


@dataclass(frozen=True)
class ReciprocalRoleModel:
    user_id: str
    persona_label: str = DEFAULT_HYPERPARAMS.default_persona_label
    persona_about_user_claims: tuple[ReciprocalClaim, ...] = ()
    user_about_persona_claims: tuple[ReciprocalClaim, ...] = ()
    reciprocal_claim_groups: tuple[ReciprocalClaimGroup, ...] = ()
    unresolved_uncertainty_points: tuple[UncertaintyPoint, ...] = ()
    high_gain_candidates: tuple[InformationGainCandidate, ...] = ()
    safety_boundaries: tuple[str, ...] = ()
    contradiction_cooldown: int = 0
    contradiction_turn_ids: tuple[int, ...] = ()
    last_consolidated_turn_id: str = ""
    version: str = DEFAULT_HYPERPARAMS.hyperparams_version

    @classmethod
    def empty(
        cls,
        *,
        user_id: str,
        hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
    ) -> "ReciprocalRoleModel":
        return cls(user_id=user_id, persona_label=hyperparams.default_persona_label, version=hyperparams.hyperparams_version)

    def all_claims(self) -> tuple[ReciprocalClaim, ...]:
        return self.persona_about_user_claims + self.user_about_persona_claims

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "persona_label": self.persona_label,
            "persona_about_user_claims": [claim.to_dict() for claim in self.persona_about_user_claims],
            "user_about_persona_claims": [claim.to_dict() for claim in self.user_about_persona_claims],
            "reciprocal_claim_groups": [group.to_dict() for group in self.reciprocal_claim_groups],
            "unresolved_uncertainty_points": [point.to_dict() for point in self.unresolved_uncertainty_points],
            "high_gain_candidates": [candidate.to_dict() for candidate in self.high_gain_candidates],
            "safety_boundaries": list(self.safety_boundaries),
            "contradiction_cooldown": self.contradiction_cooldown,
            "contradiction_turn_ids": list(self.contradiction_turn_ids),
            "last_consolidated_turn_id": self.last_consolidated_turn_id,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReciprocalRoleModel":
        return cls(
            user_id=str(payload.get("user_id", "")),
            persona_label=str(payload.get("persona_label", DEFAULT_HYPERPARAMS.default_persona_label)),
            persona_about_user_claims=tuple(ReciprocalClaim.from_dict(item) for item in _objects(payload.get("persona_about_user_claims"))),
            user_about_persona_claims=tuple(ReciprocalClaim.from_dict(item) for item in _objects(payload.get("user_about_persona_claims"))),
            reciprocal_claim_groups=tuple(ReciprocalClaimGroup.from_dict(item) for item in _objects(payload.get("reciprocal_claim_groups"))),
            unresolved_uncertainty_points=tuple(UncertaintyPoint.from_dict(item) for item in _objects(payload.get("unresolved_uncertainty_points"))),
            high_gain_candidates=tuple(InformationGainCandidate.from_dict(item) for item in _objects(payload.get("high_gain_candidates"))),
            safety_boundaries=_strings(payload.get("safety_boundaries")),
            contradiction_cooldown=int(payload.get("contradiction_cooldown", 0) or 0),
            contradiction_turn_ids=tuple(int(item) for item in _strings(payload.get("contradiction_turn_ids"))),
            last_consolidated_turn_id=str(payload.get("last_consolidated_turn_id", "")),
            version=str(payload.get("version", DEFAULT_HYPERPARAMS.hyperparams_version)),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReciprocalRoleModel":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("ReciprocalRoleModel JSON must decode to an object")
        return cls.from_dict(payload)


def apply_model_patch(
    model: ReciprocalRoleModel,
    *,
    turn_id: str,
    turn_index: int = 0,
    group_updates: Sequence[ReciprocalClaimGroup] = (),
    claims: Sequence[ReciprocalClaim] = (),
    uncertainty_points: Sequence[UncertaintyPoint] = (),
    candidates: Sequence[InformationGainCandidate] = (),
    safety_boundaries: Sequence[str] = (),
    direct_probe_turn_ids: Sequence[str] = (),
    contradiction_triggered: bool = False,
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> ReciprocalRoleModel:
    groups = {group.group_id: group for group in model.reciprocal_claim_groups}
    for group in group_updates:
        groups[group.group_id] = group
    first_claims = list(model.persona_about_user_claims)
    second_claims = list(model.user_about_persona_claims)
    for claim in claims:
        _ensure_plain(claim)
        claim = _enforce_confidence_ceiling(claim, direct_probe_turn_ids=direct_probe_turn_ids, hyperparams=hyperparams)
        target = first_claims if claim.target_axis == "persona_about_user" else second_claims
        target.append(claim)
        group = groups.get(claim.group_id) or ReciprocalClaimGroup(
            group_id=claim.group_id,
            target_axis=claim.target_axis,
            topic_label=claim.group_id,
            created_turn_id=turn_id,
            updated_turn_id=turn_id,
        )
        members = _unique((*group.member_claim_ids, claim.claim_id))
        groups[claim.group_id] = replace(group, member_claim_ids=members, updated_turn_id=turn_id)
    all_claims = first_claims + second_claims
    groups = {gid: _resolve_group_status(group, all_claims, turn_id=turn_id) for gid, group in groups.items()}
    contradiction_turn_ids = tuple(
        turn for turn in model.contradiction_turn_ids if turn_index - turn <= hyperparams.cooldown_window_turns
    )
    cooldown = max(0, model.contradiction_cooldown - 1)
    if contradiction_triggered:
        contradiction_turn_ids = (*contradiction_turn_ids, turn_index)
        if len(contradiction_turn_ids) > hyperparams.cooldown_threshold:
            cooldown = hyperparams.cooldown_turns
            first_claims = [_downgrade_claim(claim) for claim in first_claims]
            second_claims = [_downgrade_claim(claim) for claim in second_claims]
    return replace(
        model,
        persona_about_user_claims=tuple(first_claims[-hyperparams.max_claims_per_axis:]),
        user_about_persona_claims=tuple(second_claims[-hyperparams.max_claims_per_axis:]),
        reciprocal_claim_groups=tuple(sorted(groups.values(), key=lambda item: item.group_id))[-hyperparams.max_groups_per_axis * 2:],
        unresolved_uncertainty_points=tuple((*model.unresolved_uncertainty_points, *uncertainty_points))[-hyperparams.max_uncertainty_points:],
        high_gain_candidates=tuple((*model.high_gain_candidates, *candidates))[-hyperparams.max_candidates:],
        safety_boundaries=_unique((*model.safety_boundaries, *safety_boundaries))[-hyperparams.max_boundaries:],
        contradiction_cooldown=cooldown,
        contradiction_turn_ids=contradiction_turn_ids,
        last_consolidated_turn_id=turn_id if claims or uncertainty_points or candidates else model.last_consolidated_turn_id,
    )


def promote_claim_with_evidence(model: ReciprocalRoleModel, *, group_id: str, claim_id: str, turn_id: str) -> ReciprocalRoleModel:
    first: list[ReciprocalClaim] = []
    second: list[ReciprocalClaim] = []
    for claim in model.persona_about_user_claims:
        first.append(_promoted_if_match(claim, claim_id=claim_id, turn_id=turn_id))
    for claim in model.user_about_persona_claims:
        second.append(_promoted_if_match(claim, claim_id=claim_id, turn_id=turn_id))
    all_claims = first + second
    groups = tuple(
        _resolve_group_status(group, all_claims, turn_id=turn_id) if group.group_id == group_id else group
        for group in model.reciprocal_claim_groups
    )
    return replace(model, persona_about_user_claims=tuple(first), user_about_persona_claims=tuple(second), reciprocal_claim_groups=groups)


def mark_group_contradicted(
    model: ReciprocalRoleModel,
    *,
    group_id: str,
    turn_id: str,
    turn_index: int = 0,
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> ReciprocalRoleModel:
    def mark(claim: ReciprocalClaim) -> ReciprocalClaim:
        if claim.group_id != group_id:
            return claim
        return replace(claim, status="contradicted", confidence_band=lower_band(claim.confidence_band), updated_turn_id=turn_id)

    return apply_model_patch(
        replace(
            model,
            persona_about_user_claims=tuple(mark(claim) for claim in model.persona_about_user_claims),
            user_about_persona_claims=tuple(mark(claim) for claim in model.user_about_persona_claims),
        ),
        turn_id=turn_id,
        turn_index=turn_index,
        contradiction_triggered=True,
        hyperparams=hyperparams,
    )


def _promoted_if_match(claim: ReciprocalClaim, *, claim_id: str, turn_id: str) -> ReciprocalClaim:
    if claim.claim_id != claim_id:
        return claim
    return replace(claim, confidence_band=raise_band(claim.confidence_band), updated_turn_id=turn_id)


def _resolve_group_status(group: ReciprocalClaimGroup, claims: Sequence[ReciprocalClaim], *, turn_id: str) -> ReciprocalClaimGroup:
    members = [claim for claim in claims if claim.claim_id in set(group.member_claim_ids)]
    if not members:
        return group
    if all(claim.status == "contradicted" for claim in members):
        return replace(group, status="contradicted", updated_turn_id=turn_id)
    confirmed = [claim for claim in members if claim.status == "confirmed"]
    if confirmed:
        return replace(group, status="resolved", updated_turn_id=turn_id)
    levels = sorted((BAND_ORDER.get(claim.confidence_band, 0) for claim in members), reverse=True)
    if len(levels) >= 2 and levels[0] >= levels[1] + 1:
        return replace(group, status="converging", updated_turn_id=turn_id)
    return replace(group, status="open", updated_turn_id=turn_id)


def _enforce_confidence_ceiling(
    claim: ReciprocalClaim,
    *,
    direct_probe_turn_ids: Sequence[str],
    hyperparams: M122Hyperparams,
) -> ReciprocalClaim:
    reasons = list(claim.confidence_adjustment_reasons)
    band = normalize_band(claim.confidence_band)
    if claim.target_axis == "persona_about_user" and band == "high" and not any(ref.is_direct_quote for ref in claim.evidence_refs):
        band = "medium"
        reasons.append("high_first_order_requires_direct_user_quote")
    if claim.target_axis == "user_about_persona":
        direct_ref_turns = {ref.turn_id for ref in claim.evidence_refs if ref.is_direct_quote}
        has_recent_probe = bool(direct_ref_turns & set(direct_probe_turn_ids))
        if band == "high" and not has_recent_probe:
            band = "medium"
            reasons.append("high_second_order_requires_recent_direct_probe")
        if not claim.evidence_refs and band in {"medium", "high"}:
            band = "low"
            reasons.append("second_order_without_quote_capped_low")
    return replace(claim, confidence_band=band, confidence_adjustment_reasons=tuple(reasons))


def _ensure_plain(claim: ReciprocalClaim) -> None:
    findings = lint_text(claim.claim_text_plain, section="claim_text_plain")
    if findings:
        tokens = ", ".join(finding.token for finding in findings)
        raise ValueError(f"M12.2 claim_text_plain failed plain-language lint: {tokens}")


def _downgrade_claim(claim: ReciprocalClaim) -> ReciprocalClaim:
    return replace(
        claim,
        confidence_band=lower_band(claim.confidence_band),
        confidence_adjustment_reasons=(*claim.confidence_adjustment_reasons, "contradiction_cooldown_downgrade"),
    )


def _axis(value: object) -> TargetAxis:
    text = str(value or "persona_about_user")
    return text if text in {"persona_about_user", "user_about_persona"} else "persona_about_user"  # type: ignore[return-value]


def _claim_status(value: object) -> ClaimStatus:
    text = str(value or "inferred_hypothesis")
    return text if text in {"inferred_hypothesis", "confirmed", "contradicted", "insufficient_evidence"} else "inferred_hypothesis"  # type: ignore[return-value]


def _group_status(value: object) -> GroupStatus:
    text = str(value or "open")
    return text if text in {"open", "converging", "resolved", "contradicted"} else "open"  # type: ignore[return-value]


def _point_status(value: object) -> PointStatus:
    text = str(value or "open")
    return text if text in {"open", "resolved", "unsafe", "stale"} else "open"  # type: ignore[return-value]


def _candidate_kind(value: object) -> CandidateKind:
    text = str(value or "no_action")
    return text if text in {"ask_question", "clarify_self", "demonstrate_consistency", "defer", "no_action"} else "no_action"  # type: ignore[return-value]


def _consent(value: object) -> ConsentRequirement:
    text = str(value or "none")
    return text if text in {"none", "soft_check", "explicit_permission"} else "none"  # type: ignore[return-value]


def _refs(value: object) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(EvidenceRef.from_any(item) for item in value if str(item or "").strip() or isinstance(item, Mapping))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)
