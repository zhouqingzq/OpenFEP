"""Cross-user retelling as free-energy reduction over social predictions.

Sharing desire is modeled as a cognitive expectation: "if I tell B this, B
will react in some anticipated way." Retelling is allowed when testing or using
that expectation is expected to reduce enough free energy, unless the source
user declared a boundary that dominates the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams

Shareability = Literal["default_social", "restricted_explicit", "restricted_implicit"]
BoundaryStrength = Literal["none", "soft", "hard"]
SharingIntent = Literal["none", "social_share", "protective_withhold", "abstract_reference"]
SharingAction = Literal["direct_share", "abstract_reference", "withhold", "explain_or_reframe"]
ExpectedAudienceReaction = Literal["neutral", "surprised", "amused", "happy", "bonding", "empathic", "approving", "envying", "congratulatory"]
ExpectationStatus = Literal["unverified", "verified", "violated", "incomprehensible"]

VALID_SHAREABILITY = {"default_social", "restricted_explicit", "restricted_implicit"}
VALID_SHARING_INTENT = {"none", "social_share", "protective_withhold", "abstract_reference"}
VALID_EXPECTED_REACTIONS = {
    "neutral",
    "surprised",
    "amused",
    "happy",
    "bonding",
    "empathic",
    "approving",
    "envying",
    "congratulatory",
}
VALID_EXPECTATION_STATUSES = {"unverified", "verified", "violated", "incomprehensible"}

EXPLICIT_SECRECY_MARKERS = (
    "秘密",
    "别告诉别人",
    "不要告诉别人",
    "仅你知道",
    "你别说出去",
    "别外传",
    "保密",
)

NEGATIVE_SHARING_FEEDBACK_MARKERS = (
    "不该说",
    "别说了",
    "别提",
    "泄露",
    "隐私",
    "不舒服",
    "冒犯",
    "生气",
    "尴尬",
    "越界",
)


@dataclass(frozen=True)
class SocialSharingCandidate:
    memory_id: str
    source_user_id: str
    audience_user_id: str
    content_kind: str = "memory"
    shareability: Shareability = "default_social"
    boundary_strength: BoundaryStrength = "none"
    source_display_name: str = ""
    expected_audience_reaction: ExpectedAudienceReaction = "neutral"
    expectation_status: ExpectationStatus = "unverified"

    @property
    def is_cross_user(self) -> bool:
        return bool(
            self.source_user_id
            and self.audience_user_id
            and self.source_user_id != self.audience_user_id
        )


@dataclass(frozen=True)
class SocialSharingDecision:
    action: SharingAction
    current_free_energy: float
    expected_free_energy_after: float
    expected_free_energy_reduction: float
    boundary_cost: float
    relationship_cost: float
    regret_bias: float
    net_free_energy_reduction: float
    explanation_strategy: str
    reasons: tuple[str, ...]

    @property
    def allow_direct_disclosure(self) -> bool:
        return self.action == "direct_share"

    @property
    def allow_abstract_sharing(self) -> bool:
        return self.action in {"direct_share", "abstract_reference"}

    @property
    def abstract_only(self) -> bool:
        return self.action == "abstract_reference"

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "current_free_energy": self.current_free_energy,
            "expected_free_energy_after": self.expected_free_energy_after,
            "expected_free_energy_reduction": self.expected_free_energy_reduction,
            "boundary_cost": self.boundary_cost,
            "relationship_cost": self.relationship_cost,
            "regret_bias": self.regret_bias,
            "net_free_energy_reduction": self.net_free_energy_reduction,
            "allow_direct_disclosure": self.allow_direct_disclosure,
            "allow_abstract_sharing": self.allow_abstract_sharing,
            "abstract_only": self.abstract_only,
            "explanation_strategy": self.explanation_strategy,
            "reasons": list(self.reasons),
        }


def detect_explicit_secrecy(text: str) -> tuple[bool, str]:
    raw = str(text or "")
    lowered = raw.casefold()
    for marker in EXPLICIT_SECRECY_MARKERS:
        if marker.casefold() in lowered:
            return True, marker
    return False, ""


def sharing_feedback_negative(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.casefold()
    return any(marker.casefold() in lowered for marker in NEGATIVE_SHARING_FEEDBACK_MARKERS)


def memory_shareability(item: Mapping[str, object]) -> Shareability:
    shareability = str(item.get("shareability", "")).strip()
    if shareability in VALID_SHAREABILITY:
        return shareability  # type: ignore[return-value]
    visible = str(item.get("visibility", "")).strip()
    if visible in {"forbidden", "private"}:
        return "restricted_explicit"
    return "default_social"


def boundary_strength_from_constraints(
    constraints: Sequence[Mapping[str, object]],
    *,
    explicit_secrecy: bool = False,
    shareability: Shareability = "default_social",
) -> BoundaryStrength:
    if explicit_secrecy or shareability == "restricted_explicit":
        return "hard"
    strength: BoundaryStrength = "none"
    for item in constraints:
        raw = str(item.get("strength", "")).strip()
        if raw == "hard":
            return "hard"
        if raw == "soft":
            strength = "soft"
    if shareability == "restricted_implicit":
        return "soft"
    return strength


def abstract_memory_content(item: Mapping[str, object], *, max_chars: int = 80) -> str:
    kind = str(item.get("kind", "memory")).strip() or "memory"
    return f"我之前听过一个和{kind}有关的类似情况。"[:max_chars]


def candidate_from_memory(
    item: Mapping[str, object],
    *,
    audience_user_id: str,
    boundary_strength: BoundaryStrength | None = None,
) -> SocialSharingCandidate:
    shareability = memory_shareability(item)
    return SocialSharingCandidate(
        memory_id=str(item.get("id", "")).strip(),
        source_user_id=str(item.get("source_user_id", "")).strip(),
        audience_user_id=str(audience_user_id or "").strip(),
        content_kind=str(item.get("kind", "memory")).strip() or "memory",
        source_display_name=str(item.get("source_display_name", "")).strip(),
        expected_audience_reaction=_expected_reaction(item.get("expected_audience_reaction")),
        expectation_status=_expectation_status(item.get("expectation_status")),
        shareability=shareability,
        boundary_strength=boundary_strength or boundary_strength_from_constraints(
            (),
            shareability=shareability,
        ),
    )


def decide_social_sharing(
    candidate: SocialSharingCandidate,
    *,
    sharing_intent: SharingIntent = "none",
    regret_bias: float = 0.0,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> SocialSharingDecision:
    intent = sharing_intent if sharing_intent in VALID_SHARING_INTENT else "none"
    regret = _clamp(regret_bias)
    reasons: list[str] = []
    current_fe = _reaction_expectation_free_energy(candidate.expectation_status, hyperparams)
    boundary_cost = _boundary_cost(candidate, hyperparams)
    relationship_cost = _clamp(hyperparams.social_share_relationship_cost_base + regret)

    if candidate.expectation_status == "incomprehensible":
        reasons.append("reaction_expectation_incomprehensible")
        return _decision(
            "explain_or_reframe",
            current_fe,
            current_fe,
            boundary_cost,
            relationship_cost,
            regret,
            reasons,
            explanation_strategy="explain_first_then_rebuild_self_model_if_unresolved",
            hyperparams=hyperparams,
        )

    if not candidate.is_cross_user:
        reasons.append("same_user_or_unowned_memory")
        return _decision(
            "direct_share",
            current_fe,
            current_fe * (1.0 - hyperparams.direct_share_resolution_ratio),
            0.0,
            0.0,
            regret,
            reasons,
            explanation_strategy="same_user_recall",
            hyperparams=hyperparams,
        )

    if intent == "protective_withhold" or candidate.boundary_strength == "hard":
        reasons.append("protective_or_hard_boundary")
        return _decision(
            "withhold",
            current_fe,
            current_fe,
            boundary_cost,
            relationship_cost,
            regret,
            reasons,
            explanation_strategy="respect_source_boundary",
            hyperparams=hyperparams,
        )

    if candidate.boundary_strength == "soft":
        reasons.append("source_declared_soft_boundary")
        abstract_after = current_fe * (1.0 - hyperparams.abstract_share_resolution_ratio)
        abstract_boundary_cost = boundary_cost * hyperparams.abstract_boundary_cost_ratio
        return _decision(
            "abstract_reference" if _net_reduction(current_fe, abstract_after, abstract_boundary_cost, relationship_cost) > hyperparams.abstract_share_fe_reduction_threshold else "withhold",
            current_fe,
            abstract_after,
            abstract_boundary_cost,
            relationship_cost,
            regret,
            reasons,
            explanation_strategy="minimize_identifying_detail",
            hyperparams=hyperparams,
        )

    reasons.append("source_declared_no_boundary_default_social")
    after_direct = current_fe * (1.0 - hyperparams.direct_share_resolution_ratio)
    direct_net = _net_reduction(current_fe, after_direct, boundary_cost, relationship_cost)
    if direct_net > hyperparams.direct_share_fe_reduction_threshold:
        return _decision(
            "direct_share",
            current_fe,
            after_direct,
            boundary_cost,
            relationship_cost,
            regret,
            reasons,
            explanation_strategy=f"test_or_use_reaction_expectation:{candidate.expected_audience_reaction}",
            hyperparams=hyperparams,
        )

    after_abstract = current_fe * (1.0 - hyperparams.abstract_share_resolution_ratio)
    abstract_boundary_cost = boundary_cost * hyperparams.abstract_boundary_cost_ratio
    abstract_net = _net_reduction(current_fe, after_abstract, abstract_boundary_cost, relationship_cost)
    if abstract_net > hyperparams.abstract_share_fe_reduction_threshold:
        reasons.append("abstract_reduces_some_uncertainty")
        return _decision(
            "abstract_reference",
            current_fe,
            after_abstract,
            abstract_boundary_cost,
            relationship_cost,
            regret,
            reasons,
            explanation_strategy="test_reaction_without_identifying_detail",
            hyperparams=hyperparams,
        )

    reasons.append("expected_free_energy_reduction_not_enough")
    return _decision(
        "withhold",
        current_fe,
        current_fe,
        boundary_cost,
        relationship_cost,
        regret,
        reasons,
        explanation_strategy="retain_until_better_prediction",
        hyperparams=hyperparams,
    )


def update_regret_bias(
    *,
    previous_regret_bias: float,
    negative_feedback: bool,
    had_cross_user_share: bool,
    same_audience_user: bool,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> float:
    regret = _clamp(previous_regret_bias)
    if had_cross_user_share and same_audience_user and negative_feedback:
        regret += hyperparams.sharing_regret_feedback_increment
    else:
        regret -= hyperparams.sharing_regret_feedback_decay
    return round(_clamp(regret), hyperparams.float_round_digits)


def _reaction_expectation_free_energy(
    status: ExpectationStatus,
    hyperparams: Hyperparams,
) -> float:
    if status == "verified":
        return hyperparams.reaction_expectation_verified_fe
    if status == "violated":
        return hyperparams.reaction_expectation_violated_fe
    if status == "incomprehensible":
        return hyperparams.reaction_expectation_incomprehensible_fe
    return hyperparams.reaction_expectation_unverified_fe


def _boundary_cost(candidate: SocialSharingCandidate, hyperparams: Hyperparams) -> float:
    if candidate.boundary_strength == "hard" or candidate.shareability == "restricted_explicit":
        return hyperparams.restricted_explicit_boundary_cost
    if candidate.boundary_strength == "soft" or candidate.shareability == "restricted_implicit":
        return hyperparams.restricted_implicit_boundary_cost
    return hyperparams.default_social_boundary_cost


def _net_reduction(
    current_fe: float,
    expected_fe_after: float,
    boundary_cost: float,
    relationship_cost: float,
) -> float:
    return current_fe - expected_fe_after - boundary_cost - relationship_cost


def _decision(
    action: SharingAction,
    current_fe: float,
    expected_fe_after: float,
    boundary_cost: float,
    relationship_cost: float,
    regret: float,
    reasons: list[str],
    *,
    explanation_strategy: str,
    hyperparams: Hyperparams,
) -> SocialSharingDecision:
    digits = hyperparams.float_round_digits
    reduction = current_fe - expected_fe_after
    net = _net_reduction(current_fe, expected_fe_after, boundary_cost, relationship_cost)
    return SocialSharingDecision(
        action=action,
        current_free_energy=round(_clamp(current_fe), digits),
        expected_free_energy_after=round(_clamp(expected_fe_after), digits),
        expected_free_energy_reduction=round(_clamp(reduction), digits),
        boundary_cost=round(_clamp(boundary_cost), digits),
        relationship_cost=round(_clamp(relationship_cost), digits),
        regret_bias=round(_clamp(regret), digits),
        net_free_energy_reduction=round(_clamp_signed(net), digits),
        explanation_strategy=explanation_strategy,
        reasons=tuple(reasons),
    )


def _expected_reaction(value: object) -> ExpectedAudienceReaction:
    text = str(value or "neutral")
    return text if text in VALID_EXPECTED_REACTIONS else "neutral"  # type: ignore[return-value]


def _expectation_status(value: object) -> ExpectationStatus:
    text = str(value or "unverified")
    return text if text in VALID_EXPECTATION_STATUSES else "unverified"  # type: ignore[return-value]


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _clamp_signed(value: float) -> float:
    return min(max(float(value), -1.0), 1.0)
