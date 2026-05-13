"""M12.1 bounded personality profile state.

The profile is a ledger-style state object, not a factual biography.  Every
non-empty section is either an inferred hypothesis with evidence references or
an explicit insufficient-evidence marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Literal, Mapping, Sequence

from .hyperparams import (
    CONFIDENCE_BY_LEVEL,
    CONFIDENCE_ORDER,
    DEFAULT_HYPERPARAMS,
    M121Hyperparams,
    SECTION_KINDS,
)

ConfidenceBand = Literal["low", "med", "high"]
ReportStatus = Literal["draft", "linter_failed", "ready", "stale", "superseded"]
SectionKind = Literal[
    "step_1",
    "step_2",
    "step_3",
    "step_4",
    "step_5",
    "step_6",
    "step_7",
    "step_8",
]
ClaimState = Literal["inferred_hypothesis", "confirmed_by_user", "insufficient_evidence"]


@dataclass(frozen=True)
class EvidenceQuoteRef:
    turn_id: str
    quote_id: str

    @property
    def ref_id(self) -> str:
        return f"{self.turn_id}:{self.quote_id}" if self.turn_id else self.quote_id

    def to_dict(self) -> dict[str, str]:
        return {"turn_id": self.turn_id, "quote_id": self.quote_id}

    @classmethod
    def from_any(cls, value: object, *, default_turn_id: str = "") -> "EvidenceQuoteRef":
        if isinstance(value, Mapping):
            return cls(turn_id=str(value.get("turn_id", default_turn_id)), quote_id=str(value.get("quote_id", "")))
        text = str(value or "")
        if ":" in text:
            turn, quote = text.split(":", 1)
            return cls(turn_id=turn, quote_id=quote)
        return cls(turn_id=default_turn_id, quote_id=text)


@dataclass(frozen=True)
class InsufficientEvidence:
    reason: str
    evidence_refs: tuple[EvidenceQuoteRef, ...] = ()
    confidence_band: ConfidenceBand = "low"
    hyperparams_version: str = DEFAULT_HYPERPARAMS.hyperparams_version
    last_updated_turn_id: str = ""
    claim_state: ClaimState = "insufficient_evidence"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "insufficient_evidence",
            "reason": self.reason,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "hyperparams_version": self.hyperparams_version,
            "last_updated_turn_id": self.last_updated_turn_id,
            "claim_state": self.claim_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "InsufficientEvidence":
        return cls(
            reason=str(payload.get("reason", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=_band(payload.get("confidence_band")),
            hyperparams_version=str(payload.get("hyperparams_version", DEFAULT_HYPERPARAMS.hyperparams_version)),
            last_updated_turn_id=str(payload.get("last_updated_turn_id", "")),
        )


@dataclass(frozen=True)
class SectionBase:
    evidence_refs: tuple[EvidenceQuoteRef, ...] = ()
    confidence_band: ConfidenceBand = "low"
    hyperparams_version: str = DEFAULT_HYPERPARAMS.hyperparams_version
    last_updated_turn_id: str = ""
    claim_state: ClaimState = "inferred_hypothesis"

    def base_dict(self) -> dict[str, object]:
        return {
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "hyperparams_version": self.hyperparams_version,
            "last_updated_turn_id": self.last_updated_turn_id,
            "claim_state": self.claim_state,
        }


@dataclass(frozen=True)
class PersonalitySummary(SectionBase):
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"status": "inferred_hypothesis", "summary": self.summary, **self.base_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalitySummary":
        return cls(summary=str(payload.get("summary", "")), **_base_kwargs(payload))


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    content_summary: str
    evidence_refs: tuple[EvidenceQuoteRef, ...]
    confidence_band: ConfidenceBand
    claim_state: ClaimState = "inferred_hypothesis"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "content_summary": self.content_summary,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "claim_state": self.claim_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceItem":
        return cls(
            kind=str(payload.get("kind", "")),
            content_summary=str(payload.get("content_summary", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=_band(payload.get("confidence_band")),
            claim_state=_claim_state(payload.get("claim_state")),
        )


@dataclass(frozen=True)
class EvidenceExtraction(SectionBase):
    evidence_items: tuple[EvidenceItem, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceExtraction":
        return cls(
            evidence_items=tuple(
                EvidenceItem.from_dict(item)
                for item in _object_list(payload.get("evidence_items"))
            ),
            **_base_kwargs(payload),
        )


@dataclass(frozen=True)
class PredictionSystemAccount(SectionBase):
    wants: str = ""
    fears: str = ""
    hypersensitive_to: tuple[str, ...] = ()
    ignores: tuple[str, ...] = ()
    default_interpretation: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "wants": self.wants,
            "fears": self.fears,
            "hypersensitive_to": list(self.hypersensitive_to),
            "ignores": list(self.ignores),
            "default_interpretation": self.default_interpretation,
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionSystemAccount":
        return cls(
            wants=str(payload.get("wants", "")),
            fears=str(payload.get("fears", "")),
            hypersensitive_to=_strings(payload.get("hypersensitive_to")),
            ignores=_strings(payload.get("ignores")),
            default_interpretation=str(payload.get("default_interpretation", "")),
            **_base_kwargs(payload),
        )


@dataclass(frozen=True)
class CoreBelief:
    core_belief: Literal["about_self", "about_others", "about_world"]
    content_summary: str
    evidence_refs: tuple[EvidenceQuoteRef, ...]
    confidence_band: ConfidenceBand
    claim_state: ClaimState = "inferred_hypothesis"

    def to_dict(self) -> dict[str, object]:
        return {
            "core_belief": self.core_belief,
            "content_summary": self.content_summary,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "claim_state": self.claim_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CoreBelief":
        kind = str(payload.get("core_belief", "about_self"))
        if kind not in {"about_self", "about_others", "about_world"}:
            kind = "about_self"
        return cls(
            core_belief=kind,  # type: ignore[arg-type]
            content_summary=str(payload.get("content_summary", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=_band(payload.get("confidence_band")),
            claim_state=_claim_state(payload.get("claim_state")),
        )


@dataclass(frozen=True)
class CoreBeliefSet(SectionBase):
    about_self: CoreBelief | None = None
    about_others: CoreBelief | None = None
    about_world: CoreBelief | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "about_self": self.about_self.to_dict() if self.about_self else None,
            "about_others": self.about_others.to_dict() if self.about_others else None,
            "about_world": self.about_world.to_dict() if self.about_world else None,
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CoreBeliefSet":
        return cls(
            about_self=_maybe_core_belief(payload.get("about_self")),
            about_others=_maybe_core_belief(payload.get("about_others")),
            about_world=_maybe_core_belief(payload.get("about_world")),
            **_base_kwargs(payload),
        )


@dataclass(frozen=True)
class DefenseItem:
    defense_kind: str
    protects_what: str
    short_term_benefit: str
    long_term_cost: str
    evidence_refs: tuple[EvidenceQuoteRef, ...]
    confidence_band: ConfidenceBand
    claim_state: ClaimState = "inferred_hypothesis"

    def to_dict(self) -> dict[str, object]:
        return {
            "defense_kind": self.defense_kind,
            "protects_what": self.protects_what,
            "short_term_benefit": self.short_term_benefit,
            "long_term_cost": self.long_term_cost,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
            "claim_state": self.claim_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DefenseItem":
        return cls(
            defense_kind=str(payload.get("defense_kind", "")),
            protects_what=str(payload.get("protects_what", "")),
            short_term_benefit=str(payload.get("short_term_benefit", "")),
            long_term_cost=str(payload.get("long_term_cost", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=_band(payload.get("confidence_band")),
            claim_state=_claim_state(payload.get("claim_state")),
        )


@dataclass(frozen=True)
class EmotionAndDefenses(SectionBase):
    dominant_emotional_baseline: str = ""
    threat_response: str = ""
    defenses: tuple[DefenseItem, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "dominant_emotional_baseline": self.dominant_emotional_baseline,
            "threat_response": self.threat_response,
            "defenses": [item.to_dict() for item in self.defenses],
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EmotionAndDefenses":
        return cls(
            dominant_emotional_baseline=str(payload.get("dominant_emotional_baseline", "")),
            threat_response=str(payload.get("threat_response", "")),
            defenses=tuple(DefenseItem.from_dict(item) for item in _object_list(payload.get("defenses"))),
            **_base_kwargs(payload),
        )


@dataclass(frozen=True)
class RelationshipTarget:
    kind: str
    why: str
    evidence_refs: tuple[EvidenceQuoteRef, ...]
    confidence_band: ConfidenceBand

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "why": self.why,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RelationshipTarget":
        return cls(
            kind=str(payload.get("kind", "")),
            why=str(payload.get("why", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
            confidence_band=_band(payload.get("confidence_band")),
        )


@dataclass(frozen=True)
class RelationshipPatterns(SectionBase):
    close_relationship_role: str = ""
    recurring_loop_summary: str = ""
    conflict_style: str = "flee"
    drawn_to: RelationshipTarget | None = None
    clashes_with: RelationshipTarget | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "close_relationship_role": self.close_relationship_role,
            "recurring_loop_summary": self.recurring_loop_summary,
            "conflict_style": self.conflict_style,
            "drawn_to": self.drawn_to.to_dict() if self.drawn_to else None,
            "clashes_with": self.clashes_with.to_dict() if self.clashes_with else None,
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RelationshipPatterns":
        return cls(
            close_relationship_role=str(payload.get("close_relationship_role", "")),
            recurring_loop_summary=str(payload.get("recurring_loop_summary", "")),
            conflict_style=str(payload.get("conflict_style", "flee")),
            drawn_to=_maybe_relationship_target(payload.get("drawn_to")),
            clashes_with=_maybe_relationship_target(payload.get("clashes_with")),
            **_base_kwargs(payload),
        )


@dataclass(frozen=True)
class CoreLoopStage:
    loop_stage: Literal[
        "trigger_event",
        "interpretation",
        "emotion",
        "action",
        "outcome",
        "belief_reinforcement",
    ]
    content_summary: str
    evidence_refs: tuple[EvidenceQuoteRef, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "loop_stage": self.loop_stage,
            "content_summary": self.content_summary,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CoreLoopStage":
        stage = str(payload.get("loop_stage", "trigger_event"))
        if stage not in CORE_LOOP_STAGES:
            stage = "trigger_event"
        return cls(
            loop_stage=stage,  # type: ignore[arg-type]
            content_summary=str(payload.get("content_summary", "")),
            evidence_refs=_refs(payload.get("evidence_refs")),
        )


CORE_LOOP_STAGES = (
    "trigger_event",
    "interpretation",
    "emotion",
    "action",
    "outcome",
    "belief_reinforcement",
)


@dataclass(frozen=True)
class CoreLoop(SectionBase):
    stages: tuple[CoreLoopStage, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"status": "inferred_hypothesis", "stages": [stage.to_dict() for stage in self.stages], **self.base_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CoreLoop":
        return cls(stages=tuple(CoreLoopStage.from_dict(item) for item in _object_list(payload.get("stages"))), **_base_kwargs(payload))


@dataclass(frozen=True)
class GrowthHints(SectionBase):
    stable_parts: tuple[str, ...] = ()
    fragile_spots: tuple[str, ...] = ()
    soft_spots: tuple[str, ...] = ()
    communication_styles_likely_accepted: tuple[str, ...] = ()
    communication_styles_that_trigger_defenses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "inferred_hypothesis",
            "stable_parts": list(self.stable_parts),
            "fragile_spots": list(self.fragile_spots),
            "soft_spots": list(self.soft_spots),
            "communication_styles_likely_accepted": list(self.communication_styles_likely_accepted),
            "communication_styles_that_trigger_defenses": list(self.communication_styles_that_trigger_defenses),
            **self.base_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GrowthHints":
        return cls(
            stable_parts=_strings(payload.get("stable_parts")),
            fragile_spots=_strings(payload.get("fragile_spots")),
            soft_spots=_strings(payload.get("soft_spots")),
            communication_styles_likely_accepted=_strings(payload.get("communication_styles_likely_accepted")),
            communication_styles_that_trigger_defenses=_strings(payload.get("communication_styles_that_trigger_defenses")),
            **_base_kwargs(payload),
        )


SectionValue = (
    PersonalitySummary
    | EvidenceExtraction
    | PredictionSystemAccount
    | CoreBeliefSet
    | EmotionAndDefenses
    | RelationshipPatterns
    | CoreLoop
    | GrowthHints
    | InsufficientEvidence
)


@dataclass(frozen=True)
class PersonalityProfile:
    user_id: str
    display_name_hint: str = ""
    step_1_summary: PersonalitySummary | InsufficientEvidence | None = None
    step_2_evidence: EvidenceExtraction | InsufficientEvidence | None = None
    step_3_prediction_system_account: PredictionSystemAccount | InsufficientEvidence | None = None
    step_4_core_beliefs: CoreBeliefSet | InsufficientEvidence | None = None
    step_5_emotion_and_defenses: EmotionAndDefenses | InsufficientEvidence | None = None
    step_6_relationship_patterns: RelationshipPatterns | InsufficientEvidence | None = None
    step_7_core_loop: CoreLoop | InsufficientEvidence | None = None
    step_8_growth_hints: GrowthHints | InsufficientEvidence | None = None
    section_freshness: dict[str, str] = field(default_factory=dict)
    section_last_insufficient: dict[str, dict[str, object]] = field(default_factory=dict)
    hyperparams_version: str = DEFAULT_HYPERPARAMS.hyperparams_version
    last_full_report_turn_id: str = ""
    last_full_report_status: ReportStatus = "draft"

    def section_for(self, section_kind: str) -> SectionValue | None:
        return getattr(self, SECTION_ATTRS[section_kind])

    def with_section(self, section_kind: str, section: SectionValue, *, turn_id: str) -> "PersonalityProfile":
        data = self.to_dict()
        attr = SECTION_ATTRS[section_kind]
        data[attr] = section.to_dict()
        freshness = dict(self.section_freshness)
        freshness[section_kind] = turn_id
        data["section_freshness"] = freshness
        if not isinstance(section, InsufficientEvidence):
            insuff = dict(self.section_last_insufficient)
            insuff.pop(section_kind, None)
            data["section_last_insufficient"] = insuff
        return PersonalityProfile.from_dict(data)

    def with_insufficient(self, section_kind: str, marker: InsufficientEvidence, *, keep_prior_section: bool = True) -> "PersonalityProfile":
        data = self.to_dict()
        existing = self.section_for(section_kind)
        if existing is None or isinstance(existing, InsufficientEvidence) or not keep_prior_section:
            data[SECTION_ATTRS[section_kind]] = marker.to_dict()
        insuff = dict(self.section_last_insufficient)
        insuff[section_kind] = marker.to_dict()
        data["section_last_insufficient"] = insuff
        freshness = dict(self.section_freshness)
        freshness[section_kind] = marker.last_updated_turn_id
        data["section_freshness"] = freshness
        return PersonalityProfile.from_dict(data)

    def with_report_state(self, *, turn_id: str, report_status: ReportStatus) -> "PersonalityProfile":
        return replace(self, last_full_report_turn_id=turn_id, last_full_report_status=report_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name_hint": self.display_name_hint,
            "step_1_summary": _section_dict(self.step_1_summary),
            "step_2_evidence": _section_dict(self.step_2_evidence),
            "step_3_prediction_system_account": _section_dict(self.step_3_prediction_system_account),
            "step_4_core_beliefs": _section_dict(self.step_4_core_beliefs),
            "step_5_emotion_and_defenses": _section_dict(self.step_5_emotion_and_defenses),
            "step_6_relationship_patterns": _section_dict(self.step_6_relationship_patterns),
            "step_7_core_loop": _section_dict(self.step_7_core_loop),
            "step_8_growth_hints": _section_dict(self.step_8_growth_hints),
            "section_freshness": dict(sorted(self.section_freshness.items())),
            "section_last_insufficient": dict(sorted(self.section_last_insufficient.items())),
            "hyperparams_version": self.hyperparams_version,
            "last_full_report_turn_id": self.last_full_report_turn_id,
            "last_full_report_status": self.last_full_report_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "PersonalityProfile":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("PersonalityProfile JSON must decode to an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalityProfile":
        return cls(
            user_id=str(payload.get("user_id", "")),
            display_name_hint=str(payload.get("display_name_hint", "")),
            step_1_summary=_section(payload.get("step_1_summary"), PersonalitySummary),
            step_2_evidence=_section(payload.get("step_2_evidence"), EvidenceExtraction),
            step_3_prediction_system_account=_section(payload.get("step_3_prediction_system_account"), PredictionSystemAccount),
            step_4_core_beliefs=_section(payload.get("step_4_core_beliefs"), CoreBeliefSet),
            step_5_emotion_and_defenses=_section(payload.get("step_5_emotion_and_defenses"), EmotionAndDefenses),
            step_6_relationship_patterns=_section(payload.get("step_6_relationship_patterns"), RelationshipPatterns),
            step_7_core_loop=_section(payload.get("step_7_core_loop"), CoreLoop),
            step_8_growth_hints=_section(payload.get("step_8_growth_hints"), GrowthHints),
            section_freshness={str(k): str(v) for k, v in _mapping(payload.get("section_freshness")).items()},
            section_last_insufficient={
                str(k): dict(v)
                for k, v in _mapping(payload.get("section_last_insufficient")).items()
                if isinstance(v, Mapping)
            },
            hyperparams_version=str(payload.get("hyperparams_version", DEFAULT_HYPERPARAMS.hyperparams_version)),
            last_full_report_turn_id=str(payload.get("last_full_report_turn_id", "")),
            last_full_report_status=_report_status(payload.get("last_full_report_status")),
        )


SECTION_ATTRS: dict[str, str] = {
    "step_1": "step_1_summary",
    "step_2": "step_2_evidence",
    "step_3": "step_3_prediction_system_account",
    "step_4": "step_4_core_beliefs",
    "step_5": "step_5_emotion_and_defenses",
    "step_6": "step_6_relationship_patterns",
    "step_7": "step_7_core_loop",
    "step_8": "step_8_growth_hints",
}


def bounded_confidence_band(
    requested_band: str,
    *,
    existing_band: str = "low",
    evidence_refs: Sequence[EvidenceQuoteRef] = (),
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> ConfidenceBand:
    requested_level = CONFIDENCE_ORDER.get(str(requested_band), 0)
    if requested_level >= CONFIDENCE_ORDER["high"]:
        distinct_turns = {ref.turn_id for ref in evidence_refs if ref.turn_id}
        if (
            len(evidence_refs) < hyperparams.high_confidence_min_evidence_refs
            or len(distinct_turns) < hyperparams.high_confidence_min_distinct_turns
        ):
            requested_level = CONFIDENCE_ORDER["med"]
    elif requested_level >= CONFIDENCE_ORDER["med"] and len(evidence_refs) < hyperparams.med_confidence_min_evidence_refs:
        requested_level = CONFIDENCE_ORDER["low"]
    prior = CONFIDENCE_ORDER.get(str(existing_band), 0)
    max_next = min(prior + hyperparams.per_run_band_promotion_levels, CONFIDENCE_ORDER["high"])
    return CONFIDENCE_BY_LEVEL[min(requested_level, max_next)]  # type: ignore[return-value]


def _section_dict(value: SectionValue | None) -> dict[str, object] | None:
    return value.to_dict() if value is not None else None


def _section(value: object, cls: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    if str(value.get("status", "")) == "insufficient_evidence":
        return InsufficientEvidence.from_dict(value)
    return cls.from_dict(value)


def _base_kwargs(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "evidence_refs": _refs(payload.get("evidence_refs")),
        "confidence_band": _band(payload.get("confidence_band")),
        "hyperparams_version": str(payload.get("hyperparams_version", DEFAULT_HYPERPARAMS.hyperparams_version)),
        "last_updated_turn_id": str(payload.get("last_updated_turn_id", "")),
        "claim_state": _claim_state(payload.get("claim_state")),
    }


def _refs(value: object) -> tuple[EvidenceQuoteRef, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(EvidenceQuoteRef.from_any(item) for item in value if str(item or "").strip() or isinstance(item, Mapping))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _object_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _band(value: object) -> ConfidenceBand:
    text = str(value or "low")
    return text if text in {"low", "med", "high"} else "low"  # type: ignore[return-value]


def _claim_state(value: object) -> ClaimState:
    text = str(value or "inferred_hypothesis")
    if text in {"inferred_hypothesis", "confirmed_by_user", "insufficient_evidence"}:
        return text  # type: ignore[return-value]
    return "inferred_hypothesis"


def _report_status(value: object) -> ReportStatus:
    text = str(value or "draft")
    if text in {"draft", "linter_failed", "ready", "stale", "superseded"}:
        return text  # type: ignore[return-value]
    return "draft"


def _maybe_core_belief(value: object) -> CoreBelief | None:
    return CoreBelief.from_dict(value) if isinstance(value, Mapping) else None


def _maybe_relationship_target(value: object) -> RelationshipTarget | None:
    return RelationshipTarget.from_dict(value) if isinstance(value, Mapping) else None


def section_kinds() -> tuple[str, ...]:
    return SECTION_KINDS
