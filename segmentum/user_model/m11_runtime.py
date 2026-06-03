"""Per-turn M11 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from segmentum.user_model.m17_calibration import (
    PredictionCalibrationState,
    update_prediction_calibration,
)

from .evidence_cards import UserModelEvidenceCard, evidence_cards_from_user_model, prompt_safe_cards
from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams
from .llm_extractor import noop_extraction, validate_extractor_output
from .prediction_ledger import PredictionEntry, UserPredictionLedger, apply_prediction_updates
from .reliability_ledger import (
    ReliabilityJudgment,
    SourceReliabilityLedger,
    ReliabilityUpdate,
    update_reliability,
)
from .user_model import UserModel, apply_claims_to_user_model, apply_contradictions
from .value_composer import ValueComposition, compose_value

Extractor = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class M11RuntimeConfig:
    m11_user_model_enabled: bool = False
    persona_kind: str = "legacy"

    @classmethod
    def for_persona(cls, *, persona_kind: str) -> "M11RuntimeConfig":
        return cls(
            m11_user_model_enabled=persona_kind == "new",
            persona_kind=persona_kind,
        )


@dataclass(frozen=True)
class ReplyPolicyAdjustment:
    adjustment: str
    caused_by_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"adjustment": self.adjustment, "caused_by_id": self.caused_by_id, "reason": self.reason}


@dataclass(frozen=True)
class M11RuntimeState:
    user_model: UserModel
    prediction_ledger: UserPredictionLedger = field(default_factory=UserPredictionLedger)
    reliability_ledger: SourceReliabilityLedger = field(default_factory=SourceReliabilityLedger.empty)
    prediction_calibration: PredictionCalibrationState = field(default_factory=PredictionCalibrationState)

    @classmethod
    def clean(cls, *, user_id: str, display_name: str = "") -> "M11RuntimeState":
        return cls(user_model=UserModel(user_id=user_id, display_name=display_name))

    def to_dict(self) -> dict[str, object]:
        return {
            "user_model": self.user_model.to_dict(),
            "prediction_ledger": self.prediction_ledger.to_dict(),
            "reliability_ledger": self.reliability_ledger.to_dict(),
            "prediction_calibration": self.prediction_calibration.to_dict(),
        }


@dataclass(frozen=True)
class M11TurnResult:
    enabled: bool
    state_before: dict[str, object]
    state_after: dict[str, object]
    extractor_output: dict[str, object]
    reliability_ledger_updates: tuple[ReliabilityUpdate, ...]
    memory_value_compositions: tuple[ValueComposition, ...]
    evidence_cards: tuple[UserModelEvidenceCard, ...]
    prompt_safe_evidence_cards: tuple[dict[str, str], ...]
    reply_policy_effects: tuple[ReplyPolicyAdjustment, ...]
    quarantined_hypotheses: tuple[str, ...]
    legacy_memory_rows_before: tuple[Mapping[str, object], ...] = ()
    legacy_memory_rows_after: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "extractor_output": self.extractor_output,
            "reliability_ledger_updates": [item.to_dict() for item in self.reliability_ledger_updates],
            "memory_value_compositions": [item.to_dict() for item in self.memory_value_compositions],
            "evidence_cards": [item.to_dict() for item in self.evidence_cards],
            "prompt_safe_evidence_cards": list(self.prompt_safe_evidence_cards),
            "reply_policy_effects": [item.to_dict() for item in self.reply_policy_effects],
            "quarantined_hypotheses": list(self.quarantined_hypotheses),
            "legacy_memory_rows_before": [dict(row) for row in self.legacy_memory_rows_before],
            "legacy_memory_rows_after": [dict(row) for row in self.legacy_memory_rows_after],
        }


def run_m11_turn(
    state: M11RuntimeState,
    *,
    user_id: str,
    turn_id: int,
    current_turn_quotes: Mapping[str, str] | None = None,
    last_turn_summaries: Sequence[Mapping[str, object]] = (),
    extractor: Extractor | None = None,
    config: M11RuntimeConfig | None = None,
    legacy_memory_rows: Sequence[Mapping[str, object]] = (),
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[M11RuntimeState, M11TurnResult]:
    config = config or M11RuntimeConfig()
    before = state.to_dict()
    legacy_snapshot = tuple(dict(row) for row in legacy_memory_rows)
    if not config.m11_user_model_enabled:
        result = M11TurnResult(
            enabled=False,
            state_before=before,
            state_after=before,
            extractor_output=noop_extraction(),
            reliability_ledger_updates=(),
            memory_value_compositions=(),
            evidence_cards=(),
            prompt_safe_evidence_cards=(),
            reply_policy_effects=(),
            quarantined_hypotheses=(),
            legacy_memory_rows_before=legacy_snapshot,
            legacy_memory_rows_after=legacy_snapshot,
        )
        return state, result

    snapshot = _bounded_snapshot(state, current_turn_quotes or {}, last_turn_summaries)
    extractor_output = dict((extractor or (lambda _: noop_extraction()))(snapshot))
    open_prediction_ids = {
        entry.prediction_id
        for entry in state.prediction_ledger.entries
        if state.prediction_ledger.latest_status(entry.prediction_id) == "pending"
    }
    hypothesis_ids = {hyp.hypothesis_id for hyp in state.user_model.all_hypotheses()}
    judgment_ids = {
        entry.prediction_id
        for entry in state.prediction_ledger.entries
        if entry.validation_status in {"confirmed", "violated", "uncertain"}
    }
    validated = validate_extractor_output(
        extractor_output,
        snapshot_prediction_ids=open_prediction_ids,
        snapshot_hypothesis_ids=hypothesis_ids,
        snapshot_judgment_ids=judgment_ids,
    )

    turn_key = str(turn_id)
    next_model = apply_claims_to_user_model(
        state.user_model,
        list(validated["claims_made"]),
        turn_id=turn_key,
        hyperparams=hyperparams,
    )
    next_model = apply_contradictions(
        next_model,
        list(validated["contradiction_detections"]),
        turn_id=turn_key,
    )
    reliability_judgments = _reliability_judgments_from_extraction(validated, turn_id=turn_id)
    next_reliability, reliability_updates = update_reliability(
        state.reliability_ledger,
        reliability_judgments,
        current_turn_id=turn_id,
        hyperparams=hyperparams,
    )
    next_model = next_model.with_reliability(
        {
            domain: entry.reliability
            for domain, entry in next_reliability.entries_by_domain.items()
        }
    )
    current_hypothesis_ids = {hyp.hypothesis_id for hyp in next_model.all_hypotheses()}
    next_prediction_ledger = apply_prediction_updates(
        state.prediction_ledger,
        turn_id=turn_id,
        proposals=list(validated["prediction_proposals"]),
        judgments=list(validated["prediction_judgments"]),
        known_hypothesis_ids=current_hypothesis_ids,
        known_judgment_ids=judgment_ids,
        calibration_need_band=str(validated["calibration_need_band"]),  # type: ignore[arg-type]
        hyperparams=hyperparams,
        type_precision_by_type={
            key: calibration.precision_ema
            for key, calibration in state.prediction_calibration.by_type.items()
        },
        type_sample_count_by_type={
            key: calibration.sample_count
            for key, calibration in state.prediction_calibration.by_type.items()
        },
    )
    next_prediction_calibration = state.prediction_calibration
    for entry in next_prediction_ledger.entries:
        if entry.turn_id != turn_id or entry.event_kind not in {"judgment", "expiration"}:
            continue
        if entry.settlement_outcome not in {"confirmed", "violated"}:
            continue
        next_prediction_calibration = update_prediction_calibration(
            next_prediction_calibration,
            prediction_type=entry.prediction_type,
            committed_confidence=entry.committed_confidence,
            outcome=entry.settlement_outcome,
            brier_score=entry.m17_brier_score,
            turn_id=turn_id,
        )
    compositions = _compose_for_claims(
        validated,
        reliability=next_reliability,
        hyperparams=hyperparams,
    )
    recent_judgments = tuple(
        entry
        for entry in next_prediction_ledger.entries
        if entry.event_kind in {"judgment", "expiration"} and entry.turn_id >= turn_id - hyperparams.default_prediction_expiry_turns
    )
    cards = evidence_cards_from_user_model(next_model, recent_judgments=recent_judgments, hyperparams=hyperparams)
    policy_effects = derive_reply_policy_effects(
        next_model,
        next_prediction_ledger,
        cards=cards,
        hyperparams=hyperparams,
    )
    quarantined = tuple(
        hyp.hypothesis_id
        for hyp in next_model.all_hypotheses()
        if hyp.permitted_use == "forbidden" or hyp.contradiction_refs
    )
    next_state = M11RuntimeState(
        user_model=next_model,
        prediction_ledger=next_prediction_ledger,
        reliability_ledger=next_reliability,
        prediction_calibration=next_prediction_calibration,
    )
    result = M11TurnResult(
        enabled=True,
        state_before=before,
        state_after=next_state.to_dict(),
        extractor_output=validated,
        reliability_ledger_updates=reliability_updates,
        memory_value_compositions=compositions,
        evidence_cards=cards,
        prompt_safe_evidence_cards=prompt_safe_cards(cards),
        reply_policy_effects=policy_effects,
        quarantined_hypotheses=quarantined,
        legacy_memory_rows_before=legacy_snapshot,
        legacy_memory_rows_after=legacy_snapshot,
    )
    return next_state, result


def derive_reply_policy_effects(
    model: UserModel,
    ledger: UserPredictionLedger,
    *,
    cards: Sequence[UserModelEvidenceCard],
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[ReplyPolicyAdjustment, ...]:
    effects: list[ReplyPolicyAdjustment] = []
    for hyp in model.preference_hypotheses:
        reliability = model.source_reliability_by_domain.get(hyp.domain, hyperparams.prior_mean)
        if (
            reliability >= hyperparams.strong_reliability_threshold
            and len(hyp.evidence_refs) >= hyperparams.brevity_hypothesis_min_refs
            and hyp.confidence_band == "high"
        ):
            effects.append(
                ReplyPolicyAdjustment(
                    adjustment="prefer_shorter_reply",
                    caused_by_id=hyp.hypothesis_id,
                    reason="reliable_active_preference",
                )
            )
    prediction_judgments = _recent_prediction_judgments(ledger, hyperparams=hyperparams)
    violated_candidates = [
        entry
        for entry in prediction_judgments
        if entry.prediction_type in {"intent_prediction", "preference_prediction"}
        and entry.settlement_outcome == "violated"
        and entry.m17_prediction_error is not None
    ]
    if violated_candidates:
        entry = max(
            violated_candidates,
            key=lambda row: (int(row.turn_id), float(row.m17_prediction_error or 0.0)),
        )
        effects.append(
            ReplyPolicyAdjustment(
                adjustment="ask_clarifying_question",
                caused_by_id=entry.prediction_id,
                reason="recent_intent_or_preference_prediction_violated",
            )
        )
    if not any(effect.adjustment == "prefer_shorter_reply" for effect in effects):
        confirmed_brevity_predictions = _confirmed_brevity_prediction_effect(
            prediction_judgments,
            hyperparams=hyperparams,
        )
        if confirmed_brevity_predictions is not None:
            effects.append(confirmed_brevity_predictions)
    social_reliability = model.source_reliability_by_domain.get("social_relationship_claims", hyperparams.prior_mean)
    if social_reliability < hyperparams.low_reliability_threshold:
        effects.append(
            ReplyPolicyAdjustment(
                adjustment="soften_social_evidence_language",
                caused_by_id="domain:social_relationship_claims",
                reason="low_domain_reliability",
            )
        )
    for card in cards:
        if card.memory_id.startswith("repair:") and card.confidence_band == "high":
            effects.append(
                ReplyPolicyAdjustment(
                    adjustment="prioritize_relationship_repair",
                    caused_by_id=card.memory_id,
                    reason="strong_active_repair_hypothesis",
                )
            )
            break
    return tuple(effects)


def _recent_prediction_judgments(
    ledger: UserPredictionLedger,
    *,
    hyperparams: Hyperparams,
) -> tuple[PredictionEntry, ...]:
    latest_turn = max((int(entry.turn_id) for entry in ledger.entries), default=0)
    window_floor = max(0, latest_turn - hyperparams.reply_policy_recent_prediction_window_turns + 1)
    return tuple(
        entry
        for entry in ledger.entries
        if entry.event_kind == "judgment" and int(entry.turn_id) >= window_floor
    )


def _confirmed_brevity_prediction_effect(
    entries: Sequence[PredictionEntry],
    *,
    hyperparams: Hyperparams,
) -> ReplyPolicyAdjustment | None:
    by_summary: dict[str, list[PredictionEntry]] = {}
    for entry in entries:
        if entry.prediction_type != "preference_prediction":
            continue
        if entry.settlement_outcome != "confirmed":
            continue
        if entry.m17_prediction_error is None and entry.m17_brier_score is None:
            continue
        summary_key = _brevity_prediction_summary_key(entry.predicted_value_summary)
        if not summary_key:
            continue
        by_summary.setdefault(summary_key, []).append(entry)
    qualified_groups = [
        rows
        for rows in by_summary.values()
        if len(rows) >= hyperparams.reply_policy_confirmed_preference_min_count
    ]
    if not qualified_groups:
        return None
    latest_group = max(
        qualified_groups,
        key=lambda rows: max((int(row.turn_id), float(row.m17_brier_score or 0.0)) for row in rows),
    )
    latest_entry = max(latest_group, key=lambda row: (int(row.turn_id), float(row.m17_brier_score or 0.0)))
    return ReplyPolicyAdjustment(
        adjustment="prefer_shorter_reply",
        caused_by_id=latest_entry.prediction_id,
        reason="repeated_confirmed_preference_prediction",
    )


def _brevity_prediction_summary_key(summary: object) -> str:
    text = " ".join(str(summary or "").strip().casefold().split())
    if not text:
        return ""
    stems = (
        "short reply",
        "shorter repl",
        "brief repl",
        "concise repl",
        "compact repl",
        "short responses",
        "shorter responses",
        "brief responses",
        "concise responses",
        "compact responses",
        "简短",
        "更短",
        "简洁",
        "精简",
        "少一点",
        "短一点",
    )
    if any(stem in text for stem in stems):
        return text
    return ""


def _bounded_snapshot(
    state: M11RuntimeState,
    current_turn_quotes: Mapping[str, str],
    last_turn_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "user_id": state.user_model.user_id,
        "current_turn_quotes": dict(current_turn_quotes),
        "last_turn_summaries": [dict(row) for row in last_turn_summaries[-5:]],
        "active_hypotheses": [
            {
                "hypothesis_id": hyp.hypothesis_id,
                "content_summary": hyp.content_summary,
                "confidence_band": hyp.confidence_band,
                "permitted_use": hyp.permitted_use,
            }
            for hyp in state.user_model.all_hypotheses()
        ],
        "open_predictions": [
            {
                "prediction_id": entry.prediction_id,
                "prediction_type": entry.prediction_type,
                "predicted_value_summary": entry.predicted_value_summary,
                "confidence_band": entry.confidence_band,
                "committed_confidence": entry.committed_confidence,
                "created_at_turn": entry.created_at_turn,
                "expires_after_turns": entry.expires_after_turns,
            }
            for entry in state.prediction_ledger.entries
            if state.prediction_ledger.latest_status(entry.prediction_id) == "pending"
        ],
    }


def _reliability_judgments_from_extraction(payload: Mapping[str, object], *, turn_id: int) -> tuple[ReliabilityJudgment, ...]:
    contradictions = {
        str(row.get("claim_id", ""))
        for row in payload.get("contradiction_detections", [])
        if isinstance(row, Mapping)
    }
    judgments: list[ReliabilityJudgment] = []
    for claim in payload.get("claims_made", []):
        if not isinstance(claim, Mapping):
            continue
        claim_id = str(claim.get("id", ""))
        status = "violated" if claim_id in contradictions else "confirmed"
        judgments.append(
            ReliabilityJudgment(
                judgment_id=f"claim:{claim_id}:{turn_id}",
                turn_id=turn_id,
                domain=str(claim.get("domain", "task_requirements")),
                status=status,  # type: ignore[arg-type]
                modality=str(claim.get("modality", "factual")),  # type: ignore[arg-type]
                evidence_refs=tuple(str(q) for q in claim.get("evidence_quote_ids", [])),
                evidence_text="",
            )
        )
    return tuple(judgments)


def _compose_for_claims(
    payload: Mapping[str, object],
    *,
    reliability: SourceReliabilityLedger,
    hyperparams: Hyperparams,
) -> tuple[ValueComposition, ...]:
    contradictions = {
        str(row.get("claim_id", ""))
        for row in payload.get("contradiction_detections", [])
        if isinstance(row, Mapping)
    }
    compositions: list[ValueComposition] = []
    for claim in payload.get("claims_made", []):
        if not isinstance(claim, Mapping):
            continue
        domain = str(claim.get("domain", "task_requirements"))
        compositions.append(
            compose_value(
                memory_value_band=str(payload.get("memory_value_band", "low")),  # type: ignore[arg-type]
                confidence_band=str(claim.get("confidence_band", "low")),  # type: ignore[arg-type]
                source_reliability=reliability.reliability_for(domain).reliability,
                recency_weight=1.0,
                contradiction_unresolved=str(claim.get("id", "")) in contradictions,
                privacy_or_safety_flag=str(claim.get("permitted_use", "")) == "forbidden",
                hyperparams=hyperparams,
            )
        )
    return tuple(compositions)
