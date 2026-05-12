"""Per-turn M12.0 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

from segmentum.cognitive_events import CognitiveEventBus

from .evidence_cards import IdentityEvidenceCard, cards_to_prompt_safe_memory_evidence
from .hyperparams import DEFAULT_HYPERPARAMS, M12Hyperparams
from .identity_claim_ledger import IdentityClaimLedger, IdentityClaimLedgerEntry
from .identity_conflict_detector import ConflictRecord, detect_identity_conflicts
from .identity_profile import AliasObservation, ContinuityCue, IdentityProfile, profile_state_hash
from .llm_identity_extractor import noop_extraction, validate_extractor_output
from .reply_policy import ReplyPolicyDecision, select_reply_policy
from .strangeness_signal import (
    IdentityStrangenessSignal,
    build_strangeness_signal,
    signal_to_self_thought_event,
)

Extractor = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class M12RuntimeConfig:
    m12_identity_continuity_enabled: bool = False
    persona_kind: str = "legacy"


@dataclass(frozen=True)
class M12RuntimeState:
    profiles_by_user: dict[str, IdentityProfile] = field(default_factory=dict)
    claim_ledger: IdentityClaimLedger = field(default_factory=IdentityClaimLedger)
    conflict_records: tuple[ConflictRecord, ...] = ()

    @classmethod
    def clean(cls) -> "M12RuntimeState":
        return cls()

    def to_dict(self) -> dict[str, object]:
        return {
            "profiles_by_user": {
                user_id: profile.to_dict()
                for user_id, profile in self.profiles_by_user.items()
            },
            "claim_ledger": self.claim_ledger.to_dict(),
            "conflict_records": [record.to_dict() for record in self.conflict_records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "M12RuntimeState":
        raw_profiles = payload.get("profiles_by_user", {})
        if not isinstance(raw_profiles, Mapping):
            raw_profiles = {}
        raw_conflicts = payload.get("conflict_records", [])
        if not isinstance(raw_conflicts, list):
            raw_conflicts = []
        return cls(
            profiles_by_user={
                str(user_id): IdentityProfile.from_dict(item)
                for user_id, item in raw_profiles.items()
                if isinstance(item, Mapping)
            },
            claim_ledger=IdentityClaimLedger.from_dict(
                payload.get("claim_ledger", {}) if isinstance(payload.get("claim_ledger"), Mapping) else {}
            ),
            conflict_records=tuple(
                ConflictRecord.from_dict(item)
                for item in raw_conflicts
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True)
class M12TurnResult:
    enabled: bool
    state_before: dict[str, object]
    state_after: dict[str, object]
    extractor_output: dict[str, object]
    conflict_records_created: tuple[ConflictRecord, ...]
    strangeness_signal: IdentityStrangenessSignal | None
    reply_policy: ReplyPolicyDecision
    evidence_cards: tuple[IdentityEvidenceCard, ...]
    prompt_safe_evidence_cards: tuple[dict[str, str], ...]
    entity_binding_context: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "extractor_output": self.extractor_output,
            "conflict_records_created": [item.to_dict() for item in self.conflict_records_created],
            "strangeness_signal": self.strangeness_signal.to_dict() if self.strangeness_signal else None,
            "reply_policy": self.reply_policy.to_dict(),
            "evidence_cards": [item.prompt_safe_dict() for item in self.evidence_cards],
            "prompt_safe_evidence_cards": list(self.prompt_safe_evidence_cards),
            "entity_binding_context": dict(self.entity_binding_context),
        }


def run_m12_turn(
    state: M12RuntimeState,
    *,
    user_id: str,
    display_name: str,
    turn_id: str,
    current_turn_quotes: Mapping[str, str] | None = None,
    m11_readonly_summary: Mapping[str, object] | None = None,
    legacy_aliases: Sequence[str] = (),
    extractor: Extractor | None = None,
    config: M12RuntimeConfig | None = None,
    hyperparams: M12Hyperparams = DEFAULT_HYPERPARAMS,
    event_bus: CognitiveEventBus | None = None,
    session_id: str = "live",
    persona_id: str = "default",
    cycle: int = 0,
    event_sequence_index: int = 0,
    identity_anchored_action: bool = False,
) -> tuple[M12RuntimeState, M12TurnResult]:
    config = config or M12RuntimeConfig()
    before = state.to_dict()
    if not config.m12_identity_continuity_enabled:
        result = M12TurnResult(
            enabled=False,
            state_before=before,
            state_after=before,
            extractor_output=noop_extraction(),
            conflict_records_created=(),
            strangeness_signal=None,
            reply_policy=ReplyPolicyDecision("accept", ("m12_disabled",)),
            evidence_cards=(),
            prompt_safe_evidence_cards=(),
            entity_binding_context={},
        )
        return state, result

    profile = state.profiles_by_user.get(user_id, IdentityProfile(user_id=user_id, display_name=display_name))
    profile = _seed_legacy_aliases(profile, legacy_aliases=legacy_aliases, turn_id=turn_id)
    snapshot = _bounded_snapshot(
        profile=profile,
        current_turn_quotes=current_turn_quotes or {},
        m11_readonly_summary=m11_readonly_summary or {},
    )
    extraction = dict((extractor or (lambda _: noop_extraction()))(snapshot))
    validated = validate_extractor_output(extraction)

    claims = [item for item in validated.get("identity_claims", []) if isinstance(item, Mapping)][
        : hyperparams.max_claims_per_turn
    ]
    cue_rows = [item for item in validated.get("continuity_cues", []) if isinstance(item, Mapping)][
        : hyperparams.max_cues_per_turn
    ]
    cues = tuple(
        ContinuityCue(
            cue_id=str(row.get("id", "")),
            cue_kind=str(row.get("cue_kind", "history")),
            content_summary=str(row.get("content_summary", ""))[:120],
            evidence_refs=tuple(str(ref) for ref in row.get("evidence_quote_ids", []) if isinstance(row.get("evidence_quote_ids"), list)),
            confidence_band=str(row.get("confidence_band", "low")),
            supports=str(row.get("supports", "weakens")),
            source_turn_id=turn_id,
        )
        for row in cue_rows
    )
    updated_profile = _apply_cues(profile, cues, turn_id=turn_id, hyperparams=hyperparams)
    ledger = state.claim_ledger
    all_conflicts = list(state.conflict_records)
    created_conflicts: list[ConflictRecord] = []
    for claim in claims:
        alias = str(claim.get("asserted_alias", "")).strip()
        if not alias:
            continue
        prior_hash = profile_state_hash(updated_profile)
        updated_profile = _apply_alias_claim(
            updated_profile,
            alias=alias,
            modality=str(claim.get("modality", "factual")),
            turn_id=turn_id,
            claim_id=str(claim.get("id", "")),
            confidence_band=str(claim.get("confidence_band", "low")),
            hyperparams=hyperparams,
        )
        post_hash = profile_state_hash(updated_profile)
        conflicts = detect_identity_conflicts(
            turn_id=turn_id,
            claim_id=str(claim.get("id", "")),
            claimant_user_id=user_id,
            asserted_alias=alias,
            modality=str(claim.get("modality", "factual")),
            profiles_by_user={**state.profiles_by_user, user_id: updated_profile},
        )
        if conflicts:
            created_conflicts.extend(conflicts)
            all_conflicts.extend(conflicts)
            updated_profile = replace(
                updated_profile,
                identity_state="conflicted",
                contradiction_refs=tuple(
                    dict.fromkeys([*updated_profile.contradiction_refs, *[row.conflict_id for row in conflicts]])
                ),
            )
        ledger = ledger.append(
            IdentityClaimLedgerEntry(
                claim_id=str(claim.get("id", "")),
                turn_id=turn_id,
                claimant_user_id=user_id,
                asserted_alias=alias,
                modality=str(claim.get("modality", "factual")),
                evidence_quote_ids=tuple(
                    str(ref) for ref in claim.get("evidence_quote_ids", []) if isinstance(claim.get("evidence_quote_ids"), list)
                ),
                band_at_claim_time=str(claim.get("confidence_band", "low")),
                prior_state_snapshot_ref=prior_hash,
                post_state_snapshot_ref=post_hash,
                validation_status="conflicted" if conflicts else updated_profile.identity_state,
                conflict_record_id=conflicts[0].conflict_id if conflicts else "",
            )
        )

    strangeness = build_strangeness_signal(
        turn_id=turn_id,
        claim_alias=updated_profile.aliases_observed[-1].alias_text if updated_profile.aliases_observed else "",
        claimant_user_id=user_id,
        profile=updated_profile,
        conflicts=created_conflicts,
        current_turn_cues=cues,
        signal_count_this_turn=0,
        hyperparams=hyperparams,
    )
    if strangeness is not None and event_bus is not None:
        event_bus.publish(
            signal_to_self_thought_event(
                strangeness,
                session_id=session_id,
                persona_id=persona_id,
                turn_id=turn_id,
                cycle=cycle,
                sequence_index=event_sequence_index,
            )
        )
    open_conflicts = tuple(
        row for row in all_conflicts if row.resolution_status in {"open", "probed"}
    )
    reply_policy = select_reply_policy(
        profile=updated_profile,
        active_conflicts=open_conflicts,
        strangeness_signal=strangeness,
        identity_anchored_action=identity_anchored_action,
    )
    cards = cards_to_prompt_safe_memory_evidence(
        profile=updated_profile,
        open_conflicts=open_conflicts,
        hyperparams=hyperparams,
    )
    entity_binding_context = _build_entity_binding_context(profile=updated_profile)
    next_state = M12RuntimeState(
        profiles_by_user={**state.profiles_by_user, user_id: updated_profile},
        claim_ledger=ledger,
        conflict_records=tuple(all_conflicts),
    )
    result = M12TurnResult(
        enabled=True,
        state_before=before,
        state_after=next_state.to_dict(),
        extractor_output=validated,
        conflict_records_created=tuple(created_conflicts),
        strangeness_signal=strangeness,
        reply_policy=reply_policy,
        evidence_cards=cards,
        prompt_safe_evidence_cards=tuple(card.prompt_safe_dict() for card in cards if card.permitted_use != "forbidden"),
        entity_binding_context=entity_binding_context,
    )
    return next_state, result


def _seed_legacy_aliases(profile: IdentityProfile, *, legacy_aliases: Sequence[str], turn_id: str) -> IdentityProfile:
    out = profile
    known = {obs.alias_text.strip().lower() for obs in out.aliases_observed}
    for idx, alias in enumerate(legacy_aliases):
        alias_text = str(alias).strip()
        if not alias_text or alias_text.lower() in known:
            continue
        obs = AliasObservation(
            alias_observation_id=f"legacy:{out.user_id}:{idx}:{alias_text.lower()}",
            alias_text=alias_text,
            first_seen_turn=turn_id,
            last_seen_turn=turn_id,
            seen_count=1,
            modality="factual",
            permitted_use="observe",
        )
        out = replace(
            out,
            aliases_observed=(*out.aliases_observed, obs),
            identity_state="unverified" if out.identity_state == "unverified" else out.identity_state,
        )
        known.add(alias_text.lower())
    return out


def _apply_alias_claim(
    profile: IdentityProfile,
    *,
    alias: str,
    modality: str,
    turn_id: str,
    claim_id: str,
    confidence_band: str,
    hyperparams: M12Hyperparams,
) -> IdentityProfile:
    existing = list(profile.aliases_observed)
    lower = alias.strip().lower()
    updated = False
    for idx, item in enumerate(existing):
        if item.alias_text.strip().lower() != lower:
            continue
        existing[idx] = replace(item, last_seen_turn=turn_id, seen_count=item.seen_count + 1, modality=modality)
        updated = True
        break
    if not updated:
        existing.append(
            AliasObservation(
                alias_observation_id=f"alias:{profile.user_id}:{turn_id}:{claim_id}",
                alias_text=alias,
                first_seen_turn=turn_id,
                last_seen_turn=turn_id,
                seen_count=1,
                modality=modality,
                permitted_use="accept" if modality == "factual" else "observe",
            )
        )
    existing = existing[-hyperparams.max_alias_observations :]
    if profile.identity_state == "retracted":
        next_state = "retracted"
    elif modality in {"roleplay", "joke", "hypothetical"}:
        next_state = "unverified"
    elif profile.binding_confidence_band == "high" and confidence_band in {"med", "high"}:
        next_state = "corroborated"
    else:
        next_state = "asserted"
    return replace(
        profile,
        aliases_observed=tuple(existing),
        identity_state=next_state,
        last_updated_turn_id=turn_id,
    )


def _apply_cues(
    profile: IdentityProfile,
    cues: Sequence[ContinuityCue],
    *,
    turn_id: str,
    hyperparams: M12Hyperparams,
) -> IdentityProfile:
    merged = (*profile.continuity_evidence, *cues)
    merged = merged[-hyperparams.max_continuity_cues :]
    bind_cues = [cue for cue in merged if cue.supports == "binds" and cue.confidence_band in {"med", "high"}]
    binds = len(bind_cues)
    distinct_bind_turns = len({cue.source_turn_id or turn_id for cue in bind_cues})
    contradictions = sum(
        1 for cue in merged if cue.supports == "contradicts" and cue.confidence_band in {"med", "high"}
    )
    if contradictions >= hyperparams.contradict_demote_threshold:
        band = "low"
        state = "conflicted"
    elif (
        binds >= hyperparams.bind_promotion_threshold
        and distinct_bind_turns >= hyperparams.min_distinct_turns_for_binding_from_cues
    ):
        band = "high"
        state = "corroborated"
    elif binds >= 1:
        band = "med"
        state = "asserted"
    else:
        band = profile.binding_confidence_band
        state = profile.identity_state
    style_cues = tuple(cue for cue in merged if cue.cue_kind == "style")
    relationship = tuple(
        dict.fromkeys(
            [*profile.known_relationship_facts, *[cue.content_summary for cue in merged if cue.cue_kind == "relationship"]]
        )
    )[-hyperparams.max_relationship_facts :]
    return replace(
        profile,
        continuity_evidence=tuple(merged),
        style_habit_cues=style_cues[-hyperparams.max_continuity_cues :],
        known_relationship_facts=relationship,
        binding_confidence_band=band,
        identity_state=state,
        last_updated_turn_id=turn_id,
    )


def _build_entity_binding_context(*, profile: IdentityProfile) -> dict[str, object]:
    alias = profile.aliases_observed[-1].alias_text if profile.aliases_observed else ""
    return {
        "current_user_id": profile.user_id,
        "claimed_alias": alias,
        "identity_state": profile.identity_state,
        "binding_confidence_band": profile.binding_confidence_band,
    }


def _bounded_snapshot(
    *,
    profile: IdentityProfile,
    current_turn_quotes: Mapping[str, str],
    m11_readonly_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "user_id": profile.user_id,
        "display_name": profile.display_name,
        "current_turn_quotes": dict(current_turn_quotes),
        "active_aliases": [
            {
                "alias_text": row.alias_text,
                "modality": row.modality,
                "seen_count": row.seen_count,
            }
            for row in profile.aliases_observed[-8:]
        ],
        "recent_continuity_cues": [
            {
                "cue_id": cue.cue_id,
                "cue_kind": cue.cue_kind,
                "supports": cue.supports,
                "confidence_band": cue.confidence_band,
            }
            for cue in profile.continuity_evidence[-12:]
        ],
        "m11_readonly_summary": dict(m11_readonly_summary),
    }
