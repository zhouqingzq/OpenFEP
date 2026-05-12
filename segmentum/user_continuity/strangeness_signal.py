"""M12.0 identity strangeness signal and bus payload conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from segmentum.cognitive_events import make_cognitive_event

from .hyperparams import DEFAULT_HYPERPARAMS, M12Hyperparams
from .identity_conflict_detector import ConflictRecord
from .identity_profile import ContinuityCue, IdentityProfile


@dataclass(frozen=True)
class IdentityStrangenessSignal:
    event_id: str
    event_type: str
    source: str
    trigger: str
    target_alias: str
    target_user_id: str
    strangeness_band: str
    proposed_intervention: str
    evidence_event_ids: tuple[str, ...]
    budget_cost: int
    ttl: int

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "trigger": self.trigger,
            "target_alias": self.target_alias,
            "target_user_id": self.target_user_id,
            "strangeness_band": self.strangeness_band,
            "proposed_intervention": self.proposed_intervention,
            "evidence_event_ids": list(self.evidence_event_ids),
            "budget_cost": int(self.budget_cost),
            "ttl": int(self.ttl),
        }


def build_strangeness_signal(
    *,
    turn_id: str,
    claim_alias: str,
    claimant_user_id: str,
    profile: IdentityProfile,
    conflicts: Sequence[ConflictRecord],
    current_turn_cues: Sequence[ContinuityCue],
    signal_count_this_turn: int,
    hyperparams: M12Hyperparams = DEFAULT_HYPERPARAMS,
) -> IdentityStrangenessSignal | None:
    if signal_count_this_turn >= hyperparams.strangeness_signal_rate_limit_per_turn:
        return None
    major = [record for record in conflicts if record.severity_band == "major"]
    med_or_high_contradictions = [
        cue
        for cue in current_turn_cues
        if cue.supports == "contradicts" and cue.confidence_band in {"med", "high"}
    ]
    if major:
        trigger = "identity_conflict"
        band = "high"
        intervention = "ask"
        evidence_ids = tuple(record.conflict_id for record in major)
    elif med_or_high_contradictions and profile.identity_state == "corroborated":
        trigger = "continuity_contradiction"
        band = "med"
        intervention = "probe"
        evidence_ids = tuple(cue.cue_id for cue in med_or_high_contradictions)
    else:
        return None
    return IdentityStrangenessSignal(
        event_id=f"ids:{turn_id}:{claimant_user_id}:{len(evidence_ids)}",
        event_type="IdentityStrangenessSignal",
        source="user_continuity",
        trigger=trigger,
        target_alias=claim_alias,
        target_user_id=claimant_user_id,
        strangeness_band=band,
        proposed_intervention=intervention,
        evidence_event_ids=evidence_ids,
        budget_cost=hyperparams.strangeness_budget_cost,
        ttl=hyperparams.strangeness_ttl_turns,
    )


def signal_to_self_thought_event(
    signal: IdentityStrangenessSignal,
    *,
    session_id: str,
    persona_id: str,
    turn_id: str,
    cycle: int,
    sequence_index: int,
) -> object:
    salience_map = {"low": 0.45, "med": 0.7, "high": 0.9}
    priority_map = {"low": 0.45, "med": 0.7, "high": 0.9}
    return make_cognitive_event(
        event_type="SelfThoughtEvent",
        turn_id=turn_id,
        cycle=cycle,
        session_id=session_id,
        persona_id=persona_id,
        source="user_continuity",
        sequence_index=sequence_index,
        salience=salience_map.get(signal.strangeness_band, 0.6),
        priority=priority_map.get(signal.strangeness_band, 0.6),
        ttl=signal.ttl,
        payload={
            "trigger": signal.trigger,
            "target_gap_id": f"identity:{signal.target_alias}",
            "confidence": signal.strangeness_band,
            "subtype": signal.event_type,
            "proposed_intervention": signal.proposed_intervention,
            "evidence_event_ids": list(signal.evidence_event_ids),
            "budget_cost": int(signal.budget_cost),
        },
    )
