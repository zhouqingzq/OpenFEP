"""M9.0 runtime: SubjectState patches via StatePatchProposal / StateCommitEvent."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Mapping

from .state_patch import (
    StatePatchProposal,
    make_state_commit,
    propose_patch,
)
from .subject_state import SubjectState

if TYPE_CHECKING:
    from .agent import SegmentAgent


SUBJECT_OWNER = "subject_state_owner"


def proposals_from_memory_interference(
    *,
    interference_payload: Mapping[str, object],
    source_event_id: str,
) -> list[StatePatchProposal]:
    """Emit durable-state proposals when retrieval shows interference."""
    if not interference_payload.get("detected"):
        return []
    sev = float(interference_payload.get("severity", 0.0) or 0.0)
    delta_id = round(0.03 + 0.07 * sev, 4)
    delta_maint = round(0.025 + 0.05 * sev, 4)
    conf = min(1.0, 0.38 + sev * 0.45)
    return [
        propose_patch(
            target_state="SubjectState",
            field_path="identity_tension_level",
            operation="increment",
            value_summary=str(delta_id),
            source_event_id=source_event_id,
            reason="m9_memory_interference",
            confidence=conf,
            ttl=4,
        ),
        propose_patch(
            target_state="SubjectState",
            field_path="maintenance_pressure",
            operation="increment",
            value_summary=str(delta_maint),
            source_event_id=source_event_id,
            reason="m9_memory_interference",
            confidence=conf,
            ttl=4,
        ),
    ]


def _apply_increment(subject: SubjectState, proposal: StatePatchProposal) -> SubjectState:
    delta = float(proposal.value_summary)
    if proposal.field_path == "identity_tension_level":
        return replace(
            subject,
            identity_tension_level=min(1.0, float(subject.identity_tension_level) + delta),
        )
    if proposal.field_path == "maintenance_pressure":
        return replace(
            subject,
            maintenance_pressure=min(1.0, float(subject.maintenance_pressure) + delta),
        )
    return subject


def process_subject_patches_for_turn(
    agent: SegmentAgent,
    proposals: list[StatePatchProposal],
    publish_event: Callable[..., None],
    *,
    cycle: int,
    interference_event_id: str,
) -> list[dict[str, object]]:
    """Record proposals, optionally commit SubjectState increments, publish commits."""
    trace: list[dict[str, object]] = []
    log = agent.state_patch_log
    for proposal in proposals:
        log.record_proposal(proposal, cycle=cycle)
        publish_event(
            "StatePatchProposal",
            "m9_state_patch_runtime",
            dict(proposal.to_dict()),
            salience=0.82,
            priority=0.88,
            ttl=2,
        )
        accept = (
            proposal.confidence >= 0.45
            and proposal.target_state == "SubjectState"
            and proposal.operation == "increment"
            and proposal.field_path in ("identity_tension_level", "maintenance_pressure")
        )
        if accept:
            agent.subject_state = _apply_increment(agent.subject_state, proposal)
            commit = make_state_commit(
                proposal,
                accepted=True,
                owner=SUBJECT_OWNER,
                reason="m9_confidence_gate",
                committed_summary=f"{proposal.field_path}+{proposal.value_summary}",
            )
        else:
            commit = make_state_commit(
                proposal,
                accepted=False,
                owner=SUBJECT_OWNER,
                reason="m9_rejected_policy",
                committed_summary=proposal.value_summary,
            )
        log.record_commit(commit)
        publish_event(
            "StateCommitEvent",
            SUBJECT_OWNER,
            dict(commit.to_dict()),
            salience=0.75,
            priority=0.85,
            ttl=2,
        )
        trace.append({"proposal": proposal.to_dict(), "commit": commit.to_dict()})
    if trace:
        agent.latest_m9_state_patch_trace = {
            "interference_event_id": interference_event_id,
            "turn_trace": trace,
        }
    return trace
