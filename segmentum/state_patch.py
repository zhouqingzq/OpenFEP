"""M9.0 StatePatchProposal and StateCommitEvent MVP.

Durable self-cognition changes must go through an auditable patch/commit
path instead of opaque side effects.  FEP, memory conflict, self-thought,
and outcome feedback propose state changes; the cognitive loop (via a
named state owner) accepts, rejects, or defers them.

Uses first for ``SubjectState`` and selected ``SelfModel`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence
from uuid import uuid4


PatchOperation = Literal["set", "merge", "remove", "increment"]
CommitDecision = Literal["accepted", "rejected", "deferred"]


@dataclass(frozen=True)
class StatePatchProposal:
    """A proposal to change a durable self-state field.

    Proposals are emitted by FEP, memory conflict detection, self-thought,
    or outcome feedback.  They must be committed by a named state owner
    before they take effect.
    """

    proposal_id: str = ""
    target_state: str = ""  # e.g. "SubjectState", "SelfModel"
    operation: PatchOperation = "set"
    field_path: str = ""  # dotted path, e.g. "identity.name" or "preferences"
    value_summary: str = ""  # human-readable summary of the proposed value
    source_event_id: str = ""  # the event that triggered this proposal
    reason: str = ""
    confidence: float = 0.0
    ttl: int = 3  # cycles before the proposal expires if uncommitted

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "target_state": self.target_state,
            "operation": self.operation,
            "field_path": self.field_path,
            "value_summary": self.value_summary,
            "source_event_id": self.source_event_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StatePatchProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            target_state=str(payload.get("target_state", "")),
            operation=str(payload.get("operation", "set")),
            field_path=str(payload.get("field_path", "")),
            value_summary=str(payload.get("value_summary", "")),
            source_event_id=str(payload.get("source_event_id", "")),
            reason=str(payload.get("reason", "")),
            confidence=float(payload.get("confidence", 0.0)),
            ttl=int(payload.get("ttl", 3)),
        )


@dataclass(frozen=True)
class StateCommitEvent:
    """An auditable record of a state patch being committed or rejected.

    Every durable self-cognition change must produce one of these so that
    the decision, owner, and reason are preserved in trace.
    """

    commit_id: str = ""
    proposal_id: str = ""
    accepted: bool = False
    decision: CommitDecision = "rejected"
    owner: str = ""  # named state owner, e.g. "subject_state_owner"
    reason: str = ""
    committed_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "commit_id": self.commit_id,
            "proposal_id": self.proposal_id,
            "accepted": self.accepted,
            "decision": self.decision,
            "owner": self.owner,
            "reason": self.reason,
            "committed_summary": self.committed_summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateCommitEvent":
        return cls(
            commit_id=str(payload.get("commit_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            accepted=bool(payload.get("accepted", False)),
            decision=str(payload.get("decision", "rejected")),
            owner=str(payload.get("owner", "")),
            reason=str(payload.get("reason", "")),
            committed_summary=str(payload.get("committed_summary", "")),
        )


@dataclass
class StatePatchLog:
    """Audit trail of all proposals and their commit outcomes."""

    proposals: list[StatePatchProposal] = field(default_factory=list)
    commits: list[StateCommitEvent] = field(default_factory=list)
    _proposal_created_at_cycle: dict[str, int] = field(default_factory=dict)

    def record_proposal(self, proposal: StatePatchProposal, *, cycle: int = 0) -> None:
        self.proposals.append(proposal)
        self._proposal_created_at_cycle[proposal.proposal_id] = cycle

    def record_commit(self, commit: StateCommitEvent) -> None:
        self.commits.append(commit)

    def pending_proposals(self) -> list[StatePatchProposal]:
        committed_ids = {c.proposal_id for c in self.commits}
        return [p for p in self.proposals if p.proposal_id not in committed_ids]

    def rejected_proposals(self) -> list[tuple[StatePatchProposal, StateCommitEvent]]:
        result: list[tuple[StatePatchProposal, StateCommitEvent]] = []
        prop_map = {p.proposal_id: p for p in self.proposals}
        for commit in self.commits:
            if not commit.accepted and commit.proposal_id in prop_map:
                result.append((prop_map[commit.proposal_id], commit))
        return result

    def accepted_patches(self) -> list[tuple[StatePatchProposal, StateCommitEvent]]:
        result: list[tuple[StatePatchProposal, StateCommitEvent]] = []
        prop_map = {p.proposal_id: p for p in self.proposals}
        for commit in self.commits:
            if commit.accepted and commit.proposal_id in prop_map:
                result.append((prop_map[commit.proposal_id], commit))
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "commits": [c.to_dict() for c in self.commits],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StatePatchLog":
        proposals_raw = payload.get("proposals", [])
        proposals = [
            StatePatchProposal.from_dict(p)
            for p in (proposals_raw if isinstance(proposals_raw, list) else [])
            if isinstance(p, Mapping)
        ]
        commits_raw = payload.get("commits", [])
        commits = [
            StateCommitEvent.from_dict(c)
            for c in (commits_raw if isinstance(commits_raw, list) else [])
            if isinstance(c, Mapping)
        ]
        return cls(proposals=proposals, commits=commits)


# ── Commit function ─────────────────────────────────────────────────────


def make_state_commit(
    proposal: StatePatchProposal,
    *,
    accepted: bool,
    owner: str,
    reason: str = "",
    committed_summary: str = "",
) -> StateCommitEvent:
    """Produce a StateCommitEvent for a proposal.

    The owner must be a named state owner from the StateOwnership table.
    Rejected or deferred patches are visible in trace.
    """
    decision: CommitDecision = (
        "accepted" if accepted
        else "rejected" if reason else "deferred"
    )
    return StateCommitEvent(
        commit_id=f"commit:{uuid4().hex[:12]}",
        proposal_id=proposal.proposal_id,
        accepted=accepted,
        decision=decision,
        owner=owner,
        reason=reason or ("accepted" if accepted else "rejected without reason"),
        committed_summary=committed_summary or proposal.value_summary,
    )


def propose_patch(
    *,
    target_state: str,
    field_path: str,
    value_summary: str,
    source_event_id: str = "",
    reason: str = "",
    confidence: float = 0.5,
    operation: PatchOperation = "set",
    ttl: int = 3,
) -> StatePatchProposal:
    """Create a StatePatchProposal with a deterministic proposal_id."""
    proposal_id = (
        f"patch:{target_state}:{field_path}:{uuid4().hex[:8]}"
    )
    return StatePatchProposal(
        proposal_id=proposal_id,
        target_state=target_state,
        operation=operation,
        field_path=field_path,
        value_summary=value_summary,
        source_event_id=source_event_id,
        reason=reason,
        confidence=confidence,
        ttl=ttl,
    )


def expire_proposals(
    log: StatePatchLog,
    *,
    current_cycle: int = 0,
    owner: str = "system",
) -> list[StateCommitEvent]:
    """Expire proposals whose effective TTL has been exhausted.

    Effective TTL = proposal.ttl - (current_cycle - cycle_proposal_was_recorded).
    Proposals with effective_ttl <= 0 are rejected with an expiration reason.
    """
    pending = log.pending_proposals()
    expired_commits: list[StateCommitEvent] = []
    for proposal in pending:
        created = log._proposal_created_at_cycle.get(proposal.proposal_id, current_cycle)
        effective_ttl = proposal.ttl - (current_cycle - created)
        if effective_ttl <= 0:
            commit = make_state_commit(
                proposal,
                accepted=False,
                owner=owner,
                reason=f"expired: ttl={proposal.ttl}, {current_cycle - created} cycles elapsed at cycle {current_cycle}",
                committed_summary=proposal.value_summary,
            )
            log.record_commit(commit)
            expired_commits.append(commit)
    return expired_commits
