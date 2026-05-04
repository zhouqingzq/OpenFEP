"""M9.0 R5: StatePatchProposal / StateCommitEvent MVP tests."""

import pytest

from segmentum.state_patch import (
    StateCommitEvent,
    StatePatchLog,
    StatePatchProposal,
    expire_proposals,
    make_state_commit,
    propose_patch,
)


# ── R5.1: Proposal creation ─────────────────────────────────────────────

def test_propose_patch_creates_valid_proposal():
    """propose_patch() creates proposals with all required fields."""
    proposal = propose_patch(
        target_state="SubjectState",
        field_path="identity.preferred_name",
        value_summary="用户希望被叫'老周'",
        source_event_id="evt_042",
        reason="用户明确要求称呼偏好",
        confidence=0.9,
    )
    assert proposal.proposal_id.startswith("patch:")
    assert proposal.target_state == "SubjectState"
    assert proposal.field_path == "identity.preferred_name"
    assert proposal.operation == "set"
    assert proposal.confidence == 0.9
    assert proposal.ttl == 3
    assert proposal.source_event_id == "evt_042"


def test_propose_patch_with_merge_operation():
    """Merge operation is supported for list/dict fields."""
    proposal = propose_patch(
        target_state="SelfModel",
        field_path="preferences",
        value_summary="添加偏好：安静的环境",
        operation="merge",
        confidence=0.7,
    )
    assert proposal.operation == "merge"


# ── R5.2: Commit creation ───────────────────────────────────────────────

def test_memory_patch_proposal_requires_commit_owner():
    """A StatePatchProposal alone does NOT mutate state.

    The commit must have a named owner from the StateOwnership table.
    """
    proposal = propose_patch(
        target_state="SubjectState",
        field_path="identity.name",
        value_summary="用户的名字是周青",
        confidence=0.95,
    )

    # The proposal exists but has NOT been committed yet
    log = StatePatchLog()
    log.record_proposal(proposal)

    pending = log.pending_proposals()
    assert len(pending) == 1
    assert pending[0].proposal_id == proposal.proposal_id

    # No commit yet → proposal is pending, not applied
    assert len(log.commits) == 0


def test_commit_with_named_owner():
    """Commit must name the state owner."""
    proposal = propose_patch(
        target_state="SubjectState",
        field_path="identity.name",
        value_summary="周青",
        confidence=0.95,
    )
    commit = make_state_commit(
        proposal,
        accepted=True,
        owner="subject_state_owner",
        reason="用户自己陈述了名字",
        committed_summary="SubjectState.identity.name = '周青'",
    )
    assert commit.accepted
    assert commit.owner == "subject_state_owner"
    assert commit.decision == "accepted"
    assert commit.proposal_id == proposal.proposal_id


def test_rejected_commit_visible_in_trace():
    """Rejected patches are recorded in the audit log, not silently dropped."""
    proposal = propose_patch(
        target_state="SelfModel",
        field_path="identity.narrative",
        value_summary="低置信度假设",
        confidence=0.2,
    )
    commit = make_state_commit(
        proposal,
        accepted=False,
        owner="slow_persona_model_owner",
        reason="置信度过低，不足以修改SelfModel",
    )
    assert not commit.accepted
    assert commit.decision == "rejected"
    assert commit.owner == "slow_persona_model_owner"
    assert "置信度过低" in commit.reason


def test_deferred_commit_visible_in_trace():
    """Deferred patches remain pending and visible."""
    proposal = propose_patch(
        target_state="SubjectState",
        field_path="identity.preferred_name",
        value_summary="待确认：称呼偏好",
        confidence=0.5,
    )
    commit = make_state_commit(
        proposal,
        accepted=False,
        owner="subject_state_owner",
        reason="",
    )
    assert commit.decision == "deferred"
    assert not commit.accepted


# ── R5.3: Patch audit log ───────────────────────────────────────────────

def test_patch_log_tracks_proposals_and_commits():
    """StatePatchLog records both proposals and their commit outcomes."""
    log = StatePatchLog()

    p1 = propose_patch(
        target_state="SubjectState", field_path="a",
        value_summary="变更A", confidence=0.9,
    )
    p2 = propose_patch(
        target_state="SelfModel", field_path="b",
        value_summary="变更B", confidence=0.3,
    )

    log.record_proposal(p1)
    log.record_proposal(p2)

    assert len(log.proposals) == 2
    assert len(log.pending_proposals()) == 2

    c1 = make_state_commit(p1, accepted=True, owner="owner_a", reason="good")
    log.record_commit(c1)

    assert len(log.pending_proposals()) == 1
    assert len(log.accepted_patches()) == 1

    c2 = make_state_commit(p2, accepted=False, owner="owner_b", reason="low conf")
    log.record_commit(c2)

    assert len(log.pending_proposals()) == 0
    assert len(log.accepted_patches()) == 1
    assert len(log.rejected_proposals()) == 1


def test_accepted_patches_are_auditable():
    """Accepted patches carry provenance: source, reason, confidence, owner."""
    proposal = propose_patch(
        target_state="SubjectState",
        field_path="identity.preferred_language",
        value_summary="用户偏好中文交流",
        source_event_id="event_chat_001",
        reason="用户在对话中表达了语言偏好",
        confidence=0.85,
    )
    log = StatePatchLog()
    log.record_proposal(proposal)
    commit = make_state_commit(
        proposal,
        accepted=True,
        owner="subject_state_owner",
        reason="用户明确陈述，置信度高",
    )
    log.record_commit(commit)

    accepted = log.accepted_patches()
    assert len(accepted) == 1
    prop, comm = accepted[0]
    assert prop.source_event_id == "event_chat_001"
    assert prop.confidence == 0.85
    assert comm.owner == "subject_state_owner"
    assert comm.accepted


def test_rejected_patches_visible_in_audit():
    """Rejected/deferred patches are visible — changes are not opaque."""
    proposal = propose_patch(
        target_state="SelfModel",
        field_path="commitments.promise",
        value_summary="承诺每天登录",
        confidence=0.2,
        source_event_id="evt_999",
    )
    log = StatePatchLog()
    log.record_proposal(proposal)
    commit = make_state_commit(
        proposal, accepted=False, owner="persona_owner",
        reason="低置信度推测，不写入SelfModel",
    )
    log.record_commit(commit)

    rejected = log.rejected_proposals()
    assert len(rejected) == 1
    _, comm = rejected[0]
    assert not comm.accepted
    assert "低置信度" in comm.reason


# ── R5.4: TTL expiration ────────────────────────────────────────────────

def test_expired_proposals_auto_rejected():
    """Proposals past their effective TTL are automatically rejected with trace."""
    log = StatePatchLog()

    # p1: ttl=1, created at cycle 0; at cycle 2, effective_ttl = 1-2 = -1 → expired
    p1 = propose_patch(
        target_state="SubjectState", field_path="x",
        value_summary="过期变更", confidence=0.5, ttl=1,
    )
    # p2: ttl=10, created at cycle 0; at cycle 2, effective_ttl = 10-2 = 8 → still valid
    p2 = propose_patch(
        target_state="SubjectState", field_path="y",
        value_summary="有效变更", confidence=0.8, ttl=10,
    )

    log.record_proposal(p1, cycle=0)
    log.record_proposal(p2, cycle=0)

    expired = expire_proposals(log, current_cycle=2, owner="system")
    assert len(expired) == 1
    assert expired[0].proposal_id == p1.proposal_id
    assert not expired[0].accepted
    assert "expired" in expired[0].reason

    pending = log.pending_proposals()
    assert len(pending) == 1
    assert pending[0].proposal_id == p2.proposal_id


# ── R5.5: Serialization ─────────────────────────────────────────────────

def test_proposal_round_trip():
    """StatePatchProposal serializes and deserializes without loss."""
    original = propose_patch(
        target_state="SubjectState",
        field_path="identity.name",
        value_summary="周青",
        source_event_id="evt_001",
        reason="用户自述",
        confidence=0.95,
        ttl=5,
    )
    restored = StatePatchProposal.from_dict(original.to_dict())
    assert restored.proposal_id == original.proposal_id
    assert restored.target_state == original.target_state
    assert restored.field_path == original.field_path
    assert restored.confidence == original.confidence
    assert restored.ttl == original.ttl


def test_commit_round_trip():
    """StateCommitEvent serializes and deserializes without loss."""
    original = make_state_commit(
        propose_patch(
            target_state="SubjectState",
            field_path="test",
            value_summary="test",
        ),
        accepted=True,
        owner="test_owner",
        reason="test reason",
        committed_summary="test summary",
    )
    restored = StateCommitEvent.from_dict(original.to_dict())
    assert restored.commit_id == original.commit_id
    assert restored.accepted == original.accepted
    assert restored.owner == original.owner
    assert restored.reason == original.reason


def test_patch_log_round_trip():
    """StatePatchLog serializes and deserializes with all entries."""
    log = StatePatchLog()
    p = propose_patch(
        target_state="SubjectState", field_path="f",
        value_summary="v", confidence=0.8,
    )
    log.record_proposal(p)
    c = make_state_commit(p, accepted=True, owner="o", reason="r")
    log.record_commit(c)

    restored = StatePatchLog.from_dict(log.to_dict())
    assert len(restored.proposals) == 1
    assert len(restored.commits) == 1
    assert restored.accepted_patches()[0][0].proposal_id == p.proposal_id


def test_patch_does_not_mutate_state_directly():
    """A StatePatchProposal is a proposal, not a mutation.

    Durable self-cognition changes are no longer opaque side effects.
    The proposal must be committed by a named owner.
    """
    proposal = propose_patch(
        target_state="SelfModel",
        field_path="identity.narrative",
        value_summary="假设用户是工程师",
        confidence=0.3,
    )

    # The proposal carries data but does not mutate anything
    assert proposal.target_state == "SelfModel"
    assert proposal.value_summary == "假设用户是工程师"

    # Without a commit, this data never becomes state
    log = StatePatchLog()
    log.record_proposal(proposal)
    assert len(log.pending_proposals()) == 1
    assert len(log.accepted_patches()) == 0

    # Only after a commit does it take effect
    commit = make_state_commit(
        proposal, accepted=True,
        owner="slow_persona_model_owner",
        reason="has corroboration across turns",
    )
    log.record_commit(commit)
    assert len(log.accepted_patches()) == 1
