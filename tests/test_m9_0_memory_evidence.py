"""M9.0 R1: Unified Memory Evidence Surface tests."""

import pytest

from segmentum.memory_anchored import AnchoredMemoryItem
from segmentum.memory_evidence import (
    MemoryEvidence,
    evidence_from_anchored_item,
    evidence_from_hypothesis,
    evidence_from_retrieval,
    evidence_summary_for_prompt,
    unify_evidence,
)


# ── R1.1: Evidence structure completeness ────────────────────────────────

def test_memory_evidence_has_all_fifteen_fields():
    """MemoryEvidence exposes all 15 required fields."""
    ev = MemoryEvidence()
    fields = [
        "memory_id", "memory_class", "source_turn_id", "source_utterance_id",
        "speaker", "status", "confidence", "retrieval_score", "cue_match",
        "salience", "value_score", "decay_state", "conflict_status",
        "permitted_use", "content_summary",
    ]
    for f in fields:
        assert hasattr(ev, f), f"Missing field: {f}"


def test_memory_evidence_classification_properties():
    """is_user_statement vs is_verified_fact distinguish source reliability."""
    user_assertion = MemoryEvidence(
        memory_class="user_assertion", status="asserted", confidence=0.9,
    )
    verified = MemoryEvidence(
        memory_class="anchored_fact", status="corroborated", confidence=0.9,
    )
    hypothesis = MemoryEvidence(
        memory_class="inferred_hypothesis", status="hypothesis", confidence=0.3,
    )

    assert user_assertion.is_user_statement
    assert not user_assertion.is_verified_fact
    assert verified.is_verified_fact
    assert not verified.is_user_statement
    assert hypothesis.is_hypothesis
    assert not hypothesis.is_verified_fact


def test_can_be_asserted_excludes_low_confidence():
    """Low-confidence evidence cannot be asserted as fact."""
    low = MemoryEvidence(
        memory_class="anchored_fact", status="asserted", confidence=0.4,
        permitted_use="explicit_fact",
    )
    assert not low.can_be_asserted


def test_can_be_asserted_excludes_retracted():
    """Retracted evidence can never be asserted."""
    retracted = MemoryEvidence(
        memory_class="anchored_fact", status="retracted", confidence=0.9,
        permitted_use="explicit_fact",
    )
    assert not retracted.can_be_asserted


def test_can_be_asserted_excludes_unresolved_conflict():
    """Conflicting evidence cannot be asserted directly."""
    conflicted = MemoryEvidence(
        memory_class="anchored_fact", status="asserted", confidence=0.9,
        permitted_use="explicit_fact", conflict_status="unresolved",
    )
    assert not conflicted.can_be_asserted


def test_can_be_asserted_excludes_pruned():
    """Pruned/decayed evidence cannot be asserted."""
    pruned = MemoryEvidence(
        memory_class="anchored_fact", status="asserted", confidence=0.9,
        permitted_use="explicit_fact", decay_state="pruned",
    )
    assert not pruned.can_be_asserted


def test_should_be_cautious_for_hypothesis():
    """Hypotheses trigger cautious stance."""
    hyp = MemoryEvidence(
        memory_class="inferred_hypothesis", permitted_use="cautious_hypothesis",
    )
    assert hyp.should_be_cautious


# ── R1.2: Converters produce correct MemoryEvidence ──────────────────────

def test_evidence_from_anchored_item_user_assertion():
    """User-asserted anchored items become user_assertion evidence class."""
    item = AnchoredMemoryItem(
        memory_id="m007",
        speaker="user",
        utterance_id="u001",
        turn_id="turn_0001",
        proposition="用户叫周青",
        status="asserted",
        confidence=1.0,
        visibility="explicit",
        memory_type="user_asserted_fact",
    )
    ev = evidence_from_anchored_item(item)
    assert ev.memory_class == "user_assertion"
    assert ev.speaker == "user"
    assert ev.permitted_use == "explicit_fact"
    assert ev.content_summary == "用户叫周青"
    assert ev.confidence == 1.0


def test_evidence_from_anchored_item_hypothesis():
    """Hypothesis-type anchored items map to cautious_hypothesis permitted use."""
    item = AnchoredMemoryItem(
        speaker="system",
        proposition="用户可能是老师",
        status="hypothesis",
        visibility="explicit",
        memory_type="system_inferred_hypothesis",
        confidence=0.4,
    )
    ev = evidence_from_anchored_item(item)
    assert ev.memory_class == "inferred_hypothesis"
    assert ev.permitted_use == "cautious_hypothesis"


def test_evidence_from_anchored_item_agent_utterance():
    """Agent self-utterances go to strategy_only, never explicit."""
    item = AnchoredMemoryItem(
        speaker="agent",
        proposition="我说过我喜欢安静",
        status="asserted",
        visibility="strategy_only",
        memory_type="agent_self_utterance",
        confidence=0.5,
    )
    ev = evidence_from_anchored_item(item)
    assert ev.memory_class == "agent_self_utterance"
    assert ev.permitted_use == "strategy_only"


def test_evidence_from_retrieval_maps_fields():
    """Retrieval result dicts are mapped to MemoryEvidence correctly."""
    entry = {
        "episode_id": "ep_042",
        "speaker": "user",
        "turn_id": "turn_0003",
        "proposition": "用户喜欢Python",
        "confidence": 0.8,
        "retrieval_score": 0.72,
        "value_score": 0.6,
        "salience": 0.7,
    }
    ev = evidence_from_retrieval(entry, cue_match="Python")
    assert ev.memory_class == "retrieved_episodic"
    assert ev.memory_id == "ep_042"
    assert ev.content_summary == "用户喜欢Python"
    assert ev.retrieval_score == 0.72
    assert ev.cue_match == "Python"


def test_evidence_from_hypothesis_is_low_confidence():
    """Inferred hypotheses always have cautious settings."""
    ev = evidence_from_hypothesis(
        "用户可能在北京工作", confidence=0.35, cue_match="工作地点",
    )
    assert ev.memory_class == "inferred_hypothesis"
    assert ev.status == "hypothesis"
    assert ev.confidence == 0.35
    assert ev.permitted_use == "cautious_hypothesis"
    assert ev.decay_state == "fading"


# ── R1.3: Unification ───────────────────────────────────────────────────

def test_memory_evidence_unifies_anchored_retrieved_and_hypothesis_sources():
    """Anchored facts, retrieved episodic, and hypotheses can coexist in
    one evidence list without losing source or confidence."""
    anchored = AnchoredMemoryItem(
        speaker="user", proposition="用户叫周青",
        status="asserted", visibility="explicit",
        memory_type="user_asserted_fact", confidence=1.0,
    )
    retrieval = {
        "episode_id": "ep_001", "proposition": "周青昨天说喜欢Python",
        "confidence": 0.75, "retrieval_score": 0.7,
    }

    combined = unify_evidence(
        anchored_items=[anchored],
        retrieval_results=[retrieval],
        hypotheses=["周青可能是开发者"],
    )

    assert len(combined) == 3

    classes = {e.memory_class for e in combined}
    assert "user_assertion" in classes
    assert "retrieved_episodic" in classes
    assert "inferred_hypothesis" in classes

    for ev in combined:
        assert ev.content_summary, f"Evidence {ev.memory_id} has no content_summary"
        assert ev.memory_class, f"Evidence {ev.memory_id} has no memory_class"
        assert ev.permitted_use, f"Evidence {ev.memory_id} has no permitted_use"


def test_unify_evidence_sorts_explicit_facts_first():
    """Explicit facts appear before hypotheses and strategy-only items."""
    anchored = AnchoredMemoryItem(
        speaker="user", proposition="事实A",
        status="asserted", visibility="explicit",
        memory_type="user_asserted_fact", confidence=1.0,
    )
    hyp_item = AnchoredMemoryItem(
        speaker="system", proposition="假设B",
        status="hypothesis", memory_type="system_inferred_hypothesis",
        confidence=0.3,
    )

    combined = unify_evidence(anchored_items=[hyp_item, anchored])

    assert combined[0].permitted_use == "explicit_fact"
    assert combined[0].content_summary == "事实A"
    assert combined[1].permitted_use == "cautious_hypothesis"


# ── R1.4: Prompt safety ─────────────────────────────────────────────────

def test_evidence_summary_for_prompt_is_prompt_safe():
    """evidence_summary_for_prompt() produces no raw object dumps."""
    evidence = [
        MemoryEvidence(
            memory_id="m1", memory_class="anchored_fact",
            permitted_use="explicit_fact", content_summary="用户叫周青",
            confidence=1.0, speaker="user",
        ),
        MemoryEvidence(
            memory_id="m2", memory_class="inferred_hypothesis",
            permitted_use="cautious_hypothesis", content_summary="用户可能是老师",
            confidence=0.3, decay_state="fading",
        ),
    ]
    text = evidence_summary_for_prompt(evidence)
    assert "[Memory Evidence]" in text
    assert "用户叫周青" in text
    assert "用户可能是老师" in text
    assert "memory_id" not in text
    assert "source_turn_id" not in text
    assert "source_utterance_id" not in text


def test_evidence_summary_for_prompt_filters_forbidden():
    """Forbidden evidence items are excluded from prompt summary."""
    evidence = [
        MemoryEvidence(
            memory_id="m1", permitted_use="forbidden", content_summary="秘密",
        ),
        MemoryEvidence(
            memory_id="m2", permitted_use="explicit_fact", content_summary="公开信息",
            confidence=1.0,
        ),
    ]
    text = evidence_summary_for_prompt(evidence)
    assert "公开信息" in text
    assert "秘密" not in text


# ── R1.5: Serialization round-trip ──────────────────────────────────────

def test_memory_evidence_round_trip():
    """MemoryEvidence serializes and deserializes without loss."""
    original = MemoryEvidence(
        memory_id="ev_001",
        memory_class="anchored_fact",
        source_turn_id="turn_0001",
        source_utterance_id="utt_001",
        speaker="user",
        status="corroborated",
        confidence=0.95,
        retrieval_score=0.82,
        cue_match="名字",
        salience=0.7,
        value_score=0.65,
        decay_state="active",
        conflict_status="none",
        permitted_use="explicit_fact",
        content_summary="用户的名字是周青",
    )
    restored = MemoryEvidence.from_dict(original.to_dict())
    for field_name in [
        "memory_id", "memory_class", "source_turn_id", "source_utterance_id",
        "speaker", "status", "confidence", "retrieval_score", "cue_match",
        "salience", "value_score", "decay_state", "conflict_status",
        "permitted_use", "content_summary",
    ]:
        assert getattr(restored, field_name) == getattr(original, field_name), (
            f"Field {field_name} mismatch"
        )


def test_generation_can_distinguish_user_statement_from_verified_fact():
    """Generation code can distinguish 'user said X' from 'X is verified'."""
    user_said = MemoryEvidence(
        memory_class="user_assertion", status="asserted", confidence=0.9,
        permitted_use="explicit_fact", content_summary="用户说喜欢编程",
    )
    verified = MemoryEvidence(
        memory_class="anchored_fact", status="corroborated", confidence=0.95,
        permitted_use="explicit_fact", content_summary="用户完成了M8",
    )

    assert user_said.is_user_statement
    assert not user_said.is_verified_fact
    assert verified.is_verified_fact
    assert not verified.is_user_statement

    # Both can be asserted but carry different provenance
    assert user_said.can_be_asserted
    assert verified.can_be_asserted
    assert user_said.memory_class != verified.memory_class
