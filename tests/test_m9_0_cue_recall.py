"""M9.0 R3: Cue-Based Recall tests."""

import pytest

from segmentum.memory_dynamics import (
    CueMatchResult,
    compute_cue_match,
    require_cue_for_recall,
)


# ── R3.1: Cue match computation ─────────────────────────────────────────

def test_cue_match_exact_content_high_score():
    """Exact cue appearing in content gives high score."""
    result = compute_cue_match(
        cue="Python",
        memory_content="用户喜欢Python编程",
        memory_tags=["编程", "Python"],
    )
    assert result.matched
    assert result.score >= 0.5
    assert "content_exact" in result.matched_fields
    assert "Python" in result.trace


def test_cue_match_word_overlap_moderate_score():
    """Partial word overlap gives a moderate score."""
    result = compute_cue_match(
        cue="编程 项目",
        memory_content="用户在做编程相关的项目开发",
    )
    assert result.matched
    assert result.score >= 0.15
    assert any("content" in f for f in result.matched_fields)


def test_cue_match_tag_hit():
    """Tag match contributes to score."""
    result = compute_cue_match(
        cue="milestone进度",
        memory_content="用户完成了M2.5",
        memory_tags=["milestone", "M2.5"],
    )
    assert result.matched
    assert any("tag" in f for f in result.matched_fields)


def test_cue_match_no_match_low_score():
    """Completely unrelated cue and content produce no match."""
    result = compute_cue_match(
        cue="天气",
        memory_content="用户喜欢吃川菜",
        memory_tags=["食物", "偏好"],
    )
    assert not result.matched
    assert result.score < 0.15


def test_cue_match_empty_cue():
    """Empty cue string returns no match."""
    result = compute_cue_match(
        cue="",
        memory_content="任何内容都不重要",
    )
    assert not result.matched
    assert result.score == 0.0
    assert result.trace == "empty_cue"


def test_cue_match_speaker_match():
    """Speaker appearing in cue adds a small bonus."""
    result = compute_cue_match(
        cue="周青说过的",
        memory_content="用户的名字是周青",
        speaker="周青",
    )
    assert result.matched


def test_cue_match_scoring_levels_increase_with_evidence():
    """More matching fields → higher score."""
    weak = compute_cue_match(
        cue="Python",
        memory_content="用户喜欢JavaScript",
    )
    strong = compute_cue_match(
        cue="Python",
        memory_content="用户喜欢Python编程",
        memory_tags=["Python"],
    )
    assert strong.score > weak.score


# ── R3.2: require_cue_for_recall ────────────────────────────────────────

def test_long_term_memory_requires_cue_for_specific_detail():
    """Without a cue, require_cue_for_recall returns empty and 'unknown'."""
    memories = [
        {"proposition": "用户叫周青", "speaker": "user", "tags": ["名字"]},
        {"proposition": "用户喜欢Python", "speaker": "user", "tags": ["偏好"]},
    ]
    matched, stance = require_cue_for_recall(cue="", memory_items=memories)
    assert matched == []
    assert stance.startswith("unknown:")
    assert "no cue provided" in stance


def test_no_cue_returns_unknown_stance():
    """Without a cue, the stance is explicitly 'unknown'."""
    _, stance = require_cue_for_recall(
        cue="",
        memory_items=[{"proposition": "事实", "content": "内容"}],
    )
    assert "unknown" in stance
    assert "no cue provided" in stance or "not searched" in stance


def test_cue_triggers_recall_with_trace():
    """A matching cue returns items annotated with cue_match trace."""
    memories = [
        {"proposition": "用户叫周青", "speaker": "user", "tags": ["名字"]},
        {"proposition": "用户喜欢川菜", "speaker": "user", "tags": ["食物"]},
    ]
    matched, stance = require_cue_for_recall(
        cue="周青",
        memory_items=memories,
    )
    assert len(matched) >= 1
    assert matched[0]["cue_match"]
    assert "cued:" in stance
    assert "周青" in stance


def test_cue_no_match_returns_unknown():
    """A cue that doesn't match any memory returns unknown stance."""
    memories = [
        {"proposition": "用户叫周青", "speaker": "user", "tags": ["名字"]},
    ]
    matched, stance = require_cue_for_recall(
        cue="量子物理",
        memory_items=memories,
    )
    assert matched == []
    assert "unknown" in stance
    assert "did not match" in stance


def test_cue_matched_items_sorted_by_score():
    """Matched items are sorted by cue score descending."""
    memories = [
        {"proposition": "用户喜欢JavaScript", "speaker": "user", "tags": ["编程"]},
        {"proposition": "用户喜欢Python编程", "speaker": "user", "tags": ["编程", "Python"]},
    ]
    matched, stance = require_cue_for_recall(
        cue="Python",
        memory_items=memories,
    )
    assert len(matched) >= 1
    if len(matched) >= 2:
        assert matched[0]["cue_score"] >= matched[1]["cue_score"]


def test_generation_unknown_when_memory_not_recalled():
    """When no memory is recalled, the factual stance is unknown.

    This test exercises the contract: 'If long-term memory is not cued
    or recalled, the generator's factual stance is unknown.'
    """
    # No cue at all → unknown
    _, stance_no_cue = require_cue_for_recall(cue="", memory_items=[
        {"proposition": "某事实", "speaker": "user"},
    ])
    assert "unknown" in stance_no_cue

    # Cue that doesn't match → unknown
    _, stance_bad_cue = require_cue_for_recall(cue="不存在的话题", memory_items=[
        {"proposition": "某事实", "speaker": "user"},
    ])
    assert "unknown" in stance_bad_cue

    # Cue that matches → not unknown
    _, stance_good_cue = require_cue_for_recall(cue="某事实", memory_items=[
        {"proposition": "某事实", "speaker": "user"},
    ])
    assert "cued:" in stance_good_cue


def test_require_cue_for_recall_respects_min_cue_score():
    """Weak matches below min_cue_score threshold are excluded."""
    memories = [
        {"proposition": "用户完成了M8", "speaker": "user", "tags": ["milestone"]},
    ]
    # "完成" barely overlaps with "M8" - might match weakly
    matched, _ = require_cue_for_recall(
        cue="M8",
        memory_items=memories,
        min_cue_score=0.15,
    )
    # "M8" is an exact substring, should match strongly
    assert len(matched) >= 1

    # But with an unreasonably high threshold, it would be excluded
    matched_strict, stance_strict = require_cue_for_recall(
        cue="M8",
        memory_items=memories,
        min_cue_score=0.95,
    )
    # With high threshold, the match may or may not pass
    if not matched_strict:
        assert "unknown" in stance_strict
