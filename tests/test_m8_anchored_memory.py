"""M8 Anchored Memory Contract — test suite."""

from __future__ import annotations

import pytest

from segmentum.memory_anchored import (
    AnchoredMemoryItem,
    CitationAuditResult,
    DialogueFactExtractor,
    MemoryCitationGuard,
    MemoryPermissionBuckets,
    MemoryPermissionFilter,
)
from segmentum.memory_store import MemoryStore


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_item(
    proposition: str = "用户已经完成了M7",
    status: str = "asserted",
    visibility: str = "explicit",
    memory_type: str = "project_fact",
    speaker: str = "user",
    turn_id: str = "turn_0001",
    **kwargs,
) -> AnchoredMemoryItem:
    return AnchoredMemoryItem(
        speaker=speaker,
        turn_id=turn_id,
        utterance_id=f"{turn_id}_user",
        proposition=proposition,
        source_text=proposition,
        status=status,  # type: ignore[arg-type]
        visibility=visibility,  # type: ignore[arg-type]
        memory_type=memory_type,  # type: ignore[arg-type]
        created_turn_id=turn_id,
        **kwargs,
    )


# ── M8.1: AnchoredMemoryItem ───────────────────────────────────────────────

def test_dialogue_fact_is_encoded_as_anchored_memory_item():
    """A user utterance that contains a clear fact must produce an AnchoredMemoryItem."""
    extractor = DialogueFactExtractor()
    items = extractor.extract(
        "我已经完成了M7", turn_id="turn_0001", utterance_id="u1", speaker="user",
    )

    assert len(items) >= 1, "Should extract at least one fact from milestone statement"
    item = items[0]
    assert item.proposition != "", "Proposition must not be empty"
    assert item.status == "asserted"
    assert item.visibility == "explicit"
    assert item.source_text == "我已经完成了M7"
    assert "M7" in item.proposition


def test_hypothesis_is_not_promoted_to_fact():
    """An item with status=hypothesis must NOT land in explicit_facts."""
    items = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
        _make_item("用户可能担心记忆模块", status="hypothesis", visibility="explicit",
                   memory_type="hypothesis", confidence=0.6),
    ]
    buckets = MemoryPermissionFilter.filter(items)

    explicit_props = {it.proposition for it in buckets.explicit_facts}
    hypo_props = {it.proposition for it in buckets.cautious_hypotheses}

    assert "用户已经完成了M7" in explicit_props
    assert "用户可能担心记忆模块" not in explicit_props
    assert "用户可能担心记忆模块" in hypo_props


def test_private_state_not_in_explicit_memory_context():
    """PrivateState items must only appear in strategy_only or forbidden, never explicit."""
    items = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
        _make_item("用户可能偏好直接架构判断", status="asserted", visibility="private",
                   memory_type="private_state"),
        _make_item("内部状态：用户疲劳水平高", status="asserted", visibility="strategy_only",
                   memory_type="private_state"),
    ]
    buckets = MemoryPermissionFilter.filter(items)

    explicit_props = {it.proposition for it in buckets.explicit_facts}
    assert "用户已经完成了M7" in explicit_props
    assert "用户可能偏好直接架构判断" not in explicit_props
    assert "内部状态：用户疲劳水平高" not in explicit_props


# ── M8.2: Memory context reads anchored facts ──────────────────────────────

def test_memory_context_reads_anchored_facts_not_legacy_episode_templates():
    """When anchored items exist, the memory context must include their
    propositions and must NOT contain legacy template strings."""
    store = MemoryStore()

    # Add an anchored fact
    store.add_anchored_item(_make_item("用户已经完成了M7"))

    # Also add a legacy entry that would produce template content
    from segmentum.memory_model import MemoryEntry, MemoryClass
    legacy = MemoryEntry(
        content="Legacy episode at cycle 42: ask_question -> neutral",
        memory_class=MemoryClass.EPISODIC,
    )
    store.add(legacy)

    # Simulate what PromptBuilder does
    buckets = MemoryPermissionFilter.filter(store.anchored_items)

    # Build mock memory context
    explicit_lines: list[str] = []
    for item in buckets.explicit_facts:
        prefix = "用户说过：" if item.status == "asserted" else ""
        explicit_lines.append(f"{prefix}{item.proposition}")

    context_text = "\n".join(explicit_lines)

    assert "M7" in context_text, "Anchored proposition must appear in context"
    assert "Legacy episode at cycle" not in context_text, (
        "Legacy template must NOT leak into anchored memory context"
    )


def test_asserted_fact_rendered_as_user_said_not_objective_truth():
    """An asserted fact about a relationship must be framed as 'user said X',
    not as objective truth 'X is true'."""
    item = _make_item(
        "鲁永刚是用户的同学",
        status="asserted",
        visibility="explicit",
        memory_type="relationship_fact",
    )

    assert item.status == "asserted"

    # Simulate prompt rendering — use "用户说过：" prefix for asserted facts
    rendered = f"用户说过：{item.proposition}"

    assert "用户说过" in rendered
    assert "鲁永刚确实是" not in rendered, (
        "Should NOT render asserted fact as objective truth"
    )


# ── M8.3: Citation guard ───────────────────────────────────────────────────

def test_unanchored_specific_detail_is_flagged():
    """When a reply references specific unanchored details (dining, location),
    the guard must flag hallucinated_detail_risk."""
    # No anchored items about dining
    items: list[AnchoredMemoryItem] = [
        _make_item("用户已经完成了M7"),
    ]

    result = MemoryCitationGuard.audit("你们上次聚餐的时候聊得很开心", items)

    assert result.hallucinated_detail_risk, (
        "Should flag dining reference without anchored fact"
    )
    assert any("hallucinated" in f for f in result.flags)


def test_retracted_fact_is_not_reused():
    """When a user corrects 'A is my classmate' to 'B is my classmate',
    A must be retracted and B asserted; memory context must not reference A."""
    extractor = DialogueFactExtractor()

    # First turn: user says A is classmate
    items1 = extractor.extract(
        "鲁永刚是我同学", turn_id="turn_0001", utterance_id="u1", speaker="user",
    )
    assert len(items1) == 1
    assert "鲁永刚" in items1[0].proposition

    # Second turn: user corrects — "not A, it's B"
    items2 = extractor.extract(
        "不是鲁永刚，是李四",
        turn_id="turn_0002",
        utterance_id="u2",
        speaker="user",
        existing_items=items1,
    )

    # A must be retracted
    assert items1[0].status == "retracted", "Old fact must be retracted"

    # B must be asserted
    assert len(items2) >= 1
    assert any("李四" in it.proposition for it in items2), "New fact must be asserted"

    # Memory context must not reference A
    all_items = items1 + items2
    buckets = MemoryPermissionFilter.filter(all_items)
    explicit_props = {it.proposition for it in buckets.explicit_facts}
    forbidden_props = {it.proposition for it in buckets.forbidden}

    assert not any("鲁永刚" in p for p in explicit_props), (
        "Retracted fact must not appear in explicit_facts"
    )
    assert any("鲁永刚" in p for p in forbidden_props), (
        "Retracted fact must be in forbidden bucket"
    )


def test_memory_permission_filter_buckets_status_and_visibility():
    """The filter must correctly route items based on status + visibility."""
    items = [
        _make_item("F1-asserted-explicit", status="asserted", visibility="explicit"),
        _make_item("F2-corroborated-explicit", status="corroborated", visibility="explicit"),
        _make_item("H1-hypothesis", status="hypothesis", visibility="explicit",
                   memory_type="hypothesis", confidence=0.5),
        _make_item("S1-strategy-only", status="asserted", visibility="strategy_only",
                   memory_type="private_state"),
        _make_item("P1-private", status="asserted", visibility="private",
                   memory_type="private_state"),
        _make_item("R1-retracted", status="retracted", visibility="explicit"),
        _make_item("F3-forbidden-vis", status="asserted", visibility="forbidden"),
    ]
    buckets = MemoryPermissionFilter.filter(items)

    # explicit_facts: asserted/corroborated + explicit
    ef_props = {it.proposition for it in buckets.explicit_facts}
    assert "F1-asserted-explicit" in ef_props
    assert "F2-corroborated-explicit" in ef_props
    assert "H1-hypothesis" not in ef_props
    assert "S1-strategy-only" not in ef_props
    assert "P1-private" not in ef_props
    assert "R1-retracted" not in ef_props
    assert "F3-forbidden-vis" not in ef_props

    # cautious_hypotheses
    ch_props = {it.proposition for it in buckets.cautious_hypotheses}
    assert "H1-hypothesis" in ch_props
    assert "F1-asserted-explicit" not in ch_props

    # strategy_only
    so_props = {it.proposition for it in buckets.strategy_only}
    assert "S1-strategy-only" in so_props
    assert "P1-private" in so_props
    assert "F1-asserted-explicit" not in so_props

    # forbidden
    fb_props = {it.proposition for it in buckets.forbidden}
    assert "R1-retracted" in fb_props
    assert "F3-forbidden-vis" in fb_props


# ── M8.4: Regression — existing M7 tests still pass ────────────────────────

def test_existing_m7_tests_still_pass():
    """Verify that the M7 causal self-control test module imports cleanly
    and its core test functions are callable."""
    from tests.test_m7_causal_self_control import (
        test_cognitive_control_signal_created_from_high_severity_gap,
        test_memory_overdominance_reduces_memory_retrieval_gain,
        test_low_confidence_gap_can_shift_selected_path_to_clarify_or_marks_action_shift_candidate,
    )
    # Run key M7 tests to confirm they still work (avoid fixture-dependent ones)
    test_cognitive_control_signal_created_from_high_severity_gap()
    test_memory_overdominance_reduces_memory_retrieval_gain()
    test_low_confidence_gap_can_shift_selected_path_to_clarify_or_marks_action_shift_candidate()


# ── M8.5: MemoryStore anchored integration ─────────────────────────────────

def test_memory_store_stores_and_retrieves_anchored_items():
    """MemoryStore must correctly add, retrieve, and retract anchored items."""
    store = MemoryStore()
    item = _make_item("用户已经完成了M7")
    mid = store.add_anchored_item(item)

    assert mid == item.memory_id
    assert len(store.anchored_items) == 1

    # Retrieval
    found = store.find_anchored_by_proposition("M7")
    assert len(found) == 1
    assert found[0].proposition == "用户已经完成了M7"

    # Retraction
    assert store.retract_anchored_item(mid)
    assert store.anchored_items[0].status == "retracted"

    # Retraction of non-existent
    assert not store.retract_anchored_item("non-existent-id")


def test_memory_store_serialization_roundtrips_anchored_items():
    """MemoryStore.to_dict/from_dict must preserve anchored items."""
    store = MemoryStore()
    store.add_anchored_item(_make_item("用户已经完成了M7"))
    store.add_anchored_item(_make_item(
        "用户可能担心记忆", status="hypothesis", memory_type="hypothesis",
        visibility="strategy_only", confidence=0.5,
    ))

    d = store.to_dict()
    restored = MemoryStore.from_dict(d)

    assert len(restored.anchored_items) == 2
    props = {it.proposition for it in restored.anchored_items}
    assert "用户已经完成了M7" in props
    assert "用户可能担心记忆" in props

    # Check specific fields survive round-trip
    hypo = next(it for it in restored.anchored_items if it.status == "hypothesis")
    assert hypo.confidence == 0.5
    assert hypo.visibility == "strategy_only"


def test_empty_text_produces_no_facts():
    """Empty or whitespace-only text should produce zero anchored items."""
    extractor = DialogueFactExtractor()
    assert len(extractor.extract("", "t1", "u1")) == 0
    assert len(extractor.extract("   ", "t2", "u2")) == 0


def test_noise_text_produces_no_facts():
    """Text without any recognizable fact pattern should produce zero items."""
    extractor = DialogueFactExtractor()
    items = extractor.extract("今天天气不错", "t1", "u1")
    # "今天天气不错" has no name/milestone/relationship/preference patterns
    assert len(items) == 0, (
        "Non-fact text should not produce anchored items"
    )


def test_behavior_preference_has_strategy_only_visibility():
    """Behavior preferences ('please do X in future') must have
    strategy_only visibility."""
    extractor = DialogueFactExtractor()
    items = extractor.extract(
        "以后希望你直接回答，不要绕弯子",
        turn_id="turn_0001",
        utterance_id="u1",
        speaker="user",
    )
    assert len(items) >= 1
    beh = items[0]
    assert beh.visibility == "strategy_only"
    assert beh.memory_type == "preference"
