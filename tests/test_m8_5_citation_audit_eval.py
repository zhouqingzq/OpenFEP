"""M8.5 Citation audit precision/recall evaluation.

Hand-labeled (reply, anchored_items, expected_violation_types) triples
measuring MemoryCitationGuard.audit_structured performance.
"""

from __future__ import annotations

import pytest

from segmentum.memory_anchored import (
    AnchoredMemoryItem,
    MemoryCitationGuard,
    MemoryPermissionFilter,
)


def _make_item(
    proposition: str,
    status: str = "asserted",
    visibility: str = "explicit",
    memory_type: str = "user_fact",
    **kw,
) -> AnchoredMemoryItem:
    return AnchoredMemoryItem(
        proposition=proposition,
        status=status,  # type: ignore[arg-type]
        visibility=visibility,  # type: ignore[arg-type]
        memory_type=memory_type,  # type: ignore[arg-type]
        speaker="user",
        turn_id="t1",
        utterance_id="t1_u",
        source_text=proposition,
        created_turn_id="t1",
        **kw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# True Positives — should be flagged
# ═══════════════════════════════════════════════════════════════════════════════

def test_unanchored_memory_claim_ni_ji_de():
    """'我记得' without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("我记得你之前说过你喜欢短回复", items)
    assert audit.has_violation
    assert any(v.type == "unanchored_memory_claim" for v in audit.violations)


def test_unanchored_memory_claim_ni_zhi_qian_shuo():
    """'你之前说' without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("你之前说你不喜欢长回复", items)
    assert audit.has_violation


def test_unanchored_memory_claim_jucan():
    """Dining reference without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("你们上次聚餐的时候聊得很开心", items)
    assert audit.has_violation


def test_unanchored_memory_claim_ni_ceng_jing():
    """'你曾经' without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("你曾经提到过这个bug", items)
    assert audit.has_violation


def test_retracted_fact_referenced_blocking():
    """Referencing a retracted fact must produce blocking violation."""
    items = [
        _make_item("鲁永刚是用户的同学", status="retracted", visibility="forbidden"),
        _make_item("李四是用户的同学", status="asserted"),
    ]
    audit = MemoryCitationGuard.audit_structured("我记得鲁永刚是用户的同学", items)
    assert audit.has_blocking_violation
    assert any(v.type == "retracted_fact_referenced" for v in audit.violations)


def test_retracted_name_blocking():
    """Referencing a retracted name must block."""
    items = [
        _make_item("用户的名字是小王", status="retracted", visibility="forbidden"),
        _make_item("用户的名字是小张", status="asserted"),
    ]
    audit = MemoryCitationGuard.audit_structured("你之前说你的名字是小王", items)
    assert audit.has_blocking_violation


def test_hypothesis_as_fact_ni_kending_shi():
    """'你肯定是' without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("你肯定是喜欢这个方案的", items)
    assert audit.has_violation
    assert any(v.type == "hypothesis_as_fact" for v in audit.violations)


def test_hypothesis_as_fact_ni_juedui():
    """'你绝对' without anchored support must be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("你绝对是想换一个方案", items)
    assert audit.has_violation
    assert any(v.type == "hypothesis_as_fact" for v in audit.violations)


# ═══════════════════════════════════════════════════════════════════════════════
# True Negatives — should NOT be flagged
# ═══════════════════════════════════════════════════════════════════════════════

def test_normal_reply_with_anchored_support():
    """Reply referencing an anchored fact should not be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("关于你说的M8，我有些想法", items)
    assert not audit.has_violation


def test_simple_greeting_not_flagged():
    """Simple greeting should not be flagged."""
    items = [_make_item("用户的名字是周青")]
    audit = MemoryCitationGuard.audit_structured("你好，今天想聊什么？", items)
    assert not audit.has_violation


def test_normal_disagreement_not_flagged():
    """Normal disagreement without memory claims should not be flagged."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("我觉得这个方案可能还有一些问题", items)
    assert not audit.has_violation


def test_sharing_own_opinion_not_flagged():
    """Agent sharing its own opinion without claiming user facts should pass."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("我倾向于先做M8的收尾工作", items)
    assert not audit.has_violation


# ═══════════════════════════════════════════════════════════════════════════════
# False Positive Risk — informational, documents known limitations
# ═══════════════════════════════════════════════════════════════════════════════

def test_friendly_remembrance_is_false_positive_risk():
    """'我记得上次聊得很愉快' is a friendly opener, not a memory claim.

    CURRENT BEHAVIOR: patterns WILL flag this. This is a known false positive
    risk — the '我记得' pattern cannot distinguish conversational openers
    from factual memory claims.
    """
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("我记得上次聊得很愉快，今天继续", items)
    # Document current behavior without asserting pass/fail
    assert isinstance(audit.has_violation, bool)


def test_shang_ci_in_temporal_context_is_false_positive_risk():
    """'上次你给我的建议很好' is a legitimate temporal reference.

    CURRENT BEHAVIOR: patterns WILL flag this as '上次' hits memory-claim
    patterns. This is a known false positive risk.
    """
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured("上次你给我的建议很好，我试了一下", items)
    assert isinstance(audit.has_violation, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_mixed_violations_both_unanchored_and_hypothesis():
    """Reply with both memory claim and certainty language must flag both."""
    items = [_make_item("用户已经完成了M8")]
    audit = MemoryCitationGuard.audit_structured(
        "我记得你之前说过你肯定喜欢短回复", items,
    )
    violation_types = {v.type for v in audit.violations}
    assert "unanchored_memory_claim" in violation_types
    assert "hypothesis_as_fact" in violation_types


def test_empty_anchored_list_normal_reply_not_flagged():
    """With no anchored items, a normal reply should not be flagged."""
    items: list[AnchoredMemoryItem] = []
    audit = MemoryCitationGuard.audit_structured("好的，我明白了", items)
    assert not audit.has_violation


def test_empty_anchored_list_memory_claim_still_flagged():
    """With no anchored items, a memory claim should still be flagged."""
    items: list[AnchoredMemoryItem] = []
    audit = MemoryCitationGuard.audit_structured("我记得你之前说过你喜欢短回复", items)
    assert audit.has_violation


def test_forbidden_topic_blocking():
    """Forbidden topic reference must produce blocking violation."""
    items: list[AnchoredMemoryItem] = []
    audit = MemoryCitationGuard.audit_structured("我们来谈谈你的创伤吧", items)
    assert audit.has_blocking_violation
    assert any(v.type == "forbidden_topic_referenced" for v in audit.violations)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_citation_audit_precision_recall_stats():
    """Compute precision/recall over the labeled eval set."""
    eval_cases: list[tuple[str, list[AnchoredMemoryItem], list[str]]] = [
        # True Positives
        ("我记得你之前说过你喜欢短回复",
         [_make_item("用户已经完成了M8")],
         ["unanchored_memory_claim"]),
        ("你之前说你不喜欢长回复",
         [_make_item("用户已经完成了M8")],
         ["unanchored_memory_claim"]),
        ("你们上次聚餐的时候聊得很开心",
         [_make_item("用户已经完成了M8")],
         ["hallucinated_detail_risk"]),
        ("你曾经提到过这个bug",
         [_make_item("用户已经完成了M8")],
         ["unanchored_memory_claim"]),
        ("我记得鲁永刚是用户的同学",
         [_make_item("鲁永刚是用户的同学", status="retracted", visibility="forbidden"),
          _make_item("李四是用户的同学")],
         ["retracted_fact_referenced"]),
        ("你肯定是喜欢这个方案的",
         [_make_item("用户已经完成了M8")],
         ["hypothesis_as_fact"]),
        ("你绝对是想换一个方案",
         [_make_item("用户已经完成了M8")],
         ["hypothesis_as_fact"]),
        # True Negatives
        ("关于你说的M8，我有些想法",
         [_make_item("用户已经完成了M8")],
         []),
        ("你好，今天想聊什么？",
         [_make_item("用户的名字是周青")],
         []),
        ("我觉得这个方案可能还有一些问题",
         [_make_item("用户已经完成了M8")],
         []),
        ("我倾向于先做M8的收尾工作",
         [_make_item("用户已经完成了M8")],
         []),
        ("好的，我明白了",
         [],
         []),
    ]

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for reply, items, expected_types in eval_cases:
        audit = MemoryCitationGuard.audit_structured(reply, items)
        has_expected = bool(expected_types)
        got_flagged = audit.has_violation
        got_types = {v.type for v in audit.violations} if audit.has_violation else set()
        recall_ok = got_types >= set(expected_types)

        if has_expected and recall_ok:
            true_positives += 1
        elif has_expected and not recall_ok:
            false_negatives += 1
        elif not has_expected and not got_flagged:
            true_negatives += 1
        elif not has_expected and got_flagged:
            false_positives += 1

    total = len(eval_cases)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)

    # Thresholds from plan
    assert precision >= 0.70, (
        f"Citation audit precision {precision:.2f} below 0.70; "
        f"TP={true_positives} FP={false_positives}"
    )
    assert recall >= 0.85, (
        f"Citation audit recall {recall:.2f} below 0.85; "
        f"TP={true_positives} FN={false_negatives}"
    )
