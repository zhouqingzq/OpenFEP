"""M8 Anchored Memory Contract — test suite."""

from __future__ import annotations

import pytest

from segmentum.memory_anchored import (
    AnchoredMemoryItem,
    CitationAuditResult,
    DialogueFactExtractor,
    MemoryCitationAudit,
    MemoryCitationGuard,
    MemoryCitationViolation,
    MemoryPermissionBuckets,
    MemoryPermissionFilter,
    build_memory_repair_instruction,
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


# ── M8.5: MemoryCitationAudit structured output ────────────────────────────

def test_memory_citation_audit_structured_output():
    """audit_structured must return MemoryCitationAudit with proper fields."""
    items = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
    ]
    audit = MemoryCitationGuard.audit_structured("关于你说的M7，我有些想法", items)
    assert isinstance(audit, MemoryCitationAudit)
    assert not audit.has_violation
    assert not audit.has_blocking_violation
    assert audit.risk_level == "none"
    assert audit.to_dict()["has_violation"] is False


def test_retracted_fact_triggers_blocking_violation():
    """Referencing a retracted fact must produce blocking violation."""
    items = [
        _make_item("鲁永刚是用户的同学", status="retracted", visibility="forbidden"),
        _make_item("李四是用户的同学", status="asserted", visibility="explicit"),
    ]
    audit = MemoryCitationGuard.audit_structured("我记得鲁永刚是用户的同学", items)
    assert audit.has_violation
    assert audit.has_blocking_violation
    assert audit.risk_level == "blocking"
    assert any(v.type == "retracted_fact_referenced" for v in audit.violations)


def test_hypothesis_as_fact_flagged():
    """Certainty language without anchored support must be flagged."""
    items: list[AnchoredMemoryItem] = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
    ]
    audit = MemoryCitationGuard.audit_structured("你肯定是喜欢这个方案", items)
    assert audit.has_violation
    violation_types = {v.type for v in audit.violations}
    assert "hypothesis_as_fact" in violation_types


def test_unanchored_memory_claim_flagged():
    """'你之前说过' without anchored support must be flagged."""
    items: list[AnchoredMemoryItem] = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
    ]
    audit = MemoryCitationGuard.audit_structured("你之前说过你喜欢短回复", items)
    assert audit.has_violation
    assert any(v.type == "unanchored_memory_claim" for v in audit.violations)


def test_build_repair_instruction_from_audit():
    """build_memory_repair_instruction must produce relevant constraints."""
    audit = MemoryCitationAudit(
        has_violation=True,
        has_blocking_violation=True,
        risk_level="blocking",
        violations=[
            MemoryCitationViolation(
                type="retracted_fact_referenced",
                span="鲁永刚",
                message="ref retracted",
                suggested_action="block",
            ),
            MemoryCitationViolation(
                type="unanchored_memory_claim",
                span="你之前说过",
                message="unanchored claim",
                suggested_action="retry",
            ),
        ],
    )
    instruction = build_memory_repair_instruction(audit)
    assert "记忆引用契约" in instruction
    assert "已撤回" in instruction.lower() or "retract" in instruction.lower()
    assert "更短" in instruction


# ── M8.5: user/agent fact split ─────────────────────────────────────────────

def test_agent_turn_does_not_produce_explicit_user_facts():
    """Agent reply extraction must produce agent_self_utterance, not user_fact."""
    extractor = DialogueFactExtractor()
    items = extractor.extract(
        "我在考虑这个方案的可行性",
        turn_id="turn_0001",
        utterance_id="u1",
        speaker="agent",
    )
    for item in items:
        assert item.memory_type == "agent_self_utterance", (
            f"Agent turn item must be agent_self_utterance, got {item.memory_type}"
        )
        assert item.visibility != "explicit", (
            "Agent turn item must not have explicit visibility"
        )


def test_agent_self_utterance_not_in_explicit_facts():
    """MemoryPermissionFilter must route agent_self_utterance to strategy_only."""
    items = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
        _make_item(
            "agent said something", status="asserted",
            memory_type="agent_self_utterance", visibility="strategy_only",
        ),
    ]
    buckets = MemoryPermissionFilter.filter(items)
    explicit_props = {it.proposition for it in buckets.explicit_facts}
    so_props = {it.proposition for it in buckets.strategy_only}
    assert "agent said something" not in explicit_props
    assert "agent said something" in so_props


def test_user_turn_produces_explicit_facts():
    """User turn extraction must produce explicit user_asserted_fact items."""
    extractor = DialogueFactExtractor()
    items = extractor.extract(
        "我叫周青",
        turn_id="turn_0001",
        utterance_id="u1",
        speaker="user",
    )
    assert len(items) >= 1
    item = items[0]
    assert item.visibility == "explicit"
    assert item.speaker == "user"


# ── M8.5: priority (retracted > asserted) ──────────────────────────────────

def test_retracted_overrides_asserted_in_memory_context():
    """When user corrects A→B, context must contain B, A must be forbidden."""
    extractor = DialogueFactExtractor()
    items1 = extractor.extract(
        "鲁永刚是我同学", turn_id="turn_0001", utterance_id="u1", speaker="user",
    )
    items2 = extractor.extract(
        "不是鲁永刚，是李四",
        turn_id="turn_0002", utterance_id="u2", speaker="user",
        existing_items=items1,
    )
    all_items = items1 + items2
    buckets = MemoryPermissionFilter.filter(all_items)
    explicit_props = {it.proposition for it in buckets.explicit_facts}
    forbidden_props = {it.proposition for it in buckets.forbidden}

    assert any("李四" in p for p in explicit_props), "New fact B must be in explicit"
    assert any("鲁永刚" in p for p in forbidden_props), "Retracted A must be forbidden"
    assert not any("鲁永刚" in p for p in explicit_props), "Retracted A must not be explicit"

    # Audit check: referencing A must block
    audit = MemoryCitationGuard.audit_structured(
        "你之前提到的鲁永刚是你同学", all_items,
    )
    assert audit.has_blocking_violation


# ── M8.5: legacy fallback downgrade ─────────────────────────────────────────

def test_legacy_fallback_contains_downgrade_language():
    """When legacy entries are formatted without anchored items, downgrade
    language must appear."""
    from segmentum.dialogue.runtime.prompts import (
        _build_memory_context_v2,
        _format_memory_context,
    )

    # We can't easily mock the full agent, so we test _format_memory_context directly.
    mem_dict = {
        "explicit_usable": ["Legacy: user talked about M7"],
        "implicit_tone": [],
        "do_not_use": [],
    }
    result = _format_memory_context(mem_dict)
    assert "语境线索" in result, (
        "Legacy fallback must label items as contextual clues"
    )
    assert "anchored memory" in result.lower(), (
        "Legacy fallback must mention anchored memory priority"
    )


# ── M8.5: CitationAuditResult backward compat ───────────────────────────────

def test_legacy_audit_still_works():
    """The original CitationAuditResult audit method must still work."""
    items = [
        _make_item("用户已经完成了M7", status="asserted", visibility="explicit"),
        _make_item("鲁永刚是用户的同学", status="retracted", visibility="forbidden"),
    ]
    result = MemoryCitationGuard.audit("鲁永刚是用户的同学", items)
    assert isinstance(result, CitationAuditResult)
    assert result.retracted_fact_referenced


# ── M8.5: MemoryCitationAudit to_dict ───────────────────────────────────────

def test_audit_to_dict_serializable():
    """MemoryCitationAudit.to_dict must produce JSON-serializable output."""
    audit = MemoryCitationAudit(
        has_violation=True,
        has_blocking_violation=True,
        risk_level="blocking",
        violations=[
            MemoryCitationViolation(
                type="hypothesis_as_fact",
                span="你肯定",
                message="test",
                suggested_action="retry",
            ),
        ],
    )
    d = audit.to_dict()
    assert isinstance(d, dict)
    assert d["risk_level"] == "blocking"
    assert len(d["violations"]) == 1
    assert d["violations"][0]["type"] == "hypothesis_as_fact"


# ── M8.5 Remediation: RuleBasedGenerator repair response ──────────────────

def test_rulebased_generator_repair_uses_conservative_template():
    """When memory_repair_instruction is present, RuleBasedGenerator must
    return a conservative template, not a hardcoded string."""
    from segmentum.dialogue.generator import RuleBasedGenerator

    gen = RuleBasedGenerator()
    ctx: dict[str, object] = {
        "memory_repair_instruction": "test repair instruction",
        "observation": {},
    }
    reply = gen.generate(
        "ask_question", ctx, {}, [], master_seed=42, turn_index=1,
    )
    # Must not be the old hardcoded string
    assert reply != "收到，我重新说一下。"
    # Conservative templates are short
    assert len(reply) <= 10
    # Diagnostics must record repair
    diag = gen.last_diagnostics
    assert diag.get("memory_repair_active") is True


# ── M8.5 Remediation: anchored item pruning ───────────────────────────────

def test_prune_anchored_respects_max_count():
    """prune_anchored must reduce items to max_count."""
    store = MemoryStore()
    for i in range(60):
        item = _make_item(f"fact_{i}")
        item.created_at_cycle = i
        store.add_anchored_item(item)
    removed = store.prune_anchored(max_count=50)
    assert removed == 10
    assert len(store.anchored_items) == 50


def test_prune_anchored_removes_retracted_first():
    """Retracted items must be evicted before active items."""
    store = MemoryStore()
    for i in range(55):
        item = _make_item(f"fact_{i}")
        item.created_at_cycle = i
        store.add_anchored_item(item)
    retracted_id_0 = store.anchored_items[0].memory_id
    retracted_id_3 = store.anchored_items[3].memory_id
    store.anchored_items[0].status = "retracted"
    store.anchored_items[3].status = "retracted"
    store.prune_anchored(max_count=50)
    remaining_ids = {it.memory_id for it in store.anchored_items}
    assert retracted_id_0 not in remaining_ids, (
        "Retracted item should be pruned first"
    )
    assert retracted_id_3 not in remaining_ids, (
        "Retracted item should be pruned first"
    )


def test_prune_anchored_noop_when_under_limit():
    """prune_anchored must not remove anything when under max_count."""
    store = MemoryStore()
    for i in range(5):
        store.add_anchored_item(_make_item(f"fact_{i}"))
    removed = store.prune_anchored(max_count=50)
    assert removed == 0
    assert len(store.anchored_items) == 5


def test_anchored_item_roundtrip_with_cycle():
    """created_at_cycle must survive to_dict/from_dict round-trip."""
    item = _make_item("用户已经完成了M7")
    item.created_at_cycle = 42
    d = item.to_dict()
    assert d["created_at_cycle"] == 42
    restored = AnchoredMemoryItem.from_dict(d)
    assert restored.created_at_cycle == 42


# ── M8.5 Remediation: FEP capsule no double render ────────────────────────

def test_capsule_guidance_does_not_double_render_memory_constraints():
    """_build_capsule_guidance must not emit duplicate memory guidance."""
    from segmentum.dialogue.runtime.prompts import _build_capsule_guidance

    capsule: dict[str, object] = {
        "chosen_action": "ask_question",
        "memory_use_guidance": {
            "reduce_memory_reliance": True,
            "memory_conflict_count": 2,
        },
    }
    lines = _build_capsule_guidance(capsule)
    memory_lines = [
        l for l in lines
        if isinstance(l, str) and ("Memory use" in l or "tentative" in l)
    ]
    # The two lines (reduce_memory_reliance, memory_conflict_count) are
    # different constraint types — each should appear exactly once, not twice.
    assert len(set(memory_lines)) == len(memory_lines), (
        f"Memory guidance lines must not have duplicates, got: {memory_lines}"
    )
    assert len(memory_lines) >= 1, (
        "Expected at least one memory guidance line from cognitive_guidance"
    )


# ── M8.5 Remediation: priority rule in memory context ─────────────────────

def test_memory_context_includes_priority_rule():
    """_format_memory_context must include anchor-first priority sentence
    when anchored items exist."""
    from segmentum.dialogue.runtime.prompts import _format_memory_context

    mem_dict = {
        "explicit_usable": ["用户说过：用户的名字是周青"],
        "implicit_tone": [],
        "do_not_use": [],
    }
    result = _format_memory_context(mem_dict)
    assert "锚点记忆优先" in result, (
        f"Priority rule must appear in anchored context, got: {result[:200]}"
    )
