"""M8.9 R3: Response Evidence Contract tests."""

import pytest

from segmentum.memory_anchored import (
    AnchoredMemoryItem,
    MemoryCitationGuard,
    MemoryPermissionFilter,
    ResponseEvidenceContract,
    build_response_evidence_contract,
)


# ── R3.1: Evidence contract structure ────────────────────────────────────

def test_evidence_contract_has_all_eight_buckets():
    """ResponseEvidenceContract has all 8 required bucket fields."""
    contract = ResponseEvidenceContract()
    assert hasattr(contract, "current_input_facts")
    assert hasattr(contract, "anchored_facts")
    assert hasattr(contract, "retrieved_memories")
    assert hasattr(contract, "external_evidence")
    assert hasattr(contract, "unverified_claims")
    assert hasattr(contract, "unknowns")
    assert hasattr(contract, "forbidden_assumptions")
    assert hasattr(contract, "style_constraints")


def test_evidence_contract_separates_known_unverified_unknown():
    """Known facts go to anchored_facts, hypotheses to unverified_claims,
    unknowns stay in unknowns bucket."""
    contract = ResponseEvidenceContract(
        current_input_facts=("用户说自己叫周青",),
        anchored_facts=("周青喜欢Python", "M8项目已完成"),
        unverified_claims=("周青也许在北京工作",),
        unknowns=("周青的生日",),
    )
    assert "用户说自己叫周青" in contract.current_input_facts
    assert "周青喜欢Python" in contract.anchored_facts
    assert "周青也许在北京工作" in contract.unverified_claims
    assert "周青的生日" in contract.unknowns


def test_evidence_contract_is_claim_supported():
    """is_claim_supported() finds claims in known buckets."""
    contract = ResponseEvidenceContract(
        anchored_facts=("用户喜欢编程",),
        current_input_facts=("今天天气不错",),
    )
    assert contract.is_claim_supported("用户喜欢编程")
    assert contract.is_claim_supported("今天天气不错")
    assert not contract.is_claim_supported("用户住在火星")


# ── R3.2: Builder ────────────────────────────────────────────────────────

def test_build_evidence_contract_from_anchored_items():
    """build_response_evidence_contract() routes asserted facts and
    hypotheses correctly."""
    items = [
        AnchoredMemoryItem(
            proposition="用户叫李四",
            status="asserted",
            visibility="explicit",
            memory_type="user_asserted_fact",
        ),
        AnchoredMemoryItem(
            proposition="李四可能是学生",
            status="hypothesis",
            visibility="explicit",
            memory_type="hypothesis",
        ),
        AnchoredMemoryItem(
            proposition="秘密信息",
            status="retracted",
            visibility="forbidden",
        ),
    ]
    contract = build_response_evidence_contract(
        turn_id="turn_0001",
        current_turn_text="我叫李四，我喜欢编程",
        anchored_items=items,
        retrieved_memory_texts=["李四之前提到过Java"],
    )
    assert "用户叫李四" in contract.anchored_facts
    assert "李四可能是学生" in contract.unverified_claims
    assert "秘密信息" in contract.forbidden_assumptions
    assert "李四之前提到过Java" in contract.retrieved_memories
    assert "我叫李四，我喜欢编程" in contract.current_input_facts


def test_build_evidence_contract_unknowns_on_empty_retrieval():
    """No retrieved memories → unknowns includes long_term_memory_not_cued."""
    contract = build_response_evidence_contract(
        turn_id="turn_0002",
        anchored_items=[],
        retrieved_memory_texts=[],
    )
    assert any("long_term_memory_not_cued" in u.lower() for u in contract.unknowns)


# ── R3.3: Citation guard against evidence contract ───────────────────────

def test_citation_audit_against_evidence_contract_flags_unsupported_claim():
    """audit_against_evidence_contract() flags memory claims with no evidence."""
    contract = ResponseEvidenceContract(
        anchored_facts=("用户叫周青",),
        unknowns=("周青的生日",),
    )
    # Reply claims an unknown detail
    reply = "我记得你的生日是5月20日"
    audit = MemoryCitationGuard.audit_against_evidence_contract(reply, contract)
    assert audit.has_violation
    violation_types = {v.type for v in audit.violations}
    assert "unanchored_memory_claim" in violation_types


def test_citation_audit_blocks_forbidden_assumption_reference():
    """Forbidden assumptions in the reply → blocking violation."""
    contract = ResponseEvidenceContract(
        forbidden_assumptions=("创伤经历",),
    )
    reply = "你之前提到过你的创伤经历"
    audit = MemoryCitationGuard.audit_against_evidence_contract(reply, contract)
    assert audit.has_blocking_violation


def test_citation_audit_passes_with_neutral_reply():
    """A reply that only references anchored facts passes audit."""
    contract = ResponseEvidenceContract(
        anchored_facts=("用户叫周青",),
    )
    reply = "你好，周青！今天想聊什么？"
    audit = MemoryCitationGuard.audit_against_evidence_contract(reply, contract)
    assert not audit.has_blocking_violation


def test_generator_does_not_promote_unknown_memory_detail_to_fact():
    """A reply claiming an unknown detail is flagged."""
    contract = ResponseEvidenceContract(
        unknowns=("用户的具体地址",),
    )
    # Reply directly references the exact unknown proposition text
    reply = "我记得你说过用户的具体地址在上海"
    audit = MemoryCitationGuard.audit_against_evidence_contract(reply, contract)
    assert audit.has_violation


# ── R3.4: Prompt safety ──────────────────────────────────────────────────

def test_prompt_receives_evidence_contract_not_raw_memory_dump():
    """to_compact_prompt() output does not contain raw object dumps."""
    contract = ResponseEvidenceContract(
        anchored_facts=("用户喜欢Python",),
        unverified_claims=("用户可能是老师",),
    )
    text = contract.to_compact_prompt()
    assert "[Evidence Boundary]" in text
    assert "用户喜欢Python" in text
    assert "do NOT assert as fact" in text
    # Must not contain raw dict dumps
    assert "memory_id" not in text
    assert "source_text" not in text
    assert "created_at_cycle" not in text


def test_evidence_contract_to_compact_prompt_produces_prompt_safe_text():
    """to_compact_prompt() contains no sensitive markers."""
    contract = ResponseEvidenceContract(
        current_input_facts=("你好",),
        unverified_claims=("用户可能是老师",),
        forbidden_assumptions=("API key", "token"),
    )
    text = contract.to_compact_prompt()
    # Must not contain code fences or raw dump markers
    assert "```" not in text
    assert "API key" in text  # forbidden assumptions SHOULD be listed as constraint
    # Unverified claims trigger the "do NOT assert as fact" constraint
    assert "do NOT assert" in text


def test_style_constraints_bucket_does_not_promote_unknown_to_fact():
    """Style constraints are in their own bucket, not mixed into fact buckets."""
    contract = ResponseEvidenceContract(
        anchored_facts=("用户叫周青",),
        style_constraints=("保持温暖语气", "简短回复"),
    )
    assert "保持温暖语气" in contract.style_constraints
    assert "保持温暖语气" not in contract.anchored_facts
    assert "保持温暖语气" not in contract.current_input_facts


def test_evidence_contract_to_dict():
    """to_dict() produces serializable output."""
    contract = ResponseEvidenceContract(
        contract_id="test_001",
        turn_id="turn_0001",
        current_input_facts=("你好",),
        anchored_facts=("用户叫周青",),
        unknowns=("生日",),
        style_constraints=("简短",),
    )
    d = contract.to_dict()
    assert d["contract_id"] == "test_001"
    assert "你好" in d["current_input_facts"]
    assert "用户叫周青" in d["anchored_facts"]
    assert "生日" in d["unknowns"]
    assert "简短" in d["style_constraints"]
