"""M8.9 R2: Memory Write Intent tests."""

import pytest

from segmentum.memory_anchored import (
    AnchoredMemoryItem,
    MemoryWriteIntent,
    WriteIntentTrace,
)
from segmentum.cognitive_events import (
    COGNITIVE_EVENT_TYPES,
    COGNITIVE_EVENT_CONSUMERS,
)


# ── R2.1: WriteIntentTrace ───────────────────────────────────────────────

def test_write_intent_trace_survives_serialization():
    """WriteIntentTrace round-trips through to_dict/from_dict."""
    trace = WriteIntentTrace(
        source_event_id="evt_001",
        source_turn_id="turn_0001",
        source_utterance_id="turn_0001_user",
        source_speaker="user",
        extraction_cycle=5,
        committed_at_cycle=5,
    )
    d = trace.to_dict()
    restored = WriteIntentTrace.from_dict(d)
    assert restored.source_event_id == "evt_001"
    assert restored.source_turn_id == "turn_0001"
    assert restored.source_utterance_id == "turn_0001_user"
    assert restored.source_speaker == "user"
    assert restored.extraction_cycle == 5
    assert restored.committed_at_cycle == 5


def test_write_intent_trace_defaults():
    """WriteIntentTrace has sensible defaults."""
    trace = WriteIntentTrace()
    assert trace.source_event_id == ""
    assert trace.source_turn_id == ""
    assert trace.source_speaker == ""


# ── R2.2: MemoryWriteIntent ──────────────────────────────────────────────

def test_memory_write_intent_has_source_trace():
    """A MemoryWriteIntent carries source turn, utterance, and speaker."""
    trace = WriteIntentTrace(
        source_event_id="evt_002",
        source_turn_id="turn_0003",
        source_utterance_id="turn_0003_user",
        source_speaker="user",
        extraction_cycle=3,
    )
    item = AnchoredMemoryItem(
        memory_id="mem_001",
        speaker="user",
        proposition="用户叫周青",
        status="asserted",
        visibility="explicit",
    )
    intent = MemoryWriteIntent(
        intent_id="mwi_mem_001",
        item=item,
        trace=trace,
        operation="create",
        reason="dialogue_fact_extraction",
    )
    assert intent.item is not None
    assert intent.item.proposition == "用户叫周青"
    assert intent.trace is not None
    assert intent.trace.source_turn_id == "turn_0003"
    assert intent.trace.source_speaker == "user"
    assert intent.operation == "create"


def test_memory_write_intent_preserves_asserted_vs_corroborated_status():
    """Asserted and corroborated items survive write-intent wrapping with
    status intact."""
    asserted = AnchoredMemoryItem(
        proposition="周青喜欢Python",
        status="asserted",
        visibility="explicit",
    )
    corroborated = AnchoredMemoryItem(
        proposition="M8.9已完成",
        status="corroborated",
        visibility="explicit",
    )
    intent_a = MemoryWriteIntent(item=asserted, trace=WriteIntentTrace())
    intent_c = MemoryWriteIntent(item=corroborated, trace=WriteIntentTrace())
    assert intent_a.item is not None
    assert intent_a.item.status == "asserted"
    assert intent_c.item is not None
    assert intent_c.item.status == "corroborated"


def test_memory_write_intent_rejected_operation():
    """A rejected intent records the reason and keeps the item."""
    item = AnchoredMemoryItem(proposition="可疑声明", status="hypothesis")
    intent = MemoryWriteIntent(
        item=item,
        operation="reject",
        rejected_reason="hypothesis_cannot_become_fact",
    )
    assert intent.operation == "reject"
    assert intent.rejected_reason == "hypothesis_cannot_become_fact"


def test_memory_write_intent_serialization():
    """MemoryWriteIntent to_dict/from_dict round-trips."""
    trace = WriteIntentTrace(
        source_event_id="evt_003",
        source_turn_id="turn_0002",
        source_speaker="user",
    )
    item = AnchoredMemoryItem(proposition="test", status="asserted")
    intent = MemoryWriteIntent(
        intent_id="mwi_test",
        item=item,
        trace=trace,
        operation="create",
        reason="test",
    )
    d = intent.to_dict()
    restored = MemoryWriteIntent.from_dict(d)
    assert restored.intent_id == "mwi_test"
    assert restored.operation == "create"
    assert restored.item is not None
    assert restored.item.proposition == "test"
    assert restored.trace is not None
    assert restored.trace.source_turn_id == "turn_0002"


# ── R2.3: Event type registration ────────────────────────────────────────

def test_dialogue_fact_extraction_event_type_registered():
    """DialogueFactExtractionEvent is in COGNITIVE_EVENT_TYPES."""
    assert "DialogueFactExtractionEvent" in COGNITIVE_EVENT_TYPES


def test_memory_write_result_event_type_registered():
    """MemoryWriteResultEvent is in COGNITIVE_EVENT_TYPES."""
    assert "MemoryWriteResultEvent" in COGNITIVE_EVENT_TYPES


def test_extraction_event_has_state_update_consumer():
    """DialogueFactExtractionEvent is consumed by state_update."""
    consumers = COGNITIVE_EVENT_CONSUMERS.get("DialogueFactExtractionEvent", ())
    assert "state_update" in consumers
    assert "trace" in consumers


def test_write_result_event_has_trace_consumer():
    """MemoryWriteResultEvent is consumed by trace and evaluation."""
    consumers = COGNITIVE_EVENT_CONSUMERS.get("MemoryWriteResultEvent", ())
    assert "trace" in consumers
    assert "evaluation" in consumers


# ── R2.4: Backward compatibility ─────────────────────────────────────────

def test_legacy_add_anchored_item_still_works():
    """Direct add_anchored_item() still functions for backward compat."""
    from segmentum.memory_store import MemoryStore
    store = MemoryStore()
    item = AnchoredMemoryItem(proposition="legacy test", status="asserted")
    mid = store.add_anchored_item(item)
    assert mid == item.memory_id
    assert len(store.anchored_items) == 1


def test_commit_write_intent_preserves_item_in_store():
    """commit_write_intent() stores item with write-intent provenance tags."""
    from segmentum.memory_store import MemoryStore
    store = MemoryStore()
    item = AnchoredMemoryItem(
        proposition="用户叫李四",
        status="asserted",
        visibility="explicit",
        memory_type="user_asserted_fact",
    )
    trace = WriteIntentTrace(
        source_event_id="evt_010",
        source_turn_id="turn_0010",
        source_speaker="user",
        extraction_cycle=10,
    )
    intent = MemoryWriteIntent(
        intent_id="mwi_test_store",
        item=item,
        trace=trace,
        operation="create",
    )
    mid, op = store.commit_write_intent(intent)
    assert mid == item.memory_id
    assert op == "create"
    assert len(store.anchored_items) == 1
    stored = store.anchored_items[0]
    assert stored.proposition == "用户叫李四"
    # Provenance tags must be present
    tag_str = " ".join(stored.tags)
    assert "write_intent_id:mwi_test_store" in tag_str
    assert "source_turn:turn_0010" in tag_str
    assert "source_speaker:user" in tag_str


def test_commit_write_intent_rejects_on_reject_operation():
    """commit_write_intent() with operation=reject does not store the item."""
    from segmentum.memory_store import MemoryStore
    store = MemoryStore()
    item = AnchoredMemoryItem(proposition="should be rejected")
    intent = MemoryWriteIntent(
        item=item,
        trace=WriteIntentTrace(),
        operation="reject",
        rejected_reason="hypothesis_as_fact",
    )
    mid, op = store.commit_write_intent(intent)
    assert op == "reject"
    assert len(store.anchored_items) == 0


def test_commit_write_intent_errors_on_non_intent():
    """commit_write_intent() raises TypeError for non-MemoryWriteIntent."""
    from segmentum.memory_store import MemoryStore
    store = MemoryStore()
    with pytest.raises(TypeError):
        store.commit_write_intent({"not": "an intent"})  # type: ignore[arg-type]


# ── R2.5: Existing M8 tests compatibility check ─────────────────────────

def test_existing_m8_tests_structure_still_valid():
    """Key M8 concepts (AnchoredMemoryItem, extractor, permission filter,
    citation guard) remain importable and functional."""
    from segmentum.memory_anchored import (
        DialogueFactExtractor,
        MemoryPermissionFilter,
        MemoryCitationGuard,
    )
    # Basic anchored item creation still works
    item = AnchoredMemoryItem(
        speaker="user",
        proposition="我叫周青",
        status="asserted",
        visibility="explicit",
        memory_type="user_asserted_fact",
    )
    assert item.proposition == "我叫周青"
    assert item.speaker == "user"

    # Permission filter still categorizes
    from segmentum.memory_anchored import MemoryPermissionBuckets
    buckets = MemoryPermissionFilter.filter([item])
    assert isinstance(buckets, MemoryPermissionBuckets)
    assert len(buckets.explicit_facts) == 1
    assert len(buckets.cautious_hypotheses) == 0

    # Citation guard still works
    guard = MemoryCitationGuard()
    # Legacy audit
    result = guard.audit("你好，我叫周青", [item])
    assert result is not None
    # Structured audit (M8.5)
    structured = guard.audit_structured("你好，我叫周青", [item])
    assert structured is not None
    # No blocking violations for honest reply
    assert not structured.has_blocking_violation

    # Dialogue fact extractor still functional
    extractor = DialogueFactExtractor()
    facts = extractor.extract(
        text="我是李四，我喜欢Python编程",
        turn_id="turn_0001",
        utterance_id="turn_0001_user",
        speaker="user",
    )
    assert len(facts) >= 1  # at least name fact
    for fact in facts:
        assert fact.speaker == "user"
