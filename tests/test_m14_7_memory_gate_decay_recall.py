from __future__ import annotations

from segmentum.dialogue.runtime.m13_memory_efe import normalize_expectations_for_efe
from segmentum.dialogue.runtime.m14_7_memory_decay import apply_memory_decay_tick
from segmentum.dialogue.runtime.m14_7_memory_gate import MemoryGate, MemoryWriteIntent, memory_intent_fingerprint
from segmentum.dialogue.runtime.m14_7_recall_scoring import score_recall_candidate
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


def test_gate_rejects_low_value_missing_evidence_write() -> None:
    intent = MemoryWriteIntent(
        target="long_term",
        kind="episode",
        content="thin",
        confidence=0.4,
        evidence_refs=[],
        value_proxy=0.0,
        surprise_proxy=0.0,
        proposer="test",
        source="test",
    )

    decision = MemoryGate().evaluate(intent)

    assert decision.commit is False
    assert "missing_evidence_refs" in decision.violation_codes
    assert "gate_score_below_threshold" in decision.violation_codes


def test_gate_commits_high_surprise_with_evidence() -> None:
    intent = MemoryWriteIntent(
        target="short_term",
        kind="episode",
        content="specific evidence-backed turn outcome",
        confidence=0.9,
        evidence_refs=["turn_1", "mem_1"],
        value_proxy=0.7,
        surprise_proxy=0.8,
        proposer="test",
        source="test",
    )

    decision = MemoryGate().evaluate(intent)

    assert decision.commit is True
    assert decision.write_score >= decision.threshold


def test_gate_rejects_recent_duplicate_fingerprint() -> None:
    intent = MemoryWriteIntent(
        target="short_term",
        kind="episode",
        content="same structural episode",
        confidence=0.9,
        evidence_refs=["turn_1", "mem_1"],
        value_proxy=0.8,
        surprise_proxy=0.8,
        proposer="test",
        source="test",
    )

    decision = MemoryGate().evaluate(
        intent,
        recent_intent_fingerprints={memory_intent_fingerprint(intent)},
    )

    assert decision.commit is False
    assert "duplicate_of_recent_episode" in decision.violation_codes


def test_decay_tick_lowers_salience_and_exempts_identity_rows() -> None:
    state = {
        "long_term_memory": [
            {"id": "ltm_decay", "content": "ordinary", "salience": 0.5, "last_decay_at": NOW - 86400},
            {"id": "ltm_identity", "content": "identity", "salience": 0.5, "identity_relevance": 0.8},
        ],
        "open_items": [{"id": "oi", "status": "open"}],
    }

    result = apply_memory_decay_tick(state, now=NOW, turn_index=3)

    assert result.rows_touched == 1
    assert result.rows_exempted == 1
    assert state["long_term_memory"][0]["salience"] < 0.5
    assert state["long_term_memory"][1]["salience"] == 0.5
    assert state["open_items"][0]["status"] == "open"


def test_recall_scoring_cancels_archived_and_uses_precision() -> None:
    active_low = {"id": "a", "content": "benchmark result", "salience": 0.8, "precision": 0.2}
    active_high = {"id": "b", "content": "benchmark result", "salience": 0.8, "precision": 0.9}
    archived = {"id": "c", "content": "benchmark result", "salience": 0.8, "precision": 0.9, "status": "archived"}

    assert score_recall_candidate(archived, query=["benchmark"], now=NOW, retrieved_context={}) == 0.0
    assert score_recall_candidate(active_high, query=["benchmark"], now=NOW, retrieved_context={}) > score_recall_candidate(
        active_low, query=["benchmark"], now=NOW, retrieved_context={}
    )


def test_memory_efe_filters_archived_bound_memory() -> None:
    state = {
        "pending_expectations": [
            {
                "id": "exp_1",
                "status": "pending",
                "content": "check benchmark",
                "due_at_epoch": NOW - 1000,
                "bound_memory_ids": ["ltm_archived"],
                "evidence_refs": ["ltm_archived"],
                "confidence": 0.9,
            }
        ],
        "long_term_memory": [{"id": "ltm_archived", "content": "benchmark", "status": "archived", "salience": 0.8}],
    }

    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")

    assert normalized.eligible_for_efe == []
    assert normalized.diagnostic_only[0].ineligibility_reason == "bound_memories_archived"


def test_path_b_thinking_write_routes_through_gate(tmp_path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "gate"), llm=None)
    state = runtime.store.load()

    runtime._apply_thinking_writes(
        state,
        {
            "reply": "ok",
            "memory_writes": [
                {
                    "target": "long_term",
                    "kind": "fact",
                    "content": "high value remembered detail",
                    "confidence": 0.95,
                    "salience": 0.9,
                    "evidence_refs": ["turn_1", "mem_1"],
                }
            ],
        },
        user_text="user detail",
        now=NOW,
        turn_index=2,
    )

    events = state.get("memory_gate_audit_tail", [])
    assert any(event["type"] == "MemoryGateCommitEvent" for event in events)
    assert state["long_term_memory"]
