from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import M13DriveEvaluator, default_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state, set_initiative_user_opt_in
from segmentum.dialogue.runtime.m13_memory_efe import evaluate_memory_efe
from segmentum.dialogue.runtime.m15_episode_ledger import (
    EpisodeLedger,
    aggregate_fe_components,
    build_episode,
    self_consistency_proxy_from_state,
    state_fingerprint,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


class _ShortLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "鎰忚瘑涓诲惊鐜" in system_prompt:
            return {
                "expectation_results": [{"id": "exp_1", "status": "confirmed"}],
                "memory_search_keywords": ["status"],
                "temporal_assessment": {},
            }
        if "涓婅疆鍥炲鍚庢灉璇勪及" in system_prompt:
            return {
                "reaction": "uptake",
                "confidence": 0.7,
                "reason_codes": ["continues_thread"],
            }
        return {
            "reply": "好的。",
            "reply_action": "answer",
            "llm_thinking_result": {},
            "memory_writes": [
                {
                    "content": "User asked for a status update.",
                    "kind": "episode",
                    "confidence": 0.9,
                    "salience": 0.8,
                    "evidence_refs": ["turn_0001"],
                    "value_proxy": 0.8,
                    "surprise_proxy": 0.65,
                }
            ],
        }


def _episode_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_episode_written_on_user_turn_with_components_and_gate(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "persona"), llm=_ShortLLM())
    result = runtime.run_turn("项目 status 更新一下", turn_index=0, now=NOW)

    episode = result.diagnostics["m15_episode"]
    assert episode["phase"] == "user_turn"
    assert episode["action"] == "answer"
    assert episode["engineering_proxy_label"] == "mvp_local_episode_ledger"
    assert set(episode["components_before"]) >= {
        "sharing_fe",
        "memory_efe_f_memory",
        "reward_net_proxy",
        "expectation_prediction_error_proxy",
        "self_consistency_proxy",
    }
    assert isinstance(episode["delta_fe_proxy"], float)
    assert "reply" not in episode
    assert episode["memory_gate_decision"]["events"]

    rows = _episode_rows(runtime.store.root / "memory_dynamics_episodes.jsonl")
    assert len([row for row in rows if row.get("record_type") == "episode"]) == 1


def test_idle_cognitive_tick_writes_idle_episode(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "idle"), llm=None)
    runtime.run_idle_cognitive_tick(turn_index=3, idle_seconds=120.0, now=NOW)

    ledger = EpisodeLedger(runtime.store.root)
    episodes = ledger.recent(5)
    assert len(episodes) == 1
    assert episodes[0].phase == "idle_tick"
    assert episodes[0].action == "idle_wait"
    assert episodes[0].outcome_summary in {"ignored", "settled"}


def test_settled_event_is_addendum_not_mutation_of_jsonl(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "settle"), llm=_ShortLLM())
    first = runtime.run_turn("项目 status", turn_index=0, now=NOW)
    second = runtime.run_turn("好的，继续", turn_index=1, now=NOW + 60)

    rows = _episode_rows(runtime.store.root / "memory_dynamics_episodes.jsonl")
    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    addenda = [row for row in rows if row.get("record_type") == "settlement_addendum"]
    assert len(episode_rows) == 2
    assert addenda
    assert addenda[-1]["type"] == "MemoryDynamicsEpisodeSettledEvent"
    assert addenda[-1]["episode_id"] == first.diagnostics["m15_episode"]["episode_id"]
    assert second.diagnostics["m15_episode"]["episode_id"] != first.diagnostics["m15_episode"]["episode_id"]


def test_state_fingerprint_deterministic_and_structural(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "fingerprint")
    state = store.load()
    first = state_fingerprint(state)
    assert first == state_fingerprint(json.loads(json.dumps(state)))

    state["m13_drive_state"] = default_m13_drive_state()
    state["m13_drive_state"]["boredom"] = {"boredom_level": 0.8}
    assert state_fingerprint(state) != first


def test_self_consistency_proxy_uses_structured_limits_not_keyword_text() -> None:
    keyword_only = {
        "m12_1_user_personality": {
            "latest_reports_by_user": {
                "u1": {
                    "summary": "known_limit insufficient wording should not count without structured fields",
                }
            }
        }
    }
    structured = {
        "self_cognition": {
            "self_continuity": {
                "baseline_known_limits": [
                    {"limit": "needs more grounded evidence", "source": "review"}
                ]
            }
        }
    }

    assert self_consistency_proxy_from_state(keyword_only) == 0.0
    assert self_consistency_proxy_from_state(structured) > 0.0


def test_ledger_by_fingerprint_returns_latest_first(tmp_path: Path) -> None:
    state = {"m13_drive_state": default_m13_drive_state(), "temporal_state": {}}
    components = aggregate_fe_components(state)
    ledger = EpisodeLedger(tmp_path)
    ep1 = build_episode(
        at=NOW,
        turn_index=1,
        phase="user_turn",
        state=state,
        action="answer",
        action_trigger="user_message",
        evidence_refs=[],
        components_before=components,
        components_after=components,
        outcome_summary="settled",
    )
    ep2 = build_episode(
        at=NOW + 1,
        turn_index=2,
        phase="user_turn",
        state=state,
        action="answer",
        action_trigger="user_message",
        evidence_refs=[],
        components_before=components,
        components_after=components,
        outcome_summary="settled",
    )
    ledger.append(ep1)
    ledger.append(ep2)
    assert [episode.episode_id for episode in ledger.by_fingerprint(ep1.state_fingerprint, limit=2)] == [
        ep2.episode_id,
        ep1.episode_id,
    ]


def test_drive_pull_bonus_bounded_within_clamp(tmp_path: Path) -> None:
    state = {"m13_drive_state": default_m13_drive_state(), "temporal_state": {}}
    fp = state_fingerprint(state)
    before = aggregate_fe_components(state)
    after = {**before, "expectation_prediction_error_proxy": 0.0}
    ledger = EpisodeLedger(tmp_path)
    for index in range(3):
        ledger.append(
            build_episode(
                at=NOW + index,
                turn_index=index,
                phase="user_turn",
                state=state,
                action="answer",
                action_trigger="user_message",
                evidence_refs=[],
                components_before={**before, "expectation_prediction_error_proxy": 1.0},
                components_after=after,
                outcome_summary="confirmed",
            )
        )
    evaluation = M13DriveEvaluator().evaluate(
        user_text="status",
        user_id="default_user",
        turn_id="turn_0001",
        turn_index=1,
        conscious_plan={},
        memory_dynamics={"control_guidance": {}},
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state["m13_drive_state"],
        episode_ledger=ledger,
        current_state_fingerprint=fp,
    )
    bonus = evaluation.scores_by_action["answer"].get("m15_episode_ledger_pull_bonus", 0.0)
    assert 0.0 < bonus <= 0.1


def test_memory_efe_outreach_margin_reads_worse_ledger_history(tmp_path: Path) -> None:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["enabled"] = True
    initiative["implicit_idle_delivery"] = True
    m13["initiative"] = initiative
    state = {
        "pending_expectations": [
            {
                "id": "exp_a",
                "verify_on": "next_user_turn",
                "status": "pending",
                "content": "user promised benchmark result",
                "confidence": 0.95,
                "due_at_epoch": NOW - 3600,
                "expected_window_seconds": 900,
                "evidence_refs": ["mem_a"],
                "bound_memory_ids": ["ltm_a"],
            }
        ],
        "long_term_memory": [
            {
                "id": "ltm_a",
                "content": "user promised benchmark result",
                "confidence": 0.95,
                "salience": 0.9,
                "evidence_refs": ["mem_a"],
            }
        ],
        "open_items": [],
        "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_index": 4},
        "m13_drive_state": m13,
    }
    ledger = EpisodeLedger(tmp_path)
    base = aggregate_fe_components(state)
    worse = {**base, "expectation_prediction_error_proxy": 1.0}
    ledger.append(
        build_episode(
            at=NOW - 10,
            turn_index=4,
            phase="proactive_turn",
            state=state,
            action="proactive_outreach",
            action_trigger="memory_efe_outreach",
            evidence_refs=["ltm_a"],
            components_before=base,
            components_after=worse,
            outcome_summary="violated",
        )
    )
    result = evaluate_memory_efe(
        state,
        phase="idle",
        now=NOW,
        turn_index=5,
        user_active=False,
        retrieved_memories=state["long_term_memory"],
        episode_ledger=ledger,
    )
    assert result.policy_costs["ledger_outreach_margin_requirement_delta"] > 0


def test_prediction_settlement_addendum_is_written_with_m17_fields(tmp_path: Path) -> None:
    ledger = EpisodeLedger(tmp_path)

    event = ledger.append_prediction_settlement_addendum(
        at=NOW,
        turn_index=3,
        source_episode_id="ep:source",
        prediction_id="pred:p1",
        prediction_type="intent_prediction",
        outcome="violated",
        committed_confidence=0.78,
        prediction_error=1.514128,
        brier_score=0.6084,
        evidence_refs=["e1", "e2"],
        reason_codes=["counterfactual_failed"],
    )

    assert event["type"] == "PredictionSettlementAddendum"
    rows = _episode_rows(ledger.path)
    assert any(row.get("type") == "PredictionSettlementAddendum" and row.get("prediction_id") == "pred:p1" for row in rows)
