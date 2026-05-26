from __future__ import annotations

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m15_meta_control import apply_trigger_suppression_intent
from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, build_episode, aggregate_fe_components


def test_idle_cognitive_tick_suppression_applies_to_memory_efe_outreach(tmp_path) -> None:
    store_root = tmp_path / "sess"
    store_root.mkdir()
    ledger = EpisodeLedger(store_root)
    state = {"m13_drive_state": default_m13_drive_state()}
    state["m13_drive_state"]["meta_control_intents"] = {
        "active": [
            {
                "intent_id": "meta_1",
                "intent_kind": "suppress_action_trigger_for_n_turns",
                "payload": {"action_trigger": "idle_cognitive_tick", "ttl_turns": 3},
                "turn_index": 1,
                "at": 1_700_000_000,
            }
        ],
        "consumed": [],
        "recent_detections": [],
    }
    updated, intent, events = apply_trigger_suppression_intent(
        state,
        action_trigger="memory_efe_outreach",
        now=1_700_000_100,
        turn_index=2,
    )
    assert intent is not None
    assert any(event.get("type") == "MetaControlInterventionAppliedEvent" for event in events)
