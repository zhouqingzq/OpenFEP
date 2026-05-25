from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_4_implicit_idle import run_streamlit_implicit_idle_proactive
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


def _m13_enabled() -> dict[str, object]:
    m13 = set_initiative_user_opt_in({}, enabled=True)
    m13 = normalize_m13_drive_state(m13)
    initiative = dict(m13["initiative"])
    initiative["enabled"] = True
    initiative["implicit_idle_delivery"] = True
    initiative["proactive_policy_profile"] = "streamlit_open_chat"
    m13["initiative"] = initiative
    return m13


class _NoLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **_: object) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("idle cognitive tick must not call the LLM")


def test_idle_cognitive_tick_emits_composite_event_and_target(tmp_path: Path) -> None:
    llm = _NoLLM()
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "tick_target"), llm=llm)
    state = runtime.store.load()
    state.update(
        {
            "open_items": [
                {
                    "id": "oi_due",
                    "status": "open",
                    "title": "benchmark follow-up",
                    "scheduled_intent_id": "intent_benchmark",
                    "due_at_epoch": NOW - 10_000,
                    "expected_window_seconds": 900,
                    "evidence_refs": ["mem_oi"],
                    "confidence": 0.95,
                }
            ],
            "long_term_memory": [{"id": "mem_oi", "content": "benchmark result thread", "salience": 0.8}],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": _m13_enabled(),
        }
    )
    runtime.store.save(state)

    result = runtime.run_idle_cognitive_tick(turn_index=5, idle_seconds=999, now=NOW)

    assert llm.calls == 0
    assert result["selected_target"]["trigger"] == "memory_efe_outreach"
    assert result["reject_reason"] == ""
    tick = next(event for event in result["events"] if event["type"] == "IdleCognitiveTickEvent")
    assert tick["retrieved_ids"]
    assert tick["memory_efe_should_outreach"] is True
    assert tick["bands"].keys() == {"boredom_band", "reward_band", "behavior_band", "relation_band"}


def test_idle_cognitive_tick_does_not_increase_boredom_from_silence_alone(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "tick_silence"), llm=None)
    state = runtime.store.load()
    m13 = _m13_enabled()
    m13["boredom"]["boredom_level"] = 0.12
    state.update(
        {
            "open_items": [],
            "pending_expectations": [],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200},
            "m13_drive_state": m13,
        }
    )
    runtime.store.save(state)

    runtime.run_idle_cognitive_tick(turn_index=6, idle_seconds=999, now=NOW)

    after = normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])
    assert after["boredom"]["boredom_level"] == 0.12


class _FakeChat:
    mvp_runtime_active = True

    def __init__(self, tick: dict[str, object]) -> None:
        self.tick = tick
        self.propose_calls = 0
        self.audit_events: list[dict[str, object]] = []

    def has_agent(self) -> bool:
        return True

    def read_mvp_state_dict(self) -> dict[str, object]:
        return {"temporal_state": {"last_user_turn_at": NOW - 999}}

    def read_initiative_status(self) -> dict[str, object]:
        return {
            "user_opt_in": True,
            "enabled": True,
            "implicit_idle_delivery": True,
            "idle_threshold_seconds": 120,
            "proactive_policy_profile": "streamlit_open_chat",
        }

    def run_idle_cognitive_tick(self, **_: object) -> dict[str, object]:
        return self.tick

    def maybe_propose_proactive_turn(self, **kwargs: object) -> dict[str, object]:
        self.propose_calls += 1
        assert kwargs["preselected_target"]["trigger"] == "memory_efe_outreach"
        return {"proposal": {"proposal_id": "prop_1"}, "suppression_reason_code": ""}

    def run_proactive_turn(self, proposal_id: str, *, speaker_name: str = "") -> object:
        class Response:
            reply = "follow-up"
            diagnostics: dict[str, object] = {}

        assert proposal_id == "prop_1"
        return Response()

    def append_m14_4_implicit_idle_audit(self, event: dict[str, object]) -> None:
        self.audit_events.append(event)


def test_streamlit_idle_uses_tick_reject_reason_before_proposal() -> None:
    chat = _FakeChat({"selected_target": None, "reject_reason": "memory_efe_below_outreach_margin", "events": []})

    result = run_streamlit_implicit_idle_proactive(chat, session_state={}, now=NOW, now_mono=10)

    assert result.attempted is True
    assert result.delivered is False
    assert result.suppression_reason_code == "memory_efe_below_outreach_margin"
    assert chat.propose_calls == 0


def test_streamlit_idle_passes_tick_target_to_m13_3() -> None:
    chat = _FakeChat(
        {
            "selected_target": {
                "trigger": "memory_efe_outreach",
                "traceable_expectation_id": "exp_1",
                "evidence_refs": ["mem_1"],
                "ordinary_language_intent": "Follow up on exp_1",
                "source_kind": "pending_expectation",
            },
            "reject_reason": "",
            "events": [],
        }
    )

    result = run_streamlit_implicit_idle_proactive(chat, session_state={}, now=NOW, now_mono=10)

    assert result.delivered is True
    assert chat.propose_calls == 1
