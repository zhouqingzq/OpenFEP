from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from segmentum.dialogue.runtime.chat import ChatInterface
from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import (
    BOUNDED_DEFAULT_PROFILE,
    STREAMLIT_OPEN_CHAT_PROFILE,
    ProactiveTurnProposal,
    default_initiative_state,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    set_initiative_implicit_idle_delivery,
    set_initiative_proactive_policy_profile,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m14_4_implicit_idle import (
    compute_idle_seconds,
    run_streamlit_implicit_idle_proactive,
    should_attempt_implicit_idle_proactive,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_900_000_000


@pytest.fixture(autouse=True)
def _enable_legacy_streamlit_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTS_LEGACY_STREAMLIT_SCHEDULER", "1")
    monkeypatch.delenv("M16_RUNNER", raising=False)


def _m13_enabled(*, profile: str = BOUNDED_DEFAULT_PROFILE, implicit_idle: bool = True) -> dict[str, object]:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    m13 = set_initiative_implicit_idle_delivery(m13, enabled=implicit_idle)
    m13 = set_initiative_proactive_policy_profile(m13, profile=profile)
    return m13


def _locked_proposal(now: int = NOW) -> ProactiveTurnProposal:
    return ProactiveTurnProposal(
        proposal_id="prop_locked",
        created_at=now,
        source="m13_initiative_policy",
        trigger="memory_efe_outreach",
        trigger_evidence_refs=["mem_1"],
        urgency_band="medium",
        expected_user_value_band="medium",
        risk_band="low",
        proposed_action="answer",
        proposed_topic="traceable expectation",
        ordinary_language_intent="Follow up on the unresolved expectation: traceable expectation",
        expires_at=now + 300,
        cooldown_cost=2,
        traceable_expectation_id="exp_1",
        source_kind="pending_expectation",
        selection_reason_codes=["memory_efe_should_outreach"],
    )


def test_fresh_session_uses_m14_5_conservative_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEGMENTUM_PROACTIVE_PROFILE", raising=False)
    initiative = default_initiative_state()
    assert initiative["implicit_idle_delivery"] is False
    assert initiative["proactive_policy_profile"] == BOUNDED_DEFAULT_PROFILE
    assert initiative["max_proactive_per_session"] == 1


def test_streamlit_open_chat_profile_respects_session_limit_by_default() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["proactive_count_this_session"] = 1
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "session_limit_reached"


def test_streamlit_open_chat_profile_skips_session_limit_only_with_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("SEGMENTUM_STREAMLIT_OPEN_CHAT_RELAX_PROACTIVE_CAPS", "1")
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["proactive_count_this_session"] = 1
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check.proposal is not None
    assert check.proposal.proposal_id == "prop_locked"


def test_locked_memory_efe_without_refs_rejected_as_untraceable() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    proposal = _locked_proposal()
    proposal.source = "queued_outreach"
    proposal.trigger_evidence_refs = []
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=proposal,
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "no_traceable_proactive_target"


def test_locked_generic_open_item_memory_efe_rejected_as_untraceable() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    proposal = _locked_proposal()
    proposal.source = "queued_outreach"
    proposal.traceable_expectation_id = "item_001"
    proposal.trigger_evidence_refs = ["item_001"]
    proposal.source_kind = "open_item"
    proposal.proposed_topic = "unclear user intent"
    proposal.ordinary_language_intent = "Follow up on the unresolved expectation: unclear user intent"
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=proposal,
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "no_traceable_proactive_target"


def test_streamlit_open_chat_profile_respects_cooldown_by_default() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["last_proactive_turn_index"] = 2
    initiative["cooldown_turns"] = 2
    initiative["cooldown_until_timestamp"] = NOW + 120
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "cooldown_active"


def test_streamlit_open_chat_profile_skips_cooldown_only_with_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("SEGMENTUM_STREAMLIT_OPEN_CHAT_RELAX_PROACTIVE_CAPS", "1")
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["last_proactive_turn_index"] = 2
    initiative["cooldown_turns"] = 2
    initiative["cooldown_until_timestamp"] = NOW + 120
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check.proposal is not None


def test_bounded_default_profile_still_enforces_session_limit_and_cooldown() -> None:
    state = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _m13_enabled(profile=BOUNDED_DEFAULT_PROFILE),
    }
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["proactive_count_this_session"] = 1
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "session_limit_reached"

    initiative["proactive_count_this_session"] = 0
    initiative["last_proactive_turn_index"] = 2
    initiative["cooldown_turns"] = 2
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check2 = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        locked_proposal=_locked_proposal(),
        llm=None,
    )
    assert check2.proposal is None
    assert check2.suppression_reason_code == "cooldown_active"


def test_compute_idle_seconds_from_last_user_turn_at_with_fallback() -> None:
    assert compute_idle_seconds({"temporal_state": {"last_user_turn_at": 100}}, now=160) == 60
    assert compute_idle_seconds({"temporal_state": {"last_turn_at": 120}}, now=160) == 40
    assert compute_idle_seconds({"temporal_state": {}}, now=160) == 0


def test_implicit_idle_request_false_when_idle_below_threshold() -> None:
    ok, reason = should_attempt_implicit_idle_proactive(
        {},
        now_mono=10,
        idle_seconds=20,
        initiative_status={
            "user_opt_in": True,
            "enabled": True,
            "implicit_idle_delivery": True,
            "idle_threshold_seconds": 30,
        },
        mvp_runtime_active=True,
        has_agent=True,
    )
    assert ok is False
    assert reason == "idle_time_too_short"


@dataclass
class _FakeResponse:
    reply: str
    diagnostics: dict[str, object]


class _FakeChat:
    mvp_runtime_active = True

    def __init__(self) -> None:
        self.state = {"temporal_state": {"last_user_turn_at": NOW - 300}}
        self.initiative = {
            "user_opt_in": True,
            "enabled": True,
            "implicit_idle_delivery": True,
            "idle_threshold_seconds": 120,
            "proactive_policy_profile": STREAMLIT_OPEN_CHAT_PROFILE,
        }
        self.audit_events: list[dict[str, object]] = []
        self.propose_calls = 0
        self.run_calls = 0

    def has_agent(self) -> bool:
        return True

    def read_mvp_state_dict(self) -> dict[str, object]:
        return self.state

    def read_initiative_status(self) -> dict[str, object]:
        return self.initiative

    def maybe_propose_proactive_turn(self, **kwargs: object) -> dict[str, object]:
        self.propose_calls += 1
        assert kwargs["implicit_idle_request"] is True
        assert float(kwargs["idle_seconds"]) >= 120
        return {"proposal": {"proposal_id": "prop_1"}, "suppression_reason_code": ""}

    def run_proactive_turn(self, proposal_id: str, *, speaker_name: str = "") -> _FakeResponse:
        self.run_calls += 1
        assert proposal_id == "prop_1"
        return _FakeResponse(reply="跟进一下刚才那个未闭合的问题。", diagnostics={})

    def append_m14_4_implicit_idle_audit(self, event: dict[str, object]) -> None:
        self.audit_events.append(event)


def test_implicit_idle_proposal_when_idle_above_threshold_and_open_page() -> None:
    chat = _FakeChat()
    session: dict[str, object] = {}
    result = run_streamlit_implicit_idle_proactive(
        chat,
        session_state=session,
        now=NOW,
        now_mono=100,
        speaker_name="zq",
        min_attempt_interval_seconds=45,
    )
    assert result.attempted is True
    assert result.delivered is True
    assert result.rerun_requested is True
    assert chat.propose_calls == 1
    assert chat.run_calls == 1
    assert chat.audit_events[-1]["type"] == "M14ImplicitIdleProactiveCheckEvent"
    assert chat.audit_events[-1]["proposal_id"] == "prop_1"


def test_user_typing_suppresses_implicit_idle_check_before_proposal() -> None:
    chat = _FakeChat()
    result = run_streamlit_implicit_idle_proactive(
        chat,
        session_state={},
        now=NOW,
        now_mono=100,
        user_typing=True,
    )
    assert result.attempted is False
    assert result.suppression_reason_code == "user_active"
    assert chat.propose_calls == 0


def test_throttle_prevents_back_to_back_implicit_idle_proposals() -> None:
    chat = _FakeChat()
    session: dict[str, object] = {}
    first = run_streamlit_implicit_idle_proactive(
        chat,
        session_state=session,
        now=NOW,
        now_mono=100,
        min_attempt_interval_seconds=45,
    )
    second = run_streamlit_implicit_idle_proactive(
        chat,
        session_state=session,
        now=NOW + 1,
        now_mono=120,
        min_attempt_interval_seconds=45,
    )
    assert first.delivered is True
    assert second.attempted is False
    assert second.suppression_reason_code == "implicit_idle_attempt_throttled"
    assert chat.propose_calls == 1


def test_chat_interface_suppressed_proactive_does_not_advance_turn_index() -> None:
    class _Runtime:
        def run_proactive_turn(self, *, proposal_id: str, turn_index: int, speaker_name: str = "") -> SimpleNamespace:
            assert proposal_id == "prop_1"
            assert turn_index == 4
            return SimpleNamespace(
                reply="",
                action="proactive_suppressed",
                diagnostics={"suppression_reason": "delivery_assessor_reject"},
                followup_replies=[],
            )

    class _Safety:
        def enforce(self, text: str, obs_channels: dict[str, object]) -> tuple[str, list[object]]:
            return text, []

    traits = SimpleNamespace(to_dict=lambda: {"openness": 0.0})
    profile = SimpleNamespace(
        openness=0.0,
        conscientiousness=0.0,
        extraversion=0.0,
        agreeableness=0.0,
        neuroticism=0.0,
    )
    chat = ChatInterface.__new__(ChatInterface)
    chat._use_mvp_runtime = True
    chat._mvp_runtime = _Runtime()
    chat._agent = SimpleNamespace(
        slow_variable_learner=SimpleNamespace(state=SimpleNamespace(traits=traits)),
        self_model=SimpleNamespace(personality_profile=profile),
    )
    chat._safety = _Safety()
    chat._last_obs_channels = {}
    chat._last_action = "answer"
    chat._turn_index = 4
    chat._transcript = []
    chat._session_id = "s"

    response = chat.run_proactive_turn("prop_1")

    assert response.reply == ""
    assert response.diagnostics["suppression_reason"] == "delivery_assessor_reject"
    assert chat._turn_index == 4
    assert chat._transcript == []


def test_policy_profile_persists_in_m13_drive_state(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess"), llm=None)
    status = runtime.set_initiative_proactive_policy_profile(STREAMLIT_OPEN_CHAT_PROFILE)
    assert status["proactive_policy_profile"] == STREAMLIT_OPEN_CHAT_PROFILE
    reloaded = normalize_initiative_state(
        normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])["initiative"]
    )
    assert reloaded["proactive_policy_profile"] == STREAMLIT_OPEN_CHAT_PROFILE

    runtime.set_initiative_user_opt_in(False)
    reloaded_off = normalize_initiative_state(
        normalize_m13_drive_state(runtime.store.load()["m13_drive_state"])["initiative"]
    )
    assert reloaded_off["proactive_policy_profile"] == BOUNDED_DEFAULT_PROFILE


def test_no_keyword_cue_or_vague_later_open_item_creates_implicit_target() -> None:
    state = {
        "open_items": [
            {
                "id": "oi_later",
                "status": "open",
                "title": "generic sunset",
                "next_check": "later",
                "content": "later",
            }
        ],
        "temporal_state": {"last_user_text": "later later later"},
        "m13_drive_state": _m13_enabled(profile=STREAMLIT_OPEN_CHAT_PROFILE),
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=3,
        implicit_idle_request=True,
        idle_seconds=999,
        llm=None,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "no_traceable_proactive_target"
