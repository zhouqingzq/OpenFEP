"""Shared fixtures for M16.1 gateway/runner tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_1_background_continuity import set_background_continuity_opt_in
from segmentum.dialogue.runtime.m16_api import M16Gateway
from segmentum.dialogue.runtime.m16_runner import ConsciousnessRunner
from segmentum.dialogue.runtime.m16_runtime_bridge import M16SessionBridge
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from tests.test_mvp_dialogue_runtime import FakeJSONLLM


NOW = 1_900_000_000


class _Clock:
    def __init__(self, start: int = NOW) -> None:
        self.value = int(start)

    def __call__(self) -> int:
        return self.value

    def advance(self, seconds: int) -> int:
        self.value += int(seconds)
        return self.value


def full_opted_state() -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [],
        "short_term_memory": [],
        "long_term_memory": [],
        "pending_expectations": [],
        "self_cognition": {"patch_history": []},
        "temporal_state": {
            "last_turn_at": NOW - 120,
            "last_user_turn_at": NOW - 120,
            "last_turn_index": 1,
            "last_reply": "ok",
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    m13 = set_background_continuity_opt_in(m13, enabled=True, runner_kind="m16_gateway_runner")
    state["m13_drive_state"] = m13
    return state


def build_stack(
    tmp_path: Path,
    *,
    persona_id: str = "p",
    session_id: str = "s",
    llm: Any | None = None,
    clock: _Clock | None = None,
) -> tuple[M16SessionBridge, M16WsHub, ConsciousnessRunner, _Clock]:
    clk = clock or _Clock()
    store = MVPStateStore(tmp_path)
    store.save(full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=llm or FakeJSONLLM())  # type: ignore[arg-type]
    bridge = M16SessionBridge(
        persona_id=persona_id,
        session_id=session_id,
        session_root=tmp_path,
        runtime=runtime,
        clock=clk,
    )
    hub = M16WsHub(persona_id=persona_id, session_id=session_id, clock=clk)
    runner = ConsciousnessRunner(bridge=bridge, hub=hub, clock=clk, silence_tick_seconds=1)
    return bridge, hub, runner, clk


M16_DEV_HEADERS = {"Authorization": "Bearer m16-test-token"}


def build_gateway(
    tmp_path: Path,
    *,
    persona_id: str = "p",
    session_id: str = "s",
    llm: Any | None = None,
    clock: _Clock | None = None,
) -> tuple[M16Gateway, M16SessionBridge, ConsciousnessRunner, _Clock]:
    clk = clock or _Clock()
    gateway = M16Gateway(
        dev_token="m16-test-token",
        llm_factory=lambda: llm or FakeJSONLLM(),
        clock=clk,
        session_root_resolver=lambda _p, _s: tmp_path,
    )
    handle = gateway.get_or_create_session(persona_id, session_id)
    handle.bridge.store.save(full_opted_state())
    runner = gateway.ensure_runner(handle)
    runner.silence_tick_seconds = 1
    return gateway, handle.bridge, runner, clk
