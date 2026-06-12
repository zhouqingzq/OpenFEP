from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.m16_api import M16Gateway
from segmentum.dialogue.runtime.mvp_loop import default_openrouter_client
from segmentum.dialogue.runtime.m16_runner import ConsciousnessRunner
from segmentum.dialogue.runtime.m16_runtime_bridge import M16SessionBridge
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from tests.m16_1_test_helpers import NOW, full_opted_state


def test_default_openrouter_client_reads_secrets_file(tmp_path: Path, monkeypatch) -> None:
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    config_path = secrets_dir / "openrouter.json"
    config_path.write_text(
        '{"api_key":"test-key","model":"deepseek/deepseek-v4-flash","base_url":"https://openrouter.ai/api/v1"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("segmentum.dialogue.runtime.mvp_loop.openrouter_secrets_path", lambda: config_path)
    client = default_openrouter_client()
    assert client is not None
    assert client.api_key == "test-key"


def test_default_openrouter_client_returns_none_without_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "segmentum.dialogue.runtime.mvp_loop.openrouter_secrets_path",
        lambda: Path("/nonexistent/openrouter.json"),
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert default_openrouter_client() is None


def test_gateway_session_uses_default_llm_factory(tmp_path: Path, monkeypatch) -> None:
    class _LLM:
        api_key = "test-key"

    monkeypatch.setattr(
        "segmentum.dialogue.runtime.m16_api.default_openrouter_client",
        lambda: _LLM(),
    )
    gateway = M16Gateway(session_root_resolver=lambda _p, _s: tmp_path)
    handle = gateway.get_or_create_session("p", "s")
    assert isinstance(handle.bridge.runtime.llm, _LLM)
    assert handle.bridge.runtime.persona_name == "p"


def test_snapshot_includes_llm_runtime_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "segmentum.dialogue.runtime.m16_runtime_bridge.llm_configuration_status_with_source",
        lambda llm: {"available": False, "reason": "llm_unavailable", "config_source": ""},
    )
    store = MVPStateStore(tmp_path)
    store.save(full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=None)
    bridge = M16SessionBridge(
        persona_id="p",
        session_id="s",
        session_root=tmp_path,
        runtime=runtime,
        clock=lambda: NOW,
    )
    snapshot = bridge.snapshot()
    llm = snapshot["runtime_hints"]["llm"]
    assert llm["available"] is False
    assert llm["reason"] == "llm_unavailable"
    assert llm["config_source"] == ""


def test_runner_publishes_error_when_turn_fails(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=None)
    bridge = M16SessionBridge(
        persona_id="p",
        session_id="s",
        session_root=tmp_path,
        runtime=runtime,
        clock=lambda: NOW,
    )
    hub = M16WsHub(persona_id="p", session_id="s", clock=lambda: NOW)
    runner = ConsciousnessRunner(bridge=bridge, hub=hub, clock=lambda: NOW, silence_tick_seconds=60)

    def _boom(_text: str, *, turn_index: int, now: int) -> object:
        raise AttributeError("llm_unavailable")

    runner._inline_run_turn = _boom  # type: ignore[method-assign]
    bridge.append_client_input(text="hello", correlation_id="corr_fail")
    step = runner.run_once_for_tests(now=NOW)
    assert any(msg.get("kind") == "Error" for msg in step.actuation_messages)
    payload = next(msg for msg in step.actuation_messages if msg.get("kind") == "Error")["payload"]
    assert payload["code"] == "turn_failed"
