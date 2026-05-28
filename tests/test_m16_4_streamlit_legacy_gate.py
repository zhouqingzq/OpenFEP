from __future__ import annotations

from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m16_streamlit_legacy import (
    ENV_LEGACY_SCHEDULER,
    ENV_RUNNER_MODE,
    STREAMLIT_LEGACY_BANNER_MARKERS,
    consciousness_gateway_url,
    legacy_streamlit_scheduler_enabled,
    runner_mode_active,
    streamlit_legacy_banner_markdown,
    streamlit_scheduling_allowed,
)

APP_PY = Path(__file__).resolve().parents[1] / "segmentum" / "dialogue" / "runtime" / "app.py"


def test_implicit_idle_disabled_by_default_when_runner_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_RUNNER_MODE, "1")
    monkeypatch.delenv(ENV_LEGACY_SCHEDULER, raising=False)
    assert runner_mode_active() is True
    assert streamlit_scheduling_allowed() is False
    assert legacy_streamlit_scheduler_enabled() is False


def test_streamlit_banner_present() -> None:
    source = APP_PY.read_text(encoding="utf-8")
    assert "streamlit_legacy_banner_markdown" in source
    banner = streamlit_legacy_banner_markdown(gateway_url="http://127.0.0.1:8765")
    for marker in STREAMLIT_LEGACY_BANNER_MARKERS:
        assert marker in banner.lower() or marker in banner


def test_app_does_not_schedule_idle_tick_when_legacy_scheduler_off() -> None:
    source = APP_PY.read_text(encoding="utf-8")
    assert "streamlit_scheduling_allowed()" in source
    assert "_run_streamlit_implicit_idle_delivery_tick()" in source
    assert "if streamlit_scheduling_allowed():" in source
    assert "record_background_streamlit_ping()" in source


def test_legacy_scheduler_flag_restores_old_behavior_for_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LEGACY_SCHEDULER, "1")
    monkeypatch.delenv(ENV_RUNNER_MODE, raising=False)
    assert legacy_streamlit_scheduler_enabled() is True
    assert streamlit_scheduling_allowed() is True


def test_runner_mode_overrides_legacy_scheduler_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LEGACY_SCHEDULER, "1")
    monkeypatch.setenv(ENV_RUNNER_MODE, "1")
    assert runner_mode_active() is True
    assert legacy_streamlit_scheduler_enabled() is False
    assert streamlit_scheduling_allowed() is False


def test_consciousness_gateway_url_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTS_CONSCIOUSNESS_GATEWAY_URL", "http://127.0.0.1:8765")
    assert consciousness_gateway_url() == "http://127.0.0.1:8765"


def test_m14_4_helper_suppressed_when_legacy_scheduler_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_LEGACY_SCHEDULER, raising=False)
    monkeypatch.delenv(ENV_RUNNER_MODE, raising=False)

    class _Chat:
        def append_m14_4_implicit_idle_audit(self, event: dict[str, object]) -> None:
            self.event = event

    chat = _Chat()
    from segmentum.dialogue.runtime.m14_4_implicit_idle import run_streamlit_implicit_idle_proactive

    result = run_streamlit_implicit_idle_proactive(chat, session_state={}, now_mono=0.0)
    assert result.attempted is False
    assert result.suppression_reason_code == "m16_legacy_scheduler_off"
