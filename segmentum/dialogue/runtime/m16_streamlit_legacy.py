"""M16.4 Streamlit legacy scheduler gate helpers.

Streamlit reruns must not schedule Path B cognition after M16.4 unless an
operator explicitly opts into the legacy scheduler. Runner mode always wins.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ENV_LEGACY_SCHEDULER = "SEGMENTS_LEGACY_STREAMLIT_SCHEDULER"
ENV_RUNNER_MODE = "M16_RUNNER"
ENV_GATEWAY_URL = "SEGMENTS_CONSCIOUSNESS_GATEWAY_URL"

DEFAULT_LEGACY_SCHEDULER = "0"

STREAMLIT_LEGACY_BANNER_MARKERS = ("ui/web", "ui/tui", "legacy")


def _env_source(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def runner_mode_active(env: Mapping[str, str] | None = None) -> bool:
    source = _env_source(env)
    return str(source.get(ENV_RUNNER_MODE, "")).strip() == "1"


def legacy_streamlit_scheduler_enabled(env: Mapping[str, str] | None = None) -> bool:
    if runner_mode_active(env):
        return False
    source = _env_source(env)
    return str(source.get(ENV_LEGACY_SCHEDULER, DEFAULT_LEGACY_SCHEDULER)).strip() == "1"


def streamlit_scheduling_allowed(env: Mapping[str, str] | None = None) -> bool:
    """True only when legacy Streamlit scheduling is explicitly enabled."""
    return legacy_streamlit_scheduler_enabled(env)


def consciousness_gateway_url(env: Mapping[str, str] | None = None) -> str:
    source = _env_source(env)
    return str(source.get(ENV_GATEWAY_URL, "")).strip()


def streamlit_legacy_banner_markdown(*, gateway_url: str | None = None) -> str:
    url = consciousness_gateway_url() if gateway_url is None else str(gateway_url).strip()
    gateway_hint = f" Gateway: `{url}`." if url else ""
    return (
        "**Legacy Streamlit I/O only.** Path B cognition is scheduled by the M16 "
        f"Consciousness Runner, not Streamlit reruns.{gateway_hint} "
        "Use **`ui/web`** or **`ui/tui`** with "
        "`python -m segmentum.dialogue.runtime.m16_api`. "
        "Streamlit idle/proactive scheduling is off unless "
        f"`{ENV_LEGACY_SCHEDULER}=1` and `{ENV_RUNNER_MODE}` is not set."
    )


def legacy_scheduler_suppression_result() -> dict[str, Any]:
    return {
        "attempted": False,
        "delivered": False,
        "suppression_reason_code": "m16_legacy_scheduler_off",
    }
