from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from segmentum.dialogue.runtime.m16_api import create_app
from segmentum.dialogue.runtime.m16_protocol import SCHEMA_VERSION
from segmentum.version import __version__
from tests.m16_1_test_helpers import M16_DEV_HEADERS, build_gateway


def test_post_input_appends_event_without_inline_run_turn(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    app = create_app(gateway)
    calls: list[str] = []

    def _track_run_turn(text: str, **kwargs: object) -> object:
        calls.append(text)
        raise AssertionError("run_turn must not run on gateway request thread")

    with patch.object(bridge.runtime, "run_turn", side_effect=_track_run_turn):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/personas/p/sessions/s/input",
                json={
                    "text": "hello gateway",
                    "correlation_id": "corr_http",
                    "ingress_evidence_band": "structured_partial",
                },
                headers=M16_DEV_HEADERS,
            )
    assert resp.status_code == 202
    assert resp.json()["accepted"] is True
    assert resp.json()["event_id"]
    assert calls == []
    events = bridge.event_store.query_events(event_types={"ClientInputCommittedEvent"})
    assert len(events) == 1
    assert events[0]["payload"]["text"] == "hello gateway"
    assert events[0]["payload"]["ingress_evidence_band"] == "structured_partial"


def test_runner_survives_gateway_restart_with_durable_events(tmp_path: Path) -> None:
    gateway_a, bridge, runner_a, clk = build_gateway(tmp_path)
    app_a = create_app(gateway_a)
    with TestClient(app_a) as client:
        resp = client.post(
            "/v1/personas/p/sessions/s/input",
            json={"text": "durable", "correlation_id": "corr_durable"},
            headers=M16_DEV_HEADERS,
        )
        assert resp.status_code == 202
        event_id = resp.json()["event_id"]
    step_a = runner_a.run_once_for_tests(now=clk())
    assert bridge.is_event_processed(event_id)

    gateway_b = build_gateway(tmp_path)[0]
    runner_b = gateway_b.ensure_runner(gateway_b.get_or_create_session("p", "s"))
    step_b = runner_b.run_once_for_tests(now=clk())
    assert not any(msg.get("kind") == "AssistantMessageCommitted" for msg in step_b.actuation_messages)
    snapshot = gateway_b.get_or_create_session("p", "s").bridge.snapshot()
    assert snapshot["runtime_hints"]["last_turn_index"] >= 1


def test_health_and_runner_status_routes(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    app = create_app(gateway)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["version"] == __version__
        assert health.json()["protocol_version"] == SCHEMA_VERSION
        status = client.get("/v1/personas/p/sessions/s/runner/status")
        assert status.status_code == 200
        assert status.json()["runner"]["runner_kind"] == "m16_gateway_runner"
