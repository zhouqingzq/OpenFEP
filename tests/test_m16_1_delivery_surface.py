from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from segmentum.dialogue.runtime.m14_1_background_continuity import enqueue_outreach_proposal
from segmentum.dialogue.runtime.m16_api import create_app
from segmentum.dialogue.runtime.m16_protocol import DELIVERY_SURFACE_READY_TTL_SECONDS
from tests.m16_1_test_helpers import build_gateway, build_stack
from tests.test_m14_2_self_loop_daemon import _DeliveryLLM


def _enqueue_test_outreach(tmp_path: Path, *, now: int) -> None:
    proposal = {
        "proposal_id": "prop_m16_test",
        "trigger": "scheduled_outreach",
        "source_intent_id": "sint_m16_test",
        "evidence_refs": ["env_due"],
        "traceable_expectation_id": "sint_m16_test",
        "ordinary_language_intent": "Send the scheduled follow-up.",
        "persona_id": "p",
        "session_id": "s",
    }
    enqueue_outreach_proposal(
        tmp_path,
        proposal=proposal,
        now=now,
        ttl_seconds=86400,
        source_intent_id="sint_m16_test",
    )


def test_ws_ready_marks_delivery_surface_available(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    hub.register_subscriber()
    event_id = bridge.append_perception_event(
        "DeliverySurfaceReadyEvent",
        {"client_id": "test"},
        source="test",
        correlation_id="corr_ready",
    )
    step = runner.run_once_for_tests(now=clk())
    assert any(row.get("delivery_surface_ready") for row in step.processed if isinstance(row, dict))
    allowed, _ = hub.outbox_drain_allowed(now=clk())
    assert allowed
    assert event_id


def test_ws_connect_without_ready_does_not_drain_outbox(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path, llm=_DeliveryLLM())  # type: ignore[arg-type]
    _enqueue_test_outreach(tmp_path, now=clk())
    app = create_app(gateway)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/personas/p/sessions/s/stream") as ws:
            ws.receive_json()
            ws.receive_json()
            step = runner.run_once_for_tests(now=clk())
            drain = next((row for row in step.processed if row.get("phase") == "outbox_drain"), {})
            assert drain.get("drained") is not True


def test_outbox_drains_through_m13_3_when_surface_ready(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path, llm=_DeliveryLLM())  # type: ignore[arg-type]
    _enqueue_test_outreach(tmp_path, now=clk())
    hub = gateway.get_or_create_session("p", "s").hub
    hub.register_subscriber()
    hub.mark_delivery_surface_ready(correlation_id="corr_drain")
    bridge.append_perception_event(
        "DeliverySurfaceReadyEvent",
        {"client_id": "test"},
        source="test",
        correlation_id="corr_drain",
    )
    step = runner.run_once_for_tests(now=clk())
    drain = next(row for row in step.processed if row.get("phase") == "outbox_drain")
    assert drain.get("drained") is True
    assert any(msg.get("kind") == "ProactiveMessageCommitted" for msg in step.actuation_messages)


def test_delivery_surface_ready_ttl_blocks_stale_drain(tmp_path: Path) -> None:
    _, hub, runner, clk = build_stack(tmp_path)
    hub.register_subscriber()
    hub.mark_delivery_surface_ready(correlation_id="corr_stale")
    clk.advance(DELIVERY_SURFACE_READY_TTL_SECONDS + 1)
    allowed, code = hub.outbox_drain_allowed(now=clk())
    assert not allowed
    assert code == "delivery_surface_not_ready"
