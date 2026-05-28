"""M16.1 FastAPI/WebSocket gateway for the Path B consciousness runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from segmentum.dialogue.runtime.m14_2_self_loop import default_session_root
from segmentum.dialogue.runtime.m16_protocol import (
    ENGINEERING_PROXY_LABEL,
    MAX_CORRELATION_ID_CHARS,
    MAX_INPUT_TEXT_CHARS,
    SCHEMA_VERSION,
    bounded_snapshot_shape,
    build_ws_client_message,
    gateway_mutation_allowed,
    runner_control_payload_is_bounded,
    validate_ws_client_message,
)
from segmentum.dialogue.runtime.m16_runner import ConsciousnessRunner
from segmentum.dialogue.runtime.m16_runtime_bridge import M16SessionBridge, RUNNER_KIND
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    default_openrouter_client,
    llm_configuration_status_with_source,
    openrouter_secrets_path,
)


def _resolve_session_llm(gateway: "M16Gateway") -> Any | None:
    if gateway.llm_factory is not None:
        return gateway.llm_factory()
    return default_openrouter_client()


class CreateSessionBody(BaseModel):
    correlation_id: str = Field(max_length=MAX_CORRELATION_ID_CHARS)
    session_id: str = Field(default="", max_length=120)


class ClientInputBody(BaseModel):
    text: str = Field(max_length=MAX_INPUT_TEXT_CHARS)
    correlation_id: str = Field(max_length=MAX_CORRELATION_ID_CHARS)
    speaker_name: str = Field(default="", max_length=64)


class RunnerControlBody(BaseModel):
    correlation_id: str = Field(max_length=MAX_CORRELATION_ID_CHARS)
    command: str
    reason: str = Field(default="", max_length=160)


@dataclass
class M16SessionHandle:
    persona_id: str
    session_id: str
    session_root: Path
    bridge: M16SessionBridge
    hub: M16WsHub
    runner: ConsciousnessRunner | None = None


@dataclass
class M16Gateway:
    dev_token: str = ""
    llm_factory: Callable[[], Any] | None = None
    clock: Callable[[], int] | None = None
    session_root_resolver: Callable[[str, str], Path] | None = None
    sessions: dict[tuple[str, str], M16SessionHandle] = field(default_factory=dict)

    def resolve_session_root(self, persona_id: str, session_id: str) -> Path:
        if self.session_root_resolver is not None:
            return Path(self.session_root_resolver(persona_id, session_id))
        return default_session_root(persona_id, session_id)

    def get_session(self, persona_id: str, session_id: str) -> M16SessionHandle:
        key = (persona_id, session_id)
        handle = self.sessions.get(key)
        if handle is None:
            raise KeyError(key)
        return handle

    def get_or_create_session(self, persona_id: str, session_id: str) -> M16SessionHandle:
        key = (persona_id, session_id)
        existing = self.sessions.get(key)
        if existing is not None:
            return existing
        root = self.resolve_session_root(persona_id, session_id)
        store = MVPStateStore(root)
        llm = _resolve_session_llm(self)
        runtime = MVPDialogueRuntime(store=store, llm=llm)
        bridge = M16SessionBridge(
            persona_id=persona_id,
            session_id=session_id,
            session_root=root,
            runtime=runtime,
            clock=self.clock,
        )
        hub = M16WsHub(persona_id=persona_id, session_id=session_id, clock=self.clock)
        handle = M16SessionHandle(
            persona_id=persona_id,
            session_id=session_id,
            session_root=root,
            bridge=bridge,
            hub=hub,
        )
        self.sessions[key] = handle
        return handle

    def ensure_runner(self, handle: M16SessionHandle) -> ConsciousnessRunner:
        if handle.runner is None:
            handle.runner = ConsciousnessRunner(
                bridge=handle.bridge,
                hub=handle.hub,
                clock=self.clock,
            )
        return handle.runner


def _client_host(request: Request) -> str:
    return request.client.host if request.client else ""


def _check_mutation(request: Request, gateway: M16Gateway) -> None:
    ok, reason = gateway_mutation_allowed(
        client_host=_client_host(request),
        authorization_header=request.headers.get("authorization"),
        configured_dev_token=gateway.dev_token or None,
    )
    if not ok:
        raise HTTPException(status_code=403, detail=reason)


def _append_gateway_audit(
    bridge: M16SessionBridge,
    *,
    typ: str,
    correlation_id: str = "",
    **fields: Any,
) -> None:
    bridge.append_runner_audit(typ=typ, correlation_id=correlation_id, **fields)


def create_app(gateway: M16Gateway | None = None) -> FastAPI:
    gw = gateway or M16Gateway()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway = gw
        yield

    app = FastAPI(title="Segmentum Consciousness Gateway", version=SCHEMA_VERSION, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION}

    @app.post("/v1/personas/{persona_id}/sessions")
    async def create_session(persona_id: str, body: CreateSessionBody, request: Request) -> JSONResponse:
        _check_mutation(request, gw)
        session_id = body.session_id.strip() or f"sess_{uuid.uuid4().hex[:12]}"
        handle = gw.get_or_create_session(persona_id, session_id)
        return JSONResponse(
            status_code=201,
            content={
                "persona_id": persona_id,
                "session_id": session_id,
                "schema_version": SCHEMA_VERSION,
                "correlation_id": body.correlation_id[:MAX_CORRELATION_ID_CHARS],
            },
        )

    @app.get("/v1/personas/{persona_id}/sessions/{session_id}")
    async def session_metadata(persona_id: str, session_id: str) -> dict[str, Any]:
        try:
            handle = gw.get_session(persona_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session_not_found") from exc
        runner = handle.runner
        status = runner.status() if runner is not None else None
        return {
            "persona_id": persona_id,
            "session_id": session_id,
            "schema_version": SCHEMA_VERSION,
            "runner": status.to_dict() if status is not None else {"running": False},
            "llm": llm_configuration_status_with_source(handle.bridge.runtime.llm),
            "delivery_surface": {
                "ws_subscribed": handle.hub.delivery_state().ws_subscribed,
                "delivery_surface_ready_at": handle.hub.delivery_state().delivery_surface_ready_at,
            },
        }

    @app.post("/v1/personas/{persona_id}/sessions/{session_id}/input")
    async def post_input(persona_id: str, session_id: str, body: ClientInputBody, request: Request) -> JSONResponse:
        _check_mutation(request, gw)
        handle = gw.get_or_create_session(persona_id, session_id)
        event_id = handle.bridge.append_client_input(
            text=body.text,
            correlation_id=body.correlation_id,
            speaker_name=body.speaker_name,
        )
        _append_gateway_audit(
            handle.bridge,
            typ="GatewayInputAcceptedEvent",
            correlation_id=body.correlation_id,
            event_id=event_id,
        )
        runner = gw.ensure_runner(handle)
        if not runner.status().running:
            runner.start()
        runner.nudge()
        return JSONResponse(
            status_code=202,
            content={
                "accepted": True,
                "event_id": event_id,
                "persona_id": persona_id,
                "session_id": session_id,
                "correlation_id": body.correlation_id,
                "schema_version": SCHEMA_VERSION,
            },
        )

    @app.get("/v1/personas/{persona_id}/sessions/{session_id}/snapshot")
    async def get_snapshot(persona_id: str, session_id: str) -> dict[str, Any]:
        handle = gw.get_or_create_session(persona_id, session_id)
        snapshot = handle.bridge.snapshot()
        shape_errors = bounded_snapshot_shape(snapshot)
        if shape_errors:
            raise HTTPException(status_code=500, detail=",".join(shape_errors))
        return {"schema_version": SCHEMA_VERSION, **snapshot}

    @app.post("/v1/personas/{persona_id}/sessions/{session_id}/runner/start")
    async def runner_start(persona_id: str, session_id: str, body: RunnerControlBody, request: Request) -> JSONResponse:
        _check_mutation(request, gw)
        payload_errors = runner_control_payload_is_bounded({"command": body.command, "reason": body.reason})
        if payload_errors or body.command != "start":
            raise HTTPException(status_code=400, detail="invalid_runner_command")
        handle = gw.get_or_create_session(persona_id, session_id)
        runner = gw.ensure_runner(handle)
        status = runner.start()
        return JSONResponse(
            status_code=202,
            content={
                "command": "start",
                "runner": status.to_dict(),
                "correlation_id": body.correlation_id,
            },
        )

    @app.post("/v1/personas/{persona_id}/sessions/{session_id}/runner/stop")
    async def runner_stop(persona_id: str, session_id: str, body: RunnerControlBody, request: Request) -> JSONResponse:
        _check_mutation(request, gw)
        if body.command != "stop":
            raise HTTPException(status_code=400, detail="invalid_runner_command")
        handle = gw.get_or_create_session(persona_id, session_id)
        runner = gw.ensure_runner(handle)
        status = runner.stop()
        return JSONResponse(
            status_code=202,
            content={
                "command": "stop",
                "runner": status.to_dict(),
                "correlation_id": body.correlation_id,
            },
        )

    @app.get("/v1/personas/{persona_id}/sessions/{session_id}/runner/status")
    async def runner_status(persona_id: str, session_id: str) -> dict[str, Any]:
        handle = gw.get_or_create_session(persona_id, session_id)
        runner = handle.runner
        status = runner.status() if runner is not None else None
        return {
            "schema_version": SCHEMA_VERSION,
            "persona_id": persona_id,
            "session_id": session_id,
            "runner": status.to_dict() if status is not None else {"running": False, "runner_kind": RUNNER_KIND},
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }

    @app.websocket("/v1/personas/{persona_id}/sessions/{session_id}/stream")
    async def ws_stream(websocket: WebSocket, persona_id: str, session_id: str) -> None:
        await websocket.accept()
        handle = gw.get_or_create_session(persona_id, session_id)
        queue = handle.hub.register_subscriber()
        client_id = f"ws_{uuid.uuid4().hex[:10]}"
        correlation_id = f"ws-connect:{client_id}"
        handle.bridge.append_perception_event(
            "DeliverySurfaceConnectedEvent",
            {"client_id": client_id},
            source="m16_gateway",
            correlation_id=correlation_id,
        )
        _append_gateway_audit(
            handle.bridge,
            typ="GatewayClientConnectedEvent",
            correlation_id=correlation_id,
            client_id=client_id,
        )
        snapshot = handle.bridge.snapshot()
        handle.hub.subscribed_snapshot_message(snapshot=snapshot)

        async def _inbound() -> None:
            while True:
                raw = await websocket.receive_text()
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        handle.hub.build_and_publish(
                            kind="Error",
                            payload={"code": "invalid_json", "message": "payload must be JSON object"},
                        )
                    )
                    continue
                if not isinstance(row, dict):
                    continue
                errors = validate_ws_client_message(row)
                if errors:
                    await websocket.send_json(
                        handle.hub.build_and_publish(
                            kind="Error",
                            payload={"code": "invalid_message", "errors": errors[:8]},
                        )
                    )
                    continue
                kind = str(row.get("kind", "") or "")
                msg_correlation = str(row.get("correlation_id", "") or "")
                payload = dict(row.get("payload") or {})
                if kind == "Subscribe":
                    snapshot = handle.bridge.snapshot()
                    handle.hub.subscribed_snapshot_message(snapshot=snapshot)
                elif kind == "Ping":
                    handle.hub.build_and_publish(kind="RunnerHealth", payload={"pong": True})
                elif kind == "ClientInput":
                    text = str(payload.get("text", "") or "")[:MAX_INPUT_TEXT_CHARS]
                    speaker_name = str(payload.get("speaker_name", "") or "")[:64]
                    event_id = handle.bridge.append_client_input(
                        text=text,
                        correlation_id=msg_correlation,
                        source="m16_ws",
                        speaker_name=speaker_name,
                    )
                    _append_gateway_audit(
                        handle.bridge,
                        typ="GatewayInputAcceptedEvent",
                        correlation_id=msg_correlation,
                        event_id=event_id,
                    )
                    gw.ensure_runner(handle).nudge()
                elif kind == "DeliverySurfaceReady":
                    handle.bridge.append_perception_event(
                        "DeliverySurfaceReadyEvent",
                        {"client_id": client_id},
                        source="m16_ws",
                        correlation_id=msg_correlation,
                    )
                    handle.hub.mark_delivery_surface_ready(correlation_id=msg_correlation)
                    handle.bridge.append_perception_event(
                        "OutboxDeliverySurfaceAvailableEvent",
                        {"client_id": client_id},
                        source="m16_gateway",
                        correlation_id=msg_correlation,
                    )
                    gw.ensure_runner(handle).nudge()
                elif kind == "Unsubscribe":
                    return

        async def _outbound() -> None:
            while True:
                message = await queue.get()
                await websocket.send_json(message)

        inbound_task = asyncio.create_task(_inbound())
        outbound_task = asyncio.create_task(_outbound())
        try:
            await inbound_task
        except WebSocketDisconnect:
            pass
        finally:
            inbound_task.cancel()
            outbound_task.cancel()
            handle.hub.unregister_subscriber(queue)
            handle.hub.clear_delivery_surface_ready()
            handle.bridge.append_perception_event(
                "DeliverySurfaceDisconnectedEvent",
                {"client_id": client_id},
                source="m16_gateway",
                correlation_id=correlation_id,
            )
            _append_gateway_audit(
                handle.bridge,
                typ="GatewayClientDisconnectedEvent",
                correlation_id=correlation_id,
                client_id=client_id,
            )

    web_dist = Path(__file__).resolve().parents[3] / "ui" / "web" / "dist"
    if web_dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="m16_web_static")

    return app


def _serve_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M16 consciousness gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--persona", default="")
    parser.add_argument("--session", default="")
    parser.add_argument("--dev-token", default="")
    args = parser.parse_args(argv)

    llm_status = llm_configuration_status_with_source(_resolve_session_llm(M16Gateway()))
    if not bool(llm_status.get("available")):
        logging.getLogger(__name__).warning(
            "M16 gateway LLM unavailable (%s). Configure %s or set OPENROUTER_API_KEY.",
            llm_status.get("reason", "llm_unavailable"),
            openrouter_secrets_path(),
        )
    else:
        logging.getLogger(__name__).info(
            "M16 gateway LLM ready (%s)",
            llm_status.get("config_source") or "configured",
        )

    import uvicorn

    gateway = M16Gateway(dev_token=args.dev_token)
    if args.persona and args.session:
        gateway.get_or_create_session(args.persona, args.session)
    app = create_app(gateway)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(_serve_main())
