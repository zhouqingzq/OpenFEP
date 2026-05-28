"""M16.1 operator CLI for consciousness gateway and runner."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from segmentum.dialogue.runtime.m14_2_self_loop import default_session_root
from segmentum.dialogue.runtime.m16_api import M16Gateway, create_app
from segmentum.dialogue.runtime.m16_runner import ConsciousnessRunner
from segmentum.dialogue.runtime.m16_runtime_bridge import M16SessionBridge
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore, OpenRouterJSONClient


def _resolve_session_root(args: argparse.Namespace) -> Path:
    if args.session_root:
        return Path(args.session_root).resolve()
    if not args.persona or not args.session:
        raise SystemExit("either --session-root or both --persona and --session are required")
    return default_session_root(args.persona, args.session)


def _build_bridge(args: argparse.Namespace, *, clock: Any = None) -> M16SessionBridge:
    root = _resolve_session_root(args)
    persona_id = args.persona or "default"
    session_id = args.session or root.name
    store = MVPStateStore(root)
    llm = OpenRouterJSONClient.available() and OpenRouterJSONClient() or None
    runtime = MVPDialogueRuntime(store=store, llm=llm)
    return M16SessionBridge(
        persona_id=persona_id,
        session_id=session_id,
        session_root=root,
        runtime=runtime,
        clock=clock,
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    gateway = M16Gateway(dev_token=args.dev_token)
    if args.persona and args.session:
        gateway.get_or_create_session(args.persona, args.session)
    app = create_app(gateway)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_runner(args: argparse.Namespace) -> int:
    bridge = _build_bridge(args)
    hub = M16WsHub(persona_id=bridge.persona_id, session_id=bridge.session_id)
    runner = ConsciousnessRunner(bridge=bridge, hub=hub, tick_interval_seconds=args.tick_interval)
    if args.action == "start":
        status = runner.start()
        if not status.running:
            print(f"runner start refused: {status.last_error}", file=sys.stderr)
            return 1
        print(status.to_dict())
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            runner.stop()
        return 0
    if args.action == "stop":
        bridge.append_perception_event(
            "RunnerControlCommandEvent",
            {"command": "stop", "reason": args.reason or "cli_stop"},
            source="m16_cli",
            correlation_id=f"cli-stop:{uuid.uuid4().hex[:8]}",
        )
        print({"requested": "stop"})
        return 0
    status = runner.status()
    print(status.to_dict())
    return 0


def _cmd_session_create(args: argparse.Namespace) -> int:
    persona_id = args.persona
    session_id = args.session_id.strip() or f"sess_{uuid.uuid4().hex[:12]}"
    root = default_session_root(persona_id, session_id)
    MVPStateStore(root)
    print({"persona_id": persona_id, "session_id": session_id, "session_root": str(root)})
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="consciousness", description="M16 consciousness runner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start HTTP/WS gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--persona", default="")
    serve.add_argument("--session", default="")
    serve.add_argument("--dev-token", default="")
    serve.set_defaults(func=_cmd_serve)

    runner = sub.add_parser("runner", help="Start/stop/status standalone runner")
    runner.add_argument("action", choices=["start", "stop", "status"])
    runner.add_argument("--session-root", default="")
    runner.add_argument("--persona", default="")
    runner.add_argument("--session", default="")
    runner.add_argument("--tick-interval", type=int, default=2)
    runner.add_argument("--reason", default="")
    runner.set_defaults(func=_cmd_runner)

    create = sub.add_parser("session", help="Session helpers")
    create_sub = create.add_subparsers(dest="session_action", required=True)
    session_create = create_sub.add_parser("create", help="Create MVP session directory")
    session_create.add_argument("--persona", required=True)
    session_create.add_argument("--session-id", default="")
    session_create.set_defaults(func=_cmd_session_create)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
