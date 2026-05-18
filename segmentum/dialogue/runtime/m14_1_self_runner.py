"""M14.1 background self-continuity runner (inline daemon + CLI)."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from segmentum.dialogue.runtime.m14_1_background_continuity import (
    INLINE_RUNNER_IDLE_DEATH_SECONDS,
    M14_1_ENGINEERING_PROXY_LABEL,
    MIN_TICK_SECONDS,
    read_runner_lock,
    release_runner_lock,
    try_acquire_runner_lock,
)


class BackgroundSelfRunner:
    """Schedules MVP background idle ticks; does not generate user-visible text."""

    def __init__(
        self,
        runtime: Any,
        *,
        session_root: Path,
        runner_kind: str = "inline",
        tick_interval_seconds: int = 90,
    ) -> None:
        self._runtime = runtime
        self._session_root = Path(session_root)
        self._runner_kind = runner_kind
        self._tick_interval = max(MIN_TICK_SECONDS, int(tick_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock_info: Any = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        now = int(time.time())
        ok, info = try_acquire_runner_lock(self._session_root, runner_kind=self._runner_kind, now=now)
        if not ok:
            self._runtime.append_background_audit(
                {
                    "type": "BackgroundRunnerCollisionEvent",
                    "at": now,
                    "existing_pid": getattr(info, "pid", 0),
                    "runner_kind": self._runner_kind,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
            return
        self._lock_info = info
        self._stop.clear()
        self._runtime.append_background_audit(
            {
                "type": "BackgroundRunnerStartEvent",
                "at": now,
                "runner_kind": self._runner_kind,
                "pid": info.pid if info else 0,
                "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
            }
        )
        self._thread = threading.Thread(target=self._loop, name="m14_1_background_self", daemon=True)
        self._thread.start()

    def stop(self, *, drain_wait_seconds: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, drain_wait_seconds))
            self._thread = None
        now = int(time.time())
        self._runtime.append_background_audit(
            {
                "type": "BackgroundRunnerStopEvent",
                "at": now,
                "runner_kind": self._runner_kind,
                "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
            }
        )
        release_runner_lock(self._session_root)

    def status(self) -> dict[str, Any]:
        lock = read_runner_lock(self._session_root)
        return {
            "runner_kind": self._runner_kind,
            "running": self._thread is not None and self._thread.is_alive(),
            "tick_interval_seconds": self._tick_interval,
            "lock_pid": lock.pid if lock else 0,
            "lock_host": lock.host if lock else "",
        }

    def record_streamlit_ping(self) -> None:
        self._runtime.record_streamlit_ping()

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._runtime.run_background_self_tick(runner_kind=self._runner_kind)
            except Exception as exc:  # pragma: no cover - runner resilience
                self._runtime.append_background_audit(
                    {
                        "type": "BackgroundIdleTickEvent",
                        "at": int(time.time()),
                        "skip_reason": "tick_error",
                        "detail": str(exc)[:240],
                        "runner_kind": self._runner_kind,
                        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                    }
                )
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, self._tick_interval - elapsed)
            if self._stop.wait(sleep_for):
                break
            if self._runner_kind == "inline":
                if self._runtime.inline_runner_should_stop(
                    idle_death_seconds=INLINE_RUNNER_IDLE_DEATH_SECONDS
                ):
                    break


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M14.1 background self-continuity CLI runner")
    parser.add_argument("--session-root", required=True, help="MVP session directory")
    parser.add_argument("--persona", default="", help="Persona id (diagnostics only)")
    parser.add_argument("--tick-interval", type=int, default=90)
    args = parser.parse_args(argv)
    session_root = Path(args.session_root).resolve()
    from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore, OpenRouterJSONClient

    store = MVPStateStore(session_root)
    llm = OpenRouterJSONClient.available() and OpenRouterJSONClient() or None
    runtime = MVPDialogueRuntime(store=store, llm=llm)
    runner = BackgroundSelfRunner(
        runtime,
        session_root=session_root,
        runner_kind="cli",
        tick_interval_seconds=args.tick_interval,
    )

    def _handle_signal(*_a: object) -> None:
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    runner.start()
    if not runner.status().get("running"):
        return 1
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(_cli_main())
