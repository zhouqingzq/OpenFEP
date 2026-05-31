from __future__ import annotations

import asyncio
import threading

from segmentum.dialogue.runtime.m16_turn_progress import build_turn_progress_payload
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub


def test_publish_from_runner_thread_reaches_async_subscriber() -> None:
    asyncio.run(_publish_from_runner_thread_reaches_async_subscriber())


async def _publish_from_runner_thread_reaches_async_subscriber() -> None:
    hub = M16WsHub(persona_id="胡桃", session_id="demo")
    queue = hub.register_subscriber()

    async def consume() -> dict:
        return await asyncio.wait_for(queue.get(), timeout=2.0)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    def publish_from_runner() -> None:
        hub.build_and_publish(
            kind="AuditEvent",
            payload=build_turn_progress_payload(turn_index=1, stage="claimed", percent=1),
        )

    thread = threading.Thread(target=publish_from_runner)
    thread.start()
    thread.join(timeout=2.0)

    message = await consumer
    assert message["kind"] == "AuditEvent"
    assert message["payload"]["audit_type"] == "turn_progress"
    assert message["payload"]["percent"] == 1
