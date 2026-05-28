import { describe, expect, it, vi } from "vitest";

import {
  SCHEMA_VERSION,
  validateInboundServerMessage,
  validateOutboundClientMessage,
  WS_CLIENT_KINDS,
} from "../src/index.js";
import { SchemaVersionError, ValidationError } from "../src/errors.js";
import { baseMessage, createTestClient, MockWebSocket, mockFetch } from "./helpers.js";

describe("websocket stream", () => {
  it("test_ws_subscribe_receives_snapshot", async () => {
    const fetchImpl = mockFetch({});
    const client = createTestClient(fetchImpl);
    const stream = client.connectStream({ autoReconnect: false });
    const snapshots: unknown[] = [];
    stream.on("sessionSnapshot", (msg) => snapshots.push(msg.payload));
    await stream.connect();
    const ws = MockWebSocket.instances[0];
    expect(ws).toBeDefined();
    const subscribe = JSON.parse(ws.sent[0] ?? "{}") as { kind: string };
    expect(subscribe.kind).toBe("Subscribe");
    ws.simulateMessage(baseMessage("Subscribed", { schema_version: SCHEMA_VERSION }));
    expect(JSON.parse(ws.sent[1] ?? "{}").kind).toBe("DeliverySurfaceReady");
    ws.simulateMessage(
      baseMessage("SessionSnapshot", {
        chat_tail: [{ event: "assistant_message", text: "hi" }],
      }),
    );
    expect(snapshots.length).toBe(1);
    await stream.disconnect();
  });

  it("test_ws_reconnect_resubscribes_and_resyncs", async () => {
    vi.useFakeTimers();
    const snapshotCalls: number[] = [];
    const fetchImpl = mockFetch({
      "/snapshot": () => {
        snapshotCalls.push(Date.now());
        return new Response(
          JSON.stringify({
            schema_version: SCHEMA_VERSION,
            persona_id: "p",
            session_id: "s",
            chat_tail: [],
          }),
          { status: 200 },
        );
      },
    });
    const client = createTestClient(fetchImpl);
    const stream = client.connectStream({ autoReconnect: true, maxBackoffMs: 1000 });
    const resyncs: unknown[] = [];
    stream.on("resync", (snap) => resyncs.push(snap));
    await stream.connect();
    const first = MockWebSocket.instances[0];
    first.simulateMessage(baseMessage("Subscribed"));
    first.close(1006, "boom");
    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();
    await Promise.resolve();
    const second = MockWebSocket.instances[1];
    expect(second).toBeDefined();
    expect(JSON.parse(second.sent[0] ?? "{}").kind).toBe("Subscribe");
    expect(snapshotCalls.length).toBeGreaterThan(0);
    expect(resyncs.length).toBeGreaterThan(0);
    await stream.disconnect();
    vi.useRealTimers();
  });

  it("test_delivery_ack_serialization", async () => {
    const client = createTestClient(mockFetch({}));
    const stream = client.connectStream({ autoReconnect: false });
    await stream.connect();
    const ws = MockWebSocket.instances[0];
    ws.simulateMessage(baseMessage("Subscribed"));
    await stream.sendDeliveryAck("assistant:evt_1");
    const ack = JSON.parse(ws.sent.at(-1) ?? "{}") as { kind: string; payload: { delivery_id: string } };
    expect(ack.kind).toBe("DeliveryAck");
    expect(ack.payload.delivery_id).toBe("assistant:evt_1");
    validateOutboundClientMessage(ack);
    await stream.disconnect();
  });

  it("test_ws_client_input_serializes_speaker_name", async () => {
    const client = createTestClient(mockFetch({}));
    const stream = client.connectStream({ autoReconnect: false });
    await stream.connect();
    const ws = MockWebSocket.instances[0];
    ws.simulateMessage(baseMessage("Subscribed"));
    await stream.sendClientInput("hello", { speaker_name: "zq" });
    const input = JSON.parse(ws.sent.at(-1) ?? "{}") as {
      kind: string;
      payload: { text: string; speaker_name?: string };
    };
    expect(input.kind).toBe("ClientInput");
    expect(input.payload).toMatchObject({ text: "hello", speaker_name: "zq" });
    validateOutboundClientMessage(input);
    await stream.disconnect();
  });
});

describe("validation", () => {
  it("test_invalid_server_message_is_rejected", () => {
    expect(() =>
      validateInboundServerMessage(
        {
          schema_version: SCHEMA_VERSION,
          message_id: "m16s_bad",
          persona_id: "p",
          session_id: "s",
          at: 1,
          kind: "NotARealKind",
          payload: {},
        },
        "strict",
      ),
    ).toThrow(ValidationError);
  });

  it("test_schema_version_mismatch_surfaces_error", () => {
    const bad = baseMessage("RunnerHealth", {});
    bad.schema_version = "m99.0" as typeof SCHEMA_VERSION;
    expect(() => validateInboundServerMessage(bad, "strict")).toThrow(SchemaVersionError);
    const lenient = validateInboundServerMessage(bad, "lenient");
    expect(lenient.message).toBeNull();
    expect(lenient.warnings).toContain("schema_version_mismatch");
  });

  it("test_forbidden_internal_payload_keys_are_rejected", () => {
    const bad = baseMessage("AssistantMessageCommitted", {
      text: "hi",
      conscious_plan: { task: "secret" },
    });
    expect(() => validateInboundServerMessage(bad, "strict")).toThrow(ValidationError);
    const lenient = validateInboundServerMessage(bad, "lenient");
    expect(lenient.message).toBeNull();
    expect(lenient.warnings.some((w) => w.startsWith("forbidden:"))).toBe(true);
  });

  it("test_client_does_not_serialize_delivery_surface_disconnected", () => {
    expect(WS_CLIENT_KINDS).not.toContain("DeliverySurfaceDisconnected");
    expect(() =>
      validateOutboundClientMessage({
        schema_version: SCHEMA_VERSION,
        message_id: "m16c_bad",
        persona_id: "p",
        session_id: "s",
        at: 1,
        kind: "DeliverySurfaceDisconnected",
        correlation_id: "corr",
        payload: {},
      }),
    ).toThrow(ValidationError);
  });
});
