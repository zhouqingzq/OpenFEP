import { describe, expect, it } from "vitest";

import { ChatModel } from "../src/chat/model.js";
import { SCHEMA_VERSION, type WsServerMessage } from "@segments/consciousness-client";

describe("delivery ack discipline", () => {
  it("should ack only once per delivery_id", () => {
    const model = new ChatModel();
    const message: WsServerMessage = {
      schema_version: SCHEMA_VERSION,
      message_id: "m16s_a1",
      persona_id: "p",
      session_id: "s",
      at: 1,
      kind: "ProactiveMessageCommitted",
      payload: { text: "follow up", delivery_id: "proactive:prop1", proposal_id: "prop1" },
    };
    const first = model.ingestWsMessage(message);
    expect(first.shouldAck).toBe(true);
    expect(model.shouldAckDelivery("proactive:prop1")).toBe(true);
    model.markAcked("proactive:prop1");
    expect(model.shouldAckDelivery("proactive:prop1")).toBe(false);
    const replay = model.ingestWsMessage(message);
    expect(replay.duplicate).toBe(true);
    expect(replay.shouldAck).toBe(false);
  });

  it("does not ack when delivery_id missing", () => {
    const model = new ChatModel();
    const result = model.ingestWsMessage({
      schema_version: SCHEMA_VERSION,
      message_id: "m16s_x",
      persona_id: "p",
      session_id: "s",
      at: 1,
      kind: "AssistantMessageCommitted",
      payload: { text: "hi" },
    });
    expect(result.shouldAck).toBe(false);
  });
});
