import { describe, expect, it } from "vitest";

import { ChatModel } from "../src/chat/model.js";
import { SCHEMA_VERSION, type WsServerMessage } from "@segments/consciousness-client";

function actuation(kind: "AssistantMessageCommitted" | "ProactiveMessageCommitted", deliveryId: string, text: string): WsServerMessage {
  return {
    schema_version: SCHEMA_VERSION,
    message_id: `m16s_${deliveryId}`,
    persona_id: "p",
    session_id: "s",
    at: 1_900_000_000,
    kind,
    payload: { text, delivery_id: deliveryId, turn_index: 2 },
  };
}

describe("ChatModel", () => {
  it("dedupes ws actuation by delivery_id", () => {
    const model = new ChatModel();
    const first = model.ingestWsMessage(actuation("AssistantMessageCommitted", "assistant:evt1", "你好"));
    expect(first.message?.text).toBe("你好");
    expect(first.shouldAck).toBe(true);
    const second = model.ingestWsMessage(actuation("AssistantMessageCommitted", "assistant:evt1", "你好"));
    expect(second.duplicate).toBe(true);
    expect(second.shouldAck).toBe(false);
    expect(model.listMessages()).toHaveLength(1);
  });

  it("resync snapshot skips messages already shown via ws", () => {
    const model = new ChatModel();
    model.ingestWsMessage(actuation("AssistantMessageCommitted", "assistant:evt1", "same reply"));
    const added = model.ingestSnapshot([
      { event: "assistant_message", text: "same reply", turn_index: 2 },
    ]);
    expect(added).toHaveLength(0);
    expect(model.listMessages()).toHaveLength(1);
  });

  it("tracks user pending messages locally", () => {
    const model = new ChatModel();
    const user = model.addPendingUser("hello", "corr_1");
    expect(user.role).toBe("user");
    expect(model.listMessages()).toHaveLength(1);
  });
});
