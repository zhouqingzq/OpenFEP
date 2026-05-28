import type { ChatTailRow, WsServerMessage } from "@segments/consciousness-client";

export type MessageRole = "user" | "assistant" | "proactive";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  turnIndex?: number;
  at?: number;
  serverMessageId?: string;
}

export interface IngestWsResult {
  message: ChatMessage | null;
  shouldAck: boolean;
  deliveryId: string;
  duplicate: boolean;
}

function roleFromTailRow(row: ChatTailRow): MessageRole {
  const event = String(row.event ?? "");
  if (event === "proactive_turn") {
    return "proactive";
  }
  if (event === "assistant_message") {
    return "assistant";
  }
  return "user";
}

function syntheticId(prefix: string, parts: Array<string | number | undefined>): string {
  return `${prefix}:${parts.filter((p) => p !== undefined && p !== "").join(":")}`;
}

export class ChatModel {
  private messages: ChatMessage[] = [];
  private renderedDeliveryIds = new Set<string>();
  private ackedDeliveryIds = new Set<string>();

  listMessages(): readonly ChatMessage[] {
    return this.messages;
  }

  renderedIds(): ReadonlySet<string> {
    return this.renderedDeliveryIds;
  }

  ackedIds(): ReadonlySet<string> {
    return this.ackedDeliveryIds;
  }

  clear(): void {
    this.messages = [];
    this.renderedDeliveryIds.clear();
    this.ackedDeliveryIds.clear();
  }

  ingestSnapshot(chatTail: ChatTailRow[] | undefined): ChatMessage[] {
    const added: ChatMessage[] = [];
    for (const row of chatTail ?? []) {
      const text = String(row.text ?? "").trim();
      if (!text) {
        continue;
      }
      const role = roleFromTailRow(row);
      if (this.hasSimilarMessage(role, text, row.turn_index)) {
        continue;
      }
      const id = syntheticId("snapshot", [role, row.turn_index, row.at, text.slice(0, 32)]);
      if (this.renderedDeliveryIds.has(id)) {
        continue;
      }
      const message: ChatMessage = {
        id,
        role,
        text,
        turnIndex: row.turn_index,
        at: row.at,
      };
      this.messages.push(message);
      this.renderedDeliveryIds.add(id);
      added.push(message);
    }
    return added;
  }

  addPendingUser(text: string, correlationId: string): ChatMessage {
    const message: ChatMessage = {
      id: syntheticId("user", [correlationId]),
      role: "user",
      text,
      at: Math.floor(Date.now() / 1000),
    };
    this.messages.push(message);
    return message;
  }

  ingestWsMessage(message: WsServerMessage): IngestWsResult {
    if (message.kind === "AssistantMessageCommitted" || message.kind === "ProactiveMessageCommitted") {
      return this.ingestActuation(message);
    }
    return { message: null, shouldAck: false, deliveryId: "", duplicate: false };
  }

  private ingestActuation(message: WsServerMessage): IngestWsResult {
    const payload = message.payload ?? {};
    const text = String(payload.text ?? "").trim();
    const deliveryId = String(payload.delivery_id ?? "").trim();
    const displayId = deliveryId || `msg:${message.message_id}`;
    if (!text) {
      return { message: null, shouldAck: false, deliveryId, duplicate: false };
    }
    if (this.renderedDeliveryIds.has(displayId)) {
      return { message: null, shouldAck: false, deliveryId, duplicate: true };
    }
    const role: MessageRole = message.kind === "ProactiveMessageCommitted" ? "proactive" : "assistant";
    const chatMessage: ChatMessage = {
      id: displayId,
      role,
      text,
      turnIndex: typeof payload.turn_index === "number" ? payload.turn_index : undefined,
      at: message.at,
      serverMessageId: message.message_id,
    };
    this.messages.push(chatMessage);
    this.renderedDeliveryIds.add(displayId);
    const shouldAck = Boolean(deliveryId) && !this.ackedDeliveryIds.has(deliveryId);
    return { message: chatMessage, shouldAck, deliveryId, duplicate: false };
  }

  markAcked(deliveryId: string): void {
    if (!deliveryId) {
      return;
    }
    this.ackedDeliveryIds.add(deliveryId);
  }

  shouldAckDelivery(deliveryId: string): boolean {
    return Boolean(deliveryId) && this.renderedDeliveryIds.has(deliveryId) && !this.ackedDeliveryIds.has(deliveryId);
  }

  private hasSimilarMessage(role: MessageRole, text: string, turnIndex?: number): boolean {
    return this.messages.some(
      (message) =>
        message.role === role &&
        message.text === text &&
        (turnIndex === undefined || message.turnIndex === turnIndex),
    );
  }
}
