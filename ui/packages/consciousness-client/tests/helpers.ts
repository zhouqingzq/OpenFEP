import { SCHEMA_VERSION, WS_CLIENT_KINDS, type WsServerMessage } from "../src/types.js";
import { createConsciousnessClient, validateInboundServerMessage, validateOutboundClientMessage } from "../src/index.js";

export class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  sent: string[] = [];

  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(code = 1000, reason = ""): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }

  simulateMessage(message: WsServerMessage | Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  static reset(): void {
    MockWebSocket.instances = [];
  }
}

export function baseMessage(kind: string, payload: Record<string, unknown> = {}): WsServerMessage {
  return {
    schema_version: SCHEMA_VERSION,
    message_id: `m16s_${kind}`,
    persona_id: "p",
    session_id: "s",
    at: 1_900_000_000,
    kind: kind as WsServerMessage["kind"],
    payload,
  };
}

export function createTestClient(fetchImpl: typeof fetch) {
  MockWebSocket.reset();
  return createConsciousnessClient({
    baseUrl: "http://127.0.0.1:8765",
    personaId: "p",
    sessionId: "s",
    fetchImpl,
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });
}

export function mockFetch(handlers: Record<string, () => Response>): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    const key = `${method} ${url}`;
    for (const [pattern, handler] of Object.entries(handlers)) {
      if (key.startsWith(pattern) || url.includes(pattern)) {
        return handler();
      }
    }
    return new Response(JSON.stringify({ detail: "not_found" }), { status: 404 });
  }) as typeof fetch;
}
