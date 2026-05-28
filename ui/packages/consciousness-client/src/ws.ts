import {
  buildWsStreamUrl,
  ConsciousnessHttpClient,
  newCorrelationId,
  newMessageId,
  type HttpClientConfig,
} from "./http.js";
import { ReconnectBackoff, sleep } from "./reconnect.js";
import {
  SCHEMA_VERSION,
  type StreamConnectOptions,
  type StreamEventMap,
  type StreamEventName,
  type ValidationMode,
  type ValidationWarning,
  type WsClientMessage,
  type WsServerMessage,
} from "./types.js";
import { validateInboundServerMessage, validateOutboundClientMessage } from "./validate.js";

type StreamListener<K extends StreamEventName> = (...args: StreamEventMap[K]) => void;

const WS_OPEN = 1;

type AnyStreamListener = (...args: never[]) => void;

export class ConsciousnessStream {
  private readonly httpConfig: HttpClientConfig;
  private readonly http: ConsciousnessHttpClient;
  private readonly validationMode: ValidationMode;
  private readonly WebSocketImpl: new (url: string, protocols?: string | string[]) => WebSocket;
  private readonly options: Required<Pick<StreamConnectOptions, "autoReconnect" | "maxBackoffMs">> &
    Pick<StreamConnectOptions, "resumeFromMessageId">;

  private ws: WebSocket | null = null;
  private listeners = new Map<StreamEventName, Set<AnyStreamListener>>();
  private closedByUser = false;
  private subscribed = false;
  private reconnecting = false;
  private backoff = new ReconnectBackoff({ maxMs: 30_000 });

  constructor(
    httpConfig: HttpClientConfig,
    options: StreamConnectOptions = {},
    validationMode: ValidationMode = "strict",
  ) {
    this.httpConfig = httpConfig;
    this.http = new ConsciousnessHttpClient(httpConfig);
    this.validationMode = validationMode;
    this.WebSocketImpl = httpConfig.WebSocketImpl ?? WebSocket;
    this.options = {
      autoReconnect: options.autoReconnect ?? true,
      maxBackoffMs: options.maxBackoffMs ?? 30_000,
      resumeFromMessageId: options.resumeFromMessageId,
    };
    this.backoff = new ReconnectBackoff({ maxMs: this.options.maxBackoffMs });
  }

  on<K extends StreamEventName>(event: K, listener: StreamListener<K>): this {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set<AnyStreamListener>());
    }
    this.listeners.get(event)!.add(listener as AnyStreamListener);
    return this;
  }

  off<K extends StreamEventName>(event: K, listener: StreamListener<K>): this {
    this.listeners.get(event)?.delete(listener as AnyStreamListener);
    return this;
  }

  private emit<K extends StreamEventName>(event: K, ...args: StreamEventMap[K]): void {
    for (const listener of this.listeners.get(event) ?? []) {
      (listener as StreamListener<K>)(...args);
    }
  }

  async connect(): Promise<void> {
    this.closedByUser = false;
    await this.openSocket();
  }

  private async openSocket(): Promise<void> {
    const url = buildWsStreamUrl(this.httpConfig);
    const ws = new this.WebSocketImpl(url);
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => {
        this.emit("open");
        this.sendClientMessage("Subscribe", {
          resume_from_message_id: this.options.resumeFromMessageId ?? "",
        })
          .then(() => resolve())
          .catch(reject);
      };
      ws.onerror = () => {
        reject(new Error("websocket connection failed"));
      };
      ws.onclose = (event) => {
        this.subscribed = false;
        this.emit("close", { code: event.code, reason: event.reason });
        void this.handleDisconnect(event.code, event.reason);
      };
      ws.onmessage = (event) => {
        void this.handleMessage(String(event.data ?? ""));
      };
    });
  }

  private async handleDisconnect(code?: number, reason?: string): Promise<void> {
    if (this.closedByUser || !this.options.autoReconnect) {
      return;
    }
    if (this.reconnecting) {
      return;
    }
    this.reconnecting = true;
    while (!this.closedByUser && this.options.autoReconnect) {
      const delay = this.backoff.nextDelayMs();
      await sleep(delay);
      if (this.closedByUser) {
        break;
      }
      try {
        await this.openSocket();
        const snapshot = await this.http.getSnapshot();
        this.emit("resync", snapshot);
        this.reconnecting = false;
        this.backoff.reset();
        return;
      } catch {
        // keep backing off
      }
    }
    this.reconnecting = false;
    void code;
    void reason;
  }

  private async handleMessage(raw: string): Promise<void> {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw) as unknown;
    } catch {
      if (this.validationMode === "strict") {
        throw new Error("invalid websocket json");
      }
      this.emit("validationWarning", {
        code: "invalid_server_message",
        message: "invalid websocket json",
      } satisfies ValidationWarning);
      return;
    }
    const result = validateInboundServerMessage(parsed, this.validationMode);
    if (result.warnings.length && !result.message) {
      this.emit("validationWarning", {
        code: result.warnings.includes("schema_version_mismatch")
          ? "schema_version_mismatch"
          : "invalid_server_message",
        message: "inbound message rejected",
        details: result.warnings,
      } satisfies ValidationWarning);
      return;
    }
    if (!result.message) {
      return;
    }
    const message = result.message;
    this.emit("raw", message);
    switch (message.kind) {
      case "Subscribed":
        this.subscribed = true;
        this.emit("subscribed", message);
        await this.sendClientMessage("DeliverySurfaceReady", {});
        break;
      case "SessionSnapshot":
        this.emit("sessionSnapshot", message);
        break;
      case "UserMessageAccepted":
        this.emit("userMessageAccepted", message);
        break;
      case "AssistantMessageCommitted":
        this.emit("assistantMessage", message);
        break;
      case "ProactiveMessageCommitted":
        this.emit("proactiveMessage", message);
        break;
      case "AuditEvent":
        this.emit("auditEvent", message);
        break;
      case "RunnerHealth":
        this.emit("runnerHealth", message);
        break;
      case "RunnerSuppression":
        this.emit("suppression", message);
        break;
      case "Error":
        this.emit("error", message);
        break;
      default:
        break;
    }
  }

  private async sendClientMessage(
    kind: WsClientMessage["kind"],
    payload: Record<string, unknown>,
  ): Promise<void> {
    const row: WsClientMessage = {
      schema_version: SCHEMA_VERSION,
      message_id: newMessageId(),
      persona_id: this.httpConfig.personaId,
      session_id: this.httpConfig.sessionId,
      at: Math.floor(Date.now() / 1000),
      kind,
      correlation_id: newCorrelationId("ws"),
      payload,
    };
    validateOutboundClientMessage(row);
    if (!this.ws || this.ws.readyState !== WS_OPEN) {
      throw new Error("websocket is not open");
    }
    this.ws.send(JSON.stringify(row));
  }

  async sendDeliveryAck(deliveryId: string): Promise<void> {
    await this.sendClientMessage("DeliveryAck", { delivery_id: deliveryId.slice(0, 120) });
  }

  async sendClientInput(text: string): Promise<void> {
    await this.sendClientMessage("ClientInput", { text: text.slice(0, 8000) });
  }

  async disconnect(): Promise<void> {
    this.closedByUser = true;
    this.options.autoReconnect = false;
    if (this.ws && this.ws.readyState === WS_OPEN) {
      try {
        await this.sendClientMessage("Unsubscribe", {});
      } catch {
        // socket may already be closing
      }
      this.ws.close();
    }
    this.ws = null;
    this.subscribed = false;
  }

  get isSubscribed(): boolean {
    return this.subscribed;
  }
}
