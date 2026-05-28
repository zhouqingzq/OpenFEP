import {
  createConsciousnessClient,
  isProactiveMessage,
  isAssistantMessage,
  isSuppressionEvent,
  type ConsciousnessClient,
  type ConsciousnessStream,
  type SnapshotResponse,
  type WsServerMessage,
} from "@segments/consciousness-client";

import { ChatModel } from "./chat/model.js";
import { ChatView } from "./chat/view.js";
import { badgeLabel, connectionBadge } from "./connection/status.js";
import { DiagnosticsPanel } from "./diagnostics/panel.js";

const STORAGE_PERSONA = "m16_web_persona_id";
const STORAGE_SESSION = "m16_web_session_id";

export interface SessionConfig {
  personaId: string;
  sessionId: string;
  baseUrl: string;
  authToken?: string;
}

export function loadSessionConfig(): SessionConfig {
  const params = new URLSearchParams(window.location.search);
  const personaId = params.get("persona") || localStorage.getItem(STORAGE_PERSONA) || "胡桃";
  const sessionId = params.get("session") || localStorage.getItem(STORAGE_SESSION) || "web_demo";
  localStorage.setItem(STORAGE_PERSONA, personaId);
  localStorage.setItem(STORAGE_SESSION, sessionId);
  return {
    personaId,
    sessionId,
    baseUrl: window.location.origin,
    authToken: import.meta.env.VITE_M16_AUTH_TOKEN || undefined,
  };
}

export class ConsciousnessWebApp {
  readonly config: SessionConfig;
  readonly model = new ChatModel();
  readonly client: ConsciousnessClient;

  private stream: ConsciousnessStream | null = null;
  private reconnecting = false;
  private wsOpen = false;
  private subscribed = false;
  private snapshotTimer: number | null = null;

  private badgeEl!: HTMLElement;
  private runnerStatusEl!: HTMLElement;
  private chatView!: ChatView;
  private diagnostics!: DiagnosticsPanel;

  constructor(config: SessionConfig) {
    this.config = config;
    this.client = createConsciousnessClient({
      baseUrl: config.baseUrl,
      personaId: config.personaId,
      sessionId: config.sessionId,
      authToken: config.authToken,
      validationMode: "strict",
    });
  }

  mount(root: HTMLElement): void {
    root.innerHTML = `
      <div class="layout">
        <aside class="sidebar">
          <h1>Consciousness Chat</h1>
          <div class="session-fields">
            <label>Persona<input id="persona-input" /></label>
            <label>Session<input id="session-input" /></label>
            <button type="button" id="apply-session">应用会话</button>
          </div>
          <div class="connection-badge" id="connection-badge">offline</div>
          <div class="runner-controls">
            <button type="button" id="start-runner">Start runner</button>
            <button type="button" id="stop-runner">Stop runner</button>
            <button type="button" id="refresh-status">Runner status</button>
            <button type="button" id="copy-debug">Copy snapshot JSON</button>
          </div>
          <pre class="runner-status" id="runner-status"></pre>
          <div id="diagnostics-root"></div>
        </aside>
        <main id="chat-root"></main>
      </div>
    `;

    this.badgeEl = root.querySelector("#connection-badge") as HTMLElement;
    this.runnerStatusEl = root.querySelector("#runner-status") as HTMLElement;
    const personaInput = root.querySelector("#persona-input") as HTMLInputElement;
    const sessionInput = root.querySelector("#session-input") as HTMLInputElement;
    personaInput.value = this.config.personaId;
    sessionInput.value = this.config.sessionId;

    this.chatView = new ChatView({
      root: root.querySelector("#chat-root") as HTMLElement,
      onSend: (text) => void this.sendUserMessage(text),
    });
    this.diagnostics = new DiagnosticsPanel(root.querySelector("#diagnostics-root") as HTMLElement);

    root.querySelector("#apply-session")?.addEventListener("click", () => {
      const persona = personaInput.value.trim() || "胡桃";
      const session = sessionInput.value.trim() || "web_demo";
      localStorage.setItem(STORAGE_PERSONA, persona);
      localStorage.setItem(STORAGE_SESSION, session);
      const url = new URL(window.location.href);
      url.searchParams.set("persona", persona);
      url.searchParams.set("session", session);
      window.location.href = url.toString();
    });
    root.querySelector("#start-runner")?.addEventListener("click", () => void this.startRunner());
    root.querySelector("#stop-runner")?.addEventListener("click", () => void this.stopRunner());
    root.querySelector("#refresh-status")?.addEventListener("click", () => void this.refreshRunnerStatus());
    root.querySelector("#copy-debug")?.addEventListener("click", () => void this.copySnapshot());

    void this.bootstrap();
  }

  private updateBadge(): void {
    const badge = connectionBadge({
      wsOpen: this.wsOpen,
      subscribed: this.subscribed,
      reconnecting: this.reconnecting,
    });
    this.badgeEl.textContent = badgeLabel(badge);
    this.badgeEl.dataset.state = badge;
    this.chatView.setComposerEnabled(badge === "live" || badge === "resyncing");
  }

  private async bootstrap(): Promise<void> {
    this.updateBadge();
    try {
      await this.client.health();
    } catch (error) {
      this.runnerStatusEl.textContent = `gateway unreachable: ${String(error)}`;
      this.updateBadge();
    }
    await this.refreshRunnerStatus();
    await this.resyncSnapshot();
    await this.connectStream();
    this.snapshotTimer = window.setInterval(() => void this.resyncSnapshot(), 30_000);
  }

  private async connectStream(): Promise<void> {
    this.reconnecting = false;
    this.wsOpen = false;
    this.subscribed = false;
    this.updateBadge();

    const stream = this.client.connectStream({ autoReconnect: true, maxBackoffMs: 30_000 });
    this.stream = stream;

    stream.on("open", () => {
      this.wsOpen = true;
      this.updateBadge();
    });
    stream.on("close", () => {
      this.wsOpen = false;
      this.subscribed = false;
      this.reconnecting = true;
      this.updateBadge();
    });
    stream.on("subscribed", () => {
      this.subscribed = true;
      this.reconnecting = false;
      this.updateBadge();
    });
    stream.on("resync", (snapshot) => {
      this.reconnecting = false;
      this.applySnapshot(snapshot, { fromResync: true });
      this.updateBadge();
    });
    stream.on("sessionSnapshot", (msg) => {
      const payload = msg.payload as unknown as SnapshotResponse;
      this.applySnapshot({ ...payload, schema_version: payload.schema_version ?? "m16.0" }, { fromResync: false });
    });
    stream.on("assistantMessage", (msg) => {
      if (isAssistantMessage(msg)) {
        void this.handleActuation(msg);
      }
    });
    stream.on("proactiveMessage", (msg) => {
      if (isProactiveMessage(msg)) {
        void this.handleActuation(msg);
      }
    });
    stream.on("suppression", (msg) => {
      if (isSuppressionEvent(msg)) {
        const reason = String(msg.payload?.reason_code ?? "unknown");
        this.chatView.showSuppressionToast(reason);
      }
    });
    stream.on("runnerHealth", (msg) => this.diagnostics.pushAudit(msg));
    stream.on("auditEvent", (msg) => this.diagnostics.pushAudit(msg));

    await stream.connect();
  }

  private applySnapshot(snapshot: SnapshotResponse, options: { fromResync: boolean }): void {
    this.diagnostics.updateSnapshot(snapshot);
    if (options.fromResync) {
      const added = this.model.ingestSnapshot(snapshot.chat_tail);
      for (const message of added) {
        this.chatView.appendMessage(message);
      }
      return;
    }
    if (this.model.listMessages().length === 0) {
      for (const message of this.model.ingestSnapshot(snapshot.chat_tail)) {
        this.chatView.appendMessage(message);
      }
    }
  }

  private async handleActuation(message: WsServerMessage): Promise<void> {
    const result = this.model.ingestWsMessage(message);
    if (result.duplicate || !result.message) {
      return;
    }
    this.chatView.appendMessage(result.message);
    if (result.shouldAck && this.stream) {
      await this.stream.sendDeliveryAck(result.deliveryId);
      this.model.markAcked(result.deliveryId);
    }
  }

  private async sendUserMessage(text: string): Promise<void> {
    const pending = this.model.addPendingUser(text, `local_${Date.now()}`);
    this.chatView.appendMessage(pending);
    await this.client.sendUserInput(text);
  }

  private async resyncSnapshot(): Promise<void> {
    const snapshot = await this.client.getSnapshot();
    this.applySnapshot(snapshot, { fromResync: this.model.listMessages().length > 0 });
    this.diagnostics.updateSnapshot(snapshot);
  }

  private async startRunner(): Promise<void> {
    const result = await this.client.startRunner("web_ui");
    this.runnerStatusEl.textContent = JSON.stringify(result.runner, null, 2);
  }

  private async stopRunner(): Promise<void> {
    const result = await this.client.stopRunner("web_ui");
    this.runnerStatusEl.textContent = JSON.stringify(result.runner, null, 2);
  }

  private async refreshRunnerStatus(): Promise<void> {
    const status = await this.client.getRunnerStatus();
    this.runnerStatusEl.textContent = JSON.stringify(status.runner, null, 2);
  }

  private async copySnapshot(): Promise<void> {
    if (!this.diagnostics.snapshotText()) {
      await this.resyncSnapshot();
    }
    const text = this.diagnostics.snapshotText();
    await navigator.clipboard.writeText(text);
  }
}
