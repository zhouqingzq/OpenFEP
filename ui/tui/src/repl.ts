import type {
  ConsciousnessClient,
  ConsciousnessStream,
  SnapshotResponse,
  WsServerMessage,
} from "@segments/consciousness-client";
import {
  createConsciousnessClient,
  isAssistantMessage,
  isProactiveMessage,
  isSuppressionEvent,
} from "@segments/consciousness-client";

import { commandHelpText, parseReplLine } from "./commands.js";
import { AuditTail } from "./render/audit_tail.js";
import { formatStatusBar } from "./render/status_bar.js";
import {
  formatTranscriptLine,
  transcriptRoleFromEvent,
  type TranscriptLine,
} from "./render/transcript.js";

export interface ReplOptions {
  personaId: string;
  sessionId: string;
  gatewayUrl: string;
  authToken?: string;
  color?: boolean;
  input?: AsyncIterable<string>;
  output?: Pick<Console, "log" | "error">;
}

export class ConsciousnessRepl {
  readonly client: ConsciousnessClient;
  readonly auditTail = new AuditTail();
  private stream: ConsciousnessStream | null = null;
  private wsOpen = false;
  private subscribed = false;
  private reconnecting = false;
  private runnerPhase = "unknown";
  private latestSnapshot: SnapshotResponse | null = null;
  private renderedMessageKeys = new Set<string>();
  private renderedDeliveryIds = new Set<string>();
  private ackedDeliveryIds = new Set<string>();
  private readonly color: boolean;
  private readonly out: Pick<Console, "log" | "error">;

  constructor(private readonly options: ReplOptions) {
    this.color = options.color !== false;
    this.out = options.output ?? console;
    this.client = createConsciousnessClient({
      baseUrl: options.gatewayUrl,
      personaId: options.personaId,
      sessionId: options.sessionId,
      authToken: options.authToken,
      validationMode: "strict",
    });
  }

  async run(lines: AsyncIterable<string>): Promise<void> {
    await this.bootstrap();
    this.printStatusLine();
    this.out.log(commandHelpText());
    for await (const line of lines) {
      const parsed = parseReplLine(line);
      if (parsed.kind === "empty") {
        continue;
      }
      if (parsed.kind === "command") {
        const shouldContinue = await this.handleCommand(parsed.command ?? "help");
        if (!shouldContinue) {
          break;
        }
        continue;
      }
      if (parsed.text) {
        await this.sendUserMessage(parsed.text);
      }
    }
    await this.client.disconnectStream();
  }

  private async bootstrap(): Promise<void> {
    try {
      await this.client.health();
    } catch (error) {
      this.out.error(`gateway offline: ${String(error)} (REST-only /status still available)`);
    }
    await this.refreshRunnerStatus();
    try {
      await this.resyncSnapshot();
    } catch (error) {
      this.out.error(`snapshot unavailable: ${String(error)}`);
    }
    await this.connectStream();
  }

  private async connectStream(): Promise<void> {
    this.reconnecting = false;
    this.wsOpen = false;
    this.subscribed = false;
    const stream = this.client.connectStream({ autoReconnect: true, maxBackoffMs: 30_000 });
    this.stream = stream;
    stream.on("open", () => {
      this.wsOpen = true;
      this.printStatusLine();
    });
    stream.on("close", () => {
      this.wsOpen = false;
      this.subscribed = false;
      this.reconnecting = true;
      this.printStatusLine();
    });
    stream.on("subscribed", () => {
      this.subscribed = true;
      this.reconnecting = false;
      this.printStatusLine();
    });
    stream.on("resync", (snapshot) => {
      this.reconnecting = false;
      this.applySnapshot(snapshot, { fromResync: true });
      this.printStatusLine();
    });
    stream.on("sessionSnapshot", (msg) => {
      const payload = msg.payload as unknown as SnapshotResponse;
      this.applySnapshot({ ...payload, schema_version: payload.schema_version ?? "m16.0" }, { fromResync: false });
    });
    stream.on("assistantMessage", (msg) => void this.handleActuation(msg));
    stream.on("proactiveMessage", (msg) => void this.handleActuation(msg));
    stream.on("suppression", (msg) => {
      if (isSuppressionEvent(msg)) {
        const line = this.auditTail.push(msg);
        if (line) {
          this.printLine(line);
        }
      }
    });
    stream.on("runnerHealth", (msg) => {
      const line = this.auditTail.push(msg);
      if (line) {
        this.printLine(line);
      }
    });
    stream.on("auditEvent", (msg) => {
      const line = this.auditTail.push(msg);
      if (line) {
        this.printLine(line);
      }
    });
    await stream.connect();
  }

  private applySnapshot(snapshot: SnapshotResponse, options: { fromResync: boolean }): void {
    this.latestSnapshot = snapshot;
    for (const row of snapshot.chat_tail ?? []) {
      const text = String(row.text ?? "").trim();
      if (!text) {
        continue;
      }
      const role = transcriptRoleFromEvent(String(row.event ?? ""));
      const key = `snapshot:${role}:${row.turn_index ?? ""}:${row.at ?? ""}:${text.slice(0, 32)}`;
      if (this.renderedMessageKeys.has(key)) {
        continue;
      }
      this.renderedMessageKeys.add(key);
      const line: TranscriptLine = { role, text, at: row.at, meta: row.turn_index ? `turn=${row.turn_index}` : undefined };
      this.printLine(line);
    }
    if (options.fromResync) {
      this.printStatusLine();
    }
  }

  private async handleActuation(message: WsServerMessage): Promise<void> {
    if (!isAssistantMessage(message) && !isProactiveMessage(message)) {
      return;
    }
    const payload = message.payload ?? {};
    const text = String(payload.text ?? "").trim();
    const deliveryId = String(payload.delivery_id ?? "").trim();
    const displayId = deliveryId || `msg:${message.message_id}`;
    if (!text || this.renderedDeliveryIds.has(displayId)) {
      return;
    }
    this.renderedDeliveryIds.add(displayId);
    this.renderedMessageKeys.add(displayId);
    const role = message.kind === "ProactiveMessageCommitted" ? "proactive" : "assistant";
    this.printLine({ role, text, at: message.at });
    if (deliveryId && !this.ackedDeliveryIds.has(deliveryId) && this.stream) {
      await this.stream.sendDeliveryAck(deliveryId);
      this.ackedDeliveryIds.add(deliveryId);
    }
  }

  private async sendUserMessage(text: string): Promise<void> {
    const key = `local:user:${text}:${Math.floor(Date.now() / 1000)}`;
    this.renderedMessageKeys.add(key);
    this.printLine({ role: "user", text, at: Math.floor(Date.now() / 1000) });
    await this.client.sendUserInput(text);
  }

  private async handleCommand(command: string): Promise<boolean> {
    switch (command) {
      case "status":
        await this.refreshRunnerStatus();
        this.printStatusLine();
        return true;
      case "snapshot":
        await this.resyncSnapshot();
        this.out.log(JSON.stringify(this.latestSnapshot, null, 2));
        return true;
      case "debug":
        await this.buildDebugBlock();
        return true;
      case "quit":
        return false;
      case "help":
      default:
        this.out.log(commandHelpText());
        return true;
    }
  }

  private async resyncSnapshot(): Promise<void> {
    this.latestSnapshot = await this.client.getSnapshot();
  }

  private async refreshRunnerStatus(): Promise<void> {
    try {
      const status = await this.client.getRunnerStatus();
      const runner = status.runner;
      if (runner.last_error) {
        this.runnerPhase = `error:${runner.last_error}`;
      } else {
        this.runnerPhase = runner.running ? "running" : "stopped";
      }
    } catch {
      this.runnerPhase = "unavailable";
    }
  }

  private async buildDebugBlock(): Promise<void> {
    if (!this.latestSnapshot) {
      await this.resyncSnapshot();
    }
    const block = {
      persona_id: this.options.personaId,
      session_id: this.options.sessionId,
      gateway_url: this.options.gatewayUrl,
      runner_phase: this.runnerPhase,
      snapshot: this.latestSnapshot,
      recent_audits: this.auditTail.list().slice(-12),
    };
    this.out.log(JSON.stringify(block, null, 2));
  }

  private printLine(line: TranscriptLine): void {
    this.out.log(formatTranscriptLine(line, { color: this.color }));
  }

  private printStatusLine(): void {
    this.out.log(
      formatStatusBar({
        personaId: this.options.personaId,
        sessionId: this.options.sessionId,
        gatewayUrl: this.options.gatewayUrl,
        wsOpen: this.wsOpen,
        subscribed: this.subscribed,
        reconnecting: this.reconnecting,
        runnerPhase: this.runnerPhase,
      }),
    );
  }
}

export async function runRepl(options: ReplOptions): Promise<void> {
  const repl = new ConsciousnessRepl(options);
  const input = options.input ?? readStdinLines();
  await repl.run(input);
}

async function* readStdinLines(): AsyncIterable<string> {
  const { createInterface } = await import("node:readline");
  const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  try {
    for await (const line of rl) {
      yield line;
    }
  } finally {
    rl.close();
  }
}
