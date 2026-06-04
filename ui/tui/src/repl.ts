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
import type { Interface } from "node:readline";

import { commandHelpText, parseReplLine } from "./commands.js";
import { AuditTail, DELIVERY_SURFACE_REFRESH_SECONDS } from "./render/audit_tail.js";
import { formatReplyProgressLine } from "./render/reply_progress.js";
import { formatStatusBar } from "./render/status_bar.js";
import { ThinkingIndicator } from "./render/thinking_indicator.js";
import {
  formatTranscriptLine,
  resolveTranscriptLabelWidth,
  transcriptRoleFromEvent,
  type TranscriptLabels,
  type TranscriptLine,
} from "./render/transcript.js";
import { createReplOutput, clearReadlineSubmittedEcho, clearReplInlineStatus, emitReplInlineStatus, emitReplMessage, withTimeout, waitForAbortSignal, type ReplOutput } from "./repl_io.js";
import { createStdinLineSource, REPL_PROMPT, type StdinLineSource } from "./stdin_lines.js";

const INPUT_TIMEOUT_MS = 15_000;
const RUNNER_TIMEOUT_MS = 20_000;
const SUBSCRIBE_TIMEOUT_MS = 15_000;
const THINKING_TIMEOUT_MS = 300_000;

export interface ReplOptions {
  personaId: string;
  sessionId: string;
  gatewayUrl: string;
  userName?: string;
  authToken?: string;
  color?: boolean;
  input?: AsyncIterable<string>;
  readline?: Interface | null;
  prompt?: string;
  output?: Pick<Console, "log" | "error">;
  autoStartRunner?: boolean;
  verboseAudit?: boolean;
}

export class ConsciousnessRepl {
  readonly client: ConsciousnessClient;
  readonly auditTail: AuditTail;
  private stream: ConsciousnessStream | null = null;
  private wsOpen = false;
  private subscribed = false;
  private reconnecting = false;
  private runnerPhase = "unknown";
  private latestSnapshot: SnapshotResponse | null = null;
  private renderedMessageKeys = new Set<string>();
  private renderedDeliveryIds = new Set<string>();
  private ackedDeliveryIds = new Set<string>();
  private deliverySurfaceTimer: ReturnType<typeof setInterval> | null = null;
  private readonly color: boolean;
  private readonly out: ReplOutput;
  private readonly autoStartRunner: boolean;
  private readonly prompt: string;
  private readonly userDisplayName: string;
  private rl: Interface | null;
  private subscribeAbort: AbortController | null = null;
  private readonly thinking: ThinkingIndicator;
  private awaitingAssistantReply = false;
  private replyProgressPercent = 0;

  constructor(private readonly options: ReplOptions) {
    this.color = options.color !== false;
    this.out = createReplOutput(options.output);
    this.autoStartRunner = options.autoStartRunner !== false;
    this.prompt = options.prompt ?? REPL_PROMPT;
    this.userDisplayName = (options.userName ?? "你").trim() || "你";
    this.rl = options.readline ?? null;
    this.auditTail = new AuditTail({ verbose: options.verboseAudit === true });
    this.client = createConsciousnessClient({
      baseUrl: options.gatewayUrl,
      personaId: options.personaId,
      sessionId: options.sessionId,
      authToken: options.authToken,
      validationMode: "strict",
    });
    this.thinking = new ThinkingIndicator(
      {
        render: (line) => this.renderThinkingLine(line),
        clear: () => this.clearThinkingLine(),
      },
      500,
      THINKING_TIMEOUT_MS,
      () => {
        this.awaitingAssistantReply = false;
        this.emitError("assistant reply timed out (still processing on gateway?)");
      },
    );
  }

  async run(lines: AsyncIterable<string>): Promise<void> {
    try {
      await this.bootstrap();
      this.emitLine(commandHelpText());
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
          this.queueUserMessage(parsed.text);
        }
      }
    } finally {
      this.thinking.stop();
      this.clearDeliverySurfaceTimer();
      await this.client.disconnectStream();
    }
  }

  private async bootstrap(): Promise<void> {
    try {
      await withTimeout(this.client.health(), INPUT_TIMEOUT_MS, "health check");
    } catch (error) {
      this.emitError(`gateway offline: ${String(error)} (REST-only /status still available)`);
    }
    await this.refreshRunnerStatus();
    try {
      await withTimeout(this.resyncSnapshot(), INPUT_TIMEOUT_MS, "snapshot");
      this.warnIfLlmUnavailable();
      this.warnIfUserUnknown();
    } catch (error) {
      this.emitError(`snapshot unavailable: ${String(error)}`);
    }
    await this.connectStream();
    if (this.autoStartRunner) {
      await this.ensureRunnerReady("startup");
    }
    this.emitStatusLine();
  }

  private async connectStream(): Promise<void> {
    this.reconnecting = false;
    this.wsOpen = false;
    this.subscribed = false;
    this.subscribeAbort = new AbortController();
    const stream = this.client.connectStream({ autoReconnect: true, maxBackoffMs: 30_000 });
    this.stream = stream;

    stream.on("open", () => {
      this.wsOpen = true;
    });
    stream.on("close", () => {
      this.wsOpen = false;
      this.subscribed = false;
      this.reconnecting = true;
      this.clearDeliverySurfaceTimer();
    });
    stream.on("subscribed", () => {
      this.subscribed = true;
      this.reconnecting = false;
      this.subscribeAbort?.abort();
      void this.refreshRunnerStatus();
      void this.refreshDeliverySurface("subscribed");
      this.startDeliverySurfaceTimer();
    });
    stream.on("resync", (snapshot) => {
      this.reconnecting = false;
      this.applySnapshot(snapshot, { fromResync: true });
      void this.refreshDeliverySurface("resync");
    });
    stream.on("sessionSnapshot", (msg) => {
      const payload = msg.payload as unknown as SnapshotResponse;
      this.applySnapshot({ ...payload, schema_version: payload.schema_version ?? "m16.0" }, { fromResync: false });
    });
    stream.on("turnCompleted", (msg) => {
      const payload = (msg.payload ?? {}) as Record<string, unknown>;
      if (payload.visible_reply_emitted === false) {
        this.stopThinkingIndicator();
      }
    });
    stream.on("assistantMessage", (msg) => void this.handleActuation(msg));
    stream.on("proactiveMessage", (msg) => void this.handleActuation(msg));
    stream.on("suppression", (msg) => {
      if (isSuppressionEvent(msg)) {
        const line = this.auditTail.push(msg);
        if (line) {
          this.emitTranscript(line);
        }
      }
    });
    stream.on("runnerHealth", (msg) => {
      const line = this.auditTail.push(msg);
      if (line) {
        this.emitTranscript(line);
      }
    });
    stream.on("auditEvent", (msg) => {
      if (this.handleTurnProgress(msg)) {
        return;
      }
      const line = this.auditTail.push(msg);
      if (line) {
        this.emitTranscript(line);
      }
    });
    stream.on("error", (msg) => {
      this.stopThinkingIndicator();
      const payload = msg.payload ?? {};
      const code = String(payload.code ?? payload.message ?? "error");
      const detail = String(payload.detail ?? payload.message ?? "").trim();
      this.emitError(detail ? `gateway error ${code}: ${detail}` : `gateway error: ${code}`);
    });

    await stream.connect();
    if (this.subscribeAbort) {
      await waitForAbortSignal(this.subscribeAbort.signal, SUBSCRIBE_TIMEOUT_MS, "websocket subscribe");
    }
  }

  private startDeliverySurfaceTimer(): void {
    this.clearDeliverySurfaceTimer();
    this.deliverySurfaceTimer = setInterval(() => {
      void this.refreshDeliverySurface("heartbeat");
    }, DELIVERY_SURFACE_REFRESH_SECONDS * 1000);
  }

  private clearDeliverySurfaceTimer(): void {
    if (this.deliverySurfaceTimer) {
      clearInterval(this.deliverySurfaceTimer);
      this.deliverySurfaceTimer = null;
    }
  }

  private async refreshDeliverySurface(reason: string): Promise<void> {
    if (!this.stream || !this.subscribed) {
      return;
    }
    try {
      await this.stream.signalDeliverySurfaceReady();
    } catch (error) {
      this.emitError(`delivery surface refresh failed (${reason}): ${String(error)}`);
    }
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
      const line: TranscriptLine = {
        role,
        text,
        at: row.at,
        meta: row.turn_index ? `turn=${row.turn_index}` : undefined,
      };
      this.emitTranscript(line);
    }
    if (options.fromResync) {
      return;
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
    if (message.kind === "AssistantMessageCommitted") {
      this.stopThinkingIndicator();
    }
    this.emitTranscript({ role, text, at: message.at });
    if (deliveryId && !this.ackedDeliveryIds.has(deliveryId) && this.stream) {
      await this.stream.sendDeliveryAck(deliveryId);
      this.ackedDeliveryIds.add(deliveryId);
    }
  }

  private queueUserMessage(text: string): void {
    clearReadlineSubmittedEcho(this.rl, this.prompt);
    this.replyProgressPercent = 0;
    const key = `local:user:${text}:${Math.floor(Date.now() / 1000)}`;
    this.renderedMessageKeys.add(key);
    this.emitTranscript({ role: "user", text, at: Math.floor(Date.now() / 1000) });
    this.startThinkingIndicator();
    void this.dispatchUserMessage(text);
  }

  private async dispatchUserMessage(text: string): Promise<void> {
    try {
      await withTimeout(
        this.client.sendUserInput(text, {
          speaker_name: this.userDisplayName === "你" ? undefined : this.userDisplayName,
        }),
        INPUT_TIMEOUT_MS,
        "send input",
      );
      await this.refreshRunnerStatus();
    } catch (error) {
      this.stopThinkingIndicator();
      this.emitError(`send failed: ${String(error)}`);
    }
  }

  private async handleCommand(command: string): Promise<boolean> {
    switch (command) {
      case "status":
        await this.refreshRunnerStatus();
        this.emitStatusLine();
        return true;
      case "start-runner":
        await this.startRunner();
        return true;
      case "snapshot":
        await this.resyncSnapshot();
        this.emitLine(JSON.stringify(this.latestSnapshot, null, 2));
        return true;
      case "debug":
        await this.buildDebugBlock();
        return true;
      case "quit":
        return false;
      case "help":
      default:
        this.emitLine(commandHelpText());
        return true;
    }
  }

  private async resyncSnapshot(): Promise<void> {
    this.latestSnapshot = await this.client.getSnapshot();
  }

  private warnIfLlmUnavailable(): void {
    const llm = (this.latestSnapshot?.runtime_hints as Record<string, unknown> | undefined)?.llm;
    if (!llm || typeof llm !== "object") {
      return;
    }
    const row = llm as Record<string, unknown>;
    if (row.available === true) {
      return;
    }
    const reason = String(row.reason ?? "llm_unavailable");
    const source = String(row.config_source ?? "").trim();
    const hint = source
      ? `Check ${source} and restart m16_api.`
      : "Add secrets/openrouter.json at the repo root and restart m16_api.";
    this.emitError(`LLM unavailable (${reason}). ${hint}`);
  }

  private async refreshRunnerStatus(): Promise<void> {
    try {
      const status = await withTimeout(this.client.getRunnerStatus(), INPUT_TIMEOUT_MS, "runner status");
      const runner = status.runner;
      if (runner.running) {
        this.runnerPhase = "running";
      } else if (runner.last_error) {
        this.runnerPhase = `error:${runner.last_error}`;
      } else {
        this.runnerPhase = "stopped";
      }
    } catch {
      this.runnerPhase = "unavailable";
    }
  }

  private async ensureRunnerReady(context: string): Promise<boolean> {
    await this.refreshRunnerStatus();
    if (this.runnerPhase === "running") {
      return true;
    }
    if (!this.autoStartRunner) {
      return false;
    }
    this.emitLine(`runner stopped; auto-starting (${context})...`);
    return this.startRunner({ quiet: true });
  }

  private async startRunner(options?: { quiet?: boolean }): Promise<boolean> {
    try {
      const result = await withTimeout(this.client.startRunner("tui"), RUNNER_TIMEOUT_MS, "start runner");
      const runner = result.runner;
      if (runner.running) {
        this.runnerPhase = "running";
      } else if (runner.last_error) {
        this.runnerPhase = `error:${runner.last_error}`;
      } else {
        this.runnerPhase = "stopped";
      }
      if (!options?.quiet) {
        this.emitLine(JSON.stringify(result.runner, null, 2));
      } else if (this.runnerPhase === "running") {
        this.emitLine("runner started");
      } else {
        this.emitError(`runner start incomplete: ${this.runnerPhase}`);
      }
      return this.runnerPhase === "running";
    } catch (error) {
      this.emitError(`start-runner failed: ${String(error)}`);
      await this.refreshRunnerStatus();
      return false;
    }
  }

  private async buildDebugBlock(): Promise<void> {
    if (!this.latestSnapshot) {
      await this.resyncSnapshot();
    }
    const sessionRoot = `artifacts/mvp_personas/${this.options.personaId}/sessions/${this.options.sessionId}`;
    const block = {
      persona_id: this.options.personaId,
      session_id: this.options.sessionId,
      gateway_url: this.options.gatewayUrl,
      runner_phase: this.runnerPhase,
      ws_subscribed: this.subscribed,
      snapshot: this.latestSnapshot,
      recent_audits: this.auditTail.list().slice(-12),
      debug_paths: {
        conversation_log: `${sessionRoot}/conversation_log.jsonl`,
        memory_dynamics: `${sessionRoot}/memory_dynamics_episodes.jsonl`,
        environment_events: `${sessionRoot}/environment_events.jsonl`,
      },
      full_mind_debug_hint: `python -c "from pathlib import Path; from segmentum.dialogue.runtime.mvp_loop import MVPStateStore; from segmentum.dialogue.runtime.mind_debug_bundle import build_mind_debug_bundle_text; root=Path('${sessionRoot}'); store=MVPStateStore(root); state=store.load(); print(build_mind_debug_bundle_text(session_root=root, persona_name='${this.options.personaId}', session_id='${this.options.sessionId}', state=state, observability=state.get('observability_summary', {})))"`,
    };
    this.emitLine(JSON.stringify(block, null, 2));
  }

  private warnIfUserUnknown(): void {
    const hints = this.latestSnapshot?.runtime_hints as Record<string, unknown> | undefined;
    const known = hints?.user_known === true;
    const snapshotName = String(hints?.user_display_name ?? "").trim();
    if (this.userDisplayName !== "你") {
      return;
    }
    if (known && snapshotName && snapshotName !== "default_user") {
      return;
    }
    this.emitLine("提示: 她还不确定你是谁；可用 --user-name 你的名字，或在对话里直接自我介绍。");
  }

  private transcriptLabels(): TranscriptLabels {
    return {
      user: this.userDisplayName,
      assistant: this.options.personaId,
      proactive: this.options.personaId,
    };
  }

  private emitTranscript(line: TranscriptLine): void {
    const labels = this.transcriptLabels();
    this.emitLine(
      formatTranscriptLine(line, {
        color: this.color,
        labels,
        labelWidth: resolveTranscriptLabelWidth(labels),
      }),
    );
  }

  private emitStatusLine(): void {
    this.emitLine(
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

  private emitLine(message: string): void {
    emitReplMessage(this.rl, () => this.out.log(message), { prompt: this.prompt });
  }

  private emitError(message: string): void {
    emitReplMessage(this.rl, () => this.out.error(message), { prompt: this.prompt });
  }

  private startThinkingIndicator(): void {
    this.awaitingAssistantReply = true;
    this.replyProgressPercent = 0;
    const personaName = this.options.personaId;
    this.thinking.start({
      formatLine: () => formatReplyProgressLine(personaName, this.replyProgressPercent),
    });
  }

  private handleTurnProgress(message: WsServerMessage): boolean {
    const payload = message.payload ?? {};
    if (String(payload.audit_type ?? "") !== "turn_progress") {
      return false;
    }
    const percent = Number(payload.percent ?? 0);
    if (!Number.isFinite(percent)) {
      return true;
    }
    this.replyProgressPercent = Math.max(this.replyProgressPercent, Math.min(100, Math.round(percent)));
    if (this.awaitingAssistantReply) {
      this.thinking.touch();
    }
    return true;
  }

  private stopThinkingIndicator(): void {
    if (!this.awaitingAssistantReply && !this.thinking.isActive()) {
      return;
    }
    this.awaitingAssistantReply = false;
    this.thinking.stop();
  }

  private renderThinkingLine(line: string): void {
    emitReplInlineStatus(this.rl, line, { prompt: this.prompt });
  }

  private clearThinkingLine(): void {
    clearReplInlineStatus(this.rl, { prompt: this.prompt });
  }
}

export async function runRepl(options: ReplOptions): Promise<void> {
  if (options.input) {
    const repl = new ConsciousnessRepl(options);
    await repl.run(options.input);
    return;
  }
  const stdin = createStdinLineSource(options.prompt);
  const repl = new ConsciousnessRepl({ ...options, readline: stdin.readline, prompt: stdin.prompt });
  await repl.run(stdin.lines());
}
