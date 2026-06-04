import {
  type CreateSessionRequest,
  type CreateSessionResponse,
  type HealthResponse,
  type PostInputRequest,
  type PostInputResponse,
  type RunnerControlRequest,
  type RunnerStatusResponse,
  type SessionMetadataResponse,
  type SnapshotResponse,
  SCHEMA_VERSION,
} from "./types.js";
import { GatewayError } from "./errors.js";

export interface HttpClientConfig {
  baseUrl: string;
  personaId: string;
  sessionId: string;
  authToken?: string;
  fetchImpl?: typeof fetch;
  WebSocketImpl?: new (url: string, protocols?: string | string[]) => WebSocket;
}

export function newCorrelationId(prefix = "corr"): string {
  const suffix =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  return `${prefix}_${suffix}`.slice(0, 120);
}

export function newMessageId(prefix = "m16c"): string {
  const suffix =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  return `${prefix}_${suffix}`.slice(0, 120);
}

function trimBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

function sessionBase(config: HttpClientConfig): string {
  const base = trimBaseUrl(config.baseUrl);
  const persona = encodeURIComponent(config.personaId);
  const session = encodeURIComponent(config.sessionId);
  return `${base}/v1/personas/${persona}/sessions/${session}`;
}

function headers(config: HttpClientConfig, extra?: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {
    Accept: "application/json",
    ...extra,
  };
  if (config.authToken) {
    out.Authorization = `Bearer ${config.authToken}`;
  }
  return out;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) {
    return {};
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { raw: text };
  }
}

function reasonCodeFromBody(body: unknown): string {
  if (!body || typeof body !== "object") {
    return "gateway_error";
  }
  const row = body as Record<string, unknown>;
  return String(row.reason_code ?? row.detail ?? row.code ?? "gateway_error").slice(0, 160);
}

async function expectJson<T>(
  config: HttpClientConfig,
  response: Response,
  allowedStatuses: number[],
): Promise<T> {
  const body = await readJson(response);
  if (!allowedStatuses.includes(response.status)) {
    throw new GatewayError(`HTTP ${response.status}`, {
      status: response.status,
      reasonCode: reasonCodeFromBody(body),
      body,
    });
  }
  return body as T;
}

export class ConsciousnessHttpClient {
  private readonly config: HttpClientConfig;

  constructor(config: HttpClientConfig) {
    this.config = config;
  }

  private fetch(input: string, init?: RequestInit): Promise<Response> {
    const impl = this.config.fetchImpl ?? globalThis.fetch.bind(globalThis);
    return impl(input, init);
  }

  async health(): Promise<HealthResponse> {
    const url = `${trimBaseUrl(this.config.baseUrl)}/health`;
    const response = await this.fetch(url, { method: "GET", headers: headers(this.config) });
    return expectJson<HealthResponse>(this.config, response, [200]);
  }

  async createSession(body: CreateSessionRequest): Promise<CreateSessionResponse> {
    const base = trimBaseUrl(this.config.baseUrl);
    const persona = encodeURIComponent(this.config.personaId);
    const url = `${base}/v1/personas/${persona}/sessions`;
    const response = await this.fetch(url, {
      method: "POST",
      headers: headers(this.config, { "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    return expectJson<CreateSessionResponse>(this.config, response, [201]);
  }

  async getSession(): Promise<SessionMetadataResponse> {
    const url = sessionBase(this.config);
    const response = await this.fetch(url, { method: "GET", headers: headers(this.config) });
    return expectJson<SessionMetadataResponse>(this.config, response, [200]);
  }

  async getSnapshot(): Promise<SnapshotResponse> {
    const url = `${sessionBase(this.config)}/snapshot`;
    const response = await this.fetch(url, { method: "GET", headers: headers(this.config) });
    return expectJson<SnapshotResponse>(this.config, response, [200]);
  }

  async postInput(
    text: string,
    metadata?: Partial<Pick<PostInputRequest, "correlation_id" | "speaker_name" | "group_turn_envelope">>,
  ): Promise<PostInputResponse> {
    const url = `${sessionBase(this.config)}/input`;
    const payload: PostInputRequest = {
      text: text.slice(0, 8000),
      correlation_id: metadata?.correlation_id ?? newCorrelationId("in"),
    };
    const speakerName = String(metadata?.speaker_name ?? "").trim();
    if (speakerName) {
      payload.speaker_name = speakerName.slice(0, 64);
    }
    if (metadata?.group_turn_envelope && typeof metadata.group_turn_envelope === "object") {
      payload.group_turn_envelope = metadata.group_turn_envelope;
    }
    const response = await this.fetch(url, {
      method: "POST",
      headers: headers(this.config, { "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    return expectJson<PostInputResponse>(this.config, response, [202]);
  }

  async startRunner(reason = ""): Promise<{ command: string; runner: RunnerStatusResponse["runner"]; correlation_id: string }> {
    return this.runnerControl({ command: "start", reason });
  }

  async stopRunner(reason = ""): Promise<{ command: string; runner: RunnerStatusResponse["runner"]; correlation_id: string }> {
    return this.runnerControl({ command: "stop", reason });
  }

  async getRunnerStatus(): Promise<RunnerStatusResponse> {
    const url = `${sessionBase(this.config)}/runner/status`;
    const response = await this.fetch(url, { method: "GET", headers: headers(this.config) });
    return expectJson<RunnerStatusResponse>(this.config, response, [200]);
  }

  private async runnerControl(options: {
    command: RunnerControlRequest["command"];
    reason?: string;
  }): Promise<{ command: string; runner: RunnerStatusResponse["runner"]; correlation_id: string }> {
    const url = `${sessionBase(this.config)}/runner/${options.command === "status" ? "status" : options.command}`;
    const correlation_id = newCorrelationId("runner");
    if (options.command === "status") {
      return this.getRunnerStatus().then((row) => ({
        command: "status",
        runner: row.runner,
        correlation_id,
      }));
    }
    const payload: RunnerControlRequest = {
      correlation_id,
      command: options.command,
      reason: (options.reason ?? "").slice(0, 160),
    };
    const response = await this.fetch(url, {
      method: "POST",
      headers: headers(this.config, { "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    return expectJson(this.config, response, [202]);
  }
}

export function buildWsStreamUrl(config: HttpClientConfig): string {
  const httpBase = trimBaseUrl(config.baseUrl);
  const wsBase = httpBase.replace(/^http/i, "ws");
  const persona = encodeURIComponent(config.personaId);
  const session = encodeURIComponent(config.sessionId);
  return `${wsBase}/v1/personas/${persona}/sessions/${session}/stream`;
}

export { SCHEMA_VERSION };
