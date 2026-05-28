import { ConsciousnessHttpClient, buildWsStreamUrl, newCorrelationId, newMessageId } from "./http.js";
import { ConsciousnessStream } from "./ws.js";
import {
  type ConsciousnessClientOptions,
  type StreamConnectOptions,
  type WsServerMessage,
} from "./types.js";

export class ConsciousnessClient {
  readonly personaId: string;
  readonly sessionId: string;
  readonly baseUrl: string;

  private readonly http: ConsciousnessHttpClient;
  private readonly options: ConsciousnessClientOptions;
  private activeStream: ConsciousnessStream | null = null;

  constructor(options: ConsciousnessClientOptions) {
    this.options = options;
    this.personaId = options.personaId;
    this.sessionId = options.sessionId;
    this.baseUrl = options.baseUrl;
    this.http = new ConsciousnessHttpClient(this.httpConfig());
  }

  private httpConfig() {
    return {
      baseUrl: this.options.baseUrl,
      personaId: this.options.personaId,
      sessionId: this.options.sessionId,
      authToken: this.options.authToken,
      fetchImpl: this.options.fetchImpl,
      WebSocketImpl: this.options.WebSocketImpl,
    };
  }

  health() {
    return this.http.health();
  }

  createSession(body: Parameters<ConsciousnessHttpClient["createSession"]>[0]) {
    return this.http.createSession(body);
  }

  getSession() {
    return this.http.getSession();
  }

  getSnapshot() {
    return this.http.getSnapshot();
  }

  postInput(text: string, metadata?: Parameters<ConsciousnessHttpClient["postInput"]>[1]) {
    return this.http.postInput(text, metadata);
  }

  sendUserInput(text: string, metadata?: Parameters<ConsciousnessHttpClient["postInput"]>[1]) {
    return this.postInput(text, metadata);
  }

  startRunner(reason?: string) {
    return this.http.startRunner(reason);
  }

  stopRunner(reason?: string) {
    return this.http.stopRunner(reason);
  }

  getRunnerStatus() {
    return this.http.getRunnerStatus();
  }

  connectStream(options?: StreamConnectOptions): ConsciousnessStream {
    const stream = new ConsciousnessStream(
      this.httpConfig(),
      options,
      this.options.validationMode ?? "strict",
    );
    this.activeStream = stream;
    return stream;
  }

  async disconnectStream(): Promise<void> {
    await this.activeStream?.disconnect();
    this.activeStream = null;
  }
}

export function createConsciousnessClient(options: ConsciousnessClientOptions): ConsciousnessClient {
  return new ConsciousnessClient(options);
}

export function isAssistantMessage(message: WsServerMessage): boolean {
  return message.kind === "AssistantMessageCommitted";
}

export function isProactiveMessage(message: WsServerMessage): boolean {
  return message.kind === "ProactiveMessageCommitted";
}

export function isSuppressionEvent(message: WsServerMessage): boolean {
  return message.kind === "RunnerSuppression";
}

export {
  ConsciousnessHttpClient,
  ConsciousnessStream,
  buildWsStreamUrl,
  newCorrelationId,
  newMessageId,
};
export * from "./types.js";
export * from "./errors.js";
export { ReconnectBackoff, sleep } from "./reconnect.js";
export {
  validateOutboundClientMessage,
  validateInboundServerMessage,
  FORBIDDEN_ACTUATION_PAYLOAD_KEYS,
} from "./validate-lite.js";
