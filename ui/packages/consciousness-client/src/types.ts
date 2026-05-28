/** M16.0 wire protocol constants and TypeScript types. */

export const SCHEMA_VERSION = "m16.0" as const;

export const WS_CLIENT_KINDS = [
  "Subscribe",
  "Ping",
  "ClientInput",
  "DeliverySurfaceReady",
  "DeliveryAck",
  "Unsubscribe",
] as const;

export const WS_SERVER_KINDS = [
  "Subscribed",
  "SessionSnapshot",
  "UserMessageAccepted",
  "AssistantMessageCommitted",
  "ProactiveMessageCommitted",
  "AuditEvent",
  "RunnerHealth",
  "RunnerSuppression",
  "Error",
] as const;

export type WsClientKind = (typeof WS_CLIENT_KINDS)[number];
export type WsServerKind = (typeof WS_SERVER_KINDS)[number];

export interface WsEnvelopeBase {
  schema_version: typeof SCHEMA_VERSION;
  message_id: string;
  persona_id: string;
  session_id: string;
  at: number;
  kind: string;
  payload: Record<string, unknown>;
}

export interface WsClientMessage extends WsEnvelopeBase {
  correlation_id: string;
  kind: WsClientKind;
}

export interface WsServerMessage extends WsEnvelopeBase {
  kind: WsServerKind;
}

export interface ChatTailRow {
  event?: string;
  text?: string;
  turn_index?: number;
  at?: number;
}

export interface SessionSnapshotPayload {
  persona_id?: string;
  session_id?: string;
  chat_tail?: ChatTailRow[];
  runtime_hints?: Record<string, unknown>;
}

export interface AssistantMessagePayload {
  text?: string;
  turn_index?: number;
  delivery_id?: string;
}

export interface ProactiveMessagePayload {
  text?: string;
  proposal_id?: string;
  delivery_id?: string;
}

export interface SuppressionPayload {
  reason_code?: string;
}

export interface DeliveryAckPayload {
  delivery_id: string;
}

export interface ClientInputPayload {
  text: string;
}

export interface SubscribePayload {
  resume_from_message_id?: string;
}

export interface HealthResponse {
  status: string;
  schema_version: string;
}

export interface CreateSessionRequest {
  correlation_id: string;
  session_id?: string;
}

export interface CreateSessionResponse {
  persona_id: string;
  session_id: string;
  schema_version: string;
  correlation_id: string;
}

export interface PostInputRequest {
  text: string;
  correlation_id: string;
}

export interface PostInputResponse {
  accepted: boolean;
  event_id: string;
  persona_id: string;
  session_id: string;
  correlation_id: string;
  schema_version: string;
}

export interface RunnerControlRequest {
  correlation_id: string;
  command: "start" | "stop" | "status";
  reason?: string;
}

export interface RunnerStatusPayload {
  running: boolean;
  pid?: number;
  runner_kind?: string;
  last_health_at?: number;
  last_tick_at?: number;
  steps_total?: number;
  last_error?: string;
}

export interface RunnerStatusResponse {
  schema_version: string;
  persona_id: string;
  session_id: string;
  runner: RunnerStatusPayload;
  engineering_proxy_label?: string;
}

export interface SessionMetadataResponse {
  persona_id: string;
  session_id: string;
  schema_version: string;
  runner: RunnerStatusPayload;
  delivery_surface?: {
    ws_subscribed?: boolean;
    delivery_surface_ready_at?: number;
  };
}

export interface SnapshotResponse extends SessionSnapshotPayload {
  schema_version: string;
}

export type ValidationMode = "strict" | "lenient";

export interface ConsciousnessClientOptions {
  baseUrl: string;
  personaId: string;
  sessionId: string;
  authToken?: string;
  validationMode?: ValidationMode;
  fetchImpl?: typeof fetch;
  WebSocketImpl?: new (url: string, protocols?: string | string[]) => WebSocket;
}

export interface StreamConnectOptions {
  autoReconnect?: boolean;
  maxBackoffMs?: number;
  resumeFromMessageId?: string;
}

export interface ValidationWarning {
  code: "invalid_server_message" | "forbidden_payload_key" | "schema_version_mismatch";
  message: string;
  details?: string[];
}

export type StreamEventMap = {
  open: [];
  close: [{ code?: number; reason?: string }];
  subscribed: [WsServerMessage];
  sessionSnapshot: [WsServerMessage];
  userMessageAccepted: [WsServerMessage];
  assistantMessage: [WsServerMessage];
  proactiveMessage: [WsServerMessage];
  auditEvent: [WsServerMessage];
  runnerHealth: [WsServerMessage];
  suppression: [WsServerMessage];
  error: [WsServerMessage];
  resync: [SnapshotResponse];
  validationWarning: [ValidationWarning];
  raw: [WsServerMessage];
};

export type StreamEventName = keyof StreamEventMap;
