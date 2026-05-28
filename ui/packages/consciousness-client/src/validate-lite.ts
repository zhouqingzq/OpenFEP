import { SCHEMA_VERSION, type ValidationMode, type WsClientMessage, type WsServerMessage } from "./types.js";
import { SchemaVersionError, ValidationError } from "./errors.js";

export const FORBIDDEN_ACTUATION_PAYLOAD_KEYS = new Set(
  [
    "system_prompt",
    "user_prompt",
    "raw_prompt",
    "raw_prompt_text",
    "full_prompt",
    "prompt_text",
    "conscious_markdown",
    "full_conscious_markdown",
    "conscious_plan",
    "memory_dynamics",
    "full_memory_dump",
    "memory_dump",
    "short_term_memory",
    "long_term_memory",
    "llm_thinking_result",
    "diagnostics",
    "internal_patch",
    "patch_payload",
    "m13_drive_state",
    "meta_control_intents",
  ].map((k) => k.toLowerCase()),
);

const CLIENT_KINDS = new Set([
  "Subscribe",
  "Ping",
  "ClientInput",
  "DeliverySurfaceReady",
  "DeliveryAck",
  "Unsubscribe",
]);

const SERVER_KINDS = new Set([
  "Subscribed",
  "SessionSnapshot",
  "UserMessageAccepted",
  "AssistantMessageCommitted",
  "ProactiveMessageCommitted",
  "AuditEvent",
  "RunnerHealth",
  "RunnerSuppression",
  "Error",
]);

const ACTUATION_SERVER_KINDS = new Set([
  "AssistantMessageCommitted",
  "ProactiveMessageCommitted",
  "UserMessageAccepted",
  "AuditEvent",
  "RunnerSuppression",
]);

function envelopeErrors(row: Record<string, unknown>, requireCorrelation: boolean): string[] {
  const errors: string[] = [];
  if (row.schema_version !== SCHEMA_VERSION) {
    errors.push("schema_version");
  }
  for (const key of ["message_id", "session_id", "persona_id", "at", "kind", "payload"]) {
    if (!(key in row)) {
      errors.push(`missing:${key}`);
    }
  }
  if (requireCorrelation && !String(row.correlation_id ?? "").trim()) {
    errors.push("missing:correlation_id");
  }
  if (typeof row.payload !== "object" || row.payload === null || Array.isArray(row.payload)) {
    errors.push("payload_not_object");
  }
  return errors;
}

function forbiddenPayloadErrors(payload: Record<string, unknown>, prefix = ""): string[] {
  const errors: string[] = [];
  for (const [key, value] of Object.entries(payload)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (FORBIDDEN_ACTUATION_PAYLOAD_KEYS.has(key.toLowerCase())) {
      errors.push(`forbidden:${path}`);
    }
    if (value && typeof value === "object" && !Array.isArray(value)) {
      errors.push(...forbiddenPayloadErrors(value as Record<string, unknown>, path));
    }
  }
  return errors;
}

export function assertSchemaVersion(row: Record<string, unknown>): void {
  const version = String(row.schema_version ?? "");
  if (version !== SCHEMA_VERSION) {
    throw new SchemaVersionError(SCHEMA_VERSION, version || "<missing>");
  }
}

export function validateOutboundClientMessage(row: unknown): WsClientMessage {
  if (!row || typeof row !== "object") {
    throw new ValidationError("client message must be an object", ["not_object"]);
  }
  const msg = row as Record<string, unknown>;
  const errors = envelopeErrors(msg, true);
  if (!CLIENT_KINDS.has(String(msg.kind ?? ""))) {
    errors.push("invalid:kind");
  }
  if (errors.length) {
    throw new ValidationError("invalid outbound client message", errors);
  }
  return msg as unknown as WsClientMessage;
}

export interface InboundValidationResult {
  message: WsServerMessage | null;
  warnings: string[];
}

export function validateInboundServerMessage(
  row: unknown,
  mode: ValidationMode = "strict",
): InboundValidationResult {
  const warnings: string[] = [];
  if (!row || typeof row !== "object") {
    if (mode === "strict") {
      throw new ValidationError("server message must be an object", ["not_object"]);
    }
    warnings.push("not_object");
    return { message: null, warnings };
  }
  const msg = row as Record<string, unknown>;
  try {
    assertSchemaVersion(msg);
  } catch (err) {
    if (mode === "strict") {
      throw err;
    }
    warnings.push("schema_version_mismatch");
    return { message: null, warnings };
  }
  const errors = envelopeErrors(msg, false);
  if (!SERVER_KINDS.has(String(msg.kind ?? ""))) {
    errors.push("invalid:kind");
  }
  if (errors.length) {
    if (mode === "strict") {
      throw new ValidationError("invalid inbound server message", errors);
    }
    warnings.push(...errors);
    return { message: null, warnings };
  }
  const kind = String(msg.kind ?? "");
  if (ACTUATION_SERVER_KINDS.has(kind) && msg.payload && typeof msg.payload === "object") {
    const forbidden = forbiddenPayloadErrors(msg.payload as Record<string, unknown>);
    if (forbidden.length) {
      if (mode === "strict") {
        throw new ValidationError("inbound actuation payload contains forbidden keys", forbidden);
      }
      warnings.push(...forbidden);
      return { message: null, warnings };
    }
  }
  return { message: msg as unknown as WsServerMessage, warnings };
}
