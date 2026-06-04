import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SCHEMA_VERSION, type ValidationMode, type WsClientMessage, type WsServerMessage } from "./types.js";
import { SchemaVersionError, ValidationError } from "./errors.js";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const schemasDir = join(packageRoot, "../../../schemas/m16");

/** Keys that must never appear on inbound actuation payloads (mirrors m16_protocol.py). */
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

const ACTUATION_SERVER_KINDS = new Set([
  "AssistantMessageCommitted",
  "ProactiveMessageCommitted",
  "UserMessageAccepted",
  "TurnCompleted",
  "AuditEvent",
  "RunnerSuppression",
]);

function loadSchema(name: string): Record<string, unknown> {
  const text = readFileSync(join(schemasDir, name), "utf8");
  return JSON.parse(text) as Record<string, unknown>;
}

let ajvInstance: Ajv2020 | null = null;
let validateClientFn: ReturnType<Ajv2020["compile"]> | null = null;
let validateServerFn: ReturnType<Ajv2020["compile"]> | null = null;

function getAjv(): Ajv2020 {
  if (ajvInstance) {
    return ajvInstance;
  }
  ajvInstance = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajvInstance);
  validateClientFn = ajvInstance.compile(loadSchema("ws_client_messages.schema.json"));
  validateServerFn = ajvInstance.compile(loadSchema("ws_server_messages.schema.json"));
  return ajvInstance;
}

export function schemasDirectory(): string {
  return schemasDir;
}

export function resetValidatorsForTests(): void {
  ajvInstance = null;
  validateClientFn = null;
  validateServerFn = null;
}

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

export function findForbiddenPayloadKeys(payload: Record<string, unknown>): string[] {
  return forbiddenPayloadErrors(payload);
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
  getAjv();
  if (validateClientFn && !validateClientFn(msg)) {
    for (const err of validateClientFn.errors ?? []) {
      errors.push(`${err.instancePath || "root"}:${err.message ?? "invalid"}`);
    }
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
  getAjv();
  if (validateServerFn && !validateServerFn(msg)) {
    for (const err of validateServerFn.errors ?? []) {
      errors.push(`${err.instancePath || "root"}:${err.message ?? "invalid"}`);
    }
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

export function compileSchemasForCi(): { clientOk: boolean; serverOk: boolean } {
  getAjv();
  return { clientOk: Boolean(validateClientFn), serverOk: Boolean(validateServerFn) };
}
