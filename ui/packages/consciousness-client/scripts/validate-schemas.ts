import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  compileSchemasForCi,
  validateInboundServerMessage,
  validateOutboundClientMessage,
} from "../src/validate.js";
import { SCHEMA_VERSION } from "../src/types.js";

const compiled = compileSchemasForCi();
if (!compiled.clientOk || !compiled.serverOk) {
  throw new Error("failed to compile M16 JSON schemas");
}

const subscribe = {
  schema_version: SCHEMA_VERSION,
  message_id: "m16c_sub",
  persona_id: "p",
  session_id: "s",
  at: 1_900_000_000,
  kind: "Subscribe",
  correlation_id: "corr_sub",
  payload: { resume_from_message_id: "" },
};

validateOutboundClientMessage(subscribe);

const snapshot = {
  schema_version: SCHEMA_VERSION,
  message_id: "m16s_snap",
  persona_id: "p",
  session_id: "s",
  at: 1_900_000_000,
  kind: "SessionSnapshot",
  payload: { chat_tail: [], runtime_hints: {} },
};

const inbound = validateInboundServerMessage(snapshot, "strict");
if (!inbound.message) {
  throw new Error("expected snapshot message");
}

void readFileSync(join(process.cwd(), "../../../schemas/m16/ws_client_messages.schema.json"), "utf8");
void readFileSync(join(process.cwd(), "../../../schemas/m16/ws_server_messages.schema.json"), "utf8");

console.log("M16 schema roundtrips OK");
