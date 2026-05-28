import { describe, expect, it } from "vitest";

import { AuditTail } from "../src/render/audit_tail.js";
import { SCHEMA_VERSION, type WsServerMessage } from "@segments/consciousness-client";

function healthMessage(ready: boolean, reason = ""): WsServerMessage {
  return {
    schema_version: SCHEMA_VERSION,
    message_id: "m16s_health",
    persona_id: "p",
    session_id: "s",
    at: 100,
    kind: "RunnerHealth",
    payload: { delivery_surface_ready: ready, delivery_surface_reason: reason },
  };
}

function suppressionMessage(reason: string, at: number): WsServerMessage {
  return {
    schema_version: SCHEMA_VERSION,
    message_id: `m16s_sup_${at}`,
    persona_id: "p",
    session_id: "s",
    at,
    kind: "RunnerSuppression",
    payload: { reason_code: reason },
  };
}

describe("AuditTail", () => {
  it("hides RunnerHealth unless verbose", () => {
    const tail = new AuditTail();
    expect(tail.push(healthMessage(false, "delivery_surface_not_ready"))).toBeNull();
    const verbose = new AuditTail({ verbose: true });
    expect(verbose.push(healthMessage(false, "delivery_surface_not_ready"))?.text).toContain(
      "delivery_surface_not_ready",
    );
  });

  it("dedupes repeated suppression codes", () => {
    const tail = new AuditTail();
    expect(tail.push(suppressionMessage("delivery_surface_not_ready", 100))?.text).toBe(
      "delivery_surface_not_ready",
    );
    expect(tail.push(suppressionMessage("delivery_surface_not_ready", 101))).toBeNull();
    expect(tail.push(suppressionMessage("cooldown_active", 110))?.text).toBe("cooldown_active");
  });
});
