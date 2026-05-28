import { describe, expect, it } from "vitest";

import { createConsciousnessClient } from "../src/index.js";

const integrationEnabled = process.env.M16_INTEGRATION === "1";

describe.skipIf(!integrationEnabled)("M16 gateway integration", () => {
  it("health and snapshot against local gateway", async () => {
    const baseUrl = process.env.M16_BASE_URL ?? "http://127.0.0.1:8765";
    const personaId = process.env.M16_PERSONA ?? "demo";
    const sessionId = process.env.M16_SESSION ?? "demo";
    const client = createConsciousnessClient({
      baseUrl,
      personaId,
      sessionId,
      authToken: process.env.M16_AUTH_TOKEN,
    });
    const health = await client.health();
    expect(health.schema_version).toBe("m16.0");
    const snapshot = await client.getSnapshot();
    expect(snapshot.session_id).toBe(sessionId);
  });
});
