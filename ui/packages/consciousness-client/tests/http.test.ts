import { describe, expect, it } from "vitest";

import { ConsciousnessHttpClient } from "../src/http.js";
import { mockFetch } from "./helpers.js";

describe("http client", () => {
  it("test_http_post_input_expects_202", async () => {
    const fetchImpl = mockFetch({
      "POST ": () =>
        new Response(
          JSON.stringify({
            accepted: true,
            event_id: "evt_1",
            persona_id: "p",
            session_id: "s",
            correlation_id: "corr_in",
            schema_version: "m16.0",
          }),
          { status: 202 },
        ),
    });
    const http = new ConsciousnessHttpClient({
      baseUrl: "http://127.0.0.1:8765",
      personaId: "p",
      sessionId: "s",
      fetchImpl,
    });
    const result = await http.postInput("hello", { correlation_id: "corr_in" });
    expect(result.accepted).toBe(true);
    expect(result.event_id).toBe("evt_1");
  });
});
