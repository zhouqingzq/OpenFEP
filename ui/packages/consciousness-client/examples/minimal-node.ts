/**
 * Minimal Node example: connect to a local M16.1 gateway, print events, send one stdin line.
 *
 * Usage:
 *   npm run build && npm run example
 *
 * Env:
 *   M16_BASE_URL (default http://127.0.0.1:8765)
 *   M16_PERSONA (default demo)
 *   M16_SESSION (default demo)
 *   M16_AUTH_TOKEN (optional Bearer token)
 */

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { createConsciousnessClient, isAssistantMessage, isProactiveMessage } from "../src/index.js";

async function main(): Promise<void> {
  const baseUrl = process.env.M16_BASE_URL ?? "http://127.0.0.1:8765";
  const personaId = process.env.M16_PERSONA ?? "demo";
  const sessionId = process.env.M16_SESSION ?? "demo";
  const authToken = process.env.M16_AUTH_TOKEN;

  const client = createConsciousnessClient({
    baseUrl,
    personaId,
    sessionId,
    authToken,
  });

  const health = await client.health();
  console.log("health", health);

  const stream = client.connectStream({ autoReconnect: true });
  stream.on("assistantMessage", (msg) => {
    if (isAssistantMessage(msg)) {
      console.log("[assistant]", msg.payload.text ?? "");
    }
  });
  stream.on("proactiveMessage", (msg) => {
    if (isProactiveMessage(msg)) {
      console.log("[proactive]", msg.payload.text ?? "");
    }
  });
  stream.on("suppression", (msg) => console.log("[suppression]", msg.payload));
  stream.on("auditEvent", (msg) => console.log("[audit]", msg.kind, msg.payload));
  stream.on("resync", (snap) => console.log("[resync]", snap.chat_tail?.length ?? 0, "rows"));

  await stream.connect();
  console.log("connected; type a line and press Enter (empty line to quit)");

  const rl = createInterface({ input, output });
  while (true) {
    const line = (await rl.question("> ")).trim();
    if (!line) {
      break;
    }
    const accepted = await client.sendUserInput(line);
    console.log("input accepted", accepted.event_id);
  }
  rl.close();
  await client.disconnectStream();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
