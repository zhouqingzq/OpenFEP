# @segments/consciousness-client

Shared TypeScript SDK for the M16 consciousness gateway (HTTP perception + WebSocket actuation).

## Install (workspace)

```bash
cd ui/packages/consciousness-client
npm install
npm run build
npm test
npm run validate-schemas
```

## Quick start

```typescript
import { createConsciousnessClient } from "@segments/consciousness-client";

const client = createConsciousnessClient({
  baseUrl: "http://127.0.0.1:8765",
  personaId: "demo",
  sessionId: "demo",
  authToken: process.env.M16_AUTH_TOKEN, // optional off-localhost
});

const stream = client.connectStream();
stream.on("assistantMessage", (msg) => console.log(msg.payload.text));
await stream.connect();
await client.sendUserInput("hello");
```

## Boundary rules

- SDK validates all WS messages against `schemas/m16/*.json`.
- SDK never schedules cognition, computes idle eligibility, or parses user text with keyword heuristics.
- SDK never emits server-owned lifecycle events such as `DeliverySurfaceDisconnectedEvent`.
- `postInput` treats HTTP `202 Accepted` as success.

## Scripts

| Script | Purpose |
|--------|---------|
| `npm run build` | ESM + `.d.ts` to `dist/` |
| `npm test` | Vitest unit tests |
| `M16_INTEGRATION=1 npm run test:integration` | Optional local gateway smoke test |
| `npm run validate-schemas` | CI schema compile + fixture roundtrips |
| `npm run example` | `examples/minimal-node.ts` against local gateway |

## Schema source of truth

```text
schemas/m16/http.openapi.yaml
schemas/m16/ws_client_messages.schema.json
schemas/m16/ws_server_messages.schema.json
segmentum/dialogue/runtime/m16_protocol.py
```
