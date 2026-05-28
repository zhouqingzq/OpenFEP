# @segments/consciousness-client (M16.0 scaffold)

M16.0 defines the wire protocol only. **Do not add runtime code here until M16.2.**

## Intended exports (M16.2)

```text
createConsciousnessClient({ baseUrl, personaId, sessionId })
ConsciousnessClient
ConsciousnessStream
types for WS/HTTP messages (from schemas/m16)
validateOutboundMessage / validateInboundMessage
```

## Boundary rules

- HTTP `postInput` treats `202 Accepted` as success.
- WebSocket client sends `Subscribe`, then `DeliverySurfaceReady`.
- SDK never schedules cognition, computes idle eligibility, or parses user text
  with keyword heuristics.
- Reconnect must resync via REST snapshot before relying on WS tail alone.

## Schema source of truth

```text
schemas/m16/http.openapi.yaml
schemas/m16/ws_client_messages.schema.json
schemas/m16/ws_server_messages.schema.json
schemas/m16/perception_events.schema.json
schemas/m16/actuation_events.schema.json
segmentum/dialogue/runtime/m16_protocol.py
```

## Consumers

- `ui/web` (M16.3)
- `ui/tui` (M16.4)
