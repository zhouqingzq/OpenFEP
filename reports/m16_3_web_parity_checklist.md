# M16.3 Web Parity Checklist

Path B web thin client (`ui/web`) vs Streamlit MVP chat (`segmentum/dialogue/runtime/app.py`).

Legend: **Web** = M16.3 surface; **Streamlit** = legacy UI; **Runner** = M16.1 gateway + consciousness runner.

| Capability | Web (M16.3) | Streamlit | Notes |
|------------|-------------|-----------|-------|
| Send user message | Yes — HTTP `POST /input` via SDK | Yes — inline chat path | Web never calls `run_turn` in browser |
| Receive assistant reply | Yes — WS `AssistantMessageCommitted` | Yes — synchronous turn render | Web requires runner + delivery surface |
| Proactive message display | Yes — WS `ProactiveMessageCommitted` | Yes — implicit idle / proactive bubble | Web shows server text only; no client scheduling |
| Suppression reason display | Yes — toast with `reason_code` string | Yes — sidebar / diagnostics | Web does not interpret codes semantically |
| Connection loss + resume | Yes — SDK reconnect + snapshot `resync` | Partial — rerun-driven | Web dedupes by `delivery_id` |
| Session persistence on refresh | Yes — query string + `localStorage` | Yes — Streamlit session state | Persona/session fields editable in sidebar |
| Delivery ack tracing | Yes — `DeliveryAck` after render | No equivalent UI ack | Supports M14.2 idempotency tracing |
| Runner start/stop/status | Yes — sidebar HTTP controls | Partial — daemon CLI / hidden toggles | Web operator panel only |
| Mind debug bundle | Partial — copy REST snapshot JSON | Yes — full bundle export | No filesystem log scrape in browser |
| Persona material admin | No | Yes | Deferred to later admin UI |
| M13/M14 diagnostic dashboards | Partial — bounded audit tail + snapshot hints | Yes — rich Streamlit sidebar | Web read-only, no TS cognition logic |
| Idle tick scheduling | **No (by design)** | Yes (legacy rerun scheduler) | **Streamlit must not be acceptance path for proactive** |
| Keyword / regex user semantics | **No** | Must not add in engineering layer | Both rely on server-side LLM JSON |

## Acceptance path (M16.3)

```text
terminal A: python -m segmentum.dialogue.runtime.m16_api --persona 胡桃 --session <id>
terminal B: cd ui/web && npm run dev
browser: http://127.0.0.1:5173/?persona=胡桃&session=<id>
```

Streamlit may remain open for admin, but proactive acceptance uses runner + web delivery surface only.

## Deferred to M16.4+

- Streamlit deprecation banner and TUI client
- Full mind debug bundle REST route (if added later to gateway)
- OAuth / multi-user auth
- Persona file management UI
