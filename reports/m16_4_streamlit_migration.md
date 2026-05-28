# M16.4 Streamlit Migration Guide

## Why UI-driven scheduling failed

Path B cognition was originally wired through Streamlit reruns: implicit idle
proactive delivery (M14.4), idle introspection ticks (M13.4/M13.5), queued
outreach drain, and background Streamlit pings all fired from `app.py` on every
rerun or fragment timer.

Symptoms visible in mind debug bundles and operator logs:

- duplicate or competing cognition loops when a page stayed open alongside CLI
  daemons or manual continues,
- proactive attempts throttled or suppressed with opaque `reason_code` values
  while the UI still looked "idle",
- session state and runner state diverging because Streamlit reruns are not a
  durable scheduler,
- delivery surface readiness missing for M14.2 idempotency (no client-side
  `DeliveryAck` in Streamlit).

M16 moves orchestration to `ConsciousnessRunner` behind the M16 gateway. UI
clients subscribe and publish input only.

## Quick start: gateway + web chat

```bash
# Terminal A — gateway + runner bridge
python -m segmentum.dialogue.runtime.m16_api --host 127.0.0.1 --port 8765

# Terminal B — web thin client (Vite proxies /v1 and /health to gateway)
cd ui/web
npm install
npm run dev

# Browser
# http://127.0.0.1:5173/?persona=胡桃&session=<session_id>
```

Optional env:

```text
SEGMENTS_CONSCIOUSNESS_GATEWAY_URL=http://127.0.0.1:8765
M16_RUNNER=1
```

## Quick start: gateway + TUI

```bash
# Terminal A — same gateway command as above

# Terminal B — terminal client
cd ui/tui
npm install
npm run build
npx consciousness-tui --persona 胡桃 --session <session_id> --gateway http://127.0.0.1:8765
```

Operator commands inside the TUI:

```text
/status
/snapshot
/debug
/quit
```

If the gateway is offline, the TUI still supports REST-only `/status` polling
and prints a clear offline message; WS auto-reconnect resumes when the gateway
returns.

## Legacy Streamlit adapter

```bash
streamlit run segmentum/dialogue/runtime/app.py
```

After M16.4, Streamlit shows a startup banner and **does not schedule** idle
ticks, implicit proactive delivery, queued outreach drain, or background ping
unless legacy mode is explicitly enabled:

```text
SEGMENTS_LEGACY_STREAMLIT_SCHEDULER=1
```

Runner mode always wins over the legacy flag:

```text
M16_RUNNER=1
SEGMENTS_LEGACY_STREAMLIT_SCHEDULER=1   # scheduling still OFF in Streamlit
```

## Dual-run cautions

Never run two schedulers for the same persona/session:

```text
BAD: M16 runner active + Streamlit legacy scheduler on same session
BAD: Streamlit implicit idle + manual daemon drain on same session without gates
GOOD: one runner, one delivery surface (web or TUI), Streamlit closed or legacy scheduler off
```

Guard implementation: `segmentum/dialogue/runtime/m16_streamlit_legacy.py`.

## Feature parity matrix

| Capability | Web (M16.3) | TUI (M16.4) | Streamlit (legacy) |
|------------|-------------|-------------|---------------------|
| Send user input | HTTP POST | HTTP POST | Inline chat |
| Receive assistant/proactive | WS + ack | WS + ack | Inline render |
| Audit / suppression tail | Sidebar panel | Colored stdout | Sidebar / diagnostics |
| Runner start/stop/status | HTTP buttons | `/status` | Partial / hidden |
| Mind debug bundle | Snapshot JSON copy | `/debug` block | Full export |
| Persona admin | No | No | Yes |
| Schedules cognition | **No** | **No** | Only if `SEGMENTS_LEGACY_STREAMLIT_SCHEDULER=1` |
| Delivery ack tracing | Yes | Yes | No |

See also `reports/m16_3_web_parity_checklist.md`.

## Operator troubleshooting

| Symptom | Check |
|---------|--------|
| No proactive messages | Runner phase via `/status`; suppression `reason_code` in audit tail |
| Duplicate replies | Two schedulers — stop Streamlit legacy scheduler or close Streamlit |
| WS offline, HTTP ok | Gateway reachable; firewall; restart `m16_api` |
| Runner lock / stale session | Stop runner via HTTP; verify session JSON not locked by another process |
| LLM errors | Same env as MVP (`OPENAI_*` / configured provider); gateway logs |
| `m16_legacy_scheduler_off` in audits | Expected when legacy scheduler is off; use runner + web/TUI |

Structured suppression codes are audit events only; clients display the string
and do not interpret semantics with keyword lists.

## Post-M16 stable seams

```text
Perception: M16 HTTP/WS + M14.2 event store
Orchestration: ConsciousnessRunner
Actuation: WS actuation events + M13.3 delivery
Legacy UI: Streamlit adapter only (I/O, not scheduler)
```

Future milestones (M17+) may add hosted deployment, auth, and persona admin UI
without moving cognition back into the frontend.
