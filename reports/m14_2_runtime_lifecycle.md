# M14.2 Runtime Lifecycle Contract

## Responsibility Boundary

M14.2 keeps four responsibilities separate.

Environment loop: Streamlit, CLI, clock, and future local adapters append facts about the outside runtime. Examples are user message committed, UI ping, clock wake, session close, runner start/stop, and delivery surface available. This layer never calls an LLM, mutates MVP cognition state, or writes visible assistant text.

Durable event store / message bus: `environment_events.jsonl` and `environment_event_claims.jsonl` provide append, claim, ack, fail, and query. Claims have leases, so a stopped daemon does not permanently lose work. The bus is coordination storage only, not an interpretation layer.

MVP self loop daemon: `python -m segmentum.dialogue.runtime.m14_2_self_loop --persona <id> --session <id>` is the primary overnight path. It claims events, extracts durable scheduled outreach intents from explicit user requests, checks due intents, uses existing M14 reflection helpers and M13 proposal construction, updates MVP state through existing store locks, and creates pending outbox entries. It does not write user-visible chat text.

Delivery / outbox loop: `queued_outreach.jsonl` is the canonical local outbox. Streamlit may announce `OutboxDeliverySurfaceAvailableEvent` and drain one pending entry, but the drain calls M13.3 `evaluate_proactive_initiative` and `run_proactive_turn`; it does not bypass delivery assessment, safety suppression, cooldowns, or reply validation.

## Why Inline Streamlit Is Not Acceptance

Streamlit reruns are UI lifecycle events. A browser tab can close, pause, or rerun without representing a real clock wake. Overnight behavior is accepted only through the standalone daemon, because the daemon can run while Streamlit is closed, can recover claimed-but-unacked events by lease, and owns periodic clock wake processing. Enabling background self-continuity in the UI sets opt-in budgets and records `runner_kind=standalone_daemon`; it does not start the inline dev fallback runner automatically.

## Event Flow

User message: `ChatInterface` commits the turn through `MVPDialogueRuntime`, then appends `UserMessageCommittedEvent`. The daemon claims that event, runs deterministic scheduled-intent extraction, writes `scheduled_intents.jsonl`, and creates or refreshes a linked `open_items` entry with `type="scheduled_outreach"`.

UI ping: Streamlit appends `UIPingEvent` and updates legacy M14.1 heartbeat fields for status. The ping is not a cognition tick.

Clock wake: the daemon appends `ClockWakeEvent`, claims it, queries due scheduled intents, and appends `ScheduledIntentDueEvent` markers for traceability.

Due intent: the daemon marks the intent `preparing`, runs the existing M14.0 idle introspection turn under M14.1 background budgets (memory recall, conscious idle plan, named-owner patches), then appends exactly one canonical outbox entry keyed by `source_intent_id`. It marks the intent `prepared`.

Queued outreach: Streamlit or another delivery surface appends `OutboxDeliverySurfaceAvailableEvent`, then drains at most one pending outbox row through M13.3. Successful delivery marks the outbox `delivered`, the scheduled intent `delivered`, and the linked open item `closed`.

## Durable Files

Append-only:

- `environment_events.jsonl`
- `environment_event_claims.jsonl`
- `scheduled_intents.jsonl`
- `conversation_log.jsonl`

Canonical outbox:

- `queued_outreach.jsonl`

State files written under the MVP store lock:

- `open_items.json`
- `m13_drive_state.json`
- other existing MVP state files only through pre-existing owners or runtime paths

M14.2 does not write M11 or M12 ledgers, Path A state, M10 state, or `self_basic_facts.json`.

## Crash And Restart Behavior

Crash after event append before claim: event remains `pending`.

Crash after claim before ack: the claim lease expires and the next daemon can reclaim it.

Crash after scheduled intent creation before open item update: intent creation is idempotent by `source_event_id`; the daemon refreshes the linked open item on replay.

Crash after intent marked preparing before outbox append: due-intent recovery treats `preparing` as due work and retries preparation.

Crash after outbox append before intent marked prepared: outbox creation is idempotent by `source_intent_id`; replay reuses the existing row and marks the intent prepared.

Crash after delivery assessor approval before status update: delivery is keyed by `proposal_id`; replay sees the outbox status and linked intent status before considering another pending row.

## Invariants Against A Second Generation Path

- The event bus has no LLM dependency and no MVP state write API.
- Scheduled intent extraction is deterministic for explicit English and Chinese local-language requests; ambiguous LLM classification is not enabled in this MVP implementation.
- The self-loop daemon creates outbox proposals only. It never writes visible assistant chat text.
- The canonical outbox requires `delivery_policy.require_m13_3_assessor=true`.
- Delivery calls the existing M13.3 locked proposal path and `run_proactive_turn`.
- Engineering logs use `engineering_proxy_label="mvp_local_decoupled_self_loop"` and operational wording.
