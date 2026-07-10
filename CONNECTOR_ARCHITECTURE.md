# Connector Architecture

Connector Contract `0.1` separates Segmentum cognition from external chat
platforms. Telegram is the first adapter, but no shared runtime behavior should
depend on Telegram concepts.

Canonical adapter imports live under `segmentum.connectors`, for example
`segmentum.connectors.telegram`. Historical module paths remain available only
for compatibility.

## Boundary

Every adapter implements:

```python
class ConnectorAdapter(Protocol):
    platform: str
    persona_id: str
    account_scope: str
    target_store_file: str
    capabilities: ConnectorCapabilities

    def normalize_event(self, event) -> NormalizedConnectorInput | None: ...
    def target_from_payload(self, payload) -> ConnectorDeliveryTarget | None: ...
    def deliver(self, *, target, text) -> ConnectorDeliveryReceipt: ...
```

The adapter owns:

- platform authentication and API calls;
- raw webhook, polling, or event parsing;
- platform identity, mention, reply, thread, and channel semantics;
- reconstruction of a persisted platform delivery target;
- final outbound message delivery.

The shared `ConnectorRuntime` owns:

- routing normalized input into persona/session state;
- invoking the shared dialogue runner;
- persisting delivery targets before cognition runs;
- suppressing duplicate delivery after a target is marked delivered;
- returning bounded platform-neutral processing results.

## Identity Rules

Use stable namespaced identifiers:

```text
session:     <platform>:<account_scope>:<surface_kind>:<surface_id>
participant: <platform>:<account_scope>:<participant_kind>:<participant_id>
```

Platform adapters must preserve stable IDs across restarts. Display names are
not identity keys.

## Capability Declaration

Adapters declare support for direct messages, groups, threads, explicit
mentions, reply links, edits, and proactive delivery. Shared behavior must use
these declarations instead of assuming every platform behaves like Telegram.

## Adding A Platform

1. Implement the Connector Contract in a platform adapter.
2. Normalize platform events into bounded `NormalizedConnectorInput` values.
3. Reuse `ConnectorRuntime`; do not copy its ingestion or delivery loop.
4. Add connector conformance tests that use a fake platform API.
5. Add platform-specific integration tests for identity, mentions, replies,
   threads, delivery idempotency, and restart continuity.
