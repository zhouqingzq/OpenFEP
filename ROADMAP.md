# Segmentum Product Roadmap

Segmentum now uses semantic product versions. Historical `M*` documents remain
as engineering evidence, but new work is scoped by release version.

## Version Policy

- Product releases use SemVer prereleases: `0.1.0-alpha.1`, `0.1.0-alpha.2`, and so on.
- Python distribution metadata uses the equivalent PEP 440 form, such as `0.1.0a1`.
- Wire protocols and Connector Contract versions evolve independently from the product version.
- Alpha releases may change internal state and APIs, but persisted-state migrations must be explicit.

## 0.1 Alpha: Usable Cognitive Runtime

Current baseline: `0.1.0-alpha.1`.
Known limitations: `KNOWN_LIMITATIONS.md`.

Included:

- persistent persona and dialogue state;
- direct and group conversation behavior;
- selective memory and privacy boundaries;
- proactive and background cognition foundations;
- HTTP/WebSocket gateway, web UI, TUI, and TypeScript client;
- Connector Contract `0.1`;
- Telegram reference adapter on the shared Connector Runtime.

Next `0.1.x` work:

- harden connector retries, delivery recovery, and observability;
- measure reply correctness, silence correctness, latency, and cost;
- remove remaining product-facing milestone terminology;
- validate the alpha with controlled real-user deployments.

## 0.2 Alpha: Multi-Platform

- ship at least one non-Telegram reference adapter;
- add connector conformance tests and capability negotiation;
- support platform-specific authentication and deployment configuration;
- provide common admin controls, consent, export, and deletion behavior.

## 0.3 Alpha: Hosted Operations

- multi-tenant isolation and access control;
- deployment packaging, backup, restore, and migration tooling;
- operational dashboards and incident-safe delivery queues;
- persona administration independent of legacy Streamlit.

## Release Gate

A release advances based on product behavior and operational evidence, not on
the count of internal capabilities. Each alpha should publish:

- supported platforms and connector contract version;
- behavior, latency, delivery, privacy, and cost results;
- state or API compatibility notes;
- known limitations and rollback instructions
  (see `KNOWN_LIMITATIONS.md` for the current alpha).
