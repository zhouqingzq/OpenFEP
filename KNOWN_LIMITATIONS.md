# Known Limitations — `0.1.0-alpha.1`

Status: alpha freeze notes for the current product baseline.
Supported connector contract: `0.1`.
Supported external chat platform: Telegram (reference adapter only).

This page is the product-facing limitation list for release gate review.
It is not a research backlog.

## Intended Use

`0.1.0-alpha.1` is for controlled local or single-operator deployments:

- Path B runtime via M16 gateway + web UI / TUI;
- Telegram direct messages as the primary external surface;
- trusted operators who can inspect audit logs and restart the process.

It is **not** a multi-tenant hosted product and **not** a general public bot
platform.

## Group Chat Is Unstable

Direct-message behavior is the safer alpha surface.

Group chat has fixture-level acceptance for speaker separation, addressing,
memory privacy boundaries, and multi-party reply selection. Real-LLM group
runs still show high variance on:

- whether the assistant should reply or stay silent;
- addressee / targeting confidence;
- same-turn clarify vs no-reply decisions.

Treat group chat as **experimental**. Do not use public or high-stakes groups
as the first validation surface.

## Proactive Outreach Is Conservative

Background cognition and proactive delivery exist, but the product policy is
intentionally tight:

- silence, boredom, or repetition alone must not trigger outreach;
- outreach needs a traceable memory-backed expectation, bounded policy checks,
  and delivery gates;
- blocked delivery should leave structured `reason_code` audit evidence.

Expect missed reminders and delayed follow-ups more often than spam. That is
preferred for alpha. Mis-sends are still a residual risk and must be watched in
any real-user trial.

## Platform Scope

- Only Telegram is shipped as a reference external adapter on Connector
  Contract `0.1`.
- Discord, Slack, Feishu, WeCom, and other platforms are out of scope until
  `0.2`.
- Web and TUI talk to the local/gateway runtime; they are operator surfaces,
  not additional social platforms.

## No Multi-Tenant Hosted Operations

Missing from this alpha:

- tenant isolation and access control;
- hosted packaging, backup/restore, and migration tooling;
- operator dashboards and incident-safe multi-tenant delivery queues;
- persona administration independent of legacy Streamlit.

Legacy Streamlit is I/O-only after M16.4 and is not the production scheduler.
Do not enable `SEGMENTS_LEGACY_STREAMLIT_SCHEDULER` in real deployments.

## Measurement Gaps

The following release-gate numbers are not yet published as a stable baseline:

- reply correctness and silence correctness under real users;
- p50 / p95 latency and cost per turn;
- delivery success, duplicate suppression, and recovery after restart;
- privacy-incident count from live traffic.

Until those exist, advance only with controlled trials and explicit rollback.

## Privacy And Memory Caveats

Fixture tests cover DM-only facts not being reused publicly and group recall
privacy policy. Residual risks remain:

- long-context model drift can still surface unsupported detail;
- operator-visible diagnostics must not be copied into user-visible replies;
- export/deletion admin flows are not yet a product feature (`0.2` / `0.3`).

Any suspected cross-session or DM-to-group leakage is a stop-ship class issue
for further alpha expansion.

## Rollback

If a trial misbehaves:

1. stop the Telegram connector / gateway process;
2. disable proactive delivery surfaces for the affected persona/session;
3. restore persona/session state from the last known-good backup if persistence
   was corrupted;
4. keep audit logs and turn dumps for diagnosis; do not delete them to “clean”
   the incident.

There is no automated multi-tenant rollback in this alpha.

## What Comes Next

See `ROADMAP.md`. The immediate `0.1.x` bar is operational evidence from
controlled real-user deployments, not additional research milestones.
