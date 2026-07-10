# Prompt Directory Conventions

This directory is the historical archive for milestone work prompts.

Starting with product version `0.1.0-alpha.1`, new planning and releases use
semantic versions instead of new `M*` milestones. Existing milestone names,
schemas, tests, and reports remain unchanged where they are compatibility or
audit identifiers.

## Naming

Use:

```text
M{major}.{minor}_Work_Prompt.md
```

Examples:

```text
M6.0_Work_Prompt.md
M8.9_Work_Prompt.md
M10.0_Work_Prompt.md
```

For older milestones that historically used compact numbering, this directory now prefers expanded dotted milestone names. New or migrated implementation prompts should use `Work_Prompt`, not `Implementation_Prompt`.

## Current Roadmap Boundary

The post-M8 MVP architecture hardening path is:

```text
M8.9:  MVP Architecture Contract Hardening
M9.0:  Memory Dynamics Integration
M10.0: Self-Initiated Exploration Agenda
M11.0: User Generative Model And Value Memory Dynamics
M12.0: User Identity Continuity Model
M12.1: Mechanistic Personality Model And Plain-Language Report
M12.2: Bidirectional Free-Energy And Second-Order Role Cognition
M13.0: MVP-Local Behavioral Pull
M13.1: MVP-Local Boredom And Exploration Bias
M13.2: MVP-Local Affective Reward Proxy And Tolerance
M13.3: UI-Level Bounded Initiative
M13.4: UI Idle Tick And Introspection Entry Point
M13.5: Idle Cognitive Refresh Tick
M13.6: Memory-Backed Expected Free Energy Bridge
M14.0: Conscious Idle Reflector And Self-Cognition Patch
M14.1: Background Self-Continuity And Persistent Idle Loop
M14.2: Decoupled Environment Loop And Durable Overnight Outreach
M14.3: Traceable Proactive Delivery Alignment And Safety Diagnostics
M14.4: Streamlit Implicit Idle Proactive Delivery
M14.5: Production Proactive Policy Hardening
M14.6: Idle Plan And Structural Selector Alignment, Diagnose Observability
M14.7: Path B Memory Gate, Decay, And Precision-Weighted Recall
M15.0: Episode Ledger And Free-Energy Proxy Trajectory
M15.1: Consolidation And Forgetting Loop
M15.2: Meta-Control Intervention Layer
M15.3: Open-Item And Pending-Expectation Cleanup Meta-Control
M16.0: Consciousness Runner Architecture And Wire Protocol — **landed** (`reports/m16_0_consciousness_runner_contract.md`, `m16_protocol.py`, OpenAPI/JSON schemas).
M16.1: Python Consciousness Runner Service And Gateway — **landed** (`m16_runner.py`, `m16_api.py`, `m16_cli.py`, bridge/ws hub, acceptance tests). Streamlit remains legacy/non-acceptance for proactive scheduling.
M16.2: TypeScript Consciousness Client SDK — **landed** (`ui/packages/consciousness-client`, HTTP + WS + ajv validation, vitest).
M16.3: TypeScript Web Thin Frontend — **landed** (`ui/web`, Vite + consciousness-client, delivery ack + resync tests).
M16.4: TypeScript TUI Client And Streamlit Deprecation — **landed** (`ui/tui`, `m16_streamlit_legacy.py`, migration guide, Streamlit scheduler gated off by default).
M17.0: Prediction Lock And Confidence Contract
M17.1: Prediction Settlement And Deterministic Error
M17.2: Type Precision EMA And Memory Gate Integration
M17.3: Offline Calibration And Memory-Ablation Harness
M17.4: Bundle Evidence Aggregation And Anti-TopK Decision Policy
M17.5: Free Energy Surrogate Contract And Decomposition
M17.6: Verification-To-Memory Credit Assignment
M17.7: Surprise-Gated Reconsolidation
M17.8: Path-First Memory Substrate
M17.9: Local Field Potential And Non-TopK Recall
M17.10: Goal Priors And Adaptive Compute Control
M17.11: Closed-Loop Field Validation And Ablation Harness
M17.12: Path B Field Bridge And Unified Memory Recall Runtime
M18.0: Group Chat Readiness Contract And Acceptance Boundary
M18.1: Multi-Participant Identity Contract
M18.2: Addressee And Target-Resolution Graph
M18.3: Group Transcript Ownership And Durable Replay
M18.4: Group-Safe Memory, Privacy, And Cross-User Recall Boundaries
M18.5: Multi-Party Reply Policy, Turn-Taking, And Social Continuity
M18.6: Group Chat Acceptance Harness And Held-Out Validation
M18.7: Group Turn Semantic Attribution Contract — adds `addressee_hypothesis` + `reaction_attribution_hypothesis` bounded fields to the conscious-loop JSON (v2 attributes; the LLM is the only source per CLAUDE.md). Frozen state surface `state["m18_7_attribution_hypotheses"]` (≤ 8 entries, rolling window) is the M20.4 hand-off contract. Three bus events: `AddresseeHypothesisAdmitted` / `ReactionAttributionHypothesisAdmitted` / `AttributionHypothesisSkipped` (fast_chat + group_turn_binding + empty fields, the M20.3 §3.1a precedent). M18.7 alone records structured fields; the user-perceived same-turn capability requires M20.4 + M20.4.1. **landed**.
M18.7.1: Held-Out Calibration For LLM Confidence — runs a new semantic fixture (`tests/fixtures/m18_7_1_held_out_calibration.json`, 12 turns with hand-annotated ground truth) through the M18.7 conscious-loop pipeline; computes 10-bin reliability diagrams / ECE / Brier / accuracy; emits a frozen 5-value drift-signal enum; surfaces `state["m18_7_1_calibration"]` with `candidate_admit_min` / `candidate_tie_breaker_min` (decision authority remains with M20.4 — the frozen caveat string `"decision belongs to M20.4; M18.7.1 only surfaces"` is the binding contract). M18.7.1 does NOT mutate `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` / `M20_4_TIE_BREAKER_CONFIDENCE_MIN`; it only surfaces the data-driven recommendation. M18.7.1 is structural-only (pure functions, fixture shape, runner with FakeJSONLLM); real-LLM replay is an extension action, not a gate. Path A / M10 / `conversation_loop.py` remain untouched. **landed**. v2 (2026-06-09, `prompts/M18.7.1_Harness_V2_Design.md`): adds three scoring modes for the reaction field — `by_pid` (default, recommended; scores `reaction_to_participant_id` + `is_about_assistant_claim` joint correctness with pid normalization), `by_turn_id_resolved` (v1 turn_id scoring with runner-time placeholder resolution for `turn_<assistant_prior_turn_id>` etc.), and `by_turn_id_v1` (byte-identical v1). PID normalization collapses `assistant` / `hutao` / `clawdgroupchat_bot` to canonical `bot`; runner default is `by_turn_id_v1` (v1 back-compat); CLI default is `by_pid` (Q1). Mode C is byte-identical to v1 to honor D6 (no silent report shift).
M18.7.2: M18.7 Minimal-Prompt Call Site — adds a dedicated `~1.5-2.0k-char` minimal-prompt LLM call site (`build_m18_7_minimal_prompt` in `m18_7_attribution.py`, stage string `"m18_7_2_minimal"`, registered in `_AUXILIARY_LLM_STAGES` for a 12s timeout / 0 retries) that runs in `run_turn` *between* the M18.5 reply-policy decision and the conscious loop, decoupled from the conscious loop. The conscious-loop's M18.7 v2 attrs segment is removed (`build_conscious_loop_prompt` no longer requests `addressee_hypothesis` / `reaction_attribution_hypothesis`); M18.7.2 is the sole source. The minimal fill writes to the **same** `state["m18_7_attribution_hypotheses"]` surface via a new orchestrator `emit_m18_7_2_attribution_for_turn` that reuses `normalize_*` / `build_state_entry` / `record_m18_7_attribution_hypotheses` unchanged and stamps `source: "m18_7_2_minimal"` on every entry. M20.4 producer, M20.4.1 same-turn gate, and M18.7.1 calibration runner all read the same surface and "just work" once M18.7.2 populates it (the previously dead path comes alive end-to-end). Three new bus events: `M18_7_2_AddresseeHypothesisAdmitted` / `M18_7_2_ReactionAttributionHypothesisAdmitted` / `M18_7_2_MinimalDegraded` (the degraded event is emitted on LLM-call failure with a `M12-pre`-style empty-`{}` fallback; `run_turn` does NOT crash on M18.7.2 failures). Acceptance: real-LLM replay of the M18.7.1 held-out fixture must surface `m18_7_attribution_hypotheses` length > 0, addressee / reaction `n_present > 0`, and ECE / Brier values that are real (not the prior degenerate 0.0/0.0/0.0). New tests in `tests/test_m18_7_2_minimal_attribution.py` (23 tests: pure-function prompt builder, orchestrator, bus event shapes, integration with FakeJSONLLM, regression for `source` field, regression for degraded fallback). Path A / M10 / `conversation_loop.py` / M20.4 threshold constants all remain untouched. **landed**.
M19.0: Self-Expectation Formation, Indirect Observation, And Fast Mismatch Memory
M19.1: Traceable Self-Repair Expectations And Free-Energy Guidance Bridge
M19.2: Natural-Scene Settlement, Prediction-Error Writeback, And Shadow Validation
M19.3: Slow Self-Cognition Consolidation, Downgrade, And Revocation
M20.0: ActiveCommitment Meta-Contract, Registry, And Observable Vocabulary — schema-only; freezes owner map and observable enum v1; no settler, no promotion.
M20.1: Settler Protocol And Observable Settlement Runtime — adds deterministic / llm_judge / hybrid / silent settlers; emits ActiveCommitmentSettled and NoSettlementMade; writes only to owner.observability.
M20.2: Graded Correction, Revocation, And Slow Promotion Runtime — routes (outcome, magnitude) through frozen levels (microadjust / next_turn / same_turn / slow_promote / revoke / expire) to existing owner write paths; no new long-term state, no duplication of M19.x / M9.0 / M17.4 / M15.1 logic.
M20.3: Policy Admission, Same-Turn Surface Horizon, Runtime Loop Invariants, And Settler Migration — adds PolicyProducer (non-LLM policy admission), `runtime_mode_state` owner (registry v2), `horizon` attribute (same_turn_surface fast settler with pre-send block for runtime_mode_state only), `LoopInvariants` minimum loop coverage (no fast_chat skip), and M20.1.1 migration of M17.1 / M19.0 / M13.2 / M15.0 / M18.5 onto the M20.1 runtime. 6+1+1 master acceptance fixture. Three internal Tiers (A / B / C), all required.
```

M8.9 is a bridge milestone. It does not replace the original roadmap; it locks state ownership, memory write intent, and generation evidence boundaries before M9-M12 expand the system.

M11.0 was repurposed from the earlier "Conscious Projection Runtime" scope to the user-modeling track. M12 is split into M12.0 (identity continuity, alias claims, impersonation strangeness, entity binding), M12.1 (mechanistic personality model with the eight-step plain-language report), and M12.2 (reciprocal role cognition). M12.1 and M12.2 read upstream state read-only and never write back into M11 or M12.0 ledgers.

M13 is an MVP-local drive layer for the real UI chat path. It intentionally does not wait for full Path A / Path B unification. The sequence is M13.0 behavioral pull, M13.1 boredom and exploration bias, M13.2 affective reward proxy and tolerance, M13.3 bounded UI initiative, and M13.4 the UI idle tick plus introspection entry point.

M13.6 adds a memory-backed expected-free-energy bridge between traceable pending expectations/open items and bounded Path B policy guidance. Silence is not a direct boredom or outreach input: it only matters when it leaves a concrete memory-backed expectation due or unresolved. M14.0 builds on the M13.4 plumbing to add the conscious idle reflector: idle ticks can trigger a JSON-only conscious plan that proposes self-cognition patches, memory consolidation intents, and open-item updates, and may hand off to the existing M13.3 outreach pipeline only when M13.6 allows outreach. M14.1 makes that loop budgeted and persistent. M14.2 separates environment observation, event storage, self-loop preparation, and delivery; UI renders and reports runtime state, it is not the lifecycle owner for overnight self-continuity. Explicit later-message requests must become durable scheduled intents plus outbox entries, never just natural-language follow-up notes. M14.3 aligns proactive delivery around traceable expectations with evidence refs; vague open-item `next_check` tokens alone must not trigger M13.3 proposals, and blocked proactive delivery must emit structured `reason_code` audit events. M14.4 wires Streamlit reruns to `evaluate_proactive_initiative(implicit_idle_request=True)` so an open chat page can deliver a proactive bubble after silence without manual continue; it adds `proactive_policy_profile=streamlit_open_chat` to relax session/cooldown caps for demos only. M14.5 restores conservative defaults for production. M13.5 adds the silence-period idle cognitive refresh tick (recall, memory EFE, M13 band re-eval, target selection). M14.6 aligns idle reflector plans with `select_proactive_target` and fixes diagnose/UI observability. M14.7 completes Path B memory gate, decay, and precision-weighted recall (M9 series ends at M9.0), but its scored rows remain a coarse retrieval surface rather than proof of multi-memory conjunctive behavior. M15.0 adds the episode ledger and free-energy proxy trajectory. M15.1 adds consolidation and forgetting. M15.2 adds meta-control intervention intents (demo applies on `streamlit_open_chat`; production audit-only by default). M15.3 adds bounded cleanup meta-control for stale open items and pending expectations, plus strict traceability alignment in M13.6, idle bound-memory recall seeding, and active-session-only queued outreach defaults. M16 inverts UI control: a Python consciousness runner owns the self-loop; FastAPI/WebSocket expose perception and actuation only; TypeScript Web and TUI clients subscribe and publish input without scheduling cognition. After M16.4, Streamlit is legacy I/O only and must not drive idle or proactive ticks by default. M17.0-M17.4 close the verifiable prediction-error memory loop on the Path B dialogue side: lock numeric predictions before response, settle them by `prediction_id`, learn prediction-type precision, feed measured error/confirmation/novelty into memory dynamics, verify the policy with offline calibration plus ablation replay, and then require bundle-level evidence aggregation plus best-single counterfactual baselines before claiming any behavior that simple top-k item ranking could not have produced. M17.5-M17.10 start the core-runtime memory-field track: first make the free-energy surrogate contract explicit, then link verification back into memory credit, gate reconsolidation by reuse surprise, promote path objects above raw episodes, build bounded local field summaries beyond fixed top-k, and finally let active goals shape priors while adaptive compute scales effort with uncertainty and conflict. M17.11 validates that loop honestly on held-out fixtures. M17.12 is the bridge milestone that makes Path B consume the same path/field substrate as the core runtime so the two half-loops become one demo path. Path A and M10 are explicitly out of scope.
M19 starts the Path B self-expectation correction track. M19.0 forms bounded
`self_response_expectation` objects from conscious-loop / idle-reflector
structured output, observes indirect outcome mismatch, and accumulates fast
revocable `mismatch_memory_fast` without writing `self_cognition`. M19.1 turns
stable mismatches into falsifiable `self_repair_expectation` objects and bridges
them into `repair_bias`, `behavioral_pull`, and M13 traction proposals. M19.2
settles those expectations only in naturally recurring contexts and writes
`prediction_error_delta` back into fast memory and traction proposals. M19.3
slowly promotes repeated, naturally confirmed repair patterns into bounded
`self_cognition.calibrated_tendencies` / `repair_priors` with downgrade and
revocation. User correction and `reply_validation` remain auxiliary evidence
only; the primary signal is internal self-expectation mismatch.

M18 starts the group-chat readiness track. M18.0 freezes the acceptance boundary so "speaker switching" cannot be mistaken for true multi-party capability, and it also freezes the minimal structured group-turn input assumptions plus the bounded operating envelope. This is a readiness track, not a full "excellent group performance" track. M18.1 makes participant identity first-class across runtime, transcript references, and memory writes, while separating identity ownership from transcript persistence. M18.2 replaces single-target binding with a bounded addressee/target graph that consumes reply-to, mention, quote, and recent ownership structure when available. M18.3 makes transcript ownership durable and replay-safe, including compatibility migration from legacy single-user rows. M18.4 hardens cross-user memory/privacy boundaries for group conditions with an explicit source-audience-disclosure policy matrix. M18.5 adds deterministic multi-party reply policy, turn-taking precedence, bounded intentional non-reply, and unresolved-thread continuity. M18.6 provides end-to-end scripted and held-out validation before any claim that Hu Tao is truly ready for group chat, and requires pass criteria to distinguish structured assertions from rubric-based judgments.

M20 starts the Path B unified-commitment meta-contract track. M20.0 is
schema-only: it freezes `ActiveCommitment`, `CommitmentRegistry` v1
(owner map for `policy_state` / `m13_drive_state` / `m15_episode_ledger` /
`mismatch_memory_fast` / `self_repair_expectation` /
`self_cognition_calibrated_tendencies` / `user_prediction_ledger` /
`memory_dynamics_control_guidance` / `outreach_intent_registry` /
`group_addressee_graph`), and `Observable` vocabulary v1
(`expectation_outcome_match` / `prediction_error_band` / `repair_bias_band` /
`behavioral_pull_shift` / `mismatch_type_band` / `pacing_match` /
`identity_voice_match` / `boundary_handled` / `initiative_timing_match` /
`silent_then_resolved` / `traction_delta_band`). M20.0 does not implement
settlers, promotion, or revocation; it only validates admissions and emits
`ActiveCommitmentCreated` / `ActiveCommitmentRejected` audit events. M20.1
builds the settler protocol (deterministic / llm_judge / hybrid / silent)
against the frozen registry, with a single `Settler` interface and bounded
reference implementations (e.g. `IdentityVoiceMatchLLMJudgeSettler` wraps
the existing M19.x surface-consistency call; `BehavioralPullShiftSilentSettler`
is a `silent` reference). M20.1 writes only to `owner.observability[commit_id]`,
a new additive diagnostic surface. M20.2 builds the graded correction and
slow-promotion runtime: a pure dispatcher maps `(outcome, magnitude,
source_kind, owner_id)` to one of six frozen levels (microadjust / next_turn /
same_turn / slow_promote / revoke / expire) and routes to existing owner
write paths (M19.1 traction, M19.3 slow promotion, M9.0 control_guidance,
M17.4 precision EMA, M15.1 episode aggregation). M20.2 introduces no new
long-term state bucket and does not duplicate any existing owner mutation
logic. `same_turn` corrections are advisory only; visible reply text still
flows through the conscious loop, thinking, validation, and the M19.x
surface-consistency check. Later M17 / M19 milestones consume the registry
through bounded adapter producers; they do not refactor M19.0/19.1 storage
shapes. M20.1.1 (in M20.3) migrates the existing per-loop settlers (M17.1,
M19.0 outcome settlement, M13.2 band check, M15.0 episode aggregation,
M18.5 boundary judgment) onto the new runtime.

M20.3 is the operational layer that closes the five gaps the
M20.0–M20.2 review identified. It (1) adds a non-LLM `PolicyProducer` that
admits `ActiveCommitment` rows with `source_kind = "policy"` from the
command surface (turn-scoped `/status` vs durable `/mode` / `/persona` /
group ingress), runtime mode flags, and a new bounded
`correcting_assistant_identity` conscious-loop field; (2) bumps
`CommitmentRegistry` to v2 (additive over v1) with a new writable
`runtime_mode_state` owner (distinct from the read-only `policy_state`)
that has `accepts_same_turn_block = true` and `accepts_policy_correction
= true`; (3) adds a v2 `horizon` attribute on `ActiveCommitment` and a
same-turn-surface settler fast path whose pre-send gate can BLOCK only
when the violator is `runtime_mode_state`, with a post-send advisory
path that writes only to next-turn `control_guidance`; (4) adds a
runtime invariant module `LoopInvariants.enforce_minimum_loop_coverage`
with a hard rule matrix (rule A: policy ≥ 1 per external turn; rule B:
`runtime_mode_state` ≥ 1 when `surface_intent ∈ {chat, bot}`) that emits
`MinimumLoopCoverageMissed` audit events on every external turn
(including fast_chat) so the meta-loop cannot be skipped to zero; and
(5) migrates the 5 existing per-loop settlers (M17.1, M19.0, M13.2,
M15.0, M18.5) onto the M20.1 runtime via thin adapters that must agree
with the existing owner audit events. M20.3 also adds the M20.2 §2
special case **registry v2 exception table** so policy-source corrections
routed to `runtime_mode_state` are not expired by the frozen
"policy → expire" rule. M20.3 is structured as a single milestone with
three internal Tiers (A: PolicyProducer + mode owner + same_turn_surface
+ LoopInvariants + 6+1+1 sub-scenarios; B: M17.1 / M19.0 / M18.5
migration; C: M13.2 / M15.0 thin adapters + outcome-agreement tests);
all three Tiers are required for M20.3 acceptance. The 6+1+1 master
acceptance fixture is the M20-series total acceptance: base 6 steps
(eventual consistency) plus an identity sub-scenario (strict timing,
predict → observe → block) and a fast_chat sub-scenario (strict
invariants, no `MinimumLoopCoverageMissed`). M20.3 closes the loop on
the engineering side; the conscious loop, thinking, reply validation,
and M19.x surface-consistency check all still run unchanged.

## Migration Note

Root-level `M2.*_Implementation_Prompt.md` and `M6.*_Implementation_Prompt.md` files were moved into this directory and renamed to `M*_Work_Prompt.md` so milestone prompts have one canonical location.

Legacy lowercase M3/M4 prompt files were also renamed to expanded dotted names, for example:

```text
m410_work_prompt.md -> M4.10_Work_Prompt.md
m411_acceptance_criteria.md -> M4.11_Acceptance_Criteria.md
```

When two historical files already covered the same milestone, the older compact prompt was kept with `Legacy` in the filename rather than overwritten.
