# M6.0 Event And State Schema Draft

This draft defines MVP schemas for M6.1-M6.7. The schemas are intentionally
small, JSON-safe, and adapter-oriented. They do not replace the existing M5
runtime, diagnostics, memory, prompt, or policy systems.

## Field Contract Legend

Each field below specifies:

- Source: where the value comes from.
- Consumer: first planned reader.
- JSON safety: required representation.
- Prompt entry: whether the value may enter prompt text.
- Scope: persona/session scope.

## CognitiveEvent MVP

Typed pseudo-code:

```python
class CognitiveEvent:
    schema_version: int
    event_id: str
    event_type: str
    persona_id: str
    session_id: str
    turn_index: int
    cycle: int
    timestamp_utc: str
    source_module: str
    producer: str
    payload: dict[str, object]
    planned_consumers: list[str]
    prompt_eligible: bool
    sensitivity: str
```

Field contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | event adapter constant | event reader | integer | no | global |
| `event_id` | event adapter deterministic id or UUID | trace joiner | string | no | session |
| `event_type` | emission point, such as `observation`, `decision`, `prompt_capsule`, `generation`, `outcome`, `integration` | state reducer | string enum | no, except filtered labels | session |
| `persona_id` | resolved persona profile before artifact access | artifact writer and isolation guard | stable string | no | persona |
| `session_id` | runtime session context | artifact writer | string under persona | no | session |
| `turn_index` | `run_conversation` | trace writer | integer | no | session |
| `cycle` | `SegmentAgent.cycle` | trace writer and audits | integer | no | persona/session |
| `timestamp_utc` | event adapter clock | artifact writer | ISO-8601 string | no | session |
| `source_module` | emitter path, for example `segmentum/dialogue/observer.py` | audit tests | string | no | global |
| `producer` | emitter object/function name | audit tests | string | no | global |
| `payload` | bounded emitter-specific summary | state reducer | JSON primitives, arrays, objects; no object refs | only through filtered derived fields | session |
| `planned_consumers` | event registration table | event bus validation | list of strings | no | global |
| `prompt_eligible` | event type policy | prompt filter | boolean | no direct raw event entry | session |
| `sensitivity` | event type policy | prompt filter and artifact writer | `public`, `internal`, or `sensitive` | only `public` after filtering | persona/session |

Event type MVP:

- `observation`: produced after `DialogueObserver.observe`; consumed by state
  reducer and trace writer.
- `decision`: produced after `SegmentAgent.decision_cycle_from_dict`; consumed by
  path adapter and trace writer.
- `prompt_capsule`: produced after `build_fep_prompt_capsule`; consumed by prompt
  guidance audit and trace writer.
- `generation`: produced after `ResponseGenerator.generate`; consumed by trace
  writer and outcome joiner.
- `outcome`: produced after `classify_dialogue_outcome`; consumed by state reducer
  and outcome feedback audit.
- `integration`: produced after `SegmentAgent.integrate_outcome`; consumed by
  trace writer and memory/learning audit.

## CognitiveStateMVP

Typed pseudo-code:

```python
class CognitiveStateMVP:
    schema_version: int
    persona_id: str
    session_id: str
    turn_index: int
    observation_summary: dict[str, float]
    attention: dict[str, object]
    workspace: dict[str, object]
    affective: AffectiveStateMVP
    decision_summary: dict[str, object]
    metacognitive_summary: dict[str, object]
    previous_outcome: str
    uncertainty: dict[str, object]
    prompt_safe_summary: dict[str, object]
    source_event_ids: list[str]
```

Section contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | state reducer constant | trace reader | integer | no | global |
| `persona_id` | event identity fields | artifact writer | stable string | no | persona |
| `session_id` | event identity fields | artifact writer | string | no | session |
| `turn_index` | event identity fields | trace reader | integer | no | session |
| `observation_summary` | `DialogueObservation.channels` | affective reducer and trace | numeric map, rounded | only selected channel labels through prompt-safe summary | session |
| `attention` | `AttentionTrace` and `DecisionDiagnostics.attention_*` | conscious projection | bounded dict of selected/dropped channels and load | selected labels may enter | session |
| `workspace` | `GlobalWorkspaceState` and conscious report payload | conscious projection | bounded dict of accessible/carry-over channels | accessible channel names may enter | session |
| `affective` | `AffectiveStateMVP` section | meta-control and prompt guidance | bounded dict | only regulated labels, no raw narrative | session |
| `decision_summary` | `DecisionDiagnostics.chosen` and policy margins | path adapter and trace | bounded dict, no object refs | selected prompt-safe fields may enter | session |
| `metacognitive_summary` | `MetaCognitiveLayer` review payload | meta-control | bounded dict | prompt-safe guidance only | persona/session |
| `previous_outcome` | `classify_dialogue_outcome` owner | state reducer and prompt capsule | string enum | yes, normalized | session |
| `uncertainty` | FEP capsule margins and prediction error labels | meta-control | bounded dict | yes, labels only | session |
| `prompt_safe_summary` | prompt filter over state | `PromptBuilder` | strings, numbers, short lists | yes | session |
| `source_event_ids` | event reducer | trace audit | list of strings | no | session |

## AffectiveStateMVP

Typed pseudo-code:

```python
class AffectiveStateMVP:
    arousal: float
    valence: float
    regulation_need: str
    social_tension: float
    maintenance_pressure: float
    prior_outcome_bias: str
    evidence: dict[str, object]
```

Field contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `arousal` | conflict tension, stress/body channels, prediction error | meta-control | float 0..1 | no raw number unless filtered | session |
| `valence` | emotional tone and previous outcome | prompt guidance | float -1..1 | no raw number unless filtered | session |
| `regulation_need` | affective reducer | prompt guidance | string enum | yes | session |
| `social_tension` | observation `conflict_tension`, social signals | meta-control | float 0..1 | label only | session |
| `maintenance_pressure` | existing body/homeostasis and observation stress | meta-control | float 0..1 | label only | persona/session |
| `prior_outcome_bias` | previous outcome label | state reducer | string enum | yes, normalized | session |
| `evidence` | bounded source names and values | trace audit | short JSON dict | no | session |

`AffectiveStateMVP` is a lightweight maintenance signal. It is not a full emotion
simulator, a personality replacement, or an unbounded mood narrative.

## CognitivePath Read-only View

Typed pseudo-code:

```python
class CognitivePath:
    chosen_action: str
    ranked_options: list[dict[str, object]]
    policy_margin: float
    efe_margin: float
    dominant_components: list[str]
    source_diagnostics_id: str
```

Field contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `chosen_action` | `DecisionDiagnostics.chosen.choice` | conscious projection and trace | string | yes | session |
| `ranked_options` | `DecisionDiagnostics.ranked_options` | conscious projection and audit | list of bounded dicts | only top labels through capsule | session |
| `policy_margin` | `FEPPromptCapsule.policy_margin` or derived from ranked scores | meta-control | float | label only | session |
| `efe_margin` | `FEPPromptCapsule.efe_margin` | meta-control | float | label only | session |
| `dominant_components` | option policy components | conscious projection | list of strings | yes, if filtered | session |
| `source_diagnostics_id` | trace joiner | audit | string | no | session |

`CognitivePath` is an adapter over existing `ranked_options`. It is read-only and
must not re-rank, mutate, or override policy selection.

## MetaControlGuidance MVP

Typed pseudo-code:

```python
class MetaControlGuidance:
    guidance_id: str
    mode: str
    prompt_pressure: str
    caution_flags: list[str]
    suggested_style_constraints: list[str]
    forbidden_prompt_payloads: list[str]
    source_state_id: str
```

Field contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `guidance_id` | meta-control adapter | trace writer | string | no | session |
| `mode` | cognitive state and metacognitive summary | prompt guidance | string enum | yes | session |
| `prompt_pressure` | affective and uncertainty labels | `PromptBuilder` | string enum | yes | session |
| `caution_flags` | metacognitive review and boundary filters | prompt filter | list of strings | yes, after filtering | session |
| `suggested_style_constraints` | prompt guidance policy | `PromptBuilder` | short list of strings | yes | persona/session |
| `forbidden_prompt_payloads` | fixed boundary policy | prompt audit | list of strings | no direct prompt inclusion | global |
| `source_state_id` | state reducer | trace audit | string | no | session |

Meta-control first affects prompt conditioning. It does not replace core policy
selection.

## TurnTrace Required Fields

Typed pseudo-code:

```python
class TurnTrace:
    schema_version: int
    trace_id: str
    persona_id: str
    session_id: str
    turn_index: int
    cycle: int
    event_ids: list[str]
    observation_event_id: str
    decision_event_id: str
    generation_event_id: str
    outcome_event_id: str | None
    integration_event_id: str | None
    cognitive_state: dict[str, object]
    cognitive_path: dict[str, object]
    meta_control: dict[str, object]
    fep_prompt_capsule: dict[str, object]
    conscious_artifacts: dict[str, str]
    outcome: str | None
    tests: dict[str, object]
```

Field contracts:

| Field | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | trace writer constant | trace reader | integer | no | global |
| `trace_id` | trace writer | artifact writer | string | no | session |
| `persona_id` | resolved profile | isolation guard | stable string | no | persona |
| `session_id` | runtime session | artifact writer | string | no | session |
| `turn_index` | runtime turn | artifact writer | integer | no | session |
| `cycle` | `SegmentAgent.cycle` | audit | integer | no | persona/session |
| `event_ids` | event bus | audit | list of strings | no | session |
| `observation_event_id` | event bus | trace joiner | string | no | session |
| `decision_event_id` | event bus | trace joiner | string | no | session |
| `generation_event_id` | event bus | trace joiner | string | no | session |
| `outcome_event_id` | event bus | trace joiner | string or null | no | session |
| `integration_event_id` | event bus | trace joiner | string or null | no | session |
| `cognitive_state` | state reducer | conscious projection | JSON object | only `prompt_safe_summary` | session |
| `cognitive_path` | path adapter | conscious projection | JSON object | filtered summary only | session |
| `meta_control` | meta-control adapter | prompt guidance audit | JSON object | filtered guidance only | session |
| `fep_prompt_capsule` | existing capsule builder | prompt audit | JSON object | yes, existing bounded path | session |
| `conscious_artifacts` | artifact writer | audit | path map strings | no | persona/session |
| `outcome` | outcome owner | outcome feedback | string or null | yes, normalized | session |
| `tests` | test runner or audit harness | report generator | JSON object | no | global |

## Conscious.md Schema

`Conscious.md` is session-scoped and may be rewritten or rolled forward each
turn. It is a human-readable current/session context projection, not a decision
source.

Markdown sections:

- `# Conscious Context`
- `Identity`: persona id, display name, session id, schema version.
- `Current Turn`: turn index, current user-facing topic, bounded observation
  summary.
- `Consciously Accessible Now`: workspace accessible channels and carry-over
  labels.
- `Decision Path Summary`: selected action and short ranked-path explanation.
- `Affective Maintenance`: bounded regulation/tension labels.
- `Prompt Guidance`: prompt-safe constraints only.
- `Outcome Feedback`: previous outcome label and caveat that it is correlation
  evidence, not causal proof.
- `Evidence`: source `TurnTrace` id and timestamp.

Section contract:

| Section | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `Identity` | `profile.json` and session runtime | artifact reader | markdown text from JSON-safe ids | persona id no; display name yes | persona/session |
| `Current Turn` | `CognitiveStateMVP` | human reviewer | rendered from JSON-safe state | selected summary yes | session |
| `Consciously Accessible Now` | workspace state | human reviewer and optional prompt excerpt | rendered short list | yes, after filtering | session |
| `Decision Path Summary` | `CognitivePath` | human reviewer | rendered short text | yes, bounded | session |
| `Affective Maintenance` | `AffectiveStateMVP` | human reviewer and prompt guidance | rendered labels | yes, labels only | session |
| `Prompt Guidance` | `MetaControlGuidance` | human reviewer | rendered bounded guidance | yes | session |
| `Outcome Feedback` | outcome owner | human reviewer | rendered enum label | yes | session |
| `Evidence` | `TurnTrace` | audit | trace id and timestamp | no | session |

## Self-consciousness.md Schema And Update Policy

`Self-consciousness.md` is a long-term persona-scoped self-prior. It is
slow-moving, cross-session within one persona, and isolated from other personas.
It is not persisted memory, policy truth, diagnostics truth, or a prompt dump.

Markdown sections:

- `# Self-consciousness`
- `Persona Identity`: stable `persona_id`, display name history, schema version.
- `Stable Self-prior`: durable self-description distilled from accepted evidence.
- `Continuity Commitments`: bounded commitments used for consistency checks.
- `Interaction Tendencies`: slow traits and surface tendencies summarized from
  existing persona/profile systems.
- `Known Boundaries`: things the persona should not claim as lived biography
  without evidence.
- `Consolidation Evidence`: accepted session summaries and timestamps.
- `Update Log`: slow updates with reason, source session, and gate result.

Section contract:

| Section | Source | Consumer | JSON safety | Prompt entry | Scope |
| --- | --- | --- | --- | --- | --- |
| `Persona Identity` | `profile.json` | isolation guard | markdown from JSON-safe ids | display name only | persona |
| `Stable Self-prior` | slow consolidation gate | prompt-safe identity excerpt | bounded prose | yes, filtered | persona |
| `Continuity Commitments` | self model and accepted summaries | metacognitive review and prompt excerpt | bounded list | yes, filtered | persona |
| `Interaction Tendencies` | slow traits, surface profile, accepted summaries | prompt builder | bounded prose/list | yes, filtered | persona |
| `Known Boundaries` | artifact policy | prompt filter | bounded list | yes | persona |
| `Consolidation Evidence` | session summary acceptance | audit | ids, timestamps, hashes | no | persona |
| `Update Log` | slow consolidation writer | audit | bounded list | no | persona |

Update policy:

- Resolve `persona_id` before reading or writing any self-conscious artifact.
- Reject writes if the session path is not under the same `persona_id`.
- Consolidate from `turn_summaries` or session summaries only after a slow gate:
  minimum evidence count, no unresolved safety violation, no cross-persona source,
  and no direct contradiction with protected identity boundaries.
- Do not update every turn. Per-turn data belongs in `Conscious.md`,
  `conscious_trace.jsonl`, and `turn_summaries`.
- Store rejected update attempts in audit metadata, not in prompt text.
- Never use display name as the only directory identity.
