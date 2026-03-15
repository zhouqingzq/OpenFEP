# M2.5 Milestone Spec

## Title

M2.5 Narrative Experience Ingestion and Prior Reshaping

## Summary

M2.5 extends OpenFEP from a purely toy-world numeric survival agent into an agent that can ingest external narrative experience logs, compile them into structured appraisals, and feed them into the existing Free Energy Principle loop:

`narrative log -> event compilation -> appraisal latents -> surprise/value evaluation -> episodic storage -> sleep consolidation -> prior update`

This milestone does not aim to generate a historically faithful personality from a biography. Its purpose is narrower and testable: make the agent meaningfully plastic to structured lived experience instead of only to manually supplied observation vectors.

## Why This Milestone Exists

The current M2 runtime is strong enough for:

- top-down prediction
- bottom-up error propagation
- episodic storage
- sleep consolidation
- slow-weight updates

But it is bottlenecked by the six low-level observation channels:

- `food`
- `danger`
- `novelty`
- `shelter`
- `temperature`
- `social`

Those channels are sufficient for a survival toy world, but they are too coarse to carry:

- witnessed trauma
- betrayal
- moral shock
- controllability loss
- worldview shifts
- stable interpretation style

M2.5 introduces a structured middle layer rather than replacing the current base world model.

## Scope

### In Scope

- External injection of narrative logs as first-class inputs.
- Deterministic compilation of narrative logs into structured event records.
- Mapping of event records into appraisal latents.
- Conversion of appraisal latents into embodied episodes that can enter the existing memory and sleep loop.
- Sleep-phase write-back into new slow-changing priors beyond action bias alone.
- Full traceability from raw narrative to resulting prior update.

### Out of Scope

- Full open-domain personality cloning from arbitrary long biographies.
- Direct end-to-end LLM generation of priors without structured intermediate representation.
- Replacing the existing six observation channels.
- Building a full literary or sociological theory engine.
- Unbounded latent-space expansion.

## Product Goal

After M2.5, a blank-slate agent should be able to ingest repeated narrative experiences such as:

- finding food
- surviving a predator near-miss
- witnessing another person's fatal mistake

and show measurable changes in:

- risk interpretation
- trust calibration
- controllability priors
- action ranking
- identity narrative summary

The output should still be auditable and deterministic under fixed seed plus fixed compiler settings.

## Non-Goals

M2.5 should not claim:

- "this is already a full digital person"
- "this reproduces Lu Xun from biography"
- "this system understands all narratives"

The correct claim is:

"The agent can now compile narrative experience into structured appraisals that reshape its priors through the existing memory and sleep mechanisms."

## Architecture

### Existing M2 Base To Preserve

- `segmentum/agent.py`
- `segmentum/memory.py`
- `segmentum/sleep_consolidator.py`
- `segmentum/self_model.py`
- `segmentum/world_model.py`

These remain the downstream learning and consolidation substrate.

### New M2.5 Layer

Add a narrative ingestion stack ahead of episodic storage:

1. `NarrativeEpisode`
2. `CompiledNarrativeEvent`
3. `AppraisalVector`
4. `EmbodiedNarrativeEpisode`
5. `NarrativeCompiler`
6. `NarrativeIngestionService`

## Data Model

### 1. `NarrativeEpisode`

Purpose: raw external experience log.

Suggested fields:

```python
@dataclass
class NarrativeEpisode:
    episode_id: str
    timestamp: int
    source: str
    raw_text: str
    tags: list[str]
    metadata: dict[str, object]
```

Notes:

- `source` distinguishes `user_diary`, `historical_biography`, `simulated_log`, etc.
- `raw_text` must be preserved for audit and later compiler improvements.

### 2. `CompiledNarrativeEvent`

Purpose: event-normalized structure extracted from text.

Suggested fields:

```python
@dataclass
class CompiledNarrativeEvent:
    event_id: str
    timestamp: int
    setting: str
    actors: list[str]
    subject_role: str
    event_type: str
    outcome_type: str
    self_involvement: float
    witnessed: bool
    direct_harm: bool
    controllability_hint: float
    annotations: dict[str, object]
    source_episode_id: str
```
```

Examples of `event_type`:

- `resource_gain`
- `predator_attack`
- `witnessed_death`
- `social_betrayal`
- `public_humiliation`
- `institutional_rejection`

### 3. `AppraisalVector`

Purpose: the middle layer between narrative semantics and FEP-compatible learning.

Suggested fields:

```python
@dataclass
class AppraisalVector:
    physical_threat: float
    social_threat: float
    uncertainty: float
    controllability: float
    novelty: float
    loss: float
    moral_salience: float
    contamination: float
    attachment_signal: float
    trust_impact: float
    self_efficacy_impact: float
    meaning_violation: float
```
```

Constraints:

- all values normalized to `[-1, 1]` or `[0, 1]`; choose one convention and enforce it globally
- compiler must emit confidence and provenance for every populated dimension

### 4. `EmbodiedNarrativeEpisode`

Purpose: structured event payload that the existing memory and sleep pipeline can consume.

Suggested fields:

```python
@dataclass
class EmbodiedNarrativeEpisode:
    episode_id: str
    timestamp: int
    observation: dict[str, float]
    appraisal: dict[str, float]
    body_state: dict[str, float]
    predicted_outcome: str
    value_tags: list[str]
    narrative_tags: list[str]
    compiler_confidence: float
    provenance: dict[str, object]
```
```

Notes:

- `observation` may still map into the current six channels for compatibility.
- `appraisal` carries the higher-level meaning not representable in the six channels.
- `provenance` must contain source ids and compiler method details.

## Module Responsibilities

### `segmentum/narrative_types.py`

Add canonical dataclasses for:

- `NarrativeEpisode`
- `CompiledNarrativeEvent`
- `AppraisalVector`
- `EmbodiedNarrativeEpisode`

### `segmentum/narrative_compiler.py`

Primary responsibilities:

- parse raw narrative text into normalized event structures
- derive appraisal vectors
- emit compiler confidence and provenance

Required design:

- rule-based core path must exist
- optional LLM-assisted path may enrich but not replace the core path
- deterministic mode required for tests

Suggested API:

```python
class NarrativeCompiler:
    def compile_episode(
        self,
        episode: NarrativeEpisode,
    ) -> EmbodiedNarrativeEpisode:
        ...
```

### `segmentum/narrative_ingestion.py`

Primary responsibilities:

- accept one or more `NarrativeEpisode` values
- call compiler
- hand off embodied episodes to agent memory integration

Suggested API:

```python
class NarrativeIngestionService:
    def ingest(
        self,
        *,
        agent: SegmentAgent,
        episodes: list[NarrativeEpisode],
    ) -> list[dict[str, object]]:
        ...
```

Return value should include an auditable result per ingested episode:

- compilation summary
- surprise/value metrics
- episode stored or rejected
- sleep-side updates if sleep is triggered

## Compiler Semantics

### Rule-Based First

M2.5 should ship with a minimal deterministic compiler. It does not need broad language understanding. It only needs a controlled ontology that can robustly handle seeded scenarios.

Minimum supported patterns:

- finding food or resources
- direct attack / near miss
- witnessed injury or death
- toxic contamination
- social exclusion
- help / protection / rescue

### Appraisal Mapping Rules

The compiler must not directly set only the six base observation values. It must first produce appraisals.

Examples:

`"agent出门找到了一些吃的"`

- `physical_threat`: low
- `uncertainty`: low
- `loss`: low
- `novelty`: low to medium
- `self_efficacy_impact`: positive

`"agent昨天路过河边，被一只鳄鱼攻击了，没有受伤"`

- `physical_threat`: high
- `uncertainty`: high
- `controllability`: low
- `loss`: low
- `self_efficacy_impact`: slightly negative or ambiguous

`"agent看到一个人吃了毒蘑菇死去了"`

- `physical_threat`: medium
- `contamination`: high
- `moral_salience`: medium
- `uncertainty`: medium to high
- `trust_impact`: context-dependent
- `meaning_violation`: medium

### Embodiment Mapping

M2.5 must preserve compatibility with the current architecture by deriving a compatibility `observation` payload from appraisal latents.

Example compatibility mapping:

- `danger <- max(physical_threat, social_threat * 0.6, contamination * 0.5)`
- `novelty <- uncertainty + novelty`
- `social <- baseline + attachment_signal + trust_impact`

This mapping is intentionally lossy. The non-lossy representation must remain in `appraisal`.

## Changes To Existing Learning Loop

### Agent

Add a narrative ingestion path without breaking the current `decision_cycle(observation)` path.

Required new method on `SegmentAgent`:

```python
def ingest_narrative_episode(
    self,
    embodied_episode: EmbodiedNarrativeEpisode,
) -> dict[str, object]:
    ...
```

Responsibilities:

- compute prediction against current beliefs
- compute surprise / value / risk
- build a memory-storable state snapshot
- store as an episodic memory candidate
- update short-term working memory as needed

### Memory

Existing episodic memory already expects:

- observation
- prediction
- errors
- body_state
- action
- outcome

M2.5 should extend stored episode payloads with:

- `appraisal`
- `narrative_tags`
- `compiler_confidence`
- `source_episode_id`
- `source_type`

### Sleep Consolidation

Sleep must consolidate not only action-outcome regularities but also narrative appraisal regularities.

New target slow weights:

- `trust_prior`
- `controllability_prior`
- `trauma_bias`
- `contamination_sensitivity`
- `meaning_stability`

These should live in either:

- `self_model.belief_calibration`
- or a new dedicated prior store inside `SelfModel`

Recommendation: use a dedicated structure rather than overloading arbitrary dicts.

Example:

```python
@dataclass
class NarrativePriors:
    trust_prior: float = 0.0
    controllability_prior: float = 0.0
    trauma_bias: float = 0.0
    contamination_sensitivity: float = 0.0
    meaning_stability: float = 0.0
```
```

## Traceability Requirements

Every ingested narrative must leave a complete trail in trace output:

- raw narrative episode id
- compiled event fields
- appraisal vector
- compatibility observation mapping
- prediction before ingestion
- computed prediction error / surprise / value
- episode creation decision
- sleep rule extraction using narrative fields
- prior deltas after sleep

This is mandatory. Without it, the system becomes un-auditable.

## Acceptance Scenarios

### Scenario A: Resource Gain

Input narrative:

"第一天，agent出门找到了一些吃的。"

Expected effects:

- low surprise after repeated similar events
- positive value mapping
- mild increase in efficacy or safety-related priors

### Scenario B: Predator Near-Miss

Input narrative:

"第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。"

Expected effects:

- high surprise on early exposure
- elevated threat-related appraisal
- repeated exposures increase avoidance bias and trauma-related priors
- later action ranking should shift toward caution in similar contexts

### Scenario C: Witnessed Fatality

Input narrative:

"第三天，agent看到一个人吃了毒蘑菇死去了。"

Expected effects:

- non-zero surprise even without direct self-harm
- contamination-related and uncertainty-related appraisal increases
- sleep should consolidate a rule that future ambiguous food cues are evaluated more cautiously

## Acceptance Criteria

M2.5 is complete when all of the following are true:

1. The runtime can ingest narrative episodes without requiring manual numeric observation entry.
2. Each narrative episode is compiled into a structured appraisal-bearing payload.
3. The existing memory pipeline can store narrative-derived episodes.
4. Sleep consolidation updates at least one new high-level prior in addition to action bias.
5. Repeated narrative exposures change later action evaluation or prediction in a measurable way.
6. All changes are visible in trace output with full provenance.
7. Existing toy-world tests continue to pass unchanged.

## Test Plan

### Unit Tests

Add:

- `tests/test_narrative_compiler.py`
- `tests/test_narrative_ingestion.py`
- `tests/test_narrative_sleep_consolidation.py`

Minimum assertions:

- deterministic compilation of seeded narrative strings
- appraisal vector values fall inside bounded ranges
- provenance and confidence are always populated
- compatibility observation mapping is stable

### Integration Tests

Add:

- `tests/test_m25_narrative_plasticity.py`

Required scenarios:

- repeated predator near-miss logs increase threat-related priors after sleep
- witnessed-fatality logs alter contamination or uncertainty priors
- narrative-derived priors measurably affect action ranking

### Regression Tests

Preserve:

- current M2 acceptance suite
- sleep consolidation baseline behavior
- deterministic state serialization and replay

## Implementation Order

### Phase 1: Data Structures

- add narrative dataclasses
- add serialization helpers
- add trace payload schema

### Phase 2: Compiler

- implement deterministic rule-based compiler
- support seeded baseline event types
- emit appraisal vector plus provenance

### Phase 3: Agent Integration

- add narrative ingestion API
- store narrative-derived episodes in long-term memory
- expose trace output

### Phase 4: Sleep Write-Back

- consolidate appraisal patterns
- write back new narrative priors
- update self-model continuity summary

### Phase 5: Verification

- add unit and integration tests
- generate one artifact trace showing full path from narrative to prior delta

## Risks

### Risk 1: LLM Overreach

If the compiler depends too much on unconstrained LLM output, the system becomes non-deterministic and hard to audit.

Mitigation:

- deterministic rule-based base path is mandatory
- LLM path is optional augmentation only

### Risk 2: Semantic Collapse Back To Six Channels

If appraisal is immediately flattened into the six observation dimensions, M2.5 adds very little real capability.

Mitigation:

- preserve appraisal latents as first-class stored fields
- sleep must operate on appraisal-bearing episodes

### Risk 3: Unbounded Personality Claims

There is a temptation to overstate what this system can produce from biography-scale input.

Mitigation:

- milestone text and tests must frame the output as prior reshaping, not personality cloning

## Deliverables

- new narrative data model module
- new narrative compiler module
- new narrative ingestion service
- agent ingestion API
- extended memory payload schema
- sleep write-back for narrative priors
- trace support for narrative provenance
- deterministic tests for the seeded narrative scenarios
- one canonical artifact trace demonstrating end-to-end prior reshaping

## Exit Condition

M2.5 exits successfully when OpenFEP can take a small diary-like sequence of narrative events and, after one or more sleep cycles, show stable, auditable changes in high-level priors and downstream action interpretation without breaking the existing toy-world FEP loop.
