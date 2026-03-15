# M2.6 Milestone Spec

## Title

M2.6 Personality Trait Space & Trait-Driven Decision Making

## Summary

M2.6 extends OpenFEP from a survival agent with rudimentary narrative priors into an agent whose accumulated experience shapes a structured personality profile (Big Five traits). These traits systematically modulate decision-making, producing measurably different behavioral patterns for different life histories.

`narrative history -> appraisal accumulation -> personality trait extraction -> drive/policy modulation -> divergent behavior`

This milestone does not aim to perfectly replicate a real human personality. Its purpose is narrower and testable: make the agent's behavioral profile a stable, auditable function of its accumulated narrative experience, mediated by a psychologically grounded trait space.

## Why This Milestone Exists

M2.5 introduced narrative ingestion and prior reshaping, but the personality representation remains thin:

- `NarrativePriors` has only 5 dimensions (trust, controllability, trauma, contamination, meaning)
- `IdentityTraits` has only 2 dimensions (risk_aversion, resource_conservatism)
- These 7 total dimensions cannot capture the range of behavioral variation seen in real personalities
- The coupling between traits and decision-making is ad-hoc rather than systematic

M2.6 adds a psychologically grounded personality layer (Big Five / OCEAN) that:

1. Emerges from accumulated narrative experience
2. Systematically modulates drives, priors, and policy evaluation
3. Produces measurably different behavior for different life histories
4. Is fully serializable, auditable, and deterministic

## Scope

### In Scope

- `PersonalityProfile` dataclass with Big Five traits (openness, conscientiousness, extraversion, agreeableness, neuroticism), each on [0, 1] scale with 0.5 as population mean.
- Narrative-to-personality compilation: mapping appraisal vectors to Big Five trait deltas during sleep consolidation.
- Systematic personality-decision coupling: Big Five traits modulate drive urgencies, strategic priors, and identity bias in the policy evaluator.
- Personality profile serialization, round-trip, and trace output.
- Acceptance tests showing divergent behavior from divergent narrative histories.

### Out of Scope

- Multi-agent social interaction or Theory of Mind (reserved for M2.7+).
- Expanding the action space or environment beyond current 7 actions / 6 channels.
- LLM-based personality inference from free-text biography.
- Clinical-grade personality assessment or psychometric validation.
- Replacing or removing existing NarrativePriors (they coexist and feed into Big Five extraction).

## Product Goal

After M2.6, two agents initialized identically but fed different narrative histories should:

1. Develop measurably different Big Five personality profiles.
2. Make different action choices in the same environment state.
3. Show personality-consistent behavioral patterns (e.g., high-neuroticism agent avoids danger more; high-extraversion agent seeks contact more).
4. Maintain full auditability from narrative input to personality profile to action choice.

## Architecture

### Existing Modules Preserved

- `segmentum/agent.py` - extended with personality coupling
- `segmentum/self_model.py` - extended with PersonalityProfile
- `segmentum/narrative_compiler.py` - extended with personality signal extraction
- `segmentum/drives.py` - extended with personality modulation
- All existing M2.5 narrative pipeline remains intact

### New/Modified Components

#### 1. `PersonalityProfile` (in `self_model.py`)

```python
@dataclass(slots=True)
class PersonalityProfile:
    openness: float = 0.5          # curiosity, novelty-seeking, tolerance of ambiguity
    conscientiousness: float = 0.5  # planning, resource conservation, fatigue tolerance
    extraversion: float = 0.5      # social drive, activity level, positive affect
    agreeableness: float = 0.5     # trust, cooperation, conflict avoidance
    neuroticism: float = 0.5       # threat sensitivity, stress reactivity, negative affect
```

Each trait is on [0, 1] with 0.5 as neutral/population mean. Traits drift slowly through narrative experience accumulation during sleep consolidation.

#### 2. Personality Signal Extraction (in `narrative_compiler.py`)

The compiler extracts personality-relevant signals from each appraisal vector:

| Appraisal Dimension | Big Five Trait Affected | Direction |
|---------------------|------------------------|-----------|
| physical_threat (high, survived) | neuroticism + | threat sensitization |
| controllability (high) | openness + | agency reinforcement |
| self_efficacy_impact (positive) | conscientiousness + | competence signal |
| trust_impact (positive) | agreeableness + | social reward |
| trust_impact (negative) | agreeableness - | social punishment |
| social_threat (high) | neuroticism + | social anxiety |
| novelty (high, positive outcome) | openness + | curiosity reward |
| attachment_signal (positive) | extraversion + | social bonding |
| loss (high) | neuroticism + | loss sensitization |
| contamination (high) | conscientiousness + | caution reinforcement |

#### 3. Personality-Decision Coupling

**Drive System Modulation:**
- Openness: increases exploration drive weight, decreases comfort drive weight
- Conscientiousness: increases comfort drive weight, moderates hunger urgency
- Extraversion: increases social drive weight, increases exploration drive weight
- Agreeableness: decreases safety drive aggression, increases social drive weight
- Neuroticism: increases safety drive weight, increases thermal drive sensitivity

**Strategic Layer Modulation:**
- Openness: lowers novelty_floor (seeks more novelty)
- Conscientiousness: raises energy_floor (more resource-conservative)
- Extraversion: raises social_floor (needs more social contact)
- Agreeableness: lowers danger_ceiling (less threat-focused)
- Neuroticism: raises danger_ceiling (more threat-sensitive)

**Policy Evaluator Identity Bias:**
- Openness: bonus for scan, penalty for hide
- Conscientiousness: bonus for rest/exploit_shelter, penalty for risky forage
- Extraversion: bonus for seek_contact/scan, penalty for hide/rest
- Agreeableness: bonus for seek_contact, penalty for aggressive forage under danger
- Neuroticism: bonus for hide/exploit_shelter, penalty for forage under danger

## Data Model

### PersonalityProfile

```python
@dataclass(slots=True)
class PersonalityProfile:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    update_count: int = 0
    last_updated_tick: int = 0
```

### PersonalitySignal (extracted per narrative episode)

```python
@dataclass(frozen=True, slots=True)
class PersonalitySignal:
    openness_delta: float = 0.0
    conscientiousness_delta: float = 0.0
    extraversion_delta: float = 0.0
    agreeableness_delta: float = 0.0
    neuroticism_delta: float = 0.0
```

## Personality Update Mechanics

### Learning Rate

Personality traits update slowly during sleep consolidation:

```
new_trait = old_trait * (1 - learning_rate) + signal * learning_rate
learning_rate = base_rate / (1 + decay * update_count)
base_rate = 0.15
decay = 0.02
```

This ensures:
- Early experiences have stronger influence (personality formation)
- Later experiences still matter but cannot wildly swing established traits
- Traits converge toward stable values over many sleep cycles

### Clamp

All traits are clamped to [0.05, 0.95] to prevent degenerate extremes.

## Acceptance Scenarios

### Scenario A: Threat-Heavy History Produces High Neuroticism

Input: 8 predator attack narratives + 4 witnessed death narratives.

Expected:
- neuroticism > 0.65 after sleep consolidation
- Agent prefers hide/exploit_shelter over forage in moderate-danger environments
- Measurably more cautious than a fresh agent in the same environment

### Scenario B: Social-Positive History Produces High Extraversion + Agreeableness

Input: 8 rescue narratives + 4 resource gain narratives with social elements.

Expected:
- extraversion > 0.60 after sleep consolidation
- agreeableness > 0.60 after sleep consolidation
- Agent prefers seek_contact more than a fresh agent
- Social drive urgency is higher

### Scenario C: Divergent Histories Produce Divergent Behavior

Two agents, same seed, same environment:
- Agent A: fed threat-heavy narratives (Scenario A)
- Agent B: fed social-positive narratives (Scenario B)

Expected:
- Different action choices in at least 40% of identical decision cycles
- Agent A chooses hide/exploit_shelter more often
- Agent B chooses seek_contact/scan more often
- Both are internally consistent with their personality profiles

### Scenario D: Serialization Round-Trip

Expected:
- PersonalityProfile survives JSON serialization and deserialization
- Restored agent produces identical decisions to pre-serialization agent

## Acceptance Criteria

M2.6 is complete when all of the following are true:

1. PersonalityProfile with Big Five traits exists and is integrated into SelfModel.
2. Narrative experience accumulation during sleep consolidation updates personality traits.
3. Personality traits measurably modulate drive urgencies, strategic priors, and identity bias.
4. Two agents with different narrative histories produce different action choices in the same environment.
5. PersonalityProfile survives serialization round-trip.
6. All changes are visible in trace output.
7. All existing tests continue to pass unchanged.

## Test Plan

### Unit Tests

- `tests/test_m26_personality.py`
  - PersonalityProfile creation, serialization, round-trip
  - PersonalitySignal extraction from appraisal vectors
  - Personality update mechanics (learning rate, decay, clamping)
  - Drive modulation by personality
  - Strategic layer modulation by personality

### Integration Tests

- `tests/test_m26_acceptance.py`
  - Scenario A: threat history → high neuroticism → cautious behavior
  - Scenario B: social history → high extraversion → social behavior
  - Scenario C: divergent histories → divergent actions
  - Scenario D: serialization round-trip preserves personality and behavior

### Regression Tests

- All existing M2.5 tests pass unchanged
- All existing baseline regressions pass unchanged

## Implementation Order

### Phase 1: Data Structures
- Add PersonalityProfile and PersonalitySignal to self_model.py
- Add serialization/deserialization
- Integrate into SelfModel

### Phase 2: Narrative → Personality Compilation
- Add personality signal extraction to narrative_compiler.py
- Add personality update logic to agent sleep consolidation
- Wire appraisal accumulation → personality trait deltas

### Phase 3: Personality → Decision Coupling
- Modify DriveSystem to accept personality modulation
- Modify StrategicLayer priors to incorporate personality
- Extend PolicyEvaluator.identity_bias with personality terms

### Phase 4: Tests & Verification
- Write unit and acceptance tests
- Run full regression suite
- Generate artifact trace showing personality divergence

## Risks

### Risk 1: Over-Coupling
If personality influence is too strong, the agent may become trapped in personality-consistent but survival-suboptimal behavior.

Mitigation: personality modulation is additive and bounded (max +-0.15 per trait contribution). Survival priors always dominate in critical states.

### Risk 2: Trait Instability
If learning rate is too high, personality may oscillate wildly between sleep cycles.

Mitigation: decaying learning rate + trait clamping + exponential moving average.

### Risk 3: Breaking Existing Behavior
New personality modulation could change baseline agent behavior and break existing tests.

Mitigation: default PersonalityProfile has all traits at 0.5 (neutral), which produces zero modulation. Existing tests use default profiles.

## Deliverables

- PersonalityProfile and PersonalitySignal data structures
- Personality signal extraction in narrative compiler
- Personality update during sleep consolidation
- Drive system personality modulation
- Strategic layer personality modulation
- Policy evaluator personality coupling
- Full serialization support
- Unit and acceptance tests
- Milestone spec document

## Exit Condition

M2.6 exits successfully when OpenFEP can take two different narrative histories and, after sleep consolidation, produce two agents with measurably different Big Five personality profiles that drive measurably different action choices in the same environment, all fully auditable and deterministic under fixed seed.
