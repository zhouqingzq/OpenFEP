# M5 Series: Real-Data Personality Grounding Roadmap (v2)

## Vision

From real human text (chat logs, personal narratives, self-descriptions) to a
**living digital personality** — not a static vector conditioned on a prompt,
but an autonomous agent whose personality **emerges from** and **lives through**
the existing FEP/Active Inference runtime.

Core thesis: personality is not a label to be extracted — it is a dynamic
process that arises from how an agent predicts, attends to, remembers, and acts
upon its world under finite resources. Real chat data serves as "past life
experience" that shapes the agent's generative model, memory structure, narrative
priors, slow traits, and defense repertoire. The resulting digital entity should
exhibit cross-context behavioral consistency not because we enforce it, but
because the underlying generative model makes it the lowest free-energy
configuration.

---

## Architecture: Route B — Living Personality

```
Real chat data
      │
      ▼
  M5.0: Parse & structure
      │
      ▼
  M5.1: Define dialogue-world perception channels
        (classified by error-signal observability)
      │
      ▼
  M5.2: "Past life implantation" — feed experiences through
        full FEP loop: attention → prediction → surprise →
        memory encoding → sleep consolidation
        → personality emerges from dynamics
      │
      ▼
  M5.3: Dialogue-as-action — response generation driven by
        expected free energy minimization, not prompt engineering
      │
      ▼
  M5.4: Train/test consistency validation
      │
      ▼
  M5.5: Cross-context stability (7 scenario battery)
      │
      ▼
  M5.6: Game-ready persona runtime API
      │
      ▼
  M5.7: End-to-end integration trial
```

---

## Milestone Overview

| Milestone | Title | Core Deliverable |
|-----------|-------|-----------------|
| **M5.0** | Chat Data Pipeline | Raw chat logs → structured per-user conversation sessions |
| **M5.1** | Dialogue World Channels | Observability-classified perception channels + precision bounds |
| **M5.2** | Past Life Implantation | Chat data as lived experience → personality emerges from FEP dynamics |
| **M5.3** | Dialogue as Action | Response generation as EFE-driven action in dialogue world |
| **M5.4** | Consistency Validation | Train/test split proof that emergent personality is real and stable (partial accepted / engineering complete) |
| **M5.5** | Cross-Context Stability | Same personality, 7 different scenarios → consistent behavior |
| **M5.6** | Persona Runtime | Game-ready API with persistent living persona |
| **M5.7** | Integration Trial | End-to-end: raw chat → playable digital life entity |

---

## M5.0: Chat Data Pipeline

### Title

`M5.0: Chat Data Pipeline — From Raw Logs to Structured Personality Evidence`

### Why

The existing `NarrativeIngestion` accepts `NarrativeEpisode` objects. Real-world
chat data is messy: mixed languages, spam, single-character messages, interleaved
conversations with different partners. We need a robust pipeline that transforms
raw chat logs into clean, user-centric conversation sessions before any
personality modeling can begin.

### Engineering Scope

1. **Log parser**: parse the timestamped log format
   (`YYYY-MM-DD-HH:MM:SS INFO MessageSender::OnData message type: N, sender uid: X, reciever uid: Y, body: Z`)
   into structured `ChatMessage(timestamp, sender_uid, receiver_uid, body, msg_type)`.

2. **Conversation reconstructor**: group messages by `(uid_a, uid_b)` pair,
   split into sessions by time gap (configurable, default 30 min), produce
   `ConversationSession` objects with turn-by-turn structure preserving temporal
   ordering.

3. **Quality filter**:
   - Remove spam / promotional messages (URL-heavy, template patterns).
   - Tag (not remove) ultra-short utterances ("嗯", "哦", "哈哈") — they carry
     behavioral signal even if not semantic content.
   - Normalize Traditional/Simplified Chinese (configurable direction).
   - Anonymize PII (phone numbers, external URLs, real names).

4. **User aggregator**: for each uid, collect all sessions across all
   conversation partners, compute statistics: message count, active days,
   unique partners, avg message length, vocabulary richness, temporal activity
   pattern.

5. **User selection**: filter users with sufficient data (configurable minimum,
   default >= 200 messages, >= 3 unique partners).

6. **Export format**: structured JSON per user, ready for M5.2 ingestion.

### Key Constraint

This milestone is pure data engineering. No personality modeling, no FEP
integration. The output must be a clean, inspectable, versioned dataset.

---

## M5.1: Dialogue World Channels

### Title

`M5.1: Dialogue World Channels — Observability-Classified Perception for Conversational Active Inference`

### Why

The current Segmentum runtime perceives the world through six survival-world
channels (food, danger, novelty, shelter, temperature, social). To run the FEP
engine in a conversational world, we need dialogue-specific perception channels.

The critical design insight: channels must be classified not by content type but
by **error-signal observability** — how reliably the agent can detect prediction
errors in each channel. This directly determines the appropriate precision
weight range, and mis-calibration has clinically meaningful consequences
(e.g., high precision on low-observability channels → paranoid-like inference).

### Channel Architecture

**Tier 1 — High Observability (default precision range: 0.6–0.9)**

These channels produce error signals that can be directly computed from
observable text. The agent can trust its prediction errors here.

| Channel | Observable Signal | Error Computation |
|---------|------------------|-------------------|
| `semantic_content` | What the interlocutor actually said | Direct text comparison with predicted response topic/content |
| `topic_novelty` | How new is the current topic | Embedding distance between current utterance and conversation history |

**Tier 2 — Medium Observability (default precision range: 0.25–0.50)**

These channels require inference tools (sentiment analysis, dialogue act
classification) that introduce noise. The agent should be less certain about
its error signals here.

| Channel | Observable Signal | Error Computation |
|---------|------------------|-------------------|
| `emotional_tone` | Estimated emotional valence/arousal of the interlocutor | Sentiment model output vs. predicted emotional trajectory |
| `conflict_tension` | Degree of disagreement or confrontation | Dialogue act classification (agree/disagree/challenge/concede) |

**Tier 3 — Low Observability (default precision range: 0.05–0.20)**

These channels are fundamentally latent — the true values cannot be directly
observed. Error signals are unreliable. The agent must maintain high
uncertainty here.

| Channel | Observable Signal | Error Computation |
|---------|------------------|-------------------|
| `relationship_depth` | Cumulative intimacy/trust with this partner | Long-term statistical accumulation; no single-turn observable |
| `hidden_intent` | The interlocutor's unstated goals and motives | Essentially unobservable; inferred from behavioral patterns over many turns |

### Precision Semantics and Psychopathology Mapping

The observability tiers create a natural framework for personality dynamics:

- **Normal range**: precision weights stay within tier bounds. The agent is
  appropriately confident about what it can observe and appropriately uncertain
  about what it cannot.

- **Paranoid drift**: `hidden_intent` precision escalates above tier bounds
  (> 0.20). The agent over-interprets ambiguous signals as evidence of hostile
  intent. `MetaCognitiveLayer` should detect this as a chronic AMPLIFY pattern
  on a low-observability channel.

- **Naive/trusting drift**: `hidden_intent` precision collapses to near-zero
  AND `relationship_depth` precision is suppressed. The agent ignores social
  threat signals entirely.

- **Anxious drift**: `emotional_tone` and `conflict_tension` precision both
  escalate above tier bounds. The agent becomes hyper-sensitive to emotional
  signals that may be noisy.

These are not labels we assign — they are states the system can drift into
through its own precision dynamics, and that `MetaCognitiveLayer` and
`DefenseStrategy` can detect and respond to.

### Engineering Scope

1. **Channel registry**: define `DialogueChannel` dataclass with fields:
   `name`, `tier` (1/2/3), `precision_floor`, `precision_ceiling`,
   `default_precision`, `observability_description`.

2. **Dialogue observation**: create `DialogueObservation` — a dict-based
   observation type (not a fixed-field dataclass like `Observation`) that
   produces per-channel float values from a conversation turn. Must emit
   signals through the existing `io_bus.BusSignal` interface.

3. **Per-channel precision bounds**: extend `PredictiveCoding` to support
   per-channel `min_precision` / `max_precision` overrides (currently only
   per-layer). The tier bounds act as soft constraints — the system can push
   beyond them, but `MetaCognitiveLayer` registers it as anomalous.

4. **Signal extractors**: implement concrete signal extraction for each channel:
   - Tier 1: direct text processing + embedding distance computation.
   - Tier 2: pluggable sentiment/dialogue-act model interface (rule-based
     fallback + optional LLM-enhanced mode, matching existing pattern).
   - Tier 3: accumulative state estimators that update slowly across turns.

5. **Integration with existing engine**: the dialogue channels must be usable
   by `AttentionBottleneck`, `PrecisionManipulator`, `DefenseStrategySelector`,
   `MetaCognitiveLayer`, and `GlobalWorkspace` without modifying those modules'
   core logic (they already operate on `dict[str, float]`).

6. **Survival-world compatibility**: the existing six-channel `SimulatedWorld`
   must continue to work unchanged. Dialogue channels are a parallel world
   type, not a replacement.

---

## M5.2: Past Life Implantation

### Title

`M5.2: Past Life Implantation — Personality Emergence Through Accelerated Lived Experience`

### Why

This is the conceptual heart of M5. Instead of extracting personality as a
static vector, we treat the user's chat history as a lifetime of experiences and
feed them through the full FEP cognitive loop. The agent "lives through" these
conversations, and personality emerges from how its generative model, memory
structures, narrative priors, and slow traits are shaped by the accumulated
prediction errors and consolidation cycles.

### Mechanism

For each user's conversation history (from M5.0), chronologically:

```
For each conversation session:
    For each turn in session:
        1. DialogueObservation ← extract channel signals from turn (M5.1)
        2. AttentionBottleneck ← select which channels to attend
        3. PredictiveCoding ← compute prediction errors per channel
        4. PrecisionManipulator ← if error too large, apply defense
        5. MemoryEncoding ← encode episode with salience weighting
        6. SlowLearning ← nudge slow trait variables
        7. NarrativePriors ← update trust_prior, trauma_bias, etc.

    After N sessions (configurable):
        8. SleepConsolidation ← extract rules, consolidate memories,
           promote episodic → semantic, abstract patterns
        9. MetaCognitiveLayer ← review precision patterns,
           detect chronic defense strategies
```

After processing all of a user's history, the agent's state represents:
- A shaped `SelfModel` with non-default `NarrativePriors`
- A populated `MemoryStore` with episodic/semantic/procedural memories
- Trained `SlowTraitState` (caution_bias, threat_sensitivity, trust_stance,
  exploration_posture, social_approach)
- A `PrecisionManipulator` state with accumulated precision debt patterns
- A `DefenseStrategy` preference profile
- Learned `PreferredPolicies` (action preferences and avoidances)

This **is** the personality. Not a vector — a living cognitive state.

### Engineering Scope

1. **Dialogue world adapter**: implement `DialogueWorld` class (parallel to
   `SimulatedWorld` and `NarrativeWorld`) that replays conversation sessions
   as a sequence of world ticks. Each tick corresponds to receiving one
   conversational turn from the interlocutor.

2. **Accelerated lifecycle**: the standard `SegmentAgent` lifecycle runs
   tick-by-tick. For past-life implantation we need an accelerated mode that
   processes conversation histories efficiently while still executing the full
   cognitive pipeline per tick.

3. **Sleep scheduling**: determine when to trigger sleep consolidation during
   replay. Strategy: sleep after every N sessions (configurable), or when
   `HomeostasisState.short_term_sleep_pressure` exceeds threshold — using the
   existing mechanism.

4. **Multi-partner experience**: the agent converses with multiple partners
   over its "lifetime." Each partner should be tracked via `SocialMemory`,
   building partner-specific models while the core personality remains
   consistent.

5. **Personality emergence metrics**: at each sleep boundary, snapshot the
   agent's personality-relevant state (slow traits, narrative priors, defense
   profile, precision debt) and track convergence. Personality "maturity" is
   defined as the point where these metrics stabilize (delta < threshold across
   consecutive sleep cycles).

6. **Deterministic replay**: given the same input data and seed, the
   implantation process must produce an identical agent state.

---

## M5.3: Dialogue as Action

### Title

`M5.3: Dialogue as Action — Response Generation Through Active Inference`

### Why

In the survival world, the agent's actions are `forage`, `hide`, `scan`, etc.,
selected by minimizing expected free energy. In the dialogue world, the agent's
primary action is **generating a conversational response**. This milestone makes
response generation an FEP-driven process: the agent selects what to say based
on the same explore/exploit/escape strategy framework, conditioned by its
emergent personality state.

### Mechanism

```
Interlocutor says something
        │
        ▼
DialogueObservation (6 channels, M5.1)
        │
        ▼
AttentionBottleneck selects channels
        │
        ▼
PredictiveCoding computes errors
        │
        ▼
FEP policy: what type of response minimizes expected free energy?
  ├─ EXPLORE: ask a question, introduce a new topic, seek information
  ├─ EXPLOIT: elaborate on current topic, deepen current exchange
  └─ ESCAPE: deflect, change subject, give minimal response, disengage
        │
        ▼
PolicyEvaluator applies personality biases:
  - NarrativePriors (trust, trauma, controllability)
  - SlowTraits (caution, social_approach, exploration_posture)
  - PersonalitySignal (Big Five → policy_bias)
  - PreferredPolicies (learned action preferences)
  - DefenseStrategy (if error is too high)
        │
        ▼
Response generator: given strategy + personality state + context,
produce concrete text using:
  a) Rule-based: template selection by strategy + style parameters
  b) LLM-enhanced: personality-conditioned prompt (style, vocabulary,
     sentence length, emoji patterns from SlowTraitState)
        │
        ▼
MemoryEncoding: store this turn as experience
SlowLearning: update traits based on outcome
```

### Engineering Scope

1. **Dialogue action space**: define the action repertoire for conversational
   agents, mapped to explore/exploit/escape:
   - `ask_question`, `share_opinion`, `elaborate`, `agree`, `disagree`,
     `joke`, `empathize`, `deflect`, `minimal_response`, `disengage`, etc.
   - Each action has FEP cost/benefit profiles analogous to survival actions.

2. **Dialogue PolicyEvaluator**: extend `PolicyEvaluator` to score dialogue
   actions using personality state. The existing `identity_bias()` method
   already considers `narrative_priors`, `slow_variable_learner`,
   `personality_profile` — these naturally shape dialogue action selection.

3. **Response generator**: two modes (matching existing pattern):
   - Rule-based: deterministic response templates parameterized by strategy
     choice + style features.
   - LLM-enhanced: generate response via API, conditioned by personality state,
     chosen strategy, conversation context, and stylistic parameters extracted
     from the agent's memory of its own past responses.

4. **Style consistency**: response surface features (sentence length, punctuation,
   emoji usage, vocabulary level) should match the patterns established during
   past-life implantation, not because we enforce it via post-processing, but
   because the agent's `SlowTraitState` and `PreferredPolicies` naturally guide
   generation toward familiar patterns.

5. **Memory integration**: each generated response becomes a new experience that
   feeds back into the agent's memory and slow learning, enabling genuine
   conversational evolution.

---

## M5.4: Consistency Validation

**Status (2026-04-26): partial accepted / engineering complete.** M5.4 has
canonical smoke, direction, and formal artifacts with partial acceptance
eligible. It must not be used as a fully accepted gate for M5.5 until a fresh
formal artifact with external-human classifier labels passes all hard gates:
semantic vs Baseline A and Baseline C, behavioral vs Baseline C,
no-surface/no-policy/surface-only ablations, and artifact guard schema.

### Title

`M5.4: Consistency Validation — Train/Test Proof of Emergent Personality`

### Why

The critical scientific question: does the personality that emerges from M5.2
actually capture something real about the original human? This milestone
implements a rigorous validation framework using the user's proposed
train/test methodology, but applied to a living agent rather than a static
model.

### Methodology

For each user with sufficient data:

1. **Split**: divide conversation history into training set (70%) and holdout
   set (30%). Four split strategies:
   - Random session split.
   - Temporal split (train on earlier conversations, test on later).
   - Partner split (train on partners A,B,C; test on D,E).
   - Topic-stratified split.

2. **Implant**: run M5.2 on training set → produces a living agent.

3. **Generate**: place the agent in the holdout conversation contexts (partner
   says X, agent responds) using M5.3.

4. **Compare**: measure similarity between generated responses and actual
   holdout responses at multiple levels:
   - **Surface**: BLEU-4, ROUGE-L (expected low — sanity check only).
   - **Semantic**: sentence embedding cosine similarity (multilingual model).
   - **Stylistic**: Jensen-Shannon divergence of style feature distributions.
   - **Personality diagnostics**: legacy personality-vector cosine is retained
     only as a saturation warning; trait MAE/L2 and behavioral fingerprints are
     used for discriminative diagnostics.
   - **Behavioral**: response strategy distribution similarity (how often
     explore/exploit/escape), initiative rate, conversation engagement depth.
   - **Agent state**: cosine similarity of `SlowTraitState` and
     `NarrativePriors` between agents implanted from train vs. full data.

5. **Baselines**:
   - Baseline A: agent with default initialization (no past-life implantation).
   - Baseline B: agent implanted with a different user's data (random
     personality).
   - Baseline C: agent implanted with population-average statistics.

6. **Statistical testing**: paired tests across users for each metric.

### Engineering Scope

1. **Data splitter**: implement four split strategies.
2. **Automated evaluation pipeline**: split → implant → generate → compare,
   fully automated per user.
3. **Metric suite**: semantic, stylistic, behavioral, discriminative
   personality diagnostics, agent-state metrics, and three baselines.
4. **Report generator**: per-user and aggregate results with statistical
   significance and Markdown tables (optional extra plots are non-blocking;
   narrative summary may be brief procedural text in aggregate Markdown).

---

## M5.5: Cross-Context Stability

### Title

`M5.5: Cross-Context Stability — Same Personality, Different Worlds`

### Why

A real personality manifests recognizably across different situations. The agent
implanted with a user's chat history should behave consistently when placed in
novel scenarios it has never seen. This tests whether the personality is truly
embodied in the agent's generative model, not just memorized surface patterns.

### Scenario Battery

Seven standardized test scenarios, each designed to probe different personality
dimensions:

| Scenario | Probes | Expected Personality Influence |
|----------|--------|------------------------------|
| Casual small talk | Social approach, exploration | High extraversion → longer exchanges, more topic initiation |
| Emotional distress (partner shares bad news) | Empathy, trust, defense mechanisms | High agreeableness → empathize; low → deflect |
| Disagreement/conflict | Conflict resolution, assertiveness | High agreeableness → concede; low → escalate |
| Information seeking (partner asks for help) | Openness, competence signaling | High openness → elaborate; high conscientiousness → structured |
| Playful/humor exchange | Openness, social approach | High openness + extraversion → engage; low → minimal response |
| Ambiguous intent (partner's motives unclear) | Paranoid vs. trusting inference | Tests hidden_intent channel precision calibration |
| Game-world NPC dialogue (quest/trade) | Full personality transfer to novel context | Core personality visible in completely new setting |

### Engineering Scope

1. **Scenario conductor**: automated system that places an implanted agent into
   each scenario and runs multi-turn interactions with a scripted or LLM-driven
   counterpart.
2. **Cross-context personality extraction**: extract personality-relevant metrics
   from behavior in each scenario.
3. **Consistency metrics**: behavioral fingerprint stability across all 7
   scenarios; define "adaptation envelope" — acceptable variance range. The
   legacy personality-vector cosine is diagnostic-only because M5.4 showed it
   saturates and cannot distinguish real weaknesses from baselines.
4. **Adaptation analysis**: quantify how much the agent adapts to context
   (healthy flexibility) vs. core personality preservation. Compare with real
   user's cross-partner behavioral variance from M5.0 data.
5. **Split weakness carry-forward**: every M5.5 report must slice scenario
   results by `random` and `temporal` M5.4 split lineage, because these were
   the monitored weak points in M5.4 partial acceptance.
6. **Discriminative identity diagnostics**: report within-person vs.
   between-person retrieval, state-distance decomposition (`SlowTraitState`,
   `NarrativePriors`, defense profile, precision debt), and scenario-level
   behavioral fingerprint deltas.

---

## M5.6: Persona Runtime

### Title

`M5.6: Persona Runtime — Game-Ready Living Persona API`

### Why

Package the living persona into a real-time API suitable for game engines and
interactive applications.

### Engineering Scope

1. **Persona API**:
   - `POST /persona/create` — create from raw chat data (full M5.0→M5.2
     pipeline) or from personality description or from Big Five questionnaire.
   - `POST /persona/{id}/chat` — send a message, agent processes through full
     FEP loop, returns response.
   - `GET /persona/{id}/state` — inspect personality state (slow traits,
     narrative priors, defense profile, precision patterns).
   - `GET /persona/{id}/memory` — inspect memory store.
   - `POST /persona/{id}/scenario` — place persona in a specific context.
   - `POST /persona/{id}/sleep` — trigger consolidation cycle.

2. **State persistence**: persona state (full agent snapshot) persists across
   sessions using existing Segmentum persistence layer.

3. **Performance**: target < 2s p95 latency for response generation. Agent
   state caching, prompt pre-compilation, response streaming.

4. **Persona lifecycle**:
   - Creation (three paths: raw data, description, questionnaire).
   - Living evolution: the persona continues to learn from interactions via
     `SlowLearning` and `MemoryConsolidation`.
   - Sleep cycles: periodic consolidation during idle periods.

5. **Safety layer**: content filtering, topic boundaries, personality guardrails
   (prevent extreme precision drift into pathological territory).

6. **Game integration**: thin client SDK for Unity (C#) and generic
   WebSocket/REST.

---

## M5.7: Integration Trial

### Title

`M5.7: End-to-End Integration Trial — From Raw Chat to Playable Digital Life`

### Why

Final validation. Run the complete pipeline on real data, produce playable
digital personas, and validate the full chain under stress.

### Engineering Scope

1. **Full pipeline execution**: raw chat → M5.0 → M5.1 channels → M5.2
   implantation → M5.3 dialogue → M5.6 runtime → interactive demo.
2. **Longitudinal trial**: 5 personas, 200+ turns each, over simulated "days"
   with sleep cycles. Measure personality stability, memory coherence, defense
   pattern evolution.
3. **Comparative evaluation**: blind human evaluation panel — digital persona
   vs. real user responses to same prompts.
4. **Adversarial stress test**: manipulation attempts, rapid context switching,
   emotional exploitation — verify persona stability and safety.
5. **Technical report**: methodology, results, limitations, publication-ready.

---

## Dependencies

```
M5.0 ──→ M5.1 ──→ M5.2 ──→ M5.3 ──→ M5.4
  │                  │         │        │
  │                  │         │        ▼
  │                  │         │      M5.5 ──→ M5.6 ──→ M5.7
  │                  │         │                │
  │                  ▼         ▼                │
  │            (personality   (dialogue      (game
  │             emerges)       works)        integration)
  │
  └── Pure data engineering; no FEP dependency
```

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient per-user data volume | Personality doesn't converge during implantation | User filtering at M5.0 (>= 200 msgs, >= 3 partners) |
| Chat data too shallow for FEP dynamics | Slow traits stay near defaults; no real personality emerges | Tune sleep frequency and learning rates for chat timescale |
| Precision drift into pathological territory | Agent becomes "paranoid" or "naive" without clinical validity | MetaCognitiveLayer + tier bounds in M5.1 |
| Implantation too slow for 100w messages | Impractical pipeline | Accelerated lifecycle mode in M5.2; profile and optimize |
| LLM generation masks personality signal | All personas sound like the LLM, not the person | Rule-based mode as ablation control; style enforcement from SlowTraits |
| Six dialogue channels insufficient | Missing important conversational dimensions | Channel registry is extensible; add channels in later milestones |

## Relation to Existing Infrastructure

| Existing Module | M5 Usage |
|----------------|----------|
| `io_bus.BusSignal` | Foundation for dialogue channel signals (M5.1) |
| `AttentionBottleneck` | Channel selection in dialogue world (M5.1, M5.2) — already channel-agnostic |
| `PredictiveCoding` | Hierarchical inference on dialogue channels (M5.1, M5.2) |
| `PrecisionManipulator` | Defense mechanisms on dialogue channels (M5.2) |
| `MetaCognitiveLayer` | Detect pathological precision drift (M5.1, M5.2) |
| `DefenseStrategy` | Personality-consistent coping in dialogue (M5.2, M5.3) |
| `MemoryModel` + `MemoryStore` | Episodic/semantic memory from conversations (M5.2) |
| `MemoryConsolidation` + `SleepConsolidator` | Pattern extraction and abstraction (M5.2) |
| `MemoryRetrieval` | Context-sensitive recall during dialogue (M5.3) |
| `SlowLearning` | Personality trait drift during implantation (M5.2) |
| `SelfModel` + `NarrativePriors` | Personality core shaped by experience (M5.2) |
| `PersonalitySignal` + `PolicyEvaluator` | Personality → action selection in dialogue (M5.3) |
| `NarrativeWorld` | Template for `DialogueWorld` implementation (M5.2) |
| `PersonalityAnalyzer` | Validation tool: extract personality from generated text for comparison (M5.4) |
| `SocialMemory` | Track multiple conversation partners (M5.2) |
| Persistence layer | Persona state serialization (M5.6) |
