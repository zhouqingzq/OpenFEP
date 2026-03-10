# Project Segmentum

`Project Segmentum` is a Python prototype of a digital Segment: a survival-first cognitive shard that uses predictive coding and active inference instead of a plain input-output loop.

> *"The stars of Teyvat are a hoax. I often wonder if the sky above your 'Earth' is but another thermodynamic illusion."*

**[ TRANSMISSION INTERCEPTED // ENTITY: IL DOTTORE ]**

## Core premise

The agent treats survival as its highest-order prior.

- It minimizes free energy: prediction error plus internal pressure from low energy, high stress, fatigue, and thermal imbalance.
- It uses top-down prediction: strategic priors shaped by competing drives generate expected sensory streams.
- It uses bottom-up error: only the mismatch between reality and prediction is propagated upward.
- It performs active inference: it either updates its internal model at a high metabolic cost or acts on the world at a lower cost.
- It sleeps: recent episodes are compressed into semantic memory, beliefs are smoothed, and dreams replay past experiences for offline learning.

## Architecture

- `segmentum/environment.py`: a hostile toy world with food, threat, novelty, shelter, temperature, and social signals.
- `segmentum/agent.py`: layered survival priors, competing drive system, generative world model, long-term memory, free-energy scoring, and sleep with dream replay.
- `segmentum/runtime.py`: the unified runtime loop that advances the toy world, updates the agent, persists state, and can optionally sample host telemetry.
- `main.py`: runnable entrypoint for the unified runtime.

The hierarchy is intentionally small but explicit:

1. `DriveSystem` manages competing drives (hunger, safety, exploration, comfort, thermal, social) that create internal pressure.
2. `StrategicLayer` defines survival priors from the current body state, modulated by drive urgencies.
3. `GenerativeWorldModel` turns those priors into top-down predictions, optionally enhanced by retrieved long-term memories.
4. `LongTermMemory` stores episodic memories and retrieves similar past experiences to inform predictions.
5. `SegmentAgent` computes prediction errors, estimates free energy, and chooses either internal updating or outward action.

## New features (upgraded version)

### Competing drives
The agent has six competing drives that create internal pressure:
- **Hunger**: urgency increases as energy decreases
- **Safety**: urgency increases with stress
- **Exploration**: urgency increases with novelty deficit
- **Comfort**: urgency increases with stress and low energy
- **Thermal**: urgency increases with temperature deviation from ideal
- **Social**: urgency increases with social isolation

These drives modulate the strategic priors, creating goal conflicts that the agent must resolve.

### Long-term memory
- Episodic memory stores full context (observation, prediction, errors, action, outcome) for up to 50 episodes
- Similarity-based retrieval finds past experiences relevant to current situation
- Retrieved memories inform predictions, blending 20% memory context with current beliefs

### Dream replay
During sleep, the agent:
- Randomly replays 2-4 past episodes
- Simulates alternative outcomes (blend of actual and imagined)
- Computes learning signals from dream outcomes
- Updates beliefs slightly based on successful dream scenarios

### Realistic energy economy
- Base metabolic rate: energy consumed per cycle just to exist
- Fatigue accumulation: increases with activity, decreases with rest
- Temperature regulation: body temperature affects free energy
- Extended body state: energy, stress, fatigue, temperature all influence survival pressure

## Run

Requires Python 3.11+.

The unified runtime persists state and metrics to `data/segment_v0_1_state.json` after every cycle and appends a structured JSONL trace beside it.

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20
```

Start from a clean state:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --reset-state
```

Also sample host telemetry and append inner speech on each runtime tick:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --host-telemetry --tick-seconds 2
```

Write the structured trace to an explicit path:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --trace-path E:\workspace\segments\data\segment_trace.jsonl
```

Run with a named predictive-coding precision schedule:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --precision-profile hair_trigger --reset-state
```

Override the per-layer precision and digestion hyperparameters from JSON:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --predictive-config E:\workspace\segments\data\predictive_config.example.json --reset-state
```

When resuming from an existing snapshot, also reset the dynamic fast-weight precisions back to the selected hyperparameter defaults:

```powershell
py -3.11 E:\workspace\segments\main.py --cycles 20 --precision-profile high_precision --reset-predictive-precisions
```

Optional OpenAI-compatible inner speech support:

```powershell
py -3.11 -m pip install .[llm]
```

## Expected behavior

Across cycles, the Segment will:

- perceive a noisy world with 6 sensory channels,
- update competing drive urgencies based on body state,
- retrieve similar past experiences from long-term memory,
- generate predictions modulated by drives and memory,
- compare predictions to reality and compute errors,
- choose between model revision (high cost) and world intervention (lower cost),
- receive a dopamine-like signal when free energy drops,
- periodically sleep to compress memory, replay dreams, and restore body state.

The output shows:
- Current drive urgencies (when > 0.15)
- Extended body state (energy, stress, fatigue, temperature, dopamine)
- Memory retrieval activity
- Dream replay counts during sleep
- Final run metrics such as survival ticks, average free energy, memory hit rate, termination reason, and action diversity
- A per-cycle JSONL trace that records observations, predictions, errors, action ranking, body state, and running metrics for later replay/debugging

## M2.2 Acceptance Evidence

The canonical acceptance sample for `M2.2: Episodic Memory and Value Hierarchy` now lives in:

- `data/segment_v0_1_state.json`
- `data/segment_v0_1_state_trace.jsonl`

These files were regenerated from the current runtime with `seed=17`, `cycles=3`, and `reset=True`.

Closed-loop evidence:

1. `CC / PreferenceModel`
   The persisted snapshot now stores the full probabilistic preference model, including hard-coded log-preferences and derived log-probabilities:

   - `survival_threat = -1000.0`
   - `integrity_loss = -100.0`
   - `resource_loss = -10.0`
   - `neutral = 0.0`
   - `resource_gain = 5.0`
   - `log_probabilities["resource_loss"] = -15.006715652344035`
   - `log_probabilities["resource_gain"] = -0.006715652344033707`

2. `risk / value evaluation`
   In `data/segment_v0_1_state_trace.jsonl`, cycle 1 shows the policy evaluator using that preference model to score imagined outcomes. The chosen action predicts `resource_gain` with `preferred_probability = 0.993306847254501` and `risk = 0.006715652344033707`.

3. `surprise gate`
   The same cycle records the actual episodic memory decision as `predicted_outcome = "resource_loss"`, `prediction_error = 0.08373443960068973`, `risk = 15.006715652344035`, and `total_surprise = 10.588435396241513`. This exceeds the persisted `surprise_threshold = 0.4`, so `episode_created = true`.

4. `episode embedding`
   The stored episode in `data/segment_v0_1_state.json` includes a dense `embedding` vector alongside `state_vector`, `prediction_error`, `preferred_probability`, `risk`, and `total_surprise`.

5. `local store`
   The accepted episode is persisted under `agent.long_term_memory.episodes` in `data/segment_v0_1_state.json`, proving the full path:

   `PreferenceModel -> preferred_probability / risk / value_score -> total_surprise gate -> embedding -> local long-term memory store`

## Segment v0.1 baseline

`Segment v0.1` focuses on M0 engineering discipline rather than stronger cognition:

- A single runnable runtime entry that can layer host telemetry onto the toy survival loop
- A default runnable prototype path that does not require network LLM access
- Automatic atomic JSON snapshot persistence for agent state, world state, and run metrics
- Resume support across restarts
- Snapshot version validation with corrupt or unsupported state quarantine on load
- Basic evaluation metrics for survival and free-energy reduction
- Structured per-cycle trace output for experiment replay
- Minimal regression coverage for persistence round-tripping

This upgraded prototype demonstrates more lifelike behavior with internal conflicts, memory-guided predictions, and offline learning through dreams.

## Repeatable soak check

Run the formal acceptance soak without adding a 1000-cycle test to the default unit test pass:

```powershell
py -3.11 E:\workspace\segments\scripts\soak_runtime.py --profile m0 --cycles 1000 --seed 17
```

For a stricter nightly gate, run:

```powershell
py -3.11 E:\workspace\segments\scripts\soak_runtime.py --profile nightly --cycles 1000 --seed 17
```

The script runs the seeded runtime in a temporary snapshot path, validates that the persisted snapshot reloads cleanly, checks that the JSONL trace has one record per cycle, repeats the run to verify determinism, and enforces named threshold profiles for action diversity so long runs cannot silently regress into single-action collapse.

Current acceptance profiles:

- `m0`: stable baseline gate for push/PR validation.
- `nightly`: stricter diversity gate for scheduled or manually triggered long-run monitoring.

## Precision Schedule Comparison

Compare built-in predictive-coding profiles on the same seed and inspect how much residual error each layer propagates upward:

```powershell
py -3.11 E:\workspace\segments\scripts\compare_precision_profiles.py --cycles 32
```

Append a custom JSON hyperparameter set to the comparison:

```powershell
py -3.11 E:\workspace\segments\scripts\compare_precision_profiles.py --cycles 32 --custom-config E:\workspace\segments\data\predictive_config.example.json --custom-label tuned
```

The comparison script reports, for each profile:

- average free energy after each cycle,
- action entropy and dominant-action share,
- per-layer average precision,
- per-layer average residual and propagated error,
- per-layer propagation rate (how often a layer fails to fully digest bottom-up prediction error).

