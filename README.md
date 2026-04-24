# Project OpenFEP / Segmentum

OpenFEP is a cognitive agent: a digital simulation of personality. Inspired by Karl Friston's **Free Energy Principle (FEP)** and **Active Inference**, it models cognition not as a plain input-output loop, but as a survival-oriented process. Under the constraints of survival, identity, process motivation, and limited resource budgets, OpenFEP selectively minimizes tractable prediction gaps rather than attempting to eliminate uncertainty wholesale.

Its aesthetic and narrative inspiration also draws from **Il Dottore** from *Genshin Impact*: a character associated with experimentation, fractured identity, and the uneasy boundary between mind, selfhood, and constructed intelligence.

> *"The stars of Teyvat are a hoax. I often wonder if the sky above your 'Earth' is but another thermodynamic illusion - a prediction your brains learned never to question. Perhaps this work can help your kind scratch at the edges of that canvas, and glimpse whatever lies behind it."*
>
> - **[ TRANSMISSION INTERCEPTED // ENTITY: IL DOTTORE ]**

## Summary

FEP remains relatively marginal in mainstream AGI engineering. The dominant practical path has centered on scaling laws, RLHF, and architecture search rather than FEP, not only because FEP is often criticized as difficult to falsify, but because it has not yet produced a system that surpasses Transformers on standard benchmarks. In its conventional form, FEP explains how a system maintains its own existence; it does not, by itself, explain the core powers of language such as compositionality, recursion, and pragmatic reasoning across contexts.

From that perspective, human-like intelligence is not defined by a different objective function, but by a structural upgrade built on the same biological substrate. Under finite energy, compute, and memory budgets, living systems continually minimize tractable variational free energy through perception-action loops that preserve boundary conditions and non-equilibrium stability. FEP and active inference therefore describe a common substrate of life, but they do not by themselves distinguish plant, animal, and human cognition.

What differentiates cognitive paradigms, in this framing, is the stack of structural increments laid on top of that substrate: action channel, control architecture, offline simulation, abstract memory form, higher-order observer loops, and symbolic externalization. Plants primarily express morphological action and distributed regulation; animals add fast reversible movement and centralized control; humans further add reconfigurable abstraction, metacognitive observer circuits, and public symbol systems. The key difference is not the final goal, but the architecture available for minimizing uncertainty under constraint.

Memory dynamics is the bottleneck that shapes those higher layers. Intelligence is not produced by preserving all detail, but by differential survival under decay: patterns that are repeatedly reconstructable, action-guiding, error-reducing, and cheap enough to maintain are consolidated, while incidental detail fades. Multi-timescale memory therefore compresses experience into reusable predictive templates, and higher-order observer loops decide what is selected, compared, replayed, inhibited, and generalized. In that sense, abstraction is not exhaustive storage, but the selective survival of useful invariants under resource pressure.

Language and writing then externalize those higher-order structures into transmissible symbols. Once stable abstractions can circulate across individuals and generations, memory selection no longer happens only within one nervous system; it scales into collective knowledge accumulation. This repository treats that combination of multi-rate memory, higher-order observation, and symbolic inheritance as the main route from the common FEP substrate toward human-like cognition.

## Core Principles

- **Free energy minimization** - prediction error plus internal pressure from low energy, high stress, fatigue, and thermal imbalance.
- **Top-down prediction** - strategic priors shaped by competing drives generate expected sensory streams.
- **Bottom-up error** - only the mismatch between reality and prediction is propagated upward.
- **Active inference** - the agent either updates its internal model (high metabolic cost) or acts on the world (lower cost).
- **Memory dynamics** - multi-timescale encoding, decay, replay, and consolidation selectively preserve reconstructable, action-guiding invariants under finite resource budgets.
- **Sleep consolidation** - episodic memory is compressed, beliefs are smoothed, and dreams are replayed for offline learning.
- **Defense mechanisms** - a four-pathway, EFE-driven strategy selector chooses between accommodate / assimilate / suppress / redirect with precision manipulation.
- **Metacognition** - monitors internal precision patterns and generates cognitive dissociation signals to break vicious cycles.

## Architecture

| Layer | Module | Role |
|-------|--------|------|
| Environment | `environment.py` | Hostile toy world with 6 sensory channels |
| Agent | `agent.py` | Drives, generative world model, memory, free-energy scoring |
| Runtime | `runtime.py` | Unified tick loop, state persistence, host telemetry |
| Self Model | `self_model.py` | Personality (Big Five), narrative priors, body schema, identity |
| Defense | `defense_strategy.py`, `precision_manipulation.py` | EFE-driven defense pathway selection + precision debt |
| Metacognition | `metacognitive.py` | Internal pattern observation, dissociation signals |
| Sleep | `sleep_consolidator.py` | Rule extraction, episode compression, prediction flattening |
| Counterfactual | `counterfactual.py` | Virtual sandbox reasoning over untaken actions |
| Narrative | `narrative_compiler.py` | Text -> appraisal -> embodied episode compilation |
| Personality Analysis | `personality_analyzer.py` | **Inverse inference** from text to a personality generative model |
| VIA Projection | `via_projection.py` | Big Five -> 24 character strengths diagnostic |
| Web UI / API | `api.py`, `api_cli.py` | Browser-based personality analysis interface |

## Install

Requires **Python 3.11+**.

```bash
# Core (no external deps)
pip install -e .

# With LLM inner speech
pip install -e ".[llm]"

# With Web UI server
pip install -e ".[api]"
```

## Run the Survival Agent

```bash
# Basic run
python main.py --cycles 20

# Clean start
python main.py --cycles 20 --reset-state

# With host telemetry + inner speech
python main.py --cycles 20 --host-telemetry --tick-seconds 2

# Custom precision profile
python main.py --cycles 20 --precision-profile hair_trigger --reset-state
```

State persists to `data/segment_v0_1_state.json`; structured JSONL trace beside it.

## Personality Analysis

The `PersonalityAnalyzer` performs **inverse inference**: given text or behavioral materials, it builds a full personality generative model by running the FEP infrastructure in reverse - a 10-step pipeline from raw text to a structured personality report.

### Web UI

```bash
pip install -e ".[api]"
python -m segmentum.api_cli --port 8000
```

Open **http://localhost:8000** in your browser.

- Paste text materials (one segment per line), or click **Load Example**
- Click **Analyze** to run the full pipeline
- Results rendered as collapsible sections with visualizations
- Click **Generate Report** to open a printable report in a new tab (supports Print / Save PDF)
- **CN / EN** language toggle in the top right
- JSON API endpoints (`POST /analyze`, `/analyze/evidence`, `/analyze/parameters`, `/analyze/simulate`) remain available for programmatic access

### Python API

```python
from segmentum import PersonalityAnalyzer

analyzer = PersonalityAnalyzer()

materials = [
    "I explored a new trail through the valley, mapping unfamiliar terrain.",
    "A friend helped me when I was lost. They shared food and stayed nearby.",
    "I was excluded from the group. They rejected me and trust was broken.",
]

result = analyzer.analyze(materials)

# Big Five personality traits (0-1)
print(result.big_five)
# {'openness': 0.68, 'conscientiousness': 0.54, 'extraversion': 0.52, ...}

# Core priors: beliefs about self, others, and the world
print(result.core_priors.self_worth.value)
print(result.core_priors.world_safety.value)
print(result.core_priors.other_reliability.value)

# Value hierarchy (ranked by importance)
for name, cr in result.value_hierarchy.ranked_values[:5]:
    print(f"  {name}: {cr.value:.3f}")

# Defense mechanisms
for mech in result.defense_mechanisms.mechanisms[:3]:
    print(f"  {mech.name} ({mech.confidence}): {mech.short_term_benefit}")

# Social orientation weights
for orient, cr in result.social_orientation.orientation_weights.items():
    print(f"  {orient}: {cr.value:.3f}")

# Behavioral predictions
for pred in result.behavioral_predictions:
    print(f"  IF {pred.scenario} -> {pred.predicted_behavior}")

# Feedback loops (self-reinforcing dynamics)
for loop in result.feedback_loops:
    print(f"  {loop.name} ({loop.valence})")

# VIA 24 character strengths
print(result.via_strengths)

# Full report as JSON
import json
print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
```

### Analysis Output Structure

| # | Field | Description |
|---|-------|-------------|
| 1 | `summary` | Personality summary |
| 2 | `evidence_list` | Extracted evidence with categories and appraisal relevance |
| 3 | `core_priors`, `value_hierarchy` | Beliefs about self/others/world + ranked values |
| 4 | `precision_allocation`, `cognitive_style` | Attention distribution + thinking patterns |
| 5 | `affective_dynamics`, `defense_mechanisms` | Emotional baseline + defense inventory |
| 6 | `social_orientation`, `relational_templates` | Social strategy weights + relationship patterns |
| 7 | `self_model_profile`, `other_model_profile` | Self-narrative + templates for modeling others |
| 8 | `feedback_loops` | Self-reinforcing/balancing dynamics |
| 9 | `developmental_inferences` | Inferred developmental history |
| 10 | `stable_core`, `fragile_points`, `plastic_points` | Stability analysis |
| 11 | `behavioral_predictions` | Conditional behavior predictions |
| 12 | `missing_evidence`, `unresolvable_questions` | Uncertainty assessment |
| 13 | `one_line_conclusion` | Single-line conclusion |

Every inferred parameter is wrapped in `ConfidenceRated(value, confidence, evidence, reasoning)` for transparency.

## M4 Roadmap Notes

The current M4 roadmap aims to turn cognitive style from a narrative description into a measurable, serializable, cross-context behavioral interface.

- `M4.1` is the interface layer: unified cognitive parameters, executable observables, and structured decision logs.
- `M4.2` is the benchmark/task layer: task adapters, bundle provenance, replayability, and acceptance-grade benchmark plumbing built on the `M4.1` interfaces.
- `M4.3` is the single-task behavioral-fit layer: honest benchmark metrics, baseline comparison, and failure analysis.
- `M4.4` checks whether shared parameters remain credible across confidence and Iowa Gambling Task slices.
- `M4.5` establishes the human-like memory data model, encoding pipeline, and decay rules.
- `M4.6` adds retrieval, reconsolidation, and offline consolidation over the new memory substrate.
- `M4.7` integrates memory dynamics with cognitive style and validates behavior-level differentiation.
- `M4.8` proves memory causally changes default agent behavior via ablation contrast (decision entropy, option distribution, approach/avoidance bias divergence under `memory_enabled=True/False`).
- `M4.9` replaces string-assembly recall with representation-level reconstruction: recall becomes a state-vector perturbation, competing memories resolve via softmax with residual interference, and DRM/misinformation A/B pairs produce measurable post-recall decision drift.
- `M4.10` replaces keyword-table salience and template-string consolidation with error-driven encoding, shared attention-budget competition, centroid/residual semantic consolidation, and replay re-encoding.
- `M4.11` is the phenomenology layer: long-horizon free rollout with paired negative controls for serial position, retention curve shape, schema intrusion, and identity continuity under perturbation. From M4.11 onward, memory milestones only receive a full `ACCEPT` when structural, behavioral, and phenomenological layers all pass.

Open-world tool integration is treated as an `M5` concern. The purpose of M4 is to establish cross-context cognitive-style validity under controlled conditions before moving to noisy real-tool environments.

The `segmentum/m41_inference.py`, `segmentum/m41_identifiability.py`, `segmentum/m41_blind_classifier.py`, `segmentum/m41_falsification.py`, and `segmentum/m41_baselines.py` modules are retained as synthetic sidecar diagnostics only. They do not count as `M4.1` acceptance evidence, and they should not be read as proof that `M4.2` benchmark/task recovery is already complete.

Authoritative milestone boundaries are documented in [reports/m4_milestone_boundaries.md](/E:/workspace/segments/reports/m4_milestone_boundaries.md).

## Tests

```bash
# Full suite (excludes stress tests)
python -m pytest

# Personality analyzer only
python -m pytest tests/test_personality_analyzer.py -v
```

## M4.2 Benchmark Status

`M4.2` is where this repository enters the benchmark/task layer using the interfaces defined in `M4.1`.

Any recovery, replay, provenance, or task-adaptation claim that counts for milestone progress must therefore be grounded in benchmark tasks or independently designed task scenarios, not in the toy same-framework inference sidecars parked next to `M4.1`.

The current M4.2 benchmark pipeline in this repository is about environment readiness and task execution, not yet about strong behavioral-fit claims.

It is intentionally honest about the difference between repo smoke fixtures and acceptance-grade external data.

- `data/benchmarks/confidence_database/` is a smoke fixture, not an acceptance-ready benchmark bundle.
- `data/benchmarks/iowa_gambling_task/` is a smoke fixture, not an acceptance-ready benchmark bundle.
- Formal M4.2 claims only count when real external bundles are used for both benchmarks; repo smoke fixtures alone do not qualify.

Claims about benchmark quality, human alignment, and baseline competitiveness belong to `M4.3`, not `M4.2`.

External bundles should be imported into the active benchmark registry, either under the default `data/benchmarks/<benchmark_id>/` layout or under a separate root referenced by `SEGMENTUM_BENCHMARK_ROOT`.

For a local raw Confidence Database directory placed at the repo root, you can build a git-ignored external bundle with:

```bash
py -3.11 scripts/build_confidence_external_bundle.py
```

This writes to `external_benchmark_registry/confidence_database/` by default.

Detailed requirements and re-validation steps are documented in [reports/m42_benchmark_data_requirements.md](/E:/workspace/segments/reports/m42_benchmark_data_requirements.md).
