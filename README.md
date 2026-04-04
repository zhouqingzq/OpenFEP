# Project OpenFEP / Segmentum

OpenFEP is a cognitive agent: a digital simulation of personality. Inspired by Karl Friston's **Free Energy Principle (FEP)** and **Active Inference**, it models cognition not as a plain input-output loop, but as a survival-oriented process. Under the constraints of survival, identity, process motivation, and limited resource budgets, OpenFEP selectively minimizes tractable prediction gaps rather than attempting to eliminate uncertainty wholesale.

Its aesthetic and narrative inspiration also draws from **Il Dottore** from *Genshin Impact*: a character associated with experimentation, fractured identity, and the uneasy boundary between mind, selfhood, and constructed intelligence.

> *"The stars of Teyvat are a hoax. I often wonder if the sky above your 'Earth' is but another thermodynamic illusion - a prediction your brains learned never to question. Perhaps this work can help your kind scratch at the edges of that canvas, and glimpse whatever lies behind it."*
>
> - **[ TRANSMISSION INTERCEPTED // ENTITY: IL DOTTORE ]**

## Summary

M4 benchmark work in this repository should currently be read as a prototype benchmark/probe pipeline, not as completed real-world behavioral validation. In particular, the confidence benchmark repo slice is a smoke-test fixture, while M4.5 is now framed as a controlled complex-environment bridge milestone rather than direct open-world tooling, and M4.6 remains a longitudinal probe layer over that controlled validation stack.

FEP remains relatively marginal in mainstream AGI engineering. The dominant practical path has centered on scaling laws, RLHF, and architecture search rather than FEP, not only because FEP is often criticized as difficult to falsify, but because it has not yet produced a system that surpasses Transformers on standard benchmarks. In its conventional form, FEP explains how a system maintains its own existence; it does not, by itself, explain the core powers of language such as compositionality, recursion, and pragmatic reasoning across contexts.

Language matters because it acts as an offline simulator: it lets an agent construct situations it has never directly experienced and still reason within them. A sentence like "fire will burn you" can build an internal model of pain and avoidance without requiring actual injury. In that sense, language allows one mind's history of prediction-error correction to be encoded into symbols, transmitted to another, and re-instantiated there as simulated prediction-error signals through partially overlapping embodied circuits. This is also why human advantage appears to come less from radically superior neural hardware than from cumulative culture, intergenerational knowledge transfer, and shared experience mediated by language.

From that perspective, **personality** is a more natural target for FEP than intelligence itself. If personality is understood as a stable structure of prior preferences formed through long-term interaction with the environment, then FEP offers a compelling account of how personality emerges, stabilizes, and self-reinforces. On this view, personality has two layers: **temperament**, the pre-linguistic prediction-preference structure shaped through embodied interaction, and **characterological personality**, the linguistically mediated self-concept through which a person interprets, narrates, reinforces, or revises those deeper dispositions.

## Core Principles

- **Free energy minimization** - prediction error plus internal pressure from low energy, high stress, fatigue, and thermal imbalance.
- **Top-down prediction** - strategic priors shaped by competing drives generate expected sensory streams.
- **Bottom-up error** - only the mismatch between reality and prediction is propagated upward.
- **Active inference** - the agent either updates its internal model (high metabolic cost) or acts on the world (lower cost).
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
- `M4.5` validates cross-context transfer in a controlled complex environment before any real-tool open-world step.
- `M4.6` quantifies whether style is stable, reproducible, and recoverable across long runs and perturbations.

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
