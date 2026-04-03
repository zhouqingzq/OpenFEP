# M4 Milestone Boundaries

## Purpose

This document is the authoritative boundary definition for the M4 series.
It exists to keep milestone naming, acceptance language, prompts, and reports
aligned across the repository.

## Boundary Table

| Milestone | Primary Goal | Evidence That Counts | Explicit Non-Goals |
| --- | --- | --- | --- |
| `M4.1` | Translate cognitive variables into a unified parameter, observable, and logging interface. | Schema roundtrip, executable observable registry, intervention sensitivity inside the minimal simulator, log completeness, stress-mode interface behavior. | Benchmark adapters, benchmark bundle integration, human-data claims, blind classification, parameter recovery, falsification, baseline comparison. |
| `M4.2` | Build the cognitive benchmark environment around the shared interfaces. | Benchmark registry, manifest validation, smoke-vs-external separation, protocol schemas, adapter execution, deterministic replay, leakage checks, provenance-rich artifacts. | Declaring good behavioral fit, proving human alignment, beating baselines, identifying latent parameters. |
| `M4.3` | Evaluate single-task benchmark quality and compare against baselines. | Held-out benchmark metrics, human-alignment metrics, non-circular scoring, ablations, stress tests, baseline ladder, honest failure analysis. | Cross-task shared-parameter credibility, transfer claims, longitudinal stability claims. |
| `M4.4` | Test whether shared parameters remain credible across multiple benchmark tasks. | Cross-task consistency checks, shared-parameter audits, task-to-task credibility reports. | Open-world transfer and real-tool deployment. |
| `M4.5` | Validate controlled transfer into a more complex environment. | Cross-context transfer results in a controlled non-trivial environment, failure recovery analysis. | Longitudinal stability and open-world tooling. |
| `M4.6` | Quantify stability, reproducibility, and recoverability over time and perturbation. | Long-run stability, perturbation response, recovery retention, reproducibility envelopes. | Open-world tool use and deployment claims. |

## Naming Rules

- Use `interface layer` for `M4.1`.
- Use `benchmark environment` for `M4.2`.
- Use `behavioral fit` or `benchmark quality` for `M4.3`.
- Reserve `external validation` for evidence grounded in independent external human benchmark data, not same-framework synthetic generators.
- Reserve `identifiability`, `falsification`, and `blind classification` for the specific milestone or sidecar analysis that actually evaluates those claims.

## Legacy Cleanup Map

The following items have historically blurred milestone boundaries and should be interpreted using the mapping below.

### `M4.1` acceptance evidence

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`

### `M4.2` environment evidence

- `segmentum/benchmark_registry.py`
- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_external_bundle_integration.py`
- `tests/test_m42_reproducibility.py`
- `tests/test_m42_acceptance.py`

### Synthetic sidecar analyses, not `M4.1` acceptance evidence

- `segmentum/m41_blind_classifier.py`
- `segmentum/m41_identifiability.py`
- `segmentum/m41_falsification.py`
- `segmentum/m41_baselines.py`
- `segmentum/m41_external_generator.py`
- `segmentum/m41_external_dataset.py`
- `scripts/generate_m41_external_data.py`
- `data/m41_external/`

These modules may remain useful as controlled synthetic diagnostics, but they do
not define `M4.1` acceptance and they do not, by themselves, justify external
human-data claims.

## Practical Rule

If a claim depends on benchmark tasks, it belongs no earlier than `M4.2`.

If a claim depends on held-out fit, human-alignment metrics, or baseline
comparison, it belongs no earlier than `M4.3`.
