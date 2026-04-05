# M4 Milestone Boundaries

## Purpose

This document is the authoritative boundary definition for the M4 series.
It exists to keep milestone naming, acceptance language, prompts, and reports
aligned across the repository.

## Boundary Table

| Milestone | Primary Goal | Evidence That Counts | Explicit Non-Goals |
| --- | --- | --- | --- |
| `M4.1` | Translate cognitive variables into a unified parameter, observable, and logging interface. | Schema roundtrip, executable observable registry, intervention sensitivity inside the minimal simulator, log completeness, stress-mode interface behavior. | Benchmark adapters, benchmark bundle integration, human-data claims, blind classification, parameter recovery, falsification, baseline comparison, inference engines. |
| `M4.2` | Build the cognitive benchmark and task layer around the shared interfaces. | Benchmark registry, manifest validation, smoke-vs-external separation, protocol schemas, adapter execution, deterministic replay, leakage checks, provenance-rich artifacts, and task-grounded replay/recovery setup. | Declaring good behavioral fit, proving human alignment, beating baselines, identifying latent parameters from same-framework toy sidecars. |
| `M4.3` | Demonstrate single-task behavioral fit on real external data, compare against baseline ladder, assess parameter activity. | Real-data fit metrics (Confidence DB + IGT), tiered baseline comparison, parameter sensitivity on external data, non-circular scoring, honest failure analysis. All evidence must come from external_benchmark_registry, not smoke fixtures. | Cross-task shared-parameter credibility, transfer claims, longitudinal stability claims, architecture-level rewrite of scoring function. |
| `M4.4` | Test whether shared parameters remain credible across multiple benchmark tasks. | Cross-task consistency checks, shared-parameter audits, task-to-task credibility reports. | Open-world transfer and real-tool deployment. |
| `M4.5` | Validate controlled transfer into a more complex environment. | Cross-context transfer results in a controlled non-trivial environment, failure recovery analysis. | Longitudinal stability and open-world tooling. |
| `M4.6` | Quantify stability, reproducibility, and recoverability over time and perturbation. | Long-run stability, perturbation response, recovery retention, reproducibility envelopes. | Open-world tool use and deployment claims. |

## Naming Rules

- Use `interface layer` for `M4.1`.
- Use `benchmark environment` or `benchmark/task layer` for `M4.2`.
- Use `behavioral fit` or `benchmark quality` for `M4.3`.
- Reserve `external validation` for evidence grounded in independent external human benchmark data, not same-framework synthetic generators.
- Reserve `identifiability`, `falsification`, and `blind classification` for the specific milestone or sidecar analysis that actually evaluates those claims.
- Reserve `recovery-on-task` for `M4.2+` evidence grounded in benchmark tasks or independently designed task scenarios with replay and provenance.
- Never label same-framework cross-generator results as `external_validation: true`. Use `cross_generator_synthetic` instead.

## M4.1 / M4.2 Boundary Detail

### What M4.1 owns (acceptance-grade)

| Item | Purpose |
| --- | --- |
| `segmentum/m4_cognitive_style.py` | Parameter dataclass, decision-log dataclass, observable registry, minimal simulator, intervention sensitivity |
| `segmentum/m41_audit.py` | G1-G6 gate evaluation, acceptance report generation |
| `segmentum/m41_explanations.py` | Per-record parameter-contribution explanations (interface utility) |
| `tests/test_m41_cognitive_parameters.py` | Parameter schema and intervention probe tests |
| `tests/test_m41_observables.py` | Observable registry executable evaluator tests |
| `tests/test_m41_decision_logging.py` | Decision log completeness and audit tests |
| `tests/test_m41_acceptance.py` | Acceptance report structure and gate consistency tests |

### What lives under M4.1 prefix but is NOT M4.1 acceptance evidence

These are synthetic sidecar modules. They may remain for diagnostic use but
must not be cited as M4.1 acceptance evidence or labeled with
`external_validation: true`.

| Item | What it actually does | Future home |
| --- | --- | --- |
| `segmentum/m41_inference.py` | Toy parameter recovery via per-parameter ridge regression + candidate bank. Trained and evaluated on same-framework synthetic data. | M4.3 pre-research; must be re-evaluated on benchmark tasks before any acceptance claim. |
| `segmentum/m41_blind_classifier.py` | Nearest-centroid profile classifier. Trained on internal generator, evaluated on external generator. Both generators share the same action schema and parameter semantics. | M4.3 sidecar; cross-generator ≠ external validation. |
| `segmentum/m41_baselines.py` | Same-framework baseline models for toy comparison. | M4.3 baseline ladder. |
| `segmentum/m41_falsification.py` | Same-framework intervention sensitivity checks (Cohen's d on internal synthetic series). | M4.3 falsification; must be re-run on benchmark data. |
| `segmentum/m41_identifiability.py` | Same-framework recoverability analysis. | M4.3 identifiability; needs benchmark-grounded evidence. |
| `segmentum/m41_external_generator.py` | A second synthetic data generator with softmax sampling and separate scenario pools. NOT an independent external source. | Sidecar utility; rename to `m41_synthetic_holdout_generator.py` when convenient. |
| `segmentum/m41_external_dataset.py` | Loader and normalizer for same-framework holdout data. | Sidecar utility. |
| `segmentum/m41_external_observables.py` | Alternative observable computation with shifted thresholds. | Sidecar utility. |
| `segmentum/m41_external_validation.py` | Wrapper re-exporting external task eval functions. | M4.2+ task eval. |
| `segmentum/m41_external_task_eval.py` | External benchmark bundle evaluation and leakage checks. | M4.2 canonical evidence. |
| `scripts/generate_m41_external_data.py` | Script generating same-framework synthetic holdout data. | Sidecar utility. |
| `data/m41_external/` | 1000-row synthetic holdout dataset generated by `m41_external_generator.py`. Subject IDs leak profile names. | Sidecar data; not external. |

### What M4.2 owns (acceptance-grade)

| Item | Purpose |
| --- | --- |
| `segmentum/benchmark_registry.py` | Benchmark bundle discovery, manifest validation, smoke-vs-external separation |
| `segmentum/m4_benchmarks.py` | Benchmark adapters (Confidence DB, IGT, Two-Armed Bandit), replay, bootstrap CI |
| `segmentum/m42_audit.py` | M4.2 acceptance artifact generation |
| `tests/test_m42_benchmark_adapter.py` | Adapter execution tests |
| `tests/test_m42_confidence_benchmark.py` | Confidence DB protocol and export tests |
| `tests/test_m42_external_bundle_integration.py` | Bundle provenance and leakage tests |
| `tests/test_m42_reproducibility.py` | Deterministic replay and seed tolerance tests |
| `tests/test_m42_acceptance.py` | M4.2 acceptance gate tests |

### Artifacts ownership

| Artifact | Owner | Notes |
| --- | --- | --- |
| `artifacts/m41_blind_classification.json` | Sidecar (not M4.1 acceptance) | Must use `external_validation: false` and `generator_family: cross_generator_synthetic` |
| `artifacts/m41_baseline_comparison.json` | Sidecar (not M4.1 acceptance) | Same-framework comparison only |
| `artifacts/m41_external_validation.json` | Sidecar (not M4.1 acceptance) | Claims correctly downgraded in file |
| `artifacts/m41_falsification.json` | Sidecar (not M4.1 acceptance) | Internal sensitivity, not benchmark falsification |
| `artifacts/m41_generator_separation.json` | Sidecar (not M4.1 acceptance) | JS divergence between two synthetic generators |
| `artifacts/m41_identifiability.json` | Sidecar (not M4.1 acceptance) | Same-framework recoverability |
| `artifacts/m41_task_bundle_eval.json` | Sidecar (not M4.1 acceptance) | Claims correctly downgraded in file |
| `reports/m41_acceptance_report.json` | **M4.1 acceptance** | G1-G6 + R1 only |
| `reports/m42_acceptance_report.json` | **M4.2 acceptance** | Bundle provenance, adapters, replay, leakage |

## M4.2 / M4.3 Boundary Detail

### What M4.2 delivered (environment scaffold)

| Item | Purpose |
| --- | --- |
| `segmentum/benchmark_registry.py` | Bundle discovery, manifest validation, smoke/external separation |
| `segmentum/m4_benchmarks.py` | Adapter pipeline + `_score_action_candidates` heuristic scoring |
| `segmentum/m42_audit.py` | Environment acceptance artifacts |
| `external_benchmark_registry/` | Real data: Confidence DB 825K trials, IGT 11.8K trials |
| Leakage detection, deterministic replay | Infrastructure gates |

### What M4.3 must build on top

| Item | Why it's M4.3 not M4.2 |
| --- | --- |
| External bundle fit pipeline | M4.2 scaffold runs agent through trials; M4.3 must tune/evaluate fit |
| IGT single-task fitting | M4.2 has the adapter; M4.3 must produce behavioral metrics on real IGT |
| Baseline ladder (real data) | M4.2 has no baselines; M4.3 must implement and compare |
| Parameter sensitivity on real data | M4.2 has no parameter sweep on external data |
| Failure analysis on real data | M4.2 has no failure mode reporting |

### What M4.3 inherits vs must redo

| Item | Status | Notes |
| --- | --- | --- |
| `_score_action_candidates` | Inherit as scaffold | Hand-tuned heuristic; M4.3 can tune parameters but not rewrite architecture |
| M4.1 sidecar results | Must redo on real data | blind classification, falsification, recovery — all on synthetic data |
| Current `m43_modeling.py` | Must rewrite | Hardcoded `allow_smoke_test=True`, no IGT, brute-force grid |
| Current `m43_audit.py` | Must update | Add external bundle path, IGT track, parameter sensitivity gates |

### What M4.3 owns (acceptance-grade)

| Item | Purpose |
| --- | --- |
| `segmentum/m43_modeling.py` | Single-task fitting logic (Confidence DB + IGT) |
| `segmentum/m43_baselines.py` | Independent baseline implementations |
| `segmentum/m43_audit.py` | Acceptance artifact generation |
| `tests/test_m43_single_task_fit.py` | Fit tests against real data |
| `tests/test_m43_baselines.py` | Baseline correctness tests |
| `tests/test_m43_acceptance.py` | Acceptance gate tests |

### Artifact ownership (M4.3)

| Artifact | Owner | Notes |
| --- | --- | --- |
| `artifacts/m43_confidence_fit.json` | **M4.3 acceptance** | Must use external bundle, claim_envelope: benchmark_eval |
| `artifacts/m43_igt_fit.json` | **M4.3 acceptance** | Must use external bundle, claim_envelope: benchmark_eval |
| `artifacts/m43_parameter_sensitivity.json` | **M4.3 acceptance** | Sweep on real data, external_validation: false |
| `artifacts/m43_baseline_comparison.json` | **M4.3 acceptance** | Tiered baseline ladder |
| `artifacts/m43_failure_analysis.json` | **M4.3 acceptance** | Failure modes with real examples |
| `reports/m43_acceptance_report.json` | **M4.3 acceptance** | All gates + metrics |

## Known Data Integrity Issues

1. `data/m41_external/sample_external_behavior.jsonl` subject IDs contain
   profile names (e.g. `high_exploration_low_caution-subject-1`). The code's
   blindness check only verifies that `parameter_snapshot` is empty and
   `ground_truth_*` fields are stripped during normalization, so the inference
   code path is technically blinded, but the raw data design is sloppy.

2. `artifacts/m41_blind_classification.json` previously set
   `external_validation: true` even though both generators share the same
   action schema and parameter semantics. This must be corrected to `false`.

## Practical Rules

If a claim depends on benchmark tasks, it belongs no earlier than `M4.2`.

If a claim depends on parameter recovery, replay, or provenance on tasks, it
must begin no earlier than `M4.2` and must use benchmark tasks or independent
task scenarios rather than the same-framework toy sidecars parked next to
`M4.1`.

If a claim depends on held-out fit, human-alignment metrics, or baseline
comparison, it belongs no earlier than `M4.3`.

If an artifact is labeled `external_validation: true`, the evidence must come
from data that was not generated by any code in this repository. Same-framework
cross-generator synthetic data does not qualify.
