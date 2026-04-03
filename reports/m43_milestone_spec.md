# M4.3 Milestone Specification

## Title

`M4.3: Single-Task Behavioral Fit and Baseline Comparison`

## Scope

- Use the M4.2 benchmark environment to evaluate single-task behavioral fit.
- Separate infrastructure metrics from benchmark-quality and human-alignment metrics.
- Remove circular scoring paths where the benchmark would otherwise grade the agent with its own internal heuristic.
- Compare the parameterized cognitive agent against a baseline ladder:
  - weak baselines that must be beaten,
  - competitive baselines that should be matched or exceeded when the architecture is credible,
  - an upper-bound reference used as a ceiling rather than a mandatory target.
- Produce held-out metrics, ablation, stress, failure-analysis, and acceptance artifacts.

## Non-Goals

- Cross-task shared-parameter credibility
- Controlled transfer claims
- Longitudinal stability claims

## Acceptance Gates

- `benchmark_fit`
- `human_alignment_metrics`
- `non_circular_scoring`
- `baseline_tiering`
- `competitive_parity`
- `ceiling_gap_reported`
- `sample_size_sufficient`
- `regression`
