# M17.11 Field Validation

## Overfitting Guard
- params_fit_on: train
- metrics_reported_on: held_out
- fixtures_overlap: False
- leakage_detected: False

## Held-out Metrics
- mean_fe_advantage_vs_best_single: -0.065747
- mean_fe_advantage_vs_naive_topk: -0.065747
- mean_fe_advantage_vs_field_off: 34.363906
- median_fe_advantage_vs_best_single: -0.052692
- median_fe_advantage_vs_naive_topk: -0.052692
- win_rate: 0.0
- no_gain_rate: 0.0
- regression_rate: 1.0
- p90_regression_magnitude: 0.092609

## Trajectory
- full_loop_mean_slope: -0.004369
- frozen_memory_mean_slope: 0.0

## Honesty Statement
- Outcome quantity is the M17.5 expected-free-energy surrogate, not a learned generative-model variational free energy.