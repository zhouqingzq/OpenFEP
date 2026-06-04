# M17.11 Field Validation

## Overfitting Guard
- params_fit_on: train
- metrics_reported_on: held_out
- fixtures_overlap: False
- leakage_detected: False

## Held-out Metrics
- mean_fe_advantage_vs_best_single: 0.100795
- mean_fe_advantage_vs_naive_topk: 0.118885
- mean_fe_advantage_vs_field_off: 0.246887
- median_fe_advantage_vs_best_single: 0.0
- median_fe_advantage_vs_naive_topk: 0.0
- win_rate: 0.25
- no_gain_rate: 0.0
- regression_rate: 0.0
- p90_regression_magnitude: 0.0

## Trajectory
- full_loop_mean_slope: 0.033511
- frozen_memory_mean_slope: 0.0

## Held-out Rows
- hold_conflict_hide_vs_forage: status=suppressed_naive_topk_equivalent, field=1.406443, best_single=1.406443, naive_topk=1.406443, field_off=1.58623
- hold_goal_divergent_social: status=suppressed_naive_topk_equivalent, field=1.323034, best_single=1.323034, naive_topk=1.323034, field_off=1.504795
- hold_dominant_rest: status=suppressed_naive_topk_equivalent, field=1.500621, best_single=1.500621, naive_topk=1.500621, field_off=1.682804
- hold_field_required_scan_corridor: status=field_required, field=0.973277, best_single=1.376456, naive_topk=1.448815, field_off=1.417094

## Honesty Statement
- Outcome quantity is the M17.5 expected-free-energy surrogate, not a learned generative-model variational free energy.
