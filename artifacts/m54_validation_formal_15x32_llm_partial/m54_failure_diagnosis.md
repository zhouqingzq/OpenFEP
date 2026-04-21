# M5.4 Failure Diagnosis

## Artifact Sanity

- Passed: True
- Missing files: []
- Diagnostic trace rows: 1884
- Ablation trace rows: 0
- Stale ablation-vs-Baseline-C warning: False

## Baseline C

- Baseline C win rate: 0.733015
- Baseline C win turns: 1381
- Interpretation: {'population_surface_win_rate': 1.0, 'short_reply_template_win_rate': 1.0, 'population_averaged_state_like_win_rate': 0.0}

Worst user-strategy deltas:
| user_uid | strategy | turns | mean_delta | median_delta | personality_win_rate |
| --- | --- | --- | --- | --- | --- |
| 2637171 | random | 2 | -0.426234 | -0.426234 | 0.0 |
| 2637171 | topic | 2 | -0.426234 | -0.426234 | 0.0 |
| 2637101 | temporal | 10 | -0.255082 | -0.162241 | 0.1 |
| 2637134 | temporal | 25 | -0.244263 | -0.260796 | 0.12 |
| 2637470 | partner | 212 | -0.220559 | -0.175552 | 0.179245 |
| 2637171 | temporal | 8 | -0.197185 | -0.245732 | 0.25 |
| 2637379 | partner | 10 | -0.175161 | -0.146589 | 0.4 |
| 2637470 | random | 85 | -0.17469 | -0.144048 | 0.235294 |
| 2637470 | topic | 85 | -0.17469 | -0.144048 | 0.235294 |
| 2637354 | temporal | 25 | -0.171153 | -0.214226 | 0.28 |

## Ablations

- Ablation turn trace unavailable; rerun validation to populate `ablation_trace.jsonl`.

## Behavior

- Diagnosis tags: ['classifier_definition_issue', 'state_modeling_issue']
- Real strategy distribution: {'escape': 1616, 'exploit': 188, 'explore': 80}
- Real majority coverage: 0.857749
- Balanced recall: {'personality': 0.0, 'baseline_c': 0.0, 'train_majority': 0.333333}
- Majority vs personality: {'majority_behavioral_strategy_mean': 0.887857, 'personality_behavioral_strategy_mean': 0.222103, 'majority_minus_personality': 0.665754}
- State collapse: {'evaluated': 60, 'mean_strategy_majority_coverage': 0.886691, 'policy_dominant_match_rate': 1.0}
