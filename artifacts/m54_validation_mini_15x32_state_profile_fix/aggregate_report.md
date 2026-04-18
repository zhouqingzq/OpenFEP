# M5.4 Stop-Bleed Notice

This historical artifact is fail-closed under m54_v4_stop_bleed. Current classifier labels are LLM-generated provisional data: usable for engineering/direction checks, but not for formal human-labeled acceptance. See aggregate_report.json / m54_acceptance.json for machine-readable blockers.

# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 15
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Semantic embedding gate: False
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: partial
- Hard pass: False
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace rows: 1884

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | True |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | True |
| formal_acceptance_eligible | False |
| semantic_embedding_gate | False |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | True |
| baseline_c_full_population_implant | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 12 / 3 / 0
- User delta median: 0.0063; IQR: 0.0158
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0310 | 8 | 6 | 0.0004 | 0.0218 |
| `random` | 0.0472 | 8 | 7 | 0.0086 | 0.0246 |
| `temporal` | 0.0143 | 12 | 2 | 0.0142 | 0.0178 |
| `topic` | 0.0352 | 8 | 7 | 0.0020 | 0.0269 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1884 | 0.4384 | 0.6035 | 0.1433 | 0.4530 | 0.0032 | 0.0095 | 0.0554 | 0.0149 |
| `baseline_c` | 1884 | 0.2925 | 0.3689 | 0.0971 | 0.0346 | 0.0000 | 0.0447 | 0.1472 | 0.1277 |
| `baseline_b_best` | 1884 | 0.4835 | 0.5881 | 0.0000 | 0.0196 | 0.0000 | 0.0236 | 0.0217 | 0.0027 |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.0956 | 0.0401 | 0.8457 | 0.6439 |
| `no_surface_profile` | 60 | 0.0531 | -0.0024 | 0.8460 | 0.4296 |
| `surface_only_default_agent` | 60 | 0.0906 | 0.0351 | 0.5226 | 0.6402 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.977899, 'train_default_l2': 0.213684, 'train_full_cosine': 0.977766, 'train_full_l2': 0.215356, 'train_wrong_user_cosine': 0.981327, 'train_wrong_user_l2': 0.199213}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0874 | 0.0555 | 0.0319 | 0.0151 | True |
| `behavioral_similarity_strategy` | 0.4114 | 0.4326 | -0.0212 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1116 | 0.1265 | -0.0148 | 1.0000 | False |
| `stylistic_similarity` | 0.9063 | 0.8842 | 0.0220 | 0.0000 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9778 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0874 | 0.0074 | 0.0800 | 0.0000 | True |
| `behavioral_similarity_strategy` | 0.4114 | 0.6256 | -0.2142 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1116 | 0.3303 | -0.2186 | 1.0000 | False |
| `stylistic_similarity` | 0.9063 | 0.8014 | 0.1049 | 0.0001 | True |
| `personality_similarity` | 1.0000 | 1.0000 | -0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9778 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0874, baseline_a=0.0555, p(vs_a)=0.0151, baseline_c=0.0074, p(vs_c)=0.0000
- `agent_state_similarity`: personality=0.9778, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.4114, baseline_a=0.4326, baseline_c=0.6256
- `behavioral_similarity_action11`: personality=0.1116, baseline_a=0.1265, baseline_c=0.3303
- `stylistic_similarity`: personality=0.9063, baseline_a=0.8842, baseline_c=0.8014
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | True | True | False | True |
| `topic` | False | False | False | True |
