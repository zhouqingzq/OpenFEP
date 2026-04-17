# M5.4 Validation Aggregate Report

- Users: 10 (tested: 10, skipped no strategy: 0)
- Required users: 10
- Agent state: users with metric 10, skipped 0
- Topic split: {'users_with_topic_strategy_row': 10, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 10, 'users_topic_split_valid_for_hard_gate': 10}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Semantic embedding gate: False
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: fail
- Hard pass: False
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace rows: 931

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | True |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
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

- Users positive/negative/zero: 6 / 4 / 0
- User delta median: 0.0017; IQR: 0.0094
- Pair-count distribution: {'1': 1, '2': 4, '4': 1, '5': 1, '8': 1, '10': 4, '11': 2, '13': 1, '16': 2, '17': 2, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '36': 3, '40': 1, '47': 1, '48': 1, '61': 1, '67': 2, '74': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | -0.0097 | 1 | 9 | -0.0059 | 0.0103 |
| `random` | -0.0015 | 5 | 5 | -0.0005 | 0.0143 |
| `temporal` | 0.0049 | 8 | 1 | 0.0124 | 0.0236 |
| `topic` | -0.0015 | 5 | 5 | -0.0005 | 0.0141 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: True (semantic_delta_near_zero)

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 931 | 0.5059 | 0.6928 | 0.1740 | 0.5464 | 0.1740 | -0.0002 | 0.0599 | 0.0118 |
| `baseline_c` | 931 | 0.4801 | 0.8099 | 0.1676 | 0.5930 | 0.0000 | -0.0050 | 0.1429 | 0.0333 |
| `baseline_b_best` | 931 | 0.6842 | 0.7637 | 0.0000 | 0.5860 | 0.0473 | -0.0122 | 0.0082 | 0.0053 |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 40 | 0.0494 | 0.0021 | 0.8761 | 0.6487 |
| `no_surface_profile` | 40 | 0.0454 | -0.0019 | 1.0000 | 1.0000 |
| `surface_only_default_agent` | 40 | 0.0504 | 0.0031 | 0.5596 | 0.5478 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.982287, 'train_default_l2': 0.192225, 'train_full_cosine': 0.982087, 'train_full_l2': 0.194733, 'train_wrong_user_cosine': 0.939578, 'train_wrong_user_l2': 0.38921}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0454 | 0.0473 | -0.0019 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.3761 | 0.4042 | -0.0281 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1148 | 0.1472 | -0.0325 | 1.0000 | False |
| `stylistic_similarity` | 0.8829 | 0.8799 | 0.0031 | 0.1162 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 0.5000 | False |
| `agent_state_similarity` | 0.9821 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0454 | 0.0512 | -0.0058 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.3761 | 0.2893 | 0.0868 | 0.0039 | True |
| `behavioral_similarity_action11` | 0.1148 | 0.0949 | 0.0199 | 0.4219 | False |
| `stylistic_similarity` | 0.8829 | 0.8585 | 0.0244 | 0.0010 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 0.5000 | False |
| `agent_state_similarity` | 0.9821 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0454, baseline_a=0.0473, p(vs_a)=1.0000, baseline_c=0.0512, p(vs_c)=1.0000
- `agent_state_similarity`: personality=0.9821, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.3761, baseline_a=0.4042, baseline_c=0.2893
- `behavioral_similarity_action11`: personality=0.1148, baseline_a=0.1472, baseline_c=0.0949
- `stylistic_similarity`: personality=0.8829, baseline_a=0.8799, baseline_c=0.8585
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | True | True |
| `random` | False | False | True | True |
| `temporal` | False | False | True | True |
| `topic` | False | False | False | True |
