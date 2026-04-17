# M5.4 Validation Aggregate Report

- Users: 5 (tested: 5, skipped no strategy: 0)
- Required users: 5
- Agent state: users with metric 5, skipped 0
- Topic split: {'users_with_topic_strategy_row': 5, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 5, 'users_topic_split_valid_for_hard_gate': 5}
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
- Diagnostic trace rows: 237

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
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

- Users positive/negative/zero: 3 / 2 / 0
- User delta median: 0.0071; IQR: 0.0148
- Pair-count distribution: {'0': 2, '1': 4, '2': 2, '3': 1, '6': 1, '7': 1, '8': 1, '11': 2, '14': 2, '16': 2, '56': 1, '67': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0574 | 2 | 0 | 0.0000 | 0.0332 |
| `random` | -0.0191 | 0 | 4 | -0.0230 | 0.0244 |
| `temporal` | 0.1124 | 4 | 1 | 0.0528 | 0.0533 |
| `topic` | -0.0191 | 0 | 4 | -0.0230 | 0.0244 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 237 | 0.5148 | 0.8987 | 0.1899 | 0.4827 | 0.0000 | 0.0097 | 0.0571 | 0.0259 |
| `baseline_c` | 237 | 0.5570 | 0.8608 | 0.1561 | 0.0220 | 0.0000 | 0.0547 | 0.0418 | 0.0061 |
| `baseline_b_best` | 237 | 0.2405 | 0.8523 | 0.0000 | 0.0232 | 0.0000 | 0.0270 | 0.1371 | 0.0066 |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 20 | 0.1098 | 0.0748 | 0.8358 | 0.5570 |
| `no_surface_profile` | 20 | 0.0323 | -0.0027 | 0.8783 | 0.3229 |
| `surface_only_default_agent` | 20 | 0.1212 | 0.0862 | 0.4373 | 0.5428 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.929382, 'train_default_l2': 0.4365, 'train_full_cosine': 0.928951, 'train_full_l2': 0.438591, 'train_wrong_user_cosine': 0.922267, 'train_wrong_user_l2': 0.428164}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0679 | 0.0350 | 0.0329 | 0.2188 | False |
| `behavioral_similarity_strategy` | 0.6108 | 0.5919 | 0.0189 | 0.4375 | False |
| `behavioral_similarity_action11` | 0.0960 | 0.0000 | 0.0960 | 0.5000 | False |
| `stylistic_similarity` | 0.8993 | 0.8876 | 0.0117 | 0.0312 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9290 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0679 | 0.0059 | 0.0620 | 0.0312 | True |
| `behavioral_similarity_strategy` | 0.6108 | 0.7020 | -0.0911 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0960 | 0.5448 | -0.4488 | 1.0000 | False |
| `stylistic_similarity` | 0.8993 | 0.8277 | 0.0716 | 0.0625 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9290 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0679, baseline_a=0.0350, p(vs_a)=0.2188, baseline_c=0.0059, p(vs_c)=0.0312
- `agent_state_similarity`: personality=0.9290, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.6108, baseline_a=0.5919, baseline_c=0.7020
- `behavioral_similarity_action11`: personality=0.0960, baseline_a=0.0000, baseline_c=0.5448
- `stylistic_similarity`: personality=0.8993, baseline_a=0.8876, baseline_c=0.8277
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
