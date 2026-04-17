# M5.4 Validation Aggregate Report

- Users: 3 (tested: 3, skipped no strategy: 0)
- Required users: 3
- Agent state: users with metric 3, skipped 0
- Topic split: {'users_with_topic_strategy_row': 3, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 3, 'users_topic_split_valid_for_hard_gate': 3}
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
- Diagnostic trace rows: 75

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

- Users positive/negative/zero: 1 / 2 / 0
- User delta median: -0.0053; IQR: 0.0041
- Pair-count distribution: {'0': 2, '1': 2, '2': 1, '6': 1, '7': 1, '8': 1, '11': 2, '14': 2}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 |
| `random` | -0.0314 | 0 | 3 | -0.0259 | 0.0194 |
| `temporal` | 0.0556 | 3 | 0 | 0.0528 | 0.0224 |
| `topic` | -0.0314 | 0 | 3 | -0.0259 | 0.0194 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 75 | 0.4000 | 0.7600 | 0.1467 | 0.4042 | 0.0000 | -0.0144 | 0.0678 | 0.0541 |
| `baseline_c` | 75 | 0.3333 | 0.6667 | 0.0800 | 0.0309 | 0.0000 | 0.0522 | 0.1808 | 0.0106 |
| `baseline_b_best` | 75 | 0.4400 | 0.7867 | 0.0000 | 0.0296 | 0.0000 | 0.0385 | 0.2163 | 0.0210 |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 12 | 0.0607 | 0.0172 | 0.8333 | 0.5488 |
| `no_surface_profile` | 12 | 0.0393 | -0.0042 | 0.8333 | 0.3293 |
| `surface_only_default_agent` | 12 | 0.0837 | 0.0403 | 0.2761 | 0.5259 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.928198, 'train_default_l2': 0.446334, 'train_full_cosine': 0.928198, 'train_full_l2': 0.446334, 'train_wrong_user_cosine': 0.861327, 'train_wrong_user_l2': 0.693535}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0417 | 0.0434 | -0.0018 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.4552 | 0.4327 | 0.0225 | 0.6250 | False |
| `behavioral_similarity_action11` | 0.1600 | 0.0000 | 0.1600 | 0.5000 | False |
| `stylistic_similarity` | 0.9074 | 0.8967 | 0.0107 | 0.1250 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9282 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0417 | 0.0046 | 0.0371 | 0.1250 | False |
| `behavioral_similarity_strategy` | 0.4552 | 0.6309 | -0.1758 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1600 | 0.5714 | -0.4114 | 1.0000 | False |
| `stylistic_similarity` | 0.9074 | 0.8384 | 0.0691 | 0.1250 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9282 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0417, baseline_a=0.0434, p(vs_a)=1.0000, baseline_c=0.0046, p(vs_c)=0.1250
- `agent_state_similarity`: personality=0.9282, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.4552, baseline_a=0.4327, baseline_c=0.6309
- `behavioral_similarity_action11`: personality=0.1600, baseline_a=0.0000, baseline_c=0.5714
- `stylistic_similarity`: personality=0.9074, baseline_a=0.8967, baseline_c=0.8384
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
