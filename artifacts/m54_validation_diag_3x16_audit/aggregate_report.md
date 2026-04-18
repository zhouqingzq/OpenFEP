# M5.4 Stop-Bleed Notice

This historical artifact is fail-closed under m54_v4_stop_bleed. Current classifier labels are LLM-generated provisional data: usable for engineering/direction checks, but not for formal human-labeled acceptance. See aggregate_report.json / m54_acceptance.json for machine-readable blockers.

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
- Formal Baseline C gate: False
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
| reproducibility_gate | False |
| baseline_c_full_population_implant | False |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 2 / 1 / 0
- User delta median: 0.0027; IQR: 0.0545
- Pair-count distribution: {'0': 2, '1': 2, '2': 1, '6': 1, '7': 1, '8': 1, '11': 2, '14': 2}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 |
| `random` | 0.1197 | 2 | 1 | 0.0182 | 0.0769 |
| `temporal` | -0.0391 | 0 | 3 | -0.0257 | 0.0643 |
| `topic` | 0.1197 | 2 | 1 | 0.0182 | 0.0769 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False

| Baseline | rows | action agree | strategy agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 75 | 0.4667 | 0.7067 | 0.3694 | 0.0000 | -0.0180 | 0.0989 | 0.0247 |
| `baseline_c` | 75 | 0.4667 | 0.7067 | 0.5479 | 0.0000 | -0.0339 | 0.0989 | 0.0247 |
| `baseline_b_best` | 75 | 0.6533 | 0.6933 | 0.4736 | 0.0000 | -0.0186 | 0.0487 | 0.0365 |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 12 | 0.0355 | -0.0048 | 0.7727 | 0.5985 |
| `no_surface_profile` | 12 | 0.0515 | 0.0112 | 0.7608 | 0.2245 |
| `surface_only_default_agent` | 12 | 0.0417 | 0.0014 | 0.3416 | 0.5367 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 1.0, 'train_default_l2': 0.0, 'train_full_cosine': 1.0, 'train_full_l2': 0.0, 'train_wrong_user_cosine': 1.0, 'train_wrong_user_l2': 0.0}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0904 | 0.0403 | 0.0501 | 0.3750 | False |
| `behavioral_similarity_strategy` | 0.3271 | 0.4385 | -0.1114 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0554 | 0.2083 | -0.1529 | 1.0000 | False |
| `stylistic_similarity` | 0.8947 | 0.8923 | 0.0024 | 0.3750 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0904 | 0.0530 | 0.0374 | 0.6250 | False |
| `behavioral_similarity_strategy` | 0.3271 | 0.4385 | -0.1114 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0554 | 0.2083 | -0.1529 | 1.0000 | False |
| `stylistic_similarity` | 0.8947 | 0.8870 | 0.0077 | 0.2500 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0904, baseline_a=0.0403, p(vs_a)=0.3750, baseline_c=0.0530, p(vs_c)=0.6250
- `agent_state_similarity`: personality=1.0000, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.3271, baseline_a=0.4385, baseline_c=0.4385
- `behavioral_similarity_action11`: personality=0.0554, baseline_a=0.2083, baseline_c=0.2083
- `stylistic_similarity`: personality=0.8947, baseline_a=0.8923, baseline_c=0.8870
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
