# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 10
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v5_formal_evidence (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Classifier evidence tier: repo_fixture_smoke
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: fail
- Hard pass: False
- Formal blockers: ['agent_state_differentiation_failed', 'baseline_c_gate_failed', 'behavioral_majority_baseline_matches_or_beats_personality', 'classifier_fixture_only', 'diagnostic_trace_missing', 'metric_hard_pass_failed', 'partner_gate_failed', 'semantic_vs_baseline_c_failed', 'surface_ablation_missing', 'topic_gate_failed']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: False
- Diagnostic trace gate: False
- Agent-state differentiation gate: False
- Behavioral majority baseline gate: False
- Surface ablation gate: False
- Diagnostic trace rows: 0

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | True |
| semantic_similarity_vs_baseline_a_significant_better | False |
| semantic_similarity_vs_baseline_c_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
| formal_acceptance_eligible | False |
| semantic_embedding_gate | True |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | False |
| baseline_c_leave_one_out_population_average | False |
| diagnostic_trace_gate | False |
| agent_state_differentiation_gate | False |
| behavioral_majority_baseline_gate | False |
| surface_ablation_gate | False |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 7 / 8 / 0
- User delta median: -0.0006; IQR: 0.0176
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0043 | 7 | 8 | -0.0083 | 0.0220 |
| `random` | -0.0053 | 6 | 9 | -0.0073 | 0.0245 |
| `temporal` | -0.0042 | 7 | 8 | -0.0069 | 0.0349 |
| `topic` | -0.0007 | 8 | 7 | 0.0024 | 0.0271 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 0 | {} | {} |
| `baseline_c` | 0 | {} | {} |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {}

## Debug Readiness Gate

- Passed: False
- Checks: {'train_default_l2_positive': False, 'train_wrong_user_l2_positive': False, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.3991 | 0.4006 | -0.0015 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.1281 | 0.1184 | 0.0097 | 0.2915 | False |
| `behavioral_similarity_action11` | 0.0246 | 0.0247 | -0.0001 | 1.0000 | False |
| `stylistic_similarity` | 0.8867 | 0.8898 | -0.0031 | 1.0000 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 0.1587 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.3991 | 0.4020 | -0.0029 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.1281 | 0.1187 | 0.0094 | 0.2915 | False |
| `behavioral_similarity_action11` | 0.0246 | 0.0247 | -0.0001 | 1.0000 | False |
| `stylistic_similarity` | 0.8867 | 0.8905 | -0.0038 | 1.0000 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 0.1587 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.3991, baseline_a=0.4006, p(vs_a)=1.0000, baseline_c=0.4020, p(vs_c)=1.0000
- `behavioral_similarity_strategy`: personality=0.1281, baseline_a=0.1184, p(vs_a)=0.2915, baseline_c=0.1187, p(vs_c)=0.2915
- `agent_state_similarity`: personality=1.0000, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_action11`: personality=0.0246, baseline_a=0.0247, baseline_c=0.0247
- `stylistic_similarity`: personality=0.8867, baseline_a=0.8898, baseline_c=0.8905
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | True | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
