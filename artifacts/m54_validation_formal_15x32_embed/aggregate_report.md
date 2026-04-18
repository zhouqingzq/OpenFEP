# M5.4 Stop-Bleed Notice

This historical artifact is fail-closed under m54_v4_stop_bleed. Repo-tracked classifier fixtures are smoke-only and do not support formal acceptance. See aggregate_report.json / m54_acceptance.json for machine-readable blockers.

# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 10
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Semantic embedding gate: True
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
| semantic_embedding_gate | True |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | True |
| baseline_c_full_population_implant | True |

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
- `agent_state_similarity`: personality=1.0000, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.1281, baseline_a=0.1184, baseline_c=0.1187
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
