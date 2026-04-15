# M2.10 Strict Audit Report

Final status: `ACCEPT_M210`

## Gates
- statistical_support: True
- longitudinal_stability: True
- profile_level_consistency: True
- artifact_freshness: True

## Statistical Evidence
- significant_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'mean_conditioned_prediction_error']
- effect_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'action_entropy', 'survival_score', 'mean_conditioned_prediction_error', 'narrative_consistency_proxy']

## Stability Evidence
- profiles_passing: 5 / 5
- passed_profiles: ['neutral', 'threat_sensitive', 'social_approach', 'exploratory', 'rigid_cautious']
