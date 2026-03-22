# M2.10 Strict Audit Report

Final status: `REJECT_M210`

## Gates
- statistical_support: True
- longitudinal_stability: False
- profile_level_consistency: False
- artifact_freshness: True

## Statistical Evidence
- significant_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'action_entropy', 'survival_score', 'mean_conditioned_prediction_error', 'narrative_consistency_proxy']
- effect_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'action_entropy', 'survival_score', 'mean_conditioned_prediction_error', 'narrative_consistency_proxy']

## Stability Evidence
- profiles_passing: 3 / 5
- passed_profiles: ['neutral', 'threat_sensitive', 'rigid_cautious']

## Failing Profiles
- social_approach
- exploratory
