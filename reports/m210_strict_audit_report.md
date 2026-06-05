# M2.10 Strict Audit Report

Final status: `REJECT_M210`

## Gates
- statistical_support: True
- longitudinal_stability: False
- profile_level_consistency: False
- artifact_freshness: True

## Statistical Evidence
- significant_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'action_entropy', 'survival_score']
- effect_metrics: ['caution_rate', 'exploration_rate', 'seek_contact_rate', 'action_entropy', 'survival_score', 'mean_conditioned_prediction_error']

## Stability Evidence
- profiles_passing: 0 / 5
- passed_profiles: []

## Failing Profiles
- neutral
- threat_sensitive
- social_approach
- exploratory
- rigid_cautious
