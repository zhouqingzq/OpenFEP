# M4.1 Parameter Reference

## `uncertainty_sensitivity`
- Range: `0.0..1.0`
- Default: `0.65`
- Physical meaning: Sensitivity to ambiguity and incomplete evidence.
- Decision path: Higher values depress confidence and favor inspect-like actions under uncertainty.
- Observable relationship: Mapped through high-uncertainty confidence drop and inspect/scan preference.

## `error_aversion`
- Range: `0.0..1.0`
- Default: `0.7`
- Physical meaning: Penalty applied to options with elevated expected failure cost.
- Decision path: Higher values suppress risky actions and increase conservative recovery choices after error signals.
- Observable relationship: Mapped through risky-action rejection and post-error conservative switching.

## `exploration_bias`
- Range: `0.0..1.0`
- Default: `0.55`
- Physical meaning: Preference for information-seeking or novel actions.
- Decision path: Higher values increase query/scan/inspect selection in ambiguous contexts.
- Observable relationship: Mapped through unknown-option choice rate and lower repeated-choice streaks.

## `attention_selectivity`
- Range: `0.0..1.0`
- Default: `0.6`
- Physical meaning: Degree to which attention concentrates on the strongest evidence channels.
- Decision path: Higher values improve evidence-to-choice alignment and suppress distractor actions.
- Observable relationship: Mapped through dominant-feature attention share and evidence-aligned action wins.

## `confidence_gain`
- Range: `0.0..1.0`
- Default: `0.7`
- Physical meaning: Amplification from clean evidence separation into internal confidence.
- Decision path: Higher values raise confidence and commit rate when evidence becomes decisive.
- Observable relationship: Mapped through confidence-vs-evidence slope and high-evidence commit rate.

## `update_rigidity`
- Range: `0.0..1.0`
- Default: `0.65`
- Physical meaning: Resistance to changing internal state after observed error.
- Decision path: Higher values reduce learning-step magnitude and prolong strategy persistence after error.
- Observable relationship: Mapped through lower update magnitude and slower post-error switching.

## `resource_pressure_sensitivity`
- Range: `0.0..1.0`
- Default: `0.75`
- Physical meaning: Pressure response to low energy, low budget, high stress, or little time.
- Decision path: Higher values favor low-cost recovery and conservation actions under scarcity.
- Observable relationship: Mapped through low-cost-action rate and conservation trigger threshold under pressure.

## `virtual_prediction_error_gain`
- Range: `0.0..1.0`
- Default: `0.68`
- Physical meaning: Weight placed on imagined or counterfactual prediction-error signals.
- Decision path: Higher values amplify caution when simulated losses conflict with direct evidence.
- Observable relationship: Mapped through conflict-condition avoidance and counterfactual-loss driven conservative shifts.
