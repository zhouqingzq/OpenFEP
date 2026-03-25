# M2.34 Milestone Specification

## Title

`M2.34: Narrative Hypothesis And Experiment Design`

## Scope

- Convert decision-relevant narrative ambiguity into bounded experiment design rather than leaving competing explanations as descriptive text.
- Derive observable predictions from competing hypotheses and tie those predictions to governed inquiry actions.
- Support resource-bounded approximate Bayesian-style ranking over candidate inquiries rather than claiming a globally optimal single probe.
- Support staged inquiry: when multiple high-value probes fit within current action and risk limits they may be activated in bounded parallel, and when they do not fit they may be queued, deferred, or tried in batches.
- Make experiment design causally affect prediction carry-forward, verification priority, subject-state inquiry focus, workspace emphasis, and decision explanation.
- Preserve experiment-design state across snapshot save/restore.

## Non-Goals

- Forcing exactly one inquiry action per uncertainty target in all cases.
- Globally minimizing resource expenditure independent of expected information gain.
- Proving that the chosen action is the unique globally optimal single probe over the full action space.
- Treating every candidate as executable without governance, risk, or budget screening.

## Acceptance Gates

- `competition_translation`: competing narrative hypotheses become explicit discrimination targets and derived predictions.
- `bounded_parallelism`: active experiments remain bounded by configured budget and action-registry governance; queued/deferred states exist when budget or risk pressure blocks immediate execution.
- `value_ranking`: candidates are ranked by a bounded approximate-inference trade-off over information gain, falsification value, cost, risk, and goal alignment instead of raw narrative salience alone.
- `downstream_causality`: experiment design changes at least one downstream surface among prediction ledger, verification targeting, subject-state inquiry focus, workspace focus, or action scoring.
- `governance`: only registered actions may appear as executable candidates.
- `snapshot_roundtrip`: plans, rankings, and inquiry focus survive runtime snapshot restore.
- `artifact_freshness`: milestone artifacts and acceptance report are generated in the current audit round.

## Canonical Files

- `segmentum/narrative_experiment.py`
- `segmentum/agent.py`
- `segmentum/prediction_ledger.py`
- `segmentum/verification.py`
- `segmentum/subject_state.py`
- `tests/test_m234_experiment_design.py`

## Required Evidence

- Canonical trace showing competition-to-experiment translation.
- Evidence that stable social ambiguity prefers `seek_contact`, destabilized social ambiguity defers contact and promotes `scan`, and safety ambiguity prefers `scan`.
- Evidence that active experiment plans are budget-bounded and governance-bounded rather than unbounded.
- Snapshot round-trip evidence preserving experiment design state.
- Machine-readable acceptance report in `reports/m234_acceptance_report.json`.

## Audit Focus

- No decorative experiment layer that fails to alter downstream planning or verification.
- No unbounded candidate activation beyond configured budget.
- No governance bypass that promotes unregistered actions into executable probes.
- No loss of experiment state on snapshot restore.
- Multi-plan concurrency is not itself a failure if it is bounded, ranked, causally justified, and consistent with staged finite-resource inquiry rather than an exact-optimal single-probe fantasy.
