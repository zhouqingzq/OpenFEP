# M2.34 Parallel-Probe Acceptance Rationale

## Position

`M2.34` should be audited as a finite-resource active-inference milestone, not as a single-cheapest-probe milestone.

Under that framing, bounded parallel inquiry can be acceptable because the system's objective is to reduce uncertainty and downstream free energy within available budget, not to minimize action count as an independent primary objective.

The more human-like interpretation is bounded approximate Bayesian inference: under limited time, working memory, and energy, the system performs weighted ranking over candidate probes and may test them in stages or small batches rather than solving for a unique globally optimal single probe.

## Why Parallel Plans Can Be Legitimate

1. The implementation applies an explicit active-plan budget through `max_active_plans`, so concurrency is capped rather than unbounded.
2. Candidate ranking already trades off:
   - expected information gain,
   - falsification opportunity,
   - decision relevance,
   - goal alignment,
   - risk,
   - cost.
3. Lower-ranked options are not silently merged into one opaque score; they remain inspectable as queued, deferred, or secondary active plans.
4. Downstream systems consume the ranked portfolio:
   - the prediction ledger carries experiment-linked predictions,
   - verification can prioritize experiment-linked targets,
   - subject state surfaces active inquiries,
   - decision scoring receives experiment bias.

## Observed Repository Evidence

- Stable social ambiguity prefers `seek_contact` as the top-ranked active probe.
- Social destabilization defers `seek_contact` for risk and promotes `scan` to active status.
- Safety-oriented threat ambiguity prefers `scan`.
- Governance filtering removes unregistered actions from candidate sets.
- Snapshot restore preserves experiment-plan state.

These behaviors show that the layer is performing bounded inquiry design rather than producing decorative explanations.

## Important Audit Clarification

The strict question should be:

`Does the system produce a bounded, value-ranked experiment portfolio that approximates useful inference under finite resources and is causally consumed downstream?`

It should not be:

`Does the system always collapse each uncertainty target to exactly one cheapest action?`

The first question matches active inference under budget. The second imposes a stronger optimization target than the current architecture claims.

## Residual Caution

Allowing bounded parallelism does not mean every concurrent plan is automatically justified. A stronger future gate could require that multiple active plans for the same target provide non-redundant evidence channels or meaningfully different falsification opportunities.

That refinement would strengthen `M2.34`, but it is a higher bar than simply demonstrating bounded parallel experiment design.
