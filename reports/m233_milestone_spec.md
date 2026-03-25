# M2.33 Milestone Spec

Title: Narrative Uncertainty Decomposition

Core requirement:
- Extract explicit unresolved narrative unknowns
- Generate bounded competing hypotheses
- Separate surface cues from latent causes
- Preserve only action-relevant uncertainty
- Make uncertainty causally affect downstream prediction, verification, subject state, and explanation
- Preserve structured outputs across snapshot round-trip

Expected concrete mechanisms:
- `NarrativeUncertaintyDecomposer`
- `NarrativeUnknown`
- `CompetingHypothesis`
- `LatentCauseCandidate`
- `SurfaceCue`
- `DecisionRelevanceMap`
- `NarrativeAmbiguityProfile`
- `UncertaintyDecompositionResult`

Required downstream effects:
- subject-state unresolved uncertainty surface
- prediction-ledger narrative uncertainty candidates
- verification prioritization for ambiguity-linked predictions
- explanation/runtime trace visibility

Strict audit focus:
- no fake uncertainty decomposition
- no unbounded hypothesis explosion
- no surface/latent conflation
- no persistence drop on snapshot restore
- no log-only implementation
