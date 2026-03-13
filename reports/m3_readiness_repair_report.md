# M3 Readiness Repair Report

Generated: 2026-03-13
Generator: `evals/m3_readiness_evaluation.py`

## 1. Audit Problem Summary

- Strict follow-up metrics are now the default readiness baseline; legacy M2 metrics are retained only for comparison.
- Readiness conclusions are downgraded when evidence is inherited, claimed, or not replayed in the current round.
- Compression claims are separated from generic lifecycle activity.
- Review-family schema existence is separated from runtime-validated family coverage.

## 2. Strict vs Legacy Metric Inheritance

Strict metrics
- `ICI` = 1.0000 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)
- `EAA` = 0.9725 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)
- `MUR` = 1.0000 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)
- `PSSR` = 0.9997 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)
- `CAQ` = 1.0000 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)
- `VCUS` = 1.0000 (status=INHERITED_STRICT_BASELINE, origin=reports/m2_followup_metrics.json:new_metrics, gating=True)

Legacy metrics (non-gating, comparison only)
- `ICI` = 1.0000 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)
- `EAA` = 1.0000 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)
- `MUR` = 1.0000 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)
- `PSSR` = 0.9997 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)
- `CAQ` = 1.0000 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)
- `VCUS` = 1.0000 (status=LEGACY_FOR_COMPARISON_ONLY, non_gating=True)

## 3. Evidence Status And Downgrade Rules

- `INHERITED_STRICT_BASELINE`: carried from the stricter M2 follow-up run and allowed for readiness gating.
- `LEGACY_FOR_COMPARISON_ONLY`: retained only as historical comparison and never used for gating.
- `CLAIMED_BUT_NOT_REVALIDATED`: visible in the artifact, but not upgraded to verified fact in this round.
- Automatic downgrade triggers: strict/legacy mismatch in gating, missing current-round test evidence, and insufficient runtime family validation breadth.

## 4. Compression vs Lifecycle Evidence

- `lifecycle_verification_status`: REVALIDATED_THIS_ROUND
- `lifecycle_activity_observed`: true
- `compression_specifically_verified`: true
- `compression_removed_count`: 4
- `archived_count`: 1
- `pruned_count`: 4
- `compressed_cluster_count`: 1
- `probe_results_recorded`: 3
- Report rule: memory reduction is not treated as compression verification unless `compression_removed_count > 0` and `compressed_cluster_count > 0`.

## 5. Family Coverage Boundary

- `family_schema_count`: 4
- `runtime_validated_family_count`: 4
- `family_coverage_status`: RUNTIME_DIVERSITY_VALIDATED
- Report rule: schema implemented != runtime coverage verified.

## 6. Current-Round Test Evidence

- Test evidence status: REVALIDATED_THIS_ROUND
- Current-round passed count: 34
- Current-round failed count: 0
- Current-round summary: 34 passed in 0.31s
- Carried-forward unverified test claim present: false

## 7. Final Readiness Conclusion

- `controlled_ready_status`: CONTROLLED_READY_VERIFIED
- `open_ready_status`: OPEN_READY_VERIFIED
- `final_recommendation`: OPEN_READY_VERIFIED
- Rationale: Controlled readiness is verified and lifecycle/compression runtime evidence is revalidated this round.
- Why this is more conservative: n/a

## 8. Next Evidence Needed

- Keep runtime lifecycle probes in the readiness generator so OPEN readiness remains reproducible from current-round evidence.
