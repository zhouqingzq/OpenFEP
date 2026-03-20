# M3 Readiness Repair Report

Generated: 2026-03-20
Generator: `evals/m3_readiness_evaluation.py`

## 1. Audit Problem Summary

- Strict follow-up metrics are now the default readiness baseline; legacy M2 metrics are retained only for comparison.
- Readiness conclusions are downgraded when evidence is inherited, claimed, or not replayed in the current round.
- Compression claims are separated from generic lifecycle activity.
- Review-family schema existence is separated from runtime-validated family coverage.

## 2. Strict vs Legacy Metric Inheritance

Strict metrics
- `ICI` = 1.0000 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)
- `EAA` = 0.9725 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)
- `MUR` = 1.0000 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)
- `PSSR` = 0.9997 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)
- `CAQ` = 1.0000 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)
- `VCUS` = 1.0000 (status=REVALIDATED_THIS_ROUND, origin=evals/m2_followup_repair.py:run_followup_evaluation, gating=True)

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
- `evidence_kind`: runtime_replay
- `fully_graduated`: true
- `missing_graduation_families`: none
- `limitations`: none
- Report rule: schema implemented != runtime coverage verified, and framework probes cannot satisfy runtime replay gates.

## 6. Current-Round Test Evidence

- Test evidence status: REVALIDATED_THIS_ROUND
- Test suite scope: targeted_readiness_tests_only
- Readiness targets: tests/test_m3_readiness.py, tests/test_memory.py, tests/test_counterfactual_artifact.py, tests/test_m23_ultimate_consolidation_loop.py
- Current-round passed count: 41
- Current-round failed count: 0
- Current-round summary: 41 passed in 0.82s
- Carried-forward unverified test claim present: false
- Historical regressions status: REVALIDATED_THIS_ROUND
- Historical regression checks: soak_regression_passed, snapshot_compatibility_passed, runtime_lifecycle_passed, runtime_family_coverage_passed
- Boundary: readiness targets are partial readiness checks, not a complete historical milestone regression proof.

## 7. Final Readiness Conclusion

- `pre_m3_gate_status`: READY_FOR_M3
- `pre_m3_gate_passed`: True
- `controlled_ready_status`: CONTROLLED_READY_VERIFIED
- `open_ready_status`: OPEN_READY_VERIFIED
- `final_recommendation`: READY_FOR_M3
- Rationale: Pre-M3 gate passed and all readiness gates, runtime evidence, and historical regressions are verified in the current round.
- Why this is more conservative: n/a

## 8. Next Evidence Needed

- Keep runtime lifecycle probes in the readiness generator so OPEN readiness remains reproducible from current-round evidence.
