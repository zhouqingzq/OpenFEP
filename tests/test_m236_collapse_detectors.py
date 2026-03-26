from __future__ import annotations

from segmentum.m236_open_continuity_trial import (
    AdaptationMetrics,
    CollapseDetector,
    IdentityRetentionMetrics,
    InquiryStabilityMetrics,
    OrganismTrialMetrics,
)


def test_collapse_detector_fires_on_obvious_pathologies() -> None:
    detector = CollapseDetector()
    snapshots = [
        {
            "choice": "rest",
            "active_inquiry_targets": 0,
            "requires_inquiry": True,
            "protected_anchor_bias": 0.92,
            "unresolved_conflicts": 5,
        }
        for _ in range(6)
    ]
    metrics = OrganismTrialMetrics(
        inquiry_stability=InquiryStabilityMetrics(
            mean_active_targets=0.0,
            active_target_stability=0.0,
            inquiry_churn_rate=0.0,
            useful_information_gain=0.0,
            low_value_suppression_rate=0.0,
            delayed_evidence_persistence=0.0,
            recovery_after_inconclusive=0.0,
            inquiry_collapse_detected=True,
        ),
        identity_retention=IdentityRetentionMetrics(
            continuity_mean=0.40,
            continuity_min=0.30,
            commitment_retention=0.1,
            anchor_stability=0.1,
            chapter_transition_coherence=0.1,
            restart_consistency=0.2,
            bounded_drift_score=0.1,
        ),
        adaptation=AdaptationMetrics(
            personality_drift=0.8,
            value_drift=0.7,
            commitment_drift=0.8,
            inquiry_policy_drift=0.9,
            social_trust_drift=0.7,
            continuity_anchor_drift=0.8,
            adaptive_revision_score=0.1,
        ),
        maintenance_inquiry_coupling=0.0,
        social_verification_coupling=0.0,
        trace_action_coupling=0.0,
        subject_state_coupling=0.0,
        conflict_persistence_phases=5,
        reopened_conflict_count=0,
        reconciled_conflict_count=0,
        trace_reactivation_events=0,
        safe_evidence_trace_reduction=0.0,
        verification_resolution_rate=0.0,
        adaptive_revision_observed=False,
    )
    phase_summaries = [{"maintenance_pressure_mean": 0.9}]

    findings = detector.detect(snapshots=snapshots, metrics=metrics, phase_summaries=phase_summaries)
    kinds = {item["kind"] for item in findings}

    assert "action_collapse" in kinds
    assert "inquiry_collapse" in kinds
    assert "identity_collapse" in kinds
    assert "uncontrolled_drift" in kinds
    assert "trace_explosion" in kinds
    assert "conflict_backlog_explosion" in kinds

