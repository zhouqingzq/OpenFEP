from __future__ import annotations

from typing import Any

from .m47_reacceptance import REGRESSION_TARGETS, build_m47_reacceptance_report


LEGACY_M47_ACCEPTANCE_DATA_NOTICE = (
    "Legacy M4.7 acceptance payload builder. Historical only; not the primary evidence chain."
)


def build_probe_catalog() -> dict[str, object]:
    return {
        "boundary": [
            {"id": "state_vector_boundary", "gate": "state_vector_dynamics"},
            {"id": "salience_boundary", "gate": "salience_dynamic_regulation"},
            {"id": "style_boundary", "gate": "cognitive_style_memory_integration"},
            {"id": "threat_learning_boundary", "gate": "behavioral_scenario_A_threat_learning"},
            {"id": "interference_boundary", "gate": "behavioral_scenario_B_interference"},
            {"id": "consolidation_boundary", "gate": "behavioral_scenario_C_consolidation"},
            {"id": "subtypes_boundary", "gate": "long_term_subtypes"},
            {"id": "identity_boundary", "gate": "identity_continuity_retention"},
            {"id": "misattribution_boundary", "gate": "behavioral_scenario_E_natural_misattribution"},
            {"id": "integration_boundary", "gate": "integration_interface"},
            {"id": "regression_boundary", "gate": "regression"},
        ],
        "regression_targets": list(REGRESSION_TARGETS),
    }


def build_m47_acceptance_payload(*, include_regressions: bool = False) -> dict[str, Any]:
    report = build_m47_reacceptance_report(include_regressions=include_regressions)
    return {
        "artifact_lineage": "legacy_self_attested_acceptance",
        "legacy_notice": LEGACY_M47_ACCEPTANCE_DATA_NOTICE,
        "probe_catalog": build_probe_catalog(),
        "gate_summaries": report["gate_summaries"],
        "evidence_records": report["evidence_records"],
        "regression_policy": report["regression_policy"],
    }
