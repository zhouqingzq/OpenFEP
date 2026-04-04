from __future__ import annotations

"""Legacy compatibility wrapper for M4.2+ task-bundle evaluation.

Despite the module name, this file should not redefine M4.1 acceptance.
M4.1 is the interface layer. External task bundles belong to M4.2+ benchmark
and task-layer work.
"""

from .m41_external_task_eval import (
    downgraded_claims_inventory,
    evaluation_chain_audit,
    run_external_task_bundle_evaluation,
    run_minimal_external_task_validation,
    smoke_fixture_rejection_report,
)


LEGACY_SYNTHETIC_SCOPE = "same-framework synthetic holdout sidecars"


def acceptance_scope_note() -> dict[str, str]:
    return {
        "m41_acceptance_scope": "interface_layer",
        "m42_plus_scope": "benchmark_environment_and_task_layer",
        "legacy_synthetic_scope": LEGACY_SYNTHETIC_SCOPE,
        "note": "M4.1 acceptance is interface-layer only. External task bundles and recovery-on-task begin at M4.2+, while nearby synthetic inference sidecars remain non-acceptance diagnostics.",
    }


def external_validation_scope_note() -> dict[str, str]:
    return acceptance_scope_note()


__all__ = [
    "LEGACY_SYNTHETIC_SCOPE",
    "acceptance_scope_note",
    "downgraded_claims_inventory",
    "evaluation_chain_audit",
    "external_validation_scope_note",
    "run_external_task_bundle_evaluation",
    "run_minimal_external_task_validation",
    "smoke_fixture_rejection_report",
]
