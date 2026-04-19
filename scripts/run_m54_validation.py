from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from segmentum.dialogue.lifecycle import ImplantationConfig
from segmentum.dialogue.validation.act_classifier import load_labeled_samples
from segmentum.dialogue.validation.pipeline import ValidationConfig, run_batch_validation
from segmentum.dialogue.validation.report import generate_report
from segmentum.dialogue.validation.splitter import SplitStrategy
from segmentum.dialogue.validation.statistics import scipy_wilcoxon_available


def _session_sort_key(session: dict) -> tuple[str, str]:
    start = str(session.get("start_time", ""))
    sid = str(session.get("session_id", ""))
    return start, sid


def _cap_sessions(user_dataset: dict, *, max_sessions_per_user: int | None = None) -> dict:
    if max_sessions_per_user is None:
        return user_dataset
    sessions = user_dataset.get("sessions", [])
    if not isinstance(sessions, list) or len(sessions) <= int(max_sessions_per_user):
        return user_dataset
    capped = dict(user_dataset)
    capped["sessions"] = sorted(
        [dict(item) for item in sessions if isinstance(item, dict)],
        key=_session_sort_key,
    )[: int(max_sessions_per_user)]
    profile = capped.get("profile")
    if isinstance(profile, dict):
        profile = dict(profile)
        profile["m54_max_sessions_per_user_applied"] = int(max_sessions_per_user)
        profile["m54_original_session_count"] = int(len(sessions))
        capped["profile"] = profile
    return capped


def _load_user_datasets(
    user_dir: Path,
    *,
    max_users: int | None = None,
    max_sessions_per_user: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(user_dir.glob("*.json")):
        if max_users is not None and len(rows) >= int(max_users):
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
            rows.append(_cap_sessions(payload, max_sessions_per_user=max_sessions_per_user))
    return rows


def _parse_strategies(raw: str) -> list[SplitStrategy]:
    out: list[SplitStrategy] = []
    for item in [piece.strip() for piece in raw.split(",") if piece.strip()]:
        out.append(SplitStrategy(item))
    return out


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_required_users_error(message: str) -> int | None:
    marker = "required="
    if "insufficient users for validation" not in message or marker not in message:
        return None
    tail = message.split(marker, 1)[1].strip()
    digits = []
    for ch in tail:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5.4 consistency validation.")
    parser.add_argument("--user-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m54_validation"))
    parser.add_argument("--strategies", type=str, default="random,temporal,partner,topic")
    parser.add_argument("--min-users", type=int, default=10)
    parser.add_argument("--pilot-users", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-train", type=Path, default=None)
    parser.add_argument("--classifier-gate", type=Path, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-sessions-per-user", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--diagnostic-trace", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        os.environ["SEGMENTUM_M54_PROGRESS"] = "1"
    datasets = _load_user_datasets(
        args.user_dir,
        max_users=args.max_users,
        max_sessions_per_user=args.max_sessions_per_user,
    )
    strategies = _parse_strategies(args.strategies)
    print(
        f"loaded {len(datasets)} users from {args.user_dir}"
        + (f" (max-users={args.max_users})" if args.max_users else "")
        + (f" (max-sessions-per-user={args.max_sessions_per_user})" if args.max_sessions_per_user else "")
    )
    classifier_train = load_labeled_samples(args.classifier_train) if args.classifier_train else []
    classifier_gate = load_labeled_samples(args.classifier_gate) if args.classifier_gate else []
    print(f"loaded classifier labels: train={len(classifier_train)}, gate={len(classifier_gate)}")
    if args.formal:
        if not args.classifier_train or not args.classifier_gate:
            raise ValueError("--formal requires --classifier-train and --classifier-gate")
        if args.max_users is not None and int(args.max_users) < int(args.min_users):
            raise ValueError("--formal requires --max-users >= --min-users")
        if args.max_sessions_per_user is not None and int(args.max_sessions_per_user) < 24:
            raise ValueError("--formal requires --max-sessions-per-user >= 24 so partner/topic holdouts remain viable")
        if os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise ValueError("--formal requires sentence embeddings; unset SEGMENTUM_USE_TFIDF_SEMANTIC")
        if not scipy_wilcoxon_available():
            raise ValueError("--formal requires scipy>=1.11")
        if set(item.value for item in strategies) != {"random", "temporal", "partner", "topic"}:
            raise ValueError("--formal requires strategies=random,temporal,partner,topic")
        if not args.diagnostic_trace:
            raise ValueError("--formal requires --diagnostic-trace so acceptance artifacts are auditable")
    config = ValidationConfig(
        strategies=strategies,
        min_users=int(args.min_users),
        pilot_user_count=int(args.pilot_users),
        seed=int(args.seed),
        implantation_config=ImplantationConfig(),
        classifier_train_samples=classifier_train,
        classifier_gate_samples=classifier_gate,
        classifier_dataset_origin=str(args.classifier_gate) if args.classifier_gate else "missing_formal_labels",
        formal=bool(args.formal),
        diagnostic_trace=bool(args.diagnostic_trace),
        max_cue_override_rate=0.35,
        require_classifier_provenance=True,
    )
    print("running M5.4 validation...")
    direction_auto_escalation: dict[str, object] = {"applied": False}
    try:
        reports = run_batch_validation(datasets, config)
    except ValueError as exc:
        required_users = _parse_required_users_error(str(exc))
        can_direction_escalate = (
            required_users is not None
            and not args.formal
            and args.max_users is not None
            and int(args.max_users) < int(required_users)
        )
        if not can_direction_escalate:
            raise
        expanded = _load_user_datasets(
            args.user_dir,
            max_users=int(required_users),
            max_sessions_per_user=args.max_sessions_per_user,
        )
        if len(expanded) < int(required_users):
            raise
        print(
            "pilot escalated required users; rerunning direction check "
            f"with {required_users} users instead of requested max-users={args.max_users}"
        )
        direction_auto_escalation = {
            "applied": True,
            "policy": "non-formal direction runs honor pilot escalation by rerunning with the required user count",
            "requested_max_users": int(args.max_users),
            "pilot_required_users": int(required_users),
            "rerun_user_count": int(len(expanded)),
            "stale_artifact_note": (
                "Earlier 10x32 outputs without this field should be treated as pre-final-patch "
                "or pilot-escalation stale artifacts."
            ),
        }
        datasets = expanded
        reports = run_batch_validation(datasets, config)
    print(f"validation completed for {len(reports)} users; writing report...")
    md_path = generate_report(reports, args.output)
    aggregate_path = args.output / "aggregate_report.json"
    agg = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    if direction_auto_escalation.get("applied"):
        agg["direction_auto_escalation"] = direction_auto_escalation
        _write_json(aggregate_path, agg)
        if md_path.exists():
            existing_md = md_path.read_text(encoding="utf-8")
            md_path.write_text(
                existing_md
                + "\n## Direction Auto-Escalation\n\n"
                + f"- Applied: {direction_auto_escalation['applied']}\n"
                + f"- Requested max users: {direction_auto_escalation['requested_max_users']}\n"
                + f"- Pilot required users: {direction_auto_escalation['pilot_required_users']}\n"
                + f"- Rerun user count: {direction_auto_escalation['rerun_user_count']}\n"
                + f"- Note: {direction_auto_escalation['stale_artifact_note']}\n",
                encoding="utf-8",
            )
    acceptance = {
        "milestone": "M5.4",
        "user_count": len(reports),
        "strategies": [item.value for item in strategies],
        "metric_version": agg.get("metric_version"),
        "artifact_rules_version": agg.get("artifact_rules_version"),
        "required_users": agg.get("required_users"),
        "formal_requested": bool(agg.get("formal_requested", False)),
        "hard_pass": agg.get("hard_pass"),
        "hard_pass_breakdown": agg.get("hard_pass_breakdown"),
        "pilot_gate": agg.get("pilot_gate"),
        "split_gate": agg.get("split_gate"),
        "partner_gate": agg.get("partner_gate"),
        "topic_gate": agg.get("topic_gate"),
        "reproducibility_gate": agg.get("reproducibility_gate"),
        "overall_conclusion": agg.get("overall_conclusion"),
        "formal_acceptance_eligible": agg.get("formal_acceptance_eligible"),
        "classifier_gate": agg.get("classifier_gate"),
        "classifier_evidence_tier": agg.get("classifier_evidence_tier"),
        "semantic_engine_gate": agg.get("semantic_engine_gate"),
        "statistical_gate": agg.get("statistical_gate"),
        "baseline_c_gate": agg.get("baseline_c_gate"),
        "baseline_c_builder": agg.get("baseline_c_builder"),
        "surface_ablation_gate": agg.get("surface_ablation_gate"),
        "formal_blockers": agg.get("formal_blockers"),
        "diagnostic_trace_gate": agg.get("diagnostic_trace_gate"),
        "agent_state_differentiation_gate": agg.get("agent_state_differentiation_gate"),
        "behavioral_majority_baseline_gate": agg.get("behavioral_majority_baseline_gate"),
        "direction_auto_escalation": agg.get("direction_auto_escalation", direction_auto_escalation),
        "aggregate_report_json": str(aggregate_path.as_posix()),
        "aggregate_report_md": str(md_path.as_posix()),
        "baseline_c_behavioral_failure_audit_json": agg.get("baseline_c_behavioral_failure_audit_json"),
        "diagnostic_trace_jsonl": agg.get("diagnostic_trace_jsonl"),
        "diagnostic_trace_rows": agg.get("diagnostic_trace_rows"),
    }
    acceptance_path = args.output / "m54_acceptance.json"
    _write_json(acceptance_path, acceptance)
    print(f"wrote {args.output} (including {acceptance_path})")


if __name__ == "__main__":
    main()

