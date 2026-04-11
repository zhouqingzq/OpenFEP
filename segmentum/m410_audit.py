from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .memory_consolidation import (
    compress_episodic_cluster_to_semantic_skeleton,
    constrained_replay,
)
from .memory_encoding import EncodingDynamics, EncodingDynamicsInput
from .memory_model import MemoryClass, MemoryEntry
from .memory_store import MemoryStore
from .runtime import SegmentRuntime


GATE_ORDER = (
    "encoding_dynamics_not_keywords",
    "attention_budget_competition",
    "heuristic_fallback_auditable",
    "semantic_centroid_consolidation",
    "replay_reencoding",
    "default_path_evidence",
    "milestone_docs_superseded",
)

REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path("artifacts")
M410_REPORT_PATH = REPORTS_DIR / "m410_acceptance_report.json"
M410_EVIDENCE_PATH = ARTIFACTS_DIR / "m410_dynamics_evidence.json"
M410_SUMMARY_PATH = REPORTS_DIR / "m410_acceptance_summary.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_m48_behavioral_inheritance_pytest() -> dict[str, Any]:
    root = _repo_root()
    targets = [
        root / "tests" / "test_m48_ablation_contrast.py",
        root / "tests" / "test_m48_acceptance.py",
    ]
    cmd = [sys.executable, "-m", "pytest", *[str(p) for p in targets], "-q", "--tb=no"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        return {"exit_code": completed.returncode, "argv": cmd}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"exit_code": -1, "argv": cmd, "error": str(exc)}


def _source_path(relative: str) -> Path:
    return _repo_root() / relative


def _keyword_static_check() -> dict[str, object]:
    path = _source_path("segmentum/memory_encoding.py")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("class FallbackHeuristic"))
    end = len(lines)
    for i, line in enumerate(lines[start + 1 :], start=start + 1):
        if line and not line.startswith((" ", "\t", "@")) and not line.startswith("#"):
            end = i
            break
    tokens = (
        "THREAT_KEYWORDS",
        "SOCIAL_KEYWORDS",
        "REWARD_KEYWORDS",
        "FIRST_PERSON_TOKENS",
        "TASK_ONLY_KEYWORDS",
        "OUTCOME_NEGATIVE_KEYWORDS",
        "OUTCOME_POSITIVE_KEYWORDS",
    )
    violations: list[dict[str, object]] = []
    for i, line in enumerate(lines):
        if any(token in line for token in tokens) and not (start <= i < end):
            violations.append({"line": i + 1, "text": line.strip()})
    return {
        "path": str(path),
        "fallback_block": {"start": start + 1, "end": end},
        "violations": violations,
        "passed": not violations,
    }


def _template_static_check() -> dict[str, object]:
    path = _source_path("segmentum/memory_consolidation.py")
    content = path.read_text(encoding="utf-8")
    banned = ("Semantic skeleton from", "Inferred pattern from", "Replay hypothesis from")
    hits = [item for item in banned if item in content]
    return {"path": str(path), "banned_hits": hits, "passed": not hits}


def _doc_supersession_check() -> dict[str, object]:
    docs = [
        "prompts/m45_work_prompt.md",
        "prompts/m46_work_prompt.md",
        "prompts/m46_acceptance_criteria.md",
        "prompts/m47_work_prompt.md",
        "prompts/m47_acceptance_criteria.md",
    ]
    missing: list[str] = []
    for relative in docs:
        content = _source_path(relative).read_text(encoding="utf-8")
        if "superseded by M4.10" not in content:
            missing.append(relative)
    return {"checked": docs, "missing": missing, "passed": not missing}


def build_budget_competition_evidence() -> dict[str, object]:
    signals = [
        EncodingDynamicsInput(0.05, 0.05, 0.20, attention_budget=0.85, requested_budget=1.0),
        EncodingDynamicsInput(0.08, 0.08, 0.22, attention_budget=0.85, requested_budget=1.0),
        EncodingDynamicsInput(0.85, 0.80, 0.70, attention_budget=0.85, requested_budget=1.0),
        EncodingDynamicsInput(0.75, 0.70, 0.65, attention_budget=0.85, requested_budget=1.0),
    ]
    constrained = EncodingDynamics.score_many(signals)
    unlimited = EncodingDynamics.score_many(
        [
            EncodingDynamicsInput(
                item.prediction_error,
                item.surprise,
                item.arousal,
                attention_budget=10.0,
                requested_budget=1.0,
            )
            for item in signals
        ]
    )
    constrained_strengths = [item.encoding_strength for item in constrained]
    unlimited_strengths = [item.encoding_strength for item in unlimited]
    high_retained = min(constrained_strengths[2:]) > max(constrained_strengths[:2])
    low_shrunk = max(constrained_strengths[:2]) < max(unlimited_strengths[:2])
    all_unlimited_encoded = all(value > 0.0 for value in unlimited_strengths)
    runtime = build_runtime_budget_competition_evidence()
    return {
        "constrained_strengths": constrained_strengths,
        "unlimited_strengths": unlimited_strengths,
        "constrained_audit": [item.to_dict() for item in constrained],
        "unlimited_audit": [item.to_dict() for item in unlimited],
        "runtime_competition": runtime,
        "passed": bool(high_retained and low_shrunk and all_unlimited_encoded and runtime["passed"]),
    }


def build_runtime_budget_competition_evidence(*, seed: int = 4) -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        runtime = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "segment_state.json",
            seed=seed,
            reset=True,
            memory_enabled=True,
        )
        runtime.step(verbose=False)
        diagnostics = runtime.agent.last_decision_diagnostics
    ranked = list(diagnostics.ranked_options if diagnostics is not None else [])
    event_inputs: list[dict[str, object]] = []
    signals: list[EncodingDynamicsInput] = []
    for option in ranked:
        prediction_error = max(0.01, float(option.predicted_error))
        surprise = max(
            0.01,
            float(option.action_ambiguity),
            1.0 - float(option.preferred_probability),
        )
        arousal = min(
            1.0,
            max(0.05, (float(option.risk) / 4.0) + (max(0.0, -float(option.value_score)) * 0.20)),
        )
        event_inputs.append(
            {
                "event_id": option.choice,
                "prediction_error": prediction_error,
                "surprise": surprise,
                "arousal": arousal,
            }
        )
        signals.append(
            EncodingDynamicsInput(
                prediction_error,
                surprise,
                arousal,
                attention_budget=0.85,
                requested_budget=1.0,
            )
        )
    constrained = EncodingDynamics.score_many(signals)
    unlimited = EncodingDynamics.score_many(
        [
            EncodingDynamicsInput(
                item.prediction_error,
                item.surprise,
                item.arousal,
                attention_budget=10.0,
                requested_budget=1.0,
            )
            for item in signals
        ]
    )
    thresholds = [0.001, 0.05, 0.10]
    constrained_curve = {
        str(threshold): sum(1 for item in constrained if item.encoding_strength >= threshold)
        for threshold in thresholds
    }
    unlimited_curve = {
        str(threshold): sum(1 for item in unlimited if item.encoding_strength >= threshold)
        for threshold in thresholds
    }
    constrained_events = [
        {"event_id": event["event_id"], **result.to_dict()}
        for event, result in zip(event_inputs, constrained)
    ]
    unlimited_events = [
        {"event_id": event["event_id"], **result.to_dict()}
        for event, result in zip(event_inputs, unlimited)
    ]
    return {
        "seed": seed,
        "tick": 1,
        "event_count": len(event_inputs),
        "events": event_inputs,
        "constrained_events": constrained_events,
        "unlimited_events": unlimited_events,
        "constrained_retention_curve": constrained_curve,
        "unlimited_retention_curve": unlimited_curve,
        "passed": bool(
            len(event_inputs) >= 2
            and any(float(item["attention_budget_denied"]) > 0.0 for item in constrained_events)
            and any(float(item["attention_budget_granted"]) > 0.0 for item in constrained_events)
            and constrained_curve != unlimited_curve
        ),
    }


def build_replay_drift_evidence() -> dict[str, object]:
    entries = [
        MemoryEntry(
            id=f"drift-ep-{index}",
            content=f"drift episode {index}",
            semantic_tags=["m410", "cluster"],
            context_tags=["replay"],
            state_vector=[float(index), float(index + 1)],
            salience=0.42,
            arousal=0.55,
            encoding_attention=0.9,
            novelty=0.25,
        )
        for index in range(5)
    ]
    semantic = compress_episodic_cluster_to_semantic_skeleton(entries)
    before = list(semantic.centroid or [])
    entries[-1].state_vector = [12.0, 13.0]
    entries[-1].salience = 0.99
    store = MemoryStore(entries=[*entries, semantic])
    touched = constrained_replay(store, random.Random(7), batch_size=1)
    after = list(semantic.centroid or [])
    shift = sum(abs(left - right) for left, right in zip(before, after))
    return {
        "centroid_before": before,
        "centroid_after": after,
        "centroid_shift_l1": shift,
        "touched_ids": [entry.id for entry in touched],
        "replay_trail": [
            {
                "entry_id": entry.id,
                "replay_second_pass_error": entry.replay_second_pass_error,
                "salience_delta": entry.salience_delta,
                "retention_adjustment": entry.retention_adjustment,
            }
            for entry in touched
        ],
        "semantic": semantic.to_dict(),
        "passed": bool(shift > 0.1 and touched and touched[0].salience_delta is not None),
    }


def _default_path_entry_evidence(
    entries: list[MemoryEntry],
    *,
    seed: int,
    cycles: int,
    sleep_trace_has_m410: bool,
) -> dict[str, object]:
    valid_sources = {"dynamics", "heuristic"}
    encoded_episodes = [entry for entry in entries if entry.memory_class is MemoryClass.EPISODIC]
    source_counts: dict[str, int] = {}
    invalid_source_ids: list[str] = []
    for entry in encoded_episodes:
        metadata = dict(entry.compression_metadata or {})
        source_raw = metadata.get("encoding_source")
        source = str(source_raw) if source_raw in valid_sources else "missing"
        source_counts[source] = source_counts.get(source, 0) + 1
        if source not in valid_sources:
            invalid_source_ids.append(entry.id)
    dynamics_count = source_counts.get("dynamics", 0)
    heuristic_count = source_counts.get("heuristic", 0)
    encoded_count = len(encoded_episodes)
    semantic_entries = [
        entry
        for entry in entries
        if entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}
    ]
    semantic_evidence = [
        {
            "entry_id": entry.id,
            "memory_class": entry.memory_class.value,
            "consolidation_source": entry.consolidation_source,
            "has_centroid": bool(entry.centroid),
            "has_residual": entry.residual_norm_mean is not None and entry.residual_norm_var is not None,
            "support_ids": list(entry.support_ids or []),
            "content_role": dict(entry.compression_metadata or {}).get("content_role"),
        }
        for entry in semantic_entries
    ]
    semantic_dynamic = [
        entry.id
        for entry in semantic_entries
        if entry.consolidation_source == "dynamics"
        and entry.centroid
        and entry.residual_norm_mean is not None
        and entry.residual_norm_var is not None
        and entry.support_ids
    ]
    semantic_missing_centroid = [entry.id for entry in semantic_entries if not entry.centroid]
    semantic_missing_residual = [
        entry.id
        for entry in semantic_entries
        if entry.residual_norm_mean is None or entry.residual_norm_var is None
    ]
    replay_trails = [
        entry.id
        for entry in entries
        if entry.replay_second_pass_error is not None and entry.salience_delta is not None
    ]
    replay_refreshes = []
    for entry in semantic_entries:
        refresh = dict(dict(entry.compression_metadata or {}).get("m410_replay_refresh", {}) or {})
        if refresh:
            replay_refreshes.append({"entry_id": entry.id, **refresh})
    replay_refresh_updates = [
        item["entry_id"]
        for item in replay_refreshes
        if item.get("centroid_before") != item.get("centroid_after")
        or item.get("residual_norm_mean_before") != item.get("residual_norm_mean_after")
        or item.get("residual_norm_var_before") != item.get("residual_norm_var_after")
    ]
    heuristic_share = heuristic_count / encoded_count if encoded_count else 1.0
    return {
        "seed": seed,
        "cycles": cycles,
        "entry_count": len(entries),
        "encoded_episode_count": encoded_count,
        "encoding_source_histogram": source_counts,
        "invalid_encoding_source_ids": invalid_source_ids,
        "heuristic_share": heuristic_share,
        "semantic_consolidation": semantic_evidence,
        "semantic_dynamic_ids": semantic_dynamic,
        "semantic_missing_centroid_ids": semantic_missing_centroid,
        "semantic_missing_residual_ids": semantic_missing_residual,
        "replay_touched_ids": replay_trails,
        "replay_semantic_refreshes": replay_refreshes,
        "replay_semantic_refresh_updated_ids": replay_refresh_updates,
        "sleep_trace_has_m410": sleep_trace_has_m410,
        "passed": bool(
            encoded_count > 0
            and dynamics_count > 0
            and heuristic_share <= 0.25
            and not invalid_source_ids
            and semantic_dynamic
            and not semantic_missing_centroid
            and not semantic_missing_residual
            and replay_trails
            and replay_refresh_updates
            and sleep_trace_has_m410
        ),
    }


def build_default_path_evidence(*, seed: int = 4, cycles: int = 20) -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        runtime = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "segment_state.json",
            seed=seed,
            reset=True,
            memory_enabled=True,
        )
        runtime.agent.long_term_memory.sleep_interval = 9999
        for _ in range(cycles):
            runtime.step(verbose=False)
        runtime.agent.sleep()
        support_source = None
        for semantic in runtime.agent.memory_store.entries:
            if semantic.memory_class not in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}:
                continue
            for support_id in semantic.support_ids or []:
                candidate = runtime.agent.memory_store.get(support_id)
                if candidate is not None:
                    support_source = candidate
                    break
            if support_source is not None:
                break
        if support_source is not None:
            support_source.salience = max(0.99, support_source.salience)
            constrained_replay(runtime.agent.memory_store, random.Random(seed + cycles), batch_size=1)
    entries = list(runtime.agent.memory_store.entries)
    return _default_path_entry_evidence(
        entries,
        seed=seed,
        cycles=cycles,
        sleep_trace_has_m410=bool(
            runtime.agent.narrative_trace
            and "m410_memory_store_consolidation" in runtime.agent.narrative_trace[-1]
        ),
    )


def build_m410_acceptance_report(*, seed: int = 4, cycles: int = 20) -> tuple[dict[str, object], dict[str, object]]:
    static_keywords = _keyword_static_check()
    static_templates = _template_static_check()
    docs = _doc_supersession_check()
    budget = build_budget_competition_evidence()
    replay = build_replay_drift_evidence()
    default = build_default_path_evidence(seed=seed, cycles=cycles)
    gate_summaries = {
        "encoding_dynamics_not_keywords": {
            "status": "PASS" if static_keywords["passed"] and static_templates["passed"] else "FAIL",
            "checks": {"keywords": static_keywords, "templates": static_templates},
        },
        "attention_budget_competition": {"status": "PASS" if budget["passed"] else "FAIL"},
        "heuristic_fallback_auditable": {
            "status": "PASS" if default["heuristic_share"] <= 0.25 else "FAIL",
            "heuristic_share": default["heuristic_share"],
        },
        "semantic_centroid_consolidation": {
            "status": "PASS" if default["semantic_dynamic_ids"] else "FAIL",
            "semantic_dynamic_ids": default["semantic_dynamic_ids"],
        },
        "replay_reencoding": {
            "status": "PASS" if default["replay_semantic_refresh_updated_ids"] else "FAIL",
            "replay_semantic_refresh_updated_ids": default["replay_semantic_refresh_updated_ids"],
            "synthetic_drift_support_passed": replay["passed"],
        },
        "default_path_evidence": {"status": "PASS" if default["passed"] else "FAIL"},
        "milestone_docs_superseded": {"status": "PASS" if docs["passed"] else "FAIL"},
    }
    failed = [gate for gate in GATE_ORDER if gate_summaries[gate]["status"] != "PASS"]
    structural_pass = not bool(failed)
    m48_pytest = _run_m48_behavioral_inheritance_pytest()
    m48_ok = int(m48_pytest.get("exit_code", -1)) == 0
    behavioral_pass = structural_pass and m48_ok
    failed_gates = list(failed)
    if structural_pass and not m48_ok:
        failed_gates.append("m48_behavioral_inheritance")
    status = "PASS" if not failed_gates else "FAIL"
    head = _git_head()
    if failed:
        formal_conclusion = "NOT_ACCEPTED"
    elif not m48_ok:
        formal_conclusion = "NOT_ACCEPTED"
    else:
        formal_conclusion = "PARTIAL_ACCEPT"
    behavioral_basis = (
        "inherits M4.8 default-path ablation contrast; re-verified on this tree via "
        "tests/test_m48_ablation_contrast.py and tests/test_m48_acceptance.py "
        f"(pytest exit_code={m48_pytest.get('exit_code')}, git {head or 'unknown'}); "
        "not re-executed as a separate acceptance artifact in this M4.10 audit."
    )
    report = {
        "milestone_id": "M4.10",
        "status": status,
        "formal_acceptance_conclusion": formal_conclusion,
        "structural_pass": structural_pass,
        "behavioral_pass": behavioral_pass,
        "behavioral_pass_basis": behavioral_basis,
        "behavioral_inheritance_pytest": m48_pytest,
        "phenomenological_pass": "pending(M4.11)",
        "three_layer_accept_ready": False,
        "gate_order": list(GATE_ORDER),
        "gate_summaries": gate_summaries,
        "failed_gates": failed_gates,
    }
    evidence = {
        "static_keywords": static_keywords,
        "static_templates": static_templates,
        "docs": docs,
        "budget": budget,
        "replay": replay,
        "default_path": default,
    }
    return report, evidence


def write_m410_acceptance_artifacts(
    *,
    output_root: str | Path | None = None,
    seed: int = 4,
    cycles: int = 20,
) -> dict[str, str]:
    root = Path(output_root) if output_root is not None else Path(".")
    reports_dir = root / REPORTS_DIR
    artifacts_dir = root / ARTIFACTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report, evidence = build_m410_acceptance_report(seed=seed, cycles=cycles)
    report_path = reports_dir / M410_REPORT_PATH.name
    evidence_path = artifacts_dir / M410_EVIDENCE_PATH.name
    summary_path = reports_dir / M410_SUMMARY_PATH.name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# M4.10 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Formal Acceptance Conclusion: `{report['formal_acceptance_conclusion']}` "
        f"(full three-layer `ACCEPT` awaits M4.11 phenomenology; Gate 7 rule).",
        f"- Three-layer ledger: `structural_pass={report['structural_pass']}`, "
        f"`behavioral_pass={report['behavioral_pass']}` ({report['behavioral_pass_basis']}), "
        f"`phenomenological_pass={report['phenomenological_pass']!r}`",
        f"- Failed gates: {', '.join(report['failed_gates']) if report['failed_gates'] else 'none'}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "evidence": str(evidence_path),
        "summary": str(summary_path),
    }


if __name__ == "__main__":
    paths = write_m410_acceptance_artifacts()
    print(json.dumps(paths, indent=2, sort_keys=True))
