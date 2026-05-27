"""Compact Path B mind-panel debug bundle for operator copy/paste."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.diagnose_idle_reflection import (
    summarize_log,
    verdicts_for_session,
)
from segmentum.dialogue.runtime.m13_idle import gather_idle_structural_signals
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    read_runner_lock,
    runner_lock_is_alive,
)
from segmentum.dialogue.runtime.m14_3_open_item_migration import audit_open_items_for_efe
from segmentum.dialogue.runtime.m14_4_implicit_idle import compute_idle_seconds
from segmentum.dialogue.runtime.m15_3_cleanup_control import is_strictly_traceable, summarize_strict_traceability


AUDIT_TYPES_OF_INTEREST = frozenset(
    {
        "IdleCognitiveTickEvent",
        "IdlePlanStructuralMismatchEvent",
        "IdlePlanStructuralAgreementEvent",
        "IdleIntrospectionPlanEvent",
        "IdleIntrospectionSuppressionEvent",
        "M13ProactiveSuppressionEvent",
        "M13ProactiveProposalEvent",
        "M14ImplicitIdleProactiveCheckEvent",
        "MetaControlStallDetectedEvent",
        "RepeatedFailurePathDetectedEvent",
        "SelfConsistencyTensionDetectedEvent",
        "ConsolidationRunEvent",
        "ConsolidationDeferredEvent",
        "MemoryGateRejectedEvent",
        "MemoryGateCommitEvent",
    }
)


def _fmt_ts(ts: Any) -> str:
    try:
        value = int(ts or 0)
    except (TypeError, ValueError):
        return "-"
    if value <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def _clip(text: Any, *, limit: int = 160) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value or "-"
    return value[: limit - 1] + "…"


def _join(values: Any, *, limit: int = 8) -> str:
    if values is None:
        return "-"
    if isinstance(values, str):
        return _clip(values, limit=240)
    if not isinstance(values, (list, tuple, set)):
        return _clip(values)
    parts = [str(item).strip() for item in values if str(item).strip()]
    if not parts:
        return "-"
    return ", ".join(parts[:limit])


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _recent_audit_lines(path: Path, *, limit: int = 18) -> list[str]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                event = str(row.get("event", "") or "")
                typ = str(row.get("type", "") or "")
                if typ in AUDIT_TYPES_OF_INTEREST or event in {
                    "m13_proactive_audit",
                    "m13_idle_audit",
                    "m14_idle_audit",
                    "m14_4_implicit_idle_audit",
                    "m15_consolidation_audit",
                    "m15_episode_ledger",
                }:
                    rows.append(row)
    except OSError:
        return []
    lines: list[str] = []
    for row in rows[-limit:]:
        typ = str(row.get("type", "") or row.get("event", "") or "audit")
        at = _fmt_ts(row.get("at"))
        reject = row.get("reject_reason") or row.get("reason_code") or row.get("suppression_reason_code")
        reason = row.get("reason") or row.get("skip_reason") or row.get("mismatch_reason_code")
        trigger = row.get("trigger") or row.get("action_trigger")
        extra = reject or reason or trigger or row.get("reason_code")
        lines.append(f"  [{at}] {typ} {extra or ''}".rstrip())
    return lines


def _open_items_section(open_items: Any) -> list[str]:
    lines = ["## Open items (traceability)"]
    suggestions = audit_open_items_for_efe(open_items)
    suggestion_by_id = {row.item_id: row for row in suggestions}
    rows = open_items if isinstance(open_items, list) else []
    if not rows:
        lines.append("- (none)")
        return lines
    for row in rows[:12]:
        if not isinstance(row, Mapping):
            continue
        item_id = _clip(row.get("id"), limit=80)
        if str(row.get("status", "open") or "open") != "open":
            continue
        title = _clip(row.get("title") or row.get("content") or row.get("summary"), limit=100)
        suggestion = suggestion_by_id.get(str(row.get("id", "") or "").strip())
        lines.append(
            f"- `{item_id}` title=`{title}` "
            f"strict_trace={is_strictly_traceable(row)} "
            f"next_check=`{_clip(row.get('next_check') or row.get('next_step'), limit=40)}` "
            f"evidence_refs=[{_join(row.get('evidence_refs'))}] "
            f"bound_memory_ids=[{_join(row.get('bound_memory_ids'))}] "
            f"scheduled_intent_id=`{_clip(row.get('scheduled_intent_id'), limit=40)}`"
        )
        if suggestion is not None:
            lines.append(f"  -> suggestion: {suggestion.reason_code} ({suggestion.action})")
    if len(suggestions) > len(rows[:12]):
        lines.append(f"- traceability_suggestions_total: {len(suggestions)}")
    return lines


def _expectations_section(expectations: Any) -> list[str]:
    lines = ["## Pending expectations"]
    rows = expectations if isinstance(expectations, list) else []
    if not rows:
        lines.append("- (none)")
        return lines
    active_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("status", "pending") or "pending").strip().lower() in {"", "pending", "active", "uncertain", "due"}
    ]
    expired_like = len(rows) - len(active_rows)
    lines.append(f"- active_rows_shown={min(len(active_rows), 10)} raw_total={len(rows)} folded_non_active={expired_like}")
    for row in active_rows[:10]:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"- `{_clip(row.get('id'), limit=80)}` status=`{_clip(row.get('status'), limit=20)}` "
            f"content=`{_clip(row.get('content') or row.get('summary'), limit=100)}` "
            f"evidence_refs=[{_join(row.get('evidence_refs'))}] "
            f"bound_memory_ids=[{_join(row.get('bound_memory_ids'))}]"
        )
    return lines


def _status_counts(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    items = rows if isinstance(rows, list) else []
    for row in items:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "pending") or "pending").strip().lower()
        if status.startswith("merged_into:"):
            status = "merged"
        counts[status or "pending"] = counts.get(status or "pending", 0) + 1
    return counts


def _queued_outreach_section(rows: Any, *, session_id: str = "") -> list[str]:
    lines = ["## Queued outreach (this session)"]
    items = rows if isinstance(rows, list) else []
    pending = [row for row in items if isinstance(row, Mapping) and str(row.get("status", "")) == "pending"]
    foreign = [row for row in items if isinstance(row, Mapping) and str(row.get("source_session_id", row.get("session_id", ""))) not in {"", session_id}]
    if foreign:
        lines.extend(["", "## Queued outreach (other sessions)", f"- foreign_session_excluded: {len(foreign)}"])
    if not pending:
        lines.append("- pending=0")
        return lines
    lines.append(f"- pending={len(pending)}")
    for row in pending[:8]:
        lines.append(
            f"- proposal_id=`{_clip(row.get('proposal_id'), limit=60)}` "
            f"session_id=`{_clip(row.get('session_id') or row.get('source_session_id'), limit=40)}` "
            f"trigger=`{_clip(row.get('trigger'), limit=60)}` "
            f"evidence_refs=[{_join(row.get('trigger_evidence_refs') or row.get('evidence_refs'))}] "
            f"intent=`{_clip(row.get('ordinary_language_intent'), limit=120)}`"
        )
    return lines


def _strict_traceability_section(state: Mapping[str, Any]) -> list[str]:
    summary = summarize_strict_traceability(state)
    pending_rows = state.get("pending_expectations", []) if isinstance(state.get("pending_expectations"), list) else []
    pending_counts = _status_counts(pending_rows)
    pending_raw = len(pending_rows) if isinstance(pending_rows, list) else 0
    return [
        "## Traceability (strict)",
        f"- open_items: total={summary.get('open_items_total', 0)} "
        f"strict_trace={summary.get('open_items_strict_trace', 0)} "
        f"duplicate_ids={summary.get('open_items_duplicate_local_ids', 0)}",
        f"- pending_expectations_raw_total={pending_raw} "
        f"active_total={summary.get('pending_expectations_total', 0)} "
        f"strict_trace_active={summary.get('pending_expectations_strict_trace', 0)} "
        f"duplicate_ids={summary.get('pending_expectations_duplicate_local_ids', 0)}",
        f"- pending_status_counts: expired={pending_counts.get('expired', 0)} "
        f"merged={pending_counts.get('merged', 0)} diagnostic={pending_counts.get('diagnostic_only', 0)} "
        f"active={summary.get('pending_expectations_total', 0)}",
    ]


def build_mind_debug_bundle_text(
    *,
    session_root: Path,
    persona_name: str,
    session_id: str,
    state: Mapping[str, Any],
    observability: Mapping[str, Any],
    ui_hints: Mapping[str, Any] | None = None,
    turn_index: int = 0,
) -> str:
    """Build a plain-text bundle suitable for pasting into an agent chat."""
    ui_hints = _mapping(ui_hints)
    now = int(time.time())
    log_path = session_root / "conversation_log.jsonl"
    log_summary = summarize_log(log_path)

    temporal = _mapping(state.get("temporal_state"))
    m13_state = _mapping(state.get("m13_drive_state"))
    initiative = _mapping(m13_state.get("initiative"))
    idle = _mapping(initiative.get("idle_introspection"))
    bg = _mapping(initiative.get("background_continuity"))

    last_user = int(temporal.get("last_user_turn_at", 0) or temporal.get("last_turn_at", 0) or 0)
    idle_elapsed = max(0, now - last_user) if last_user > 0 else -1
    computed_idle = compute_idle_seconds(state, now=float(now))

    lock = read_runner_lock(session_root)
    lock_alive = runner_lock_is_alive(lock)
    turn_index_state = int(temporal.get("last_turn_index", 0) or 0)
    signals = gather_idle_structural_signals(state, now=now, turn_index=turn_index_state)
    verdicts = verdicts_for_session(
        state=dict(state),
        log_summary=log_summary,
        lock_alive=lock_alive,
        has_lock=lock is not None,
        idle_elapsed=idle_elapsed,
        structural_should_run=bool(signals.should_run_llm()),
    )

    tick = _mapping(observability.get("m13_5_last_idle_cognitive_tick"))
    mismatch = _mapping(observability.get("m14_6_last_plan_selector_mismatch"))
    target = _mapping(observability.get("m14_3_last_proactive_target"))
    suppression = _mapping(observability.get("m14_3_last_proactive_suppression"))
    latest_selector = _mapping(observability.get("latest_selector_target"))
    latest_attempted = _mapping(observability.get("latest_attempted_target"))
    latest_delivered = _mapping(observability.get("latest_delivered_target"))
    latest_suppressed = _mapping(observability.get("latest_suppressed_target"))
    latest_pipeline = _mapping(observability.get("latest_pipeline_suppression"))
    last_assessment = _mapping(observability.get("last_delivery_assessment"))
    bands = _mapping(observability.get("m14_3_last_drive_band_summary"))
    intro_plan = _mapping(log_summary.get("latest_intro_plan"))
    plan_body = _mapping(intro_plan.get("plan"))
    outreach_plan = _mapping(plan_body.get("outreach_recommendation"))

    m15_trail = observability.get("m15_delta_f_trail", [])
    m15_slow = _mapping(observability.get("m15_slow_loop"))
    m15_meta = _mapping(observability.get("m15_meta_control"))
    m15_cleanup = _mapping(observability.get("m15_cleanup"))
    counts = _mapping(log_summary.get("counts"))

    lines = [
        "# Path B Mind Debug Bundle",
        f"generated_at: {_fmt_ts(now)}",
        f"persona: {_clip(persona_name, limit=80)}",
        f"session_id: {_clip(session_id, limit=120)}",
        f"session_root: {session_root}",
        "",
        "## Diagnose verdicts",
        *([f"- {code}" for code in verdicts] if verdicts else ["- (none)"]),
        "",
        "## UI / runtime hints",
        f"- chat_turn_index: {turn_index}",
        f"- temporal_last_turn_index: {turn_index_state}",
        f"- pending_user_message: {ui_hints.get('pending_user_message', '-')}",
        f"- m13_ui_turn_in_progress: {ui_hints.get('m13_ui_turn_in_progress', False)}",
        f"- idle_seconds_computed: {round(computed_idle, 3)}",
        f"- idle_seconds_since_last_user: {idle_elapsed}",
        f"- idle_threshold_seconds: {initiative.get('idle_threshold_seconds', '-')}",
        f"- last_implicit_idle_suppression: {_clip(ui_hints.get('last_implicit_idle_suppression_reason_code'), limit=80)}",
        f"- proactive_policy_profile: {_clip(initiative.get('proactive_policy_profile'), limit=40)}",
        f"- meta_control_apply_env: {ui_hints.get('meta_control_apply_env', '-')}",
        "",
        "## Temporal",
        f"- last_turn_at: {_fmt_ts(temporal.get('last_turn_at'))}",
        f"- last_user_turn_at: {_fmt_ts(temporal.get('last_user_turn_at'))}",
        f"- last_time_gap_label: {_clip(temporal.get('last_time_gap_label'), limit=40)}",
        f"- last_assistant_turn_at: {_fmt_ts(temporal.get('last_assistant_turn_at'))}",
        "",
        "## Daemon / background",
        f"- daemon_pid: {lock.pid if lock else 0} alive={lock_alive}",
        f"- health_ticks_today: {observability.get('health_ticks_today', 0)}",
        f"- background_ticks_today: {bg.get('ticks_today', 0)}",
        f"- daemon_llm_available: {observability.get('daemon_llm_available', '-')}",
        f"- daemon_llm_unavailable_reason: {_clip(observability.get('daemon_llm_unavailable_reason'), limit=80)}",
        f"- daemon_background_ran_llm: {observability.get('daemon_background_ran_llm', '-')}",
        f"- background_llm_calls_today: {observability.get('background_llm_calls_today', 0)}/"
        f"{observability.get('background_llm_budget', '?')}",
        f"- last_budget_block_reason: {_clip(observability.get('last_budget_block_reason'), limit=80)}",
        f"- last_background_skip_reason: {_clip(observability.get('last_background_skip_reason'), limit=80)}",
        f"- environment_event_status_counts: {json.dumps(_mapping(observability.get('environment_event_status_counts')), ensure_ascii=False)}",
        f"- environment_events_terminal_ratio: {observability.get('environment_events_terminal_ratio', '-')}",
        f"- environment_event_backlog_count: {observability.get('environment_event_backlog_count', 0)}",
        f"- stale_environment_event_backlog_count: {observability.get('stale_environment_event_backlog_count', 0)}",
        "",
        "## M13.5 last idle cognitive tick",
        f"- at: {_fmt_ts(tick.get('at'))}",
        f"- idle_seconds: {tick.get('idle_seconds', '-')}",
        f"- reject_reason: {_clip(tick.get('reject_reason'), limit=80)}",
        f"- memory_efe_should_outreach: {tick.get('memory_efe_should_outreach', '-')}",
        f"- memory_efe_selected_policy: {_clip(tick.get('memory_efe_selected_policy'), limit=40)}",
        f"- retrieved_ids: [{_join(tick.get('retrieved_ids'))}]",
        f"- selected_target: {json.dumps(_mapping(tick.get('selected_target')), ensure_ascii=False) if tick.get('selected_target') else '-'}",
        f"- bands: {json.dumps(_mapping(tick.get('bands')), ensure_ascii=False) if tick.get('bands') else '-'}",
        "",
        "## M14.3 proactive alignment (legacy summary)",
        f"- target_trigger: {_clip(target.get('trigger'), limit=60)}",
        f"- source_kind: {_clip(target.get('source_kind'), limit=40)}",
        f"- traceable_expectation_id: {_clip(target.get('traceable_expectation_id'), limit=80)}",
        f"- evidence_refs: [{_join(target.get('evidence_refs'))}]",
        f"- selection_reason_codes: [{_join(target.get('selection_reason_codes'))}]",
        f"- last_suppression: {_clip(suppression.get('reason_code'), limit=80)} stage={_clip(suppression.get('reason_stage'), limit=40)}",
        "",
        "## Proactive target timeline",
        f"- latest_selector_target: {json.dumps(latest_selector, ensure_ascii=False) if latest_selector else '-'}",
        f"- latest_attempted_target: {json.dumps(latest_attempted, ensure_ascii=False) if latest_attempted else '-'}",
        f"- latest_delivered_target: {json.dumps(latest_delivered, ensure_ascii=False) if latest_delivered else '-'}",
        f"- latest_suppressed_target: {json.dumps(latest_suppressed, ensure_ascii=False) if latest_suppressed else '-'}",
        f"- latest_pipeline_suppression: {json.dumps(latest_pipeline, ensure_ascii=False) if latest_pipeline else '-'}",
        f"- last_delivery_assessment: {json.dumps(last_assessment, ensure_ascii=False) if last_assessment else '-'}",
        f"- drive_bands: behavior={bands.get('behavioral_pull_band')} boredom={bands.get('boredom_band')} "
        f"reward={bands.get('affective_reward_band')} relation={bands.get('relation_path_precision_band')}",
        f"- open_item_traceability_suggestions: {observability.get('m14_3_open_item_traceability_suggestions', 0)}",
        "",
        "## M14.6 plan vs selector",
        f"- mismatch_at: {_fmt_ts(mismatch.get('at'))}",
        f"- mismatch_reason_code: {_clip(mismatch.get('mismatch_reason_code'), limit=80)}",
        f"- plan_recommendation_reason: {_clip(mismatch.get('plan_recommendation_reason'), limit=120)}",
        f"- intro_plan_at: {_fmt_ts(intro_plan.get('at'))}",
        f"- intro_should_outreach: {outreach_plan.get('should_outreach', '-')}",
        f"- intro_outreach_reason: {_clip(outreach_plan.get('reason'), limit=80)}",
        f"- intro_suggested_intent: {_clip(outreach_plan.get('suggested_intent') or outreach_plan.get('ordinary_language_intent'), limit=160)}",
        f"- intro_reflection_focus: {_clip(_mapping(plan_body.get('reflection_focus')).get('topic'), limit=120)}",
        f"- idle_last_outreach_outcome: {_clip(idle.get('last_outreach_outcome'), limit=80)}",
        "",
        "## M15.0 Delta F trail (recent)",
    ]
    if isinstance(m15_trail, list) and m15_trail:
        for row in m15_trail[:6]:
            if isinstance(row, Mapping):
                lines.append(
                    f"- turn {row.get('turn_index', '?')} {_clip(row.get('action'), limit=30)} "
                    f"DeltaF={float(row.get('delta_fe_proxy', 0) or 0):+.3f} "
                    f"outcome={_clip(row.get('outcome_summary'), limit=30)}"
                )
    else:
        lines.append("- (empty)")

    ops = _mapping(m15_slow.get("last_ops"))
    lines.extend(
        [
            "",
            "## M15.1 slow loop",
            f"- last_run_at: {_fmt_ts(m15_slow.get('last_run_at'))}",
            f"- ops: merges={ops.get('merges', 0)} promote={ops.get('promote', 0)} "
            f"abstract={ops.get('abstract', 0)} decay_touched={ops.get('decay_touched', 0)} archived={ops.get('archived', 0)}",
            f"- budget_today: {m15_slow.get('runs_today', 0)}/{m15_slow.get('budget_per_day', 6)}",
            "",
            "## M15.2 meta-control",
        ]
    )
    active_meta = m15_meta.get("active", [])
    if isinstance(active_meta, list) and active_meta:
        for row in active_meta[-5:]:
            item = _mapping(row)
            payload = _mapping(item.get("payload"))
            lines.append(
                f"- active {item.get('intent_kind')} trigger={payload.get('action_trigger')} "
                f"expires={_fmt_ts(item.get('expires_at'))}"
            )
    else:
        lines.append("- active intents: (none)")
    recent_meta = m15_meta.get("recent_detections", [])
    if isinstance(recent_meta, list):
        for row in recent_meta[-5:]:
            item = _mapping(row)
            lines.append(
                f"- detected {item.get('type')} trigger={item.get('action_trigger')} "
                f"reject={item.get('reject_reason')}"
            )

    cleanup_ops = _mapping(m15_cleanup.get("last_ops"))
    lines.extend(
        [
            "",
            "## M15.3 cleanup",
            f"- last_run_at: {_fmt_ts(m15_cleanup.get('last_run_at'))}",
            f"- last_source: {_clip(m15_cleanup.get('last_source'), limit=40)}",
            f"- ops: merged={cleanup_ops.get('merged_duplicates', 0)} "
            f"expired_pending={cleanup_ops.get('expired_pending_expectations', 0)} "
            f"diagnostic_open={cleanup_ops.get('diagnostic_open_items', 0)} "
            f"recall_deprioritized={cleanup_ops.get('recall_deprioritized', 0)}",
            f"- active_cleanup_intents: {m15_cleanup.get('cleanup_active_count', 0)}",
        ]
    )
    cleanup_active = m15_meta.get("cleanup_active", [])
    if isinstance(cleanup_active, list) and cleanup_active:
        for row in cleanup_active[-5:]:
            item = _mapping(row)
            lines.append(f"- active_cleanup {item.get('intent_kind')} detector={item.get('detector')}")
    cleanup_consumed = m15_meta.get("cleanup_consumed", [])
    if isinstance(cleanup_consumed, list) and cleanup_consumed:
        for row in cleanup_consumed[-5:]:
            item = _mapping(row)
            lines.append(
                f"- recently_applied_cleanup {item.get('intent_kind')} "
                f"at={_fmt_ts(item.get('consumed_at'))} ops={json.dumps(_mapping(item.get('ops_delta')), ensure_ascii=False)}"
            )
    cleanup_recent = m15_meta.get("cleanup_recent_detections", [])
    if isinstance(cleanup_recent, list):
        for row in cleanup_recent[-5:]:
            item = _mapping(row)
            lines.append(f"- cleanup_detected {item.get('type')} emitted={item.get('emitted_intent_id')}")

    lines.extend(
        [
            "",
            *_strict_traceability_section(state),
            "",
            *_open_items_section(state.get("open_items")),
            "",
            *_expectations_section(state.get("pending_expectations")),
            "",
            *_queued_outreach_section(ui_hints.get("queued_outreach"), session_id=session_id),
            "",
            "## Log channel counts (full scan)",
        ]
    )
    for key in sorted(counts.keys()):
        if int(counts.get(key, 0) or 0) > 0:
            lines.append(f"- {key}: {counts[key]}")

    lines.extend(["", "## Recent audit tail"])
    audit_lines = _recent_audit_lines(log_path)
    lines.extend(audit_lines if audit_lines else ["- (none)"])

    latest_skip = _mapping(log_summary.get("latest_skip"))
    latest_stateful = _mapping(log_summary.get("latest_stateful_skip"))
    lines.extend(
        [
            "",
            "## Latest skip reasons (log vs state)",
            f"- log_latest_skip: {_clip(latest_skip.get('skip_reason') or latest_skip.get('reason_code') or latest_skip.get('suppression_reason_code'), limit=80)} "
            f"at={_fmt_ts(latest_skip.get('at'))}",
            f"- log_latest_stateful_skip: {_clip(latest_stateful.get('skip_reason') or latest_stateful.get('reason_code'), limit=80)} "
            f"at={_fmt_ts(latest_stateful.get('at'))}",
            f"- state_idle_last_skip_reason: {_clip(idle.get('last_skip_reason'), limit=80)}",
            f"- state_initiative_last_suppression: {_clip(initiative.get('last_suppression_reason_code') or initiative.get('last_suppression_reason'), limit=80)}",
        ]
    )
    return "\n".join(lines) + "\n"
