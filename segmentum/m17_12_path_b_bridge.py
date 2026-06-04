from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from .adaptive_compute import decide_adaptive_compute, fixed_budget_decision
from .goal_priors import build_goal_prior_adjustment
from .memory_consolidation import ConflictType, MemoryReuseEvent
from .memory_credit import build_memory_credit_signal
from .memory_field import build_local_memory_field
from .memory_retrieval import RetrievalQuery
from .memory_store import MemoryStore


def _clamp(value: object, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(low, min(high, numeric))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: object, *, limit: int = 32) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        result = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str) and value.strip():
        result = [value.strip()]
    else:
        result = []
    return result[:limit]


def _expand_terms(*values: object, limit: int = 48) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw in _string_list(value, limit=limit):
            parts = [raw, *re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", raw)]
            for part in parts:
                token = str(part).strip()
                if not token:
                    continue
                lowered = token.casefold()
                if lowered in seen:
                    continue
                seen.add(lowered)
                expanded.append(token)
                if len(expanded) >= limit:
                    return expanded
    return expanded


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "for",
    "how",
    "i",
    "if",
    "is",
    "it",
    "of",
    "or",
    "please",
    "review",
    "should",
    "the",
    "to",
    "we",
    "whether",
    "you",
}


def _compact_semantic_terms(*values: object, limit: int = 12) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for token in _expand_terms(*values, limit=64):
        lowered = token.casefold()
        if lowered in _QUERY_STOPWORDS:
            continue
        if " " in token and len(token) > 24:
            continue
        if len(token) <= 1:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        filtered.append(token)
        if len(filtered) >= limit:
            break
    return filtered


def _legacy_rows_from_state(state: Mapping[str, object]) -> list[dict[str, object]]:
    rows = state.get("long_term_memory", [])
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        payload = dict(row)
        row_id = str(payload.get("episode_id") or payload.get("id") or f"pathb:ltm:{index:04d}").strip()
        payload.setdefault("episode_id", row_id)
        payload.setdefault("id", row_id)
        payload.setdefault("content", str(payload.get("predicted_outcome", payload.get("kind", "memory"))))
        normalized.append(payload)
    return normalized


def _store_from_state(state: Mapping[str, object], *, current_cycle: int) -> MemoryStore:
    store = MemoryStore.from_legacy_episodes(_legacy_rows_from_state(state))
    store.refresh_memory_paths(current_cycle=max(int(current_cycle), 0))
    return store


def _retrieval_query_from_recall_query(
    recall_query: Mapping[str, object] | None,
    *,
    now: int,
) -> RetrievalQuery:
    query = _mapping(recall_query)
    semantic_terms = _compact_semantic_terms(
        query.get("semantic_terms"),
        query.get("relationship_terms"),
        query.get("memory_kinds"),
        query.get("current_task"),
        query.get("next_task"),
        limit=32,
    )
    context_terms = _expand_terms(
        query.get("status_terms"),
        _mapping(query.get("entity_binding")).get("target_person"),
        list(_mapping(_mapping(query.get("entity_binding")).get("pronoun_bindings")).values()),
        limit=24,
    )
    content_terms = _expand_terms(
        semantic_terms,
        context_terms,
        query.get("current_task"),
        query.get("next_task"),
        limit=40,
    )
    return RetrievalQuery(
        semantic_tags=semantic_terms,
        context_tags=context_terms,
        content_keywords=content_terms,
        reference_cycle=max(int(now), 0),
        debug=False,
    )


def _goal_hint(recall_query: Mapping[str, object] | None) -> str:
    query = _mapping(recall_query)
    current_task = " ".join(
        _expand_terms(query.get("current_task"), query.get("next_task"), limit=16)
    ).casefold()
    if any(token in current_task for token in ("review", "verify", "check", "inspect", "evidence")):
        return "INTEGRITY"
    if any(token in current_task for token in ("plan", "control", "step", "schedule")):
        return "CONTROL"
    if any(token in current_task for token in ("resource", "budget", "time", "cost")):
        return "RESOURCES"
    if any(token in current_task for token in ("relationship", "trust", "share", "social")):
        return "SOCIAL"
    return ""


def _prediction_error_surrogate(
    recall_query: Mapping[str, object] | None,
    *,
    retrieved_count: int,
) -> float:
    query = _mapping(recall_query)
    statuses = {item.casefold() for item in _string_list(query.get("status_terms"), limit=8)}
    semantic_term_count = len(_expand_terms(query.get("semantic_terms"), limit=12))
    base = 0.10 + min(0.16, semantic_term_count * 0.02)
    if retrieved_count == 0:
        base += 0.12
    if "violated" in statuses:
        base += 0.16
    if "uncertain" in statuses:
        base += 0.10
    return round(_clamp(base), 6)


def _field_reply_strategy(field_action: str) -> str:
    action = str(field_action or "").strip().casefold()
    if action == "scan":
        return "clarify"
    if action in {"hide", "rest", "exploit_shelter"}:
        return "deflect"
    if action == "seek_contact":
        return "self_disclose"
    return "answer"


def _selected_writeback_targets(
    active_paths: list[dict[str, object]],
    *,
    selected_action: str,
    best_single_path_id: str,
    field_required: bool,
    member_memory_ids: list[str],
) -> tuple[list[str], list[str]]:
    selected_paths = [
        str(payload.get("path_id", "")).strip()
        for payload in active_paths
        if str(payload.get("dominant_action", "")).strip().casefold() == selected_action.casefold()
        and str(payload.get("path_id", "")).strip()
    ]
    if not selected_paths and best_single_path_id:
        selected_paths = [best_single_path_id]
    selected_memory_ids = sorted(
        {
            str(memory_id).strip()
            for payload in active_paths
            if str(payload.get("path_id", "")).strip() in set(selected_paths)
            for memory_id in payload.get("source_memory_ids", []) or []
            if str(memory_id).strip()
        }
    )
    if field_required and len(selected_memory_ids) < 2:
        selected_memory_ids = sorted({*selected_memory_ids, *member_memory_ids})
    return selected_memory_ids, selected_paths


def _writeback_entry_from_bridge(result: "PathBRecallBridgeResult") -> dict[str, object]:
    return {
        "committed_memory_ids": list(result.writeback_targets.get("committed_memory_ids", [])),
        "linked_path_ids": list(result.writeback_targets.get("linked_path_ids", [])),
        "selected_action": result.field_selected_action,
        "reply_strategy": result.reply_strategy,
        "counterfactual_status": result.counterfactual_status,
        "field_required": bool(result.field_required),
        "best_single_action": str(result.counterfactual_audit.get("best_single_action", "")),
        "naive_topk_action": str(result.counterfactual_audit.get("naive_topk_action", "")),
        "field_selected_action": str(result.counterfactual_audit.get("field_selected_action", "")),
    }


@dataclass(frozen=True)
class PathBRecallBridgeResult:
    retrieved_items: list[dict[str, object]]
    active_paths: list[dict[str, object]]
    local_field: dict[str, object]
    counterfactual_audit: dict[str, object]
    goal_prior: dict[str, object]
    adaptive_compute: dict[str, object]
    provenance_refs: dict[str, object]
    writeback_targets: dict[str, object]
    field_selected_action: str
    reply_strategy: str
    counterfactual_status: str
    field_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieved_items": [dict(item) for item in self.retrieved_items],
            "active_paths": [dict(item) for item in self.active_paths],
            "active_path_ids": [str(item.get("path_id", "")) for item in self.active_paths if str(item.get("path_id", ""))],
            "local_field": dict(self.local_field),
            "counterfactual_audit": dict(self.counterfactual_audit),
            "goal_prior": dict(self.goal_prior),
            "adaptive_compute": dict(self.adaptive_compute),
            "provenance_refs": dict(self.provenance_refs),
            "writeback_targets": dict(self.writeback_targets),
            "field_selected_action": self.field_selected_action,
            "reply_strategy": self.reply_strategy,
            "counterfactual_status": self.counterfactual_status,
            "field_required": bool(self.field_required),
        }


@dataclass(frozen=True)
class PathBSettlementWritebackResult:
    credit_reports: list[dict[str, object]] = field(default_factory=list)
    reconsolidation_reports: list[dict[str, object]] = field(default_factory=list)
    settled_prediction_ids: list[str] = field(default_factory=list)
    updated_path_ids: list[str] = field(default_factory=list)
    writeback_targets: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "credit_reports": [dict(item) for item in self.credit_reports],
            "reconsolidation_reports": [dict(item) for item in self.reconsolidation_reports],
            "settled_prediction_ids": list(self.settled_prediction_ids),
            "updated_path_ids": list(self.updated_path_ids),
            "writeback_targets": dict(self.writeback_targets),
        }


def build_path_b_recall_bridge(
    state: Mapping[str, object],
    recall_query: Mapping[str, object] | None,
    *,
    retrieved_items: list[dict[str, object]] | None,
    now: int,
    field_consumer_enabled: bool = True,
) -> PathBRecallBridgeResult:
    retrieved = [dict(item) for item in (retrieved_items or [])]
    store = _store_from_state(state, current_cycle=now)
    retrieval_query = _retrieval_query_from_recall_query(recall_query, now=now)
    base_retrieval_k = max(2, min(4, len(retrieved) or 3))
    base_path_k = max(2, min(4, max(2, len(store.memory_paths) or 2)))
    active_paths = store.retrieve_paths(retrieval_query, k=base_path_k)
    local_field_obj = (
        build_local_memory_field(
            active_paths,
            baseline_prediction={},
            errors={},
            body_state={},
        )
        if active_paths
        else None
    )
    goal_context = {}
    goal_hint = _goal_hint(recall_query)
    if goal_hint:
        goal_context = {"active_goal": goal_hint, "urgency_scores": {goal_hint: 0.58}}
    prediction_error = _prediction_error_surrogate(
        recall_query,
        retrieved_count=len(retrieved),
    )
    adaptive_decision = (
        decide_adaptive_compute(
            field=local_field_obj.to_dict() if local_field_obj is not None else {},
            goal_context=goal_context,
            prediction_error_surrogate=prediction_error,
            base_retrieval_k=base_retrieval_k,
            base_path_k=base_path_k,
            candidate_action_count=4,
        ).to_dict()
        if active_paths
        else fixed_budget_decision(
            base_retrieval_k=base_retrieval_k,
            base_path_k=max(1, base_path_k),
            candidate_action_limit=4,
        ).to_dict()
    )
    refined_path_k = int(adaptive_decision.get("path_neighborhood_k", base_path_k) or base_path_k)
    if active_paths and refined_path_k != base_path_k:
        active_paths = store.retrieve_paths(retrieval_query, k=refined_path_k)
        local_field_obj = build_local_memory_field(
            active_paths,
            baseline_prediction={},
            errors={},
            body_state={},
        )
    goal_prior = build_goal_prior_adjustment(
        active_goal=goal_hint or None,
        current_state={"observation": {}, "body_state": {}, "temporal_state": dict(_mapping(state.get("temporal_state")))},
        goal_context=goal_context,
    )
    local_field = local_field_obj.to_dict() if local_field_obj is not None else {}
    counterfactual_audit = dict(local_field.get("counterfactual_audit", {}))
    field_selected_action = str(counterfactual_audit.get("field_selected_action", "answer"))
    counterfactual_status = str(counterfactual_audit.get("status", "no_field"))
    field_required = bool(counterfactual_audit.get("field_required", False))
    reply_strategy = _field_reply_strategy(field_selected_action)
    committed_action = (
        field_selected_action
        if field_required and field_consumer_enabled
        else str(counterfactual_audit.get("best_single_action") or field_selected_action)
    )
    committed_memory_ids, linked_path_ids = _selected_writeback_targets(
        active_paths,
        selected_action=committed_action,
        best_single_path_id=str(counterfactual_audit.get("best_single_path_id", "")),
        field_required=field_required and field_consumer_enabled,
        member_memory_ids=[str(item) for item in local_field.get("member_memory_ids", []) if str(item)],
    )
    writeback_targets = {
        "committed_memory_ids": committed_memory_ids,
        "linked_path_ids": linked_path_ids,
        "committed_action": committed_action,
        "selected_action": field_selected_action,
    }
    provenance_refs = {
        "retrieved_memory_ids": [str(item.get("id", "")) for item in retrieved if str(item.get("id", ""))],
        "active_path_ids": [str(item.get("path_id", "")) for item in active_paths if str(item.get("path_id", ""))],
        "counterfactual_status": counterfactual_status,
    }
    return PathBRecallBridgeResult(
        retrieved_items=retrieved[: int(adaptive_decision.get("retrieval_k", base_retrieval_k) or base_retrieval_k)],
        active_paths=active_paths,
        local_field=local_field,
        counterfactual_audit=counterfactual_audit,
        goal_prior=goal_prior.to_dict() if goal_prior is not None else {},
        adaptive_compute=adaptive_decision,
        provenance_refs=provenance_refs,
        writeback_targets=writeback_targets,
        field_selected_action=field_selected_action,
        reply_strategy=reply_strategy,
        counterfactual_status=counterfactual_status,
        field_required=field_required,
    )


def merge_path_b_field_guidance(
    memory_dynamics: dict[str, object],
    bridge_result: PathBRecallBridgeResult,
    *,
    field_consumer_enabled: bool = True,
) -> dict[str, object]:
    recall = dict(_mapping(memory_dynamics.get("recall")))
    recall["active_path_ids"] = list(bridge_result.provenance_refs.get("active_path_ids", []))
    recall["counterfactual_status"] = bridge_result.counterfactual_status
    recall["best_single_baseline"] = {
        "action": bridge_result.counterfactual_audit.get("best_single_action", ""),
        "path_id": bridge_result.counterfactual_audit.get("best_single_path_id", ""),
    }
    recall["naive_topk_baseline"] = {
        "action": bridge_result.counterfactual_audit.get("naive_topk_action", ""),
    }
    recall["field_enabled"] = {
        "selected_action": bridge_result.counterfactual_audit.get("field_selected_action", ""),
        "field_required": bool(bridge_result.field_required),
    }
    memory_dynamics["recall"] = recall
    memory_dynamics["recall_bridge"] = bridge_result.to_dict()
    control = dict(_mapping(memory_dynamics.get("control_guidance")))
    contract = dict(_mapping(control.get("reply_contract")))
    contract["path_b_field_counterfactual_status"] = bridge_result.counterfactual_status
    contract["path_b_field_selected_action"] = bridge_result.field_selected_action
    contract["path_b_field_reply_strategy"] = bridge_result.reply_strategy
    contract["path_b_field_required"] = bool(bridge_result.field_required and field_consumer_enabled)
    contract["path_b_field_guided"] = bool(
        field_consumer_enabled
        and bridge_result.field_selected_action
        and bridge_result.field_selected_action == str(bridge_result.writeback_targets.get("committed_action", ""))
    )
    control["path_b_field_counterfactual_status"] = bridge_result.counterfactual_status
    control["path_b_field_selected_action"] = bridge_result.field_selected_action
    control["path_b_field_reply_strategy"] = bridge_result.reply_strategy
    if field_consumer_enabled:
        if bridge_result.reply_strategy == "clarify" and bridge_result.field_selected_action == "scan":
            contract["prefer_clarification"] = True
            control["clarification_bias"] = max(
                _clamp(control.get("clarification_bias", 0.0)),
                0.78 if bridge_result.field_required else 0.62,
            )
            if bridge_result.field_required:
                control["conflict_level"] = max(_clamp(control.get("conflict_level", 0.0)), 0.44)
        elif bridge_result.field_required and bridge_result.reply_strategy == "deflect":
            contract["prefer_boundary_safe_reply"] = True
            control["repair_bias"] = max(_clamp(control.get("repair_bias", 0.0)), 0.72)
        elif bridge_result.field_required and bridge_result.reply_strategy == "self_disclose":
            contract["prefer_relational_disclosure"] = True
        elif bridge_result.field_required:
            contract["prefer_direct_answer"] = True
    control["reply_contract"] = contract
    memory_dynamics["control_guidance"] = control
    return memory_dynamics


def register_prediction_provenance(
    state: dict[str, object],
    *,
    prediction_ids: list[str],
    bridge_result: PathBRecallBridgeResult,
    turn_index: int,
) -> None:
    bridge_state = dict(_mapping(state.get("m17_path_b_bridge")))
    provenance = dict(_mapping(bridge_state.get("prediction_provenance")))
    bridge_state["last_recall_bridge"] = bridge_result.to_dict()
    bridge_state["last_turn_index"] = int(turn_index)
    writeback_entry = _writeback_entry_from_bridge(bridge_result)
    for prediction_id in prediction_ids:
        token = str(prediction_id).strip()
        if not token:
            continue
        provenance[token] = {
            **writeback_entry,
            "prediction_id": token,
            "turn_index": int(turn_index),
        }
    bridge_state["prediction_provenance"] = provenance
    state["m17_path_b_bridge"] = bridge_state


def _settlement_support_scores(outcome: str, confidence: float) -> tuple[float, float]:
    normalized = str(outcome or "").strip()
    bounded_confidence = _clamp(confidence)
    if normalized == "confirmed":
        return bounded_confidence, 0.0
    if normalized == "violated":
        return 0.0, bounded_confidence
    if normalized == "expired":
        return 0.0, 0.12
    if normalized == "uncertain":
        return 0.18, 0.18
    return 0.12, 0.0


def apply_path_b_settlement_writeback(
    state: dict[str, object],
    *,
    ledger_entries: list[Mapping[str, object]],
    turn_index: int,
    current_user_text: str,
    now: int,
) -> PathBSettlementWritebackResult:
    bridge_state = dict(_mapping(state.get("m17_path_b_bridge")))
    prediction_provenance = dict(_mapping(bridge_state.get("prediction_provenance")))
    if not prediction_provenance:
        return PathBSettlementWritebackResult()
    store = _store_from_state(state, current_cycle=turn_index + 1)
    settled_prediction_ids: list[str] = []
    credit_reports: list[dict[str, object]] = []
    reconsolidation_reports: list[dict[str, object]] = []
    writeback_targets: dict[str, object] = {}
    context_tags = _expand_terms(current_user_text, limit=12)
    for entry in ledger_entries:
        prediction_id = str(entry.get("prediction_id", "")).strip()
        if not prediction_id:
            continue
        provenance = _mapping(prediction_provenance.get(prediction_id))
        if not provenance:
            continue
        outcome = str(
            entry.get("settlement_outcome")
            or entry.get("validation_status")
            or entry.get("status")
            or ""
        ).strip()
        if not outcome:
            continue
        support_score, contradiction_score = _settlement_support_scores(
            outcome,
            _clamp(entry.get("settlement_confidence", 0.0)),
        )
        signal = build_memory_credit_signal(
            prediction_id=prediction_id,
            semantic_provenance={
                "committed_memory_ids": list(provenance.get("committed_memory_ids", [])),
                "linked_path_ids": list(provenance.get("linked_path_ids", [])),
            },
            outcome=outcome,
            support_score=support_score,
            contradiction_score=contradiction_score,
            confidence_weight=_clamp(entry.get("committed_confidence", 0.0)),
            source_module="path_b_m17_bridge",
        )
        if signal is None:
            continue
        credit_report = store.apply_memory_credit(signal, tick=turn_index + 1)
        credit_reports.append(dict(credit_report))
        contradiction_detected = signal.outcome == "violated" or contradiction_score > support_score
        for memory_id in signal.linked_memory_ids:
            report = store.reconsolidate_entry(
                memory_id,
                current_mood="dialogue_active",
                current_context_tags=context_tags,
                current_cycle=turn_index + 1,
                current_state={"active_goals": [], "conversation_runtime": True},
                recall_artifact={"prediction_id": prediction_id, "turn_index": int(turn_index)},
                conflict_type=ConflictType.FACTUAL if contradiction_detected else None,
                cognitive_style={"update_rigidity": 0.0, "error_aversion": 0.35},
                reuse_event=MemoryReuseEvent(
                    reuse_event_id=f"pathb:{prediction_id}:{memory_id}:{turn_index + 1}",
                    memory_id=memory_id,
                    path_id=signal.linked_path_ids[0] if signal.linked_path_ids else "",
                    prediction_before_reuse={},
                    observation_after_reuse={},
                    reuse_prediction_error=max(contradiction_score, 1.0 - support_score),
                    reuse_free_energy_delta=float(signal.free_energy_delta),
                    recall_confidence=float(signal.confidence_weight),
                    contradiction_detected=contradiction_detected,
                    live_reuse=True,
                ),
            )
            reconsolidation_reports.append(report.to_dict())
        settled_prediction_ids.append(prediction_id)
        writeback_targets[prediction_id] = dict(provenance)
        prediction_provenance.pop(prediction_id, None)
    if not credit_reports and not reconsolidation_reports:
        return PathBSettlementWritebackResult()
    path_refresh = store.refresh_memory_paths(current_cycle=turn_index + 1)
    state["long_term_memory"] = store.to_legacy_episodes()
    bridge_state["prediction_provenance"] = prediction_provenance
    bridge_state["last_settlement_writeback"] = {
        "turn_index": int(turn_index),
        "at": int(now),
        "settled_prediction_ids": list(settled_prediction_ids),
        "writeback_targets": dict(writeback_targets),
        "path_refresh": dict(path_refresh),
    }
    state["m17_path_b_bridge"] = bridge_state
    updated_path_ids = sorted(
        {
            path_id
            for report in credit_reports
            for path_id in _string_list(_mapping(report.get("path_refresh")).get("updated_ids"), limit=64)
        }
    )
    return PathBSettlementWritebackResult(
        credit_reports=credit_reports,
        reconsolidation_reports=reconsolidation_reports,
        settled_prediction_ids=settled_prediction_ids,
        updated_path_ids=updated_path_ids,
        writeback_targets=writeback_targets,
    )
