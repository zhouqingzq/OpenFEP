"""M15.1 bounded consolidation and forgetting owner for Path B."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import uuid
from typing import Any, Mapping

from segmentum.dialogue.runtime.m14_7_memory_decay import (
    MAX_ARCHIVE_FLIPS_PER_TICK,
    apply_consolidation_decay_extension,
)
from segmentum.dialogue.runtime.m14_7_memory_gate import MemoryGate, MemoryWriteIntent, memory_gate_event
from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, MemoryDynamicsEpisode


ENGINEERING_PROXY_LABEL = "mvp_local_consolidation"
MAX_MERGES_PER_RUN = 4
MAX_PROMOTIONS_PER_RUN = 4
MAX_PATH_ABSTRACTIONS_PER_RUN = 2
MAX_CONSOLIDATION_RUNS_PER_DAY = 6
MIN_RUN_INTERVAL_SECONDS = 600
LTM_PROMOTION_RECALL_FLOOR = 0.4
MAX_STM_AGE_SECONDS_FOR_PROMOTION = 7 * 86400
N_PATH_OCCURRENCES_FOR_ABSTRACTION = 4
CONSOLIDATION_OPS = frozenset(
    {
        "merge_expectation",
        "merge_open_item",
        "promote_stm_to_ltm",
        "abstract_path",
        "decay_ltm_salience",
        "archive_ltm",
    }
)
DEFERRED_REASONS = frozenset(
    {
        "budget_exceeded",
        "user_turn_in_progress",
        "ledger_empty",
        "recently_ran_within_min_interval",
    }
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _epoch(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _day_key(now: int) -> str:
    return str(int(now) // 86400)


def _status_family(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "open") or "open")
    if status.startswith("merged_into:"):
        return "merged"
    if status in {"pending", "open", "due", "uncertain", ""}:
        return "open"
    return status


def _structural_duplicate(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if _status_family(a) != "open" or _status_family(b) != "open":
        return False
    if _bounded_float(a.get("confidence"), default=0.5) < 0.4:
        return False
    if _bounded_float(b.get("confidence"), default=0.5) < 0.4:
        return False
    ev_a = set(_string_list(a.get("evidence_refs"), limit=16))
    ev_b = set(_string_list(b.get("evidence_refs"), limit=16))
    bound_a = set(_string_list(a.get("bound_memory_ids"), limit=16))
    bound_b = set(_string_list(b.get("bound_memory_ids"), limit=16))
    if not ev_a or not ev_b or not bound_a or not bound_b:
        return False
    shared_evidence = ev_a & ev_b
    if not shared_evidence or not (bound_a & bound_b):
        return False
    similarity = len(shared_evidence) / max(len(ev_a), len(ev_b), 1)
    return similarity >= 0.5


def fingerprint_class(episode: MemoryDynamicsEpisode | Mapping[str, Any]) -> str:
    payload = (
        episode.state_fingerprint_payload
        if isinstance(episode, MemoryDynamicsEpisode)
        else _mapping(episode.get("state_fingerprint_payload"))
    )
    if not payload:
        fp = episode.state_fingerprint if isinstance(episode, MemoryDynamicsEpisode) else str(episode.get("state_fingerprint", ""))
        return fp[:12]
    compact = {
        key: payload.get(key)
        for key in (
            "boredom_band",
            "reward_band",
            "behavior_band",
            "relation_band",
            "memory_efe_should_outreach",
            "memory_efe_selected_policy",
            "open_items_concrete_count",
            "unsettled_pending_settlement_count",
        )
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:240]


@dataclass(frozen=True)
class ConsolidationOpResult:
    op: str
    committed: bool
    source_ids: list[str] = field(default_factory=list)
    retained_id: str = ""
    new_id: str = ""
    violation_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConsolidationRunResult:
    ran: bool
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    op_results: list[ConsolidationOpResult] = field(default_factory=list)
    deferred_reason: str = ""


def _op_event(*, result: ConsolidationOpResult, run_id: str, now: int, turn_index: int) -> dict[str, Any]:
    return {
        "type": "ConsolidationOpEvent",
        "at": now,
        "turn_index": turn_index,
        "run_id": run_id,
        "op": result.op,
        "inputs": {"source_ids": list(result.source_ids[:8])},
        "outputs": {
            "retained_id": result.retained_id,
            "new_id": result.new_id,
        },
        "committed": bool(result.committed),
        "violation_codes": list(result.violation_codes[:8]),
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def _merge_rows(rows: list[Any], *, op: str, cap: int) -> list[ConsolidationOpResult]:
    results: list[ConsolidationOpResult] = []
    for idx, row in enumerate(rows):
        if len(results) >= cap:
            break
        if not isinstance(row, dict) or str(row.get("status", "")).startswith("merged_into:"):
            continue
        row_id = str(row.get("id", "") or "")
        for other in rows[idx + 1 :]:
            if not isinstance(other, dict) or str(other.get("status", "")).startswith("merged_into:"):
                continue
            other_id = str(other.get("id", "") or "")
            if not row_id or not other_id or row_id == other_id:
                continue
            if not _structural_duplicate(row, other):
                continue
            row_created = _epoch(row.get("created_at"))
            other_created = _epoch(other.get("created_at"))
            if other_created and (not row_created or other_created < row_created):
                canonical = other
                duplicate = row
                canonical_id = other_id
                duplicate_id = row_id
            else:
                canonical = row
                duplicate = other
                canonical_id = row_id
                duplicate_id = other_id
            canonical["evidence_refs"] = _string_list(
                [
                    *(_string_list(canonical.get("evidence_refs"), limit=16)),
                    *(_string_list(duplicate.get("evidence_refs"), limit=16)),
                ],
                limit=16,
            )
            canonical["bound_memory_ids"] = _string_list(
                [
                    *(_string_list(canonical.get("bound_memory_ids"), limit=16)),
                    *(_string_list(duplicate.get("bound_memory_ids"), limit=16)),
                ],
                limit=16,
            )
            canonical["confidence"] = max(
                _bounded_float(canonical.get("confidence"), default=0.5),
                _bounded_float(duplicate.get("confidence"), default=0.5),
            )
            merged = _string_list(canonical.get("merged_from"), limit=16)
            canonical["merged_from"] = _string_list([*merged, duplicate_id], limit=16)
            duplicate["status"] = f"merged_into:{canonical_id}"
            results.append(
                ConsolidationOpResult(
                    op=op,
                    committed=True,
                    source_ids=[canonical_id, duplicate_id],
                    retained_id=canonical_id,
                )
            )
            break
    return results


def _promote_stm(state: dict[str, Any], *, now: int, turn_index: int) -> tuple[list[ConsolidationOpResult], list[dict[str, Any]]]:
    results: list[ConsolidationOpResult] = []
    events: list[dict[str, Any]] = []
    short_rows = state.get("short_term_memory", [])
    if not isinstance(short_rows, list):
        return results, events
    long_rows = state.setdefault("long_term_memory", [])
    if not isinstance(long_rows, list):
        long_rows = []
        state["long_term_memory"] = long_rows
    promoted_ids = {
        str(row.get("promoted_from", "")) for row in long_rows if isinstance(row, Mapping)
    }
    gate = MemoryGate()
    for row in short_rows:
        if len(results) >= MAX_PROMOTIONS_PER_RUN:
            break
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("id", "") or "")
        if not row_id or row_id in promoted_ids:
            continue
        recall_count = int(row.get("recall_count_session", row.get("recall_count", 0)) or 0)
        scores = row.get("recall_scores", [])
        mean_score = _bounded_float(row.get("_m14_7_recall_score"), default=0.0)
        if isinstance(scores, list) and scores:
            mean_score = sum(_bounded_float(item) for item in scores) / len(scores)
        age = max(0, int(now) - _epoch(row.get("written_at", row.get("created_at", now))))
        gate_decision = _mapping(row.get("memory_gate_decision"))
        if not gate_decision:
            for event in reversed(state.get("memory_gate_audit_tail", []) or []):
                if isinstance(event, Mapping) and str(event.get("store_id", "")) == row_id:
                    gate_decision = dict(event)
                    break
        prior_write_score = _bounded_float(gate_decision.get("write_score"), default=0.0)
        if recall_count < 2 or mean_score < LTM_PROMOTION_RECALL_FLOOR or age > MAX_STM_AGE_SECONDS_FOR_PROMOTION or prior_write_score < 0.55:
            results.append(
                ConsolidationOpResult(
                    op="promote_stm_to_ltm",
                    committed=False,
                    source_ids=[row_id],
                    violation_codes=["promotion_criteria_not_met"],
                )
            )
            continue
        evidence_refs = _string_list(row.get("evidence_refs"), limit=8)
        intent = MemoryWriteIntent(
            target="long_term",
            kind=str(row.get("kind", "episode") or "episode"),
            content=str(row.get("content", "") or "")[:400],
            confidence=_bounded_float(row.get("confidence"), default=0.5),
            evidence_refs=evidence_refs,
            identity_relevance=_bounded_float(row.get("identity_relevance")),
            value_proxy=_bounded_float(row.get("value_proxy", row.get("salience")), default=0.5),
            surprise_proxy=_bounded_float(row.get("surprise_proxy"), default=0.5),
            source="m15_consolidation",
            proposer="ConsolidationOwner",
            audit_reason="promote_stm_to_ltm",
        )
        decision = gate.evaluate(intent)
        new_id = f"ltm_promoted_{now}_{len(results)}"
        events.append(
            memory_gate_event(
                event_type="MemoryGateCommitEvent" if decision.commit else "MemoryGateRejectedEvent",
                intent=intent,
                decision=decision,
                turn_index=turn_index,
                now=now,
                store_target="long_term",
                store_id=new_id,
            )
        )
        if decision.commit:
            long_rows.append(
                {
                    "id": new_id,
                    "kind": intent.kind,
                    "content": intent.content,
                    "confidence": intent.confidence,
                    "salience": max(0.55, _bounded_float(row.get("salience"), default=0.5)),
                    "evidence_refs": evidence_refs,
                    "promoted_from": row_id,
                    "created_at": now,
                    "last_recalled_at": row.get("last_recalled_at"),
                    "recall_count": 0,
                    "value_proxy": intent.value_proxy,
                    "identity_relevance": intent.identity_relevance,
                    "memory_gate_decision": decision.to_dict(),
                    "source": "m15_consolidation",
                }
            )
        results.append(
            ConsolidationOpResult(
                op="promote_stm_to_ltm",
                committed=decision.commit,
                source_ids=[row_id],
                new_id=new_id if decision.commit else "",
                violation_codes=list(decision.violation_codes),
            )
        )
    return results, events


def _abstract_paths(state: dict[str, Any], *, ledger: EpisodeLedger, now: int, turn_index: int) -> tuple[list[ConsolidationOpResult], list[dict[str, Any]]]:
    results: list[ConsolidationOpResult] = []
    events: list[dict[str, Any]] = []
    episodes = ledger.recent(64)
    consumed_ids = set()
    try:
        for line in ledger.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, Mapping) and row.get("record_type") == "abstraction_consumed":
                consumed_ids.update(_string_list(row.get("episode_ids"), limit=64))
    except Exception:
        consumed_ids = set()
    buckets: dict[tuple[str, str], list[MemoryDynamicsEpisode]] = {}
    for episode in episodes:
        if episode.episode_id in consumed_ids:
            continue
        buckets.setdefault((fingerprint_class(episode), episode.action), []).append(episode)
    long_rows = state.setdefault("long_term_memory", [])
    if not isinstance(long_rows, list):
        long_rows = []
        state["long_term_memory"] = long_rows
    gate = MemoryGate()
    for (_fp_class, action), rows in buckets.items():
        if len(results) >= MAX_PATH_ABSTRACTIONS_PER_RUN:
            break
        if len(rows) < N_PATH_OCCURRENCES_FOR_ABSTRACTION:
            continue
        mean_delta = sum(row.delta_fe_proxy for row in rows) / len(rows)
        if mean_delta > -0.05:
            continue
        evidence_refs = _string_list([ref for row in rows for ref in row.evidence_refs], limit=8)
        content = f"habit:{action} under repeated low-delta-fe structural state"
        confidence = min(0.95, 0.45 + len(rows) * 0.05 + min(0.25, -mean_delta))
        intent = MemoryWriteIntent(
            target="long_term",
            kind="habit",
            content=content,
            confidence=confidence,
            evidence_refs=evidence_refs or [rows[0].episode_id],
            value_proxy=min(1.0, -mean_delta * 4),
            surprise_proxy=0.55,
            source="m15_consolidation",
            proposer="ConsolidationOwner",
            audit_reason="abstract_repeated_paths",
        )
        decision = gate.evaluate(intent)
        new_id = f"ltm_path_habit_{now}_{len(results)}"
        events.append(
            memory_gate_event(
                event_type="MemoryGateCommitEvent" if decision.commit else "MemoryGateRejectedEvent",
                intent=intent,
                decision=decision,
                turn_index=turn_index,
                now=now,
                store_target="long_term",
                store_id=new_id,
            )
        )
        if decision.commit:
            source_ids = [row.episode_id for row in rows[:8]]
            long_rows.append(
                {
                    "id": new_id,
                    "kind": "habit",
                    "content": content,
                    "confidence": round(confidence, 6),
                    "salience": 0.65,
                    "evidence_refs": evidence_refs or source_ids[:2],
                    "source_episode_ids": source_ids,
                    "created_at": now,
                    "source": "m15_consolidation",
                    "memory_gate_decision": decision.to_dict(),
                }
            )
            with ledger.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "abstraction_consumed",
                            "type": "MemoryDynamicsEpisodeConsumedForAbstractionEvent",
                            "at": now,
                            "turn_index": turn_index,
                            "episode_ids": source_ids,
                            "new_id": new_id,
                            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        results.append(
            ConsolidationOpResult(
                op="abstract_path",
                committed=decision.commit,
                source_ids=[row.episode_id for row in rows[:8]],
                new_id=new_id if decision.commit else "",
                violation_codes=list(decision.violation_codes),
            )
        )
    return results, events


class ConsolidationOwner:
    @staticmethod
    def maybe_run(
        state: dict[str, Any],
        *,
        now: int,
        turn_index: int,
        ledger: EpisodeLedger,
        budget: Mapping[str, Any] | None = None,
    ) -> ConsolidationRunResult:
        budget = _mapping(budget)
        triggered_by = str(budget.get("triggered_by", "idle_cognitive_tick") or "idle_cognitive_tick")
        run_id = _new_id("m15_consolidation")
        m13_state = _mapping(state.get("m13_drive_state"))
        cstate = _mapping(m13_state.get("m15_consolidation"))
        day = _day_key(now)
        runs_by_day = _mapping(cstate.get("runs_by_day"))
        runs_today = int(runs_by_day.get(day, 0) or 0)

        reason = ""
        if bool(state.get("m13_ui_turn_in_progress")):
            reason = "user_turn_in_progress"
        elif runs_today >= int(budget.get("max_runs_per_day", MAX_CONSOLIDATION_RUNS_PER_DAY) or MAX_CONSOLIDATION_RUNS_PER_DAY):
            reason = "budget_exceeded"
        elif ledger.recent(1) == []:
            reason = "ledger_empty"
        elif int(now) - int(cstate.get("last_run_at", 0) or 0) < MIN_RUN_INTERVAL_SECONDS:
            reason = "recently_ran_within_min_interval"
        if reason:
            event = {
                "type": "ConsolidationDeferredEvent",
                "at": now,
                "turn_index": turn_index,
                "reason_code": reason,
                "triggered_by": triggered_by,
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
            return ConsolidationRunResult(ran=False, run_id=run_id, events=[event], deferred_reason=reason)

        op_results: list[ConsolidationOpResult] = []
        gate_events: list[dict[str, Any]] = []
        op_results.extend(_merge_rows(state.get("pending_expectations", []), op="merge_expectation", cap=MAX_MERGES_PER_RUN))
        op_results.extend(_merge_rows(state.get("open_items", []), op="merge_open_item", cap=MAX_MERGES_PER_RUN))
        promote_results, promote_events = _promote_stm(state, now=now, turn_index=turn_index)
        op_results.extend(promote_results)
        gate_events.extend(promote_events)
        abstract_results, abstract_events = _abstract_paths(state, ledger=ledger, now=now, turn_index=turn_index)
        op_results.extend(abstract_results)
        gate_events.extend(abstract_events)

        decay = apply_consolidation_decay_extension(state, now=now, turn_index=turn_index)
        if decay.rows_touched:
            op_results.append(ConsolidationOpResult(op="decay_ltm_salience", committed=True, source_ids=[]))
        if decay.rows_archived:
            op_results.append(ConsolidationOpResult(op="archive_ltm", committed=True, source_ids=[]))

        runs_by_day[day] = runs_today + 1
        cstate["runs_by_day"] = runs_by_day
        cstate["last_run_at"] = now
        cstate["last_run_id"] = run_id
        cstate["last_ops"] = {
            "merges": sum(1 for row in op_results if row.op in {"merge_expectation", "merge_open_item"} and row.committed),
            "promote": sum(1 for row in op_results if row.op == "promote_stm_to_ltm" and row.committed),
            "abstract": sum(1 for row in op_results if row.op == "abstract_path" and row.committed),
            "decay_touched": decay.rows_touched,
            "archived": decay.rows_archived,
        }
        m13_state["m15_consolidation"] = cstate
        state["m13_drive_state"] = m13_state

        events: list[dict[str, Any]] = [
            _op_event(result=result, run_id=run_id, now=now, turn_index=turn_index)
            for result in op_results
            if result.op in CONSOLIDATION_OPS
        ]
        events.extend(gate_events)
        events.extend(decay.events)
        events.append(
            {
                "type": "ConsolidationRunEvent",
                "at": now,
                "turn_index": turn_index,
                "run_id": run_id,
                "triggered_by": triggered_by,
                "ops_attempted": len(op_results),
                "ops_committed": sum(1 for result in op_results if result.committed),
                "ops_rejected": sum(1 for result in op_results if not result.committed),
                "budget_remaining_runs_today": max(0, MAX_CONSOLIDATION_RUNS_PER_DAY - (runs_today + 1)),
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        return ConsolidationRunResult(ran=True, run_id=run_id, events=events, op_results=op_results)
