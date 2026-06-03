"""M15.0 episode ledger for local memory-dynamics trajectories.

The ledger records compact state-action-outcome paths and local free-energy
proxies.  It is intentionally append-only: settlement revisions are stored as
addenda, never by rewriting the original JSONL episode row.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ENGINEERING_PROXY_LABEL = "mvp_local_episode_ledger"
LEDGER_FILENAME = "memory_dynamics_episodes.jsonl"
RECENT_CACHE_LIMIT = 64
BY_LOOKUP_LIMIT = 64
OUTREACH_TRIGGER = "memory_efe_outreach"

FE_PROXY_WEIGHTS: dict[str, float] = {
    "sharing_fe": 0.20,
    "memory_efe_f_memory": 0.25,
    "reward_net_proxy": 0.20,
    "expectation_prediction_error_proxy": 0.20,
    "self_consistency_proxy": 0.15,
}

STATE_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "last_user_turn_index",
    "last_assistant_turn_index",
    "open_items_concrete_count",
    "unsettled_pending_settlement_count",
    "boredom_band",
    "reward_band",
    "behavior_band",
    "relation_band",
    "memory_efe_should_outreach",
    "memory_efe_selected_policy",
    "top_3_traction_actions",
    "top_3_bound_memory_ids",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _epoch(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _round(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _band(value: Any) -> str:
    number = _bounded_float(value)
    if number >= 0.67:
        return "high"
    if number >= 0.35:
        return "medium"
    return "low"


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list | tuple | set) else [value]
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _short_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _top_traction_actions(m13_state: Mapping[str, Any]) -> list[str]:
    traction = _mapping(m13_state.get("traction_by_action"))
    rows: list[tuple[str, float]] = []
    for key, value in traction.items():
        action = str(key).split("|", 1)[0].strip()
        if action:
            rows.append((action, _bounded_float(value)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return sorted({action for action, _score in rows[:3]})[:3]


def _top_bound_memory_ids(state: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in state.get("pending_expectations", []) or []:
        if not isinstance(row, Mapping):
            continue
        for key in ("bound_memory_ids", "evidence_refs"):
            ids.extend(_string_list(row.get(key), limit=8))
    m13_state = _mapping(state.get("m13_drive_state"))
    memory_efe = _mapping(m13_state.get("memory_efe") or state.get("memory_efe"))
    for row in memory_efe.get("eligible_for_efe", []) or []:
        if not isinstance(row, Mapping):
            continue
        ids.extend(_string_list(row.get("bound_memory_ids"), limit=8))
        ids.extend(_string_list(row.get("evidence_refs"), limit=8))
    return sorted(dict.fromkeys(ids))[:3]


def _open_items_concrete_count(state: Mapping[str, Any]) -> int:
    count = 0
    for row in state.get("open_items", []) or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status", "open") or "open") != "open":
            continue
        title = str(row.get("title", row.get("content", "")) or "").strip()
        refs = _string_list(row.get("evidence_refs"), limit=2)
        if title and (refs or len(title) >= 8):
            count += 1
    return count


def _unsettled_pending_count(state: Mapping[str, Any]) -> int:
    m13_state = _mapping(state.get("m13_drive_state"))
    reward = _mapping(m13_state.get("affective_reward_proxy"))
    memory_efe = _mapping(m13_state.get("memory_efe"))
    reward_pending = [
        row for row in reward.get("pending_settlements", []) or [] if isinstance(row, Mapping)
    ]
    memory_pending = [
        row for row in memory_efe.get("pending_settlements", []) or [] if isinstance(row, Mapping)
    ]
    return len(reward_pending) + len(memory_pending)


def _structured_limit_count(value: Any) -> int:
    if isinstance(value, Mapping):
        total = 0
        for key in ("known_limits", "baseline_known_limits", "limits", "constraint_records"):
            nested = value.get(key)
            if isinstance(nested, list):
                total += len([row for row in nested if row])
        return total
    if isinstance(value, list):
        return len([row for row in value if row])
    return 0


def self_consistency_proxy_from_state(state: Mapping[str, Any]) -> float:
    """Read-only M12/M12.1-derived self-consistency proxy.

    Missing M12 data is treated as zero pressure.  The helper deliberately does
    not write to M11/M12 ledgers.
    """
    m12 = _mapping(state.get("m12_user_continuity"))
    conflicts = [
        row for row in m12.get("conflict_records", []) or [] if isinstance(row, Mapping)
    ]
    open_conflicts = [
        row for row in conflicts if str(row.get("resolution_status", "open") or "open") in {"open", "probed"}
    ]
    tension_pressure = min(1.0, len(open_conflicts) / 4.0)

    m12_1 = _mapping(state.get("m12_1_user_personality"))
    structured_limit_total = _structured_limit_count(m12_1.get("known_limits"))
    self_cognition = _mapping(state.get("self_cognition"))
    structured_limit_total += _structured_limit_count(self_cognition.get("known_limits"))
    structured_limit_total += _structured_limit_count(
        _mapping(self_cognition.get("self_continuity")).get("baseline_known_limits")
    )
    reports = _mapping(m12_1.get("latest_reports_by_user"))
    for report in reports.values():
        if isinstance(report, Mapping):
            structured_limit_total += _structured_limit_count(report)
    limit_pressure = min(1.0, structured_limit_total / 4.0)
    return _round(max(tension_pressure, limit_pressure))


def state_fingerprint_payload(
    state: Mapping[str, Any],
    *,
    memory_efe_evaluation: Any | None = None,
    band_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    m13_state = _mapping(state.get("m13_drive_state"))
    temporal = _mapping(state.get("temporal_state"))
    reward = _mapping(m13_state.get("affective_reward_proxy"))
    boredom = _mapping(m13_state.get("boredom"))
    memory_efe = _mapping(m13_state.get("memory_efe"))
    bands = _mapping(band_summary)
    if memory_efe_evaluation is not None:
        should_outreach = bool(getattr(memory_efe_evaluation, "should_outreach", False))
        selected_policy = str(getattr(memory_efe_evaluation, "selected_policy", "") or "")
    else:
        should_outreach = bool(memory_efe.get("should_outreach", False))
        selected_policy = str(memory_efe.get("selected_policy", "") or "")
    payload = {
        "last_user_turn_index": _epoch(temporal.get("last_user_turn_index", temporal.get("last_turn_index"))),
        "last_assistant_turn_index": _epoch(temporal.get("last_turn_index")),
        "open_items_concrete_count": _open_items_concrete_count(state),
        "unsettled_pending_settlement_count": _unsettled_pending_count(state),
        "boredom_band": str(bands.get("boredom_band") or _band(boredom.get("boredom_level"))),
        "reward_band": str(bands.get("affective_reward_band") or bands.get("reward_band") or _band(reward.get("last_net_reward_proxy"))),
        "behavior_band": str(bands.get("behavioral_pull_band") or bands.get("behavior_band") or "low"),
        "relation_band": str(bands.get("relation_path_precision_band") or bands.get("relation_band") or "low"),
        "memory_efe.should_outreach": should_outreach,
        "memory_efe.selected_policy": selected_policy,
        "top_3_traction_actions": _top_traction_actions(m13_state),
        "top_3_bound_memory_ids": _top_bound_memory_ids(state),
    }
    return {
        "last_user_turn_index": payload["last_user_turn_index"],
        "last_assistant_turn_index": payload["last_assistant_turn_index"],
        "open_items_concrete_count": payload["open_items_concrete_count"],
        "unsettled_pending_settlement_count": payload["unsettled_pending_settlement_count"],
        "boredom_band": payload["boredom_band"],
        "reward_band": payload["reward_band"],
        "behavior_band": payload["behavior_band"],
        "relation_band": payload["relation_band"],
        "memory_efe_should_outreach": payload["memory_efe.should_outreach"],
        "memory_efe_selected_policy": payload["memory_efe.selected_policy"],
        "top_3_traction_actions": payload["top_3_traction_actions"],
        "top_3_bound_memory_ids": payload["top_3_bound_memory_ids"],
    }


def state_fingerprint(
    state: Mapping[str, Any],
    *,
    memory_efe_evaluation: Any | None = None,
    band_summary: Mapping[str, Any] | None = None,
) -> str:
    return _short_hash(
        state_fingerprint_payload(
            state,
            memory_efe_evaluation=memory_efe_evaluation,
            band_summary=band_summary,
        )
    )


def aggregate_fe_components(
    state: Mapping[str, Any],
    *,
    memory_dynamics: Mapping[str, Any] | None = None,
    memory_efe_evaluation: Any | None = None,
    reward_evaluation: Any | None = None,
    settlement_event: Mapping[str, Any] | None = None,
    conscious_plan: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    control = _mapping(_mapping(memory_dynamics).get("control_guidance"))
    sharing = _mapping(control.get("sharing_policy"))
    current_fe = sharing.get("current_free_energy")
    if current_fe is None:
        current_fe = 1.0 - _bounded_float(sharing.get("net_free_energy_reduction"), default=0.0)

    m13_state = _mapping(state.get("m13_drive_state"))
    memory_efe = _mapping(m13_state.get("memory_efe"))
    f_memory = getattr(memory_efe_evaluation, "f_memory", memory_efe.get("f_memory", 0.0))
    reward_state = _mapping(m13_state.get("affective_reward_proxy"))
    reward_net = getattr(reward_evaluation, "net_affective_reward_proxy", reward_state.get("last_net_reward_proxy", 0.0))
    prediction_error = abs(
        _round(getattr(reward_evaluation, "prediction_error_proxy", reward_state.get("last_prediction_error_proxy", 0.0)))
    )
    if settlement_event is not None and "prediction_error_proxy" in settlement_event:
        prediction_error = abs(_round(settlement_event.get("prediction_error_proxy")))
    if conscious_plan:
        statuses = [
            str(row.get("status", "") or "").lower()
            for row in conscious_plan.get("expectation_results", []) or []
            if isinstance(row, Mapping)
        ]
        if any(status == "violated" for status in statuses):
            prediction_error = max(prediction_error, 0.35)
        elif any(status == "uncertain" for status in statuses):
            prediction_error = max(prediction_error, 0.18)
        elif any(status == "confirmed" for status in statuses):
            prediction_error = min(prediction_error, 0.08)

    return {
        "sharing_fe": _round(_bounded_float(current_fe)),
        "memory_efe_f_memory": _round(_bounded_float(f_memory)),
        "reward_net_proxy": _round(_bounded_float(reward_net)),
        "expectation_prediction_error_proxy": _round(_bounded_float(prediction_error)),
        "self_consistency_proxy": self_consistency_proxy_from_state(state),
    }


def aggregate_fe_proxy(components: Mapping[str, Any]) -> float:
    total = 0.0
    for key, weight in FE_PROXY_WEIGHTS.items():
        value = _bounded_float(components.get(key), default=0.0)
        if key == "reward_net_proxy":
            value = 1.0 - value
        total += weight * value
    return _round(total)


def memory_gate_decision_from_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(event) for event in events if isinstance(event, Mapping)]
    if not rows:
        return {
            "commit": False,
            "write_score": 0.0,
            "factors": {},
            "violation_codes": ["no_memory_write_intent"],
        }
    latest = rows[-1]
    best = max(rows, key=lambda row: _bounded_float(row.get("write_score")))
    return {
        "commit": any(str(row.get("type", "")) == "MemoryGateCommitEvent" for row in rows),
        "write_score": _round(best.get("write_score")),
        "factors": dict(_mapping(best.get("factors"))),
        "violation_codes": _string_list(latest.get("violation_codes"), limit=8),
        "events": [
            {
                "type": str(row.get("type", "")),
                "store_target": str(row.get("store_target", "")),
                "store_id": str(row.get("store_id", "")),
                "write_score": _round(row.get("write_score")),
                "factors": dict(_mapping(row.get("factors"))),
                "violation_codes": _string_list(row.get("violation_codes"), limit=8),
            }
            for row in rows[-6:]
        ],
    }


@dataclass(frozen=True)
class MemoryDynamicsEpisode:
    episode_id: str
    at: int
    turn_index: int
    phase: str
    state_fingerprint: str
    action: str
    action_trigger: str
    evidence_refs: list[str]
    fe_proxy_before: float
    fe_proxy_after: float
    delta_fe_proxy: float
    components_before: dict[str, float]
    components_after: dict[str, float]
    memory_gate_decision: dict[str, Any]
    outcome_summary: str
    engineering_proxy_label: str = ENGINEERING_PROXY_LABEL
    record_type: str = "episode"
    state_fingerprint_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "episode_id": self.episode_id,
            "at": self.at,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "state_fingerprint": self.state_fingerprint,
            "state_fingerprint_payload": dict(self.state_fingerprint_payload),
            "action": self.action,
            "action_trigger": self.action_trigger,
            "evidence_refs": list(self.evidence_refs[:12]),
            "fe_proxy_before": _round(self.fe_proxy_before),
            "fe_proxy_after": _round(self.fe_proxy_after),
            "delta_fe_proxy": _round(self.delta_fe_proxy),
            "components_before": dict(self.components_before),
            "components_after": dict(self.components_after),
            "memory_gate_decision": dict(self.memory_gate_decision),
            "outcome_summary": self.outcome_summary,
            "engineering_proxy_label": self.engineering_proxy_label,
        }

    @staticmethod
    def from_mapping(row: Mapping[str, Any]) -> "MemoryDynamicsEpisode":
        return MemoryDynamicsEpisode(
            episode_id=str(row.get("episode_id", "")),
            at=_epoch(row.get("at")),
            turn_index=_epoch(row.get("turn_index")),
            phase=str(row.get("phase", "")),
            state_fingerprint=str(row.get("state_fingerprint", "")),
            action=str(row.get("action", "")),
            action_trigger=str(row.get("action_trigger", "")),
            evidence_refs=_string_list(row.get("evidence_refs"), limit=12),
            fe_proxy_before=_round(row.get("fe_proxy_before")),
            fe_proxy_after=_round(row.get("fe_proxy_after")),
            delta_fe_proxy=_round(row.get("delta_fe_proxy")),
            components_before={k: _round(v) for k, v in _mapping(row.get("components_before")).items()},
            components_after={k: _round(v) for k, v in _mapping(row.get("components_after")).items()},
            memory_gate_decision=dict(_mapping(row.get("memory_gate_decision"))),
            outcome_summary=str(row.get("outcome_summary", "")),
            state_fingerprint_payload=dict(_mapping(row.get("state_fingerprint_payload"))),
        )


def build_episode(
    *,
    at: int,
    turn_index: int,
    phase: str,
    state: Mapping[str, Any],
    action: str,
    action_trigger: str,
    evidence_refs: Iterable[Any] | None,
    components_before: Mapping[str, Any],
    components_after: Mapping[str, Any],
    memory_gate_decision: Mapping[str, Any] | None = None,
    outcome_summary: str = "uncertain",
    memory_efe_evaluation: Any | None = None,
    band_summary: Mapping[str, Any] | None = None,
) -> MemoryDynamicsEpisode:
    fp_payload = state_fingerprint_payload(
        state,
        memory_efe_evaluation=memory_efe_evaluation,
        band_summary=band_summary,
    )
    fp = _short_hash(fp_payload)
    before = {k: _round(components_before.get(k, 0.0)) for k in FE_PROXY_WEIGHTS}
    after = {k: _round(components_after.get(k, 0.0)) for k in FE_PROXY_WEIGHTS}
    fe_before = aggregate_fe_proxy(before)
    fe_after = aggregate_fe_proxy(after)
    episode_seed = {
        "at": at,
        "turn_index": turn_index,
        "phase": phase,
        "action": action,
        "trigger": action_trigger,
        "fingerprint": fp,
    }
    return MemoryDynamicsEpisode(
        episode_id=f"m15_ep_{_short_hash(episode_seed)}",
        at=int(at),
        turn_index=int(turn_index),
        phase=phase,
        state_fingerprint=fp,
        state_fingerprint_payload=fp_payload,
        action=action,
        action_trigger=action_trigger,
        evidence_refs=_string_list(list(evidence_refs or []), limit=12),
        fe_proxy_before=fe_before,
        fe_proxy_after=fe_after,
        delta_fe_proxy=_round(fe_after - fe_before),
        components_before=before,
        components_after=after,
        memory_gate_decision=dict(memory_gate_decision or {}),
        outcome_summary=outcome_summary,
    )


class EpisodeLedger:
    def __init__(self, session_root: Path, *, cache_limit: int = RECENT_CACHE_LIMIT) -> None:
        self.session_root = Path(session_root)
        self.path = self.session_root / LEDGER_FILENAME
        self.cache_limit = max(1, int(cache_limit))
        self._recent: deque[MemoryDynamicsEpisode] = deque(maxlen=self.cache_limit)
        self._loaded = False

    def _ensure(self) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _load_recent_once(self) -> None:
        if self._loaded:
            return
        self._ensure()
        rows: deque[MemoryDynamicsEpisode] = deque(maxlen=self.cache_limit)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping) and row.get("record_type", "episode") == "episode":
                    rows.append(MemoryDynamicsEpisode.from_mapping(row))
        self._recent = rows
        self._loaded = True

    def append(self, episode: MemoryDynamicsEpisode) -> None:
        self._ensure()
        self._load_recent_once()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._recent.append(episode)

    def append_settlement_event(
        self,
        *,
        episode_id: str,
        at: int,
        turn_index: int,
        new_outcome_summary: str,
        fe_proxy_after_revised: float,
        components_after_revised: Mapping[str, Any],
        settlement_event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure()
        original = self.find_episode(episode_id=episode_id)
        fe_before = original.fe_proxy_before if original is not None else 0.0
        delta = _round(fe_proxy_after_revised - fe_before)
        event = {
            "record_type": "settlement_addendum",
            "type": "MemoryDynamicsEpisodeSettledEvent",
            "episode_id": episode_id,
            "at": int(at),
            "turn_index": int(turn_index),
            "new_outcome_summary": str(new_outcome_summary or "uncertain"),
            "fe_proxy_after_revised": _round(fe_proxy_after_revised),
            "delta_fe_proxy_revised": delta,
            "components_after_revised": {
                key: _round(value) for key, value in dict(components_after_revised).items()
            },
            "settlement_event": dict(settlement_event or {}),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def append_prediction_settlement_addendum(
        self,
        *,
        at: int,
        turn_index: int,
        source_episode_id: str,
        prediction_id: str,
        prediction_type: str,
        outcome: str,
        committed_confidence: float,
        prediction_error: float | None,
        brier_score: float | None,
        evidence_refs: Iterable[Any] | None = None,
        reason_codes: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure()
        event = {
            "record_type": "prediction_settlement_addendum",
            "type": "PredictionSettlementAddendum",
            "episode_id": str(source_episode_id or ""),
            "source_episode_id": str(source_episode_id or ""),
            "prediction_id": str(prediction_id or ""),
            "prediction_type": str(prediction_type or ""),
            "outcome": str(outcome or ""),
            "committed_confidence": _round(committed_confidence),
            "m17_prediction_error": _round(prediction_error) if prediction_error is not None else None,
            "m17_brier_score": _round(brier_score) if brier_score is not None else None,
            "evidence_refs": _string_list(list(evidence_refs or []), limit=12),
            "reason_codes": _string_list(list(reason_codes or []), limit=8),
            "at": int(at),
            "turn_index": int(turn_index),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def recent(self, n: int = RECENT_CACHE_LIMIT) -> list[MemoryDynamicsEpisode]:
        self._load_recent_once()
        limit = max(0, int(n))
        return list(self._recent)[-limit:] if limit else []

    def _iter_episodes_latest_first(self) -> Iterable[MemoryDynamicsEpisode]:
        self._ensure()
        rows: list[MemoryDynamicsEpisode] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping) and row.get("record_type", "episode") == "episode":
                    rows.append(MemoryDynamicsEpisode.from_mapping(row))
        yield from reversed(rows)

    def find_episode(
        self,
        *,
        episode_id: str = "",
        turn_index: int | None = None,
        phase: str = "",
    ) -> MemoryDynamicsEpisode | None:
        for episode in reversed(self.recent(self.cache_limit)):
            if episode_id and episode.episode_id != episode_id:
                continue
            if turn_index is not None and episode.turn_index != int(turn_index):
                continue
            if phase and episode.phase != phase:
                continue
            return episode
        for episode in self._iter_episodes_latest_first():
            if episode_id and episode.episode_id != episode_id:
                continue
            if turn_index is not None and episode.turn_index != int(turn_index):
                continue
            if phase and episode.phase != phase:
                continue
            return episode
        return None

    def by_fingerprint(self, state_fingerprint: str, *, limit: int = BY_LOOKUP_LIMIT) -> list[MemoryDynamicsEpisode]:
        out: list[MemoryDynamicsEpisode] = []
        for episode in self._iter_episodes_latest_first():
            if episode.state_fingerprint == state_fingerprint:
                out.append(episode)
            if len(out) >= limit:
                break
        return out

    def by_action(self, action: str, *, limit: int = 16) -> list[MemoryDynamicsEpisode]:
        out: list[MemoryDynamicsEpisode] = []
        for episode in self._iter_episodes_latest_first():
            if episode.action == action:
                out.append(episode)
            if len(out) >= limit:
                break
        return out

    def search(self, action_trigger: str, *, limit: int = 16) -> list[MemoryDynamicsEpisode]:
        out: list[MemoryDynamicsEpisode] = []
        for episode in self._iter_episodes_latest_first():
            if episode.action_trigger == action_trigger:
                out.append(episode)
            if len(out) >= limit:
                break
        return out


def drive_pull_bonus_for_action(
    episodes: Iterable[MemoryDynamicsEpisode],
    *,
    state_fingerprint: str,
    action: str,
) -> float:
    matches = [
        episode
        for episode in episodes
        if episode.state_fingerprint == state_fingerprint and episode.action == action
    ]
    if not matches:
        return 0.0
    mean_improvement = sum(-episode.delta_fe_proxy for episode in matches) / len(matches)
    return _round(max(-0.1, min(0.1, mean_improvement * 0.2)))


def outreach_margin_history_adjustment(episodes: Iterable[MemoryDynamicsEpisode]) -> float:
    rows = [episode for episode in episodes if episode.action_trigger == OUTREACH_TRIGGER]
    if not rows:
        return 0.0
    mean_delta = sum(episode.delta_fe_proxy for episode in rows) / len(rows)
    if mean_delta <= 0.0:
        return 0.0
    return _round(min(0.12, mean_delta * 0.25))
