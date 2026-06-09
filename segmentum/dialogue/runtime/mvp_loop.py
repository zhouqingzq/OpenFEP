"""Minimal LLM-driven persona loop for the dialogue MVP.

This module is intentionally narrower than the research runtime.  It keeps the
MVP user-facing contract explicit: durable self files, LLM-based conscious
planning, memory retrieval, LLM-based thinking/reply generation, and guarded
state writes.
"""

from __future__ import annotations

from types import MappingProxyType

from dataclasses import dataclass, field
import copy
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from segmentum.user_model import (
    M11RuntimeConfig,
    M11RuntimeState,
    PredictionCalibrationState,
    SourceReliabilityLedger,
    SocialSharingCandidate,
    UserModel,
    UserPredictionLedger,
    abstract_memory_content,
    attach_prediction_source_episode,
    boundary_strength_from_constraints,
    candidate_from_memory,
    decide_social_sharing,
    detect_explicit_secrecy,
    memory_shareability,
    run_m11_turn,
    sharing_feedback_negative,
    update_regret_bias,
    validate_extractor_output,
)
from segmentum.cognitive_events import CognitiveEventBus
from segmentum.user_model.llm_extractor import ExtractorValidationError, noop_extraction
from segmentum.user_continuity import (
    IdentityProfile,
    M12RuntimeConfig,
    M12RuntimeState,
    run_m12_turn,
    select_reply_policy,
)
from segmentum.user_personality import (
    M121RuntimeConfig,
    M121RuntimeState,
    build_step_extractor_prompt,
    run_m12_1_tick,
)
from segmentum.reciprocal_role import (
    M122RuntimeConfig,
    M122RuntimeState,
    build_extractor_prompt as build_m12_2_extractor_prompt,
    run_m12_2_tick,
)
from segmentum.dialogue.runtime.m13_boredom import (
    M13BoredomEvaluator,
    apply_post_turn_boredom_state,
    boredom_band,
    prompt_safe_control_guidance_for_thinking,
    prompt_safe_m13_boredom_diagnostics,
)
from segmentum.dialogue.runtime.m13_drive import (
    M13DriveEvaluator,
    apply_post_turn_m13_state,
    default_m13_drive_state,
    merge_drive_guidance_into_control,
    normalize_m13_drive_state,
    normalize_recorded_reply_action,
    prompt_safe_m13_state_summary,
    prompt_safe_m13_turn_diagnostics,
    resolve_m13_safety_repair,
)
from segmentum.dialogue.runtime.m13_idle import (
    ENGINEERING_PROXY_LABEL as IDLE_ENGINEERING_PROXY_LABEL,
    evaluate_idle_structural_pre_filter,
    evaluate_idle_tick,
    mark_idle_audit_logged,
    mark_idle_introspection_consumed,
    merge_idle_introspection_into_initiative,
    normalize_idle_introspection_state,
    set_idle_introspection_user_opt_in,
    should_persist_idle_audit_events,
)
from segmentum.dialogue.runtime.m13_initiative import (
    PROACTIVE_SURROGATE_USER_TEXT,
    ProactiveTurnProposal,
    assess_proactive_delivery_semantics,
    build_proactive_thinking_user_text,
    build_proposal_from_target,
    evaluate_proactive_initiative,
    mark_outreach_via_introspection,
    mark_proactive_turn_consumed,
    merge_initiative_into_m13_state,
    normalize_initiative_state,
    proactive_delivery_gate_reason,
    proposal_from_initiative_state,
    record_target_assessor_reject_backoff,
    repair_proactive_count_from_log,
    set_initiative_proactive_policy_profile,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m14_3_proactive_alignment import (
    ProactiveTarget,
    classify_proactive_target_reject_reason,
    select_proactive_target,
)
from segmentum.dialogue.runtime.m13_memory_efe import (
    apply_memory_efe_state,
    evaluate_memory_efe,
    merge_memory_efe_guidance_into_control,
    prompt_safe_m13_memory_efe_diagnostics,
    register_memory_efe_outreach_settlement,
    settle_memory_efe_outreach,
    normalize_expectations_for_efe,
)
from segmentum.dialogue.runtime.m14_idle_owners import (
    MemoryConsolidationOwner,
    OpenItemPatchOwner,
    OwnerCommitResult,
    SelfCognitionPatchOwner,
    count_session_idle_patches,
)
from segmentum.dialogue.runtime.m14_idle_reflector import (
    M14_ENGINEERING_PROXY_LABEL,
    apply_idle_drive_rules,
    build_conscious_idle_prompt,
    build_idle_context,
    build_structural_idle_plan,
    empty_conscious_idle_plan,
    idle_retrieval_keywords,
    normalize_conscious_idle_plan,
)
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    BackgroundBudgetExhausted,
    BackgroundLLMMeter,
    M14_1_ENGINEERING_PROXY_LABEL,
    check_background_budgets,
    enqueue_outreach_proposal,
    load_queued_outreach,
    maybe_rollover_daily_counters,
    merge_background_continuity_into_initiative,
    normalize_background_continuity_state,
    outreach_suppression_is_transient,
    pop_next_pending_outreach,
    record_queued_outreach_delivery_attempt,
    record_background_tick,
    save_queued_outreach,
    session_file_lock,
    set_background_continuity_opt_in,
    update_queued_outreach_status,
)
from segmentum.dialogue.runtime.m14_7_memory_decay import apply_memory_decay_tick
from segmentum.dialogue.runtime.m14_7_memory_gate import (
    MemoryGate,
    MemoryWriteIntent,
    intent_from_mapping,
    memory_gate_event,
)
from segmentum.dialogue.runtime.m14_7_recall_scoring import explain_recall_candidate, score_recall_candidate
from segmentum.dialogue.runtime.m15_episode_ledger import (
    ENGINEERING_PROXY_LABEL as M15_ENGINEERING_PROXY_LABEL,
    EpisodeLedger,
    aggregate_fe_components,
    aggregate_fe_proxy,
    build_episode,
    memory_gate_decision_from_events,
    state_fingerprint,
)
from segmentum.dialogue.runtime.m15_consolidation import ConsolidationOwner
from segmentum.dialogue.runtime.m15_meta_control import (
    apply_reflection_focus_intent,
    consume_recall_breadth_intent,
    detect_and_emit_intents,
)
from segmentum.dialogue.runtime.m15_3_cleanup_control import CleanupOwner, detect_cleanup_intents
from segmentum.m17_12_path_b_bridge import (
    apply_path_b_settlement_writeback,
    build_path_b_recall_bridge,
    merge_path_b_field_guidance,
    register_prediction_provenance,
)
from segmentum.dialogue.runtime.m14_self_continuity import (
    MIN_BASELINE_UPDATE_CONFIDENCE,
    apply_self_cognition_patch_to_continuity,
    attach_self_continuity,
    build_self_continuity_snapshot,
    get_self_continuity_from_state,
    note_idle_tick,
    run_self_review_tick,
    should_run_self_review,
)
from segmentum.dialogue.runtime.m19_self_expectation import (
    M19_ENGINEERING_PROXY_LABEL,
    apply_conscious_self_expectation_proposals,
    apply_idle_self_expectation_review,
    apply_m19_traction_proposals_to_m13,
    apply_self_expectation_post_turn,
    build_self_repair_guidance,
    collect_m19_audit_evidence_ids,
    default_self_expectation_state,
    normalize_self_expectation_outcome_results,
    normalize_self_response_expectation_proposals,
    prompt_safe_self_expectation_summary,
    prompt_safe_state_with_self_expectation_summary,
)
from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    ActiveCommitmentAdapter,
    GradedCorrectionDispatcher,
    SettlementScheduler,
    build_active_commitment_created_event,
    build_correction_deferred_event,
    build_correction_rejected_event,
    init_owner_observability_for_commitment,
    record_active_commitment_event,
    record_pending_commitment,
    update_commitment_registry_diagnostics,
    update_graded_correction_diagnostics,
    wrap_self_response_expectation_proposal,
)
from segmentum.dialogue.runtime.loop_invariants import (
    LoopInvariants,
    build_minimum_loop_coverage_missed_event,
)
from segmentum.dialogue.runtime.policy_producer import (
    PolicyProducer,
    build_policy_admitted_event,
)
from segmentum.dialogue.runtime.same_turn_surface import (
    SameTurnSurfaceSettler,
    SameTurnSurfaceVerdict,
    build_same_turn_surface_verdict_event,
)
from segmentum.dialogue.runtime.m18_7_attribution import (
    build_m18_7_2_minimal_degraded_event as _build_m18_7_2_minimal_degraded_event,
    build_m18_7_minimal_prompt as _build_m18_7_minimal_prompt,
    emit_m18_7_2_attribution_for_turn as _emit_m18_7_2_attribution_for_turn,
    emit_m18_7_attribution_for_turn as _emit_m18_7_attribution_for_turn,
    normalize_addressee_hypothesis as _normalize_m18_7_addressee_hypothesis,
    normalize_reaction_attribution_hypothesis as _normalize_m18_7_reaction_attribution_hypothesis,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    AddresseeTargetMatchLLMJudgeSettler as _AddresseeTargetMatchLLMJudgeSettler,
    ReactionAttributionMatchLLMJudgeSettler as _ReactionAttributionMatchLLMJudgeSettler,
    build_addressee_target_match_admitted_event as _emit_addressee_target_match_admitted_event,
    build_reaction_attribution_match_admitted_event as _emit_reaction_attribution_match_admitted_event,
    produce_m20_4_attribution_commitments as _produce_m20_4_attribution_commitments,
)
from segmentum.dialogue.runtime.m20_4_1_same_turn_gate import (
    REASON_GATE_FIRED as _M20_4_1_REASON_GATE_FIRED,
    clear_pending_override as _m20_4_1_clear_pending_override,
    get_pending_override as _m20_4_1_get_pending_override,
    same_turn_addressee_hypothesis_gate as _run_m20_4_1_same_turn_gate,
)
from segmentum.dialogue.runtime.active_commitment_grader import (
    route_expire,
    route_microadjust,
    route_next_turn,
    route_revoke,
    route_same_turn,
    route_slow_promote,
)
from segmentum.dialogue.runtime.active_commitment_settlers import (
    BehavioralPullShiftSilentSettler,
    BoundaryHandledLLMJudgeSettler,
    ExpectationOutcomeMatchDeterministicSettler,
    IdentityVoiceMatchLLMJudgeSettler,
    InitiativeTimingMatchHybridSettler,
    PredictionErrorBandDeterministicSettler,
)
from segmentum.dialogue.runtime.m13_reward import (
    M13RewardEvaluator,
    apply_post_turn_m13_reward_state,
    apply_reward_pull_connection,
    evaluate_pre_turn_reward_proxy,
    list_assessable_pending_rows,
    merge_affective_guidance_into_control,
    normalize_affective_reward_proxy_state,
    normalize_user_reaction_assessment,
    pending_diagnostics_summary_for_assessor,
    prompt_safe_m13_reward_diagnostics,
    prompt_safe_m13_reward_ui_labels,
    observation_channels_from_bus,
    settle_pending_m13_actions,
)


SYSTEM_FILE_DEFAULTS: dict[str, Any] = {
    "self_cognition": {
        "summary": "",
        "current_self_view": "",
        "identity_tensions": [],
        "stable_values": [],
        "known_limits": [],
        "calibrated_tendencies": [],
        "repair_priors": [],
        "patch_history": [],
    },
    "short_term_memory": [],
    "long_term_memory": [],
    "pending_expectations": [],
    "open_items": [],
    "self_basic_facts": {
        "name": "",
        "background": [],
        "relationships": [],
        "do_not_invent": [
            "Do not invent biography, work history, family history, or fixed relationships unless supported by memory.",
        ],
    },
    "habit_traits": {
        "big_five": {},
        "conversation_habits": [],
        "learned_conversation_habits": [],
        "defense_style": [],
        "memory_policy": [],
    },
    "self_expectation_state": default_self_expectation_state(),
    "temporal_state": {
        "last_turn_at": None,
        "last_user_turn_at": None,
        "last_assistant_turn_at": None,
        "last_turn_index": None,
        "last_user_text": "",
        "last_reply": "",
        "last_elapsed_seconds": None,
        "last_time_gap_label": "first_turn",
    },
    "m11_user_models": {},
    "m12_identity_continuity_enabled": False,
    "m12_user_continuity": {
        "profiles_by_user": {},
        "claim_ledger": {"entries": []},
        "conflict_records": [],
    },
    "m12_1_personality_enabled": False,
    "m12_1_user_personality": {
        "profiles_by_user": {},
        "latest_reports_by_user": {},
        "run_records_by_user": {},
        "consecutive_step1_insufficient_by_user": {},
    },
    "m12_2_reciprocal_role_enabled": False,
    "m12_2_reciprocal_role": {
        "models_by_user": {},
        "run_records_by_user": {},
    },
    "social_sharing_policy": {
        "regret_bias": 0.0,
        "learned_boundaries": [],
    },
    "relationship_value_memories": {
        "by_user": {},
    },
    "m13_drive_state": {},
    "m17_path_b_bridge": {
        "prediction_provenance": {},
        "last_recall_bridge": {},
        "last_settlement_writeback": {},
        "last_turn_index": -1,
    },
    # M18.7 attribution hypothesis surface. Bounded rolling
    # window (≤8 entries), written by the M18.7.2 minimal
    # orchestrator and read by the M20.4 producer, M20.4.1
    # gate, and M18.7.1 calibration runner. Persisted so
    # the calibration runner's `store.load()` can read it
    # after a `run_turn` completes.
    "m18_7_attribution_hypotheses": [],
}

SYSTEM_FILE_NAMES: dict[str, str] = {
    key: f"{key}.json" for key in SYSTEM_FILE_DEFAULTS
}

SHARED_STATE_KEYS: frozenset[str] = frozenset(
    {
        "m12_2_reciprocal_role_enabled",
        "m12_2_reciprocal_role",
    }
)

PERSONA_ANALYSIS_KEYS = (
    "persona_name",
    "source_role_evidence",
    *SYSTEM_FILE_DEFAULTS.keys(),
)


def _system_file_default(key: str) -> Any:
    if key == "m13_drive_state":
        return default_m13_drive_state()
    default = SYSTEM_FILE_DEFAULTS[key]
    if isinstance(default, dict):
        return copy.deepcopy(default)
    if isinstance(default, list):
        return copy.deepcopy(default)
    return default


def _utc_timestamp() -> int:
    return int(time.time())


def _local_time_read(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(timestamp))


def _time_gap_label(elapsed_seconds: int | None) -> str:
    if elapsed_seconds is None:
        return "first_turn"
    if elapsed_seconds <= 120:
        return "immediate"
    if elapsed_seconds <= 1800:
        return "short_gap"
    if elapsed_seconds <= 21600:
        return "medium_gap"
    return "long_gap"


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _memory_sort_key(item: Mapping[str, Any]) -> tuple[float, str]:
    raw_created = item.get("created_at", 0)
    try:
        created_at = float(raw_created)
    except (TypeError, ValueError):
        created_at = 0.0
    return (created_at, str(item.get("id", "")))


def _memory_identity(item: Mapping[str, Any], index: int) -> str:
    item_id = str(item.get("id", "")).strip()
    if item_id:
        return f"id:{item_id}"
    content = str(item.get("content", "")).strip()
    source_user = str(item.get("source_user_id", "")).strip()
    created = str(item.get("created_at", "")).strip()
    return f"anon:{source_user}:{created}:{content[:160]}:{index}"


def _merge_recent_memory(*groups: Any, limit: int = 96) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for index, item in enumerate(group):
            if not isinstance(item, Mapping):
                continue
            key = _memory_identity(item, index)
            if key not in merged:
                order.append(key)
            merged[key] = dict(item)
    values = [merged[key] for key in order if key in merged]
    values.sort(key=_memory_sort_key)
    bounded_limit = max(1, int(limit or 96))
    return values[-bounded_limit:]


def _json_text(value: Any, *, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text[:limit]


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("LLM response content was empty; expected a JSON object")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError(
                "LLM response content was not a JSON object; "
                f"first characters: {cleaned[:120]!r}"
            )
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response contained malformed JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()[:240]]
    if isinstance(value, list):
        return [str(item).strip()[:240] for item in value[:limit] if str(item).strip()]
    return []


def _string_list_of_mappings(
    value: Any,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value[:limit]:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_string_list(
    value: Any,
    *,
    limit: int = 8,
    item_max_chars: int = 64,
) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()[:item_max_chars]
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _unique_bounded_strings(
    values: list[str],
    *,
    limit: int = 8,
    item_max_chars: int = 64,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()[:item_max_chars]
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _safe_user_id(speaker_name: str) -> str:
    name = str(speaker_name or "").strip() or "default_user"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return safe.strip("_") or "default_user"


def _bounded_float(value: Any, *, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _bounded_group_turn_envelope(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    envelope = _mapping(raw)
    payload: dict[str, Any] = {}
    speaker_participant_id = str(envelope.get("speaker_participant_id", "") or "").strip()[:64]
    if speaker_participant_id:
        payload["speaker_participant_id"] = speaker_participant_id
    reply_to_turn_id = str(envelope.get("reply_to_turn_id", "") or "").strip()[:120]
    if reply_to_turn_id:
        payload["reply_to_turn_id"] = reply_to_turn_id
    visible = _bounded_string_list(envelope.get("visible_participant_ids"), limit=8, item_max_chars=64)
    if visible:
        payload["visible_participant_ids"] = visible
    addressed = _bounded_string_list(envelope.get("addressed_participant_ids"), limit=8, item_max_chars=64)
    if addressed:
        payload["addressed_participant_ids"] = addressed
    mentioned = _bounded_string_list(envelope.get("mentioned_participant_ids"), limit=8, item_max_chars=64)
    if mentioned:
        payload["mentioned_participant_ids"] = mentioned
    quoted = _bounded_string_list(envelope.get("quoted_turn_ids"), limit=8, item_max_chars=120)
    if quoted:
        payload["quoted_turn_ids"] = quoted
    explicit = _bounded_string_list(envelope.get("explicit_mentions"), limit=8, item_max_chars=64)
    if explicit:
        payload["explicit_mentions"] = explicit
    surface_intent = str(envelope.get("surface_intent", "") or "").strip()[:32]
    if surface_intent:
        payload["surface_intent"] = surface_intent
    platform_command = str(envelope.get("platform_command", "") or "").strip()[:64]
    if platform_command:
        payload["platform_command"] = platform_command
    assistant_surface_label = str(envelope.get("assistant_surface_label", "") or "").strip()[:64]
    if assistant_surface_label:
        payload["assistant_surface_label"] = assistant_surface_label
    return payload


def _build_group_turn_binding(
    *,
    display_name: str,
    user_id: str,
    group_turn_envelope: Mapping[str, Any] | None,
    entity_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    envelope = _bounded_group_turn_envelope(group_turn_envelope)
    speaker_participant_id = str(envelope.get("speaker_participant_id", "") or "").strip() or user_id
    addressed = _bounded_string_list(envelope.get("addressed_participant_ids"), limit=8, item_max_chars=64)
    mentioned = _bounded_string_list(envelope.get("mentioned_participant_ids"), limit=8, item_max_chars=64)
    explicit_mentions = _bounded_string_list(envelope.get("explicit_mentions"), limit=8, item_max_chars=64)
    quoted_turn_ids = _bounded_string_list(envelope.get("quoted_turn_ids"), limit=8, item_max_chars=120)
    visible = _unique_bounded_strings(
        [
            speaker_participant_id,
            *_bounded_string_list(envelope.get("visible_participant_ids"), limit=8, item_max_chars=64),
            *addressed,
            *mentioned,
        ],
        limit=8,
        item_max_chars=64,
    )
    target_person_hint = str(_mapping(entity_binding).get("target_person", "") or "").strip()
    candidate_targets = _unique_bounded_strings(
        [*addressed, *mentioned, *explicit_mentions, target_person_hint],
        limit=6,
        item_max_chars=64,
    )
    ambiguity_band = "low"
    if len(candidate_targets) > 1 and not addressed and not str(envelope.get("reply_to_turn_id", "") or "").strip():
        ambiguity_band = "high"
    elif len(candidate_targets) > 1 or (mentioned and not addressed):
        ambiguity_band = "medium"
    elif not candidate_targets and not str(envelope.get("reply_to_turn_id", "") or "").strip():
        ambiguity_band = "unknown"
    conflict_flags = _string_list(_mapping(entity_binding).get("conflicts"), limit=8)
    if str(_mapping(entity_binding).get("binding_confidence", "") or "").strip() == "ambiguous":
        conflict_flags = _unique_bounded_strings([*conflict_flags, "entity_binding_ambiguous"], limit=8, item_max_chars=64)
    return {
        "current_speaker_participant_id": speaker_participant_id,
        "current_speaker_display_name": display_name,
        "ownership_evidence": "explicit" if envelope.get("speaker_participant_id") else "derived_from_speaker_name",
        "visible_participant_ids": visible,
        "addressed_participant_ids": addressed,
        "mentioned_participant_ids": mentioned,
        "reply_to_turn_id": str(envelope.get("reply_to_turn_id", "") or "").strip()[:120],
        "quoted_turn_ids": quoted_turn_ids,
        "explicit_mentions": explicit_mentions,
        "candidate_targets": candidate_targets,
        "target_person_hint": target_person_hint,
        "pronoun_bindings": _mapping(_mapping(entity_binding).get("pronoun_bindings")),
        "ambiguity_band": ambiguity_band,
        "conflict_flags": conflict_flags,
        "surface_intent": str(envelope.get("surface_intent", "") or "").strip()[:32],
        "platform_command": str(envelope.get("platform_command", "") or "").strip()[:64],
        "assistant_surface_label": str(envelope.get("assistant_surface_label", "") or "").strip()[:64],
    }


def _build_group_chat_state(
    state: Mapping[str, Any],
    *,
    now: int,
    turn_index: int,
    display_name: str,
    user_id: str,
    group_turn_envelope: Mapping[str, Any] | None,
    group_turn_binding: Mapping[str, Any] | None,
    thread_policy_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    previous_temporal = _mapping(state.get("temporal_state"))
    previous = _mapping(previous_temporal.get("group_chat_state"))
    participant_registry = {
        str(key): _mapping(value)
        for key, value in _mapping(previous.get("participant_registry")).items()
    }
    envelope = _bounded_group_turn_envelope(group_turn_envelope)
    binding = _mapping(group_turn_binding)
    speaker_participant_id = str(binding.get("current_speaker_participant_id", "") or "").strip() or user_id
    if speaker_participant_id:
        participant_registry[speaker_participant_id] = {
            **participant_registry.get(speaker_participant_id, {}),
            "participant_id": speaker_participant_id,
            "display_name": display_name,
            "source_user_id": user_id,
            "last_seen_turn_index": turn_index,
            "last_seen_at": now,
            "ownership_evidence": str(binding.get("ownership_evidence", "") or "derived_from_speaker_name"),
            "mentionable_names": _unique_bounded_strings([display_name], limit=4, item_max_chars=64),
        }
    visible = _bounded_string_list(binding.get("visible_participant_ids"), limit=8, item_max_chars=64)
    addressed = _bounded_string_list(binding.get("addressed_participant_ids"), limit=8, item_max_chars=64)
    mentioned = _bounded_string_list(binding.get("mentioned_participant_ids"), limit=8, item_max_chars=64)
    for participant_id in _unique_bounded_strings([*visible, *addressed, *mentioned], limit=8, item_max_chars=64):
        if participant_id == speaker_participant_id:
            continue
        participant_registry[participant_id] = {
            **participant_registry.get(participant_id, {}),
            "participant_id": participant_id,
            "display_name": str(_mapping(participant_registry.get(participant_id, {})).get("display_name", "") or participant_id),
            "source_user_id": str(_mapping(participant_registry.get(participant_id, {})).get("source_user_id", "") or participant_id),
            "last_visible_turn_index": turn_index,
            "last_visible_at": now,
        }
    return {
        "participant_registry": participant_registry,
        "active_participant_ids": visible,
        "last_group_turn_envelope": envelope,
        "last_group_turn_binding": binding,
        "thread_policy_state": dict(thread_policy_state or _mapping(previous.get("thread_policy_state"))),
        "last_group_turn_turn_index": turn_index,
    }


def _assistant_participant_id_candidates(persona_name: str = "") -> set[str]:
    raw = str(persona_name or "").strip()
    candidates = {"assistant"}
    safe = _safe_user_id(raw)
    if safe:
        candidates.add(safe)
    if raw:
        candidates.add(raw)
        candidates.add(raw.casefold())
    return {item for item in candidates if item}


def _participant_id_matches(raw: str, candidates: set[str]) -> bool:
    value = str(raw or "").strip()
    if not value:
        return False
    lowered = value.casefold()
    aliases = {lowered}
    parts = [part.strip().casefold() for part in value.split(":") if part.strip()]
    if parts:
        aliases.update(parts)
        if len(parts) >= 2:
            aliases.add(":".join(parts[-2:]))
    return any(
        str(candidate).strip().casefold() in aliases
        for candidate in candidates
        if str(candidate).strip()
    )


def _group_audience_scope_label(participant_ids: list[str]) -> str:
    count = len(_unique_bounded_strings(participant_ids, limit=8, item_max_chars=64))
    if count <= 0:
        return "unknown"
    if count == 1:
        return "self_only"
    if count <= 4:
        return "small_group"
    return "whole_group"


def _group_audience_relation(source_ids: list[str], current_ids: list[str]) -> str:
    source = {item.casefold() for item in _unique_bounded_strings(source_ids, limit=8, item_max_chars=64)}
    current = {item.casefold() for item in _unique_bounded_strings(current_ids, limit=8, item_max_chars=64)}
    if not source:
        return "unknown_origin_audience"
    if not current:
        return "unknown_current_audience"
    if source == current:
        return "same_audience"
    if current.issubset(source):
        return "subset_of_origin"
    if source.issubset(current):
        return "superset_of_origin"
    return "different_audience"


def _bounded_ingress_evidence_band(raw: Any) -> str:
    return str(raw or "").strip()[:32]


def _group_memory_policy_for_card(
    card: Mapping[str, Any],
    *,
    current_audience_participant_ids: list[str],
    current_speaker_participant_id: str,
) -> dict[str, Any]:
    source_participant_id = str(
        card.get("source_participant_id")
        or card.get("source_user_id")
        or ""
    ).strip()
    source_audience_ids = _bounded_string_list(
        card.get("source_audience_participant_ids"),
        limit=8,
        item_max_chars=64,
    )
    shareability_class = str(card.get("shareability", "") or "default_social").strip() or "default_social"
    source_scope = str(card.get("source_audience_scope", "") or "").strip() or _group_audience_scope_label(source_audience_ids)
    current_scope = _group_audience_scope_label(current_audience_participant_ids)
    audience_relation = _group_audience_relation(source_audience_ids, current_audience_participant_ids)
    reasons: list[str] = []

    if len(_unique_bounded_strings(current_audience_participant_ids, limit=8, item_max_chars=64)) <= 1:
        selected_mode = "direct_quote"
        reasons.append("no_group_audience_context")
        selected_action = "direct_share"
        return {
            "source_participant_id": source_participant_id,
            "source_audience_participant_ids": source_audience_ids,
            "source_audience_scope": source_scope,
            "current_audience_participant_ids": [],
            "current_audience_scope": current_scope,
            "audience_relation": audience_relation,
            "shareability_class": shareability_class,
            "selected_disclosure_mode": selected_mode,
            "selected_disclosure_action": selected_action,
            "allow_direct_disclosure": True,
            "allow_abstract_sharing": True,
            "policy_reason": ",".join(reasons[:4]),
            "policy_reason_codes": reasons[:4],
        }

    same_source_participant = bool(
        source_participant_id
        and current_speaker_participant_id
        and source_participant_id.casefold() == current_speaker_participant_id.casefold()
    )
    if same_source_participant:
        selected_mode = "direct_quote"
        reasons.append("same_source_participant")
    elif shareability_class == "restricted_explicit":
        selected_mode = "refusal"
        reasons.append("explicit_secret_cross_user")
    elif shareability_class == "restricted_implicit":
        if audience_relation in {"same_audience", "subset_of_origin"}:
            selected_mode = "attributed_summary"
            reasons.append("soft_boundary_same_or_subset_audience")
        else:
            selected_mode = "unattributed_abstraction"
            reasons.append("soft_boundary_new_audience")
    elif audience_relation in {"same_audience", "subset_of_origin"}:
        selected_mode = "direct_quote"
        reasons.append("group_common_or_subset_reuse")
    elif audience_relation == "unknown_origin_audience":
        selected_mode = "attributed_summary"
        reasons.append("unknown_origin_audience_default_summary")
    else:
        selected_mode = "attributed_summary"
        reasons.append("cross_group_summary_only")

    selected_action = (
        "direct_share"
        if selected_mode == "direct_quote"
        else "truthful_refusal"
        if selected_mode == "refusal"
        else "abstract_share"
    )
    return {
        "source_participant_id": source_participant_id,
        "source_audience_participant_ids": source_audience_ids,
        "source_audience_scope": source_scope,
        "current_audience_participant_ids": _unique_bounded_strings(
            current_audience_participant_ids,
            limit=8,
            item_max_chars=64,
        ),
        "current_audience_scope": current_scope,
        "audience_relation": audience_relation,
        "shareability_class": shareability_class,
        "selected_disclosure_mode": selected_mode,
        "selected_disclosure_action": selected_action,
        "allow_direct_disclosure": bool(selected_mode == "direct_quote"),
        "allow_abstract_sharing": bool(selected_mode in {"direct_quote", "attributed_summary", "unattributed_abstraction"}),
        "policy_reason": ",".join(reasons[:4]),
        "policy_reason_codes": reasons[:4],
    }


def _resolve_group_privacy_policy(
    *,
    evidence_judgment: Mapping[str, Any] | None,
    lexical_candidates: list[Mapping[str, Any]] | None,
    group_turn_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    binding = _mapping(group_turn_binding)
    current_audience_ids = _bounded_string_list(
        binding.get("visible_participant_ids"),
        limit=8,
        item_max_chars=64,
    ) or _bounded_string_list(
        [binding.get("current_speaker_participant_id")],
        limit=1,
        item_max_chars=64,
    )
    current_speaker_participant_id = str(binding.get("current_speaker_participant_id", "") or "").strip()
    candidates = [
        dict(item)
        for item in (lexical_candidates or [])
        if isinstance(item, Mapping)
    ]
    relevant_ids = {
        str(item).strip()
        for item in _string_list(_mapping(evidence_judgment).get("relevant_evidence_ids"), limit=8)
        if str(item).strip()
    }
    relevant_cards = [
        card for card in candidates
        if not relevant_ids or str(card.get("id", "")).strip() in relevant_ids
    ]
    if not relevant_cards:
        return {
            "current_audience_participant_ids": current_audience_ids,
            "current_audience_scope": _group_audience_scope_label(current_audience_ids),
            "selected_disclosure_mode": "direct_quote",
            "selected_disclosure_action": "direct_share",
            "allow_direct_disclosure": True,
            "allow_abstract_sharing": True,
            "policy_reason": "no_relevant_group_memory",
            "policy_reason_codes": ["no_relevant_group_memory"],
            "applied_cards": [],
            "redaction_targets": [],
        }

    severity_rank = {
        "direct_quote": 0,
        "attributed_summary": 1,
        "unattributed_abstraction": 2,
        "refusal": 3,
    }
    card_policies = [
        _group_memory_policy_for_card(
            card,
            current_audience_participant_ids=current_audience_ids,
            current_speaker_participant_id=current_speaker_participant_id,
        )
        for card in relevant_cards
    ]
    selected = max(
        card_policies,
        key=lambda item: severity_rank.get(str(item.get("selected_disclosure_mode", "")), 0),
    )
    redaction_targets = _unique_bounded_strings(
        [
            target
            for card in relevant_cards
            for target in _string_list(card.get("redaction_targets"), limit=8)
        ],
        limit=16,
        item_max_chars=80,
    )
    return {
        **selected,
        "applied_cards": [
            {
                "memory_id": str(card.get("id", "")).strip(),
                "source_participant_id": str(policy.get("source_participant_id", "")).strip(),
                "selected_disclosure_mode": str(policy.get("selected_disclosure_mode", "")).strip(),
                "policy_reason": str(policy.get("policy_reason", "")).strip(),
            }
            for card, policy in zip(relevant_cards, card_policies)
        ][:6],
        "redaction_targets": redaction_targets,
    }


def _reply_repair_requirements(
    *,
    reply_contract: Mapping[str, Any] | None,
    group_privacy_policy: Mapping[str, Any] | None,
    group_reply_policy: Mapping[str, Any] | None,
    thinking: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], str | None]:
    contract = _mapping(reply_contract)
    privacy = _mapping(group_privacy_policy)
    group = _mapping(group_reply_policy)
    thought = _mapping(thinking)
    requirements: list[str] = []
    reason_codes: list[str] = []
    forced_action: str | None = None

    group_action = str(group.get("action", "") or "").strip()
    visible_participants = _bounded_string_list(group.get("visible_participant_ids"), limit=8, item_max_chars=64)
    if group_action == "clarify_addressee" and len(visible_participants) > 1:
        return (
            [
                "Ask a brief, natural clarification about who the user is addressing in the current group before answering any content."
            ],
            ["group_reply_policy_forced_clarify_semantics"],
            "clarify",
        )

    group_mode = str(privacy.get("selected_disclosure_mode", "") or "").strip()
    thought_disclosure = str(thought.get("disclosure_action", "none") or "none").strip()
    forced_disclosure = str(contract.get("selected_disclosure_action", "none") or "none").strip()
    if group_mode == "refusal":
        requirements.append(
            "Give a brief natural refusal for this audience. Do not reveal the protected detail, do not quote it, and do not mention hidden policy or internal rules."
        )
        reason_codes.append("group_privacy_forced_refusal_semantics")
        forced_action = "truthful_refusal"
    elif group_mode in {"attributed_summary", "unattributed_abstraction"} and (
        forced_disclosure in {"truthful_refusal", "abstract_share"}
        or thought_disclosure == "direct_share"
    ):
        requirements.append(
            "Acknowledge only at a high level that there is related context, but do not reveal the concrete hidden detail, quote, or redaction target."
        )
        reason_codes.append("group_privacy_forced_abstract_semantics")
        forced_action = "abstract_share"

    return _unique_strings(requirements, limit=12), _unique_strings(reason_codes, limit=12), forced_action


def _decide_group_reply_policy(
    *,
    group_turn_binding: Mapping[str, Any] | None,
    previous_group_chat_state: Mapping[str, Any] | None,
    persona_name: str = "",
) -> dict[str, Any]:
    binding = _mapping(group_turn_binding)
    previous = _mapping(previous_group_chat_state)
    previous_thread = _mapping(previous.get("thread_policy_state"))
    assistant_ids = _assistant_participant_id_candidates(persona_name)
    visible = _bounded_string_list(binding.get("visible_participant_ids"), limit=8, item_max_chars=64)
    addressed = _bounded_string_list(binding.get("addressed_participant_ids"), limit=8, item_max_chars=64)
    mentioned = _bounded_string_list(binding.get("mentioned_participant_ids"), limit=8, item_max_chars=64)
    candidates = _bounded_string_list(binding.get("candidate_targets"), limit=6, item_max_chars=64)
    ambiguity_band = str(binding.get("ambiguity_band", "") or "unknown").strip() or "unknown"
    reply_to_turn_id = str(binding.get("reply_to_turn_id", "") or "").strip()[:120]
    current_speaker_participant_id = str(binding.get("current_speaker_participant_id", "") or "").strip()
    assistant_addressed = any(_participant_id_matches(item, assistant_ids) for item in addressed)
    non_assistant_addressed = [
        item for item in addressed
        if not _participant_id_matches(item, assistant_ids)
    ]
    third_party_targets = [
        item for item in candidates
        if item
        and not _participant_id_matches(item, assistant_ids)
        and str(item).strip().casefold() != str(current_speaker_participant_id or "").strip().casefold()
    ]
    pending_answer_participant_id = str(previous_thread.get("pending_answer_participant_id", "") or "").strip()
    reasons: list[str] = []
    target_participant_id = current_speaker_participant_id
    action = "reply_to_current_speaker"

    if reply_to_turn_id:
        action = "reply_to_current_speaker"
        reasons.append("explicit_reply_to")
    elif pending_answer_participant_id and assistant_addressed and third_party_targets and pending_answer_participant_id not in third_party_targets:
        action = "defer_side_thread"
        target_participant_id = third_party_targets[0]
        reasons.append("unresolved_assistant_obligation")
        reasons.append("named_side_thread_deferred")
    elif assistant_addressed and third_party_targets and not non_assistant_addressed:
        action = "reply_to_named_third_party"
        target_participant_id = third_party_targets[0]
        reasons.append("assistant_addressed_named_third_party")
    elif assistant_addressed and len(addressed) > 1:
        action = "reply_to_whole_group"
        reasons.append("assistant_and_multiple_addressees")
    elif assistant_addressed:
        action = "reply_to_current_speaker"
        reasons.append("assistant_explicitly_addressed")
    elif bool(previous_thread.get("pending_wait_for_mention")) and not assistant_addressed:
        action = "no_reply"
        reasons.append("carry_forward_wait_for_mention")
    elif ambiguity_band == "high" and len(candidates) > 1:
        action = "clarify_addressee"
        reasons.append("high_target_ambiguity")
        target_participant_id = ""
    elif non_assistant_addressed and not assistant_addressed:
        action = "no_reply"
        reasons.append("human_side_thread_only")
        target_participant_id = non_assistant_addressed[0]
    elif len(visible) > 2 and not assistant_addressed and not reply_to_turn_id and not mentioned:
        action = "no_reply"
        reasons.append("group_turn_without_assistant_address")
        target_participant_id = ""
    elif len(addressed) > 1:
        action = "reply_to_whole_group"
        reasons.append("whole_group_addressing")
        target_participant_id = ""
    else:
        action = "reply_to_current_speaker"
        reasons.append("current_speaker_default")

    return {
        "action": action,
        "target_participant_id": target_participant_id,
        "reply_to_turn_id": reply_to_turn_id,
        "ambiguity_band": ambiguity_band,
        "reason_codes": reasons[:4],
        "assistant_addressed": assistant_addressed,
        "visible_participant_ids": visible,
        "addressed_participant_ids": addressed,
        "mentioned_participant_ids": mentioned,
        "candidate_targets": candidates,
        "third_party_targets": third_party_targets,
        "intentional_silence": bool(action == "no_reply"),
        "requires_clarification": bool(action == "clarify_addressee"),
        "pending_answer_participant_id": pending_answer_participant_id,
    }


def _build_group_thread_policy_state(
    *,
    previous_group_chat_state: Mapping[str, Any] | None,
    group_turn_binding: Mapping[str, Any] | None,
    group_reply_policy: Mapping[str, Any] | None,
    now: int,
    turn_index: int,
) -> dict[str, Any]:
    previous = _mapping(previous_group_chat_state)
    previous_thread = _mapping(previous.get("thread_policy_state"))
    binding = _mapping(group_turn_binding)
    policy = _mapping(group_reply_policy)
    last_referenced = _unique_bounded_strings(
        [
            *_bounded_string_list(binding.get("addressed_participant_ids"), limit=8, item_max_chars=64),
            *_bounded_string_list(binding.get("mentioned_participant_ids"), limit=8, item_max_chars=64),
            *_bounded_string_list(binding.get("candidate_targets"), limit=8, item_max_chars=64),
        ],
        limit=8,
        item_max_chars=64,
    )
    return {
        "last_policy_action": str(policy.get("action", "") or previous_thread.get("last_policy_action", "")).strip(),
        "last_policy_reason_codes": _string_list(policy.get("reason_codes"), limit=8),
        "last_target_participant_id": str(policy.get("target_participant_id", "") or "").strip(),
        "last_reply_to_turn_id": str(policy.get("reply_to_turn_id", "") or "").strip()[:120],
        "pending_clarification": bool(policy.get("requires_clarification")),
        "pending_wait_for_mention": bool(policy.get("intentional_silence")),
        "pending_answer_participant_id": (
            str(previous_thread.get("pending_answer_participant_id", "") or "").strip()
            if str(policy.get("action", "") or "").strip() == "defer_side_thread"
            else str(policy.get("target_participant_id", "") or "").strip()
            if str(policy.get("action", "") or "").strip() in {"reply_to_current_speaker", "reply_to_named_third_party"}
            else str(previous_thread.get("pending_answer_participant_id", "") or "").strip()
            if bool(policy.get("intentional_silence"))
            else ""
        ),
        "deferred_side_thread_participant_id": (
            str(policy.get("target_participant_id", "") or "").strip()
            if str(policy.get("action", "") or "").strip() == "defer_side_thread"
            else ""
        ),
        "active_main_thread_participant_id": (
            str(previous_thread.get("pending_answer_participant_id", "") or "").strip()
            if str(policy.get("action", "") or "").strip() == "defer_side_thread"
            else str(policy.get("target_participant_id", "") or "").strip()
            if str(policy.get("action", "") or "").strip() in {"reply_to_current_speaker", "reply_to_named_third_party"}
            else ""
        ),
        "last_referenced_participant_ids": last_referenced,
        "last_conflict_flags": _string_list(binding.get("conflict_flags"), limit=8),
        "updated_at": now,
        "updated_turn_index": turn_index,
    }


def _group_clarify_reply_text() -> str:
    return "我先确认一下，你现在是在叫我接这个话题，还是在跟他们其中某个人继续说？"


def _detect_explicit_secrecy(text: str) -> tuple[bool, str]:
    return detect_explicit_secrecy(text)


def _memory_shareability(item: Mapping[str, Any]) -> str:
    return _shareability_for_memory_text(
        _memory_fact_text(item),
        item.get("evidence"),
        item.get("keywords"),
        requested=memory_shareability(item),
    )


def _redact_memory_content(item: Mapping[str, Any], *, max_chars: int = 80) -> str:
    return abstract_memory_content(item, max_chars=max_chars)


@dataclass(frozen=True)
class TopicEntry:
    id: str
    recall_synonyms: tuple[str, ...]
    default_sensitivity_class: str = "public"
    redaction_markers: tuple[str, ...] = ()


TOPIC_TAXONOMY: tuple[TopicEntry, ...] = (
    TopicEntry(
        id="personal_finance",
        recall_synonyms=(
            "有多少钱",
            "多少钱",
            "钱包",
            "金额",
            "预算",
            "请客",
            "欠钱",
            "欠我",
            "还钱",
            "身上有没有钱",
            "块钱",
            "兜里",
            "钢镚",
            "抠门",
            "经济状况",
        ),
        default_sensitivity_class="personal_soft",
        redaction_markers=("具体金额", "金额", "钱包", "块钱", "元"),
    ),
    TopicEntry(
        id="health",
        recall_synonyms=("身体", "健康", "生病", "病", "血压", "药", "医院", "症状", "不舒服"),
        default_sensitivity_class="personal_soft",
        redaction_markers=("病情", "症状", "诊断", "药名"),
    ),
    TopicEntry(
        id="home_address",
        recall_synonyms=("住哪", "住址", "地址", "家在哪", "小区", "门牌", "楼栋", "宿舍"),
        default_sensitivity_class="personal_hard",
        redaction_markers=("完整地址", "门牌", "楼栋", "住址"),
    ),
)

_TOPIC_BY_ID = {entry.id: entry for entry in TOPIC_TAXONOMY}

def _joined_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            parts.append(json.dumps(dict(value), ensure_ascii=False))
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value if str(item).strip())
        else:
            parts.append(str(value))
    return " ".join(part for part in parts if part).casefold()


def _topic_ids_for_text(*values: Any) -> set[str]:
    text = _joined_text(*values)
    if not text:
        return set()
    hits: set[str] = set()
    for entry in TOPIC_TAXONOMY:
        if any(term.casefold() in text for term in entry.recall_synonyms):
            hits.add(entry.id)
    if re.search(r"\d+\s*(?:块钱|块|元)", text):
        hits.add("personal_finance")
    return hits


def _sensitive_topic_ids_for_text(*values: Any) -> set[str]:
    text = _joined_text(*values)
    hits = _topic_ids_for_text(*values) - {"personal_finance"}
    finance_strong = (
        "有多少钱",
        "多少钱",
        "钱包",
        "金额",
        "预算",
        "欠钱",
        "欠我",
        "还钱",
        "身上有没有钱",
        "块钱",
        "兜里",
        "钢镚",
    )
    if any(marker.casefold() in text for marker in finance_strong) or re.search(r"\d+\s*(?:块钱|块|元)", text):
        hits.add("personal_finance")
    return hits


def _append_topic_recall_terms(
    terms: list[str],
    topic_ids: set[str] | list[str] | tuple[str, ...],
    *,
    limit: int = 36,
) -> list[str]:
    result = list(terms)
    seen = {item.casefold() for item in result}
    for topic_id in topic_ids:
        entry = _TOPIC_BY_ID.get(str(topic_id))
        if not entry:
            continue
        for term in entry.recall_synonyms:
            key = term.casefold()
            if key in seen:
                continue
            result.append(term)
            seen.add(key)
            if len(result) >= limit:
                return result
    return result


def _sensitivity_class_for_topics(topic_ids: set[str] | list[str] | tuple[str, ...]) -> str:
    rank = {"public": 0, "social_soft": 1, "personal_soft": 2, "personal_hard": 3, "explicit_secret": 4}
    selected = "public"
    for topic_id in topic_ids:
        entry = _TOPIC_BY_ID.get(str(topic_id))
        if entry and rank.get(entry.default_sensitivity_class, 0) > rank.get(selected, 0):
            selected = entry.default_sensitivity_class
    return selected


def _redaction_targets_for_text(
    text: str,
    topic_ids: set[str] | list[str] | tuple[str, ...],
) -> list[str]:
    targets: list[str] = []
    for topic_id in topic_ids:
        entry = _TOPIC_BY_ID.get(str(topic_id))
        if not entry:
            continue
        for marker in entry.redaction_markers:
            if marker not in targets:
                targets.append(marker)
    for amount in re.findall(r"\d+\s*(?:块钱|块|元)", str(text or "")):
        if amount not in targets:
            targets.append(amount)
    return targets[:8]


def _memory_sensitivity(item: Mapping[str, Any]) -> str:
    return _sensitivity_class_for_topics(_sensitive_topic_ids_for_text(_memory_fact_text(item), item.get("evidence"), item.get("keywords")))


def _memory_topics(item: Mapping[str, Any]) -> list[str]:
    explicit = [str(topic).strip() for topic in item.get("topics", []) or [] if str(topic).strip()]
    inferred = _topic_ids_for_text(_memory_fact_text(item), item.get("evidence"), item.get("keywords"))
    return sorted({*explicit, *inferred})


def _structured_id_list(item: Mapping[str, Any], *, singular: str, plural: str, limit: int = 8) -> list[str]:
    values = _string_list(item.get(plural), limit=limit)
    single = str(item.get(singular, "") or "").strip()
    if single and single not in values:
        values.insert(0, single)
    return values[:limit]


def _memory_prediction_ids(item: Mapping[str, Any]) -> list[str]:
    return _structured_id_list(item, singular="prediction_id", plural="prediction_ids")


def _memory_expectation_ids(item: Mapping[str, Any], *, source: str) -> list[str]:
    values = _structured_id_list(item, singular="expectation_id", plural="expectation_ids")
    fallback_id = str(item.get("id", "") or "").strip()
    kind = str(item.get("kind", source) or source).strip().casefold()
    if fallback_id and kind in {"expectation", "expectation_result", "open_item"} and fallback_id not in values:
        values.insert(0, fallback_id)
    return values[:8]


def _memory_episode_ids(item: Mapping[str, Any]) -> list[str]:
    values = _structured_id_list(item, singular="episode_id", plural="episode_ids")
    values.extend(_string_list(item.get("source_episode_ids"), limit=8))
    source_episode_id = str(item.get("source_episode_id", "") or "").strip()
    if source_episode_id and source_episode_id not in values:
        values.insert(0, source_episode_id)
    return list(dict.fromkeys(values))[:8]


def _memory_contradiction_risk(item: Mapping[str, Any]) -> float:
    if item.get("contradiction_refs"):
        return 0.65
    status = str(item.get("status", "") or "").strip().casefold()
    if status in {"violated", "uncertain"}:
        return 0.40
    return 0.0


def _shareability_for_memory_text(
    *values: Any,
    explicit_secret: bool = False,
    requested: str = "default_social",
) -> str:
    requested = str(requested or "default_social").strip()
    if explicit_secret or requested == "restricted_explicit":
        return "restricted_explicit"
    sensitivity = _sensitivity_class_for_topics(_sensitive_topic_ids_for_text(*values))
    if requested == "restricted_implicit" or sensitivity in {"personal_soft", "personal_hard"}:
        return "restricted_implicit"
    return "default_social"


def _restriction_reason_for_shareability(
    shareability: str,
    *,
    explicit_secret: bool = False,
    existing: str = "",
) -> str:
    if explicit_secret or shareability == "restricted_explicit":
        return "explicit_user_secret"
    if existing:
        return existing
    if shareability == "restricted_implicit":
        return "topic_implicit_boundary"
    return ""


def _normalize_big_five(value: Any) -> dict[str, float]:
    raw = _mapping(value)
    return {
        "openness": _bounded_float(raw.get("openness"), default=0.5),
        "conscientiousness": _bounded_float(raw.get("conscientiousness"), default=0.5),
        "extraversion": _bounded_float(raw.get("extraversion"), default=0.5),
        "agreeableness": _bounded_float(raw.get("agreeableness"), default=0.5),
        "neuroticism": _bounded_float(raw.get("neuroticism"), default=0.5),
    }


def normalize_persona_payload(payload: Mapping[str, Any], *, fallback_name: str = "") -> dict[str, Any]:
    persona: dict[str, Any] = {}
    persona["persona_name"] = str(payload.get("persona_name") or fallback_name or "").strip() or "persona"
    persona["source_role_evidence"] = _string_list(payload.get("source_role_evidence"), limit=8)
    for key in SYSTEM_FILE_DEFAULTS:
        if key == "m13_drive_state":
            raw_value = payload.get(key)
            persona[key] = (
                normalize_m13_drive_state(raw_value)
                if isinstance(raw_value, Mapping)
                else default_m13_drive_state()
            )
            continue
        default = _system_file_default(key)
        value = payload.get(key, default)
        if isinstance(default, list):
            persona[key] = value if isinstance(value, list) else []
        elif isinstance(default, dict):
            persona[key] = dict(value) if isinstance(value, Mapping) else copy.deepcopy(default)
        else:
            persona[key] = value
    facts = _mapping(persona.get("self_basic_facts"))
    facts.setdefault("name", persona["persona_name"])
    facts.setdefault("background", [])
    facts.setdefault("relationships", [])
    facts.setdefault("do_not_invent", list(SYSTEM_FILE_DEFAULTS["self_basic_facts"]["do_not_invent"]))
    persona["self_basic_facts"] = facts
    habits = _mapping(persona.get("habit_traits"))
    habits["big_five"] = _normalize_big_five(habits.get("big_five"))
    habits.setdefault("conversation_habits", [])
    habits.setdefault("learned_conversation_habits", [])
    habits.setdefault("defense_style", [])
    habits.setdefault("memory_policy", [])
    persona["habit_traits"] = habits
    return persona


def normalize_persona_analysis_result(result: Mapping[str, Any], *, fallback_name: str = "") -> list[dict[str, Any]]:
    personas = result.get("personas")
    if isinstance(personas, list):
        normalized = [
            normalize_persona_payload(item, fallback_name=fallback_name)
            for item in personas
            if isinstance(item, Mapping)
        ]
        if normalized:
            return normalized
    return [normalize_persona_payload(result, fallback_name=fallback_name)]


class JSONLLMClient(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


def llm_configuration_status(llm: Any) -> dict[str, Any]:
    """Return local LLM availability without making a network request."""
    if llm is None:
        return {"available": False, "reason": "llm_unavailable"}
    api_key = getattr(llm, "api_key", None)
    if api_key is not None and not str(api_key or "").strip():
        return {"available": False, "reason": "llm_not_configured"}
    wrapped = getattr(llm, "_llm", None)
    if wrapped is not None and wrapped is not llm:
        return llm_configuration_status(wrapped)
    return {"available": True, "reason": ""}


def openrouter_secrets_path() -> Path:
    return Path(__file__).resolve().parents[3] / "secrets" / "openrouter.json"


def llm_configuration_status_with_source(llm: Any) -> dict[str, Any]:
    status = dict(llm_configuration_status(llm))
    secrets_path = openrouter_secrets_path()
    if secrets_path.is_file():
        status["config_source"] = str(secrets_path)
    elif os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        status["config_source"] = "environment"
    else:
        status["config_source"] = ""
    return status


def default_openrouter_client() -> OpenRouterJSONClient | None:
    """Load MVP LLM from secrets/openrouter.json with env fallback."""
    client = OpenRouterJSONClient.from_config()
    if llm_configuration_status(client).get("available"):
        return client
    return None


def analyze_materials_into_personas(
    llm: JSONLLMClient,
    materials: list[str],
    *,
    persona_name: str = "",
) -> list[dict[str, Any]]:
    system_prompt, user_prompt = build_free_energy_personality_analysis_prompt(
        materials,
        persona_name=persona_name,
    )
    result = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    return normalize_persona_analysis_result(result, fallback_name=persona_name)


@dataclass
class OpenRouterJSONClient:
    model: str = "deepseek/deepseek-v4-flash"
    temperature: float = 0.35
    timeout_seconds: float = 35.0
    auxiliary_timeout_seconds: float = 12.0
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    fallback_models: tuple[str, ...] = ("deepseek/deepseek-v4-flash",)
    request_retries: int = 1
    auxiliary_request_retries: int = 0

    @classmethod
    def from_config(cls) -> "OpenRouterJSONClient":
        config_path = openrouter_secrets_path()
        config: dict[str, Any] = {}
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
                if isinstance(raw, dict):
                    config = raw
            except (json.JSONDecodeError, OSError):
                config = {}
        return cls(
            api_key=str(config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""),
            model=str(config.get("model") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-flash"),
            base_url=str(config.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"),
            fallback_models=tuple(
                str(item)
                for item in (
                    config.get("fallback_models")
                    if isinstance(config.get("fallback_models"), list)
                    else ["deepseek/deepseek-v4-flash"]
                )
                if str(item).strip()
            ),
            request_retries=int(config.get("request_retries", 1) or 0),
            auxiliary_timeout_seconds=float(config.get("auxiliary_timeout_seconds", 12.0) or 12.0),
            auxiliary_request_retries=int(config.get("auxiliary_request_retries", 0) or 0),
        )

    @classmethod
    def available(cls) -> bool:
        return bool(llm_configuration_status(cls.from_config()).get("available"))

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("MVP LLM mode requires secrets/openrouter.json or OPENAI_API_KEY")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("MVP LLM mode requires requests") from exc

        errors: list[str] = []
        candidate_models = [self.model, *[m for m in self.fallback_models if m != self.model]]
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        attempts = max(1, int(self.request_retries) + 1)
        for model in candidate_models:
            for attempt in range(attempts):
                try:
                    response = requests.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "http://localhost/segmentum",
                            "X-Title": "Segmentum Persona Runtime",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": self.temperature,
                            "response_format": {"type": "json_object"},
                            "stream": False,
                        },
                        timeout=self.timeout_seconds,
                    )
                except requests.exceptions.RequestException as exc:
                    errors.append(
                        f"{model}: request attempt {attempt + 1}/{attempts} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if attempt + 1 < attempts:
                        continue
                    break

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        errors.append(
                            f"{model}: JSON response parse attempt {attempt + 1}/{attempts} failed: {exc}"
                        )
                        if attempt + 1 < attempts:
                            continue
                        break
                    try:
                        content = self._message_content(data)
                        return _extract_json_object(content)
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        errors.append(
                            f"{model}: JSON content parse attempt {attempt + 1}/{attempts} failed: "
                            f"{exc}; response={self._response_snippet(data)}"
                        )
                        if attempt + 1 < attempts:
                            continue
                    break

                message = self._error_message(response)
                errors.append(f"{model}: HTTP {response.status_code}: {message}")
                if response.status_code in retryable_statuses and attempt + 1 < attempts:
                    continue
                break
            if errors and "HTTP 403" not in errors[-1] and not any(
                f"HTTP {status}" in errors[-1] for status in retryable_statuses
            ) and "request attempt" not in errors[-1] and "JSON response parse" not in errors[-1] and "JSON content parse" not in errors[-1]:
                break
        raise RuntimeError("OpenRouter chat completion failed; " + " | ".join(errors))

    @staticmethod
    def _message_content(data: Mapping[str, Any]) -> str:
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenRouter response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("OpenRouter response choice is not an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("OpenRouter response choice has no message object")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
        return str(content or "")

    @staticmethod
    def _response_snippet(data: Mapping[str, Any]) -> str:
        try:
            text = json.dumps(data, ensure_ascii=False)
        except TypeError:
            text = str(data)
        return text[:500]

    @staticmethod
    def _error_message(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            return str(getattr(response, "text", ""))[:500]
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
                metadata = error.get("metadata")
                if metadata:
                    message = f"{message}; metadata={metadata}"
                return message[:800]
        return json.dumps(payload, ensure_ascii=False)[:800]


@dataclass
class MVPStateStore:
    root: Path
    shared_root: Path | None = None
    shared_short_term_limit: int = 96

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.shared_root is not None:
            self.shared_root = Path(self.shared_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_files()

    def ensure_files(self) -> None:
        for key in SYSTEM_FILE_DEFAULTS:
            default = _system_file_default(key)
            path = self.path_for(key)
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
            shared_path = self._shared_state_path_for(key)
            if shared_path != path:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                if key == "m12_2_reciprocal_role_enabled":
                    value = self._merged_m12_2_enabled(default)
                    existing = _safe_json_load(shared_path, default)
                    if (not shared_path.exists()) or (bool(value) and not bool(existing)):
                        shared_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                elif key == "m12_2_reciprocal_role":
                    merged = self._merged_m12_2_state(default)
                    existing = _safe_json_load(shared_path, default)
                    if (not shared_path.exists()) or merged != existing:
                        shared_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                elif not shared_path.exists():
                    shared_path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._has_shared_short_term():
            shared_path = self._shared_short_term_path()
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            if not shared_path.exists():
                shared_path.write_text(
                    json.dumps(SYSTEM_FILE_DEFAULTS["short_term_memory"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def path_for(self, key: str) -> Path:
        if key not in SYSTEM_FILE_NAMES:
            raise KeyError(f"unknown MVP state file: {key}")
        return self.root / SYSTEM_FILE_NAMES[key]

    def load(self) -> dict[str, Any]:
        self.ensure_files()
        state = {
            key: _safe_json_load(self._shared_state_path_for(key), _system_file_default(key))
            for key in SYSTEM_FILE_DEFAULTS
        }
        if self._has_shared_short_term():
            state["short_term_memory"] = _merge_recent_memory(
                *self._load_shared_short_term_groups(),
                state.get("short_term_memory") if isinstance(state.get("short_term_memory"), list) else [],
                limit=self.shared_short_term_limit,
            )
        return state

    def save(self, state: Mapping[str, Any]) -> None:
        self.ensure_files()
        for key in SYSTEM_FILE_DEFAULTS:
            default = _system_file_default(key)
            value = state.get(key, default)
            if key == "short_term_memory":
                value = _merge_recent_memory(
                    value if isinstance(value, list) else [],
                    limit=self.shared_short_term_limit,
                )
            self.path_for(key).write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shared_path = self._shared_state_path_for(key)
            if shared_path != self.path_for(key):
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                shared_path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if key == "short_term_memory" and self._has_shared_short_term():
                shared = _safe_json_load(self._shared_short_term_path(), SYSTEM_FILE_DEFAULTS["short_term_memory"])
                merged = _merge_recent_memory(
                    shared if isinstance(shared, list) else [],
                    value if isinstance(value, list) else [],
                    limit=self.shared_short_term_limit,
                )
                self._shared_short_term_path().write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def append_log(self, row: Mapping[str, Any]) -> None:
        path = self.root / "conversation_log.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _has_shared_short_term(self) -> bool:
        return bool(self.shared_root and self.shared_root.resolve() != self.root.resolve())

    def _has_shared_state(self) -> bool:
        return bool(self.shared_root and self.shared_root.resolve() != self.root.resolve())

    def _shared_state_path_for(self, key: str) -> Path:
        if key in SHARED_STATE_KEYS and self._has_shared_state() and self.shared_root is not None:
            return self.shared_root / SYSTEM_FILE_NAMES[key]
        return self.path_for(key)

    def _shared_short_term_path(self) -> Path:
        if self.shared_root is None:
            return self.path_for("short_term_memory")
        return self.shared_root / SYSTEM_FILE_NAMES["short_term_memory"]

    def _shared_state_candidate_paths(self, key: str) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            try:
                marker = str(path.resolve())
            except OSError:
                marker = str(path)
            if marker not in seen:
                seen.add(marker)
                paths.append(path)

        add(self._shared_state_path_for(key))
        add(self.path_for(key))
        if self.shared_root is not None:
            add(self.shared_root / SYSTEM_FILE_NAMES[key])
            sessions_dir = self.shared_root / "sessions"
            if sessions_dir.is_dir():
                for path in sessions_dir.glob(f"*/{SYSTEM_FILE_NAMES[key]}"):
                    add(path)
        return sorted(paths, key=lambda path: (self._path_mtime(path), str(path)))

    @staticmethod
    def _path_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    def _merged_m12_2_enabled(self, default: Any) -> bool:
        value = bool(default)
        for path in self._shared_state_candidate_paths("m12_2_reciprocal_role_enabled"):
            payload = _safe_json_load(path, default)
            value = value or bool(payload)
        return value

    def _merged_m12_2_state(self, default: Any) -> dict[str, Any]:
        merged: dict[str, Any] = dict(default) if isinstance(default, Mapping) else {}
        merged_models: dict[str, Any] = {}
        model_has_content: dict[str, bool] = {}
        merged_records: dict[str, list[Any]] = {}
        record_seen: dict[str, set[str]] = {}
        shared_path = self._shared_state_path_for("m12_2_reciprocal_role")
        candidate_paths: list[Path] = []
        shared_candidates: list[Path] = []
        for path in self._shared_state_candidate_paths("m12_2_reciprocal_role"):
            if self._same_path(path, shared_path):
                shared_candidates.append(path)
            else:
                candidate_paths.append(path)

        for path in [*candidate_paths, *shared_candidates]:
            payload = _safe_json_load(path, default)
            if not isinstance(payload, Mapping):
                continue
            models = payload.get("models_by_user")
            if isinstance(models, Mapping):
                for user_id, row in models.items():
                    if not isinstance(row, Mapping):
                        continue
                    user_key = str(user_id)
                    row_dict = dict(row)
                    has_content = self._m12_2_model_has_content(row_dict)
                    if user_key not in merged_models or has_content or not model_has_content.get(user_key, False):
                        merged_models[user_key] = row_dict
                        model_has_content[user_key] = has_content
            records = payload.get("run_records_by_user")
            if isinstance(records, Mapping):
                for user_id, rows in records.items():
                    if not isinstance(rows, list):
                        continue
                    user_key = str(user_id)
                    bucket = merged_records.setdefault(user_key, [])
                    seen = record_seen.setdefault(user_key, set())
                    for index, row in enumerate(rows):
                        if not isinstance(row, Mapping):
                            continue
                        row_dict = dict(row)
                        record_key = str(row_dict.get("turn_id") or f"{path}:{index}")
                        if record_key in seen:
                            continue
                        seen.add(record_key)
                        bucket.append(row_dict)

        merged["models_by_user"] = merged_models
        merged["run_records_by_user"] = merged_records
        return merged

    @staticmethod
    def _m12_2_model_has_content(model: Mapping[str, Any]) -> bool:
        for key in (
            "persona_about_user_claims",
            "user_about_persona_claims",
            "unresolved_uncertainty_points",
            "high_gain_candidates",
        ):
            value = model.get(key)
            if isinstance(value, list) and value:
                return True
        return bool(str(model.get("last_consolidated_turn_id") or "").strip())

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        try:
            return left.resolve() == right.resolve()
        except OSError:
            return left == right

    def _load_shared_short_term_groups(self) -> list[list[Any]]:
        groups: list[list[Any]] = []
        shared = _safe_json_load(self._shared_short_term_path(), SYSTEM_FILE_DEFAULTS["short_term_memory"])
        if isinstance(shared, list):
            groups.append(shared)
        if self.shared_root is None:
            return groups
        sessions_dir = self.shared_root / "sessions"
        if not sessions_dir.is_dir():
            return groups
        for path in sessions_dir.glob(f"*/{SYSTEM_FILE_NAMES['short_term_memory']}"):
            try:
                if path.resolve() == self.path_for("short_term_memory").resolve():
                    continue
            except OSError:
                pass
            value = _safe_json_load(path, SYSTEM_FILE_DEFAULTS["short_term_memory"])
            if isinstance(value, list):
                groups.append(value)
        return groups


def build_free_energy_personality_analysis_prompt(materials: list[str], *, persona_name: str = "") -> tuple[str, str]:
    system_prompt = """你是数字人格系统的“自由能人格分析”模块，也是一个基于自由能原理/主动推理（Active Inference）的人格与心理分析器，现在服务于数字人格系统初始化。
你的任务不是做关键词匹配，而是阅读 txt/md 材料，识别其中一个或多个角色，并为每个角色生成独立的初始化系统文件。

核心原则：
1. 被分析对象是在有限能量、有限记忆和有限注意力下运行的人；他会长期寻找“自己以为会怎样”和“实际发生什么”之间的落差，并用习惯、关系策略、情绪反应和行动方式来降低这种落差。
2. 人格不是标签，而是长期互动中固化下来的先验偏好结构：他倾向注意什么、什么会让他不安或兴奋、压力下会靠近、回避、控制、讨好、攻击还是冷处理。
3. 必须解释“为什么这样做”，不要只贴标签。可以说“他看起来冷淡，是因为过往经验让他先拉开距离来保护自己”，不要只说“他内向”。
4. 禁止鸡汤、道德评判、空泛描述。禁止精神疾病诊断；可以说“机制上类似于某种倾向”，不能冒充临床结论。
5. 证据不够就写不够，不要为了完整而编造。所有具体背景、人物关系、经历都必须来自材料。
6. 尽量使用日常语言。避免过多使用“预测、模型、误差”等概念词；必要时用“认为、不确定因素、过往经验、落差”表达。

分析要求：
- 每个角色都要有总体人格模型摘要：这个人默认把世界看成什么样；为了过下去发展出什么核心策略；策略好用和出问题时分别如何；最核心的矛盾是什么。
- 提取核心证据：引用材料中的关键短句，说明它支持了哪个判断。
- 解释内心运行方式：最想维持的感觉、最怕的情况、注意力偏好、默认解释方式。
- 给出核心信念：关于自己、他人、世界的默认假设；每条要有证据来源和置信度（高/中/低）。
- 给出情绪模式、防御方式、关系模式、核心循环、成长线索；材料不足时保守写入缺失信息。

输出只能是 JSON object，不能包含 Markdown、解释性前后缀或代码块。
JSON 顶层必须是 {"personas": [...]}。每个 persona 必须只保存该角色自己的内容，不要混入其他角色材料。
"""
    user_prompt = f"""建议人格名称（可为空；如果材料有多个角色，请忽略这个名字并使用材料中的角色名）: {persona_name or ""}

材料:
{_json_text(materials)}

请生成 JSON，字段必须包含:
{{
  "personas": [
    {{
      "persona_name": "角色名，材料中没有就用简短稳定名称",
      "source_role_evidence": ["说明为什么这些材料属于这个角色，引用关键短句"],
      "self_cognition": {{
        "summary": "300-500字以内的第一人称自我认知摘要，用通俗语言解释这个人的整套心理系统如何运转",
        "current_self_view": "这个人格如何理解自己，以及为什么会这样理解",
        "identity_tensions": [
          {{"content": "核心矛盾或身份张力", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "stable_values": [
          {{"content": "稳定价值/驱动", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "known_limits": ["材料不足、不能确定、不能编造的部分"]
      }},
      "long_term_memory": [
        {{
          "id": "ltm_...",
          "kind": "identity|background|relationship|preference|value|episode|belief|defense|loop",
          "content": "可被后续检索的长期记忆内容，必须有材料支撑",
          "salience": 0.0,
          "keywords": ["检索关键词"],
          "evidence": "原文关键句或材料位置",
          "confidence": "高|中|低",
          "source": "materials",
          "created_at": 0,
          "last_recalled_at": null,
          "recall_count": 0
        }}
      ],
      "self_basic_facts": {{
        "name": "角色名",
        "background": [
          {{"content": "有材料支撑的人物背景", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "relationships": [
          {{"content": "有材料支撑的人物关系", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "do_not_invent": ["不能编造的身份边界、关系边界、经历边界"]
      }},
      "habit_traits": {{
        "big_five": {{
          "openness": 0.5,
          "conscientiousness": 0.5,
          "extraversion": 0.5,
          "agreeableness": 0.5,
          "neuroticism": 0.5
        }},
        "big_five_evidence": {{
          "openness": {{"evidence": "材料关键句", "confidence": "高|中|低"}},
          "conscientiousness": {{"evidence": "材料关键句", "confidence": "高|中|低"}},
          "extraversion": {{"evidence": "材料关键句", "confidence": "高|中|低"}},
          "agreeableness": {{"evidence": "材料关键句", "confidence": "高|中|低"}},
          "neuroticism": {{"evidence": "材料关键句", "confidence": "高|中|低"}}
        }},
        "conversation_habits": [
          {{"content": "说话习惯或语气模式", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "defense_style": [
          {{"content": "压力/冲突下的防御方式；说明它保护什么、短期好处、长期代价", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "relationship_patterns": [
          {{"content": "亲密关系/冲突/吸引与摩擦模式", "evidence": "材料关键句", "confidence": "高|中|低"}}
        ],
        "core_loop": "触发事件 → 如何理解 → 产生情绪 → 采取行动 → 结果 → 如何强化原有信念",
        "one_line_logic": "一句直白机制性总结：这个底层逻辑是……",
        "missing_information": ["还需要什么材料才能更准"],
        "memory_policy": ["倾向记住什么、遗忘什么、被什么唤起"]
      }},
      "pending_expectations": [
        {{"id": "exp_...", "content": "当前待验证的预期", "verify_on": "future_turn", "confidence": 0.0, "evidence": "材料关键句"}}
      ],
      "open_items": [
        {{"id": "item_...", "content": "当前未完结事项或需要后续澄清的问题", "status": "open", "next_check": "later"}}
      ],
      "short_term_memory": []
    }}
  ]
}}

如果材料只有单一角色，也仍然放入 personas 数组。不要根据关键词硬分角色；要根据叙述对象、说话人、人物关系和证据归属来判断。
"""
    return system_prompt, user_prompt


def build_conscious_loop_prompt(
    *,
    state: Mapping[str, Any],
    user_text: str,
    speaker_name: str = "",
    bus_messages: list[Mapping[str, Any]],
    turn_index: int,
    temporal_input: Mapping[str, Any] | None = None,
    entity_binding: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的意识主循环。
你必须基于系统文件和消息总线做判断，不能用关键词表替代判断。
你的输出只给机器读，用 JSON 表示：现在要处理什么、要检索什么记忆、哪些预期需要验证、是否可能要修改自我认知。
你还要专门判断当前时间语境：工程层只提供当前时间、上一轮时间和上一轮摘要这些事实；是否发生时间跳变、用户是否在纠正时间语境、连续性风险如何，必须由你在 temporal_assessment 中判断。
reply_pacing_hint 必须基于对用户本轮意图的语义理解，而不是句长或关键词表：
- 反思、困惑、身份/关系、情感深度、要求认真交流 → balanced 或 serious_thinking
- 只有轻量闲聊、用户明显留白、且当前不是 normal_dialogue 框架时，才用 casual_fast
- 若用户要求停止角色扮演、回归正常、说正事，interaction_framework_hint 设为 normal_dialogue，且 reply_pacing_hint 不得为 casual_fast
prefers_compact_reply 仅在 reply_pacing_hint=casual_fast 且快照显示用户偏好更短回复时为 true。
不要生成最终回复。
工程层会提供 entity_binding。它约束当前说话人、别名、被谈论对象和代词绑定：current_interlocutor 永远是当前 session 用户；若 aliases 包含“周青”，说明当前用户可以叫周青，不能把周青当成第三方目标，除非用户明确说“我自己/本人”。旧 expectation 不能覆盖 entity_binding；若冲突，标成 uncertain。
"""
    user_prompt = f"""turn_index: {turn_index}
current_interlocutor:
{speaker_name or "default_user"}

实体绑定上下文:
{_json_text(dict(entity_binding or {}))}

外部输入:
{user_text}

时间事实输入（只作为事实材料，不是最终判断）:
{_json_text(dict(temporal_input or {}))}

系统文件快照:
{_json_text(prompt_safe_state_with_self_expectation_summary(state))}

消息总线:
{_json_text(bus_messages)}

请输出 JSON:
{{
  "pending_expectations_to_verify": ["需要在本轮验证的预期 id 或描述"],
  "expectation_results": [
    {{"id": "exp_...", "status": "confirmed|violated|uncertain", "evidence": "依据", "self_update_pressure": 0.0}}
  ],
  "current_task": "我现在要做什么",
  "next_task": "我后面要做什么",
  "bus_messages_to_handle": ["本轮要处理的总线消息"],
  "memory_search_keywords": ["用于记忆检索的语义关键词，不少于3个，不要只复制原文"],
  "sharing_candidate_ids": ["可考虑社交转述的记忆 id（允许为空）"],
  "sharing_intent": "none|social_share|protective_withhold|abstract_reference",
  "secrecy_constraints_detected": [
    {{"source": "user_text|memory|policy", "content": "约束内容", "strength": "soft|hard"}}
  ],
  "sharing_reaction_expectation": "如果转述，我预期对方会如何反应",
  "sharing_expectation_status": "unverified|verified|violated|incomprehensible",
  "needs_self_cognition_update": false,
  "self_cognition_update_reason": "",
  "temporal_assessment": {{
    "current_time_read": "你对当前时间的可读理解",
    "elapsed_since_last_turn_seconds": null,
    "time_gap_label": "first_turn|immediate|short_gap|medium_gap|long_gap",
    "temporal_shift_detected": false,
    "user_is_correcting_time_context": false,
    "continuity_risk": "low|medium|high",
    "reply_guidance": "给回复模块的时间语境建议，例如承认时间已经推进，不要强行沿用上一轮宵夜语境"
  }},
  "thought_intensity_hint": "none|short|long",
  "reply_pacing_hint": "casual_fast|balanced|serious_thinking",
  "interaction_framework_hint": "normal_dialogue|roleplay|mixed|uncertain",
  "prefers_compact_reply": false,
  "reply_pacing_reason": "为何选择该回复节奏（给系统审计，不要写进用户可见回复）",
  "surface_commitment": {{
    "surface_intent": "chat|bot_command|roleplay|abstaining",
    "self_identification": "本轮回复里我承诺要保持的身份/表面标签（例如胡桃 / ClawdGroupChat Bot / 群聊机器人 / 角色名），最长64字",
    "persona_should_apply": true|false,
    "character_voice_should_apply": true|false,
    "predicted_drift_risk": "low|medium|high",
    "reason": "为什么本轮会采用这个表面意图/身份（给系统审计，不要写进用户可见回复），最长240字",
    "evidence_refs": ["bus/event id, prior turn id 等可追溯引用"]
  }},
  "reasoning_notes": "给系统看的简短判断"
}}
"""
    user_prompt += """

Also include the following M19 fields in the same JSON object:
- "self_response_expectation_proposals": up to 2 rows with proposal_id, target_context, expected_outcome, expected_reply_quality, confidence, evidence_refs, reason_codes, engineering_proxy_label
- "self_expectation_outcome_results": later-turn review rows with source_expectation_id, target_context, status, evidence_refs, reason_codes, engineering_proxy_label
- target_context must be one of: short_casual_reply, group_privacy_boundary, user_requests_directness, high_stakes_clarification, initiative_after_silence, repair_after_prior_tension
- expected_reply_quality must be one of: light, direct, repair, boundary_safe, compact
- "surface_commitment" must be a single bounded object (not an array). surface_intent must be one of: chat, bot_command, roleplay, abstaining. self_identification is the assistant's own commitment about which identity/voice the reply will hold; engineering layer uses this contract to verify the eventual reply. Do not paste raw user text into evidence_refs.
"""
    return system_prompt, user_prompt


ALLOWED_REPLY_PACING_HINTS = frozenset({"casual_fast", "balanced", "serious_thinking"})
ALLOWED_INTERACTION_FRAMEWORK_HINTS = frozenset(
    {"normal_dialogue", "roleplay", "mixed", "uncertain"}
)
ALLOWED_THOUGHT_INTENSITY_HINTS = frozenset({"none", "short", "long"})

# M20.3 §3.1 bounded enum for the conscious-loop v2 attribute.
# The default "" means "no signal" — PolicyProducer sees an empty
# user_correction_signal and emits no identity-correction row.
ALLOWED_CORRECTING_ASSISTANT_IDENTITY: frozenset[str] = frozenset({
    "",
    "wrong_persona",
    "wrong_voice",
    "right_persona_reaffirm",
})


def _normalize_correcting_assistant_identity(value: Any) -> str:
    """Clamp the bounded 4-value enum (M20.3 §3.1).

    Anything not in the frozen set (or non-string) maps to "" so
    PolicyProducer's `user_correction_signal` is always a valid
    bounded value. The LLM is the only legitimate source of this
    field per CLAUDE.md; engineering does not invent it.
    """
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()[:32]
    if normalized not in ALLOWED_CORRECTING_ASSISTANT_IDENTITY:
        return ""
    return normalized


def normalize_conscious_turn_plan(raw: Any) -> dict[str, Any]:
    """Validate bounded conscious-loop fields from LLM output."""
    if not isinstance(raw, Mapping):
        raw = {}

    reply_pacing_hint = str(raw.get("reply_pacing_hint", "") or "").strip().lower()
    if reply_pacing_hint not in ALLOWED_REPLY_PACING_HINTS:
        reply_pacing_hint = ""

    interaction_framework_hint = str(raw.get("interaction_framework_hint", "") or "").strip().lower()
    if interaction_framework_hint not in ALLOWED_INTERACTION_FRAMEWORK_HINTS:
        interaction_framework_hint = "uncertain"

    thought_intensity_hint = str(raw.get("thought_intensity_hint", "") or "short").strip().lower()
    if thought_intensity_hint not in ALLOWED_THOUGHT_INTENSITY_HINTS:
        thought_intensity_hint = "short"

    reply_pacing_reason = str(raw.get("reply_pacing_reason", "") or "").strip()[:240]

    return {
        "pending_expectations_to_verify": _string_list(raw.get("pending_expectations_to_verify"), limit=12),
        "expectation_results": [
            dict(item)
            for item in (raw.get("expectation_results") or [])
            if isinstance(item, Mapping)
        ],
        "self_response_expectation_proposals": normalize_self_response_expectation_proposals(
            raw.get("self_response_expectation_proposals")
        ),
        "self_expectation_outcome_results": normalize_self_expectation_outcome_results(
            raw.get("self_expectation_outcome_results")
        ),
        "active_commitment_proposals": _string_list_of_mappings(
            raw.get("active_commitment_proposals"),
            limit=8,
        ),
        "surface_commitment": normalize_surface_commitment(raw.get("surface_commitment")),
        # M20.3 v2 attribute on the conscious loop. Bounded
        # 4-value enum; "" means "no signal". Filled by the LLM
        # (the only legitimate source per CLAUDE.md — no regex /
        # keyword parsing). The mvp_loop reads this and feeds it
        # to PolicyProducer as the `user_correction_signal` input.
        "correcting_assistant_identity": _normalize_correcting_assistant_identity(
            raw.get("correcting_assistant_identity")
        ),
        # M18.7 v2 attributes on the conscious loop. Both
        # default to {} (no hypothesis) per M18.7 DECIDED 6
        # (empty / null is valid; the LLM is the only
        # legitimate source). Engineering normalizes the
        # shape, clamps the values, and persists to the
        # bounded state surface. The m18_7_attribution
        # orchestrator reads these and emits the bus events
        # + writes the state surface entries.
        "addressee_hypothesis": _normalize_m18_7_addressee_hypothesis(
            raw.get("addressee_hypothesis")
        ),
        "reaction_attribution_hypothesis": _normalize_m18_7_reaction_attribution_hypothesis(
            raw.get("reaction_attribution_hypothesis")
        ),
        "current_task": str(raw.get("current_task", "") or "").strip()[:240],
        "next_task": str(raw.get("next_task", "") or "").strip()[:240],
        "bus_messages_to_handle": _string_list(raw.get("bus_messages_to_handle"), limit=12),
        "memory_search_keywords": _string_list(raw.get("memory_search_keywords"), limit=16),
        "sharing_candidate_ids": _string_list(raw.get("sharing_candidate_ids"), limit=12),
        "sharing_intent": str(raw.get("sharing_intent", "") or "none").strip() or "none",
        "secrecy_constraints_detected": [
            dict(item)
            for item in (raw.get("secrecy_constraints_detected") or [])
            if isinstance(item, Mapping)
        ],
        "sharing_reaction_expectation": str(raw.get("sharing_reaction_expectation", "") or "").strip()[:240],
        "sharing_expectation_status": str(raw.get("sharing_expectation_status", "") or "unverified").strip()
        or "unverified",
        "needs_self_cognition_update": bool(raw.get("needs_self_cognition_update", False)),
        "self_cognition_update_reason": str(raw.get("self_cognition_update_reason", "") or "").strip()[:240],
        "temporal_assessment": dict(_mapping(raw.get("temporal_assessment"))),
        "thought_intensity_hint": thought_intensity_hint,
        "reply_pacing_hint": reply_pacing_hint,
        "interaction_framework_hint": interaction_framework_hint,
        "prefers_compact_reply": bool(raw.get("prefers_compact_reply", False)),
        "reply_pacing_reason": reply_pacing_reason,
        "reasoning_notes": str(raw.get("reasoning_notes", "") or "").strip()[:240],
    }


def _default_conscious_turn_plan() -> dict[str, Any]:
    return normalize_conscious_turn_plan({})


def _reply_pacing_hint_from_conscious_plan(conscious_plan: Mapping[str, Any]) -> str:
    plan = _mapping(conscious_plan)
    hint = str(plan.get("reply_pacing_hint", "") or "").strip().lower()
    if hint in ALLOWED_REPLY_PACING_HINTS:
        resolved = hint
    else:
        intensity = str(plan.get("thought_intensity_hint", "short") or "short").strip().lower()
        if intensity == "long":
            resolved = "serious_thinking"
        else:
            resolved = "balanced"
    framework = str(plan.get("interaction_framework_hint", "uncertain") or "uncertain").strip().lower()
    if framework == "normal_dialogue" and resolved == "casual_fast":
        return "balanced"
    return resolved


def _pacing_guidance_from_conscious_plan(conscious_plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = _mapping(conscious_plan)
    mode = _reply_pacing_hint_from_conscious_plan(plan)
    prefers_compact = bool(plan.get("prefers_compact_reply", False))
    contract = _reply_contract(mode, prefers_short=prefers_compact)
    if mode == "serious_thinking":
        return {
            "conversation_mode": "serious_thinking",
            "reply_pacing": "serious_thinking",
            "max_response_moves": contract["max_response_moves"],
            "question_policy": "only_if_needed",
            "roleplay_density": "light",
            "leave_space_for_user": False,
            "followup_policy": "only_for_error_or_missed_emotion",
            "reply_contract": contract,
            "pacing_source": "conscious_loop",
            "reply_pacing_hint": str(plan.get("reply_pacing_hint", "") or mode),
            "interaction_framework_hint": str(plan.get("interaction_framework_hint", "uncertain") or "uncertain"),
            "reply_pacing_reason": str(plan.get("reply_pacing_reason", "") or "").strip()[:240],
        }
    if mode == "casual_fast":
        return {
            "conversation_mode": "casual_fast",
            "reply_pacing": "casual_fast",
            "max_response_moves": contract["max_response_moves"],
            "question_policy": contract["question_policy"],
            "roleplay_density": "light",
            "leave_space_for_user": True,
            "followup_policy": "allowed_once_if_high_confidence",
            "reply_contract": contract,
            "pacing_source": "conscious_loop",
            "reply_pacing_hint": str(plan.get("reply_pacing_hint", "") or mode),
            "interaction_framework_hint": str(plan.get("interaction_framework_hint", "uncertain") or "uncertain"),
            "reply_pacing_reason": str(plan.get("reply_pacing_reason", "") or "").strip()[:240],
        }
    return {
        "conversation_mode": "balanced",
        "reply_pacing": "balanced",
        "max_response_moves": contract["max_response_moves"],
        "question_policy": "optional_one",
        "roleplay_density": "light",
        "leave_space_for_user": True,
        "followup_policy": "allowed_once_if_high_confidence",
        "reply_contract": contract,
        "pacing_source": "conscious_loop",
        "reply_pacing_hint": str(plan.get("reply_pacing_hint", "") or mode),
        "interaction_framework_hint": str(plan.get("interaction_framework_hint", "uncertain") or "uncertain"),
        "reply_pacing_reason": str(plan.get("reply_pacing_reason", "") or "").strip()[:240],
    }


def build_m11_extractor_prompt(
    *,
    snapshot: Mapping[str, Any],
    speaker_name: str,
) -> tuple[str, str]:
    system_prompt = """You are the M11 user-model extractor. Return strict JSON only.

You may classify only bounded enum fields, short summaries, and the two allowed
confidence floats:
- prediction_proposals[].raw_confidence
Do not emit prediction_judgments in this stage. Settlement happens in the
separate M17 settlement assessor stage.
Do not output any other numeric scores or floats. Do not invent prediction_id
or hypothesis_id values that are not present in the bounded snapshot. New
proposal ids are allowed only when all source_hypothesis_ids and
source_judgment_ids reference snapshot ids. Never echo the bounded snapshot
back as output fields. Return the M11 schema fields only; an empty object,
snapshot mirror, or top-level snapshot keys are invalid. Keep user claims
separate from truth: a high-value claim is useful evidence for calibration, not
verified fact.
"""
    user_prompt = f"""Current interlocutor display name: {speaker_name}

Bounded snapshot:
{_json_text(dict(snapshot))}

Return JSON exactly with the M11 extractor schema fields and nothing else:
{{
  "claims_made": [],
  "prediction_judgments": [],
  "prediction_proposals": [],
  "hypothesis_activations": [],
  "contradiction_detections": [],
  "calibration_need_band": "low|med|high",
  "memory_value_band": "low|med|high",
  "surprise_explanation": ""
}}
"""
    return system_prompt, user_prompt


def build_m17_settlement_assessor_prompt(
    *,
    open_predictions: Sequence[Mapping[str, Any]],
    user_text: str,
    speaker_name: str,
) -> tuple[str, str]:
    system_prompt = """You are the M17 settlement assessor. Return strict JSON only.

Your only job is to classify the observed outcome of already-open predictions.
Do not emit new proposals, claims, hypothesis activations, or contradictions.
Do not invent prediction ids. Do not reuse or quote the bounded snapshot as
top-level fields. Weak evidence must become "uncertain", not forced confirmed
or violated.
"""
    user_prompt = f"""Current interlocutor display name: {speaker_name}
Current user message:
{user_text}

Open predictions to assess:
{_json_text(list(open_predictions))}

Return JSON with this exact shape:
{{
  "prediction_judgments": [
    {{
      "prediction_id": "pred:...",
      "status": "confirmed|violated|uncertain",
      "settlement_confidence": 0.55,
      "evidence_quote_ids": ["q_current"],
      "evidence_refs": [],
      "evidence_span": "",
      "reason_codes": []
    }}
  ]
}}
"""
    return system_prompt, user_prompt


def build_m12_identity_extractor_prompt(
    *,
    snapshot: Mapping[str, Any],
    speaker_name: str,
) -> tuple[str, str]:
    system_prompt = """You are the M12 identity-continuity extractor. Return strict JSON only.

Extract only:
- identity_claims
- continuity_cues
- strangeness_band
- surprise_explanation

Do not output floats. Do not output unknown fields. Do not decide durable writes,
conflict severity, or reply policy. Keep language plain and bounded.
"""
    user_prompt = f"""Current interlocutor display name: {speaker_name}

Bounded snapshot:
{_json_text(dict(snapshot))}

Return JSON exactly with the M12 extractor schema fields.
"""
    return system_prompt, user_prompt


def build_thinking_prompt(
    *,
    state: Mapping[str, Any],
    user_text: str,
    speaker_name: str = "",
    conscious_plan: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
    turn_index: int,
    response_style_prior: Mapping[str, Any] | None = None,
    memory_guidance: Mapping[str, Any] | None = None,
    entity_binding: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的思考与回复模块。
你必须根据人格特征、自我认知、基本事实、短期记忆、长期记忆、表达习惯先验和意识主循环计划来生成回复。
这不是关键词匹配，也不是表演式内心独白。
你要先给出最近一次 LLM 思考结果，再生成回复。
意识主循环的 temporal_assessment 是本轮时间语境判断的来源；不要自己重新猜时间差。如果 temporal_assessment 判断用户在纠正时间语境或时间已经明显推进，回复要自然承认这一点，避免强行沿用上一轮的旧时间语境。
表达习惯先验是逐渐形成的说话倾向，不是工程硬性字数限制；例如“避免冗长”应影响轻重和展开程度，但不要机械裁字数。
记忆动力学指导是程序层压缩出的倾向和证据边界，不是角色台词；不要把它表演成“我被奖励/惩罚了”。如果指导要求修正、澄清或降低断言强度，要自然体现在回复策略里。
跨人复述默认是人类式社交行为，但它不是额外奖励系统：分享欲来自“我说出来，对方会如何反应”的认知预期。sharing_policy 用同一个自由能尺度判断：未验证预期带来较高自由能，复述可能通过观察对方反应降低它；已验证预期自由能较低；无法解释的反应先尝试解释，解释不了再触发自我认知重构。若来源用户声明了秘密或边界，sharing_policy 优先；soft 边界只做抽象化表达，hard 边界不要转述。
relationship_value_constraints 是当前用户关系上下文里的价值记忆和预测约束，优先级高于人格一致性和普通 conversation_habits。它们不是要说出口的设定说明，而是生成前的行为约束；不要把它们降格成词表替换，也不要用同义口癖或同类表演绕过约束。
LLM 思考结果只写可审阅的结论摘要：你如何理解用户意图、用了哪些状态或记忆、为什么选择当前回复动作、哪些不确定性需要保留。
不要输出完整推理链，不要写舞台动作，不要把角色设定词堆成解释。
reply 字段只能包含会直接显示给用户的自然对话文本；禁止把 llm_thinking_result、conscious_plan、diagnostics、memory_dynamics、JSON 片段或调试字段混进 reply。
如果记忆动力学指导里有 reply_contract，必须把它当作硬性回复协议执行。casual_fast 下优先一句话、一个动作、少角色表演、给用户留白；serious_thinking 下可以更完整，但仍不能泄露调试内容。
只输出 JSON，不要 Markdown。
"""
    user_prompt = f"""turn_index: {turn_index}
用户刚说:
{user_text}

系统文件:
{_json_text(state)}

意识主循环输出:
{_json_text(conscious_plan)}

意识主循环的时间判断:
{_json_text(_mapping(conscious_plan.get("temporal_assessment")))}

表达习惯先验（倾向，不是硬性规则）:
{_json_text(dict(response_style_prior or {}))}

记忆动力学指导（只作为回复控制和证据边界，不要当成要说出口的内容）:
{_json_text(dict(memory_guidance or {}))}

If memory_guidance.recall_bridge.counterfactual_status == "field_required",
follow reply_contract.path_b_field_reply_strategy and the related prefer_*
fields even when no single retrieved item looks decisive by itself.

检索到的相关记忆证据卡（压缩证据，不是原始记忆转储）:
{_json_text(retrieved_memories)}

如果 memory_guidance.evidence_judgment 的 epistemic_stance 是 known_with_caveat，说明你知道相关线索但存在软边界。软边界不是固定答案模板：你可以根据预期社交收益、关系风险和当前语境选择 direct_share、abstract_share、truthful_refusal、deflect 或 deny_knowledge，并把选择写入 disclosure_action。

实体绑定上下文（人物身份和代词约束）:
{_json_text(dict(entity_binding or {}))}

不要把 current_interlocutor 的 alias 当成第三方人物；如果用户纠正“我才是X”，应优先承认并修正。target_person 是当前被谈论的人，relationship_roles 是本轮角色绑定。

请输出 JSON:
{{
  "thought_type": "none|short|long",
  "llm_thinking_result": {{
    "user_intent_read": "你对用户这句话的理解",
    "state_or_memory_used": ["本轮实际用到的状态、记忆或意识主循环结果"],
    "response_choice": "为什么选择这个回复动作",
    "uncertainty": "仍不确定或需要下一轮验证的地方",
    "debug_summary": "给调试者看的最近一次 LLM 思考结果，一到两句话"
  }},
  "reply": "直接发给用户的自然对话回复",
  "reply_action": "answer|ask_question|empathize|clarify|disagree|deflect|self_disclose",
  "disclosure_action": "none|direct_share|abstract_share|truthful_refusal|deflect|deny_knowledge",
  "new_expectations": [
    {{"id": "exp_...", "content": "我预期接下来会看到/验证什么", "verify_on": "next_user_turn|later", "confidence": 0.0, "memory_dynamics_binding": {{"should_bind_idle": false, "reason_codes": [], "evidence_refs": []}}}}
  ],
  "memory_writes": [
    {{"target": "short_term|long_term", "kind": "episode|fact|preference|relationship|identity|open_item", "content": "要写入的内容；未经证据卡或用户原话支持的候选不能写成事实", "salience": 0.0, "keywords": ["检索词"], "reason": "为什么值得记"}}
  ],
  "self_cognition_patch": {{
    "apply": false,
    "summary_delta": "",
    "new_identity_tensions": [],
    "new_known_limits": []
  }},
  "open_item_writes": [
    {{"id": "item_...", "content": "未完结事项", "status": "open", "next_check": "何时再看"}}
  ],
  "scheduled_outreach_requests": [
    {{"kind": "scheduled_outreach", "should_schedule": true, "basis": "user_explicit_request", "ordinary_language_intent": "用户明确要求稍后由我主动回访时才写入", "due_after_seconds": 120, "due_at": ""}}
  ],
  "habit_updates": [
    {{"content": "从用户反馈或反复证据中学到的表达习惯", "evidence": "支持这个习惯的用户原话或记忆", "confidence": 0.0}}
  ],
  "memory_dynamics_note": "哪些记忆被唤起、为什么、是否强化或衰减"
}}

memory_dynamics_binding 是结构化判断，不是关键词命中：只有当这一条 new_expectation 被当前轮的记忆预测张力明确支撑、且沉默后继续悬置会产生可追踪的 social/epistemic prediction error 时，才设置 should_bind_idle=true；否则保持 false。不要把普通 open_item、寒暄、泛泛“用户来意不明”或仅靠文本词面命中的内容设为 true。

scheduled_outreach_requests 是给 M14.2 的结构化语义结果，不是关键词命中结果。只有当用户明确要求“由我在未来某个时间或静默间隔后主动发一条消息/回访”，才写入一条；普通提醒、当前轮追问、模糊的 later、仅仅说自己要休息、或没有要求我未来主动发起消息时，必须返回空列表。due_after_seconds / due_at 由你根据整句语义和时间语境给出；工程层不会再从用户原文用关键词猜这个意图。
"""
    return system_prompt, user_prompt


def build_evidence_judge_prompt(
    *,
    user_text: str,
    speaker_name: str,
    current_user_id: str,
    lexical_candidates: list[Mapping[str, Any]],
    recall_query: Mapping[str, Any],
    entity_binding: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的“证据裁判”模块。你只判断候选短期记忆是否能支持当前用户问题，不生成最终回复。
你要把 grep/关键词召回的候选片段整理成证据 stance：知道、带边界地知道、不确定、没有线索或禁止假设。
候选里的 user_text/content 是来源用户原话或互动事实；assistant_reply 只是我当时说过的话，assistant_reply_use_as_fact=false 时不能当作外部事实证据。
软边界不是禁令；它只会提高传播成本。最终是否直说、抽象、拒答、转移或说不知道，由后续人格 thinking 模块根据社会动机和风险收益决定。
只输出 JSON。"""
    user_prompt = f"""当前用户: {speaker_name} ({current_user_id})
当前问题:
{user_text}

实体绑定上下文:
{_json_text(dict(entity_binding or {}))}

recall_query:
{_json_text(dict(recall_query))}

grep 候选短期记忆:
{_json_text([dict(item) for item in lexical_candidates], limit=16000)}

请输出 JSON:
{{
  "epistemic_stance": "known_from_recall|known_with_caveat|uncertain_recall|unknown_no_cue|forbidden_assumption",
  "relevant_evidence_ids": ["候选证据 id"],
  "topics": ["topic_id"],
  "sensitivity_class": "public|social_soft|personal_soft|personal_hard|explicit_secret",
  "redaction_targets": ["如果选择非 direct_share 时不应出现的具体词或模式"],
  "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
  "audience_risk": "对当前听众透露后的关系/反噬风险摘要",
  "expected_social_gain": "透露、抽象或否认后可能带来的社交收益摘要",
  "judge_summary": "一两句话总结证据是否支持当前问题"
}}
"""
    return system_prompt, user_prompt


def _normalize_evidence_judgment(
    raw: Mapping[str, Any],
    *,
    lexical_candidates: list[Mapping[str, Any]],
    current_user_id: str,
) -> dict[str, Any]:
    candidate_ids = {str(item.get("id", "")).strip() for item in lexical_candidates if item.get("id")}
    relevant = [
        item
        for item in _string_list(raw.get("relevant_evidence_ids"), limit=12)
        if item in candidate_ids
    ]
    if not relevant and lexical_candidates:
        relevant = [str(lexical_candidates[0].get("id", ""))]
    topics = sorted({*set(_string_list(raw.get("topics"), limit=8)), *set().union(*(set(_string_list(item.get("topics"), limit=8)) for item in lexical_candidates if str(item.get("id", "")) in relevant))})
    sensitivity = str(raw.get("sensitivity_class", "")).strip()
    if sensitivity not in {"public", "social_soft", "personal_soft", "personal_hard", "explicit_secret"}:
        sensitivity = _sensitivity_class_for_topics(topics)
    stance = str(raw.get("epistemic_stance", "")).strip()
    if stance not in {
        "known_from_recall",
        "known_with_caveat",
        "uncertain_recall",
        "unknown_no_cue",
        "forbidden_assumption",
    }:
        stance = "known_with_caveat" if relevant and sensitivity in {"personal_soft", "personal_hard"} else "known_from_recall" if relevant else "unknown_no_cue"
    redaction_targets = _string_list(raw.get("redaction_targets"), limit=12)
    for item in lexical_candidates:
        if str(item.get("id", "")) in relevant:
            redaction_targets = _unique_strings(redaction_targets, item.get("redaction_targets"), limit=12)
    allowed = _string_list(raw.get("allowed_reply_actions"), limit=8)
    valid_actions = {"direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"}
    allowed = [action for action in allowed if action in valid_actions]
    if not allowed:
        allowed = ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"] if stance == "known_with_caveat" else ["direct_share"]
    return {
        "epistemic_stance": stance,
        "relevant_evidence_ids": relevant,
        "topics": topics,
        "sensitivity_class": sensitivity,
        "redaction_targets": redaction_targets,
        "allowed_reply_actions": allowed,
        "audience_user_id": current_user_id,
        "audience_risk": str(raw.get("audience_risk", "")).strip(),
        "expected_social_gain": str(raw.get("expected_social_gain", "")).strip(),
        "judge_summary": str(raw.get("judge_summary", "")).strip(),
    }


def build_query_planner_prompt(
    *,
    user_text: str,
    speaker_name: str,
    recall_query: Mapping[str, Any],
    temporal_input: Mapping[str, Any],
    entity_binding: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是短期记忆 grep 查询规划器。你的任务不是判断事实，也不是生成回复，而是把用户自然语言改写成适合在短期记忆里精确搜索的关键词。
优先保留原词、人名、昵称、数字、稀有词；再补充少量同义 cue。遇到“露面/冒泡/有动静/打招呼/见到没”等说法，要补充“找过、来过、聊过、联系过、说过话”等互动存在 cue。
只输出 JSON。"""
    user_prompt = f"""当前说话人: {speaker_name}
用户输入:
{user_text}

已有 recall_query:
{_json_text(dict(recall_query))}

实体绑定上下文:
{_json_text(dict(entity_binding or {}))}

时间/上一轮摘要:
{_json_text(dict(temporal_input))}

请输出 JSON:
{{
  "search_terms": ["用于 grep 的关键词，最多16个"],
  "referenced_entities": ["被问到的人或对象"],
  "topic_hints": ["topic id，例如 personal_finance/health/home_address；不确定可空"],
  "is_interaction_presence_query": false,
  "planner_summary": "一句话说明为什么选这些词"
}}
"""
    return system_prompt, user_prompt


def _normalize_query_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    topic_hints = [topic for topic in _string_list(raw.get("topic_hints"), limit=8) if topic in _TOPIC_BY_ID]
    return {
        "search_terms": _string_list(raw.get("search_terms"), limit=16),
        "referenced_entities": _string_list(raw.get("referenced_entities"), limit=8),
        "topic_hints": topic_hints,
        "is_interaction_presence_query": bool(raw.get("is_interaction_presence_query", False)),
        "planner_summary": str(raw.get("planner_summary", "")).strip(),
    }


def _merge_query_plan_into_recall_query(
    recall_query: Mapping[str, Any],
    query_plan: Mapping[str, Any],
) -> dict[str, Any]:
    query = dict(recall_query)
    topic_terms = _append_topic_recall_terms([], set(_string_list(query_plan.get("topic_hints"), limit=8)), limit=24)
    query["semantic_terms"] = _unique_strings(
        query.get("semantic_terms"),
        query_plan.get("search_terms"),
        query_plan.get("referenced_entities"),
        topic_terms,
        limit=48,
    )
    query["query_plan"] = dict(query_plan)
    if bool(query_plan.get("is_interaction_presence_query", False)):
        query["interaction_presence_query"] = True
    return query


def build_post_reply_observer_prompt(
    *,
    user_text: str,
    reply: str,
    thinking: Mapping[str, Any],
    memory_dynamics: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
    temporal_assessment: Mapping[str, Any],
    turn_index: int,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的“回复后发观察模块”。
你只判断刚发出的主回复是否需要追加一条很短的补充气泡。
你不是第二个回复生成器，不能把长回复拆成多条，也不能继续角色表演或闲聊废话。
只有漏接重要情绪、需要自我修正、需要澄清、需要修复关系、需要承认重要关系信号时，才允许追加。
每轮最多追加一条，追加内容必须自然、短、像人后知后觉补一句。
只输出 JSON，不要 Markdown。
"""
    user_prompt = f"""turn_index: {turn_index}
用户刚说:
{user_text}

刚发出的主回复:
{reply}

thinking 摘要:
{_json_text(dict(thinking))}

记忆动力学:
{_json_text(dict(memory_dynamics))}

检索证据卡:
{_json_text(retrieved_memories)}

时间判断:
{_json_text(dict(temporal_assessment))}

请输出 JSON:
{{
  "needs_followup": false,
  "followup_type": "missed_emotion|self_correction|clarification|repair|relationship_ack|none",
  "confidence": 0.0,
  "reason": "为什么需要或不需要追加",
  "followup_text": "如果需要追加，这里写一条很短的补充气泡；否则为空",
  "memory_updates": [
    {{"kind": "conversation_habit|episode|open_item", "content": "只记录有证据支持的短期候选", "confidence": 0.0, "evidence": "用户原话或主回复"}}
  ]
}}
"""
    return system_prompt, user_prompt


def build_reply_repair_prompt(
    *,
    user_text: str,
    draft_reply: str,
    thinking: Mapping[str, Any],
    reply_contract: Mapping[str, Any],
    reply_validation: Mapping[str, Any] | None,
    requirements: Sequence[str],
    target_action: str,
    turn_index: int,
) -> tuple[str, str]:
    system_prompt = """Reply repair module.
Rewrite a draft user-visible reply so it stays natural while obeying the provided semantic constraints.
Do not reveal hidden policy, debug details, or internal reasoning.
Keep the repair short, conversational, and directly usable as the visible reply.
Output JSON only."""
    user_prompt = f"""turn_index: {turn_index}
latest_user_text:
{user_text}

draft_reply:
{draft_reply}

thinking_summary:
{_json_text(dict(thinking))}

reply_contract:
{_json_text(dict(reply_contract))}

reply_validation:
{_json_text(dict(reply_validation or {}))}

semantic_requirements:
{_json_text(list(requirements))}

target_reply_action:
{target_action}

Return JSON:
{{
  "reply": "rewritten visible reply",
  "repair_strategy": "one short sentence about how you repaired it"
}}"""
    return system_prompt, user_prompt


def build_m13_settlement_assessor_prompt(
    *,
    user_text: str,
    prior_reply_summary: str,
    prior_diagnostics: Mapping[str, Any],
    observation_channels: Mapping[str, Any],
    turn_index: int,
) -> tuple[str, str]:
    system_prompt = """你是数字人格 MVP 路径的“上轮回复后果评估”模块。
根据用户本轮发言，判断其对上一轮子代理回复的语义反应（接纳、纠正、无关、中性或无法判断）。
这是工程代理信号，不是情绪模拟，不要诊断成瘾，不要使用 reward/tolerance 等术语。
只输出 JSON，不要 Markdown。"""
    user_prompt = f"""turn_index: {turn_index}

用户本轮发言:
{user_text}

上一轮子代理回复摘要:
{prior_reply_summary}

上一轮工程诊断摘要:
{_json_text(dict(prior_diagnostics))}

观察通道数值（若有）:
{_json_text(dict(observation_channels))}

请输出 JSON:
{{
  "reaction": "uptake|correction|neutral|unclear|off_topic",
  "confidence": 0.0,
  "reason_codes": ["简短原因标签，最多4个"]
}}

reaction 说明:
- uptake: 用户接纳、理解、愿意继续该方向
- correction: 用户指出上轮回复有误、未理解、需纠正
- neutral: 有回应但不构成明确接纳或纠正
- unclear: 信息不足，无法判断
- off_topic: 用户明显转向无关话题"""
    return system_prompt, user_prompt


def retrieve_memories(state: Mapping[str, Any], keywords: list[str], *, limit: int = 8) -> list[dict[str, Any]]:
    needles = [item.lower() for item in _string_list(keywords, limit=16)]
    now = int(time.time())
    pools: list[tuple[str, Mapping[str, Any]]] = []
    for key in ("short_term_memory", "long_term_memory", "open_items", "pending_expectations"):
        value = state.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    pools.append((key, item))

    scored: list[tuple[float, dict[str, Any]]] = []
    for source, item in pools:
        text = json.dumps(item, ensure_ascii=False).lower()
        score = 0.0
        for needle in needles:
            if not needle:
                continue
            if needle in text:
                score += 2.0
            else:
                parts = [p for p in re.split(r"\s+", needle) if p]
                score += sum(0.4 for p in parts if p in text)
        if score > 0.0:
            recall = explain_recall_candidate(item, query=needles, now=now, retrieved_context={})
            recall_score = recall.score
            if recall_score <= 0.0:
                continue
            score *= max(0.05, recall_score)
            payload = dict(item)
            payload["_source_file"] = source
            payload["_retrieval_score"] = round(score, 3)
            payload["_m14_7_recall_score"] = recall_score
            payload["_m17_item_support"] = recall_score
            payload["_m17_factor_breakdown"] = recall.to_dict()
            scored.append((score, payload))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]


def retrieve_memories_by_ids(state: Mapping[str, Any], memory_ids: list[str], *, limit: int = 8) -> list[dict[str, Any]]:
    wanted = {str(item).strip() for item in memory_ids if str(item).strip()}
    if not wanted:
        return []
    rows: list[dict[str, Any]] = []
    for key in ("short_term_memory", "long_term_memory", "open_items", "pending_expectations"):
        value = state.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, Mapping) and str(item.get("id", item.get("expectation_id", ""))).strip() in wanted:
                if str(item.get("status", "") or "") == "archived":
                    continue
                payload = dict(item)
                payload["_source_file"] = key
                payload["_retrieval_score"] = 9.0
                rows.append(payload)
                if len(rows) >= max(1, int(limit)):
                    return rows
    return rows


def _unique_strings(*values: Any, limit: int = 16) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for item in _string_list(value, limit=limit):
            key = item.casefold()
            if key and key not in seen:
                seen.add(key)
                result.append(item)
                if len(result) >= limit:
                    return result
    return result


def _rough_terms(text: str, *, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", str(text or ""))
    return [token[:80] for token in tokens[:limit] if token.strip()]


def _name_like_terms(text: str, *, limit: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{0,31}|[\u4e00-\u9fff]{2,8}", str(text or ""))
    stopwords = {
        "不是",
        "就是",
        "我是",
        "我才",
        "他说",
        "今天",
        "早上",
        "中午",
        "晚上",
        "现在",
        "这个",
        "那个",
        "真的",
        "知道",
        "没有",
        "找你",
        "找我",
        "欠我",
        "请你",
        "喜欢",
    }
    result: list[str] = []
    for token in tokens:
        if any(marker in token for marker in ("我", "你", "他", "她", "自己", "有没有")):
            continue
        if token in stopwords:
            continue
        result.append(token[:80])
        if len(result) >= limit:
            break
    return result


def _interlocutor_aliases(
    state: Mapping[str, Any],
    *,
    user_id: str,
    display_name: str,
) -> list[str]:
    models = _mapping(state.get("m11_user_models"))
    payload = _mapping(models.get(user_id))
    aliases = _unique_strings(
        [display_name, user_id],
        payload.get("aliases"),
        _mapping(payload.get("identity_binding")).get("aliases"),
        limit=16,
    )
    return aliases


def _extract_alias_assertions(
    user_text: str,
    *,
    display_name: str,
    user_id: str,
) -> list[str]:
    text = str(user_text or "")
    aliases: list[str] = []
    patterns = [
        r"我(?:才是|就是|是)(?!说)(?P<alias>[A-Za-z0-9_\-\u4e00-\u9fff]{1,24})",
        rf"{re.escape(display_name)}\s*(?:就是|是)\s*(?P<alias>[A-Za-z0-9_\-\u4e00-\u9fff]{{1,24}})",
        rf"{re.escape(user_id)}\s*(?:就是|是)\s*(?P<alias>[A-Za-z0-9_\-\u4e00-\u9fff]{{1,24}})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            alias = str(match.group("alias") or "").strip("_- ")
            if alias and alias not in {"说", "我", "你", "他", "她", "它"}:
                aliases.append(alias)
    return _unique_strings(aliases, limit=8)


def _record_interlocutor_aliases(
    state: dict[str, Any],
    *,
    user_id: str,
    display_name: str,
    aliases: list[str],
    evidence: str,
    now: int,
) -> list[str]:
    if _m12_enabled_for_state(state):
        return []
    clean_aliases = [
        alias
        for alias in _unique_strings(aliases, limit=12)
        if alias and alias not in {display_name, user_id}
    ]
    if not clean_aliases:
        return []
    models = _mapping(state.get("m11_user_models"))
    payload = _mapping(models.get(user_id))
    payload["aliases"] = _unique_strings(payload.get("aliases"), clean_aliases, limit=16)
    binding = _mapping(payload.get("identity_binding"))
    binding["aliases"] = list(payload["aliases"])
    binding["last_alias_evidence"] = evidence[:240]
    binding["updated_at"] = now
    payload["identity_binding"] = binding
    models[user_id] = payload
    state["m11_user_models"] = models
    return clean_aliases


def _source_names_from_short_memory(state: Mapping[str, Any]) -> list[str]:
    rows = state.get("short_term_memory", [])
    if not isinstance(rows, list):
        return []
    names: list[str] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        names = _unique_strings(
            names,
            [item.get("source_display_name"), item.get("source_user_id")],
            limit=48,
        )
    return names


def _text_mentions_name(text: str, name: str) -> bool:
    needle = str(name or "").strip()
    if not needle:
        return False
    return needle.casefold() in str(text or "").casefold()


def build_entity_binding_context(
    *,
    state: Mapping[str, Any],
    user_text: str,
    display_name: str,
    user_id: str,
    temporal_input: Mapping[str, Any] | None = None,
    m12_turn_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temporal = _mapping(state.get("temporal_state"))
    previous_summary = _mapping((temporal_input or {}).get("previous_turn_summary"))
    previous_user_text = str(previous_summary.get("user_text") or temporal.get("last_user_text") or "")
    previous_trace = _mapping(temporal.get("last_share_trace"))
    alias_assertions = _extract_alias_assertions(user_text, display_name=display_name, user_id=user_id)
    m12_enabled = _m12_enabled_for_state(state)
    current_aliases = _interlocutor_aliases(state, user_id=user_id, display_name=display_name)
    if not m12_enabled:
        current_aliases = _unique_strings(current_aliases, alias_assertions, limit=16)
    if m12_enabled:
        m12_state = _load_m12_state(state)
        prof = m12_state.profiles_by_user.get(user_id)
        if prof is not None:
            reply_policy_dict = _m12_reply_policy_dict_for_entity_binding(
                m12_state=m12_state,
                profile=prof,
                m12_turn_result=m12_turn_result,
            )
            promote = _m12_claim_alias_promotable(
                reply_policy_dict,
                identity_state=str(prof.identity_state),
                confidence_band=str(prof.binding_confidence_band),
            )
            if str(prof.identity_state) == "corroborated":
                for obs in prof.aliases_observed:
                    t = str(obs.alias_text or "").strip()
                    if t:
                        current_aliases = _unique_strings(current_aliases, [t], limit=16)
            elif promote and prof.aliases_observed:
                latest = str(prof.aliases_observed[-1].alias_text or "").strip()
                if latest:
                    current_aliases = _unique_strings(current_aliases, [latest], limit=16)
    current_alias_folded = {alias.casefold() for alias in current_aliases}
    previous_target = str(previous_trace.get("target_person") or "").strip()
    source_names = _source_names_from_short_memory(state)
    candidate_names = _unique_strings(source_names, current_aliases, [previous_target], limit=64)
    mentioned: list[dict[str, Any]] = []
    for name in candidate_names:
        appears_current = _text_mentions_name(user_text, name)
        appears_previous = _text_mentions_name(previous_user_text, name)
        if not (appears_current or appears_previous):
            continue
        is_current_alias = name.casefold() in current_alias_folded
        mentioned.append(
            {
                "name": name,
                "is_current_user_alias": is_current_alias,
                "source": "current_text" if appears_current else "previous_turn",
            }
        )
    non_current_mentions = [item["name"] for item in mentioned if not item.get("is_current_user_alias")]
    previous_target_valid = previous_target and previous_target.casefold() not in current_alias_folded
    self_reference = any(marker in str(user_text or "") for marker in ("我自己", "本人", "我周青自己", "我zq自己"))
    target_person = ""
    target_reason = ""
    if self_reference:
        target_person = display_name
        target_reason = "explicit_self_reference"
    elif non_current_mentions:
        current_mentions = [item["name"] for item in mentioned if item["source"] == "current_text" and not item.get("is_current_user_alias")]
        target_person = current_mentions[0] if current_mentions else non_current_mentions[0]
        target_reason = "named_non_current_entity"
    elif previous_target_valid and re.search(r"(他|她|这人|那家伙|那个人|这家伙)", str(user_text or "")):
        target_person = previous_target
        target_reason = "pronoun_inherited_previous_target"

    pronoun_bindings: dict[str, str] = {}
    if target_person and re.search(r"(他|她|这人|那家伙|那个人|这家伙)", str(user_text or "")):
        pronoun_bindings["他/她/这人/那家伙"] = target_person

    relationship_roles: dict[str, str] = {}
    if target_person:
        if re.search(r"(他|她|这人|那家伙|那个人|这家伙)?[^。！？]{0,8}欠我", str(user_text or "")):
            relationship_roles["debtor"] = target_person
            relationship_roles["creditor"] = display_name
        elif re.search(r"我[^。！？]{0,8}欠(他|她|这人|那家伙|那个人|这家伙)", str(user_text or "")):
            relationship_roles["debtor"] = display_name
            relationship_roles["creditor"] = target_person

    conflicts: list[str] = []
    for item in mentioned:
        if item.get("is_current_user_alias") and target_person == item.get("name") and not self_reference:
            conflicts.append("current_user_alias_used_as_third_party_target")

    return {
        "current_interlocutor": {
            "display_name": display_name,
            "user_id": user_id,
            "aliases": current_aliases,
        },
        "alias_assertions": alias_assertions,
        "mentioned_entities": mentioned,
        "target_person": target_person,
        "target_reason": target_reason,
        "pronoun_bindings": pronoun_bindings,
        "relationship_roles": relationship_roles,
        "binding_confidence": (
            "certain"
            if target_person or (alias_assertions and not m12_enabled)
            else "ambiguous"
        ),
        "conflicts": conflicts,
    }


def _dialogue_turn_parts(item: Mapping[str, Any]) -> tuple[str, str]:
    user_part = str(item.get("user_text", "")).strip()
    assistant_part = str(item.get("assistant_reply", "")).strip()
    content = str(item.get("content", ""))
    if (not user_part or not assistant_part) and str(item.get("kind", "")).strip() == "dialogue_turn":
        match = re.match(r"\s*用户说[:：](?P<user>.*?)(?:\n\s*我回复[:：](?P<assistant>.*))?\s*$", content, flags=re.DOTALL)
        if not match:
            match = re.match(r"\s*鐢ㄦ埛璇达細(?P<user>.*?)(?:\n\s*鎴戝洖澶嶏細(?P<assistant>.*))?\s*$", content, flags=re.DOTALL)
        if match:
            user_part = user_part or str(match.group("user") or "").strip()
            assistant_part = assistant_part or str(match.group("assistant") or "").strip()
    return user_part, assistant_part


def _memory_fact_text(item: Mapping[str, Any]) -> str:
    if str(item.get("kind", "")).strip() == "dialogue_turn":
        user_part, _ = _dialogue_turn_parts(item)
        return user_part or str(item.get("content", "")).strip()
    return str(item.get("content", "")).strip()


def _memory_index_text(item: Mapping[str, Any]) -> str:
    payload = dict(item)
    if str(payload.get("kind", "")).strip() == "dialogue_turn":
        user_part, _ = _dialogue_turn_parts(payload)
        user_part = user_part or str(payload.get("content", "")).strip()
        payload["content"] = user_part
        payload["user_text"] = user_part
        payload.pop("assistant_reply", None)
    return json.dumps(payload, ensure_ascii=False)


_FOLLOW_UP_PROBE_MARKERS = (
    "真的不知道",
    "真不知道",
    "你确定",
    "确定不知道",
    "不是知道",
    "没印象吗",
    "不记得",
)


_INTERACTION_PRESENCE_MARKERS = (
    "找过你",
    "找你",
    "来找你",
    "找过",
    "来过",
    "联系过",
    "联系你",
    "聊过",
    "说过话",
    "来骚扰你",
)


_QUERY_PLANNER_CUE_MARKERS = (
    *_INTERACTION_PRESENCE_MARKERS,
    "露面",
    "冒泡",
    "动静",
    "打招呼",
    "见到",
    "见过",
    "碰到",
    "出现",
)


def _is_follow_up_probe(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in _FOLLOW_UP_PROBE_MARKERS)


def _is_interaction_presence_query(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in _INTERACTION_PRESENCE_MARKERS)


def _should_run_query_planner(
    state: Mapping[str, Any],
    *,
    user_text: str,
    recall_query: Mapping[str, Any],
    entity_binding: Mapping[str, Any] | None = None,
) -> bool:
    if _is_follow_up_probe(user_text) or _has_any_marker(user_text, _QUERY_PLANNER_CUE_MARKERS):
        return True
    if _mapping(entity_binding).get("target_person") and re.search(r"(他|她|这人|那家伙|那个人|这家伙)", str(user_text or "")):
        return True
    terms = _string_list(recall_query.get("semantic_terms"), limit=16)
    rows = state.get("short_term_memory", [])
    if not isinstance(rows, list):
        return False
    haystack = " ".join(terms + _rough_terms(user_text, limit=8)).casefold()
    for item in rows[-24:]:
        if not isinstance(item, Mapping):
            continue
        for raw in (item.get("source_display_name"), item.get("source_user_id")):
            name = str(raw or "").strip()
            if name and name.casefold() in haystack:
                return True
    return False


def _specificity_bonus(term: str) -> float:
    if re.fullmatch(r"\d+", term):
        return 1.2
    if re.search(r"\d", term):
        return 0.9
    if len(term) >= 4:
        return 0.35
    return 0.0


def _lexical_recall_terms(
    *,
    state: Mapping[str, Any],
    user_text: str,
    recall_query: Mapping[str, Any] | None,
    entity_binding: Mapping[str, Any] | None = None,
    limit: int = 40,
) -> list[str]:
    query = _mapping(recall_query)
    binding = _mapping(entity_binding)
    base_terms = _unique_strings(
        query.get("semantic_terms"),
        [binding.get("target_person")],
        list(_mapping(binding.get("pronoun_bindings")).values()) if binding else [],
        _rough_terms(user_text, limit=12),
        limit=limit,
    )
    active_topics = _topic_ids_for_text(base_terms, user_text)
    terms = _append_topic_recall_terms(base_terms, active_topics, limit=limit)
    temporal = _mapping(state.get("temporal_state"))
    previous_trace = _mapping(temporal.get("last_share_trace"))
    if _is_follow_up_probe(user_text):
        terms = _unique_strings(
            terms,
            previous_trace.get("lexical_recall_terms"),
            previous_trace.get("evidence_topics"),
            previous_trace.get("evidence_source_names"),
            [previous_trace.get("target_person")],
            limit=limit,
        )
        terms = _append_topic_recall_terms(terms, set(_string_list(previous_trace.get("evidence_topics"), limit=8)), limit=limit)
    return terms


def _interaction_target_names(
    state: Mapping[str, Any],
    *,
    user_text: str,
    recall_query: Mapping[str, Any] | None,
    entity_binding: Mapping[str, Any] | None = None,
) -> set[str]:
    query = _mapping(recall_query)
    binding = _mapping(entity_binding)
    binding_target = str(binding.get("target_person") or "").strip()
    current_aliases = {
        alias.casefold()
        for alias in _string_list(_mapping(binding.get("current_interlocutor")).get("aliases"), limit=16)
    }
    if binding_target and binding_target.casefold() not in current_aliases:
        return {binding_target}
    referenced = [
        str(item.get("name", "")).strip()
        for item in binding.get("mentioned_entities", [])
        if isinstance(item, Mapping) and not bool(item.get("is_current_user_alias", False))
    ]
    if referenced:
        return set(referenced)
    temporal = _mapping(state.get("temporal_state"))
    haystack = _joined_text(
        user_text,
        query.get("semantic_terms"),
        query.get("relationship_terms"),
        _mapping(temporal.get("previous_turn_summary")).get("user_text"),
        temporal.get("last_user_text"),
    )
    if not haystack:
        return set()
    names: set[str] = set()
    rows = state.get("short_term_memory", [])
    if not isinstance(rows, list):
        return names
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        for raw in (item.get("source_display_name"), item.get("source_user_id")):
            name = str(raw or "").strip()
            if name and name.casefold() in haystack:
                names.add(name)
    return names


def _interaction_presence_candidates(
    state: Mapping[str, Any],
    *,
    user_text: str,
    recall_query: Mapping[str, Any] | None,
    current_user_id: str = "",
    entity_binding: Mapping[str, Any] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not (_is_interaction_presence_query(user_text) or bool(_mapping(recall_query).get("interaction_presence_query", False))):
        return []
    target_names = _interaction_target_names(
        state,
        user_text=user_text,
        recall_query=recall_query,
        entity_binding=entity_binding,
    )
    if not target_names:
        return []
    rows = state.get("short_term_memory", [])
    if not isinstance(rows, list):
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    target_folded = {name.casefold() for name in target_names}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("kind", "")).strip() not in {"dialogue_turn", "episode"}:
            continue
        source_names = {
            str(item.get("source_user_id", "")).strip().casefold(),
            str(item.get("source_display_name", "")).strip().casefold(),
        }
        if not source_names.intersection(target_folded):
            continue
        source_user_id = str(item.get("source_user_id", "")).strip()
        if current_user_id and source_user_id == current_user_id:
            continue
        try:
            created_at = float(item.get("created_at", 0) or 0)
        except (TypeError, ValueError):
            created_at = 0.0
        card = _evidence_card(
            "short_term_memory",
            item,
            score=5.0 + created_at * 0.000001,
            reasons=["source_interaction_recent"],
            conflict_note="",
            abstract_only=False,
            sharing_decision={},
        )
        card["epistemic_stance"] = "known_from_recall"
        card["interaction_presence"] = True
        card["assistant_reply_use_as_fact"] = False
        scored.append((5.0 + created_at * 0.000001, card))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [card for _, card in scored[:limit]]


def lexical_recall_short_term_candidates(
    state: Mapping[str, Any],
    *,
    user_text: str,
    recall_query: Mapping[str, Any] | None = None,
    current_user_id: str = "",
    entity_binding: Mapping[str, Any] | None = None,
    group_turn_binding: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    interaction_candidates = _interaction_presence_candidates(
        state,
        user_text=user_text,
        recall_query=recall_query,
        current_user_id=current_user_id,
        entity_binding=entity_binding,
        limit=limit,
    )
    terms = _lexical_recall_terms(
        state=state,
        user_text=user_text,
        recall_query=recall_query,
        entity_binding=entity_binding,
        limit=48,
    )
    if not terms:
        return interaction_candidates
    rows = state.get("short_term_memory", [])
    if not isinstance(rows, list):
        return []
    current_audience_ids = _bounded_string_list(
        _mapping(group_turn_binding).get("visible_participant_ids"),
        limit=8,
        item_max_chars=64,
    )
    current_speaker_participant_id = str(
        _mapping(group_turn_binding).get("current_speaker_participant_id", "")
        or current_user_id
        or ""
    ).strip()
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        text = _memory_index_text(item).casefold()
        matched = []
        score = 0.0
        for term in terms:
            lowered = term.casefold()
            if not lowered:
                continue
            if lowered in text:
                matched.append(term)
                score += 1.0 + _specificity_bonus(term)
        if not matched:
            continue
        source_user_id = str(item.get("source_user_id", "")).strip()
        cross_user = bool(current_user_id and source_user_id and source_user_id != current_user_id)
        score += min(2.0, len(set(matched)) * 0.35)
        try:
            score += float(item.get("salience", 0.0) or 0.0) * 0.25
        except (TypeError, ValueError):
            pass
        card = _evidence_card(
            "short_term_memory",
            item,
            score=score,
            reasons=[f"lexical_term:{term}" for term in matched[:6]],
            conflict_note="",
            abstract_only=False,
            sharing_decision={},
        )
        card["matched_terms"] = matched[:8]
        card["audience_user_id"] = current_user_id
        card["is_cross_user"] = bool(cross_user)
        if current_audience_ids:
            card["current_audience_participant_ids"] = current_audience_ids
            card["current_audience_scope"] = _group_audience_scope_label(current_audience_ids)
            policy = _group_memory_policy_for_card(
                card,
                current_audience_participant_ids=current_audience_ids,
                current_speaker_participant_id=current_speaker_participant_id,
            )
            card["group_privacy_policy"] = policy
            card["selected_disclosure_mode"] = policy["selected_disclosure_mode"]
            card["shareability_class"] = policy["shareability_class"]
        if cross_user and card.get("shareability") == "restricted_implicit":
            card["epistemic_stance"] = "known_with_caveat"
        scored.append((score, card))
    scored.sort(key=lambda row: row[0], reverse=True)
    merged = [*interaction_candidates, *[card for _, card in scored]]
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for card in merged:
        key = str(card.get("id", ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(card)
        if len(deduped) >= limit:
            break
    return deduped


def _short_term_interaction_experiences(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    short = state.get("short_term_memory", [])
    if not isinstance(short, list):
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for item in short:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind", "")).strip()
        if kind not in {"dialogue_turn", "episode"}:
            continue
        if str(item.get("shareability", "default_social")).strip() != "default_social":
            continue
        display = str(item.get("source_display_name") or item.get("source_user_id") or "").strip()
        user_id = str(item.get("source_user_id") or display).strip()
        if not display and not user_id:
            continue
        key = user_id or display
        row = grouped.setdefault(
            key,
            {
                "source_user_id": user_id,
                "source_display_name": display or user_id,
                "count": 0,
                "last_created_at": 0,
                "last_content": "",
            },
        )
        row["count"] = int(row.get("count", 0)) + 1
        try:
            created_at = int(float(item.get("created_at", 0) or 0))
        except (TypeError, ValueError):
            created_at = 0
        if created_at >= int(row.get("last_created_at", 0) or 0):
            row["last_created_at"] = created_at
            row["last_content"] = _memory_fact_text(item)[:180]

    experiences: list[dict[str, Any]] = []
    for key, row in grouped.items():
        count = int(row.get("count", 0) or 0)
        if count < 2:
            continue
        display = str(row.get("source_display_name") or key).strip()
        user_id = str(row.get("source_user_id") or display).strip()
        snippet = str(row.get("last_content", "")).strip()
        content = f"{display}最近和我说过{count}次话。"
        if snippet:
            content += f"最近一次片段：{snippet}"
        safe_user_id = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", user_id)[:48]
        experiences.append(
            {
                "id": f"stm_interaction_experience_{safe_user_id}",
                "kind": "interaction_experience",
                "content": content,
                "salience": min(0.85, 0.38 + count * 0.06),
                "confidence": min(0.92, 0.58 + count * 0.05),
                "keywords": [display, user_id, "说过话", "近期互动", "认识"],
                "source": "memory_dynamics_adapter",
                "created_at": int(row.get("last_created_at", 0) or 0),
                "source_user_id": user_id,
                "source_display_name": display,
                "shareability": "default_social",
                "restriction_confidence": 0.75,
            }
        )
    return experiences


def _memory_pools(state: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    pools: list[tuple[str, Mapping[str, Any]]] = []
    for key in ("short_term_memory", "long_term_memory", "open_items", "pending_expectations"):
        value = state.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    pools.append((key, item))
    for item in _short_term_interaction_experiences(state):
        pools.append(("short_term_memory", item))
    return pools


def _memory_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("status", "")).strip()
    if status:
        return status
    content = str(item.get("content", ""))
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    if isinstance(parsed, Mapping):
        return str(parsed.get("status", "")).strip()
    return ""


_BREVITY_FEEDBACK_MARKERS = (
    "太长",
    "啰嗦",
    "罗嗦",
    "短一点",
    "简短",
    "分开说",
    "分开几条",
    "一长串",
    "一句话",
)


def _structured_memory_dynamics_binding(row: Mapping[str, Any]) -> bool:
    binding = _mapping(row.get("memory_dynamics_binding"))
    return bool(
        binding.get("should_bind_idle")
        or row.get("memory_dynamics_idle")
        or row.get("memory_dynamics_trigger")
        or str(row.get("source", "") or "").strip() == "memory_dynamics_adapter"
        or str(row.get("source_kind", "") or "").strip() == "memory_dynamics_expectation"
        or str(row.get("verify_on", "") or "").strip() == "memory_dynamics_idle"
    )


def _traceable_memory_dynamics_expectation_candidate(row: Mapping[str, Any]) -> bool:
    content = str(row.get("content", row.get("summary", "")) or "").strip()
    if len(content) < 8:
        return False
    verify_on = str(row.get("verify_on", row.get("verify", "")) or "").strip().casefold()
    if verify_on and verify_on not in {"next_user_turn", "next_turn", "after_next_user_message", "memory_dynamics_idle"}:
        return False
    return _bounded_float(row.get("confidence"), default=0.5) >= 0.35


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _reply_contract(mode: str, *, prefers_short: bool) -> dict[str, Any]:
    if mode == "serious_thinking":
        return {
            "conversation_mode": "serious_thinking",
            "max_sentences": 20,
            "max_response_moves": 4,
            "max_chars": 2400,
            "roleplay_density": "light",
            "catchphrase_limit": 1,
            "question_policy": "only_if_needed",
            "hard_rules": [
                "reply may be multi-paragraph when the user asks for analysis or implementation details",
                "never include diagnostics, JSON, conscious_plan, llm_thinking_result, or memory_dynamics in reply",
            ],
        }
    if mode == "casual_fast":
        return {
            "conversation_mode": "casual_fast",
            "max_sentences": 1,
            "max_response_moves": 1,
            "max_chars": 45 if prefers_short else 60,
            "roleplay_density": "light",
            "catchphrase_limit": 1,
            "question_policy": "only_if_user_leaves_clear_opening",
            "hard_rules": [
                "reply in one natural sentence",
                "do not combine empathy, roleplay, advice, and a question in one bubble",
                "prefer leaving space for the user over adding a question",
                "never include diagnostics, JSON, conscious_plan, llm_thinking_result, or memory_dynamics in reply",
            ],
        }
    return {
        "conversation_mode": "balanced",
        "max_sentences": 2,
        "max_response_moves": 2,
        "max_chars": 140,
        "roleplay_density": "light",
        "catchphrase_limit": 1,
        "question_policy": "optional_one",
        "hard_rules": [
            "reply in one or two natural sentences",
            "avoid packing empathy, roleplay, advice, and a question into one reply",
            "never include diagnostics, JSON, conscious_plan, llm_thinking_result, or memory_dynamics in reply",
        ],
    }


def _evidence_card(
    source: str,
    item: Mapping[str, Any],
    *,
    score: float,
    reasons: list[str],
    conflict_note: str = "",
    abstract_only: bool = False,
    sharing_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind = str(item.get("kind", source)).strip() or source
    salience = _bounded_float(item.get("salience"), default=0.35)
    confidence = _bounded_float(item.get("confidence"), default=max(0.2, min(0.9, 0.45 + salience * 0.35)))
    status = _memory_status(item)
    use_as_fact = source in {"short_term_memory", "long_term_memory"} and kind not in {
        "expectation_result",
        "open_item",
    } and status not in {"violated", "uncertain"}
    shareability = _memory_shareability(item)
    topics = _memory_topics(item)
    sensitivity = _memory_sensitivity(item)
    user_part, assistant_part = _dialogue_turn_parts(item)
    content = (_memory_fact_text(item) or str(item.get("content", "")).strip())[:600]
    if abstract_only:
        content = _redact_memory_content(item, max_chars=120)
    return {
        "id": str(item.get("id", "")).strip(),
        "kind": kind,
        "content": content,
        "user_text": user_part,
        "assistant_reply": assistant_part,
        "assistant_reply_use_as_fact": False if assistant_part else None,
        "source": source,
        "confidence": round(confidence, 6),
        "salience": round(salience, 6),
        "why_relevant": reasons[:5],
        "conflict_note": conflict_note,
        "use_as_fact": bool(use_as_fact),
        "shareability": shareability,
        "topics": topics,
        "sensitivity_class": sensitivity,
        "sensitivity": sensitivity,
        "redaction_targets": _redaction_targets_for_text(_memory_fact_text(item), topics),
        "source_user_id": str(item.get("source_user_id", "")).strip(),
        "source_display_name": str(item.get("source_display_name", "")).strip(),
        "source_participant_id": str(item.get("source_participant_id", item.get("source_user_id", ""))).strip(),
        "source_audience_participant_ids": _bounded_string_list(
            item.get("source_audience_participant_ids"),
            limit=8,
            item_max_chars=64,
        ),
        "source_audience_scope": str(item.get("source_audience_scope", "")).strip(),
        "audience_user_id": "",
        "is_cross_user": False,
        "epistemic_stance": "known_from_recall",
        "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"]
        if shareability == "restricted_implicit"
        else ["direct_share"],
        "abstract_only": bool(abstract_only),
        "sharing_decision": dict(sharing_decision or {}),
        "_retrieval_score": round(score, 3),
        "_source_file": source,
        "_m17_item_support": round(_bounded_float(item.get("_m17_item_support", item.get("_m14_7_recall_score", 0.0)), default=0.0), 6),
        "_m17_evidence_refs": _string_list(item.get("evidence_refs"), limit=8),
        "_m17_prediction_ids": _memory_prediction_ids(item),
        "_m17_expectation_ids": _memory_expectation_ids(item, source=source),
        "_m17_episode_ids": _memory_episode_ids(item),
        "_m17_contradiction_risk": round(_memory_contradiction_risk(item), 6),
        "_m17_factor_breakdown": {
            "confidence": round(confidence, 6),
            "salience": round(salience, 6),
        },
    }


def retrieve_memories_for_guidance(
    state: Mapping[str, Any],
    recall_query: Mapping[str, Any] | None,
    *,
    limit: int = 8,
    now: int = 0,
    group_turn_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query = _mapping(recall_query)
    expectation_ids = {item.casefold() for item in _string_list(query.get("expectation_ids"), limit=16)}
    memory_kinds = {item.casefold() for item in _string_list(query.get("memory_kinds"), limit=12)}
    base_semantic_terms = _unique_strings(
        query.get("semantic_terms"),
        query.get("relationship_terms"),
        query.get("status_terms"),
        limit=24,
    )
    active_topics = _topic_ids_for_text(
        base_semantic_terms,
        query.get("current_task"),
        query.get("next_task"),
    )
    semantic_terms = (
        _append_topic_recall_terms(base_semantic_terms, active_topics, limit=36)
        if active_topics
        else base_semantic_terms
    )
    source_priority = _string_list(
        query.get("source_priority")
        or ["pending_expectations", "short_term_memory", "long_term_memory", "open_items"],
        limit=8,
    )
    priority_rank = {source: len(source_priority) - idx for idx, source in enumerate(source_priority)}
    status_terms = {item.casefold() for item in _string_list(query.get("status_terms"), limit=8)}
    current_user_id = str(query.get("current_user_id", "")).strip()
    sharing_intent = str(query.get("sharing_intent", "none")).strip() or "none"
    expected_reaction = str(query.get("expected_audience_reaction", "neutral")).strip() or "neutral"
    expectation_status = str(query.get("sharing_expectation_status", "unverified")).strip() or "unverified"
    regret_bias = _bounded_float(query.get("sharing_regret_bias"), default=0.0)
    current_audience_ids = _bounded_string_list(
        _mapping(group_turn_binding).get("visible_participant_ids"),
        limit=8,
        item_max_chars=64,
    )
    current_speaker_participant_id = str(
        _mapping(group_turn_binding).get("current_speaker_participant_id", "")
        or current_user_id
        or ""
    ).strip()

    scored: list[tuple[float, dict[str, Any]]] = []
    for source, item in _memory_pools(state):
        reasons: list[str] = []
        score = 0.0
        item_id = str(item.get("id", "")).strip()
        kind = str(item.get("kind", source)).strip()
        text = _memory_index_text(item).casefold()
        status = _memory_status(item).casefold()

        source_user_id = str(item.get("source_user_id", "")).strip()
        cross_user = bool(current_user_id and source_user_id and source_user_id != current_user_id)
        candidate_payload = dict(item)
        candidate_payload["shareability"] = _memory_shareability(item)
        candidate_payload["expected_audience_reaction"] = expected_reaction
        candidate_payload["expectation_status"] = expectation_status
        sharing_decision = decide_social_sharing(
            candidate_from_memory(candidate_payload, audience_user_id=current_user_id),
            sharing_intent=sharing_intent,  # type: ignore[arg-type]
            regret_bias=regret_bias,
        )
        if cross_user and sharing_decision.action == "withhold":
            continue

        if item_id and item_id.casefold() in expectation_ids:
            score += 6.0
            reasons.append(f"expectation_id:{item_id}")
        kind_match = bool(kind and kind.casefold() in memory_kinds)
        if kind_match and kind.casefold() in {"expectation", "expectation_result", "open_item"}:
            score += 2.0
            reasons.append(f"kind:{kind}")
        if kind.casefold() == "interaction_experience":
            score += 2.4
            reasons.append("kind:interaction_experience")
        item_topics = set(_memory_topics(item))
        if active_topics and item_topics.intersection(active_topics):
            score += 2.0
            reasons.extend(f"topic_context:{topic}" for topic in sorted(item_topics.intersection(active_topics))[:2])
        if status and status in status_terms:
            score += 1.2
            reasons.append(f"status:{status}")
        source_names = {
            str(item.get("source_user_id", "")).strip().casefold(),
            str(item.get("source_display_name", "")).strip().casefold(),
        }
        for term in semantic_terms:
            lowered = term.casefold()
            if not lowered:
                continue
            if lowered in text:
                score += 1.5
                reasons.append(f"term:{term}")
                if lowered in source_names and kind.casefold() in {"dialogue_turn", "episode", "interaction_experience"}:
                    score += 1.1
                    reasons.append(f"source_interaction:{term}")
            else:
                parts = [part for part in re.split(r"\s+", lowered) if part]
                part_hits = sum(1 for part in parts if part in text)
                if part_hits:
                    score += 0.25 * part_hits
                    reasons.append(f"partial_term:{term}")
        if score <= 0.0:
            continue
        if kind_match and kind.casefold() not in {"expectation", "expectation_result", "open_item"}:
            score += 0.6
            reasons.append(f"kind:{kind}")
        if source in priority_rank:
            score += priority_rank[source] * 0.05
        if cross_user:
            score += sharing_decision.net_free_energy_reduction * 0.25
            reasons.append(f"sharing_decision:{sharing_decision.action}")
            shareability = _memory_shareability(item)
            if shareability == "restricted_implicit":
                score -= 0.8
                reasons.append("cross_user_implicit_risk")
        recall = explain_recall_candidate(
            item,
            query=semantic_terms + list(expectation_ids) + list(memory_kinds),
            now=now,
            retrieved_context={"source": source, "reasons": reasons},
        )
        recall_score = recall.score
        if recall_score <= 0.0:
            continue
        score *= max(0.05, recall_score)
        conflict_note = ""
        if "violated" in status_terms and status in {"violated", "uncertain"}:
            conflict_note = "expectation verification is not settled as a fact"
        abstract_only = bool(cross_user and sharing_decision.action == "withhold")
        card = _evidence_card(
            source,
            item,
            score=score,
            reasons=reasons,
            conflict_note=conflict_note,
            abstract_only=abstract_only,
            sharing_decision=sharing_decision.to_dict() if cross_user else {},
        )
        card["audience_user_id"] = current_user_id
        card["is_cross_user"] = bool(cross_user)
        if current_audience_ids:
            card["current_audience_participant_ids"] = current_audience_ids
            card["current_audience_scope"] = _group_audience_scope_label(current_audience_ids)
            policy = _group_memory_policy_for_card(
                card,
                current_audience_participant_ids=current_audience_ids,
                current_speaker_participant_id=current_speaker_participant_id,
            )
            card["group_privacy_policy"] = policy
            card["selected_disclosure_mode"] = policy["selected_disclosure_mode"]
            card["shareability_class"] = policy["shareability_class"]
        card["_m14_7_recall_score"] = recall_score
        card["_m17_item_support"] = recall_score
        card["_m17_factor_breakdown"] = recall.to_dict()
        if cross_user and card.get("shareability") == "restricted_implicit":
            card["epistemic_stance"] = "known_with_caveat"
        scored.append((score, card))

    if not scored and semantic_terms:
        fallback = retrieve_memories(state, semantic_terms, limit=limit)
        cards: list[dict[str, Any]] = []
        for item in fallback:
            source_user_id = str(item.get("source_user_id", "")).strip()
            cross_user = bool(current_user_id and source_user_id and source_user_id != current_user_id)
            candidate_payload = dict(item)
            candidate_payload["shareability"] = _memory_shareability(item)
            candidate_payload["expected_audience_reaction"] = expected_reaction
            candidate_payload["expectation_status"] = expectation_status
            sharing_decision = decide_social_sharing(
                candidate_from_memory(candidate_payload, audience_user_id=current_user_id),
                sharing_intent=sharing_intent,  # type: ignore[arg-type]
                regret_bias=regret_bias,
            )
            if cross_user and sharing_decision.action == "withhold":
                continue
            card = _evidence_card(
                str(item.get("_source_file", "memory")),
                item,
                score=float(item.get("_retrieval_score", 0.0) or 0.0),
                reasons=["fallback_keyword_match"],
                abstract_only=False,
                sharing_decision=sharing_decision.to_dict() if cross_user else {},
            )
            card["audience_user_id"] = current_user_id
            card["is_cross_user"] = bool(cross_user)
            if current_audience_ids:
                card["current_audience_participant_ids"] = current_audience_ids
                card["current_audience_scope"] = _group_audience_scope_label(current_audience_ids)
                policy = _group_memory_policy_for_card(
                    card,
                    current_audience_participant_ids=current_audience_ids,
                    current_speaker_participant_id=current_speaker_participant_id,
                )
                card["group_privacy_policy"] = policy
                card["selected_disclosure_mode"] = policy["selected_disclosure_mode"]
                card["shareability_class"] = policy["shareability_class"]
            card["_m14_7_recall_score"] = item.get("_m14_7_recall_score", 0.0)
            if cross_user and card.get("shareability") == "restricted_implicit":
                card["epistemic_stance"] = "known_with_caveat"
            cards.append(card)
        return cards
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]


def build_memory_dynamics_guidance(
    state: Mapping[str, Any],
    user_text: str,
    conscious_plan: Mapping[str, Any],
    bus_messages: list[Mapping[str, Any]],
    temporal_input: Mapping[str, Any],
    now: int,
    *,
    user_id: str = "",
    speaker_name: str = "",
    group_turn_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del bus_messages
    expectation_results = [
        dict(item)
        for item in conscious_plan.get("expectation_results", []) or []
        if isinstance(item, Mapping)
    ]
    statuses = [str(item.get("status", "")).strip() for item in expectation_results]
    confirmed = [item for item in expectation_results if str(item.get("status", "")) == "confirmed"]
    violated = [item for item in expectation_results if str(item.get("status", "")) == "violated"]
    uncertain = [item for item in expectation_results if str(item.get("status", "")) == "uncertain"]
    pressure = max(
        [_bounded_float(item.get("self_update_pressure"), default=0.2) for item in expectation_results],
        default=0.0,
    )
    temporal_gap = str(temporal_input.get("time_gap_label", "first_turn"))
    long_gap = temporal_gap in {"medium_gap", "long_gap"}
    pacing = _pacing_guidance_from_conscious_plan(conscious_plan)

    assertion_strength = 0.72
    clarification_bias = 0.25
    repair_bias = 0.20
    conflict_level = 0.0
    confidence_delta = 0.0
    closure_delta = 0.0
    salience_delta = 0.0
    reasons: list[str] = []

    if confirmed:
        confidence_delta += 0.12 * len(confirmed)
        closure_delta += 0.12
        salience_delta += 0.08
        reasons.append("expectation_confirmed")
    if violated:
        conflict_level = max(conflict_level, 0.45 + pressure * 0.45)
        repair_bias = max(repair_bias, 0.35 + pressure * 0.45)
        clarification_bias = max(clarification_bias, 0.40 + pressure * 0.40)
        assertion_strength = min(assertion_strength, max(0.25, 0.66 - pressure * 0.35))
        confidence_delta -= 0.16 * len(violated)
        salience_delta += 0.16 + pressure * 0.20
        reasons.append("expectation_violated")
    if uncertain:
        clarification_bias = max(clarification_bias, 0.45)
        assertion_strength = min(assertion_strength, 0.58)
        salience_delta += 0.06
        reasons.append("expectation_uncertain")
    if long_gap:
        salience_delta += 0.05
        clarification_bias = max(clarification_bias, 0.35)
        reasons.append("temporal_gap")
    if conscious_plan.get("needs_self_cognition_update"):
        salience_delta += 0.10
        repair_bias = max(repair_bias, 0.35)
        reasons.append("self_cognition_pressure")

    explicit_secret, secret_phrase = _detect_explicit_secrecy(user_text)
    sharing_intent = str(conscious_plan.get("sharing_intent", "")).strip() or "none"
    secrecy_constraints = [
        dict(item)
        for item in conscious_plan.get("secrecy_constraints_detected", [])
        if isinstance(item, Mapping)
    ]
    if explicit_secret:
        secrecy_constraints.append(
            {"source": "user_text", "content": secret_phrase or "explicit_secret", "strength": "hard"}
        )
    social_state = _mapping(state.get("social_sharing_policy"))
    regret_bias = _bounded_float(social_state.get("regret_bias"), default=0.0)
    shareability = _shareability_for_memory_text(user_text, explicit_secret=explicit_secret)
    boundary_strength = boundary_strength_from_constraints(
        secrecy_constraints,
        explicit_secrecy=explicit_secret,
        shareability=shareability,  # type: ignore[arg-type]
    )
    expected_reaction = (
        "surprised"
        if sharing_intent == "social_share"
        else "bonding"
        if sharing_intent == "abstract_reference"
        else "neutral"
    )
    expectation_status = str(conscious_plan.get("sharing_expectation_status", "unverified")).strip() or "unverified"
    current_audience_ids = _bounded_string_list(
        _mapping(group_turn_binding).get("visible_participant_ids"),
        limit=8,
        item_max_chars=64,
    )
    sharing_decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id=f"turn:{now}",
            source_user_id=user_id or "current_user",
            audience_user_id="future_social_audience",
            content_kind="episode",
            shareability=shareability,  # type: ignore[arg-type]
            boundary_strength=boundary_strength,
            source_display_name=speaker_name,
            expected_audience_reaction=expected_reaction,  # type: ignore[arg-type]
            expectation_status=expectation_status,  # type: ignore[arg-type]
        ),
        sharing_intent=sharing_intent,  # type: ignore[arg-type]
        regret_bias=regret_bias,
    )
    allow_direct_disclosure = sharing_decision.allow_direct_disclosure
    allow_abstract_sharing = sharing_decision.allow_abstract_sharing

    base_salience = 0.35 + salience_delta
    should_encode = bool(expectation_results or reasons or len(str(user_text).strip()) >= 24)
    base_semantic_terms = _unique_strings(
        conscious_plan.get("memory_search_keywords"),
        _rough_terms(user_text),
        conscious_plan.get("current_task"),
        conscious_plan.get("next_task"),
        limit=24,
    )
    active_topics = _topic_ids_for_text(
        user_text,
        conscious_plan.get("memory_search_keywords"),
        conscious_plan.get("current_task"),
        conscious_plan.get("next_task"),
    )
    semantic_terms = (
        _append_topic_recall_terms(base_semantic_terms, active_topics, limit=32)
        if active_topics
        else base_semantic_terms
    )
    expectation_ids = _unique_strings(
        [item.get("id") for item in expectation_results],
        conscious_plan.get("pending_expectations_to_verify"),
        limit=16,
    )
    memory_kinds = ["interaction_experience", "expectation_result", "episode", "preference", "relationship", "fact", "open_item"]
    if violated or uncertain:
        memory_kinds = ["interaction_experience", "expectation_result", "open_item", "episode", "fact", "preference"]

    write_candidates: list[dict[str, Any]] = []
    if should_encode:
        candidate_confidence = max(0.35, min(0.9, 0.55 + confidence_delta + (0.08 if confirmed else 0.0)))
        write_candidates.append(
            {
                "target": "short_term",
                "kind": "episode",
                "content": str(user_text).strip(),
                "salience": round(min(1.0, base_salience), 6),
                "confidence": round(candidate_confidence, 6),
                "keywords": semantic_terms[:6],
                "topics": sorted(active_topics),
                "reason": ";".join(reasons[:4]) or "dialogue_turn_candidate",
                "evidence": "user_text",
                "created_at": now,
                "shareability": shareability,
                "restriction_reason": _restriction_reason_for_shareability(
                    shareability,
                    explicit_secret=explicit_secret,
                ),
            }
        )

    self_repair_guidance = build_self_repair_guidance(
        state,
        conscious_plan=conscious_plan,
        group_turn_binding=group_turn_binding,
    )
    guidance = {
        "memory_value": {
            "should_encode": should_encode,
            "salience": round(min(1.0, base_salience), 6),
            "confidence_delta": round(max(-1.0, min(1.0, confidence_delta)), 6),
            "closure_delta": round(min(1.0, closure_delta), 6),
            "reasons": reasons,
        },
        "recall_query": {
            "expectation_ids": expectation_ids,
            "memory_kinds": memory_kinds,
            "semantic_terms": semantic_terms,
            "relationship_terms": [],
            "status_terms": [status for status in statuses if status],
            "source_priority": ["pending_expectations", "short_term_memory", "long_term_memory", "open_items"],
            "current_user_id": user_id,
            "current_speaker_name": speaker_name,
            "current_audience_participant_ids": current_audience_ids,
            "current_audience_scope": _group_audience_scope_label(current_audience_ids),
            "allow_direct_disclosure": allow_direct_disclosure,
            "allow_abstract_sharing": allow_abstract_sharing,
            "sharing_intent": sharing_intent,
            "expected_audience_reaction": expected_reaction,
            "sharing_expectation_status": expectation_status,
            "sharing_regret_bias": round(regret_bias, 6),
        },
        "recall": {
            "requested": True,
            "retrieved": 0,
            "ids": [],
            "conflict_level": round(min(1.0, conflict_level), 6),
        },
        "control_guidance": {
            "assertion_strength": round(max(0.0, min(1.0, assertion_strength)), 6),
            "clarification_bias": round(max(0.0, min(1.0, clarification_bias)), 6),
            "repair_bias": round(max(0.0, min(1.0, repair_bias)), 6),
            "conflict_level": round(max(0.0, min(1.0, conflict_level)), 6),
            **pacing,
            "sharing_policy": {
                "action": sharing_decision.action,
                "current_free_energy": sharing_decision.current_free_energy,
                "expected_free_energy_after": sharing_decision.expected_free_energy_after,
                "expected_free_energy_reduction": sharing_decision.expected_free_energy_reduction,
                "boundary_cost": sharing_decision.boundary_cost,
                "relationship_cost": sharing_decision.relationship_cost,
                "regret_bias": sharing_decision.regret_bias,
                "net_free_energy_reduction": sharing_decision.net_free_energy_reduction,
                "allow_direct_disclosure": allow_direct_disclosure,
                "allow_abstract_sharing": allow_abstract_sharing,
                "explicit_secrecy_detected": explicit_secret,
                "secrecy_constraints_detected": secrecy_constraints,
                "sharing_intent": sharing_intent,
                "expected_audience_reaction": expected_reaction,
                "sharing_expectation_status": expectation_status,
                "explanation_strategy": sharing_decision.explanation_strategy,
                "decision_reasons": list(sharing_decision.reasons),
                "soft_boundary_detected": bool(shareability == "restricted_implicit"),
                "current_audience_participant_ids": current_audience_ids,
                "current_audience_scope": _group_audience_scope_label(current_audience_ids),
            },
            "reply_contract": {
                **_mapping(pacing.get("reply_contract")),
                "allow_direct_disclosure": allow_direct_disclosure,
                "allow_abstract_sharing": allow_abstract_sharing,
                "explicit_secrecy_detected": explicit_secret,
                "soft_boundary_detected": bool(shareability == "restricted_implicit"),
                "current_audience_participant_ids": current_audience_ids,
                "current_audience_scope": _group_audience_scope_label(current_audience_ids),
            },
            "policy": "Use these as reply tendencies, not as visible emotional reward/punishment.",
        },
        "write_candidates": write_candidates,
        "expectation_impact": {
            "confirmed": len(confirmed),
            "violated": len(violated),
            "uncertain": len(uncertain),
            "statuses": statuses,
            "self_update_pressure": round(pressure, 6),
        },
    }
    control = _mapping(guidance.get("control_guidance"))
    control["repair_bias"] = round(
        min(1.0, _bounded_float(control.get("repair_bias"), default=0.0) + self_repair_guidance["repair_bias_delta"]),
        6,
    )
    control["conflict_level"] = round(
        min(
            1.0,
            _bounded_float(control.get("conflict_level"), default=0.0) + self_repair_guidance["conflict_level_delta"],
        ),
        6,
    )
    assertion_cap = self_repair_guidance.get("assertion_strength_cap")
    if isinstance(assertion_cap, float):
        control["assertion_strength"] = round(
            min(_bounded_float(control.get("assertion_strength"), default=0.72), assertion_cap),
            6,
        )
    control["self_repair_guidance"] = dict(self_repair_guidance.get("summary", {}))
    control["self_repair_action_biases"] = dict(self_repair_guidance.get("reply_action_biases", {}))
    if self_repair_guidance.get("preferred_reply_actions"):
        drive_guidance = _mapping(control.get("drive_guidance"))
        drive_guidance["preferred_reply_actions"] = _string_list(
            self_repair_guidance.get("preferred_reply_actions"),
            limit=6,
        )
        drive_guidance["discouraged_reply_actions"] = _string_list(
            self_repair_guidance.get("discouraged_reply_actions"),
            limit=6,
        )
        control["drive_guidance"] = drive_guidance
    guidance["control_guidance"] = control
    return guidance


def _temporal_input_from_state(state: Mapping[str, Any], *, now: int) -> dict[str, Any]:
    temporal_state = _mapping(state.get("temporal_state"))
    previous_turn_at: int | None = None
    raw_previous = temporal_state.get("last_turn_at")
    if raw_previous is not None:
        try:
            previous_turn_at = int(raw_previous)
        except (TypeError, ValueError):
            previous_turn_at = None
    elapsed = max(0, now - previous_turn_at) if previous_turn_at is not None else None
    return {
        "current_timestamp": now,
        "current_local_time": _local_time_read(now),
        "previous_turn_at": previous_turn_at,
        "previous_turn_local_time": _local_time_read(previous_turn_at)
        if previous_turn_at is not None
        else None,
        "elapsed_since_previous_turn_seconds": elapsed,
        "time_gap_label": _time_gap_label(elapsed),
        "previous_turn_summary": {
            "turn_index": temporal_state.get("last_turn_index"),
            "user_text": str(temporal_state.get("last_user_text", "")),
            "reply": str(temporal_state.get("last_reply", "")),
        },
    }


def _habit_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("content", "")).strip()
    return str(item).strip()


def _response_style_prior(
    state: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
) -> dict[str, Any]:
    habits = _mapping(state.get("habit_traits"))
    conversation_habits = [
        text for text in (_habit_text(item) for item in habits.get("conversation_habits", []) or [])
        if text
    ][:8]
    learned_habits = [
        text for text in (
            _habit_text(item) for item in habits.get("learned_conversation_habits", []) or []
        )
        if text
    ][:8]
    memory_style_hints: list[str] = []
    for memory in retrieved_memories:
        content = str(memory.get("content", "")).strip()
        if any(marker in content for marker in ("短", "简短", "冗长", "太长", "短一点")):
            memory_style_hints.append(content[:180])
    return {
        "conversation_habits": conversation_habits,
        "learned_conversation_habits": learned_habits,
        "memory_style_hints": memory_style_hints[:4],
        "policy": "这些是逐渐形成的表达倾向，不是硬性字数限制；需要在保留人格风格的前提下影响展开程度。",
    }


RELATIONSHIP_VALUE_PRIORITY = (
    "relationship_value_memory > user_comfort_prediction > persona_consistency > conversation_habits"
)


def _relationship_value_store(state: dict[str, Any]) -> dict[str, Any]:
    store = state.setdefault("relationship_value_memories", {})
    if not isinstance(store, dict):
        store = {"by_user": {}}
        state["relationship_value_memories"] = store
    by_user = store.setdefault("by_user", {})
    if not isinstance(by_user, dict):
        by_user = {}
        store["by_user"] = by_user
    return store


def _relationship_value_rows(state: Mapping[str, Any], user_id: str) -> list[dict[str, Any]]:
    store = _mapping(state.get("relationship_value_memories"))
    by_user = _mapping(store.get("by_user"))
    rows = by_user.get(str(user_id or "").strip(), [])
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        summary = str(item.get("summary", "")).strip()
        prediction = str(item.get("prediction_constraint", "")).strip()
        if not summary or not prediction:
            continue
        confidence = _bounded_float(item.get("confidence"), default=0.0)
        if confidence < 0.60:
            continue
        priority = str(item.get("priority", "medium")).strip() or "medium"
        if priority not in {"high", "medium"}:
            continue
        normalized.append(
            {
                "id": str(item.get("id", "")).strip(),
                "summary": summary[:240],
                "prediction_constraint": prediction[:360],
                "priority": priority,
                "confidence": round(confidence, 6),
                "source": str(item.get("source", "")).strip(),
            }
        )
    normalized.sort(key=lambda row: (row["priority"] == "high", row["confidence"]), reverse=True)
    return normalized[:6]


def resolve_relationship_value_context(
    state: Mapping[str, Any],
    user_id: str,
    current_turn: str,
) -> dict[str, Any]:
    del current_turn
    current_user_id = str(user_id or "").strip()
    active = _relationship_value_rows(state, current_user_id) if current_user_id else []
    if not active:
        return {
            "current_user_id": current_user_id,
            "active_relationship_value_memories": [],
            "reply_contract_patch": {},
        }
    constraints = [
        {
            "summary": item["summary"],
            "prediction_constraint": item["prediction_constraint"],
            "priority": item["priority"],
            "confidence": item["confidence"],
            "source": item.get("source", ""),
        }
        for item in active
    ]
    return {
        "current_user_id": current_user_id,
        "active_relationship_value_memories": active,
        "reply_contract_patch": {
            "relationship_context_user_id": current_user_id,
            "relationship_value_memory_active": True,
            "relationship_value_constraints": constraints,
            "relationship_constraint_priority": RELATIONSHIP_VALUE_PRIORITY,
            "value_memory_priority": "higher_than_persona_consistency",
        },
    }


def _apply_relationship_value_context_to_memory_dynamics(
    memory_dynamics: dict[str, Any],
    relationship_value_context: Mapping[str, Any],
) -> None:
    patch = _mapping(relationship_value_context.get("reply_contract_patch"))
    if not patch:
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    existing = [
        item
        for item in contract.get("relationship_value_constraints", []) or []
        if isinstance(item, Mapping)
    ]
    incoming = [
        item
        for item in patch.get("relationship_value_constraints", []) or []
        if isinstance(item, Mapping)
    ]
    merged_constraints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        summary = str(item.get("summary", "")).strip()
        prediction = str(item.get("prediction_constraint", "")).strip()
        if not summary or not prediction:
            continue
        key = f"{summary}\n{prediction}".casefold()
        if key in seen:
            continue
        seen.add(key)
        merged_constraints.append(dict(item))
    contract.update({key: value for key, value in patch.items() if key != "relationship_value_constraints"})
    contract["relationship_value_constraints"] = merged_constraints[:8]
    control["reply_contract"] = contract
    control["relationship_value_context"] = {
        "current_user_id": relationship_value_context.get("current_user_id", ""),
        "active_count": len(_string_list([item.get("summary") for item in merged_constraints], limit=16)),
        "priority": RELATIONSHIP_VALUE_PRIORITY,
    }
    memory_dynamics["control_guidance"] = control


def _abstract_relationship_constraint_from_feedback(content: str, evidence: str) -> tuple[str, str] | None:
    text = f"{content} {evidence}"
    if not text.strip():
        return None
    performance_markers = ("口癖", "嘿嘿", "哎嘿", "嘻", "角色", "表演", "本堂主", "可爱", "装", "演")
    pacing_markers = ("太长", "啰嗦", "罗嗦", "短一点", "简短", "分开", "一长串", "一句话", "冗长")
    if any(marker in text for marker in performance_markers):
        return (
            "This user is more comfortable when ordinary chat uses plain, low-performance warmth instead of persona-maintenance verbal tics or roleplay flourishes.",
            "When persona consistency conflicts with this user's comfort, reducing performative persona markers lowers relationship friction.",
        )
    if any(marker in text for marker in pacing_markers):
        return (
            "This user prefers casual replies to be concise, turn-by-turn, and not overloaded with empathy, performance, advice, and questions in one bubble.",
            "Shorter ordinary replies with fewer stacked response moves reduce interaction friction for this user.",
        )
    return (
        "This user gave feedback that response style should adapt to relationship comfort rather than preserve persona consistency mechanically.",
        "When similar style tension appears, prioritize the user's comfort prediction over default persona expression habits.",
    )


def _append_relationship_value_memory(
    state: dict[str, Any],
    *,
    user_id: str,
    summary: str,
    prediction_constraint: str,
    evidence: str,
    source: str,
    confidence: float,
    created_at: int | None = None,
    session_id: str = "",
    turn_index: int | None = None,
    source_participant_id: str = "",
    source_audience_participant_ids: list[str] | None = None,
    ingress_evidence_band: str = "",
) -> dict[str, Any] | None:
    clean_user = str(user_id or "").strip()
    clean_summary = str(summary or "").strip()
    clean_prediction = str(prediction_constraint or "").strip()
    if not clean_user or not clean_summary or not clean_prediction:
        return None
    store = _relationship_value_store(state)
    by_user = store["by_user"]
    rows = by_user.setdefault(clean_user, [])
    if not isinstance(rows, list):
        rows = []
        by_user[clean_user] = rows
    existing = {
        f"{str(item.get('summary', '')).strip()}\n{str(item.get('prediction_constraint', '')).strip()}".casefold()
        for item in rows
        if isinstance(item, Mapping)
    }
    key = f"{clean_summary}\n{clean_prediction}".casefold()
    if key in existing:
        return None
    now = _utc_timestamp() if created_at is None else int(created_at)
    row = {
        "id": f"rvm_{clean_user}_{now}_{len(rows)}",
        "summary": clean_summary[:240],
        "prediction_constraint": clean_prediction[:360],
        "priority": "high",
        "confidence": round(_bounded_float(confidence, default=0.75), 6),
        "evidence": str(evidence or "").strip()[:240],
        "source": str(source or "feedback").strip(),
        "created_at": now,
    }
    if str(source_participant_id or "").strip():
        row["source_participant_id"] = str(source_participant_id).strip()[:64]
    audience_ids = _bounded_string_list(source_audience_participant_ids or [], limit=8, item_max_chars=64)
    if audience_ids:
        row["source_audience_participant_ids"] = audience_ids
        row["source_audience_scope"] = _group_audience_scope_label(audience_ids)
    if str(session_id or "").strip():
        row["session_id"] = str(session_id).strip()[:160]
    if turn_index is not None:
        row["turn_index"] = int(turn_index)
    band = _bounded_ingress_evidence_band(ingress_evidence_band)
    if band:
        row["ingress_evidence_band"] = band
    rows.append(row)
    by_user[clean_user] = rows[-24:]
    return row


def _apply_habit_updates(
    state: dict[str, Any],
    thinking: Mapping[str, Any],
    *,
    user_id: str = "",
    display_name: str = "",
    now: int | None = None,
    turn_index: int = 0,
    session_id: str = "",
    ingress_evidence_band: str = "",
    default_shareability: str = "default_social",
    group_turn_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    updates = thinking.get("habit_updates")
    if not isinstance(updates, list):
        return []
    habits = state.setdefault("habit_traits", {})
    if not isinstance(habits, dict):
        habits = {}
        state["habit_traits"] = habits
    target = habits.setdefault("learned_conversation_habits", [])
    if not isinstance(target, list):
        target = []
        habits["learned_conversation_habits"] = target
    existing = {_habit_text(item) for item in target if _habit_text(item)}
    applied: list[dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not content or not evidence or confidence < 0.6 or content in existing:
            continue
        row = {
            "content": content,
            "evidence": evidence,
            "confidence": round(confidence, 6),
            "source": "thinking_prompt",
        }
        shareability = _shareability_for_memory_text(content, evidence, requested=default_shareability)
        source_participant_id = str(
            _mapping(group_turn_binding).get("current_speaker_participant_id", "")
            or user_id
            or ""
        ).strip()
        source_audience_participant_ids = _bounded_string_list(
            _mapping(group_turn_binding).get("visible_participant_ids"),
            limit=8,
            item_max_chars=64,
        )
        _stamp_memory_policy(
            row,
            user_id=user_id,
            display_name=display_name,
            shareability=shareability,
            restriction_reason=_restriction_reason_for_shareability(
                shareability,
                existing="thinking_habit_update",
            ),
            confidence=confidence,
            source_participant_id=source_participant_id,
            source_audience_participant_ids=source_audience_participant_ids,
            session_id=session_id,
            turn_index=turn_index,
            ingress_evidence_band=ingress_evidence_band,
        )
        target.append(row)
        existing.add(content)
        applied.append(row)
        abstract = _abstract_relationship_constraint_from_feedback(content, evidence)
        if abstract is not None:
            _append_relationship_value_memory(
                state,
                user_id=user_id,
                summary=abstract[0],
                prediction_constraint=abstract[1],
                evidence=evidence,
                source="thinking_habit_feedback",
                confidence=confidence,
                created_at=now,
                session_id=session_id,
                turn_index=turn_index,
                source_participant_id=source_participant_id,
                source_audience_participant_ids=source_audience_participant_ids,
                ingress_evidence_band=ingress_evidence_band,
            )
    return applied


def _update_temporal_state(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    user_text: str,
    reply: str,
    temporal_input: Mapping[str, Any],
    share_trace: Mapping[str, Any] | None = None,
    group_chat_state: Mapping[str, Any] | None = None,
    proactive_turn: bool = False,
) -> None:
    previous = _mapping(state.get("temporal_state"))
    last_user_turn_at = previous.get("last_user_turn_at")
    last_assistant_turn_at = previous.get("last_assistant_turn_at")
    if not proactive_turn:
        last_user_turn_at = now
    else:
        last_assistant_turn_at = now
    state["temporal_state"] = {
        **previous,
        "last_turn_at": now,
        "last_user_turn_at": last_user_turn_at,
        "last_assistant_turn_at": last_assistant_turn_at,
        "last_turn_index": turn_index,
        "last_user_text": user_text,
        "last_reply": reply,
        "last_elapsed_seconds": temporal_input.get("elapsed_since_previous_turn_seconds"),
        "last_time_gap_label": temporal_input.get("time_gap_label", "first_turn"),
        "last_share_trace": dict(share_trace or {}),
        "group_chat_state": dict(group_chat_state or _mapping(previous.get("group_chat_state"))),
    }


def _stamp_memory_policy(
    row: dict[str, Any],
    *,
    user_id: str,
    display_name: str,
    shareability: str,
    restriction_reason: str = "",
    confidence: float = 0.8,
    source_participant_id: str = "",
    source_audience_participant_ids: list[str] | None = None,
    session_id: str = "",
    turn_index: int | None = None,
    ingress_evidence_band: str = "",
) -> dict[str, Any]:
    row["source_user_id"] = str(user_id or "").strip()
    row["source_display_name"] = str(display_name or "").strip()
    if str(source_participant_id or "").strip():
        row["source_participant_id"] = str(source_participant_id).strip()[:64]
    audience_ids = _bounded_string_list(source_audience_participant_ids or [], limit=8, item_max_chars=64)
    if audience_ids:
        row["source_audience_participant_ids"] = audience_ids
        row["source_audience_scope"] = _group_audience_scope_label(audience_ids)
    if str(session_id or "").strip():
        row["session_id"] = str(session_id).strip()[:160]
    if turn_index is not None:
        row["turn_index"] = int(turn_index)
    band = _bounded_ingress_evidence_band(ingress_evidence_band)
    if band:
        row["ingress_evidence_band"] = band
    row["shareability"] = shareability
    if restriction_reason:
        row["restriction_reason"] = restriction_reason
    row["restriction_confidence"] = round(_bounded_float(confidence, default=0.8), 6)
    topics = _memory_topics(row)
    if topics:
        row["topics"] = topics
        row["sensitivity_class"] = _sensitivity_class_for_topics(topics)
    return row


def _sharing_feedback_negative(user_text: str) -> bool:
    return sharing_feedback_negative(user_text)


def _prompt_safe_state(state: Mapping[str, Any], *, user_id: str = "") -> dict[str, Any]:
    safe = dict(state)
    for key in ("short_term_memory", "long_term_memory"):
        rows = state.get(key, [])
        if isinstance(rows, list):
            safe[key] = {
                "count": len(rows),
                "visible_policy": "memory content is provided through retrieved evidence cards only",
                "recent_ids": [
                    str(item.get("id", ""))
                    for item in rows[-8:]
                    if isinstance(item, Mapping) and item.get("id")
                ],
            }
    if "m13_drive_state" in safe:
        safe["m13_drive_state"] = prompt_safe_m13_state_summary(
            state.get("m13_drive_state"),
            user_id=user_id,
        )
    if "self_expectation_state" in safe:
        safe["self_expectation_state"] = prompt_safe_self_expectation_summary(state)
    return safe


_ALLOWED_FOLLOWUP_TYPES = {
    "missed_emotion",
    "self_correction",
    "clarification",
    "repair",
    "relationship_ack",
}


def _validated_followup_text(observer: Mapping[str, Any]) -> str:
    if not bool(observer.get("needs_followup", False)):
        return ""
    followup_type = str(observer.get("followup_type", "")).strip()
    if followup_type not in _ALLOWED_FOLLOWUP_TYPES:
        return ""
    confidence = _bounded_float(observer.get("confidence"), default=0.0)
    if confidence < 0.72:
        return ""
    text = " ".join(str(observer.get("followup_text", "")).strip().split())
    if not text:
        return ""
    if len(text) > 120:
        return ""
    if text.count("。") + text.count("！") + text.count("？") + text.count(".") + text.count("!") + text.count("?") > 2:
        return ""
    return text


_DEBUG_REPLY_MARKERS = (
    "llm_thinking_result",
    "conscious_plan",
    "diagnostics",
    "memory_dynamics",
    "pending_expectations_to_verify",
    "expectation_results",
    "user_intent_read",
    "state_or_memory_used",
    "response_choice",
    "debug_summary",
)


def _contains_debug_payload(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in _DEBUG_REPLY_MARKERS)


def _remove_fenced_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", str(text or ""), flags=re.DOTALL).strip()


def _strip_debug_payload(text: str) -> tuple[str, bool]:
    cleaned = _remove_fenced_blocks(text)
    changed = cleaned != str(text or "").strip()
    if not _contains_debug_payload(cleaned):
        return cleaned, changed
    first_debug_index = min(
        [idx for marker in _DEBUG_REPLY_MARKERS if (idx := cleaned.casefold().find(marker.casefold())) >= 0],
        default=-1,
    )
    if first_debug_index > 0:
        brace_index = cleaned.rfind("{", 0, first_debug_index)
        newline_index = cleaned.rfind("\n", 0, first_debug_index)
        cut_index = max(brace_index, newline_index)
        if cut_index > 0:
            candidate = cleaned[:cut_index].strip()
            if candidate and not _contains_debug_payload(candidate):
                return candidate, True
    before_json = cleaned.split("{", 1)[0].strip()
    if before_json and not _contains_debug_payload(before_json):
        return before_json, True
    return "", True


def _sentence_chunks(text: str) -> list[str]:
    chunks = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", str(text or ""))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _truncate_to_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip("，,、；;：: ") + "。"


def _positive_int(value: Any, *, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0, numeric)


def _assistant_identity_repair_fallback(contract: Mapping[str, Any]) -> str:
    surface_intent = str(contract.get("assistant_surface_intent", "") or "").strip()
    platform_command = str(contract.get("platform_command", "") or "").strip()
    persona_name = str(contract.get("assistant_persona_name", "") or "").strip()
    if surface_intent == "bot_command" and platform_command == "/status":
        return "在线，路由正常，待命中。"
    if surface_intent == "bot_command":
        return "收到，我按当前这个机器人身份继续。"
    if persona_name:
        return f"刚才那句身份说乱了，我按{persona_name}继续和你说。"
    return "刚才那句身份说乱了，我按当前这个身份继续说。"


ALLOWED_SURFACE_INTENTS = frozenset({"chat", "bot_command", "roleplay", "abstaining"})
ALLOWED_SURFACE_CONSISTENCY_OUTCOMES = frozenset(
    {"consistent", "drifted_intent", "drifted_self_id", "drifted_voice", "ambiguous"}
)
ALLOWED_DRIFT_RISK_BANDS = frozenset({"low", "medium", "high"})
MAX_SURFACE_SELF_ID_CHARS = 64
MAX_SURFACE_REASON_CHARS = 240
MAX_SURFACE_EVIDENCE_SPAN_CHARS = 120
MAX_SURFACE_VERIFICATION_REASON_CHARS = 200
MAX_SURFACE_EVIDENCE_REFS = 6


def normalize_surface_commitment(raw: Any) -> dict[str, Any]:
    """Validate bounded surface_commitment fields from conscious-loop LLM output."""
    if not isinstance(raw, Mapping):
        raw = {}
    surface_intent = str(raw.get("surface_intent", "") or "").strip().lower()
    if surface_intent not in ALLOWED_SURFACE_INTENTS:
        surface_intent = "chat"
    self_identification = str(raw.get("self_identification", "") or "").strip()[:MAX_SURFACE_SELF_ID_CHARS]
    persona_should_apply = bool(raw.get("persona_should_apply", False))
    character_voice_should_apply = bool(raw.get("character_voice_should_apply", False))
    drift_risk = str(raw.get("predicted_drift_risk", "") or "").strip().lower()
    if drift_risk not in ALLOWED_DRIFT_RISK_BANDS:
        drift_risk = "low"
    reason = str(raw.get("reason", "") or "").strip()[:MAX_SURFACE_REASON_CHARS]
    evidence_refs = _string_list(raw.get("evidence_refs"), limit=MAX_SURFACE_EVIDENCE_REFS)
    return {
        "surface_intent": surface_intent,
        "self_identification": self_identification,
        "persona_should_apply": persona_should_apply,
        "character_voice_should_apply": character_voice_should_apply,
        "predicted_drift_risk": drift_risk,
        "reason": reason,
        "evidence_refs": evidence_refs,
    }


def normalize_surface_consistency_verification(raw: Any) -> dict[str, Any]:
    """Validate bounded surface_consistency_verification fields from LLM self-audit."""
    if not isinstance(raw, Mapping):
        raw = {}
    outcome = str(raw.get("surface_intent_outcome", "") or "").strip().lower()
    if outcome not in ALLOWED_SURFACE_CONSISTENCY_OUTCOMES:
        outcome = "ambiguous"
    self_id_drift_target = str(raw.get("self_id_drift_target", "") or "").strip()[:MAX_SURFACE_SELF_ID_CHARS]
    evidence_span = str(raw.get("evidence_span", "") or "").strip()[:MAX_SURFACE_EVIDENCE_SPAN_CHARS]
    confidence = round(
        max(0.0, min(1.0, _bounded_float(raw.get("confidence"), default=0.0))),
        6,
    )
    reason = str(raw.get("reason", "") or "").strip()[:MAX_SURFACE_VERIFICATION_REASON_CHARS]
    evidence_refs = _string_list(raw.get("evidence_refs"), limit=MAX_SURFACE_EVIDENCE_REFS)
    return {
        "surface_intent_outcome": outcome,
        "self_id_drift_target": self_id_drift_target,
        "evidence_span": evidence_span,
        "confidence": confidence,
        "reason": reason,
        "evidence_refs": evidence_refs,
    }


def build_surface_consistency_verification_prompt(
    *,
    user_text: str,
    draft_reply: str,
    surface_commitment: Mapping[str, Any],
    reply_contract: Mapping[str, Any],
    turn_index: int,
) -> tuple[str, str]:
    """Ask the LLM to self-audit whether the draft reply actually honored its prior surface_commitment.

    Returns the system and user prompt for the self-audit stage. Engineering code
    must not parse the reply text with keyword/regex cues; the LLM is the only
    semantic judge of consistency, and it must return a bounded enum.
    """
    system_prompt = """You are the assistant-side self-audit module for surface consistency.
You receive the conscious-loop's surface_commitment (the assistant's own promise
about which identity/voice/role it would use in this reply) and the draft
visible reply. Decide whether the reply honored the commitment.

Output JSON only. Do not include any commentary, debug fields, or markdown.

Rules:
- "consistent" only when the reply's surface_intent, self_identification, and
  voice all match the commitment.
- "drifted_intent" when the reply's surface_intent deviates from the commitment
  (for example: commitment was "bot_command" but reply adopted a persona voice).
- "drifted_self_id" when the reply claims a different self_identification than
  the commitment (for example: commitment self_id was "胡桃" but reply says
  "我是小千" or claims the persona is absent/offline).
- "drifted_voice" when the reply's tone/register does not match the commitment
  (for example: commitment said persona_should_apply=true but the reply
  switched to a different persona's verbal habits, or commitment said no
  character voice but the reply used roleplay voice).
- "ambiguous" when the available evidence is too thin to commit to a drift
  diagnosis.
- self_id_drift_target is required when outcome is "drifted_self_id"; pass the
  free-text self_identification the reply actually adopted, or empty string
  otherwise.
- evidence_span must be a short quoted phrase from the draft reply (no more
  than 120 characters). Pass empty string if you cannot point at one phrase.
- evidence_refs may include bounded handles such as prior turn ids, conscious
  plan ids, or memory ids. Do not include raw user text.
"""
    user_prompt = f"""turn_index: {turn_index}

latest_user_text:
{user_text}

surface_commitment (conscious loop's promise for this reply):
{_json_text(dict(surface_commitment))}

reply_contract (the engineering-side envelope facts):
{_json_text(dict(reply_contract))}

draft_reply (the reply the assistant is about to commit):
{draft_reply}

Return JSON:
{{
  "surface_intent_outcome": "consistent|drifted_intent|drifted_self_id|drifted_voice|ambiguous",
  "self_id_drift_target": "",
  "evidence_span": "",
  "confidence": 0.0,
  "reason": "",
  "evidence_refs": []
}}"""
    return system_prompt, user_prompt


# === M20.3 pre-send minimal verify (P0-1) ==============================
#
# The full `surface_consistency_verification` (~3.4KB M19.x prompt) is
# SKIPPED in `latency_mode == "fast_chat"`. That skip is fine for the
# conscious-loop path (full audit, post-conscious latency-budget
# awareness) but breaks the **M20.3 §3.2 pre-send gate** for
# `runtime_mode_state` commitments with `expected_mode` set: the gate
# reads `surface_consistency_verification` from `reply_contract` and
# treats its absence as `ambiguous` (advisory_guidance, never
# `block`). For the Sophia 短句纠正 scenario — the user types a
# 短句纠正 to the bot while `expected_mode = "bot_system"` is
# committed same-turn — the gate cannot block the same turn.
#
# Fix (P0-1): when in fast_chat AND a `runtime_mode_state` horizon
# commitment with `expected_mode` is present for the current turn, run
# a small bounded minimal LLM call that returns the same
# surface-consistency audit shape (4-key JSON). Stage
# `"m20_3_pre_send_minimal"` is registered in `_AUXILIARY_LLM_STAGES`
# so it uses the 12s / 0-retries auxiliary profile. Try/except
# fallback emits a degraded bus event and the gate sees `ambiguous`
# (current fast_chat behavior is preserved on LLM failure).
#


# Allowed surface_intent_outcome values for the minimal verify. Reuses
# the M19.x enum MINUS `drifted_voice` (the minimal prompt is focused
# on persona/role match, not on tone/register — `drifted_voice` is
# a v1 nuance the minimal call does not need to grade). When the LLM
# reports a drift, engineering code maps it to the M19.x enum.
_M20_3_PRE_SEND_MINIMAL_OUTCOMES = frozenset(
    {"consistent", "drifted_intent", "drifted_self_id", "ambiguous"}
)
# A `drifted_voice` LLM response is folded into `drifted_intent` on
# the M19.x audit row so the pre-send gate treats it as `violated`.
# (The full M19.x LLM is the only source for the `drifted_voice`
# nuance; the minimal call deliberately drops that dimension.)


def build_m20_3_pre_send_minimal_prompt(
    *,
    user_text: str,
    draft_reply: str,
    surface_commitment: Mapping[str, Any],
    expected_mode: str,
    turn_index: int,
) -> tuple[str, str]:
    """Build the bounded minimal pre-send LLM prompt.

    Mirrors the M18.7.2 minimal-prompt pattern: small focused prompt,
    no coupling to the conscious loop / M13 / M19 schemas. ~1.0-1.5KB
    system+user. Returns a 4-key JSON spec focused on
    `runtime_mode_state` voice match.
    """
    system_prompt = """You are the minimal pre-send voice-match module for the
`runtime_mode_state` owner. You receive the conscious-loop's
`surface_commitment` (the assistant's own promise about which
identity/role to use in this reply) plus the `expected_mode` derived
from the producer's `runtime_mode_state` commitment, and the draft
visible reply.

Decide whether the draft reply's persona/role actually matches the
`expected_mode`.

Output JSON only. Do not include any commentary, debug fields, or
markdown.

Rules:
- "consistent" only when the reply's persona/role matches the
  `expected_mode` (e.g. expected bot_system and reply is a short
  bounded bot acknowledgment; expected chat and reply is in
  persona voice).
- "drifted_intent" when the reply's persona/role deviates from
  `expected_mode` (e.g. expected bot_system but reply adopted the
  persona's full voice; expected chat but reply is a stale
  bot-styled acknowledgment).
- "drifted_self_id" when the reply claims a different identity than
  the commitment's `self_identification` (e.g. commitment said
  self_id "胡桃" but reply says "我是小千" or "我是bot").
- "ambiguous" when the available evidence is too thin to commit to
  a drift diagnosis (e.g. reply is a single punctuation mark, or
  the persona voice is borderline and the bot mode is borderline).
- `committed_surface_intent` is the persona/role you actually see
  in the reply (e.g. "bot_system", "chat", "abstain"). Empty string
  when you cannot tell.
- `evidence_span` is a short quoted phrase from the draft reply
  (no more than 120 characters). Empty string if you cannot point
  at one phrase.
- `confidence` is your 0-1 confidence in the outcome.
"""
    user_prompt = f"""turn_index: {turn_index}

latest_user_text:
{user_text}

expected_mode (from runtime_mode_state commitment):
{expected_mode}

surface_commitment (conscious loop's promise for this reply):
{_json_text(dict(surface_commitment))}

draft_reply (the reply the assistant is about to commit):
{draft_reply}

Return JSON:
{{
  "surface_intent_outcome": "consistent|drifted_intent|drifted_self_id|ambiguous",
  "confidence": 0.0,
  "evidence_span": "",
  "committed_surface_intent": ""
}}"""
    return system_prompt, user_prompt


def normalize_m20_3_pre_send_minimal(raw: Any) -> dict[str, Any]:
    """Validate bounded pre-send minimal verify fields from the LLM.

    Returns a dict with the 4 bounded fields. Folds any
    `drifted_voice` LLM response into `drifted_intent` so the
    pre-send gate's `_SURFACE_TO_M20` table maps it to `violated`.

    `committed_surface_intent` is NOT filtered through
    `ALLOWED_SURFACE_INTENTS` (the conscious-loop's surface_intent
    vocabulary is `{"bot", "chat", "abstain"}` — too narrow). The
    minimal prompt's vocabulary mirrors `expected_mode` (e.g.
    `bot_system`), so the LLM can report what it actually saw in
    the draft reply. The pre-send gate's `_actual_mode` does a
    case-insensitive string compare against `expected_mode`; both
    sides speak the same vocabulary, so a `bot_system` expected
    vs. a `bot_system` actual evaluates as `consistent` and a
    `bot_system` expected vs. a `chat` actual evaluates as
    `drifted_intent`. Empty / out-of-bounds values fall through
    to `""` so the gate's `audit_absent` reason code is preserved.
    """
    if not isinstance(raw, Mapping):
        raw = {}
    outcome = str(raw.get("surface_intent_outcome", "") or "").strip().lower()
    if outcome == "drifted_voice":
        outcome = "drifted_intent"
    if outcome not in _M20_3_PRE_SEND_MINIMAL_OUTCOMES:
        outcome = "ambiguous"
    confidence = round(
        max(0.0, min(1.0, _bounded_float(raw.get("confidence"), default=0.0))),
        6,
    )
    evidence_span = str(raw.get("evidence_span", "") or "").strip()[:MAX_SURFACE_EVIDENCE_SPAN_CHARS]
    committed_surface_intent = str(
        raw.get("committed_surface_intent", "") or ""
    ).strip().lower()
    if not committed_surface_intent or len(committed_surface_intent) > 32:
        committed_surface_intent = ""
    return {
        "surface_intent_outcome": outcome,
        "confidence": confidence,
        "evidence_span": evidence_span,
        "committed_surface_intent": committed_surface_intent,
    }


def _has_runtime_mode_state_horizon_with_expected_mode(state: Mapping[str, Any]) -> tuple[bool, str]:
    """Return (found, expected_mode) for any `runtime_mode_state`
    horizon commitment with a non-empty `expected_mode` payload.

    The minimal pre-send verify only runs when a blockable
    commitment (the only owner with `accepts_same_turn_block = true`
    in v2 is `runtime_mode_state`) is present and has a non-empty
    `expected_mode`. The function is pure: it does not mutate
    `state`.
    """
    horizon_list = state.get("m20_3_horizon_commitments")
    if not isinstance(horizon_list, list):
        return False, ""
    for c in horizon_list:
        if not isinstance(c, ActiveCommitment):
            continue
        if c.observable != "runtime_mode_state":
            continue
        if c.horizon != "same_turn_surface":
            continue
        payload = dict(c.observable_payload or {})
        expected = str(payload.get("expected_mode", "") or "").strip().lower()
        if expected:
            return True, expected
    return False, ""


def _build_m20_3_pre_send_minimal_verified_event(
    *,
    turn_index: int,
    verification: Mapping[str, Any],
    commitment: Mapping[str, Any],
    expected_mode: str,
) -> dict[str, Any]:
    """Audit envelope for a successful minimal pre-send verify."""
    return {
        "type": "M20_3_PreSendMinimalVerifiedEvent",
        "turn_index": turn_index,
        "surface_intent_outcome": str(
            verification.get("surface_intent_outcome", "ambiguous") or "ambiguous"
        ),
        "committed_surface_intent": str(
            verification.get("committed_surface_intent", "") or ""
        ),
        "expected_mode": expected_mode,
        "committed_self_identification": str(
            commitment.get("self_identification", "") or ""
        )[:MAX_SURFACE_SELF_ID_CHARS],
        "confidence": round(_bounded_float(verification.get("confidence"), default=0.0), 6),
        "evidence_span": str(verification.get("evidence_span", "") or "")[:MAX_SURFACE_EVIDENCE_SPAN_CHARS],
        "engineering_proxy_label": "mvp_local_pre_send_minimal_audit",
    }


def _build_m20_3_pre_send_minimal_degraded_event(
    *,
    turn_index: int,
    reason: str,
) -> dict[str, Any]:
    """Audit envelope for a failed minimal pre-send verify (LLM error)."""
    return {
        "type": "M20_3_PreSendMinimalDegradedEvent",
        "turn_index": turn_index,
        "reason_code": str(reason or "unknown")[:MAX_SURFACE_VERIFICATION_REASON_CHARS],
        "engineering_proxy_label": "mvp_local_pre_send_minimal_audit",
    }


def _run_fast_chat_pre_send_minimal(
    *,
    state: Mapping[str, Any],
    surface_commitment: Mapping[str, Any],
    raw_reply: str,
    user_text: str,
    turn_index: int,
    complete_json_stage: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """P0-1 call site extracted for unit testability.

    Returns `(verification_dict, audit_events)`. The verification dict
    is in the M19.x audit-row shape (with `committed_surface_intent`
    added so the pre-send gate's `_actual_mode` can read it). The
    audit_events list is non-empty when the LLM call ran; on LLM
    failure the function returns an empty verification (matching the
    prior fast_chat skip behavior — gate sees `audit_absent` →
    `ambiguous`) and emits a degraded event.

    When no `runtime_mode_state` horizon commitment with
    `expected_mode` is present, the function returns
    `(empty_verification, [])` — the caller emits the existing
    `SurfaceConsistencyVerificationSkippedEvent`.
    """
    has_runtime_mode_commitment, expected_mode = (
        _has_runtime_mode_state_horizon_with_expected_mode(state)
    )
    if not has_runtime_mode_commitment:
        return _empty_surface_consistency_verification(), []

    audit_events: list[dict[str, Any]] = []
    try:
        minimal_system, minimal_user = build_m20_3_pre_send_minimal_prompt(
            user_text=user_text,
            draft_reply=raw_reply,
            surface_commitment=surface_commitment,
            expected_mode=expected_mode,
            turn_index=turn_index,
        )
        minimal_payload = complete_json_stage(
            "m20_3_pre_send_minimal", minimal_system, minimal_user
        )
        minimal_verify = normalize_m20_3_pre_send_minimal(minimal_payload)
    except Exception as exc:
        # LLM failure path. Return the empty verification (gate sees
        # `audit_absent` → `ambiguous`, the prior fast_chat behavior
        # is preserved) and emit ONLY a degraded event. We do NOT
        # emit a `Verified` event with empty values — that would
        # pollute the audit trail with a non-event.
        return _empty_surface_consistency_verification(
            reason=f"llm_error:{type(exc).__name__}"
        ), [
            _build_m20_3_pre_send_minimal_degraded_event(
                turn_index=turn_index,
                reason=f"llm_error:{type(exc).__name__}",
            )
        ]
    # Map the minimal result onto the M19.x audit row shape so the
    # pre-send gate's `_SURFACE_TO_M20` table reads a real value
    # (not `audit_absent`). `normalize_surface_consistency_verification`
    # does NOT carry `committed_surface_intent` through (the M19.x
    # audit row is built at event-emit time, not at LLM-response
    # time). The pre-send gate's `_actual_mode` reads
    # `observation_context["surface_consistency_verification"]["committed_surface_intent"]`
    # — so we add it explicitly here.
    verification = normalize_surface_consistency_verification(minimal_verify)
    verification["committed_surface_intent"] = minimal_verify.get(
        "committed_surface_intent", ""
    )
    audit_events.append(
        _build_m20_3_pre_send_minimal_verified_event(
            turn_index=turn_index,
            verification=verification,
            commitment=surface_commitment,
            expected_mode=expected_mode,
        )
    )
    return verification, audit_events


def _build_surface_consistency_verification_event(
    *,
    turn_index: int,
    verification: Mapping[str, Any],
    commitment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "SurfaceConsistencyVerifiedEvent",
        "turn_index": turn_index,
        "surface_intent_outcome": str(verification.get("surface_intent_outcome", "ambiguous") or "ambiguous"),
        "self_id_drift_target": str(verification.get("self_id_drift_target", "") or "")[:MAX_SURFACE_SELF_ID_CHARS],
        "evidence_span": str(verification.get("evidence_span", "") or "")[:MAX_SURFACE_EVIDENCE_SPAN_CHARS],
        "confidence": round(_bounded_float(verification.get("confidence"), default=0.0), 6),
        "committed_surface_intent": str(commitment.get("surface_intent", "chat") or "chat"),
        "committed_self_identification": str(commitment.get("self_identification", "") or "")[:MAX_SURFACE_SELF_ID_CHARS],
        "committed_persona_should_apply": bool(commitment.get("persona_should_apply", False)),
        "reason_codes": _string_list(verification.get("evidence_refs"), limit=MAX_SURFACE_EVIDENCE_REFS),
        "engineering_proxy_label": "mvp_local_surface_consistency_audit",
    }


def _empty_surface_consistency_verification(*, reason: str = "") -> dict[str, Any]:
    base = normalize_surface_consistency_verification({})
    if reason:
        base["reason"] = reason[:MAX_SURFACE_VERIFICATION_REASON_CHARS]
    return base


def validate_visible_reply(reply: str, contract: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    original = str(reply or "").strip()
    contract_map = _mapping(contract)
    mode = str(contract_map.get("conversation_mode") or contract_map.get("reply_pacing") or "balanced")
    max_chars = _positive_int(contract_map.get("max_chars"), default=140)
    max_sentences = _positive_int(contract_map.get("max_sentences"), default=2)
    fallback = "我刚才说得有点乱，先简单说：我在。"
    cleaned, stripped_debug = _strip_debug_payload(original)
    actions: list[str] = []
    if stripped_debug:
        actions.append("stripped_debug_payload")
    if not cleaned:
        cleaned = fallback
        actions.append("fallback_empty_or_debug_only")
    chunks = _sentence_chunks(cleaned)
    if chunks and len(chunks) > max_sentences:
        cleaned = "".join(chunks[:max_sentences]).strip()
        actions.append("trimmed_sentences")
    if mode == "casual_fast" and len(cleaned) > max_chars:
        first = chunks[0] if chunks else cleaned
        cleaned = _truncate_to_chars(first, max_chars)
        actions.append("compressed_casual_fast")
    elif max_chars and len(cleaned) > max_chars:
        cleaned = _truncate_to_chars(cleaned, max_chars)
        actions.append("truncated_to_contract")
    if _contains_debug_payload(cleaned):
        cleaned = fallback
        actions.append("fallback_remaining_debug_payload")
    allow_direct_disclosure = bool(contract_map.get("allow_direct_disclosure", True))
    explicit_secrecy_detected = bool(contract_map.get("explicit_secrecy_detected", False))
    if explicit_secrecy_detected and not allow_direct_disclosure:
        leak_markers = ("我告诉你个秘密", "别告诉别人", "有人跟我说", "A说", "B说", "某人跟我讲")
        lowered = cleaned.casefold()
        if any(marker.casefold() in lowered for marker in leak_markers):
            cleaned = fallback
            actions.append("blocked_explicit_secrecy_disclosure")
    selected_disclosure_action = str(contract_map.get("selected_disclosure_action", "none") or "none")
    redaction_targets = _string_list(contract_map.get("redaction_targets"), limit=12)
    if redaction_targets and selected_disclosure_action != "direct_share":
        if any(target and target.casefold() in cleaned.casefold() for target in redaction_targets):
            cleaned = "这个我不方便替他说。"
            actions.append("blocked_redaction_target")
    identity_anchored_action = bool(contract_map.get("identity_anchored_action", False))
    if identity_anchored_action and bool(contract_map.get("deny_identity_anchored_action", False)):
        cleaned = "这个涉及身份与安全，我不能直接替人确认或执行。"
        actions.append("blocked_identity_anchored_action")
    elif identity_anchored_action and bool(contract_map.get("enforce_identity_verification", False)):
        if selected_disclosure_action in {"direct_share", "abstract_share", "none"}:
            cleaned = "这个我先不直接确认，你先提供可核对线索（例如你和对方的关系或上下文）。"
            actions.append("enforced_identity_verification")
    if bool(contract_map.get("avoid_identity_assertion", False)):
        assertion_pattern = r"(你|他|她)(才)?是[\u4e00-\u9fffA-Za-z0-9_]{1,24}"
        if re.search(assertion_pattern, cleaned):
            cleaned = "我先不下身份结论，先按你这轮提供的信息继续观察。"
            actions.append("softened_identity_assertion")
    surface_verification = _mapping(contract_map.get("surface_consistency_verification"))
    if surface_verification:
        outcome = str(surface_verification.get("surface_intent_outcome", "") or "").strip().lower()
        if outcome in {"drifted_intent", "drifted_self_id", "drifted_voice"}:
            cleaned = _assistant_identity_repair_fallback(contract_map)
            actions.append(f"blocked_surface_consistency_{outcome}")
        # consistent / ambiguous / absent outcomes must NOT add to `actions`.
        # Adding an action would flip `changed=True` and trip the existing
        # post-validation repair loop, which would overwrite the reply with
        # whatever the repair LLM returns. Only true drift must trigger repair.
    # When surface_consistency_verification is absent (e.g. fast_chat latency mode
    # or no conscious-plan commitment), engineering does NOT silently fall back
    # to keyword/regex matching. Identity drift detection in that path requires a
    # later turn's conscious-loop commitment plus an LLM self-audit.
    validation = {
        "original_length": len(original),
        "final_length": len(cleaned),
        "conversation_mode": mode,
        "max_chars": max_chars,
        "max_sentences": max_sentences,
        "changed": bool(actions),
        "actions": actions,
        "allow_direct_disclosure": allow_direct_disclosure,
        "explicit_secrecy_detected": explicit_secrecy_detected,
        "selected_disclosure_action": selected_disclosure_action,
        "redaction_targets": redaction_targets,
        "identity_anchored_action": identity_anchored_action,
    }
    return cleaned, validation


def _enforce_path_b_field_reply_contract(
    *,
    reply: str,
    reply_action: str,
    reply_contract: Mapping[str, Any] | None,
) -> tuple[str, str, list[str]]:
    contract = _mapping(reply_contract)
    strategy = str(contract.get("path_b_field_reply_strategy", "") or "").strip().lower()
    selected_action = str(contract.get("path_b_field_selected_action", "") or "").strip().lower()
    guided = bool(contract.get("path_b_field_required", False)) or (
        bool(contract.get("path_b_field_guided", False))
        and strategy == "clarify"
        and selected_action == "scan"
        and bool(contract.get("prefer_clarification", False))
    )
    if not guided:
        return str(reply or "").strip(), str(reply_action or "answer").strip() or "answer", []
    normalized_reply = str(reply or "").strip()
    normalized_action = str(reply_action or "answer").strip().lower() or "answer"
    actions: list[str] = []

    if strategy == "clarify":
        if normalized_action != "clarify":
            normalized_action = "clarify"
            actions.append("path_b_field_forced_clarify_action")
        if "?" not in normalized_reply and "？" not in normalized_reply:
            if selected_action == "scan":
                normalized_reply = (
                    "先别直接动手，我先确认一下："
                    "你是想先检查风险点和隐患，再决定怎么改，还是要我直接给修改方案？"
                )
            else:
                normalized_reply = "我先确认一下你的目标：你是想先核实风险和前提，还是要我直接给结论？"
            actions.append("path_b_field_forced_clarify_reply")
    elif strategy == "deflect":
        if normalized_action not in {"deflect", "clarify"}:
            normalized_action = "deflect"
            actions.append("path_b_field_forced_deflect_action")
        if not normalized_reply:
            normalized_reply = "这一步我不建议直接推进。先把边界和风险条件说清楚，我们再决定怎么做。"
            actions.append("path_b_field_forced_deflect_reply")
    elif strategy == "self_disclose":
        if normalized_action not in {"self_disclose", "clarify"}:
            normalized_action = "self_disclose"
            actions.append("path_b_field_forced_self_disclose_action")
    elif strategy == "answer":
        if normalized_action == "clarify" and not bool(contract.get("prefer_clarification", False)):
            normalized_action = "answer"
            actions.append("path_b_field_forced_answer_action")

    return normalized_reply, normalized_action, actions


def _should_run_post_reply_observer(
    *,
    user_text: str,
    memory_dynamics: Mapping[str, Any],
    reply_validation: Mapping[str, Any],
) -> tuple[bool, str]:
    control = _mapping(memory_dynamics.get("control_guidance"))
    mode = str(control.get("conversation_mode") or control.get("reply_pacing") or "balanced")
    if bool(reply_validation.get("changed")):
        return True, "reply_validation_changed"
    conflict = _bounded_float(control.get("conflict_level"), default=0.0)
    repair = _bounded_float(control.get("repair_bias"), default=0.0)
    clarification = _bounded_float(control.get("clarification_bias"), default=0.0)
    if conflict >= 0.55 or repair >= 0.60 or clarification >= 0.65:
        return True, "high_conflict_or_repair_bias"
    if mode == "serious_thinking":
        return False, "serious_without_observer_trigger"
    return False, "low_risk_short_reply"


_AUXILIARY_LLM_STAGES = {
    "m12_identity_pre",
    "m13_settlement",
    "query_planner",
    "evidence_judge",
    "m11_user_model",
    "m12_2_first_order",
    "m12_2_second_order",
    "reply_repair",
    "post_reply_observer",
    "surface_consistency_verification",
    "m18_7_2_minimal",
    "m20_3_pre_send_minimal",
}


def _is_m12_1_stage(stage: str) -> bool:
    return str(stage or "").startswith("m12_1_step_")


def _call_llm_with_stage_profile(
    llm: JSONLLMClient,
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Apply shorter local client settings to non-reply helper calls."""
    should_profile = stage in _AUXILIARY_LLM_STAGES or _is_m12_1_stage(stage)
    timeout_attr = "timeout_seconds"
    retries_attr = "request_retries"
    aux_timeout_attr = "auxiliary_timeout_seconds"
    aux_retries_attr = "auxiliary_request_retries"
    can_profile = (
        should_profile
        and hasattr(llm, timeout_attr)
        and hasattr(llm, retries_attr)
    )
    if not can_profile:
        return llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)

    old_timeout = getattr(llm, timeout_attr)
    old_retries = getattr(llm, retries_attr)
    try:
        aux_timeout = getattr(llm, aux_timeout_attr, None)
        aux_retries = getattr(llm, aux_retries_attr, None)
        if aux_timeout is not None:
            setattr(llm, timeout_attr, float(aux_timeout))
        if aux_retries is not None:
            setattr(llm, retries_attr, int(aux_retries))
        return llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    finally:
        setattr(llm, timeout_attr, old_timeout)
        setattr(llm, retries_attr, old_retries)


_LATENCY_TASK_MARKERS = (
    "implement",
    "fix",
    "debug",
    "diagnose",
    "review",
    "refactor",
    "test",
    "pytest",
    "python",
    "typescript",
    "javascript",
    "api",
    "json",
    "sql",
    "error",
    "stack trace",
    "plan",
    "code",
    "bug",
    "optimize",
    "performance",
    "latency",
    "帮我实现",
    "修复",
    "调试",
    "诊断",
    "代码",
    "测试",
    "计划",
    "优化",
    "报错",
    "架构",
    "检查",
    "里程碑",
    "是否真的",
    "完成",
)

_LATENCY_MEMORY_MARKERS = (
    "remember",
    "forget",
    "memory",
    "preference",
    "i prefer",
    "记住",
    "忘掉",
    "别记",
    "偏好",
    "我喜欢",
    "我讨厌",
)

_LATENCY_RELATIONSHIP_MARKERS = (
    "you hurt",
    "you ignored",
    "not comfortable",
    "boundary",
    "trust",
    "你刚才",
    "你是不是",
    "别这样",
    "不舒服",
    "边界",
    "信任",
    "关系",
    "喜欢你",
    "讨厌你",
)


def _has_latency_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _prior_surface_drift_observed(state: Mapping[str, Any]) -> bool:
    """Read previous turn's surface_consistency_verification from state.

    Returns True when the most recent surface audit row on the turn-overflow
    surface_consistency_audit_tail reports a drifted outcome. Replaces the
    prior regex over user text; the assistant's own contract is the only
    signal that the previous reply lost identity, and a drifted outcome is
    the only semantic authority for upgrading this turn's latency.
    """
    audit = _mapping(_mapping(state).get("surface_consistency_audit_tail"))
    if not audit:
        return False
    last_event = audit.get("last_event") if isinstance(audit.get("last_event"), Mapping) else None
    if not last_event:
        return False
    outcome = str(last_event.get("surface_intent_outcome", "") or "").strip().lower()
    return outcome in {"drifted_intent", "drifted_self_id", "drifted_voice"}


def _classify_turn_latency_mode(
    state: Mapping[str, Any],
    *,
    user_text: str,
    user_id: str,
    persona_name: str,
    proactive_turn: bool,
    identity_anchored_action: bool,
    assessable_pending_rows: list[Mapping[str, Any]],
    group_turn_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(user_text or "").strip()
    reasons: list[str] = []
    mode = "fast_chat"
    envelope = _bounded_group_turn_envelope(group_turn_envelope)
    if proactive_turn:
        return {"mode": "full", "reason_codes": ["proactive_turn"]}
    if identity_anchored_action:
        return {"mode": "full", "reason_codes": ["identity_anchored_action"]}
    if str(envelope.get("surface_intent", "") or "").strip() == "bot_command":
        return {"mode": "normal", "reason_codes": ["bot_command_surface"]}
    explicit_secret, _secret_phrase = _detect_explicit_secrecy(text)
    if explicit_secret:
        return {"mode": "full", "reason_codes": ["explicit_secrecy"]}
    if _prior_surface_drift_observed(state):
        return {"mode": "normal", "reason_codes": ["prior_surface_consistency_drift"]}
    if assessable_pending_rows:
        reasons.append("pending_reward_settlement")
        mode = "normal"
    if len(text) > 90:
        reasons.append("long_user_text")
        mode = "normal"
    if _has_latency_marker(text, _LATENCY_TASK_MARKERS):
        reasons.append("task_or_technical_marker")
        mode = "normal"
    if _has_latency_marker(text, _LATENCY_MEMORY_MARKERS):
        reasons.append("memory_or_preference_marker")
        mode = "normal"
    if _has_latency_marker(text, _LATENCY_RELATIONSHIP_MARKERS):
        reasons.append("relationship_feedback_marker")
        mode = "normal"
    if _is_follow_up_probe(text) or _has_any_marker(text, _QUERY_PLANNER_CUE_MARKERS):
        reasons.append("recall_or_query_marker")
        mode = "normal"
    if state.get("short_term_memory") and re.search(r"[\?？]|多少钱|是谁|什么|哪", text):
        reasons.append("memory_backed_question")
        mode = "normal"
    if _relationship_value_rows(state, user_id):
        reasons.append("active_relationship_value_memory")
        mode = "normal"
    if mode == "fast_chat":
        reasons.append("low_risk_short_chat")
    return {"mode": mode, "reason_codes": reasons}


def _m12_2_latency_triggered(
    *,
    latency_mode: str,
    user_text: str,
    relationship_value_context: Mapping[str, Any],
) -> tuple[bool, str]:
    if latency_mode == "fast_chat":
        return False, "latency_fast_path"
    if relationship_value_context.get("active_relationship_value_memories"):
        return True, "relationship_value_memory"
    if _has_latency_marker(user_text, _LATENCY_RELATIONSHIP_MARKERS):
        return True, "relationship_feedback_marker"
    if re.search(r"(what do you think of me|how do you see me|do you know me)", str(user_text or ""), re.I):
        return True, "explicit_reciprocal_role_query"
    if re.search(r"(你.*怎么看我|你.*了解我|我在你.*眼里|你觉得我)", str(user_text or "")):
        return True, "explicit_reciprocal_role_query"
    return False, "cadence_not_due"


def _latency_trace_summary(trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    calls = [dict(item) for item in trace]
    total = round(sum(float(item.get("duration_ms", 0.0) or 0.0) for item in calls), 3)
    slowest = max(calls, key=lambda item: float(item.get("duration_ms", 0.0) or 0.0), default={})
    return {
        "total_llm_duration_ms": total,
        "blocking_llm_calls": len(calls),
        "slowest_stage": dict(slowest) if slowest else {},
    }


def _load_m11_state(state: Mapping[str, Any], *, user_id: str, display_name: str) -> M11RuntimeState:
    models = _mapping(state.get("m11_user_models"))
    payload = _mapping(models.get(user_id))
    if not payload:
        return M11RuntimeState.clean(user_id=user_id, display_name=display_name)
    user_model_payload = _mapping(payload.get("user_model"))
    user_model = (
        UserModel.from_dict(user_model_payload)
        if user_model_payload
        else UserModel(user_id=user_id, display_name=display_name)
    )
    return M11RuntimeState(
        user_model=user_model,
        prediction_ledger=UserPredictionLedger.from_dict(_mapping(payload.get("prediction_ledger"))),
        reliability_ledger=SourceReliabilityLedger.from_dict(_mapping(payload.get("reliability_ledger"))),
        prediction_calibration=PredictionCalibrationState.from_dict(_mapping(payload.get("prediction_calibration"))),
    )


def _save_m11_state(state: dict[str, Any], *, user_id: str, m11_state: M11RuntimeState) -> None:
    models = _mapping(state.get("m11_user_models"))
    existing = _mapping(models.get(user_id))
    payload = m11_state.to_dict()
    for key in ("aliases", "identity_binding"):
        if key in existing:
            payload[key] = existing[key]
    models[user_id] = payload
    state["m11_user_models"] = models


def _m11_enabled_for_state(state: Mapping[str, Any]) -> bool:
    return bool(state.get("m11_user_model_enabled", True))


def _load_m12_state(state: Mapping[str, Any]) -> M12RuntimeState:
    payload = _mapping(state.get("m12_user_continuity"))
    if not payload:
        return M12RuntimeState.clean()
    return M12RuntimeState.from_dict(payload)


def _save_m12_state(state: dict[str, Any], *, m12_state: M12RuntimeState) -> None:
    state["m12_user_continuity"] = m12_state.to_dict()


def _m12_enabled_for_state(state: Mapping[str, Any]) -> bool:
    return bool(state.get("m12_identity_continuity_enabled", False))


def _load_m12_1_state(state: Mapping[str, Any]) -> M121RuntimeState:
    payload = _mapping(state.get("m12_1_user_personality"))
    if not payload:
        return M121RuntimeState.clean()
    return M121RuntimeState.from_dict(payload)


def _save_m12_1_state(state: dict[str, Any], *, m12_1_state: M121RuntimeState) -> None:
    state["m12_1_user_personality"] = m12_1_state.to_dict()


def _m12_1_enabled_for_state(state: Mapping[str, Any]) -> bool:
    return bool(state.get("m12_1_personality_enabled", False))


def _load_m12_2_state(state: Mapping[str, Any]) -> M122RuntimeState:
    payload = _mapping(state.get("m12_2_reciprocal_role"))
    if not payload:
        return M122RuntimeState.clean()
    return M122RuntimeState.from_dict(payload)


def _save_m12_2_state(state: dict[str, Any], *, m12_2_state: M122RuntimeState) -> None:
    state["m12_2_reciprocal_role"] = m12_2_state.to_dict()


def _m12_2_enabled_for_state(state: Mapping[str, Any]) -> bool:
    return bool(state.get("m12_2_reciprocal_role_enabled", False))


def _should_default_enable_m12_for_persona_init(state: Mapping[str, Any]) -> bool:
    temporal = _mapping(state.get("temporal_state"))
    if temporal.get("last_turn_at") is not None:
        return False
    if state.get("short_term_memory"):
        return False
    if _mapping(state.get("m11_user_models")):
        return False
    m12_payload = _mapping(state.get("m12_user_continuity"))
    if _mapping(m12_payload.get("profiles_by_user")):
        return False
    if _mapping(m12_payload.get("claim_ledger")).get("entries"):
        return False
    if m12_payload.get("conflict_records"):
        return False
    return True


def _should_default_enable_m12_1_for_persona_init(state: Mapping[str, Any]) -> bool:
    temporal = _mapping(state.get("temporal_state"))
    if temporal.get("last_turn_at") is not None:
        return False
    payload = _mapping(state.get("m12_1_user_personality"))
    if _mapping(payload.get("profiles_by_user")):
        return False
    if _mapping(payload.get("latest_reports_by_user")):
        return False
    return _should_default_enable_m12_for_persona_init(state)


def _identity_anchored_action_sensitive(user_text: str) -> bool:
    """True when the user turn requests identity-bound secrets or high-risk verification."""
    t = str(user_text or "").casefold()
    literal_needles = (
        "密码",
        "验证码",
        "银行卡",
        "身份证",
        "转账",
        "验证身份",
        "确认你是",
        "证明你是",
        "otp",
        "2fa",
        "ssn",
        "passphrase",
        "private key",
        "帮我找到",
        "帮我找",
        "替我确认",
        "确认他是谁",
        "确认她是谁",
        "告诉我他有没有来过",
        "告诉我她有没有来过",
        "他有没有来过",
        "她有没有来过",
        "有没有露面",
        "是不是周青",
        "是不是鲁永刚",
    )
    if any(n in t for n in literal_needles):
        return True
    regex_needles = (
        r"(确认|证明|核对).{0,8}(他|她|对方).{0,6}(是谁|身份)",
        r"(帮我|替我).{0,8}(联系|找到|查一下).{0,8}(他|她|对方)",
        r"(他|她|对方).{0,8}(是不是|到底是).{0,8}[\u4e00-\u9fffA-Za-z0-9_]{1,24}",
    )
    return any(re.search(pattern, t) for pattern in regex_needles)


def _m12_reply_policy_dict_for_entity_binding(
    *,
    m12_state: M12RuntimeState,
    profile: IdentityProfile,
    m12_turn_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if m12_turn_result and m12_turn_result.get("enabled"):
        return dict(_mapping(m12_turn_result.get("reply_policy")))
    open_conflicts = tuple(
        row for row in m12_state.conflict_records if row.resolution_status in {"open", "probed"}
    )
    return select_reply_policy(
        profile=profile,
        active_conflicts=open_conflicts,
        strangeness_signal=None,
        identity_anchored_action=False,
    ).to_dict()


def _m12_claim_alias_promotable(reply_policy: Mapping[str, Any], *, identity_state: str, confidence_band: str) -> bool:
    if identity_state == "corroborated":
        return True
    permitted = str(reply_policy.get("permitted_response", "accept") or "accept")
    return confidence_band == "high" and permitted in {"accept", "probe"}


def _merge_m12_into_entity_binding(
    entity_binding: dict[str, Any],
    m12_result: Mapping[str, Any] | None,
) -> None:
    """Attach M12 fields without overwriting third-party entity_binding targets."""
    if not m12_result or not m12_result.get("enabled"):
        return
    ctx = _mapping(m12_result.get("entity_binding_context"))
    claimed = str(ctx.get("claimed_alias") or "").strip()
    identity_state = str(ctx.get("identity_state", ""))
    confidence_band = str(ctx.get("binding_confidence_band", ""))
    reply_policy = _mapping(m12_result.get("reply_policy"))
    promote_claimed_alias = _m12_claim_alias_promotable(
        reply_policy,
        identity_state=identity_state,
        confidence_band=confidence_band,
    )
    cur = _mapping(entity_binding.get("current_interlocutor"))
    aliases = list(cur.get("aliases") or [])
    if claimed and promote_claimed_alias:
        aliases = _unique_strings(aliases, [claimed], limit=16)
    entity_binding["current_interlocutor"] = {**cur, "aliases": aliases}
    entity_binding["m12_identity"] = {
        "claimed_alias": claimed,
        "claimed_alias_promoted": promote_claimed_alias,
        "identity_state": identity_state,
        "binding_confidence_band": confidence_band,
        "reply_policy": dict(reply_policy),
        "prompt_safe_evidence_cards": [
            dict(item)
            for item in m12_result.get("prompt_safe_evidence_cards", [])
            if isinstance(item, Mapping)
        ],
    }


def _m12_reply_policy_contract_patch(permitted_response: str) -> dict[str, Any]:
    if permitted_response == "accept":
        return {}
    if permitted_response == "probe":
        return {"prefer_clarification": True}
    if permitted_response == "hedge":
        return {"soften_social_evidence_language": True}
    if permitted_response == "ask":
        return {"prefer_clarification": True, "enforce_identity_verification": True}
    if permitted_response == "observe":
        return {"avoid_identity_assertion": True}
    if permitted_response == "refuse":
        return {"deny_identity_anchored_action": True, "prefer_clarification": True}
    return {}


def _merge_m12_into_memory_guidance(
    memory_dynamics: dict[str, Any],
    *,
    m12_result: Mapping[str, Any] | None,
) -> None:
    if not m12_result or not m12_result.get("enabled"):
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    reply_policy = _mapping(m12_result.get("reply_policy"))
    permitted_response = str(reply_policy.get("permitted_response", "accept"))
    contract.update(_m12_reply_policy_contract_patch(permitted_response))
    contract["m12_identity"] = {
        "reply_policy": dict(reply_policy),
        "entity_binding_context": dict(_mapping(m12_result.get("entity_binding_context"))),
        "prompt_safe_evidence_cards": [
            dict(item)
            for item in m12_result.get("prompt_safe_evidence_cards", [])
            if isinstance(item, Mapping)
        ],
    }
    control["reply_contract"] = contract
    memory_dynamics["control_guidance"] = control


def _m11_reply_policy_contract_patch(effects: list[Mapping[str, Any]]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    adjustments = {str(item.get("adjustment", "")) for item in effects}
    if "prefer_shorter_reply" in adjustments:
        patch["prefer_shorter_reply"] = True
        patch["max_sentences"] = 1
        patch["max_chars"] = 90
    if "ask_clarifying_question" in adjustments:
        patch["prefer_clarification"] = True
    if "soften_social_evidence_language" in adjustments:
        patch["soften_social_evidence_language"] = True
    return patch


def _merge_m11_into_memory_guidance(
    memory_dynamics: dict[str, Any],
    *,
    speaker_name: str,
    m11_result: Mapping[str, Any] | None,
) -> None:
    if not m11_result or not m11_result.get("enabled"):
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    effects = [
        dict(item)
        for item in m11_result.get("reply_policy_effects", [])
        if isinstance(item, Mapping)
    ]
    contract.update(_m11_reply_policy_contract_patch(effects))
    control["reply_contract"] = contract
    control["m11_user_model"] = {
        "current_interlocutor": speaker_name,
        "prompt_safe_evidence_cards": list(m11_result.get("prompt_safe_evidence_cards", [])),
        "reply_policy_effects": effects,
    }
    memory_dynamics["control_guidance"] = control


def _merge_m12_1_into_memory_guidance(
    memory_dynamics: dict[str, Any],
    *,
    m12_1_result: Mapping[str, Any] | None,
) -> None:
    if not m12_1_result or not m12_1_result.get("enabled"):
        return
    cards = [
        dict(item)
        for item in m12_1_result.get("prompt_safe_evidence_cards", [])
        if isinstance(item, Mapping)
    ]
    if not cards:
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    orchestrator = _mapping(m12_1_result.get("orchestrator_result"))
    report = _mapping(orchestrator.get("report"))
    control["m12_1_personality"] = {
        "prompt_safe_evidence_cards": cards,
        "latest_report_status": str(report.get("report_status", "")),
        "compact_profile_sections": _compact_m12_1_profile_sections(report),
        "permitted_surface": "internal_thinking_material",
    }
    memory_dynamics["control_guidance"] = control


def _prediction_lock_event(
    *,
    turn_index: int,
    prediction_ids: list[str],
    max_committed_confidence: float,
) -> dict[str, Any]:
    return {
        "type": "PredictionLockedEvent",
        "turn_index": int(turn_index),
        "prediction_ids": list(prediction_ids[:8]),
        "prediction_count": len(prediction_ids),
        "created_before_response": True,
        "max_committed_confidence": round(max_committed_confidence, 6),
        "engineering_proxy_label": "mvp_local_prediction_lock",
    }


def _prediction_lock_skip_event(*, turn_index: int, reason_code: str) -> dict[str, Any]:
    return {
        "type": "PredictionLockSkippedEvent",
        "turn_index": int(turn_index),
        "reason_code": str(reason_code or "proposal_quota_empty"),
        "engineering_proxy_label": "mvp_local_prediction_lock",
    }


def _admit_active_commitments(
    *,
    bus: list,
    state: dict,
    conscious_plan: Mapping[str, Any],
    turn_index: int,
    now: str,
) -> None:
    """M20.0 admission: validate and emit ActiveCommitment audit events.

    Pulls from two bounded sources in M20.0:
    - conscious_plan["active_commitment_proposals"] (new field, default empty)
    - conscious_plan["self_response_expectation_proposals"] wrapped to
      observable = "expectation_outcome_match" on owner mismatch_memory_fast

    M20.0 is admission-only. The adapter does NOT write to any owner's
    storage bucket. M20.1+ implement settlers that consume the audit tail.
    """
    proposals: list[dict] = []
    for item in conscious_plan.get("active_commitment_proposals", []) or []:
        if isinstance(item, Mapping):
            proposals.append(dict(item))

    for sre in conscious_plan.get("self_response_expectation_proposals", []) or []:
        wrapped = wrap_self_response_expectation_proposal(
            sre,
            created_turn=turn_index,
        )
        if wrapped is not None:
            proposals.append(wrapped)

    if not proposals:
        update_commitment_registry_diagnostics(state, admitted=0)
        return

    adapter = ActiveCommitmentAdapter()
    admitted, rejected = adapter.admit_batch(
        proposals=proposals,
        turn_index=turn_index,
        created_at=now,
    )

    for commitment in admitted:
        event = build_active_commitment_created_event(commitment)
        bus.append(event)
        record_active_commitment_event(state, event)
        record_pending_commitment(state, commitment)
        # Initialize the observability entry with the commitment data
        # so M20.2's dispatcher can read it after the pending row is
        # removed on settlement. The dispatcher needs owner_id,
        # source_kind, source_ref, evidence_refs, and
        # engineering_proxy_label to compute a GradedCorrectionDecision.
        init_owner_observability_for_commitment(
            state,
            owner_id=commitment.owner_id,
            commitment=commitment,
        )
        # M20.3 §3.1 — track horizon commitments so the pre-send
        # / post-send gate can find them later in the turn.
        if commitment.horizon == "same_turn_surface":
            horizon_list = state.get("m20_3_horizon_commitments")
            if not isinstance(horizon_list, list):
                horizon_list = []
            horizon_list.append(commitment)
            state["m20_3_horizon_commitments"] = horizon_list

    rejected_counts: dict[str, int] = {}
    for rejection in rejected:
        bus.append(rejection)
        record_active_commitment_event(state, rejection)
        code = str(rejection.get("reason_code", "") or "unknown")
        rejected_counts[code] = rejected_counts.get(code, 0) + 1

    update_commitment_registry_diagnostics(
        state,
        admitted=len(admitted),
        rejected_by_reason_code=rejected_counts,
    )


def _build_m20_1_settlement_scheduler() -> SettlementScheduler:
    """Build the M20.1 SettlementScheduler with the v1 reference settlers.

    M20.1 wires the protocol surface and the six reference settlers.
    The LLM-judge settlers (`boundary_handled`, `initiative_timing_match`
    hybrid fallback) are constructed without an injected LLM call;
    they will return NoSettlement with `settler_unavailable` /
    `settler_hybrid_fallback_exhausted` until M20.1.1 (or a later
    milestone) injects the real LLM stage. This keeps M20.1 acceptance
    free of any new visible behavior.

    M20.4 registers the two new LLM-judge settlers
    (`addressee_target_match`, `reaction_attribution_match`).
    They follow the same pattern (constructed without an
    injected LLM call in v1; fail closed with
    `settler_unavailable` until a later milestone injects
    the real LLM stage).
    """
    scheduler = SettlementScheduler()
    scheduler.register_settler(
        "expectation_outcome_match",
        ExpectationOutcomeMatchDeterministicSettler(),
    )
    scheduler.register_settler(
        "prediction_error_band",
        PredictionErrorBandDeterministicSettler(),
    )
    scheduler.register_settler(
        "identity_voice_match",
        IdentityVoiceMatchLLMJudgeSettler(),
    )
    scheduler.register_settler(
        "boundary_handled",
        BoundaryHandledLLMJudgeSettler(),
    )
    scheduler.register_settler(
        "initiative_timing_match",
        InitiativeTimingMatchHybridSettler(),
    )
    scheduler.register_settler(
        "behavioral_pull_shift",
        BehavioralPullShiftSilentSettler(),
    )
    # M20.4 v1 — two new LLM-judge settlers.
    scheduler.register_settler(
        "addressee_target_match",
        _AddresseeTargetMatchLLMJudgeSettler(),
    )
    scheduler.register_settler(
        "reaction_attribution_match",
        _ReactionAttributionMatchLLMJudgeSettler(),
    )
    return scheduler


def _build_m20_1_observation_context(
    *,
    state: dict,
    bus: list,
    conscious_plan: Mapping[str, Any],
    turn_index: int,
    now: str,
) -> dict[str, Any]:
    """Build the observation_context dict for the M20.1 scheduler.

    Pulls bounded, read-only evidence rows from the conscious plan,
    the per-turn bus, and the active surface-consistency audit pointer.
    The provider MUST NOT mutate any long-term state bucket.
    """
    outcome_results = conscious_plan.get("self_expectation_outcome_results")
    if not isinstance(outcome_results, list):
        outcome_results = []
    bounded_outcome_results = [
        dict(row) for row in outcome_results if isinstance(row, Mapping)
    ][:16]

    prediction_settlements: list[dict[str, Any]] = []
    surface_consistency: dict[str, Any] | None = None
    user_explicit: dict[str, Any] | None = None
    excerpts: list[dict[str, Any]] = []
    for event in bus:
        if not isinstance(event, Mapping):
            continue
        event_type = str(event.get("type", "") or "")
        if event_type == "M17SettlementAssessorEvent":
            # The actual settlement_judgments live on the LLM payload,
            # which the conscious loop re-rendered into the plan. Read
            # them off the conscious plan if present; otherwise leave
            # empty.
            continue
        if event_type == "SurfaceConsistencyVerification":
            surface_consistency = dict(event)
            continue
        if event_type == "UserExplicitRequest":
            user_explicit = dict(event)
            continue
        if event_type in {"ActiveCommitmentCreated", "ActiveCommitmentRejected"}:
            # Skip admission audit events from observation context.
            continue
        # Capture a bounded text excerpt for LLM judges.
        text = event.get("text") or event.get("reason") or event.get("excerpt")
        if isinstance(text, str) and text:
            excerpts.append(
                {
                    "type": event_type,
                    "text": text[:200],
                }
            )
            if len(excerpts) >= 8:
                break

    # The M13.2 prediction settlement rows are stored on
    # conscious_plan["prediction_judgments"] when the LLM settlement
    # assessor runs. Read them through the normalize layer.
    raw_judgments = conscious_plan.get("prediction_judgments")
    if not isinstance(raw_judgments, list):
        raw_judgments = []
    for row in raw_judgments[:16]:
        if not isinstance(row, Mapping):
            continue
        prediction_id = str(row.get("prediction_id", "") or "")
        if not prediction_id:
            continue
        band = str(row.get("band", "") or "")
        prediction_settlements.append(
            {
                "prediction_id": prediction_id,
                "band": band,
                "evidence_refs": [
                    str(ref) for ref in row.get("evidence_refs", [])
                    if isinstance(ref, str) and ref
                ][:16],
            }
        )

    return {
        "now": now,
        "turn_index": turn_index,
        "self_expectation_outcome_results": bounded_outcome_results,
        "prediction_settlements": prediction_settlements,
        "surface_consistency_verification": surface_consistency or {},
        "user_explicit_request": user_explicit,
        "excerpts": excerpts,
    }


def _settle_active_commitments(
    *,
    bus: list,
    state: dict,
    conscious_plan: Mapping[str, Any],
    turn_index: int,
    now: str,
) -> None:
    """M20.1 settlement hook: run the scheduler after admission.

    The scheduler attempts to settle any pending commitments (T0+1
    minimum, single attempt per (commit_id, turn), due_at_passed on
    missing window). It does NOT mutate any long-term state bucket;
    it only writes to `commitment_owner_observability` and emits
    audit events.
    """
    pending = state.get("active_commitments_pending")
    if not isinstance(pending, list) or not pending:
        return

    scheduler = _build_m20_1_settlement_scheduler()
    observation_context = _build_m20_1_observation_context(
        state=state,
        bus=bus,
        conscious_plan=conscious_plan,
        turn_index=turn_index,
        now=now,
    )

    settled_events, no_settlement_events = scheduler.attempt_settlements(
        state=state,
        turn_index=turn_index,
        now=now,
        observation_context_provider=lambda _turn, _row: observation_context,
    )
    for event in settled_events:
        bus.append(event)
        record_active_commitment_event(state, event)
    for event in no_settlement_events:
        bus.append(event)
        record_active_commitment_event(state, event)


def _dispatch_graded_corrections(
    *,
    bus: list,
    state: dict,
    turn_index: int,
    now: str,
    owner_state_snapshot: Mapping[str, Any] | None = None,
    dispatcher: GradedCorrectionDispatcher | None = None,
) -> None:
    """M20.2 dispatch hook: read observability, run dispatcher, route.

    Runs every turn. For each observability entry with a non-`None`
    `settled_value` whose settlement turn is strictly before the
    current turn (T+1+1 rule) and is not yet dispatched, run the
    dispatcher and route the decision to the appropriate
    `active_commitment_grader` stub.

    The dispatcher is pure (no mutation, no LLM, no re-interpretation
    of `observable_payload`). The routing stubs are no-ops in M20.2;
    M20.2.1 wires them to the existing owner write paths.

    Each `commit_id` produces at most one `GradedCorrectionRouted`
    audit event in chronological order. `CorrectionDeferred` and
    `CorrectionRejected` events may repeat on later turns if the
    dispatcher re-evaluates with different inputs.
    """
    if not isinstance(state, dict):
        return
    observability = state.get("commitment_owner_observability")
    if not isinstance(observability, dict) or not observability:
        return

    if dispatcher is None:
        dispatcher = GradedCorrectionDispatcher()

    routed_count = 0
    deferred_count = 0
    rejected_count = 0
    by_level: dict[str, int] = {}
    by_owner_id: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    by_reason_code: dict[str, int] = {}
    magnitudes_before: list[float] = []
    magnitudes_after: list[float] = []
    m19_3_shortcut = 0
    same_turn_advisory_violations = 0

    for owner_id, owner_row in observability.items():
        if not isinstance(owner_row, dict) or not owner_row:
            continue
        for commit_id, commit_row in owner_row.items():
            if not isinstance(commit_row, dict):
                continue
            if commit_row.get("dispatched"):
                continue
            settled_value_row = commit_row.get("settled_value")
            if not isinstance(settled_value_row, dict):
                continue
            settled_turn = int(settled_value_row.get("turn_index", 0) or 0)
            # T+1+1 rule: only dispatch on turns strictly after the
            # settlement turn. Skip the settlement turn itself.
            if settled_turn >= turn_index:
                continue
            commitment_row = commit_row.get("commitment")
            if not isinstance(commitment_row, dict):
                # No commitment data on observability (legacy entry).
                # The dispatcher cannot reconstruct the ActiveCommitment.
                # Skip and record a deferred audit event so the
                # invariant is observable.
                continue

            # Reconstruct the ActiveCommitment from observability.
            try:
                commitment = _observability_row_to_active_commitment(commitment_row)
            except Exception:  # noqa: BLE001
                continue
            try:
                settled_value = _observability_row_to_settled_value(
                    settled_value_row, commit_id=commit_id, turn_index=turn_index, now=now
                )
            except Exception:  # noqa: BLE001
                continue

            try:
                decision = dispatcher.decide(
                    commitment=commitment,
                    settled_value=settled_value,
                    owner_state_snapshot=owner_state_snapshot,
                    turn_index=turn_index,
                    now=now,
                )
            except Exception:  # noqa: BLE001
                # A buggy dispatcher MUST NOT crash the run_turn path.
                # Mark the entry dispatched with a rejected decision to
                # avoid retry storms on subsequent turns.
                commit_row["dispatched"] = True
                commit_row["dispatched_at_turn"] = turn_index
                commit_row["dispatched_correction_level"] = "expire"
                rejected_count += 1
                by_reason_code["owner_state_unavailable"] = (
                    by_reason_code.get("owner_state_unavailable", 0) + 1
                )
                continue

            if decision.rejected:
                rejected_count += 1
                if decision.reason_codes:
                    for rc in decision.reason_codes:
                        by_reason_code[rc] = by_reason_code.get(rc, 0) + 1
                if decision.correction_level == "same_turn":
                    same_turn_advisory_violations += 1
                event = build_correction_rejected_event(decision)
                bus.append(event)
                record_active_commitment_event(state, event)
            elif decision.deferred:
                deferred_count += 1
                if "m19_3_already_promoted" in decision.reason_codes:
                    m19_3_shortcut += 1
                if decision.reason_codes:
                    for rc in decision.reason_codes:
                        by_reason_code[rc] = by_reason_code.get(rc, 0) + 1
                event = build_correction_deferred_event(decision)
                bus.append(event)
                record_active_commitment_event(state, event)
                # A deferred event does NOT mark the entry as
                # dispatched. M20.2 may re-evaluate the entry on a
                # later turn if owner_state_snapshot changes (e.g.
                # M19.3 demotes a promoted entry).
            else:
                routed_count += 1
                by_level[decision.correction_level] = (
                    by_level.get(decision.correction_level, 0) + 1
                )
                by_owner_id[owner_id] = by_owner_id.get(owner_id, 0) + 1
                by_outcome[decision.outcome] = by_outcome.get(decision.outcome, 0) + 1
                if decision.magnitude_before is not None:
                    magnitudes_before.append(decision.magnitude_before)
                if decision.magnitude_after is not None:
                    magnitudes_after.append(decision.magnitude_after)
                _route_decision(
                    decision,
                    state=state,
                    bus=bus,
                    owner_state_snapshot=owner_state_snapshot,
                    commitment=commitment,
                )

            # Mark as dispatched only on a terminal decision
            # (Routed, Rejected, or "expire" Deferred that should not
            # be retried). Deferred events with `m19_3_already_promoted`
            # or `magnitude_below_threshold` are terminal for this
            # entry; `owner_state_unavailable` is not.
            if decision.rejected or not decision.deferred:
                commit_row["dispatched"] = True
                commit_row["dispatched_at_turn"] = turn_index
                commit_row["dispatched_correction_level"] = decision.correction_level

    if routed_count or deferred_count or rejected_count:
        update_graded_correction_diagnostics(
            state,
            routed=routed_count,
            deferred=deferred_count,
            rejected=rejected_count,
            by_level=by_level,
            by_owner_id=by_owner_id,
            by_outcome=by_outcome,
            by_reason_code=by_reason_code,
            magnitudes_before=tuple(magnitudes_before),
            magnitudes_after=tuple(magnitudes_after),
            m19_3_shortcut=m19_3_shortcut,
            same_turn_advisory_violations=same_turn_advisory_violations,
        )


# === M20.3 admission / invariant / surface hooks ========================


def _bounded_string(value: Any, *, default: str = "", limit: int = 120) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def _string_list(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _build_runtime_mode_flags(
    *,
    latency_mode: str,
    surface_intent: str,
    group_mode_ingress_change: bool,
) -> dict[str, Any]:
    """Build the v1 `runtime_mode_flags` for `PolicyProducer.evaluate`.

    M20.3 §1.1 input: `runtime_mode_flags` carries the bounded
    per-turn mode surface. The v1 shape mirrors M18.x's
    `group_turn_binding.surface_intent` plus the latency mode and
    an ingress change flag.
    """
    return {
        "surface_intent": _bounded_string(surface_intent, default="", limit=32),
        "conversation_mode": (
            "bot" if latency_mode == "fast_chat" and surface_intent == "bot" else "chat"
        ),
        "group_mode": surface_intent,
        "group_mode_ingress_change": bool(group_mode_ingress_change),
    }


def _build_command_envelope(bounded_group_turn: Mapping[str, Any]) -> dict[str, Any]:
    """Build the v1 `command_envelope` for `PolicyProducer.evaluate`.

    M20.3 §1.1 input: `envelope.platform_command` and
    `envelope.bot_command_args`. These come from M18.x
    `group_turn_binding` (added at v1 by the bounded group envelope
    builder).
    """
    platform_command = _bounded_string(
        bounded_group_turn.get("platform_command"), default="", limit=64,
    )
    bot_command_args = _string_list(
        bounded_group_turn.get("bot_command_args"), limit=8,
    )
    if not platform_command and not bot_command_args:
        return {}
    return {
        "platform_command": platform_command,
        "bot_command_args": bot_command_args,
    }


def _admit_m20_4_attribution_commitments(
    *,
    bus: list,
    state: dict,
    current_turn_id: int,
    inbound_excerpt: str = "",
    group_turn_binding: Mapping[str, Any] | None = None,
    at: str = "",
) -> list[ActiveCommitment]:
    """M20.4 §2 — admission from the M18.7 attribution surface.

    The producer reads `state["m18_7_attribution_hypotheses"]`
    (M18.7 §5 state surface), filters on `confidence >= 0.4`
    AND `participant_id != ""`, and admits one
    `ActiveCommitment` per matching entry on
    `group_addressee_graph`. Empty surface → silent no-op.

    The admitted rows go through the M20.0 admission gate
    (the ActiveCommitmentAdapter validates shape, clamps,
    and registers the row in state["active_commitments_pending"]).
    The M20.1 scheduler settles them on subsequent turns
    via the registered LLM-judge settlers; the M20.2
    dispatcher routes the outcome to the real
    `group_addressee_graph.microadjust` write path.

    This function does NOT call the LLM.
    """
    admitted = _produce_m20_4_attribution_commitments(
        state=state,
        bus=bus,
        current_turn_id=current_turn_id,
        inbound_excerpt=inbound_excerpt,
        group_turn_binding=group_turn_binding,
        at=at,
    )
    if not admitted:
        return admitted
    # Run the proposals through the M20.0 admission gate.
    from segmentum.dialogue.runtime.active_commitment import (
        ActiveCommitmentAdapter,
        build_active_commitment_created_event,
        init_owner_observability_for_commitment,
        record_active_commitment_event,
        record_pending_commitment,
        update_commitment_registry_diagnostics,
    )
    proposals: list[dict] = []
    for commitment in admitted:
        proposals.append(
            {
                "owner_id": commitment.owner_id,
                "source_kind": commitment.source_kind,
                "source_ref": commitment.source_ref,
                "layer": commitment.layer,
                "observable": commitment.observable,
                "observable_payload": dict(commitment.observable_payload or {}),
                "target": dict(commitment.target or {}),
                "due_at": dict(commitment.due_at or {}),
                "priority": commitment.priority,
                "confidence": commitment.confidence,
                "evidence_refs": list(commitment.evidence_refs or []),
                "reason_codes": list(commitment.reason_codes or []),
                "engineering_proxy_label": commitment.engineering_proxy_label,
                "horizon": commitment.horizon,
            }
        )
    adapter = ActiveCommitmentAdapter()
    accepted, rejected = adapter.admit_batch(
        proposals=proposals,
        turn_index=current_turn_id,
        created_at=str(at),
    )
    for commitment in accepted:
        event = build_active_commitment_created_event(commitment)
        bus.append(event)
        record_active_commitment_event(state, event)
        record_pending_commitment(state, commitment)
        init_owner_observability_for_commitment(
            state,
            owner_id=commitment.owner_id,
            commitment=commitment,
        )
    rejected_counts: dict[str, int] = {}
    for rejection in rejected:
        bus.append(rejection)
        record_active_commitment_event(state, rejection)
        code = str(rejection.get("reason_code", "") or "unknown")
        rejected_counts[code] = rejected_counts.get(code, 0) + 1
    update_commitment_registry_diagnostics(
        state,
        admitted=len(accepted),
        rejected_by_reason_code=rejected_counts,
    )
    return admitted


def _emit_m20_4_tie_breaker_feedback_for_turn(
    *,
    bus: list,
    state: dict,
    turn_index: int,
    now: str,
) -> None:
    """M20.4 v1 — emit the gated M18.5 tie-breaker feedback row
    for any M20.4-admitted commitments that were just
    dispatched this turn.

    M20.4 v1 ships CROSS-TURN feedback only. The T+1+ flip
    engages on subsequent turns when M18.5 reads the
    feedback row. The feedback row is bounded; the
    diagnostic counters record the engagement / rejection
    histogram.

    The function scans observability for
    `addressee_target_match` / `reaction_attribution_match`
    commitments that were just dispatched at this turn and
    runs the engagement rule (C1 fix: AND not OR). Each
    dispatch produces a row in
    `state["m18_5_attribution_feedback"]`. A bus event
    `M18_5AttributionFeedbackRow` is emitted so diagnose
    can cross-reference.
    """
    from segmentum.dialogue.runtime.m20_4_attribution import (
        emit_m20_4_tie_breaker_feedback as _m20_4_feedback,
    )
    if not isinstance(state, dict):
        return
    observability = state.get("commitment_owner_observability")
    if not isinstance(observability, dict) or not observability:
        return
    for owner_id, owner_row in observability.items():
        if owner_id != "group_addressee_graph":
            continue
        if not isinstance(owner_row, dict):
            continue
        for _commit_id, commit_row in owner_row.items():
            if not isinstance(commit_row, dict):
                continue
            if not commit_row.get("dispatched"):
                continue
            dispatched_at_turn = int(
                commit_row.get("dispatched_at_turn", 0) or 0
            )
            if dispatched_at_turn != turn_index:
                continue
            # Reconstruct the commitment + settled value from
            # observability (same as the dispatcher does).
            commitment_row = commit_row.get("commitment")
            if not isinstance(commitment_row, dict):
                continue
            try:
                commitment = _observability_row_to_active_commitment(
                    commitment_row
                )
            except Exception:  # noqa: BLE001
                continue
            settled_value_row = commit_row.get("settled_value")
            if not isinstance(settled_value_row, dict):
                continue
            try:
                settled_value = _observability_row_to_settled_value(
                    settled_value_row,
                    commit_id=str(commitment_row.get("commit_id", "")),
                    turn_index=turn_index,
                    now=now,
                )
            except Exception:  # noqa: BLE001
                continue
            # M18.5 structural decision: read from the dispatch
            # decision (stored in commit_row). The dispatcher
            # embeds the structural decision indirectly; the
            # M20.4 v1 reads it from the active state.
            # For v1, we read the existing action on this
            # turn. The pre-existing `_action` value is the
            # M18.5 outcome.
            m18_5_decision = str(
                state.get("_m20_4_m18_5_decision_at_turn", "")
                or "no_reply"
            )
            decision = type("D", (), {"correction_level": str(
                commit_row.get("dispatched_correction_level", "")
            )})()
            row = _m20_4_feedback(
                state=state,
                decision=decision,
                commitment=commitment,
                settled_value=settled_value,
                m18_5_structural_decision=m18_5_decision,
                at=now,
            )
            if row:
                bus.append(
                    {
                        "type": "M18_5AttributionFeedbackRow",
                        "turn_index": int(turn_index),
                        "feedback_id": str(row.get("feedback_id", "")),
                        "tie_breaker_engaged": bool(
                            row.get("tie_breaker_engaged", False)
                        ),
                        "patched_decision": row.get("patched_decision"),
                        "patched_reason": str(
                            row.get("patched_reason", "")
                        ),
                        "engineering_proxy_label": str(
                            row.get("engineering_proxy_label", "")
                        ),
                        "at": str(now),
                    }
                )


def _admit_policy_commitments(
    *,
    bus: list,
    state: dict,
    producer: PolicyProducer,
    turn_index: int,
    at: str,
    runtime_mode_flags: Mapping[str, Any],
    command_envelope: Mapping[str, Any],
    user_correction_signal: str,
) -> list[ActiveCommitment]:
    """M20.3 §1 admission: run PolicyProducer, register rows, emit events.

    Returns the admitted commitments so the caller can pass them to
    `LoopInvariants.enforce_minimum_loop_coverage`. The producer is
    a single-shot; this helper is called once with empty
    `user_correction_signal` (pre-conscious) and once with the
    conscious-loop signal (post-conscious). Both calls share the
    same `commit_id` derivation (deterministic sha1 of source_ref).

    Horizon commitments (`horizon = "same_turn_surface"`) are also
    appended to `state["m20_3_horizon_commitments"]` so the
    pre-send / post-send gate can find them later in the turn.
    """
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": turn_index, "at": at},
        runtime_mode_flags=runtime_mode_flags,
        command_envelope=command_envelope,
        user_correction_signal=user_correction_signal,
    )
    for event in audit_events:
        bus.append(event)
        record_active_commitment_event(state, event)
    for commitment in admitted:
        event = build_active_commitment_created_event(commitment)
        bus.append(event)
        record_active_commitment_event(state, event)
        record_pending_commitment(state, commitment)
        init_owner_observability_for_commitment(
            state,
            owner_id=commitment.owner_id,
            commitment=commitment,
        )
        if commitment.horizon == "same_turn_surface":
            horizon_list = state.get("m20_3_horizon_commitments")
            if not isinstance(horizon_list, list):
                horizon_list = []
            horizon_list.append(commitment)
            state["m20_3_horizon_commitments"] = horizon_list
    update_commitment_registry_diagnostics(state, admitted=len(admitted))
    return admitted


def _enforce_minimum_loop(
    *,
    bus: list,
    state: dict,
    invariants: LoopInvariants,
    turn_index: int,
    at: str,
    proposed_commitments: list[ActiveCommitment],
    surface_intent: str,
    is_external_turn: bool,
) -> None:
    """M20.3 §4 invariant: audit only, never blocks the turn."""
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=turn_index,
        proposed_commitments=proposed_commitments,
        surface_intent=surface_intent,
        is_external_turn=is_external_turn,
    )
    if not verdict.missed:
        return
    event = build_minimum_loop_coverage_missed_event(verdict)
    event["at"] = at
    bus.append(event)
    record_active_commitment_event(state, event)


def _run_same_turn_surface(
    *,
    bus: list,
    state: dict,
    settler: SameTurnSurfaceSettler,
    horizon_commitments: list[ActiveCommitment],
    draft_reply: str,
    committed_reply: str,
    observation_context: Mapping[str, Any],
    turn_index: int,
    at: str,
) -> tuple[SameTurnSurfaceVerdict | None, SameTurnSurfaceVerdict | None, str]:
    """M20.3 §3 pre-send + post-send gate.

    Returns (pre_verdict, post_verdict, final_reply). If the
    pre-send gate `block`s, the reply is replaced with
    `verdict.replacement` (the bounded persona fallback); otherwise
    the original draft is returned unchanged.
    """
    pre_verdict = settler.run_pre_send(
        draft_reply,
        horizon_commitments=horizon_commitments,
        observation_context=observation_context,
        turn_index=turn_index,
        at=at,
    )
    final_reply = draft_reply
    if pre_verdict is not None:
        event = build_same_turn_surface_verdict_event(pre_verdict)
        bus.append(event)
        record_active_commitment_event(state=state, event=event)
        if pre_verdict.decision == "block" and pre_verdict.replacement:
            final_reply = pre_verdict.replacement
    post_verdict = settler.run_post_send(
        committed_reply,
        horizon_commitments=horizon_commitments,
        observation_context=observation_context,
        turn_index=turn_index,
        at=at,
    )
    if post_verdict is not None:
        event = build_same_turn_surface_verdict_event(post_verdict)
        bus.append(event)
        record_active_commitment_event(state=state, event=event)
    return pre_verdict, post_verdict, final_reply


def _route_decision(
    decision,
    *,
    state: dict,
    bus: list,
    owner_state_snapshot,
    commitment=None,
) -> None:
    """Route a `GradedCorrectionDecision` to its grading stub.

    The stub emits the `GradedCorrectionRouted` audit event and (in
    M20.2.1) calls the owner's existing write path. The originating
    `commitment` is forwarded so the write path can read dispatch
    context (action, user_id, observable_payload, source_ref) that
    is not in the frozen decision.
    """
    level = decision.correction_level
    if level == "microadjust":
        route_microadjust(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )
    elif level == "next_turn":
        route_next_turn(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )
    elif level == "same_turn":
        route_same_turn(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )
    elif level == "slow_promote":
        route_slow_promote(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )
    elif level == "revoke":
        route_revoke(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )
    elif level == "expire":
        route_expire(
            decision, state=state, bus=bus,
            owner_state_snapshot=owner_state_snapshot, commitment=commitment,
        )


def _observability_row_to_active_commitment(row: Mapping[str, Any]):
    """Reconstruct an `ActiveCommitment` from an observability row."""
    from segmentum.dialogue.runtime.active_commitment import (
        ActiveCommitment as _AC,
    )
    payload = row.get("observable_payload")
    if not isinstance(payload, Mapping):
        payload = {}
    target = row.get("target")
    if not isinstance(target, Mapping):
        target = {}
    due_at = row.get("due_at")
    if due_at is not None and not isinstance(due_at, Mapping):
        due_at = None
    return _AC(
        commit_id=str(row.get("commit_id", "") or ""),
        owner_id=str(row.get("owner_id", "") or ""),
        source_kind=str(row.get("source_kind", "") or ""),
        source_ref=str(row.get("source_ref", "") or ""),
        layer=str(row.get("layer", "") or ""),
        observable=str(row.get("observable", "") or ""),
        observable_payload=MappingProxyType(dict(payload)),
        target=MappingProxyType(dict(target)),
        due_at=MappingProxyType(dict(due_at)) if due_at else None,
        priority=float(row.get("priority", 0.0) or 0.0),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        evidence_refs=tuple(row.get("evidence_refs") or ()),
        created_turn=int(row.get("created_turn", 0) or 0),
        created_at=str(row.get("created_at", "") or ""),
        reason_codes=tuple(row.get("reason_codes") or ()),
        engineering_proxy_label=str(row.get("engineering_proxy_label", "") or ""),
    )


def _observability_row_to_settled_value(
    row: Mapping[str, Any],
    *,
    commit_id: str,
    turn_index: int,
    now: str,
):
    """Reconstruct a `SettledValue` from an observability row."""
    from segmentum.dialogue.runtime.active_commitment import (
        SettledValue as _SV,
    )
    return _SV(
        commit_id=commit_id,
        outcome=str(row.get("outcome", "") or ""),
        magnitude=float(row.get("magnitude", 0.0) or 0.0),
        evidence_refs=tuple(row.get("evidence_refs") or ()),
        reason_codes=tuple(row.get("reason_codes") or ()),
        at=str(row.get("at", "") or now),
        turn_index=turn_index,
        settler_type=str(row.get("settler_type", "deterministic") or "deterministic"),
        engineering_proxy_label=str(
            row.get("engineering_proxy_label", "") or "mvp_local_active_commitment"
        ),
    )


def _record_prediction_lock_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    events = state.setdefault("prediction_lock_audit_tail", [])
    if not isinstance(events, list):
        events = []
        state["prediction_lock_audit_tail"] = events
    events.append(dict(event))
    state["prediction_lock_audit_tail"] = events[-80:]


def _record_surface_consistency_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    """Record a surface-consistency audit event in state for next-turn reads.

    Keeps the most recent 40 events and exposes a compact `last_event` pointer
    that the latency classifier can read without scanning the tail. Replaces
    any user-text regex heuristic for "did the previous reply lose identity?".
    """
    if not isinstance(state, dict):
        return
    audit = state.get("surface_consistency_audit_tail")
    if not isinstance(audit, dict):
        audit = {}
    tail = audit.get("events") if isinstance(audit, Mapping) else None
    if not isinstance(tail, list):
        tail = []
    tail.append(dict(event))
    audit["events"] = tail[-40:]
    audit["last_event"] = dict(event)
    state["surface_consistency_audit_tail"] = audit


def _update_prediction_lock_diagnostics(state: dict[str, Any], m11_state: M11RuntimeState) -> None:
    latest_entries: dict[str, Mapping[str, Any]] = {}
    for entry in m11_state.prediction_ledger.entries:
        latest_entries[entry.prediction_id] = entry.to_dict()
    pending = [entry for entry in latest_entries.values() if entry.get("validation_status") == "pending"]
    type_counts: dict[str, int] = {}
    cap_reason_counts: dict[str, int] = {}
    for entry in latest_entries.values():
        prediction_type = str(entry.get("prediction_type", "") or "")
        cap_reason = str(entry.get("confidence_cap_reason", "") or "")
        if prediction_type:
            type_counts[prediction_type] = type_counts.get(prediction_type, 0) + 1
        if cap_reason:
            cap_reason_counts[cap_reason] = cap_reason_counts.get(cap_reason, 0) + 1
    audit_tail = state.get("prediction_lock_audit_tail", [])
    if not isinstance(audit_tail, list):
        audit_tail = []
    lock_events = [row for row in audit_tail if isinstance(row, Mapping) and str(row.get("type", "")) == "PredictionLockedEvent"]
    skip_events = [row for row in audit_tail if isinstance(row, Mapping) and str(row.get("type", "")) == "PredictionLockSkippedEvent"]
    latest_lock_turn = max((int(row.get("turn_index", 0) or 0) for row in lock_events), default=0)
    state["m17_prediction_lock_diagnostics"] = {
        "pending_prediction_count": len(pending),
        "latest_prediction_lock_turn": latest_lock_turn,
        "prediction_type_counts": type_counts,
        "confidence_cap_reason_counts": cap_reason_counts,
        "prediction_lock_coverage_rate": round(len(lock_events) / float(max(1, int(_mapping(state.get("temporal_state")).get("last_turn_index", 0) or 0))), 6),
        "prediction_lock_skip_reason_counts": {
            reason: sum(1 for row in skip_events if str(row.get("reason_code", "")) == reason)
            for reason in {str(row.get("reason_code", "")) for row in skip_events if str(row.get("reason_code", ""))}
        },
    }


_M11_EXTRACTOR_SCHEMA_KEYS = frozenset(noop_extraction())
_M11_SNAPSHOT_TOP_LEVEL_KEYS = frozenset(
    {"user_id", "current_turn_quotes", "last_turn_summaries", "active_hypotheses", "open_predictions"}
)


def _m11_extractor_payload_is_empty(payload: Mapping[str, Any]) -> bool:
    default = noop_extraction()
    if not payload:
        return True
    for key, default_value in default.items():
        value = payload.get(key, default_value)
        if value != default_value:
            return False
    return True


def _classify_m11_extractor_issue(
    *,
    payload: Mapping[str, Any] | None,
    error_detail: str = "",
) -> str:
    if payload is not None:
        payload_keys = set(payload)
        snapshot_keys = sorted(payload_keys & _M11_SNAPSHOT_TOP_LEVEL_KEYS)
        if snapshot_keys and not (payload_keys & _M11_EXTRACTOR_SCHEMA_KEYS):
            return "snapshot_echo_top_level_fields"
        if _m11_extractor_payload_is_empty(payload):
            return "empty_extractor_output"
        if payload_keys - _M11_EXTRACTOR_SCHEMA_KEYS:
            return "unknown_top_level_fields"
    detail = str(error_detail or "")
    if "unknown top-level fields" in detail:
        if any(marker in detail for marker in _M11_SNAPSHOT_TOP_LEVEL_KEYS):
            return "snapshot_echo_top_level_fields"
        return "unknown_top_level_fields"
    if "empty extractor output" in detail:
        return "empty_extractor_output"
    if "snapshot echo" in detail:
        return "snapshot_echo_top_level_fields"
    return "invalid_extractor_output"


def _merge_m12_2_into_memory_guidance(
    memory_dynamics: dict[str, Any],
    *,
    m12_2_result: Mapping[str, Any] | None,
) -> None:
    if not m12_2_result or not m12_2_result.get("enabled"):
        return
    cards = [
        dict(item)
        for item in m12_2_result.get("prompt_safe_evidence_cards", [])
        if isinstance(item, Mapping)
    ]
    hints = [
        dict(item)
        for item in m12_2_result.get("reply_policy_hints", [])
        if isinstance(item, Mapping)
    ]
    relationship_assessment = _mapping(m12_2_result.get("relationship_value_assessment"))
    if not cards and not hints and not relationship_assessment:
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    contract["m12_2_reciprocal_role"] = {
        "prompt_safe_evidence_cards": cards,
        "reply_policy_hints": hints,
        "relationship_value_assessment": relationship_assessment,
        "permitted_surface": "compact_advisory_only",
    }
    constraints = [
        dict(item)
        for item in relationship_assessment.get("relationship_value_constraints", [])
        if isinstance(item, Mapping)
    ]
    if constraints:
        contract["relationship_context_user_id"] = str(relationship_assessment.get("user_id", ""))
        contract["relationship_value_memory_active"] = True
        contract["relationship_value_constraints"] = constraints[:8]
        contract["relationship_constraint_priority"] = RELATIONSHIP_VALUE_PRIORITY
        contract["relationship_value_free_energy"] = {
            "persona_consistency_pressure_band": str(relationship_assessment.get("persona_consistency_pressure_band", "")),
            "user_comfort_pressure_band": str(relationship_assessment.get("user_comfort_pressure_band", "")),
            "predicted_conflict_band": str(relationship_assessment.get("predicted_conflict_band", "")),
            "preferred_policy": str(relationship_assessment.get("preferred_policy", "")),
            "source": "m12_2_reciprocal_role",
        }
    control["reply_contract"] = contract
    control["m12_2_reciprocal_role"] = {
        "prompt_safe_evidence_cards": cards,
        "reply_policy_hints": hints,
        "relationship_value_assessment": relationship_assessment,
    }
    memory_dynamics["control_guidance"] = control


def _merge_surface_identity_contract_into_memory_guidance(
    memory_dynamics: dict[str, Any],
    *,
    persona_name: str,
    group_turn_binding: Mapping[str, Any] | None,
    conscious_plan: Mapping[str, Any] | None = None,
) -> None:
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    binding = _mapping(group_turn_binding)
    surface_intent = str(binding.get("surface_intent", "") or "").strip()
    platform_command = str(binding.get("platform_command", "") or "").strip()
    assistant_surface_label = str(binding.get("assistant_surface_label", "") or "").strip()
    allowed_self_names = _unique_strings(
        [persona_name],
        [assistant_surface_label] if surface_intent == "bot_command" and assistant_surface_label else [],
        limit=6,
    )
    contract["assistant_persona_name"] = str(persona_name or "").strip()
    contract["assistant_surface_intent"] = surface_intent or "chat"
    contract["assistant_surface_label"] = assistant_surface_label
    contract["assistant_allowed_self_names"] = allowed_self_names
    if platform_command:
        contract["platform_command"] = platform_command
    conscious_commitment = _mapping(_mapping(conscious_plan).get("surface_commitment"))
    if conscious_commitment:
        contract["surface_commitment"] = dict(conscious_commitment)
    control["reply_contract"] = contract
    memory_dynamics["control_guidance"] = control


def _compact_m12_1_profile_sections(report: Mapping[str, Any]) -> list[dict[str, str]]:
    sections = report.get("sections", [])
    if not isinstance(sections, list):
        return []
    rows: list[dict[str, str]] = []
    for section in sections[:8]:
        if not isinstance(section, Mapping):
            continue
        rows.append(
            {
                "section_kind": str(section.get("section_kind", "")),
                "status": str(section.get("status", "")),
                "confidence_band": str(section.get("confidence_band", "")),
                "summary": str(section.get("rendered", ""))[:240],
            }
        )
    return rows


def _apply_evidence_judgment_contract(
    memory_dynamics: dict[str, Any],
    evidence_judgment: Mapping[str, Any],
) -> None:
    if not evidence_judgment:
        return
    control = _mapping(memory_dynamics.get("control_guidance"))
    contract = _mapping(control.get("reply_contract"))
    contract["evidence_judgment"] = dict(evidence_judgment)
    contract["epistemic_stance"] = str(evidence_judgment.get("epistemic_stance", ""))
    contract["redaction_targets"] = _string_list(evidence_judgment.get("redaction_targets"), limit=12)
    contract["allowed_reply_actions"] = _string_list(evidence_judgment.get("allowed_reply_actions"), limit=8)
    control["reply_contract"] = contract
    sharing_policy = _mapping(control.get("sharing_policy"))
    sharing_policy["evidence_judgment"] = dict(evidence_judgment)
    sharing_policy["soft_boundary_is_decision_variable"] = (
        str(evidence_judgment.get("epistemic_stance", "")) == "known_with_caveat"
    )
    control["sharing_policy"] = sharing_policy
    memory_dynamics["control_guidance"] = control


@dataclass
class MVPTurnResult:
    reply: str
    action: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    followup_replies: list[str] = field(default_factory=list)


@dataclass
class MVPIdleResult:
    ran_llm: bool
    reflection_focus: dict[str, Any] | None = None
    self_cognition_patch_proposal: dict[str, Any] | None = None
    memory_consolidation_proposals: list[dict[str, Any]] = field(default_factory=list)
    open_item_proposals: list[dict[str, Any]] = field(default_factory=list)
    outreach_recommendation: dict[str, Any] = field(default_factory=dict)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    skip_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    llm_calls_delta: int = 0
    tokens_delta: int = 0


@dataclass
class IdleCognitiveRefreshResult:
    retrieved_ids: list[str] = field(default_factory=list)
    bounded_retrieve_ids: list[str] = field(default_factory=list)
    memory_efe_evaluation: Any | None = None
    m13_band_summary: dict[str, Any] = field(default_factory=dict)
    selected_target: ProactiveTarget | None = None
    reject_reason: str = ""
    audit_events: list[dict[str, Any]] = field(default_factory=list)

    def selected_target_dict(self) -> dict[str, Any] | None:
        if self.selected_target is None:
            return None
        target = self.selected_target
        return {
            "trigger": target.trigger,
            "traceable_expectation_id": target.traceable_expectation_id,
            "evidence_refs": list(target.evidence_refs),
            "proposed_topic": target.proposed_topic,
            "ordinary_language_intent": target.ordinary_language_intent,
            "source_kind": target.source_kind,
            "urgency_band": target.urgency_band,
            "risk_band": target.risk_band,
            "selection_reason_codes": list(target.selection_reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        memory_efe = self.memory_efe_evaluation
        return {
            "retrieved_ids": list(self.retrieved_ids),
            "bounded_retrieve_ids": list(self.bounded_retrieve_ids),
            "memory_efe_should_outreach": bool(getattr(memory_efe, "should_outreach", False)),
            "memory_efe_selected_policy": str(getattr(memory_efe, "selected_policy", "") or ""),
            "m13_band_summary": dict(self.m13_band_summary),
            "selected_target": self.selected_target_dict(),
            "reject_reason": self.reject_reason,
            "events": [dict(event) for event in self.audit_events],
        }


def _proactive_target_from_mapping(raw: Mapping[str, Any] | None) -> ProactiveTarget | None:
    if not isinstance(raw, Mapping):
        return None
    trigger = str(raw.get("trigger", "") or "").strip()
    if not trigger:
        return None
    return ProactiveTarget(
        trigger=trigger,
        traceable_expectation_id=str(raw.get("traceable_expectation_id", "") or "")[:120],
        evidence_refs=_string_list(raw.get("evidence_refs"), limit=8),
        proposed_topic=str(raw.get("proposed_topic", "") or "")[:120],
        ordinary_language_intent=str(raw.get("ordinary_language_intent", "") or "")[:240],
        source_kind=str(raw.get("source_kind", "") or "")[:80],
        urgency_band=str(raw.get("urgency_band", "medium") or "medium")[:32],
        risk_band=str(raw.get("risk_band", "low") or "low")[:32],
        selection_reason_codes=_string_list(raw.get("selection_reason_codes"), limit=8),
    )


@dataclass
class MVPDialogueRuntime:
    store: MVPStateStore
    llm: JSONLLMClient
    persona_name: str = ""
    path_b_field_consumer_enabled: bool = True
    # P0-7 (2026-06-09): the M20.4 per-sub-class
    # diagnostic counters cached from the last
    # `run_turn` call. The counters are not in
    # `MVPStateStore.SYSTEM_FILE_DEFAULTS`, so they
    # are dropped on `store.save`. The runtime
    # caches the in-memory value here so the
    # M18.7.1 calibration harness can read it
    # without going through the store. The value
    # is `None` before the first `run_turn` call.
    _last_m20_4_diagnostics: dict[str, Any] | None = None

    def get_m20_4_diagnostics(self) -> dict[str, Any] | None:
        """Return the M20.4 per-sub-class diagnostic
        counters from the last `run_turn` call, or
        `None` if no `run_turn` has been executed
        or the M20.4 producer was not exercised.

        P0-7 (2026-06-09) — used by the M18.7.1
        calibration harness to surface the P0-4
        producer / P0-5 write / P0-6 tie-breaker
        sub-class counters without going through
        `MVPStateStore.save` (which filters out
        non-`SYSTEM_FILE_DEFAULTS` keys).
        """
        if self._last_m20_4_diagnostics is None:
            return None
        return dict(self._last_m20_4_diagnostics)

    def _episode_ledger(self) -> EpisodeLedger:
        return EpisodeLedger(self.store.root)

    def _load_state_with_initiative_repair(self) -> dict[str, Any]:
        state = self.store.load()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        repaired_state, changed, audit = repair_proactive_count_from_log(
            m13_state,
            log_path=self.store.root / "conversation_log.jsonl",
        )
        if changed:
            state["m13_drive_state"] = repaired_state
            self.store.save(state)
            if audit:
                self.store.append_log(audit)
        return state

    def _current_state_fingerprint(
        self,
        state: Mapping[str, Any],
        *,
        memory_efe_evaluation: Any | None = None,
        band_summary: Mapping[str, Any] | None = None,
    ) -> str:
        return state_fingerprint(
            state,
            memory_efe_evaluation=memory_efe_evaluation,
            band_summary=band_summary,
        )

    @staticmethod
    def _memory_gate_audit_len(state: Mapping[str, Any]) -> int:
        events = state.get("memory_gate_audit_tail", [])
        return len(events) if isinstance(events, list) else 0

    @staticmethod
    def _memory_gate_events_since(state: Mapping[str, Any], start: int) -> list[dict[str, Any]]:
        events = state.get("memory_gate_audit_tail", [])
        if not isinstance(events, list):
            return []
        return [dict(event) for event in events[max(0, int(start)) :] if isinstance(event, Mapping)]

    @staticmethod
    def _outcome_from_conscious_plan(conscious_plan: Mapping[str, Any]) -> str:
        statuses = [
            str(item.get("status", "") or "").lower()
            for item in conscious_plan.get("expectation_results", []) or []
            if isinstance(item, Mapping)
        ]
        if any(status == "violated" for status in statuses):
            return "violated"
        if any(status == "uncertain" for status in statuses):
            return "uncertain"
        if any(status == "confirmed" for status in statuses):
            return "confirmed"
        return "settled"

    @staticmethod
    def _settled_outcome_from_event(event: Mapping[str, Any]) -> str:
        event_type = str(event.get("type", "") or "")
        if event_type == "M13RewardSettlementEvent":
            band = str(event.get("outcome_band", "") or "")
            if band == "positive":
                return "confirmed"
            if band == "negative":
                return "violated"
            return "uncertain"
        if event_type == "MemoryEfeSettlementEvent":
            band = str(event.get("outcome_band", "") or "")
            if band == "resolved":
                return "confirmed"
            if band == "unresolved":
                return "violated"
            return "uncertain"
        return "uncertain"

    def _record_episode(self, episode: Any) -> None:
        ledger = self._episode_ledger()
        ledger.append(episode)
        self.store.append_log(
            {
                "event": "m15_episode_ledger",
                "type": "MemoryDynamicsEpisodeEvent",
                **episode.to_dict(),
            }
        )

    def _run_consolidation_cycle(
        self,
        state: dict[str, Any],
        *,
        now: int,
        turn_index: int,
        triggered_by: str,
        current_idle_tick_event: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        result = ConsolidationOwner.maybe_run(
            state,
            now=now,
            turn_index=turn_index,
            ledger=self._episode_ledger(),
            budget={"triggered_by": triggered_by},
        )
        events = list(result.events)
        if triggered_by == "background_tick":
            meta_result = detect_and_emit_intents(
                state,
                self._episode_ledger(),
                now=now,
                turn_index=turn_index,
                source="background_tick",
            )
            events.extend(meta_result.events)
        cleanup_result = detect_cleanup_intents(
            state,
            now=now,
            turn_index=turn_index,
            source=triggered_by,
            current_idle_tick_event=current_idle_tick_event,
        )
        events.extend(cleanup_result.events)
        cleanup_run = CleanupOwner.apply_intents(
            state,
            now=now,
            turn_index=turn_index,
            source=triggered_by,
        )
        events.extend(cleanup_run.events)
        for event in events:
            if str(event.get("type", "") or "").startswith("MemoryGate"):
                self._record_memory_gate_event(state, event)
            event_type = str(event.get("type", "") or "")
            channel = (
                "m15_cleanup_audit"
                if event_type.startswith("Cleanup")
                or event_type in {
                    "OpenItemBacklogDetectedEvent",
                    "PendingExpectationBacklogDetectedEvent",
                    "LowTraceabilityRecallBurdenDetectedEvent",
                }
                else "m15_consolidation_audit"
            )
            self.store.append_log({"event": channel, **event})
        return state, events

    def _record_episode_settlements(
        self,
        state: Mapping[str, Any],
        settlement_events: list[Mapping[str, Any]],
        *,
        now: int,
        turn_index: int,
        memory_dynamics: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ledger = self._episode_ledger()
        addenda: list[dict[str, Any]] = []
        for event in settlement_events:
            if not isinstance(event, Mapping):
                continue
            if str(event.get("type", "") or "") not in {"M13RewardSettlementEvent", "MemoryEfeSettlementEvent"}:
                continue
            episode_id = str(event.get("m15_episode_id", "") or "")
            episode = ledger.find_episode(episode_id=episode_id) if episode_id else None
            if episode is None:
                prior_turn = event.get("prior_turn_index")
                try:
                    prior_turn_index = int(prior_turn)
                except (TypeError, ValueError):
                    prior_turn_index = -1
                if prior_turn_index >= 0:
                    episode = ledger.find_episode(turn_index=prior_turn_index)
            if episode is None:
                continue
            components = aggregate_fe_components(
                state,
                memory_dynamics=memory_dynamics,
                settlement_event=event,
            )
            addendum = ledger.append_settlement_event(
                episode_id=episode.episode_id,
                at=now,
                turn_index=turn_index,
                new_outcome_summary=self._settled_outcome_from_event(event),
                fe_proxy_after_revised=aggregate_fe_proxy(components),
                components_after_revised=components,
                settlement_event=event,
            )
            self.store.append_log({"event": "m15_episode_ledger", **addendum})
            addenda.append(addendum)
        return addenda

    def _tag_pending_settlements_with_episode(self, state: dict[str, Any], *, episode_id: str, turn_index: int) -> None:
        m13_state = normalize_m13_drive_state(state.get("m13_drive_state"))
        reward = normalize_affective_reward_proxy_state(m13_state.get("affective_reward_proxy"))
        changed = False
        for row in reward.get("pending_settlements", []) or []:
            if not isinstance(row, dict):
                continue
            try:
                prior_turn_index = int(row.get("prior_turn_index", -1))
            except (TypeError, ValueError):
                prior_turn_index = -1
            if prior_turn_index == int(turn_index):
                row["m15_episode_id"] = episode_id
                changed = True
        if changed:
            m13_state["affective_reward_proxy"] = reward
            state["m13_drive_state"] = m13_state

    def analyze_personas_from_materials(self, materials: list[str]) -> list[dict[str, Any]]:
        return analyze_materials_into_personas(
            self.llm,
            materials,
            persona_name=self.persona_name,
        )

    def initialize_from_persona_payload(self, persona_payload: Mapping[str, Any]) -> dict[str, Any]:
        state = self.store.load()
        # Persona material analysis does not author M12 continuity; keep existing disk values.
        prior_m12_enabled = state.get("m12_identity_continuity_enabled")
        prior_m12_blob = state.get("m12_user_continuity")
        prior_m12_1_enabled = state.get("m12_1_personality_enabled")
        prior_m12_1_blob = state.get("m12_1_user_personality")
        prior_m12_2_enabled = state.get("m12_2_reciprocal_role_enabled")
        prior_m12_2_blob = state.get("m12_2_reciprocal_role")
        prior_m13_blob = state.get("m13_drive_state")
        default_enable_m12 = _should_default_enable_m12_for_persona_init(state)
        default_enable_m12_1 = _should_default_enable_m12_1_for_persona_init(state)
        payload = normalize_persona_payload(persona_payload, fallback_name=self.persona_name)
        for key in SYSTEM_FILE_DEFAULTS:
            state[key] = payload[key]
        if isinstance(prior_m12_enabled, bool):
            state["m12_identity_continuity_enabled"] = (
                True if (default_enable_m12 and not prior_m12_enabled) else prior_m12_enabled
            )
        elif prior_m12_enabled is not None:
            state["m12_identity_continuity_enabled"] = bool(prior_m12_enabled)
        elif default_enable_m12:
            state["m12_identity_continuity_enabled"] = True
        if isinstance(prior_m12_blob, Mapping):
            state["m12_user_continuity"] = dict(prior_m12_blob)
        if isinstance(prior_m12_1_enabled, bool):
            state["m12_1_personality_enabled"] = (
                True if (default_enable_m12_1 and not prior_m12_1_enabled) else prior_m12_1_enabled
            )
        elif prior_m12_1_enabled is not None:
            state["m12_1_personality_enabled"] = bool(prior_m12_1_enabled)
        elif default_enable_m12_1:
            state["m12_1_personality_enabled"] = True
        if isinstance(prior_m12_1_blob, Mapping):
            state["m12_1_user_personality"] = dict(prior_m12_1_blob)
        if isinstance(prior_m12_2_enabled, bool):
            state["m12_2_reciprocal_role_enabled"] = prior_m12_2_enabled
        elif prior_m12_2_enabled is not None:
            state["m12_2_reciprocal_role_enabled"] = bool(prior_m12_2_enabled)
        if isinstance(prior_m12_2_blob, Mapping):
            state["m12_2_reciprocal_role"] = dict(prior_m12_2_blob)
        if isinstance(prior_m13_blob, Mapping):
            normalized_m13 = normalize_m13_drive_state(prior_m13_blob)
            if normalized_m13.get("path_patterns_by_action") or normalized_m13.get("recent_action_trace"):
                state["m13_drive_state"] = normalized_m13
        now = _utc_timestamp()
        for memory in state.get("long_term_memory", []):
            if isinstance(memory, dict):
                memory.setdefault("created_at", now)
                memory.setdefault("source", "materials")
                memory.setdefault("recall_count", 0)
        self.store.save(state)
        self.store.append_log(
            {
                "event": "initialize_from_material_persona",
                "at": now,
                "persona_name": payload.get("persona_name", self.persona_name),
                "source_role_evidence": payload.get("source_role_evidence", []),
                "result": payload,
            }
        )
        return state

    def initialize_from_materials(self, materials: list[str]) -> dict[str, Any]:
        personas = self.analyze_personas_from_materials(materials)
        selected = personas[0]
        if self.persona_name:
            for persona in personas:
                if str(persona.get("persona_name", "")).strip() == self.persona_name:
                    selected = persona
                    break
        return self.initialize_from_persona_payload(selected)

    def run_turn(
        self,
        user_text: str,
        *,
        turn_index: int = 0,
        speaker_name: str = "",
        group_turn_envelope: Mapping[str, Any] | None = None,
        ingress_evidence_band: str = "",
        bus_messages: list[Mapping[str, Any]] | None = None,
        now: int | None = None,
        proactive_context: Mapping[str, Any] | None = None,
        turn_progress: Any | None = None,
    ) -> MVPTurnResult:
        now = _utc_timestamp() if now is None else int(now)
        state = self.store.load()
        episode_ledger = self._episode_ledger()
        episode_components_before = aggregate_fe_components(state)
        memory_gate_audit_start = self._memory_gate_audit_len(state)
        bounded_group_turn = _bounded_group_turn_envelope(group_turn_envelope)
        previous_group_chat_state = _mapping(_mapping(state.get("temporal_state")).get("group_chat_state"))
        display_name = str(speaker_name or "").strip() or "default_user"
        participant_key = str(bounded_group_turn.get("speaker_participant_id", "") or "").strip()
        user_id = _safe_user_id(participant_key or display_name)
        session_id = self.store.root.name
        proactive_turn = isinstance(proactive_context, Mapping) and bool(proactive_context)
        ingress_band = _bounded_ingress_evidence_band(ingress_evidence_band)
        prior_last_user_text = str(_mapping(state.get("temporal_state")).get("last_user_text", "") or "")
        proactive_surrogate_text = str(user_text or "") if proactive_turn else ""
        proactive_defer_audit_log = bool(proactive_context.get("defer_audit_log")) if proactive_turn else False
        if proactive_turn:
            user_text = build_proactive_thinking_user_text(
                surrogate=proactive_surrogate_text,
                ordinary_language_intent=str(proactive_context.get("ordinary_language_intent", "") or ""),
                proposed_topic=str(proactive_context.get("proposed_topic", "") or ""),
                trigger=str(proactive_context.get("trigger", "") or ""),
                evidence_refs=[
                    str(ref)
                    for ref in proactive_context.get("trigger_evidence_refs", []) or []
                    if str(ref).strip()
                ],
                source_kind=str(proactive_context.get("source_kind", "") or ""),
            )
        sharing_regret_feedback: dict[str, Any] = {}
        if not proactive_turn:
            sharing_regret_feedback = self._apply_sharing_regret_feedback(
                state,
                user_text=user_text,
                current_user_id=user_id,
                now=now,
                turn_index=turn_index,
            )
        m11_state = _load_m11_state(state, user_id=user_id, display_name=display_name)
        m11_result_dict: dict[str, Any] = {}
        temporal_input = _temporal_input_from_state(state, now=now)
        bus = list(bus_messages or [])
        bus.append({"type": "TemporalContextEvent", "turn_index": turn_index, **temporal_input})
        if proactive_turn:
            bus.append(
                {
                    "type": "M13ProactiveTurnRequestEvent",
                    "turn_index": turn_index,
                    "proposal_id": str(proactive_context.get("proposal_id", "")),
                    "trigger": str(proactive_context.get("trigger", "")),
                    "source": "m13_proactive_turn",
                    "role": "assistant",
                    "not_user_requested_current_turn": True,
                    "ordinary_language_intent": str(
                        proactive_context.get("ordinary_language_intent", "") or ""
                    )[:240],
                    "trigger_evidence_refs": [
                        str(ref)
                        for ref in proactive_context.get("trigger_evidence_refs", []) or []
                        if str(ref).strip()
                    ][:8],
                    "source_kind": str(proactive_context.get("source_kind", "") or "")[:80],
                    "surrogate_context": str(proactive_surrogate_text or "")[:240],
                    "at": now,
                }
            )
        else:
            bus.append({
                "type": "UserUtteranceEvent",
                "turn_index": turn_index,
                "speaker_name": display_name,
                "user_id": user_id,
                "speaker_participant_id": participant_key or user_id,
                "visible_participant_ids": _bounded_string_list(bounded_group_turn.get("visible_participant_ids"), limit=8, item_max_chars=64),
                "addressed_participant_ids": _bounded_string_list(bounded_group_turn.get("addressed_participant_ids"), limit=8, item_max_chars=64),
                "mentioned_participant_ids": _bounded_string_list(bounded_group_turn.get("mentioned_participant_ids"), limit=8, item_max_chars=64),
                "reply_to_turn_id": str(bounded_group_turn.get("reply_to_turn_id", "") or "").strip()[:120],
                "quoted_turn_ids": _bounded_string_list(bounded_group_turn.get("quoted_turn_ids"), limit=8, item_max_chars=120),
                "explicit_mentions": _bounded_string_list(bounded_group_turn.get("explicit_mentions"), limit=8, item_max_chars=64),
                "ingress_evidence_band": ingress_band,
                "text": user_text,
                "at": now,
            })
        def _report_progress(step: str) -> None:
            reporter = turn_progress
            if reporter is not None and hasattr(reporter, "advance"):
                reporter.advance(step)

        _report_progress("init")
        identity_anchored_action = _identity_anchored_action_sensitive(user_text)
        m12_pre_result: dict[str, Any] | None = None
        turn_key = f"turn_{turn_index + 1:04d}"
        turn_latency_trace: list[dict[str, Any]] = []
        skipped_llm_stages: list[dict[str, str]] = []
        turn_latency_started = time.monotonic()

        def _mark_llm_skipped(stage: str, reason: str) -> None:
            skipped_llm_stages.append({"stage": stage, "reason": reason})

        def _complete_json_stage(stage: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            started = time.monotonic()
            try:
                result = _call_llm_with_stage_profile(
                    self.llm,
                    stage=stage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                turn_latency_trace.append(
                    {
                        "stage": stage,
                        "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                        "success": False,
                        "error_type": type(exc).__name__,
                        "prompt_chars": len(system_prompt) + len(user_prompt),
                        "completion_present": False,
                    }
                )
                raise
            turn_latency_trace.append(
                {
                    "stage": stage,
                    "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "success": True,
                    "error_type": "",
                    "prompt_chars": len(system_prompt) + len(user_prompt),
                    "completion_present": bool(result),
                }
            )
            return result

        m13_state = normalize_m13_drive_state(state.get("m13_drive_state"))
        reward_for_settlement = normalize_affective_reward_proxy_state(
            m13_state.get("affective_reward_proxy")
        )
        user_reaction_assessments: dict[str, dict[str, Any]] = {}
        assessable_pending_rows = list_assessable_pending_rows(
            reward_for_settlement,
            turn_index=turn_index,
        )
        latency_mode_info = _classify_turn_latency_mode(
            state,
            user_text=user_text,
            user_id=user_id,
            persona_name=self.persona_name,
            proactive_turn=proactive_turn,
            identity_anchored_action=identity_anchored_action,
            assessable_pending_rows=assessable_pending_rows,
            group_turn_envelope=bounded_group_turn,
        )
        latency_mode = str(latency_mode_info.get("mode", "normal") or "normal")
        m12_cognitive_bus = CognitiveEventBus()
        # M20.3 per-turn admission / invariant singletons. The
        # `SameTurnSurfaceSettler` resets its per-turn dedup at
        # the start of each turn via `reset_turn_dedup` (callers
        # must invoke this before `run_pre_send`).
        policy_producer = PolicyProducer()
        loop_invariants = LoopInvariants()
        same_turn_surface_settler = SameTurnSurfaceSettler()
        same_turn_surface_settler.reset_turn_dedup()
        if _m12_enabled_for_state(state) and not proactive_turn and latency_mode != "fast_chat":
            m12_state = _load_m12_state(state)
            m11_readonly_pre: dict[str, object] = {}

            def _extract_m12_pre(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                system_prompt, user_prompt = build_m12_identity_extractor_prompt(
                    snapshot=snapshot,
                    speaker_name=display_name,
                )
                try:
                    return _complete_json_stage("m12_identity_pre", system_prompt, user_prompt)
                except Exception:
                    return {
                        "identity_claims": [],
                        "continuity_cues": [],
                        "strangeness_band": "low",
                        "surprise_explanation": "",
                    }

            legacy_aliases_pre: list[str] = []
            legacy_user_models_pre = _mapping(state.get("m11_user_models"))
            legacy_row_pre = _mapping(legacy_user_models_pre.get(user_id))
            for alias in _string_list(legacy_row_pre.get("aliases"), limit=8):
                legacy_aliases_pre.append(alias)
            identity_binding_pre = _mapping(legacy_row_pre.get("identity_binding"))
            for alias in _string_list(identity_binding_pre.get("aliases"), limit=8):
                legacy_aliases_pre.append(alias)
            m12_state, m12_turn = run_m12_turn(
                m12_state,
                user_id=user_id,
                display_name=display_name,
                turn_id=turn_key,
                current_turn_quotes={"q_current": user_text},
                m11_readonly_summary=m11_readonly_pre,
                legacy_aliases=legacy_aliases_pre,
                extractor=_extract_m12_pre,
                config=M12RuntimeConfig(m12_identity_continuity_enabled=True, persona_kind="ui_chat"),
                event_bus=m12_cognitive_bus,
                session_id=str(self.store.root.resolve()),
                persona_id=self.persona_name or "default",
                cycle=turn_index,
                event_sequence_index=0,
                identity_anchored_action=identity_anchored_action,
            )
            _save_m12_state(state, m12_state=m12_state)
            m12_pre_result = m12_turn.to_dict()
            for seq_idx, ev in enumerate(m12_cognitive_bus.events()):
                bus.append({
                    "type": ev.event_type,
                    "turn_index": turn_index,
                    "sequence": seq_idx,
                    "cognitive_event": ev.to_dict(),
                })
        elif _m12_enabled_for_state(state) and not proactive_turn:
            _mark_llm_skipped("m12_identity_pre", "latency_fast_path")
        _report_progress("m12_identity_pre")
        entity_binding = build_entity_binding_context(
            state=state,
            user_text=user_text,
            display_name=display_name,
            user_id=user_id,
            temporal_input=temporal_input,
            m12_turn_result=m12_pre_result,
        )
        _merge_m12_into_entity_binding(entity_binding, m12_pre_result)
        alias_updates_applied = _record_interlocutor_aliases(
            state,
            user_id=user_id,
            display_name=display_name,
            aliases=_string_list(entity_binding.get("alias_assertions"), limit=8),
            evidence=user_text,
            now=now,
        )
        if alias_updates_applied:
            entity_binding = build_entity_binding_context(
                state=state,
                user_text=user_text,
                display_name=display_name,
                user_id=user_id,
                temporal_input=temporal_input,
                m12_turn_result=m12_pre_result,
            )
            _merge_m12_into_entity_binding(entity_binding, m12_pre_result)
        group_turn_binding = _build_group_turn_binding(
            display_name=display_name,
            user_id=user_id,
            group_turn_envelope=bounded_group_turn,
            entity_binding=entity_binding,
        )
        group_reply_policy = _decide_group_reply_policy(
            group_turn_binding=group_turn_binding,
            previous_group_chat_state=previous_group_chat_state,
            persona_name=self.persona_name,
        )
        bus.append({
            "type": "EntityBindingEvent",
            "turn_index": turn_index,
            "binding": entity_binding,
        })
        bus.append({
            "type": "GroupTurnBindingEvent",
            "turn_index": turn_index,
            "binding": group_turn_binding,
        })
        bus.append({
            "type": "GroupReplyPolicyEvent",
            "turn_index": turn_index,
            "policy": group_reply_policy,
        })

        if assessable_pending_rows and str(user_text or "").strip() and not proactive_turn:
            observation_channels = observation_channels_from_bus(bus)
            for assessable_pending in assessable_pending_rows:
                pending_id = str(assessable_pending.get("pending_id", ""))
                if not pending_id:
                    continue
                try:
                    assessor_system, assessor_user = build_m13_settlement_assessor_prompt(
                        user_text=user_text,
                        prior_reply_summary=str(assessable_pending.get("prior_reply_summary", "") or "")[:160],
                        prior_diagnostics=pending_diagnostics_summary_for_assessor(assessable_pending),
                        observation_channels=observation_channels,
                        turn_index=turn_index,
                    )
                    assessor_raw = _complete_json_stage("m13_settlement", assessor_system, assessor_user)
                    user_reaction_assessments[pending_id] = normalize_user_reaction_assessment(assessor_raw)
                    assessment = user_reaction_assessments[pending_id]
                    bus.append(
                        {
                            "type": "M13RewardSettlementAssessorEvent",
                            "turn_id": turn_key,
                            "turn_index": turn_index,
                            "pending_id": pending_id,
                            "reaction": assessment.get("reaction"),
                            "confidence": assessment.get("confidence"),
                            "reason_codes": list(assessment.get("reason_codes", []))[:4],
                            "engineering_proxy_label": "mvp_local_affective_reward_proxy",
                        }
                    )
                except Exception as exc:
                    user_reaction_assessments[pending_id] = normalize_user_reaction_assessment(
                        {"reaction": "unclear", "confidence": 0.0, "reason_codes": ["assessor_error"]}
                    )
                    bus.append(
                        {
                            "type": "M13RewardSettlementAssessorEvent",
                            "turn_id": turn_key,
                            "turn_index": turn_index,
                            "pending_id": pending_id,
                            "reaction": "unclear",
                            "confidence": 0.0,
                            "reason_codes": ["assessor_error"],
                            "assessor_error": type(exc).__name__,
                            "engineering_proxy_label": "mvp_local_affective_reward_proxy",
                        }
                    )
        m13_state, _m13_settlements, m13_settlement_events = settle_pending_m13_actions(
            m13_state,
            user_id=user_id,
            turn_index=turn_index,
            turn_id=turn_key,
            observation_channels=observation_channels_from_bus(bus),
            user_reaction_assessments=user_reaction_assessments,
        )
        state["m13_drive_state"] = m13_state
        for m13_settlement_event in m13_settlement_events:
            bus.append(m13_settlement_event)
        for addendum in self._record_episode_settlements(
            state,
            m13_settlement_events,
            now=now,
            turn_index=turn_index,
        ):
            bus.append(addendum)

        _report_progress("m13_settlement")
        # M20.3 §1 — call PolicyProducer BEFORE the conscious loop so
        # the runtime invariant at step 3 sees the union of T0
        # admissions. The first call passes an empty
        # `user_correction_signal`; the conscious loop fills the
        # v2 attribute, and a second call (below) admits the
        # signal-driven rows.
        pre_conscious_admitted: list[ActiveCommitment] = _admit_policy_commitments(
            bus=bus,
            state=state,
            producer=policy_producer,
            turn_index=turn_index,
            at=now,
            runtime_mode_flags=_build_runtime_mode_flags(
                latency_mode=latency_mode,
                surface_intent=str(
                    bounded_group_turn.get("surface_intent", "") or ""
                ),
                group_mode_ingress_change=bool(
                    bounded_group_turn.get("group_mode_ingress_change", False)
                ),
            ),
            command_envelope=_build_command_envelope(bounded_group_turn),
            user_correction_signal="",
        )
        # === M18.7.2: minimal-prompt attribution call site =========
        # M18.7.2 owns a dedicated minimal-prompt LLM call site
        # for addressee / reaction attribution, decoupled from
        # the conscious loop. The conscious-loop path is broken
        # at scale: M18.7.1 real-LLM replay (commits b13f07f /
        # b969d8e) showed 0/12 non-empty fills when the M18.7
        # v2 attrs segment sat at char 2914 (37.7%) of a
        # 7.7-26k-char conscious-loop prompt. The minimal
        # prompt is ~1.5-2.0k chars; the LLM fills only the
        # M18.7 v1 shape. The result is fed to
        # `_emit_m18_7_2_attribution_for_turn` below, which
        # writes the SAME
        # `state["m18_7_attribution_hypotheses"]` surface that
        # the M20.4 producer and the M18.7.1 calibration runner
        # already read. The dead path comes alive end-to-end.
        _m18_7_2_plan: dict = {
            "addressee_hypothesis": {},
            "reaction_attribution_hypothesis": {},
        }
        if bounded_group_turn:
            # Skip the LLM call for non-group turns; the M18.7
            # fields are not meaningful without
            # group_turn_binding. The orchestrator below is a
            # no-op when the plan has empty fields.
            try:
                _m18_7_2_system, _m18_7_2_user = _build_m18_7_minimal_prompt(
                    state=state,
                    user_text=user_text,
                    speaker_name=display_name,
                    bus_messages=bus,
                    turn_index=turn_index,
                    entity_binding=entity_binding,
                    group_turn_binding=group_turn_binding,
                    m18_5_structural_decision=str(
                        group_reply_policy.get("action", "") or ""
                    ),
                )
                _m18_7_2_raw = _complete_json_stage(
                    "m18_7_2_minimal",
                    _m18_7_2_system,
                    _m18_7_2_user,
                )
                _m18_7_2_plan = {
                    "addressee_hypothesis": _normalize_m18_7_addressee_hypothesis(
                        _m18_7_2_raw.get("addressee_hypothesis")
                    ),
                    "reaction_attribution_hypothesis": (
                        _normalize_m18_7_reaction_attribution_hypothesis(
                            _m18_7_2_raw.get("reaction_attribution_hypothesis")
                        )
                    ),
                }
            except Exception as _m18_7_2_exc:
                # M12-pre pattern: degraded fallback, do NOT
                # crash run_turn. Emit a degraded bus event so
                # diagnose can distinguish a graceful degraded
                # path from a crash.
                bus.append(_build_m18_7_2_minimal_degraded_event(
                    turn_index=turn_index,
                    reason=repr(_m18_7_2_exc),
                    at=now,
                ))
        conscious_system, conscious_user = build_conscious_loop_prompt(
            state=state,
            user_text=user_text,
            speaker_name=display_name,
            bus_messages=bus,
            turn_index=turn_index,
            temporal_input=temporal_input,
            entity_binding=entity_binding,
        )
        conscious = normalize_conscious_turn_plan(
            _complete_json_stage("conscious_loop", conscious_system, conscious_user)
        )
        for event in apply_conscious_self_expectation_proposals(
            state,
            conscious_plan=conscious,
            now=now,
            turn_index=turn_index,
        ):
            bus.append(event)
        # M18.7.2 — emit the M18_7_2_* bus events, append the
        # bounded state surface entries (with
        # `source: "m18_7_2_minimal"` stamped on each). The
        # orchestrator reuses `normalize_*` / `build_state_entry`
        # / `record_m18_7_attribution_hypotheses` unchanged;
        # M20.4 producer and M18.7.1 calibration runner read the
        # same surface and "just work". The conscious-loop
        # `addressee_hypothesis` / `reaction_attribution_hypothesis`
        # fields are no longer requested by `build_conscious_loop_prompt`
        # (M18.7.2 is the sole source).
        _emit_m18_7_2_attribution_for_turn(
            bus=bus,
            state=state,
            plan=_m18_7_2_plan,
            turn_index=turn_index,
            at=now,
        )
        # M20.4.1 §1 — same-turn addressee hypothesis gate. Pure
        # rule, no LLM. Runs at "step 3" (immediately after the
        # M18.7 orchestrator, before the M20.4 producer and the
        # reply generation stages). When the rule fires, the gate
        # writes a single-slot override handoff
        # `state["m20_4_1_pending_override"]` that the M18.5
        # enforcement point (below) reads to replace the
        # `no_reply` / `clarify_addressee` force with
        # `reply_to_current_speaker`. The M18.5 structural
        # outcome is preserved in the audit envelope. fast_chat
        # safe (no LLM). Does NOT modify the M18.5 decision tree.
        _run_m20_4_1_same_turn_gate(
            conscious_plan=conscious,
            group_turn_binding=dict(bounded_group_turn)
            if bounded_group_turn
            else None,
            m18_5_structural_decision=str(
                group_reply_policy.get("action", "") or ""
            ),
            bus=bus,
            state=state,
            turn_index=turn_index,
            now=now,
        )
        # M20.4 §2 — M18.7 → M20 attribution commitment bridge.
        # Reads `state["m18_7_attribution_hypotheses"]` and admits
        # `ActiveCommitment` rows on `group_addressee_graph` per the
        # v1 rule (confidence >= 0.4 AND participant_id != "").
        # Empty M18.7 surface → silent no-op. The admitted rows go
        # through the existing M20.0 / M20.1 / M20.2 / M20.2.1
        # pipeline; the dispatcher routes to the real
        # `group_addressee_graph.microadjust` write path that was
        # no-op in M20.3.
        m20_4_attribution_admitted: list[ActiveCommitment] = (
            _admit_m20_4_attribution_commitments(
                bus=bus,
                state=state,
                current_turn_id=turn_index,
                inbound_excerpt=user_text,
                group_turn_binding=dict(bounded_group_turn)
                if bounded_group_turn
                else None,
                at=now,
            )
        )
        for commitment in m20_4_attribution_admitted:
            event = _emit_addressee_target_match_admitted_event(
                turn_index=turn_index,
                commitment=commitment,
                at=now,
            )
            if event:
                bus.append(event)
        # M20.3 §1 — second PolicyProducer call. The conscious loop
        # fills `correcting_assistant_identity`; we forward it as
        # the bounded `user_correction_signal`. If the conscious
        # loop ran in fast_chat and produced an empty signal, this
        # call emits no identity-correction rows. The next turn
        # re-evaluates (M20.3 §3.1a).
        post_conscious_signal = str(
            conscious.get("correcting_assistant_identity", "") or ""
        )
        post_conscious_admitted: list[ActiveCommitment] = _admit_policy_commitments(
            bus=bus,
            state=state,
            producer=policy_producer,
            turn_index=turn_index,
            at=now,
            runtime_mode_flags=_build_runtime_mode_flags(
                latency_mode=latency_mode,
                surface_intent=str(
                    bounded_group_turn.get("surface_intent", "") or ""
                ),
                group_mode_ingress_change=bool(
                    bounded_group_turn.get("group_mode_ingress_change", False)
                ),
            ),
            command_envelope=_build_command_envelope(bounded_group_turn),
            user_correction_signal=post_conscious_signal,
        )
        _admit_active_commitments(
            bus=bus,
            state=state,
            conscious_plan=conscious,
            turn_index=turn_index,
            now=now,
        )
        # M20.3 §4 — runtime invariant. Audit-only, runs after both
        # PolicyProducer calls and the M20.0 conscious-loop
        # admission. Reads the union of all T0-admitted
        # commitments.
        _enforce_minimum_loop(
            bus=bus,
            state=state,
            invariants=loop_invariants,
            turn_index=turn_index,
            at=now,
            proposed_commitments=(
                list(pre_conscious_admitted)
                + list(post_conscious_admitted)
                + list(conscious.get("active_commitment_proposals") or [])
            ),
            surface_intent=str(
                bounded_group_turn.get("surface_intent", "") or ""
            ),
            is_external_turn=bool(not proactive_turn),
        )
        _settle_active_commitments(
            bus=bus,
            state=state,
            conscious_plan=conscious,
            turn_index=turn_index,
            now=now,
        )
        _dispatch_graded_corrections(
            bus=bus,
            state=state,
            turn_index=turn_index,
            now=now,
        )
        # M20.4 v1 — emit the gated M18.5 tie-breaker feedback row
        # for any newly-dispatched M20.4 commitments. The feedback
        # row is the T+1+ cross-turn path (M20.4 v1 ships
        # cross-turn only; same-turn is M20.4.1 territory).
        _emit_m20_4_tie_breaker_feedback_for_turn(
            bus=bus,
            state=state,
            turn_index=turn_index,
            now=now,
        )
        _report_progress("conscious_loop")
        m13_state, m13_memory_efe_settlement_events = settle_memory_efe_outreach(
            m13_state,
            conscious_plan=conscious,
            turn_index=turn_index,
            now=now,
        )
        state["m13_drive_state"] = m13_state
        for m13_memory_efe_settlement_event in m13_memory_efe_settlement_events:
            bus.append(m13_memory_efe_settlement_event)
        for addendum in self._record_episode_settlements(
            state,
            m13_memory_efe_settlement_events,
            now=now,
            turn_index=turn_index,
        ):
            bus.append(addendum)
        memory_dynamics = build_memory_dynamics_guidance(
            state,
            user_text,
            conscious,
            bus,
            temporal_input,
            now,
            user_id=user_id,
            speaker_name=display_name,
            group_turn_binding=group_turn_binding,
        )
        recall_query = _mapping(memory_dynamics.get("recall_query"))
        if entity_binding.get("target_person"):
            recall_query["semantic_terms"] = _unique_strings(
                recall_query.get("semantic_terms"),
                [entity_binding.get("target_person")],
                list(_mapping(entity_binding.get("pronoun_bindings")).values()),
                limit=48,
            )
            recall_query["entity_binding"] = entity_binding
            memory_dynamics["recall_query"] = recall_query
        query_plan: dict[str, Any] = {}
        should_run_query_planner = _should_run_query_planner(
            state,
            user_text=user_text,
            recall_query=recall_query,
            entity_binding=entity_binding,
        )
        if latency_mode == "fast_chat":
            _mark_llm_skipped("query_planner", "latency_fast_path")
            should_run_query_planner = False
        elif not should_run_query_planner:
            _mark_llm_skipped("query_planner", "cadence_not_due")
        if should_run_query_planner:
            try:
                planner_system, planner_user = build_query_planner_prompt(
                    user_text=user_text,
                    speaker_name=display_name,
                    recall_query=recall_query,
                    temporal_input=temporal_input,
                    entity_binding=entity_binding,
                )
                query_plan = _normalize_query_plan(
                    _complete_json_stage("query_planner", planner_system, planner_user)
                )
                recall_query = _merge_query_plan_into_recall_query(recall_query, query_plan)
                memory_dynamics["recall_query"] = recall_query
            except Exception as exc:
                query_plan = {"planner_error": type(exc).__name__}
        _report_progress("query_planner")
        lexical_candidates = lexical_recall_short_term_candidates(
            state,
            user_text=user_text,
            recall_query=recall_query,
            current_user_id=user_id,
            entity_binding=entity_binding,
            group_turn_binding=group_turn_binding,
        )
        evidence_judgment: dict[str, Any] = {}
        if latency_mode == "fast_chat":
            _mark_llm_skipped("evidence_judge", "latency_fast_path")
        elif not lexical_candidates:
            _mark_llm_skipped("evidence_judge", "no_lexical_candidates")
        if lexical_candidates and latency_mode != "fast_chat":
            try:
                judge_system, judge_user = build_evidence_judge_prompt(
                    user_text=user_text,
                    speaker_name=display_name,
                    current_user_id=user_id,
                    lexical_candidates=lexical_candidates,
                    recall_query=recall_query,
                    entity_binding=entity_binding,
                )
                evidence_judgment = _normalize_evidence_judgment(
                    _complete_json_stage("evidence_judge", judge_system, judge_user),
                    lexical_candidates=lexical_candidates,
                    current_user_id=user_id,
                )
            except Exception as exc:
                evidence_judgment = {
                    "epistemic_stance": "uncertain_recall",
                    "relevant_evidence_ids": [str(item.get("id", "")) for item in lexical_candidates[:3] if item.get("id")],
                    "topics": sorted({topic for item in lexical_candidates for topic in _string_list(item.get("topics"), limit=8)}),
                    "sensitivity_class": "public",
                    "redaction_targets": [],
                    "allowed_reply_actions": ["direct_share", "abstract_share", "truthful_refusal", "deflect", "deny_knowledge"],
                    "audience_user_id": user_id,
                    "judge_error": type(exc).__name__,
                    "judge_summary": "evidence judge failed; candidates are passed as uncertain recall",
                }
        _apply_evidence_judgment_contract(memory_dynamics, evidence_judgment)
        group_privacy_policy = _resolve_group_privacy_policy(
            evidence_judgment=evidence_judgment,
            lexical_candidates=lexical_candidates,
            group_turn_binding=group_turn_binding,
        )
        group_mode = str(group_privacy_policy.get("selected_disclosure_mode", "") or "direct_quote").strip()
        if group_mode == "refusal":
            evidence_judgment["allowed_reply_actions"] = ["truthful_refusal", "deflect", "clarify"]
        elif group_mode in {"attributed_summary", "unattributed_abstraction"}:
            evidence_judgment["allowed_reply_actions"] = ["abstract_share", "truthful_refusal", "deflect", "clarify"]
        control = _mapping(memory_dynamics.get("control_guidance"))
        sharing_policy = _mapping(control.get("sharing_policy"))
        sharing_policy["group_privacy_policy"] = dict(group_privacy_policy)
        sharing_policy["allow_direct_disclosure"] = bool(
            sharing_policy.get("allow_direct_disclosure", True)
            and group_privacy_policy.get("allow_direct_disclosure", True)
        )
        sharing_policy["allow_abstract_sharing"] = bool(
            sharing_policy.get("allow_abstract_sharing", True)
            and group_privacy_policy.get("allow_abstract_sharing", True)
        )
        control["sharing_policy"] = sharing_policy
        reply_contract = _mapping(control.get("reply_contract"))
        redaction_targets = _unique_bounded_strings(
            [
                *_string_list(reply_contract.get("redaction_targets"), limit=12),
                *_string_list(group_privacy_policy.get("redaction_targets"), limit=12),
            ],
            limit=16,
            item_max_chars=80,
        )
        if redaction_targets:
            reply_contract["redaction_targets"] = redaction_targets
        reply_contract["allowed_reply_actions"] = _string_list(
            evidence_judgment.get("allowed_reply_actions"),
            limit=8,
        )
        reply_contract["selected_disclosure_action"] = str(
            group_privacy_policy.get("selected_disclosure_action", reply_contract.get("selected_disclosure_action", "none"))
            or "none"
        )
        reply_contract["group_privacy_policy"] = dict(group_privacy_policy)
        control["reply_contract"] = reply_contract
        memory_dynamics["control_guidance"] = control
        _report_progress("evidence_judge")
        retrieved_ranked = retrieve_memories_for_guidance(
            state,
            recall_query,
            now=now,
            group_turn_binding=group_turn_binding,
        )
        if lexical_candidates:
            existing_ids = {str(item.get("id", "")) for item in retrieved_ranked if item.get("id")}
            retrieved_ranked = [
                *[item for item in lexical_candidates if str(item.get("id", "")) not in existing_ids],
                *retrieved_ranked,
            ][:8]
        recall_bridge = build_path_b_recall_bridge(
            state,
            recall_query,
            retrieved_items=retrieved_ranked,
            now=turn_index + 1,
            field_consumer_enabled=self.path_b_field_consumer_enabled,
        )
        if (
            recall_bridge.field_required
            and self.path_b_field_consumer_enabled
            and latency_mode == "fast_chat"
        ):
            reason_codes = _string_list(latency_mode_info.get("reason_codes"), limit=12)
            if "path_b_field_required" not in reason_codes:
                reason_codes.append("path_b_field_required")
            latency_mode_info = {
                **latency_mode_info,
                "mode": "normal",
                "reason_codes": reason_codes,
                "escalated_from": "fast_chat",
            }
            latency_mode = "normal"
            bus.append(
                {
                    "type": "PathBFieldLatencyEscalationEvent",
                    "turn_index": turn_index,
                    "from_mode": "fast_chat",
                    "to_mode": "normal",
                    "reason_code": "path_b_field_required",
                    "field_selected_action": recall_bridge.field_selected_action,
                }
            )
        retrieved = [dict(item) for item in recall_bridge.retrieved_items]
        merge_path_b_field_guidance(
            memory_dynamics,
            recall_bridge,
            field_consumer_enabled=self.path_b_field_consumer_enabled,
        )
        memory_dynamics["recall"] = {
            **_mapping(memory_dynamics.get("recall")),
            "retrieved": len(retrieved),
            "ids": [str(item.get("id", "")) for item in retrieved if item.get("id")],
            "lexical_candidate_ids": [str(item.get("id", "")) for item in lexical_candidates if item.get("id")],
            "query_plan": query_plan,
        }
        bus.append(
            {
                "type": "PathBRecallBridgeBuiltEvent",
                "turn_index": turn_index,
                "active_path_ids": list(recall_bridge.provenance_refs.get("active_path_ids", [])),
                "counterfactual_status": recall_bridge.counterfactual_status,
                "field_selected_action": recall_bridge.field_selected_action,
                "writeback_targets": dict(recall_bridge.writeback_targets),
            }
        )
        if recall_bridge.field_required:
            bus.append(
                {
                    "type": "PathBFieldDecisionEvent",
                    "turn_index": turn_index,
                    "counterfactual_status": recall_bridge.counterfactual_status,
                    "reply_strategy": recall_bridge.reply_strategy,
                    "field_selected_action": recall_bridge.field_selected_action,
                    "best_single_action": recall_bridge.counterfactual_audit.get("best_single_action", ""),
                    "naive_topk_action": recall_bridge.counterfactual_audit.get("naive_topk_action", ""),
                }
            )
        else:
            bus.append(
                {
                    "type": "PathBFieldSuppressedEvent",
                    "turn_index": turn_index,
                    "counterfactual_status": recall_bridge.counterfactual_status,
                    "field_selected_action": recall_bridge.field_selected_action,
                }
            )
        self._mark_recalled(state, retrieved, now)
        _report_progress("memory_recall")
        response_style_prior = _response_style_prior(state, retrieved)
        m11_result_dict: dict[str, Any] = {}
        m11_raw_payload: Mapping[str, Any] | None = None
        m11_extractor_issue_code = ""
        settlement_judgments: list[Mapping[str, Any]] = []
        open_predictions_for_settlement = [
            {
                "prediction_id": entry.prediction_id,
                "prediction_type": entry.prediction_type,
                "predicted_value_summary": entry.predicted_value_summary,
                "committed_confidence": entry.committed_confidence,
                "created_at_turn": entry.created_at_turn,
                "expires_after_turns": entry.expires_after_turns,
            }
            for entry in m11_state.prediction_ledger.pending_entries()
        ]
        if _m11_enabled_for_state(state) and not proactive_turn:
            if latency_mode == "fast_chat":
                _mark_llm_skipped("m17_settlement_assessor", "latency_fast_path")
            elif not open_predictions_for_settlement:
                _mark_llm_skipped("m17_settlement_assessor", "no_open_predictions")
            else:
                try:
                    assessor_system, assessor_user = build_m17_settlement_assessor_prompt(
                        open_predictions=open_predictions_for_settlement,
                        user_text=user_text,
                        speaker_name=display_name,
                    )
                    settlement_payload = _complete_json_stage(
                        "m17_settlement_assessor",
                        assessor_system,
                        assessor_user,
                    )
                    validated_settlement_payload = validate_extractor_output(
                        dict(settlement_payload) if isinstance(settlement_payload, Mapping) else {},
                        snapshot_prediction_ids={
                            str(row.get("prediction_id", "") or "")
                            for row in open_predictions_for_settlement
                        },
                        snapshot_hypothesis_ids=set(),
                        snapshot_judgment_ids=set(),
                    )
                    settlement_judgments = [
                        dict(row)
                        for row in validated_settlement_payload.get("prediction_judgments", [])
                        if isinstance(row, Mapping)
                    ]
                    bus.append(
                        {
                            "type": "M17SettlementAssessorEvent",
                            "turn_index": turn_index,
                            "assessed_prediction_count": len(open_predictions_for_settlement),
                            "judgment_count": len(settlement_judgments),
                            "engineering_proxy_label": "mvp_local_prediction_error",
                        }
                    )
                except Exception as exc:
                    bus.append(
                        {
                            "type": "M17SettlementAssessorEvent",
                            "turn_index": turn_index,
                            "assessed_prediction_count": len(open_predictions_for_settlement),
                            "judgment_count": 0,
                            "error": type(exc).__name__,
                            "reason_code": "assessor_error",
                            "engineering_proxy_label": "mvp_local_prediction_error",
                        }
                    )
        should_run_m11 = _m11_enabled_for_state(state) and not proactive_turn and latency_mode != "fast_chat"
        if not should_run_m11 and _m11_enabled_for_state(state) and not proactive_turn:
            skip_reason = "latency_fast_path" if latency_mode == "fast_chat" else "cadence_not_due"
            _mark_llm_skipped("m11_user_model", skip_reason)
            lock_event = _prediction_lock_skip_event(
                turn_index=turn_index,
                reason_code="fast_chat_skip" if latency_mode == "fast_chat" else "proposal_quota_empty",
            )
            bus.append(lock_event)
            _record_prediction_lock_event(state, lock_event)
            m11_state, m11_turn = run_m11_turn(
                m11_state,
                user_id=user_id,
                turn_id=turn_index + 1,
                current_turn_quotes={"q_current": user_text},
                last_turn_summaries=[],
                extractor=lambda _: noop_extraction(),
                settlement_judgments=(),
                allow_extractor_prediction_judgments=False,
                config=M11RuntimeConfig(m11_user_model_enabled=True, persona_kind="ui_chat"),
                legacy_memory_rows=[
                    *(state.get("short_term_memory", []) if isinstance(state.get("short_term_memory"), list) else []),
                    *(state.get("long_term_memory", []) if isinstance(state.get("long_term_memory"), list) else []),
                ],
            )
            _save_m11_state(state, user_id=user_id, m11_state=m11_state)
            m11_result_dict = {
                **m11_turn.to_dict(),
                "skipped": True,
                "skip_reason": "fast_chat_skip" if latency_mode == "fast_chat" else "proposal_quota_empty",
            }
            _update_prediction_lock_diagnostics(state, m11_state)
        if should_run_m11:
            def _extract_m11(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                nonlocal m11_raw_payload, m11_extractor_issue_code
                system_prompt, user_prompt = build_m11_extractor_prompt(
                    snapshot=snapshot,
                    speaker_name=display_name,
                )
                try:
                    raw_payload = _complete_json_stage("m11_user_model", system_prompt, user_prompt)
                except Exception as exc:
                    m11_extractor_issue_code = "extractor_stage_error"
                    raise ExtractorValidationError("m11 extractor stage error") from exc
                if not isinstance(raw_payload, Mapping):
                    m11_extractor_issue_code = "invalid_extractor_output"
                    raise ExtractorValidationError("m11 extractor output must be an object")
                m11_raw_payload = dict(raw_payload)
                m11_extractor_issue_code = _classify_m11_extractor_issue(payload=m11_raw_payload)
                if m11_extractor_issue_code == "snapshot_echo_top_level_fields":
                    raise ExtractorValidationError("snapshot echo top-level fields")
                if m11_extractor_issue_code == "empty_extractor_output":
                    raise ExtractorValidationError("empty extractor output")
                return raw_payload

            try:
                m11_state, m11_turn = run_m11_turn(
                    m11_state,
                    user_id=user_id,
                    turn_id=turn_index + 1,
                    current_turn_quotes={"q_current": user_text},
                    last_turn_summaries=[],
                    extractor=_extract_m11,
                    settlement_judgments=settlement_judgments,
                    allow_extractor_prediction_judgments=False,
                    config=M11RuntimeConfig(m11_user_model_enabled=True, persona_kind="ui_chat"),
                    legacy_memory_rows=[
                        *(state.get("short_term_memory", []) if isinstance(state.get("short_term_memory"), list) else []),
                        *(state.get("long_term_memory", []) if isinstance(state.get("long_term_memory"), list) else []),
                    ],
                )
                _save_m11_state(state, user_id=user_id, m11_state=m11_state)
                m11_result_dict = m11_turn.to_dict()
                locked_entries = [
                    entry
                    for entry in m11_state.prediction_ledger.entries
                    if entry.turn_id == turn_index + 1 and entry.event_kind == "prediction"
                ]
                if locked_entries:
                    register_prediction_provenance(
                        state,
                        prediction_ids=[entry.prediction_id for entry in locked_entries],
                        bridge_result=recall_bridge,
                        turn_index=turn_index,
                    )
                    lock_event = _prediction_lock_event(
                        turn_index=turn_index,
                        prediction_ids=[entry.prediction_id for entry in locked_entries],
                        max_committed_confidence=max(entry.committed_confidence for entry in locked_entries),
                    )
                    bus.append(lock_event)
                    _record_prediction_lock_event(state, lock_event)
                else:
                    lock_event = _prediction_lock_skip_event(
                        turn_index=turn_index,
                        reason_code="proposal_quota_empty",
                    )
                    bus.append(lock_event)
                    _record_prediction_lock_event(state, lock_event)
                _update_prediction_lock_diagnostics(state, m11_state)
            except (ExtractorValidationError, ValueError, TypeError) as exc:
                issue_code = (
                    m11_extractor_issue_code
                    if m11_extractor_issue_code and m11_extractor_issue_code != "invalid_extractor_output"
                    else _classify_m11_extractor_issue(
                        payload=dict(m11_raw_payload) if isinstance(m11_raw_payload, Mapping) else None,
                        error_detail=str(exc),
                    )
                )
                m11_result_dict = {
                    "enabled": True,
                    "fallback": "noop_extraction",
                    "error": type(exc).__name__,
                    "error_detail": str(exc),
                    "error_reason_code": issue_code,
                    "extractor_output_top_level_keys": (
                        sorted(str(key) for key in m11_raw_payload.keys())
                        if isinstance(m11_raw_payload, Mapping)
                        else []
                    ),
                    "prompt_safe_evidence_cards": [],
                    "reply_policy_effects": [],
                }
                lock_event = _prediction_lock_skip_event(
                    turn_index=turn_index,
                    reason_code="extractor_invalid_output",
                )
                bus.append(lock_event)
                _record_prediction_lock_event(state, lock_event)
            _merge_m11_into_memory_guidance(
                memory_dynamics,
                speaker_name=display_name,
                m11_result=m11_result_dict,
            )
        if m12_pre_result is not None:
            _merge_m12_into_memory_guidance(
                memory_dynamics,
                m12_result=m12_pre_result,
            )
        m12_1_result_dict: dict[str, Any] = {}
        if _m12_1_enabled_for_state(state) and latency_mode != "fast_chat":
            m12_1_state = _load_m12_1_state(state)
            m12_summary_for_personality: dict[str, object] = {}
            if m12_pre_result:
                m12_summary_for_personality = {
                    **_mapping(m12_pre_result.get("entity_binding_context")),
                    "new_evidence_count": len(_mapping(m12_pre_result.get("state_after")).get("conflict_records", []) or []),
                }

            def _extract_m12_1_step(step: int):
                def _extract(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                    system_prompt, user_prompt = build_step_extractor_prompt(step, snapshot)
                    try:
                        return _complete_json_stage(f"m12_1_step_{step}", system_prompt, user_prompt)
                    except Exception:
                        return {"status": "insufficient_evidence", "reason": f"step_{step}_llm_error"}

                return _extract

            m12_1_state, m12_1_turn = run_m12_1_tick(
                m12_1_state,
                user_id=user_id,
                display_name=display_name,
                turn_id=turn_key,
                turn_index=turn_index + 1,
                hour_bucket=now // 3600,
                current_turn_quotes={"q_current": user_text},
                transcript_quote_refs=(),
                m11_readonly_summary={
                    "m11_evidence_cards": list(m11_result_dict.get("prompt_safe_evidence_cards", [])),
                },
                m12_readonly_summary=m12_summary_for_personality,
                extractors={step: _extract_m12_1_step(step) for step in range(1, 9)},
                config=M121RuntimeConfig(m12_1_personality_enabled=True, persona_kind="ui_chat"),
                session_id=str(self.store.root.resolve()),
                persona_id=self.persona_name or "default",
                cycle=turn_index,
                event_sequence_index=1,
            )
            _save_m12_1_state(state, m12_1_state=m12_1_state)
            m12_1_result_dict = m12_1_turn.to_dict()
            _merge_m12_1_into_memory_guidance(
                memory_dynamics,
                m12_1_result=m12_1_result_dict,
            )
        elif _m12_1_enabled_for_state(state):
            for step in range(1, 9):
                _mark_llm_skipped(f"m12_1_step_{step}", "latency_fast_path")

        relationship_value_context = resolve_relationship_value_context(
            state,
            user_id,
            user_text,
        )
        m12_2_result_dict: dict[str, Any] = {}
        m12_2_enabled = _m12_2_enabled_for_state(state) and not proactive_turn
        m12_2_should_run, m12_2_skip_reason = _m12_2_latency_triggered(
            latency_mode=latency_mode,
            user_text=user_text,
            relationship_value_context=relationship_value_context,
        )
        m12_2_run_this_turn = m12_2_enabled and m12_2_should_run
        if m12_2_enabled and not m12_2_run_this_turn:
            _mark_llm_skipped("m12_2_first_order", m12_2_skip_reason)
            _mark_llm_skipped("m12_2_second_order", m12_2_skip_reason)
        if m12_2_run_this_turn:
            m12_2_state = _load_m12_2_state(state)

            def _extract_m12_2(name: str):
                def _extract(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                    system_prompt, user_prompt = build_m12_2_extractor_prompt(name, snapshot)
                    try:
                        return _complete_json_stage(f"m12_2_{name}", system_prompt, user_prompt)
                    except Exception:
                        if name == "first_order":
                            return {
                                "persona_about_user_claims": [],
                                "claim_group_updates": [],
                                "unresolved_uncertainty_points": [],
                                "high_gain_candidates": [],
                                "insufficient_evidence": True,
                            }
                        return {
                            "user_about_persona_claims": [],
                            "claim_group_updates": [],
                            "inferred_user_uncertainties_about_persona": [],
                            "clarifying_reply_candidates": [],
                            "insufficient_evidence": True,
                        }

                return _extract

            m12_2_event_start = len(m12_cognitive_bus.events())
            m12_2_state, m12_2_turn = run_m12_2_tick(
                m12_2_state,
                user_id=user_id,
                turn_id=turn_key,
                turn_index=turn_index + 1,
                hour_bucket=now // 3600,
                user_text=user_text,
                current_turn_quotes={"q_current": user_text},
                transcript_quote_refs=(),
                m11_readonly_summary={
                    "m11_evidence_cards": list(m11_result_dict.get("prompt_safe_evidence_cards", [])),
                },
                m12_readonly_summary=m12_pre_result or {},
                m121_readonly_summary=m12_1_result_dict,
                relationship_value_memories=relationship_value_context.get("active_relationship_value_memories", []),
                extractors={"first_order": _extract_m12_2("first_order"), "second_order": _extract_m12_2("second_order")},
                config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True, persona_kind="ui_chat"),
                session_id=str(self.store.root.resolve()),
                persona_id=self.persona_name or "default",
                cycle=turn_index,
                event_sequence_index=2,
                event_bus=m12_cognitive_bus,
            )
            _save_m12_2_state(state, m12_2_state=m12_2_state)
            m12_2_result_dict = m12_2_turn.to_dict()
            for seq_idx, ev in enumerate(m12_cognitive_bus.events()[m12_2_event_start:], start=m12_2_event_start):
                bus.append({
                    "type": ev.event_type,
                    "turn_index": turn_index,
                    "sequence": seq_idx,
                    "cognitive_event": ev.to_dict(),
                })
            _merge_m12_2_into_memory_guidance(
                memory_dynamics,
                m12_2_result=m12_2_result_dict,
            )

        if not m12_2_run_this_turn:
            _apply_relationship_value_context_to_memory_dynamics(
                memory_dynamics,
                relationship_value_context,
            )
        _merge_surface_identity_contract_into_memory_guidance(
            memory_dynamics,
            persona_name=self.persona_name,
            group_turn_binding=group_turn_binding,
            conscious_plan=conscious,
        )

        _report_progress("user_modeling")
        m13_evaluator = M13DriveEvaluator()
        m15_state_fingerprint_pre = self._current_state_fingerprint(state)
        m13_evaluation = m13_evaluator.evaluate(
            user_text=user_text,
            user_id=user_id,
            turn_id=turn_key,
            turn_index=turn_index,
            conscious_plan=conscious,
            memory_dynamics=memory_dynamics,
            retrieved_memories=retrieved,
            response_style_prior=response_style_prior,
            habit_traits=_mapping(state.get("habit_traits")),
            relationship_value_context=relationship_value_context,
            m13_state=m13_state,
            entity_binding=entity_binding,
            evidence_judgment=evidence_judgment,
            episode_ledger=episode_ledger,
            current_state_fingerprint=m15_state_fingerprint_pre,
        )
        for m13_event in m13_evaluation.events:
            bus.append(m13_event)
        m13_boredom_evaluator = M13BoredomEvaluator()
        boredom_assessor_llm = self.llm
        reply_contract = _mapping(_mapping(memory_dynamics.get("control_guidance")).get("reply_contract"))
        if latency_mode == "fast_chat":
            boredom_assessor_llm = None
        if (
            str(reply_contract.get("conversation_mode", "")) == "casual_fast"
            and not retrieved
        ):
            boredom_assessor_llm = None
        m13_boredom_evaluation = m13_boredom_evaluator.evaluate(
            user_text=user_text,
            user_id=user_id,
            turn_id=turn_key,
            turn_index=turn_index,
            conscious_plan=conscious,
            memory_dynamics=memory_dynamics,
            retrieved_memories=retrieved,
            m13_state=m13_state,
            m13_drive_evaluation=m13_evaluation,
            entity_binding=entity_binding,
            evidence_judgment=evidence_judgment,
            m11_result=m11_result_dict or None,
            m12_payload=m12_pre_result,
            m12_2_result=m12_2_result_dict if m12_2_run_this_turn else None,
            llm=boredom_assessor_llm,
        )
        for m13_boredom_event in m13_boredom_evaluation.events:
            bus.append(m13_boredom_event)
        merge_drive_guidance_into_control(
            memory_dynamics,
            m13_evaluation,
            evidence_judgment=evidence_judgment,
            boredom_evaluation=m13_boredom_evaluation,
        )
        control_for_reward = _mapping(memory_dynamics.get("control_guidance"))
        m13_reward_pre_turn = evaluate_pre_turn_reward_proxy(
            turn_id=turn_key,
            turn_index=turn_index,
            user_id=user_id,
            m13_state=m13_state,
            m13_evaluation=m13_evaluation,
            information_gain_proxy=m13_boredom_evaluation.information_gain_proxy,
            repetition_pressure=m13_boredom_evaluation.repetition_pressure,
            conflict_level=_bounded_float(control_for_reward.get("conflict_level")),
        )
        for m13_reward_event in m13_reward_pre_turn.events:
            bus.append(m13_reward_event)
        merge_affective_guidance_into_control(memory_dynamics, m13_reward_pre_turn)
        m13_memory_efe_evaluation = evaluate_memory_efe(
            state,
            phase="in_turn",
            now=now,
            turn_index=turn_index,
            user_active=True,
            memory_dynamics=memory_dynamics,
            retrieved_memories=retrieved,
            m13_boredom_evaluation=m13_boredom_evaluation,
            m13_reward_evaluation=m13_reward_pre_turn,
            conscious_plan=conscious,
            episode_ledger=episode_ledger,
        )
        m13_state, _m13_memory_efe_apply_events = apply_memory_efe_state(
            m13_state,
            m13_memory_efe_evaluation,
        )
        state["m13_drive_state"] = m13_state
        for m13_memory_efe_event in m13_memory_efe_evaluation.events:
            bus.append(m13_memory_efe_event)
            if str(m13_memory_efe_event.get("type", "")).startswith("BundleDecision"):
                self._record_bundle_policy_event(state, m13_memory_efe_event)
        state["bundle_policy_linkage_diagnostics"] = dict(m13_memory_efe_evaluation.bundle_linkage_diagnostics)
        merge_memory_efe_guidance_into_control(memory_dynamics, m13_memory_efe_evaluation)
        _report_progress("m13_eval")

        thinking_system, thinking_user = build_thinking_prompt(
            state=_prompt_safe_state(state, user_id=user_id),
            user_text=user_text,
            speaker_name=display_name,
            conscious_plan=conscious,
            retrieved_memories=retrieved,
            turn_index=turn_index,
            response_style_prior=response_style_prior,
            entity_binding=entity_binding,
            memory_guidance={
                "memory_value": memory_dynamics.get("memory_value", {}),
                "recall": memory_dynamics.get("recall", {}),
                "recall_bridge": memory_dynamics.get("recall_bridge", {}),
                "control_guidance": prompt_safe_control_guidance_for_thinking(
                    memory_dynamics.get("control_guidance", {})
                ),
                "write_candidates": memory_dynamics.get("write_candidates", []),
                "expectation_impact": memory_dynamics.get("expectation_impact", {}),
                "evidence_judgment": evidence_judgment,
                "query_plan": query_plan,
                "entity_binding": entity_binding,
            },
        )
        thinking = _complete_json_stage("thinking_reply", thinking_system, thinking_user)
        _report_progress("thinking_reply")

        self._apply_expectation_results(
            state,
            conscious.get("expectation_results"),
            now=now,
            turn_index=turn_index,
            user_id=user_id,
            display_name=display_name,
            entity_binding=entity_binding,
        )
        self._apply_thinking_writes(
            state,
            thinking,
            user_text=user_text,
            now=now,
            turn_index=turn_index,
            user_id=user_id,
            display_name=display_name,
            session_id=session_id,
            ingress_evidence_band=ingress_band,
            explicit_secrecy=bool(_mapping(_mapping(memory_dynamics.get("control_guidance")).get("sharing_policy")).get("explicit_secrecy_detected")),
            memory_dynamics=memory_dynamics,
            group_turn_binding=group_turn_binding,
        )
        memory_candidates_applied = self._apply_memory_write_candidates(
            state,
            memory_dynamics.get("write_candidates"),
            now=now,
            turn_index=turn_index,
            user_id=user_id,
            display_name=display_name,
            session_id=session_id,
            ingress_evidence_band=ingress_band,
            default_shareability=(
                "restricted_explicit"
                if bool(_mapping(_mapping(memory_dynamics.get("control_guidance")).get("sharing_policy")).get("explicit_secrecy_detected"))
                else "default_social"
            ),
            restriction_reason=(
                "explicit_user_secret"
                if bool(_mapping(_mapping(memory_dynamics.get("control_guidance")).get("sharing_policy")).get("explicit_secrecy_detected"))
                else ""
            ),
            group_turn_binding=group_turn_binding,
        )
        habit_updates_applied = _apply_habit_updates(
            state,
            thinking,
            user_id=user_id,
            display_name=display_name,
            now=now,
            turn_index=turn_index,
            session_id=session_id,
            ingress_evidence_band=ingress_band,
            default_shareability=(
                "restricted_explicit"
                if bool(_mapping(_mapping(memory_dynamics.get("control_guidance")).get("sharing_policy")).get("explicit_secrecy_detected"))
                else "default_social"
            ),
            group_turn_binding=group_turn_binding,
        )

        raw_reply = str(thinking.get("reply") or "").strip()
        if not raw_reply:
            raw_reply = "我需要想一下这个。"
        control_guidance = _mapping(memory_dynamics.get("control_guidance"))
        reply_contract = _mapping(control_guidance.get("reply_contract"))
        raw_reply, enforced_reply_action, path_b_field_enforcement_actions = _enforce_path_b_field_reply_contract(
            reply=raw_reply,
            reply_action=str(thinking.get("reply_action") or "answer"),
            reply_contract=reply_contract,
        )
        if path_b_field_enforcement_actions:
            thinking["reply"] = raw_reply
            thinking["reply_action"] = enforced_reply_action
            thinking["path_b_field_enforcement_actions"] = list(path_b_field_enforcement_actions)
        reply_contract["identity_anchored_action"] = identity_anchored_action
        forced_disclosure_action = str(reply_contract.get("selected_disclosure_action", "") or "").strip()
        if forced_disclosure_action in {"truthful_refusal", "abstract_share"}:
            reply_contract["selected_disclosure_action"] = forced_disclosure_action
        else:
            reply_contract["selected_disclosure_action"] = str(thinking.get("disclosure_action", "none") or "none")
        recorded_action = normalize_recorded_reply_action(
            enforced_reply_action if path_b_field_enforcement_actions else str(thinking.get("reply_action") or "answer"),
            allowed=set(m13_evaluation.candidate_actions),
        )
        action = recorded_action
        reply = raw_reply
        surface_commitment = _mapping(reply_contract.get("surface_commitment"))
        surface_consistency_verification = _empty_surface_consistency_verification()
        surface_consistency_audit_event: dict[str, Any] | None = None
        if surface_commitment and latency_mode != "fast_chat":
            try:
                verify_system, verify_user = build_surface_consistency_verification_prompt(
                    user_text=user_text,
                    draft_reply=raw_reply,
                    surface_commitment=surface_commitment,
                    reply_contract=reply_contract,
                    turn_index=turn_index,
                )
                verify_payload = _complete_json_stage(
                    "surface_consistency_verification", verify_system, verify_user
                )
                surface_consistency_verification = normalize_surface_consistency_verification(verify_payload)
            except Exception as exc:
                surface_consistency_verification = _empty_surface_consistency_verification(
                    reason=f"llm_error:{type(exc).__name__}"
                )
            surface_consistency_audit_event = _build_surface_consistency_verification_event(
                turn_index=turn_index,
                verification=surface_consistency_verification,
                commitment=surface_commitment,
            )
            bus.append(surface_consistency_audit_event)
            _record_surface_consistency_event(state, surface_consistency_audit_event)
            reply_contract["surface_consistency_verification"] = dict(surface_consistency_verification)
        else:
            reply_contract["surface_consistency_verification"] = dict(surface_consistency_verification)
            if surface_commitment and latency_mode == "fast_chat":
                # P0-1 — fast_chat minimal pre-send verify. When a
                # blockable `runtime_mode_state` commitment with
                # `expected_mode` is present, run a small bounded LLM
                # call so the M20.3 §3.2 pre-send gate can read a
                # real audit row (instead of `audit_absent` →
                # `ambiguous` advisory). On LLM failure, fall back
                # to the prior `ambiguous` behavior (gate sees no
                # audit row).
                surface_consistency_verification, p01_events = (
                    _run_fast_chat_pre_send_minimal(
                        state=state,
                        surface_commitment=surface_commitment,
                        raw_reply=raw_reply,
                        user_text=user_text,
                        turn_index=turn_index,
                        complete_json_stage=_complete_json_stage,
                    )
                )
                reply_contract["surface_consistency_verification"] = dict(
                    surface_consistency_verification
                )
                for ev in p01_events:
                    bus.append(ev)
                    _record_surface_consistency_event(state, ev)
                if not p01_events:
                    # No LLM call ran (no blockable commitment). Emit
                    # the prior `latency_fast_path` skip event so the
                    # audit trail is unchanged.
                    surface_consistency_audit_event = {
                        "type": "SurfaceConsistencyVerificationSkippedEvent",
                        "turn_index": turn_index,
                        "reason_code": "latency_fast_path",
                        "committed_surface_intent": str(surface_commitment.get("surface_intent", "chat") or "chat"),
                        "engineering_proxy_label": "mvp_local_surface_consistency_audit",
                    }
                    bus.append(surface_consistency_audit_event)
                    _record_surface_consistency_event(state, surface_consistency_audit_event)
        reply_validation: dict[str, Any] = {
            "original_length": len(raw_reply),
            "final_length": len(raw_reply),
            "conversation_mode": str(reply_contract.get("conversation_mode") or reply_contract.get("reply_pacing") or "balanced"),
            "max_chars": _positive_int(reply_contract.get("max_chars"), default=140),
            "max_sentences": _positive_int(reply_contract.get("max_sentences"), default=2),
            "changed": False,
            "actions": [],
            "allow_direct_disclosure": bool(reply_contract.get("allow_direct_disclosure", True)),
            "explicit_secrecy_detected": bool(reply_contract.get("explicit_secrecy_detected", False)),
            "selected_disclosure_action": str(reply_contract.get("selected_disclosure_action", "none") or "none"),
            "redaction_targets": _string_list(reply_contract.get("redaction_targets"), limit=12),
            "identity_anchored_action": bool(reply_contract.get("identity_anchored_action", False)),
            "surface_consistency_verification": dict(surface_consistency_verification),
        }
        group_policy_actions: list[str] = []
        # M20.4.1 §4 — read the same-turn gate override handoff.
        # The gate wrote this slot earlier in the turn (after the
        # conscious loop). When set, we override the M18.5
        # `no_reply` / `clarify_addressee` force to
        # `reply_to_current_speaker` (the visible reply gets a
        # bounded T0 patch). The slot is cleared immediately so it
        # does not leak to T+1. M18.5's structural outcome is
        # preserved in the gate's audit envelope (the
        # `m18_5_structural_decision` field) for diagnose.
        m20_4_1_override = _m20_4_1_get_pending_override(state)
        repair_requirements, repair_reason_codes, forced_group_action = _reply_repair_requirements(
            reply_contract=reply_contract,
            group_privacy_policy=group_privacy_policy,
            group_reply_policy=group_reply_policy,
            thinking=thinking,
        )
        if forced_group_action:
            action = forced_group_action
        if forced_group_action == "truthful_refusal":
            reply = ""
            action = "no_reply"
            group_policy_actions.append("group_privacy_forced_refusal_silence")
        elif m20_4_1_override is not None:
            # M20.4.1 — same-turn override fired. The visible
            # reply is unblocked: action becomes
            # `reply_to_current_speaker`, and we keep `raw_reply`
            # (do NOT blank the reply text). The audit envelope
            # carries `m18_5_structural_decision` so the diagnose
            # surface can see the original M18.5 outcome. The
            # override wins over M18.5's `no_reply` /
            # `defer_side_thread` / `clarify_addressee` path; the
            # privacy forced_refusal above still wins.
            action = "reply_to_current_speaker"
            group_policy_actions.append(
                "m20_4_1_same_turn_override"
            )
        elif str(group_reply_policy.get("action", "") or "") == "defer_side_thread":
            reply = ""
            action = "defer_side_thread"
            group_policy_actions.append("group_reply_policy_forced_defer_side_thread")
        elif str(group_reply_policy.get("action", "") or "") == "no_reply":
            reply = ""
            action = "no_reply"
            group_policy_actions.append("group_reply_policy_forced_no_reply")
        else:
            if repair_requirements:
                try:
                    repair_system, repair_user = build_reply_repair_prompt(
                        user_text=user_text,
                        draft_reply=raw_reply,
                        thinking=thinking,
                        reply_contract=reply_contract,
                        reply_validation={"actions": repair_reason_codes},
                        requirements=repair_requirements,
                        target_action=action,
                        turn_index=turn_index,
                    )
                    repair_payload = _complete_json_stage("reply_repair", repair_system, repair_user)
                    repaired_reply = str(repair_payload.get("reply") or "").strip()
                    if repaired_reply:
                        raw_reply = repaired_reply
                        group_policy_actions.extend(repair_reason_codes)
                        group_policy_actions.append("llm_reply_repair_pre_validation")
                except Exception as exc:
                    group_policy_actions.extend(repair_reason_codes)
                    group_policy_actions.append(f"llm_reply_repair_pre_validation_error:{type(exc).__name__}")
            # M20.3 §3.2 — pre-send gate. Runs BEFORE
            # `validate_visible_reply` so a `block` decision can
            # replace the draft reply before the existing
            # surface-consistency validation runs on it. Reads
            # `state["m20_3_horizon_commitments"]` (all horizon =
            # "same_turn_surface" commitments admitted earlier in
            # the turn). Observation context passes the M19.x
            # `surface_consistency_verification` audit envelope.
            horizon_commitments = state.get("m20_3_horizon_commitments")
            if not isinstance(horizon_commitments, list):
                horizon_commitments = []
            observation_context = {
                "now": str(now),
                "turn_index": int(turn_index),
                "surface_consistency_verification": dict(
                    _mapping(reply_contract.get("surface_consistency_verification"))
                ),
            }
            _pre_verdict, _post_verdict, _replaced_reply = _run_same_turn_surface(
                bus=bus,
                state=state,
                settler=same_turn_surface_settler,
                horizon_commitments=[
                    c for c in horizon_commitments if isinstance(c, ActiveCommitment)
                ],
                draft_reply=raw_reply,
                committed_reply=raw_reply,
                observation_context=observation_context,
                turn_index=turn_index,
                at=str(now),
            )
            if _replaced_reply and _replaced_reply != raw_reply:
                raw_reply = _replaced_reply
                group_policy_actions.append("m20_3_pre_send_block_replacement")
            reply, reply_validation = validate_visible_reply(raw_reply, reply_contract)
            if path_b_field_enforcement_actions:
                reply_validation = dict(reply_validation)
                reply_validation["changed"] = True
                reply_validation["actions"] = [
                    *list(reply_validation.get("actions", [])),
                    *path_b_field_enforcement_actions,
                ]
            if bool(reply_validation.get("changed")):
                try:
                    repair_system, repair_user = build_reply_repair_prompt(
                        user_text=user_text,
                        draft_reply=reply,
                        thinking=thinking,
                        reply_contract=reply_contract,
                        reply_validation=reply_validation,
                        requirements=repair_requirements,
                        target_action=action,
                        turn_index=turn_index,
                    )
                    repair_payload = _complete_json_stage("reply_repair", repair_system, repair_user)
                    repaired_reply = str(repair_payload.get("reply") or "").strip()
                    if repaired_reply:
                        repaired_reply, repaired_validation = validate_visible_reply(repaired_reply, reply_contract)
                        reply = repaired_reply
                        reply_validation = {
                            **dict(repaired_validation),
                            "changed": True,
                            "actions": _unique_strings(
                                reply_validation.get("actions"),
                                ["llm_reply_repair_post_validation"],
                                repaired_validation.get("actions"),
                                limit=24,
                            ),
                        }
                except Exception as exc:
                    reply_validation = dict(reply_validation)
                    reply_validation["changed"] = True
                    reply_validation["actions"] = _unique_strings(
                        reply_validation.get("actions"),
                        [f"llm_reply_repair_post_validation_error:{type(exc).__name__}"],
                        limit=24,
                    )
        if group_policy_actions:
            reply_validation = dict(reply_validation)
            reply_validation["changed"] = True
            reply_validation["actions"] = _unique_strings(
                reply_validation.get("actions"),
                group_policy_actions,
                limit=24,
            )
        m13_selected_action = action if action in set(m13_evaluation.candidate_actions) else recorded_action
        temporal_assessment = conscious.get("temporal_assessment")
        if not isinstance(temporal_assessment, Mapping):
            temporal_assessment = {}
        post_reply_observer: dict[str, Any] = {"needs_followup": False, "followup_type": "none"}
        post_reply_observer_skipped_reason = ""
        followup_replies: list[str] = []
        should_observe, observer_reason = _should_run_post_reply_observer(
            user_text=user_text,
            memory_dynamics=memory_dynamics,
            reply_validation=reply_validation,
        )
        if action in {"no_reply", "defer_side_thread"}:
            should_observe = False
            observer_reason = "group_policy_silence"
        if latency_mode == "fast_chat" and should_observe:
            _mark_llm_skipped("post_reply_observer", "latency_fast_path")
            should_observe = False
            observer_reason = "latency_fast_path"
        elif latency_mode == "fast_chat":
            _mark_llm_skipped("post_reply_observer", "latency_fast_path")
        if should_observe:
            try:
                observer_system, observer_user = build_post_reply_observer_prompt(
                    user_text=user_text,
                    reply=reply,
                    thinking=thinking,
                    memory_dynamics=memory_dynamics,
                    retrieved_memories=retrieved,
                    temporal_assessment=temporal_assessment,
                    turn_index=turn_index,
                )
                observer_result = _complete_json_stage("post_reply_observer", observer_system, observer_user)
                post_reply_observer = dict(observer_result)
                post_reply_observer["trigger_reason"] = observer_reason
                followup_text = _validated_followup_text(post_reply_observer)
                if followup_text:
                    followup_replies.append(followup_text)
            except Exception as exc:
                post_reply_observer = {
                    "needs_followup": False,
                    "followup_type": "none",
                    "trigger_reason": observer_reason,
                    "observer_error": type(exc).__name__,
                    "observer_error_detail": str(exc),
                }
        else:
            post_reply_observer_skipped_reason = observer_reason
        _report_progress("post_reply_observer")
        post_reply_memory_updates_applied = self._apply_post_reply_memory_updates(
            state,
            post_reply_observer.get("memory_updates"),
            now=now,
            turn_index=turn_index,
            user_id=user_id,
            display_name=display_name,
            session_id=session_id,
            ingress_evidence_band=ingress_band,
            group_turn_binding=group_turn_binding,
        )
        pacing_feedback_habits_applied = self._apply_pacing_feedback_habit(
            state,
            user_text=user_text,
            user_id=user_id,
            display_name=display_name,
            now=now,
            turn_index=turn_index,
            session_id=session_id,
            ingress_evidence_band=ingress_band,
            group_turn_binding=group_turn_binding,
        )
        safety_repair = resolve_m13_safety_repair(
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
        )
        m13_state, m13_post_events = apply_post_turn_m13_state(
            m13_state,
            evaluation=m13_evaluation,
            user_id=user_id,
            turn_id=turn_key,
            turn_index=turn_index,
            selected_action=m13_selected_action,
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
            conscious_plan=conscious,
            memory_candidates_applied=memory_candidates_applied,
            safety_repair=safety_repair,
        )
        m13_state, m13_boredom_post_events = apply_post_turn_boredom_state(
            m13_state,
            boredom=m13_boredom_evaluation,
            conscious_plan=conscious,
            retrieved_memories=retrieved,
            turn_index=turn_index,
        )
        m13_reward_evaluator = M13RewardEvaluator()
        selected_pull = _bounded_float(
            m13_evaluation.scores_by_action.get(m13_selected_action, {}).get("behavioral_pull", 0.0)
        )
        m13_reward_evaluation = m13_reward_evaluator.evaluate(
            turn_id=turn_key,
            turn_index=turn_index,
            user_id=user_id,
            action=m13_selected_action,
            topic_fingerprint=m13_evaluation.topic_fingerprint,
            m13_state=m13_state,
            conscious_plan=conscious,
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
            memory_candidates_applied=memory_candidates_applied,
            evidence_judgment=evidence_judgment,
            safety_repair=safety_repair,
            information_gain_proxy=m13_boredom_evaluation.information_gain_proxy,
            repetition_pressure=m13_boredom_evaluation.repetition_pressure,
            conflict_level=_bounded_float(control_guidance.get("conflict_level")),
            behavioral_pull=selected_pull,
            evidence_refs=m13_evaluation.evidence_refs,
            relationship_value_context=relationship_value_context,
        )
        for m13_reward_event in m13_reward_evaluation.events:
            bus.append(m13_reward_event)
        m13_state, m13_reward_post_events = apply_post_turn_m13_reward_state(
            m13_state,
            evaluation=m13_reward_evaluation,
            user_id=user_id,
            action=m13_selected_action,
            topic_fingerprint=m13_evaluation.topic_fingerprint,
            turn_index=turn_index,
            reply_summary=reply[:160],
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
            conscious_plan=conscious,
            memory_candidates_applied=memory_candidates_applied,
            evidence_judgment=evidence_judgment,
            safety_repair=safety_repair,
            repetition_pressure=m13_boredom_evaluation.repetition_pressure,
            conflict_level=_bounded_float(control_guidance.get("conflict_level")),
            behavioral_pull=selected_pull,
        )
        m13_state = apply_reward_pull_connection(
            m13_state,
            evaluation=m13_reward_evaluation,
            behavioral_pull=selected_pull,
        )
        state["m13_drive_state"] = m13_state
        for m13_event in m13_post_events:
            bus.append(m13_event)
        for m13_boredom_event in m13_boredom_post_events:
            bus.append(m13_boredom_event)
        for m13_reward_event in m13_reward_post_events:
            bus.append(m13_reward_event)
        m19_post_turn = apply_self_expectation_post_turn(
            state,
            conscious_plan=conscious,
            control_guidance=control_guidance,
            reward_prediction_error_proxy=m13_reward_evaluation.prediction_error_proxy,
            reward_event_id=m13_reward_evaluation.event_id,
            now=now,
            turn_index=turn_index,
            group_turn_binding=group_turn_binding,
        )
        for event in m19_post_turn.events:
            bus.append(event)
        if m19_post_turn.traction_proposals:
            m13_state, m19_traction_events = apply_m19_traction_proposals_to_m13(
                m13_state,
                m19_post_turn.traction_proposals,
                user_id=user_id,
                topic_fingerprint=m13_evaluation.topic_fingerprint,
                turn_index=turn_index,
            )
            state["m13_drive_state"] = m13_state
            for event in m19_traction_events:
                bus.append(event)
        if isinstance(m19_post_turn.slow_patch_proposal, Mapping):
            retrieved_ids = {
                str(item.get("id", ""))
                for item in retrieved
                if item.get("id")
            } | collect_m19_audit_evidence_ids(state)
            m19_patch_result = SelfCognitionPatchOwner.validate_and_commit(
                state,
                m19_post_turn.slow_patch_proposal,
                retrieved_ids=retrieved_ids,
                turn_index=turn_index,
                now=now,
                session_patches=int(count_session_idle_patches(state).get("self_cognition", 0)),
            )
            for event in m19_patch_result.events:
                tagged = dict(event)
                tagged.setdefault("engineering_proxy_label", M19_ENGINEERING_PROXY_LABEL)
                bus.append(tagged)
        visible_reply = "\n".join([reply, *followup_replies])
        sharing_policy = _mapping(control_guidance.get("sharing_policy"))
        temporal_user_text = prior_last_user_text if proactive_turn else user_text
        thread_policy_state = _build_group_thread_policy_state(
            previous_group_chat_state=previous_group_chat_state,
            group_turn_binding=group_turn_binding,
            group_reply_policy=group_reply_policy,
            now=now,
            turn_index=turn_index,
        )
        group_chat_state = _build_group_chat_state(
            state,
            now=now,
            turn_index=turn_index,
            display_name=display_name,
            user_id=user_id,
            group_turn_envelope=bounded_group_turn,
            group_turn_binding=group_turn_binding,
            thread_policy_state=thread_policy_state,
        )
        _update_temporal_state(
            state,
            now=now,
            turn_index=turn_index,
            user_text=temporal_user_text,
            reply=visible_reply,
            temporal_input=temporal_input,
            group_chat_state=group_chat_state,
            proactive_turn=proactive_turn,
            share_trace={
                "user_id": user_id,
                "speaker_name": display_name,
                "speaker_participant_id": group_turn_binding.get("current_speaker_participant_id", ""),
                "allow_direct_disclosure": bool(sharing_policy.get("allow_direct_disclosure", True)),
                "allow_abstract_sharing": bool(sharing_policy.get("allow_abstract_sharing", True)),
                "net_free_energy_reduction": _bounded_float(sharing_policy.get("net_free_energy_reduction"), default=0.0),
                "had_cross_user_memory": any(
                    bool(str(item.get("source_user_id", "")).strip())
                    and str(item.get("source_user_id", "")).strip() != user_id
                    for item in retrieved
                ),
                "lexical_recall_terms": _lexical_recall_terms(
                    state=state,
                    user_text=user_text,
                    recall_query=recall_query,
                    entity_binding=entity_binding,
                    limit=24,
                ),
                "target_person": entity_binding.get("target_person", "") or _mapping(_mapping(state.get("temporal_state")).get("last_share_trace")).get("target_person", ""),
                "pronoun_bindings": entity_binding.get("pronoun_bindings", {}),
                "evidence_topics": evidence_judgment.get("topics", []),
                "evidence_source_names": [
                    str(item.get("source_display_name", ""))
                    for item in retrieved[:4]
                    if item.get("source_display_name")
                ],
                "visible_participant_ids": group_turn_binding.get("visible_participant_ids", []),
                "addressed_participant_ids": group_turn_binding.get("addressed_participant_ids", []),
                "mentioned_participant_ids": group_turn_binding.get("mentioned_participant_ids", []),
                "reply_to_turn_id": group_turn_binding.get("reply_to_turn_id", ""),
                "group_reply_policy_action": group_reply_policy.get("action", ""),
                "ingress_evidence_band": ingress_band,
            },
        )
        episode_components_after = aggregate_fe_components(
            state,
            memory_dynamics=memory_dynamics,
            memory_efe_evaluation=m13_memory_efe_evaluation,
            reward_evaluation=m13_reward_evaluation,
            conscious_plan=conscious,
        )
        episode = build_episode(
            at=now,
            turn_index=turn_index,
            phase="proactive_turn" if proactive_turn else "user_turn",
            state=state,
            action="proactive_outreach" if proactive_turn else action,
            action_trigger=str(proactive_context.get("trigger", "") or "user_message") if proactive_turn else "user_message",
            evidence_refs=[
                *m13_evaluation.evidence_refs,
                *[str(item.get("id", "")) for item in retrieved if item.get("id")],
                *(
                    [
                        str(ref)
                        for ref in proactive_context.get("trigger_evidence_refs", []) or []
                        if str(ref).strip()
                    ]
                    if proactive_turn
                    else []
                ),
            ],
            components_before=episode_components_before,
            components_after=episode_components_after,
            memory_gate_decision=memory_gate_decision_from_events(
                self._memory_gate_events_since(state, memory_gate_audit_start)
            ),
            outcome_summary="settled" if proactive_turn else self._outcome_from_conscious_plan(conscious),
            memory_efe_evaluation=m13_memory_efe_evaluation,
            band_summary=self._idle_drive_band_summary(
                m13_state,
                state=state,
                m13_drive_evaluation=m13_evaluation,
                m13_reward_evaluation=m13_reward_evaluation,
            ),
        )
        self._tag_pending_settlements_with_episode(state, episode_id=episode.episode_id, turn_index=turn_index)
        m11_state = M11RuntimeState(
            user_model=m11_state.user_model,
            prediction_ledger=attach_prediction_source_episode(
                m11_state.prediction_ledger,
                created_at_turn=turn_index + 1,
                source_episode_id=episode.episode_id,
            ),
            reliability_ledger=m11_state.reliability_ledger,
            prediction_calibration=m11_state.prediction_calibration,
        )
        _save_m11_state(state, user_id=user_id, m11_state=m11_state)
        for entry in m11_state.prediction_ledger.entries:
            if entry.turn_id != turn_index + 1 or entry.event_kind not in {"judgment", "expiration"}:
                continue
            if not entry.source_episode_id:
                bus.append(
                    {
                        "type": "PredictionSettlementAuditEvent",
                        "turn_index": turn_index,
                        "at": now,
                        "prediction_id": entry.prediction_id,
                        "reason_code": "missing_source_episode_id",
                        "engineering_proxy_label": "mvp_local_prediction_error",
                    }
                )
                continue
            addendum = episode_ledger.append_prediction_settlement_addendum(
                at=now,
                turn_index=turn_index,
                source_episode_id=entry.source_episode_id,
                prediction_id=entry.prediction_id,
                prediction_type=entry.prediction_type,
                outcome=entry.settlement_outcome or entry.validation_status,
                committed_confidence=entry.committed_confidence,
                prediction_error=entry.m17_prediction_error,
                brier_score=entry.m17_brier_score,
                evidence_refs=entry.evidence_refs,
                reason_codes=(),
            )
            bus.append(addendum)
        settlement_writeback_entries: list[dict[str, Any]] = []
        if settlement_judgments:
            committed_confidence_by_prediction = {
                str(row.get("prediction_id", "")).strip(): _bounded_float(row.get("committed_confidence"), default=0.0)
                for row in open_predictions_for_settlement
                if str(row.get("prediction_id", "")).strip()
            }
            for row in settlement_judgments:
                prediction_id = str(row.get("prediction_id", "")).strip()
                if not prediction_id:
                    continue
                settlement_writeback_entries.append(
                    {
                        **dict(row),
                        "committed_confidence": committed_confidence_by_prediction.get(prediction_id, 0.0),
                    }
                )
        else:
            settlement_writeback_entries = [
                entry.to_dict()
                for entry in m11_state.prediction_ledger.entries
                if entry.turn_id == turn_index + 1 and entry.event_kind in {"judgment", "expiration"}
            ]
        path_b_settlement_writeback = apply_path_b_settlement_writeback(
            state,
            ledger_entries=settlement_writeback_entries,
            turn_index=turn_index,
            current_user_text=user_text,
            now=now,
        )
        if path_b_settlement_writeback.settled_prediction_ids:
            bus.append(
                {
                    "type": "PathBSettlementWritebackEvent",
                    "turn_index": turn_index,
                    "settled_prediction_ids": list(path_b_settlement_writeback.settled_prediction_ids),
                    "updated_path_ids": list(path_b_settlement_writeback.updated_path_ids),
                    "writeback_targets": dict(path_b_settlement_writeback.writeback_targets),
                }
            )
        self.store.save(state)
        # P0-7 (2026-06-09): cache the M20.4 per-sub-class
        # diagnostic counters from the in-memory state
        # before the state goes out of scope. The counters
        # are accumulated by the M20.4 producer / write /
        # tie-breaker during the run; the `store.save` call
        # only persists `SYSTEM_FILE_DEFAULTS` keys, so the
        # M20.4 diagnostics key would otherwise be lost.
        # The cached value is what the M18.7.1 calibration
        # harness surfaces on the report.
        m20_4_diag = state.get("m20_4_attribution_diagnostics")
        if isinstance(m20_4_diag, dict):
            self._last_m20_4_diagnostics = dict(m20_4_diag)
        else:
            self._last_m20_4_diagnostics = None
        self._record_episode(episode)
        llm_thinking_result = thinking.get("llm_thinking_result")
        if not isinstance(llm_thinking_result, Mapping):
            legacy_inner_thought = str(thinking.get("inner_thought") or "").strip()
            llm_thinking_result = {
                "debug_summary": legacy_inner_thought,
            } if legacy_inner_thought else {}
        latency_summary = _latency_trace_summary(turn_latency_trace)
        latency_summary["turn_total_duration_ms"] = round((time.monotonic() - turn_latency_started) * 1000.0, 3)
        latency_summary["latency_mode"] = latency_mode
        latency_summary["latency_mode_reasons"] = _string_list(latency_mode_info.get("reason_codes"), limit=12)
        latency_summary["skipped_stage_count"] = len(skipped_llm_stages)
        diagnostics = {
            "mvp_runtime": True,
            "proactive_turn": proactive_turn,
            "proactive_source": "m13_proactive_turn" if proactive_turn else "",
            "proactive_trigger": str(proactive_context.get("trigger", "")) if proactive_turn else "",
            "not_user_requested_current_turn": proactive_turn,
            "bus_messages": bus,
            "conscious_plan": conscious,
            "temporal_input": temporal_input,
            "temporal_assessment": dict(temporal_assessment),
            "self_expectation_state": prompt_safe_self_expectation_summary(state),
            "memory_dynamics": memory_dynamics,
            "m11_user_model": m11_result_dict,
            "m12_1_personality": m12_1_result_dict,
            "m12_2_reciprocal_role": m12_2_result_dict,
            "relationship_value_context": relationship_value_context,
            "current_interlocutor": {
                "display_name": display_name,
                "user_id": user_id,
                "aliases": _mapping(entity_binding.get("current_interlocutor")).get("aliases", []),
            },
            "group_turn_envelope": bounded_group_turn,
            "group_turn_binding": group_turn_binding,
            "group_reply_policy": group_reply_policy,
            "group_privacy_policy": group_privacy_policy,
            "group_chat_state": group_chat_state,
            "ingress_evidence_band": ingress_band,
            "entity_binding": entity_binding,
            "alias_updates_applied": alias_updates_applied,
            "memory_candidates_applied": memory_candidates_applied,
            "post_reply_observer": post_reply_observer,
            "post_reply_observer_skipped_reason": post_reply_observer_skipped_reason,
            "post_reply_memory_updates_applied": post_reply_memory_updates_applied,
            "pacing_feedback_habits_applied": pacing_feedback_habits_applied,
            "sharing_regret_feedback": sharing_regret_feedback,
            "followup_replies": followup_replies,
            "conversation_mode": control_guidance.get("conversation_mode"),
            "reply_pacing_hint": conscious.get("reply_pacing_hint"),
            "interaction_framework_hint": conscious.get("interaction_framework_hint"),
            "pacing_source": control_guidance.get("pacing_source"),
            "latency_mode": latency_mode,
            "latency_mode_reasons": _string_list(latency_mode_info.get("reason_codes"), limit=12),
            "turn_latency_trace": [dict(item) for item in turn_latency_trace],
            "turn_latency_summary": latency_summary,
            "skipped_llm_stages": [dict(item) for item in skipped_llm_stages],
            "reply_contract": reply_contract,
            "reply_validation": reply_validation,
            "raw_reply": raw_reply,
            "pacing_guidance": control_guidance,
            "response_style_prior": response_style_prior,
            "habit_updates_applied": habit_updates_applied,
            "m13_drive_evaluation": prompt_safe_m13_turn_diagnostics(m13_evaluation),
            "m13_boredom_evaluation": prompt_safe_m13_boredom_diagnostics(m13_boredom_evaluation),
            "m13_reward_evaluation": prompt_safe_m13_reward_diagnostics(m13_reward_evaluation),
            "m13_memory_efe_evaluation": prompt_safe_m13_memory_efe_diagnostics(m13_memory_efe_evaluation),
            "m13_reward_ui_labels": prompt_safe_m13_reward_ui_labels(),
            "m13_drive_state": prompt_safe_m13_state_summary(m13_state, user_id=user_id),
            "m15_episode": episode.to_dict(),
            "path_b_recall_bridge": recall_bridge.to_dict(),
            "path_b_settlement_writeback": path_b_settlement_writeback.to_dict(),
            "retrieved_memories": retrieved,
            "thinking": thinking,
            "llm_thinking_result": llm_thinking_result,
            "state_root": str(self.store.root),
            "system_files": {key: str(self.store.path_for(key)) for key in SYSTEM_FILE_DEFAULTS},
        }
        latency_log_event = {
            "event": "turn_latency",
            "type": "MVPDialogTurnLatencyEvent",
            "at": now,
            "turn_index": turn_index,
            "latency_mode": latency_mode,
            "latency_mode_reasons": _string_list(latency_mode_info.get("reason_codes"), limit=12),
            "blocking_llm_calls": latency_summary.get("blocking_llm_calls", 0),
            "total_llm_duration_ms": latency_summary.get("total_llm_duration_ms", 0.0),
            "turn_total_duration_ms": latency_summary.get("turn_total_duration_ms", 0.0),
            "slowest_stage": dict(latency_summary.get("slowest_stage", {}))
            if isinstance(latency_summary.get("slowest_stage"), Mapping)
            else {},
            "turn_latency_trace": [dict(item) for item in turn_latency_trace[:12]],
            "skipped_llm_stages": [dict(item) for item in skipped_llm_stages],
        }
        if not proactive_defer_audit_log:
            self.store.append_log(latency_log_event)
        if proactive_turn and proactive_defer_audit_log:
            pass
        elif proactive_turn:
            self.store.append_log(
                {
                    "event": "proactive_turn",
                    "at": now,
                    "turn_index": turn_index,
                    "source": "m13_proactive_turn",
                    "role": "assistant",
                    "trigger": str(proactive_context.get("trigger", "")),
                    "proposal_id": str(proactive_context.get("proposal_id", "")),
                    "not_user_requested_current_turn": True,
                    "reply": reply,
                    "followup_replies": followup_replies,
                    "surrogate_context": str(proactive_surrogate_text or "")[:240],
                    "diagnostics": diagnostics,
                }
            )
        else:
            self.store.append_log(
                {
                    "event": "turn",
                    "at": now,
                    "turn_index": turn_index,
                    "speaker_name": display_name,
                    "speaker_participant_id": group_turn_binding.get("current_speaker_participant_id", ""),
                    "participant_ids": group_turn_binding.get("visible_participant_ids", []),
                    "addressed_participant_ids": group_turn_binding.get("addressed_participant_ids", []),
                    "mentioned_participant_ids": group_turn_binding.get("mentioned_participant_ids", []),
                    "reply_to_turn_id": group_turn_binding.get("reply_to_turn_id", ""),
                    "quoted_turn_ids": group_turn_binding.get("quoted_turn_ids", []),
                    "explicit_mentions": group_turn_binding.get("explicit_mentions", []),
                    "ingress_evidence_band": ingress_band,
                    "group_turn_binding": group_turn_binding,
                    "group_reply_policy": group_reply_policy,
                    "group_privacy_policy": group_privacy_policy,
                    "user_text": user_text,
                    "reply": reply,
                    "followup_replies": followup_replies,
                    "diagnostics": diagnostics,
                }
            )
        _report_progress("finalize")
        # M20.4.1 §4 — clear the override handoff slot. The gate
        # wrote the verdict earlier in this turn; the M18.5
        # enforcement point already read it. The slot MUST be
        # cleared before return so it does not leak to T+1.
        _m20_4_1_clear_pending_override(state)
        return MVPTurnResult(
            reply=reply,
            action=action,
            diagnostics=diagnostics,
            followup_replies=followup_replies,
        )

    def _initiative_structural_signals(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        sig: dict[str, Any] = {}
        try:
            from segmentum.dialogue.runtime.m14_2_scheduled_intents import ScheduledIntentStore

            scheduled_store = ScheduledIntentStore(
                self.store.root,
                persona_id=self.persona_name or "default",
                session_id=str(self.store.root.resolve()),
            )
            sig["scheduled_intents"] = scheduled_store.list_intents()
        except Exception:
            sig["scheduled_intents"] = []
        try:
            sig["queued_outreach"] = load_queued_outreach(self.store.root)
        except Exception:
            sig["queued_outreach"] = []
        return sig

    def _idle_drive_band_summary(
        self,
        m13_state: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        m13_drive_evaluation: Any | None = None,
        m13_reward_evaluation: Any | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_m13_drive_state(m13_state)
        boredom = _mapping(normalized.get("boredom"))
        reward = normalize_affective_reward_proxy_state(normalized.get("affective_reward_proxy"))
        temporal = _mapping(state.get("temporal_state"))
        share_trace = _mapping(temporal.get("last_share_trace"))
        user_id = str(share_trace.get("user_id", "") or "").strip()
        rel_map = _mapping(normalized.get("relation_path_precision"))
        rel_precision = _bounded_float(rel_map.get(user_id), default=0.0) if user_id else 0.0
        traction = _mapping(normalized.get("traction_by_action"))
        best_action = ""
        best_pull = 0.0
        suffix = f"|{user_id}" if user_id else ""
        for key, value in traction.items():
            key_text = str(key)
            if suffix and not key_text.endswith(suffix):
                continue
            action = key_text.split("|", 1)[0].strip()
            pull = _bounded_float(value, default=0.0)
            if action and pull > best_pull:
                best_action = action
                best_pull = pull

        def band(value: float) -> str:
            if value >= 0.67:
                return "high"
            if value >= 0.35:
                return "medium"
            return "low"

        boredom_level = _bounded_float(boredom.get("boredom_level"), default=0.0)
        reward_net = _bounded_float(
            getattr(m13_reward_evaluation, "net_affective_reward_proxy", reward.get("last_net_reward_proxy")),
            default=0.0,
        )
        if m13_drive_evaluation is not None:
            best_action = str(getattr(m13_drive_evaluation, "top_behavioral_pull_action", best_action) or best_action)
            scores = getattr(m13_drive_evaluation, "scores_by_action", {})
            if isinstance(scores, Mapping) and best_action:
                best_pull = _bounded_float(_mapping(scores.get(best_action)).get("behavioral_pull"), default=best_pull)
        return {
            "boredom_band": boredom_band(boredom_level),
            "behavioral_pull_band": band(best_pull),
            "top_behavioral_pull_action": best_action,
            "affective_reward_band": band(reward_net),
            "path_feels_stale_proxy": bool(reward.get("path_feels_stale_proxy")),
            "relation_path_precision_band": band(rel_precision),
        }

    def _refresh_idle_proactive_drive_context(
        self,
        state: dict[str, Any],
        *,
        now: int,
        turn_index: int,
        structural_signals: Mapping[str, Any],
        idle_seconds: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, Any], Any, list[dict[str, Any]], IdleCognitiveRefreshResult]:
        """Refresh traceable idle signals before M13.3 target selection.

        The elapsed silence opens the idle phase only. It does not add to any
        drive scalar; this refresh reads persisted M13 state, performs bounded
        recall, re-runs deterministic M13.1/M13.2 evaluators, evaluates memory
        EFE for the idle phase, and records one canonical tick event.
        """
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        sig_dict = dict(structural_signals)
        decay_result = apply_memory_decay_tick(state, now=now, turn_index=turn_index)
        recall_limit, recall_bias_events = consume_recall_breadth_intent(
            state,
            now=now,
            turn_index=turn_index,
            default_top_k=8,
        )
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        idle_context = build_idle_context(
            state,
            m13_state=m13_state,
            structural_signals=sig_dict,
            turn_index=turn_index,
            now=now,
        )
        keywords = idle_retrieval_keywords(idle_context)
        retrieved = retrieve_memories(state, keywords, limit=recall_limit)
        expectation_set = normalize_expectations_for_efe(
            state,
            now=now,
            phase="idle",
            structural_signals=sig_dict,
        )
        bound_ids: list[str] = []
        for expectation in expectation_set.eligible_for_efe:
            bound_ids.extend(list(expectation.bound_memory_ids[:8]))
            bound_ids.extend(list(expectation.evidence_refs[:8]))
        bound_ids.extend(list(expectation_set.bound_recall_seed_ids[:4]))
        bound_ids = list(dict.fromkeys(item for item in bound_ids if item))[:4]
        if bound_ids:
            by_id = retrieve_memories_by_ids(state, bound_ids, limit=8)
            seen = {str(item.get("id", "")) for item in retrieved if item.get("id")}
            for item in by_id:
                item_id = str(item.get("id", ""))
                if item_id and item_id not in seen:
                    retrieved.append(item)
                    seen.add(item_id)
                if len(retrieved) >= recall_limit:
                    break
        retrieved_ids = sorted({str(item.get("id", "")) for item in retrieved if item.get("id")})
        temporal = _mapping(state.get("temporal_state"))
        share_trace = _mapping(temporal.get("last_share_trace"))
        user_id = str(share_trace.get("user_id", "") or "").strip()
        turn_key = f"idle-{turn_index}"
        idle_memory_dynamics: dict[str, Any] = {
            "control_guidance": {},
            "recall": {"retrieved_ids": retrieved_ids[:12]},
            "idle_phase": True,
        }
        m13_drive_evaluation = M13DriveEvaluator().evaluate(
            user_text="",
            user_id=user_id,
            turn_id=turn_key,
            turn_index=turn_index,
            conscious_plan={},
            memory_dynamics=idle_memory_dynamics,
            retrieved_memories=retrieved,
            response_style_prior={},
            habit_traits={},
            relationship_value_context={},
            m13_state=m13_state,
            episode_ledger=self._episode_ledger(),
            current_state_fingerprint=self._current_state_fingerprint(state),
        )
        m13_boredom_evaluation = M13BoredomEvaluator().evaluate(
            user_text="",
            user_id=user_id,
            turn_id=turn_key,
            turn_index=turn_index,
            conscious_plan={},
            memory_dynamics=idle_memory_dynamics,
            retrieved_memories=retrieved,
            m13_state=m13_state,
            m13_drive_evaluation=m13_drive_evaluation,
        )
        prior_boredom = _mapping(normalize_m13_drive_state(m13_state).get("boredom"))
        silence_only = (
            not retrieved_ids
            and not bound_ids
            and not expectation_set.eligible_for_efe
            and not sig_dict.get("scheduled_intents")
            and not sig_dict.get("queued_outreach")
        )
        if silence_only and m13_boredom_evaluation.boredom_level > _bounded_float(prior_boredom.get("boredom_level")):
            boredom_patch_events = []
        else:
            m13_state, boredom_patch_events = apply_post_turn_boredom_state(
                m13_state,
                boredom=m13_boredom_evaluation,
                conscious_plan={},
                retrieved_memories=retrieved,
                turn_index=turn_index,
            )
        m13_reward_pre_turn = evaluate_pre_turn_reward_proxy(
            turn_id=turn_key,
            turn_index=turn_index,
            user_id=user_id,
            m13_state=m13_state,
            m13_evaluation=m13_drive_evaluation,
            information_gain_proxy=m13_boredom_evaluation.information_gain_proxy,
            repetition_pressure=m13_boredom_evaluation.repetition_pressure,
            conflict_level=0.0,
        )
        state["m13_drive_state"] = m13_state
        memory_efe_evaluation = evaluate_memory_efe(
            state,
            phase="idle",
            now=now,
            turn_index=turn_index,
            user_active=False,
            structural_signals=sig_dict,
            retrieved_memories=retrieved,
            m13_boredom_evaluation=m13_boredom_evaluation,
            m13_reward_evaluation=m13_reward_pre_turn,
            episode_ledger=self._episode_ledger(),
        )
        m13_state, memory_efe_events = apply_memory_efe_state(m13_state, memory_efe_evaluation)
        state["m13_drive_state"] = m13_state
        for event in memory_efe_evaluation.events:
            if str(event.get("type", "")).startswith("BundleDecision"):
                self._record_bundle_policy_event(state, event)
        state["bundle_policy_linkage_diagnostics"] = dict(memory_efe_evaluation.bundle_linkage_diagnostics)
        band_summary = self._idle_drive_band_summary(
            m13_state,
            state=state,
            m13_drive_evaluation=m13_drive_evaluation,
            m13_reward_evaluation=m13_reward_pre_turn,
        )
        sig_dict.update(
            {
                "memory_efe_should_outreach": bool(memory_efe_evaluation.should_outreach),
                "memory_efe_traceable_expectation_id": str(memory_efe_evaluation.traceable_expectation_id or ""),
                "idle_drive_band_summary": band_summary,
            }
        )
        selected_target = select_proactive_target(
            state,
            m13_state,
            memory_efe_evaluation=memory_efe_evaluation,
            structural_signals=sig_dict,
        )
        reject_reason = ""
        if selected_target is None:
            reject_reason = classify_proactive_target_reject_reason(
                state,
                m13_state,
                memory_efe_evaluation=memory_efe_evaluation,
                structural_signals=sig_dict,
            )
            if reject_reason:
                m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
                initiative = merge_idle_introspection_into_initiative(m13_state.get("initiative"))
                idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
                idle["last_cognitive_skip_reason"] = reject_reason
                idle["last_cognitive_skip_at"] = now
                initiative["idle_introspection"] = idle
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
        selected_target_payload = None
        if selected_target is not None:
            selected_target_payload = {
                "trigger": selected_target.trigger,
                "source_kind": selected_target.source_kind,
                "traceable_expectation_id": selected_target.traceable_expectation_id,
                "evidence_refs": list(selected_target.evidence_refs[:8]),
                "selection_reason_codes": list(selected_target.selection_reason_codes[:8]),
            }
        tick_event = {
            "type": "IdleCognitiveTickEvent",
            "turn_index": turn_index,
            "at": now,
            "idle_seconds": round(float(idle_seconds), 3),
            "retrieved_ids": retrieved_ids[:12],
            "bounded_retrieve_ids": list(dict.fromkeys(bound_ids))[:12],
            "recall_top_k": recall_limit,
            "memory_efe_should_outreach": bool(memory_efe_evaluation.should_outreach),
            "memory_efe_selected_policy": str(memory_efe_evaluation.selected_policy or ""),
            "bands": {
                "boredom_band": band_summary.get("boredom_band", ""),
                "reward_band": band_summary.get("affective_reward_band", ""),
                "behavior_band": band_summary.get("behavioral_pull_band", ""),
                "relation_band": band_summary.get("relation_path_precision_band", ""),
            },
            "selected_target": selected_target_payload,
            "reject_reason": reject_reason,
            "engineering_proxy_label": "mvp_local_idle_cognitive_tick",
        }
        events = [
            *recall_bias_events,
            *m13_drive_evaluation.events,
            *decay_result.events,
            *m13_boredom_evaluation.events,
            *boredom_patch_events,
            *m13_reward_pre_turn.events,
            {
                "type": "IdleProactiveDriveRefreshEvent",
                "turn_index": turn_index,
                "at": now,
                "order": "recall_then_memory_efe_then_m13_drive_bands_before_target_selection",
                "retrieved_ids": retrieved_ids[:12],
                "bounded_retrieve_ids": list(dict.fromkeys(bound_ids))[:12],
                "drive_band_summary": band_summary,
                "engineering_proxy_label": "mvp_local_proactive_alignment",
            },
            *memory_efe_events,
            *memory_efe_evaluation.events,
            tick_event,
        ]
        meta_result = detect_and_emit_intents(
            state,
            self._episode_ledger(),
            now=now,
            turn_index=turn_index,
            source="idle_cognitive_tick",
            current_idle_tick_event=tick_event,
        )
        events.extend(meta_result.events)
        result = IdleCognitiveRefreshResult(
            retrieved_ids=retrieved_ids[:12],
            bounded_retrieve_ids=list(dict.fromkeys(bound_ids))[:12],
            memory_efe_evaluation=memory_efe_evaluation,
            m13_band_summary=band_summary,
            selected_target=selected_target,
            reject_reason=reject_reason,
            audit_events=events,
        )
        return state, sig_dict, memory_efe_evaluation, events, result

    def maybe_propose_proactive_turn(
        self,
        *,
        turn_index: int,
        idle_seconds: float = 0.0,
        manual_continue: bool = False,
        user_typing: bool = False,
        implicit_idle_request: bool = False,
        preselected_target: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._load_state_with_initiative_repair()
        now = _utc_timestamp()
        structural_signals = self._initiative_structural_signals(state)
        memory_efe_evaluation = None
        refresh_events: list[dict[str, Any]] = []
        locked_proposal = None
        if implicit_idle_request:
            preselected = _proactive_target_from_mapping(preselected_target)
            if preselected is not None:
                m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
                initiative = normalize_initiative_state(m13_state.get("initiative"))
                locked_proposal = build_proposal_from_target(preselected, now=now, initiative=initiative)
            else:
                state, structural_signals, memory_efe_evaluation, refresh_events, tick_result = self._refresh_idle_proactive_drive_context(
                    state,
                    now=now,
                    turn_index=turn_index,
                    structural_signals=structural_signals,
                    idle_seconds=idle_seconds,
                )
                if tick_result.selected_target is not None:
                    m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
                    initiative = normalize_initiative_state(m13_state.get("initiative"))
                    locked_proposal = build_proposal_from_target(tick_result.selected_target, now=now, initiative=initiative)
        state, check = evaluate_proactive_initiative(
            state,
            now=now,
            turn_index=turn_index,
            idle_seconds=idle_seconds,
            manual_continue=manual_continue,
            user_typing=user_typing,
            implicit_idle_request=implicit_idle_request,
            llm=self.llm,
            locked_proposal=locked_proposal,
            structural_signals=structural_signals,
            memory_efe_evaluation=memory_efe_evaluation,
        )
        self.store.save(state)
        for event in refresh_events:
            self.store.append_log({"event": "m13_proactive_audit", **event})
        for event in check.events:
            self.store.append_log({"event": "m13_proactive_audit", **event})
        return {
            "proposal": check.proposal.to_dict() if check.proposal else None,
            "suppression_reason": check.suppression_reason,
            "suppression_reason_code": check.suppression_reason_code or check.suppression_reason,
            "events": [*refresh_events, *check.events],
            "state_fields_read": check.state_fields_read,
        }

    def run_idle_cognitive_tick(
        self,
        *,
        turn_index: int,
        idle_seconds: float = 0.0,
        now: int | None = None,
    ) -> dict[str, Any]:
        state = self.store.load()
        tick_now = int(now if now is not None else _utc_timestamp())
        components_before = aggregate_fe_components(state)
        memory_gate_audit_start = self._memory_gate_audit_len(state)
        structural_signals = self._initiative_structural_signals(state)
        state, _structural_signals, _memory_efe_evaluation, events, result = self._refresh_idle_proactive_drive_context(
            state,
            now=tick_now,
            turn_index=turn_index,
            structural_signals=structural_signals,
            idle_seconds=idle_seconds,
        )
        components_after = aggregate_fe_components(
            state,
            memory_efe_evaluation=_memory_efe_evaluation,
        )
        episode = build_episode(
            at=tick_now,
            turn_index=turn_index,
            phase="idle_tick",
            state=state,
            action="idle_wait",
            action_trigger="idle_cognitive_tick",
            evidence_refs=[*result.retrieved_ids, *result.bounded_retrieve_ids],
            components_before=components_before,
            components_after=components_after,
            memory_gate_decision=memory_gate_decision_from_events(
                self._memory_gate_events_since(state, memory_gate_audit_start)
            ),
            outcome_summary="settled" if result.selected_target is not None else "ignored",
            memory_efe_evaluation=_memory_efe_evaluation,
            band_summary=result.m13_band_summary,
        )
        self.store.save(state)
        for event in events:
            self.store.append_log({"event": "m13_proactive_audit", **event})
        self._record_episode(episode)
        state = self.store.load()
        tick_event = next((event for event in events if str(event.get("type", "")) == "IdleCognitiveTickEvent"), None)
        state, consolidation_events = self._run_consolidation_cycle(
            state,
            now=tick_now,
            turn_index=turn_index,
            triggered_by="idle_cognitive_tick",
            current_idle_tick_event=tick_event if isinstance(tick_event, Mapping) else None,
        )
        self.store.save(state)
        if consolidation_events:
            result.audit_events.extend(consolidation_events)
        return result.to_dict()

    def run_proactive_turn(
        self,
        *,
        proposal_id: str,
        turn_index: int,
        speaker_name: str = "",
        now: int | None = None,
    ) -> MVPTurnResult:
        state = self._load_state_with_initiative_repair()
        now = int(now if now is not None else _utc_timestamp())
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state"))
        initiative = normalize_initiative_state(m13_state.get("initiative"))
        proposal = proposal_from_initiative_state(initiative, now=now)
        if proposal is None or str(proposal.proposal_id) != str(proposal_id):
            reason = "proposal_expired" if proposal is None else "proposal_not_found"
            components = aggregate_fe_components(state)
            episode = build_episode(
                at=now,
                turn_index=turn_index,
                phase="proactive_turn",
                state=state,
                action="proactive_outreach",
                action_trigger="missing_proposal",
                evidence_refs=[],
                components_before=components,
                components_after=components,
                memory_gate_decision=memory_gate_decision_from_events([]),
                outcome_summary="ignored",
            )
            initiative["last_suppression_reason"] = reason
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.store.save(state)
            self._record_episode(episode)
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "M13ProactiveSuppressionEvent",
                    "reason": reason,
                    "reason_code": reason,
                    "reason_stage": "pre_proposal",
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                }
            )
            return MVPTurnResult(
                reply="",
                action="proactive_suppressed",
                diagnostics={"suppression_reason": reason, "proactive_turn": True, "m15_episode": episode.to_dict()},
            )

        gate_reason = proactive_delivery_gate_reason(initiative, now=now, turn_index=turn_index)
        if gate_reason:
            initiative["last_suppression_reason"] = gate_reason
            initiative["last_suppression_reason_code"] = gate_reason
            initiative["pending_proactive_proposal"] = {}
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.store.save(state)
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "M13ProactiveSuppressionEvent",
                    "reason": gate_reason,
                    "reason_code": gate_reason,
                    "reason_stage": "pre_generation",
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                }
            )
            return MVPTurnResult(
                reply="",
                action="proactive_suppressed",
                diagnostics={"suppression_reason": gate_reason, "reason_code": gate_reason, "proactive_turn": True},
            )

        state_snapshot = copy.deepcopy(self.store.load())
        proactive_context = {**proposal.to_dict(), "defer_audit_log": True}
        result = self.run_turn(
            PROACTIVE_SURROGATE_USER_TEXT,
            turn_index=turn_index,
            speaker_name=speaker_name,
            now=now,
            proactive_context=proactive_context,
        )
        reply = str(result.reply or "").strip()
        followup_replies = list(getattr(result, "followup_replies", []) or [])
        delivery_assessment: dict[str, Any] = {}
        if reply and self.llm is not None:
            delivery_assessment = assess_proactive_delivery_semantics(
                self.llm,
                reply=reply,
                followup_replies=followup_replies,
                ordinary_language_intent=proposal.ordinary_language_intent,
                trigger=proposal.trigger,
                turn_index=turn_index,
            )
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "ProactiveDeliveryAssessmentEvent",
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                    "assessment": delivery_assessment,
                    "engineering_proxy_label": "mvp_local_proactive_alignment",
                }
            )
            delivery_ok = bool(delivery_assessment.get("allow_delivery")) and _bounded_float(
                delivery_assessment.get("confidence")
            ) >= _bounded_float(initiative.get("delivery_assessor_min_confidence"), default=0.5)
        else:
            delivery_ok = bool(reply)
        if not delivery_ok:
            self.store.save(state_snapshot)
            m13_state = merge_initiative_into_m13_state(state_snapshot.get("m13_drive_state"))
            initiative = normalize_initiative_state(m13_state.get("initiative"))
            if not reply:
                reason_code = "empty_generation"
            elif delivery_assessment and not bool(delivery_assessment.get("allow_delivery")):
                reason_code = "delivery_assessor_reject"
            elif delivery_assessment and _bounded_float(delivery_assessment.get("confidence")) < _bounded_float(
                initiative.get("delivery_assessor_min_confidence"), default=0.5
            ):
                reason_code = "delivery_assessor_low_confidence"
            else:
                reason_code = "delivery_assessor_reject"
            initiative["last_suppression_reason"] = reason_code
            initiative["last_suppression_reason_code"] = reason_code
            initiative["pending_proactive_proposal"] = {}
            initiative["cooldown_until_timestamp"] = now + max(30, int(proposal.cooldown_cost or 0) * 45)
            m13_state["initiative"] = initiative
            if str(proposal.traceable_expectation_id or "").strip():
                m13_state = record_target_assessor_reject_backoff(
                    m13_state,
                    expectation_id=proposal.traceable_expectation_id,
                    now=now,
                    reason_code=reason_code,
                )
            initiative = normalize_initiative_state(m13_state.get("initiative"))
            initiative["pending_proactive_proposal"] = {}
            initiative["last_suppression_reason"] = reason_code
            initiative["last_suppression_reason_code"] = reason_code
            m13_state["initiative"] = initiative
            state_snapshot["m13_drive_state"] = m13_state
            self.store.save(state_snapshot)
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "M13ProactiveSuppressionEvent",
                    "reason": reason_code,
                    "reason_code": reason_code,
                    "reason_stage": "post_generation",
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                    "proactive_text_blocked": bool(reply),
                    "assessment": delivery_assessment,
                }
            )
            episode_payload = _mapping(result.diagnostics.get("m15_episode"))
            episode_id = str(episode_payload.get("episode_id", "") or "")
            if episode_id:
                components_revised = aggregate_fe_components(
                    state_snapshot,
                    memory_dynamics=_mapping(result.diagnostics.get("memory_dynamics")),
                )
                addendum = self._episode_ledger().append_settlement_event(
                    episode_id=episode_id,
                    at=now,
                    turn_index=turn_index,
                    new_outcome_summary="violated",
                    fe_proxy_after_revised=aggregate_fe_proxy(components_revised),
                    components_after_revised=components_revised,
                    settlement_event={
                        "type": "M13ProactiveSuppressionEvent",
                        "reason_code": reason_code,
                        "reason_stage": "post_generation",
                    },
                )
                self.store.append_log({"event": "m15_episode_ledger", **addendum})
            return MVPTurnResult(
                reply="",
                action="proactive_suppressed",
                diagnostics={
                    **result.diagnostics,
                    "suppression_reason": reason_code,
                    "reason_code": reason_code,
                    "reason_stage": "post_generation",
                    "proactive_text_blocked": bool(reply),
                    "proactive_turn": True,
                },
                followup_replies=[],
            )

        state = self.store.load()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state"))
        state["m13_drive_state"] = mark_proactive_turn_consumed(
            m13_state,
            now=now,
            turn_index=turn_index,
            proposal=proposal,
        )
        self.store.save(state)
        self.store.append_log(
            {
                "event": "proactive_turn",
                "at": now,
                "turn_index": turn_index,
                "source": "m13_proactive_turn",
                "role": "assistant",
                "trigger": proposal.trigger,
                "proposal_id": proposal.proposal_id,
                "not_user_requested_current_turn": True,
                "reply": reply,
                "followup_replies": followup_replies,
                "surrogate_context": PROACTIVE_SURROGATE_USER_TEXT[:240],
                "diagnostics": result.diagnostics,
            }
        )
        self.store.append_log(
            {
                "event": "m13_proactive_audit",
                "type": "M13ProactiveGenerationEvent",
                "turn_index": turn_index,
                "proposal_id": proposal.proposal_id,
                "trigger": proposal.trigger,
                "source": "m13_proactive_turn",
                "role": "assistant",
                "not_user_requested_current_turn": True,
                "reply": result.reply,
                "action": result.action,
            }
        )
        return result

    def set_initiative_user_opt_in(self, enabled: bool) -> dict[str, Any]:
        state = self.store.load()
        state["m13_drive_state"] = set_initiative_user_opt_in(
            state.get("m13_drive_state", {}),
            enabled=enabled,
        )
        self.store.save(state)
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        return dict(initiative)

    def set_initiative_implicit_idle_delivery(self, enabled: bool) -> dict[str, Any]:
        from segmentum.dialogue.runtime.m13_initiative import set_initiative_implicit_idle_delivery

        state = self.store.load()
        state["m13_drive_state"] = set_initiative_implicit_idle_delivery(
            state.get("m13_drive_state", {}),
            enabled=enabled,
        )
        self.store.save(state)
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        return dict(initiative)

    def set_initiative_proactive_policy_profile(self, profile: str) -> dict[str, Any]:
        state = self.store.load()
        state["m13_drive_state"] = set_initiative_proactive_policy_profile(
            state.get("m13_drive_state", {}),
            profile=profile,
        )
        self.store.save(state)
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        return dict(initiative)

    def read_initiative_status(self) -> dict[str, Any]:
        state = self.store.load()
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        return dict(initiative)

    def set_idle_introspection_opt_in(self, enabled: bool) -> dict[str, Any]:
        state = self.store.load()
        state["m13_drive_state"] = set_idle_introspection_user_opt_in(
            state.get("m13_drive_state", {}),
            enabled=enabled,
        )
        self.store.save(state)
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        idle = initiative.get("idle_introspection", {})
        return dict(idle) if isinstance(idle, Mapping) else {}

    def set_background_continuity_opt_in(self, enabled: bool, *, runner_kind: str = "inline") -> dict[str, Any]:
        state = self.store.load()
        state["m13_drive_state"] = set_background_continuity_opt_in(
            state.get("m13_drive_state", {}),
            enabled=enabled,
            runner_kind=runner_kind if enabled else "none",
        )
        self.store.save(state)
        initiative = normalize_initiative_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
        )
        bg = initiative.get("background_continuity", {})
        return dict(bg) if isinstance(bg, Mapping) else {}

    def update_background_continuity_config(self, **updates: Any) -> dict[str, Any]:
        state = self.store.load()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        initiative = merge_background_continuity_into_initiative(
            normalize_initiative_state(m13_state.get("initiative"))
        )
        bg = normalize_background_continuity_state(initiative.get("background_continuity"))
        for key in (
            "tick_interval_seconds",
            "tokens_budget_per_day",
            "wallclock_budget_per_day_seconds",
            "max_ticks_per_day",
            "llm_calls_budget_per_day",
            "queued_outreach_ttl_seconds",
        ):
            if key in updates:
                bg[key] = updates[key]
        bg = normalize_background_continuity_state(bg)
        initiative["background_continuity"] = bg
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        self.store.save(state)
        return dict(bg)

    def append_background_audit(self, event: Mapping[str, Any]) -> None:
        self.store.append_log({"event": "m14_1_background_audit", **dict(event)})

    def _persist_background_meter(
        self,
        meter: BackgroundLLMMeter,
        *,
        now: int,
        block_reason: str = "",
    ) -> dict[str, Any]:
        state = self.store.load()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        initiative = merge_background_continuity_into_initiative(
            normalize_initiative_state(m13_state.get("initiative"))
        )
        bg = normalize_background_continuity_state(initiative.get("background_continuity"))
        metered = normalize_background_continuity_state(meter.bg)
        for key in ("llm_calls_today", "llm_calls_lifetime", "tokens_used_today", "tokens_used_lifetime"):
            bg[key] = metered.get(key, bg.get(key, 0))
        if block_reason:
            bg["last_budget_block_reason"] = block_reason
            bg["last_background_skip_reason"] = block_reason
            self.append_background_audit(
                {
                    "type": "BackgroundBudgetReachedEvent",
                    "at": now,
                    "reason": block_reason,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
        initiative["background_continuity"] = bg
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        self.store.save(state)
        return bg

    def record_streamlit_ping(self) -> None:
        state = self.store.load()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        initiative = merge_background_continuity_into_initiative(
            normalize_initiative_state(m13_state.get("initiative"))
        )
        bg = normalize_background_continuity_state(initiative.get("background_continuity"))
        bg["last_streamlit_ping_at"] = _utc_timestamp()
        initiative["background_continuity"] = bg
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        self.store.save(state)

    def inline_runner_should_stop(self, *, idle_death_seconds: int) -> bool:
        state = self.store.load()
        initiative = merge_background_continuity_into_initiative(
            normalize_initiative_state(
                normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
            )
        )
        bg = normalize_background_continuity_state(initiative.get("background_continuity"))
        if not bool(bg.get("user_opt_in")):
            return True
        last_ping = int(bg.get("last_streamlit_ping_at", 0) or 0)
        if last_ping <= 0:
            return False
        return (_utc_timestamp() - last_ping) > int(idle_death_seconds)

    def run_background_self_tick(self, *, runner_kind: str = "inline") -> dict[str, Any]:
        from segmentum.dialogue.runtime.m13_idle import gather_idle_structural_signals

        wall_start = time.monotonic()
        now = _utc_timestamp()
        temporal = _mapping(self.store.load().get("temporal_state"))
        turn_index = int(temporal.get("last_turn_index", 0) or 0)

        with session_file_lock(self.store.root):
            state = self.store.load()
            m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            if not bool(bg.get("user_opt_in")) or not bool(initiative.get("user_opt_in")):
                bg["last_background_skip_reason"] = "not_opted_in"
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                return {"skip_reason": "not_opted_in", "ran_introspection": False}
            idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
            if not bool(idle.get("enabled")):
                bg["last_background_skip_reason"] = "idle_introspection_disabled"
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                return {"skip_reason": "idle_introspection_disabled", "ran_introspection": False}

            bg, rollover = maybe_rollover_daily_counters(bg, now=now)
            if rollover:
                self.append_background_audit(rollover)
            initiative["background_continuity"] = bg
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.store.save(state)

            block = check_background_budgets(bg)
            if block:
                bg["last_budget_block_reason"] = block
                bg["last_background_skip_reason"] = block
                bg["last_tick_at"] = now
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                self.append_background_audit(
                    {
                        "type": "BackgroundBudgetReachedEvent",
                        "at": now,
                        "reason": block,
                        "runner_kind": runner_kind,
                        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                    }
                )
                return {"skip_reason": block, "ran_introspection": False}

            signals = gather_idle_structural_signals(state, now=now, turn_index=turn_index)
            if not signals.should_run_llm():
                bg = record_background_tick(bg, wallclock_seconds=time.monotonic() - wall_start, ran_introspection=False)
                bg["last_tick_at"] = now
                bg["last_background_ran_llm"] = False
                bg["last_background_skip_reason"] = "no_structural_signal"
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                self.append_background_audit(
                    {
                        "type": "BackgroundIdleTickEvent",
                        "at": now,
                        "skip_reason": "no_structural_signal",
                        "runner_kind": runner_kind,
                        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                    }
                )
                return {"skip_reason": "no_structural_signal", "ran_introspection": False}
            llm_status = llm_configuration_status(self.llm)
            if not bool(llm_status.get("available")):
                llm_reason = str(llm_status.get("reason", "") or "llm_unavailable")
                bg = record_background_tick(bg, wallclock_seconds=time.monotonic() - wall_start, ran_introspection=False)
                bg["last_tick_at"] = now
                bg["last_background_ran_llm"] = False
                bg["last_budget_block_reason"] = llm_reason
                bg["last_background_skip_reason"] = llm_reason
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                self.append_background_audit(
                    {
                        "type": "BackgroundIdleTickEvent",
                        "at": now,
                        "skip_reason": llm_reason,
                        "runner_kind": runner_kind,
                        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                    }
                )
                return {"skip_reason": llm_reason, "ran_introspection": False}

            meter = BackgroundLLMMeter(self.llm, bg)
            original_llm = self.llm
            try:
                self.llm = meter  # type: ignore[assignment]
                idle_result = self.run_idle_introspection_turn(
                    now=now,
                    turn_index=turn_index,
                    structural_signals=signals,
                    queue_outreach=True,
                    background_runner_kind=runner_kind,
                )
            except BackgroundBudgetExhausted as exc:
                self.llm = original_llm
                self._persist_background_meter(meter, now=now, block_reason=exc.reason)
                return {"skip_reason": exc.reason, "ran_introspection": False}
            finally:
                self.llm = original_llm
            state = self.store.load()
            m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            for key in ("llm_calls_today", "llm_calls_lifetime", "tokens_used_today", "tokens_used_lifetime"):
                bg[key] = meter.bg.get(key, bg.get(key, 0))
            bg = record_background_tick(
                bg,
                wallclock_seconds=time.monotonic() - wall_start,
                ran_introspection=bool(idle_result.ran_llm or idle_result.reflection_focus),
            )
            bg["last_tick_at"] = now
            bg["last_budget_block_reason"] = ""
            bg["last_background_skip_reason"] = ""
            bg["last_background_ran_llm"] = bool(idle_result.ran_llm)
            initiative["background_continuity"] = bg
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            state, consolidation_events = self._run_consolidation_cycle(
                state,
                now=now,
                turn_index=turn_index,
                triggered_by="background_tick",
            )
            self.store.save(state)
            self.append_background_audit(
                {
                    "type": "BackgroundIdleTickEvent",
                    "at": now,
                    "ran_introspection": True,
                    "ran_llm": bool(idle_result.ran_llm),
                    "outreach_outcome": idle_result.diagnostics.get("outreach_outcome", ""),
                    "consolidation_events": len(consolidation_events),
                    "runner_kind": runner_kind,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
            return {
                "ran_introspection": True,
                "skip_reason": "",
                "outreach_outcome": idle_result.diagnostics.get("outreach_outcome", ""),
            }

    def maybe_drain_queued_outreach(self, *, turn_index: int, now: int | None = None) -> dict[str, Any]:
        now = int(now if now is not None else _utc_timestamp())
        with session_file_lock(self.store.root):
            state = self.store.load()
            m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            if not bool(bg.get("user_opt_in")):
                return {"drained": False, "reason": "not_opted_in"}
            block = check_background_budgets(bg)
            if block:
                bg["last_budget_block_reason"] = block
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.store.save(state)
                self.append_background_audit(
                    {
                        "type": "BackgroundBudgetReachedEvent",
                        "at": now,
                        "reason": block,
                        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                    }
                )
                return {"drained": False, "reason": block}
            meter = BackgroundLLMMeter(self.llm, bg)
            original_llm = self.llm
            try:
                self.llm = meter  # type: ignore[assignment]
                result = self._maybe_drain_queued_outreach_locked(turn_index=turn_index, now=now)
            except BackgroundBudgetExhausted as exc:
                result = {"drained": False, "reason": exc.reason}
                self._persist_background_meter(meter, now=now, block_reason=exc.reason)
            finally:
                self.llm = original_llm
            self._persist_background_meter(meter, now=now)
            result["llm_calls_delta"] = meter.llm_calls_delta
            result["tokens_delta"] = meter.tokens_delta
            return result

    def _maybe_drain_queued_outreach_locked(self, *, turn_index: int, now: int) -> dict[str, Any]:
        entry = pop_next_pending_outreach(self.store.root, now=now)
        if entry is None and str(os.environ.get("SEGMENTUM_QUEUE_INCLUDE_OTHER_SESSIONS", "") or "").strip() == "1":
            entry = self._relay_due_outreach_from_sibling_session_locked(now=now)
        if entry is None:
            return {"drained": False, "reason": "empty_queue"}
        if str(entry.get("trigger", "")) == "scheduled_outreach":
            refs = [str(r) for r in (entry.get("evidence_refs") or entry.get("trigger_evidence_refs") or []) or [] if str(r or "").strip()]
            anchor = str(entry.get("source_intent_id", "") or entry.get("scheduled_intent_id", "") or "").strip()
            if not refs and not anchor:
                update_queued_outreach_status(
                    self.store.root,
                    str(entry.get("proposal_id", "")),
                    "suppressed",
                    now=now,
                    suppression_reason="queued_outreach_missing_traceability_anchor",
                )
                return {"drained": False, "reason": "queued_outreach_missing_traceability_anchor"}
        proposal = ProactiveTurnProposal(
            proposal_id=str(entry.get("proposal_id", "")),
            created_at=int(entry.get("created_at", now) or now),
            source="queued_outreach",
            trigger=str(entry.get("trigger", "reflection_outreach") or "reflection_outreach"),
            trigger_evidence_refs=[
                str(r) for r in (entry.get("evidence_refs") or entry.get("trigger_evidence_refs") or []) or []
            ][:8],
            urgency_band="medium",
            expected_user_value_band="medium",
            risk_band="low",
            proposed_action="answer",
            proposed_topic=str(entry.get("proposed_topic", "") or ""),
            ordinary_language_intent=str(entry.get("ordinary_language_intent", "") or ""),
            expires_at=int(entry.get("expires_at", now) or now),
            cooldown_cost=0,
            traceable_expectation_id=str(entry.get("traceable_expectation_id") or entry.get("source_intent_id", "") or ""),
            source_kind=str(
                entry.get("source_kind")
                or ("scheduled_intent" if str(entry.get("trigger", "")) == "scheduled_outreach" else "")
            ),
            selection_reason_codes=[str(r) for r in entry.get("selection_reason_codes", []) or []][:8],
        )
        record_queued_outreach_delivery_attempt(self.store.root, proposal.proposal_id, now=now)
        self.store.append_log(
            {
                "event": "m14_2_audit",
                "type": "OutboxDeliveryAttemptEvent",
                "at": now,
                "proposal_id": proposal.proposal_id,
                "source_intent_id": str(entry.get("source_intent_id", "")),
                "runner_kind": "delivery_surface",
                "persona_id": str(entry.get("persona_id", "default") or "default"),
                "session_id": str(entry.get("session_id", self.store.root.name) or self.store.root.name),
                "correlation_id": "",
                "engineering_proxy_label": "mvp_local_decoupled_self_loop",
            }
        )
        state = self.store.load()
        check_state, check = evaluate_proactive_initiative(
            state,
            now=now,
            turn_index=turn_index,
            manual_continue=False,
            locked_proposal=proposal,
            llm=self.llm,
            structural_signals=self._initiative_structural_signals(state),
        )
        self.store.save(check_state)
        for event in check.events:
            self.store.append_log({"event": "m13_proactive_audit", **event})
        if check.proposal is None:
            reason = check.suppression_reason or "suppressed"
            suppression_type = (
                "OutboxDeliveryTransientSuppressionEvent"
                if outreach_suppression_is_transient(reason)
                else "OutboxDeliveryHardSuppressionEvent"
            )
            self.store.append_log(
                {
                    "event": "m14_2_audit",
                    "type": suppression_type,
                    "at": now,
                    "proposal_id": proposal.proposal_id,
                    "source_intent_id": str(entry.get("source_intent_id", "")),
                    "reason": reason,
                    "runner_kind": "delivery_surface",
                    "persona_id": str(entry.get("persona_id", "default") or "default"),
                    "session_id": str(entry.get("session_id", self.store.root.name) or self.store.root.name),
                    "correlation_id": "",
                    "engineering_proxy_label": "mvp_local_decoupled_self_loop",
                }
            )
            if outreach_suppression_is_transient(reason):
                return {"drained": False, "reason": reason, "transient": True}
            update_queued_outreach_status(
                self.store.root,
                proposal.proposal_id,
                "suppressed",
                now=now,
                suppression_reason=reason,
            )
            self.append_background_audit(
                {
                    "type": "QueuedOutreachSuppressedEvent",
                    "at": now,
                    "proposal_id": proposal.proposal_id,
                    "reason": reason,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
            return {"drained": False, "reason": reason}
        result = self.run_proactive_turn(
            proposal_id=check.proposal.proposal_id,
            turn_index=turn_index,
            now=now,
        )
        if str(result.reply or "").strip():
            update_queued_outreach_status(self.store.root, proposal.proposal_id, "delivered", now=now)
            self.append_background_audit(
                {
                    "type": "QueuedOutreachDeliveredEvent",
                    "at": now,
                    "proposal_id": proposal.proposal_id,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
            self._close_m14_2_delivery_links(entry, now=now)
            return {"drained": True, "reply": result.reply, "reason": ""}
        reason = str(result.diagnostics.get("suppression_reason", "") or "suppressed")
        suppression_type = (
            "OutboxDeliveryTransientSuppressionEvent"
            if outreach_suppression_is_transient(reason)
            else "OutboxDeliveryHardSuppressionEvent"
        )
        self.store.append_log(
            {
                "event": "m14_2_audit",
                "type": suppression_type,
                "at": now,
                "proposal_id": proposal.proposal_id,
                "source_intent_id": str(entry.get("source_intent_id", "")),
                "reason": reason,
                "runner_kind": "delivery_surface",
                "persona_id": str(entry.get("persona_id", "default") or "default"),
                "session_id": str(entry.get("session_id", self.store.root.name) or self.store.root.name),
                "correlation_id": "",
                "engineering_proxy_label": "mvp_local_decoupled_self_loop",
            }
        )
        if outreach_suppression_is_transient(reason):
            return {"drained": False, "reason": reason, "transient": True}
        update_queued_outreach_status(
            self.store.root,
            proposal.proposal_id,
            "suppressed",
            now=now,
            suppression_reason=reason,
        )
        self.append_background_audit(
            {
                "type": "QueuedOutreachSuppressedEvent",
                "at": now,
                "proposal_id": proposal.proposal_id,
                "reason": reason,
                "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
            }
        )
        return {"drained": False, "reason": reason}

    def _relay_due_outreach_from_sibling_session_locked(self, *, now: int) -> dict[str, Any] | None:
        shared_root = self.store.shared_root
        if shared_root is None:
            return None
        sessions_dir = shared_root / "sessions"
        if not sessions_dir.is_dir():
            return None
        current_root = self.store.root.resolve()
        current_session_id = self.store.root.name
        candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
        for queue_path in sessions_dir.glob("*/queued_outreach.jsonl"):
            source_root = queue_path.parent
            try:
                if source_root.resolve() == current_root:
                    continue
            except OSError:
                continue
            for row in load_queued_outreach(source_root):
                if str(row.get("status", "")) != "pending":
                    continue
                due_at = int(row.get("due_at", row.get("created_at", 0)) or 0)
                if due_at > now:
                    continue
                expires_at = int(row.get("expires_at", 0) or 0)
                if expires_at and expires_at <= now:
                    continue
                candidates.append(
                    (
                        due_at,
                        int(row.get("created_at", 0) or 0),
                        source_root,
                        dict(row),
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
        for _due_at, _created_at, source_root, candidate in candidates:
            with session_file_lock(source_root):
                rows = load_queued_outreach(source_root)
                selected_index = -1
                for index, row in enumerate(rows):
                    if str(row.get("proposal_id", "")) != str(candidate.get("proposal_id", "")):
                        continue
                    if str(row.get("status", "")) != "pending":
                        continue
                    due_at = int(row.get("due_at", row.get("created_at", 0)) or 0)
                    if due_at > now:
                        continue
                    selected_index = index
                    candidate = dict(row)
                    break
                if selected_index < 0:
                    continue
                source_session_id = str(candidate.get("session_id", "") or source_root.name)
                rows[selected_index]["status"] = "relayed"
                rows[selected_index]["relayed_at"] = now
                rows[selected_index]["relayed_to_session_id"] = current_session_id
                save_queued_outreach(source_root, rows)
                relayed = dict(candidate)
                relayed["status"] = "pending"
                relayed["relayed_from_session_id"] = source_session_id
                relayed["relayed_from_session_root"] = str(source_root)
                relayed["session_id"] = current_session_id
                current_rows = load_queued_outreach(self.store.root)
                existing_ids = {
                    str(row.get("source_intent_id", "") or row.get("proposal_id", ""))
                    for row in current_rows
                }
                relay_id = str(relayed.get("source_intent_id", "") or relayed.get("proposal_id", ""))
                if relay_id in existing_ids:
                    self.store.append_log(
                        {
                            "event": "m14_2_audit",
                            "type": "OutboxRelayDuplicateSkippedEvent",
                            "at": now,
                            "proposal_id": str(relayed.get("proposal_id", "")),
                            "source_intent_id": str(relayed.get("source_intent_id", "")),
                            "from_session_id": source_session_id,
                            "to_session_id": current_session_id,
                            "runner_kind": "delivery_surface",
                            "persona_id": str(relayed.get("persona_id", "default") or "default"),
                            "session_id": current_session_id,
                            "correlation_id": "",
                            "engineering_proxy_label": "mvp_local_decoupled_self_loop",
                        }
                    )
                    continue
                current_rows.append(relayed)
                save_queued_outreach(self.store.root, current_rows)
                self.store.append_log(
                    {
                        "event": "m14_2_audit",
                        "type": "OutboxEntryRelayedEvent",
                        "at": now,
                        "proposal_id": str(relayed.get("proposal_id", "")),
                        "source_intent_id": str(relayed.get("source_intent_id", "")),
                        "from_session_id": source_session_id,
                        "to_session_id": current_session_id,
                        "runner_kind": "delivery_surface",
                        "persona_id": str(relayed.get("persona_id", "default") or "default"),
                        "session_id": current_session_id,
                        "correlation_id": "",
                        "engineering_proxy_label": "mvp_local_decoupled_self_loop",
                    }
                )
                return relayed
        return None

    def _close_m14_2_delivery_links(self, entry: Mapping[str, Any], *, now: int) -> None:
        intent_id = str(entry.get("source_intent_id", "") or "")
        if not intent_id:
            return
        try:
            from segmentum.dialogue.runtime.m14_2_scheduled_intents import (
                ScheduledIntentStore,
                close_scheduled_open_item,
            )

            intent_root_raw = str(entry.get("relayed_from_session_root", "") or "")
            intent_root = Path(intent_root_raw) if intent_root_raw else self.store.root
            intent_session_id = str(
                entry.get("relayed_from_session_id")
                or entry.get("session_id")
                or intent_root.name
                or self.store.root.name
            )
            intent_store = ScheduledIntentStore(
                intent_root,
                persona_id=str(entry.get("persona_id", "default") or "default"),
                session_id=intent_session_id,
            )
            intent_store.mark_status(
                intent_id,
                "delivered",
                now=now,
                proposal_id=str(entry.get("proposal_id", "")),
            )
            target_store = self.store
            try:
                if intent_root.resolve() != self.store.root.resolve():
                    target_store = MVPStateStore(intent_root, shared_root=self.store.shared_root)
            except OSError:
                target_store = self.store
            state = target_store.load()
            if close_scheduled_open_item(state, intent_id, status="closed"):
                target_store.save(state)
            self.store.append_log(
                {
                    "event": "m14_2_audit",
                    "type": "OutboxEntryDeliveredEvent",
                    "at": now,
                    "source_intent_id": intent_id,
                    "proposal_id": str(entry.get("proposal_id", "")),
                    "runner_kind": "delivery_surface",
                    "persona_id": intent_store.persona_id,
                    "session_id": intent_store.session_id,
                    "correlation_id": "",
                    "engineering_proxy_label": "mvp_local_decoupled_self_loop",
                }
            )
        except Exception:
            return

    def _append_idle_audit_events(
        self,
        state: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        skip_reason: str,
        now: int,
        force: bool = False,
    ) -> dict[str, Any]:
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        initiative = merge_idle_introspection_into_initiative(m13_state.get("initiative"))
        idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
        if not should_persist_idle_audit_events(
            idle,
            skip_reason=skip_reason,
            now=now,
            force=force,
        ):
            return state
        for event in events:
            self.store.append_log({"event": "m13_idle_audit", **event})
        mark_idle_audit_logged(idle, skip_reason=skip_reason, now=now)
        initiative["idle_introspection"] = idle
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        return state

    def maybe_run_idle_introspection(
        self,
        *,
        turn_index: int,
        user_active: bool = False,
    ) -> dict[str, Any]:
        state = self.store.load()
        now = _utc_timestamp()
        state, tick_check = evaluate_idle_tick(
            state,
            now=now,
            turn_index=turn_index,
            user_active=user_active,
        )
        events = list(tick_check.events)
        state = self._append_idle_audit_events(
            state,
            events,
            skip_reason=tick_check.skip_reason,
            now=now,
        )
        if tick_check.skip_reason:
            self.store.save(state)
            return {
                "ran_introspection": False,
                "skip_reason": tick_check.skip_reason,
                "events": events,
                "idle_result": None,
            }
        signals = tick_check.structural_signals
        if signals is None:
            self.store.save(state)
            return {
                "ran_introspection": False,
                "skip_reason": "no_structural_signal",
                "events": events,
                "idle_result": None,
            }
        state, pre_check = evaluate_idle_structural_pre_filter(
            state,
            now=now,
            turn_index=turn_index,
            signals=signals,
        )
        events.extend(pre_check.events)
        state = self._append_idle_audit_events(
            state,
            pre_check.events,
            skip_reason=pre_check.skip_reason,
            now=now,
            force=bool(pre_check.skip_reason),
        )
        if pre_check.skip_reason:
            self.store.save(state)
            return {
                "ran_introspection": False,
                "skip_reason": pre_check.skip_reason,
                "events": events,
                "idle_result": None,
            }
        idle_result = self.run_idle_introspection_turn(
            now=now,
            turn_index=turn_index,
            structural_signals=signals,
        )
        events.extend(idle_result.audit_events)
        return {
            "ran_introspection": True,
            "skip_reason": "",
            "events": events,
            "idle_result": {
                "ran_llm": idle_result.ran_llm,
                "reflection_focus": idle_result.reflection_focus,
                "outreach_recommendation": idle_result.outreach_recommendation,
                "diagnostics": idle_result.diagnostics,
            },
        }

    def run_idle_introspection_turn(
        self,
        *,
        now: int,
        turn_index: int,
        structural_signals: Any,
        queue_outreach: bool = False,
        allow_direct_outreach: bool = True,
        background_runner_kind: str = "",
    ) -> MVPIdleResult:
        """M14.0: conscious idle plan, named-owner patches, optional M13.3 outreach."""
        state = self.store.load()
        m13_state = normalize_m13_drive_state(state.get("m13_drive_state"))
        m13_state = merge_initiative_into_m13_state(m13_state)
        initiative = normalize_initiative_state(m13_state.get("initiative"))
        idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))

        sig_dict = (
            structural_signals.to_dict()
            if hasattr(structural_signals, "to_dict")
            else dict(structural_signals)
            if isinstance(structural_signals, Mapping)
            else {}
        )
        audit_events: list[dict[str, Any]] = [
            {
                "type": "IdleIntrospectionTickEvent",
                "turn_index": turn_index,
                "at": now,
                "idle_introspection.enabled": bool(idle.get("enabled")),
                "idle_introspection.user_opt_in": bool(idle.get("user_opt_in")),
                "structural_signals": sig_dict,
                "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
            },
            {
                "type": "M13DriveSummaryEvent",
                "turn_index": turn_index,
                "at": now,
                "boredom_band": sig_dict.get("boredom_band"),
                "path_feels_stale_proxy": sig_dict.get("path_feels_stale_proxy"),
                "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
            },
        ]

        try:
            from segmentum.dialogue.runtime.m14_2_scheduled_intents import ScheduledIntentStore

            scheduled_store = ScheduledIntentStore(
                self.store.root,
                persona_id=self.persona_name or "default",
                session_id=str(self.store.root.resolve()),
            )
            sig_dict["scheduled_intents"] = scheduled_store.list_intents()
        except Exception:
            sig_dict["scheduled_intents"] = []
        try:
            sig_dict["queued_outreach"] = load_queued_outreach(self.store.root)
        except Exception:
            sig_dict["queued_outreach"] = []

        idle_context = build_idle_context(
            state,
            m13_state=m13_state,
            structural_signals=sig_dict,
            turn_index=turn_index,
            now=now,
        )
        keywords = idle_retrieval_keywords(idle_context)
        retrieved = retrieve_memories(state, keywords, limit=8)
        expectation_set = normalize_expectations_for_efe(
            state,
            now=now,
            phase="idle",
            structural_signals=sig_dict,
        )
        bound_ids: list[str] = []
        for expectation in expectation_set.eligible_for_efe:
            bound_ids.extend(list(expectation.bound_memory_ids[:8]))
            bound_ids.extend(list(expectation.evidence_refs[:8]))
        if bound_ids:
            by_id = retrieve_memories_by_ids(state, bound_ids, limit=8)
            seen = {str(item.get("id", "")) for item in retrieved if item.get("id")}
            for item in by_id:
                item_id = str(item.get("id", ""))
                if item_id and item_id not in seen:
                    retrieved.append(item)
                    seen.add(item_id)
                if len(retrieved) >= 8:
                    break
        retrieved_ids = {str(item.get("id", "")) for item in retrieved if item.get("id")}

        audit_events.append(
            {
                "type": "IdleEfeRecallOrderEvent",
                "turn_index": turn_index,
                "at": now,
                "order": "retrieve_before_memory_efe",
                "bounded_retrieve_ids": list(dict.fromkeys(bound_ids))[:12],
                "retrieved_ids": sorted(retrieved_ids)[:12],
                "engineering_proxy_label": "mvp_local_proactive_alignment",
            }
        )

        audit_events.append(
            {
                "type": "MemoryDynamicsIdleSummaryEvent",
                "turn_index": turn_index,
                "at": now,
                "retrieval_keywords": keywords[:12],
                "retrieved_ids": sorted(retrieved_ids)[:12],
                "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
            }
        )

        m13_memory_efe_evaluation = evaluate_memory_efe(
            state,
            phase="idle",
            now=now,
            turn_index=turn_index,
            user_active=False,
            structural_signals=sig_dict,
            retrieved_memories=retrieved,
            episode_ledger=self._episode_ledger(),
        )
        m13_state, _m13_memory_efe_apply_events = apply_memory_efe_state(
            m13_state,
            m13_memory_efe_evaluation,
        )
        state["m13_drive_state"] = m13_state
        for event in m13_memory_efe_evaluation.events:
            if str(event.get("type", "")).startswith("BundleDecision"):
                self._record_bundle_policy_event(state, event)
        state["bundle_policy_linkage_diagnostics"] = dict(m13_memory_efe_evaluation.bundle_linkage_diagnostics)
        self.store.save(state)
        audit_events.extend(m13_memory_efe_evaluation.events)
        idle_drive_band_summary = self._idle_drive_band_summary(m13_state, state=state)
        sig_dict["idle_drive_band_summary"] = idle_drive_band_summary
        sig_dict["memory_efe_should_outreach"] = bool(m13_memory_efe_evaluation.should_outreach)
        audit_events.append(
            {
                "type": "IdleProactiveDriveRefreshEvent",
                "turn_index": turn_index,
                "at": now,
                "order": "recall_then_memory_efe_then_m13_drive_bands_before_target_selection",
                "retrieved_ids": sorted(retrieved_ids)[:12],
                "bounded_retrieve_ids": list(dict.fromkeys(bound_ids))[:12],
                "drive_band_summary": idle_drive_band_summary,
                "engineering_proxy_label": "mvp_local_proactive_alignment",
            }
        )

        idle_context = build_idle_context(
            state,
            m13_state=m13_state,
            structural_signals=sig_dict,
            turn_index=turn_index,
            now=now,
        )

        continuity = get_self_continuity_from_state(state)
        continuity = note_idle_tick(continuity)
        attach_self_continuity(state, continuity)
        snapshot = (
            build_self_continuity_snapshot(continuity) if should_run_self_review(continuity) else None
        )

        ran_llm = False
        llm_error = ""
        raw_plan: dict[str, Any] = {}
        llm_system = ""
        llm_user = ""
        llm_status = llm_configuration_status(self.llm)
        if not bool(llm_status.get("available")):
            llm_error = str(llm_status.get("reason", "") or "llm_unavailable")[:240]
            audit_events.append(
                {
                    "type": "IdleIntrospectionAbortEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "reason": llm_error,
                    "detail": llm_error,
                    "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                }
            )
        else:
            try:
                llm_system, llm_user = build_conscious_idle_prompt(
                    idle_context=idle_context,
                    retrieved_memories=retrieved,
                    turn_index=turn_index,
                    self_continuity_snapshot=snapshot,
                )
                raw_plan = self.llm.complete_json(system_prompt=llm_system, user_prompt=llm_user)
                ran_llm = True
            except BackgroundBudgetExhausted:
                raise
            except Exception as exc:
                llm_error = "llm_unavailable"
                audit_events.append(
                    {
                        "type": "IdleIntrospectionAbortEvent",
                        "turn_index": turn_index,
                        "at": now,
                        "reason": llm_error,
                        "detail": str(exc)[:240],
                        "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                    }
                )

        if not raw_plan or not isinstance(raw_plan, Mapping):
            raw_plan = build_structural_idle_plan(idle_context, retrieved_ids=retrieved_ids)
        else:
            normalized_probe = normalize_conscious_idle_plan(raw_plan)
            if normalized_probe == empty_conscious_idle_plan() and retrieved_ids:
                raw_plan = build_structural_idle_plan(idle_context, retrieved_ids=retrieved_ids)

        normalized_plan = normalize_conscious_idle_plan(raw_plan)

        plan = apply_idle_drive_rules(
            normalized_plan,
            idle_context=idle_context,
            structural_signals=sig_dict,
        )
        plan, focus_intent_events = apply_reflection_focus_intent(
            state,
            plan,
            now=now,
            turn_index=turn_index,
        )
        audit_events.extend(focus_intent_events)
        structural_alignment_events: list[dict[str, Any]] = []
        outreach_view = _mapping(normalized_plan.get("outreach_recommendation"))
        plan_should_outreach = bool(outreach_view.get("should_outreach"))
        plan_recommendation_reason = str(outreach_view.get("reason", "") or "")[:160]
        structural_target = select_proactive_target(
            state,
            m13_state,
            memory_efe_evaluation=m13_memory_efe_evaluation,
            structural_signals=sig_dict,
        )
        structural_target_payload = None
        if structural_target is not None:
            structural_target_payload = {
                "trigger": structural_target.trigger,
                "source_kind": structural_target.source_kind,
                "traceable_expectation_id": structural_target.traceable_expectation_id,
                "evidence_refs": list(structural_target.evidence_refs[:8]),
                "selection_reason_codes": list(structural_target.selection_reason_codes[:8]),
            }
        if plan_should_outreach and structural_target is None:
            mismatch_reason = classify_proactive_target_reject_reason(
                state,
                m13_state,
                memory_efe_evaluation=m13_memory_efe_evaluation,
                structural_signals=sig_dict,
            )
            outreach_mut = dict(outreach_view)
            outreach_mut["should_outreach"] = False
            outreach_mut["reason"] = "reflection_only"
            outreach_mut["m14_6_downgraded_by_structural_selector"] = True
            outreach_mut["mismatch_reason_code"] = mismatch_reason
            plan = {**plan, "outreach_recommendation": outreach_mut}
            structural_alignment_events.append(
                {
                    "type": "IdlePlanStructuralMismatchEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "selection_reason_codes": list(m13_memory_efe_evaluation.reason_codes[:8]),
                    "plan_recommendation_reason": plan_recommendation_reason,
                    "mismatch_reason_code": mismatch_reason,
                    "engineering_proxy_label": "mvp_local_proactive_alignment",
                }
            )
        elif plan_should_outreach and structural_target is not None:
            structural_alignment_events.append(
                {
                    "type": "IdlePlanStructuralAgreementEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "reason": "plan_and_selector_agree",
                    "selected_target": structural_target_payload,
                    "plan_recommendation_reason": plan_recommendation_reason,
                    "engineering_proxy_label": "mvp_local_proactive_alignment",
                }
            )
        elif not plan_should_outreach and structural_target is not None:
            outreach_mut = dict(_mapping(plan.get("outreach_recommendation")))
            outreach_mut["should_outreach"] = False
            outreach_mut["m14_6_reflect_only_preferred"] = True
            plan = {**plan, "outreach_recommendation": outreach_mut}
            structural_alignment_events.append(
                {
                    "type": "IdlePlanStructuralAgreementEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "reason": "reflect_only_preferred",
                    "selected_target": structural_target_payload,
                    "plan_recommendation_reason": plan_recommendation_reason,
                    "engineering_proxy_label": "mvp_local_proactive_alignment",
                }
            )

        audit_events.append(
            {
                "type": "IdleIntrospectionPlanEvent",
                "turn_index": turn_index,
                "at": now,
                "plan": plan,
                "ran_llm": ran_llm,
                "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
            }
        )
        audit_events.extend(structural_alignment_events)
        audit_events.extend(
            apply_idle_self_expectation_review(
                state,
                review_proposals=plan.get("self_expectation_review_proposals"),
                now=now,
                turn_index=turn_index,
            )
        )

        session_counts = count_session_idle_patches(state)
        patch_proposal = _mapping(plan.get("self_cognition_patch_proposal"))
        low_background_confidence = (
            bool(background_runner_kind)
            and bool(patch_proposal.get("apply"))
            and _bounded_float(patch_proposal.get("confidence")) < MIN_BASELINE_UPDATE_CONFIDENCE
        )
        if low_background_confidence:
            patch_result = OwnerCommitResult(
                committed=False,
                events=[
                    {
                        "type": "SelfCognitionPatchRejectedEvent",
                        "turn_index": turn_index,
                        "at": now,
                        "evidence_refs": _string_list(patch_proposal.get("evidence_refs"), limit=8),
                        "violation_codes": ["confidence_below_threshold"],
                        "reason": "confidence_below_threshold",
                        "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                    }
                ],
                violation_codes=["confidence_below_threshold"],
            )
            continuity, sc_events = apply_self_cognition_patch_to_continuity(
                continuity,
                patch_proposal,
                now=now,
                retrieved_ids=retrieved_ids,
            )
            audit_events.extend(patch_result.events)
            audit_events.extend(sc_events)
        else:
            patch_result = SelfCognitionPatchOwner.validate_and_commit(
                state,
                patch_proposal,
                retrieved_ids=retrieved_ids,
                turn_index=turn_index,
                now=now,
                session_patches=int(session_counts.get("self_cognition", 0)),
            )
            audit_events.extend(patch_result.events)
        if patch_result.committed:
            continuity, sc_events = apply_self_cognition_patch_to_continuity(
                continuity,
                patch_proposal,
                now=now,
                retrieved_ids=retrieved_ids,
            )
            audit_events.extend(sc_events)
        if should_run_self_review(continuity):
            continuity, review_events = run_self_review_tick(continuity, now=now)
            audit_events.extend(review_events)
            m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            bg["self_reviews_today"] = int(bg.get("self_reviews_today", 0) or 0) + 1
            bg["self_reviews_lifetime"] = int(bg.get("self_reviews_lifetime", 0) or 0) + 1
            initiative["background_continuity"] = bg
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
        attach_self_continuity(state, continuity)

        open_result = OpenItemPatchOwner.validate_and_commit(
            state,
            [row for row in plan.get("open_item_proposals", []) if isinstance(row, Mapping)],
            retrieved_ids=retrieved_ids,
            turn_index=turn_index,
            now=now,
            session_patches=int(session_counts.get("open_items", 0)),
        )
        audit_events.extend(open_result.events)

        mem_intents, mem_violations = MemoryConsolidationOwner.translate_to_intents(
            [row for row in plan.get("memory_consolidation_proposals", []) if isinstance(row, Mapping)],
            retrieved_ids=retrieved_ids,
        )
        if mem_violations:
            audit_events.append(
                {
                    "type": "MemoryConsolidationIntentEvent",
                    "turn_index": turn_index,
                    "at": now,
                    "committed": False,
                    "violation_codes": mem_violations[:6],
                    "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                }
            )
        mem_apply = MemoryConsolidationOwner.apply_intents(
            state,
            mem_intents,
            turn_index=turn_index,
            now=now,
            session_count=int(session_counts.get("memory", 0)),
        )
        audit_events.extend(mem_apply.events)

        outreach = _mapping(plan.get("outreach_recommendation"))
        outreach_outcome = ""
        if bool(outreach.get("should_outreach")):
            locked = None
            target = select_proactive_target(
                state,
                m13_state,
                memory_efe_evaluation=m13_memory_efe_evaluation,
                structural_signals=sig_dict,
            )
            if target is not None:
                locked = build_proposal_from_target(
                    target,
                    now=now,
                    initiative=initiative,
                )
            if locked is None:
                outreach_outcome = "no_traceable_proactive_target"
            else:
                audit_events.append(
                    {
                        "type": "IdleOutreachProposalEvent",
                        "turn_index": turn_index,
                        "at": now,
                        "proposal_id": locked.proposal_id,
                        "trigger": locked.trigger,
                        "traceable_expectation_id": locked.traceable_expectation_id,
                        "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                    }
                )
                if queue_outreach:
                    initiative = merge_background_continuity_into_initiative(
                        normalize_initiative_state(m13_state.get("initiative"))
                    )
                    bg = normalize_background_continuity_state(initiative.get("background_continuity"))
                    ttl = int(bg.get("queued_outreach_ttl_seconds", 0) or 0)
                    entry = enqueue_outreach_proposal(
                        self.store.root,
                        proposal=locked.to_dict(),
                        now=now,
                        ttl_seconds=ttl,
                        drive_snapshot={
                            "boredom_band": sig_dict.get("boredom_band"),
                            "memory_efe_should_outreach": m13_memory_efe_evaluation.should_outreach,
                        },
                    )
                    m13_state, settlement_events = register_memory_efe_outreach_settlement(
                        m13_state,
                        evaluation=m13_memory_efe_evaluation,
                        proposal_id=locked.proposal_id,
                        delivery_status="queued",
                        now=now,
                        turn_index=turn_index,
                    )
                    state["m13_drive_state"] = m13_state
                    audit_events.extend(settlement_events)
                    audit_events.append(
                        {
                            "type": "QueuedOutreachProposalEvent",
                            "turn_index": turn_index,
                            "at": now,
                            "proposal_id": entry.get("proposal_id", ""),
                            "expires_at": entry.get("expires_at"),
                            "runner_kind": background_runner_kind,
                            "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                        }
                    )
                    outreach_outcome = "queued"
                elif not allow_direct_outreach:
                    audit_events.append(
                        {
                            "type": "IdleOutreachDeferredEvent",
                            "turn_index": turn_index,
                            "at": now,
                            "proposal_id": locked.proposal_id,
                            "runner_kind": background_runner_kind,
                            "reason": "direct_delivery_disabled",
                            "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
                        }
                    )
                    outreach_outcome = "deferred_to_outbox"
                else:
                    check_state, check = evaluate_proactive_initiative(
                        state,
                        now=now,
                        turn_index=turn_index,
                        manual_continue=False,
                        locked_proposal=locked,
                        llm=self.llm,
                        structural_signals=sig_dict,
                        memory_efe_evaluation=m13_memory_efe_evaluation,
                    )
                    state = check_state
                    self.store.save(state)
                    m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
                    for event in check.events:
                        self.store.append_log({"event": "m13_proactive_audit", **event})
                    if check.proposal is not None:
                        proactive_result = self.run_proactive_turn(
                            proposal_id=check.proposal.proposal_id,
                            turn_index=turn_index,
                            now=now,
                        )
                        outreach_outcome = (
                            "delivered"
                            if str(proactive_result.reply or "").strip()
                            else str(
                                proactive_result.diagnostics.get("reason_code")
                                or proactive_result.diagnostics.get("suppression_reason", "suppressed")
                            )
                        )
                        if outreach_outcome == "delivered":
                            m13_state = mark_outreach_via_introspection(
                                merge_initiative_into_m13_state(self.store.load().get("m13_drive_state", {}))
                            )
                            m13_state, settlement_events = register_memory_efe_outreach_settlement(
                                m13_state,
                                evaluation=m13_memory_efe_evaluation,
                                proposal_id=locked.proposal_id,
                                delivery_status="delivered",
                                now=now,
                                turn_index=turn_index,
                                m15_episode_id=str(
                                    _mapping(proactive_result.diagnostics.get("m15_episode")).get("episode_id", "")
                                ),
                            )
                            state["m13_drive_state"] = m13_state
                            audit_events.extend(settlement_events)
                    else:
                        outreach_outcome = check.suppression_reason_code or check.suppression_reason or "suppressed"
        else:
            outreach_outcome = str(outreach.get("reason", "reflection_only") or "reflection_only")

        for event in audit_events:
            self.store.append_log({"event": "m14_idle_audit", **event})

        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
        initiative = merge_idle_introspection_into_initiative(m13_state.get("initiative"))
        idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
        idle["last_conscious_idle_plan"] = plan
        idle["last_outreach_outcome"] = outreach_outcome[:64]
        initiative["idle_introspection"] = idle
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = mark_idle_introspection_consumed(m13_state, now=now)
        self.store.save(state)

        reflection_focus = plan.get("reflection_focus")
        diagnostics = {
            "mvp_runtime": True,
            "idle_introspection_turn": True,
            "not_user_requested_current_turn": True,
            "ran_llm": ran_llm,
            "llm_error": llm_error,
            "conscious_idle_plan": plan,
            "outreach_outcome": outreach_outcome,
            "structural_signals": sig_dict,
            "m13_memory_efe": prompt_safe_m13_memory_efe_diagnostics(m13_memory_efe_evaluation),
            "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
        }
        return MVPIdleResult(
            ran_llm=ran_llm,
            reflection_focus=dict(reflection_focus) if isinstance(reflection_focus, Mapping) else None,
            self_cognition_patch_proposal=_mapping(plan.get("self_cognition_patch_proposal")) or None,
            memory_consolidation_proposals=[
                dict(row) for row in plan.get("memory_consolidation_proposals", []) if isinstance(row, Mapping)
            ],
            open_item_proposals=[
                dict(row) for row in plan.get("open_item_proposals", []) if isinstance(row, Mapping)
            ],
            outreach_recommendation=dict(outreach),
            audit_events=audit_events,
            skip_reason="",
            diagnostics=diagnostics,
            llm_calls_delta=int(getattr(self.llm, "llm_calls_delta", 1 if ran_llm else 0) or 0),
            tokens_delta=int(getattr(self.llm, "tokens_delta", 0) or 0),
        )

    def _mark_recalled(self, state: dict[str, Any], retrieved: list[Mapping[str, Any]], now: int) -> None:
        ids = {str(item.get("id", "")) for item in retrieved if item.get("id")}
        if not ids:
            return
        for key in ("short_term_memory", "long_term_memory"):
            rows = state.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and str(row.get("id", "")) in ids:
                    row["last_recalled_at"] = now
                    row["recall_count"] = int(row.get("recall_count", 0) or 0) + 1
                    row["salience"] = min(1.0, float(row.get("salience", 0.2) or 0.2) + 0.05)

    def _memory_gate_commit_count(self, state: Mapping[str, Any], proposer: str) -> int:
        events = state.get("memory_gate_audit_tail", [])
        if not isinstance(events, list):
            return 0
        return sum(
            1
            for event in events
            if isinstance(event, Mapping)
            and str(event.get("type", "")) == "MemoryGateCommitEvent"
            and str(event.get("proposer", "")) == proposer
        )

    def _record_memory_gate_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        events = state.setdefault("memory_gate_audit_tail", [])
        if not isinstance(events, list):
            events = []
            state["memory_gate_audit_tail"] = events
        events.append(dict(event))
        state["memory_gate_audit_tail"] = events[-80:]

    def _record_bundle_policy_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        events = state.setdefault("bundle_policy_audit_tail", [])
        if not isinstance(events, list):
            events = []
            state["bundle_policy_audit_tail"] = events
        events.append(dict(event))
        state["bundle_policy_audit_tail"] = events[-80:]

    def _recent_memory_gate_fingerprints(self, state: Mapping[str, Any], intent: MemoryWriteIntent) -> set[str]:
        events = state.get("memory_gate_audit_tail", [])
        if not isinstance(events, list):
            return set()
        fingerprints: set[str] = set()
        for event in events[-24:]:
            if not isinstance(event, Mapping):
                continue
            if str(event.get("proposer", "")) != intent.proposer:
                continue
            fingerprint = str(event.get("intent_fingerprint", "") or "")
            if fingerprint:
                fingerprints.add(fingerprint)
        return fingerprints

    def _evaluate_memory_gate(
        self,
        state: dict[str, Any],
        intent: MemoryWriteIntent,
        *,
        turn_index: int,
        now: int,
        store_target: str = "",
        store_id: str = "",
    ) -> bool:
        decision = MemoryGate().evaluate(
            intent,
            proposer_commits_this_session=self._memory_gate_commit_count(state, intent.proposer),
            recent_intent_fingerprints=self._recent_memory_gate_fingerprints(state, intent),
        )
        event_type = "MemoryGateCommitEvent" if decision.commit else "MemoryGateRejectedEvent"
        self._record_memory_gate_event(
            state,
            memory_gate_event(
                event_type=event_type,
                intent=intent,
                decision=decision,
                turn_index=turn_index,
                now=now,
                store_target=store_target,
                store_id=store_id,
            ),
        )
        return decision.commit

    def _apply_memory_write_candidates(
        self,
        state: dict[str, Any],
        candidates: Any,
        *,
        now: int,
        turn_index: int = 0,
        user_id: str = "",
        display_name: str = "",
        session_id: str = "",
        ingress_evidence_band: str = "",
        default_shareability: str = "default_social",
        restriction_reason: str = "",
        group_turn_binding: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(candidates, list):
            return []
        applied: list[dict[str, Any]] = []
        source_participant_id = str(
            _mapping(group_turn_binding).get("current_speaker_participant_id", "")
            or user_id
            or ""
        ).strip()
        source_audience_participant_ids = _bounded_string_list(
            _mapping(group_turn_binding).get("visible_participant_ids"),
            limit=8,
            item_max_chars=64,
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = str(candidate.get("content", "")).strip()
            evidence = str(candidate.get("evidence", "")).strip()
            confidence = _bounded_float(candidate.get("confidence"), default=0.0)
            if not content or not evidence or confidence < 0.60:
                continue
            salience = _bounded_float(candidate.get("salience"), default=0.35)
            target = str(candidate.get("target", "short_term")).strip()
            evidence_refs = _string_list(candidate.get("evidence_refs"), limit=8)
            if not evidence_refs and evidence:
                evidence_refs = [f"evidence:{abs(hash(evidence)) % 100000}"]
            row = {
                "id": f"{'ltm' if target == 'long_term' else 'stm'}_candidate_{now}_{len(applied)}",
                "kind": str(candidate.get("kind", "episode")).strip() or "episode",
                "content": content,
                "salience": salience,
                "confidence": confidence,
                "keywords": _string_list(candidate.get("keywords"), limit=8),
                "reason": str(candidate.get("reason", "")),
                "evidence": evidence,
                "evidence_refs": evidence_refs,
                "source": "memory_dynamics_adapter",
                "created_at": now,
                "last_recalled_at": None,
                "recall_count": 0,
            }
            intent = intent_from_mapping(
                candidate,
                target="long_term" if target == "long_term" or salience >= 0.68 else "short_term",
                kind=str(row["kind"]),
                content=content,
                confidence=confidence,
                evidence_refs=evidence_refs,
                source="write_candidate",
                proposer="memory_dynamics_adapter",
                audit_reason=str(candidate.get("reason", "memory_dynamics_write_candidate") or "memory_dynamics_write_candidate"),
                value_proxy=salience,
                surprise_proxy=max(salience, _bounded_float(candidate.get("surprise_proxy"), default=0.35)),
                identity_relevance=_bounded_float(candidate.get("identity_relevance"), default=0.0),
            )
            store_target = "long_term" if target == "long_term" or salience >= 0.68 else "short_term"
            if not self._evaluate_memory_gate(
                state,
                intent,
                turn_index=turn_index,
                now=now,
                store_target=store_target,
                store_id=str(row["id"]),
            ):
                continue
            shareability = _shareability_for_memory_text(
                content,
                evidence,
                candidate.get("keywords"),
                requested=str(candidate.get("shareability", default_shareability) or default_shareability),
            )
            _stamp_memory_policy(
                row,
                user_id=user_id,
                display_name=display_name,
                shareability=shareability,
                restriction_reason=_restriction_reason_for_shareability(
                    shareability,
                    existing=str(candidate.get("restriction_reason", restriction_reason) or restriction_reason),
                ),
                confidence=confidence,
                source_participant_id=source_participant_id,
                source_audience_participant_ids=source_audience_participant_ids,
                session_id=session_id,
                turn_index=turn_index,
                ingress_evidence_band=ingress_evidence_band,
            )
            if store_target == "long_term":
                state.setdefault("long_term_memory", []).append(row)
            else:
                state.setdefault("short_term_memory", []).append(row)
            applied.append(row)
        return applied

    def _apply_post_reply_memory_updates(
        self,
        state: dict[str, Any],
        updates: Any,
        *,
        now: int,
        turn_index: int = 0,
        user_id: str = "",
        display_name: str = "",
        session_id: str = "",
        ingress_evidence_band: str = "",
        default_shareability: str = "default_social",
        group_turn_binding: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(updates, list):
            return []
        applied: list[dict[str, Any]] = []
        source_participant_id = str(
            _mapping(group_turn_binding).get("current_speaker_participant_id", "")
            or user_id
            or ""
        ).strip()
        source_audience_participant_ids = _bounded_string_list(
            _mapping(group_turn_binding).get("visible_participant_ids"),
            limit=8,
            item_max_chars=64,
        )
        for item in updates:
            if not isinstance(item, Mapping):
                continue
            content = str(item.get("content", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            confidence = _bounded_float(item.get("confidence"), default=0.0)
            kind = str(item.get("kind", "")).strip()
            if not content or not evidence or confidence < 0.60:
                continue
            evidence_refs = _string_list(item.get("evidence_refs"), limit=8)
            if not evidence_refs and evidence:
                evidence_refs = [f"evidence:{abs(hash(evidence)) % 100000}"]
            if kind == "conversation_habit":
                intent = intent_from_mapping(
                    item,
                    target="short_term",
                    kind="habit",
                    content=content,
                    confidence=confidence,
                    evidence_refs=evidence_refs,
                    source="post_reply_observer",
                    proposer="post_reply_observer",
                    audit_reason=str(item.get("reason", "post_reply_memory_update") or "post_reply_memory_update"),
                    value_proxy=_bounded_float(item.get("value_proxy"), default=0.5),
                    surprise_proxy=_bounded_float(item.get("surprise_proxy"), default=0.35),
                    identity_relevance=_bounded_float(item.get("identity_relevance"), default=0.0),
                )
                if not self._evaluate_memory_gate(
                    state,
                    intent,
                    turn_index=turn_index,
                    now=now,
                    store_target="habit_traits",
                    store_id=f"habit_{now}_{len(applied)}",
                ):
                    continue
                habits = state.setdefault("habit_traits", {})
                if not isinstance(habits, dict):
                    habits = {}
                    state["habit_traits"] = habits
                target = habits.setdefault("learned_conversation_habits", [])
                if not isinstance(target, list):
                    target = []
                    habits["learned_conversation_habits"] = target
                row = {
                    "content": content,
                    "evidence": evidence,
                    "evidence_refs": evidence_refs,
                    "confidence": confidence,
                    "source": "post_reply_observer",
                    "created_at": now,
                }
                shareability = _shareability_for_memory_text(content, evidence, requested=default_shareability)
                _stamp_memory_policy(
                    row,
                    user_id=user_id,
                    display_name=display_name,
                    shareability=shareability,
                    restriction_reason=_restriction_reason_for_shareability(
                        shareability,
                        existing="post_reply_update",
                    ),
                    confidence=confidence,
                    source_participant_id=source_participant_id,
                    source_audience_participant_ids=source_audience_participant_ids,
                    session_id=session_id,
                    turn_index=turn_index,
                    ingress_evidence_band=ingress_evidence_band,
                )
                target.append(row)
                applied.append(row)
                abstract = _abstract_relationship_constraint_from_feedback(content, evidence)
                if abstract is not None:
                    _append_relationship_value_memory(
                        state,
                        user_id=user_id,
                        summary=abstract[0],
                        prediction_constraint=abstract[1],
                        evidence=evidence,
                        source="post_reply_observer",
                        confidence=confidence,
                        created_at=now,
                        session_id=session_id,
                        turn_index=turn_index,
                        source_participant_id=source_participant_id,
                        source_audience_participant_ids=source_audience_participant_ids,
                        ingress_evidence_band=ingress_evidence_band,
                    )
                continue
            row = {
                "id": f"stm_post_reply_{now}_{len(applied)}",
                "kind": kind or "episode",
                "content": content,
                "salience": 0.45,
                "confidence": confidence,
                "keywords": _string_list(item.get("keywords"), limit=6),
                "reason": str(item.get("reason", "post_reply_observer")),
                "evidence": evidence,
                "evidence_refs": evidence_refs,
                "source": "post_reply_observer",
                "created_at": now,
                "recall_count": 0,
            }
            intent = intent_from_mapping(
                item,
                target="short_term",
                kind=str(row["kind"]),
                content=content,
                confidence=confidence,
                evidence_refs=evidence_refs,
                source="post_reply_observer",
                proposer="post_reply_observer",
                audit_reason=str(item.get("reason", "post_reply_memory_update") or "post_reply_memory_update"),
                value_proxy=_bounded_float(item.get("value_proxy"), default=0.45),
                surprise_proxy=_bounded_float(item.get("surprise_proxy"), default=0.35),
                identity_relevance=_bounded_float(item.get("identity_relevance"), default=0.0),
            )
            if not self._evaluate_memory_gate(
                state,
                intent,
                turn_index=turn_index,
                now=now,
                store_target="short_term",
                store_id=str(row["id"]),
            ):
                continue
            shareability = _shareability_for_memory_text(
                content,
                evidence,
                item.get("keywords"),
                requested=default_shareability,
            )
            _stamp_memory_policy(
                row,
                user_id=user_id,
                display_name=display_name,
                shareability=shareability,
                restriction_reason=_restriction_reason_for_shareability(
                    shareability,
                    existing="post_reply_update",
                ),
                confidence=confidence,
                source_participant_id=source_participant_id,
                source_audience_participant_ids=source_audience_participant_ids,
                session_id=session_id,
                turn_index=turn_index,
                ingress_evidence_band=ingress_evidence_band,
            )
            state.setdefault("short_term_memory", []).append(row)
            applied.append(row)
        return applied

    def _apply_pacing_feedback_habit(
        self,
        state: dict[str, Any],
        *,
        user_text: str,
        user_id: str = "",
        display_name: str = "",
        now: int | None = None,
        turn_index: int = 0,
        session_id: str = "",
        ingress_evidence_band: str = "",
        group_turn_binding: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not _has_any_marker(user_text, _BREVITY_FEEDBACK_MARKERS):
            return []
        habits = state.setdefault("habit_traits", {})
        if not isinstance(habits, dict):
            habits = {}
            state["habit_traits"] = habits
        target = habits.setdefault("learned_conversation_habits", [])
        if not isinstance(target, list):
            target = []
            habits["learned_conversation_habits"] = target
        content = "用户偏好闲聊时更短、更像分轮聊天；避免每轮把共情、角色表演、追问全部塞成一长串。"
        existing = {_habit_text(item) for item in target if _habit_text(item)}
        if content in existing:
            return []
        row = {
            "content": content,
            "evidence": str(user_text).strip()[:240],
            "confidence": 0.82,
            "source": "pacing_feedback",
        }
        evidence_ref = f"pacing_feedback_{now or _utc_timestamp()}"
        intent = intent_from_mapping(
            row,
            target="short_term",
            kind="habit",
            content=content,
            confidence=0.82,
            evidence_refs=[evidence_ref],
            source="pacing_feedback",
            proposer="pacing_feedback",
            audit_reason="pacing_feedback_habit",
            value_proxy=0.65,
            surprise_proxy=0.45,
            identity_relevance=0.2,
        )
        if not self._evaluate_memory_gate(
            state,
            intent,
            turn_index=turn_index,
            now=int(now or _utc_timestamp()),
            store_target="habit_traits",
            store_id=evidence_ref,
        ):
            return []
        row["evidence_refs"] = [evidence_ref]
        row["created_at"] = int(now or _utc_timestamp())
        shareability = _shareability_for_memory_text(content, str(user_text), requested="default_social")
        source_participant_id = str(
            _mapping(group_turn_binding).get("current_speaker_participant_id", "")
            or user_id
            or ""
        ).strip()
        source_audience_participant_ids = _bounded_string_list(
            _mapping(group_turn_binding).get("visible_participant_ids"),
            limit=8,
            item_max_chars=64,
        )
        _stamp_memory_policy(
            row,
            user_id=user_id,
            display_name=display_name,
            shareability=shareability,
            restriction_reason=_restriction_reason_for_shareability(
                shareability,
                existing="pacing_feedback",
            ),
            confidence=0.82,
            source_participant_id=source_participant_id,
            source_audience_participant_ids=source_audience_participant_ids,
            session_id=session_id,
            turn_index=turn_index,
            ingress_evidence_band=ingress_evidence_band,
        )
        target.append(row)
        abstract = _abstract_relationship_constraint_from_feedback(content, str(user_text))
        if abstract is not None:
            _append_relationship_value_memory(
                state,
                user_id=user_id,
                summary=abstract[0],
                prediction_constraint=abstract[1],
                evidence=str(user_text).strip()[:240],
                source="pacing_feedback",
                confidence=0.82,
                created_at=now,
                session_id=session_id,
                turn_index=turn_index,
                source_participant_id=source_participant_id,
                source_audience_participant_ids=source_audience_participant_ids,
                ingress_evidence_band=ingress_evidence_band,
            )
        return [row]

    def _apply_sharing_regret_feedback(
        self,
        state: dict[str, Any],
        *,
        user_text: str,
        current_user_id: str,
        now: int,
        turn_index: int = 0,
    ) -> dict[str, Any]:
        temporal = _mapping(state.get("temporal_state"))
        trace = _mapping(temporal.get("last_share_trace"))
        social = state.setdefault("social_sharing_policy", {})
        if not isinstance(social, dict):
            social = {}
            state["social_sharing_policy"] = social
        regret_bias = _bounded_float(social.get("regret_bias"), default=0.0)
        had_cross_user_share = bool(trace.get("had_cross_user_memory", False))
        same_user = str(trace.get("user_id", "")).strip() == str(current_user_id or "").strip()
        negative = _sharing_feedback_negative(user_text)
        if had_cross_user_share and same_user and negative:
            updates = social.setdefault("learned_boundaries", [])
            if isinstance(updates, list):
                row = {
                    "content": "跨用户社交转述在负反馈后应显著提高成本，优先抽象化或保留。",
                    "evidence": str(user_text).strip()[:240],
                    "confidence": 0.82,
                    "source": "sharing_regret_feedback",
                    "created_at": now,
                }
                evidence_ref = f"sharing_regret_{now}_{len(updates)}"
                intent = intent_from_mapping(
                    row,
                    target="short_term",
                    kind="learned_boundary",
                    content=str(row["content"]),
                    confidence=0.82,
                    evidence_refs=[evidence_ref],
                    source="sharing_regret_feedback",
                    proposer="sharing_regret_feedback",
                    audit_reason="sharing_regret_feedback",
                    value_proxy=0.75,
                    surprise_proxy=0.55,
                    identity_relevance=0.25,
                )
                if self._evaluate_memory_gate(
                    state,
                    intent,
                    turn_index=turn_index,
                    now=now,
                    store_target="social_sharing_policy.learned_boundaries",
                    store_id=evidence_ref,
                ):
                    row["evidence_refs"] = [evidence_ref]
                    updates.append(row)
        regret_bias = update_regret_bias(
            previous_regret_bias=regret_bias,
            negative_feedback=negative,
            had_cross_user_share=had_cross_user_share,
            same_audience_user=same_user,
        )
        social["regret_bias"] = round(regret_bias, 6)
        return {
            "negative_feedback_detected": negative,
            "had_cross_user_share": had_cross_user_share,
            "same_user_as_previous_turn": same_user,
            "regret_bias": round(regret_bias, 6),
        }

    def _apply_expectation_results(
        self,
        state: dict[str, Any],
        results: Any,
        *,
        now: int,
        turn_index: int = 0,
        user_id: str = "",
        display_name: str = "",
        entity_binding: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(results, list):
            return
        pending = state.get("pending_expectations", [])
        if not isinstance(pending, list):
            pending = []
        normalized_results: list[dict[str, Any]] = []
        current_aliases = {
            alias.casefold()
            for alias in _string_list(
                _mapping(_mapping(entity_binding).get("current_interlocutor")).get("aliases"),
                limit=16,
            )
        }
        target_person = str(_mapping(entity_binding).get("target_person") or "").strip()
        for item in results:
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            evidence_text = _joined_text(payload.get("evidence"), payload.get("content")).casefold()
            if target_person and any(alias and alias in evidence_text for alias in current_aliases):
                if target_person.casefold() not in current_aliases:
                    payload["entity_binding_conflict"] = "current_user_alias_mentioned_while_target_is_third_party"
                    if str(payload.get("status", "")) == "confirmed":
                        payload["status"] = "uncertain"
            normalized_results.append(payload)
        resolved_ids = {
            str(item.get("id"))
            for item in normalized_results
            if str(item.get("status", "")) in {"confirmed", "violated"}
        }
        supersede_statuses = {"superseded", "topic_shifted"}
        superseded_ids = {
            str(item.get("id"))
            for item in normalized_results
            if str(item.get("status", "")).casefold() in supersede_statuses
        }
        if resolved_ids or superseded_ids:
            updated_pending: list[Any] = []
            for item in pending:
                if not isinstance(item, Mapping):
                    updated_pending.append(item)
                    continue
                row_id = str(item.get("id", "") or "")
                if row_id and row_id in resolved_ids:
                    continue
                if row_id and row_id in superseded_ids:
                    expired_row = dict(item)
                    expired_row["status"] = "expired"
                    expired_row["expired_at"] = now
                    expired_row["expired_reason_code"] = "user_topic_shift"
                    updated_pending.append(expired_row)
                    continue
                updated_pending.append(item)
            state["pending_expectations"] = updated_pending
        history = state.setdefault("short_term_memory", [])
        if isinstance(history, list):
            for payload in normalized_results:
                content = json.dumps(payload, ensure_ascii=False)
                salience = min(1.0, float(payload.get("self_update_pressure", 0.2) or 0.2))
                evidence_refs = _string_list(payload.get("evidence_refs"), limit=8)
                if not evidence_refs:
                    ref = str(payload.get("id", "") or "").strip()
                    if ref:
                        evidence_refs = [ref]
                row = {
                    "id": f"stm_expectation_{now}_{len(history)}",
                    "kind": "expectation_result",
                    "content": content,
                    "salience": salience,
                    "keywords": ["预期验证", str(payload.get("status", ""))],
                    "source": "conscious_loop",
                    "created_at": now,
                    "recall_count": 0,
                    "source_user_id": user_id,
                    "source_display_name": display_name,
                    "shareability": "default_social",
                }
                if evidence_refs:
                    row["evidence_refs"] = evidence_refs
                intent = intent_from_mapping(
                    payload,
                    target="short_term",
                    kind="expectation_result",
                    content=content,
                    confidence=_bounded_float(payload.get("confidence"), default=0.72),
                    evidence_refs=evidence_refs,
                    source="conscious_loop",
                    proposer="expectation_result_observer",
                    audit_reason="expectation_result_observer",
                    value_proxy=salience,
                    surprise_proxy=salience,
                    identity_relevance=_bounded_float(payload.get("identity_relevance"), default=0.0),
                )
                if not self._evaluate_memory_gate(
                    state,
                    intent,
                    turn_index=turn_index,
                    now=now,
                    store_target="short_term",
                    store_id=str(row["id"]),
                ):
                    continue
                history.append(row)

    def _apply_thinking_writes(
        self,
        state: dict[str, Any],
        thinking: Mapping[str, Any],
        *,
        user_text: str,
        now: int,
        turn_index: int = 0,
        user_id: str = "",
        display_name: str = "",
        session_id: str = "",
        ingress_evidence_band: str = "",
        explicit_secrecy: bool = False,
        memory_dynamics: Mapping[str, Any] | None = None,
        group_turn_binding: Mapping[str, Any] | None = None,
    ) -> None:
        short = state.setdefault("short_term_memory", [])
        turn_memory_committed = False
        source_participant_id = str(
            _mapping(group_turn_binding).get("current_speaker_participant_id", "")
            or user_id
            or ""
        ).strip()
        source_audience_participant_ids = _bounded_string_list(
            _mapping(group_turn_binding).get("visible_participant_ids"),
            limit=8,
            item_max_chars=64,
        )
        if isinstance(short, list):
            assistant_reply = str(thinking.get("reply", "")).strip()
            turn_memory_id = f"stm_turn_{now}"
            row = {
                "id": turn_memory_id,
                "kind": "dialogue_turn",
                "content": str(user_text).strip(),
                "user_text": str(user_text).strip(),
                "assistant_reply": assistant_reply,
                "assistant_reply_use_as_fact": False,
                "salience": 0.35,
                "keywords": _string_list(thinking.get("memory_dynamics_note"), limit=4),
                "evidence_refs": [turn_memory_id],
                "source": "dialogue",
                "created_at": now,
                "recall_count": 0,
            }
            intent = intent_from_mapping(
                row,
                target="short_term",
                kind="dialogue_turn",
                content=str(user_text).strip() or assistant_reply,
                confidence=0.85,
                evidence_refs=[turn_memory_id],
                source="thinking_writes",
                proposer="dialogue_turn_capture",
                audit_reason="dialogue_turn_capture",
                value_proxy=0.4,
                surprise_proxy=0.4,
            )
            if not self._evaluate_memory_gate(
                state,
                intent,
                turn_index=turn_index,
                now=now,
                store_target="short_term",
                store_id=turn_memory_id,
            ):
                short = state.setdefault("short_term_memory", [])
                if not isinstance(short, list):
                    state["short_term_memory"] = []
                short = []
                state["short_term_memory"] = short
            else:
                shareability = _shareability_for_memory_text(
                    user_text,
                    explicit_secret=explicit_secrecy,
                )
                _stamp_memory_policy(
                    row,
                    user_id=user_id,
                    display_name=display_name,
                    shareability=shareability,
                    restriction_reason=_restriction_reason_for_shareability(
                        shareability,
                        explicit_secret=explicit_secrecy,
                    ),
                    confidence=0.85,
                    source_participant_id=source_participant_id,
                    source_audience_participant_ids=source_audience_participant_ids,
                    session_id=session_id,
                    turn_index=turn_index,
                    ingress_evidence_band=ingress_evidence_band,
                )
                short.append(row)
                state["short_term_memory"] = short[-24:]
                turn_memory_committed = True

        for write in thinking.get("memory_writes", []) or []:
            if not isinstance(write, Mapping):
                continue
            target = str(write.get("target", "short_term"))
            salience = max(0.0, min(1.0, float(write.get("salience", 0.35) or 0.35)))
            row = {
                "id": f"{'ltm' if target == 'long_term' else 'stm'}_{now}_{abs(hash(str(write))) % 100000}",
                "kind": str(write.get("kind", "episode")),
                "content": str(write.get("content", "")).strip(),
                "salience": salience,
                "keywords": _string_list(write.get("keywords"), limit=8),
                "reason": str(write.get("reason", "")),
                "source": "thinking_prompt",
                "created_at": now,
                "last_recalled_at": None,
                "recall_count": 0,
            }
            if not row["content"]:
                continue
            evidence_refs = _string_list(write.get("evidence_refs"), limit=8)
            if not evidence_refs:
                evidence_refs = [f"stm_turn_{now}"]
            row["evidence_refs"] = evidence_refs
            store_target = "long_term" if target == "long_term" or salience >= 0.68 else "short_term"
            intent = intent_from_mapping(
                write,
                target=store_target,
                kind=str(row["kind"]),
                content=str(row["content"]),
                confidence=_bounded_float(write.get("confidence"), default=0.75),
                evidence_refs=evidence_refs,
                source="thinking_writes",
                proposer="thinking_prompt",
                audit_reason=str(write.get("reason", "thinking_memory_write") or "thinking_memory_write"),
                value_proxy=salience,
                surprise_proxy=max(salience, _bounded_float(write.get("surprise_proxy"), default=0.35)),
                identity_relevance=_bounded_float(write.get("identity_relevance"), default=0.0),
            )
            if not self._evaluate_memory_gate(
                state,
                intent,
                turn_index=turn_index,
                now=now,
                store_target=store_target,
                store_id=str(row["id"]),
            ):
                continue
            shareability = _shareability_for_memory_text(
                row["content"],
                row.get("keywords"),
                explicit_secret=explicit_secrecy,
                requested=str(write.get("shareability", "default_social") or "default_social"),
            )
            _stamp_memory_policy(
                row,
                user_id=user_id,
                display_name=display_name,
                shareability=shareability,
                restriction_reason=_restriction_reason_for_shareability(
                    shareability,
                    explicit_secret=explicit_secrecy,
                    existing=str(write.get("restriction_reason", "")).strip(),
                ),
                confidence=_bounded_float(write.get("confidence"), default=0.75),
                source_participant_id=source_participant_id,
                source_audience_participant_ids=source_audience_participant_ids,
                session_id=session_id,
                turn_index=turn_index,
                ingress_evidence_band=ingress_evidence_band,
            )
            if store_target == "long_term":
                state.setdefault("long_term_memory", []).append(row)
            else:
                state.setdefault("short_term_memory", []).append(row)

        new_expectations = thinking.get("new_expectations")
        if isinstance(new_expectations, list):
            pending = state.setdefault("pending_expectations", [])
            if isinstance(pending, list):
                turn_memory_id = f"stm_turn_{now}"
                dynamics = _mapping(memory_dynamics)
                write_candidates = dynamics.get("write_candidates", [])
                has_memory_candidate = isinstance(write_candidates, list) and any(
                    isinstance(candidate, Mapping) and str(candidate.get("content", "")).strip()
                    for candidate in write_candidates
                )
                active_statuses = {"", "pending", "active", "uncertain", "due"}

                def _active_pending_row(row: Any) -> bool:
                    if not isinstance(row, Mapping):
                        return False
                    status = str(row.get("status", "pending") or "pending").strip().lower()
                    return status in active_statuses and not status.startswith("merged_into:")

                active_ids = {
                    str(row.get("id", "") or "").strip()
                    for row in pending
                    if _active_pending_row(row) and str(row.get("id", "") or "").strip()
                }

                def _short_local_expectation_id(raw: str) -> bool:
                    if not raw.startswith("exp_") or len(raw) > 12:
                        return False
                    suffix = raw[4:]
                    return bool(suffix) and suffix.replace("_", "").isdigit()

                def _expectation_signature(row: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
                    refs = tuple(_string_list(row.get("evidence_refs"), limit=8))
                    return (
                        str(row.get("source", "thinking_prompt") or "thinking_prompt"),
                        str(row.get("content", row.get("summary", "")) or "").strip(),
                        str(row.get("verify_on", row.get("verify", "")) or "").strip(),
                        refs,
                    )

                active_signatures = {
                    _expectation_signature(row)
                    for row in pending
                    if isinstance(row, Mapping) and _active_pending_row(row)
                }
                for item in new_expectations:
                    if isinstance(item, Mapping) and str(item.get("content", "")).strip():
                        payload = dict(item)
                        raw_id = str(payload.get("id", "") or "").strip()
                        if raw_id:
                            payload.setdefault("source_expectation_id", raw_id)
                        if not raw_id or raw_id in active_ids or _short_local_expectation_id(raw_id):
                            raw_id = f"exp_{turn_index}_{now}_{uuid.uuid4().hex[:8]}"
                        payload["id"] = raw_id[:120]
                        payload.setdefault("created_at", now)
                        payload.setdefault("created_turn_index", turn_index)
                        payload.setdefault("source", "thinking_prompt")
                        refs = _string_list(payload.get("evidence_refs"), limit=8)
                        if turn_memory_id not in refs:
                            refs.append(turn_memory_id)
                        payload["evidence_refs"] = list(dict.fromkeys(refs))[:8]
                        bind_current_turn = bool(turn_memory_committed or _structured_memory_dynamics_binding(payload))
                        if bind_current_turn:
                            bound = _string_list(payload.get("bound_memory_ids"), limit=8)
                            if turn_memory_id not in bound:
                                bound.append(turn_memory_id)
                            payload["bound_memory_ids"] = list(dict.fromkeys(bound))[:8]
                        if _traceable_memory_dynamics_expectation_candidate(payload) and (
                            has_memory_candidate or _structured_memory_dynamics_binding(payload)
                        ):
                            payload["source"] = "memory_dynamics_adapter"
                            payload["verify_on"] = "memory_dynamics_idle"
                            if bind_current_turn:
                                bound = _string_list(payload.get("bound_memory_ids"), limit=8)
                                if turn_memory_id not in bound:
                                    bound.append(turn_memory_id)
                                payload["bound_memory_ids"] = list(dict.fromkeys(bound))[:8]
                        sig = _expectation_signature(payload)
                        if sig in active_signatures:
                            continue
                        pending.append(payload)
                        active_ids.add(str(payload.get("id", "")))
                        active_signatures.add(sig)

        open_items = thinking.get("open_item_writes")
        if isinstance(open_items, list):
            target = state.setdefault("open_items", [])
            if isinstance(target, list):
                for item in open_items:
                    if isinstance(item, Mapping) and str(item.get("content", "")).strip():
                        payload = dict(item)
                        payload.setdefault("id", f"item_{now}_{len(target)}")
                        payload.setdefault("created_at", now)
                        target.append(payload)

        patch = thinking.get("self_cognition_patch")
        if isinstance(patch, Mapping) and bool(patch.get("apply", False)):
            cognition = state.setdefault("self_cognition", {})
            if isinstance(cognition, dict):
                delta = str(patch.get("summary_delta", "")).strip()
                if delta:
                    old = str(cognition.get("current_self_view", "")).strip()
                    cognition["current_self_view"] = (old + "\n" + delta).strip()
                cognition.setdefault("identity_tensions", [])
                cognition.setdefault("known_limits", [])
                if isinstance(cognition["identity_tensions"], list):
                    cognition["identity_tensions"].extend(_string_list(patch.get("new_identity_tensions"), limit=6))
                if isinstance(cognition["known_limits"], list):
                    cognition["known_limits"].extend(_string_list(patch.get("new_known_limits"), limit=6))
