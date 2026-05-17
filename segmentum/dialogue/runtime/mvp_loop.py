"""Minimal LLM-driven persona loop for the dialogue MVP.

This module is intentionally narrower than the research runtime.  It keeps the
MVP user-facing contract explicit: durable self files, LLM-based conscious
planning, memory retrieval, LLM-based thinking/reply generation, and guarded
state writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from segmentum.user_model import (
    M11RuntimeConfig,
    M11RuntimeState,
    SourceReliabilityLedger,
    SocialSharingCandidate,
    UserModel,
    UserPredictionLedger,
    abstract_memory_content,
    boundary_strength_from_constraints,
    candidate_from_memory,
    decide_social_sharing,
    detect_explicit_secrecy,
    memory_shareability,
    run_m11_turn,
    sharing_feedback_negative,
    update_regret_bias,
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
from segmentum.dialogue.runtime.m13_initiative import (
    PROACTIVE_SURROGATE_USER_TEXT,
    evaluate_proactive_initiative,
    mark_proactive_turn_consumed,
    merge_initiative_into_m13_state,
    normalize_initiative_state,
    proposal_from_initiative_state,
    proactive_visible_text_is_safe,
    set_initiative_user_opt_in,
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
    "temporal_state": {
        "last_turn_at": None,
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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    fallback_models: tuple[str, ...] = ("deepseek/deepseek-v4-flash",)
    request_retries: int = 1

    @classmethod
    def from_config(cls) -> "OpenRouterJSONClient":
        config_path = Path(__file__).resolve().parents[3] / "secrets" / "openrouter.json"
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
        )

    @classmethod
    def available(cls) -> bool:
        return bool(cls.from_config().api_key)

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
{_json_text(state)}

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
  "reasoning_notes": "给系统看的简短判断"
}}
"""
    return system_prompt, user_prompt


def build_m11_extractor_prompt(
    *,
    snapshot: Mapping[str, Any],
    speaker_name: str,
) -> tuple[str, str]:
    system_prompt = """You are the M11 user-model extractor. Return strict JSON only.

You may classify only bounded enum fields and short summaries. Do not output
floats or numeric scores. Do not invent prediction_id or hypothesis_id values
that are not present in the bounded snapshot. New proposal ids are allowed only
when all source_hypothesis_ids and source_judgment_ids reference snapshot ids.
Keep user claims separate from truth: a high-value claim is useful evidence for
calibration, not verified fact.
"""
    user_prompt = f"""Current interlocutor display name: {speaker_name}

Bounded snapshot:
{_json_text(dict(snapshot))}

Return JSON exactly with the M11 extractor schema fields.
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
    {{"id": "exp_...", "content": "我预期接下来会看到/验证什么", "verify_on": "next_user_turn|later", "confidence": 0.0}}
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
  "habit_updates": [
    {{"content": "从用户反馈或反复证据中学到的表达习惯", "evidence": "支持这个习惯的用户原话或记忆", "confidence": 0.0}}
  ],
  "memory_dynamics_note": "哪些记忆被唤起、为什么、是否强化或衰减"
}}
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
            payload = dict(item)
            payload["_source_file"] = source
            payload["_retrieval_score"] = round(score, 3)
            scored.append((score, payload))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]


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

_SERIOUS_MARKERS = (
    "代码",
    "实现",
    "计划",
    "架构",
    "技术",
    "工程",
    "评估",
    "复盘",
    "测试",
    "接口",
    "schema",
    "api",
    "pytest",
    "debug",
    "bug",
    "修复",
    "原因",
    "分析",
    "设计",
    "方案",
    "修改",
    "提交",
)

_CASUAL_MARKERS = (
    "晚上好",
    "早上好",
    "吃",
    "撑",
    "陪",
    "陪伴",
    "聊天",
    "母亲节",
    "家里",
    "孩子",
    "开心",
    "难过",
    "想你",
    "睡觉",
    "晚安",
    "聊会",
    "聊一会",
    "唉",
    "单挑",
    "看你",
    "好了",
)


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text or "").casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _learned_prefers_short_turns(state: Mapping[str, Any]) -> bool:
    habits = _mapping(state.get("habit_traits"))
    learned = habits.get("learned_conversation_habits", []) or []
    combined = " ".join(_habit_text(item) for item in learned)
    return _has_any_marker(combined, _BREVITY_FEEDBACK_MARKERS)


def _compact_text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _punctuation_count(text: str) -> int:
    return len(re.findall(r"[。！？!?；;，,、.]", str(text or "")))


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


def _pacing_guidance(
    state: Mapping[str, Any],
    user_text: str,
    temporal_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(user_text or "")
    serious = _has_any_marker(text, _SERIOUS_MARKERS)
    casual = _has_any_marker(text, _CASUAL_MARKERS)
    brevity_feedback = _has_any_marker(text, _BREVITY_FEEDBACK_MARKERS)
    prefers_short = _learned_prefers_short_turns(state) or brevity_feedback
    compact_len = _compact_text_len(text)
    punctuation = _punctuation_count(text)
    temporal = _mapping(temporal_input)
    short_gap = str(temporal.get("time_gap_label", "")) in {"immediate", "short_gap"}
    short_casual_shape = compact_len <= 18 and punctuation <= 3
    medium_casual_shape = compact_len <= 34 and punctuation <= 4 and short_gap
    if serious and not brevity_feedback:
        contract = _reply_contract("serious_thinking", prefers_short=prefers_short)
        return {
            "conversation_mode": "serious_thinking",
            "reply_pacing": "serious_thinking",
            "max_response_moves": contract["max_response_moves"],
            "question_policy": "only_if_needed",
            "roleplay_density": "light",
            "leave_space_for_user": False,
            "followup_policy": "only_for_error_or_missed_emotion",
            "reply_contract": contract,
        }
    if casual or prefers_short or short_casual_shape or medium_casual_shape:
        contract = _reply_contract("casual_fast", prefers_short=prefers_short)
        return {
            "conversation_mode": "casual_fast",
            "reply_pacing": "casual_fast",
            "max_response_moves": contract["max_response_moves"],
            "question_policy": contract["question_policy"],
            "roleplay_density": "light",
            "leave_space_for_user": True,
            "followup_policy": "allowed_once_if_high_confidence",
            "reply_contract": contract,
        }
    contract = _reply_contract("balanced", prefers_short=prefers_short)
    return {
        "conversation_mode": "balanced",
        "reply_pacing": "balanced",
        "max_response_moves": contract["max_response_moves"],
        "question_policy": "optional_one",
        "roleplay_density": "light",
        "leave_space_for_user": True,
        "followup_policy": "allowed_once_if_high_confidence",
        "reply_contract": contract,
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
    }


def retrieve_memories_for_guidance(
    state: Mapping[str, Any],
    recall_query: Mapping[str, Any] | None,
    *,
    limit: int = 8,
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
    pacing = _pacing_guidance(state, user_text, temporal_input)

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

    return {
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
            },
            "reply_contract": {
                **_mapping(pacing.get("reply_contract")),
                "allow_direct_disclosure": allow_direct_disclosure,
                "allow_abstract_sharing": allow_abstract_sharing,
                "explicit_secrecy_detected": explicit_secret,
                "soft_boundary_detected": bool(shareability == "restricted_implicit"),
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
    rows.append(row)
    by_user[clean_user] = rows[-24:]
    return row


def _apply_habit_updates(
    state: dict[str, Any],
    thinking: Mapping[str, Any],
    *,
    user_id: str = "",
    now: int | None = None,
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
) -> None:
    state["temporal_state"] = {
        "last_turn_at": now,
        "last_turn_index": turn_index,
        "last_user_text": user_text,
        "last_reply": reply,
        "last_elapsed_seconds": temporal_input.get("elapsed_since_previous_turn_seconds"),
        "last_time_gap_label": temporal_input.get("time_gap_label", "first_turn"),
        "last_share_trace": dict(share_trace or {}),
    }


def _stamp_memory_policy(
    row: dict[str, Any],
    *,
    user_id: str,
    display_name: str,
    shareability: str,
    restriction_reason: str = "",
    confidence: float = 0.8,
) -> dict[str, Any]:
    row["source_user_id"] = str(user_id or "").strip()
    row["source_display_name"] = str(display_name or "").strip()
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
class MVPDialogueRuntime:
    store: MVPStateStore
    llm: JSONLLMClient
    persona_name: str = ""

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
        bus_messages: list[Mapping[str, Any]] | None = None,
        now: int | None = None,
        proactive_context: Mapping[str, Any] | None = None,
    ) -> MVPTurnResult:
        now = _utc_timestamp() if now is None else int(now)
        state = self.store.load()
        display_name = str(speaker_name or "").strip() or "default_user"
        user_id = _safe_user_id(display_name)
        proactive_turn = isinstance(proactive_context, Mapping) and bool(proactive_context)
        sharing_regret_feedback = self._apply_sharing_regret_feedback(
            state,
            user_text=user_text,
            current_user_id=user_id,
            now=now,
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
                    "surrogate_context": str(user_text or "")[:240],
                    "at": now,
                }
            )
        else:
            bus.append({
                "type": "UserUtteranceEvent",
                "turn_index": turn_index,
                "speaker_name": display_name,
                "user_id": user_id,
                "text": user_text,
                "at": now,
            })
        identity_anchored_action = _identity_anchored_action_sensitive(user_text)
        m12_pre_result: dict[str, Any] | None = None
        turn_key = f"turn_{turn_index + 1:04d}"
        m12_cognitive_bus = CognitiveEventBus()
        if _m12_enabled_for_state(state):
            m12_state = _load_m12_state(state)
            m11_readonly_pre: dict[str, object] = {}

            def _extract_m12_pre(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                system_prompt, user_prompt = build_m12_identity_extractor_prompt(
                    snapshot=snapshot,
                    speaker_name=display_name,
                )
                try:
                    return self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
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
        bus.append({
            "type": "EntityBindingEvent",
            "turn_index": turn_index,
            "binding": entity_binding,
        })

        m13_state = normalize_m13_drive_state(state.get("m13_drive_state"))
        reward_for_settlement = normalize_affective_reward_proxy_state(
            m13_state.get("affective_reward_proxy")
        )
        user_reaction_assessments: dict[str, dict[str, Any]] = {}
        assessable_pending_rows = list_assessable_pending_rows(
            reward_for_settlement,
            turn_index=turn_index,
        )
        if assessable_pending_rows and str(user_text or "").strip():
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
                    assessor_raw = self.llm.complete_json(
                        system_prompt=assessor_system,
                        user_prompt=assessor_user,
                    )
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

        conscious_system, conscious_user = build_conscious_loop_prompt(
            state=state,
            user_text=user_text,
            speaker_name=display_name,
            bus_messages=bus,
            turn_index=turn_index,
            temporal_input=temporal_input,
            entity_binding=entity_binding,
        )
        conscious = self.llm.complete_json(system_prompt=conscious_system, user_prompt=conscious_user)
        memory_dynamics = build_memory_dynamics_guidance(
            state,
            user_text,
            conscious,
            bus,
            temporal_input,
            now,
            user_id=user_id,
            speaker_name=display_name,
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
        if _should_run_query_planner(
            state,
            user_text=user_text,
            recall_query=recall_query,
            entity_binding=entity_binding,
        ):
            try:
                planner_system, planner_user = build_query_planner_prompt(
                    user_text=user_text,
                    speaker_name=display_name,
                    recall_query=recall_query,
                    temporal_input=temporal_input,
                    entity_binding=entity_binding,
                )
                query_plan = _normalize_query_plan(
                    self.llm.complete_json(system_prompt=planner_system, user_prompt=planner_user)
                )
                recall_query = _merge_query_plan_into_recall_query(recall_query, query_plan)
                memory_dynamics["recall_query"] = recall_query
            except Exception as exc:
                query_plan = {"planner_error": type(exc).__name__}
        lexical_candidates = lexical_recall_short_term_candidates(
            state,
            user_text=user_text,
            recall_query=recall_query,
            current_user_id=user_id,
            entity_binding=entity_binding,
        )
        evidence_judgment: dict[str, Any] = {}
        if lexical_candidates:
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
                    self.llm.complete_json(system_prompt=judge_system, user_prompt=judge_user),
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
        retrieved = retrieve_memories_for_guidance(
            state,
            recall_query,
        )
        if lexical_candidates:
            existing_ids = {str(item.get("id", "")) for item in retrieved if item.get("id")}
            retrieved = [
                *[item for item in lexical_candidates if str(item.get("id", "")) not in existing_ids],
                *retrieved,
            ][:8]
        memory_dynamics["recall"] = {
            **_mapping(memory_dynamics.get("recall")),
            "retrieved": len(retrieved),
            "ids": [str(item.get("id", "")) for item in retrieved if item.get("id")],
            "lexical_candidate_ids": [str(item.get("id", "")) for item in lexical_candidates if item.get("id")],
            "query_plan": query_plan,
        }
        self._mark_recalled(state, retrieved, now)
        response_style_prior = _response_style_prior(state, retrieved)
        m11_result_dict: dict[str, Any] = {}
        if _m11_enabled_for_state(state):
            def _extract_m11(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                system_prompt, user_prompt = build_m11_extractor_prompt(
                    snapshot=snapshot,
                    speaker_name=display_name,
                )
                try:
                    return self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
                except Exception:
                    return noop_extraction()

            try:
                m11_state, m11_turn = run_m11_turn(
                    m11_state,
                    user_id=user_id,
                    turn_id=turn_index + 1,
                    current_turn_quotes={"q_current": user_text},
                    last_turn_summaries=[],
                    extractor=_extract_m11,
                    config=M11RuntimeConfig(m11_user_model_enabled=True, persona_kind="ui_chat"),
                    legacy_memory_rows=[
                        *(state.get("short_term_memory", []) if isinstance(state.get("short_term_memory"), list) else []),
                        *(state.get("long_term_memory", []) if isinstance(state.get("long_term_memory"), list) else []),
                    ],
                )
                _save_m11_state(state, user_id=user_id, m11_state=m11_state)
                m11_result_dict = m11_turn.to_dict()
            except (ExtractorValidationError, ValueError, TypeError) as exc:
                m11_result_dict = {
                    "enabled": True,
                    "fallback": "noop_extraction",
                    "error": type(exc).__name__,
                    "error_detail": str(exc),
                    "prompt_safe_evidence_cards": [],
                    "reply_policy_effects": [],
                }
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
        if _m12_1_enabled_for_state(state):
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
                        return self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
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

        relationship_value_context = resolve_relationship_value_context(
            state,
            user_id,
            user_text,
        )
        m12_2_result_dict: dict[str, Any] = {}
        m12_2_enabled = _m12_2_enabled_for_state(state)
        if m12_2_enabled:
            m12_2_state = _load_m12_2_state(state)

            def _extract_m12_2(name: str):
                def _extract(snapshot: Mapping[str, object]) -> Mapping[str, object]:
                    system_prompt, user_prompt = build_m12_2_extractor_prompt(name, snapshot)
                    try:
                        return self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
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

        if not m12_2_enabled:
            _apply_relationship_value_context_to_memory_dynamics(
                memory_dynamics,
                relationship_value_context,
            )

        m13_evaluator = M13DriveEvaluator()
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
        )
        for m13_event in m13_evaluation.events:
            bus.append(m13_event)
        m13_boredom_evaluator = M13BoredomEvaluator()
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
            m12_2_result=m12_2_result_dict if m12_2_enabled else None,
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
        thinking = self.llm.complete_json(system_prompt=thinking_system, user_prompt=thinking_user)

        self._apply_expectation_results(
            state,
            conscious.get("expectation_results"),
            user_id=user_id,
            display_name=display_name,
            entity_binding=entity_binding,
        )
        self._apply_thinking_writes(
            state,
            thinking,
            user_text=user_text,
            now=now,
            user_id=user_id,
            display_name=display_name,
            explicit_secrecy=bool(_mapping(_mapping(memory_dynamics.get("control_guidance")).get("sharing_policy")).get("explicit_secrecy_detected")),
        )
        memory_candidates_applied = self._apply_memory_write_candidates(
            state,
            memory_dynamics.get("write_candidates"),
            now=now,
            user_id=user_id,
            display_name=display_name,
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
        )
        habit_updates_applied = _apply_habit_updates(
            state,
            thinking,
            user_id=user_id,
            now=now,
        )

        raw_reply = str(thinking.get("reply") or "").strip()
        if not raw_reply:
            raw_reply = "我需要想一下这个。"
        control_guidance = _mapping(memory_dynamics.get("control_guidance"))
        reply_contract = _mapping(control_guidance.get("reply_contract"))
        reply_contract["identity_anchored_action"] = identity_anchored_action
        reply_contract["selected_disclosure_action"] = str(thinking.get("disclosure_action", "none") or "none")
        reply, reply_validation = validate_visible_reply(raw_reply, reply_contract)
        action = normalize_recorded_reply_action(
            str(thinking.get("reply_action") or "answer"),
            allowed=set(m13_evaluation.candidate_actions),
        )
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
                observer_result = self.llm.complete_json(
                    system_prompt=observer_system,
                    user_prompt=observer_user,
                )
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
        post_reply_memory_updates_applied = self._apply_post_reply_memory_updates(
            state,
            post_reply_observer.get("memory_updates"),
            now=now,
            user_id=user_id,
            display_name=display_name,
        )
        pacing_feedback_habits_applied = self._apply_pacing_feedback_habit(
            state,
            user_text=user_text,
            user_id=user_id,
            now=now,
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
            selected_action=action,
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
            m13_evaluation.scores_by_action.get(action, {}).get("behavioral_pull", 0.0)
        )
        m13_reward_evaluation = m13_reward_evaluator.evaluate(
            turn_id=turn_key,
            turn_index=turn_index,
            user_id=user_id,
            action=action,
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
            action=action,
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
        visible_reply = "\n".join([reply, *followup_replies])
        sharing_policy = _mapping(control_guidance.get("sharing_policy"))
        _update_temporal_state(
            state,
            now=now,
            turn_index=turn_index,
            user_text=user_text,
            reply=visible_reply,
            temporal_input=temporal_input,
            share_trace={
                "user_id": user_id,
                "speaker_name": display_name,
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
            },
        )
        self.store.save(state)
        llm_thinking_result = thinking.get("llm_thinking_result")
        if not isinstance(llm_thinking_result, Mapping):
            legacy_inner_thought = str(thinking.get("inner_thought") or "").strip()
            llm_thinking_result = {
                "debug_summary": legacy_inner_thought,
            } if legacy_inner_thought else {}
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
            "reply_contract": reply_contract,
            "reply_validation": reply_validation,
            "raw_reply": raw_reply,
            "pacing_guidance": control_guidance,
            "response_style_prior": response_style_prior,
            "habit_updates_applied": habit_updates_applied,
            "m13_drive_evaluation": prompt_safe_m13_turn_diagnostics(m13_evaluation),
            "m13_boredom_evaluation": prompt_safe_m13_boredom_diagnostics(m13_boredom_evaluation),
            "m13_reward_evaluation": prompt_safe_m13_reward_diagnostics(m13_reward_evaluation),
            "m13_reward_ui_labels": prompt_safe_m13_reward_ui_labels(),
            "m13_drive_state": prompt_safe_m13_state_summary(m13_state, user_id=user_id),
            "retrieved_memories": retrieved,
            "thinking": thinking,
            "llm_thinking_result": llm_thinking_result,
            "state_root": str(self.store.root),
            "system_files": {key: str(self.store.path_for(key)) for key in SYSTEM_FILE_DEFAULTS},
        }
        if proactive_turn:
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
                    "surrogate_context": str(user_text or "")[:240],
                    "diagnostics": diagnostics,
                }
            )
        else:
            self.store.append_log(
                {
                    "event": "turn",
                    "at": now,
                    "turn_index": turn_index,
                    "user_text": user_text,
                    "reply": reply,
                    "followup_replies": followup_replies,
                    "diagnostics": diagnostics,
                }
            )
        return MVPTurnResult(
            reply=reply,
            action=action,
            diagnostics=diagnostics,
            followup_replies=followup_replies,
        )

    def maybe_propose_proactive_turn(
        self,
        *,
        turn_index: int,
        idle_seconds: float = 0.0,
        manual_continue: bool = False,
        user_typing: bool = False,
        implicit_idle_request: bool = False,
    ) -> dict[str, Any]:
        state = self.store.load()
        now = _utc_timestamp()
        state, check = evaluate_proactive_initiative(
            state,
            now=now,
            turn_index=turn_index,
            idle_seconds=idle_seconds,
            manual_continue=manual_continue,
            user_typing=user_typing,
            implicit_idle_request=implicit_idle_request,
        )
        self.store.save(state)
        for event in check.events:
            self.store.append_log({"event": "m13_proactive_audit", **event})
        return {
            "proposal": check.proposal.to_dict() if check.proposal else None,
            "suppression_reason": check.suppression_reason,
            "events": check.events,
            "state_fields_read": check.state_fields_read,
        }

    def run_proactive_turn(
        self,
        *,
        proposal_id: str,
        turn_index: int,
        speaker_name: str = "",
    ) -> MVPTurnResult:
        state = self.store.load()
        now = _utc_timestamp()
        m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state"))
        initiative = normalize_initiative_state(m13_state.get("initiative"))
        proposal = proposal_from_initiative_state(initiative, now=now)
        if proposal is None or str(proposal.proposal_id) != str(proposal_id):
            reason = "proposal_expired" if proposal is None else "proposal_not_found"
            initiative["last_suppression_reason"] = reason
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.store.save(state)
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "M13ProactiveSuppressionEvent",
                    "reason": reason,
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                }
            )
            return MVPTurnResult(
                reply="",
                action="proactive_suppressed",
                diagnostics={"suppression_reason": reason, "proactive_turn": True},
            )

        result = self.run_turn(
            PROACTIVE_SURROGATE_USER_TEXT,
            turn_index=turn_index,
            speaker_name=speaker_name,
            now=now,
            proactive_context=proposal.to_dict(),
        )
        reply = str(result.reply or "").strip()
        if reply and not proactive_visible_text_is_safe(reply):
            result = MVPTurnResult(
                reply="",
                action="proactive_suppressed",
                diagnostics={
                    **result.diagnostics,
                    "suppression_reason": "safety_risk",
                    "proactive_text_blocked": True,
                },
            )
            initiative["last_suppression_reason"] = "safety_risk"
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.store.save(state)
            self.store.append_log(
                {
                    "event": "m13_proactive_audit",
                    "type": "M13ProactiveSuppressionEvent",
                    "reason": "safety_risk",
                    "proposal_id": proposal_id,
                    "turn_index": turn_index,
                }
            )
            return result

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

    def _apply_memory_write_candidates(
        self,
        state: dict[str, Any],
        candidates: Any,
        *,
        now: int,
        user_id: str = "",
        display_name: str = "",
        default_shareability: str = "default_social",
        restriction_reason: str = "",
    ) -> list[dict[str, Any]]:
        if not isinstance(candidates, list):
            return []
        applied: list[dict[str, Any]] = []
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
            row = {
                "id": f"{'ltm' if target == 'long_term' else 'stm'}_candidate_{now}_{len(applied)}",
                "kind": str(candidate.get("kind", "episode")).strip() or "episode",
                "content": content,
                "salience": salience,
                "confidence": confidence,
                "keywords": _string_list(candidate.get("keywords"), limit=8),
                "reason": str(candidate.get("reason", "")),
                "evidence": evidence,
                "source": "memory_dynamics_adapter",
                "created_at": now,
                "last_recalled_at": None,
                "recall_count": 0,
            }
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
            )
            if target == "long_term" or salience >= 0.68:
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
        user_id: str = "",
        display_name: str = "",
        default_shareability: str = "default_social",
    ) -> list[dict[str, Any]]:
        if not isinstance(updates, list):
            return []
        applied: list[dict[str, Any]] = []
        for item in updates:
            if not isinstance(item, Mapping):
                continue
            content = str(item.get("content", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            confidence = _bounded_float(item.get("confidence"), default=0.0)
            kind = str(item.get("kind", "")).strip()
            if not content or not evidence or confidence < 0.60:
                continue
            if kind == "conversation_habit":
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
                "source": "post_reply_observer",
                "created_at": now,
                "recall_count": 0,
            }
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
        now: int | None = None,
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
            )
        return [row]

    def _apply_sharing_regret_feedback(
        self,
        state: dict[str, Any],
        *,
        user_text: str,
        current_user_id: str,
        now: int,
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
                updates.append(
                    {
                        "content": "跨用户社交转述在负反馈后应显著提高成本，优先抽象化或保留。",
                        "evidence": str(user_text).strip()[:240],
                        "confidence": 0.82,
                        "source": "sharing_regret_feedback",
                        "created_at": now,
                    }
                )
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
        if resolved_ids:
            state["pending_expectations"] = [
                item for item in pending
                if not isinstance(item, Mapping) or str(item.get("id")) not in resolved_ids
            ]
        history = state.setdefault("short_term_memory", [])
        if isinstance(history, list):
            for payload in normalized_results:
                history.append(
                    {
                        "id": f"stm_expectation_{_utc_timestamp()}_{len(history)}",
                        "kind": "expectation_result",
                        "content": json.dumps(payload, ensure_ascii=False),
                        "salience": min(1.0, float(payload.get("self_update_pressure", 0.2) or 0.2)),
                        "keywords": ["预期验证", str(payload.get("status", ""))],
                        "source": "conscious_loop",
                        "created_at": _utc_timestamp(),
                        "recall_count": 0,
                        "source_user_id": user_id,
                        "source_display_name": display_name,
                        "shareability": "default_social",
                    }
                )

    def _apply_thinking_writes(
        self,
        state: dict[str, Any],
        thinking: Mapping[str, Any],
        *,
        user_text: str,
        now: int,
        user_id: str = "",
        display_name: str = "",
        explicit_secrecy: bool = False,
    ) -> None:
        short = state.setdefault("short_term_memory", [])
        if isinstance(short, list):
            assistant_reply = str(thinking.get("reply", "")).strip()
            row = {
                "id": f"stm_turn_{now}",
                "kind": "dialogue_turn",
                "content": str(user_text).strip(),
                "user_text": str(user_text).strip(),
                "assistant_reply": assistant_reply,
                "assistant_reply_use_as_fact": False,
                "salience": 0.35,
                "keywords": _string_list(thinking.get("memory_dynamics_note"), limit=4),
                "source": "dialogue",
                "created_at": now,
                "recall_count": 0,
            }
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
            )
            short.append(row)
            state["short_term_memory"] = short[-24:]

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
            )
            if target == "long_term" or salience >= 0.68:
                state.setdefault("long_term_memory", []).append(row)
            else:
                state.setdefault("short_term_memory", []).append(row)

        new_expectations = thinking.get("new_expectations")
        if isinstance(new_expectations, list):
            pending = state.setdefault("pending_expectations", [])
            if isinstance(pending, list):
                for item in new_expectations:
                    if isinstance(item, Mapping) and str(item.get("content", "")).strip():
                        payload = dict(item)
                        payload.setdefault("id", f"exp_{now}_{len(pending)}")
                        payload.setdefault("created_at", now)
                        pending.append(payload)

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
