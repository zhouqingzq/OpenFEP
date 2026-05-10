"""Minimal LLM-driven persona loop for the dialogue MVP.

This module is intentionally narrower than the research runtime.  It keeps the
MVP user-facing contract explicit: durable self files, LLM-based conscious
planning, memory retrieval, LLM-based thinking/reply generation, and guarded
state writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Protocol


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
}

SYSTEM_FILE_NAMES: dict[str, str] = {
    key: f"{key}.json" for key in SYSTEM_FILE_DEFAULTS
}

PERSONA_ANALYSIS_KEYS = (
    "persona_name",
    "source_role_evidence",
    *SYSTEM_FILE_DEFAULTS.keys(),
)


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


def _bounded_float(value: Any, *, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


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
    for key, default in SYSTEM_FILE_DEFAULTS.items():
        value = payload.get(key, default)
        if isinstance(default, list):
            persona[key] = value if isinstance(value, list) else []
        elif isinstance(default, dict):
            persona[key] = dict(value) if isinstance(value, Mapping) else dict(default)
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

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_files()

    def ensure_files(self) -> None:
        for key, default in SYSTEM_FILE_DEFAULTS.items():
            path = self.path_for(key)
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def path_for(self, key: str) -> Path:
        if key not in SYSTEM_FILE_NAMES:
            raise KeyError(f"unknown MVP state file: {key}")
        return self.root / SYSTEM_FILE_NAMES[key]

    def load(self) -> dict[str, Any]:
        self.ensure_files()
        return {
            key: _safe_json_load(self.path_for(key), default)
            for key, default in SYSTEM_FILE_DEFAULTS.items()
        }

    def save(self, state: Mapping[str, Any]) -> None:
        self.ensure_files()
        for key, default in SYSTEM_FILE_DEFAULTS.items():
            value = state.get(key, default)
            self.path_for(key).write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def append_log(self, row: Mapping[str, Any]) -> None:
        path = self.root / "conversation_log.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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
    bus_messages: list[Mapping[str, Any]],
    turn_index: int,
    temporal_input: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的意识主循环。
你必须基于系统文件和消息总线做判断，不能用关键词表替代判断。
你的输出只给机器读，用 JSON 表示：现在要处理什么、要检索什么记忆、哪些预期需要验证、是否可能要修改自我认知。
你还要专门判断当前时间语境：工程层只提供当前时间、上一轮时间和上一轮摘要这些事实；是否发生时间跳变、用户是否在纠正时间语境、连续性风险如何，必须由你在 temporal_assessment 中判断。
不要生成最终回复。
"""
    user_prompt = f"""turn_index: {turn_index}
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


def build_thinking_prompt(
    *,
    state: Mapping[str, Any],
    user_text: str,
    conscious_plan: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
    turn_index: int,
    response_style_prior: Mapping[str, Any] | None = None,
    memory_guidance: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的思考与回复模块。
你必须根据人格特征、自我认知、基本事实、短期记忆、长期记忆、表达习惯先验和意识主循环计划来生成回复。
这不是关键词匹配，也不是表演式内心独白。
你要先给出最近一次 LLM 思考结果，再生成回复。
意识主循环的 temporal_assessment 是本轮时间语境判断的来源；不要自己重新猜时间差。如果 temporal_assessment 判断用户在纠正时间语境或时间已经明显推进，回复要自然承认这一点，避免强行沿用上一轮的旧时间语境。
表达习惯先验是逐渐形成的说话倾向，不是工程硬性字数限制；例如“避免冗长”应影响轻重和展开程度，但不要机械裁字数。
记忆动力学指导是程序层压缩出的倾向和证据边界，不是角色台词；不要把它表演成“我被奖励/惩罚了”。如果指导要求修正、澄清或降低断言强度，要自然体现在回复策略里。
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


def _memory_pools(state: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    pools: list[tuple[str, Mapping[str, Any]]] = []
    for key in ("short_term_memory", "long_term_memory", "open_items", "pending_expectations"):
        value = state.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    pools.append((key, item))
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
) -> dict[str, Any]:
    kind = str(item.get("kind", source)).strip() or source
    salience = _bounded_float(item.get("salience"), default=0.35)
    confidence = _bounded_float(item.get("confidence"), default=max(0.2, min(0.9, 0.45 + salience * 0.35)))
    status = _memory_status(item)
    use_as_fact = source in {"short_term_memory", "long_term_memory"} and kind not in {
        "expectation_result",
        "open_item",
    } and status not in {"violated", "uncertain"}
    return {
        "id": str(item.get("id", "")).strip(),
        "kind": kind,
        "content": str(item.get("content", "")).strip()[:600],
        "source": source,
        "confidence": round(confidence, 6),
        "salience": round(salience, 6),
        "why_relevant": reasons[:5],
        "conflict_note": conflict_note,
        "use_as_fact": bool(use_as_fact),
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
    semantic_terms = _unique_strings(
        query.get("semantic_terms"),
        query.get("relationship_terms"),
        query.get("status_terms"),
        limit=24,
    )
    source_priority = _string_list(
        query.get("source_priority")
        or ["pending_expectations", "short_term_memory", "long_term_memory", "open_items"],
        limit=8,
    )
    priority_rank = {source: len(source_priority) - idx for idx, source in enumerate(source_priority)}
    status_terms = {item.casefold() for item in _string_list(query.get("status_terms"), limit=8)}

    scored: list[tuple[float, dict[str, Any]]] = []
    for source, item in _memory_pools(state):
        reasons: list[str] = []
        score = 0.0
        item_id = str(item.get("id", "")).strip()
        kind = str(item.get("kind", source)).strip()
        text = json.dumps(item, ensure_ascii=False).casefold()
        status = _memory_status(item).casefold()

        if item_id and item_id.casefold() in expectation_ids:
            score += 6.0
            reasons.append(f"expectation_id:{item_id}")
        kind_match = bool(kind and kind.casefold() in memory_kinds)
        if kind_match and kind.casefold() in {"expectation", "expectation_result", "open_item"}:
            score += 2.0
            reasons.append(f"kind:{kind}")
        if status and status in status_terms:
            score += 1.2
            reasons.append(f"status:{status}")
        for term in semantic_terms:
            lowered = term.casefold()
            if not lowered:
                continue
            if lowered in text:
                score += 1.5
                reasons.append(f"term:{term}")
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
        conflict_note = ""
        if "violated" in status_terms and status in {"violated", "uncertain"}:
            conflict_note = "expectation verification is not settled as a fact"
        scored.append(
            (
                score,
                _evidence_card(source, item, score=score, reasons=reasons, conflict_note=conflict_note),
            )
        )

    if not scored and semantic_terms:
        fallback = retrieve_memories(state, semantic_terms, limit=limit)
        return [
            _evidence_card(
                str(item.get("_source_file", "memory")),
                item,
                score=float(item.get("_retrieval_score", 0.0) or 0.0),
                reasons=["fallback_keyword_match"],
            )
            for item in fallback
        ]
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]


def build_memory_dynamics_guidance(
    state: Mapping[str, Any],
    user_text: str,
    conscious_plan: Mapping[str, Any],
    bus_messages: list[Mapping[str, Any]],
    temporal_input: Mapping[str, Any],
    now: int,
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

    base_salience = 0.35 + salience_delta
    should_encode = bool(expectation_results or reasons or len(str(user_text).strip()) >= 24)
    semantic_terms = _unique_strings(
        conscious_plan.get("memory_search_keywords"),
        _rough_terms(user_text),
        conscious_plan.get("current_task"),
        conscious_plan.get("next_task"),
        limit=16,
    )
    expectation_ids = _unique_strings(
        [item.get("id") for item in expectation_results],
        conscious_plan.get("pending_expectations_to_verify"),
        limit=16,
    )
    memory_kinds = ["expectation_result", "episode", "preference", "relationship", "fact", "open_item"]
    if violated or uncertain:
        memory_kinds = ["expectation_result", "open_item", "episode", "fact", "preference"]

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
                "reason": ";".join(reasons[:4]) or "dialogue_turn_candidate",
                "evidence": "user_text",
                "created_at": now,
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


def _apply_habit_updates(state: dict[str, Any], thinking: Mapping[str, Any]) -> list[dict[str, Any]]:
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
    return applied


def _update_temporal_state(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    user_text: str,
    reply: str,
    temporal_input: Mapping[str, Any],
) -> None:
    state["temporal_state"] = {
        "last_turn_at": now,
        "last_turn_index": turn_index,
        "last_user_text": user_text,
        "last_reply": reply,
        "last_elapsed_seconds": temporal_input.get("elapsed_since_previous_turn_seconds"),
        "last_time_gap_label": temporal_input.get("time_gap_label", "first_turn"),
    }


def _prompt_safe_state(state: Mapping[str, Any]) -> dict[str, Any]:
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
    validation = {
        "original_length": len(original),
        "final_length": len(cleaned),
        "conversation_mode": mode,
        "max_chars": max_chars,
        "max_sentences": max_sentences,
        "changed": bool(actions),
        "actions": actions,
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
    relationship_terms = ("陪伴", "想要你", "需要你", "你这样的", "朋友", "在我身边", "认真记住")
    if any(term in str(user_text or "") for term in relationship_terms):
        return True, "relationship_signal"
    if mode == "serious_thinking":
        return False, "serious_without_observer_trigger"
    return False, "low_risk_short_reply"


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
        payload = normalize_persona_payload(persona_payload, fallback_name=self.persona_name)
        for key in SYSTEM_FILE_DEFAULTS:
            state[key] = payload[key]
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
        bus_messages: list[Mapping[str, Any]] | None = None,
        now: int | None = None,
    ) -> MVPTurnResult:
        now = _utc_timestamp() if now is None else int(now)
        state = self.store.load()
        temporal_input = _temporal_input_from_state(state, now=now)
        bus = list(bus_messages or [])
        bus.append({"type": "TemporalContextEvent", "turn_index": turn_index, **temporal_input})
        bus.append({"type": "UserUtteranceEvent", "turn_index": turn_index, "text": user_text, "at": now})

        conscious_system, conscious_user = build_conscious_loop_prompt(
            state=state,
            user_text=user_text,
            bus_messages=bus,
            turn_index=turn_index,
            temporal_input=temporal_input,
        )
        conscious = self.llm.complete_json(system_prompt=conscious_system, user_prompt=conscious_user)
        memory_dynamics = build_memory_dynamics_guidance(
            state,
            user_text,
            conscious,
            bus,
            temporal_input,
            now,
        )
        retrieved = retrieve_memories_for_guidance(
            state,
            _mapping(memory_dynamics.get("recall_query")),
        )
        memory_dynamics["recall"] = {
            **_mapping(memory_dynamics.get("recall")),
            "retrieved": len(retrieved),
            "ids": [str(item.get("id", "")) for item in retrieved if item.get("id")],
        }
        self._mark_recalled(state, retrieved, now)
        response_style_prior = _response_style_prior(state, retrieved)

        thinking_system, thinking_user = build_thinking_prompt(
            state=_prompt_safe_state(state),
            user_text=user_text,
            conscious_plan=conscious,
            retrieved_memories=retrieved,
            turn_index=turn_index,
            response_style_prior=response_style_prior,
            memory_guidance={
                "memory_value": memory_dynamics.get("memory_value", {}),
                "recall": memory_dynamics.get("recall", {}),
                "control_guidance": memory_dynamics.get("control_guidance", {}),
                "write_candidates": memory_dynamics.get("write_candidates", []),
                "expectation_impact": memory_dynamics.get("expectation_impact", {}),
            },
        )
        thinking = self.llm.complete_json(system_prompt=thinking_system, user_prompt=thinking_user)

        self._apply_expectation_results(state, conscious.get("expectation_results"))
        self._apply_thinking_writes(state, thinking, user_text=user_text, now=now)
        memory_candidates_applied = self._apply_memory_write_candidates(
            state,
            memory_dynamics.get("write_candidates"),
            now=now,
        )
        habit_updates_applied = _apply_habit_updates(state, thinking)

        raw_reply = str(thinking.get("reply") or "").strip()
        if not raw_reply:
            raw_reply = "我需要想一下这个。"
        control_guidance = _mapping(memory_dynamics.get("control_guidance"))
        reply_contract = _mapping(control_guidance.get("reply_contract"))
        reply, reply_validation = validate_visible_reply(raw_reply, reply_contract)
        action = str(thinking.get("reply_action") or "answer")
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
        )
        pacing_feedback_habits_applied = self._apply_pacing_feedback_habit(
            state,
            user_text=user_text,
        )
        visible_reply = "\n".join([reply, *followup_replies])
        _update_temporal_state(
            state,
            now=now,
            turn_index=turn_index,
            user_text=user_text,
            reply=visible_reply,
            temporal_input=temporal_input,
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
            "bus_messages": bus,
            "conscious_plan": conscious,
            "temporal_input": temporal_input,
            "temporal_assessment": dict(temporal_assessment),
            "memory_dynamics": memory_dynamics,
            "memory_candidates_applied": memory_candidates_applied,
            "post_reply_observer": post_reply_observer,
            "post_reply_observer_skipped_reason": post_reply_observer_skipped_reason,
            "post_reply_memory_updates_applied": post_reply_memory_updates_applied,
            "pacing_feedback_habits_applied": pacing_feedback_habits_applied,
            "followup_replies": followup_replies,
            "conversation_mode": control_guidance.get("conversation_mode"),
            "reply_contract": reply_contract,
            "reply_validation": reply_validation,
            "raw_reply": raw_reply,
            "pacing_guidance": control_guidance,
            "response_style_prior": response_style_prior,
            "habit_updates_applied": habit_updates_applied,
            "retrieved_memories": retrieved,
            "thinking": thinking,
            "llm_thinking_result": llm_thinking_result,
            "state_root": str(self.store.root),
            "system_files": {key: str(self.store.path_for(key)) for key in SYSTEM_FILE_DEFAULTS},
        }
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
                target.append(row)
                applied.append(row)
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
            state.setdefault("short_term_memory", []).append(row)
            applied.append(row)
        return applied

    def _apply_pacing_feedback_habit(
        self,
        state: dict[str, Any],
        *,
        user_text: str,
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
        return [row]

    def _apply_expectation_results(self, state: dict[str, Any], results: Any) -> None:
        if not isinstance(results, list):
            return
        pending = state.get("pending_expectations", [])
        if not isinstance(pending, list):
            pending = []
        resolved_ids = {
            str(item.get("id"))
            for item in results
            if isinstance(item, Mapping) and str(item.get("status", "")) in {"confirmed", "violated"}
        }
        if resolved_ids:
            state["pending_expectations"] = [
                item for item in pending
                if not isinstance(item, Mapping) or str(item.get("id")) not in resolved_ids
            ]
        history = state.setdefault("short_term_memory", [])
        if isinstance(history, list):
            for item in results:
                if isinstance(item, Mapping):
                    history.append(
                        {
                            "id": f"stm_expectation_{_utc_timestamp()}_{len(history)}",
                            "kind": "expectation_result",
                            "content": json.dumps(dict(item), ensure_ascii=False),
                            "salience": min(1.0, float(item.get("self_update_pressure", 0.2) or 0.2)),
                            "keywords": ["预期验证", str(item.get("status", ""))],
                            "source": "conscious_loop",
                            "created_at": _utc_timestamp(),
                            "recall_count": 0,
                        }
                    )

    def _apply_thinking_writes(self, state: dict[str, Any], thinking: Mapping[str, Any], *, user_text: str, now: int) -> None:
        short = state.setdefault("short_term_memory", [])
        if isinstance(short, list):
            short.append(
                {
                    "id": f"stm_turn_{now}",
                    "kind": "dialogue_turn",
                    "content": f"用户说：{user_text}\n我回复：{thinking.get('reply', '')}",
                    "salience": 0.35,
                    "keywords": _string_list(thinking.get("memory_dynamics_note"), limit=4),
                    "source": "dialogue",
                    "created_at": now,
                    "recall_count": 0,
                }
            )
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
