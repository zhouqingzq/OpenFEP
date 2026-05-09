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
        "defense_style": [],
        "memory_policy": [],
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
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
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
        data: dict[str, Any] | None = None
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

                message = self._error_message(response)
                errors.append(f"{model}: HTTP {response.status_code}: {message}")
                if response.status_code in retryable_statuses and attempt + 1 < attempts:
                    continue
                break
            if data is not None:
                break
            if errors and "HTTP 403" not in errors[-1] and not any(
                f"HTTP {status}" in errors[-1] for status in retryable_statuses
            ) and "request attempt" not in errors[-1] and "JSON response parse" not in errors[-1]:
                break
        if data is None:
            raise RuntimeError("OpenRouter chat completion failed; " + " | ".join(errors))
        content = data["choices"][0]["message"]["content"]
        return _extract_json_object(content)

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
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的意识主循环。
你必须基于系统文件和消息总线做判断，不能用关键词表替代判断。
你的输出只给机器读，用 JSON 表示：现在要处理什么、要检索什么记忆、哪些预期需要验证、是否可能要修改自我认知。
不要生成最终回复。
"""
    user_prompt = f"""turn_index: {turn_index}
外部输入:
{user_text}

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
) -> tuple[str, str]:
    system_prompt = """你是数字人格系统的思考与回复模块。
你必须根据人格特征、自我认知、基本事实、短期记忆、长期记忆和意识主循环计划来生成回复。
这不是关键词匹配。你需要模拟这个人格在当下的判断：是否短思考、长思考、是否要更新记忆、设定什么预期、是否需要改写自我认知。
只输出 JSON，不要 Markdown。
"""
    user_prompt = f"""turn_index: {turn_index}
用户刚说:
{user_text}

系统文件:
{_json_text(state)}

意识主循环输出:
{_json_text(conscious_plan)}

检索到的相关记忆:
{_json_text(retrieved_memories)}

请输出 JSON:
{{
  "thought_type": "none|short|long",
  "inner_thought": "给系统看的内心判断，不直接展示给用户",
  "reply": "直接发给用户的自然对话回复",
  "reply_action": "answer|ask_question|empathize|clarify|disagree|deflect|self_disclose",
  "new_expectations": [
    {{"id": "exp_...", "content": "我预期接下来会看到/验证什么", "verify_on": "next_user_turn|later", "confidence": 0.0}}
  ],
  "memory_writes": [
    {{"target": "short_term|long_term", "kind": "episode|fact|preference|relationship|identity|open_item", "content": "要写入的内容", "salience": 0.0, "keywords": ["检索词"], "reason": "为什么值得记"}}
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
  "memory_dynamics_note": "哪些记忆被唤起、为什么、是否强化或衰减"
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


@dataclass
class MVPTurnResult:
    reply: str
    action: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
    ) -> MVPTurnResult:
        now = _utc_timestamp()
        bus = list(bus_messages or [])
        bus.append({"type": "UserUtteranceEvent", "turn_index": turn_index, "text": user_text, "at": now})
        state = self.store.load()

        conscious_system, conscious_user = build_conscious_loop_prompt(
            state=state,
            user_text=user_text,
            bus_messages=bus,
            turn_index=turn_index,
        )
        conscious = self.llm.complete_json(system_prompt=conscious_system, user_prompt=conscious_user)
        keywords = _string_list(conscious.get("memory_search_keywords"), limit=16)
        retrieved = retrieve_memories(state, keywords or [user_text])
        self._mark_recalled(state, retrieved, now)

        thinking_system, thinking_user = build_thinking_prompt(
            state=state,
            user_text=user_text,
            conscious_plan=conscious,
            retrieved_memories=retrieved,
            turn_index=turn_index,
        )
        thinking = self.llm.complete_json(system_prompt=thinking_system, user_prompt=thinking_user)

        self._apply_expectation_results(state, conscious.get("expectation_results"))
        self._apply_thinking_writes(state, thinking, user_text=user_text, now=now)
        self.store.save(state)

        reply = str(thinking.get("reply") or "").strip()
        if not reply:
            reply = "我需要想一下这个。"
        action = str(thinking.get("reply_action") or "answer")
        diagnostics = {
            "mvp_runtime": True,
            "bus_messages": bus,
            "conscious_plan": conscious,
            "retrieved_memories": retrieved,
            "thinking": thinking,
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
                "diagnostics": diagnostics,
            }
        )
        return MVPTurnResult(reply=reply, action=action, diagnostics=diagnostics)

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
