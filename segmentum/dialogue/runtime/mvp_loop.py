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


class JSONLLMClient(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


@dataclass
class OpenRouterJSONClient:
    model: str = "deepseek/deepseek-v4-flash"
    temperature: float = 0.35
    timeout_seconds: float = 35.0
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    fallback_models: tuple[str, ...] = ("deepseek/deepseek-v4-flash",)

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
        for model in candidate_models:
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
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code == 200:
                data = response.json()
                break
            message = self._error_message(response)
            errors.append(f"{model}: HTTP {response.status_code}: {message}")
            if response.status_code not in {403, 502, 503}:
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
    system_prompt = """你是数字人格系统的“自由能人格分析”模块。
你的任务不是做关键词匹配，而是从材料中推断一个可持续对话人格的初始系统文件。
请把人格理解为：在资源有限、记忆有限、不确定性持续存在的条件下，一个人会如何维持自我认知、关系预期、行动风格和情绪稳定。

只输出 JSON，不要 Markdown。不要编造材料中没有支撑的具体履历。没有证据时写成不确定或留空。
"""
    user_prompt = f"""人格名称: {persona_name or "未命名人格"}

材料:
{_json_text(materials)}

请生成 JSON，字段必须包含:
{{
  "self_cognition": {{
    "summary": "第一人称自我认知摘要",
    "current_self_view": "这个人格如何理解自己",
    "identity_tensions": ["可能的身份冲突或不稳定点"],
    "stable_values": ["稳定价值/驱动"],
    "known_limits": ["已知限制，不知道的地方"]
  }},
  "long_term_memory": [
    {{
      "id": "ltm_...",
      "kind": "identity|background|relationship|preference|value|episode",
      "content": "可被后续检索的记忆内容",
      "salience": 0.0,
      "keywords": ["检索关键词"],
      "source": "materials",
      "created_at": 0,
      "last_recalled_at": null,
      "recall_count": 0
    }}
  ],
  "self_basic_facts": {{
    "name": "{persona_name}",
    "background": ["有材料支撑的人物背景"],
    "relationships": ["有材料支撑的人物关系"],
    "do_not_invent": ["不能编造的身份边界"]
  }},
  "habit_traits": {{
    "big_five": {{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}},
    "conversation_habits": ["说话习惯"],
    "defense_style": ["不确定/冲突时的防御方式"],
    "memory_policy": ["倾向记住什么、遗忘什么、被什么唤起"]
  }},
  "pending_expectations": [],
  "open_items": [],
  "short_term_memory": []
}}
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

    def initialize_from_materials(self, materials: list[str]) -> dict[str, Any]:
        system_prompt, user_prompt = build_free_energy_personality_analysis_prompt(
            materials,
            persona_name=self.persona_name,
        )
        result = self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        state = self.store.load()
        for key in SYSTEM_FILE_DEFAULTS:
            if key in result:
                state[key] = result[key]
        now = _utc_timestamp()
        for memory in state.get("long_term_memory", []):
            if isinstance(memory, dict):
                memory.setdefault("created_at", now)
                memory.setdefault("source", "materials")
                memory.setdefault("recall_count", 0)
        self.store.save(state)
        self.store.append_log({"event": "initialize_from_materials", "at": now, "result": result})
        return state

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
