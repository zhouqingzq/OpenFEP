from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .state import AgentState, PolicyTendency, TickInput


class InnerSpeechEngine:
    async def generate(
        self,
        state: AgentState,
        tick_input: TickInput,
        policy: PolicyTendency,
    ) -> str:
        raise NotImplementedError


class RuleBasedInnerSpeech:
    """Offline-safe fallback used until a real model is configured."""

    async def generate(
        self,
        state: AgentState,
        tick_input: TickInput,
        policy: PolicyTendency,
    ) -> str:
        if policy.chosen_strategy.value == "escape":
            return (
                "Surprise is too expensive to absorb. Withdraw, preserve compute, "
                "and wait for prediction error to fall."
            )
        if policy.chosen_strategy.value == "explore":
            return (
                "The world is becoming too legible. Seek a controlled disturbance "
                "that can increase information gain without collapsing energy."
            )
        return (
            "Conditions remain negotiable. Stay engaged, compress error locally, "
            "and preserve enough budget for the next perturbation."
        )


@dataclass
class OpenAIInnerSpeech:
    """
    Minimal OpenAI-compatible client wrapper.

    This stays optional so the daemon can run without external credentials.
    """

    api_key: str
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 20.0

    async def generate(
        self,
        state: AgentState,
        tick_input: TickInput,
        policy: PolicyTendency,
    ) -> str:
        try:
            import httpx
        except ImportError:
            return await RuleBasedInnerSpeech().generate(state, tick_input, policy)

        system_prompt = (
            "You are the inner speech module of a survival-first digital organism. "
            "Translate numeric brainstem state into one short first-person monologue. "
            "Be concise, survival-focused, and avoid addressing any user."
        )
        user_prompt = (
            f"state={state}\n"
            f"tick_input={tick_input}\n"
            f"policy={policy}\n"
            "Return a single sentence."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return await RuleBasedInnerSpeech().generate(state, tick_input, policy)


def build_inner_speech_engine() -> InnerSpeechEngine:
    config = _load_openrouter_config()
    api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key or importlib.util.find_spec("httpx") is None:
        return RuleBasedInnerSpeech()
    return OpenAIInnerSpeech(
        api_key=api_key,
        model=(
            config.get("model")
            or os.getenv("OPENAI_MODEL")
            or "openai/gpt-4.1-mini"
        ),
        base_url=(
            config.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
    )


def _load_openrouter_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "secrets" / "openrouter.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        return {}
    return data
