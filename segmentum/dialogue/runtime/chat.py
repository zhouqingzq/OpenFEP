from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..conversation_loop import run_conversation
from ..fep_prompt import normalize_dialogue_outcome
from ..generator import LLMGenerator, ResponseGenerator, RuleBasedGenerator
from ..observer import DialogueObserver

if TYPE_CHECKING:
    from ...agent import SegmentAgent


@dataclass
class ChatRequest:
    user_text: str
    override_traits: dict[str, float] | None = None
    override_precisions: dict[str, float] | None = None


@dataclass
class ChatResponse:
    reply: str
    action: str
    observation: dict[str, float]
    delta_traits: dict[str, float]
    delta_big_five: dict[str, float]
    diagnostics: dict[str, object]
    safety_checks: list[Any]
    turn_index: int
    llm_latency_ms: float = 0.0


def _llm_api_key_available() -> bool:
    import json
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "secrets" / "openrouter.json"
    if not config_path.exists():
        return False
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(cfg.get("api_key"))


class _PromptInjector:
    """Generator wrapper that injects a fresh PromptBuilder system prompt
    before each generate() call inside run_conversation."""

    def __init__(self, real_gen: ResponseGenerator, chat_iface: "ChatInterface") -> None:
        self._real = real_gen
        self._chat_iface = chat_iface

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return getattr(self._real, "last_diagnostics", {})

    @last_diagnostics.setter
    def last_diagnostics(self, value: dict[str, object]) -> None:
        self._real.last_diagnostics = value

    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history: Any,
        *,
        master_seed: int,
        turn_index: int,
    ) -> str:
        agent = self._chat_iface._agent
        builder = self._chat_iface._prompt_builder
        if isinstance(self._real, LLMGenerator) and builder is not None and agent is not None:
            obs = dialogue_context.get("observation")
            if isinstance(obs, dict):
                emotional = float(obs.get("emotional_tone", 0.5))
                conflict = float(obs.get("conflict_tension", 0.0))
                hidden_intent = float(obs.get("hidden_intent", 0.5))
            else:
                emotional = 0.5
                conflict = 0.0
                hidden_intent = 0.5
            current_turn = str(dialogue_context.get("current_turn", ""))
            # FEP: previous outcome and decision uncertainty from ChatInterface
            previous_outcome = normalize_dialogue_outcome(self._chat_iface._last_outcome)
            efe_margin = float(dialogue_context.get("efe_margin", self._chat_iface._last_efe_margin))
            fep_capsule = dialogue_context.get("fep_prompt_capsule")
            if isinstance(fep_capsule, dict):
                if fep_capsule.get("previous_outcome") in (None, "", "neutral"):
                    fep_capsule["previous_outcome"] = previous_outcome
                fep_capsule = dict(fep_capsule)
            system_prompt = builder.build_system_prompt(
                agent, action, emotional, conflict,
                turn_index=turn_index,
                conversation_history=conversation_history,
                current_turn=current_turn,
                hidden_intent=hidden_intent,
                previous_outcome=previous_outcome,
                efe_margin=efe_margin,
                fep_capsule=fep_capsule if isinstance(fep_capsule, dict) else None,
            )
            user_message = builder.build_user_message(current_turn, conversation_history)
            self._real.system_prompt = system_prompt
            self._real.user_message = user_message
        return self._real.generate(
            action, dialogue_context, personality_state, conversation_history,
            master_seed=master_seed, turn_index=turn_index,
        )


class ChatInterface:
    def __init__(
        self,
        *,
        use_llm: bool | None = None,
        generator: ResponseGenerator | None = None,
        observer: DialogueObserver | None = None,
        persona_name: str = "",
    ) -> None:
        from .dashboard import DashboardCollector
        from .safety import SafetyLayer

        self._agent: SegmentAgent | None = None
        self._observer = observer or DialogueObserver()
        self._safety = SafetyLayer()
        self._dashboard = DashboardCollector()
        self._turn_index: int = 0
        self._baseline_traits: dict[str, float] = {}
        self._baseline_big_five: dict[str, float] = {}
        self._persona_name = persona_name

        # FEP reasoning bridge: track previous turn state for outcome classification
        self._last_action: str = ""
        self._last_obs_channels: dict[str, float] = {}
        self._last_outcome: str = ""
        self._last_efe_margin: float = 1.0

        if use_llm is None:
            use_llm = _llm_api_key_available()
        self._use_llm = use_llm

        if generator is not None:
            self._generator = generator
            self._use_llm = isinstance(generator, LLMGenerator)
        elif self._use_llm:
            self._generator = LLMGenerator()
        else:
            self._generator = RuleBasedGenerator()

        self._prompt_builder: Any = None
        if self._use_llm:
            from .prompts import PromptBuilder
            self._prompt_builder = PromptBuilder(persona_name=persona_name)

    # ── LLM config ────────────────────────────────────────────────────

    @property
    def use_llm(self) -> bool:
        return self._use_llm

    @property
    def generator_type(self) -> str:
        return "llm" if self._use_llm else "rule"

    def set_temperature(self, temperature: float) -> None:
        if isinstance(self._generator, LLMGenerator):
            self._generator.temperature = float(temperature)

    def set_model(self, model: str) -> None:
        if isinstance(self._generator, LLMGenerator):
            self._generator.model = model

    def get_temperature(self) -> float:
        if isinstance(self._generator, LLMGenerator):
            return self._generator.temperature
        return 0.0

    def get_model(self) -> str:
        if isinstance(self._generator, LLMGenerator):
            return self._generator.model
        return "rule-based"

    # ── Agent management ───────────────────────────────────────────────

    @property
    def agent(self) -> "SegmentAgent | None":
        return self._agent

    @property
    def persona_name(self) -> str:
        return self._persona_name

    @persona_name.setter
    def persona_name(self, name: str) -> None:
        self._persona_name = name
        if self._prompt_builder is not None:
            self._prompt_builder.persona_name = name

    def set_agent(self, agent: "SegmentAgent", *, persona_name: str = "") -> None:
        self._agent = agent
        if persona_name:
            self._persona_name = persona_name
            if self._prompt_builder is not None:
                self._prompt_builder.persona_name = persona_name
        self._record_baseline()
        self._turn_index = 0
        self._dashboard = type(self._dashboard)()
        self._last_action = ""
        self._last_obs_channels = {}
        self._last_outcome = "neutral"
        self._last_efe_margin = 1.0

    def has_agent(self) -> bool:
        return self._agent is not None

    # ── Chat ──────────────────────────────────────────────────────────

    def send(self, request: ChatRequest) -> ChatResponse:
        if self._agent is None:
            raise RuntimeError("No persona loaded. Create or load a persona first.")

        if request.override_traits:
            for name, value in request.override_traits.items():
                self.set_trait(name, value)
        if request.override_precisions:
            for channel, value in request.override_precisions.items():
                self.set_precision(channel, value)

        pre_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        pp = self._agent.self_model.personality_profile
        pre_big_five = {
            "openness": pp.openness, "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion, "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }

        turn_seed = (42 + self._turn_index * 7919) % (2**31)
        turns = run_conversation(
            self._agent, [request.user_text],
            generator=_PromptInjector(self._generator, self),
            observer=self._observer, partner_uid=0, session_id="m56_live",
            master_seed=turn_seed,
            initial_prior_observation=self._last_obs_channels or None,
            initial_last_action=self._last_action or None,
        )
        turn = turns[0] if turns else None
        if turn is None:
            raise RuntimeError("Conversation loop returned no turns")

        post_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        post_big_five = {
            "openness": pp.openness, "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion, "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        delta_traits = {k: round(post_traits[k] - pre_traits.get(k, 0.0), 6) for k in post_traits}
        delta_big_five = {k: round(post_big_five[k] - pre_big_five.get(k, 0.0), 6) for k in post_big_five}

        obs_channels = turn.observation or {}
        safe_text, checks = self._safety.enforce(turn.text, obs_channels)

        llm_latency = 0.0
        if isinstance(self._generator, LLMGenerator):
            diag = self._generator.last_diagnostics
            llm_latency = float(diag.get("llm_latency_ms", 0.0))

        # FEP bridge: classify previous outcome for next turn
        if self._last_action and self._last_obs_channels and obs_channels:
            from ..outcome import classify_dialogue_outcome
            try:
                self._last_outcome = normalize_dialogue_outcome(classify_dialogue_outcome(
                    self._last_action,
                    obs_channels,
                    {},
                    previous_observation=self._last_obs_channels,
                ))
            except Exception:
                self._last_outcome = "neutral"
        else:
            self._last_outcome = "neutral"
        # Store for next classification
        self._last_action = str(turn.action or "")
        self._last_obs_channels = dict(obs_channels)
        fep_capsule = {}
        if isinstance(turn.generation_diagnostics, dict):
            maybe_capsule = turn.generation_diagnostics.get("fep_prompt_capsule")
            if isinstance(maybe_capsule, dict):
                fep_capsule = dict(maybe_capsule)
                try:
                    self._last_efe_margin = float(fep_capsule.get("efe_margin", 1.0) or 1.0)
                except (TypeError, ValueError):
                    self._last_efe_margin = 1.0

        generation_diagnostics = dict(turn.generation_diagnostics or {})
        llm_generation = (
            dict(self._generator.last_diagnostics)
            if isinstance(self._generator, LLMGenerator)
            else {}
        )
        generation_diagnostics["fep_prompt_capsule"] = fep_capsule
        generation_diagnostics["llm_generation"] = llm_generation
        generation_diagnostics["selected_action"] = turn.action or "ask_question"

        self._dashboard.snapshot(self._agent)
        self._turn_index += 1

        return ChatResponse(
            reply=safe_text, action=turn.action or "ask_question",
            observation=obs_channels, delta_traits=delta_traits,
            delta_big_five=delta_big_five,
            diagnostics=generation_diagnostics,
            safety_checks=checks, turn_index=self._turn_index,
            llm_latency_ms=llm_latency,
        )

    def chat(self, user_text: str) -> str:
        return self.send(ChatRequest(user_text=user_text)).reply

    # ── Manual overrides ──────────────────────────────────────────────

    def set_trait(self, trait_name: str, value: float) -> None:
        if self._agent is None:
            raise RuntimeError("No persona loaded")
        setattr(self._agent.slow_variable_learner.state.traits, trait_name,
                max(0.05, min(0.95, float(value))))

    def set_precision(self, channel: str, value: float) -> None:
        if self._agent is None:
            raise RuntimeError("No persona loaded")
        self._agent.precision_manipulator.channel_precisions[channel] = max(0.05, min(2.0, float(value)))

    def reset_to_baseline(self) -> None:
        if self._agent is None:
            raise RuntimeError("No persona loaded")
        for k, v in self._baseline_traits.items():
            setattr(self._agent.slow_variable_learner.state.traits, k, v)
        pp = self._agent.self_model.personality_profile
        for k, v in self._baseline_big_five.items():
            if hasattr(pp, k):
                setattr(pp, k, v)

    # ── State access ──────────────────────────────────────────────────

    def get_memory(self, *, limit: int = 20) -> list[dict[str, object]]:
        if self._agent is None:
            return []
        store = getattr(self._agent, "memory_store", None)
        if store is None:
            return []
        entries = store.episodic_entries()[-limit:]
        return [{"id": getattr(e, "entry_id", ""),
                 "tags": list(getattr(e, "tags", [])),
                 "salience": getattr(e, "salience", 0.0)} for e in entries]

    def get_full_state(self) -> dict[str, object]:
        return self._agent.to_dict() if self._agent else {}

    def trigger_sleep(self) -> dict[str, object]:
        if self._agent is None:
            raise RuntimeError("No persona loaded")
        summary = self._agent.sleep()
        if hasattr(summary, "to_dict"):
            return summary.to_dict()
        return summary if isinstance(summary, dict) else {"sleep_result": str(summary)}

    def get_dashboard(self) -> Any:
        return self._dashboard

    # ── Internal ──────────────────────────────────────────────────────

    def _record_baseline(self) -> None:
        if self._agent is None:
            return
        self._baseline_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        pp = self._agent.self_model.personality_profile
        self._baseline_big_five = {
            "openness": pp.openness, "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion, "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
