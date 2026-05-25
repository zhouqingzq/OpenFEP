from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..conversation_loop import run_conversation
from ..fep_prompt import normalize_dialogue_outcome
from ..generator import LLMGenerator, ResponseGenerator, RuleBasedGenerator
from ..observer import DialogueObserver
from ..turn_trace import ConsciousMarkdownWriter
from ..types import TranscriptUtterance
from .mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    OpenRouterJSONClient,
    SHARED_STATE_KEYS,
    SYSTEM_FILE_NAMES,
)
from .m15_episode_ledger import EpisodeLedger

if TYPE_CHECKING:
    from ...agent import SegmentAgent

_logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    user_text: str
    speaker_name: str = ""
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
    followup_replies: list[str] = field(default_factory=list)


def _sanitize_dir_component(raw: str, *, max_len: int = 48) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(raw or ""))
    s = s.strip("_") or "default"
    return s[:max_len]


def _seed_mvp_session_store_if_needed(persona_root: Path, session_root: Path) -> None:
    """Copy persona-root MVP JSON into a per-browser-tab session folder when empty.

    Avoids two Streamlit sessions sharing one MVPStateStore (race on save).
    """
    if session_root.exists():
        try:
            if any(session_root.iterdir()):
                return
        except OSError:
            return
    try:
        session_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    shared_file_names = {SYSTEM_FILE_NAMES[key] for key in SHARED_STATE_KEYS}
    for fname in SYSTEM_FILE_NAMES.values():
        if fname in shared_file_names:
            continue
        src = (
            _latest_temporal_state_seed(persona_root)
            if fname == SYSTEM_FILE_NAMES["temporal_state"]
            else persona_root / fname
        )
        dst = session_root / fname
        if src.is_file() and not dst.exists():
            try:
                shutil.copy2(src, dst)
            except OSError:
                continue


def _latest_temporal_state_seed(persona_root: Path) -> Path:
    fallback = persona_root / SYSTEM_FILE_NAMES["temporal_state"]
    candidates = [fallback]
    sessions_root = persona_root / "sessions"
    if sessions_root.is_dir():
        candidates.extend(sessions_root.glob(f"*/{SYSTEM_FILE_NAMES['temporal_state']}"))
    best_path = fallback
    best_turn_at = _temporal_seed_turn_at(fallback)
    for path in candidates:
        turn_at = _temporal_seed_turn_at(path)
        if turn_at > best_turn_at:
            best_path = path
            best_turn_at = turn_at
    return best_path


def _temporal_seed_turn_at(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1
    try:
        return int(payload.get("last_turn_at", -1))
    except (AttributeError, TypeError, ValueError):
        return -1


def _llm_api_key_available() -> bool:
    import json
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "secrets" / "openrouter.json"
    if not config_path.exists():
        return False
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
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
            evidence_contract = dialogue_context.get("evidence_contract")
            system_prompt = builder.build_system_prompt(
                agent, action, emotional, conflict,
                turn_index=turn_index,
                conversation_history=conversation_history,
                current_turn=current_turn,
                hidden_intent=hidden_intent,
                previous_outcome=previous_outcome,
                efe_margin=efe_margin,
                fep_capsule=fep_capsule if isinstance(fep_capsule, dict) else None,
                evidence_contract=evidence_contract,
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
        enable_conscious_trace: bool = False,
        conscious_root: str | Path | None = None,
        session_id: str = "m56_live",
        use_mvp_runtime: bool = True,
        mvp_root: str | Path | None = None,
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
        self._transcript: list[TranscriptUtterance] = []
        self._session_id = str(session_id or "m56_live")
        self._use_mvp_runtime = bool(use_mvp_runtime)
        self._mvp_root = Path(mvp_root) if mvp_root is not None else (
            Path(__file__).resolve().parents[3] / "artifacts" / "mvp_personas"
        )
        self._mvp_runtime: MVPDialogueRuntime | None = None
        self._background_runner: Any = None
        self._enable_conscious_trace = bool(enable_conscious_trace)
        if conscious_root is None:
            conscious_root = (
                Path(__file__).resolve().parents[3]
                / "artifacts"
                / "conscious"
            )
        self._conscious_writer = (
            ConsciousMarkdownWriter(conscious_root)
            if self._enable_conscious_trace
            else None
        )

        # FEP reasoning bridge: track previous turn state for outcome classification
        self._last_action: str = ""
        self._last_obs_channels: dict[str, float] = {}
        self._last_outcome: str = ""
        self._last_efe_margin: float = 1.0
        self._last_response_diagnostics: dict[str, object] = {}

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
        self._maybe_enable_mvp_llm_runtime()
        return "llm" if self._use_llm else "rule"

    @property
    def mvp_runtime_active(self) -> bool:
        self._maybe_enable_mvp_llm_runtime()
        return self._mvp_runtime is not None

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
        self._ensure_runtime_fields()
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
        self._last_response_diagnostics = {}
        self._transcript = []
        self._mvp_runtime = self._build_mvp_runtime() if self._use_llm and self._use_mvp_runtime else None
        if self._conscious_writer is not None:
            session_dir = self._conscious_writer.session_dir(
                self._resolved_persona_id(), self._session_id
            )
            (session_dir / "conscious_trace.jsonl").unlink(missing_ok=True)
            (session_dir / "Conscious.md").unlink(missing_ok=True)

    def has_agent(self) -> bool:
        return self._agent is not None

    # ── Chat ──────────────────────────────────────────────────────────

    def send(self, request: ChatRequest) -> ChatResponse:
        self._ensure_runtime_fields()
        if self._agent is None:
            raise RuntimeError("No persona loaded. Create or load a persona first.")
        self._maybe_enable_mvp_llm_runtime()

        if request.override_traits:
            for name, value in request.override_traits.items():
                self.set_trait(name, value)
        if request.override_precisions:
            for channel, value in request.override_precisions.items():
                self.set_precision(channel, value)

        if self._mvp_runtime is not None:
            return self._send_mvp(request)
        if self._use_mvp_runtime:
            raise RuntimeError(
                "MVP LLM runtime is not active. Check secrets/openrouter.json and restart Streamlit."
            )

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
            observer=self._observer, partner_uid=0, session_id=self._session_id,
            master_seed=turn_seed,
            initial_prior_observation=self._last_obs_channels or None,
            initial_last_action=self._last_action or None,
            initial_transcript=list(self._transcript),
            persona_id=self._resolved_persona_id(),
            conscious_writer=self._conscious_writer,
            turn_index_offset=self._turn_index,
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
        repaired_text = self._repair_high_conflict_reply(
            request.user_text,
            turn.text,
        )
        safe_text, checks = self._safety.enforce(repaired_text, obs_channels)

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
        self._transcript.append(
            TranscriptUtterance(role="interlocutor", text=request.user_text)
        )
        self._transcript.append(TranscriptUtterance(role="agent", text=safe_text))
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

        response = ChatResponse(
            reply=safe_text, action=turn.action or "ask_question",
            observation=obs_channels, delta_traits=delta_traits,
            delta_big_five=delta_big_five,
            diagnostics=generation_diagnostics,
            safety_checks=checks, turn_index=self._turn_index,
            llm_latency_ms=llm_latency,
        )
        self._last_response_diagnostics = dict(generation_diagnostics)
        return response

    def chat(self, user_text: str) -> str:
        return self.send(ChatRequest(user_text=user_text)).reply

    def bootstrap_mvp_from_materials(self, materials: list[str] | str) -> dict[str, object]:
        self._ensure_runtime_fields()
        if self._mvp_runtime is None:
            if not (self._use_llm and self._use_mvp_runtime):
                return {}
            self._mvp_runtime = self._build_mvp_runtime()
        payload = [materials] if isinstance(materials, str) else list(materials)
        return self._mvp_runtime.initialize_from_materials(payload)

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

    def sync_transcript_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        pending_user_text: str | None = None,
    ) -> None:
        self._ensure_runtime_fields()
        transcript: list[TranscriptUtterance] = []
        filtered = list(messages)
        if (
            pending_user_text
            and filtered
            and filtered[-1].get("role") == "user"
            and filtered[-1].get("text") == pending_user_text
        ):
            filtered = filtered[:-1]
        for message in filtered:
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            role = "interlocutor" if message.get("role") == "user" else "agent"
            transcript.append(TranscriptUtterance(role=role, text=text))
        self._transcript = transcript

    def get_conscious_markdown(self) -> str:
        self._ensure_runtime_fields()
        path = self.conscious_markdown_path()
        if path is None or not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def get_conscious_trace_rows(self, *, limit: int = 20) -> list[dict[str, object]]:
        self._ensure_runtime_fields()
        path = self.conscious_trace_path()
        if path is None or not path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows[-limit:]

    def conscious_markdown_path(self) -> Path | None:
        self._ensure_runtime_fields()
        if self._conscious_writer is None:
            return None
        return (
            self._conscious_writer.session_dir(
                self._resolved_persona_id(), self._session_id
            )
            / "Conscious.md"
        )

    def conscious_trace_path(self) -> Path | None:
        self._ensure_runtime_fields()
        if self._conscious_writer is None:
            return None
        return (
            self._conscious_writer.session_dir(
                self._resolved_persona_id(), self._session_id
            )
            / "conscious_trace.jsonl"
        )

    def latest_response_diagnostics(self) -> dict[str, object]:
        self._ensure_runtime_fields()
        return dict(self._last_response_diagnostics)

    def read_mvp_state_dict(self) -> dict[str, object] | None:
        self._ensure_runtime_fields()
        runtime = self._mvp_runtime
        store = getattr(runtime, "store", None) if runtime is not None else None
        if store is None:
            return None
        try:
            state = store.load()
        except Exception:
            return None
        return copy.deepcopy(state) if isinstance(state, dict) else None

    def read_conversation_log(self, *, limit: int = 24) -> list[dict[str, object]]:
        """Recent MVP conversation_log.jsonl rows (newest last)."""
        self._ensure_runtime_fields()
        runtime = self._mvp_runtime
        store = getattr(runtime, "store", None) if runtime is not None else None
        if store is None:
            return []
        path = store.root / "conversation_log.jsonl"
        if not path.is_file():
            return []
        rows: list[dict[str, object]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        except (OSError, json.JSONDecodeError):
            return []
        cap = max(1, int(limit))
        return rows[-cap:]

    def set_bounded_proactive_opt_in(self, enabled: bool) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.set_initiative_user_opt_in(bool(enabled)))

    def set_implicit_idle_delivery(self, enabled: bool) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.set_initiative_implicit_idle_delivery(bool(enabled)))

    def set_proactive_policy_profile(self, profile: str) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.set_initiative_proactive_policy_profile(str(profile)))

    def read_initiative_status(self) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.read_initiative_status())

    def append_m14_4_implicit_idle_audit(self, event: dict[str, object]) -> None:
        self._ensure_runtime_fields()
        runtime = self._mvp_runtime
        store = getattr(runtime, "store", None) if runtime is not None else None
        if store is None:
            return
        try:
            store.append_log({"event": "m14_4_implicit_idle_audit", **dict(event)})
        except Exception as exc:  # pragma: no cover - UI guardrail
            _logger.warning("failed to append M14.4 implicit idle audit: %s", exc)

    def set_idle_introspection_opt_in(self, enabled: bool) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.set_idle_introspection_opt_in(bool(enabled)))

    def maybe_run_idle_introspection(
        self,
        *,
        user_active: bool = False,
    ) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {
                "ran_introspection": False,
                "skip_reason": "disabled",
                "events": [],
                "idle_result": None,
            }
        turn_index = max(0, int(self._turn_index))
        return dict(
            self._mvp_runtime.maybe_run_idle_introspection(
                turn_index=turn_index,
                user_active=user_active,
            )
        )

    def read_idle_introspection_status(self) -> dict[str, object]:
        state = self.read_mvp_state_dict()
        if not state:
            return {}
        m13 = state.get("m13_drive_state", {})
        if not isinstance(m13, dict):
            return {}
        initiative = m13.get("initiative", {})
        if not isinstance(initiative, dict):
            return {}
        idle = initiative.get("idle_introspection", {})
        return dict(idle) if isinstance(idle, dict) else {}

    def read_background_continuity_status(self) -> dict[str, object]:
        state = self.read_mvp_state_dict()
        if not state:
            return {}
        m13 = state.get("m13_drive_state", {})
        if not isinstance(m13, dict):
            return {}
        initiative = m13.get("initiative", {})
        if not isinstance(initiative, dict):
            return {}
        bg = initiative.get("background_continuity", {})
        return dict(bg) if isinstance(bg, dict) else {}

    def set_background_continuity_opt_in(self, enabled: bool) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        if not enabled:
            self.stop_self_loop_daemon()
        runner_kind = "standalone_daemon" if enabled else "none"
        bg = dict(
            self._mvp_runtime.set_background_continuity_opt_in(
                bool(enabled),
                runner_kind=runner_kind,
            )
        )
        self._stop_background_runner()
        return bg

    def read_self_loop_daemon_status(self) -> dict[str, object]:
        """Runner lock + process liveness for the M14.2 standalone daemon."""
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {"status": "unavailable", "pid": 0, "runner_kind": "none", "running": False}
        from segmentum.dialogue.runtime.m14_1_background_continuity import read_runner_lock, runner_lock_is_alive

        lock = read_runner_lock(self._mvp_runtime.store.root)
        alive = runner_lock_is_alive(lock)
        if lock is None:
            status = "stopped"
        elif alive:
            status = "running"
        else:
            status = "stale"
        return {
            "status": status,
            "running": alive,
            "pid": int(lock.pid if lock else 0),
            "runner_kind": str(lock.runner_kind if lock else ""),
            "host": str(lock.host if lock else ""),
            "started_at": int(lock.started_at if lock else 0),
        }

    def start_self_loop_daemon(self, *, wait_seconds: float = 6.0) -> dict[str, object]:
        """Spawn ``m14_2_self_loop``; PID is read from ``runner.lock`` (not Streamlit's)."""
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {"ok": False, "reason": "mvp_inactive"}
        from segmentum.dialogue.runtime.m14_1_background_continuity import (
            read_runner_lock,
            release_runner_lock,
            runner_lock_is_alive,
        )

        store = self._mvp_runtime.store
        lock = read_runner_lock(store.root)
        if runner_lock_is_alive(lock):
            return {
                "ok": True,
                "reason": "already_running",
                "pid": int(lock.pid if lock else 0),
            }

        if lock is not None and not runner_lock_is_alive(lock):
            release_runner_lock(store.root)

        persona = self._resolved_persona_id()
        session = self._session_id
        project_root = Path(__file__).resolve().parents[3]
        cmd = [
            sys.executable,
            "-m",
            "segmentum.dialogue.runtime.m14_2_self_loop",
            "--persona",
            persona,
            "--session",
            session,
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=os.environ.copy(),
                creationflags=creationflags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            return {"ok": False, "reason": f"spawn_failed:{exc}"[:120]}

        deadline = time.monotonic() + max(1.0, float(wait_seconds))
        while time.monotonic() < deadline:
            lock = read_runner_lock(store.root)
            if runner_lock_is_alive(lock):
                return {
                    "ok": True,
                    "reason": "started",
                    "pid": int(lock.pid if lock else 0),
                    "spawn_pid": int(proc.pid),
                }
            if proc.poll() is not None:
                return {
                    "ok": False,
                    "reason": "process_exited_early",
                    "spawn_pid": int(proc.pid),
                    "exit_code": int(proc.returncode or 0),
                }
            time.sleep(0.2)

        return {
            "ok": False,
            "reason": "lock_not_acquired",
            "spawn_pid": int(proc.pid),
            "detail": "daemon may still be starting; refresh in a few seconds",
        }

    def stop_self_loop_daemon(self) -> dict[str, object]:
        """Stop standalone daemon using ``runner.lock`` PID; clears stale locks."""
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        self._stop_background_runner()
        if self._mvp_runtime is None:
            return {"ok": False, "reason": "mvp_inactive"}
        from segmentum.dialogue.runtime.m14_1_background_continuity import (
            read_runner_lock,
            release_runner_lock,
            runner_lock_is_alive,
        )

        store = self._mvp_runtime.store
        lock = read_runner_lock(store.root)
        if lock is None:
            return {"ok": True, "reason": "not_running", "pid": 0}

        pid = int(lock.pid)
        stopped = False
        if runner_lock_is_alive(lock):
            try:
                if sys.platform == "win32":
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
                stopped = True
            except OSError:
                if sys.platform == "win32":
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            check=False,
                            capture_output=True,
                        )
                        stopped = True
                    except OSError:
                        stopped = False
        release_runner_lock(store.root)
        return {"ok": True, "reason": "stopped" if stopped else "lock_cleared", "pid": pid}

    def update_background_continuity_config(self, **updates: object) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {}
        return dict(self._mvp_runtime.update_background_continuity_config(**updates))

    def read_queued_outreach(self, *, include_other_sessions: bool = False) -> list[dict[str, object]]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return []
        from segmentum.dialogue.runtime.m14_1_background_continuity import load_queued_outreach

        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        roots = [self._mvp_runtime.store.root]
        shared_root = self._mvp_runtime.store.shared_root
        if include_other_sessions and shared_root is not None:
            sessions_dir = shared_root / "sessions"
            if sessions_dir.is_dir():
                roots.extend(path for path in sessions_dir.iterdir() if path.is_dir())
        for root in roots:
            try:
                root_key = str(root.resolve())
            except OSError:
                root_key = str(root)
            if root_key in seen:
                continue
            seen.add(root_key)
            for row in load_queued_outreach(root):
                item = dict(row)
                item.setdefault("session_id", root.name)
                item.setdefault("source_session_id", root.name)
                rows.append(item)
        return rows

    def read_m14_2_environment_events(self, *, limit: int = 20) -> list[dict[str, object]]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return []
        from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore

        store = EnvironmentEventStore(
            self._mvp_runtime.store.root,
            persona_id=self._resolved_persona_id(),
            session_id=self._session_id,
        )
        return [dict(row) for row in store.query_events(limit=limit)]

    def read_m14_2_scheduled_intents(self) -> list[dict[str, object]]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return []
        from segmentum.dialogue.runtime.m14_2_scheduled_intents import ScheduledIntentStore

        store = ScheduledIntentStore(
            self._mvp_runtime.store.root,
            persona_id=self._resolved_persona_id(),
            session_id=self._session_id,
        )
        return [dict(row) for row in store.list_intents()]

    def read_m14_2_observability_summary(self) -> dict[str, object]:
        """Aggregate M14.2 daemon, environment bus, idle introspection, and budgets for the UI."""
        from datetime import datetime

        now = int(time.time())
        today = datetime.fromtimestamp(now).date().isoformat()
        daemon = self.read_self_loop_daemon_status()
        idle = self.read_idle_introspection_status()
        bg = self.read_background_continuity_status()

        clock_wake_acked_today = 0
        last_clock_wake_at = 0
        user_message_pending = 0
        ui_audit_pending = 0
        for row in self.read_m14_2_environment_events(limit=400):
            event_type = str(row.get("event_type", "") or "")
            status = str(row.get("status", "") or "")
            at = int(row.get("at", 0) or 0)
            event_day = datetime.fromtimestamp(at).date().isoformat() if at > 0 else ""
            if event_type == "ClockWakeEvent" and status == "acked" and event_day == today:
                clock_wake_acked_today += 1
                last_clock_wake_at = max(last_clock_wake_at, at)
            if event_type == "UserMessageCommittedEvent" and status == "pending":
                user_message_pending += 1
            if event_type in {"UIPingEvent", "OutboxDeliverySurfaceAvailableEvent"} and status == "pending":
                ui_audit_pending += 1

        active_intent_statuses = {"pending", "preparing", "prepared", "awaiting_delivery"}
        scheduled_intents_active = 0
        for row in self.read_m14_2_scheduled_intents():
            if str(row.get("status", "") or "") in active_intent_statuses:
                scheduled_intents_active += 1

        queued_outreach = sum(
            1
            for row in self.read_queued_outreach()
            if str(row.get("status", "") or "") == "pending"
        )

        health_ticks_today = 0
        last_health_at = 0
        last_proactive_target: dict[str, object] = {}
        last_proactive_suppression: dict[str, object] = {}
        last_drive_band_summary: dict[str, object] = {}
        last_idle_cognitive_tick: dict[str, object] = {}
        last_idle_plan_mismatch: dict[str, object] = {}
        mvp_runtime = getattr(self, "_mvp_runtime", None)
        m15_delta_f_trail: list[dict[str, object]] = []
        if mvp_runtime is not None:
            try:
                ledger = EpisodeLedger(mvp_runtime.store.root)
                m15_delta_f_trail = [
                    {
                        "turn_index": episode.turn_index,
                        "phase": episode.phase,
                        "action": episode.action,
                        "delta_fe_proxy": episode.delta_fe_proxy,
                        "outcome_summary": episode.outcome_summary,
                    }
                    for episode in reversed(ledger.recent(5))
                ]
            except Exception:
                m15_delta_f_trail = []
        for row in self.read_conversation_log(limit=300):
            if str(row.get("event", "")) != "m14_2_audit":
                if str(row.get("type", "")) == "IdleCognitiveTickEvent":
                    target = row.get("selected_target", {})
                    bands = row.get("bands", {})
                    last_idle_cognitive_tick = {
                        "at": int(row.get("at", 0) or 0),
                        "idle_seconds": row.get("idle_seconds", 0),
                        "retrieved_ids": row.get("retrieved_ids", []),
                        "bounded_retrieve_ids": row.get("bounded_retrieve_ids", []),
                        "recall_top_k": row.get("recall_top_k", 0),
                        "memory_efe_should_outreach": bool(row.get("memory_efe_should_outreach", False)),
                        "memory_efe_selected_policy": str(row.get("memory_efe_selected_policy", "") or ""),
                        "reject_reason": str(row.get("reject_reason", "") or ""),
                        "selected_target": dict(target) if isinstance(target, dict) else {},
                        "bands": dict(bands) if isinstance(bands, dict) else {},
                    }
                    if isinstance(target, dict) and target:
                        last_proactive_target = {
                            "trigger": str(target.get("trigger", "") or ""),
                            "traceable_expectation_id": str(target.get("traceable_expectation_id", "") or ""),
                            "source_kind": str(target.get("source_kind", "") or ""),
                            "selection_reason_codes": target.get("selection_reason_codes", []),
                            "evidence_refs": target.get("evidence_refs", []),
                        }
                    elif str(row.get("reject_reason", "") or ""):
                        last_proactive_suppression = {
                            "reason": str(row.get("reject_reason", "") or ""),
                            "reason_code": str(row.get("reject_reason", "") or ""),
                            "reason_stage": "idle_cognitive_tick",
                        }
                    if isinstance(bands, dict):
                        last_drive_band_summary = {
                            "behavioral_pull_band": bands.get("behavior_band", ""),
                            "boredom_band": bands.get("boredom_band", ""),
                            "affective_reward_band": bands.get("reward_band", ""),
                            "relation_path_precision_band": bands.get("relation_band", ""),
                        }
                if str(row.get("type", "")) == "IdlePlanStructuralMismatchEvent":
                    last_idle_plan_mismatch = {
                        "at": int(row.get("at", 0) or 0),
                        "mismatch_reason_code": str(row.get("mismatch_reason_code", "") or ""),
                        "plan_recommendation_reason": str(row.get("plan_recommendation_reason", "") or ""),
                        "selection_reason_codes": row.get("selection_reason_codes", []),
                    }
                if str(row.get("type", "")) == "M13ProactiveProposalEvent":
                    last_proactive_target = {
                        "trigger": str(row.get("trigger", "") or ""),
                        "traceable_expectation_id": str(row.get("traceable_expectation_id", "") or ""),
                        "ordinary_language_intent": str(row.get("ordinary_language_intent", "") or "")[:160],
                        "source_kind": str(row.get("source_kind", "") or ""),
                        "selection_reason_codes": row.get("selection_reason_codes", []),
                        "evidence_refs": row.get("trigger_evidence_refs", []),
                    }
                if str(row.get("type", "")) == "M13ProactiveSuppressionEvent":
                    if str(row.get("reason_stage", "") or "") == "pre_proposal":
                        last_proactive_target = {}
                    last_proactive_suppression = {
                        "reason": str(row.get("reason", "") or ""),
                        "reason_code": str(row.get("reason_code", row.get("reason", "")) or ""),
                        "reason_stage": str(row.get("reason_stage", "") or ""),
                    }
                if str(row.get("type", "")) == "IdleProactiveDriveRefreshEvent":
                    summary = row.get("drive_band_summary", {})
                    last_drive_band_summary = dict(summary) if isinstance(summary, dict) else {}
                continue
            if str(row.get("type", "")) != "SelfLoopDaemonHealthEvent":
                continue
            at = int(row.get("at", 0) or 0)
            if at > 0 and datetime.fromtimestamp(at).date().isoformat() == today:
                health_ticks_today += 1
                last_health_at = max(last_health_at, at)

        reflection_count = int(idle.get("reflection_count_this_session", 0) or 0)
        reflection_max = int(idle.get("max_per_session", 4) or 4)
        m14_3_traceability_suggestions = 0
        legacy_vague_open_item_proactive = False
        try:
            from segmentum.dialogue.runtime.m14_3_open_item_migration import audit_open_items_for_efe

            state = mvp_runtime.store.load() if mvp_runtime is not None else {}
            m13_state = state.get("m13_drive_state", {}) if isinstance(state, dict) else {}
            initiative = (
                m13_state.get("initiative", {})
                if isinstance(m13_state, dict)
                else {}
            )
            legacy_vague_open_item_proactive = bool(
                initiative.get("legacy_vague_open_item_proactive")
            ) if isinstance(initiative, dict) else False
            m14_3_traceability_suggestions = len(audit_open_items_for_efe(state.get("open_items", []))) if isinstance(state, dict) else 0
        except Exception:
            m14_3_traceability_suggestions = 0
            state = mvp_runtime.store.load() if mvp_runtime is not None else {}
            m13_state = state.get("m13_drive_state", {}) if isinstance(state, dict) else {}

        consolidation = {}
        if isinstance(m13_state, dict):
            consolidation = m13_state.get("m15_consolidation", {}) if isinstance(m13_state.get("m15_consolidation"), dict) else {}
        slow_loop = {
            "last_run_at": int(consolidation.get("last_run_at", 0) or 0),
            "last_run_id": str(consolidation.get("last_run_id", "") or ""),
            "last_ops": dict(consolidation.get("last_ops", {})) if isinstance(consolidation.get("last_ops"), dict) else {},
            "runs_today": 0,
            "budget_per_day": 6,
        }
        runs_by_day = consolidation.get("runs_by_day", {}) if isinstance(consolidation, dict) else {}
        if isinstance(runs_by_day, dict):
            today_key = str(int(time.time()) // 86400)
            slow_loop["runs_today"] = int(runs_by_day.get(today_key, 0) or 0)
        meta_control_raw = m13_state.get("meta_control_intents", {}) if isinstance(m13_state, dict) else {}
        meta_control = {
            "active": [],
            "recent_detections": [],
            "cleanup_active": [],
            "cleanup_recent_detections": [],
        }
        if isinstance(meta_control_raw, dict):
            meta_control["active"] = [
                {
                    "intent_id": str(row.get("intent_id", "") or ""),
                    "intent_kind": str(row.get("intent_kind", "") or ""),
                    "detector": str(row.get("detector", "") or ""),
                    "payload": dict(row.get("payload", {})) if isinstance(row.get("payload"), dict) else {},
                    "expires_at": int(row.get("expires_at", 0) or 0),
                }
                for row in meta_control_raw.get("active", []) or []
                if isinstance(row, dict)
            ][-8:]
            meta_control["recent_detections"] = [
                {
                    "type": str(row.get("type", "") or ""),
                    "action_trigger": str(row.get("action_trigger", "") or ""),
                    "tension_id": str(row.get("tension_id", "") or ""),
                    "reject_reason": str(row.get("reject_reason", "") or ""),
                    "failure_count": row.get("failure_count", ""),
                    "stable_tick_count": row.get("stable_tick_count", ""),
                }
                for row in meta_control_raw.get("recent_detections", []) or []
                if isinstance(row, dict)
            ][-8:]
            meta_control["cleanup_active"] = [
                {
                    "intent_id": str(row.get("intent_id", "") or ""),
                    "intent_kind": str(row.get("intent_kind", "") or ""),
                    "detector": str(row.get("detector", "") or ""),
                    "payload": dict(row.get("payload", {})) if isinstance(row.get("payload"), dict) else {},
                    "expires_at": int(row.get("expires_at", 0) or 0),
                }
                for row in meta_control_raw.get("cleanup_active", []) or []
                if isinstance(row, dict)
            ][-8:]
            meta_control["cleanup_recent_detections"] = [
                {
                    "type": str(row.get("type", "") or ""),
                    "low_traceability_count": row.get("low_traceability_count", ""),
                    "candidate_count": row.get("candidate_count", ""),
                    "emitted_intent_id": str(row.get("emitted_intent_id", "") or ""),
                }
                for row in meta_control_raw.get("cleanup_recent_detections", []) or []
                if isinstance(row, dict)
            ][-8:]
        cleanup = m13_state.get("m15_cleanup", {}) if isinstance(m13_state, dict) else {}
        if not isinstance(cleanup, dict):
            cleanup = {}

        return {
            "daemon": daemon,
            "clock_wake_acked_today": clock_wake_acked_today,
            "last_clock_wake_at": last_clock_wake_at,
            "health_ticks_today": health_ticks_today,
            "last_health_at": last_health_at,
            "user_message_pending": user_message_pending,
            "ui_audit_pending": ui_audit_pending,
            "scheduled_intents_active": scheduled_intents_active,
            "queued_outreach": queued_outreach,
            "reflection_count": reflection_count,
            "reflection_max": reflection_max,
            "last_introspection_at": int(idle.get("last_introspection_at", 0) or 0),
            "last_idle_skip_reason": str(idle.get("last_skip_reason", "") or ""),
            "last_outreach_outcome": str(idle.get("last_outreach_outcome", "") or ""),
            "background_llm_calls_today": int(bg.get("llm_calls_today", 0) or 0),
            "background_llm_budget": int(bg.get("llm_calls_budget_per_day", 80) or 80),
            "background_tokens_today": int(bg.get("tokens_used_today", 0) or 0),
            "background_tokens_budget": int(bg.get("tokens_budget_per_day", 30000) or 30000),
            "last_budget_block_reason": str(bg.get("last_budget_block_reason", "") or ""),
            "m14_3_last_proactive_target": last_proactive_target,
            "m14_3_last_proactive_suppression": last_proactive_suppression,
            "m14_3_last_drive_band_summary": last_drive_band_summary,
            "m13_5_last_idle_cognitive_tick": last_idle_cognitive_tick,
            "m14_6_last_plan_selector_mismatch": last_idle_plan_mismatch,
            "m15_delta_f_trail": m15_delta_f_trail,
            "m15_slow_loop": slow_loop,
            "m15_meta_control": meta_control,
            "m15_cleanup": {
                "last_run_at": int(cleanup.get("last_run_at", 0) or 0),
                "last_source": str(cleanup.get("last_source", "") or ""),
                "last_ops": dict(cleanup.get("last_ops", {})) if isinstance(cleanup.get("last_ops"), dict) else {},
                "cleanup_active_count": int(cleanup.get("cleanup_active_count", 0) or 0),
                "cleanup_consumed_count": int(cleanup.get("cleanup_consumed_count", 0) or 0),
            },
            "m14_3_open_item_traceability_suggestions": m14_3_traceability_suggestions,
            "m14_3_legacy_vague_open_item_proactive": legacy_vague_open_item_proactive,
        }

    def read_mind_debug_bundle(self, *, ui_hints: Mapping[str, object] | None = None) -> str:
        """Plain-text debug bundle for operator copy/paste."""
        from segmentum.dialogue.runtime.mind_debug_bundle import build_mind_debug_bundle_text

        self._ensure_runtime_fields()
        runtime = self._mvp_runtime
        if runtime is None:
            return "# Path B Mind Debug Bundle\n- MVP runtime inactive\n"
        state = runtime.store.load()
        observability = self.read_m14_2_observability_summary()
        hints = dict(ui_hints or {})
        hints.setdefault("queued_outreach", self.read_queued_outreach())
        hints.setdefault(
            "meta_control_apply_env",
            str(os.environ.get("SEGMENTUM_META_CONTROL_APPLY", "") or "").strip() or "0",
        )
        return build_mind_debug_bundle_text(
            session_root=runtime.store.root,
            persona_name=self._resolved_persona_id(),
            session_id=self._session_id,
            state=state if isinstance(state, dict) else {},
            observability=observability,
            ui_hints=hints,
            turn_index=int(getattr(self, "_turn_index", 0) or 0),
        )

    def _start_background_runner(self) -> None:
        """Development-only inline fallback; not the M14.2 overnight acceptance path."""
        from segmentum.dialogue.runtime.m14_1_self_runner import BackgroundSelfRunner

        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return
        self._stop_background_runner()
        store = self._mvp_runtime.store
        bg = self.read_background_continuity_status()
        interval = int(bg.get("tick_interval_seconds", 90) or 90)
        self._background_runner = BackgroundSelfRunner(
            self._mvp_runtime,
            session_root=store.root,
            persona_id=self._resolved_persona_id(),
            session_id=self._session_id,
            runner_kind="inline_dev_fallback",
            tick_interval_seconds=interval,
        )
        self._background_runner.start()

    def _stop_background_runner(self) -> None:
        if self._background_runner is not None:
            self._background_runner.stop()
            self._background_runner = None

    def _append_m14_2_environment_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        source: str,
        correlation_id: str | None = None,
    ) -> None:
        runtime = self._mvp_runtime
        if runtime is None:
            return
        try:
            from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore

            store = EnvironmentEventStore(
                runtime.store.root,
                persona_id=self._resolved_persona_id(),
                session_id=self._session_id,
            )
            event_id = store.append_event(
                event_type,
                payload,
                source=source,
                correlation_id=correlation_id,
            )
            runtime.store.append_log(
                {
                    "event": "m14_2_audit",
                    "type": "EnvironmentEventAppendedEvent",
                    "event_id": event_id,
                    "event_type": event_type,
                    "runner_kind": "streamlit_adapter",
                    "persona_id": self._resolved_persona_id(),
                    "session_id": self._session_id,
                    "correlation_id": correlation_id or "",
                    "engineering_proxy_label": "mvp_local_decoupled_self_loop",
                }
            )
        except Exception as exc:
            _logger.warning("failed to append M14.2 environment event %s: %s", event_type, exc)
            return

    def record_background_streamlit_ping(self) -> None:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return
        self._append_m14_2_environment_event(
            "UIPingEvent",
            {"surface": "streamlit_chat", "turn_index": self._turn_index},
            source="streamlit",
            correlation_id=f"ui-ping:{int(time.time())}",
        )
        self._mvp_runtime.record_streamlit_ping()
        if self._background_runner is not None:
            self._background_runner.record_streamlit_ping()

    def maybe_drain_queued_outreach(self) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {"drained": False, "reason": "disabled"}
        self._append_m14_2_environment_event(
            "OutboxDeliverySurfaceAvailableEvent",
            {"surface": "streamlit_chat", "turn_index": self._turn_index},
            source="streamlit",
            correlation_id=f"delivery-surface:{self._turn_index}:{int(time.time())}",
        )
        return dict(self._mvp_runtime.maybe_drain_queued_outreach(turn_index=self._turn_index))

    def maybe_propose_proactive_turn(
        self,
        *,
        manual_continue: bool = False,
        idle_seconds: float = 0.0,
        user_typing: bool = False,
        implicit_idle_request: bool = False,
        preselected_target: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {"proposal": None, "suppression_reason": "disabled", "events": []}
        return dict(
            self._mvp_runtime.maybe_propose_proactive_turn(
                turn_index=self._turn_index,
                idle_seconds=idle_seconds,
                manual_continue=manual_continue,
                user_typing=user_typing,
                implicit_idle_request=implicit_idle_request,
                preselected_target=preselected_target,
            )
        )

    def run_idle_cognitive_tick(self, *, idle_seconds: float = 0.0) -> dict[str, object]:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None:
            return {"selected_target": None, "reject_reason": "disabled", "events": []}
        return dict(
            self._mvp_runtime.run_idle_cognitive_tick(
                turn_index=self._turn_index,
                idle_seconds=idle_seconds,
            )
        )

    def run_proactive_turn(self, proposal_id: str, *, speaker_name: str = "") -> ChatResponse:
        self._ensure_runtime_fields()
        self._maybe_enable_mvp_llm_runtime()
        if self._mvp_runtime is None or self._agent is None:
            raise RuntimeError("MVP runtime is not initialized")
        pre_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        pp = self._agent.self_model.personality_profile
        pre_big_five = {
            "openness": pp.openness,
            "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion,
            "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        obs_channels = dict(self._last_obs_channels or {})
        start = time.monotonic()
        try:
            result = self._mvp_runtime.run_proactive_turn(
                proposal_id=str(proposal_id),
                turn_index=self._turn_index,
                speaker_name=speaker_name,
            )
            reply = str(result.reply or "")
            followup_replies = list(getattr(result, "followup_replies", []) or [])
            action = str(result.action or "proactive")
            diagnostics = dict(result.diagnostics)
        except Exception as exc:
            reply = ""
            followup_replies = []
            action = "proactive_error"
            diagnostics = {
                "mvp_runtime": True,
                "proactive_turn": True,
                "llm_error": type(exc).__name__,
                "llm_error_detail": str(exc),
            }
        llm_latency = round((time.monotonic() - start) * 1000.0, 3)
        safe_text, checks = self._safety.enforce(reply, obs_channels)
        safe_followups: list[str] = []
        followup_safety_checks: list[Any] = []
        for followup in followup_replies:
            safe_followup, followup_checks = self._safety.enforce(followup, obs_channels)
            if safe_followup.strip():
                safe_followups.append(safe_followup)
            followup_safety_checks.extend(followup_checks)
        diagnostics["safe_followup_replies"] = safe_followups
        diagnostics["proactive_turn"] = True
        diagnostics["not_user_requested_current_turn"] = True
        if safe_text.strip():
            self._transcript.append(TranscriptUtterance(role="agent", text=safe_text))
            for followup in safe_followups:
                self._transcript.append(TranscriptUtterance(role="agent", text=followup))
        self._last_action = action
        self._turn_index += 1
        post_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        post_big_five = {
            "openness": pp.openness,
            "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion,
            "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        delta_traits = {k: round(post_traits[k] - pre_traits.get(k, 0.0), 6) for k in post_traits}
        delta_big_five = {k: round(post_big_five[k] - pre_big_five.get(k, 0.0), 6) for k in post_big_five}
        response = ChatResponse(
            reply=safe_text,
            action=action,
            observation=obs_channels,
            delta_traits=delta_traits,
            delta_big_five=delta_big_five,
            diagnostics=diagnostics,
            safety_checks=[*checks, *followup_safety_checks],
            turn_index=self._turn_index,
            llm_latency_ms=llm_latency,
            followup_replies=safe_followups,
        )
        self._last_response_diagnostics = dict(diagnostics)
        return response

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

    def _send_mvp(self, request: ChatRequest) -> ChatResponse:
        if self._agent is None or self._mvp_runtime is None:
            raise RuntimeError("MVP runtime is not initialized")

        pre_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        pp = self._agent.self_model.personality_profile
        pre_big_five = {
            "openness": pp.openness, "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion, "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        obs_obj = self._observer.observe(
            current_turn=request.user_text,
            conversation_history=list(self._transcript),
            partner_uid=0,
            session_context={},
            session_id=self._session_id,
            turn_index=self._turn_index,
            speaker_uid=0,
            timestamp=None,
        )
        obs_channels = dict(obs_obj.channels)
        start = time.monotonic()
        try:
            result = self._mvp_runtime.run_turn(
                request.user_text,
                turn_index=self._turn_index,
                speaker_name=request.speaker_name,
                bus_messages=[
                    {
                        "type": "ObservationEvent",
                        "channels": obs_channels,
                        "last_action": self._last_action,
                        "last_outcome": self._last_outcome,
                    }
                ],
            )
            reply = result.reply
            followup_replies = list(getattr(result, "followup_replies", []) or [])
            action = result.action
            diagnostics = dict(result.diagnostics)
        except Exception as exc:
            reply = f"[MVP LLM 调用失败：{exc}]"
            followup_replies = []
            action = "llm_error"
            diagnostics = {
                "mvp_runtime": True,
                "llm_error": type(exc).__name__,
                "llm_error_detail": str(exc),
            }
        llm_latency = round((time.monotonic() - start) * 1000.0, 3)
        safe_text, checks = self._safety.enforce(reply, obs_channels)
        safe_followups: list[str] = []
        followup_safety_checks: list[Any] = []
        for followup in followup_replies:
            safe_followup, followup_checks = self._safety.enforce(followup, obs_channels)
            if safe_followup.strip():
                safe_followups.append(safe_followup)
            followup_safety_checks.extend(followup_checks)
        diagnostics["safe_followup_replies"] = safe_followups
        diagnostics["followup_safety_checks_count"] = len(followup_safety_checks)

        self._last_action = action
        self._last_obs_channels = dict(obs_channels)
        self._last_outcome = "neutral"
        self._transcript.append(
            TranscriptUtterance(role="interlocutor", text=request.user_text)
        )
        self._transcript.append(TranscriptUtterance(role="agent", text=safe_text))
        for followup in safe_followups:
            self._transcript.append(TranscriptUtterance(role="agent", text=followup))
        self._dashboard.snapshot(self._agent)
        self._turn_index += 1
        scheduled_outreach_requests: list[dict[str, object]] = []
        thinking_payload = diagnostics.get("thinking") if isinstance(diagnostics, dict) else None
        if isinstance(thinking_payload, dict):
            raw_requests = thinking_payload.get("scheduled_outreach_requests")
            if isinstance(raw_requests, dict):
                raw_iterable = [raw_requests]
            elif isinstance(raw_requests, list):
                raw_iterable = raw_requests
            else:
                raw_iterable = []
            for raw_request in raw_iterable[:3]:
                if isinstance(raw_request, dict):
                    scheduled_outreach_requests.append(dict(raw_request))
        self._append_m14_2_environment_event(
            "UserMessageCommittedEvent",
            {
                "user_text": request.user_text,
                "speaker_name": request.speaker_name,
                "turn_index": self._turn_index,
                "scheduled_outreach_requests": scheduled_outreach_requests,
            },
            source="streamlit",
            correlation_id=(
                f"user-message:{self._turn_index}:"
                f"{hashlib.sha1(request.user_text.encode('utf-8')).hexdigest()[:16]}"
            ),
        )

        post_traits = self._agent.slow_variable_learner.state.traits.to_dict()
        post_big_five = {
            "openness": pp.openness, "conscientiousness": pp.conscientiousness,
            "extraversion": pp.extraversion, "agreeableness": pp.agreeableness,
            "neuroticism": pp.neuroticism,
        }
        delta_traits = {k: round(post_traits[k] - pre_traits.get(k, 0.0), 6) for k in post_traits}
        delta_big_five = {k: round(post_big_five[k] - pre_big_five.get(k, 0.0), 6) for k in post_big_five}
        diagnostics["selected_action"] = action
        diagnostics["llm_generation"] = {
            "mvp_runtime": True,
            "llm_latency_ms": llm_latency,
        }
        response = ChatResponse(
            reply=safe_text,
            action=action,
            observation=obs_channels,
            delta_traits=delta_traits,
            delta_big_five=delta_big_five,
            diagnostics=diagnostics,
            safety_checks=[*checks, *followup_safety_checks],
            turn_index=self._turn_index,
            llm_latency_ms=llm_latency,
            followup_replies=safe_followups,
        )
        self._last_response_diagnostics = dict(diagnostics)
        return response

    def _build_mvp_runtime(self) -> MVPDialogueRuntime:
        persona_id = self._resolved_persona_id()
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in persona_id).strip("_") or "default"
        persona_root = self._mvp_root / safe_id
        sess = _sanitize_dir_component(self._session_id, max_len=48)
        session_root = persona_root / "sessions" / sess
        _seed_mvp_session_store_if_needed(persona_root, session_root)
        return MVPDialogueRuntime(
            store=MVPStateStore(session_root, shared_root=persona_root),
            llm=OpenRouterJSONClient.from_config(),
            persona_name=self._persona_name,
        )

    def _maybe_enable_mvp_llm_runtime(self) -> None:
        self._ensure_runtime_fields()
        if not self._use_mvp_runtime or self._mvp_runtime is not None:
            return
        if not OpenRouterJSONClient.available():
            return
        self._use_llm = True
        if not isinstance(self._generator, LLMGenerator):
            self._generator = LLMGenerator()
        if self._prompt_builder is None:
            from .prompts import PromptBuilder

            self._prompt_builder = PromptBuilder(persona_name=self._persona_name)
        self._mvp_runtime = self._build_mvp_runtime()

    def _resolved_persona_id(self) -> str:
        return self._persona_name.strip() or "default"

    def _ensure_runtime_fields(self) -> None:
        if not hasattr(self, "_transcript"):
            self._transcript = []
        if not hasattr(self, "_session_id"):
            self._session_id = "m56_live"
        if not hasattr(self, "_enable_conscious_trace"):
            self._enable_conscious_trace = False
        if not hasattr(self, "_conscious_writer"):
            self._conscious_writer = None
        if not hasattr(self, "_use_mvp_runtime"):
            self._use_mvp_runtime = True
        if not hasattr(self, "_mvp_root"):
            self._mvp_root = Path(__file__).resolve().parents[3] / "artifacts" / "mvp_personas"
        if not hasattr(self, "_mvp_runtime"):
            self._mvp_runtime = None
        if not hasattr(self, "_last_response_diagnostics"):
            self._last_response_diagnostics = {}
        if not hasattr(self, "_background_runner"):
            self._background_runner = None

    def _repair_high_conflict_reply(self, user_text: str, reply: str) -> str:
        text = user_text.strip()
        threat_markers = ("打死你", "揍你", "杀了你", "弄死你", "气死我", "气死了")
        if not any(marker in text for marker in threat_markers):
            return reply
        recent_text = "\n".join(item["text"] for item in self._transcript[-8:])
        game_context = any(
            marker in recent_text or marker in text
            for marker in ("原神", "星穹", "重返未来", "抽到", "角色", "等级", "60级")
        )
        if game_context:
            return (
                "我刚才接岔了，先认错。你说的 60 是游戏等级，不是年龄，"
                "我不该顺嘴乱扯；别真动手，我们回到原神/星铁/重返未来这条线。"
            )
        return "我刚才可能把话接歪了，先停一下认错。你现在明显不爽，我先把语气放低。"
