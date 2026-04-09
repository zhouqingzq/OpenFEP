from __future__ import annotations

from typing import Any

from .agent import SegmentAgent
from .m4_cognitive_style import CognitiveStyleParameters
from .memory_encoding import SalienceConfig
from .memory_state import AgentStateVector, MemoryAwareAgentMixin


class MemoryAwareSegmentAgent(SegmentAgent, MemoryAwareAgentMixin):
    def __init__(
        self,
        *args,
        memory_cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
        memory_cycle_interval: int = 5,
        memory_salience_config: SalienceConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_memory_awareness(
            memory_store=self.long_term_memory.ensure_memory_store(),
            state_vector=AgentStateVector.from_dict(
                dict(getattr(self.long_term_memory, "agent_state_vector", {}))
            ),
            cognitive_style=memory_cognitive_style or getattr(self.long_term_memory, "memory_cognitive_style", None),
            memory_cycle_interval=memory_cycle_interval or getattr(self.long_term_memory, "memory_cycle_interval", 5),
            salience_config=memory_salience_config,
        )

    def sync_memory_awareness_to_long_term_memory(self) -> None:
        self.long_term_memory.memory_store = self.memory_store
        self.long_term_memory.episodes = self.memory_store.to_legacy_episodes()
        self.long_term_memory.agent_state_vector = self.agent_state_vector.to_dict()
        self.long_term_memory.memory_cognitive_style = self.memory_cognitive_style.to_dict()
        self.long_term_memory.memory_cycle_interval = self.memory_cycle_interval

    def encode_cycle_memory(self, raw_input: dict[str, Any], cycle: int):
        entry = super().encode_cycle_memory(raw_input, cycle)
        self.sync_memory_awareness_to_long_term_memory()
        return entry

    def run_memory_consolidation_if_due(self, cycle: int, *, rng: Any | None = None):
        report = super().run_memory_consolidation_if_due(cycle, rng=rng)
        self.sync_memory_awareness_to_long_term_memory()
        return report

    def reconsolidate_after_recall(
        self,
        entry_id: str,
        *,
        current_mood: str | None = None,
        current_context_tags: list[str] | None = None,
        current_cycle: int | None = None,
        recall_artifact: Any = None,
        conflict_type: Any = None,
    ):
        report = super().reconsolidate_after_recall(
            entry_id,
            current_mood=current_mood,
            current_context_tags=current_context_tags,
            current_cycle=current_cycle,
            recall_artifact=recall_artifact,
            conflict_type=conflict_type,
        )
        self.sync_memory_awareness_to_long_term_memory()
        return report

    def to_dict(self) -> dict[str, object]:
        self.sync_memory_awareness_to_long_term_memory()
        payload = super().to_dict()
        payload["agent_state_vector"] = self.agent_state_vector.to_dict()
        payload["memory_cognitive_style"] = self.memory_cognitive_style.to_dict()
        payload["memory_cycle_interval"] = self.memory_cycle_interval
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict | None,
        rng=None,
        predictive_hyperparameters=None,
        reset_predictive_precisions: bool = False,
    ) -> "MemoryAwareSegmentAgent":
        base_agent = SegmentAgent.from_dict(
            payload,
            rng=rng,
            predictive_hyperparameters=predictive_hyperparameters,
            reset_predictive_precisions=reset_predictive_precisions,
        )
        agent = cls(rng=base_agent.rng)
        agent.__dict__.update(base_agent.__dict__)
        agent.init_memory_awareness(
            memory_store=agent.long_term_memory.ensure_memory_store(),
            state_vector=AgentStateVector.from_dict(
                dict(payload.get("agent_state_vector", {})) if isinstance(payload, dict) else {}
            ),
            cognitive_style=dict(payload.get("memory_cognitive_style", {}))
            if isinstance(payload, dict) and isinstance(payload.get("memory_cognitive_style"), dict)
            else getattr(agent.long_term_memory, "memory_cognitive_style", None),
            memory_cycle_interval=int(payload.get("memory_cycle_interval", 5))
            if isinstance(payload, dict)
            else 5,
        )
        agent.sync_memory_awareness_to_long_term_memory()
        return agent
