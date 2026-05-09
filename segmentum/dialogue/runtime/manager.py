from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..lifecycle import ImplantationConfig, implant_personality
from ..maturity import big_five_to_slow_traits
from ..observer import DialogueObserver
from ..world import DialogueWorld

if TYPE_CHECKING:
    from ...agent import SegmentAgent


SUPPORTED_MATERIAL_EXTENSIONS = {".txt", ".md"}


def safe_persona_name(name: str, *, fallback: str = "persona") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name or ""))
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def unique_persona_name(candidate: str, existing: set[str]) -> str:
    base = safe_persona_name(candidate)
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def read_material_file_bytes(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_MATERIAL_EXTENSIONS:
        raise ValueError("Material file must be a .txt or .md file")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Material file must be UTF-8 encoded") from exc
    if not text.strip():
        raise ValueError("Material file is empty")
    return text


class PersonaManager:
    def __init__(self, storage_dir: str | Path = "personas") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ── Three Creation Paths ──────────────────────────────────────────────

    def create_from_chat_data(
        self,
        user_dataset: dict,
        *,
        config: ImplantationConfig | None = None,
        seed: int = 42,
    ) -> "SegmentAgent":
        from ...agent import SegmentAgent

        observer = DialogueObserver()
        world = DialogueWorld(user_dataset, observer, seed=seed)
        agent = SegmentAgent()
        implant_personality(agent, world, config or ImplantationConfig())
        return agent

    def create_from_description(
        self,
        description: str,
        *,
        use_narrative_ingestion: bool = False,
    ) -> "SegmentAgent":
        from ...agent import SegmentAgent
        from ...personality_analyzer import PersonalityAnalyzer

        agent = SegmentAgent()
        analyzer = PersonalityAnalyzer()
        result = analyzer.analyze([description])
        bf = result.big_five
        self._apply_big_five(agent, bf)
        self._ingest_description_value_memory(agent, description)

        if use_narrative_ingestion:
            from ...narrative_ingestion import NarrativeIngestionService
            from ...narrative_types import NarrativeEpisode

            episode = NarrativeEpisode(
                episode_id=f"desc_{int(time.time())}",
                timestamp=int(time.time()),
                source="description",
                raw_text=description,
            )
            svc = NarrativeIngestionService()
            svc.ingest(agent=agent, episodes=[episode])

        return agent

    @staticmethod
    def _ingest_description_value_memory(agent: "SegmentAgent", description: str) -> None:
        from ...memory import suppress_legacy_memory_warnings
        from ...value_memory import ValueMemoryExtractor

        extractor = ValueMemoryExtractor()
        evaluations = extractor.extract(description, source_id=f"desc_{int(time.time())}")
        if not evaluations:
            return
        memory = agent.long_term_memory
        timestamp = int(time.time())
        staged = list(memory.episodes)
        for index, evaluation in enumerate(evaluations):
            payload = {
                "episode_id": f"desc-value-{timestamp}-{index}",
                "timestamp": timestamp + index,
                "cycle": timestamp + index,
                "action": "apply_value_memory",
                "predicted_outcome": evaluation.candidate_kind,
                "value_label": evaluation.candidate_kind,
                "value_score": evaluation.value_memory_score,
                "future_path_utility": evaluation.future_path_utility,
                "reuse_gain": evaluation.score_breakdown.future_reuse_gain,
                "error_avoidance_gain": evaluation.score_breakdown.error_avoidance_gain,
                "maintenance_cost": evaluation.score_breakdown.maintenance_cost,
                "prediction_error": 0.0,
                "risk": 0.0,
                "total_surprise": max(0.05, abs(evaluation.future_path_utility)),
                "weighted_surprise": max(0.05, abs(evaluation.future_path_utility)),
                "content": evaluation.candidate.summary,
                "memory_class": "semantic",
                "source_type": "inference",
                "semantic_tags": [
                    "value_memory",
                    evaluation.candidate_kind,
                    *evaluation.candidate.trigger_conditions[:2],
                ],
                "context_tags": list(evaluation.candidate.applicability_bounds[:3]),
                "compression_metadata": {
                    "value_memory": evaluation.to_dict(),
                    "description_source": True,
                },
                "support_count": max(1, len(evaluation.candidate.evidence_refs)),
                "support": max(1, len(evaluation.candidate.evidence_refs)),
            }
            staged.append(payload)
        with suppress_legacy_memory_warnings():
            memory._commit_episode_projection(staged)
        agent.sync_memory_awareness_to_long_term_memory()

    def create_from_questionnaire(
        self,
        big_five: dict[str, float],
    ) -> "SegmentAgent":
        from ...agent import SegmentAgent

        agent = SegmentAgent()
        self._apply_big_five(agent, big_five)
        return agent

    def create_from_material_analysis(
        self,
        persona_payload: Mapping[str, Any],
    ) -> "SegmentAgent":
        """Create a persona from an LLM-produced material-analysis payload.

        This intentionally does not call PersonalityAnalyzer or value-memory
        keyword extraction.  The file-material path is LLM-only; durable MVP
        self files are written by MVPDialogueRuntime from the same payload.
        """
        from ...agent import SegmentAgent

        habit_traits = persona_payload.get("habit_traits", {})
        big_five = {}
        if isinstance(habit_traits, Mapping):
            maybe_big_five = habit_traits.get("big_five", {})
            if isinstance(maybe_big_five, Mapping):
                big_five = dict(maybe_big_five)
        agent = SegmentAgent()
        self._apply_big_five(agent, big_five)
        return agent

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, agent: "SegmentAgent", name: str) -> Path:
        path = self._storage_dir / f"{name}.json"
        state = agent.to_dict()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> "SegmentAgent":
        from ...agent import SegmentAgent

        path = self._storage_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Persona '{name}' not found at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SegmentAgent.from_dict(payload)

    def list_personas(self) -> list[str]:
        return sorted(
            p.stem for p in self._storage_dir.glob("*.json") if p.is_file()
        )

    def delete(self, name: str) -> bool:
        path = self._storage_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _apply_big_five(agent: "SegmentAgent", big_five: dict[str, float]) -> None:
        from ...slow_learning import SlowTraitState

        mapped = big_five_to_slow_traits(
            openness=float(big_five.get("openness", 0.5)),
            conscientiousness=float(big_five.get("conscientiousness", 0.5)),
            extraversion=float(big_five.get("extraversion", 0.5)),
            agreeableness=float(big_five.get("agreeableness", 0.5)),
            neuroticism=float(big_five.get("neuroticism", 0.5)),
        )
        # Set slow traits
        agent.slow_variable_learner.state.traits = SlowTraitState(**mapped)

        # Sync personality profile
        pp = agent.self_model.personality_profile
        pp.openness = float(big_five.get("openness", 0.5))
        pp.conscientiousness = float(big_five.get("conscientiousness", 0.5))
        pp.extraversion = float(big_five.get("extraversion", 0.5))
        pp.agreeableness = float(big_five.get("agreeableness", 0.5))
        pp.neuroticism = float(big_five.get("neuroticism", 0.5))

        # Sync precision manipulator
        agent.precision_manipulator.update_personality(
            neuroticism=pp.neuroticism,
            openness=pp.openness,
            extraversion=pp.extraversion,
            agreeableness=pp.agreeableness,
            conscientiousness=pp.conscientiousness,
        )

        # Sync defense strategy selector
        agent.defense_strategy_selector.update_personality(
            neuroticism=pp.neuroticism,
            openness=pp.openness,
            extraversion=pp.extraversion,
            conscientiousness=pp.conscientiousness,
            agreeableness=pp.agreeableness,
        )
