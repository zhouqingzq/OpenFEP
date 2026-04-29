from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..lifecycle import ImplantationConfig, implant_personality
from ..maturity import big_five_to_slow_traits
from ..observer import DialogueObserver
from ..world import DialogueWorld

if TYPE_CHECKING:
    from ...agent import SegmentAgent


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

    def create_from_questionnaire(
        self,
        big_five: dict[str, float],
    ) -> "SegmentAgent":
        from ...agent import SegmentAgent

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
