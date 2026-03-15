from __future__ import annotations

from .agent import SegmentAgent
from .narrative_compiler import NarrativeCompiler
from .narrative_types import NarrativeEpisode


class NarrativeIngestionService:
    def __init__(self, compiler: NarrativeCompiler | None = None) -> None:
        self.compiler = compiler or NarrativeCompiler()

    def ingest(
        self,
        *,
        agent: SegmentAgent,
        episodes: list[NarrativeEpisode],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for episode in episodes:
            embodied = self.compiler.compile_episode(episode)
            result = agent.ingest_narrative_episode(embodied)
            results.append(
                {
                    "episode_id": episode.episode_id,
                    "raw_text": episode.raw_text,
                    "compilation": embodied.to_dict(),
                    "ingestion": result,
                }
            )

        if results and agent.should_sleep():
            sleep_summary = agent.sleep()
            sleep_payload = {
                "sleep_cycle_id": sleep_summary.sleep_cycle_id,
                "rules_extracted": sleep_summary.rules_extracted,
                "threat_updates": sleep_summary.threat_updates,
                "preference_updates": sleep_summary.preference_updates,
                "narrative_prior_updates": dict(
                    agent.self_model.narrative_priors.to_dict()
                ),
            }
            for result in results:
                result["sleep"] = sleep_payload

        return results
