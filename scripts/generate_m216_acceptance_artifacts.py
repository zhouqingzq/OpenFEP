from __future__ import annotations

import json
import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_ingestion import NarrativeIngestionService
from segmentum.narrative_types import NarrativeEpisode


ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    service = NarrativeIngestionService()
    agent = SegmentAgent(rng=random.Random(4))
    episodes = [
        NarrativeEpisode(
            episode_id="social:1",
            timestamp=1,
            source="world-alpha",
            raw_text="A trusted ally shared food and stayed nearby.",
            tags=["social_event", "cooperation", "repair"],
            metadata={
                "counterpart_id": "ally_alice",
                "counterpart_name": "Alice",
                "trust_impact": 0.8,
                "attachment_signal": 0.5,
                "repair": True,
                "event_type": "social_contact",
            },
        ),
        NarrativeEpisode(
            episode_id="social:2",
            timestamp=2,
            source="world-alpha",
            raw_text="A rival threatened the shelter entrance.",
            tags=["social_event", "betrayal", "rupture"],
            metadata={
                "counterpart_id": "rival_bob",
                "counterpart_name": "Bob",
                "trust_impact": -0.8,
                "social_threat": 0.9,
                "rupture": True,
                "event_type": "threat",
            },
        ),
        NarrativeEpisode(
            episode_id="social:3",
            timestamp=3,
            source="world-alpha",
            raw_text="The rival offered a cautious apology.",
            tags=["social_event", "repair", "apology"],
            metadata={
                "counterpart_id": "rival_bob",
                "counterpart_name": "Bob",
                "trust_impact": 0.35,
                "social_threat": 0.1,
                "repair": True,
                "event_type": "repair_attempt",
            },
        ),
    ]
    continuity_rows: list[dict[str, object]] = []
    for episode in episodes:
        result = service.ingest(agent=agent, episodes=[episode])[0]
        continuity_rows.append(
            {
                "episode_id": episode.episode_id,
                "counterpart_id": episode.metadata["counterpart_id"],
                "ingestion": result["ingestion"],
                "social_memory": agent.social_memory.to_dict(),
            }
        )
    _write_jsonl(ARTIFACTS_DIR / "m216_social_continuity_trace.jsonl", continuity_rows)

    rupture_agent = SegmentAgent(rng=random.Random(6))
    rupture_agent.social_memory.observe_counterpart(
        other_id="rival_bob",
        tick=1,
        appraisal={
            "trust_impact": -0.9,
            "social_threat": 0.9,
            "attachment_signal": -0.4,
            "uncertainty": 0.3,
        },
        metadata={"counterpart_name": "Bob", "rupture": True},
        tags=["betrayal", "rupture"],
        event_type="threat",
    )
    trust_after_rupture = rupture_agent.social_memory.others["rival_bob"].trust
    threat_after_rupture = rupture_agent.social_memory.others["rival_bob"].threat
    rupture_agent.social_memory.observe_counterpart(
        other_id="rival_bob",
        tick=2,
        appraisal={
            "trust_impact": 0.35,
            "social_threat": 0.1,
            "attachment_signal": 0.2,
            "uncertainty": 0.2,
        },
        metadata={"counterpart_name": "Bob", "repair": True},
        tags=["repair", "apology"],
        event_type="repair_attempt",
    )
    repaired = rupture_agent.social_memory.others["rival_bob"]
    before_agent = SegmentAgent(rng=random.Random(5))
    after_agent = SegmentAgent(rng=random.Random(5))
    after_agent.social_memory.observe_counterpart(
        other_id="ally_alice",
        tick=1,
        appraisal={"trust_impact": 0.9, "attachment_signal": 0.6, "uncertainty": 0.1},
        metadata={"counterpart_name": "Alice", "repair": True},
        tags=["cooperation", "repair"],
        event_type="social_contact",
    )
    after_agent.social_memory.observe_counterpart(
        other_id="rival_bob",
        tick=2,
        appraisal={
            "trust_impact": -0.9,
            "social_threat": 0.9,
            "attachment_signal": -0.3,
            "uncertainty": 0.2,
        },
        metadata={"counterpart_name": "Bob", "rupture": True},
        tags=["betrayal", "rupture"],
        event_type="threat",
    )
    observation = Observation(
        food=0.3,
        danger=0.1,
        novelty=0.3,
        shelter=0.3,
        temperature=0.5,
        social=0.05,
    )
    before_diag = before_agent.decision_cycle(observation)["diagnostics"]
    after_diag = after_agent.decision_cycle(observation)["diagnostics"]
    repair_benchmark = {
        "trust_after_rupture": trust_after_rupture,
        "threat_after_rupture": threat_after_rupture,
        "trust_after_repair": repaired.trust,
        "threat_after_repair": repaired.threat,
        "hide_score_before_social": before_diag.policy_scores["hide"],
        "hide_score_after_rupture": after_diag.policy_scores["hide"],
        "seek_contact_before_social": before_diag.policy_scores["seek_contact"],
        "seek_contact_after_rupture": after_diag.policy_scores["seek_contact"],
        "social_focus": after_diag.social_focus,
        "social_alerts": after_diag.social_alerts,
    }
    _write_json(ARTIFACTS_DIR / "m216_trust_repair_benchmark.json", repair_benchmark)

    restored = SegmentAgent.from_dict(
        json.loads(json.dumps(after_agent.to_dict())),
        rng=random.Random(9),
    )
    restored_diag = restored.decision_cycle(observation)["diagnostics"]

    report = {
        "milestone": "M2.16",
        "status": "PASS",
        "gates": {
            "persistent_others_with_distinct_profiles": len(agent.social_memory.others) >= 2
            and agent.social_memory.others["ally_alice"].trust
            > agent.social_memory.others["rival_bob"].trust,
            "social_expectations_transfer": bool(restored_diag.social_focus)
            and "ally_alice" in restored.social_memory.others,
            "rupture_changes_action_policy": after_diag.policy_scores["hide"]
            > before_diag.policy_scores["hide"],
            "repair_partially_restores_trust": repaired.trust > trust_after_rupture
            and repaired.trust < 0.8,
            "social_memory_survives_restart": restored.social_memory.to_dict()
            == after_agent.social_memory.to_dict(),
        },
        "summary": {
            "other_ids": sorted(agent.social_memory.others),
            "restored_social_focus": restored_diag.social_focus,
            "restored_social_alerts": restored_diag.social_alerts,
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    _write_json(REPORTS_DIR / "m216_acceptance_report.json", report)


if __name__ == "__main__":
    main()
