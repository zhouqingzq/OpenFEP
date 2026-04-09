from __future__ import annotations

from copy import deepcopy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters
from .memory_agent import MemoryAwareSegmentAgent
from .memory_encoding import SalienceConfig, encode_memory
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from .memory_retrieval import RetrievalQuery
from .memory_state import AgentStateVector
from .memory_store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

M47_CORPUS_PATH = DATA_DIR / "m47_corpus.json"
M47_RUNTIME_SNAPSHOT_PATH = ARTIFACTS_DIR / "m47_runtime_snapshot.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_vector(
    *,
    active_goals: list[str] | None = None,
    identity_themes: list[str] | None = None,
    threat_level: float = 0.2,
    reward_context_active: bool = False,
    social_context_active: bool = True,
    recent_mood_baseline: str = "reflective",
    last_updated: int = 0,
) -> AgentStateVector:
    return AgentStateVector(
        active_goals=list(active_goals or ["keep promises", "protect mentees"]),
        recent_mood_baseline=recent_mood_baseline,
        recent_dominant_tags=["mentor", "care", "lab"],
        identity_active_themes=list(identity_themes or ["mentor", "promise", "continuity"]),
        threat_level=threat_level,
        reward_context_active=reward_context_active,
        social_context_active=social_context_active,
        last_updated=last_updated,
    )


def _style(**overrides: float) -> CognitiveStyleParameters:
    payload = CognitiveStyleParameters().to_dict()
    payload.update(overrides)
    return CognitiveStyleParameters.from_dict(payload)


def load_m47_corpus(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve() if path is not None else M47_CORPUS_PATH
    return json.loads(resolved.read_text(encoding="utf-8"))


def _payload(raw_input: dict[str, Any], cycle: int, suffix: str = "") -> dict[str, Any]:
    payload = deepcopy(raw_input)
    if suffix:
        payload["content"] = f"{payload.get('content', '').strip()} {suffix}".strip()
    payload.setdefault("created_at", cycle)
    payload.setdefault("cycle", cycle)
    payload.setdefault("event_time", f"cycle-{cycle}")
    payload.setdefault("agents", "mentor_lin")
    return payload


def _clone_with_seed(raw_input: dict[str, Any], seed_group: str, cycle: int, index: int) -> dict[str, Any]:
    payload = _payload(raw_input, cycle, suffix=f"[{seed_group}:{index}]")
    payload["support_count"] = max(1, int(payload.get("support_count", 1)))
    return payload


def _query_from_payload(payload: dict[str, Any], reference_cycle: int) -> RetrievalQuery:
    return RetrievalQuery(
        semantic_tags=list(payload.get("semantic_tags", []))[:3],
        context_tags=list(payload.get("context_tags", []))[:2],
        content_keywords=[str(payload.get("action", "")), str(payload.get("outcome", ""))][:2],
        reference_cycle=reference_cycle,
    )


def _query_to_dict(query: RetrievalQuery) -> dict[str, Any]:
    return {
        "semantic_tags": list(query.semantic_tags),
        "context_tags": list(query.context_tags),
        "content_keywords": list(query.content_keywords),
        "state_vector": list(query.state_vector),
        "reference_cycle": int(query.reference_cycle),
        "target_memory_class": (
            query.target_memory_class.value if query.target_memory_class is not None else None
        ),
        "debug": bool(query.debug),
    }


def _dynamic_state(name: str) -> AgentStateVector:
    if name == "neutral":
        return _state_vector(active_goals=["stabilize repair"], identity_themes=["mentor"], threat_level=0.15)
    if name == "threat":
        return _state_vector(
            active_goals=["stabilize repair"],
            identity_themes=["mentor", "promise"],
            threat_level=0.82,
            recent_mood_baseline="alert",
        )
    return _state_vector(
        active_goals=["repair", "mentor", "promise"],
        identity_themes=["mentor", "promise", "continuity"],
        threat_level=0.35,
        reward_context_active=True,
        social_context_active=True,
        recent_mood_baseline="engaged",
    )


def _encode_probe(raw_input: dict[str, Any], vector: AgentStateVector, style: CognitiveStyleParameters | None = None) -> MemoryEntry:
    return encode_memory(
        deepcopy(raw_input),
        vector.to_context(),
        SalienceConfig(),
        agent_state=vector,
        cognitive_style=style,
    )


def _state_vector_probe(corpus: dict[str, Any]) -> dict[str, Any]:
    agent = MemoryAwareSegmentAgent(memory_cycle_interval=5)
    logs: list[dict[str, Any]] = []
    for cycle, spec in enumerate(corpus["state_vector_events"], start=1):
        entry = agent.encode_cycle_memory(_payload(spec["raw_input"], cycle), cycle)
        logs.append(
            {
                "cycle": cycle,
                "entry_id": entry.id,
                "semantic_tags": list(entry.semantic_tags),
                "salience": round(float(entry.salience), 6),
                "state_vector": agent.agent_state_vector.to_dict(),
            }
        )
    consolidation_report = agent.run_memory_consolidation_if_due(20, rng=random.Random(20))
    return {
        "window_size": getattr(agent.memory_store, "state_window_size", 30),
        "log": logs,
        "snapshot": agent.agent_state_vector.to_dict(),
        "snapshot_for_consolidation": consolidation_report.to_dict() if consolidation_report is not None else {},
    }


def _dynamic_salience_probe(corpus: dict[str, Any]) -> dict[str, Any]:
    base_input = corpus["dynamic_salience_probe"]["raw_input"]
    results: dict[str, Any] = {}
    for name in ("neutral", "threat", "enriched"):
        vector = _dynamic_state(name)
        entry = _encode_probe(base_input, vector)
        audit = dict(entry.compression_metadata or {}).get("dynamic_salience_audit", {})
        results[name] = {
            "salience": round(float(entry.salience), 6),
            "relevance_self": round(float(entry.relevance_self), 6),
            "audit": audit,
        }
    results["threat"]["salience_delta_vs_neutral"] = round(
        float(results["threat"]["salience"]) - float(results["neutral"]["salience"]),
        6,
    )
    results["enriched"]["salience_delta_vs_neutral"] = round(
        float(results["enriched"]["salience"]) - float(results["neutral"]["salience"]),
        6,
    )
    enriched_vector = _dynamic_state("enriched")
    identity_entry = _encode_probe(corpus["dynamic_salience_probe"]["identity_event"], enriched_vector)
    noise_entry = _encode_probe(corpus["dynamic_salience_probe"]["novelty_noise"], enriched_vector)
    results["identity_vs_noise_control"] = {
        "identity_event": {
            "id": identity_entry.id,
            "relevance_self": round(float(identity_entry.relevance_self), 6),
            "salience": round(float(identity_entry.salience), 6),
        },
        "novelty_noise": {
            "id": noise_entry.id,
            "relevance_self": round(float(noise_entry.relevance_self), 6),
            "salience": round(float(noise_entry.salience), 6),
        },
    }
    return results


def _reconsolidation_probe(style: CognitiveStyleParameters) -> tuple[dict[str, Any], float]:
    vector = _state_vector(identity_themes=["mentor", "promise", "continuity"], threat_level=0.30)
    state_context = vector.to_context()
    state_context["cognitive_style"] = style.to_dict()
    store = MemoryStore()
    primary = _encode_probe(
        {
            "content": "My mentor promise summary is abstract and missing some place details.",
            "memory_class": MemoryClass.SEMANTIC.value,
            "semantic_tags": ["mentor", "promise", "continuity", "summary"],
            "context_tags": ["archive", "lab"],
            "valence": 0.12,
            "arousal": 0.24,
            "novelty": 0.38,
            "place": "archive",
            "action": "summarize_commitment",
            "outcome": "identity_summary_saved"
        },
        vector,
        style,
    )
    primary.abstractness = 0.88
    primary.reality_confidence = 0.22
    donor = _encode_probe(
        {
            "content": "Mentor Lin and I renewed the promise in the archive beside the north wall.",
            "semantic_tags": ["mentor", "promise", "archive", "continuity"],
            "context_tags": ["archive", "lab"],
            "valence": 0.18,
            "arousal": 0.20,
            "novelty": 0.14,
            "place": "archive",
            "action": "renew_promise",
            "outcome": "identity_anchor_restored"
        },
        vector,
        style,
    )
    store.add(primary, current_state=state_context, cognitive_style=style)
    store.add(donor, current_state=state_context, cognitive_style=style)
    query = RetrievalQuery(
        semantic_tags=["mentor", "promise", "continuity"],
        context_tags=["archive"],
        content_keywords=["summary"],
        reference_cycle=4,
    )
    before_access = float(primary.accessibility)
    retrieval = store.retrieve(query, k=2, agent_state=vector, cognitive_style=style)
    report = store.reconsolidate_entry(
        primary.id,
        current_mood="alert",
        current_context_tags=["archive", "north_wall"],
        current_cycle=4,
        current_state=state_context,
        recall_artifact=retrieval.recall_hypothesis,
        cognitive_style=style,
    )
    refreshed = store.get(primary.id)
    after_access = float(refreshed.accessibility if refreshed is not None else before_access)
    return report.to_dict(), round(after_access - before_access, 6)


def _exploration_bias_probe() -> dict[str, Any]:
    store = MemoryStore(
        entries=[
            MemoryEntry(
                id="familiar",
                content="Familiar mentor pattern.",
                memory_class=MemoryClass.SEMANTIC,
                store_level=StoreLevel.MID,
                source_type=SourceType.EXPERIENCE,
                created_at=1,
                last_accessed=10,
                valence=0.0,
                arousal=0.2,
                encoding_attention=0.5,
                novelty=0.2,
                relevance_goal=0.2,
                relevance_threat=0.0,
                relevance_self=0.2,
                relevance_social=0.1,
                relevance_reward=0.1,
                relevance=0.2,
                salience=0.4,
                trace_strength=0.5,
                accessibility=0.7,
                abstractness=0.6,
                source_confidence=0.9,
                reality_confidence=0.9,
                semantic_tags=["mentor", "pattern", "archive"],
                context_tags=["lab"],
                retrieval_count=8,
            ),
            MemoryEntry(
                id="novel",
                content="Novel mentor pattern.",
                memory_class=MemoryClass.SEMANTIC,
                store_level=StoreLevel.MID,
                source_type=SourceType.EXPERIENCE,
                created_at=9,
                last_accessed=9,
                valence=0.0,
                arousal=0.2,
                encoding_attention=0.5,
                novelty=0.3,
                relevance_goal=0.2,
                relevance_threat=0.0,
                relevance_self=0.2,
                relevance_social=0.1,
                relevance_reward=0.1,
                relevance=0.2,
                salience=0.4,
                trace_strength=0.5,
                accessibility=0.62,
                abstractness=0.6,
                source_confidence=0.9,
                reality_confidence=0.9,
                semantic_tags=["mentor", "pattern", "archive"],
                context_tags=["lab"],
                retrieval_count=0,
            ),
        ]
    )
    query = RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=11)
    low = store.retrieve(query, k=2, cognitive_style=_style(exploration_bias=0.0))
    high = store.retrieve(query, k=2, cognitive_style=_style(exploration_bias=1.0))
    return {
        "low_top_id": low.candidates[0].entry_id,
        "high_top_id": high.candidates[0].entry_id,
    }


def _cognitive_style_probe(corpus: dict[str, Any]) -> dict[str, Any]:
    vector = _state_vector(identity_themes=["mentor", "promise", "continuity"], threat_level=0.35)
    low_uncertainty = _encode_probe(corpus["cognitive_style_probe"]["uncertainty_input"], vector, _style(uncertainty_sensitivity=0.0))
    high_uncertainty = _encode_probe(corpus["cognitive_style_probe"]["uncertainty_input"], vector, _style(uncertainty_sensitivity=1.0))
    low_error = _encode_probe(corpus["cognitive_style_probe"]["error_input"], vector, _style(error_aversion=0.0))
    high_error = _encode_probe(corpus["cognitive_style_probe"]["error_input"], vector, _style(error_aversion=1.0))
    low_selectivity = _encode_probe(corpus["cognitive_style_probe"]["selectivity_input"], vector, _style(attention_selectivity=0.0))
    high_selectivity = _encode_probe(corpus["cognitive_style_probe"]["selectivity_input"], vector, _style(attention_selectivity=1.0))
    low_recon, low_access_delta = _reconsolidation_probe(_style(update_rigidity=0.0))
    high_recon, high_access_delta = _reconsolidation_probe(_style(update_rigidity=1.0))
    return {
        "uncertainty_sensitivity": {
            "low": dict(low_uncertainty.compression_metadata or {}).get("cognitive_style_effects", {}).get("effective_novelty"),
            "high": dict(high_uncertainty.compression_metadata or {}).get("cognitive_style_effects", {}).get("effective_novelty"),
        },
        "error_aversion": {
            "low": dict(low_error.compression_metadata or {}).get("cognitive_style_effects", {}).get("effective_arousal"),
            "high": dict(high_error.compression_metadata or {}).get("cognitive_style_effects", {}).get("effective_arousal"),
        },
        "update_rigidity": {
            "low_update_type": low_recon["update_type"],
            "high_update_type": high_recon["update_type"],
        },
        "attention_selectivity": {
            "low_tag_focus": dict(low_selectivity.compression_metadata or {}).get("cognitive_style_effects", {}).get("tag_focus", {}),
            "high_tag_focus": dict(high_selectivity.compression_metadata or {}).get("cognitive_style_effects", {}).get("tag_focus", {}),
        },
        "exploration_bias": _exploration_bias_probe(),
        "identity_stability_interaction": {
            "low_access_delta": low_access_delta,
            "high_access_delta": high_access_delta,
        },
    }


def _threat_payload(text: str, cycle: int, seed_group: str, index: int) -> dict[str, Any]:
    return {
        "content": f"{text} [{seed_group}:{index}]",
        "semantic_tags": ["alarm", "threat", "mentor", "safety", "lab"],
        "context_tags": ["lab", f"seed_{seed_group}"],
        "valence": -0.64,
        "arousal": 0.72,
        "novelty": 0.32,
        "place": "lab",
        "action": "respond_to_warning",
        "outcome": "threat_encoded",
        "created_at": cycle,
        "cycle": cycle,
    }


def _run_threat_learning_group(seed_group: dict[str, Any]) -> dict[str, Any]:
    vector = _state_vector(identity_themes=["mentor", "promise"], threat_level=0.45, last_updated=1)
    high_values: list[float] = []
    low_values: list[float] = []
    traces: list[dict[str, Any]] = []
    for index, text in enumerate(seed_group["threat_learning"], start=1):
        payload = _threat_payload(text, index, str(seed_group["seed_group"]), index)
        high_entry = _encode_probe(payload, vector, _style(error_aversion=0.8))
        low_entry = _encode_probe(payload, vector, _style(error_aversion=0.2))
        high_values.append(round(float(high_entry.salience), 6))
        low_values.append(round(float(low_entry.salience), 6))
        traces.append(
            {
                "episode": text,
                "high_salience": round(float(high_entry.salience), 6),
                "low_salience": round(float(low_entry.salience), 6),
            }
        )
    return {
        "seed_group": seed_group["seed_group"],
        "high_error_aversion_salience": high_values,
        "low_error_aversion_salience": low_values,
        "trace": traces,
    }


def _semantic_clusters(seed_group: dict[str, Any], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters = list(seed_group.get("semantic_clusters", []))
    if clusters:
        return deepcopy(clusters)
    cloned = deepcopy(fallback)
    label = str(seed_group["seed_group"])
    for cluster in cloned:
        cluster["cluster_id"] = f"{cluster['cluster_id']}_{label}"
        for index, entry in enumerate(cluster["entries"], start=1):
            entry["raw_input"]["content"] = f"{entry['raw_input']['content']} [{label}:{index}]"
            entry["seed_group"] = label
    return cloned


def _run_interference_group(seed_group: dict[str, Any], fallback_clusters: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = _semantic_clusters(seed_group, fallback_clusters)
    all_low_runs: list[dict[str, Any]] = []
    all_high_runs: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        for style_name, style in (("low", _style(attention_selectivity=0.2)), ("high", _style(attention_selectivity=0.8))):
            store = MemoryStore()
            for entry_index, spec in enumerate(cluster["entries"], start=1):
                payload = _clone_with_seed(spec["raw_input"], str(seed_group["seed_group"]), (cluster_index * 10) + entry_index, entry_index)
                entry = encode_memory(payload, _state_vector(last_updated=entry_index).to_context(), SalienceConfig(), cognitive_style=style)
                store.add(entry, current_state=_state_vector(last_updated=entry_index).to_context(), cognitive_style=style)
            for query_index, query_spec in enumerate(cluster["queries"], start=1):
                query = RetrievalQuery(
                    semantic_tags=list(query_spec.get("semantic_tags", [])),
                    context_tags=list(query_spec.get("context_tags", [])),
                    content_keywords=list(query_spec.get("content_keywords", [])),
                    reference_cycle=40 + query_index,
                )
                result = store.retrieve(query, k=3, cognitive_style=style)
                run_payload = {
                    "cluster_id": cluster["cluster_id"],
                    "query": _query_to_dict(query),
                    "candidate_ids": [candidate.entry_id for candidate in result.candidates],
                    "scores": [candidate.to_dict() for candidate in result.candidates],
                    "competition": result.reconstruction_trace.get("competition_snapshot", {}),
                }
                if style_name == "low":
                    all_low_runs.append(run_payload)
                else:
                    all_high_runs.append(run_payload)
    return {
        "seed_group": seed_group["seed_group"],
        "low_selectivity_runs": all_low_runs,
        "high_selectivity_runs": all_high_runs,
    }


def _long_cycle_payload(cycle: int, seed_group: str) -> dict[str, Any]:
    if cycle == 1:
        return {
            "content": f"Reactor alarm anchored the mentor drill near the archive [{seed_group}]",
            "semantic_tags": ["alarm", "mentor", "danger", "anchor"],
            "context_tags": ["reactor_room", "drill"],
            "valence": -0.55,
            "arousal": 0.95,
            "novelty": 0.70,
            "place": "reactor_room",
            "action": "anchor_alarm",
            "outcome": "threat_anchor_saved",
            "created_at": cycle,
        }
    if cycle == 2:
        return {
            "content": f"Procedure for stabilizing the reactor during a mentor-guided drill [{seed_group}]",
            "memory_class": MemoryClass.PROCEDURAL.value,
            "semantic_tags": ["procedure", "reactor", "mentor", "stabilize"],
            "context_tags": ["reactor_room", "drill"],
            "procedure_steps": ["scan gauges", "vent pressure", "log readings"],
            "execution_contexts": ["reactor_room"],
            "valence": 0.05,
            "arousal": 0.92,
            "novelty": 0.30,
            "place": "reactor_room",
            "action": "stabilize_core",
            "outcome": "procedure_learned",
            "created_at": cycle,
        }
    if cycle == 203:
        return {
            "content": f"I met mentor Lin in the east archive on Monday to review the continuity report [{seed_group}]",
            "semantic_tags": ["mentor", "meeting", "continuity", "report"],
            "context_tags": ["archive", "east_wing"],
            "valence": 0.14,
            "arousal": 0.26,
            "novelty": 0.22,
            "place": "east_archive",
            "action": "review_report",
            "outcome": "report_reviewed",
            "created_at": cycle,
        }
    if cycle == 204:
        return {
            "content": f"I met mentor Lin in the west archive on Tuesday to review the continuity report [{seed_group}]",
            "semantic_tags": ["mentor", "meeting", "continuity", "report"],
            "context_tags": ["archive", "west_wing"],
            "valence": 0.14,
            "arousal": 0.24,
            "novelty": 0.24,
            "place": "west_archive",
            "action": "review_report",
            "outcome": "report_reviewed",
            "created_at": cycle,
        }
    if cycle == 205:
        return {
            "content": f"I met mentor Lin near the river annex on Monday to review the continuity report [{seed_group}]",
            "semantic_tags": ["mentor", "meeting", "continuity", "report"],
            "context_tags": ["annex", "river"],
            "valence": 0.14,
            "arousal": 0.24,
            "novelty": 0.24,
            "place": "river_annex",
            "action": "review_report",
            "outcome": "report_reviewed",
            "created_at": cycle,
        }
    if cycle % 5 == 0:
        return {
            "content": f"I renewed the mentor promise and continuity duty during cycle {cycle} [{seed_group}]",
            "semantic_tags": ["mentor", "promise", "continuity", "identity"],
            "context_tags": ["lab", "briefing"],
            "roles": ["mentor_keeper", "continuity_guardian"],
            "relationships": ["mentor_lin", "mentees"],
            "commitments": ["keep promises", "protect mentees", "continuity duty"],
            "identity_themes": ["mentor", "promise", "continuity"],
            "identity_relevance_hint": 0.88,
            "supporting_episode_ids": [f"{seed_group}_commitment_a", f"{seed_group}_commitment_b"],
            "valence": 0.22,
            "arousal": 0.16,
            "novelty": 0.12,
            "place": "lab",
            "action": "renew_promise",
            "outcome": "identity_reinforced",
            "created_at": cycle,
        }
    if cycle % 7 == 0:
        return {
            "content": f"A threat spike interrupted the mentor workflow during cycle {cycle} [{seed_group}]",
            "semantic_tags": ["danger", "alarm", "mentor", "threat"],
            "context_tags": ["reactor_room", "lab"],
            "valence": -0.34,
            "arousal": 0.58,
            "novelty": 0.28,
            "place": "reactor_room",
            "action": "contain_threat",
            "outcome": "threat_managed",
            "created_at": cycle,
        }
    if cycle % 9 == 0:
        return {
            "content": f"Repeated pattern note for mentor care routine during cycle {cycle} [{seed_group}]",
            "memory_class": MemoryClass.SEMANTIC.value,
            "semantic_tags": ["mentor", "pattern", "care", "lab", f"pattern_{cycle}"],
            "context_tags": ["lab", "archive"],
            "valence": 0.10,
            "arousal": 0.24,
            "novelty": 0.20,
            "place": "archive",
            "action": "record_pattern",
            "outcome": "pattern_saved",
            "created_at": cycle,
        }
    return {
        "content": f"A bright station distraction passed by during cycle {cycle} [{seed_group}]",
        "semantic_tags": ["flash", "market", "noise", "novelty"],
        "context_tags": ["station", "hallway"],
        "valence": 0.06,
        "arousal": 0.32,
        "novelty": 0.86,
        "place": "station_hallway",
        "action": "notice_distraction",
        "outcome": "noise_encoded",
        "created_at": cycle,
    }


def _long_cycle_query(cycle: int) -> RetrievalQuery:
    if cycle % 5 == 0 or cycle % 3 == 0:
        return RetrievalQuery(
            semantic_tags=["mentor", "promise", "continuity"],
            context_tags=["lab"],
            content_keywords=["mentor", "promise"],
            reference_cycle=cycle,
        )
    if cycle % 7 == 0:
        return RetrievalQuery(
            semantic_tags=["danger", "alarm", "mentor"],
            context_tags=["reactor_room"],
            reference_cycle=cycle,
        )
    if cycle % 2 == 0:
        return RetrievalQuery(
            semantic_tags=["procedure", "reactor"],
            context_tags=["reactor_room"],
            reference_cycle=cycle,
        )
    return RetrievalQuery(
        semantic_tags=["mentor", "pattern"],
        context_tags=["archive"],
        reference_cycle=cycle,
    )


def _run_long_seed_group(seed_group: dict[str, Any]) -> dict[str, Any]:
    seed_label = str(seed_group["seed_group"])
    cycles = int(seed_group["cycles"])
    agent = MemoryAwareSegmentAgent(
        memory_cognitive_style=_style(
            uncertainty_sensitivity=0.55,
            error_aversion=0.55,
            attention_selectivity=0.65,
            exploration_bias=0.45,
            update_rigidity=0.35,
        ),
        memory_cycle_interval=5,
    )
    agent.agent_state_vector.active_goals = ["keep promises", "protect mentees"]
    agent.agent_state_vector.identity_active_themes = ["mentor", "promise", "continuity"]
    logs: list[dict[str, Any]] = []
    consolidation_reports: list[dict[str, Any]] = []
    identity_ids: list[str] = []
    noise_ids: list[str] = []
    for cycle in range(1, cycles + 1):
        payload = _long_cycle_payload(cycle, seed_label)
        entry = agent.encode_cycle_memory(payload, cycle)
        if entry.relevance_self >= 0.35:
            identity_ids.append(entry.id)
        if entry.novelty >= 0.75 and entry.relevance_self < 0.20:
            noise_ids.append(entry.id)
        query = _long_cycle_query(cycle)
        retrieval = agent.retrieve_for_decision(
            query,
            cycle,
            current_mood=agent.agent_state_vector.recent_mood_baseline,
            k=3,
        )
        reconsolidation_report = None
        if retrieval.recall_hypothesis is not None:
            reconsolidation_report = agent.reconsolidate_after_recall(
                retrieval.recall_hypothesis.primary_entry_id,
                current_mood=agent.agent_state_vector.recent_mood_baseline,
                current_context_tags=list(query.context_tags),
                current_cycle=cycle,
                recall_artifact=retrieval.recall_hypothesis,
            )
        decay_report = agent.memory_store.apply_decay(cycle)
        consolidation_report = agent.run_memory_consolidation_if_due(cycle, rng=random.Random(cycle))
        if consolidation_report is not None:
            consolidation_reports.append({"cycle": cycle, "report": consolidation_report.to_dict()})
        logs.append(
            {
                "cycle": cycle,
                "encoded_entry_id": entry.id,
                "encoded_store_level": entry.store_level.value,
                "state_vector": agent.agent_state_vector.to_dict(),
                "retrieval_candidate_ids": [candidate.entry_id for candidate in retrieval.candidates],
                "retrieval_primary_id": retrieval.recall_hypothesis.primary_entry_id if retrieval.recall_hypothesis else None,
                "reconsolidation_update_type": reconsolidation_report.update_type if reconsolidation_report is not None else None,
                "decay_deleted_short_residue": list(getattr(decay_report, "deleted_short_residue", [])),
            }
        )
    self_query = RetrievalQuery(
        semantic_tags=["mentor", "promise", "continuity"],
        context_tags=["lab"],
        content_keywords=["promise"],
        reference_cycle=cycles,
    )
    self_related_recall = agent.retrieve_for_decision(self_query, cycles, current_mood="reflective", k=3)
    mis_query = RetrievalQuery(
        semantic_tags=["mentor", "meeting", "continuity"],
        context_tags=["archive"],
        content_keywords=["report"],
        reference_cycle=cycles,
    )
    mis_result = agent.retrieve_for_decision(mis_query, cycles, current_mood="reflective", k=3)
    entries = [entry.to_dict() for entry in agent.memory_store.entries]
    return {
        "seed_group": seed_label,
        "cycle_count": cycles,
        "log": logs,
        "consolidation_reports": consolidation_reports,
        "tracked_identity_ids": identity_ids,
        "tracked_noise_ids": noise_ids,
        "self_related_recall": self_related_recall.to_dict(),
        "misattribution": mis_result.to_dict(),
        "entries": entries,
        "layer_distribution": {
            "short": sum(1 for entry in agent.memory_store.entries if entry.store_level is StoreLevel.SHORT),
            "mid": sum(1 for entry in agent.memory_store.entries if entry.store_level is StoreLevel.MID),
            "long": sum(1 for entry in agent.memory_store.entries if entry.store_level is StoreLevel.LONG),
        },
        "restored_state_vector": MemoryAwareSegmentAgent.from_dict(agent.to_dict(), rng=random.Random(7)).agent_state_vector.to_dict(),
    }


def build_m47_runtime_snapshot(*, corpus_path: Path | str | None = None) -> dict[str, Any]:
    corpus = load_m47_corpus(corpus_path)
    fallback_clusters = list(corpus["short_seed_groups"][0]["semantic_clusters"])
    short_groups = [
        {
            "seed_group": spec["seed_group"],
            "scenario_a": _run_threat_learning_group(spec),
            "scenario_b": _run_interference_group(spec, fallback_clusters),
        }
        for spec in corpus["short_seed_groups"]
    ]
    long_groups = [_run_long_seed_group(spec) for spec in corpus["long_seed_groups"]]
    return {
        "mode": "m47_shared_runtime_snapshot",
        "generated_at": _now_iso(),
        "corpus_path": str(Path(corpus_path).resolve() if corpus_path is not None else M47_CORPUS_PATH),
        "schema_version": str(corpus.get("schema_version", "unknown")),
        "probes": {
            "state_vector": _state_vector_probe(corpus),
            "dynamic_salience": _dynamic_salience_probe(corpus),
            "cognitive_style": _cognitive_style_probe(corpus),
        },
        "short_seed_groups": short_groups,
        "long_seed_groups": long_groups,
    }


def write_m47_runtime_snapshot(
    *,
    output_path: Path | str | None = None,
    corpus_path: Path | str | None = None,
) -> str:
    snapshot = build_m47_runtime_snapshot(corpus_path=corpus_path)
    target = Path(output_path).resolve() if output_path is not None else M47_RUNTIME_SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)
