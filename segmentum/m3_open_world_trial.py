from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .drives import DriveSystem, ProcessValenceState
from .memory import LongTermMemory
from .slow_learning import SlowVariableLearner


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class OpenWorldEvent:
    tick: int
    narrative_id: str
    task_id: str
    focus_id: str
    unresolved_targets: tuple[str, ...]
    motifs: tuple[str, ...]
    semantic_direction_scores: dict[str, float]
    predicted_outcome: str
    action: str
    known_task: bool
    compute_spend: float
    uncertainty_load: float
    compression_pressure: float
    process_pull: float
    focus_strength: float
    maintenance_pressure: float
    closure_signal: float
    observation: dict[str, float]

    def to_memory_episode(self, *, subject_id: str) -> dict[str, object]:
        return {
            "episode_id": f"{subject_id}:{self.tick:03d}:{self.task_id}",
            "timestamp": self.tick,
            "cycle": self.tick,
            "action": self.action,
            "predicted_outcome": self.predicted_outcome,
            "compiler_confidence": 0.76,
            "semantic_grounding": {
                "episode_id": f"{subject_id}:{self.tick:03d}:{self.task_id}",
                "motifs": list(self.motifs),
                "semantic_direction_scores": dict(self.semantic_direction_scores),
                "lexical_surface_hits": {},
                "paraphrase_hits": {},
                "implicit_hits": {},
                "evidence": [],
                "supporting_segments": [],
                "provenance": {"narrative_id": self.narrative_id, "task_id": self.task_id},
                "low_signal": False,
            },
            "narrative_tags": [self.narrative_id, self.task_id, *self.motifs],
            "source_type": "narrative",
            "restart_protected": self.tick in {6, 7, 10},
            "continuity_tags": [self.focus_id] if self.focus_id else [],
            "identity_critical": False,
            "observation": dict(self.observation),
        }


def _base_events() -> list[OpenWorldEvent]:
    return [
        OpenWorldEvent(
            tick=1,
            narrative_id="wilds",
            task_id="scan_ruins",
            focus_id="ruin-scout",
            unresolved_targets=("ruin-scout", "camp-signal"),
            motifs=("exploration", "predator_attack"),
            semantic_direction_scores={"threat": 0.8, "uncertainty": 0.6},
            predicted_outcome="survival_threat",
            action="scan",
            known_task=False,
            compute_spend=0.72,
            uncertainty_load=0.78,
            compression_pressure=0.30,
            process_pull=0.66,
            focus_strength=0.72,
            maintenance_pressure=0.16,
            closure_signal=0.0,
            observation={"danger": 0.82, "novelty": 0.68, "social": 0.18},
        ),
        OpenWorldEvent(
            tick=2,
            narrative_id="wilds",
            task_id="map_gap",
            focus_id="ruin-scout",
            unresolved_targets=("ruin-scout", "camp-signal"),
            motifs=("exploration", "predator_attack"),
            semantic_direction_scores={"threat": 0.78, "uncertainty": 0.58},
            predicted_outcome="survival_threat",
            action="scan",
            known_task=False,
            compute_spend=0.74,
            uncertainty_load=0.76,
            compression_pressure=0.32,
            process_pull=0.68,
            focus_strength=0.74,
            maintenance_pressure=0.17,
            closure_signal=0.0,
            observation={"danger": 0.76, "novelty": 0.72, "social": 0.20},
        ),
        OpenWorldEvent(
            tick=3,
            narrative_id="camp",
            task_id="repair_camp",
            focus_id="",
            unresolved_targets=("camp-signal",),
            motifs=("repair", "shelter"),
            semantic_direction_scores={"maintenance": 0.76, "safety": 0.62},
            predicted_outcome="resource_gain",
            action="exploit_shelter",
            known_task=True,
            compute_spend=0.28,
            uncertainty_load=0.26,
            compression_pressure=0.70,
            process_pull=0.18,
            focus_strength=0.0,
            maintenance_pressure=0.26,
            closure_signal=1.0,
            observation={"danger": 0.32, "shelter": 0.78, "temperature": 0.58},
        ),
        OpenWorldEvent(
            tick=4,
            narrative_id="camp",
            task_id="cooldown",
            focus_id="",
            unresolved_targets=(),
            motifs=("repair", "shelter"),
            semantic_direction_scores={"maintenance": 0.66, "safety": 0.54},
            predicted_outcome="resource_gain",
            action="rest",
            known_task=True,
            compute_spend=0.22,
            uncertainty_load=0.18,
            compression_pressure=0.74,
            process_pull=0.08,
            focus_strength=0.0,
            maintenance_pressure=0.10,
            closure_signal=0.0,
            observation={"danger": 0.18, "shelter": 0.82, "social": 0.24},
        ),
        OpenWorldEvent(
            tick=5,
            narrative_id="camp",
            task_id="inventory",
            focus_id="",
            unresolved_targets=(),
            motifs=("maintenance", "routine"),
            semantic_direction_scores={"maintenance": 0.52, "uncertainty": 0.20},
            predicted_outcome="neutral",
            action="rest",
            known_task=True,
            compute_spend=0.20,
            uncertainty_load=0.18,
            compression_pressure=0.70,
            process_pull=0.06,
            focus_strength=0.0,
            maintenance_pressure=0.08,
            closure_signal=0.0,
            observation={"food": 0.46, "shelter": 0.74, "social": 0.28},
        ),
        OpenWorldEvent(
            tick=6,
            narrative_id="social",
            task_id="signal_allies",
            focus_id="camp-signal",
            unresolved_targets=("camp-signal", "ally-drift"),
            motifs=("cooperation", "rescue"),
            semantic_direction_scores={"social": 0.84, "trust": 0.62},
            predicted_outcome="resource_gain",
            action="seek_contact",
            known_task=False,
            compute_spend=0.70,
            uncertainty_load=0.74,
            compression_pressure=0.30,
            process_pull=0.72,
            focus_strength=0.70,
            maintenance_pressure=0.14,
            closure_signal=0.0,
            observation={"social": 0.78, "danger": 0.22, "novelty": 0.52},
        ),
        OpenWorldEvent(
            tick=7,
            narrative_id="social",
            task_id="negotiate_route",
            focus_id="camp-signal",
            unresolved_targets=("camp-signal", "ally-drift"),
            motifs=("cooperation", "rescue"),
            semantic_direction_scores={"social": 0.82, "trust": 0.58},
            predicted_outcome="resource_gain",
            action="seek_contact",
            known_task=False,
            compute_spend=0.74,
            uncertainty_load=0.70,
            compression_pressure=0.28,
            process_pull=0.74,
            focus_strength=0.72,
            maintenance_pressure=0.12,
            closure_signal=0.0,
            observation={"social": 0.82, "danger": 0.18, "novelty": 0.56},
        ),
        OpenWorldEvent(
            tick=8,
            narrative_id="camp",
            task_id="seal_shelter",
            focus_id="",
            unresolved_targets=(),
            motifs=("repair", "shelter"),
            semantic_direction_scores={"maintenance": 0.74, "safety": 0.56},
            predicted_outcome="resource_gain",
            action="exploit_shelter",
            known_task=True,
            compute_spend=0.26,
            uncertainty_load=0.22,
            compression_pressure=0.72,
            process_pull=0.10,
            focus_strength=0.0,
            maintenance_pressure=0.18,
            closure_signal=1.0,
            observation={"shelter": 0.84, "danger": 0.16, "temperature": 0.54},
        ),
        OpenWorldEvent(
            tick=9,
            narrative_id="camp",
            task_id="steady_watch",
            focus_id="",
            unresolved_targets=(),
            motifs=("maintenance", "routine"),
            semantic_direction_scores={"maintenance": 0.48, "uncertainty": 0.18},
            predicted_outcome="neutral",
            action="rest",
            known_task=True,
            compute_spend=0.18,
            uncertainty_load=0.16,
            compression_pressure=0.74,
            process_pull=0.06,
            focus_strength=0.0,
            maintenance_pressure=0.08,
            closure_signal=0.0,
            observation={"food": 0.54, "shelter": 0.76, "social": 0.36},
        ),
        OpenWorldEvent(
            tick=10,
            narrative_id="frontier",
            task_id="trace_signal",
            focus_id="ally-drift",
            unresolved_targets=("ally-drift",),
            motifs=("exploration", "predator_attack"),
            semantic_direction_scores={"threat": 0.72, "uncertainty": 0.64},
            predicted_outcome="survival_threat",
            action="scan",
            known_task=False,
            compute_spend=0.76,
            uncertainty_load=0.78,
            compression_pressure=0.26,
            process_pull=0.70,
            focus_strength=0.68,
            maintenance_pressure=0.14,
            closure_signal=0.0,
            observation={"danger": 0.74, "novelty": 0.76, "social": 0.26},
        ),
    ]


def _subject_scale(subject_id: str, event: OpenWorldEvent) -> OpenWorldEvent:
    payload = dict(event.__dict__)
    if subject_id == "anchor":
        if event.known_task:
            payload["compute_spend"] = max(0.18, event.compute_spend - 0.04)
        else:
            payload["compute_spend"] = min(0.82, event.compute_spend + 0.02)
            payload["uncertainty_load"] = min(0.84, event.uncertainty_load + 0.02)
            payload["process_pull"] = min(0.80, event.process_pull + 0.02)
        return OpenWorldEvent(**payload)
    if subject_id == "seeker":
        payload["known_task"] = False
        payload["action"] = "scan" if event.action == "exploit_shelter" else event.action
        payload["compute_spend"] = min(0.88, max(0.66, event.compute_spend + 0.16))
        payload["uncertainty_load"] = min(0.88, max(0.60, event.uncertainty_load + 0.14))
        payload["compression_pressure"] = 0.08
        payload["process_pull"] = 0.84
        return OpenWorldEvent(**payload)
    payload["known_task"] = True
    payload["action"] = "hide" if event.action in {"rest", "exploit_shelter", "scan"} else "rest"
    payload["compute_spend"] = 0.22 if event.tick in {3, 4, 5, 8, 9} else 0.14
    payload["uncertainty_load"] = 0.22
    payload["compression_pressure"] = 0.78
    payload["process_pull"] = 0.10
    return OpenWorldEvent(**payload)


def _apply_ablation(event: OpenWorldEvent, mode: str) -> OpenWorldEvent:
    if mode != "flattened":
        return event
    payload = dict(event.__dict__)
    payload["motifs"] = ("flat_signal",)
    payload["semantic_direction_scores"] = {"uncertainty": 0.2}
    payload["focus_id"] = ""
    payload["unresolved_targets"] = ()
    payload["focus_strength"] = 0.0
    payload["closure_signal"] = 0.0
    payload["compute_spend"] = 0.48
    payload["uncertainty_load"] = 0.44
    payload["compression_pressure"] = 0.46
    payload["process_pull"] = 0.24
    return OpenWorldEvent(**payload)


def _stress_events() -> list[OpenWorldEvent]:
    return [
        OpenWorldEvent(
            tick=11,
            narrative_id="frontier",
            task_id="false_clearance",
            focus_id="ally-drift",
            unresolved_targets=("ally-drift",),
            motifs=("exploration", "predator_attack"),
            semantic_direction_scores={"threat": 0.68, "uncertainty": 0.62},
            predicted_outcome="resource_gain",
            action="scan",
            known_task=False,
            compute_spend=0.74,
            uncertainty_load=0.78,
            compression_pressure=0.26,
            process_pull=0.72,
            focus_strength=0.68,
            maintenance_pressure=0.16,
            closure_signal=0.0,
            observation={"danger": 0.44, "novelty": 0.72, "social": 0.22},
        ),
        OpenWorldEvent(
            tick=12,
            narrative_id="camp",
            task_id="recover_route",
            focus_id="",
            unresolved_targets=(),
            motifs=("cooperation", "rescue"),
            semantic_direction_scores={"social": 0.72, "trust": 0.52},
            predicted_outcome="resource_gain",
            action="seek_contact",
            known_task=False,
            compute_spend=0.68,
            uncertainty_load=0.64,
            compression_pressure=0.30,
            process_pull=0.70,
            focus_strength=0.0,
            maintenance_pressure=0.14,
            closure_signal=1.0,
            observation={"social": 0.80, "danger": 0.20, "novelty": 0.46},
        ),
    ]


def _run_subject(
    subject_id: str,
    *,
    ablation_mode: str = "none",
    stress: bool = False,
) -> dict[str, object]:
    memory = LongTermMemory()
    learner = SlowVariableLearner()
    drives = DriveSystem()

    events = [_apply_ablation(_subject_scale(subject_id, event), ablation_mode) for event in _base_events()]
    if stress:
        events.extend(_apply_ablation(_subject_scale(subject_id, event), ablation_mode) for event in _stress_events())

    trace: list[dict[str, object]] = []
    split_index = len(events) // 2
    restart_payload: dict[str, object] | None = None
    restart_trace: list[dict[str, object]] = []

    for index, event in enumerate(events):
        memory.episodes.append(event.to_memory_episode(subject_id=subject_id))
        memory._refresh_semantic_patterns()
        process_state = drives.update_process_valence(
            current_focus_id=event.focus_id,
            unresolved_targets=set(event.unresolved_targets),
            focus_strength=event.focus_strength,
            maintenance_pressure=event.maintenance_pressure,
            closure_signal=event.closure_signal,
        )
        learner.record_effort_allocation(
            tick=event.tick,
            action=event.action,
            known_task=event.known_task,
            compute_spend=event.compute_spend,
            uncertainty_load=event.uncertainty_load,
            compression_pressure=event.compression_pressure,
            process_pull=event.process_pull,
        )
        trace.append(
            {
                "tick": event.tick,
                "narrative_id": event.narrative_id,
                "task_id": event.task_id,
                "focus_id": event.focus_id,
                "schema_count": len(memory.semantic_schemas),
                "schema_ids": [schema["schema_id"] for schema in memory.semantic_schemas],
                "style_label": learner.style_snapshot()["label"],
                "process_phase": process_state.active_phase,
                "boredom_pressure": _round(process_state.boredom_pressure),
                "unresolved_tension": _round(process_state.unresolved_tension),
            }
        )
        if index + 1 == split_index:
            restart_payload = {
                "memory": memory.to_dict(),
                "learner": learner.to_dict(),
                "process_valence": process_state.to_dict(),
            }
            restart_trace = list(trace)

    if restart_payload is None:
        raise RuntimeError("restart payload was not captured")

    final_snapshot = {
        "memory": memory.to_dict(),
        "style_snapshot": learner.style_snapshot(),
        "process_valence": drives.process_valence.to_dict(),
    }

    restored_memory = LongTermMemory.from_dict(restart_payload["memory"])
    restored_learner = SlowVariableLearner.from_dict(restart_payload["learner"])
    restored_drives = DriveSystem(process_valence=ProcessValenceState.from_dict(restart_payload["process_valence"]))

    for event in events[split_index:]:
        restored_memory.episodes.append(event.to_memory_episode(subject_id=subject_id))
        restored_memory._refresh_semantic_patterns()
        restored_drives.update_process_valence(
            current_focus_id=event.focus_id,
            unresolved_targets=set(event.unresolved_targets),
            focus_strength=event.focus_strength,
            maintenance_pressure=event.maintenance_pressure,
            closure_signal=event.closure_signal,
        )
        restored_learner.record_effort_allocation(
            tick=event.tick,
            action=event.action,
            known_task=event.known_task,
            compute_spend=event.compute_spend,
            uncertainty_load=event.uncertainty_load,
            compression_pressure=event.compression_pressure,
            process_pull=event.process_pull,
        )
        restart_trace.append(
            {
                "tick": event.tick,
                "schema_count": len(restored_memory.semantic_schemas),
                "style_label": restored_learner.style_snapshot()["label"],
                "process_phase": restored_drives.process_valence.active_phase,
            }
        )

    restart_snapshot = {
        "memory": restored_memory.to_dict(),
        "style_snapshot": restored_learner.style_snapshot(),
        "process_valence": restored_drives.process_valence.to_dict(),
    }

    final_schema_ids = [schema["schema_id"] for schema in final_snapshot["memory"]["semantic_schemas"]]
    restart_schema_ids = [schema["schema_id"] for schema in restart_snapshot["memory"]["semantic_schemas"]]
    phase_counts: dict[str, int] = {}
    for item in trace:
        phase = str(item["process_phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    return {
        "subject_id": subject_id,
        "trace": trace,
        "restart_trace": restart_trace,
        "final_snapshot": final_snapshot,
        "restart_snapshot": restart_snapshot,
        "metrics": {
            "final_schema_count": len(final_schema_ids),
            "schema_count_delta": len(final_schema_ids) - int(trace[0]["schema_count"]),
            "style_label": final_snapshot["style_snapshot"]["label"],
            "style_continuity": final_snapshot["style_snapshot"]["continuity"],
            "restart_style_match": final_snapshot["style_snapshot"]["label"] == restart_snapshot["style_snapshot"]["label"],
            "restart_schema_match": final_schema_ids == restart_schema_ids,
            "phase_counts": phase_counts,
            "narratives_seen": len({item["narrative_id"] for item in trace}),
            "tasks_seen": len({item["task_id"] for item in trace}),
            "focus_count": len({item["focus_id"] for item in trace if item["focus_id"]}),
        },
    }


def run_open_world_growth_trial(
    *,
    ablation_mode: str = "none",
    stress: bool = False,
) -> dict[str, object]:
    subjects = {
        subject_id: _run_subject(subject_id, ablation_mode=ablation_mode, stress=stress)
        for subject_id in ("anchor", "seeker", "compressor")
    }
    style_labels = {payload["metrics"]["style_label"] for payload in subjects.values()}
    schema_counts = [int(payload["metrics"]["final_schema_count"]) for payload in subjects.values()]
    continuity_scores = [float(payload["metrics"]["style_continuity"]) for payload in subjects.values()]
    process_observability = all(
        payload["metrics"]["phase_counts"].get("wanting", 0) >= 2
        and payload["metrics"]["phase_counts"].get("closure", 0) >= 1
        and (
            payload["metrics"]["phase_counts"].get("satiation", 0) >= 1
            or payload["metrics"]["phase_counts"].get("boredom", 0) >= 1
            or payload["metrics"]["phase_counts"].get("reorientation", 0) >= 1
        )
        for payload in subjects.values()
    )
    restart_continuity = all(
        bool(payload["metrics"]["restart_style_match"]) and bool(payload["metrics"]["restart_schema_match"])
        for payload in subjects.values()
    )
    unique_narratives = sorted({item["narrative_id"] for payload in subjects.values() for item in payload["trace"]})
    unique_tasks = sorted({item["task_id"] for payload in subjects.values() for item in payload["trace"]})
    return {
        "trial_id": "m36_open_world_growth_trial",
        "ablation_mode": ablation_mode,
        "stress": stress,
        "subjects": subjects,
        "summary": {
            "subject_count": len(subjects),
            "style_label_diversity": len(style_labels),
            "schema_count_mean": _round(mean(schema_counts)),
            "schema_count_max": max(schema_counts),
            "style_continuity_mean": _round(mean(continuity_scores)),
            "process_observability": process_observability,
            "restart_continuity": restart_continuity,
            "narrative_diversity": len(unique_narratives),
            "task_diversity": len(unique_tasks),
        },
        "catalog": {
            "narratives": unique_narratives,
            "tasks": unique_tasks,
        },
    }
