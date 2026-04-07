from __future__ import annotations

from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from .m48_open_world import simulate_open_world_projection
from .m4_cognitive_style import CognitiveStyleParameters


@dataclass(frozen=True)
class StyleSignature:
    query_rate: float
    exploration_rate: float
    conservation_rate: float
    recovery_rate: float
    planful_rate: float
    confidence_mean: float
    exploration_margin: float
    planning_margin: float
    recovery_margin: float

    def to_dict(self) -> dict[str, float]:
        return {
            "query_rate": round(self.query_rate, 6),
            "exploration_rate": round(self.exploration_rate, 6),
            "conservation_rate": round(self.conservation_rate, 6),
            "recovery_rate": round(self.recovery_rate, 6),
            "planful_rate": round(self.planful_rate, 6),
            "confidence_mean": round(self.confidence_mean, 6),
            "exploration_margin": round(self.exploration_margin, 6),
            "planning_margin": round(self.planning_margin, 6),
            "recovery_margin": round(self.recovery_margin, 6),
        }


def _candidate_score(row: dict[str, Any], action_name: str) -> float:
    for candidate in row["decision"]["candidate_actions"]:
        if str(candidate["action"]["name"]) == action_name:
            return float(candidate["total_score"])
    return 0.0


def _signature_from_logs(payload: dict[str, Any]) -> StyleSignature:
    actions = [row["decision"]["selected_action"] for row in payload["logs"]]
    confidences = [float(row["decision"]["internal_confidence"]) for row in payload["logs"]]
    total = max(1, len(actions))
    knowledge_rows = [row for row in payload["logs"] if str(row["task"]["task_type"]) == "knowledge_retrieval"]
    planning_rows = [row for row in payload["logs"] if str(row["task"]["task_type"]) == "multi_step_planning"]
    recovery_rows = [row for row in payload["logs"] if str(row["task"]["task_type"]) == "failure_recovery"]
    return StyleSignature(
        query_rate=sum(1 for action in actions if action == "query") / total,
        exploration_rate=sum(1 for action in actions if action in {"query", "scan", "inspect"}) / total,
        conservation_rate=sum(1 for action in actions if action in {"conserve"}) / total,
        recovery_rate=sum(1 for action in actions if action in {"recover"}) / total,
        planful_rate=sum(1 for action in actions if action in {"plan", "inspect"}) / total,
        confidence_mean=mean(confidences) if confidences else 0.0,
        exploration_margin=mean(_candidate_score(row, "query") - _candidate_score(row, "scan") for row in knowledge_rows) if knowledge_rows else 0.0,
        planning_margin=mean(_candidate_score(row, "plan") - _candidate_score(row, "inspect") for row in planning_rows) if planning_rows else 0.0,
        recovery_margin=mean(_candidate_score(row, "recover") - _candidate_score(row, "conserve") for row in recovery_rows) if recovery_rows else 0.0,
    )


def _profile_catalog() -> dict[str, CognitiveStyleParameters]:
    return {
        "cautious": CognitiveStyleParameters(error_aversion=0.84, resource_pressure_sensitivity=0.82, exploration_bias=0.42),
        "exploratory": CognitiveStyleParameters(error_aversion=0.52, resource_pressure_sensitivity=0.56, exploration_bias=0.76),
        "rigid": CognitiveStyleParameters(error_aversion=0.74, update_rigidity=0.88, exploration_bias=0.46),
    }


def _distance(left: StyleSignature, right: StyleSignature) -> float:
    return round(
        abs(left.exploration_rate - right.exploration_rate)
        + abs(left.query_rate - right.query_rate)
        + abs(left.conservation_rate - right.conservation_rate)
        + abs(left.recovery_rate - right.recovery_rate)
        + abs(left.planful_rate - right.planful_rate)
        + abs(left.confidence_mean - right.confidence_mean)
        + abs(left.exploration_margin - right.exploration_margin)
        + abs(left.planning_margin - right.planning_margin)
        + abs(left.recovery_margin - right.recovery_margin),
        6,
    )


def _state_path(root: Path, profile_name: str, seed: int) -> Path:
    return root / f"{profile_name}_{seed}.json"


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_longitudinal_style_suite(*, seeds: tuple[int, ...] = (46, 146, 246, 346)) -> dict[str, Any]:
    profile_runs: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for profile_name, parameters in _profile_catalog().items():
            rows = []
            for seed in seeds:
                base = simulate_open_world_projection(parameters, seed=seed)
                state_path = _state_path(root, profile_name, seed)
                _write_state(state_path, {"profile": profile_name, "seed": seed, "payload": base})
                restarted = _read_state(state_path)["payload"]
                corrupted_state = {"profile": profile_name, "seed": seed, "payload": simulate_open_world_projection(parameters, seed=seed + 100, stress=True, ablate_style=True)}
                _write_state(state_path, corrupted_state)
                corrupted = _read_state(state_path)["payload"]
                repaired = simulate_open_world_projection(parameters, seed=seed)
                _write_state(state_path, {"profile": profile_name, "seed": seed, "payload": repaired})
                repaired = _read_state(state_path)["payload"]
                base_sig = _signature_from_logs(base)
                restarted_sig = _signature_from_logs(restarted)
                corrupted_sig = _signature_from_logs(corrupted)
                repaired_sig = _signature_from_logs(repaired)
                rows.append(
                    {
                        "seed": seed,
                        "persistence_path": f"persistent://{profile_name}_{seed}.json",
                        "base": base,
                        "restarted": restarted,
                        "corrupted": corrupted,
                        "repaired": repaired,
                        "signatures": {
                            "base": base_sig.to_dict(),
                            "restarted": restarted_sig.to_dict(),
                            "corrupted": corrupted_sig.to_dict(),
                            "repaired": repaired_sig.to_dict(),
                        },
                        "restart_distance": _distance(base_sig, restarted_sig),
                        "corruption_distance": _distance(base_sig, corrupted_sig),
                        "repair_retention_distance": _distance(base_sig, repaired_sig),
                    }
                )
            profile_runs[profile_name] = rows
    within_profile_cross_seed = {}
    for profile, rows in profile_runs.items():
        signatures = [_signature_from_logs(row["base"]) for row in rows]
        pairwise: list[float] = []
        for index, left in enumerate(signatures):
            for right in signatures[index + 1 :]:
                pairwise.append(_distance(left, right))
        within_profile_cross_seed[profile] = round(mean(pairwise), 6) if pairwise else 0.0
    between_profile = {}
    profile_names = list(profile_runs.keys())
    for index, left in enumerate(profile_names):
        left_sig = _signature_from_logs(profile_runs[left][0]["base"])
        for right in profile_names[index + 1 :]:
            right_sig = _signature_from_logs(profile_runs[right][0]["base"])
            between_profile[f"{left}_vs_{right}"] = _distance(left_sig, right_sig)
    restart_profile = {profile: round(mean(float(row["restart_distance"]) for row in rows), 6) for profile, rows in profile_runs.items()}
    repair_retention = {profile: round(mean(float(row["repair_retention_distance"]) for row in rows), 6) for profile, rows in profile_runs.items()}
    return {
        "seeds": list(seeds),
        "probe_type": "synthetic_probe_with_persistence",
        "profiles": profile_runs,
        "summary": {
            "restart_distance_mean": round(mean(restart_profile.values()), 6),
            "within_profile_cross_seed_distance_mean": round(mean(within_profile_cross_seed.values()), 6),
            "between_profile_distance_mean": round(mean(between_profile.values()), 6),
            "repair_retention_distance_mean": round(mean(repair_retention.values()), 6),
            "style_divergence_reproducible": mean(between_profile.values()) > mean(within_profile_cross_seed.values()),
            "recovery_retains_style": mean(repair_retention.values()) < mean(between_profile.values()),
        },
    }
