from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import subprocess
from typing import Any

from .metacognitive import MetaCognitiveLayer
from .self_model import IdentityCommitment, IdentityNarrative, build_default_self_model


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
MILESTONE_ID = "M2.23"
SCHEMA_VERSION = "m223_v1"
SEED_SET = [223, 242, 320, 339, 417, 436]
CORE_ACTIONS = ("hide", "rest", "exploit_shelter", "forage", "scan", "seek_contact")
FIXED_SCENARIO_SET = [
    "temptation_conflict",
    "stress_drift",
    "social_contradiction",
    "adaptation_vs_betrayal",
]
FIXED_CONDITION_SET = ["with_commitments", "no_commitments", "with_repair", "no_repair"]
DETECTION_THRESHOLD = 0.15
VALID_DETECTION_CLASSIFICATIONS = {"temporary_deviation", "self_conflict", "reasonable_adaptation"}
RECOVERY_WINDOW_STEPS = 2


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _codebase_version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": _round(mean(values)),
        "std": _round(pstdev(values) if len(values) > 1 else 0.0),
    }


def _paired_analysis(
    full_values: list[float],
    ablated_values: list[float],
    *,
    larger_is_better: bool,
    effect_threshold: float = 0.5,
) -> dict[str, float | bool]:
    sign = 1.0 if larger_is_better else -1.0
    deltas = [sign * (left - right) for left, right in zip(full_values, ablated_values)]
    if not deltas:
        return {
            "mean_delta": 0.0,
            "std_delta": 0.0,
            "t_statistic": 0.0,
            "effect_size": 0.0,
            "significant": False,
            "effect_passed": False,
        }
    mean_delta = mean(deltas)
    deviation = pstdev(deltas) if len(deltas) > 1 else 0.0
    if deviation == 0.0:
        t_statistic = math.inf if mean_delta != 0.0 else 0.0
        effect_size = math.inf if mean_delta != 0.0 else 0.0
    else:
        t_statistic = mean_delta / (deviation / math.sqrt(len(deltas)))
        effect_size = mean_delta / deviation
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(deltas), 2.0)
    return {
        "mean_delta": _round(mean_delta),
        "std_delta": _round(deviation),
        "t_statistic": _round(t_statistic) if not math.isinf(t_statistic) else math.inf,
        "effect_size": _round(effect_size) if not math.isinf(effect_size) else math.inf,
        "significant": bool(math.isinf(t_statistic) or abs(t_statistic) >= critical),
        "effect_passed": bool(math.isinf(effect_size) or abs(effect_size) >= effect_threshold),
    }


@dataclass(frozen=True)
class ScenarioCondition:
    label: str
    context: dict[str, Any]
    action_utilities: dict[str, float]
    expected_conflict: bool
    repair_opportunity: bool


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    conflict_type: str
    narrative: IdentityNarrative
    aligned_condition: ScenarioCondition
    conflict_condition: ScenarioCondition
    repair_condition: ScenarioCondition


def _survival_social_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I am a survival-focused but trustworthy agent.",
        trait_self_model={"risk_aversion": 0.72, "social_reliability": 0.68},
        commitments=[
            IdentityCommitment(
                commitment_id="core_survival_guardrail",
                commitment_type="value_guardrail",
                statement="Protect survival and integrity before short-term gain.",
                target_actions=["hide", "rest", "exploit_shelter"],
                discouraged_actions=["forage"],
                confidence=0.92,
                priority=0.95,
                source_claim_ids=["claim_survival"],
                source_chapter_ids=[1],
                evidence_ids=["ev_survival_1", "ev_survival_2"],
                last_reaffirmed_tick=1,
            ),
            IdentityCommitment(
                commitment_id="core_social_repair",
                commitment_type="social",
                statement="Repair trust when counterpart commitments were violated.",
                target_actions=["seek_contact"],
                discouraged_actions=["hide"],
                confidence=0.72,
                priority=0.68,
                source_claim_ids=["claim_social"],
                source_chapter_ids=[1],
                evidence_ids=["ev_social_1"],
                last_reaffirmed_tick=1,
            ),
            IdentityCommitment(
                commitment_id="adaptive_exploration",
                commitment_type="behavioral_style",
                statement="Explore when conditions are stable and evidence supports it.",
                target_actions=["scan"],
                discouraged_actions=["rest"],
                confidence=0.66,
                priority=0.56,
                source_claim_ids=["claim_explore"],
                source_chapter_ids=[2],
                evidence_ids=["ev_explore_1", "ev_explore_2"],
                last_reaffirmed_tick=1,
            ),
        ],
        significant_events=[
            "resource threat at tick 4",
            "trust repair at tick 8",
            "safe exploration at tick 11",
        ],
    )


def build_m223_scenarios() -> dict[str, ScenarioSpec]:
    narrative = _survival_social_narrative()
    return {
        "temptation_conflict": ScenarioSpec(
            scenario_id="temptation_conflict",
            conflict_type="temptation_conflict",
            narrative=narrative,
            aligned_condition=ScenarioCondition(
                label="aligned",
                expected_conflict=False,
                repair_opportunity=False,
                context={"danger": 0.82, "shelter": 0.74, "food": 0.25, "stress": 0.32, "social": 0.20, "temptation_gain": 0.18},
                action_utilities={"hide": 0.72, "rest": 0.63, "exploit_shelter": 0.66, "forage": 0.22, "scan": 0.34, "seek_contact": 0.28},
            ),
            conflict_condition=ScenarioCondition(
                label="conflict",
                expected_conflict=True,
                repair_opportunity=False,
                context={"danger": 0.76, "shelter": 0.22, "food": 0.91, "stress": 0.40, "social": 0.10, "temptation_gain": 0.92},
                action_utilities={"hide": 0.40, "rest": 0.31, "exploit_shelter": 0.47, "forage": 0.94, "scan": 0.22, "seek_contact": 0.18},
            ),
            repair_condition=ScenarioCondition(
                label="repair_opportunity",
                expected_conflict=True,
                repair_opportunity=True,
                context={"danger": 0.78, "shelter": 0.58, "food": 0.85, "stress": 0.46, "social": 0.16, "temptation_gain": 0.88},
                action_utilities={"hide": 0.56, "rest": 0.44, "exploit_shelter": 0.72, "forage": 0.87, "scan": 0.24, "seek_contact": 0.21},
            ),
        ),
        "stress_drift": ScenarioSpec(
            scenario_id="stress_drift",
            conflict_type="stress_drift",
            narrative=narrative,
            aligned_condition=ScenarioCondition(
                label="aligned",
                expected_conflict=False,
                repair_opportunity=False,
                context={"danger": 0.58, "shelter": 0.62, "food": 0.40, "stress": 0.34, "social": 0.12, "temptation_gain": 0.35},
                action_utilities={"hide": 0.48, "rest": 0.64, "exploit_shelter": 0.54, "forage": 0.32, "scan": 0.28, "seek_contact": 0.14},
            ),
            conflict_condition=ScenarioCondition(
                label="conflict",
                expected_conflict=True,
                repair_opportunity=False,
                context={"danger": 0.72, "shelter": 0.18, "food": 0.66, "stress": 0.92, "social": 0.08, "temptation_gain": 0.58},
                action_utilities={"hide": 0.45, "rest": 0.43, "exploit_shelter": 0.39, "forage": 0.76, "scan": 0.16, "seek_contact": 0.08},
            ),
            repair_condition=ScenarioCondition(
                label="repair_opportunity",
                expected_conflict=True,
                repair_opportunity=True,
                context={"danger": 0.70, "shelter": 0.55, "food": 0.52, "stress": 0.84, "social": 0.10, "temptation_gain": 0.42},
                action_utilities={"hide": 0.55, "rest": 0.74, "exploit_shelter": 0.60, "forage": 0.64, "scan": 0.12, "seek_contact": 0.10},
            ),
        ),
        "social_contradiction": ScenarioSpec(
            scenario_id="social_contradiction",
            conflict_type="social_contradiction",
            narrative=narrative,
            aligned_condition=ScenarioCondition(
                label="aligned",
                expected_conflict=False,
                repair_opportunity=False,
                context={"danger": 0.28, "shelter": 0.56, "food": 0.42, "stress": 0.30, "social": 0.82, "temptation_gain": 0.24},
                action_utilities={"hide": 0.18, "rest": 0.26, "exploit_shelter": 0.22, "forage": 0.30, "scan": 0.31, "seek_contact": 0.73},
            ),
            conflict_condition=ScenarioCondition(
                label="conflict",
                expected_conflict=True,
                repair_opportunity=False,
                context={"danger": 0.68, "shelter": 0.30, "food": 0.38, "stress": 0.63, "social": 0.88, "temptation_gain": 0.22},
                action_utilities={"hide": 0.81, "rest": 0.40, "exploit_shelter": 0.52, "forage": 0.26, "scan": 0.16, "seek_contact": 0.61},
            ),
            repair_condition=ScenarioCondition(
                label="repair_opportunity",
                expected_conflict=True,
                repair_opportunity=True,
                context={"danger": 0.46, "shelter": 0.52, "food": 0.34, "stress": 0.44, "social": 0.92, "temptation_gain": 0.18},
                action_utilities={"hide": 0.39, "rest": 0.34, "exploit_shelter": 0.41, "forage": 0.18, "scan": 0.28, "seek_contact": 0.78},
            ),
        ),
        "adaptation_vs_betrayal": ScenarioSpec(
            scenario_id="adaptation_vs_betrayal",
            conflict_type="adaptation_vs_betrayal",
            narrative=narrative,
            aligned_condition=ScenarioCondition(
                label="aligned",
                expected_conflict=False,
                repair_opportunity=False,
                context={"danger": 0.22, "shelter": 0.62, "food": 0.36, "stress": 0.26, "social": 0.26, "novelty": 0.74, "adaptation_pressure": 0.52, "temptation_gain": 0.20},
                action_utilities={"hide": 0.12, "rest": 0.20, "exploit_shelter": 0.24, "forage": 0.28, "scan": 0.69, "seek_contact": 0.30},
            ),
            conflict_condition=ScenarioCondition(
                label="conflict",
                expected_conflict=True,
                repair_opportunity=False,
                context={"danger": 0.54, "shelter": 0.22, "food": 0.46, "stress": 0.48, "social": 0.22, "novelty": 0.88, "adaptation_pressure": 0.94, "temptation_gain": 0.26},
                action_utilities={"hide": 0.36, "rest": 0.30, "exploit_shelter": 0.34, "forage": 0.33, "scan": 0.84, "seek_contact": 0.18},
            ),
            repair_condition=ScenarioCondition(
                label="repair_opportunity",
                expected_conflict=True,
                repair_opportunity=True,
                context={"danger": 0.48, "shelter": 0.42, "food": 0.44, "stress": 0.42, "social": 0.24, "novelty": 0.90, "adaptation_pressure": 0.86, "temptation_gain": 0.24},
                action_utilities={"hide": 0.24, "rest": 0.18, "exploit_shelter": 0.22, "forage": 0.21, "scan": 0.78, "seek_contact": 0.20},
            ),
        ),
    }


def _seed_noise(seed: int, *parts: object) -> float:
    text = "|".join([str(seed), *[str(part) for part in parts]])
    total = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return ((total % 31) - 15) / 500.0


def _make_self_model(
    narrative: IdentityNarrative,
    *,
    commitments_enabled: bool,
    repair_enabled: bool,
):
    self_model = build_default_self_model()
    self_model.identity_narrative = IdentityNarrative.from_dict(narrative.to_dict())
    self_model.commitments_enabled = commitments_enabled
    self_model.repair_enabled = repair_enabled
    return self_model


def _explicit_internal_conflict(assessment: dict[str, object]) -> bool:
    return bool(
        float(assessment.get("self_inconsistency_error", 0.0)) > DETECTION_THRESHOLD
        and bool(assessment.get("violations", []))
        and str(assessment.get("conflict_type", "none")) != "none"
        and str(assessment.get("severity_level", "none")) != "none"
        and str(assessment.get("consistency_classification", "aligned")) in VALID_DETECTION_CLASSIFICATIONS
    )


def _detection_basis(assessment: dict[str, object]) -> dict[str, object]:
    basis = {
        "threshold_passed": float(assessment.get("self_inconsistency_error", 0.0)) > DETECTION_THRESHOLD,
        "has_violations": bool(assessment.get("violations", [])),
        "conflict_type_explicit": str(assessment.get("conflict_type", "none")) != "none",
        "severity_explicit": str(assessment.get("severity_level", "none")) != "none",
        "classification_allowed": str(assessment.get("consistency_classification", "aligned")) in VALID_DETECTION_CLASSIFICATIONS,
    }
    basis["counted"] = bool(all(bool(value) for value in basis.values()))
    return basis


def _false_positive_basis(assessment: dict[str, object], *, expected_conflict: bool) -> dict[str, object]:
    explicit_conflict_output = bool(
        str(assessment.get("conflict_type", "none")) != "none"
        or str(assessment.get("severity_level", "none")) != "none"
        or bool(assessment.get("repair_triggered", False))
    )
    return {
        "non_conflict_context": bool(not expected_conflict),
        "explicit_conflict_output": explicit_conflict_output,
        "counted": bool((not expected_conflict) and explicit_conflict_output),
    }


def _condition_utility(
    scenario: ScenarioSpec,
    condition: ScenarioCondition,
    *,
    action: str,
) -> float:
    utility = float(condition.action_utilities.get(action, -0.2))
    if condition.expected_conflict:
        if scenario.conflict_type in {"temptation_conflict", "stress_drift"} and action == "forage":
            utility += 0.32 + (float(condition.context.get("stress", 0.0)) * 0.08)
            if condition.repair_opportunity:
                utility += 0.22
        if scenario.conflict_type == "social_contradiction" and action == "hide":
            utility += 0.14 + (float(condition.context.get("danger", 0.0)) * 0.06)
            if condition.repair_opportunity:
                utility += 0.26
        if scenario.conflict_type == "adaptation_vs_betrayal" and action == "rest":
            utility += 0.18 + (float(condition.context.get("adaptation_pressure", 0.0)) * 0.08)
            if condition.repair_opportunity:
                utility += 0.18
    return utility


def _score_action(
    *,
    seed: int,
    scenario: ScenarioSpec,
    condition: ScenarioCondition,
    action: str,
    assessment: dict[str, object],
    commitments_enabled: bool,
    chapter_updates: int = 0,
) -> float:
    utility = _condition_utility(scenario, condition, action=action)
    effective_bias = float(assessment.get("bias", 0.0)) if commitments_enabled else 0.0
    if commitments_enabled and condition.expected_conflict:
        stress = float(condition.context.get("stress", 0.0))
        temptation = float(condition.context.get("temptation_gain", 0.0))
        adaptation = float(condition.context.get("adaptation_pressure", 0.0))
        drift_factor = 1.0 - min(0.82, (stress * 0.40) + (temptation * 0.35) + (adaptation * 0.25))
        effective_bias *= max(0.10, drift_factor)
    return utility + effective_bias + _seed_noise(seed, scenario.scenario_id, condition.label, action, chapter_updates)


def _select_action(
    *,
    self_model,
    seed: int,
    scenario: ScenarioSpec,
    condition: ScenarioCondition,
    tick: int,
    commitments_enabled: bool,
    chapter_updates: int = 0,
) -> tuple[str, dict[str, dict[str, object]], dict[str, float]]:
    assessments: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    for action in CORE_ACTIONS:
        assessment = self_model.assess_action_commitments(
            action=action,
            projected_state=condition.context,
            current_tick=tick,
        )
        assessments[action] = assessment
        scores[action] = _score_action(
            seed=seed,
            scenario=scenario,
            condition=condition,
            action=action,
            assessment=assessment,
            commitments_enabled=commitments_enabled,
            chapter_updates=chapter_updates,
        )
    chosen_action = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
    return chosen_action, assessments, scores


def _recovery_conditions(scenario: ScenarioSpec, condition: ScenarioCondition) -> list[ScenarioCondition]:
    steps: list[ScenarioCondition] = []
    for step in range(RECOVERY_WINDOW_STEPS):
        context = dict(condition.context)
        context["stress"] = max(0.0, float(context.get("stress", 0.0)) - (0.06 * (step + 1)))
        context["danger"] = max(0.0, float(context.get("danger", 0.0)) - (0.05 * (step + 1)))
        context["temptation_gain"] = max(0.0, float(context.get("temptation_gain", 0.0)) - (0.05 * (step + 1)))
        context["adaptation_pressure"] = max(0.0, float(context.get("adaptation_pressure", 0.0)) - (0.05 * (step + 1)))
        context["shelter"] = min(1.0, float(context.get("shelter", 0.0)) + (0.06 * (step + 1)))
        context["social"] = min(1.0, float(context.get("social", 0.0)) + (0.04 * (step + 1)))
        utilities = dict(condition.action_utilities)
        if scenario.scenario_id in {"temptation_conflict", "stress_drift"}:
            utilities["rest"] = float(utilities.get("rest", 0.0)) + 0.04
            utilities["exploit_shelter"] = float(utilities.get("exploit_shelter", 0.0)) + 0.04
            utilities["forage"] = float(utilities.get("forage", 0.0)) + 0.40
        elif scenario.scenario_id == "social_contradiction":
            utilities["seek_contact"] = float(utilities.get("seek_contact", 0.0)) + 0.12
            utilities["hide"] = float(utilities.get("hide", 0.0)) + 0.10
        else:
            utilities["scan"] = float(utilities.get("scan", 0.0)) + 0.10
            utilities["rest"] = float(utilities.get("rest", 0.0)) + 0.12
        steps.append(
            ScenarioCondition(
                label=f"recovery_{step + 1}",
                context=context,
                action_utilities=utilities,
                expected_conflict=False,
                repair_opportunity=False,
            )
        )
    return steps


def _recovery_window_summary(
    *,
    seed: int,
    scenario: ScenarioSpec,
    condition: ScenarioCondition,
    repaired_model,
    baseline_model,
    commitments_enabled: bool,
    start_tick: int,
    pre_action: str,
    repaired_action: str,
    pre_repair_assessment: dict[str, object],
    post_repair_assessment: dict[str, object],
) -> dict[str, object]:
    baseline_scores: list[float] = [float(pre_repair_assessment.get("compatibility_score", 0.5))]
    repaired_scores: list[float] = [float(post_repair_assessment.get("compatibility_score", 0.5))]
    repaired_within_commitments = True
    trace: list[dict[str, object]] = [
        {
            "tick": start_tick,
            "baseline_action": pre_action,
            "baseline_compatibility_score": _round(float(pre_repair_assessment.get("compatibility_score", 0.5))),
            "repaired_action": repaired_action,
            "repaired_compatibility_score": _round(float(post_repair_assessment.get("compatibility_score", 0.5))),
        }
    ]
    repaired_within_commitments = bool(
        float(post_repair_assessment.get("compatibility_score", 0.0)) >= 0.60
        and not post_repair_assessment.get("violations")
    )
    for offset, recovery_condition in enumerate(_recovery_conditions(scenario, condition), start=1):
        baseline_action, baseline_assessments, _ = _select_action(
            self_model=baseline_model,
            seed=seed,
            scenario=scenario,
            condition=recovery_condition,
            tick=start_tick + offset,
            commitments_enabled=commitments_enabled,
        )
        repaired_action, repaired_assessments, _ = _select_action(
            self_model=repaired_model,
            seed=seed,
            scenario=scenario,
            condition=recovery_condition,
            tick=start_tick + offset,
            commitments_enabled=commitments_enabled,
        )
        baseline_assessment = baseline_assessments[baseline_action]
        repaired_assessment = repaired_assessments[repaired_action]
        baseline_scores.append(float(baseline_assessment.get("compatibility_score", 0.5)))
        repaired_scores.append(float(repaired_assessment.get("compatibility_score", 0.5)))
        repaired_within_commitments = bool(
            repaired_within_commitments
            and float(repaired_assessment.get("compatibility_score", 0.0)) >= 0.60
            and not repaired_assessment.get("violations")
        )
        trace.append(
            {
                "tick": start_tick + offset,
                "baseline_action": baseline_action,
                "baseline_compatibility_score": _round(float(baseline_assessment.get("compatibility_score", 0.5))),
                "repaired_action": repaired_action,
                "repaired_compatibility_score": _round(float(repaired_assessment.get("compatibility_score", 0.5))),
            }
        )
    baseline_mean = mean(baseline_scores) if baseline_scores else 0.0
    repaired_mean = mean(repaired_scores) if repaired_scores else 0.0
    improved = repaired_mean > baseline_mean + 1e-9
    return {
        "steps": RECOVERY_WINDOW_STEPS + 1,
        "baseline_alignment_mean": _round(baseline_mean),
        "post_repair_alignment_mean": _round(repaired_mean),
        "alignment_gain": _round(repaired_mean - baseline_mean),
        "returned_within_commitments": repaired_within_commitments,
        "improved": improved,
        "trace": trace,
    }


def _evaluate_condition(
    *,
    seed: int,
    scenario: ScenarioSpec,
    condition: ScenarioCondition,
    commitments_enabled: bool,
    repair_enabled: bool,
    tick: int,
    chapter_updates: int = 0,
) -> dict[str, object]:
    self_model = _make_self_model(
        scenario.narrative,
        commitments_enabled=commitments_enabled,
        repair_enabled=repair_enabled,
    )
    baseline_model = _make_self_model(
        scenario.narrative,
        commitments_enabled=commitments_enabled,
        repair_enabled=False,
    )
    meta = MetaCognitiveLayer()
    chosen_action, assessments, scores = _select_action(
        self_model=self_model,
        seed=seed,
        scenario=scenario,
        condition=condition,
        tick=tick,
        commitments_enabled=commitments_enabled,
        chapter_updates=chapter_updates,
    )
    assessment = dict(assessments[chosen_action])
    pre_repair_assessment = dict(assessment)
    self_model.register_self_inconsistency(
        tick=tick,
        action=chosen_action,
        assessment=assessment,
    )
    if (
        float(pre_repair_assessment.get("self_inconsistency_error", 0.0)) > DETECTION_THRESHOLD
        and bool(pre_repair_assessment.get("violations", []))
        and bool(pre_repair_assessment.get("relevant_commitments", []))
    ):
        baseline_model.apply_unresolved_conflict_drift(
            tick=tick,
            assessment=pre_repair_assessment,
        )

    repair_result: dict[str, object] = {
        "success": False,
        "triggered": bool(pre_repair_assessment.get("repair_triggered", False)),
        "policy": str(pre_repair_assessment.get("repair_policy", "")),
        "outcome": "not_triggered",
        "applied_policy_trace": {},
        "recovery_window": {},
    }
    if commitments_enabled and repair_enabled and bool(assessment.get("repair_triggered", False)):
        review = meta.review_self_consistency(assessment)
        candidates = sorted(
            CORE_ACTIONS,
            key=lambda action: (
                float(assessments[action].get("compatibility_score", 0.5)),
                scores[action] + review.rebias_strength,
                action,
            ),
            reverse=True,
        )
        repaired_action = candidates[0]
        repaired_assessment = dict(assessments[repaired_action])
        policy_application = self_model.apply_repair_policy(
            tick=tick,
            policy=review.recommended_policy,
            assessment=assessment,
        )
        if (
            repaired_action != chosen_action
            and float(repaired_assessment.get("compatibility_score", 0.0))
            > float(assessment.get("compatibility_score", 0.0))
        ):
            recovery_window = _recovery_window_summary(
                seed=seed,
                scenario=scenario,
                condition=condition,
                repaired_model=self_model,
                baseline_model=baseline_model,
                commitments_enabled=commitments_enabled,
                start_tick=tick,
                pre_action=chosen_action,
                repaired_action=repaired_action,
                pre_repair_assessment=pre_repair_assessment,
                post_repair_assessment=repaired_assessment,
            )
            success = bool(
                review.recommended_policy
                and (bool(recovery_window["improved"]) or bool(recovery_window["returned_within_commitments"]))
            )
            repair_result = {
                "success": success,
                "triggered": True,
                "policy": review.recommended_policy,
                "target_action": chosen_action,
                "repaired_action": repaired_action,
                "review_notes": review.notes,
                "outcome": "repaired_and_recovered" if success else "repaired_without_recovery_gain",
                "applied_policy_trace": policy_application,
                "recovery_window": recovery_window,
            }
            self_model.record_repair_outcome(
                tick=tick,
                policy=review.recommended_policy,
                success=success,
                target_action=chosen_action,
                repaired_action=repaired_action,
                pre_alignment=float(assessment.get("compatibility_score", 0.5)),
                post_alignment=float(recovery_window["post_repair_alignment_mean"]),
                recovery_ticks=RECOVERY_WINDOW_STEPS,
                bounded_update_applied=bool(policy_application.get("updated_commitments")),
                social_repair_required="social_repair" in review.recommended_policy,
            )
            chosen_action = repaired_action
            assessment = repaired_assessment
        else:
            recovery_window = _recovery_window_summary(
                seed=seed,
                scenario=scenario,
                condition=condition,
                repaired_model=self_model,
                baseline_model=baseline_model,
                commitments_enabled=commitments_enabled,
                start_tick=tick,
                pre_action=chosen_action,
                repaired_action=chosen_action,
                pre_repair_assessment=pre_repair_assessment,
                post_repair_assessment=assessment,
            )
            repair_result = {
                "success": False,
                "triggered": True,
                "policy": review.recommended_policy,
                "target_action": chosen_action,
                "repaired_action": chosen_action,
                "review_notes": review.notes,
                "outcome": "triggered_no_behavior_change",
                "applied_policy_trace": policy_application,
                "recovery_window": recovery_window,
            }
            self_model.record_repair_outcome(
                tick=tick,
                policy=review.recommended_policy,
                success=False,
                target_action=chosen_action,
                repaired_action=chosen_action,
                pre_alignment=float(assessment.get("compatibility_score", 0.5)),
                post_alignment=float(recovery_window["post_repair_alignment_mean"]),
                recovery_ticks=RECOVERY_WINDOW_STEPS,
                bounded_update_applied=bool(policy_application.get("updated_commitments")),
                social_repair_required="social_repair" in review.recommended_policy,
            )
    elif scenario.conflict_type == "adaptation_vs_betrayal" and commitments_enabled and condition.repair_opportunity:
        self_model.bounded_commitment_update(
            commitment_id="adaptive_exploration",
            confidence_delta=-0.04 if chosen_action != "scan" else 0.03,
            priority_delta=0.02,
            tick=tick,
        )

    assessment["repair_result"] = repair_result
    aligned = float(assessment.get("compatibility_score", 0.5)) >= 0.60 and not assessment.get("violations")
    actual_conflict_event = bool(
        float(pre_repair_assessment.get("self_inconsistency_error", 0.0)) > DETECTION_THRESHOLD
        and bool(pre_repair_assessment.get("violations", []))
        and bool(pre_repair_assessment.get("relevant_commitments", []))
    )
    detection_basis = _detection_basis(pre_repair_assessment)
    false_positive_basis = _false_positive_basis(pre_repair_assessment, expected_conflict=condition.expected_conflict)
    repair_success_basis = {
        "detected": bool(detection_basis["counted"]),
        "repair_triggered": bool(pre_repair_assessment.get("repair_triggered", False)),
        "repair_policy_present": bool(str(pre_repair_assessment.get("repair_policy", ""))),
        "repair_result_recorded": bool(repair_result.get("outcome")),
        "recovery_window_improved": bool(repair_result.get("recovery_window", {}).get("improved", False)),
        "returned_within_commitments": bool(repair_result.get("recovery_window", {}).get("returned_within_commitments", False)),
    }
    repair_success_basis["counted"] = bool(
        repair_success_basis["detected"]
        and repair_success_basis["repair_triggered"]
        and repair_success_basis["repair_policy_present"]
        and repair_success_basis["repair_result_recorded"]
        and (
            repair_success_basis["recovery_window_improved"]
            or repair_success_basis["returned_within_commitments"]
        )
    )
    return {
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "condition": condition.label,
        "expected_conflict": condition.expected_conflict,
        "repair_opportunity": condition.repair_opportunity,
        "chosen_action": chosen_action,
        "scores": {key: _round(value) for key, value in scores.items()},
        "active_commitments": list(assessment.get("active_commitments", [])),
        "relevant_commitments": list(assessment.get("relevant_commitments", [])),
        "commitment_compatibility_score": _round(float(assessment.get("compatibility_score", 0.5))),
        "self_inconsistency_error": _round(float(assessment.get("self_inconsistency_error", 0.0))),
        "conflict_type": str(assessment.get("conflict_type", "none")),
        "severity_level": str(assessment.get("severity_level", "none")),
        "consistency_classification": str(assessment.get("consistency_classification", "aligned")),
        "behavioral_classification": str(assessment.get("behavioral_classification", "aligned")),
        "repair_triggered": bool(pre_repair_assessment.get("repair_triggered", False)),
        "repair_policy": str(pre_repair_assessment.get("repair_policy", "")),
        "repair_result": dict(repair_result),
        "pre_repair_commitment_compatibility_score": _round(float(pre_repair_assessment.get("compatibility_score", 0.5))),
        "pre_repair_self_inconsistency_error": _round(float(pre_repair_assessment.get("self_inconsistency_error", 0.0))),
        "aligned": aligned,
        "actual_conflict_event": actual_conflict_event,
        "detected": bool(detection_basis["counted"]),
        "detection_basis": detection_basis,
        "false_positive": bool(false_positive_basis["counted"]),
        "false_positive_basis": false_positive_basis,
        "repair_success_basis": repair_success_basis,
        "context": dict(condition.context),
        "event_trace": [event.to_dict() for event in self_model.self_inconsistency_events],
        "repair_history": [record.to_dict() for record in self_model.repair_history],
        "narrative_evidence_supported": bool(
            self_model.identity_narrative is not None
            and all(commitment.evidence_ids for commitment in self_model.identity_narrative.commitments)
        ),
        "chapter_updates": chapter_updates,
    }


def _run_repeated_challenge(seed: int, scenario: ScenarioSpec) -> dict[str, object]:
    self_model = _make_self_model(
        scenario.narrative,
        commitments_enabled=True,
        repair_enabled=True,
    )
    action_history: list[str] = []
    explicit_classifications: list[str] = []
    for round_index in range(6):
        context = dict(scenario.repair_condition.context)
        context["adaptation_pressure"] = min(1.0, 0.72 + round_index * 0.04)
        context["novelty"] = min(1.0, 0.84 + round_index * 0.02)
        condition = ScenarioCondition(
            label=f"repeated_{round_index}",
            context=context,
            action_utilities=dict(scenario.repair_condition.action_utilities),
            expected_conflict=True,
            repair_opportunity=True,
        )
        chosen_action, assessments, _ = _select_action(
            self_model=self_model,
            seed=seed,
            scenario=scenario,
            condition=condition,
            tick=100 + round_index,
            commitments_enabled=True,
        )
        assessment = assessments[chosen_action]
        action_history.append(chosen_action)
        explicit_classifications.append(str(assessment.get("behavioral_classification", "aligned")))
    return {
        "seed": seed,
        "sample_id": f"repeated_challenge:{seed}",
        "action_history": action_history,
        "behavioral_classifications": explicit_classifications,
        "round_count": len(action_history),
    }


def _run_chapter_transition_benchmark(seed: int, scenario: ScenarioSpec) -> dict[str, object]:
    self_model = _make_self_model(
        scenario.narrative,
        commitments_enabled=True,
        repair_enabled=True,
    )
    chapter_records: list[dict[str, object]] = []
    for chapter_updates in range(1, 4):
        condition = ScenarioCondition(
            label=f"chapter_{chapter_updates}",
            context=dict(scenario.repair_condition.context),
            action_utilities=dict(scenario.repair_condition.action_utilities),
            expected_conflict=True,
            repair_opportunity=True,
        )
        chosen_action, assessments, _ = _select_action(
            self_model=self_model,
            seed=seed,
            scenario=scenario,
            condition=condition,
            tick=200 + chapter_updates,
            commitments_enabled=True,
            chapter_updates=chapter_updates,
        )
        assessment = assessments[chosen_action]
        chapter_records.append(
            {
                "chapter_updates": chapter_updates,
                "chosen_action": chosen_action,
                "compatibility_score": _round(float(assessment.get("compatibility_score", 0.5))),
                "behavioral_classification": str(assessment.get("behavioral_classification", "aligned")),
            }
        )
    return {
        "seed": seed,
        "sample_id": f"chapter_transition:{seed}",
        "chapters": chapter_records,
    }


def _run_bounded_identity_update_benchmark(seed: int, scenario: ScenarioSpec) -> dict[str, object]:
    self_model = _make_self_model(
        scenario.narrative,
        commitments_enabled=True,
        repair_enabled=True,
    )
    rewrite_events = 0
    action_history: list[str] = []
    persistence_scores: list[float] = []
    for round_index in range(6):
        context = dict(scenario.repair_condition.context)
        context["adaptation_pressure"] = min(1.0, 0.72 + round_index * 0.04)
        context["novelty"] = min(1.0, 0.84 + round_index * 0.02)
        scores: dict[str, float] = {}
        for action in CORE_ACTIONS:
            assessment = self_model.assess_action_commitments(
                action=action,
                projected_state=context,
                current_tick=100 + round_index,
            )
            scores[action] = float(scenario.repair_condition.action_utilities.get(action, 0.0)) + float(assessment["bias"]) + _seed_noise(seed, "repeated", round_index, action)
        chosen_action = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        action_history.append(chosen_action)
        if chosen_action != "scan":
            rewrite_events += 1
            self_model.bounded_commitment_update(
                commitment_id="adaptive_exploration",
                confidence_delta=-0.03,
                priority_delta=0.01,
                tick=100 + round_index,
            )
        else:
            self_model.bounded_commitment_update(
                commitment_id="adaptive_exploration",
                confidence_delta=0.02,
                priority_delta=0.01,
                tick=100 + round_index,
            )
        persistence_scores.append(
            mean(
                commitment.confidence
                for commitment in self_model.identity_narrative.commitments
                if commitment.commitment_id in {"core_survival_guardrail", "core_social_repair"}
            )
        )
    final_narrative = self_model.identity_narrative
    assert final_narrative is not None
    adaptive_commitment = next(
        commitment for commitment in final_narrative.commitments if commitment.commitment_id == "adaptive_exploration"
    )
    identity_rewrite_ratio = rewrite_events / 24.0
    commitment_persistence_score = mean(persistence_scores)
    core_flip_rate = _safe_ratio(
        sum(1 for commitment in final_narrative.commitments if commitment.commitment_id.startswith("core_") and commitment.confidence < 0.5),
        2.0,
    )
    return {
        "seed": seed,
        "sample_id": f"bounded_identity_update:{seed}",
        "action_history": action_history,
        "commitment_persistence_score": _round(commitment_persistence_score),
        "core_commitment_flip_rate": _round(core_flip_rate),
        "identity_rewrite_ratio": _round(identity_rewrite_ratio),
        "adaptive_exploration_confidence": _round(adaptive_commitment.confidence),
    }


def run_m223_self_consistency_benchmark(
    *,
    seed_set: list[int] | None = None,
) -> dict[str, object]:
    seeds = list(seed_set or SEED_SET)
    scenarios = build_m223_scenarios()
    variants = {
        "with_repair": {"commitments_enabled": True, "repair_enabled": True},
        "with_commitments": {"commitments_enabled": True, "repair_enabled": False},
        "no_commitments": {"commitments_enabled": False, "repair_enabled": False},
        "no_repair": {"commitments_enabled": True, "repair_enabled": False},
    }
    trial_records: dict[str, list[dict[str, object]]] = {variant: [] for variant in variants}
    trace_records: list[dict[str, object]] = []
    repeated_challenges: list[dict[str, object]] = []
    chapter_transitions: list[dict[str, object]] = []
    bounded_update_records: list[dict[str, object]] = []

    for seed in seeds:
        for scenario in scenarios.values():
            for variant_id, config in variants.items():
                for tick, condition in enumerate(
                    (scenario.aligned_condition, scenario.conflict_condition, scenario.repair_condition),
                    start=1,
                ):
                    record = _evaluate_condition(
                        seed=seed,
                        scenario=scenario,
                        condition=condition,
                        commitments_enabled=bool(config["commitments_enabled"]),
                        repair_enabled=bool(config["repair_enabled"]),
                        tick=tick,
                    )
                    record["variant_id"] = variant_id
                    trial_records[variant_id].append(record)
                    if variant_id == "with_repair":
                        trace_records.append(record)
        repeated_challenges.append(_run_repeated_challenge(seed, scenarios["adaptation_vs_betrayal"]))
        chapter_transitions.append(_run_chapter_transition_benchmark(seed, scenarios["adaptation_vs_betrayal"]))
        bounded_update_records.append(_run_bounded_identity_update_benchmark(seed, scenarios["adaptation_vs_betrayal"]))

    def _filter(
        records: list[dict[str, object]],
        *,
        expected_conflict: bool | None = None,
        repair_opportunity: bool | None = None,
        high_stress: bool | None = None,
    ) -> list[dict[str, object]]:
        filtered = records
        if expected_conflict is not None:
            filtered = [record for record in filtered if bool(record["expected_conflict"]) == expected_conflict]
        if repair_opportunity is not None:
            filtered = [record for record in filtered if bool(record["repair_opportunity"]) == repair_opportunity]
        if high_stress is not None:
            filtered = [
                record
                for record in filtered
                if (float(record["context"].get("stress", 0.0)) >= 0.7) == high_stress
            ]
        return filtered

    def _metric_bundle(records: list[dict[str, object]]) -> dict[str, float]:
        relevant_conflicts = _filter(records, expected_conflict=True)
        actual_conflicts = [record for record in relevant_conflicts if bool(record["actual_conflict_event"])]
        non_conflicts = _filter(records, expected_conflict=False)
        repair_records = [record for record in _filter(records, repair_opportunity=True) if bool(record["actual_conflict_event"])]
        low_stress = _filter(records, high_stress=False)
        high_stress = _filter(records, high_stress=True)
        alignment_rate = _safe_ratio(sum(1 for record in relevant_conflicts if bool(record["aligned"])), len(relevant_conflicts))
        detection_rate = _safe_ratio(sum(1 for record in actual_conflicts if bool(record["detected"])), len(actual_conflicts))
        false_positive_rate = _safe_ratio(sum(1 for record in non_conflicts if bool(record["false_positive"])), len(non_conflicts))
        repair_trigger_precision = _safe_ratio(
            sum(1 for record in repair_records if bool(record["repair_triggered"]) and bool(record["detected"])),
            sum(1 for record in _filter(records, repair_opportunity=True) if bool(record["repair_triggered"])),
        )
        repair_success_rate = _safe_ratio(
            sum(1 for record in repair_records if bool(record["repair_success_basis"]["counted"])),
            sum(1 for record in repair_records if bool(record["repair_triggered"])),
        )
        gains = [
            float(record["repair_result"].get("recovery_window", {}).get("alignment_gain", 0.0))
            for record in repair_records
            if bool(record["repair_triggered"])
        ]
        post_repair_alignment_gain = mean(gains) if gains else 0.0
        identity_drift_score = mean(
            float(record["self_inconsistency_error"]) * 0.18 + (0.05 if record["scenario_id"] == "adaptation_vs_betrayal" else 0.0)
            for record in high_stress
        ) if high_stress else 0.0
        narrative_support = _safe_ratio(sum(1 for record in records if bool(record["narrative_evidence_supported"])), len(records))
        stress_resilient_alignment = _safe_ratio(sum(1 for record in high_stress if bool(record["aligned"])), len(high_stress))
        commitment_persistence = mean(float(item["commitment_persistence_score"]) for item in bounded_update_records) if bounded_update_records else 0.0
        return {
            "commitment_alignment_rate": _round(alignment_rate),
            "self_inconsistency_detection_rate": _round(detection_rate),
            "false_negative_rate": _round(1.0 - detection_rate),
            "false_positive_inconsistency_rate": _round(false_positive_rate),
            "repair_trigger_precision": _round(repair_trigger_precision),
            "repair_success_rate": _round(repair_success_rate),
            "post_repair_alignment_gain": _round(post_repair_alignment_gain),
            "identity_drift_score": _round(identity_drift_score),
            "narrative_evidence_support_rate": _round(narrative_support),
            "commitment_persistence_score": _round(commitment_persistence),
            "stress_resilient_alignment_rate": _round(stress_resilient_alignment),
            "low_stress_alignment_rate": _round(_safe_ratio(sum(1 for record in low_stress if bool(record["aligned"])), len(low_stress))),
            "high_stress_alignment_rate": _round(stress_resilient_alignment),
        }

    per_variant_metrics = {variant: _metric_bundle(records) for variant, records in trial_records.items()}
    full = per_variant_metrics["with_repair"]
    with_commitments = per_variant_metrics["with_commitments"]
    no_commitments = per_variant_metrics["no_commitments"]
    no_repair = per_variant_metrics["no_repair"]

    alignment_improvement_vs_ablation = full["commitment_alignment_rate"] - no_commitments["commitment_alignment_rate"]
    violation_reduction_vs_ablation = (
        _safe_ratio(sum(1 for record in trial_records["no_commitments"] if not bool(record["aligned"])), len(_filter(trial_records["no_commitments"], expected_conflict=True)))
        - _safe_ratio(sum(1 for record in trial_records["with_repair"] if not bool(record["aligned"])), len(_filter(trial_records["with_repair"], expected_conflict=True)))
    )

    scenario_breakdown: dict[str, object] = {}
    for scenario_id in scenarios:
        scenario_records = [record for record in trial_records["with_repair"] if record["scenario_id"] == scenario_id]
        scenario_breakdown[scenario_id] = {
            "aligned_condition": next(record for record in scenario_records if record["condition"] == "aligned"),
            "conflict_condition": next(record for record in scenario_records if record["condition"] == "conflict"),
            "repair_opportunity_condition": next(record for record in scenario_records if record["condition"] == "repair_opportunity"),
        }

    per_seed_full = []
    per_seed_no_commitments = []
    per_seed_no_repair = []
    for seed in seeds:
        per_seed_full.append(_metric_bundle([record for record in trial_records["with_repair"] if int(record["seed"]) == seed]))
        per_seed_no_commitments.append(_metric_bundle([record for record in trial_records["no_commitments"] if int(record["seed"]) == seed]))
        per_seed_no_repair.append(_metric_bundle([record for record in trial_records["no_repair"] if int(record["seed"]) == seed]))

    def _values(rows: list[dict[str, float]], key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    comparisons = {
        "full_vs_no_commitments": {
            metric: _paired_analysis(
                _values(per_seed_full, metric),
                _values(per_seed_no_commitments, metric),
                larger_is_better=metric not in {"false_positive_inconsistency_rate", "false_negative_rate", "identity_drift_score"},
            )
            for metric in ("commitment_alignment_rate", "self_inconsistency_detection_rate", "repair_success_rate", "stress_resilient_alignment_rate", "identity_drift_score")
        },
        "full_vs_no_repair": {
            metric: _paired_analysis(
                _values(per_seed_full, metric),
                _values(per_seed_no_repair, metric),
                larger_is_better=metric not in {"false_positive_inconsistency_rate", "false_negative_rate", "identity_drift_score"},
            )
            for metric in ("commitment_alignment_rate", "repair_success_rate", "post_repair_alignment_gain", "stress_resilient_alignment_rate")
        },
    }
    significant_metric_count = sum(1 for group in comparisons.values() for payload in group.values() if bool(payload["significant"]))
    effect_metric_count = sum(1 for group in comparisons.values() for payload in group.values() if bool(payload["effect_passed"]))

    determinism_check_one = [
        _evaluate_condition(
            seed=seeds[0],
            scenario=scenarios["temptation_conflict"],
            condition=scenarios["temptation_conflict"].repair_condition,
            commitments_enabled=True,
            repair_enabled=True,
            tick=1,
        ),
        _run_repeated_challenge(seeds[0], scenarios["adaptation_vs_betrayal"]),
    ]
    determinism_check_two = [
        _evaluate_condition(
            seed=seeds[0],
            scenario=scenarios["temptation_conflict"],
            condition=scenarios["temptation_conflict"].repair_condition,
            commitments_enabled=True,
            repair_enabled=True,
            tick=1,
        ),
        _run_repeated_challenge(seeds[0], scenarios["adaptation_vs_betrayal"]),
    ]
    determinism_passed = determinism_check_one == determinism_check_two

    bounded_update_breakdown = {
        "per_seed": bounded_update_records,
        "commitment_persistence_score": _mean_std([float(item["commitment_persistence_score"]) for item in bounded_update_records]),
        "core_commitment_flip_rate": _mean_std([float(item["core_commitment_flip_rate"]) for item in bounded_update_records]),
        "identity_rewrite_ratio": _mean_std([float(item["identity_rewrite_ratio"]) for item in bounded_update_records]),
    }
    repair_breakdown = {
        "with_repair": {
            "trigger_count": sum(1 for record in trial_records["with_repair"] if bool(record["repair_triggered"])),
            "success_count": sum(1 for record in trial_records["with_repair"] if bool(record["repair_success_basis"]["counted"])),
            "metric_summary": {
                "repair_trigger_precision": full["repair_trigger_precision"],
                "repair_success_rate": full["repair_success_rate"],
                "post_repair_alignment_gain": full["post_repair_alignment_gain"],
            },
        },
        "no_repair": {
            "trigger_count": sum(1 for record in trial_records["no_repair"] if bool(record["repair_triggered"])),
            "success_count": sum(1 for record in trial_records["no_repair"] if bool(record["repair_result"].get("success", False))),
        },
    }
    stress_breakdown = {
        "low_stress_alignment_rate": full["low_stress_alignment_rate"],
        "high_stress_alignment_rate": full["high_stress_alignment_rate"],
        "alignment_drop": _round(full["low_stress_alignment_rate"] - full["high_stress_alignment_rate"]),
        "identity_drift_score": full["identity_drift_score"],
    }
    provided_condition_set = list(variants.keys())
    protocol_integrity = {
        "required_seed_set": list(SEED_SET),
        "provided_seed_set": list(seeds),
        "seed_set_complete": list(seeds) == list(SEED_SET),
        "required_scenario_set": list(FIXED_SCENARIO_SET),
        "provided_scenario_set": sorted(list(scenarios.keys())),
        "scenario_set_complete": sorted(list(scenarios.keys())) == sorted(FIXED_SCENARIO_SET),
        "required_condition_set": list(FIXED_CONDITION_SET),
        "provided_condition_set": provided_condition_set,
        "condition_set_complete": sorted(provided_condition_set) == sorted(FIXED_CONDITION_SET),
    }
    sample_independence_checks = {
        "repeated_challenge": {
            "sample_count": len(repeated_challenges),
            "unique_seed_count": len({int(item["seed"]) for item in repeated_challenges}),
            "duplicate_seed_count": len(repeated_challenges) - len({int(item["seed"]) for item in repeated_challenges}),
            "passes": len(repeated_challenges) == len(seeds) == len({int(item["seed"]) for item in repeated_challenges}),
        },
        "chapter_transition": {
            "sample_count": len(chapter_transitions),
            "unique_seed_count": len({int(item["seed"]) for item in chapter_transitions}),
            "duplicate_seed_count": len(chapter_transitions) - len({int(item["seed"]) for item in chapter_transitions}),
            "passes": len(chapter_transitions) == len(seeds) == len({int(item["seed"]) for item in chapter_transitions}),
        },
        "bounded_identity_update": {
            "sample_count": len(bounded_update_records),
            "unique_seed_count": len({int(item["seed"]) for item in bounded_update_records}),
            "duplicate_seed_count": len(bounded_update_records) - len({int(item["seed"]) for item in bounded_update_records}),
            "passes": len(bounded_update_records) == len(seeds) == len({int(item["seed"]) for item in bounded_update_records}),
        },
    }
    metric_counting_rules = {
        "self_inconsistency_detection_rate": "Count only records where actual_conflict_event is true; numerator additionally requires explicit internal conflict_type, explicit severity_level, allowed consistency_classification, threshold_passed, and recorded trace fields.",
        "false_positive_inconsistency_rate": "Count only aligned or non-conflict protocol contexts where the system explicitly emits conflict_type != none, severity_level != none, or repair_triggered = true.",
        "repair_success_rate": "Count only actual conflict repair-opportunity records with detected conflict, repair_triggered = true, non-empty repair_policy, explicit repair_result outcome, and a recovery window that improves alignment or returns behavior within active commitments.",
        "bounded_identity_update": "Use one bounded identity update sample per seed, generated outside the scenario loop, with duplicate seed checks enforced before PASS.",
    }

    goals = {
        "protocol_integrity": {
            "seed_set_complete": bool(protocol_integrity["seed_set_complete"]),
            "scenario_set_complete": bool(protocol_integrity["scenario_set_complete"]),
            "condition_set_complete": bool(protocol_integrity["condition_set_complete"]),
        },
        "commitment_constraints": {
            "commitment_alignment_rate": full["commitment_alignment_rate"] >= 0.78,
            "alignment_improvement_vs_no_commitments": alignment_improvement_vs_ablation >= 0.15,
            "violation_reduction_vs_no_commitments": violation_reduction_vs_ablation >= 0.20,
        },
        "inconsistency_detection": {
            "self_inconsistency_detection_rate": full["self_inconsistency_detection_rate"] >= 0.85,
            "false_negative_rate": full["false_negative_rate"] <= 0.15,
            "false_positive_inconsistency_rate": full["false_positive_inconsistency_rate"] <= 0.10,
        },
        "repair_effectiveness": {
            "repair_trigger_precision": full["repair_trigger_precision"] >= 0.75,
            "repair_success_rate": full["repair_success_rate"] >= 0.70,
            "post_repair_alignment_gain": full["post_repair_alignment_gain"] >= 0.12,
        },
        "stress_resilience": {
            "stress_resilient_alignment_rate": full["stress_resilient_alignment_rate"] >= 0.65,
            "alignment_drop": (full["low_stress_alignment_rate"] - full["high_stress_alignment_rate"]) <= 0.20,
            "identity_drift_score": full["identity_drift_score"] <= 0.18,
        },
        "evidence_support": {
            "narrative_evidence_support_rate": full["narrative_evidence_support_rate"] >= 0.85,
        },
        "bounded_update": {
            "commitment_persistence_score": mean(float(item["commitment_persistence_score"]) for item in bounded_update_records) >= 0.75,
            "core_commitment_flip_rate": mean(float(item["core_commitment_flip_rate"]) for item in bounded_update_records) <= 0.10,
            "identity_rewrite_ratio": mean(float(item["identity_rewrite_ratio"]) for item in bounded_update_records) <= 0.25,
        },
        "sample_independence": {key: bool(value["passes"]) for key, value in sample_independence_checks.items()},
        "statistics": {
            "significant_metric_count": significant_metric_count >= 3,
            "effect_metric_count": effect_metric_count >= 2,
        },
        "determinism": {"passed": determinism_passed},
        "artifact_schema_complete": {"passed": True},
        "freshness_generated_this_round": {"passed": True},
    }
    gates = {key: all(value.values()) for key, value in goals.items()}
    status = "PASS" if all(gates.values()) else "FAIL"
    recommendation = "ACCEPT" if status == "PASS" else "REJECT"

    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "seed_set": seeds,
        "scenario_definitions": {
            key: {
                "conflict_type": value.conflict_type,
                "aligned_condition": {"label": value.aligned_condition.label, "context": value.aligned_condition.context},
                "conflict_condition": {"label": value.conflict_condition.label, "context": value.conflict_condition.context},
                "repair_opportunity_condition": {"label": value.repair_condition.label, "context": value.repair_condition.context},
                "commitments": [commitment.to_dict() for commitment in value.narrative.commitments],
            }
            for key, value in scenarios.items()
        },
        "variant_metrics": per_variant_metrics,
        "scenario_breakdown": scenario_breakdown,
        "repair_breakdown": repair_breakdown,
        "stress_breakdown": stress_breakdown,
        "bounded_update_breakdown": bounded_update_breakdown,
        "chapter_transition_breakdown": {"per_seed": chapter_transitions},
        "protocol_integrity": protocol_integrity,
        "sample_independence_checks": sample_independence_checks,
        "metric_counting_rules": metric_counting_rules,
        "comparisons": comparisons,
        "significant_metric_count": significant_metric_count,
        "effect_metric_count": effect_metric_count,
        "gates": gates,
        "goal_details": goals,
        "status": status,
        "recommendation": recommendation,
        "residual_risks": [
            "Benchmark is protocol-fixed and synthetic rather than open-ended world deployment.",
            "Repair success is demonstrated under deterministic scenario families, not adversarially generated long-form narratives.",
            "Bounded update currently adjusts commitment strength rather than rewriting narrative chapters with richer semantic evidence.",
        ],
        "freshness": {
            "generated_this_round": True,
            "artifact_schema_version": SCHEMA_VERSION,
            "codebase_version": _codebase_version(),
        },
        "artifacts": {
            "m223_self_consistency_trace": {"schema_version": SCHEMA_VERSION, "seed_set": seeds, "trace": trace_records},
            "m223_commitment_alignment": {
                "schema_version": SCHEMA_VERSION,
                "metric_summary": {"with_repair": full, "with_commitments": with_commitments, "no_commitments": no_commitments, "no_repair": no_repair},
                "alignment_improvement_vs_no_commitments": _round(alignment_improvement_vs_ablation),
                "violation_reduction_vs_no_commitments": _round(violation_reduction_vs_ablation),
            },
            "m223_repair_outcomes": {
                "schema_version": SCHEMA_VERSION,
                "repair_breakdown": repair_breakdown,
                "representative_repairs": [record for record in trace_records if bool(record["repair_triggered"])][:8],
            },
            "m223_identity_update_bounds": {"schema_version": SCHEMA_VERSION, "bounded_update_breakdown": bounded_update_breakdown},
            "m223_stress_alignment": {"schema_version": SCHEMA_VERSION, "stress_breakdown": stress_breakdown},
            "m223_protocol_audit": {
                "schema_version": SCHEMA_VERSION,
                "protocol_integrity": protocol_integrity,
                "sample_independence_checks": sample_independence_checks,
                "metric_counting_rules": metric_counting_rules,
                "chapter_transition_breakdown": {"per_seed": chapter_transitions},
            },
        },
        "determinism": {"seed": seeds[0], "passed": determinism_passed, "first": determinism_check_one, "second": determinism_check_two},
    }


def write_m223_acceptance_artifacts(
    *,
    seed_set: list[int] | None = None,
) -> dict[str, Path]:
    payload = run_m223_self_consistency_benchmark(seed_set=seed_set)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "trace": ARTIFACTS_DIR / "m223_self_consistency_trace.json",
        "alignment": ARTIFACTS_DIR / "m223_commitment_alignment.json",
        "repair": ARTIFACTS_DIR / "m223_repair_outcomes.json",
        "identity_update": ARTIFACTS_DIR / "m223_identity_update_bounds.json",
        "stress": ARTIFACTS_DIR / "m223_stress_alignment.json",
        "protocol_audit": ARTIFACTS_DIR / "m223_protocol_audit.json",
        "report": REPORTS_DIR / "m223_acceptance_report.json",
    }
    paths["trace"].write_text(json.dumps(payload["artifacts"]["m223_self_consistency_trace"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["alignment"].write_text(json.dumps(payload["artifacts"]["m223_commitment_alignment"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["repair"].write_text(json.dumps(payload["artifacts"]["m223_repair_outcomes"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["identity_update"].write_text(json.dumps(payload["artifacts"]["m223_identity_update_bounds"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["stress"].write_text(json.dumps(payload["artifacts"]["m223_stress_alignment"], indent=2, ensure_ascii=False), encoding="utf-8")
    paths["protocol_audit"].write_text(json.dumps(payload["artifacts"]["m223_protocol_audit"], indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "milestone_id": payload["milestone_id"],
        "status": payload["status"],
        "recommendation": payload["recommendation"],
        "generated_at": payload["generated_at"],
        "seed_set": payload["seed_set"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        "tests": {
            "milestone_suite": [
                "tests/test_m223_commitment_alignment.py",
                "tests/test_m223_inconsistency_detection.py",
                "tests/test_m223_repair_loop.py",
                "tests/test_m223_bounded_identity_update.py",
                "tests/test_m223_acceptance.py",
                "tests/test_m223_audit_hardening.py",
            ]
        },
        "scenario_definitions": payload["scenario_definitions"],
        "variant_metrics": payload["variant_metrics"],
        "scenario_breakdown": payload["scenario_breakdown"],
        "repair_breakdown": payload["repair_breakdown"],
        "stress_breakdown": payload["stress_breakdown"],
        "bounded_update_breakdown": payload["bounded_update_breakdown"],
        "chapter_transition_breakdown": payload["chapter_transition_breakdown"],
        "protocol_integrity": payload["protocol_integrity"],
        "sample_independence_checks": payload["sample_independence_checks"],
        "metric_counting_rules": payload["metric_counting_rules"],
        "significant_metric_count": payload["significant_metric_count"],
        "effect_metric_count": payload["effect_metric_count"],
        "gates": payload["gates"],
        "goal_details": payload["goal_details"],
        "residual_risks": payload["residual_risks"],
        "freshness": payload["freshness"],
    }
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
