from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Mapping

from ...agent import SegmentAgent
from ..conversation_loop import run_conversation
from ..lifecycle import ImplantationConfig, implant_personality
from ..observer import DialogueObserver
from ..world import DialogueWorld
from .act_classifier import DialogueActClassifier, validate_act_classifier
from .act_classifier_eval_sets import DEFAULT_CLASSIFIER_EVAL_SAMPLES
from .baselines import (
    build_population_average_agent,
    clone_agent_template,
    create_average_agent,
    create_default_agent,
    create_wrong_agent,
    select_wrong_users,
)
from .metrics import (
    SimilarityResult,
    agent_state_similarity,
    behavioral_similarity,
    personality_similarity,
    semantic_similarity,
    stylistic_similarity,
    surface_similarity,
)
from .splitter import SplitStrategy, split_user_data


@dataclass(slots=True)
class ValidationConfig:
    """M5.4 validation options.

    ``skip_population_average_implant``: when True, Baseline C uses profile-only
    :func:`create_average_agent` instead of one full implant per user to build a population
    average agent (much faster; for unit tests). Production runs should use False.
    """

    strategies: list[SplitStrategy]
    train_ratio: float = 0.70
    seed: int = 42
    min_holdout_sessions: int = 3
    min_users: int = 10
    pilot_user_count: int = 5
    pilot_sd_threshold: float = 0.04
    min_users_if_high_sd: int = 15
    implantation_config: ImplantationConfig = field(default_factory=ImplantationConfig)
    wrong_user_uids: list[int] | None = None
    skip_population_average_implant: bool = False


@dataclass(slots=True)
class ValidationReport:
    user_uid: int
    per_strategy: dict[str, dict]
    aggregate: dict[str, object]
    conclusion: str


def _dataset_subset(user_dataset: Mapping[str, object], sessions: list[dict]) -> dict:
    return {
        "uid": int(user_dataset.get("uid", 0)),
        "profile": dict(user_dataset.get("profile", {})) if isinstance(user_dataset.get("profile"), Mapping) else {},
        "sessions": list(sessions),
    }


def _pair_partner_and_real_replies(
    session: Mapping[str, object],
    user_uid: int,
) -> tuple[list[str], list[str]]:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return [], []
    interlocutor_turns: list[str] = []
    real_replies: list[str] = []
    pending_idx: int | None = None
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        body = str(turn.get("body", "")).strip()
        if not body:
            continue
        sender_uid = int(turn.get("sender_uid", 0))
        if sender_uid != user_uid:
            interlocutor_turns.append(body)
            real_replies.append("")
            pending_idx = len(real_replies) - 1
        elif pending_idx is not None and not real_replies[pending_idx]:
            real_replies[pending_idx] = body
            pending_idx = None
    paired_interlocutor: list[str] = []
    paired_real: list[str] = []
    for partner_line, real_line in zip(interlocutor_turns, real_replies):
        if real_line:
            paired_interlocutor.append(partner_line)
            paired_real.append(real_line)
    return paired_interlocutor, paired_real


def _session_partner_uid(user_uid: int, session: Mapping[str, object]) -> int:
    uid_a = int(session.get("uid_a", user_uid))
    uid_b = int(session.get("uid_b", user_uid))
    if uid_a == user_uid:
        return uid_b
    return uid_a


def _generate_from_sessions(
    agent: SegmentAgent,
    holdout_sessions: list[dict],
    *,
    user_uid: int,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    observer = DialogueObserver()
    generated_texts: list[str] = []
    real_texts: list[str] = []
    for idx, session in enumerate(holdout_sessions):
        interlocutor_turns, real_replies = _pair_partner_and_real_replies(session, user_uid)
        if not interlocutor_turns or not real_replies:
            continue
        turns = run_conversation(
            agent,
            interlocutor_turns,
            observer=observer,
            partner_uid=_session_partner_uid(user_uid, session),
            session_id=str(session.get("session_id", f"holdout:{idx}")),
            master_seed=int(seed) + idx,
        )
        for turn, real in zip(turns, real_replies):
            generated_texts.append(str(turn.text))
            real_texts.append(str(real))
    classifier = DialogueActClassifier()
    preds_g = classifier.predict_batch(generated_texts)
    preds_r = classifier.predict_batch(real_texts)
    gen_11 = [item.label_11 for item in preds_g]
    real_11 = [item.label_11 for item in preds_r]
    gen_3 = [item.label_3 for item in preds_g]
    real_3 = [item.label_3 for item in preds_r]
    return generated_texts, real_texts, gen_11, real_11, gen_3, real_3


def _metrics_bundle(
    generated_texts: list[str],
    real_texts: list[str],
    gen_11: list[str],
    real_11: list[str],
    gen_3: list[str],
    real_3: list[str],
) -> dict[str, SimilarityResult]:
    return {
        "surface_similarity": surface_similarity(generated_texts, real_texts),
        "semantic_similarity": semantic_similarity(generated_texts, real_texts),
        "stylistic_similarity": stylistic_similarity(generated_texts, real_texts),
        "personality_similarity": personality_similarity(generated_texts, real_texts),
        "behavioral_similarity_strategy": behavioral_similarity(gen_3, real_3, granularity="strategy"),
        "behavioral_similarity_action11": behavioral_similarity(gen_11, real_11, granularity="action11"),
    }


def _result_values(results: dict[str, SimilarityResult]) -> dict[str, float]:
    return {name: float(item.value) for name, item in results.items()}


def run_validation(
    user_dataset: dict,
    config: ValidationConfig,
    all_user_profiles: list[dict] | None = None,
    *,
    population_average_template: SegmentAgent | None = None,
) -> ValidationReport:
    user_uid = int(user_dataset.get("uid", 0))
    all_profiles = list(all_user_profiles or [])
    full_agent = SegmentAgent()
    full_world = DialogueWorld(user_dataset, DialogueObserver(), seed=int(config.seed))
    implant_personality(full_agent, full_world, config.implantation_config)

    per_strategy: dict[str, dict] = {}
    strategy_personality_means: dict[str, float] = {}
    strategy_baseline_a_means: dict[str, float] = {}
    strategy_baseline_b_means: dict[str, float] = {}
    strategy_baseline_c_means: dict[str, float] = {}
    strategy_behavioral_p_means: dict[str, float] = {}
    strategy_behavioral_c_means: dict[str, float] = {}

    classifier_eval = validate_act_classifier(DEFAULT_CLASSIFIER_EVAL_SAMPLES)
    behavioral_hard_enabled = bool(classifier_eval.get("passed_3class_gate", False))

    for strategy in config.strategies:
        split = split_user_data(
            user_dataset,
            strategy,
            train_ratio=float(config.train_ratio),
            seed=int(config.seed),
        )
        strategy_key = strategy.value
        if len(split.holdout_sessions) < int(config.min_holdout_sessions):
            per_strategy[strategy_key] = {
                "skipped": True,
                "reason": "insufficient_holdout_sessions",
                "split_metadata": split.split_metadata,
            }
            continue
        train_dataset = _dataset_subset(user_dataset, split.train_sessions)
        train_agent = SegmentAgent()
        train_world = DialogueWorld(train_dataset, DialogueObserver(), seed=int(config.seed))
        implant_personality(train_agent, train_world, config.implantation_config)
        generated, real, g11, r11, g3, r3 = _generate_from_sessions(
            train_agent,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed),
        )
        personality_metrics = _metrics_bundle(generated, real, g11, r11, g3, r3)
        personality_metrics["agent_state_similarity"] = agent_state_similarity(train_agent, full_agent)

        baseline_a_agent = create_default_agent(seed=int(config.seed) + 101)
        a_gen, a_real, ag11, ar11, ag3, ar3 = _generate_from_sessions(
            baseline_a_agent,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed) + 101,
        )
        baseline_a_metrics = _metrics_bundle(a_gen, a_real, ag11, ar11, ag3, ar3)

        chosen_wrong = select_wrong_users(
            user_dataset,
            [item for item in all_profiles if int(item.get("uid", -1)) != user_uid],
            k=3,
            seed=int(config.seed) + 202,
        )
        b_runs: list[dict[str, float]] = []
        for wrong in chosen_wrong:
            wrong_agent = create_wrong_agent(wrong, config.implantation_config, seed=int(config.seed) + int(wrong.get("_wrong_user_uid", 0)))
            b_gen, b_real, bg11, br11, bg3, br3 = _generate_from_sessions(
                wrong_agent,
                split.holdout_sessions,
                user_uid=user_uid,
                seed=int(config.seed) + 303,
            )
            b_runs.append(_result_values(_metrics_bundle(b_gen, b_real, bg11, br11, bg3, br3)))
        baseline_b_values: dict[str, float] = {}
        if b_runs:
            metric_names = sorted(set().union(*[set(item.keys()) for item in b_runs]))
            for name in metric_names:
                baseline_b_values[name] = float(mean([run.get(name, 0.0) for run in b_runs]))

        strat_idx = config.strategies.index(strategy)
        strat_seed = int(config.seed) + 404 + strat_idx * 31
        if population_average_template is not None:
            baseline_c_agent = clone_agent_template(population_average_template, seed=strat_seed)
        else:
            baseline_c_agent = create_average_agent(all_profiles, seed=strat_seed)
        c_gen, c_real, cg11, cr11, cg3, cr3 = _generate_from_sessions(
            baseline_c_agent,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed) + 404,
        )
        baseline_c_metrics = _metrics_bundle(c_gen, c_real, cg11, cr11, cg3, cr3)

        personality_values = _result_values(personality_metrics)
        baseline_a_values = _result_values(baseline_a_metrics)
        baseline_c_values = _result_values(baseline_c_metrics)
        per_strategy[strategy_key] = {
            "split_metadata": split.split_metadata,
            "metrics_without_baselines": ["agent_state_similarity"],
            "behavioral_hard_metric_enabled": behavioral_hard_enabled,
            "behavioral_labeling": "dialogue_act_classifier_11class_and_3strategy",
            "classifier_validation": classifier_eval,
            "personality_metrics": personality_values,
            "baseline_a_metrics": baseline_a_values,
            "baseline_b_metrics": baseline_b_values,
            "baseline_c_metrics": baseline_c_values,
            "baseline_b_selected_wrong_uids": [int(item.get("_wrong_user_uid", -1)) for item in chosen_wrong],
        }
        strategy_personality_means[strategy_key] = float(personality_values.get("semantic_similarity", 0.0))
        strategy_baseline_a_means[strategy_key] = float(baseline_a_values.get("semantic_similarity", 0.0))
        strategy_baseline_b_means[strategy_key] = float(baseline_b_values.get("semantic_similarity", 0.0))
        strategy_baseline_c_means[strategy_key] = float(baseline_c_values.get("semantic_similarity", 0.0))
        strategy_behavioral_p_means[strategy_key] = float(
            personality_values.get("behavioral_similarity_strategy", 0.0)
        )
        strategy_behavioral_c_means[strategy_key] = float(
            baseline_c_values.get("behavioral_similarity_strategy", 0.0)
        )

    valid = [value for value in per_strategy.values() if not value.get("skipped", False)]
    conclusion = "skipped_all_strategies" if not valid else "completed"
    aggregate = {
        "pipeline_status": conclusion,
        "metric_version": "m54_v3",
        "behavioral_labeling": "dialogue_act_classifier_11class_and_3strategy",
        "strategy_count": int(len(valid)),
        "semantic_personality_mean": float(mean(strategy_personality_means.values())) if strategy_personality_means else 0.0,
        "semantic_baseline_a_mean": float(mean(strategy_baseline_a_means.values())) if strategy_baseline_a_means else 0.0,
        "semantic_baseline_b_mean": float(mean(strategy_baseline_b_means.values())) if strategy_baseline_b_means else 0.0,
        "semantic_baseline_c_mean": float(mean(strategy_baseline_c_means.values())) if strategy_baseline_c_means else 0.0,
        "behavioral_personality_mean": float(mean(strategy_behavioral_p_means.values()))
        if strategy_behavioral_p_means
        else 0.0,
        "behavioral_baseline_c_mean": float(mean(strategy_behavioral_c_means.values()))
        if strategy_behavioral_c_means
        else 0.0,
        "behavioral_hard_metric_enabled": behavioral_hard_enabled,
    }
    return ValidationReport(
        user_uid=user_uid,
        per_strategy=per_strategy,
        aggregate=aggregate,
        conclusion=conclusion,
    )


def run_pilot_validation(
    user_datasets: list[dict],
    config: ValidationConfig,
    *,
    population_average_template: SegmentAgent | None = None,
) -> dict[str, object]:
    if not user_datasets:
        return {
            "pilot_user_count": 0,
            "semantic_diff_mean": 0.0,
            "semantic_diff_sd": 0.0,
            "behavioral_diff_sd": 0.0,
            "sd_threshold": float(config.pilot_sd_threshold),
            "suggested_min_users": int(config.min_users),
            "escalated": False,
            "pilot_metrics_used": [],
        }
    pilot_count = min(len(user_datasets), max(3, int(config.pilot_user_count)))
    pilot_users = user_datasets[:pilot_count]
    if population_average_template is not None:
        pop_template = population_average_template
    elif config.skip_population_average_implant:
        pop_template = None
    else:
        pop_template = build_population_average_agent(
            user_datasets, config.implantation_config, seed=int(config.seed)
        )
    semantic_diffs: list[float] = []
    behavioral_diffs: list[float] = []
    classifier_gate_any = False
    for item in pilot_users:
        report = run_validation(
            item,
            config,
            all_user_profiles=user_datasets,
            population_average_template=pop_template,
        )
        personality = float(report.aggregate.get("semantic_personality_mean", 0.0))
        baseline_a = float(report.aggregate.get("semantic_baseline_a_mean", 0.0))
        semantic_diffs.append(personality - baseline_a)
        beh_p = float(report.aggregate.get("behavioral_personality_mean", 0.0))
        beh_c = float(report.aggregate.get("behavioral_baseline_c_mean", 0.0))
        behavioral_diffs.append(beh_p - beh_c)
        if report.aggregate.get("behavioral_hard_metric_enabled"):
            classifier_gate_any = True
    sem_sd = pstdev(semantic_diffs) if len(semantic_diffs) > 1 else 0.0
    beh_sd = pstdev(behavioral_diffs) if len(behavioral_diffs) > 1 else 0.0
    sem_escalate = sem_sd > float(config.pilot_sd_threshold)
    beh_escalate = classifier_gate_any and (beh_sd > float(config.pilot_sd_threshold))
    suggested_sem = (
        int(config.min_users_if_high_sd) if sem_escalate else int(config.min_users)
    )
    suggested_beh = (
        int(config.min_users_if_high_sd) if beh_escalate else int(config.min_users)
    )
    suggested = max(suggested_sem, suggested_beh)
    escalated = bool(sem_escalate or beh_escalate)
    pilot_metrics_used = ["semantic_vs_baseline_a_mean_diff"]
    if classifier_gate_any:
        pilot_metrics_used.append("behavioral_vs_baseline_c_mean_diff_when_classifier_hard_enabled")
    return {
        "pilot_user_count": int(pilot_count),
        "semantic_diff_mean": round(float(mean(semantic_diffs)) if semantic_diffs else 0.0, 6),
        "semantic_diff_sd": round(float(sem_sd), 6),
        "behavioral_diff_sd": round(float(beh_sd), 6),
        "sd_threshold": float(config.pilot_sd_threshold),
        "suggested_min_users": int(suggested),
        "escalated": escalated,
        "semantic_escalated": bool(sem_escalate),
        "behavioral_escalated": bool(beh_escalate),
        "pilot_metrics_used": pilot_metrics_used,
    }


def run_batch_validation(
    user_datasets: list[dict],
    config: ValidationConfig,
) -> list[ValidationReport]:
    if config.skip_population_average_implant:
        pop_template = None
    else:
        pop_template = build_population_average_agent(
            user_datasets, config.implantation_config, seed=int(config.seed)
        )
    pilot = run_pilot_validation(
        user_datasets, config, population_average_template=pop_template
    )
    required_users = int(pilot.get("suggested_min_users", config.min_users))
    if len(user_datasets) < required_users:
        raise ValueError(
            f"insufficient users for validation: have={len(user_datasets)}, required={required_users}"
        )
    reports: list[ValidationReport] = []
    for item in user_datasets:
        report = run_validation(
            item,
            config,
            all_user_profiles=user_datasets,
            population_average_template=pop_template,
        )
        report.aggregate["pilot"] = pilot
        reports.append(report)
    return reports

