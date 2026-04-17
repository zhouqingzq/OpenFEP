from __future__ import annotations

from dataclasses import dataclass, field
import os
from statistics import mean, pstdev
from typing import Mapping

from ...agent import SegmentAgent
from ..conversation_loop import run_conversation
from ..lifecycle import ImplantationConfig, implant_personality
from ..observer import DialogueObserver
from ..world import DialogueWorld
from ..actions import DIALOGUE_ACTION_STRATEGY_MAP
from .act_classifier import DialogueActClassifier, validate_act_classifier
from .act_classifier_eval_sets import (
    DEFAULT_CLASSIFIER_GATE_SAMPLES,
    DEFAULT_CLASSIFIER_TRAIN_SAMPLES,
)
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
    semantic_pair_scores,
    semantic_similarity,
    stylistic_similarity,
    surface_similarity,
)
from .splitter import SplitStrategy, split_user_data
from .surface_profile import (
    DialogueSurfaceProfile,
    attach_surface_profile,
    average_surface_profiles,
    build_surface_profile,
)


def _progress(message: str) -> None:
    if os.environ.get("SEGMENTUM_M54_PROGRESS"):
        print(message, flush=True)


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
    classifier_train_samples: list[dict[str, str]] = field(default_factory=list)
    classifier_gate_samples: list[dict[str, str]] = field(default_factory=list)
    classifier_dataset_origin: str = "missing_formal_labels"
    formal: bool = False
    diagnostic_trace: bool = False


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
    classifier: DialogueActClassifier,
    diagnostic_trace: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[dict[str, object]]]:
    observer = DialogueObserver()
    generated_texts: list[str] = []
    real_texts: list[str] = []
    gen_11: list[str] = []
    gen_3: list[str] = []
    trace_rows: list[dict[str, object]] = []
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
            action = str(turn.action or "ask_question")
            strategy = str(turn.strategy or DIALOGUE_ACTION_STRATEGY_MAP.get(action, "explore"))
            generated_texts.append(str(turn.text))
            real_texts.append(str(real))
            gen_11.append(action)
            gen_3.append(strategy)
            if diagnostic_trace:
                diagnostics = turn.diagnostics
                chosen = getattr(diagnostics, "chosen", None) if diagnostics is not None else None
                priors = getattr(agent.self_model, "narrative_priors", None)
                trace_rows.append(
                    {
                        "session_id": str(session.get("session_id", f"holdout:{idx}")),
                        "partner_uid": int(_session_partner_uid(user_uid, session)),
                        "turn_index": int(turn.turn_index),
                        "generated_text": str(turn.text),
                        "real_text": str(real),
                        "generated_chars": int(len(str(turn.text))),
                        "real_chars": int(len(str(real))),
                        "action": action,
                        "strategy": strategy,
                        "dominant_component": str(getattr(chosen, "dominant_component", "")),
                        "slow_traits": agent.slow_variable_learner.state.traits.to_dict(),
                        "narrative_priors": priors.to_dict() if hasattr(priors, "to_dict") else {},
                    }
                )
    preds_r = classifier.predict_batch(real_texts)
    real_11 = [item.label_11 for item in preds_r]
    real_3 = [item.label_3 for item in preds_r]
    return generated_texts, real_texts, gen_11, real_11, gen_3, real_3, trace_rows


def _unpack_generation_result(
    result: object,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[dict[str, object]]]:
    values = tuple(result) if isinstance(result, tuple) else tuple()  # type: ignore[arg-type]
    if len(values) == 6:
        generated, real, g11, r11, g3, r3 = values
        return list(generated), list(real), list(g11), list(r11), list(g3), list(r3), []
    if len(values) == 7:
        generated, real, g11, r11, g3, r3, trace = values
        return list(generated), list(real), list(g11), list(r11), list(g3), list(r3), list(trace)
    raise ValueError("generation result must contain 6 or 7 fields")


def _call_generate_from_sessions(
    agent: SegmentAgent,
    holdout_sessions: list[dict],
    *,
    user_uid: int,
    seed: int,
    classifier: DialogueActClassifier,
    diagnostic_trace: bool,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[dict[str, object]]]:
    try:
        result = _generate_from_sessions(
            agent,
            holdout_sessions,
            user_uid=user_uid,
            seed=seed,
            classifier=classifier,
            diagnostic_trace=diagnostic_trace,
        )
    except TypeError as exc:
        if "diagnostic_trace" not in str(exc):
            raise
        result = _generate_from_sessions(
            agent,
            holdout_sessions,
            user_uid=user_uid,
            seed=seed,
            classifier=classifier,
        )
    return _unpack_generation_result(result)


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


def _result_details(results: dict[str, SimilarityResult]) -> dict[str, dict[str, object]]:
    return {name: dict(item.details) for name, item in results.items()}


def _surface_profile_summary(profile: DialogueSurfaceProfile) -> dict[str, object]:
    payload = profile.to_dict()
    return {
        "source": payload.get("source"),
        "reply_count": payload.get("reply_count"),
        "avg_reply_chars": payload.get("avg_reply_chars"),
        "median_reply_chars": payload.get("median_reply_chars"),
        "ultra_short_ratio": payload.get("ultra_short_ratio"),
        "top_tokens": list(payload.get("top_tokens", []))[:8],
        "opening_phrases": list(payload.get("opening_phrases", []))[:5],
        "strategy_counts": dict(payload.get("strategy_counts", {})),
    }


def _semantic_trace_rows(
    *,
    strategy_key: str,
    personality_trace: list[dict[str, object]],
    baseline_a_trace: list[dict[str, object]],
    baseline_c_trace: list[dict[str, object]],
    personality_scores: list[float],
    baseline_a_scores: list[float],
    baseline_c_scores: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = min(
        len(personality_trace),
        len(baseline_a_trace),
        len(baseline_c_trace),
        len(personality_scores),
        len(baseline_a_scores),
        len(baseline_c_scores),
    )
    for idx in range(total):
        p = dict(personality_trace[idx])
        a = dict(baseline_a_trace[idx])
        c = dict(baseline_c_trace[idx])
        rows.append(
            {
                "strategy": strategy_key,
                "pair_index": idx,
                "session_id": p.get("session_id"),
                "partner_uid": p.get("partner_uid"),
                "turn_index": p.get("turn_index"),
                "real_text": p.get("real_text"),
                "personality_text": p.get("generated_text"),
                "baseline_a_text": a.get("generated_text"),
                "baseline_c_text": c.get("generated_text"),
                "personality_semantic_pair_score": round(float(personality_scores[idx]), 6),
                "baseline_a_semantic_pair_score": round(float(baseline_a_scores[idx]), 6),
                "baseline_c_semantic_pair_score": round(float(baseline_c_scores[idx]), 6),
                "personality_vs_a_pair_delta": round(
                    float(personality_scores[idx]) - float(baseline_a_scores[idx]),
                    6,
                ),
                "personality_vs_c_pair_delta": round(
                    float(personality_scores[idx]) - float(baseline_c_scores[idx]),
                    6,
                ),
                "personality_action": p.get("action"),
                "personality_strategy": p.get("strategy"),
                "baseline_a_action": a.get("action"),
                "baseline_a_strategy": a.get("strategy"),
                "baseline_c_action": c.get("action"),
                "baseline_c_strategy": c.get("strategy"),
                "dominant_component": p.get("dominant_component"),
                "reply_length_bucket": _reply_length_bucket(int(p.get("generated_chars", 0) or 0)),
                "slow_traits": p.get("slow_traits", {}),
                "narrative_priors": p.get("narrative_priors", {}),
            }
        )
    return rows


def _reply_length_bucket(length: int) -> str:
    if length <= 3:
        return "ultra_short"
    if length <= 12:
        return "short"
    if length <= 40:
        return "medium"
    return "long"


def run_validation(
    user_dataset: dict,
    config: ValidationConfig,
    all_user_profiles: list[dict] | None = None,
    *,
    population_average_template: SegmentAgent | None = None,
    wrong_agent_cache: dict[int, tuple[SegmentAgent, DialogueSurfaceProfile]] | None = None,
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

    classifier_train = list(config.classifier_train_samples or DEFAULT_CLASSIFIER_TRAIN_SAMPLES)
    classifier_gate = list(config.classifier_gate_samples or DEFAULT_CLASSIFIER_GATE_SAMPLES)
    classifier = DialogueActClassifier(classifier_train or None)
    classifier_eval = validate_act_classifier(
        train_samples=classifier_train,
        gate_samples=classifier_gate,
        classifier=classifier,
        dataset_origin=str(config.classifier_dataset_origin),
    )
    behavioral_hard_enabled = bool(classifier_eval.get("passed_3class_gate", False))
    population_surface_profile = average_surface_profiles(
        (
            build_surface_profile(item, classifier=None, source=f"population:{int(item.get('uid', -1))}")
            for item in all_profiles
            if isinstance(item, Mapping)
        ),
        source="population_average",
    )

    for strategy in config.strategies:
        split = split_user_data(
            user_dataset,
            strategy,
            train_ratio=float(config.train_ratio),
            seed=int(config.seed),
        )
        strategy_key = strategy.value
        if strategy == SplitStrategy.TOPIC and bool(split.split_metadata.get("topic_split_not_applicable")):
            per_strategy[strategy_key] = {
                "skipped": True,
                "reason": "topic_split_not_applicable",
                "eligible_for_hard_gate": False,
                "split_metadata": split.split_metadata,
            }
            continue
        if len(split.holdout_sessions) < int(config.min_holdout_sessions):
            per_strategy[strategy_key] = {
                "skipped": True,
                "reason": "insufficient_holdout_sessions",
                "eligible_for_hard_gate": False,
                "split_metadata": split.split_metadata,
            }
            continue
        train_dataset = _dataset_subset(user_dataset, split.train_sessions)
        train_surface_profile = build_surface_profile(
            train_dataset,
            classifier=classifier,
            source=f"{strategy_key}:train",
        )
        train_agent = SegmentAgent()
        train_world = DialogueWorld(train_dataset, DialogueObserver(), seed=int(config.seed))
        implant_personality(train_agent, train_world, config.implantation_config)
        agent_state_result = agent_state_similarity(train_agent, full_agent)
        agent_state_result.details["agent_state_measured_before_holdout_generation"] = True
        train_agent_for_generation = clone_agent_template(
            train_agent,
            seed=int(config.seed) + 17 + config.strategies.index(strategy),
        )
        attach_surface_profile(train_agent_for_generation, train_surface_profile)
        generated, real, g11, r11, g3, r3, p_trace = _call_generate_from_sessions(
            train_agent_for_generation,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed),
            classifier=classifier,
            diagnostic_trace=bool(config.diagnostic_trace),
        )
        personality_metrics = _metrics_bundle(generated, real, g11, r11, g3, r3)
        personality_metrics["agent_state_similarity"] = agent_state_result

        baseline_a_agent = create_default_agent(seed=int(config.seed) + 101)
        a_gen, a_real, ag11, ar11, ag3, ar3, a_trace = _call_generate_from_sessions(
            baseline_a_agent,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed) + 101,
            classifier=classifier,
            diagnostic_trace=bool(config.diagnostic_trace),
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
            wrong_uid = int(wrong.get("_wrong_user_uid", wrong.get("uid", 0)))
            cache_key = int(wrong_uid)
            cached_wrong = wrong_agent_cache.get(cache_key) if wrong_agent_cache is not None else None
            if cached_wrong is None:
                wrong_template = create_wrong_agent(
                    wrong,
                    config.implantation_config,
                    seed=int(config.seed) + wrong_uid,
                )
                wrong_profile = build_surface_profile(wrong, classifier=classifier, source="wrong_user")
                if wrong_agent_cache is not None:
                    wrong_agent_cache[cache_key] = (wrong_template, wrong_profile)
            else:
                wrong_template, wrong_profile = cached_wrong
            wrong_agent = clone_agent_template(
                wrong_template,
                seed=int(config.seed) + 303 + wrong_uid + config.strategies.index(strategy),
            )
            attach_surface_profile(wrong_agent, wrong_profile)
            b_gen, b_real, bg11, br11, bg3, br3, _b_trace = _call_generate_from_sessions(
                wrong_agent,
                split.holdout_sessions,
                user_uid=user_uid,
                seed=int(config.seed) + 303,
                classifier=classifier,
                diagnostic_trace=False,
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
        attach_surface_profile(baseline_c_agent, population_surface_profile)
        c_gen, c_real, cg11, cr11, cg3, cr3, c_trace = _call_generate_from_sessions(
            baseline_c_agent,
            split.holdout_sessions,
            user_uid=user_uid,
            seed=int(config.seed) + 404,
            classifier=classifier,
            diagnostic_trace=bool(config.diagnostic_trace),
        )
        baseline_c_metrics = _metrics_bundle(c_gen, c_real, cg11, cr11, cg3, cr3)

        personality_values = _result_values(personality_metrics)
        baseline_a_values = _result_values(baseline_a_metrics)
        baseline_c_values = _result_values(baseline_c_metrics)
        diagnostic_rows: list[dict[str, object]] = []
        if config.diagnostic_trace:
            diagnostic_rows = _semantic_trace_rows(
                strategy_key=strategy_key,
                personality_trace=p_trace,
                baseline_a_trace=a_trace,
                baseline_c_trace=c_trace,
                personality_scores=semantic_pair_scores(generated, real),
                baseline_a_scores=semantic_pair_scores(a_gen, a_real),
                baseline_c_scores=semantic_pair_scores(c_gen, c_real),
            )
        per_strategy[strategy_key] = {
            "skipped": False,
            "reason": None,
            "eligible_for_hard_gate": True,
            "split_metadata": split.split_metadata,
            "metrics_without_baselines": ["agent_state_similarity"],
            "behavioral_hard_metric_enabled": behavioral_hard_enabled,
            "behavioral_labeling": "generated_action_direct_real_reply_classifier",
            "classifier_validation": classifier_eval,
            "personality_metrics": personality_values,
            "personality_metric_details": _result_details(personality_metrics),
            "baseline_a_metrics": baseline_a_values,
            "baseline_a_metric_details": _result_details(baseline_a_metrics),
            "baseline_b_metrics": baseline_b_values,
            "baseline_c_metrics": baseline_c_values,
            "baseline_c_metric_details": _result_details(baseline_c_metrics),
            "baseline_b_selected_wrong_uids": [int(item.get("_wrong_user_uid", -1)) for item in chosen_wrong],
            "surface_profile": _surface_profile_summary(train_surface_profile),
            "baseline_a_surface_profile": {"source": "empty", "reply_count": 0},
            "baseline_c_surface_profile": _surface_profile_summary(population_surface_profile),
            "diagnostic_trace": diagnostic_rows,
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
        "behavioral_labeling": "generated_action_direct_real_reply_classifier",
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
        "classifier_validation": classifier_eval,
        "skip_population_average_implant": bool(config.skip_population_average_implant),
        "requested_strategies": [item.value for item in config.strategies],
        "formal_requested": bool(config.formal),
        "diagnostic_trace_enabled": bool(config.diagnostic_trace),
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
    wrong_agent_cache: dict[int, tuple[SegmentAgent, DialogueSurfaceProfile]] | None = None,
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
            "required_users": int(config.min_users),
        }
    pilot_count = min(len(user_datasets), max(3, int(config.pilot_user_count)))
    _progress(f"running pilot validation on {pilot_count} users...")
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
    for idx, item in enumerate(pilot_users):
        _progress(f"  pilot user {idx + 1}/{pilot_count} (uid={int(item.get('uid', -1))})")
        report = run_validation(
            item,
            config,
            all_user_profiles=user_datasets,
            population_average_template=pop_template,
            wrong_agent_cache=wrong_agent_cache,
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
        "required_users": int(suggested),
        "escalated": escalated,
        "semantic_escalated": bool(sem_escalate),
        "behavioral_escalated": bool(beh_escalate),
        "pilot_metrics_used": pilot_metrics_used,
    }


def run_batch_validation(
    user_datasets: list[dict],
    config: ValidationConfig,
) -> list[ValidationReport]:
    wrong_agent_cache: dict[int, tuple[SegmentAgent, DialogueSurfaceProfile]] = {}
    if config.skip_population_average_implant:
        pop_template = None
    else:
        pop_template = build_population_average_agent(
            user_datasets, config.implantation_config, seed=int(config.seed)
        )
    pilot = run_pilot_validation(
        user_datasets,
        config,
        population_average_template=pop_template,
        wrong_agent_cache=wrong_agent_cache,
    )
    required_users = int(pilot.get("suggested_min_users", config.min_users))
    if len(user_datasets) < required_users:
        raise ValueError(
            f"insufficient users for validation: have={len(user_datasets)}, required={required_users}"
        )
    reports: list[ValidationReport] = []
    total = len(user_datasets)
    _progress(f"running full validation on {total} users...")
    for idx, item in enumerate(user_datasets):
        _progress(f"  validation user {idx + 1}/{total} (uid={int(item.get('uid', -1))})")
        report = run_validation(
            item,
            config,
            all_user_profiles=user_datasets,
            population_average_template=pop_template,
            wrong_agent_cache=wrong_agent_cache,
        )
        report.aggregate["pilot"] = pilot
        report.aggregate["required_users"] = required_users
        report.aggregate["skip_population_average_implant"] = bool(config.skip_population_average_implant)
        report.aggregate["requested_strategies"] = [strategy.value for strategy in config.strategies]
        reports.append(report)
    return reports

