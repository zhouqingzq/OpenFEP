"""M2.4 Counterfactual Learning — Sophisticated Active Inference.

Implements a "mental sandbox" where the agent replays high-surprise episodes,
imagines alternative actions, evaluates them via Expected Free Energy (EFE),
and absorbs the resulting insights into policy biases.

Pipeline:  Sleep Consolidation → Counterfactual Engine → Insight Absorption → Wake
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from math import exp, sqrt
from statistics import mean

from .constants import ACTION_BODY_EFFECTS, ACTION_COSTS
from .environment import clamp
from .preferences import PreferenceModel
from .world_model import GenerativeWorldModel


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

ABSORPTION_THRESHOLD = 0.60
SIGNIFICANCE_THRESHOLD = 0.50
DEFAULT_LEARNING_RATE = 0.08
MAX_UPDATE_STEP = 0.15
DEFAULT_ENERGY_COST_PER_NODE = 0.005
DEFAULT_ENERGY_BUDGET = 0.10
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_CONFIDENCE_DECAY = 0.85
DEFAULT_COOLING_CONFIRMATIONS = 2
DEFAULT_REVIEW_PASS_RATE = 0.60
DEFAULT_REVIEW_BENEFIT_THRESHOLD = 0.02


@dataclass
class CounterfactualInsight:
    """A single counterfactual observation: an untaken action would have been better."""

    source_episode_cycle: int
    original_action: str
    counterfactual_action: str
    original_efe: float
    counterfactual_efe: float
    efe_delta: float
    confidence: float
    state_context: dict[str, object]
    cluster_id: int | None
    timestamp: int
    absorbed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_episode_cycle": self.source_episode_cycle,
            "original_action": self.original_action,
            "counterfactual_action": self.counterfactual_action,
            "original_efe": self.original_efe,
            "counterfactual_efe": self.counterfactual_efe,
            "efe_delta": self.efe_delta,
            "confidence": self.confidence,
            "state_context": dict(self.state_context),
            "cluster_id": self.cluster_id,
            "timestamp": self.timestamp,
            "absorbed": self.absorbed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CounterfactualInsight:
        cluster_id = payload.get("cluster_id")
        return cls(
            source_episode_cycle=int(payload.get("source_episode_cycle", 0)),
            original_action=str(payload.get("original_action", "")),
            counterfactual_action=str(payload.get("counterfactual_action", "")),
            original_efe=float(payload.get("original_efe", 0.0)),
            counterfactual_efe=float(payload.get("counterfactual_efe", 0.0)),
            efe_delta=float(payload.get("efe_delta", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            state_context=dict(payload.get("state_context") or {}),
            cluster_id=int(cluster_id) if cluster_id is not None else None,
            timestamp=int(payload.get("timestamp", 0)),
            absorbed=bool(payload.get("absorbed", False)),
        )


@dataclass
class CounterfactualSummary:
    """Statistics from one counterfactual phase."""

    episodes_evaluated: int = 0
    branches_explored: int = 0
    insights_generated: int = 0
    insights_absorbed: int = 0
    energy_spent: float = 0.0
    policy_updates: int = 0
    skipped_reason: str = ""
    counterfactual_log: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes_evaluated": self.episodes_evaluated,
            "branches_explored": self.branches_explored,
            "insights_generated": self.insights_generated,
            "insights_absorbed": self.insights_absorbed,
            "energy_spent": self.energy_spent,
            "policy_updates": self.policy_updates,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class BranchResult:
    """Result of evaluating a single counterfactual branch."""

    action: str
    predicted_efe: float
    confidence: float
    steps_completed: int
    pruned: bool = False


# ---------------------------------------------------------------------------
# Forward Generative Model (B-matrix analogue)
# ---------------------------------------------------------------------------


class ForwardGenerativeModel:
    """Internal world simulator built from learned causal rules and body effects.

    Uses :class:`GenerativeWorldModel` action effects for state transitions and
    computes confidence from distance to known episode states.
    """

    def __init__(
        self,
        *,
        world_model: GenerativeWorldModel,
        preference_model: PreferenceModel,
        known_episodes: list[dict[str, object]],
        confidence_decay: float = DEFAULT_CONFIDENCE_DECAY,
        unknown_region_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.world_model = world_model
        self.preference_model = preference_model
        self.confidence_decay = confidence_decay
        self.unknown_region_threshold = unknown_region_threshold

        # Pre-compute observation+body feature vectors for confidence estimation.
        self._known_vectors: list[list[float]] = []
        for ep in known_episodes:
            vec = self._state_to_vector(ep)
            if vec:
                self._known_vectors.append(vec)

    # -- public API ----------------------------------------------------------

    def predict_step(
        self,
        state: dict[str, object],
        action: str,
    ) -> tuple[dict[str, object], float]:
        """Predict next state from ``(state, action)``.

        Returns ``(next_state, confidence)`` where confidence ∈ (0, 1].
        """
        obs = _float_dict(state.get("observation", {}))
        body = _float_dict(state.get("body_state", {}))

        # Sensory prediction via world model action effects
        imagined_obs = self.world_model.imagine_action(action, obs)

        # Body state projection
        cost = ACTION_COSTS.get(action, 0.05)
        effects = ACTION_BODY_EFFECTS.get(action, {})
        energy = clamp(
            float(body.get("energy", 0.5))
            - cost - 0.015
            + effects.get("energy_delta", 0.0)
        )
        stress = clamp(
            float(body.get("stress", 0.25)) + effects.get("stress_delta", 0.0)
        )
        fatigue = clamp(
            float(body.get("fatigue", 0.2))
            + 0.08
            + effects.get("fatigue_delta", 0.0)
        )
        temperature = clamp(
            float(body.get("temperature", 0.48))
            + effects.get("temperature_delta", 0.0)
        )

        energy_delta = energy - float(body.get("energy", 0.5))
        stress_delta = stress - float(body.get("stress", 0.25))
        fatigue_delta = fatigue - float(body.get("fatigue", 0.2))
        temperature_delta = temperature - float(body.get("temperature", 0.48))

        next_body: dict[str, float] = {
            "energy": energy,
            "stress": stress,
            "fatigue": fatigue,
            "temperature": temperature,
        }
        predicted_outcome: dict[str, float] = {
            "energy_delta": energy_delta,
            "stress_delta": stress_delta,
            "fatigue_delta": fatigue_delta,
            "temperature_delta": temperature_delta,
            "free_energy_drop": energy_delta - max(0.0, stress_delta) * 0.5,
        }

        next_state: dict[str, object] = {
            "observation": imagined_obs,
            "body_state": next_body,
            "predicted_outcome": predicted_outcome,
        }

        confidence = self._compute_confidence(state)
        return next_state, confidence

    def predict_multistep(
        self,
        state: dict[str, object],
        actions: list[str],
    ) -> list[tuple[dict[str, object], float]]:
        """Multi-step forward prediction with cumulative confidence decay."""
        results: list[tuple[dict[str, object], float]] = []
        current = state
        cumulative_confidence = 1.0
        for action in actions:
            next_state, step_confidence = self.predict_step(current, action)
            cumulative_confidence *= step_confidence * self.confidence_decay
            results.append((next_state, cumulative_confidence))
            current = next_state
        return results

    def compute_efe(self, predicted_state: dict[str, object]) -> float:
        """Expected Free Energy = pragmatic_value + epistemic_value.

        Higher EFE ⟹ worse state.  The agent prefers actions that minimise EFE.
        """
        evaluation = self.preference_model.evaluate_state(predicted_state)
        pragmatic_value = evaluation.risk

        confidence = self._compute_confidence(predicted_state)
        epistemic_value = max(0.0, 1.0 - confidence) * 0.5

        return pragmatic_value + epistemic_value

    # -- internals -----------------------------------------------------------

    def _compute_confidence(self, state: dict[str, object]) -> float:
        """Distance-based confidence: close to known states ⟹ high confidence."""
        if not self._known_vectors:
            return 0.1

        query = self._state_to_vector(state)
        if not query:
            return 0.1

        min_distance = float("inf")
        for known in self._known_vectors:
            length = min(len(query), len(known))
            if length == 0:
                continue
            dist = sqrt(
                sum((query[i] - known[i]) ** 2 for i in range(length)) / length
            )
            min_distance = min(min_distance, dist)

        if min_distance == float("inf"):
            return 0.1

        return max(0.05, min(1.0, exp(-min_distance * 3.0)))

    @staticmethod
    def _state_to_vector(state: dict[str, object]) -> list[float]:
        """Build a compact feature vector from observation + body_state."""
        obs = state.get("observation", {})
        body = state.get("body_state", {})
        if not isinstance(obs, dict) or not obs:
            return []
        vec = [float(obs.get(k, 0.0)) for k in sorted(obs.keys())]
        if isinstance(body, dict):
            vec.extend(float(body.get(k, 0.0)) for k in sorted(body.keys()))
        return vec


# ---------------------------------------------------------------------------
# Counterfactual Engine
# ---------------------------------------------------------------------------


class CounterfactualEngine:
    """Replays high-surprise episodes with alternative actions and scores them."""

    def __init__(
        self,
        *,
        forward_model: ForwardGenerativeModel,
        preference_model: PreferenceModel,
        max_depth: int = 3,
        max_branches: int = 3,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        energy_cost_per_node: float = DEFAULT_ENERGY_COST_PER_NODE,
        energy_budget: float = DEFAULT_ENERGY_BUDGET,
        surprise_threshold: float = 0.40,
    ) -> None:
        self.forward_model = forward_model
        self.preference_model = preference_model
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.confidence_threshold = confidence_threshold
        self.energy_cost_per_node = energy_cost_per_node
        self.energy_budget = energy_budget
        self.surprise_threshold = surprise_threshold
        self._energy_spent = 0.0

    def run(
        self,
        *,
        episodes: list[dict[str, object]],
        current_cycle: int,
        agent_energy: float,
        rng: random.Random,
    ) -> tuple[list[CounterfactualInsight], CounterfactualSummary]:
        """Run counterfactual reasoning.  Returns (insights, summary)."""
        summary = CounterfactualSummary()

        # Energy gate: refuse to run when energy is critically low.
        if agent_energy < 0.30:
            summary.skipped_reason = "energy_below_30pct"
            return [], summary

        # Adaptive depth based on energy.
        effective_depth = self.max_depth
        if agent_energy < 0.50:
            effective_depth = 1

        # Select high-surprise episodes.
        candidates = [
            ep
            for ep in episodes
            if float(ep.get("total_surprise", 0.0)) > self.surprise_threshold
        ]
        if not candidates:
            summary.skipped_reason = "no_high_surprise_episodes"
            return [], summary

        candidates.sort(key=lambda ep: -float(ep.get("total_surprise", 0.0)))

        self._energy_spent = 0.0
        insights: list[CounterfactualInsight] = []

        for episode in candidates:
            if self._energy_spent + self.energy_cost_per_node > self.energy_budget:
                break
            episode_insights, branches = self._evaluate_episode(
                episode=episode,
                depth=effective_depth,
                current_cycle=current_cycle,
                rng=rng,
            )
            insights.extend(episode_insights)
            summary.episodes_evaluated += 1
            summary.branches_explored += branches

        summary.insights_generated = len(insights)
        summary.energy_spent = self._energy_spent
        return insights, summary

    def _evaluate_episode(
        self,
        *,
        episode: dict[str, object],
        depth: int,
        current_cycle: int,
        rng: random.Random,
    ) -> tuple[list[CounterfactualInsight], int]:
        original_action = str(
            episode.get("action_taken", episode.get("action", ""))
        )
        episode_cycle = int(episode.get("timestamp", episode.get("cycle", 0)))

        state_context: dict[str, object] = {
            "observation": dict(episode.get("observation") or {}),
            "body_state": dict(episode.get("body_state") or {}),
            "prediction": dict(episode.get("prediction") or {}),
            "errors": dict(episode.get("errors") or {}),
        }

        # Original action EFE from *actual* outcome.
        original_efe_state: dict[str, object] = {
            "observation": dict(episode.get("observation") or {}),
            "body_state": dict(episode.get("body_state") or {}),
            "predicted_outcome": dict(
                episode.get("outcome", episode.get("outcome_state", {})) or {}
            ),
        }
        original_efe = self.forward_model.compute_efe(original_efe_state)

        cluster_id = episode.get("cluster_id")

        # Enumerate alternative actions, prioritised by policy bias.
        alternative_actions = [a for a in ACTION_COSTS if a != original_action]
        if isinstance(cluster_id, int):
            alternative_actions.sort(
                key=lambda a: -self.forward_model.world_model.get_policy_bias(
                    cluster_id, a
                )
            )
        alternative_actions = alternative_actions[: self.max_branches]

        insights: list[CounterfactualInsight] = []
        branches_explored = 0

        for alt_action in alternative_actions:
            if self._energy_spent + self.energy_cost_per_node > self.energy_budget:
                break

            self._energy_spent += self.energy_cost_per_node
            branches_explored += 1

            # First step.
            next_state, confidence = self.forward_model.predict_step(
                state_context, alt_action
            )
            if confidence < self.confidence_threshold:
                continue  # Prune low-confidence branch.

            best_efe = self.forward_model.compute_efe(next_state)
            total_confidence = confidence

            # Multi-step rollout.
            if depth > 1:
                current = next_state
                for _step in range(1, depth):
                    if self._energy_spent + self.energy_cost_per_node > self.energy_budget:
                        break
                    self._energy_spent += self.energy_cost_per_node
                    step_state, step_conf = self.forward_model.predict_step(
                        current, alt_action
                    )
                    total_confidence *= (
                        step_conf * self.forward_model.confidence_decay
                    )
                    if total_confidence < self.confidence_threshold:
                        break
                    step_efe = self.forward_model.compute_efe(step_state)
                    best_efe = min(best_efe, step_efe)
                    current = step_state

            efe_delta = best_efe - original_efe
            effective_confidence = total_confidence
            if efe_delta >= 0:
                fallback_delta = self._counterfactual_fallback_delta(
                    episode=episode,
                    predicted_state=next_state,
                )
                if fallback_delta is None:
                    continue
                efe_delta = fallback_delta
                best_efe = original_efe + fallback_delta
                effective_confidence = min(
                    total_confidence,
                    self.confidence_threshold + 0.36,
                )

            insights.append(
                CounterfactualInsight(
                    source_episode_cycle=episode_cycle,
                    original_action=original_action,
                    counterfactual_action=alt_action,
                    original_efe=original_efe,
                    counterfactual_efe=best_efe,
                    efe_delta=efe_delta,
                    confidence=effective_confidence,
                    state_context=state_context,
                    cluster_id=(
                        int(cluster_id) if isinstance(cluster_id, int) else None
                    ),
                    timestamp=current_cycle,
                )
            )

        return insights, branches_explored

    def _counterfactual_fallback_delta(
        self,
        *,
        episode: dict[str, object],
        predicted_state: dict[str, object],
    ) -> float | None:
        predicted_outcome_label = str(episode.get("predicted_outcome", "")).lower()
        episode_risk = float(episode.get("risk", 0.0))
        if predicted_outcome_label not in {"survival_threat", "integrity_loss"} and episode_risk < 3.0:
            return None

        actual_outcome = _float_dict(episode.get("outcome", episode.get("outcome_state", {})))
        actual_observation = _float_dict(episode.get("observation", {}))
        imagined_observation = _float_dict(predicted_state.get("observation", {}))
        predicted_outcome = _float_dict(predicted_state.get("predicted_outcome", {}))

        actual_harm = (
            max(0.0, -float(actual_outcome.get("energy_delta", 0.0))) * 1.2
            + max(0.0, float(actual_outcome.get("stress_delta", 0.0))) * 1.0
            + max(0.0, float(actual_outcome.get("fatigue_delta", 0.0))) * 0.4
            + float(actual_observation.get("danger", 0.0)) * 1.4
            - float(actual_observation.get("shelter", 0.0)) * 0.3
        )
        counterfactual_harm = (
            max(0.0, -float(predicted_outcome.get("energy_delta", 0.0))) * 1.1
            + max(0.0, float(predicted_outcome.get("stress_delta", 0.0))) * 0.9
            + max(0.0, float(predicted_outcome.get("fatigue_delta", 0.0))) * 0.3
            + float(imagined_observation.get("danger", 0.0)) * 1.2
            - float(imagined_observation.get("shelter", 0.0)) * 0.3
        )
        improvement = actual_harm - counterfactual_harm
        if improvement <= 0.08:
            return None
        return -improvement


# ---------------------------------------------------------------------------
# Insight Absorber
# ---------------------------------------------------------------------------


class InsightAbsorber:
    """Write qualifying counterfactual insights back into policy biases.

    Absorption rules:
    - Only insights with ``confidence > absorption_threshold`` are accepted.
    - Only insights with ``efe_delta < -significance_threshold`` are accepted.
    - Updates go to ``world_model.policy_biases``, never to core beliefs.
    - Each absorption is logged for full causal traceability.
    """

    def __init__(
        self,
        *,
        absorption_threshold: float = ABSORPTION_THRESHOLD,
        significance_threshold: float = SIGNIFICANCE_THRESHOLD,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_update_step: float = MAX_UPDATE_STEP,
        cooling_confirmations: int = DEFAULT_COOLING_CONFIRMATIONS,
        review_pass_rate: float = DEFAULT_REVIEW_PASS_RATE,
        review_benefit_threshold: float = DEFAULT_REVIEW_BENEFIT_THRESHOLD,
    ) -> None:
        self.absorption_threshold = absorption_threshold
        self.significance_threshold = significance_threshold
        self.learning_rate = learning_rate
        self.max_update_step = max_update_step
        self.cooling_confirmations = max(1, cooling_confirmations)
        self.review_pass_rate = review_pass_rate
        self.review_benefit_threshold = review_benefit_threshold
        self.log: list[dict[str, object]] = []

    def absorb(
        self,
        insights: list[CounterfactualInsight],
        world_model: GenerativeWorldModel,
        *,
        preference_model: PreferenceModel | None = None,
    ) -> int:
        """Absorb qualifying insights.  Returns count absorbed."""
        absorbed_count = 0
        for insight in insights:
            if insight.absorbed:
                continue
            if insight.confidence < self.absorption_threshold:
                self._log_rejection(insight, "confidence_below_threshold")
                continue
            if insight.efe_delta > -self.significance_threshold:
                self._log_rejection(insight, "efe_delta_not_significant")
                continue
            if insight.cluster_id is None:
                self._log_rejection(insight, "no_cluster_id")
                continue

            buffer_key = (
                f"{insight.cluster_id}:{insight.original_action}:{insight.counterfactual_action}"
            )
            candidate_buffer = world_model.counterfactual_candidate_buffer.setdefault(
                buffer_key,
                {
                    "confirmations": 0.0,
                    "cumulative_confidence": 0.0,
                    "best_delta": 0.0,
                    "last_timestamp": float(insight.timestamp),
                    "cluster_id": float(insight.cluster_id),
                    "original_action": insight.original_action,
                    "counterfactual_action": insight.counterfactual_action,
                },
            )
            candidate_buffer["confirmations"] = float(candidate_buffer.get("confirmations", 0.0)) + 1.0
            candidate_buffer["cumulative_confidence"] = float(
                candidate_buffer.get("cumulative_confidence", 0.0)
            ) + float(insight.confidence)
            candidate_buffer["best_delta"] = max(
                float(candidate_buffer.get("best_delta", 0.0)),
                abs(float(insight.efe_delta)),
            )
            candidate_buffer["last_timestamp"] = float(insight.timestamp)
            self._accumulate_candidate_state(candidate_buffer, insight)

            if (
                insight.confidence < (self.absorption_threshold + 0.15)
                and candidate_buffer["confirmations"] < float(self.cooling_confirmations)
            ):
                self.log.append(
                    {
                        "type": "buffered_candidate",
                        "reason": "cooling_gate",
                        "source_episode_cycle": insight.source_episode_cycle,
                        "cluster_id": insight.cluster_id,
                        "original_action": insight.original_action,
                        "counterfactual_action": insight.counterfactual_action,
                        "efe_delta": insight.efe_delta,
                        "confidence": insight.confidence,
                        "confirmations": candidate_buffer["confirmations"],
                        "timestamp": insight.timestamp,
                    }
                )
                continue

            review = self._review_candidate(
                candidate_buffer,
                insight,
                world_model,
                preference_model=preference_model,
            )
            self.log.append(review)
            if not bool(review.get("passed")):
                self.log.append(
                    {
                        "type": "buffered_candidate",
                        "reason": "review_veto",
                        "source_episode_cycle": insight.source_episode_cycle,
                        "cluster_id": insight.cluster_id,
                        "original_action": insight.original_action,
                        "counterfactual_action": insight.counterfactual_action,
                        "efe_delta": insight.efe_delta,
                        "confidence": insight.confidence,
                        "confirmations": candidate_buffer["confirmations"],
                        "timestamp": insight.timestamp,
                    }
                )
                continue

            raw_delta = self.learning_rate * max(
                abs(insight.efe_delta),
                float(candidate_buffer.get("best_delta", 0.0)),
            )
            review_scale = max(
                0.5,
                min(
                    1.5,
                    float(review.get("average_benefit", 0.0))
                    / max(self.review_benefit_threshold, 1e-6),
                ),
            )
            delta = min(raw_delta * review_scale, self.max_update_step)

            new_cf_bias = world_model.adjust_policy_bias(
                insight.cluster_id, insight.counterfactual_action, delta
            )
            new_orig_bias = world_model.adjust_policy_bias(
                insight.cluster_id, insight.original_action, -delta
            )
            new_cf_penalty = world_model.adjust_preference_penalty(
                insight.cluster_id, insight.counterfactual_action, delta * 0.60
            )
            new_orig_penalty = world_model.adjust_preference_penalty(
                insight.cluster_id, insight.original_action, -(delta * 0.90)
            )

            cf_biases = world_model.counterfactual_biases
            cf_biases[insight.counterfactual_action] = max(
                -1.0,
                min(1.0, cf_biases.get(insight.counterfactual_action, 0.0) + delta),
            )
            cf_biases[insight.original_action] = max(
                -1.0,
                min(1.0, cf_biases.get(insight.original_action, 0.0) - delta),
            )

            world_model.counterfactual_candidate_buffer.pop(buffer_key, None)
            insight.absorbed = True
            absorbed_count += 1

            self.log.append(
                {
                    "type": "absorption",
                    "source_episode_cycle": insight.source_episode_cycle,
                    "cluster_id": insight.cluster_id,
                    "original_action": insight.original_action,
                    "counterfactual_action": insight.counterfactual_action,
                    "efe_delta": insight.efe_delta,
                    "confidence": insight.confidence,
                    "policy_delta": delta,
                    "new_cf_bias": new_cf_bias,
                    "new_orig_bias": new_orig_bias,
                    "new_cf_preference_penalty": new_cf_penalty,
                    "new_orig_preference_penalty": new_orig_penalty,
                    "confirmations": candidate_buffer.get("confirmations", 1.0),
                    "review_average_benefit": review.get("average_benefit", 0.0),
                    "review_pass_rate": review.get("pass_rate", 0.0),
                    "timestamp": insight.timestamp,
                }
            )

        return absorbed_count

    def _accumulate_candidate_state(
        self,
        candidate_buffer: dict[str, object],
        insight: CounterfactualInsight,
    ) -> None:
        observation = _float_dict(insight.state_context.get("observation", {}))
        body_state = _float_dict(insight.state_context.get("body_state", {}))
        counts = float(candidate_buffer.get("confirmations", 1.0))
        if observation:
            existing = _float_dict(candidate_buffer.get("prototype_observation", {}))
            if not existing:
                candidate_buffer["prototype_observation"] = dict(observation)
            else:
                candidate_buffer["prototype_observation"] = {
                    key: (
                        (existing.get(key, 0.0) * max(0.0, counts - 1.0))
                        + observation.get(key, 0.0)
                    )
                    / max(1.0, counts)
                    for key in sorted(set(existing) | set(observation))
                }
        if body_state:
            existing = _float_dict(candidate_buffer.get("prototype_body_state", {}))
            if not existing:
                candidate_buffer["prototype_body_state"] = dict(body_state)
            else:
                candidate_buffer["prototype_body_state"] = {
                    key: (
                        (existing.get(key, 0.0) * max(0.0, counts - 1.0))
                        + body_state.get(key, 0.0)
                    )
                    / max(1.0, counts)
                    for key in sorted(set(existing) | set(body_state))
                }

    def _review_candidate(
        self,
        candidate_buffer: dict[str, object],
        insight: CounterfactualInsight,
        world_model: GenerativeWorldModel,
        *,
        preference_model: PreferenceModel | None,
    ) -> dict[str, object]:
        observation = _float_dict(candidate_buffer.get("prototype_observation", {}))
        body_state = _float_dict(candidate_buffer.get("prototype_body_state", {}))
        if not observation:
            observation = _float_dict(insight.state_context.get("observation", {}))
        if not body_state:
            body_state = _float_dict(insight.state_context.get("body_state", {}))

        perturbations = self._build_review_perturbations(observation, body_state)
        benefits: list[float] = []
        for perturbed_observation, perturbed_body in perturbations:
            original_score = self._project_candidate_benefit(
                observation=perturbed_observation,
                body_state=perturbed_body,
                action=insight.original_action,
                world_model=world_model,
                preference_model=preference_model,
            )
            counterfactual_score = self._project_candidate_benefit(
                observation=perturbed_observation,
                body_state=perturbed_body,
                action=insight.counterfactual_action,
                world_model=world_model,
                preference_model=preference_model,
            )
            benefits.append(counterfactual_score - original_score)

        pass_rate = (
            sum(1 for benefit in benefits if benefit > self.review_benefit_threshold) / len(benefits)
            if benefits
            else 0.0
        )
        average_benefit = mean(benefits) if benefits else 0.0
        requires_confirmation = insight.confidence < (self.absorption_threshold + 0.15)
        return {
            "type": "candidate_review",
            "cluster_id": insight.cluster_id,
            "original_action": insight.original_action,
            "counterfactual_action": insight.counterfactual_action,
            "confirmations": candidate_buffer.get("confirmations", 0.0),
            "average_benefit": average_benefit,
            "pass_rate": pass_rate,
            "requires_confirmation": requires_confirmation,
            "passed": (
                (
                    not requires_confirmation
                    or candidate_buffer.get("confirmations", 0.0) >= float(self.cooling_confirmations)
                )
                and average_benefit > self.review_benefit_threshold
                and pass_rate >= self.review_pass_rate
            ),
        }

    def _build_review_perturbations(
        self,
        observation: dict[str, float],
        body_state: dict[str, float],
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        obs_variants = (
            {},
            {"danger": 0.04, "food": -0.02, "shelter": -0.02},
            {"danger": -0.03, "food": 0.01, "novelty": 0.02},
        )
        body_variants = (
            {},
            {"energy": -0.03, "stress": 0.04},
            {"energy": 0.02, "fatigue": -0.02},
        )
        perturbations: list[tuple[dict[str, float], dict[str, float]]] = []
        for obs_delta, body_delta in zip(obs_variants, body_variants):
            perturbations.append(
                (
                    {
                        key: clamp(float(observation.get(key, 0.0)) + float(obs_delta.get(key, 0.0)))
                        for key in sorted(set(observation) | set(obs_delta))
                    },
                    {
                        key: clamp(float(body_state.get(key, 0.0)) + float(body_delta.get(key, 0.0)))
                        for key in sorted(set(body_state) | set(body_delta))
                    },
                )
            )
        return perturbations

    def _project_candidate_benefit(
        self,
        *,
        observation: dict[str, float],
        body_state: dict[str, float],
        action: str,
        world_model: GenerativeWorldModel,
        preference_model: PreferenceModel | None,
    ) -> float:
        imagined = world_model.imagine_action(action, observation)
        cost = ACTION_COSTS.get(action, 0.05)
        effects = ACTION_BODY_EFFECTS.get(action, {})
        next_body = {
            "energy": clamp(float(body_state.get("energy", 0.5)) - cost - 0.015 + effects.get("energy_delta", 0.0)),
            "stress": clamp(float(body_state.get("stress", 0.25)) + effects.get("stress_delta", 0.0)),
            "fatigue": clamp(float(body_state.get("fatigue", 0.2)) + 0.08 + effects.get("fatigue_delta", 0.0)),
            "temperature": clamp(float(body_state.get("temperature", 0.48)) + effects.get("temperature_delta", 0.0)),
        }
        heuristic_score = (
            (1.0 - float(imagined.get("danger", 0.0))) * 1.30
            + float(imagined.get("shelter", 0.0)) * 0.45
            + float(next_body.get("energy", 0.0)) * 0.85
            - float(next_body.get("stress", 0.0)) * 0.70
            - float(next_body.get("fatigue", 0.0)) * 0.25
            - abs(float(next_body.get("temperature", 0.5)) - 0.5) * 0.20
        )
        if preference_model is None:
            return heuristic_score
        preference = preference_model.evaluate_state(
            {
                "observation": imagined,
                "body_state": next_body,
                "predicted_outcome": {
                    "energy_delta": next_body["energy"] - float(body_state.get("energy", 0.5)),
                    "stress_delta": next_body["stress"] - float(body_state.get("stress", 0.25)),
                    "fatigue_delta": next_body["fatigue"] - float(body_state.get("fatigue", 0.2)),
                    "temperature_delta": next_body["temperature"] - float(body_state.get("temperature", 0.48)),
                },
            }
        )
        return heuristic_score + preference.value_score - (preference.risk * 0.05)
    def _log_rejection(
        self,
        insight: CounterfactualInsight,
        reason: str,
    ) -> None:
        self.log.append(
            {
                "type": "rejection",
                "reason": reason,
                "source_episode_cycle": insight.source_episode_cycle,
                "original_action": insight.original_action,
                "counterfactual_action": insight.counterfactual_action,
                "efe_delta": insight.efe_delta,
                "confidence": insight.confidence,
            }
        )


# ---------------------------------------------------------------------------
# Top-level integration helper
# ---------------------------------------------------------------------------


def run_counterfactual_phase(
    *,
    agent_energy: float,
    current_cycle: int,
    episodes: list[dict[str, object]],
    world_model: GenerativeWorldModel,
    preference_model: PreferenceModel,
    rng: random.Random,
    max_depth: int = 3,
    energy_budget: float = DEFAULT_ENERGY_BUDGET,
    surprise_threshold: float = 0.40,
) -> tuple[list[CounterfactualInsight], CounterfactualSummary]:
    """Run the full counterfactual phase: engine + absorption.

    Called at the end of sleep consolidation, after slow-weight updates.
    """
    forward_model = ForwardGenerativeModel(
        world_model=world_model,
        preference_model=preference_model,
        known_episodes=episodes,
    )
    engine = CounterfactualEngine(
        forward_model=forward_model,
        preference_model=preference_model,
        max_depth=max_depth,
        energy_budget=energy_budget,
        surprise_threshold=surprise_threshold,
    )
    absorber = InsightAbsorber()

    insights, summary = engine.run(
        episodes=episodes,
        current_cycle=current_cycle,
        agent_energy=agent_energy,
        rng=rng,
    )

    absorbed = absorber.absorb(
        insights,
        world_model,
        preference_model=preference_model,
    )
    summary.insights_absorbed = absorbed
    summary.policy_updates = absorbed
    summary.counterfactual_log = list(absorber.log)

    # 验收 M2.4: 日志中出现「虚拟沙盒推演」记录
    if summary.episodes_evaluated > 0 or summary.branches_explored > 0:
        summary.counterfactual_log.insert(
            0,
            {
                "type": "virtual_sandbox_reasoning",
                "label": "虚拟沙盒推演",
                "episodes_evaluated": summary.episodes_evaluated,
                "branches_explored": summary.branches_explored,
                "insights_generated": summary.insights_generated,
                "insights_absorbed": summary.insights_absorbed,
                "energy_spent": summary.energy_spent,
                "skipped_reason": summary.skipped_reason or "",
            },
        )

    return insights, summary

class CounterfactualLearning:
    """Named M2 counterfactual surface wrapping the sleep-phase helper."""

    def run(
        self,
        *,
        agent_energy: float,
        current_cycle: int,
        episodes: list[dict[str, object]],
        world_model: GenerativeWorldModel,
        preference_model: PreferenceModel,
        rng: random.Random,
        max_depth: int = 3,
        energy_budget: float = DEFAULT_ENERGY_BUDGET,
        surprise_threshold: float = 0.40,
    ) -> tuple[list[CounterfactualInsight], CounterfactualSummary]:
        return run_counterfactual_phase(
            agent_energy=agent_energy,
            current_cycle=current_cycle,
            episodes=episodes,
            world_model=world_model,
            preference_model=preference_model,
            rng=rng,
            max_depth=max_depth,
            energy_budget=energy_budget,
            surprise_threshold=surprise_threshold,
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
