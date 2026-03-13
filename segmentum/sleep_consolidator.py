from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, replace
from statistics import mean
from typing import Callable

from .action_schema import ActionSchema, action_name, ensure_action_schema
from .types import (
    ModelUpdate,
    SemanticMemoryEntry,
    SequenceCondition,
    SequenceStep,
    SleepConsolidationResult,
    SleepRule,
)

HARMFUL_OUTCOMES = {"survival_threat", "integrity_loss", "resource_loss"}
MINIMUM_RULE_DOMINANCE = 0.60

_SLEEP_REFINE_SYSTEM_PROMPT = (
    "You are the sleep consolidation module of a survival-first digital organism "
    "governed by the Free Energy Principle.  During an offline sleep phase you "
    "receive heuristic causal rules extracted from high-surprise episodic memories "
    "and the raw episode summaries they were derived from.\n\n"
    "Your task:\n"
    "1. Review each candidate rule for causal plausibility.\n"
    "2. Adjust the `confidence` field (0.05–0.99) if the episode evidence warrants "
    "   a higher or lower confidence than the heuristic estimate.\n"
    "3. You may merge near-duplicate rules (same cluster+action) by keeping the "
    "   stronger one and raising its support count.\n"
    "4. You may add *at most one* new rule if you identify a clear causal pattern "
    "   that the heuristic missed.  New rules must use type 'risk_pattern' or "
    "   'opportunity_pattern'.\n"
    "5. Never remove all rules — at least one must survive.\n\n"
    "Return ONLY a JSON array of rule objects.  Each object must have exactly these "
    "fields: rule_id, type, cluster, action, observed_outcome, confidence, support, "
    "average_surprise, average_prediction_error, timestamp.  No markdown, no "
    "commentary — just the JSON array."
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class HeuristicSleepExtractor:
    """Default deterministic extractor: validates and passes rules through.

    This is the built-in backend for the LLM extraction stage.  When no
    external LLM is configured the pipeline still exercises this stage,
    ensuring confidence values are clamped and the extraction contract
    is honoured.  Replace with :class:`SleepLLMExtractor` for LLM-backed
    refinement.
    """

    def __call__(
        self,
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> list[SleepRule]:
        return [
            replace(rule, confidence=_clamp(rule.confidence, 0.05, 0.99))
            for rule in rules
        ]


class SleepLLMExtractor:
    """LLM-backed rule extraction stage.

    Wraps an external LLM backend (callable or object with
    ``refine_sleep_rules``) and delegates rule refinement to it.
    Falls back to the heuristic identity if the LLM call fails.
    """

    def __init__(self, llm=None) -> None:
        self.llm = llm

    def __call__(
        self,
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> list[SleepRule]:
        if self.llm is None:
            return rules

        summary = {
            "rule_count": len(rules),
            "episode_count": len(episodes),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "type": rule.type,
                    "cluster": rule.cluster,
                    "action": rule.action,
                    "observed_outcome": rule.observed_outcome,
                    "confidence": rule.confidence,
                    "support": rule.support,
                }
                for rule in rules
            ],
            "episodes": [
                {
                    "cluster_id": payload.get("cluster_id"),
                    "action": action_name(payload.get("action_taken", payload.get("action"))),
                    "predicted_outcome": payload.get("predicted_outcome"),
                    "prediction_error": payload.get("prediction_error"),
                    "total_surprise": payload.get("total_surprise"),
                }
                for payload in episodes
            ],
        }
        try:
            if callable(self.llm):
                refined = self.llm(summary=summary, rules=rules, episodes=episodes)
            elif hasattr(self.llm, "refine_sleep_rules"):
                refined = self.llm.refine_sleep_rules(
                    summary=summary,
                    rules=rules,
                    episodes=episodes,
                )
            else:
                return rules
        except Exception:
            return rules

        if not isinstance(refined, list) or not all(
            isinstance(rule, SleepRule) for rule in refined
        ):
            return rules
        if not refined:
            return rules
        return refined


class LLMSleepRuleRefiner:
    """Call an OpenAI-compatible API to refine heuristic sleep rules.

    Falls back to returning the original rules if the API is unreachable,
    returns malformed output, or no credentials are configured.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "openai/gpt-4.1-mini")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(
        self,
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> str:
        rule_dicts = [r.to_dict() for r in rules]
        episode_summaries = [
            {
                "cluster_id": p.get("cluster_id"),
                "action": action_name(p.get("action_taken", p.get("action"))),
                "predicted_outcome": p.get("predicted_outcome"),
                "prediction_error": round(float(p.get("prediction_error", 0)), 4),
                "total_surprise": round(float(p.get("total_surprise", 0)), 4),
            }
            for p in episodes
        ]
        return json.dumps(
            {"rules": rule_dicts, "episodes": episode_summaries},
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _parse_rules(text: str, fallback: list[SleepRule]) -> list[SleepRule]:
        """Parse LLM JSON response into SleepRule list."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return fallback
        if not isinstance(data, list) or not data:
            return fallback
        parsed: list[SleepRule] = []
        for item in data:
            if not isinstance(item, dict):
                return fallback
            try:
                parsed.append(
                    SleepRule(
                        rule_id=str(item["rule_id"]),
                        type=str(item["type"]),
                        cluster=int(item["cluster"]),
                        action=action_name(item["action"]),
                        observed_outcome=str(item["observed_outcome"]),
                        confidence=_clamp(float(item["confidence"]), 0.05, 0.99),
                        support=max(1, int(item["support"])),
                        average_surprise=float(item["average_surprise"]),
                        average_prediction_error=float(item["average_prediction_error"]),
                        timestamp=int(item["timestamp"]),
                        narrative_insight=str(item.get("narrative_insight", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                return fallback
        return parsed or fallback

    def refine_sleep_rules(
        self,
        *,
        summary: dict[str, object],
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> list[SleepRule]:
        """Synchronous HTTP call to refine rules via LLM."""
        if not self.api_key:
            return rules
        try:
            import httpx
        except ImportError:
            return rules

        user_prompt = self._build_user_prompt(rules, episodes)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SLEEP_REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            ) as client:
                response = client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_rules(content, rules)
        except Exception:
            return rules


def build_sleep_llm_extractor() -> SleepLLMExtractor | None:
    """Create a SleepLLMExtractor backed by a real LLM if credentials exist."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        try:
            from .llm import _load_openrouter_config

            config = _load_openrouter_config()
            api_key = config.get("api_key", "")
        except Exception:
            pass
    if not api_key:
        return None
    refiner = LLMSleepRuleRefiner(api_key=api_key)
    return SleepLLMExtractor(llm=refiner)


class SleepConsolidator:
    """Extract sleep rules from surprising replay via a two-stage pipeline.

    Stage 1 (heuristic): deterministic statistical rule extraction from
    clustered episodes.

    Stage 2 (LLM extraction): always executed.  Defaults to
    :class:`HeuristicSleepExtractor` (deterministic pass-through with
    confidence validation).  When a real LLM backend is provided, this
    stage refines rules using the model.
    """

    def __init__(
        self,
        *,
        surprise_threshold: float,
        minimum_support: int = 3,
        llm_extractor: Callable[[list[SleepRule], list[dict[str, object]]], list[SleepRule]]
        | None = None,
    ) -> None:
        self.surprise_threshold = surprise_threshold
        self.minimum_support = max(1, minimum_support)
        self.llm_extractor: Callable[
            [list[SleepRule], list[dict[str, object]]], list[SleepRule]
        ] = llm_extractor or HeuristicSleepExtractor()

    def consolidate(
        self,
        *,
        sleep_cycle_id: int,
        current_cycle: int,
        episodes: list[dict[str, object]],
        transition_statistics: dict[str, dict[str, float]],
        outcome_distributions: dict[str, dict[str, float]],
    ) -> SleepConsolidationResult:
        candidate_episodes = [
            payload
            for payload in episodes
            if float(payload.get("total_surprise", 0.0)) >= self.surprise_threshold
        ]
        rules = self._extract_rules(
            candidate_episodes,
            sleep_cycle_id=sleep_cycle_id,
            current_cycle=current_cycle,
            transition_statistics=transition_statistics,
            outcome_distributions=outcome_distributions,
        )
        rules_before_llm = list(rules)
        llm_used = not isinstance(self.llm_extractor, HeuristicSleepExtractor)
        if rules:
            try:
                refined = list(self.llm_extractor(rules, candidate_episodes))
                if refined:
                    rules = refined
            except Exception:
                pass
        rules = self._resolve_conflicts(rules)
        semantic_entries = self._semantic_entries(rules)
        model_updates = self._model_updates(rules)
        return SleepConsolidationResult(
            rules=rules,
            semantic_memory_entries=semantic_entries,
            model_updates=model_updates,
            llm_used=llm_used,
            rules_before_llm=rules_before_llm,
        )

    def _extract_rules(
        self,
        episodes: list[dict[str, object]],
        *,
        sleep_cycle_id: int,
        current_cycle: int,
        transition_statistics: dict[str, dict[str, float]],
        outcome_distributions: dict[str, dict[str, float]],
    ) -> list[SleepRule]:
        return self._extract_single_rules(
            episodes,
            sleep_cycle_id=sleep_cycle_id,
            current_cycle=current_cycle,
            transition_statistics=transition_statistics,
            outcome_distributions=outcome_distributions,
        ) + self._extract_sequence_rules(
            episodes,
            sleep_cycle_id=sleep_cycle_id,
            current_cycle=current_cycle,
            transition_statistics=transition_statistics,
            outcome_distributions=outcome_distributions,
        )

    def _extract_single_rules(
        self,
        episodes: list[dict[str, object]],
        *,
        sleep_cycle_id: int,
        current_cycle: int,
        transition_statistics: dict[str, dict[str, float]],
        outcome_distributions: dict[str, dict[str, float]],
    ) -> list[SleepRule]:
        grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
        action_totals: dict[tuple[int, str], int] = defaultdict(int)
        for payload in episodes:
            cluster = payload.get("cluster_id")
            action = action_name(payload.get("action_taken", payload.get("action", "")))
            outcome = str(payload.get("predicted_outcome", "neutral"))
            if not isinstance(cluster, int) or not action:
                continue
            grouped[(cluster, action, outcome)].append(payload)
            action_totals[(cluster, action)] += 1

        rules: list[SleepRule] = []
        for (cluster, action, outcome), payloads in sorted(grouped.items()):
            support = len(payloads)
            if support < self.minimum_support:
                continue
            average_surprise = mean(float(payload.get("total_surprise", 0.0)) for payload in payloads)
            average_prediction_error = mean(
                float(payload.get("prediction_error", 0.0)) for payload in payloads
            )
            dominance = support / max(1, action_totals[(cluster, action)])
            if dominance < MINIMUM_RULE_DOMINANCE:
                continue
            outcome_alignment = float(
                outcome_distributions.get(f"{cluster}:{action}", {}).get(outcome, 0.0)
            )
            transition_support = 1.0 if transition_statistics.get(f"{cluster}:{action}") else 0.0
            confidence = _clamp(
                0.30
                + min(0.20, 0.07 * max(0, support - 1))
                + (0.25 * dominance)
                + (0.15 * min(1.0, average_surprise / max(self.surprise_threshold, 1e-9)))
                + (0.05 * outcome_alignment)
                + (0.05 * transition_support),
                0.05,
                0.99,
            )
            rule_type = "risk_pattern" if outcome in HARMFUL_OUTCOMES else "opportunity_pattern"
            rules.append(
                SleepRule(
                    rule_id=f"sleep-{sleep_cycle_id}-{cluster}-{action}-{outcome}",
                    type=rule_type,
                    cluster=cluster,
                    action=action,
                    observed_outcome=outcome,
                    confidence=confidence,
                    support=support,
                    average_surprise=average_surprise,
                    average_prediction_error=average_prediction_error,
                    timestamp=current_cycle,
                    action_descriptor=ensure_action_schema(
                        payloads[0].get("action_taken", action)
                    ).to_dict(),
                )
            )
        return rules

    def _extract_sequence_rules(
        self,
        episodes: list[dict[str, object]],
        *,
        sleep_cycle_id: int,
        current_cycle: int,
        transition_statistics: dict[str, dict[str, float]],
        outcome_distributions: dict[str, dict[str, float]],
    ) -> list[SleepRule]:
        if len(episodes) < 3:
            return []

        def _has_explicit_tick(payload: dict[str, object]) -> bool:
            return "timestamp" in payload or "cycle" in payload

        def _tick(payload: dict[str, object]) -> int:
            return int(payload.get("timestamp", payload.get("cycle", 0)))

        timestamped_episodes = [payload for payload in episodes if _has_explicit_tick(payload)]
        if len(timestamped_episodes) < 3:
            return []

        sorted_episodes = sorted(timestamped_episodes, key=_tick)
        window_ticks = 10
        min_repeat = max(3, self.minimum_support)
        grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
        for payload in sorted_episodes:
            cluster = payload.get("cluster_id")
            action = action_name(payload.get("action_taken", payload.get("action", "")))
            outcome = str(payload.get("predicted_outcome", "neutral"))
            if not isinstance(cluster, int) or not action:
                continue
            grouped[(cluster, action, outcome)].append(payload)

        rules: list[SleepRule] = []
        for (cluster, action, outcome), payloads in sorted(grouped.items()):
            key = f"{cluster}:{action}"
            # Sequence rules are a fallback for replay bursts that do not yet
            # have stable slow-weight support in the world model.
            if transition_statistics.get(key) or outcome_distributions.get(key):
                continue
            if len(payloads) < min_repeat:
                continue

            best_window: list[dict[str, object]] = []
            left = 0
            for right in range(len(payloads)):
                while _tick(payloads[right]) - _tick(payloads[left]) > window_ticks:
                    left += 1
                window = payloads[left : right + 1]
                if len(window) > len(best_window):
                    best_window = window

            if len(best_window) < min_repeat:
                continue

            support = len(best_window)
            average_surprise = mean(float(payload.get("total_surprise", 0.0)) for payload in best_window)
            average_prediction_error = mean(
                float(payload.get("prediction_error", 0.0)) for payload in best_window
            )
            confidence = _clamp(
                0.50
                + (0.08 * max(0, support - min_repeat))
                + (0.12 * min(1.0, average_surprise / max(self.surprise_threshold, 1e-9))),
                0.05,
                0.99,
            )
            rules.append(
                SleepRule(
                    rule_id=f"sleep-seq-{sleep_cycle_id}-{cluster}-{action}-{outcome}-x{support}",
                    type="sequence_pattern",
                    cluster=cluster,
                    action=action,
                    observed_outcome=outcome,
                    confidence=confidence,
                    support=support,
                    average_surprise=average_surprise,
                    average_prediction_error=average_prediction_error,
                    timestamp=current_cycle,
                    action_descriptor=ensure_action_schema(
                        best_window[0].get("action_taken", action)
                    ).to_dict(),
                    sequence_condition=SequenceCondition(
                        steps=[SequenceStep(action_name=action, outcome=outcome)],
                        window_ticks=window_ticks,
                        min_occurrences=support,
                    ),
                )
            )
        return rules

    def _resolve_conflicts(self, rules: list[SleepRule]) -> list[SleepRule]:
        """Detect and resolve contradictory rules for the same cluster+action.

        When two rules target the same (cluster, action) but predict opposing
        outcome types (one risk_pattern, one opportunity_pattern), only the
        rule with the stronger evidence (confidence * support) survives.
        Ties are broken in favour of risk (safety-first).
        """
        groups: dict[tuple[int, str], list[SleepRule]] = defaultdict(list)
        for rule in rules:
            groups[(rule.cluster, rule.action)].append(rule)

        resolved: list[SleepRule] = []
        for (_cluster, _action), group in sorted(groups.items()):
            risk_rules = [r for r in group if r.type == "risk_pattern"]
            opportunity_rules = [r for r in group if r.type == "opportunity_pattern"]

            if risk_rules and opportunity_rules:
                best_risk = max(risk_rules, key=lambda r: (r.confidence * r.support, r.support))
                best_opp = max(opportunity_rules, key=lambda r: (r.confidence * r.support, r.support))
                risk_strength = best_risk.confidence * best_risk.support
                opp_strength = best_opp.confidence * best_opp.support
                if risk_strength >= opp_strength:
                    resolved.append(best_risk)
                else:
                    resolved.append(best_opp)
            else:
                resolved.extend(group)
        return resolved

    def _semantic_entries(self, rules: list[SleepRule]) -> list[SemanticMemoryEntry]:
        return [
            SemanticMemoryEntry(
                rule_id=rule.rule_id,
                rule_type=rule.type,
                cluster=rule.cluster,
                action=rule.action,
                confidence=rule.confidence,
                timestamp=rule.timestamp,
                observed_outcome=rule.observed_outcome,
                support=rule.support,
            )
            for rule in rules
        ]

    def _model_updates(self, rules: list[SleepRule]) -> list[ModelUpdate]:
        updates: list[ModelUpdate] = []
        for rule in rules:
            if rule.type in {"risk_pattern", "sequence_pattern"}:
                repeat_factor = 1.0
                threat_cap = 0.55
                preference_cap = 1.2
                if rule.type == "sequence_pattern" and rule.sequence_condition is not None:
                    repeat_factor = min(rule.sequence_condition.min_occurrences / 3.0, 2.5)
                    threat_cap = 0.75
                    preference_cap = 2.0
                updates.append(
                    ModelUpdate(
                        update_type="threat_prior",
                        cluster=rule.cluster,
                        action=rule.action,
                        delta=min(
                            threat_cap,
                            0.14 * rule.confidence * rule.support * repeat_factor,
                        ),
                        target=str(rule.cluster),
                        rule_id=rule.rule_id,
                    )
                )
                updates.append(
                    ModelUpdate(
                        update_type="preference_penalty",
                        cluster=rule.cluster,
                        action=rule.action,
                        delta=-min(
                            preference_cap,
                            0.25 * rule.confidence * rule.support * repeat_factor,
                        ),
                        target=f"{rule.cluster}:{rule.action}",
                        rule_id=rule.rule_id,
                    )
                )
                continue
            updates.append(
                ModelUpdate(
                    update_type="preference_penalty",
                    cluster=rule.cluster,
                    action=rule.action,
                    delta=min(0.40, 0.10 * rule.confidence * rule.support),
                    target=f"{rule.cluster}:{rule.action}",
                    rule_id=rule.rule_id,
                )
            )
        return updates

class SleepConsolidation:
    """Named M2 sleep surface that wraps the existing consolidator."""

    def __init__(
        self,
        *,
        surprise_threshold: float,
        minimum_support: int = 3,
        llm_extractor: Callable[[list[SleepRule], list[dict[str, object]]], list[SleepRule]]
        | None = None,
    ) -> None:
        self._consolidator = SleepConsolidator(
            surprise_threshold=surprise_threshold,
            minimum_support=minimum_support,
            llm_extractor=llm_extractor,
        )

    def consolidate(
        self,
        *,
        sleep_cycle_id: int,
        current_cycle: int,
        episodes: list[dict[str, object]],
        transition_statistics: dict[str, dict[str, float]],
        outcome_distributions: dict[str, dict[str, float]],
    ) -> SleepConsolidationResult:
        return self._consolidator.consolidate(
            sleep_cycle_id=sleep_cycle_id,
            current_cycle=current_cycle,
            episodes=episodes,
            transition_statistics=transition_statistics,
            outcome_distributions=outcome_distributions,
        )

    run = consolidate
