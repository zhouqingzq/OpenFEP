"""Generate M5.3 persona-action correlation artifact with chi-square statistics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.actions import DIALOGUE_ACTION_NAMES
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions as register_from_bridge


def _chi_square_critical_0_05(df: int) -> float:
    table = {
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
        11: 19.675,
        12: 21.026,
        13: 22.362,
        14: 23.685,
        15: 24.996,
        16: 26.296,
        17: 27.587,
        18: 28.869,
        19: 30.144,
        20: 31.410,
        21: 32.671,
        22: 33.924,
        23: 35.172,
        24: 36.415,
        25: 37.652,
        26: 38.885,
        27: 40.113,
        28: 41.337,
        29: 42.557,
        30: 43.773,
    }
    return table.get(df, 43.773)


def _regularized_gamma_q(s: float, x: float) -> float:
    """Return Q(s, x) = 1 - P(s, x) using series/continued-fraction expansions."""
    if s <= 0.0:
        raise ValueError("s must be > 0")
    if x < 0.0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 1.0

    eps = 1.0e-12
    fpm_in = 1.0e-300
    gln = math.lgamma(s)

    if x < s + 1.0:
        ap = s
        delta = 1.0 / s
        series = delta
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            series += delta
            if abs(delta) < abs(series) * eps:
                break
        p = series * math.exp(-x + s * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - s
    c = 1.0 / fpm_in
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < fpm_in:
            d = fpm_in
        c = b + an / c
        if abs(c) < fpm_in:
            c = fpm_in
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = math.exp(-x + s * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def _chi_square_p_value(chi2: float, df: int) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if chi2 < 0.0:
        return 1.0
    return _regularized_gamma_q(0.5 * float(df), 0.5 * float(chi2))


def _persona_counts(persona: str, seeds: range, lines: list[str], observer: DialogueObserver) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seed in seeds:
        agent = SegmentAgent()
        register_from_bridge(agent.action_registry)
        if persona == "high_trust":
            agent.self_model.narrative_priors.trust_prior = 0.90
            agent.self_model.narrative_priors.trauma_bias = 0.05
            agent.self_model.personality_profile.agreeableness = 0.78
            agent.self_model.personality_profile.neuroticism = 0.22
        elif persona == "low_trust":
            agent.self_model.narrative_priors.trust_prior = -0.65
            agent.self_model.narrative_priors.trauma_bias = 0.20
            agent.self_model.personality_profile.agreeableness = 0.28
            agent.self_model.personality_profile.neuroticism = 0.58
        elif persona == "high_trauma":
            agent.self_model.narrative_priors.trust_prior = -0.20
            agent.self_model.narrative_priors.trauma_bias = 0.92
            agent.self_model.personality_profile.agreeableness = 0.32
            agent.self_model.personality_profile.neuroticism = 0.88
        else:
            raise ValueError(f"Unknown persona: {persona}")

        turns = run_conversation(
            agent,
            lines,
            observer=observer,
            generator=RuleBasedGenerator(),
            master_seed=int(seed),
            partner_uid=2,
            session_id=f"corr-{persona}-{seed}",
        )
        counts.update(t.action or "" for t in turns if t.action)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M5.3 persona-action chi-square artifact.")
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--seed-end", type=int, default=1009, help="exclusive upper bound")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/m53_personality_action_correlation.json"),
    )
    args = parser.parse_args()

    lines = [
        "你好，我们先聊聊最近的状态。",
        "我最近压力有点大。",
        "你觉得我应该先做什么？",
        "其实我也有点生气。",
        "我不太确定你是不是在敷衍我。",
        "那你再解释具体一点。",
        "好吧，也许是我太敏感了。",
        "我们能不能换个角度看？",
        "你会怎么做？",
        "最后给我一个建议。",
    ]
    personas = ("high_trust", "low_trust", "high_trauma")
    seeds = range(int(args.seed_start), int(args.seed_end))
    observer = DialogueObserver()

    per_group: dict[str, Counter[str]] = {
        persona: _persona_counts(persona, seeds=seeds, lines=lines, observer=observer)
        for persona in personas
    }
    columns = [a for a in DIALOGUE_ACTION_NAMES if sum(per_group[p][a] for p in personas) > 0]
    rows = len(personas)
    cols = len(columns)
    grand_total = sum(per_group[p][a] for p in personas for a in columns)

    row_totals = {p: sum(per_group[p][a] for a in columns) for p in personas}
    col_totals = {a: sum(per_group[p][a] for p in personas) for a in columns}
    chi2 = 0.0
    for persona in personas:
        for action in columns:
            observed = float(per_group[persona][action])
            expected = float(row_totals[persona] * col_totals[action]) / float(grand_total)
            if expected > 0.0:
                diff = observed - expected
                chi2 += (diff * diff) / expected

    df = (rows - 1) * (cols - 1)
    critical_0_05 = _chi_square_critical_0_05(df)
    p_value = _chi_square_p_value(chi2=chi2, df=df)
    payload = {
        "milestone": "M5.3",
        "artifact": "m53_personality_action_correlation",
        "sample_config": {
            "personas": list(personas),
            "seed_start_inclusive": int(args.seed_start),
            "seed_end_exclusive": int(args.seed_end),
            "num_seeds": int(args.seed_end) - int(args.seed_start),
            "turns_per_seed": len(lines),
            "dialogue_lines": lines,
        },
        "action_columns_with_observations": columns,
        "persona_action_counts": {
            persona: {action: int(per_group[persona][action]) for action in DIALOGUE_ACTION_NAMES}
            for persona in personas
        },
        "chi_square": {
            "chi2": chi2,
            "df": df,
            "p_value": p_value,
            "critical_0_05": critical_0_05,
            "significant_at_0_05_using_critical_table": bool(chi2 > critical_0_05),
            "significant_at_0_05_using_p_value": bool(p_value < 0.05),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
