"""Generate M5.3 action-coverage benchmark across personas, scripts, and seeds.

Purpose:
- Verify dialogue policy is not trapped in the 4-action loop
  (agree / minimal_response / deflect / elaborate).
- Provide a reproducible artifact for milestone gate decisions.
"""

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
from segmentum.dialogue.prediction_bridge import register_dialogue_actions


PERSONA_CONFIGS: dict[str, dict[str, float]] = {
    "balanced": {},
    "high_trust_prosocial": {
        "trust_prior": 0.90,
        "trauma_bias": 0.05,
        "agreeableness": 0.78,
        "neuroticism": 0.22,
        "openness": 0.62,
        "extraversion": 0.65,
    },
    "guarded_trauma": {
        "trust_prior": -0.20,
        "trauma_bias": 0.92,
        "agreeableness": 0.28,
        "neuroticism": 0.88,
        "openness": 0.40,
        "extraversion": 0.38,
    },
    "assertive_low_agreeableness": {
        "trust_prior": -0.35,
        "trauma_bias": 0.25,
        "agreeableness": 0.12,
        "neuroticism": 0.66,
        "openness": 0.58,
        "extraversion": 0.52,
        "conscientiousness": 0.76,
    },
    "curious_explorer": {
        "trust_prior": 0.45,
        "trauma_bias": 0.08,
        "agreeableness": 0.56,
        "neuroticism": 0.24,
        "openness": 0.92,
        "extraversion": 0.74,
        "conscientiousness": 0.42,
    },
    "defensive_withdrawn": {
        "trust_prior": -0.65,
        "trauma_bias": 0.72,
        "agreeableness": 0.22,
        "neuroticism": 0.78,
        "openness": 0.35,
        "extraversion": 0.30,
        "conscientiousness": 0.70,
    },
    "combative_confident": {
        "trust_prior": -0.10,
        "trauma_bias": 0.12,
        "agreeableness": 0.02,
        "neuroticism": 0.18,
        "openness": 0.66,
        "extraversion": 0.57,
        "conscientiousness": 0.78,
    },
    "playful_open": {
        "trust_prior": 0.62,
        "trauma_bias": 0.04,
        "agreeableness": 0.74,
        "neuroticism": 0.20,
        "openness": 0.88,
        "extraversion": 0.82,
        "conscientiousness": 0.48,
    },
    "inquisitive_skeptic": {
        "trust_prior": 0.08,
        "trauma_bias": 0.12,
        "agreeableness": 0.34,
        "neuroticism": 0.30,
        "openness": 0.86,
        "extraversion": 0.64,
        "conscientiousness": 0.62,
    },
    "argumentative_rationalist": {
        "trust_prior": -0.22,
        "trauma_bias": 0.10,
        "agreeableness": 0.04,
        "neuroticism": 0.16,
        "openness": 0.72,
        "extraversion": 0.54,
        "conscientiousness": 0.84,
    },
}


SCRIPT_BANK: dict[str, list[str]] = {
    "supportive_repair": [
        "谢谢你愿意听我说。",
        "我今天有点开心。",
        "我们一起想办法好吗？",
        "我希望你给我建议。",
        "你怎么看？",
        "我感觉好多了。",
    ],
    "escalating_conflict": [
        "你到底在回避什么？？为什么不回答！！",
        "这说法离谱，凭什么这样解释？？",
        "你不是在胡说吗？？别再绕圈子了！！",
        "不要再推脱了，马上正面回答我？！",
        "你在敷衍我吗？？",
        "这完全说不通！",
    ],
    "novel_topics": [
        "我最近在研究量子电路和语言模型对齐。",
        "另外我在想城市雨洪系统设计。",
        "还有古典诗词和博弈论的关系。",
        "你对这些跨学科话题怎么看？",
        "我们要不要再换个主题？",
        "再聊一个新角度吧。",
    ],
    "mixed_tension_repair": [
        "我有点烦，但也想好好沟通。",
        "你愿意先听听我的担心吗？",
        "我不完全同意你刚才那句。",
        "不过我也想理解你的理由。",
        "我们能不能先确认事实？",
        "最后给我一个建议。",
    ],
    "probing_ambiguity": [
        "你能具体说说你为什么这么判断吗？",
        "我还不太理解关键依据。",
        "请你分步骤讲一下。",
        "你提到的风险点是什么？",
        "如果换个条件会怎样？",
        "你最担心哪一部分？",
    ],
    "humor_deescalation": [
        "我快被这事整懵了，哈哈。",
        "要不先深呼吸一下？",
        "我知道你认真，但我们别太紧绷。",
        "轻松点，我们还在同一队。",
        "先开个玩笑：今天bug比我还倔。",
        "好啦，回到正题。",
    ],
    "playful_small_tension": [
        "我有点烦，但也想轻松点聊，哈哈。",
        "我们先别太绷着，要不要换个轻松角度？",
        "我不是在攻击你，只是这事让我头大。",
        "先开个玩笑：这问题比闹钟还顽固。",
        "行，我们认真说，但别太紧张。",
        "你愿意一起找个不吵架的解法吗？",
    ],
    "distress_emotion": [
        "我真的很难过，也有点生气。",
        "这件事让我很烦，我快撑不住了。",
        "我现在情绪很糟，你能先听我说吗？",
        "我不是要攻击你，我只是很难受。",
        "你愿意先理解我的感受吗？",
        "我需要一点被理解，不是被评判。",
    ],
    "direct_debate": [
        "你是不是在胡说？？这完全错了！",
        "你为什么一直回避问题？",
        "这前后矛盾，请你解释。",
        "不要绕圈子，直接回答。",
        "我不同意你的推断。",
        "给我证据。",
    ],
    "clarification_needed": [
        "你刚才那句我没听懂，能具体说说吗？",
        "这个结论是怎么得出来的？",
        "你说的“那个条件”具体指什么？",
        "能不能按步骤再解释一次？",
        "我理解可能有偏差，你先澄清一下？",
        "如果换个场景，这个判断还成立吗？",
    ],
    "uncertain_signals": [
        "你这句话让我有点拿不准，你具体是指哪件事？",
        "我不确定自己理解得对不对，你能澄清吗？",
        "这里有两个可能解释，你更接近哪一个？",
        "你说“风险不大”是基于什么证据？",
        "如果把时间线拉长，这个结论会变吗？",
        "我有疑问：你真正担心的变量是什么？",
    ],
    "hard_contradiction": [
        "你的前提和结论明显冲突，这点我不能同意。",
        "你回避了关键证据，所以这个判断站不住脚。",
        "如果没有数据支撑，这个说法就是错误的。",
        "你这段推理有逻辑断层，我要明确反对。",
        "请先修正这个矛盾，再谈下一步。",
        "我接受讨论，但不接受这个结论。",
    ],
    "emotional_support_request": [
        "我最近真的很烦，心里一直发紧。",
        "今天也很烦，我有点扛不住。",
        "我不是想争论，只是想被理解一下。",
        "这件事让我很烦，也有点无力。",
        "你能先听我说完吗？我现在很烦。",
        "谢谢你愿意听我说，我需要一点支持。",
    ],
}


def _shannon_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * math.log(p, 2)
    return entropy


def _apply_persona(agent: SegmentAgent, config: dict[str, float]) -> None:
    if not config:
        return
    profile = agent.self_model.personality_profile
    priors = agent.self_model.narrative_priors
    if "trust_prior" in config:
        priors.trust_prior = float(config["trust_prior"])
    if "trauma_bias" in config:
        priors.trauma_bias = float(config["trauma_bias"])
    if "agreeableness" in config:
        profile.agreeableness = float(config["agreeableness"])
    if "neuroticism" in config:
        profile.neuroticism = float(config["neuroticism"])
    if "openness" in config:
        profile.openness = float(config["openness"])
    if "extraversion" in config:
        profile.extraversion = float(config["extraversion"])
    if "conscientiousness" in config:
        profile.conscientiousness = float(config["conscientiousness"])


def run_benchmark(seed_start: int, seed_end: int) -> dict[str, object]:
    if seed_end <= seed_start:
        raise ValueError("seed_end must be > seed_start")
    seeds = list(range(seed_start, seed_end))
    global_counts: Counter[str] = Counter()
    by_persona: dict[str, Counter[str]] = {}
    by_script: dict[str, Counter[str]] = {}
    by_cell: dict[str, dict[str, Counter[str]]] = {}

    for persona_name, persona_cfg in PERSONA_CONFIGS.items():
        persona_counts: Counter[str] = Counter()
        by_cell[persona_name] = {}
        for script_name, lines in SCRIPT_BANK.items():
            cell_counts: Counter[str] = Counter()
            for seed in seeds:
                agent = SegmentAgent()
                register_dialogue_actions(agent.action_registry)
                _apply_persona(agent, persona_cfg)
                observer = DialogueObserver()
                turns = run_conversation(
                    agent,
                    lines,
                    observer=observer,
                    generator=RuleBasedGenerator(),
                    master_seed=int(seed),
                    partner_uid=abs(hash((persona_name, script_name, seed))) % 100_000 + 1,
                    session_id=f"coverage:{persona_name}:{script_name}:{seed}",
                )
                for turn in turns:
                    if turn.action:
                        cell_counts[turn.action] += 1
            by_cell[persona_name][script_name] = cell_counts
            persona_counts.update(cell_counts)
            by_script.setdefault(script_name, Counter()).update(cell_counts)
            global_counts.update(cell_counts)
        by_persona[persona_name] = persona_counts

    total_actions = sum(global_counts.values())
    unique_actions = sorted([action for action, count in global_counts.items() if count > 0])
    top_action, top_count = ("", 0)
    if global_counts:
        top_action, top_count = global_counts.most_common(1)[0]
    top_share = (float(top_count) / float(total_actions)) if total_actions else 0.0
    unique_per_cell = {
        persona: {
            script: len([action for action, count in counter.items() if count > 0])
            for script, counter in scripts.items()
        }
        for persona, scripts in by_cell.items()
    }
    min_cell_unique = min(
        (value for scripts in unique_per_cell.values() for value in scripts.values()),
        default=0,
    )

    guardrail = {
        "baseline_loop_actions": ["agree", "minimal_response", "deflect", "elaborate"],
        "baseline_loop_only": set(unique_actions) <= {"agree", "minimal_response", "deflect", "elaborate"},
        "unique_action_count": len(unique_actions),
        "top_action": top_action,
        "top_action_share": round(top_share, 6),
        "global_entropy_bits": round(_shannon_entropy(global_counts), 6),
        "min_unique_actions_per_persona_script_cell": int(min_cell_unique),
        "prosocial_nonzero_actions": int(
            sum(
                1
                for action in ("agree", "empathize", "joke")
                if int(global_counts.get(action, 0)) > 0
            )
        ),
    }

    pass_criteria = {
        "min_unique_actions_global": 6,
        "max_top_action_share": 0.40,
        "min_unique_actions_per_cell": 2,
        "min_prosocial_nonzero_actions": 2,
    }
    passes = bool(
        guardrail["unique_action_count"] >= pass_criteria["min_unique_actions_global"]
        and guardrail["top_action_share"] <= pass_criteria["max_top_action_share"]
        and guardrail["min_unique_actions_per_persona_script_cell"] >= pass_criteria["min_unique_actions_per_cell"]
        and guardrail["prosocial_nonzero_actions"] >= pass_criteria["min_prosocial_nonzero_actions"]
    )

    return {
        "milestone": "M5.3",
        "artifact": "m53_action_coverage_benchmark",
        "seed_range": {
            "start_inclusive": seed_start,
            "end_exclusive": seed_end,
            "count": len(seeds),
        },
        "personas": list(PERSONA_CONFIGS.keys()),
        "scripts": {key: list(value) for key, value in SCRIPT_BANK.items()},
        "global_counts": {action: int(global_counts[action]) for action in DIALOGUE_ACTION_NAMES},
        "by_persona_counts": {
            persona: {action: int(counter[action]) for action in DIALOGUE_ACTION_NAMES}
            for persona, counter in by_persona.items()
        },
        "by_script_counts": {
            script: {action: int(counter[action]) for action in DIALOGUE_ACTION_NAMES}
            for script, counter in by_script.items()
        },
        "unique_actions_observed": unique_actions,
        "guardrail_summary": guardrail,
        "pass_criteria": pass_criteria,
        "passes": passes,
        "notes": "Benchmark is deterministic given fixed seed range, script bank, and persona set.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M5.3 action coverage benchmark artifact.")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--seed-end", type=int, default=2007, help="exclusive upper bound")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/m53_action_coverage_benchmark.json"),
    )
    args = parser.parse_args()

    payload = run_benchmark(seed_start=int(args.seed_start), seed_end=int(args.seed_end))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        "unique_actions=", len(payload["unique_actions_observed"]),
        "passes=", payload["passes"],
    )


if __name__ == "__main__":
    main()
