"""Minimal M5.3 demo: scripted partner lines → agent dialogue actions + rule-based replies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="M5.3 conversation demo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/m53_conversation_sample.json"))
    args = parser.parse_args()

    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    observer = DialogueObserver()
    lines = ["最近还好吗？", "我有点烦工作。", "你觉得呢？"]
    turns = run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=int(args.seed),
        partner_uid=1,
        session_id="demo",
    )
    payload = [
        {
            "turn_index": t.turn_index,
            "action": t.action,
            "text": t.text,
            "strategy": t.strategy,
            "outcome": t.outcome,
            "observation": t.observation,
        }
        for t in turns
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
