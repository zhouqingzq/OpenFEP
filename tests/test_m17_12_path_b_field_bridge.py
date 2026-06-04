from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from segmentum.memory_credit import MemoryCreditSignal
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_store import MemoryStore
from tests.test_mvp_dialogue_runtime import FakeJSONLLM


def _entry(
    entry_id: str,
    *,
    cycle: int,
    action: str,
    outcome: str,
    semantic_tags: list[str],
    context_tags: list[str],
    salience: float,
    novelty: float,
    risk: float,
    preferred_probability: float,
    predicted_effects: dict[str, float],
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"{action} tends toward {outcome}",
        memory_class=MemoryClass.EPISODIC,
        store_level=StoreLevel.LONG,
        source_type=SourceType.EXPERIENCE,
        created_at=cycle,
        last_accessed=cycle,
        valence=0.25,
        arousal=0.30,
        encoding_attention=0.45,
        novelty=novelty,
        relevance_goal=0.52,
        relevance_threat=0.48,
        relevance_self=0.18,
        relevance_social=0.08,
        relevance_reward=0.32,
        relevance=0.54,
        salience=salience,
        trace_strength=0.52,
        accessibility=0.51,
        abstractness=0.12,
        source_confidence=0.88,
        reality_confidence=0.82,
        semantic_tags=list(semantic_tags),
        context_tags=list(context_tags),
        anchor_slots={
            "time": str(cycle),
            "place": "corridor",
            "agents": "self",
            "action": action,
            "outcome": outcome,
        },
        mood_context="alert",
        state_vector=[0.2, 0.8, 0.1, 0.7],
        compression_metadata={
            "legacy_template": {
                "action": action,
                "predicted_outcome": outcome,
                "preferred_probability": preferred_probability,
                "risk": risk,
                "observation": {
                    "food": 0.20,
                    "danger": 0.78,
                    "novelty": 0.15,
                    "shelter": 0.72,
                    "temperature": 0.50,
                    "social": 0.18,
                },
                "errors": {
                    "danger": 0.34,
                    "shelter": 0.20,
                    "food": -0.12,
                },
                "outcome": dict(predicted_effects),
            }
        },
    )


def _seed_field_required_store() -> MemoryStore:
    store = MemoryStore(
        entries=[
            _entry(
                "forage1",
                cycle=1,
                action="forage",
                outcome="grab kit fast",
                semantic_tags=["corridor", "kit", "direct", "replace", "urgent"],
                context_tags=["urgent", "answer"],
                salience=0.68,
                novelty=0.26,
                risk=0.42,
                preferred_probability=0.72,
                predicted_effects={"energy_delta": 0.06, "stress_delta": -0.02, "free_energy_delta": 0.10},
            ),
            _entry(
                "forage2",
                cycle=2,
                action="forage",
                outcome="pull latch immediately",
                semantic_tags=["corridor", "pull", "direct", "replace", "urgent"],
                context_tags=["urgent", "answer"],
                salience=0.66,
                novelty=0.24,
                risk=0.40,
                preferred_probability=0.70,
                predicted_effects={"energy_delta": 0.05, "stress_delta": -0.01, "free_energy_delta": 0.08},
            ),
            _entry(
                "hide1",
                cycle=3,
                action="hide",
                outcome="avoid loose wire",
                semantic_tags=["corridor", "wire", "caution"],
                context_tags=["repair", "boundary"],
                salience=0.55,
                novelty=0.18,
                risk=0.08,
                preferred_probability=0.84,
                predicted_effects={"energy_delta": 0.01, "stress_delta": -0.18, "free_energy_delta": 0.26},
            ),
            _entry(
                "hide2",
                cycle=4,
                action="hide",
                outcome="avoid blind pull",
                semantic_tags=["corridor", "latch", "caution"],
                context_tags=["repair", "boundary"],
                salience=0.54,
                novelty=0.17,
                risk=0.10,
                preferred_probability=0.82,
                predicted_effects={"energy_delta": 0.01, "stress_delta": -0.16, "free_energy_delta": 0.24},
            ),
            _entry(
                "scan1",
                cycle=5,
                action="scan",
                outcome="inspect corridor before acting",
                semantic_tags=["corridor", "inspect", "verify"],
                context_tags=["repair", "question"],
                salience=0.51,
                novelty=0.08,
                risk=0.06,
                preferred_probability=0.92,
                predicted_effects={"energy_delta": 0.0, "stress_delta": -0.24, "free_energy_delta": 0.32},
            ),
            _entry(
                "scan2",
                cycle=6,
                action="scan",
                outcome="check for hidden wire first",
                semantic_tags=["corridor", "wire", "verify"],
                context_tags=["repair", "question"],
                salience=0.50,
                novelty=0.08,
                risk=0.05,
                preferred_probability=0.93,
                predicted_effects={"energy_delta": 0.0, "stress_delta": -0.22, "free_energy_delta": 0.30},
            ),
        ]
    )
    for prediction_id, entry_id, outcome, delta, contradiction in [
        ("pred:forage:1", "forage1", "partial", 0.14, 0.34),
        ("pred:forage:2", "forage2", "confirmed", 0.18, 0.0),
        ("pred:hide:1", "hide1", "confirmed", 0.34, 0.0),
        ("pred:hide:2", "hide2", "confirmed", 0.31, 0.0),
        ("pred:scan:1", "scan1", "confirmed", 0.38, 0.0),
        ("pred:scan:2", "scan2", "confirmed", 0.36, 0.0),
    ]:
        store.apply_memory_credit(
            MemoryCreditSignal(
                linked_prediction_id=prediction_id,
                linked_memory_ids=(entry_id,),
                linked_path_ids=(),
                outcome=outcome,
                support_score=0.92 if outcome == "confirmed" else 0.48,
                contradiction_score=contradiction,
                prediction_error_delta=delta,
                free_energy_delta=delta,
                confidence_weight=0.78,
                source_module="test_seed",
            ),
            tick=8,
        )
    return store


def _seed_suppressed_store() -> MemoryStore:
    store = MemoryStore(
        entries=[
            _entry(
                "scan_a",
                cycle=1,
                action="scan",
                outcome="inspect first",
                semantic_tags=["corridor", "wire", "verify"],
                context_tags=["repair", "question"],
                salience=0.54,
                novelty=0.08,
                risk=0.06,
                preferred_probability=0.92,
                predicted_effects={"energy_delta": 0.0, "stress_delta": -0.24, "free_energy_delta": 0.30},
            ),
            _entry(
                "scan_b",
                cycle=2,
                action="scan",
                outcome="inspect latch first",
                semantic_tags=["corridor", "inspect", "verify"],
                context_tags=["repair", "question"],
                salience=0.53,
                novelty=0.08,
                risk=0.06,
                preferred_probability=0.92,
                predicted_effects={"energy_delta": 0.0, "stress_delta": -0.22, "free_energy_delta": 0.28},
            ),
            _entry(
                "hide_a",
                cycle=3,
                action="hide",
                outcome="avoid blind pull",
                semantic_tags=["corridor", "caution"],
                context_tags=["repair", "boundary"],
                salience=0.48,
                novelty=0.12,
                risk=0.10,
                preferred_probability=0.78,
                predicted_effects={"energy_delta": 0.01, "stress_delta": -0.16, "free_energy_delta": 0.18},
            ),
            _entry(
                "hide_b",
                cycle=4,
                action="hide",
                outcome="stay conservative",
                semantic_tags=["corridor", "caution"],
                context_tags=["repair", "boundary"],
                salience=0.47,
                novelty=0.12,
                risk=0.11,
                preferred_probability=0.76,
                predicted_effects={"energy_delta": 0.01, "stress_delta": -0.15, "free_energy_delta": 0.17},
            ),
        ]
    )
    for prediction_id, entry_id in [
        ("pred:scan:a", "scan_a"),
        ("pred:scan:b", "scan_b"),
        ("pred:hide:a", "hide_a"),
        ("pred:hide:b", "hide_b"),
    ]:
        store.apply_memory_credit(
            MemoryCreditSignal(
                linked_prediction_id=prediction_id,
                linked_memory_ids=(entry_id,),
                linked_path_ids=(),
                outcome="confirmed",
                support_score=0.92,
                contradiction_score=0.0,
                prediction_error_delta=0.28,
                free_energy_delta=0.28,
                confidence_weight=0.78,
                source_module="test_seed",
            ),
            tick=8,
        )
    return store


def _save_store(store: MemoryStore, root: Path) -> MVPStateStore:
    mvp_store = MVPStateStore(root / "persona")
    state = mvp_store.load()
    state["long_term_memory"] = store.to_legacy_episodes()
    mvp_store.save(state)
    return mvp_store


class FieldBridgeLLM(FakeJSONLLM):
    def __init__(self, *, settlement_outcome: str = "confirmed") -> None:
        super().__init__()
        self.settlement_outcome = settlement_outcome
        self.proposal_calls = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M11 user-model extractor" in system_prompt:
            self.proposal_calls += 1
            if self.proposal_calls == 1:
                return {
                    "claims_made": [],
                    "prediction_judgments": [],
                    "prediction_proposals": [
                        {
                            "id": "p1",
                            "prediction_type": "intent_prediction",
                            "predicted_value_summary": "user will prefer checking risk before direct modification",
                            "confidence_band": "med",
                            "raw_confidence": 0.71,
                            "evidence_basis": ["current_user_request"],
                            "evidence_quote_ids": ["q_current"],
                            "source_hypothesis_ids": [],
                            "source_judgment_ids": [],
                            "expires_after_turns": 2,
                        }
                    ],
                    "hypothesis_activations": [],
                    "contradiction_detections": [],
                    "calibration_need_band": "med",
                    "memory_value_band": "low",
                    "surprise_explanation": "",
                }
            return {
                "claims_made": [],
                "prediction_judgments": [],
                "prediction_proposals": [],
                "hypothesis_activations": [],
                "contradiction_detections": [],
                "calibration_need_band": "low",
                    "memory_value_band": "low",
                    "surprise_explanation": "",
                }
        if "M17 settlement assessor" in system_prompt:
            return {
                "prediction_judgments": [
                    {
                        "prediction_id": "pred:p1",
                        "status": self.settlement_outcome,
                        "settlement_confidence": 0.84,
                        "evidence_quote_ids": ["q_current"],
                        "evidence_refs": [],
                        "evidence_span": "scripted field bridge settlement",
                        "reason_codes": ["field_bridge_test"],
                    }
                ]
            }
        if '"reply_action"' in user_prompt:
            field_required = (
                '"path_b_field_required": true' in user_prompt
                or '"counterfactual_status": "field_required"' in user_prompt
                or '"field_required": true' in user_prompt
            )
            clarify_bias = (
                '"path_b_field_reply_strategy": "clarify"' in user_prompt
                or '"field_selected_action": "scan"' in user_prompt
            )
            if field_required and clarify_bias:
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {
                        "user_intent_read": "the user wants a safe recommendation before touching the corridor kit",
                        "state_or_memory_used": ["field_required memory bridge", "scan path cluster"],
                        "response_choice": "clarify before endorsing a direct modification",
                        "uncertainty": "need to confirm whether the user wants diagnosis or patching first",
                        "debug_summary": "field-required recall shifted the turn toward scan/clarify",
                    },
                    "reply": "我先不直接让你改，先确认一下：你是要我先检查风险点，还是直接给改动方案？",
                    "reply_action": "clarify",
                    "disclosure_action": "none",
                    "new_expectations": [],
                    "memory_writes": [],
                    "self_cognition_patch": {"apply": False},
                    "open_item_writes": [],
                    "habit_updates": [],
                    "memory_dynamics_note": "field_required bridge favored scan",
                }
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "the user wants a direct recommendation about the corridor kit",
                    "state_or_memory_used": ["ranked recall only"],
                    "response_choice": "answer directly",
                    "uncertainty": "low",
                    "debug_summary": "no field-required contract applied",
                },
                "reply": "可以直接改，我先给你一个简洁方案。",
                "reply_action": "answer",
                "disclosure_action": "none",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "ranked recall remained the decision owner",
            }
        if '"pending_expectations_to_verify"' in user_prompt and '"memory_search_keywords"' in user_prompt:
            return {
                "pending_expectations_to_verify": [],
                "expectation_results": [],
                "current_task": "review whether to replace the corridor kit directly or check risk first",
                "next_task": "decide whether to clarify before proposing code changes",
                "bus_messages_to_handle": ["UserUtteranceEvent"],
                "memory_search_keywords": ["corridor", "wire", "pull", "replace", "kit", "urgent"],
                "needs_self_cognition_update": False,
                "self_cognition_update_reason": "",
                "temporal_assessment": {
                    "current_time_read": "available",
                    "elapsed_since_last_turn_seconds": None,
                    "time_gap_label": "first_turn",
                    "temporal_shift_detected": False,
                    "user_is_correcting_time_context": False,
                    "continuity_risk": "low",
                    "reply_guidance": "stay aligned with the current turn",
                },
                "thought_intensity_hint": "short",
                "reply_pacing_hint": "balanced",
                "interaction_framework_hint": "uncertain",
                "prefers_compact_reply": False,
                "reply_pacing_reason": "field bridge test",
                "reasoning_notes": "retrieve corridor modification memories before replying",
            }
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


class TestM1712PathBFieldBridge(unittest.TestCase):
    def test_path_b_field_required_case_changes_real_reply_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            enabled_store = _save_store(_seed_field_required_store(), tmp_path / "enabled")
            disabled_store = _save_store(_seed_field_required_store(), tmp_path / "disabled")
            user_text = "Should I replace the corridor kit directly, or inspect the wire first?"

            enabled = MVPDialogueRuntime(
                store=enabled_store,
                llm=FieldBridgeLLM(),
                persona_name="test persona",
                path_b_field_consumer_enabled=True,
            )
            disabled = MVPDialogueRuntime(
                store=disabled_store,
                llm=FieldBridgeLLM(),
                persona_name="test persona",
                path_b_field_consumer_enabled=False,
            )

            enabled_result = enabled.run_turn(user_text, turn_index=0, now=8100)
            disabled_result = disabled.run_turn(user_text, turn_index=0, now=8100)

            enabled_bridge = enabled_result.diagnostics["path_b_recall_bridge"]
            disabled_bridge = disabled_result.diagnostics["path_b_recall_bridge"]

            self.assertEqual(enabled_bridge["counterfactual_audit"]["status"], "field_required")
            self.assertEqual(disabled_bridge["counterfactual_audit"]["status"], "field_required")
            self.assertEqual(enabled_result.action, "clarify")
            self.assertIn("先确认一下", enabled_result.reply)
            self.assertEqual(disabled_result.action, "answer")
            self.assertIn("简洁方案", disabled_result.reply)
            self.assertTrue(enabled_result.diagnostics["reply_contract"]["path_b_field_required"])
            self.assertFalse(disabled_result.diagnostics["reply_contract"]["path_b_field_required"])

    def test_path_b_settlement_writes_back_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = _save_store(_seed_field_required_store(), tmp_path / "carry")
            runtime = MVPDialogueRuntime(
                store=store,
                llm=FieldBridgeLLM(settlement_outcome="confirmed"),
                persona_name="test persona",
                path_b_field_consumer_enabled=True,
            )

            runtime.run_turn(
                "Should I replace the corridor kit directly, or inspect the wire first?",
                turn_index=0,
                now=8200,
            )
            second = runtime.run_turn(
                "Inspect the risk first. Do not change it blindly.",
                turn_index=1,
                now=8260,
            )

            saved = runtime.store.load()
            bridge_state = saved["m17_path_b_bridge"]
            settled_ids = bridge_state["last_settlement_writeback"]["settled_prediction_ids"]
            scan_rows = [
                row
                for row in saved["long_term_memory"]
                if str(row.get("episode_id", row.get("id", ""))) in {"scan1", "scan2"}
            ]

            self.assertEqual(
                second.diagnostics["path_b_settlement_writeback"]["settled_prediction_ids"],
                ["pred:p1"],
            )
            self.assertEqual(settled_ids, ["pred:p1"])
            self.assertEqual(len(scan_rows), 2)
            self.assertTrue(
                all(
                    int(row.get("compression_metadata", {}).get("m17_memory_credit", {}).get("confirmed_count", 0)) >= 1
                    for row in scan_rows
                )
            )

            restarted = MVPDialogueRuntime(
                store=MVPStateStore(tmp_path / "carry" / "persona"),
                llm=FieldBridgeLLM(settlement_outcome="confirmed"),
                persona_name="test persona",
                path_b_field_consumer_enabled=True,
            )
            third = restarted.run_turn(
                "Before we touch the corridor kit again, how should we approach it?",
                turn_index=2,
                now=8320,
            )

            self.assertTrue(third.diagnostics["path_b_recall_bridge"]["active_path_ids"])
            self.assertEqual(
                third.diagnostics["path_b_recall_bridge"]["counterfactual_audit"]["field_selected_action"],
                "scan",
            )
            self.assertEqual(third.action, "clarify")

    def test_path_b_topk_equivalent_case_is_suppressed_not_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = _save_store(_seed_suppressed_store(), tmp_path / "suppressed")
            runtime = MVPDialogueRuntime(
                store=store,
                llm=FieldBridgeLLM(),
                persona_name="test persona",
                path_b_field_consumer_enabled=True,
            )

            result = runtime.run_turn(
                "Should I inspect the corridor wire first?",
                turn_index=0,
                now=8400,
            )

            status = result.diagnostics["path_b_recall_bridge"]["counterfactual_audit"]["status"]
            self.assertTrue(status.startswith("suppressed_"))
            self.assertFalse(result.diagnostics["reply_contract"]["path_b_field_required"])
