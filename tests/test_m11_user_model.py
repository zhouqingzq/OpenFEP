import importlib
import inspect
import json
import random
from pathlib import Path

import pytest

from segmentum.user_model import (
    M11RuntimeConfig,
    M11RuntimeState,
    UserModel,
    compose_value,
    run_m11_turn,
    validate_extractor_output,
)
from segmentum.user_model.evidence_cards import evidence_cards_from_user_model, prompt_safe_cards
from segmentum.user_model.hyperparams import DEFAULT_HYPERPARAMS
from segmentum.user_model.llm_extractor import ExtractorValidationError, EXTRACTOR_OUTPUT_SCHEMA, EXTRACTOR_PROMPT_TEMPLATE
from segmentum.user_model.prediction_ledger import UserPredictionLedger, apply_prediction_updates
from segmentum.user_model.reliability_ledger import (
    ReliabilityJudgment,
    SourceReliabilityLedger,
    update_reliability,
)
from segmentum.user_model.user_model import apply_claims_to_user_model


def _claim(idx: int, *, domain: str = "self_reported_preferences", modality: str = "factual", band: str = "high") -> dict[str, object]:
    return {
        "id": f"claim:{idx}",
        "domain": domain,
        "modality": modality,
        "content_summary": "prefers concise replies",
        "evidence_quote_ids": [f"q{idx}"],
        "confidence_band": band,
    }


def _extractor(*, claims=None, judgments=None, proposals=None, contradictions=None, memory_value_band="high"):
    payload = {
        "claims_made": claims or [],
        "prediction_judgments": judgments or [],
        "prediction_proposals": proposals or [],
        "hypothesis_activations": [],
        "contradiction_detections": contradictions or [],
        "calibration_need_band": "med",
        "memory_value_band": memory_value_band,
        "surprise_explanation": "diagnostic phrase that must not reach prompt",
    }
    return lambda snapshot: payload


def test_m11_does_not_import_memory_dynamics_or_value_memory_or_memory_retrieval():
    root = Path("segmentum/user_model")
    forbidden = ("memory_dynamics", "value_memory", "memory_retrieval")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"segmentum.{name}" in text or f"..{name}" in text or f"import {name}" in text for name in forbidden), path
    importlib.import_module("segmentum.user_model")
    for module in ("segmentum.memory_dynamics", "segmentum.value_memory", "segmentum.memory_retrieval"):
        importlib.sys.modules.pop(module, None)
    importlib.import_module("segmentum.user_model.m11_runtime")


def test_extractor_schema_has_no_number_or_float_fields():
    text = json.dumps(EXTRACTOR_OUTPUT_SCHEMA, sort_keys=True) + EXTRACTOR_PROMPT_TEMPLATE
    forbidden = ("float", "number", "value_score", "source_reliability", "prediction_error", "retrieval_score")
    assert not any(word in text for word in forbidden)


def test_extractor_output_with_any_float_is_rejected():
    payload = _extractor(claims=[{**_claim(1), "confidence": 0.8}])({})
    with pytest.raises(ExtractorValidationError):
        validate_extractor_output(payload)


def test_extractor_output_with_unknown_field_is_rejected():
    payload = _extractor()({})
    payload["extra"] = "nope"
    with pytest.raises(ExtractorValidationError):
        validate_extractor_output(payload)


def test_extractor_output_referencing_unknown_prediction_id_is_rejected():
    payload = _extractor(judgments=[{"prediction_id": "pred:missing", "status": "confirmed", "evidence_quote_ids": ["q"]}])({})
    with pytest.raises(ExtractorValidationError):
        validate_extractor_output(payload, snapshot_prediction_ids={"pred:known"})


def test_user_model_distinguishes_claim_hypothesis_fact_and_unknown():
    model = UserModel(user_id="u")
    model = apply_claims_to_user_model(model, [_claim(1), _claim(2, modality="roleplay")], turn_id="1")
    kinds = {row.claim_kind for row in model.claim_history_summary}
    assert "user_stated_claim" in kinds
    assert "unknown" in kinds
    assert model.preference_hypotheses[0].claim_kind == "inferred_hypothesis"


def test_user_model_round_trip_and_confidence_band_not_float():
    model = apply_claims_to_user_model(UserModel(user_id="u"), [_claim(1)], turn_id="1")
    restored = UserModel.from_json(model.to_json())
    assert restored.to_dict() == model.to_dict()
    assert isinstance(restored.preference_hypotheses[0].confidence_band, str)


def test_low_evidence_personality_inference_stays_in_low_confidence_band():
    model = apply_claims_to_user_model(
        UserModel(user_id="u"),
        [_claim(1, domain="emotional_state", band="high")],
        turn_id="1",
    )
    assert model.cognitive_style_hypotheses[0].confidence_band != "high"


def test_reliability_monotone_under_only_confirmed():
    ledger = SourceReliabilityLedger.empty()
    series = []
    for turn in range(1, 20):
        ledger, _ = update_reliability(
            ledger,
            [ReliabilityJudgment(f"j{turn}", turn, "technical_claims", "confirmed")],
            current_turn_id=turn,
        )
        series.append(ledger.reliability_for("technical_claims").reliability)
    assert series == sorted(series)


def test_reliability_bounded_in_unit_interval():
    for seed in range(100):
        rng = random.Random(seed)
        ledger = SourceReliabilityLedger.empty()
        for turn in range(1, 30):
            status = "confirmed" if rng.random() > 0.35 else "violated"
            ledger, _ = update_reliability(
                ledger,
                [ReliabilityJudgment(f"{seed}:{turn}", turn, "task_requirements", status)],
                current_turn_id=turn,
            )
            value = ledger.reliability_for("task_requirements").reliability
            assert 0.0 <= value <= 1.0


def test_reliability_domain_isolation():
    ledger = SourceReliabilityLedger.empty()
    base = ledger.reliability_for("emotional_state").reliability
    ledger, _ = update_reliability(
        ledger,
        [ReliabilityJudgment("j", 1, "technical_claims", "confirmed")],
        current_turn_id=1,
    )
    assert ledger.reliability_for("emotional_state").reliability == base


def test_reliability_single_lie_cannot_destroy_long_history():
    ledger = SourceReliabilityLedger.empty()
    for turn in range(1, 51):
        ledger, _ = update_reliability(ledger, [ReliabilityJudgment(f"c{turn}", turn, "self_reported_history", "confirmed")], current_turn_id=turn)
    before = ledger.reliability_for("self_reported_history").reliability
    ledger, _ = update_reliability(ledger, [ReliabilityJudgment("lie", 51, "self_reported_history", "violated")], current_turn_id=51)
    after = ledger.reliability_for("self_reported_history").reliability
    assert after > 0.5
    assert before - after <= DEFAULT_HYPERPARAMS.max_delta_per_turn


def test_reliability_per_turn_delta_bound():
    ledger = SourceReliabilityLedger.empty()
    before = ledger.reliability_for("task_requirements").reliability
    ledger, _ = update_reliability(
        ledger,
        [ReliabilityJudgment(f"v{i}", 1, "task_requirements", "violated") for i in range(12)],
        current_turn_id=1,
    )
    after = ledger.reliability_for("task_requirements").reliability
    assert abs(after - before) <= DEFAULT_HYPERPARAMS.max_delta_per_turn


def test_reliability_decay_under_silence():
    ledger = SourceReliabilityLedger.empty()
    for turn in range(1, 20):
        ledger, _ = update_reliability(ledger, [ReliabilityJudgment(f"c{turn}", turn, "technical_claims", "confirmed")], current_turn_id=turn)
    ledger, _ = update_reliability(ledger, [], current_turn_id=20 + (5 * DEFAULT_HYPERPARAMS.reliability_half_life_turns))
    value = ledger.reliability_for("technical_claims").reliability
    assert abs(value - DEFAULT_HYPERPARAMS.prior_mean) <= 0.10


def test_reliability_unaffected_by_roleplay_or_joke_modality():
    ledger = SourceReliabilityLedger.empty()
    before = ledger.reliability_for("self_reported_history").reliability
    ledger, _ = update_reliability(
        ledger,
        [
            ReliabilityJudgment("r", 1, "self_reported_history", "violated", modality="roleplay"),
            ReliabilityJudgment("j", 1, "self_reported_history", "violated", modality="joke"),
        ],
        current_turn_id=1,
    )
    assert ledger.reliability_for("self_reported_history").reliability == before


def test_reliability_is_independent_of_evidence_text_content():
    a, _ = update_reliability(
        SourceReliabilityLedger.empty(),
        [ReliabilityJudgment("x", 1, "technical_claims", "confirmed", evidence_text="alpha")],
        current_turn_id=1,
    )
    b, _ = update_reliability(
        SourceReliabilityLedger.empty(),
        [ReliabilityJudgment("x", 1, "technical_claims", "confirmed", evidence_text="totally different")],
        current_turn_id=1,
    )
    assert a.to_dict() == b.to_dict()


def test_prediction_ledger_records_confirmed_violated_and_uncertain():
    ledger = UserPredictionLedger()
    ledger = apply_prediction_updates(
        ledger,
        turn_id=1,
        proposals=[
            {
                "id": "p1",
                "prediction_type": "intent_prediction",
                "predicted_value_summary": "user will clarify scope",
                "confidence_band": "med",
                "source_hypothesis_ids": [],
                "source_judgment_ids": [],
                "expires_after_turns": 1,
            }
        ],
        judgments=[],
        known_hypothesis_ids=set(),
        known_judgment_ids=set(),
    )
    assert ledger.predictions_by_status(status="pending", current_turn_id=1, last_n_turns=2)
    ledger = apply_prediction_updates(
        ledger,
        turn_id=2,
        proposals=[],
        judgments=[{"prediction_id": "pred:p1", "status": "confirmed", "evidence_quote_ids": ["q"]}],
        known_hypothesis_ids=set(),
        known_judgment_ids=set(),
    )
    assert ledger.predictions_by_status(status="confirmed", current_turn_id=2, last_n_turns=2)


def test_prediction_proposal_acceptance_and_expiration_are_deterministic():
    raw = [
        {
            "id": "p1",
            "prediction_type": "intent_prediction",
            "predicted_value_summary": "one",
            "confidence_band": "med",
            "source_hypothesis_ids": ["h1"],
            "source_judgment_ids": [],
            "expires_after_turns": 1,
        },
        {
            "id": "p2",
            "prediction_type": "intent_prediction",
            "predicted_value_summary": "two",
            "confidence_band": "med",
            "source_hypothesis_ids": ["missing"],
            "source_judgment_ids": [],
            "expires_after_turns": 1,
        },
    ]
    a = apply_prediction_updates(UserPredictionLedger(), turn_id=1, proposals=raw, judgments=[], known_hypothesis_ids={"h1"}, known_judgment_ids=set())
    b = apply_prediction_updates(UserPredictionLedger(), turn_id=1, proposals=raw, judgments=[], known_hypothesis_ids={"h1"}, known_judgment_ids=set())
    assert a.to_dict() == b.to_dict()
    assert [p.rejection_reason for p in a.proposals] == ["", "unknown_source_id"]
    expired = apply_prediction_updates(a, turn_id=3, proposals=[], judgments=[], known_hypothesis_ids={"h1"}, known_judgment_ids=set())
    assert expired.latest_status("pred:p1") == "uncertain"
    assert any(p.rejection_reason == "expired" for p in expired.proposals)


def test_value_score_rises_monotonically_with_reliability_at_fixed_band():
    series = [
        compose_value(
            memory_value_band="med",
            confidence_band="med",
            source_reliability=i / 10,
            recency_weight=1.0,
            contradiction_unresolved=False,
            privacy_or_safety_flag=False,
        ).value_score
        for i in range(11)
    ]
    assert series == sorted(series)


def test_privacy_flag_prevents_long_term_promotion_regardless_of_band():
    out = compose_value(
        memory_value_band="high",
        confidence_band="high",
        source_reliability=1.0,
        recency_weight=1.0,
        contradiction_unresolved=False,
        privacy_or_safety_flag=True,
    )
    assert out.write_target != "long_term_user_model"


def test_contradiction_unresolved_lowers_value_score():
    plain = compose_value(memory_value_band="high", confidence_band="high", source_reliability=0.8, recency_weight=1.0, contradiction_unresolved=False, privacy_or_safety_flag=False)
    contradicted = compose_value(memory_value_band="high", confidence_band="high", source_reliability=0.8, recency_weight=1.0, contradiction_unresolved=True, privacy_or_safety_flag=False)
    assert contradicted.value_score < plain.value_score


def test_prompt_facing_evidence_cards_never_contain_floats_or_extractor_json():
    state = M11RuntimeState.clean(user_id="u")
    for turn in range(1, 4):
        state, _ = run_m11_turn(
            state,
            user_id="u",
            turn_id=turn,
            extractor=_extractor(claims=[_claim(turn)]),
            config=M11RuntimeConfig(m11_user_model_enabled=True),
        )
    cards = evidence_cards_from_user_model(state.user_model)
    prompt = json.dumps(prompt_safe_cards(cards), ensure_ascii=False)
    assert "0." not in prompt
    assert "diagnostic phrase" not in prompt
    assert "surprise_explanation" not in prompt


def test_dynamics_layer_runs_without_any_llm_call_using_stub_extractor():
    state = M11RuntimeState.clean(user_id="u")
    next_state, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=_extractor(claims=[_claim(1)]),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
    )
    assert next_state.user_model.preference_hypotheses
    assert result.evidence_cards


def test_reply_shortens_when_brevity_preference_reliable_and_active():
    state = M11RuntimeState.clean(user_id="u")
    result = None
    for turn in range(1, 6):
        state, result = run_m11_turn(
            state,
            user_id="u",
            turn_id=turn,
            extractor=_extractor(claims=[{**_claim(1), "evidence_quote_ids": [f"q{turn}"]}]),
            config=M11RuntimeConfig(m11_user_model_enabled=True),
        )
    assert any(effect.adjustment == "prefer_shorter_reply" for effect in result.reply_policy_effects)


def test_reply_asks_clarification_when_intent_prediction_just_violated():
    state = M11RuntimeState.clean(user_id="u")
    state, _ = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=_extractor(
            proposals=[
                {
                    "id": "p1",
                    "prediction_type": "intent_prediction",
                    "predicted_value_summary": "user will ask for code",
                    "confidence_band": "med",
                    "source_hypothesis_ids": [],
                    "source_judgment_ids": [],
                    "expires_after_turns": 2,
                }
            ]
        ),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
    )
    state, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=2,
        extractor=_extractor(judgments=[{"prediction_id": "pred:p1", "status": "violated", "evidence_quote_ids": ["q2"]}]),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
    )
    assert any(effect.adjustment == "ask_clarifying_question" for effect in result.reply_policy_effects)


def test_reply_softens_when_domain_reliability_below_threshold():
    state = M11RuntimeState.clean(user_id="u")
    state, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=_extractor(claims=[_claim(1, domain="social_relationship_claims")], contradictions=[{"claim_id": "claim:1", "conflicts_with_memory_id": "", "severity_band": "major"}]),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
    )
    assert any(effect.adjustment == "soften_social_evidence_language" for effect in result.reply_policy_effects)


def test_existing_persona_with_m11_disabled_byte_identical_to_baseline():
    state = M11RuntimeState.clean(user_id="u")
    legacy = [{"id": "m1", "content": "unchanged"}]
    _, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=lambda _: (_ for _ in ()).throw(AssertionError("extractor called")),
        config=M11RuntimeConfig(m11_user_model_enabled=False),
        legacy_memory_rows=legacy,
    )
    assert result.state_before == result.state_after
    assert result.legacy_memory_rows_before == result.legacy_memory_rows_after


def test_m11_enabled_does_not_modify_any_legacy_memory_row():
    state = M11RuntimeState.clean(user_id="u")
    legacy = [{"id": "m1", "content": "unchanged"}]
    _, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=_extractor(claims=[_claim(1)]),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
        legacy_memory_rows=legacy,
    )
    assert result.legacy_memory_rows_before == result.legacy_memory_rows_after


def test_existing_chat_retrieval_path_unchanged_with_m11_enabled():
    state = M11RuntimeState.clean(user_id="u")
    retrieval_candidates = [{"id": "r1", "score": 0.8}, {"id": "r2", "score": 0.7}]
    _, result = run_m11_turn(
        state,
        user_id="u",
        turn_id=1,
        extractor=_extractor(claims=[_claim(1)]),
        config=M11RuntimeConfig(m11_user_model_enabled=True),
        legacy_memory_rows=retrieval_candidates,
    )
    assert list(result.legacy_memory_rows_after) == retrieval_candidates


def test_static_determinism_contract_no_forbidden_content_matching():
    modules = [
        "segmentum.user_model.reliability_ledger",
        "segmentum.user_model.prediction_ledger",
        "segmentum.user_model.value_composer",
        "segmentum.user_model.evidence_cards",
    ]
    for module_name in modules:
        source = inspect.getsource(importlib.import_module(module_name))
        assert "import re" not in source
        assert ".lower()" not in source
        assert "os.environ" not in source
        assert "datetime" not in source


def test_value_scores_are_byte_identical_across_runs_with_same_extractor_fixture():
    state_a = M11RuntimeState.clean(user_id="u")
    state_b = M11RuntimeState.clean(user_id="u")
    fixture = _extractor(claims=[_claim(1)], memory_value_band="high")
    _, a = run_m11_turn(state_a, user_id="u", turn_id=1, extractor=fixture, config=M11RuntimeConfig(m11_user_model_enabled=True))
    _, b = run_m11_turn(state_b, user_id="u", turn_id=1, extractor=fixture, config=M11RuntimeConfig(m11_user_model_enabled=True))
    assert [x.to_dict() for x in a.memory_value_compositions] == [x.to_dict() for x in b.memory_value_compositions]
    assert [x.to_dict() for x in a.reliability_ledger_updates] == [x.to_dict() for x in b.reliability_ledger_updates]
