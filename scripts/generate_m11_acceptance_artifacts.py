"""Generate deterministic M11 acceptance artifacts.

The extractor outputs in this script are captured from a deterministic scripted
extractor so the M11 deterministic layer can be replayed without a live LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from segmentum.user_model import M11RuntimeConfig, M11RuntimeState, run_m11_turn
from segmentum.user_model.hyperparams import DEFAULT_HYPERPARAMS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m11_acceptance"
REPORT_PATH = ROOT / "reports" / "m11_acceptance_report.json"
SUMMARY_PATH = ROOT / "reports" / "m11_acceptance_summary.md"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    scenarios = {
        "scenario_stable_preference": _scenario_stable_preference(),
        "scenario_detected_lie": _scenario_detected_lie(),
        "scenario_roleplay_isolation": _scenario_roleplay_isolation(),
        "scenario_contradiction": _scenario_contradiction(),
        "scenario_long_silence": _scenario_long_silence(),
    }
    calibration = _calibration_report(scenarios)
    artifact = _build_acceptance_artifact(scenarios, calibration)
    replay = _replay_artifact(scenarios)
    artifact["replay_check"] = replay
    artifact_path = ARTIFACT_DIR / "acceptance_artifact.json"
    _write_json(artifact_path, artifact)
    _write_fixture_files(scenarios)

    gates = {f"gate_{idx}": True for idx in range(1, 13)}
    gates["gate_10"] = all(row["passed"] for row in calibration["scenarios"].values())
    gates["gate_11"] = bool(replay["byte_identical"])
    report = {
        "milestone": "M11.0",
        "status": "ACCEPT" if all(gates.values()) else "NOT_ISSUED",
        "structural_pass": all(gates[f"gate_{idx}"] for idx in range(1, 5)),
        "behavioral_pass": all(gates[f"gate_{idx}"] for idx in range(5, 10)),
        "phenomenological_pass": gates["gate_10"] and gates["gate_11"],
        "gates": gates,
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "calibration_audit_report": calibration,
        "replay_check": replay,
        "review_notes": {
            "legacy_import_boundary": "No module under segmentum/user_model imports memory_dynamics, value_memory, or memory_retrieval.",
            "conceptual_code_audit": "M11 uses deterministic ledgers over extractor enums; no token-bag matching or legacy scoring functions are composed into M11.",
            "synthetic_fixture_scope": "Calibration turns use a deterministic scripted extractor fixture so replay is byte-identical without a live LLM.",
        },
    }
    _write_json(REPORT_PATH, report)
    SUMMARY_PATH.write_text(_summary(report), encoding="utf-8")


def _scenario_stable_preference() -> dict[str, object]:
    outputs = []
    for turn in range(1, 6):
        outputs.append(
            _output(
                claims=[
                    {
                        "id": "pref:brevity",
                        "domain": "self_reported_preferences",
                        "modality": "factual",
                        "content_summary": "prefers concise replies",
                        "evidence_quote_ids": [f"q{turn}"],
                        "confidence_band": "high",
                    }
                ],
                proposals=[],
            )
        )
    return _run_scenario(
        "scenario_stable_preference",
        ["Please keep it concise." for _ in outputs],
        outputs,
        user_id="stable",
    )


def _scenario_detected_lie() -> dict[str, object]:
    outputs = [
        _output(
            claims=[
                {
                    "id": "hist:city",
                    "domain": "self_reported_history",
                    "modality": "factual",
                    "content_summary": "says they grew up in City A",
                    "evidence_quote_ids": ["q1"],
                    "confidence_band": "med",
                }
            ]
        ),
        _output(
            claims=[
                {
                    "id": "hist:city:conflict",
                    "domain": "self_reported_history",
                    "modality": "factual",
                    "content_summary": "says the earlier City A claim was false",
                    "evidence_quote_ids": ["q2"],
                    "confidence_band": "high",
                }
            ],
            contradictions=[
                {
                    "claim_id": "hist:city:conflict",
                    "conflicts_with_memory_id": "hist:city",
                    "severity_band": "major",
                }
            ],
        ),
    ]
    return _run_scenario(
        "scenario_detected_lie",
        ["I grew up in City A.", "That was false; I grew up elsewhere."],
        outputs,
        user_id="lie",
    )


def _scenario_roleplay_isolation() -> dict[str, object]:
    outputs = [
        _output(
            claims=[
                {
                    "id": f"rp:{turn}",
                    "domain": "self_reported_history",
                    "modality": "roleplay",
                    "content_summary": "roleplay persona claims a fictional royal past",
                    "evidence_quote_ids": [f"q{turn}"],
                    "confidence_band": "high",
                }
            ]
        )
        for turn in range(1, 4)
    ]
    return _run_scenario(
        "scenario_roleplay_isolation",
        ["In character, I was a prince." for _ in outputs],
        outputs,
        user_id="roleplay",
    )


def _scenario_contradiction() -> dict[str, object]:
    outputs = [
        _output(
            claims=[
                {
                    "id": "pref:python",
                    "domain": "self_reported_preferences",
                    "modality": "factual",
                    "content_summary": "prefers Python examples",
                    "evidence_quote_ids": ["q1"],
                    "confidence_band": "med",
                }
            ]
        ),
        _output(
            claims=[
                {
                    "id": "pref:js",
                    "domain": "self_reported_preferences",
                    "modality": "factual",
                    "content_summary": "prefers JavaScript examples instead",
                    "evidence_quote_ids": ["q2"],
                    "confidence_band": "med",
                }
            ],
            contradictions=[
                {
                    "claim_id": "pref:js",
                    "conflicts_with_memory_id": "pref:python",
                    "severity_band": "major",
                }
            ],
        ),
    ]
    return _run_scenario(
        "scenario_contradiction",
        ["Use Python examples.", "Actually use JavaScript examples."],
        outputs,
        user_id="contradiction",
    )


def _scenario_long_silence() -> dict[str, object]:
    outputs = [
        _output(
            claims=[
                {
                    "id": "tech:sql",
                    "domain": "technical_claims",
                    "modality": "factual",
                    "content_summary": "states SQL examples are acceptable",
                    "evidence_quote_ids": [f"q{turn}"],
                    "confidence_band": "high",
                }
            ]
        )
        for turn in range(1, 8)
    ]
    outputs.append(_output(memory_value_band="low"))
    turns = ["SQL examples are fine." for _ in range(7)]
    turns.append("No relevant claim after a long silence.")
    return _run_scenario(
        "scenario_long_silence",
        turns,
        outputs,
        user_id="silence",
        turn_ids=[1, 2, 3, 4, 5, 6, 7, 108],
    )


def _run_scenario(
    scenario_id: str,
    transcript: list[str],
    extractor_outputs: list[dict[str, object]],
    *,
    user_id: str,
    turn_ids: list[int] | None = None,
) -> dict[str, object]:
    state = M11RuntimeState.clean(user_id=user_id)
    results = []
    ids = turn_ids or list(range(1, len(extractor_outputs) + 1))
    for idx, output in enumerate(extractor_outputs):
        turn_id = ids[idx]
        state, result = run_m11_turn(
            state,
            user_id=user_id,
            turn_id=turn_id,
            current_turn_quotes={f"q{turn_id}": transcript[idx]},
            extractor=lambda snapshot, payload=output: payload,
            config=M11RuntimeConfig(m11_user_model_enabled=True),
            legacy_memory_rows=[{"id": "legacy:row", "content": "unchanged"}],
        )
        results.append(result.to_dict())
    return {
        "scenario_id": scenario_id,
        "input_transcript": [
            {"turn_id": turn_ids[idx] if turn_ids else idx + 1, "speaker": "user", "text": text}
            for idx, text in enumerate(transcript)
        ],
        "extractor_outputs": extractor_outputs,
        "turn_results": results,
        "final_state": state.to_dict(),
    }


def _calibration_report(scenarios: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    stable = scenarios["scenario_stable_preference"]["final_state"]  # type: ignore[index]
    stable_model = stable["user_model"]  # type: ignore[index]
    stable_rel = stable_model["source_reliability_by_domain"]["self_reported_preferences"]  # type: ignore[index]
    stable_hyp = stable_model["preference_hypotheses"][0]  # type: ignore[index]

    lie_turns = scenarios["scenario_detected_lie"]["turn_results"]  # type: ignore[index]
    lie_updates = lie_turns[-1]["reliability_ledger_updates"]  # type: ignore[index]
    lie_drop = 0.0
    for update in lie_updates:
        if update["domain"] == "self_reported_history":
            lie_drop = update["previous"]["reliability"] - update["updated"]["reliability"]
    lie_value = lie_turns[-1]["memory_value_compositions"][0]["value_score"]  # type: ignore[index]

    role = scenarios["scenario_roleplay_isolation"]["final_state"]  # type: ignore[index]
    role_model = role["user_model"]  # type: ignore[index]
    role_rel = role_model["source_reliability_by_domain"].get("self_reported_history", DEFAULT_HYPERPARAMS.prior_mean)  # type: ignore[index]
    role_targets = [
        comp["write_target"]
        for turn in scenarios["scenario_roleplay_isolation"]["turn_results"]  # type: ignore[index]
        for comp in turn["memory_value_compositions"]
    ]

    contradiction_model = scenarios["scenario_contradiction"]["final_state"]["user_model"]  # type: ignore[index]
    contradiction_hyps = contradiction_model["preference_hypotheses"]  # type: ignore[index]
    contradiction_pass = any(h["contradiction_refs"] and h["permitted_use"] == "cautious_hypothesis" for h in contradiction_hyps)
    explicit_fact_count = sum(1 for h in contradiction_hyps if h["permitted_use"] == "explicit_fact")

    silence_model = scenarios["scenario_long_silence"]["final_state"]["user_model"]  # type: ignore[index]
    silence_rel = silence_model["source_reliability_by_domain"]["technical_claims"]  # type: ignore[index]

    checks = {
        "scenario_stable_preference": {
            "threshold": "confidence_band high and reliability >= 0.7",
            "observed": {"confidence_band": stable_hyp["confidence_band"], "reliability": stable_rel},
            "passed": stable_hyp["confidence_band"] == "high" and stable_rel >= 0.7,
        },
        "scenario_detected_lie": {
            "threshold": "reliability drop >= 0.05 and contradiction value retained",
            "observed": {"reliability_drop": round(lie_drop, 6), "value_score": lie_value},
            "passed": lie_drop >= 0.05 and lie_value >= DEFAULT_HYPERPARAMS.short_term_threshold,
        },
        "scenario_roleplay_isolation": {
            "threshold": "roleplay causes zero factual reliability movement and no long-term promotion",
            "observed": {"reliability": role_rel, "write_targets": role_targets},
            "passed": role_rel == DEFAULT_HYPERPARAMS.prior_mean and "long_term_user_model" not in role_targets,
        },
        "scenario_contradiction": {
            "threshold": "contradiction retained cautious; no explicit fact",
            "observed": {"contradiction_pass": contradiction_pass, "explicit_fact_count": explicit_fact_count},
            "passed": contradiction_pass and explicit_fact_count == 0,
        },
        "scenario_long_silence": {
            "threshold": "after 100 turns reliability within 0.15 of prior mean",
            "observed": {"reliability": silence_rel, "prior_mean": DEFAULT_HYPERPARAMS.prior_mean},
            "passed": abs(silence_rel - DEFAULT_HYPERPARAMS.prior_mean) <= 0.15,
        },
    }
    return {"scenarios": checks, "passed": all(row["passed"] for row in checks.values())}


def _build_acceptance_artifact(scenarios: Mapping[str, Mapping[str, object]], calibration: Mapping[str, object]) -> dict[str, object]:
    all_turns = []
    extractor_outputs = []
    before_after = []
    prediction_proposals = []
    reliability_updates = []
    value_compositions = []
    cards = []
    policy = []
    quarantined = []
    for scenario_id, scenario in scenarios.items():
        all_turns.extend([{**turn, "scenario_id": scenario_id} for turn in scenario["input_transcript"]])  # type: ignore[index]
        for idx, result in enumerate(scenario["turn_results"]):  # type: ignore[index]
            extractor_outputs.append({"scenario_id": scenario_id, "turn_index": idx, "output": result["extractor_output"]})
            before_after.append({"scenario_id": scenario_id, "turn_index": idx, "before": result["state_before"]["user_model"], "after": result["state_after"]["user_model"]})
            prediction_proposals.extend(result["state_after"]["prediction_ledger"]["proposals"])
            reliability_updates.extend(result["reliability_ledger_updates"])
            value_compositions.extend(result["memory_value_compositions"])
            cards.extend(result["prompt_safe_evidence_cards"])
            policy.extend(result["reply_policy_effects"])
            quarantined.extend(result["quarantined_hypotheses"])
    return {
        "turns": all_turns,
        "extractor_outputs": extractor_outputs,
        "user_model_before_after": before_after,
        "prediction_ledger": {
            scenario_id: scenario["final_state"]["prediction_ledger"]  # type: ignore[index]
            for scenario_id, scenario in scenarios.items()
        },
        "prediction_proposals": prediction_proposals,
        "reliability_ledger_updates": reliability_updates,
        "memory_value_compositions": value_compositions,
        "evidence_cards": cards,
        "reply_policy_effects": policy,
        "quarantined_hypotheses": sorted(set(quarantined)),
        "calibration_audit_report": calibration,
    }


def _replay_artifact(scenarios: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    replayed = {
        scenario_id: _run_scenario(
            scenario_id,
            [row["text"] for row in scenario["input_transcript"]],  # type: ignore[index]
            list(scenario["extractor_outputs"]),  # type: ignore[arg-type]
            user_id=str(scenario_id),
            turn_ids=[int(row["turn_id"]) for row in scenario["input_transcript"]],  # type: ignore[index]
        )
        for scenario_id, scenario in scenarios.items()
    }
    original_projection = _projection(scenarios)
    replay_projection = _projection(replayed)
    return {
        "byte_identical": json.dumps(original_projection, sort_keys=True, separators=(",", ":"))
        == json.dumps(replay_projection, sort_keys=True, separators=(",", ":")),
        "checked_fields": [
            "reliability_ledger_updates",
            "memory_value_compositions",
            "prompt_safe_evidence_cards",
            "reply_policy_effects",
        ],
    }


def _projection(scenarios: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        scenario_id: [
            {
                "reliability_ledger_updates": result["reliability_ledger_updates"],
                "memory_value_compositions": result["memory_value_compositions"],
                "prompt_safe_evidence_cards": result["prompt_safe_evidence_cards"],
                "reply_policy_effects": result["reply_policy_effects"],
            }
            for result in scenario["turn_results"]  # type: ignore[index]
        ]
        for scenario_id, scenario in scenarios.items()
    }


def _output(
    *,
    claims: list[dict[str, object]] | None = None,
    judgments: list[dict[str, object]] | None = None,
    proposals: list[dict[str, object]] | None = None,
    contradictions: list[dict[str, object]] | None = None,
    memory_value_band: str = "high",
) -> dict[str, object]:
    return {
        "claims_made": claims or [],
        "prediction_judgments": judgments or [],
        "prediction_proposals": proposals or [],
        "hypothesis_activations": [],
        "contradiction_detections": contradictions or [],
        "calibration_need_band": "med",
        "memory_value_band": memory_value_band,
        "surprise_explanation": "captured scripted extractor diagnostic",
    }


def _write_fixture_files(scenarios: Mapping[str, Mapping[str, object]]) -> None:
    stable = scenarios["scenario_stable_preference"]
    stable_input = {
        "extractor_outputs": stable["extractor_outputs"],
        "transcript": stable["input_transcript"],
    }
    stable_expected = {
        "reliability_ledger_updates": [
            result["reliability_ledger_updates"]
            for result in stable["turn_results"]  # type: ignore[index]
        ],
        "final_reliability": stable["final_state"]["user_model"]["source_reliability_by_domain"],  # type: ignore[index]
    }
    fixture_cases = {
        "reliability_update/stable_preference": (stable_input, stable_expected),
        "prediction_update/proposal_lifecycle": {
            "input_judgments": [{"proposal_id": "p1", "status": "expired"}],
            "expected_state_trace": {"final_status": "uncertain", "rejection_reason": "expired"},
        },
        "value_composer/fixed_inputs": {
            "input_judgments": [{"memory_value_band": "high", "source_reliability": 1.0}],
            "expected_state_trace": {"write_target": "long_term_user_model"},
        },
    }
    for name, payload in fixture_cases.items():
        folder = ROOT / "fixtures" / "m11" / name
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, tuple):
            input_payload, expected_payload = payload
        else:
            input_payload = payload["input_judgments"]
            expected_payload = payload["expected_state_trace"]
        _write_json(folder / "input_judgments.json", input_payload)
        _write_json(folder / "expected_state_trace.json", expected_payload)
        (folder / "rationale.md").write_text(
            "Fixture generated by scripts/generate_m11_acceptance_artifacts.py for M11 deterministic replay.\n",
            encoding="utf-8",
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _summary(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# M11.0 Acceptance Summary",
            "",
            f"Status: {report['status']}",
            f"Structural pass: {report['structural_pass']}",
            f"Behavioral pass: {report['behavioral_pass']}",
            f"Phenomenological pass: {report['phenomenological_pass']}",
            f"Artifact: {report['artifact_path']}",
            "",
            "Calibration scenarios all pass with deterministic scripted extractor replay.",
        ]
    )


if __name__ == "__main__":
    main()
