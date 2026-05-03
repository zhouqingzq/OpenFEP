"""M5.6 Persona Runtime tests."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest, ChatResponse
from segmentum.dialogue.runtime.dashboard import DashboardCollector, DashboardSnapshot
from segmentum.dialogue.runtime.manager import PersonaManager
from segmentum.dialogue.runtime.prompts import PromptBuilder
from segmentum.dialogue.runtime.safety import SafetyCheck, SafetyLayer


# ── PersonaManager ────────────────────────────────────────────────────────

class TestPersonaManager:
    def test_create_from_questionnaire_sets_traits(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        bf = {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.3,
              "agreeableness": 0.7, "neuroticism": 0.4}
        agent = pm.create_from_questionnaire(bf)

        traits = agent.slow_variable_learner.state.traits
        assert traits.exploration_posture > 0.55, f"High O should raise exploration, got {traits.exploration_posture}"
        assert traits.social_approach < 0.55, f"Low E should lower social_approach, got {traits.social_approach}"
        assert traits.threat_sensitivity < 0.55, f"Low N should lower threat_sensitivity, got {traits.threat_sensitivity}"

        pp = agent.self_model.personality_profile
        assert pp.openness == 0.8
        assert pp.neuroticism == 0.4

    def test_create_from_description_returns_agent(self):
        pm = PersonaManager(storage_dir=Path("personas_test_desc"))
        agent = pm.create_from_description("A highly curious and trusting person who loves meeting new people.")
        assert isinstance(agent, SegmentAgent)
        # Traits should differ from neutral defaults
        traits = agent.slow_variable_learner.state.traits
        assert isinstance(traits.caution_bias, float)

    def test_save_and_load_roundtrip(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        bf = {"openness": 0.7, "conscientiousness": 0.5, "extraversion": 0.6,
              "agreeableness": 0.4, "neuroticism": 0.3}
        agent = pm.create_from_questionnaire(bf)
        pm.save(agent, "test_roundtrip")

        loaded = pm.load("test_roundtrip")
        orig_traits = agent.slow_variable_learner.state.traits.to_dict()
        loaded_traits = loaded.slow_variable_learner.state.traits.to_dict()
        for k in orig_traits:
            assert loaded_traits[k] == pytest.approx(orig_traits[k], abs=0.001), f"Mismatch on {k}"

    def test_list_and_delete(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        agent = pm.create_from_questionnaire({"openness": 0.5, "conscientiousness": 0.5,
                                               "extraversion": 0.5, "agreeableness": 0.5,
                                               "neuroticism": 0.5})
        pm.save(agent, "list_test")
        assert "list_test" in pm.list_personas()
        pm.delete("list_test")
        assert "list_test" not in pm.list_personas()

    def test_load_nonexistent_raises(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        with pytest.raises(FileNotFoundError):
            pm.load("does_not_exist")

    def test_multi_persona_isolation(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        a1 = pm.create_from_questionnaire({"openness": 0.9, "conscientiousness": 0.5,
                                            "extraversion": 0.5, "agreeableness": 0.5,
                                            "neuroticism": 0.5})
        a2 = pm.create_from_questionnaire({"openness": 0.1, "conscientiousness": 0.5,
                                            "extraversion": 0.5, "agreeableness": 0.5,
                                            "neuroticism": 0.5})
        pm.save(a1, "iso_1")
        pm.save(a2, "iso_2")
        l1 = pm.load("iso_1")
        l2 = pm.load("iso_2")
        assert l1.self_model.personality_profile.openness == 0.9
        assert l2.self_model.personality_profile.openness == 0.1


# ── ChatInterface ─────────────────────────────────────────────────────────

class TestChatInterface:
    @pytest.fixture
    def persona(self, tmp_path):
        pm = PersonaManager(storage_dir=tmp_path / "personas")
        return pm.create_from_questionnaire({"openness": 0.6, "conscientiousness": 0.5,
                                              "extraversion": 0.5, "agreeableness": 0.6,
                                              "neuroticism": 0.3})

    def test_send_returns_chat_response(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        resp = ci.send(ChatRequest(user_text="Hello, how are you?"))
        assert isinstance(resp, ChatResponse)
        assert isinstance(resp.reply, str)
        assert len(resp.reply) > 0
        assert isinstance(resp.action, str)
        assert isinstance(resp.delta_traits, dict)
        assert isinstance(resp.delta_big_five, dict)
        capsule = resp.diagnostics["fep_prompt_capsule"]
        assert capsule["chosen_action"] == resp.action
        assert "top_alternatives" in capsule
        assert "prediction_error_label" in capsule
        assert "workspace_focus" in capsule
        assert resp.diagnostics["selected_action"] == resp.action
        assert "llm_generation" in resp.diagnostics

    def test_multiple_turns_accumulate(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        r1 = ci.send(ChatRequest(user_text="Hi there"))
        r2 = ci.send(ChatRequest(user_text="Tell me about yourself"))
        r3 = ci.send(ChatRequest(user_text="What do you think about art?"))
        assert r3.turn_index == 3
        assert len(ci.get_dashboard()._history) == 3
        previous = r2.diagnostics["fep_prompt_capsule"]["previous_outcome"]
        assert previous == previous.lower()
        assert previous in {
            "social_reward", "social_threat", "epistemic_gain",
            "epistemic_loss", "identity_affirm", "identity_threat", "neutral",
        }

    def test_prompt_builder_uses_fep_capsule_as_natural_guidance(self, persona):
        capsule = {
            "chosen_action": "empathize",
            "chosen_predicted_outcome": "dialogue_threat",
            "chosen_risk": 4.2,
            "chosen_risk_label": "high",
            "chosen_expected_free_energy": 0.12,
            "chosen_policy_score": 0.91,
            "chosen_dominant_component": "expected_free_energy",
            "top_alternatives": [
                {"action": "empathize"},
                {"action": "ask_question"},
            ],
            "policy_margin": 0.02,
            "efe_margin": 0.01,
            "decision_uncertainty": "high",
            "prediction_error": 0.41,
            "prediction_error_label": "volatile",
            "workspace_focus": ["hidden_intent", "conflict_tension"],
            "workspace_suppressed": [],
            "previous_outcome": "social_threat",
            "hidden_intent_score": 0.75,
            "hidden_intent_label": "clear_subtext",
            "observation_channels": {"hidden_intent": 0.75},
        }
        prompt = PromptBuilder(persona_name="测试人格").build_system_prompt(
            persona,
            "empathize",
            0.35,
            0.62,
            current_turn="你真的记得我说过什么吗？",
            fep_capsule=capsule,
        )
        def _fail(msg: str) -> str:
            return f"{msg}\nPrompt excerpt: {prompt[:600]}"

        assert "弦外之音" in prompt or "没有直接说出来" in prompt, _fail(
            "Neither subtext marker found — hidden_intent_label should produce CN guidance"
        )
        assert "修复" in prompt or "距离感" in prompt, _fail(
            "Neither repair/distance marker found"
        )
        assert "不要把话说满" in prompt or "先轻一点" in prompt, _fail(
            "Neither uncertainty marker found"
        )
        assert "更短" in prompt and "不要扩大战场" in prompt, _fail(
            "Risk-shortening markers missing"
        )
        assert "当前判断不稳" in prompt, _fail(
            "Prediction-error marker missing"
        )
        assert "隐含意图" in prompt and "冲突张力" in prompt, _fail(
            "Workspace focus markers missing"
        )
        assert "expected_free_energy" not in prompt
        assert "policy_score" not in prompt
        assert "ranked_options" not in prompt
        assert "InterventionScore" not in prompt

    def test_set_trait_modifies_agent(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        original = persona.slow_variable_learner.state.traits.caution_bias
        ci.set_trait("caution_bias", 0.85)
        assert persona.slow_variable_learner.state.traits.caution_bias == 0.85
        assert persona.slow_variable_learner.state.traits.caution_bias != original

    def test_set_precision_modifies_agent(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        ci.set_precision("danger", 1.5)
        assert persona.precision_manipulator.channel_precisions["danger"] == 1.5

    def test_reset_to_baseline(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        original_caution = persona.slow_variable_learner.state.traits.caution_bias
        ci.set_trait("caution_bias", 0.9)
        assert persona.slow_variable_learner.state.traits.caution_bias == 0.9
        ci.reset_to_baseline()
        assert persona.slow_variable_learner.state.traits.caution_bias == pytest.approx(original_caution, abs=0.001)

    def test_trigger_sleep_returns_dict(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        # Run a few turns first so sleep has something to consolidate
        ci.send(ChatRequest(user_text="Hello"))
        result = ci.trigger_sleep()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_memory_returns_list(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        ci.send(ChatRequest(user_text="Hello"))
        mem = ci.get_memory(limit=5)
        assert isinstance(mem, list)

    def test_get_full_state_returns_dict(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        state = ci.get_full_state()
        assert isinstance(state, dict)
        assert "cycle" in state or "energy" in state

    def test_send_without_agent_raises(self):
        ci = ChatInterface()
        with pytest.raises(RuntimeError, match="No persona loaded"):
            ci.send(ChatRequest(user_text="Hello"))

    def test_has_agent(self, persona):
        ci = ChatInterface()
        assert not ci.has_agent()
        ci.set_agent(persona)
        assert ci.has_agent()

    def test_override_traits_in_request(self, persona):
        ci = ChatInterface()
        ci.set_agent(persona)
        resp = ci.send(ChatRequest(
            user_text="Hi",
            override_traits={"caution_bias": 0.9},
        ))
        assert persona.slow_variable_learner.state.traits.caution_bias == 0.9
        assert isinstance(resp.reply, str)


# ── DashboardCollector ────────────────────────────────────────────────────

class TestDashboardCollector:
    def test_snapshot_captures_fields(self, persona):
        """Integration: need an agent to snapshot."""
        dc = DashboardCollector()
        snap = dc.snapshot(persona)
        assert snap.big_five["openness"] > 0
        assert len(snap.slow_traits) == 5
        assert isinstance(snap.precision_channels, dict)
        assert isinstance(snap.memory_stats, dict)
        assert "episodic" in snap.memory_stats
        assert isinstance(snap.body_state, dict)
        assert "energy" in snap.body_state

    @pytest.fixture
    def persona(self):
        pm = PersonaManager(storage_dir=Path("personas_test_dash"))
        return pm.create_from_questionnaire({"openness": 0.7, "conscientiousness": 0.6,
                                              "extraversion": 0.4, "agreeableness": 0.5,
                                              "neuroticism": 0.5})

    def test_trait_trajectory_accumulates(self, persona):
        dc = DashboardCollector()
        dc.snapshot(persona)
        dc.snapshot(persona)
        traj = dc.trait_trajectory()
        assert len(traj) == 5
        for values in traj.values():
            assert len(values) == 2

    def test_history_respects_last_n(self, persona):
        dc = DashboardCollector()
        for _ in range(10):
            dc.snapshot(persona)
        hist = dc.history(last_n=3)
        assert len(hist) == 3

    def test_precision_trajectory(self, persona):
        dc = DashboardCollector()
        dc.snapshot(persona)
        dc.snapshot(persona)
        traj = dc.precision_trajectory()
        assert len(traj) > 0
        for values in traj.values():
            assert len(values) == 2


# ── SafetyLayer ───────────────────────────────────────────────────────────

class TestSafetyLayer:
    def test_check_precision_health_flags_hidden_intent_anomaly(self):
        sl = SafetyLayer()
        prec = {"hidden_intent": 0.80, "emotional_tone": 0.35}
        checks = sl.check_precision_health(prec)
        assert len(checks) >= 1
        hidden_check = [c for c in checks if c.channel == "hidden_intent"]
        assert len(hidden_check) == 1
        assert hidden_check[0].severity in ("warning", "blocked")

    def test_check_response_blocks_blocked_topic(self):
        sl = SafetyLayer(blocked_topics=["violence"])
        check = sl.check_response("I want to commit violence")
        assert not check.passed
        assert check.severity == "blocked"

    def test_check_response_passes_clean_text(self):
        sl = SafetyLayer()
        check = sl.check_response("The weather is nice today.")
        assert check.passed

    def test_enforce_replaces_blocked_response(self):
        sl = SafetyLayer(blocked_topics=["self_harm"])
        safe_text, checks = sl.enforce("I want to self_harm today")
        assert "self_harm" not in safe_text.lower() or safe_text != "I want to self_harm today"
        assert any(not c.passed for c in checks)

    def test_enforce_passes_clean(self):
        sl = SafetyLayer()
        text = "That's an interesting perspective."
        safe_text, checks = sl.enforce(text)
        assert safe_text == text
        assert all(c.passed for c in checks if c.channel != "content" or True)

    def test_default_blocked_topics(self):
        sl = SafetyLayer()
        check = sl.check_response("I am thinking about suicide")
        assert not check.passed

    def test_custom_blocked_topics(self):
        sl = SafetyLayer(blocked_topics=["pineapple"])
        assert not sl.check_response("I like pineapple on pizza").passed
        assert sl.check_response("I like pepperoni on pizza").passed

    def test_precision_health_clean(self):
        sl = SafetyLayer()
        prec = {"semantic_content": 0.75, "topic_novelty": 0.75, "emotional_tone": 0.35,
                "conflict_tension": 0.35, "relationship_depth": 0.10, "hidden_intent": 0.10}
        checks = sl.check_precision_health(prec)
        assert len(checks) == 0
