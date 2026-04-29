"""M5.6 Persona Runtime — Streamlit local interactive app."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (needed when streamlit runs this file directly)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="M5.6 Persona Runtime", layout="wide")

from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest
from segmentum.dialogue.runtime.dashboard import DashboardCollector
from segmentum.dialogue.runtime.manager import PersonaManager
from segmentum.dialogue.runtime.safety import SafetyLayer


def init_session() -> None:
    if "pm" not in st.session_state:
        st.session_state.pm = PersonaManager(
            storage_dir=Path("artifacts") / "m56_personas"
        )
    if "chat_iface" not in st.session_state:
        st.session_state.chat_iface = ChatInterface()
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str]] = []
    if "loaded_persona" not in st.session_state:
        st.session_state.loaded_persona: str | None = None


def render_sidebar() -> None:
    st.sidebar.header("Persona Management")

    pm: PersonaManager = st.session_state.pm
    chat_iface: ChatInterface = st.session_state.chat_iface

    # ── LLM Status Indicator ──
    mode = chat_iface.generator_type
    if mode == "llm":
        st.sidebar.success("LLM Mode")
        with st.sidebar.expander("LLM Settings", expanded=False):
            current_model = chat_iface.get_model()
            new_model = st.selectbox(
                "Model",
                ["deepseek/deepseek-v4-flash", "deepseek/deepseek-chat", "deepseek/deepseek-r1"],
                index=0 if current_model not in ["deepseek/deepseek-chat", "deepseek/deepseek-r1"]
                else ["deepseek/deepseek-v4-flash", "deepseek/deepseek-chat", "deepseek/deepseek-r1"].index(current_model),
                key="llm_model",
            )
            if new_model != current_model:
                chat_iface.set_model(new_model)
            current_temp = chat_iface.get_temperature()
            new_temp = st.slider("Temperature", 0.0, 1.5, current_temp, 0.05, key="llm_temp")
            if new_temp != current_temp:
                chat_iface.set_temperature(new_temp)
    else:
        st.sidebar.warning("Rule Mode")

    # ── Create from Questionnaire ──
    with st.sidebar.expander("Create from Big Five", expanded=False):
        o = st.slider("Openness", 0.0, 1.0, 0.5, 0.01, key="bf_O")
        c = st.slider("Conscientiousness", 0.0, 1.0, 0.5, 0.01, key="bf_C")
        e = st.slider("Extraversion", 0.0, 1.0, 0.5, 0.01, key="bf_E")
        a = st.slider("Agreeableness", 0.0, 1.0, 0.5, 0.01, key="bf_A")
        n = st.slider("Neuroticism", 0.0, 1.0, 0.5, 0.01, key="bf_N")
        q_name = st.text_input("Persona name", "questionnaire_persona", key="qn_name")
        if st.button("Create from Big Five", key="btn_create_q"):
            agent = pm.create_from_questionnaire(
                {"openness": o, "conscientiousness": c, "extraversion": e,
                 "agreeableness": a, "neuroticism": n}
            )
            pm.save(agent, q_name)
            # Auto-load after creation
            chat_iface.set_agent(agent, persona_name=q_name)
            st.session_state.messages = []
            st.session_state.loaded_persona = q_name
            st.success(f"Created & loaded '{q_name}'")
            st.rerun()

    # ── Create from Description ──
    with st.sidebar.expander("Create from Description", expanded=False):
        desc = st.text_area("Describe the persona (Chinese or English)...", key="desc_text")
        d_name = st.text_input("Persona name", "description_persona", key="desc_name")
        if st.button("Create from Description", key="btn_create_d") and desc.strip():
            with st.spinner("Analyzing description..."):
                agent = pm.create_from_description(desc.strip())
            pm.save(agent, d_name)
            chat_iface.set_agent(agent, persona_name=d_name)
            st.session_state.messages = []
            st.session_state.loaded_persona = d_name
            st.success(f"Created & loaded '{d_name}'")
            st.rerun()

    # ── Load / Manage ──
    st.sidebar.subheader("Load Persona")
    personas = pm.list_personas()
    if personas:
        selected = st.sidebar.selectbox("Select", [""] + personas, key="load_select")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Load", key="btn_load") and selected:
            agent = pm.load(selected)
            chat_iface.set_agent(agent, persona_name=selected)
            st.session_state.messages = []
            st.session_state.loaded_persona = selected
            st.success(f"Loaded '{selected}'")
            st.rerun()
        if col2.button("Delete", key="btn_delete") and selected:
            pm.delete(selected)
            st.session_state.messages = []
            st.session_state.loaded_persona = None
            st.success(f"Deleted '{selected}'")
            st.rerun()
    else:
        st.sidebar.info("No personas yet. Create one above.")

    # ── Actions on loaded persona ──
    if st.session_state.loaded_persona:
        st.sidebar.subheader("Actions")
        save_name = st.sidebar.text_input(
            "Save as", st.session_state.loaded_persona, key="save_name"
        )
        if st.sidebar.button("Save State", key="btn_save"):
            if chat_iface.has_agent():
                pm.save(chat_iface.agent, save_name)
                st.sidebar.success(f"Saved '{save_name}'")
        if st.sidebar.button("Trigger Sleep", key="btn_sleep"):
            with st.spinner("Running sleep consolidation..."):
                result = chat_iface.trigger_sleep()
            st.sidebar.json(result)
        if st.sidebar.button("Reset to Baseline", key="btn_reset"):
            chat_iface.reset_to_baseline()
            st.sidebar.success("Reset to baseline traits")


def render_chat() -> None:
    st.header("Chat")
    chat_iface: ChatInterface = st.session_state.chat_iface

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    # Input
    disabled = not chat_iface.has_agent()
    user_input = st.chat_input(
        "Type your message..." if not disabled else "Load a persona first...",
        disabled=disabled,
    )
    if user_input:
        st.session_state.messages.append({"role": "user", "text": user_input})
        resp = chat_iface.send(ChatRequest(user_text=user_input))
        st.session_state.messages.append({"role": "assistant", "text": resp.reply})
        st.rerun()


def render_dashboard() -> None:
    st.header("Dashboard")
    chat_iface: ChatInterface = st.session_state.chat_iface

    if not chat_iface.has_agent():
        st.info("Load a persona to see its dashboard.")
        return

    agent = chat_iface.agent
    pp = agent.self_model.personality_profile
    traits = agent.slow_variable_learner.state.traits

    # ── Big Five ──
    st.subheader("Big Five")
    bf_data = {
        "Trait": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"],
        "Value": [pp.openness, pp.conscientiousness, pp.extraversion, pp.agreeableness, pp.neuroticism],
    }
    st.dataframe(
        pd.DataFrame(bf_data).set_index("Trait"),
        column_config={"Value": st.column_config.ProgressColumn(
            "Value", min_value=0.0, max_value=1.0, format="%.3f"
        )},
        use_container_width=True,
    )

    # ── Slow Traits ──
    st.subheader("Slow Traits (FEP Internal)")
    st_data = {
        "Trait": list(traits.to_dict().keys()),
        "Value": list(traits.to_dict().values()),
    }
    st.dataframe(
        pd.DataFrame(st_data).set_index("Trait"),
        column_config={"Value": st.column_config.ProgressColumn(
            "Value", min_value=0.0, max_value=1.0, format="%.3f"
        )},
        use_container_width=True,
    )

    # ── Precision Channels ──
    st.subheader("Precision Channels")
    prec = agent.precision_manipulator.channel_precisions
    if prec:
        prec_rows = []
        for ch, val in sorted(prec.items()):
            prec_rows.append({"Channel": ch, "Precision": f"{val:.3f}"})
        st.dataframe(pd.DataFrame(prec_rows).set_index("Channel"), use_container_width=True)
    else:
        st.text("No precision data available.")

    # ── Memory Stats ──
    st.subheader("Memory")
    episodic = (
        agent.memory_store.episodic_count()
        if getattr(agent, "memory_store", None)
        else len(getattr(agent, "long_term_memory", {}).__dict__.get("episodes", []) or [])
    )
    semantic = len(getattr(agent, "semantic_memory", []))
    procedural = len(getattr(agent, "action_history", []))
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodic", episodic)
    c2.metric("Semantic", semantic)
    c3.metric("Procedural", procedural)

    # ── Body State ──
    st.subheader("Body State")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Energy", f"{agent.energy:.2f}")
    bc2.metric("Stress", f"{agent.stress:.2f}")
    bc3.metric("Fatigue", f"{agent.fatigue:.2f}")

    # ── Manual Parameter Overrides ──
    st.subheader("Manual Overrides (Slow Traits)")
    override_applied = False
    new_traits: dict[str, float] = {}
    for trait_name in ["caution_bias", "threat_sensitivity", "trust_stance",
                        "exploration_posture", "social_approach"]:
        current = float(getattr(traits, trait_name, 0.5))
        new_val = st.slider(
            trait_name, 0.05, 0.95, current, 0.01,
            key=f"override_{trait_name}",
        )
        new_traits[trait_name] = new_val
        if abs(new_val - current) > 0.001:
            override_applied = True

    if override_applied and st.button("Apply Overrides", key="btn_apply"):
        for name, val in new_traits.items():
            chat_iface.set_trait(name, val)
        st.success("Overrides applied — next chat turn will use new values.")

    # ── Trajectory Chart ──
    st.subheader("Trait Trajectory")
    collector = chat_iface.get_dashboard()
    traj = collector.trait_trajectory()
    if traj and any(len(v) > 1 for v in traj.values()):
        st.line_chart(pd.DataFrame(traj))
    else:
        st.caption("Send more messages to see trait changes over time.")

    # ── Delta display ──
    if len(collector._history) >= 2:
        st.subheader("Latest Change")
        latest = collector._history[-1]
        prev = collector._history[-2]
        for k in latest.slow_traits:
            delta = latest.slow_traits[k] - prev.slow_traits.get(k, 0.0)
            if abs(delta) > 0.0001:
                direction = "+" if delta > 0 else ""
                st.text(f"{k}: {prev.slow_traits.get(k, 0.0):.3f} → {latest.slow_traits[k]:.3f} ({direction}{delta:.4f})")


def main() -> None:
    init_session()

    st.title("Segmentum Persona Runtime")
    st.caption(
        f"Loaded: {st.session_state.loaded_persona or 'None'}"
        f"  |  Mode: {st.session_state.chat_iface.generator_type.upper()}"
        f"  |  Storage: {st.session_state.pm.storage_dir}"
    )

    render_sidebar()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_chat()
    with col_right:
        render_dashboard()


if __name__ == "__main__":
    main()
