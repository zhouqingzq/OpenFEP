"""M5.6 Persona Runtime — Streamlit local interactive app."""

from __future__ import annotations

import sys
from html import escape
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
    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message: str | None = None


def inject_app_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #15171b;
            color: #f3f3f3;
        }
        [data-testid="stHeader"] {
            background: rgba(21, 23, 27, 0.94);
        }
        [data-testid="stSidebar"] {
            background: #191b20;
        }
        .main .block-container {
            max-width: 1120px;
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }
        .app-caption {
            color: #9ca3af;
            font-size: 0.86rem;
            margin: -0.4rem 0 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #2b3038;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 18px;
            border-radius: 0;
            color: #a4abb6;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #f3f3f3;
            border-bottom: 3px solid #07c160;
        }
        .wechat-panel {
            min-height: calc(100vh - 330px);
            background: #1f1f1f;
            border: 1px solid #292b30;
            border-bottom: 0;
            border-radius: 0;
            overflow: hidden;
        }
        .wechat-topbar {
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1f1f1f;
            border-bottom: 1px solid #292b30;
            font-weight: 600;
            color: #f4f4f5;
        }
        .wechat-subtitle {
            margin-left: 8px;
            color: #9ca3af;
            font-size: 0.82rem;
            font-weight: 400;
        }
        .wechat-body {
            min-height: calc(100vh - 380px);
            padding: 26px 30px 30px;
        }
        .wechat-row {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin: 24px 0;
        }
        .wechat-row.user {
            justify-content: flex-end;
        }
        .wechat-row.assistant {
            justify-content: flex-start;
        }
        .wechat-avatar {
            width: 40px;
            height: 40px;
            flex: 0 0 40px;
            border-radius: 7px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            color: #ffffff;
            user-select: none;
        }
        .wechat-row.user .wechat-avatar {
            background:
                linear-gradient(145deg, rgba(255,255,255,0.18), rgba(255,255,255,0)),
                linear-gradient(135deg, #f3c28c, #9162e4 54%, #ffd37a);
        }
        .wechat-row.assistant .wechat-avatar {
            background:
                linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0)),
                linear-gradient(135deg, #5c6470, #29303a);
        }
        .wechat-stack {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            max-width: min(60%, 720px);
        }
        .wechat-row.user .wechat-stack {
            align-items: flex-end;
        }
        .wechat-name {
            margin: -2px 0 6px;
            color: #8b8d93;
            font-size: 0.82rem;
            line-height: 1;
        }
        .wechat-bubble {
            position: relative;
            padding: 10px 14px;
            border-radius: 7px;
            line-height: 1.55;
            font-size: 1.02rem;
            white-space: pre-wrap;
            word-break: break-word;
            box-shadow: none;
        }
        .wechat-row.user .wechat-bubble {
            background: #39d98a;
            color: #07130b;
            margin-right: 1px;
        }
        .wechat-row.assistant .wechat-bubble {
            background: #303133;
            color: #eeeeee;
            border: 0;
            margin-left: 1px;
        }
        .wechat-row.user .wechat-bubble::after {
            content: "";
            position: absolute;
            top: 13px;
            right: -6px;
            border-width: 6px 0 6px 7px;
            border-style: solid;
            border-color: transparent transparent transparent #39d98a;
        }
        .wechat-row.assistant .wechat-bubble::before {
            content: "";
            position: absolute;
            top: 13px;
            left: -6px;
            border-width: 6px 7px 6px 0;
            border-style: solid;
            border-color: transparent #303133 transparent transparent;
        }
        .wechat-empty {
            display: flex;
            min-height: 240px;
            align-items: center;
            justify-content: center;
            color: #8f98a6;
            font-size: 0.94rem;
        }
        .wechat-typing {
            color: #8f98a6;
        }
        div[data-testid="stForm"] {
            background: #1f1f1f;
            border: 1px solid #393b40;
            border-radius: 10px;
            padding: 12px 16px 14px;
            margin-top: 0;
        }
        div[data-testid="stForm"] input {
            height: 48px !important;
            background: transparent !important;
            border: 0 !important;
            color: #efefef !important;
            font-size: 1rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stForm"] input:focus {
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
        }
        div[data-testid="stForm"] input::placeholder {
            color: #85888f;
        }
        div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
            min-width: 74px;
            height: 36px;
            border: 0;
            border-radius: 9px;
            background: #2f3033;
            color: #9b9da3;
            font-size: 0.95rem;
        }
        div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
            background: #07c160;
            color: #101810;
        }
        .wechat-toolbar {
            display: flex;
            align-items: center;
            gap: 24px;
            color: #a8abb2;
            font-size: 1.25rem;
            line-height: 1;
            margin-bottom: 6px;
            user-select: none;
        }
        .wechat-toolbar-spacer {
            flex: 1 1 auto;
        }
        .wechat-tool-divider {
            width: 1px;
            height: 22px;
            background: #2d2f33;
            margin-left: -8px;
        }
        @media (max-width: 760px) {
            .main .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            .wechat-body {
                padding: 18px 12px 22px;
            }
            .wechat-stack {
                max-width: 74%;
            }
            .wechat-bubble {
                font-size: 0.95rem;
            }
            .wechat-avatar {
                width: 34px;
                height: 34px;
                flex-basis: 34px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _avatar_label(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        return "AI"
    return stripped[:2]


def _message_html(role: str, text: str, *, assistant_name: str = "AI") -> str:
    safe_text = escape(text)
    if role == "user":
        return (
            '<div class="wechat-row user">'
            '<div class="wechat-stack">'
            f'<div class="wechat-bubble">{safe_text}</div>'
            "</div>"
            '<div class="wechat-avatar">我</div>'
            "</div>"
        )
    safe_name = escape(assistant_name or "AI")
    safe_avatar = escape(_avatar_label(assistant_name or "AI"))
    return (
        '<div class="wechat-row assistant">'
        f'<div class="wechat-avatar">{safe_avatar}</div>'
        '<div class="wechat-stack">'
        f'<div class="wechat-name">{safe_name}</div>'
        f'<div class="wechat-bubble">{safe_text}</div>'
        "</div>"
        "</div>"
    )


def _queue_composer_message() -> None:
    text = str(st.session_state.get("wechat_composer_input", "")).strip()
    chat_iface: ChatInterface | None = st.session_state.get("chat_iface")
    if not text or st.session_state.get("pending_user_message"):
        return
    if chat_iface is None or not chat_iface.has_agent():
        return
    st.session_state.messages.append({"role": "user", "text": text})
    st.session_state.pending_user_message = text


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
    chat_iface: ChatInterface = st.session_state.chat_iface
    loaded_name = st.session_state.loaded_persona or "未加载人格"
    assistant_name = st.session_state.loaded_persona or chat_iface.persona_name or "AI"
    pending_text = st.session_state.pending_user_message

    # Display message history in a WeChat-like conversation surface.
    message_parts: list[str] = []
    for msg in st.session_state.messages:
        message_parts.append(
            _message_html(msg["role"], msg["text"], assistant_name=assistant_name)
        )
    if pending_text:
        message_parts.append(
            _message_html(
                "assistant",
                f"{assistant_name} 正在输入...",
                assistant_name=assistant_name,
            )
        )
    if not message_parts:
        if chat_iface.has_agent():
            empty_text = "开始聊天吧"
        else:
            empty_text = "请先在左侧创建或加载一个 persona"
        message_parts.append(f'<div class="wechat-empty">{escape(empty_text)}</div>')

    st.markdown(
        (
            '<div class="wechat-panel">'
            '<div class="wechat-topbar">'
            "Chat"
            f'<span class="wechat-subtitle">{escape(loaded_name)}</span>'
            "</div>"
            '<div class="wechat-body">'
            + "".join(message_parts)
            + "</div></div>"
        ),
        unsafe_allow_html=True,
    )

    if pending_text and chat_iface.has_agent():
        with st.spinner("AI 正在回复..."):
            try:
                resp = chat_iface.send(ChatRequest(user_text=pending_text))
                st.session_state.messages.append(
                    {"role": "assistant", "text": resp.reply}
                )
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"发送失败：{exc}"}
                )
            finally:
                st.session_state.pending_user_message = None
        st.rerun()
    elif pending_text and not chat_iface.has_agent():
        st.session_state.pending_user_message = None

    disabled = not chat_iface.has_agent() or pending_text is not None
    with st.form("wechat_composer", clear_on_submit=True):
        st.markdown(
            (
                '<div class="wechat-toolbar">'
                "<span>☺</span><span>◇</span><span>▭</span><span>✂</span>"
                '<span class="wechat-toolbar-spacer"></span>'
                "<span>◉</span><span>♬</span>"
                '<span class="wechat-tool-divider"></span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        user_input = st.text_input(
            "发消息",
            placeholder="发消息" if not disabled else "先加载一个 persona...",
            label_visibility="collapsed",
            disabled=disabled,
            key="wechat_composer_input",
        )
        _, send_col = st.columns([1, 0.12])
        submitted = send_col.form_submit_button("发送", disabled=disabled)

    if submitted and user_input.strip():
        _queue_composer_message()
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
    inject_app_style()

    st.title("Segmentum Persona Runtime")
    st.markdown(
        (
            '<div class="app-caption">'
            f"Loaded: {escape(st.session_state.loaded_persona or 'None')}"
            f"  |  Mode: {escape(st.session_state.chat_iface.generator_type.upper())}"
            f"  |  Storage: {escape(str(st.session_state.pm.storage_dir))}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    render_sidebar()

    chat_tab, dashboard_tab = st.tabs(["Chat", "Dashboard"])
    with chat_tab:
        render_chat()
    with dashboard_tab:
        render_dashboard()


if __name__ == "__main__":
    main()
