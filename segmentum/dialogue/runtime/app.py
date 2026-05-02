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
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="M5.6 Persona Runtime", layout="wide")

from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest
from segmentum.dialogue.runtime.dashboard import DashboardCollector
from segmentum.dialogue.runtime.manager import PersonaManager
from segmentum.dialogue.runtime.safety import SafetyLayer


def init_session() -> None:
    if "pm" not in st.session_state:
        st.session_state.pm = PersonaManager(
            storage_dir=_project_root / "artifacts" / "m56_personas"
        )
    if "chat_iface" not in st.session_state:
        st.session_state.chat_iface = ChatInterface(
            enable_conscious_trace=True,
            conscious_root=_project_root / "artifacts" / "conscious",
            session_id="m56_live",
        )
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
        :root {
            --wechat-bg: #1f1f1f;
            --wechat-panel-bg: #202020;
            --wechat-divider: #2f2f2f;
            --bubble-other-bg: #2f3136;
            --bubble-other-text: #e8e8e8;
            --bubble-mine-bg: #3cd681;
            --bubble-mine-text: #111111;
            --name-color: #8a8a8a;
            --time-color: #7b7b7b;
            --input-bg: #202020;
            --input-border: #3a3a3a;
            --input-placeholder: #8b8b8b;
            --icon-color: #9a9a9a;
            --send-disabled-bg: #2b2b2b;
            --send-disabled-text: #7c7c7c;
        }
        .stApp {
            background: var(--wechat-bg);
            color: var(--bubble-other-text);
            font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
        }
        [data-testid="stHeader"] {
            background: rgba(31, 31, 31, 0.96);
        }
        [data-testid="stSidebar"] {
            background: #191919;
        }
        .main .block-container {
            max-width: none;
            padding: 0.6rem 1rem 1rem;
        }
        .app-caption {
            color: #777777;
            font-size: 14px;
            margin: -0.4rem 0 0.7rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--wechat-divider);
            background: var(--wechat-bg);
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px;
            padding: 0 18px;
            border-radius: 0;
            color: #8a8a8a;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #e8e8e8;
            border-bottom: 2px solid var(--bubble-mine-bg);
        }
        .wechat-panel {
            min-height: calc(100vh - 360px);
            background: var(--wechat-bg);
            border: 0;
            border-bottom: 0;
            border-radius: 0;
            overflow: hidden;
        }
        .wechat-topbar {
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--wechat-bg);
            border-bottom: 1px solid var(--wechat-divider);
            font-size: 16px;
            font-weight: 400;
            color: #e7e7e7;
        }
        .wechat-subtitle {
            margin-left: 8px;
            color: #777777;
            font-size: 14px;
            font-weight: 400;
        }
        .wechat-body {
            min-height: calc(100vh - 410px);
            max-height: calc(100vh - 360px);
            overflow-y: auto;
            padding: 20px 28px 12px;
            background: var(--wechat-bg);
            box-sizing: border-box;
        }
        .wechat-row {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 0 0 18px;
        }
        .wechat-row.user {
            flex-direction: row-reverse;
            justify-content: flex-start;
        }
        .wechat-row.assistant {
            justify-content: flex-start;
        }
        .wechat-row.user + .wechat-row.user {
            margin-top: -7px;
        }
        .wechat-avatar {
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: #ffffff;
            user-select: none;
            overflow: hidden;
        }
        .wechat-row.user .wechat-avatar {
            background:
                linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0)),
                linear-gradient(135deg, #5270df, #73b6ff 52%, #e7f0ff);
            color: #101010;
        }
        .wechat-row.assistant .wechat-avatar {
            background:
                linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0)),
                linear-gradient(135deg, #565b64, #252a31);
        }
        .wechat-stack {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            max-width: min(62vw, 460px);
        }
        .wechat-row.user .wechat-stack {
            align-items: flex-end;
        }
        .wechat-name {
            margin: -1px 0 6px;
            color: var(--name-color);
            font-size: 13px;
            line-height: 1.2;
            font-weight: 400;
        }
        .wechat-bubble {
            position: relative;
            display: inline-block;
            width: fit-content;
            max-width: min(58vw, 390px);
            min-height: 38px;
            padding: 11px 15px;
            border-radius: 8px;
            line-height: 1.5;
            font-size: 15px;
            font-weight: 400;
            letter-spacing: 0;
            white-space: pre-wrap;
            word-break: break-word;
            box-shadow: none;
            box-sizing: border-box;
        }
        .wechat-row.user .wechat-bubble {
            background: var(--bubble-mine-bg);
            color: var(--bubble-mine-text);
            margin-right: 0;
        }
        .wechat-row.assistant .wechat-bubble {
            background: var(--bubble-other-bg);
            color: var(--bubble-other-text);
            border: 0;
            margin-left: 0;
        }
        .wechat-row.user .wechat-bubble::after {
            content: "";
            position: absolute;
            top: 13px;
            right: -6px;
            border-width: 6px 0 6px 6px;
            border-style: solid;
            border-color: transparent transparent transparent var(--bubble-mine-bg);
        }
        .wechat-row.assistant .wechat-bubble::before {
            content: "";
            position: absolute;
            top: 13px;
            left: -6px;
            border-width: 6px 6px 6px 0;
            border-style: solid;
            border-color: transparent var(--bubble-other-bg) transparent transparent;
        }
        .time-divider {
            text-align: center;
            color: var(--time-color);
            font-size: 15px;
            line-height: 1;
            margin: 20px 0 18px;
        }
        .wechat-empty {
            display: flex;
            min-height: 240px;
            align-items: center;
            justify-content: center;
            color: #777777;
            font-size: 16px;
        }
        .wechat-typing {
            color: #8a8a8a;
        }
        .wechat-body::-webkit-scrollbar {
            width: 8px;
        }
        .wechat-body::-webkit-scrollbar-track {
            background: transparent;
        }
        .wechat-body::-webkit-scrollbar-thumb {
            background: #666666;
            border-radius: 8px;
        }
        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatInputContainer"],
        .stChatInputContainer {
            width: 100% !important;
            max-width: none !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        [data-testid="stChatInput"] {
            position: relative;
            width: 100% !important;
            max-width: none !important;
            height: 210px !important;
            min-height: 210px !important;
            background: var(--wechat-bg) !important;
            border: 0 !important;
            border-top: 1px solid var(--wechat-divider) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
            padding: 14px 30px 64px !important;
            margin-top: 0 !important;
            box-sizing: border-box !important;
        }
        [data-testid="stChatInput"]::before {
            content: "☻   ◇   ▭   ✂⌄   ◎   ♫";
            position: absolute;
            left: 48px;
            right: 126px;
            bottom: 27px;
            color: var(--icon-color);
            font-size: 22px;
            line-height: 1;
            letter-spacing: 0;
            white-space: nowrap;
            pointer-events: none;
        }
        [data-testid="stChatInput"]::after {
            content: "";
            position: absolute;
            right: 126px;
            bottom: 18px;
            width: 1px;
            height: 30px;
            background: var(--wechat-divider);
            pointer-events: none;
        }
        [data-testid="stChatInput"] > div {
            width: 100% !important;
            max-width: none !important;
            min-height: 132px !important;
            background: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            outline: 0 !important;
            overflow: hidden;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            width: 100% !important;
            max-width: none !important;
            min-height: 130px !important;
            background: var(--input-bg) !important;
            border: 0 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            outline: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] textarea {
            min-height: 130px !important;
            max-height: 130px !important;
            background: var(--input-bg) !important;
            border: 0 !important;
            border-radius: 12px !important;
            color: #eaeaea !important;
            -webkit-text-fill-color: #eaeaea !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            font-weight: 400 !important;
            box-shadow: none !important;
            caret-color: var(--bubble-mine-bg);
            padding: 18px 20px !important;
            resize: none !important;
            font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif !important;
            box-sizing: border-box !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--input-placeholder);
            -webkit-text-fill-color: var(--input-placeholder);
            font-size: 14px;
            line-height: 1.6;
        }
        [data-testid="stChatInput"] button {
            position: absolute;
            right: 36px;
            bottom: 18px;
            width: 80px;
            height: 40px;
            background: var(--send-disabled-bg) !important;
            border-radius: 8px !important;
            color: var(--send-disabled-text) !important;
            box-shadow: none !important;
            border: 0 !important;
            font-size: 16px !important;
            font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif !important;
        }
        [data-testid="stChatInput"] button:not(:disabled):not([aria-disabled="true"]) {
            background: #07c160 !important;
            color: #ffffff !important;
        }
        [data-testid="stChatInput"] button:not(:disabled):not([aria-disabled="true"]):hover {
            background: #07c160 !important;
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .main .block-container {
                padding-left: 0;
                padding-right: 0;
            }
            .wechat-body {
                padding: 18px 20px 12px;
            }
            .wechat-stack {
                max-width: min(70vw, 460px);
            }
            .wechat-bubble {
                font-size: 15px;
                max-width: min(70vw, 390px);
            }
            .wechat-avatar {
                width: 40px;
                height: 40px;
                flex-basis: 40px;
            }
            [data-testid="stChatInput"] {
                padding-left: 24px !important;
                padding-right: 24px !important;
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
            '<div class="wechat-avatar">我</div>'
            '<div class="wechat-stack">'
            f'<div class="wechat-bubble">{safe_text}</div>'
            "</div>"
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


def _auto_scroll_chat() -> None:
    components.html(
        """
        <script>
        const scrollChatToBottom = () => {
          const doc = window.parent.document;
          const bodies = doc.querySelectorAll(".wechat-body");
          const body = bodies[bodies.length - 1];
          if (!body) return;
          body.scrollTop = body.scrollHeight;
        };
        requestAnimationFrame(scrollChatToBottom);
        setTimeout(scrollChatToBottom, 120);
        setTimeout(scrollChatToBottom, 450);
        </script>
        """,
        height=0,
        width=0,
    )


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
    if st.session_state.messages:
        message_parts.append('<div class="time-divider">今天</div>')
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
    _auto_scroll_chat()

    if pending_text and chat_iface.has_agent():
        chat_iface.sync_transcript_from_messages(
            st.session_state.messages,
            pending_user_text=pending_text,
        )
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
    user_input = st.chat_input(
        "发消息" if not disabled else "先加载一个 persona...",
        disabled=disabled,
    )
    if user_input:
        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.pending_user_message = user_input
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


def _join_values(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "无"
    return str(value or "无")


def _latest_trace() -> dict[str, object]:
    chat_iface: ChatInterface = st.session_state.chat_iface
    rows = chat_iface.get_conscious_trace_rows(limit=1)
    return rows[-1] if rows else {}


def render_inner_world() -> None:
    st.header("内心观察")
    chat_iface: ChatInterface = st.session_state.chat_iface

    if not chat_iface.has_agent():
        st.info("Load a persona to observe its current turn trace.")
        return

    markdown = chat_iface.get_conscious_markdown()
    latest = _latest_trace()
    if not markdown or not latest:
        st.info("Send a message to generate the first conscious trace.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turn", str(latest.get("turn_id", "")))
    c2.metric("Action", str(latest.get("chosen_action", "")))
    c3.metric("Policy Margin", f"{float(latest.get('policy_margin', 0.0)):.3f}")
    c4.metric("EFE Margin", f"{float(latest.get('efe_margin', 0.0)):.3f}")

    obs = latest.get("observation_channels", {})
    if isinstance(obs, dict) and obs:
        st.subheader("Observation")
        obs_rows = [
            {"Channel": key, "Value": float(value)}
            for key, value in sorted(obs.items())
        ]
        st.dataframe(
            pd.DataFrame(obs_rows).set_index("Channel"),
            column_config={
                "Value": st.column_config.ProgressColumn(
                    "Value", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
            use_container_width=True,
        )

    st.subheader("Attention / Workspace")
    a1, a2 = st.columns(2)
    a1.markdown(
        "\n".join(
            [
                f"**Selected**  \n{_join_values(latest.get('attention_selected_channels'))}",
                f"**Workspace focus**  \n{_join_values(latest.get('workspace_focus'))}",
            ]
        )
    )
    a2.markdown(
        "\n".join(
            [
                f"**Dropped**  \n{_join_values(latest.get('attention_dropped_channels'))}",
                f"**Suppressed**  \n{_join_values(latest.get('workspace_suppressed'))}",
            ]
        )
    )

    ranked = latest.get("ranked_options", [])
    if isinstance(ranked, list) and ranked:
        st.subheader("Candidate Paths")
        rows = []
        for item in ranked:
            if isinstance(item, dict):
                rows.append(
                    {
                        "Action": item.get("action", ""),
                        "Policy": float(item.get("policy_score", 0.0)),
                        "EFE": float(item.get("expected_free_energy", 0.0)),
                        "Risk": float(item.get("risk", 0.0)),
                        "Dominant": item.get("dominant_component", ""),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    affect = latest.get("affective_state_summary", {})
    capsule = latest.get("fep_prompt_capsule", {})
    guidance = latest.get("meta_control_guidance", {})
    affective_guidance = latest.get("affective_maintenance_summary", {})
    outcome = latest.get("memory_update_signal", {})
    st.subheader("State / Prompt / Outcome")
    s1, s2, s3, s4 = st.columns(4)
    if isinstance(affect, dict):
        s1.json(affect, expanded=False)
    if isinstance(capsule, dict):
        s2.json(
            {
                "decision_uncertainty": capsule.get("decision_uncertainty", ""),
                "prediction_error_label": capsule.get("prediction_error_label", ""),
                "previous_outcome": capsule.get("previous_outcome", "neutral"),
                "hidden_intent_label": capsule.get("hidden_intent_label", ""),
            },
            expanded=False,
        )
    if isinstance(outcome, dict):
        s3.json(outcome, expanded=False)
    if isinstance(guidance, dict):
        flags = [
            key
            for key, value in sorted(guidance.items())
            if isinstance(value, bool) and value
        ]
        s4.json(
            {
                "flags": flags,
                "intensity": guidance.get("intensity", 0.0),
                "trigger_reasons": guidance.get("trigger_reasons", []),
                "affective_maintenance": affective_guidance
                if isinstance(affective_guidance, dict)
                else {},
            },
            expanded=False,
        )

    st.subheader("Conscious.md")
    st.markdown(markdown)


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

    chat_tab, dashboard_tab, inner_tab = st.tabs(["Chat", "Dashboard", "内心观察"])
    with chat_tab:
        render_chat()
    with dashboard_tab:
        render_dashboard()
    with inner_tab:
        render_inner_world()


if __name__ == "__main__":
    main()
