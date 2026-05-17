"""M5.6 Persona Runtime — Streamlit local interactive app."""

from __future__ import annotations

import base64
from collections.abc import Mapping
import hashlib
import json
import shutil
import sys
import uuid
from html import escape
from pathlib import Path

# Ensure project root is on sys.path (needed when streamlit runs this file directly)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# In-process cache: (mtime_ns, data_uri) per slot; survives Streamlit reruns.
_CHAT_AVATAR_DATA_URI_CACHE: dict[str, tuple[float, str]] = {}

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="M5.6 Persona Runtime", layout="wide")

from segmentum.dialogue.runtime.chat import ChatInterface, ChatRequest
from segmentum.dialogue.runtime.dashboard import DashboardCollector
from segmentum.dialogue.runtime.manager import (
    PersonaManager,
    read_material_file_bytes,
    unique_persona_name,
)
from segmentum.dialogue.runtime.safety import SafetyLayer
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    OpenRouterJSONClient,
    analyze_materials_into_personas,
)

_DIALOGUE_PREF_FIELDS: tuple[str, ...] = (
    "current_speaker_name",
    "chat_avatar_user_label",
    "chat_avatar_user_preset",
    "chat_avatar_assistant_label",
    "chat_avatar_assistant_preset",
)


def _read_query_did() -> str:
    qp = st.query_params
    raw = qp.get("did")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    s = str(raw or "").strip()
    return s[:64] if s else ""


def ensure_dialogue_client_id() -> None:
    """Assign a stable per-tab ``did`` query param so parallel browsers do not share MVP files."""
    s = _read_query_did()
    if len(s) >= 8:
        old = st.session_state.get("dialogue_client_id")
        st.session_state.dialogue_client_id = s
        if old is not None and old != s:
            st.session_state._dialogue_prefs_loaded = False
            st.session_state.messages = []
            st.session_state._last_persisted_prefs_blob = None
        return
    new_id = uuid.uuid4().hex[:16]
    st.query_params["did"] = new_id
    st.rerun()


def _dialogue_client_prefs_path(did: str) -> Path:
    base = _chat_avatar_dir() / "clients"
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in did).strip("_")[:64] or "client"
    return base / f"{safe}.json"


def _load_dialogue_client_prefs(did: str) -> dict[str, str]:
    path = _dialogue_client_prefs_path(did)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(k, str)}


def _maybe_persist_dialogue_client_prefs(did: str) -> None:
    if len(did) < 8:
        return
    data = {k: str(st.session_state.get(k, "")) for k in _DIALOGUE_PREF_FIELDS}
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("_last_persisted_prefs_blob") == blob:
        return
    path = _dialogue_client_prefs_path(did)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state._last_persisted_prefs_blob = blob


def _speaker_avatar_stem(speaker_name: str) -> str:
    n = str(speaker_name or "").strip() or "default_user"
    return hashlib.sha256(n.encode("utf-8")).hexdigest()[:24]


def _speakers_avatar_dir() -> Path:
    d = _chat_avatar_dir() / "speakers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_speaker_avatar_paths(speaker_name: str):
    stem = _speaker_avatar_stem(speaker_name)
    base = _speakers_avatar_dir()
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        yield base / f"{stem}{ext}"


def find_stored_chat_avatar_path_for_speaker(speaker_name: str) -> Path | None:
    for p in _iter_speaker_avatar_paths(speaker_name):
        if p.is_file():
            return p
    return None


def _speaker_avatar_cache_key(speaker_name: str) -> str:
    return f"sp:{_speaker_avatar_stem(speaker_name)}"


def _migrate_legacy_user_avatar_for_speaker(speaker_name: str) -> None:
    """If an old flat ``user.*`` avatar exists and this speaker has none, copy once."""
    if find_stored_chat_avatar_path_for_speaker(speaker_name) is not None:
        return
    leg = find_stored_chat_avatar_path("user")
    if leg is None or not leg.is_file():
        return
    dest_dir = _speakers_avatar_dir()
    dest = dest_dir / f"{_speaker_avatar_stem(speaker_name)}{leg.suffix.lower()}"
    try:
        shutil.copy2(leg, dest)
    except OSError:
        return
    _invalidate_chat_avatar_cache(_speaker_avatar_cache_key(speaker_name))


def delete_chat_avatar_image_for_speaker(speaker_name: str) -> None:
    for p in _iter_speaker_avatar_paths(speaker_name):
        p.unlink(missing_ok=True)
    _invalidate_chat_avatar_cache(_speaker_avatar_cache_key(speaker_name))


def save_chat_avatar_image_for_speaker(speaker_name: str, data: bytes) -> tuple[bool, str]:
    if len(data) > _MAX_CHAT_AVATAR_BYTES:
        return False, f"文件过大（最大 {_MAX_CHAT_AVATAR_BYTES // 1000}KB）"
    mime = _detect_image_mime(data)
    if not mime:
        return False, "仅支持 PNG / JPEG / GIF / WebP"
    delete_chat_avatar_image_for_speaker(speaker_name)
    out = _speakers_avatar_dir() / f"{_speaker_avatar_stem(speaker_name)}{_mime_to_ext(mime)}"
    out.write_bytes(data)
    _invalidate_chat_avatar_cache(_speaker_avatar_cache_key(speaker_name))
    return True, ""


def load_chat_avatar_data_uri_for_speaker(speaker_name: str) -> str | None:
    stem = _speaker_avatar_stem(speaker_name)
    cache_key = _speaker_avatar_cache_key(speaker_name)
    path = find_stored_chat_avatar_path_for_speaker(speaker_name)
    if path is None:
        _invalidate_chat_avatar_cache(cache_key)
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CHAT_AVATAR_DATA_URI_CACHE.get(cache_key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    data = path.read_bytes()
    mime = _detect_image_mime(data)
    if not mime:
        return None
    uri = f"data:{mime};base64,{base64.standard_b64encode(data).decode('ascii')}"
    _CHAT_AVATAR_DATA_URI_CACHE[cache_key] = (mtime, uri)
    return uri


def _recreate_chat_iface_for_dialogue_session(did: str) -> None:
    iface = st.session_state.chat_iface
    old_agent = getattr(iface, "agent", None)
    old_persona = getattr(iface, "persona_name", "") or st.session_state.get("loaded_persona") or ""
    key_available = OpenRouterJSONClient.available()
    replacement = ChatInterface(
        use_llm=True if key_available else None,
        persona_name=str(old_persona),
        enable_conscious_trace=True,
        conscious_root=_project_root / "artifacts" / "conscious",
        session_id=did,
    )
    if old_agent is not None:
        replacement.set_agent(old_agent, persona_name=str(old_persona))
    st.session_state.chat_iface = replacement


def init_session() -> None:
    did = str(st.session_state.get("dialogue_client_id") or "").strip()[:64]
    if len(did) < 8:
        did = "m56_live"
        st.session_state.dialogue_client_id = did

    if not st.session_state.get("_dialogue_prefs_loaded"):
        prefs = _load_dialogue_client_prefs(did)
        for k in _DIALOGUE_PREF_FIELDS:
            if k in prefs:
                st.session_state[k] = prefs[k]
        st.session_state._dialogue_prefs_loaded = True

    if "pm" not in st.session_state:
        st.session_state.pm = PersonaManager(
            storage_dir=_project_root / "artifacts" / "m56_personas"
        )
    if "chat_iface" not in st.session_state:
        st.session_state.chat_iface = ChatInterface(
            use_llm=True if OpenRouterJSONClient.available() else None,
            enable_conscious_trace=True,
            conscious_root=_project_root / "artifacts" / "conscious",
            session_id=did,
        )
    else:
        _upgrade_chat_interface_if_needed()
        if getattr(st.session_state.chat_iface, "_session_id", None) != did:
            _recreate_chat_iface_for_dialogue_session(did)
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str]] = []
    if "loaded_persona" not in st.session_state:
        st.session_state.loaded_persona: str | None = None
    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message: str | None = None
    if "pending_speaker_name" not in st.session_state:
        st.session_state.pending_speaker_name: str | None = None
    if "current_speaker_name" not in st.session_state:
        st.session_state.current_speaker_name = "测试用户"
    if "chat_avatar_user_label" not in st.session_state:
        st.session_state.chat_avatar_user_label = "我"
    if "chat_avatar_user_preset" not in st.session_state:
        st.session_state.chat_avatar_user_preset = "ocean"
    if "chat_avatar_assistant_label" not in st.session_state:
        st.session_state.chat_avatar_assistant_label = ""
    if "chat_avatar_assistant_preset" not in st.session_state:
        st.session_state.chat_avatar_assistant_preset = "slate"
    # Bump to reset st.file_uploader after save/clear so the widget does not
    # keep returning the same file and re-trigger save+rerun forever.
    if "chat_avatar_upload_nonce_user" not in st.session_state:
        st.session_state.chat_avatar_upload_nonce_user = 0
    if "chat_avatar_upload_nonce_assistant" not in st.session_state:
        st.session_state.chat_avatar_upload_nonce_assistant = 0

    sn0 = str(st.session_state.get("current_speaker_name") or "").strip()
    if sn0:
        _migrate_legacy_user_avatar_for_speaker(sn0)

    if st.session_state.get("_last_persisted_prefs_blob") is None:
        st.session_state._last_persisted_prefs_blob = json.dumps(
            {k: str(st.session_state.get(k, "")) for k in _DIALOGUE_PREF_FIELDS},
            ensure_ascii=False,
            sort_keys=True,
        )


def _upgrade_chat_interface_if_needed() -> None:
    """Replace old Streamlit-held ChatInterface instances after code updates.

    Streamlit keeps Python objects in session_state across reruns.  A previously
    created rule-mode ChatInterface can therefore keep using the old rule-based
    generator even after the source code now supports the MVP LLM runtime.
    """
    iface = st.session_state.chat_iface
    key_available = OpenRouterJSONClient.available()
    needs_upgrade = not hasattr(iface, "_maybe_enable_mvp_llm_runtime")
    needs_llm_switch = bool(key_available and getattr(iface, "_use_llm", False) is False)
    if not (needs_upgrade or needs_llm_switch):
        return

    old_agent = getattr(iface, "agent", None)
    old_persona = getattr(iface, "persona_name", "") or st.session_state.get("loaded_persona") or ""
    did = str(st.session_state.get("dialogue_client_id") or "m56_live").strip()[:64]
    if len(did) < 8:
        did = "m56_live"
    replacement = ChatInterface(
        use_llm=True if key_available else None,
        persona_name=str(old_persona),
        enable_conscious_trace=True,
        conscious_root=_project_root / "artifacts" / "conscious",
        session_id=did,
    )
    if old_agent is not None:
        replacement.set_agent(old_agent, persona_name=str(old_persona))
    st.session_state.chat_iface = replacement


# Avatar preset → CSS class inside the chat iframe (whitelist only; no user CSS injection).
CHAT_USER_AVATAR_PRESETS: dict[str, str] = {
    "ocean": "av-u-ocean",
    "mint": "av-u-mint",
    "sunset": "av-u-sunset",
    "lilac": "av-u-lilac",
    "ember": "av-u-ember",
}
CHAT_USER_AVATAR_PRESET_LABELS: dict[str, str] = {
    "ocean": "蓝紫（默认）",
    "mint": "翠绿",
    "sunset": "暖霞",
    "lilac": "淡紫",
    "ember": "赤陶",
}
CHAT_ASSISTANT_AVATAR_PRESETS: dict[str, str] = {
    "slate": "av-a-slate",
    "teal": "av-a-teal",
    "rose": "av-a-rose",
    "amber": "av-a-amber",
    "midnight": "av-a-midnight",
}
CHAT_ASSISTANT_AVATAR_PRESET_LABELS: dict[str, str] = {
    "slate": "岩灰（默认）",
    "teal": "青绿",
    "rose": "玫粉",
    "amber": "琥珀",
    "midnight": "午夜",
}


def _truncate_avatar_label(raw: str, *, max_chars: int = 4) -> str:
    text = str(raw or "").strip()
    if not text:
        return "我"
    return text[:max_chars]


def _assistant_avatar_label_text(custom_label: str, assistant_name: str) -> str:
    custom = str(custom_label or "").strip()
    if custom:
        return _truncate_avatar_label(custom, max_chars=4)
    return _avatar_label(assistant_name)


def _single_wide_char_label(label: str) -> bool:
    """Use slightly larger type for single non-ASCII glyph (e.g. one emoji)."""
    s = str(label or "").strip()
    return len(s) == 1 and ord(s[0]) > 127


_MAX_CHAT_AVATAR_BYTES = 400_000


def _chat_avatar_dir() -> Path:
    d = _project_root / "artifacts" / "chat_avatars"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_image_mime(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _mime_to_ext(mime: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(mime, ".png")


def _iter_avatar_paths(slot: str):
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        yield _chat_avatar_dir() / f"{slot}{ext}"


def find_stored_chat_avatar_path(slot: str) -> Path | None:
    for p in _iter_avatar_paths(slot):
        if p.is_file():
            return p
    return None


def _invalidate_chat_avatar_cache(slot: str) -> None:
    _CHAT_AVATAR_DATA_URI_CACHE.pop(slot, None)


def delete_chat_avatar_image(slot: str) -> None:
    for p in _iter_avatar_paths(slot):
        p.unlink(missing_ok=True)
    _invalidate_chat_avatar_cache(slot)


def save_chat_avatar_image(slot: str, data: bytes) -> tuple[bool, str]:
    if len(data) > _MAX_CHAT_AVATAR_BYTES:
        return False, f"文件过大（最大 {_MAX_CHAT_AVATAR_BYTES // 1000}KB）"
    mime = _detect_image_mime(data)
    if not mime:
        return False, "仅支持 PNG / JPEG / GIF / WebP"
    delete_chat_avatar_image(slot)
    out = _chat_avatar_dir() / f"{slot}{_mime_to_ext(mime)}"
    out.write_bytes(data)
    _invalidate_chat_avatar_cache(slot)
    return True, ""


def load_chat_avatar_data_uri(slot: str) -> str | None:
    path = find_stored_chat_avatar_path(slot)
    if path is None:
        _invalidate_chat_avatar_cache(slot)
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CHAT_AVATAR_DATA_URI_CACHE.get(slot)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    data = path.read_bytes()
    mime = _detect_image_mime(data)
    if not mime:
        return None
    uri = f"data:{mime};base64,{base64.standard_b64encode(data).decode('ascii')}"
    _CHAT_AVATAR_DATA_URI_CACHE[slot] = (mtime, uri)
    return uri


# Chat UI lives inside components.html so scroll JS runs in the same document as the
# bubbles (Streamlit ≥1.5x isolates st.markdown from component iframes).
_WECHAT_IFRAME_STYLES = """
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
    --bubble-max: 288px;
}
html, body {
    margin: 0;
    height: 100%;
    box-sizing: border-box;
}
*, *::before, *::after { box-sizing: inherit; }
body {
    font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
    background: var(--wechat-bg);
    color: var(--bubble-other-text);
}
.wechat-panel {
    height: 100%;
    min-height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--wechat-bg);
    overflow: hidden;
    border: 0;
    border-radius: 0;
}
.wechat-topbar {
    flex: 0 0 auto;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--wechat-bg);
    border-bottom: 1px solid var(--wechat-divider);
    font-size: 15px;
    font-weight: 400;
    color: #e7e7e7;
}
.wechat-subtitle {
    margin-left: 8px;
    color: #777777;
    font-size: 13px;
    font-weight: 400;
}
.wechat-body {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 16px 16px 28px;
    background: var(--wechat-bg);
    scroll-padding-bottom: 48px;
}
.wechat-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin: 0 0 14px;
}
.wechat-row.user {
    flex-direction: row-reverse;
    justify-content: flex-start;
    gap: 14px;
}
.wechat-row.assistant {
    justify-content: flex-start;
    gap: 14px;
}
.wechat-row.user + .wechat-row.user {
    margin-top: -7px;
}
.wechat-avatar {
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: #ffffff;
    user-select: none;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.35);
}
.wechat-avatar.av-single {
    font-size: 19px;
    font-weight: 500;
    line-height: 1;
}
.wechat-avatar.with-img {
    padding: 0;
    line-height: 0;
    background: transparent !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.45);
    font-size: 0;
    font-weight: 400;
}
.wechat-avatar.with-img img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 8px;
    display: block;
}
.wechat-row.user .wechat-avatar.av-u-ocean {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0)),
        linear-gradient(135deg, #5270df, #73b6ff 52%, #e7f0ff);
    color: #101010;
}
.wechat-row.user .wechat-avatar.av-u-mint {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.12), transparent),
        linear-gradient(135deg, #0d7a62, #3cd681 55%, #b8ffe8);
    color: #06221a;
}
.wechat-row.user .wechat-avatar.av-u-sunset {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.12), transparent),
        linear-gradient(135deg, #c44a2d, #f0a050 50%, #ffd8a8);
    color: #2a0f08;
}
.wechat-row.user .wechat-avatar.av-u-lilac {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.12), transparent),
        linear-gradient(135deg, #6b4fb8, #a78bfa 52%, #e8e0ff);
    color: #1e1038;
}
.wechat-row.user .wechat-avatar.av-u-ember {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.1), transparent),
        linear-gradient(135deg, #8b1538, #d94a4a 48%, #ffb59a);
    color: #1a0505;
}
.wechat-row.assistant .wechat-avatar.av-a-slate {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.12), transparent),
        linear-gradient(135deg, #565b64, #252a31);
    color: #f0f0f0;
}
.wechat-row.assistant .wechat-avatar.av-a-teal {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.1), transparent),
        linear-gradient(135deg, #0f5c5c, #1a8a8a 50%, #6ee7d8);
    color: #031a18;
}
.wechat-row.assistant .wechat-avatar.av-a-rose {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.12), transparent),
        linear-gradient(135deg, #7c3a5c, #c084b8 52%, #fce7f3);
    color: #2a0a1f;
}
.wechat-row.assistant .wechat-avatar.av-a-amber {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.1), transparent),
        linear-gradient(135deg, #8a5a12, #d4a017 48%, #fff2c4);
    color: #2a1a00;
}
.wechat-row.assistant .wechat-avatar.av-a-midnight {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.08), transparent),
        linear-gradient(135deg, #1a2a4a, #2d4a8c 45%, #5a7fd4);
    color: #e8f0ff;
}
.wechat-stack {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: fit-content;
    max-width: min(280px, 88%);
    min-width: 0;
}
.wechat-row.user .wechat-stack {
    align-items: flex-end;
}
.wechat-name {
    margin: -1px 0 4px;
    color: var(--name-color);
    font-size: 12px;
    line-height: 1.2;
    font-weight: 400;
}
.wechat-bubble {
    position: relative;
    display: inline-block;
    width: fit-content;
    max-width: min(280px, 100%);
    min-height: 34px;
    padding: 8px 11px;
    border-radius: 6px;
    line-height: 1.45;
    font-size: 14px;
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
    border-radius: 8px;
}
.wechat-row.assistant .wechat-bubble {
    background: var(--bubble-other-bg);
    color: var(--bubble-other-text);
    border: 0;
    margin-left: 0;
    border-radius: 8px;
}
/* Rotated square tail: avoids the vertical "stem" artifact of border-triangles. */
.wechat-row.user .wechat-bubble::after {
    content: "";
    position: absolute;
    width: 9px;
    height: 9px;
    top: 14px;
    right: -3px;
    background: var(--bubble-mine-bg);
    transform: rotate(45deg);
    border-radius: 0 2px 0 0;
    z-index: 0;
    pointer-events: none;
}
.wechat-row.assistant .wechat-bubble::before {
    content: "";
    position: absolute;
    width: 9px;
    height: 9px;
    top: 14px;
    left: -3px;
    background: var(--bubble-other-bg);
    transform: rotate(45deg);
    border-radius: 0 0 0 2px;
    z-index: 0;
    pointer-events: none;
}
.time-divider {
    text-align: center;
    color: var(--time-color);
    font-size: 13px;
    line-height: 1;
    margin: 16px 0 14px;
}
.wechat-empty {
    display: flex;
    min-height: 240px;
    align-items: center;
    justify-content: center;
    color: #777777;
    font-size: 14px;
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
@media (max-width: 760px) {
    .wechat-body {
        padding: 14px 12px 22px;
    }
    .wechat-stack {
        max-width: min(268px, 90%);
    }
    .wechat-bubble {
        font-size: 14px;
        max-width: min(268px, 100%);
    }
    .wechat-avatar {
        width: 36px;
        height: 36px;
        flex-basis: 36px;
    }
}
"""


def _chat_iframe_height(messages: list[dict[str, str]], *, pending: str | None) -> int:
    """Height (px) for the WeChat-style message iframe.

    Taller pane shows more history before the inner ``.wechat-body`` scrolls; the
    Streamlit ``st.chat_input`` block below is kept compact so the middle region
    dominates typical laptop viewports.
    """
    n = len(messages) + (1 if pending else 0)
    chars = sum(len(str(m.get("text", ""))) for m in messages)
    content_hint = 220 + n * 46 + min(260, chars // 140)
    cap = 620
    floor = 420
    return max(floor, min(cap, content_hint))


def _render_wechat_chat_iframe(
    message_parts_html: str,
    loaded_name: str,
    *,
    messages: list[dict[str, str]],
    pending: str | None,
) -> None:
    safe_title = escape(loaded_name)
    inner = (
        '<div class="wechat-panel">'
        '<div class="wechat-topbar">Chat'
        f'<span class="wechat-subtitle">{safe_title}</span>'
        "</div>"
        '<div class="wechat-body" id="wechat-scroll-root">'
        + message_parts_html
        + "</div></div>"
    )
    doc = (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"/>"
        "<style>"
        + _WECHAT_IFRAME_STYLES
        + "</style></head><body>"
        + inner
        + """
<script>
(function () {
  const el = document.getElementById("wechat-scroll-root");
  function toBottom() {
    if (!el) return;
    el.scrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
  }
  toBottom();
  requestAnimationFrame(function () {
    toBottom();
    requestAnimationFrame(toBottom);
  });
  [0, 16, 48, 120, 280, 600, 1200, 2200].forEach(function (t) {
    setTimeout(toBottom, t);
  });
  if (typeof ResizeObserver !== "undefined" && el) {
    var ro = new ResizeObserver(toBottom);
    ro.observe(el);
    setTimeout(function () { ro.disconnect(); }, 10000);
  }
})();
</script>
</body></html>"""
    )
    components.html(doc, height=_chat_iframe_height(messages, pending=pending), scrolling=True)


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
            height: 168px !important;
            min-height: 168px !important;
            background: var(--wechat-bg) !important;
            border: 0 !important;
            border-top: 1px solid var(--wechat-divider) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
            padding: 12px 30px 52px !important;
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
            min-height: 104px !important;
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
            min-height: 102px !important;
            background: var(--input-bg) !important;
            border: 0 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            outline: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] textarea {
            min-height: 102px !important;
            max-height: 102px !important;
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


def _message_html(
    role: str,
    text: str,
    *,
    assistant_name: str = "AI",
    user_avatar_label: str = "我",
    user_avatar_class: str = "av-u-ocean",
    assistant_avatar_label: str = "",
    assistant_avatar_class: str = "av-a-slate",
    user_image_data_uri: str | None = None,
    assistant_image_data_uri: str | None = None,
) -> str:
    safe_text = escape(text)
    if role == "user":
        if user_image_data_uri:
            return (
                '<div class="wechat-row user">'
                '<div class="wechat-avatar with-img">'
                f'<img src="{user_image_data_uri}" alt="" draggable="false"/>'
                "</div>"
                '<div class="wechat-stack">'
                f'<div class="wechat-bubble">{safe_text}</div>'
                "</div>"
                "</div>"
            )
        u_lbl = _truncate_avatar_label(user_avatar_label)
        safe_u = escape(u_lbl)
        single_cls = " av-single" if _single_wide_char_label(u_lbl) else ""
        safe_cls = escape(user_avatar_class)
        return (
            '<div class="wechat-row user">'
            f'<div class="wechat-avatar {safe_cls}{single_cls}">{safe_u}</div>'
            '<div class="wechat-stack">'
            f'<div class="wechat-bubble">{safe_text}</div>'
            "</div>"
            "</div>"
        )
    safe_name = escape(assistant_name or "AI")
    if assistant_image_data_uri:
        return (
            '<div class="wechat-row assistant">'
            '<div class="wechat-avatar with-img">'
            f'<img src="{assistant_image_data_uri}" alt="" draggable="false"/>'
            "</div>"
            '<div class="wechat-stack">'
            f'<div class="wechat-name">{safe_name}</div>'
            f'<div class="wechat-bubble">{safe_text}</div>'
            "</div>"
            "</div>"
        )
    asst_lbl = _assistant_avatar_label_text(assistant_avatar_label, assistant_name or "AI")
    safe_avatar = escape(asst_lbl)
    asst_single = " av-single" if _single_wide_char_label(asst_lbl) else ""
    safe_acls = escape(assistant_avatar_class)
    return (
        '<div class="wechat-row assistant">'
        f'<div class="wechat-avatar {safe_acls}{asst_single}">{safe_avatar}</div>'
        '<div class="wechat-stack">'
        f'<div class="wechat-name">{safe_name}</div>'
        f'<div class="wechat-bubble">{safe_text}</div>'
        "</div>"
        "</div>"
    )


def append_assistant_response_messages(
    messages: list[dict[str, str]],
    response: object,
) -> None:
    reply = str(getattr(response, "reply", "")).strip()
    if reply:
        messages.append({"role": "assistant", "text": reply})
    for followup in getattr(response, "followup_replies", []) or []:
        text = str(followup).strip()
        if text:
            messages.append({"role": "assistant", "text": text})


def render_sidebar() -> None:
    st.sidebar.header("Persona Management")

    pm: PersonaManager = st.session_state.pm
    chat_iface: ChatInterface = st.session_state.chat_iface

    # ── LLM Status Indicator ──
    mode = chat_iface.generator_type
    if mode == "llm":
        st.sidebar.success("LLM Mode")
        if getattr(chat_iface, "mvp_runtime_active", False):
            st.sidebar.caption("MVP loop: active")
        else:
            st.sidebar.error("MVP loop: inactive")
        if chat_iface.has_agent() and getattr(chat_iface, "mvp_runtime_active", False):
            proactive_opt_in = st.sidebar.checkbox(
                "Enable bounded proactive messages",
                value=bool(st.session_state.get("m13_initiative_opt_in", False)),
                key="m13_initiative_opt_in_checkbox",
                help="Off by default. Allows a manual continue button; no background autonomy.",
            )
            if proactive_opt_in != bool(st.session_state.get("m13_initiative_opt_in_synced", False)):
                chat_iface.set_bounded_proactive_opt_in(proactive_opt_in)
                st.session_state.m13_initiative_opt_in_synced = proactive_opt_in
            st.session_state.m13_initiative_opt_in = proactive_opt_in
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

    # ── Create from Material File ──
    with st.sidebar.expander("Create from Material File", expanded=False):
        material_file = st.file_uploader(
            "Select .txt or .md material",
            type=["txt", "md"],
            key="material_file",
        )
        if st.button("Create from Material File", key="btn_create_material"):
            if material_file is None:
                st.warning("Please select a .txt or .md material file first.")
            elif not OpenRouterJSONClient.available():
                st.error("Material-file initialization requires secrets/openrouter.json or OPENAI_API_KEY.")
            else:
                try:
                    material_text = read_material_file_bytes(
                        material_file.name,
                        material_file.getvalue(),
                    )
                    with st.spinner("Analyzing material with LLM..."):
                        personas = analyze_materials_into_personas(
                            OpenRouterJSONClient.from_config(),
                            [material_text],
                        )
                    existing = set(pm.list_personas())
                    created: list[str] = []
                    loaded_agent = None
                    loaded_name = ""
                    for payload in personas:
                        requested_name = str(payload.get("persona_name") or "persona")
                        persona_name = unique_persona_name(requested_name, existing)
                        existing.add(persona_name)
                        payload["persona_name"] = persona_name
                        if isinstance(payload.get("self_basic_facts"), dict):
                            payload["self_basic_facts"]["name"] = persona_name
                        agent = pm.create_from_material_analysis(payload)
                        pm.save(agent, persona_name)
                        runtime = MVPDialogueRuntime(
                            store=MVPStateStore(
                                _project_root / "artifacts" / "mvp_personas" / persona_name
                            ),
                            llm=OpenRouterJSONClient.from_config(),
                            persona_name=persona_name,
                        )
                        runtime.initialize_from_persona_payload(payload)
                        created.append(persona_name)
                        if loaded_agent is None:
                            loaded_agent = agent
                            loaded_name = persona_name
                    if loaded_agent is not None:
                        chat_iface.set_agent(loaded_agent, persona_name=loaded_name)
                        st.session_state.messages = []
                        st.session_state.loaded_persona = loaded_name
                    st.session_state.last_created_material_personas = created
                    st.success(
                        "Created: " + ", ".join(created)
                        if created
                        else "No personas were created."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Material-file initialization failed: {exc}")

        created = st.session_state.get("last_created_material_personas", [])
        if created:
            st.caption("Last created: " + ", ".join(str(item) for item in created))

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

    st.sidebar.divider()
    st.sidebar.text_input(
        "对话者姓名",
        key="current_speaker_name",
        disabled=(
            not chat_iface.has_agent()
            or st.session_state.get("pending_user_message") is not None
        ),
        placeholder="例如：张三、李四",
        help="用于区分不同对话者；M11 用户模型按此名称分桶。头像预设与图片按此姓名绑定保存。",
    )
    _did_show = str(st.session_state.get("dialogue_client_id") or "")
    st.sidebar.caption(
        "地址栏里的 `did=…` 表示本聊天窗口；在此处改姓名后偏好会写入本机，下次同一链接自动还原。"
        + (f" 当前会话：`{_did_show[:20]}…`" if len(_did_show) > 20 else f" 当前会话：`{_did_show}`")
    )

    with st.sidebar.expander("聊天头像", expanded=False):
        st.caption("仅影响 Chat 气泡旁展示；不参与推理。")
        st.text_input(
            "我的头像文字",
            key="chat_avatar_user_label",
            max_chars=4,
            help="最多 4 个字符，或单个 emoji。",
        )
        st.selectbox(
            "我的头像底色",
            options=list(CHAT_USER_AVATAR_PRESETS.keys()),
            format_func=lambda k: CHAT_USER_AVATAR_PRESET_LABELS.get(k, k),
            key="chat_avatar_user_preset",
        )
        st.text_input(
            "对方头像文字",
            key="chat_avatar_assistant_label",
            max_chars=4,
            placeholder="留空用人格名前两字",
            help="不填则用当前已加载人格名称的前两个字符。",
        )
        st.selectbox(
            "对方头像底色",
            options=list(CHAT_ASSISTANT_AVATAR_PRESETS.keys()),
            format_func=lambda k: CHAT_ASSISTANT_AVATAR_PRESET_LABELS.get(k, k),
            key="chat_avatar_assistant_preset",
        )

        st.divider()
        st.caption(
            "「我的」图片按左侧「对话者姓名」存入 artifacts/chat_avatars/speakers/；"
            "`clients/*.json` 存姓名与头像选项（同一 `did` 链接下自动恢复）。"
        )
        _spk = str(st.session_state.get("current_speaker_name") or "").strip() or "测试用户"
        _nu = int(st.session_state.get("chat_avatar_upload_nonce_user", 0))
        up_user = st.file_uploader(
            "我的头像图片",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            key=f"chat_avatar_upload_user_{_nu}",
            help=f"绑定「{_spk}」；最大 {_MAX_CHAT_AVATAR_BYTES // 1000}KB，推荐方形。",
        )
        if up_user is not None:
            ok_u, err_u = save_chat_avatar_image_for_speaker(_spk, up_user.getvalue())
            if ok_u:
                st.session_state.chat_avatar_upload_nonce_user = _nu + 1
                st.success("已保存我的头像图片")
                st.rerun()
            else:
                st.error(err_u)
        path_user = find_stored_chat_avatar_path_for_speaker(_spk)
        if path_user is not None:
            st.image(str(path_user), width=56)
            if st.button("清除我的头像图", key="btn_chat_avatar_clear_user"):
                delete_chat_avatar_image_for_speaker(_spk)
                st.session_state.chat_avatar_upload_nonce_user = int(
                    st.session_state.get("chat_avatar_upload_nonce_user", 0)
                ) + 1
                st.success("已清除")
                st.rerun()

        _na = int(st.session_state.get("chat_avatar_upload_nonce_assistant", 0))
        up_asst = st.file_uploader(
            "对方头像图片",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            key=f"chat_avatar_upload_assistant_{_na}",
            help=f"最大 {_MAX_CHAT_AVATAR_BYTES // 1000}KB，推荐方形。",
        )
        if up_asst is not None:
            ok_a, err_a = save_chat_avatar_image("assistant", up_asst.getvalue())
            if ok_a:
                st.session_state.chat_avatar_upload_nonce_assistant = _na + 1
                st.success("已保存对方头像图片")
                st.rerun()
            else:
                st.error(err_a)
        path_asst = find_stored_chat_avatar_path("assistant")
        if path_asst is not None:
            st.image(str(path_asst), width=56)
            if st.button("清除对方头像图", key="btn_chat_avatar_clear_assistant"):
                delete_chat_avatar_image("assistant")
                st.session_state.chat_avatar_upload_nonce_assistant = int(
                    st.session_state.get("chat_avatar_upload_nonce_assistant", 0)
                ) + 1
                st.success("已清除")
                st.rerun()

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

    _did = str(st.session_state.get("dialogue_client_id") or "")
    _maybe_persist_dialogue_client_prefs(_did)


def _user_message_visuals(msg: dict[str, str]) -> tuple[str, str, str | None]:
    sp = str(msg.get("speaker_name") or "").strip() or "测试用户"
    fallback_preset = str(st.session_state.get("chat_avatar_user_preset", "ocean"))
    fallback_label = str(st.session_state.get("chat_avatar_user_label", "我"))
    preset = str(msg.get("user_avatar_preset") or fallback_preset)
    label = str(msg.get("user_avatar_label") or fallback_label)
    u_cls = CHAT_USER_AVATAR_PRESETS.get(preset, CHAT_USER_AVATAR_PRESETS["ocean"])
    img = load_chat_avatar_data_uri_for_speaker(sp)
    return label, u_cls, img


def render_chat() -> None:
    chat_iface: ChatInterface = st.session_state.chat_iface
    loaded_name = st.session_state.loaded_persona or "未加载人格"
    assistant_name = st.session_state.loaded_persona or chat_iface.persona_name or "AI"
    pending_text = st.session_state.pending_user_message
    pending_speaker = (
        st.session_state.pending_speaker_name
        or st.session_state.current_speaker_name
        or "测试用户"
    )

    a_preset = str(st.session_state.get("chat_avatar_assistant_preset", "slate"))
    a_av_class = CHAT_ASSISTANT_AVATAR_PRESETS.get(a_preset, CHAT_ASSISTANT_AVATAR_PRESETS["slate"])
    a_av_custom = str(st.session_state.get("chat_avatar_assistant_label", ""))
    asst_img_uri = load_chat_avatar_data_uri("assistant")

    # Display message history in a WeChat-like conversation surface.
    message_parts: list[str] = []
    if st.session_state.messages:
        message_parts.append('<div class="time-divider">今天</div>')
    for msg in st.session_state.messages:
        text = msg["text"]
        if msg.get("role") == "user":
            u_av_label, u_av_class, user_img_uri = _user_message_visuals(msg)
        else:
            # User-side avatar args are ignored for assistant rows in ``_message_html``.
            u_av_label = str(st.session_state.get("chat_avatar_user_label", "我"))
            u_av_class = CHAT_USER_AVATAR_PRESETS.get(
                str(st.session_state.get("chat_avatar_user_preset", "ocean")),
                CHAT_USER_AVATAR_PRESETS["ocean"],
            )
            user_img_uri = None
        message_parts.append(
            _message_html(
                msg["role"],
                text,
                assistant_name=assistant_name,
                user_avatar_label=u_av_label,
                user_avatar_class=u_av_class,
                assistant_avatar_label=a_av_custom,
                assistant_avatar_class=a_av_class,
                user_image_data_uri=user_img_uri,
                assistant_image_data_uri=asst_img_uri,
            )
        )
    if pending_text:
        message_parts.append(
            _message_html(
                "assistant",
                f"{assistant_name} 正在输入...",
                assistant_name=assistant_name,
                user_avatar_label=str(st.session_state.get("chat_avatar_user_label", "我")),
                user_avatar_class=CHAT_USER_AVATAR_PRESETS.get(
                    str(st.session_state.get("chat_avatar_user_preset", "ocean")),
                    CHAT_USER_AVATAR_PRESETS["ocean"],
                ),
                assistant_avatar_label=a_av_custom,
                assistant_avatar_class=a_av_class,
                user_image_data_uri=None,
                assistant_image_data_uri=asst_img_uri,
            )
        )
    if not message_parts:
        if chat_iface.has_agent():
            empty_text = "开始聊天吧"
        else:
            empty_text = "请先在左侧创建或加载一个 persona"
        message_parts.append(f'<div class="wechat-empty">{escape(empty_text)}</div>')

    _render_wechat_chat_iframe(
        "".join(message_parts),
        loaded_name,
        messages=st.session_state.messages,
        pending=pending_text,
    )

    pending_proactive = bool(st.session_state.get("pending_proactive_continue"))
    if pending_proactive and chat_iface.has_agent() and not pending_text:
        speaker = (
            st.session_state.pending_speaker_name
            or st.session_state.current_speaker_name
            or "测试用户"
        )
        with st.spinner("胡桃正在主动续写..."):
            try:
                check = chat_iface.maybe_propose_proactive_turn(manual_continue=True)
                proposal = check.get("proposal") if isinstance(check, dict) else None
                if isinstance(proposal, dict) and proposal.get("proposal_id"):
                    resp = chat_iface.run_proactive_turn(
                        str(proposal["proposal_id"]),
                        speaker_name=str(speaker).strip() or "测试用户",
                    )
                    if str(resp.reply or "").strip():
                        append_assistant_response_messages(st.session_state.messages, resp)
                    else:
                        reason = ""
                        if isinstance(resp.diagnostics, dict):
                            reason = str(resp.diagnostics.get("suppression_reason", "") or "")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "text": (
                                    "本轮未发送主动消息"
                                    + (f"（{reason}）" if reason else "。")
                                ),
                            }
                        )
                else:
                    reason = str(check.get("suppression_reason", "") if isinstance(check, dict) else "")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "text": f"当前不适合主动续写（{reason or '已抑制'}）。",
                        }
                    )
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"主动续写失败：{exc}"}
                )
            finally:
                st.session_state.pending_proactive_continue = False
                st.session_state.pending_speaker_name = None
        st.rerun()
    elif pending_text and chat_iface.has_agent():
        chat_iface.sync_transcript_from_messages(
            st.session_state.messages,
            pending_user_text=pending_text,
        )
        with st.spinner("AI 正在回复..."):
            try:
                resp = chat_iface.send(
                    ChatRequest(
                        user_text=pending_text,
                        speaker_name=str(pending_speaker).strip() or "测试用户",
                    )
                )
                append_assistant_response_messages(st.session_state.messages, resp)
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"发送失败：{exc}"}
                )
            finally:
                st.session_state.pending_user_message = None
                st.session_state.pending_speaker_name = None
        st.rerun()
    elif pending_text and not chat_iface.has_agent():
        st.session_state.pending_user_message = None
        st.session_state.pending_speaker_name = None

    disabled = not chat_iface.has_agent() or pending_text is not None
    if (
        chat_iface.has_agent()
        and getattr(chat_iface, "mvp_runtime_active", False)
        and bool(st.session_state.get("m13_initiative_opt_in", False))
        and not pending_text
        and not pending_proactive
    ):
        if st.button(
            "让胡桃继续",
            key="btn_manual_proactive_continue",
            disabled=disabled,
            help="Manual bounded proactive message (requires opt-in above).",
        ):
            st.session_state.pending_proactive_continue = True
            st.session_state.pending_speaker_name = (
                str(st.session_state.current_speaker_name or "").strip() or "测试用户"
            )
            st.rerun()
    user_input = st.chat_input(
        "发消息" if not disabled else "先加载一个 persona...",
        disabled=disabled,
    )
    if user_input:
        speaker = str(st.session_state.current_speaker_name or "").strip() or "测试用户"
        st.session_state.messages.append(
            {
                "role": "user",
                "text": user_input,
                "speaker_name": speaker,
                "user_avatar_preset": str(st.session_state.get("chat_avatar_user_preset", "ocean")),
                "user_avatar_label": str(st.session_state.get("chat_avatar_user_label", "我")),
            }
        )
        st.session_state.pending_user_message = user_input
        st.session_state.pending_speaker_name = speaker
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
    latest_diag = chat_iface.latest_response_diagnostics()
    latest_thinking = latest_diag.get("llm_thinking_result", {})
    if not isinstance(latest_thinking, dict):
        latest_thinking = {}
    if not markdown or not latest:
        if latest_thinking:
            st.subheader("LLM 思考结果")
            st.json(latest_thinking, expanded=True)
            conscious_plan = latest_diag.get("conscious_plan", {})
            if isinstance(conscious_plan, dict) and conscious_plan:
                st.subheader("意识主循环结果")
                st.json(conscious_plan, expanded=False)
            return
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

    generation = latest.get("generation_diagnostics", {})
    trace_thinking = {}
    if isinstance(generation, dict):
        maybe_thinking = generation.get("llm_thinking_result", {})
        if isinstance(maybe_thinking, dict):
            trace_thinking = maybe_thinking
    if not trace_thinking:
        trace_thinking = latest_thinking
    if trace_thinking:
        st.subheader("LLM 思考结果")
        st.json(trace_thinking, expanded=True)

    st.subheader("Conscious.md")
    st.markdown(markdown)


def _safe_current_user_id(name: str) -> str:
    text = str(name or "").strip() or "default_user"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "default_user"


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _get_user_bucket(
    rows_by_user: object,
    *,
    speaker_name: str,
) -> tuple[dict[str, object], str]:
    rows = _as_dict(rows_by_user)
    if not rows:
        return {}, ""
    preferred = [
        _safe_current_user_id(speaker_name),
        str(speaker_name or "").strip(),
        "default_user",
    ]
    for key in preferred:
        value = rows.get(key)
        if isinstance(value, Mapping):
            return dict(value), key
    for key, value in sorted(rows.items(), key=lambda item: str(item[0])):
        if isinstance(value, Mapping):
            return dict(value), str(key)
    return {}, ""


def _mvp_substate_from_diagnostics(
    diagnostics: Mapping[str, object],
    key: str,
    state_key: str,
) -> dict[str, object]:
    section = _as_dict(diagnostics.get(key))
    state_after = _as_dict(section.get("state_after"))
    return state_after if state_after else _as_dict(section.get(state_key))


def _m12_2_has_user_models(state: Mapping[str, object]) -> bool:
    models = _as_dict(state.get("models_by_user"))
    return bool(models)


def _mvp_observation_sources(
    chat_iface: ChatInterface,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    diagnostics = chat_iface.latest_response_diagnostics()
    m121_state = _mvp_substate_from_diagnostics(
        diagnostics,
        "m12_1_personality",
        "m12_1_user_personality",
    )
    m122_state = _mvp_substate_from_diagnostics(
        diagnostics,
        "m12_2_reciprocal_role",
        "m12_2_reciprocal_role",
    )
    source = "diagnostics" if (m121_state or m122_state) else ""

    if not (m121_state and m122_state) or not _m12_2_has_user_models(m122_state):
        state = chat_iface.read_mvp_state_dict()
        if isinstance(state, dict):
            if not m121_state:
                m121_state = _as_dict(state.get("m12_1_user_personality"))
            file_m122_state = _as_dict(state.get("m12_2_reciprocal_role"))
            if not m122_state or (
                _m12_2_has_user_models(file_m122_state)
                and not _m12_2_has_user_models(m122_state)
            ):
                m122_state = file_m122_state
            if not source and (m121_state or m122_state):
                source = "system_files"
    return diagnostics, m121_state, m122_state, source


def _evidence_ref_text(value: object) -> str:
    refs: list[str] = []
    for ref in value if isinstance(value, list) else []:
        if isinstance(ref, Mapping):
            ref_id = str(ref.get("ref_id") or "").strip()
            turn_id = str(ref.get("turn_id") or "").strip()
            quote_id = str(ref.get("quote_id") or "").strip()
            refs.append(ref_id or f"{turn_id}:{quote_id}".strip(":"))
        else:
            refs.append(str(ref))
    return ", ".join(item for item in refs if item) or "无"


def _claim_table_rows(claims: object, *, limit: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for claim in _dict_rows(claims)[-limit:]:
        text = str(claim.get("claim_text_plain") or claim.get("claim_text_internal") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "判断": text,
                "置信": str(claim.get("confidence_band") or ""),
                "不确定性": str(claim.get("uncertainty_band") or ""),
                "状态": str(claim.get("status") or ""),
                "证据": _evidence_ref_text(claim.get("evidence_refs")),
            }
        )
    return rows


def _candidate_table_rows(candidates: object, *, limit: int = 6) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in _dict_rows(candidates)[-limit:]:
        action = str(item.get("plain_action") or item.get("plain_question") or "").strip()
        if not action:
            continue
        rows.append(
            {
                "待观察点": action,
                "收益": str(item.get("expected_gain_band") or ""),
                "风险": str(item.get("risk_band") or ""),
                "状态": str(item.get("status") or ("blocked" if item.get("blocked_by_safety") else "open")),
            }
        )
    return rows


def _profile_notes(profile: Mapping[str, object]) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    summary = _as_dict(profile.get("step_1_summary"))
    if summary.get("summary"):
        notes.append(("整体印象", str(summary.get("summary"))))
    prediction = _as_dict(profile.get("step_3_prediction_system_account"))
    pred_bits = [
        str(prediction.get("wants") or "").strip(),
        str(prediction.get("fears") or "").strip(),
        str(prediction.get("default_interpretation") or "").strip(),
    ]
    pred_text = " / ".join(bit for bit in pred_bits if bit)
    if pred_text:
        notes.append(("预测系统", pred_text))
    emotion = _as_dict(profile.get("step_5_emotion_and_defenses"))
    emotion_bits = [
        str(emotion.get("dominant_emotional_baseline") or "").strip(),
        str(emotion.get("threat_response") or "").strip(),
    ]
    emotion_text = " / ".join(bit for bit in emotion_bits if bit)
    if emotion_text:
        notes.append(("情绪与防御", emotion_text))
    relationship = _as_dict(profile.get("step_6_relationship_patterns"))
    rel_bits = [
        str(relationship.get("close_relationship_role") or "").strip(),
        str(relationship.get("recurring_loop_summary") or "").strip(),
        str(relationship.get("conflict_style") or "").strip(),
    ]
    rel_text = " / ".join(bit for bit in rel_bits if bit)
    if rel_text:
        notes.append(("关系模式", rel_text))
    return notes


def _free_energy_metrics(diagnostics: Mapping[str, object]) -> dict[str, object]:
    memory_dynamics = _as_dict(diagnostics.get("memory_dynamics"))
    control = _as_dict(memory_dynamics.get("control_guidance"))
    sharing = _as_dict(control.get("sharing_policy"))
    if not sharing:
        sharing = _as_dict(memory_dynamics.get("sharing_policy"))
    keys = (
        "current_free_energy",
        "expected_free_energy_after",
        "expected_free_energy_reduction",
        "net_free_energy_reduction",
        "risk_cost",
        "boundary_cost",
    )
    return {key: sharing[key] for key in keys if key in sharing}


def _mvp_observation_status(
    diagnostics: Mapping[str, object],
    mvp_state: Mapping[str, object] | None,
) -> dict[str, object]:
    m121_diag = _as_dict(diagnostics.get("m12_1_personality"))
    m122_diag = _as_dict(diagnostics.get("m12_2_reciprocal_role"))
    state = _as_dict(mvp_state)
    return {
        "has_state": bool(state or m121_diag or m122_diag),
        "m12_1_enabled": bool(
            m121_diag.get("enabled", state.get("m12_1_personality_enabled", False))
        ),
        "m12_2_enabled": bool(
            m122_diag.get("enabled", state.get("m12_2_reciprocal_role_enabled", False))
        ),
    }


def _metric_text(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value or "无")


def render_current_person_judgment() -> None:
    st.header("当前人物判断")
    chat_iface: ChatInterface = st.session_state.chat_iface

    if not chat_iface.has_agent():
        st.info("Load a persona to observe its current person judgments.")
        return

    speaker_name = str(st.session_state.get("current_speaker_name") or "").strip() or "测试用户"
    diagnostics, m121_state, m122_state, source = _mvp_observation_sources(chat_iface)
    mvp_state = chat_iface.read_mvp_state_dict()
    status = _mvp_observation_status(diagnostics, mvp_state)
    profile, profile_user_id = _get_user_bucket(
        _as_dict(m121_state).get("profiles_by_user"),
        speaker_name=speaker_name,
    )
    role_model, role_user_id = _get_user_bucket(
        _as_dict(m122_state).get("models_by_user"),
        speaker_name=speaker_name,
    )
    source_label = {"diagnostics": "最近一轮 diagnostics", "system_files": "MVP system files"}.get(source, "")
    st.caption(
        f"当前对话者：{speaker_name}"
        + (f" ｜ 观测来源：{source_label}" if source_label else "")
        + (f" ｜ 桶：{profile_user_id or role_user_id}" if (profile_user_id or role_user_id) else "")
    )

    persona_claims = _claim_table_rows(role_model.get("persona_about_user_claims"))
    user_about_persona_claims = _claim_table_rows(role_model.get("user_about_persona_claims"))
    profile_notes = _profile_notes(profile)
    free_energy = _free_energy_metrics(diagnostics)
    uncertainty_rows = _candidate_table_rows(role_model.get("unresolved_uncertainty_points"))
    candidate_rows = _candidate_table_rows(role_model.get("high_gain_candidates"))

    if not any([profile_notes, persona_claims, user_about_persona_claims, free_energy, uncertainty_rows, candidate_rows]):
        if status.get("has_state") and not (
            status.get("m12_1_enabled") or status.get("m12_2_enabled")
        ):
            st.info("当前人物判断模块未开启：M12.1 人物画像与 M12.2 相互判断都处于关闭状态。")
            st.caption("打开后需要至少一轮对话，才会生成可观察记录。")
            return
        if status.get("m12_1_enabled") or status.get("m12_2_enabled"):
            st.info("当前人物判断模块已开启，暂无可展示记录。")
            st.caption("请继续发送一轮消息，胡桃会在下一轮对话后写入新的观察判断。")
            return
        st.info("暂无观测记录")
        return

    st.subheader("自由能判断")
    if free_energy:
        cols = st.columns(min(4, len(free_energy)))
        labels = {
            "current_free_energy": "当前自由能",
            "expected_free_energy_after": "预期之后",
            "expected_free_energy_reduction": "预期降低",
            "net_free_energy_reduction": "净降低",
            "risk_cost": "风险成本",
            "boundary_cost": "边界成本",
        }
        for idx, (key, value) in enumerate(free_energy.items()):
            cols[idx % len(cols)].metric(labels.get(key, key), _metric_text(value))
    else:
        st.caption("暂无自由能数值记录；下面只展示已有判断。")

    left, right = st.columns(2)
    with left:
        st.subheader("胡桃对当前人物")
        if profile_notes:
            for title, text in profile_notes:
                st.markdown(f"**{title}**  \n{text}")
        if persona_claims:
            st.dataframe(pd.DataFrame(persona_claims), use_container_width=True, hide_index=True)
        elif not profile_notes:
            st.caption("暂无胡桃对当前人物的判断。")

    with right:
        st.subheader("胡桃猜测对方对自己")
        if user_about_persona_claims:
            st.dataframe(pd.DataFrame(user_about_persona_claims), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无对方如何判断胡桃的记录。")
        if uncertainty_rows or candidate_rows:
            st.subheader("下一步可观察点")
            rows = uncertainty_rows or candidate_rows
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    ensure_dialogue_client_id()
    init_session()
    inject_app_style()

    st.title("Segmentum Persona Runtime")
    _did_hdr = str(st.session_state.get("dialogue_client_id") or "")
    st.markdown(
        (
            '<div class="app-caption">'
            f"Loaded: {escape(st.session_state.loaded_persona or 'None')}"
            f"  |  Mode: {escape(st.session_state.chat_iface.generator_type.upper())}"
            f"  |  Storage: {escape(str(st.session_state.pm.storage_dir))}"
            f"  |  Tab: {escape(_did_hdr[:14])}{'…' if len(_did_hdr) > 14 else ''}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    render_sidebar()

    chat_tab, dashboard_tab, inner_tab, judgment_tab = st.tabs(["Chat", "Dashboard", "内心观察", "当前人物判断"])
    with chat_tab:
        render_chat()
    with dashboard_tab:
        render_dashboard()
    with inner_tab:
        render_inner_world()
    with judgment_tab:
        render_current_person_judgment()


if __name__ == "__main__":
    main()
