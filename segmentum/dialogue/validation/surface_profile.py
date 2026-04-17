from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import re
from statistics import mean
from typing import Iterable, Mapping, Protocol


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_PUNCT_CHARS = tuple(".,!?;:，。！？；：")
_STOP_TOKENS = frozenset(
    {
        "the",
        "and",
        "you",
        "that",
        "this",
        "with",
        "for",
        "了",
        "的",
        "是",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "啊",
        "吗",
        "吧",
    }
)


class _Predictor(Protocol):
    def predict(self, text: str) -> object: ...


@dataclass(slots=True)
class DialogueSurfaceProfile:
    """Train-only surface and semantic anchors used by M5.4 generation.

    The profile stores aggregate features and short snippets from training
    replies only. It intentionally does not store session ids or full transcripts.
    """

    source: str = "empty"
    reply_count: int = 0
    avg_reply_chars: float = 0.0
    median_reply_chars: int = 0
    ultra_short_ratio: float = 0.0
    punctuation_counts: dict[str, int] = field(default_factory=dict)
    opening_phrases: list[str] = field(default_factory=list)
    connector_phrases: list[str] = field(default_factory=list)
    top_tokens: list[str] = field(default_factory=list)
    action_phrases: dict[str, list[str]] = field(default_factory=dict)
    strategy_counts: dict[str, int] = field(default_factory=dict)
    partner_tokens: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DialogueSurfaceProfile":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            source=str(payload.get("source", "empty")),
            reply_count=int(payload.get("reply_count", 0) or 0),
            avg_reply_chars=float(payload.get("avg_reply_chars", 0.0) or 0.0),
            median_reply_chars=int(payload.get("median_reply_chars", 0) or 0),
            ultra_short_ratio=float(payload.get("ultra_short_ratio", 0.0) or 0.0),
            punctuation_counts={
                str(k): int(v)
                for k, v in dict(payload.get("punctuation_counts", {})).items()
                if isinstance(v, (int, float))
            },
            opening_phrases=[str(x) for x in payload.get("opening_phrases", [])],
            connector_phrases=[str(x) for x in payload.get("connector_phrases", [])],
            top_tokens=[str(x) for x in payload.get("top_tokens", [])],
            action_phrases={
                str(k): [str(x) for x in v]
                for k, v in dict(payload.get("action_phrases", {})).items()
                if isinstance(v, list)
            },
            strategy_counts={
                str(k): int(v)
                for k, v in dict(payload.get("strategy_counts", {})).items()
                if isinstance(v, (int, float))
            },
            partner_tokens={
                str(k): [str(x) for x in v]
                for k, v in dict(payload.get("partner_tokens", {})).items()
                if isinstance(v, list)
            },
        )


def tokenize_surface(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _session_partner_uid(user_uid: int, session: Mapping[str, object]) -> int:
    uid_a = _safe_int(session.get("uid_a", user_uid), user_uid)
    uid_b = _safe_int(session.get("uid_b", user_uid), user_uid)
    return uid_b if uid_a == user_uid else uid_a


def _iter_user_replies(
    sessions: Iterable[Mapping[str, object]],
    *,
    user_uid: int,
) -> Iterable[tuple[str, str]]:
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        partner_uid = str(_session_partner_uid(user_uid, session))
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            if _safe_int(turn.get("sender_uid"), -1) != int(user_uid):
                continue
            body = str(turn.get("body", "")).strip()
            if body:
                yield partner_uid, body


def _phrase(text: str, *, max_chars: int) -> str:
    compact = " ".join(str(text).strip().split())
    return compact[:max_chars]


def _top(counter: Counter[str], n: int) -> list[str]:
    return [key for key, _ in counter.most_common(max(0, int(n)))]


def build_surface_profile(
    user_dataset: Mapping[str, object],
    *,
    classifier: _Predictor | None = None,
    source: str = "train",
    max_items: int = 12,
) -> DialogueSurfaceProfile:
    user_uid = _safe_int(user_dataset.get("uid"), 0)
    sessions = user_dataset.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    replies = list(_iter_user_replies(sessions, user_uid=user_uid))
    if not replies:
        return DialogueSurfaceProfile(source=source)

    lengths = [len(text) for _, text in replies]
    sorted_lengths = sorted(lengths)
    mid = len(sorted_lengths) // 2
    median = sorted_lengths[mid] if sorted_lengths else 0
    token_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    connector_counts: Counter[str] = Counter()
    punct_counts: Counter[str] = Counter()
    action_phrases: dict[str, Counter[str]] = defaultdict(Counter)
    strategy_counts: Counter[str] = Counter()
    partner_tokens: dict[str, Counter[str]] = defaultdict(Counter)

    for partner_uid, text in replies:
        tokens = [tok for tok in tokenize_surface(text) if tok not in _STOP_TOKENS]
        token_counts.update(tokens)
        partner_tokens[partner_uid].update(tokens)
        prefix_counts[_phrase(text, max_chars=18)] += 1
        pieces = [piece.strip() for piece in re.split(r"[，。！？,.!?;；:：]", text) if piece.strip()]
        if pieces:
            connector_counts[_phrase(pieces[0], max_chars=16)] += 1
        punct_counts.update(ch for ch in text if ch in _PUNCT_CHARS)
        if classifier is not None:
            pred = classifier.predict(text)
            action = str(getattr(pred, "label_11", "elaborate"))
            strategy = str(getattr(pred, "label_3", "exploit"))
        else:
            action = "elaborate"
            strategy = "exploit"
        action_phrases[action][_phrase(text, max_chars=36)] += 1
        strategy_counts[strategy] += 1

    return DialogueSurfaceProfile(
        source=source,
        reply_count=len(replies),
        avg_reply_chars=round(float(mean(lengths)), 6),
        median_reply_chars=int(median),
        ultra_short_ratio=round(
            sum(1 for item in lengths if item <= 3) / float(max(1, len(lengths))),
            6,
        ),
        punctuation_counts={k: int(v) for k, v in punct_counts.items()},
        opening_phrases=_top(prefix_counts, max_items),
        connector_phrases=_top(connector_counts, max_items),
        top_tokens=_top(token_counts, max_items),
        action_phrases={k: _top(v, 5) for k, v in action_phrases.items()},
        strategy_counts={k: int(v) for k, v in strategy_counts.items()},
        partner_tokens={k: _top(v, 6) for k, v in partner_tokens.items()},
    )


def average_surface_profiles(
    profiles: Iterable[DialogueSurfaceProfile | Mapping[str, object]],
    *,
    source: str = "population_average",
    max_items: int = 12,
    include_surface_anchors: bool = True,
) -> DialogueSurfaceProfile:
    normalized = [
        p if isinstance(p, DialogueSurfaceProfile) else DialogueSurfaceProfile.from_dict(p)
        for p in profiles
    ]
    normalized = [p for p in normalized if p.reply_count > 0]
    if not normalized:
        return DialogueSurfaceProfile(source=source)

    total_replies = sum(p.reply_count for p in normalized)
    token_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    connector_counts: Counter[str] = Counter()
    punct_counts: Counter[str] = Counter()
    action_phrases: dict[str, Counter[str]] = defaultdict(Counter)
    strategy_counts: Counter[str] = Counter()
    for profile in normalized:
        weight = max(1, int(profile.reply_count))
        token_counts.update({token: weight for token in profile.top_tokens})
        prefix_counts.update({phrase: weight for phrase in profile.opening_phrases})
        connector_counts.update({phrase: weight for phrase in profile.connector_phrases})
        punct_counts.update(profile.punctuation_counts)
        strategy_counts.update(profile.strategy_counts)
        for action, phrases in profile.action_phrases.items():
            action_phrases[action].update({phrase: weight for phrase in phrases})

    return DialogueSurfaceProfile(
        source=source,
        reply_count=int(total_replies),
        avg_reply_chars=round(
            sum(p.avg_reply_chars * p.reply_count for p in normalized) / float(total_replies),
            6,
        ),
        median_reply_chars=int(round(mean([p.median_reply_chars for p in normalized]))),
        ultra_short_ratio=round(
            sum(p.ultra_short_ratio * p.reply_count for p in normalized) / float(total_replies),
            6,
        ),
        punctuation_counts={k: int(v) for k, v in punct_counts.items()},
        opening_phrases=_top(prefix_counts, max_items) if include_surface_anchors else [],
        connector_phrases=_top(connector_counts, max_items) if include_surface_anchors else [],
        top_tokens=_top(token_counts, max_items) if include_surface_anchors else [],
        action_phrases=(
            {k: _top(v, 5) for k, v in action_phrases.items()}
            if include_surface_anchors
            else {}
        ),
        strategy_counts={k: int(v) for k, v in strategy_counts.items()},
        partner_tokens={},
    )


def attach_surface_profile(agent: object, profile: DialogueSurfaceProfile | Mapping[str, object] | None) -> None:
    payload = (
        profile.to_dict()
        if isinstance(profile, DialogueSurfaceProfile)
        else DialogueSurfaceProfile.from_dict(profile).to_dict()
    )
    setattr(agent, "dialogue_surface_profile", payload)
