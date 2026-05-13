"""Plain-language linter for M12.2 user-facing surfaces.

This layer only blocks explicit jargon. Higher-order observers can be plugged
in to judge whether a phrase is exposing internal reasoning in context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


ALWAYS_BLOCKED = (
    "prediction error",
    "free energy",
    "expected free energy",
    "expected information gain",
    "information gain",
    "bayesian",
    "prior update",
    "likelihood function",
    "latent state",
    "hidden variable",
    "active inference",
    "predictive system",
    "predictive model",
    "预测误差",
    "自由能",
    "期望自由能",
    "信息增益",
    "贝叶斯",
    "后验更新",
    "先验更新",
    "似然函数",
    "潜在状态",
    "隐变量",
    "主动推断",
    "预测系统",
    "预测模型",
)


@dataclass(frozen=True)
class PlainLanguageFinding:
    token: str
    section: str
    rule: str
    raw_quote: str

    def to_dict(self) -> dict[str, str]:
        return {
            "token": self.token,
            "section": self.section,
            "rule": self.rule,
            "raw_quote": self.raw_quote,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlainLanguageFinding":
        return cls(
            token=str(payload.get("token", "")),
            section=str(payload.get("section", "")),
            rule=str(payload.get("rule", "")),
            raw_quote=str(payload.get("raw_quote", "")),
        )


PlainLanguageObserver = Callable[[str, str], Sequence[PlainLanguageFinding]]


def lint_text(
    text: str,
    *,
    section: str = "unknown",
    observer: PlainLanguageObserver | None = None,
) -> tuple[PlainLanguageFinding, ...]:
    source = str(text or "")
    folded = source.casefold()
    findings: list[PlainLanguageFinding] = []
    for token in ALWAYS_BLOCKED:
        idx = folded.find(token.casefold())
        if idx >= 0:
            findings.append(_finding(source, idx, token, section, "always_blocked_jargon"))
    if observer is not None:
        findings.extend(observer(source, section))
    return tuple(_dedupe(findings))


def lint_user_facing_fields(rows: Sequence[Mapping[str, object]] | Mapping[str, object]) -> tuple[PlainLanguageFinding, ...]:
    items: Sequence[Mapping[str, object]]
    items = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, Mapping)) else (rows,)  # type: ignore[assignment]
    findings: list[PlainLanguageFinding] = []
    for row in items:
        if not isinstance(row, Mapping):
            continue
        section = str(row.get("section", row.get("kind", "unknown")))
        for key in ("claim_text_plain", "plain_question", "plain_action", "plain_reason", "content_summary"):
            if key in row:
                findings.extend(lint_text(str(row.get(key, "")), section=f"{section}.{key}"))
    return tuple(_dedupe(findings))


def passes_plain_language(text: str) -> bool:
    return not lint_text(text)


def _finding(source: str, idx: int, token: str, section: str, rule: str) -> PlainLanguageFinding:
    start = max(0, idx - 24)
    end = min(len(source), idx + len(token) + 24)
    return PlainLanguageFinding(token=token, section=section, rule=rule, raw_quote=source[start:end])


def _dedupe(findings: Sequence[PlainLanguageFinding]) -> list[PlainLanguageFinding]:
    out: list[PlainLanguageFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (finding.token.casefold(), finding.section, finding.rule)
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out
