"""Plain-language linter for M12.1 user-facing surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams


@dataclass(frozen=True)
class LinterFinding:
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
    def from_dict(cls, payload: Mapping[str, object]) -> "LinterFinding":
        return cls(
            token=str(payload.get("token", "")),
            section=str(payload.get("section", "")),
            rule=str(payload.get("rule", "")),
            raw_quote=str(payload.get("raw_quote", "")),
        )


def lint_user_facing_text(
    text: str,
    *,
    section: str,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[LinterFinding, ...]:
    """Pure function over text and configured token sets."""
    source = str(text or "")
    folded = source.casefold()
    findings: list[LinterFinding] = []
    findings.extend(
        _find_tokens(
            source,
            folded,
            section=section,
            tokens=hyperparams.forbidden_user_facing_tokens_extra,
            rule="engineering_jargon",
        )
    )
    findings.extend(
        _find_tokens(
            source,
            folded,
            section=section,
            tokens=hyperparams.forbidden_clinical_label_tokens,
            rule="clinical_label",
        )
    )
    findings.extend(
        _find_tokens(
            source,
            folded,
            section=section,
            tokens=hyperparams.forbidden_moral_or_chicken_soup_tokens,
            rule="moral_or_chicken_soup",
        )
    )
    return tuple(findings)


def lint_user_facing_fields(
    fields: Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[LinterFinding, ...]:
    """Lint only explicit user-facing fields, not schema keys or logs."""
    rows: Iterable[Mapping[str, object]]
    rows = fields if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes, Mapping)) else (fields,)  # type: ignore[assignment]
    findings: list[LinterFinding] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        section = str(row.get("section", row.get("section_kind", "unknown")))
        for key in ("content_summary", "summary", "reason", "why", "why_retrieved", "protects_what", "short_term_benefit", "long_term_cost"):
            if key in row:
                findings.extend(lint_user_facing_text(str(row.get(key, "")), section=f"{section}.{key}", hyperparams=hyperparams))
    return tuple(findings)


def lint_report_dict(
    report: Mapping[str, object],
    *,
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[LinterFinding, ...]:
    """Lint the report's rendered section text only."""
    findings: list[LinterFinding] = []
    sections = report.get("sections", [])
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        return ()
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        section_kind = str(section.get("section_kind", "unknown"))
        rendered = section.get("rendered")
        if isinstance(rendered, str):
            findings.extend(lint_user_facing_text(rendered, section=section_kind, hyperparams=hyperparams))
        content = section.get("content")
        findings.extend(_lint_nested_content(content, section=section_kind, hyperparams=hyperparams))
    return tuple(findings)


def _lint_nested_content(
    value: object,
    *,
    section: str,
    hyperparams: M121Hyperparams,
) -> tuple[LinterFinding, ...]:
    findings: list[LinterFinding] = []
    if isinstance(value, str):
        findings.extend(lint_user_facing_text(value, section=section, hyperparams=hyperparams))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in {"status", "confidence_band", "evidence_refs", "claim_state", "hyperparams_version", "last_updated_turn_id", "core_belief", "defense_kind", "loop_stage"}:
                continue
            findings.extend(_lint_nested_content(child, section=f"{section}.{key_text}", hyperparams=hyperparams))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for idx, child in enumerate(value):
            findings.extend(_lint_nested_content(child, section=f"{section}[{idx}]", hyperparams=hyperparams))
    return tuple(findings)


def _find_tokens(
    source: str,
    folded: str,
    *,
    section: str,
    tokens: Sequence[str],
    rule: str,
) -> tuple[LinterFinding, ...]:
    findings: list[LinterFinding] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        needle = str(token or "")
        if not needle:
            continue
        idx = folded.find(needle.casefold())
        if idx < 0:
            continue
        key = (needle.casefold(), section)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, idx - 24)
        end = min(len(source), idx + len(needle) + 24)
        findings.append(
            LinterFinding(
                token=needle,
                section=section,
                rule=rule,
                raw_quote=source[start:end],
            )
        )
    return tuple(findings)
