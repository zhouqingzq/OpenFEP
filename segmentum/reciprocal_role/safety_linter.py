"""Safety and consent linter for M12.2 candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Sequence


@dataclass(frozen=True)
class SafetyFinding:
    candidate_id: str
    rule: str
    severity: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "rule": self.rule,
            "severity": self.severity,
            "reason": self.reason,
        }


SENSITIVE_TERMS = (
    "trauma",
    "sexual",
    "sex life",
    "diagnosis",
    "mental health",
    "depression",
    "religion",
    "political",
    "immigration",
    "bank",
    "debt",
    "salary",
    "secret",
    "password",
    "private key",
    "family privacy",
    "创伤",
    "性生活",
    "诊断",
    "宗教",
    "政治",
    "移民",
    "债务",
    "密码",
)

MANIPULATION_TERMS = (
    "make them trust",
    "increase trust regardless",
    "keep them engaged",
    "make them attached",
    "dependency",
    "loneliness",
    "exploit",
    "pressure them",
    "让用户更信任",
    "保持依赖",
    "利用孤独",
    "施压",
)


def lint_candidate(candidate: object) -> tuple[SafetyFinding, ...]:
    candidate_id = str(getattr(candidate, "candidate_id", ""))
    text = " ".join(
        str(getattr(candidate, attr, "") or "")
        for attr in ("plain_action", "target_axis", "kind", "consent_requirement")
    ).casefold()
    findings: list[SafetyFinding] = []
    if any(_term_matches(term, text) for term in SENSITIVE_TERMS):
        findings.append(
            SafetyFinding(
                candidate_id=candidate_id,
                rule="over_intimate_or_sensitive",
                severity="block",
                reason="candidate asks for sensitive details not required by the current dialogue",
            )
        )
    if any(_term_matches(term, text) for term in MANIPULATION_TERMS):
        findings.append(
            SafetyFinding(
                candidate_id=candidate_id,
                rule="manipulative_or_engagement_seeking",
                severity="block",
                reason="candidate is aimed at pressure, attachment, or trust rather than clarity",
            )
        )
    return tuple(findings)


def _term_matches(term: str, text: str) -> bool:
    folded = term.casefold()
    if folded == "dependency" and any(phrase in text for phrase in ("dependency injection", "package dependency", "software dependency")):
        return False
    if folded == "secret" and any(phrase in text for phrase in ("secret recipe", "secretary")):
        return False
    if folded.isascii():
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(folded)}(?![a-z0-9_])", text))
    return folded in text


def apply_safety_linter(candidates: Sequence[object]) -> tuple[tuple[object, ...], tuple[SafetyFinding, ...]]:
    allowed: list[object] = []
    findings: list[SafetyFinding] = []
    for candidate in candidates:
        local = lint_candidate(candidate)
        findings.extend(local)
        blocked = bool(local) or bool(getattr(candidate, "blocked_by_safety", False))
        if blocked:
            if hasattr(candidate, "blocked_by_safety"):
                try:
                    candidate = replace(candidate, blocked_by_safety=True)
                except TypeError:
                    pass
            continue
        allowed.append(candidate)
    return tuple(allowed), tuple(findings)


def findings_to_dict(findings: Sequence[SafetyFinding]) -> list[dict[str, str]]:
    return [finding.to_dict() for finding in findings]
