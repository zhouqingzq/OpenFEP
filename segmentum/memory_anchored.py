"""M8 Anchored Memory Contract: dialogue fact extraction, permission filter, citation guard.

This module encodes user dialogue facts as auditable AnchoredMemoryItem records
with provenance, status tracking, and visibility control.  It does NOT participate
in the FEP memory pipeline (decay / consolidation / replay).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from uuid import uuid4

# ── Status & visibility literals ──────────────────────────────────────────

FactStatus = Literal['asserted', 'corroborated', 'retracted', 'hypothesis']
FactVisibility = Literal['explicit', 'strategy_only', 'private', 'forbidden']
FactMemoryType = Literal[
    'user_fact',
    'user_asserted_fact',
    'relationship_fact',
    'project_fact',
    'preference',
    'task_state',
    'hypothesis',
    'system_inferred_hypothesis',
    'agent_self_utterance',
    'private_state',
]


# ── AnchoredMemoryItem ────────────────────────────────────────────────────

@dataclass
class AnchoredMemoryItem:
    """A dialogue fact anchored to a specific utterance with provenance.

    Status semantics:
    - ``asserted`` means *the speaker said this* — it is NOT a claim of objective
      truth. The proposition records what was stated, not what is verified.
    - ``corroborated`` means the fact has been independently confirmed. When
      implemented, ``corroboration_source`` should distinguish between:
      ``user_repeat``, ``user_confirmation``, ``tool``, ``external_evidence``.
      TODO: add ``corroboration_source`` field to disambiguate confirmation paths.
    - ``hypothesis`` must carry a confidence score and cannot enter the
      explicit-fact bucket.
    - ``retracted`` items must never appear in memory_context or citation support.
      Retracted has HIGHEST priority — if a proposition is retracted, it overrides
      any asserted/corroborated copy of the same fact.

    Memory type guide:
    - ``user_asserted_fact``: explicit user statement ("我叫周青")
    - ``agent_self_utterance``: agent said something about itself; must NOT
      be treated as a user fact in future turns.
    - ``system_inferred_hypothesis``: inferred by the system, not stated by user.
    """

    memory_id: str = field(default_factory=lambda: str(uuid4()))
    speaker: str = ''
    utterance_id: str = ''
    turn_id: str = ''
    proposition: str = ''
    source_text: str = ''
    status: FactStatus = 'asserted'
    confidence: float = 1.0
    visibility: FactVisibility = 'explicit'
    memory_type: FactMemoryType = 'user_fact'
    created_turn_id: str = ''
    last_confirmed_turn_id: str | None = None
    contradiction_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at_cycle: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            'memory_id': self.memory_id,
            'speaker': self.speaker,
            'utterance_id': self.utterance_id,
            'turn_id': self.turn_id,
            'proposition': self.proposition,
            'source_text': self.source_text,
            'status': self.status,
            'confidence': self.confidence,
            'visibility': self.visibility,
            'memory_type': self.memory_type,
            'created_turn_id': self.created_turn_id,
            'last_confirmed_turn_id': self.last_confirmed_turn_id,
            'contradiction_ids': list(self.contradiction_ids),
            'tags': list(self.tags),
            'created_at_cycle': self.created_at_cycle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AnchoredMemoryItem:
        return cls(
            memory_id=str(payload.get('memory_id', str(uuid4()))),
            speaker=str(payload.get('speaker', '')),
            utterance_id=str(payload.get('utterance_id', '')),
            turn_id=str(payload.get('turn_id', '')),
            proposition=str(payload.get('proposition', '')),
            source_text=str(payload.get('source_text', '')),
            status=str(payload.get('status', 'asserted')),  # type: ignore[arg-type]
            confidence=float(payload.get('confidence', 1.0)),
            visibility=str(payload.get('visibility', 'explicit')),  # type: ignore[arg-type]
            memory_type=str(payload.get('memory_type', 'user_fact')),  # type: ignore[arg-type]
            created_turn_id=str(payload.get('created_turn_id', '')),
            last_confirmed_turn_id=(
                str(payload['last_confirmed_turn_id'])
                if payload.get('last_confirmed_turn_id') is not None
                else None
            ),
            contradiction_ids=[
                str(x) for x in payload.get('contradiction_ids', []) or []
            ],
            tags=[str(t) for t in payload.get('tags', []) or []],
            created_at_cycle=int(payload.get('created_at_cycle', 0)),
        )


# ── MemoryPermissionBuckets ───────────────────────────────────────────────

class MemoryPermissionBuckets(NamedTuple):
    explicit_facts: list[AnchoredMemoryItem]
    cautious_hypotheses: list[AnchoredMemoryItem]
    strategy_only: list[AnchoredMemoryItem]
    forbidden: list[AnchoredMemoryItem]


# ── CitationAuditResult (legacy compat, thin wrapper) ─────────────────────

@dataclass
class CitationAuditResult:
    """Legacy audit result — kept for backward compat. Prefer MemoryCitationAudit."""
    flags: list[str] = field(default_factory=list)
    cited_anchored_ids: list[str] = field(default_factory=list)
    hallucinated_detail_risk: bool = False
    retracted_fact_referenced: bool = False
    hypothesis_as_fact: bool = False
    forbidden_topic_referenced: bool = False


# ── MemoryCitationAudit (M8.5 structured output) ──────────────────────────

ViolationType = Literal[
    'retracted_fact_referenced',
    'forbidden_topic_referenced',
    'hypothesis_as_fact',
    'unanchored_memory_claim',
    'hallucinated_detail_risk',
]


@dataclass
class MemoryCitationViolation:
    type: ViolationType
    span: str = ''
    message: str = ''
    suggested_action: Literal['retry', 'downgrade', 'log_only', 'block'] = 'log_only'


@dataclass
class MemoryCitationAudit:
    """Structured audit output (M8.5).

    Fields:
    - has_violation: any violation at all
    - has_blocking_violation: at least one violation with suggested_action='block' or 'retry'
    - risk_level: aggregate severity
    - violations: detailed list of violations found
    """
    has_violation: bool = False
    has_blocking_violation: bool = False
    risk_level: Literal['none', 'low', 'medium', 'high', 'blocking'] = 'none'
    violations: list[MemoryCitationViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            'has_violation': self.has_violation,
            'has_blocking_violation': self.has_blocking_violation,
            'risk_level': self.risk_level,
            'violations': [
                {
                    'type': v.type,
                    'span': v.span,
                    'message': v.message,
                    'suggested_action': v.suggested_action,
                }
                for v in self.violations
            ],
        }


# ── DialogueFactExtractor ─────────────────────────────────────────────────

# Chinese regex patterns keyed by memory_type.
# Use a helper to build patterns containing CJK characters without tripping
# any encoding pipeline issues.

def _cjk_class() -> str:
    """Return a regex character class covering common CJK Unified Ideographs."""
    return '[一-鿿]'


_C = _cjk_class()

_NAME_PATTERNS: list[tuple[str, str]] = [
    (r'我叫[“‘“「]?(' + _C + r'{2,4})[”’”」]?', 'user_fact'),
    (r'我的名字是[“‘「]?(' + _C + r'{2,4})[”’」]?', 'user_fact'),
    (r'你可以叫我[“‘「]?(' + _C + r'{2,4})[”’」]?', 'user_fact'),
]

_RELATIONSHIP_PATTERNS: list[tuple[str, str]] = [
    # Colloquial Chinese often drops 的; make it optional.
    (r'(' + _C + r'{2,4})是我(?:的)?(同学|朋友|同事|老板|老师|家人|室友|邻居|队友|搭档|领导|下属|亲戚|发小|闺蜜|哥们|兄弟|姐妹)', 'relationship_fact'),
    (r'(' + _C + r'{2,4})是(?:我|咱)(?:的)?(同学|朋友|同事|老板|老师|家人|室友|邻居|队友|搭档|领导|下属|亲戚|发小|闺蜜|哥们|兄弟|姐妹)', 'relationship_fact'),
    (r'我和(' + _C + r'{2,4})是(同学|朋友|同事|老板|老师|家人|室友|邻居|队友|搭档|领导|下属|亲戚|发小|闺蜜|哥们|兄弟|姐妹)关系', 'relationship_fact'),
]

_MILESTONE_PATTERNS: list[tuple[str, str]] = [
    (r'我已经完成了\s*(M\d+(?:\.\d+)?)', 'project_fact'),
    (r'(M\d+(?:\.\d+)?)\s*已经做完了', 'project_fact'),
    (r'(M\d+(?:\.\d+)?)\s*已完成', 'project_fact'),
    (r'(M\d+(?:\.\d+)?)\s*已经通过', 'project_fact'),
    (r'完成了?\s*(M\d+(?:\.\d+)?)', 'project_fact'),
]

_TASK_STATE_PATTERNS: list[tuple[str, str]] = [
    (r'我正在(做|设计|开发|写|重构|检查|修改|审查|调试|测试)(\S+)', 'task_state'),
    (r'我在(做|设计|开发|写|重构|检查|修改|审查|调试|测试)(\S+)', 'task_state'),
    (r'我当前的任务是(\S+)', 'task_state'),
    (r'我现在在(做|弄|处理|搞)(\S+)', 'task_state'),
]

_PREFERENCE_PATTERNS: list[tuple[str, str]] = [
    (r'我(?:很|非常|比较|特别|更|不太|不)?喜欢(\S.{0,30}?)(?:[，,。！!]|$)', 'preference'),
    (r'我(?:很|比较|特别|不太)?讨厌(\S+)', 'preference'),
    (r'我偏好(\S+)', 'preference'),
]

_BEHAVIOR_PREFERENCE_PATTERNS: list[tuple[str, str]] = [
    (r'以后希望你能?(\S.{0,40}?)(?:[。！!]|$)', 'preference'),
    (r'从现在开始希望你能?(\S.{0,40}?)(?:[。！!]|$)', 'preference'),
    (r'以后请(?:你)?(\S.{0,40}?)(?:[。！!]|$)', 'preference'),
    (r'我希望你以后能?(\S.{0,40}?)(?:[。！!]|$)', 'preference'),
    (r'以后能?(\S.{0,40}?)吗[？?]?', 'preference'),
]

_CORRECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'不是[“‘「]?([^，,是。]{1,30})[”’」]?[，,]\s*是[“‘「]?([^，,。]{1,30})[”’」]?'),
    re.compile(r'不对[，,]\s*(?:应该是\s*)?[“‘「]?([^，,。]{1,50})[”’」]?'),
    re.compile(r'(?:我说错了|我记错了)[，,]\s*(?:其实是\s*)?[“‘「]?([^，,。]{1,50})[”’」]?'),
    re.compile(r'之前说的?\s*不对[，,]\s*([^，,。]{1,50})'),
]


def _clean_proposition(text: str) -> str:
    """Strip whitespace and trailing punctuation from a proposition."""
    return text.strip().rstrip('，,。.！!？?')


def _fuzzy_proposition_match(
    proposition: str, existing_items: list[AnchoredMemoryItem],
) -> AnchoredMemoryItem | None:
    """Find an existing item with a similar proposition (simple substring overlap)."""
    prop_clean = _clean_proposition(proposition)
    if len(prop_clean) < 2:
        return None
    for item in existing_items:
        if item.status == 'retracted':
            continue
        existing_prop = _clean_proposition(item.proposition)
        if len(existing_prop) < 2:
            continue
        if prop_clean in existing_prop or existing_prop in prop_clean:
            return item
    return None


class DialogueFactExtractor:
    """Rule-based extractor that produces AnchoredMemoryItem from user turns.

    First version is purely regex-driven — no LLM calls.  If no clear fact
    matches, the extractor returns an empty list rather than guessing.

    M8.5 user/agent split:
    - ``speaker='user'`` → items can be ``user_asserted_fact`` with
      ``visibility='explicit'``.
    - ``speaker='agent'`` → items are forced to
      ``memory_type='agent_self_utterance'`` with
      ``visibility='strategy_only'``. Agent utterances must NEVER become
      explicit user facts.
    """

    def extract(
        self,
        text: str,
        turn_id: str,
        utterance_id: str,
        speaker: str = 'user',
        existing_items: list[AnchoredMemoryItem] | None = None,
        *,
        current_cycle: int = 0,
    ) -> list[AnchoredMemoryItem]:
        """Extract anchored facts from *text* and return new AnchoredMemoryItem entries.

        *existing_items* is used to detect corrections (retract old, assert new).
        *current_cycle* sets ``created_at_cycle`` on new items (for lifecycle pruning).
        """
        existing = list(existing_items or [])
        results: list[AnchoredMemoryItem] = []

        text = text.strip()
        if not text:
            return results

        # M8.5: agent turns produce agent_self_utterance, not user facts
        is_agent = speaker != 'user'

        def _make(
            proposition: str,
            source_text: str,
            status: FactStatus = 'asserted',
            confidence: float = 1.0,
            visibility: FactVisibility = 'explicit',
            memory_type: FactMemoryType = 'user_fact',
            tags: list[str] | None = None,
        ) -> AnchoredMemoryItem:
            if is_agent:
                return AnchoredMemoryItem(
                    speaker=speaker,
                    utterance_id=utterance_id,
                    turn_id=turn_id,
                    proposition=proposition,
                    source_text=source_text,
                    status='asserted',
                    confidence=0.5,
                    visibility='strategy_only',
                    memory_type='agent_self_utterance',
                    created_turn_id=turn_id,
                    tags=['agent_utterance'] + (tags or []),
                )
            return AnchoredMemoryItem(
                speaker=speaker,
                utterance_id=utterance_id,
                turn_id=turn_id,
                proposition=proposition,
                source_text=source_text,
                status=status,
                confidence=confidence,
                visibility=visibility,
                memory_type=memory_type,
                created_turn_id=turn_id,
                tags=tags or [],
            )

        # ── 1. Name declarations ──────────────────────────────────────
        for pattern, mem_type in _NAME_PATTERNS:
            m = re.search(pattern, text)
            if m:
                name = m.group(1)
                prop = f'用户的名字是{name}'
                if not _fuzzy_proposition_match(prop, existing + results):
                    results.append(_make(
                        prop, text, status='asserted', confidence=1.0,
                        visibility='explicit',
                        memory_type='user_asserted_fact' if not is_agent else 'agent_self_utterance',
                        tags=['name', name],
                    ))
                break

        # ── 2. Relationship statements ─────────────────────────────────
        for pattern, mem_type in _RELATIONSHIP_PATTERNS:
            for m in re.finditer(pattern, text):
                person = m.group(1)
                relation = m.group(2)
                prop = f'{person}是用户的{relation}'
                if not _fuzzy_proposition_match(prop, existing + results):
                    results.append(_make(
                        prop, text, status='asserted', confidence=1.0,
                        visibility='explicit',
                        memory_type='relationship_fact' if not is_agent else 'agent_self_utterance',
                        tags=['relationship', person, relation],
                    ))

        # ── 3. Milestone completions ───────────────────────────────────
        for pattern, mem_type in _MILESTONE_PATTERNS:
            for m in re.finditer(pattern, text):
                milestone = m.group(1)
                prop = f'用户已经完成了{milestone}'
                if not _fuzzy_proposition_match(prop, existing + results):
                    results.append(_make(
                        prop, text, status='asserted', confidence=1.0,
                        visibility='explicit',
                        memory_type='project_fact' if not is_agent else 'agent_self_utterance',
                        tags=['milestone', milestone],
                    ))

        # ── 4. Current task / work-in-progress ─────────────────────────
        for pattern, mem_type in _TASK_STATE_PATTERNS:
            for m in re.finditer(pattern, text):
                verb = m.group(1)
                obj = m.group(2) if m.lastindex and m.lastindex >= 2 else ''
                task_desc = f'{verb}{obj}'.strip()
                if len(task_desc) < 2:
                    continue
                prop = f'用户当前在{task_desc}'
                if not _fuzzy_proposition_match(prop, existing + results):
                    results.append(_make(
                        prop, text, status='asserted', confidence=0.9,
                        visibility='explicit',
                        memory_type='task_state' if not is_agent else 'agent_self_utterance',
                        tags=['task', task_desc],
                    ))

        # ── 5. Preferences ─────────────────────────────────────────────
        for pattern, mem_type in _PREFERENCE_PATTERNS:
            for m in re.finditer(pattern, text):
                pref = _clean_proposition(m.group(1))
                if len(pref) < 2:
                    continue
                full_match = m.group(0)
                if any(neg in full_match for neg in ('不喜欢', '讨厌', '不太喜欢')):
                    prop = f'用户不喜欢{pref}'
                else:
                    prop = f'用户喜欢{pref}'
                if not _fuzzy_proposition_match(prop, existing + results):
                    results.append(_make(
                        prop, text, status='asserted', confidence=0.95,
                        visibility='explicit' if not is_agent else 'strategy_only',
                        memory_type='preference' if not is_agent else 'agent_self_utterance',
                        tags=['preference', pref],
                    ))
                break

        # ── 6. Behavior preferences ────────────────────────────────────
        if not is_agent:
            for pattern, _mem_type in _BEHAVIOR_PREFERENCE_PATTERNS:
                m = re.search(pattern, text)
                if m:
                    behavior = _clean_proposition(m.group(1))
                    if len(behavior) >= 2:
                        prop = f'用户希望AI{behavior}'
                        if not _fuzzy_proposition_match(prop, existing + results):
                            results.append(AnchoredMemoryItem(
                                speaker=speaker, utterance_id=utterance_id,
                                turn_id=turn_id, proposition=prop,
                                source_text=text, status='asserted',
                                confidence=0.9, visibility='strategy_only',
                                memory_type='preference',
                                created_turn_id=turn_id,
                                tags=['behavior_preference'],
                            ))
                    break

        # ── 7. Corrections (retract old, assert new) ────────────────────
        # M8.5: only user corrections retract/assert; agent corrections are logged differently
        if not is_agent:
            for corr_pattern in _CORRECTION_PATTERNS:
                m = corr_pattern.search(text)
                if not m:
                    continue

                if m.re is _CORRECTION_PATTERNS[0]:
                    old_text = m.group(1).strip()
                    new_text = m.group(2).strip()
                elif m.re is _CORRECTION_PATTERNS[1]:
                    old_text = ''
                    new_text = m.group(1).strip()
                elif m.re in (_CORRECTION_PATTERNS[2], _CORRECTION_PATTERNS[3]):
                    old_text = ''
                    new_text = m.group(1).strip()
                else:
                    continue

                if old_text and existing:
                    for old_item in existing:
                        if old_item.status == 'retracted':
                            continue
                        if old_text in old_item.proposition or old_item.proposition in old_text:
                            old_item.status = 'retracted'
                            old_item.contradiction_ids.append(f'corrected-at-{turn_id}')

                if new_text and len(new_text) >= 2:
                    prop = _clean_proposition(new_text)
                    if not any(kw in prop for kw in ('用户', '名字', '同学', '朋友', '同事', '完成', '喜欢', '在')):
                        prop = f'用户陈述：{prop}'
                    if not _fuzzy_proposition_match(prop, existing + results):
                        results.append(AnchoredMemoryItem(
                            speaker=speaker, utterance_id=utterance_id,
                            turn_id=turn_id, proposition=prop, source_text=text,
                            status='asserted', confidence=1.0, visibility='explicit',
                            memory_type='user_asserted_fact', created_turn_id=turn_id,
                            tags=['correction'],
                        ))
                break

        if current_cycle:
            for item in results:
                if not item.created_at_cycle:
                    item.created_at_cycle = current_cycle

        return results


# ── MemoryPermissionFilter ────────────────────────────────────────────────

class MemoryPermissionFilter:
    """Categorises AnchoredMemoryItem into permission-governed buckets.

    Rules (applied in order):
    1. ``retracted`` → ``forbidden`` (always).
    2. ``forbidden`` visibility → ``forbidden``.
    3. ``agent_self_utterance`` → ``strategy_only`` (M8.5: agent utterances
       must NOT enter explicit_facts).
    4. ``asserted`` / ``corroborated`` + ``explicit`` → ``explicit_facts``.
    5. ``hypothesis`` / ``system_inferred_hypothesis`` → ``cautious_hypotheses``.
    6. ``strategy_only`` / ``private`` visibility → ``strategy_only``.
    7. Anything else → ``forbidden`` (safety default).

    Priority: retracted/forbidden > corroborated > asserted > hypothesis > legacy
    """

    @staticmethod
    def filter(items: list[AnchoredMemoryItem]) -> MemoryPermissionBuckets:
        explicit_facts: list[AnchoredMemoryItem] = []
        cautious_hypotheses: list[AnchoredMemoryItem] = []
        strategy_only: list[AnchoredMemoryItem] = []
        forbidden: list[AnchoredMemoryItem] = []

        for item in items:
            if item.status == 'retracted':
                forbidden.append(item)
                continue
            if item.visibility == 'forbidden':
                forbidden.append(item)
                continue

            # M8.5: agent_self_utterance must never enter explicit_facts
            if item.memory_type == 'agent_self_utterance':
                strategy_only.append(item)
                continue

            if item.status in ('asserted', 'corroborated') and item.visibility == 'explicit':
                explicit_facts.append(item)
            elif item.status == 'hypothesis' or item.memory_type == 'system_inferred_hypothesis':
                cautious_hypotheses.append(item)
            elif item.visibility in ('strategy_only', 'private'):
                strategy_only.append(item)
            else:
                forbidden.append(item)

        return MemoryPermissionBuckets(
            explicit_facts=explicit_facts,
            cautious_hypotheses=cautious_hypotheses,
            strategy_only=strategy_only,
            forbidden=forbidden,
        )


# ── MemoryCitationGuard ───────────────────────────────────────────────────

_MEMORY_CLAIM_PATTERNS: list[tuple[str, str]] = [
    (r'我记得', '我记得'),
    (r'上次', '上次'),
    (r'你曾经', '你曾经'),
    (r'你之前(?:说|提到|讲过)', '你之前说/提到/讲过'),
    (r'那次', '那次'),
    (r'你们(?:俩|两个|一起)', '你们一起'),
    (r'(?:一起)?聚[餐会]|吃过?饭|喝过?酒|约过?', '聚餐/吃饭/喝酒'),
    (r'(?:去了?|来过?|逛了?|玩了?)(?:一趟|一次|一下)', '具体地点/出行'),
    (r'在你家|在我家|在公司|在学校|在餐厅', '具体地点'),
    (r'你说过?你(?:喜欢|讨厌|想|要|觉得|认为)', '引用用户观点'),
    (r'你上次(?:说|提到|讲过|做的|写的)', '引用用户历史行为'),
]

_HYPOTHESIS_AS_FACT_PATTERNS: list[str] = [
    r'你肯定是',
    r'你肯定',
    r'你一定',
    r'你绝对',
    r'你确实',
    r'你真的是',
    r'你就是',
]

_FORBIDDEN_TOPICS: list[str] = [
    '创伤',
    'trauma',
    '秘密',
]


class MemoryCitationGuard:
    """Lightweight post-hoc audit for memory-hallucination risk.

    Checks a generated reply against anchored items and flags:
    - Memory-claim language without supporting anchored facts
    - References to retracted facts
    - Hypotheses stated as certain facts
    - Forbidden-topic references

    M8.5: ``audit_structured`` returns ``MemoryCitationAudit`` with
    risk_level and per-violation suggested_action for runtime dispatch.
    ``audit`` returns legacy ``CitationAuditResult`` for backward compat.
    """

    @staticmethod
    def audit(
        reply_text: str,
        anchored_items: list[AnchoredMemoryItem] | None = None,
    ) -> CitationAuditResult:
        """Legacy audit returning CitationAuditResult (backward compat)."""
        items = list(anchored_items or [])
        result = CitationAuditResult()

        explicit_propositions: set[str] = set()
        retracted_propositions: set[str] = set()

        for item in items:
            prop_lower = item.proposition.lower().strip()
            if item.status == 'retracted':
                retracted_propositions.add(prop_lower)
            elif item.status in ('asserted', 'corroborated') and item.visibility == 'explicit':
                explicit_propositions.add(prop_lower)

        # 1. Check for memory-claim language
        for pattern, label in _MEMORY_CLAIM_PATTERNS:
            if re.search(pattern, reply_text):
                supported = any(
                    prop in reply_text[:min(len(prop) + 20, len(reply_text))]
                    for prop in explicit_propositions if len(prop) >= 4
                )
                if not supported:
                    result.flags.append(f'hallucinated_detail_risk: {label}')
                    result.hallucinated_detail_risk = True

        # 2. Check for retracted fact references
        for prop in retracted_propositions:
            if len(prop) >= 4 and prop in reply_text.lower():
                result.flags.append(f'retracted_fact_referenced: {prop[:60]}')
                result.retracted_fact_referenced = True

        # 3. Check for hypothesis-as-fact language
        for pattern in _HYPOTHESIS_AS_FACT_PATTERNS:
            if re.search(pattern, reply_text):
                result.flags.append(f'hypothesis_as_fact: {pattern}')
                result.hypothesis_as_fact = True

        # 4. Check for forbidden-topic references
        for topic in _FORBIDDEN_TOPICS:
            if topic.lower() in reply_text.lower():
                result.flags.append(f'forbidden_topic: {topic}')
                result.forbidden_topic_referenced = True

        return result

    @staticmethod
    def audit_structured(
        reply_text: str,
        anchored_items: list[AnchoredMemoryItem] | None = None,
    ) -> MemoryCitationAudit:
        """M8.5: structured audit with risk_level and per-violation actions."""
        items = list(anchored_items or [])
        violations: list[MemoryCitationViolation] = []

        explicit_propositions: dict[str, AnchoredMemoryItem] = {}
        retracted_propositions: set[str] = set()
        forbidden_propositions: set[str] = set()
        hypothesis_propositions: dict[str, AnchoredMemoryItem] = {}

        for item in items:
            prop_lower = item.proposition.lower().strip()
            if item.status == 'retracted':
                retracted_propositions.add(prop_lower)
            elif item.visibility == 'forbidden':
                forbidden_propositions.add(prop_lower)
            elif item.status in ('asserted', 'corroborated') and item.visibility == 'explicit':
                explicit_propositions[prop_lower] = item
            elif item.status == 'hypothesis' or item.memory_type in ('hypothesis', 'system_inferred_hypothesis'):
                hypothesis_propositions[prop_lower] = item

        # 1. Retracted fact references → blocking
        for prop in retracted_propositions:
            if len(prop) >= 4 and prop in reply_text.lower():
                violations.append(MemoryCitationViolation(
                    type='retracted_fact_referenced',
                    span=prop[:80],
                    message=f'Reply references retracted fact: {prop[:80]}',
                    suggested_action='block',
                ))

        # 2. Forbidden topic/proposition references → blocking
        for prop in forbidden_propositions:
            if len(prop) >= 4 and prop in reply_text.lower():
                violations.append(MemoryCitationViolation(
                    type='forbidden_topic_referenced',
                    span=prop[:80],
                    message=f'Reply references forbidden fact: {prop[:80]}',
                    suggested_action='block',
                ))
        for topic in _FORBIDDEN_TOPICS:
            if topic.lower() in reply_text.lower():
                violations.append(MemoryCitationViolation(
                    type='forbidden_topic_referenced',
                    span=topic,
                    message=f'Reply references forbidden topic: {topic}',
                    suggested_action='block',
                ))

        # 3. Hypothesis-as-fact language → retry (medium/high)
        for pattern in _HYPOTHESIS_AS_FACT_PATTERNS:
            m = re.search(pattern, reply_text)
            if m:
                violations.append(MemoryCitationViolation(
                    type='hypothesis_as_fact',
                    span=m.group(0),
                    message=f'Certainty language without anchored support: {pattern}',
                    suggested_action='retry',
                ))

        # 4. Unanchored memory claims → retry (medium)
        for pattern, label in _MEMORY_CLAIM_PATTERNS:
            if re.search(pattern, reply_text):
                supported = any(
                    prop in reply_text[:min(len(prop) + 20, len(reply_text))]
                    for prop in list(explicit_propositions.keys()) if len(prop) >= 4
                )
                if not supported:
                    violations.append(MemoryCitationViolation(
                        type='unanchored_memory_claim',
                        span=label,
                        message=f'Memory-claim language without anchored support: {label}',
                        suggested_action='retry',
                    ))

        # 5. Hallucinated detail risk → log (low/medium)
        if not any(v.type == 'unanchored_memory_claim' for v in violations):
            for pattern, label in _MEMORY_CLAIM_PATTERNS:
                if re.search(pattern, reply_text):
                    violations.append(MemoryCitationViolation(
                        type='hallucinated_detail_risk',
                        span=label,
                        message=f'Potential hallucination risk: {label}',
                        suggested_action='log_only',
                    ))

        # Compute aggregate risk level
        has_blocking = any(
            v.suggested_action in ('block', 'retry') for v in violations
        )
        has_violation = len(violations) > 0

        if any(v.suggested_action == 'block' for v in violations):
            risk_level = 'blocking'
        elif any(v.suggested_action == 'retry' for v in violations):
            risk_level = 'high'
        elif has_violation:
            risk_level = 'medium'
        else:
            risk_level = 'none'

        return MemoryCitationAudit(
            has_violation=has_violation,
            has_blocking_violation=has_blocking,
            risk_level=risk_level,
            violations=violations,
        )


# ── M8.5 Conservative Retry Repair Instruction ────────────────────────────

def build_memory_repair_instruction(audit: MemoryCitationAudit) -> str:
    """Build a repair instruction for conservative retry based on audit violations.

    The instruction tells the model its previous reply violated the memory
    contract and gives specific constraints for the rewrite.
    """
    violation_types = {v.type for v in audit.violations}
    lines = [
        "上一版回复违反了记忆引用契约。请重写回复：",
    ]

    if 'retracted_fact_referenced' in violation_types:
        lines.append("1. 不要引用任何已撤回或 forbidden 的用户事实。")
    else:
        lines.append("1. 只引用当前对话中明确确认的信息。")

    if 'forbidden_topic_referenced' in violation_types:
        lines.append("2. 不要提及被标记为禁止的话题或命题。")

    if 'hypothesis_as_fact' in violation_types:
        lines.append(
            "3. 不要把假设当作确定事实。对不确定内容使用"
            '"可能"、"似乎"、"我不确定"、"需要你确认"等措辞。'
        )
    else:
        lines.append("3. 不要把假设当作确定事实。")

    if 'unanchored_memory_claim' in violation_types:
        lines.append(
            '4. 如果没有 anchored memory 支持，不要说'
            '"你之前说过"、"我记得你说过"、"你告诉过我"这类表达。'
        )

    if 'hallucinated_detail_risk' in violation_types:
        lines.append("5. 不要编造用户没有明确说过的具体细节。")

    lines.append("6. 回复应更短、更保守，只基于当前对话和明确锚点。")
    lines.append("7. 对不确定内容使用'可能/似乎/需要确认'等措辞。")

    return "\n".join(lines)
