"""M14.0 named owners for idle introspection patches (Path B)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m14_idle_reflector import (
    M14_ENGINEERING_PROXY_LABEL,
    MIN_PATCH_CONFIDENCE,
    subjective_language_violations,
)
from segmentum.dialogue.runtime.m13_drive import _bounded_float, _mapping, _new_id, _string_list
from segmentum.dialogue.runtime.m14_7_memory_gate import (
    MemoryGate,
    MemoryWriteIntent,
    memory_gate_event,
)

MAX_SELF_COGNITION_PATCHES_PER_SESSION = 2
MAX_OPEN_ITEM_PATCHES_PER_SESSION = 5
MAX_MEMORY_CONSOLIDATIONS_PER_SESSION = 5

_FORBIDDEN_LEDGER_PREFIXES = ("m11_", "m12_", "m12.")

_VIOLATION_CODES = frozenset(
    {
        "missing_evidence_refs",
        "evidence_not_retrieved",
        "confidence_below_threshold",
        "session_cap_reached",
        "forbidden_ledger_target",
        "subjective_language",
        "empty_summary_delta",
        "open_item_not_found",
        "invalid_op",
        "empty_memory_content",
        "m11_m12_cross_write",
    }
)


def _audit_base(*, turn_index: int, at: int) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "at": at,
        "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
    }


def _refs_intersect_retrieved(refs: list[str], retrieved_ids: set[str]) -> bool:
    if not refs:
        return False
    return bool(set(refs) & retrieved_ids)


def _forbidden_target(payload: Mapping[str, Any]) -> bool:
    blob = " ".join(str(payload.get(key, "")) for key in payload).casefold()
    return any(marker in blob for marker in _FORBIDDEN_LEDGER_PREFIXES)


@dataclass
class OwnerCommitResult:
    committed: bool
    events: list[dict[str, Any]] = field(default_factory=list)
    violation_codes: list[str] = field(default_factory=list)


class SelfCognitionPatchOwner:
    @staticmethod
    def validate_and_commit(
      state: dict[str, Any],
      proposal: Mapping[str, Any],
      *,
      retrieved_ids: set[str],
      turn_index: int,
      now: int,
      session_patches: int,
  ) -> OwnerCommitResult:
      events: list[dict[str, Any]] = []
      base = _audit_base(turn_index=turn_index, at=now)
      refs = _string_list(proposal.get("evidence_refs"), limit=8)
      events.append(
          {
              "type": "SelfCognitionPatchProposalEvent",
              **base,
              "evidence_refs": refs,
              "apply": bool(proposal.get("apply")),
              "confidence": _bounded_float(proposal.get("confidence")),
          }
      )
      violations: list[str] = []
      if not bool(proposal.get("apply")):
          return OwnerCommitResult(committed=False, events=events)

      if session_patches >= MAX_SELF_COGNITION_PATCHES_PER_SESSION:
          violations.append("session_cap_reached")
      if not refs:
          violations.append("missing_evidence_refs")
      elif not _refs_intersect_retrieved(refs, retrieved_ids):
          violations.append("evidence_not_retrieved")
      if _bounded_float(proposal.get("confidence")) < MIN_PATCH_CONFIDENCE:
          violations.append("confidence_below_threshold")
      delta = str(proposal.get("summary_delta", "") or "").strip()
      if not delta:
          violations.append("empty_summary_delta")
      if subjective_language_violations(delta):
          violations.append("subjective_language")
      if _forbidden_target(proposal):
          violations.append("m11_m12_cross_write")

      if violations:
          events.append(
              {
                  "type": "SelfCognitionPatchRejectedEvent",
                  **base,
                  "evidence_refs": refs,
                  "violation_codes": violations,
                  "reason": violations[0],
              }
          )
          return OwnerCommitResult(committed=False, events=events, violation_codes=violations)

      cognition = state.setdefault("self_cognition", {})
      if not isinstance(cognition, dict):
          cognition = {}
          state["self_cognition"] = cognition
      cognition.setdefault("patch_history", [])
      if not isinstance(cognition.get("patch_history"), list):
          cognition["patch_history"] = []

      old = str(cognition.get("current_self_view", cognition.get("summary", "")) or "").strip()
      cognition["current_self_view"] = (old + "\n" + delta).strip() if old else delta
      cognition.setdefault("identity_tensions", [])
      cognition.setdefault("known_limits", [])
      if isinstance(cognition["identity_tensions"], list):
          cognition["identity_tensions"].extend(_string_list(proposal.get("new_identity_tensions"), limit=6))
      if isinstance(cognition["known_limits"], list):
          cognition["known_limits"].extend(_string_list(proposal.get("new_known_limits"), limit=6))

      history_row = {
          "patch_id": _new_id("sc_patch"),
          "at": now,
          "turn_index": turn_index,
          "source": "m14_idle_introspection",
          "summary_delta": delta[:400],
          "evidence_refs": refs,
          "confidence": round(_bounded_float(proposal.get("confidence")), 4),
          "reason": str(proposal.get("reason", "") or "")[:240],
      }
      history = cognition["patch_history"]
      history.append(history_row)
      cognition["patch_history"] = history[-20:]

      events.append(
          {
              "type": "SelfCognitionPatchCommitEvent",
              **base,
              "evidence_refs": refs,
              "patch_id": history_row["patch_id"],
              "reason": history_row["reason"],
          }
      )
      return OwnerCommitResult(committed=True, events=events)


class OpenItemPatchOwner:
    @staticmethod
    def validate_and_commit(
      state: dict[str, Any],
      proposals: list[Mapping[str, Any]],
      *,
      retrieved_ids: set[str],
      turn_index: int,
      now: int,
      session_patches: int,
  ) -> OwnerCommitResult:
      events: list[dict[str, Any]] = []
      committed_any = False
      violations_all: list[str] = []
      base = _audit_base(turn_index=turn_index, at=now)
      items = state.get("open_items", [])
      if not isinstance(items, list):
          items = []
          state["open_items"] = items

      for proposal in proposals[:MAX_OPEN_ITEM_PATCHES_PER_SESSION]:
          if session_patches + (1 if committed_any else 0) >= MAX_OPEN_ITEM_PATCHES_PER_SESSION:
              violations_all.append("session_cap_reached")
              break
          item_id = str(proposal.get("id", "") or "")
          op = str(proposal.get("op", "update") or "update")
          events.append(
              {
                  "type": "OpenItemPatchProposalEvent",
                  **base,
                  "open_item_id": item_id,
                  "op": op,
                  "rationale": str(proposal.get("rationale", "") or "")[:240],
              }
          )
          violations: list[str] = []
          if op not in {"update", "close", "defer"}:
              violations.append("invalid_op")
          target = next((row for row in items if isinstance(row, dict) and str(row.get("id")) == item_id), None)
          if target is None:
              violations.append("open_item_not_found")
          if subjective_language_violations(str(proposal.get("rationale", ""))):
              violations.append("subjective_language")

          if violations:
              violations_all.extend(violations)
              events.append(
                  {
                      "type": "OpenItemPatchRejectedEvent",
                      **base,
                      "open_item_id": item_id,
                      "violation_codes": violations,
                      "reason": violations[0],
                  }
              )
              continue

          if not isinstance(target, dict):
              continue
          if op == "close":
              target["status"] = "closed"
          elif op == "defer":
              target["status"] = "deferred"
              target["next_check"] = "next_user_turn"
          else:
              rationale = str(proposal.get("rationale", "") or "").strip()
              if rationale:
                  target["next_check"] = rationale[:140]
          target["last_idle_patch_at"] = now
          committed_any = True
          events.append(
              {
                  "type": "OpenItemPatchCommitEvent",
                  **base,
                  "open_item_id": item_id,
                  "op": op,
                  "evidence_refs": [item_id] if item_id in retrieved_ids else [],
              }
          )

      return OwnerCommitResult(
          committed=committed_any,
          events=events,
          violation_codes=violations_all,
      )


@dataclass(frozen=True)
class MvpMemoryWriteIntent:
    """MVP-local write intent (Path B JSON state, not anchored store)."""

    intent_id: str
    target: str
    kind: str
    content: str
    confidence: float
    evidence_refs: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "target": self.target,
            "kind": self.kind,
            "content": self.content,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
            "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
        }


class MemoryConsolidationOwner:
    @staticmethod
    def translate_to_intents(
      proposals: list[Mapping[str, Any]],
      *,
      retrieved_ids: set[str],
  ) -> tuple[list[MvpMemoryWriteIntent], list[str]]:
      intents: list[MvpMemoryWriteIntent] = []
      violations: list[str] = []
      for row in proposals:
          refs = _string_list(row.get("evidence_refs"), limit=8)
          content = str(row.get("content", "") or "").strip()
          if not content:
              violations.append("empty_memory_content")
              continue
          if not refs or not _refs_intersect_retrieved(refs, retrieved_ids):
              violations.append("evidence_not_retrieved")
              continue
          if _bounded_float(row.get("confidence")) < MIN_PATCH_CONFIDENCE:
              violations.append("confidence_below_threshold")
              continue
          if _forbidden_target(row):
              violations.append("m11_m12_cross_write")
              continue
          intents.append(
              MvpMemoryWriteIntent(
                  intent_id=_new_id("mem_intent"),
                  target=str(row.get("target", "short_term")),
                  kind=str(row.get("kind", "episode")),
                  content=content[:400],
                  confidence=_bounded_float(row.get("confidence")),
                  evidence_refs=tuple(refs),
                  reason="idle_memory_consolidation",
              )
          )
          if len(intents) >= MAX_MEMORY_CONSOLIDATIONS_PER_SESSION:
              break
      return intents, violations

    @staticmethod
    def apply_intents(
      state: dict[str, Any],
      intents: list[MvpMemoryWriteIntent],
      *,
      turn_index: int,
      now: int,
      session_count: int,
  ) -> OwnerCommitResult:
      events: list[dict[str, Any]] = []
      base = _audit_base(turn_index=turn_index, at=now)
      committed = False
      applied = 0
      for intent in intents:
          if session_count + applied >= MAX_MEMORY_CONSOLIDATIONS_PER_SESSION:
              events.append(
                  {
                      "type": "MemoryConsolidationIntentEvent",
                      **base,
                      "intent_id": intent.intent_id,
                      "committed": False,
                      "violation_codes": ["session_cap_reached"],
                  }
              )
              break
          gate_intent = MemoryWriteIntent(
              target=intent.target if intent.target in {"short_term", "long_term"} else "short_term",
              kind=intent.kind,
              content=intent.content,
              confidence=intent.confidence,
              evidence_refs=list(intent.evidence_refs),
              identity_relevance=0.0,
              value_proxy=max(0.45, intent.confidence),
              surprise_proxy=max(0.35, intent.confidence * 0.75),
              source="idle_consolidation",
              proposer="MemoryConsolidationOwner",
              audit_reason=intent.reason,
              intent_id=intent.intent_id,
          )
          decision = MemoryGate().evaluate(gate_intent, proposer_commits_this_session=session_count + applied)
          if not decision.commit:
              events.append(
                  memory_gate_event(
                      event_type="MemoryGateRejectedEvent",
                      intent=gate_intent,
                      decision=decision,
                      turn_index=turn_index,
                      now=now,
                  )
              )
              events.append(
                  {
                      "type": "MemoryConsolidationIntentEvent",
                      **base,
                      "intent_id": intent.intent_id,
                      "committed": False,
                      "violation_codes": decision.violation_codes,
                  }
              )
              continue
          row = {
              "id": _new_id("stm"),
              "kind": intent.kind,
              "content": intent.content,
              "salience": round(min(1.0, 0.45 + intent.confidence * 0.35), 4),
              "keywords": list(intent.evidence_refs)[:6],
              "source": "m14_idle_introspection",
              "created_at": now,
              "turn_index": turn_index,
              "evidence_refs": list(intent.evidence_refs),
              "memory_gate_decision": decision.to_dict(),
          }
          if intent.target == "long_term":
              state.setdefault("long_term_memory", []).append(row)
          else:
              state.setdefault("short_term_memory", []).append(row)
          applied += 1
          committed = True
          events.append(
              memory_gate_event(
                  event_type="MemoryGateCommitEvent",
                  intent=gate_intent,
                  decision=decision,
                  turn_index=turn_index,
                  now=now,
                  store_target=intent.target,
                  store_id=str(row["id"]),
              )
          )
          events.append(
              {
                  "type": "MemoryConsolidationIntentEvent",
                  **base,
                  "intent_id": intent.intent_id,
                  "committed": True,
                  "target": intent.target,
                  "evidence_refs": list(intent.evidence_refs),
                  "intent": intent.to_dict(),
              }
          )
      return OwnerCommitResult(committed=committed, events=events)


def count_session_idle_patches(state: Mapping[str, Any]) -> dict[str, int]:
    cognition = _mapping(state.get("self_cognition"))
    history = cognition.get("patch_history", [])
    self_count = len(history) if isinstance(history, list) else 0
    open_count = 0
    for item in state.get("open_items", []) or []:
        if isinstance(item, Mapping) and item.get("last_idle_patch_at"):
            open_count += 1
    mem_count = 0
    for key in ("short_term_memory", "long_term_memory"):
        for row in state.get(key, []) or []:
            if isinstance(row, Mapping) and str(row.get("source", "")) == "m14_idle_introspection":
                mem_count += 1
    return {
        "self_cognition": self_count,
        "open_items": open_count,
        "memory": mem_count,
    }
