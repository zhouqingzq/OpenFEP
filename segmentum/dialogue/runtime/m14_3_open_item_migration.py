"""M14.3 deterministic open-item traceability audit and patch helpers."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_drive import _string_list
from segmentum.dialogue.runtime.m14_3_proactive_alignment import next_check_is_vague


@dataclass(frozen=True)
class OpenItemMigrationSuggestion:
    item_id: str
    action: str
    reason_code: str
    patch: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "patch": dict(self.patch),
        }


def _item_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", "") or "").strip()


def _created_at(row: Mapping[str, Any]) -> int:
    for key in ("created_at", "created_at_epoch", "opened_at", "at"):
        try:
            value = int(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _has_traceable_evidence(row: Mapping[str, Any]) -> bool:
    from segmentum.dialogue.runtime.m15_3_cleanup_control import is_strictly_traceable

    return is_strictly_traceable(row)


def audit_open_items_for_efe(open_items: Any) -> list[OpenItemMigrationSuggestion]:
    suggestions: list[OpenItemMigrationSuggestion] = []
    for row in open_items or []:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "open") or "open").strip().lower()
        if status != "open":
            continue
        item_id = _item_id(row)
        next_check = str(row.get("next_check", row.get("next_step", "")) or "").strip()
        scheduled_id = str(row.get("scheduled_intent_id", row.get("intent_id", "")) or "").strip()
        due_at = row.get("due_at_epoch") or row.get("due_at")
        if due_at and not scheduled_id:
            suggestions.append(
                OpenItemMigrationSuggestion(
                    item_id=item_id,
                    action="diagnostic_only",
                    reason_code="wall_clock_open_item_requires_scheduled_intent",
                    patch={"diagnostic_flags": ["wall_clock_open_item_requires_scheduled_intent"]},
                )
            )
            continue
        if not next_check_is_vague(next_check):
            continue
        if _has_traceable_evidence(row) and _created_at(row):
            suggestions.append(
                OpenItemMigrationSuggestion(
                    item_id=item_id,
                    action="patch_next_check",
                    reason_code="traceable_vague_open_item_can_use_next_user_turn",
                    patch={
                        "id": item_id,
                        "next_check": "next_user_turn",
                        "migration_reason": "m14_3_traceable_open_item_upgrade",
                    },
                )
            )
        else:
            suggestions.append(
                OpenItemMigrationSuggestion(
                    item_id=item_id,
                    action="diagnostic_only",
                    reason_code="vague_open_item_missing_evidence_or_created_at",
                    patch={"diagnostic_flags": ["vague_open_item_missing_evidence_or_created_at"]},
                )
            )
    return suggestions


def propose_open_item_traceability_patches(open_items: Any, *, now: int) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for suggestion in audit_open_items_for_efe(open_items):
        if suggestion.action != "patch_next_check":
            continue
        patch = {
            **suggestion.patch,
            "patched_at": int(now),
            "source": "m14_3_open_item_migration",
            "reason_code": suggestion.reason_code,
        }
        patches.append(patch)
    return patches


def apply_open_item_traceability_patches(
    state: dict[str, Any],
    patches: list[Mapping[str, Any]],
    *,
    source: str,
    reason: str,
) -> int:
    by_id = {str(patch.get("id", "") or ""): patch for patch in patches if str(patch.get("id", "") or "")}
    if not by_id:
        return 0
    applied = 0
    history = state.setdefault("open_item_patch_history", [])
    if not isinstance(history, list):
        history = []
        state["open_item_patch_history"] = history
    for row in state.get("open_items", []) or []:
        if not isinstance(row, dict):
            continue
        item_id = _item_id(row)
        patch = by_id.get(item_id)
        if not patch:
            continue
        before = {key: row.get(key) for key in ("next_check", "diagnostic_flags")}
        if patch.get("next_check"):
            row["next_check"] = str(patch.get("next_check"))
        applied += 1
        history.append(
            {
                "type": "OpenItemTraceabilityPatchEvent",
                "item_id": item_id,
                "source": source,
                "reason": reason,
                "reason_code": str(patch.get("reason_code", "")),
                "before": before,
                "after": {key: row.get(key) for key in ("next_check", "diagnostic_flags")},
                "engineering_proxy_label": "mvp_local_proactive_alignment",
            }
        )
    return applied


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit M14.3 open item traceability")
    parser.add_argument("--session-root", required=True, help="MVP session directory (MVPStateStore root)")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    from segmentum.dialogue.runtime.mvp_loop import MVPStateStore

    store = MVPStateStore(Path(args.session_root))
    state = store.load()
    suggestions = audit_open_items_for_efe(state.get("open_items", []))
    if args.apply:
        patches = propose_open_item_traceability_patches(state.get("open_items", []), now=int(time.time()))
        applied = apply_open_item_traceability_patches(
            state,
            patches,
            source="cli",
            reason="m14_3_traceability_migration",
        )
        store.save(state)
        print(json.dumps({"applied": applied, "suggestions": [s.to_dict() for s in suggestions]}, ensure_ascii=False))
    else:
        print(json.dumps({"suggestions": [s.to_dict() for s in suggestions]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())

