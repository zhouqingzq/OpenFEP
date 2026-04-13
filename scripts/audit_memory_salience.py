"""P3-8: CLI audit for memory salience decomposition (build_salience_audit / format_salience_audit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from segmentum.memory_encoding import SalienceConfig, build_salience_audit, format_salience_audit
from segmentum.memory_model import MemoryEntry


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _salience_config_from_dict(payload: dict[str, object]) -> SalienceConfig:
    base = SalienceConfig()
    rw = payload.get("relevance_weights")
    relevance_weights = dict(rw) if isinstance(rw, dict) else base.relevance_weights
    coerced_rw = {str(k): float(v) for k, v in relevance_weights.items()}
    return SalienceConfig(
        w_arousal=float(payload.get("w_arousal", base.w_arousal)),
        w_attention=float(payload.get("w_attention", base.w_attention)),
        w_novelty=float(payload.get("w_novelty", base.w_novelty)),
        w_relevance=float(payload.get("w_relevance", base.w_relevance)),
        relevance_weights=coerced_rw,
    )


def _iter_entry_dicts(root: object) -> list[dict[str, object]]:
    if isinstance(root, dict) and "arousal" in root and "encoding_attention" in root:
        return [root]
    if isinstance(root, list):
        out: list[dict[str, object]] = []
        for item in root:
            if isinstance(item, dict):
                out.extend(_iter_entry_dicts(item))
        return out
    if isinstance(root, dict):
        for key in ("entries", "memories", "memory_entries", "items"):
            child = root.get(key)
            if isinstance(child, list):
                found = _iter_entry_dicts(child)
                if found:
                    return found
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print salience audit for MemoryEntry JSON (P3-8 tooling)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to JSON (MemoryEntry object, or document containing entries[]). Use - for stdin.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with SalienceConfig fields (w_arousal, w_attention, ...).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max entries when input expands to multiple MemoryEntry-shaped objects.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only format_salience_audit lines (no JSON audit blob).",
    )
    args = parser.parse_args()

    if args.input == "-":
        raw = json.load(sys.stdin)
    else:
        raw = _load_json(Path(args.input))

    config: SalienceConfig | None = None
    if args.config:
        cfg_payload = _load_json(Path(args.config))
        if not isinstance(cfg_payload, dict):
            raise SystemExit("--config must be a JSON object")
        config = _salience_config_from_dict(cfg_payload)

    blobs = _iter_entry_dicts(raw)
    if not blobs:
        raise SystemExit("No MemoryEntry-shaped objects found (need arousal + encoding_attention fields).")

    for i, blob in enumerate(blobs[: max(1, args.limit)]):
        entry = MemoryEntry.from_dict(blob)
        audit = build_salience_audit(entry, config)
        if not args.summary_only:
            print(json.dumps({"index": i, "id": entry.id, "audit": audit}, indent=2, sort_keys=True))
        print(f"[{i}] {entry.id}: {format_salience_audit(entry, config)}")


if __name__ == "__main__":
    main()
