from __future__ import annotations

import json
from pathlib import Path


def derive_trace_path(state_path: str | Path | None) -> Path | None:
    if not state_path:
        return None
    base = Path(state_path)
    return base.with_name(f"{base.stem}_trace.jsonl")


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        self.path.unlink(missing_ok=True)

    def append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
