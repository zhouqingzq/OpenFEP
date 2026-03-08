from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


class SnapshotLoadError(RuntimeError):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class SnapshotCorruptedError(SnapshotLoadError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason="corrupt_snapshot")


class SnapshotVersionError(SnapshotLoadError):
    def __init__(self, version: object, supported_versions: set[str]) -> None:
        supported = ", ".join(sorted(supported_versions))
        super().__init__(
            f"unsupported state_version={version!r}; supported versions: {supported}",
            reason="unsupported_state_version",
        )


def atomic_write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path_str = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f"{target.stem}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def load_snapshot(
    path: str | Path,
    *,
    supported_versions: set[str],
) -> dict[str, object]:
    snapshot_path = Path(path)
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SnapshotCorruptedError(f"snapshot JSON is invalid: {exc}") from exc
    except OSError as exc:
        raise SnapshotCorruptedError(f"snapshot could not be read: {exc}") from exc

    if not isinstance(payload, dict):
        raise SnapshotCorruptedError("snapshot root must be a JSON object")

    version = payload.get("state_version")
    if version not in supported_versions:
        raise SnapshotVersionError(version, supported_versions)

    return payload


def quarantine_snapshot(path: str | Path, *, reason: str) -> Path | None:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return None

    safe_reason = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in reason
    ).strip("_") or "snapshot_error"
    quarantined = snapshot_path.with_name(
        f"{snapshot_path.stem}.{safe_reason}.{time.time_ns()}{snapshot_path.suffix}"
    )
    snapshot_path.replace(quarantined)
    return quarantined
