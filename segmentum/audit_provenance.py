from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def codebase_version(root: Path) -> str:
    output = _run_git(root, "rev-parse", "HEAD")
    if output is None:
        return "unknown"
    return output.strip() or "unknown"


def collect_codebase_provenance(root: Path) -> dict[str, object]:
    commit = codebase_version(root)
    status_output = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status_output is None:
        return {
            "git_commit": commit,
            "dirty_worktree": None,
            "binding_scope": "unknown",
            "workspace_fingerprint": None,
            "status_entries": [],
            "changed_path_count": 0,
            "reproducibility_scope": "unknown",
        }

    status_entries = [line.rstrip() for line in status_output.splitlines() if line.strip()]
    dirty = bool(status_entries)
    fingerprint = sha256()
    fingerprint.update(f"commit={commit}\n".encode("utf-8"))
    fingerprint.update(f"dirty={dirty}\n".encode("utf-8"))

    changed_paths: list[str] = []
    for entry in sorted(status_entries):
        fingerprint.update(f"status={entry}\n".encode("utf-8"))
        path_text = entry[3:] if len(entry) > 3 else ""
        if " -> " in path_text:
            _, path_text = path_text.split(" -> ", 1)
        relative_path = path_text.strip()
        if not relative_path:
            continue
        changed_paths.append(relative_path)
        absolute_path = root / relative_path
        if absolute_path.is_file():
            fingerprint.update(f"path={relative_path}\n".encode("utf-8"))
            fingerprint.update(_sha256_bytes(absolute_path.read_bytes()).encode("utf-8"))
            fingerprint.update(b"\n")
        else:
            fingerprint.update(f"path={relative_path}\nmissing\n".encode("utf-8"))

    binding_scope = "workspace_snapshot" if dirty else "frozen_commit"
    reproducibility_scope = (
        "strict_workspace_fingerprint"
        if dirty
        else "frozen_commit"
    )
    return {
        "git_commit": commit,
        "dirty_worktree": dirty,
        "binding_scope": binding_scope,
        "workspace_fingerprint": fingerprint.hexdigest(),
        "status_entries": sorted(status_entries),
        "changed_path_count": len(changed_paths),
        "changed_paths": sorted(changed_paths),
        "reproducibility_scope": reproducibility_scope,
    }


def normalize_codebase_provenance(
    value: str | dict[str, object] | None,
    *,
    root: Path,
) -> dict[str, object]:
    if isinstance(value, dict):
        provenance = dict(value)
        provenance.setdefault("git_commit", str(provenance.get("codebase_version", "unknown")))
        provenance.setdefault("dirty_worktree", None)
        provenance.setdefault("binding_scope", "provided")
        provenance.setdefault("workspace_fingerprint", None)
        provenance.setdefault("status_entries", [])
        provenance.setdefault("changed_path_count", len(list(provenance.get("changed_paths", []))))
        provenance.setdefault("reproducibility_scope", "provided")
        return provenance
    if isinstance(value, str) and value:
        return {
            "git_commit": value,
            "dirty_worktree": None,
            "binding_scope": "provided",
            "workspace_fingerprint": None,
            "status_entries": [],
            "changed_path_count": 0,
            "reproducibility_scope": "provided",
        }
    return collect_codebase_provenance(root)
