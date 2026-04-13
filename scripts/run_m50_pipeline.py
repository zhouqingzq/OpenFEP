from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import sys

from segmentum.chat_pipeline.exporter import export_user_dataset
from segmentum.chat_pipeline.parser import parse_line
from segmentum.chat_pipeline.quality_filter import QualityFilter
from segmentum.chat_pipeline.session_builder import ConversationSession, build_sessions
from segmentum.chat_pipeline.user_aggregator import UserProfile, aggregate_users


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [p for p in input_path.rglob("*") if p.is_file()]
    return sorted(files)


def _parse_messages(files: list[Path], *, progress_every: int = 100_000) -> tuple[list, dict[str, int]]:
    messages = []
    total_lines = 0
    parse_failed = 0
    non_text_filtered = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                total_lines += 1
                parsed = parse_line(line)
                if parsed is None:
                    parse_failed += 1
                elif parsed.msg_type != 0:
                    non_text_filtered += 1
                else:
                    messages.append(parsed)
                if progress_every > 0 and total_lines % progress_every == 0:
                    print(
                        (
                            f"[m5.0] progress lines={total_lines} "
                            f"parsed_success={len(messages)} "
                            f"parsed_failed={parse_failed} "
                            f"non_text_filtered={non_text_filtered}"
                        ),
                        file=sys.stderr,
                    )
    stats = {
        "total_lines": total_lines,
        "parsed_success": len(messages),
        "parsed_failed": parse_failed,
        "non_text_filtered": non_text_filtered,
    }
    return messages, stats


def _filter_sessions(
    sessions: dict[tuple[int, int], list[ConversationSession]],
    quality_filter: QualityFilter,
) -> tuple[dict[tuple[int, int], list[ConversationSession]], Counter[str]]:
    filtered: dict[tuple[int, int], list[ConversationSession]] = defaultdict(list)
    tag_counts: Counter[str] = Counter()
    for pair in sorted(sessions):
        for session in sessions[pair]:
            filtered_session = quality_filter.filter_session(session)
            tag_counts.update(filtered_session.metadata.get("filter_tag_counts", {}))
            if filtered_session.metadata.get("dropped"):
                continue
            filtered[pair].append(filtered_session)
    return dict(filtered), tag_counts


def _sessions_for_uid(
    uid: int,
    sessions: dict[tuple[int, int], list[ConversationSession]],
) -> list[ConversationSession]:
    collected: list[ConversationSession] = []
    for (uid_a, uid_b), pair_sessions in sessions.items():
        if uid in (uid_a, uid_b):
            collected.extend(pair_sessions)
    return sorted(collected, key=lambda s: (s.start_time, s.session_id))


def run_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    min_messages: int,
    min_partners: int,
    normalize: str | None,
    progress_every: int = 100_000,
) -> dict[str, object]:
    t0 = time.perf_counter()
    files = _iter_input_files(input_path)
    messages, parse_stats = _parse_messages(files, progress_every=progress_every)

    sessions = build_sessions(messages)
    total_sessions = sum(len(v) for v in sessions.values())

    quality_filter = QualityFilter(normalize_chinese=normalize)
    filtered_sessions, tag_counts = _filter_sessions(sessions, quality_filter)
    total_filtered_sessions = sum(len(v) for v in filtered_sessions.values())

    profiles = aggregate_users(
        filtered_sessions,
        min_messages=min_messages,
        min_partners=min_partners,
    )
    total_users = len(profiles)
    qualified_profiles: dict[int, UserProfile] = {uid: p for uid, p in profiles.items() if p.qualifies}
    qualified_users = len(qualified_profiles)

    users_dir = output_path / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    for uid in sorted(qualified_profiles):
        export_user_dataset(uid, qualified_profiles[uid], _sessions_for_uid(uid, filtered_sessions), users_dir)

    elapsed = round(time.perf_counter() - t0, 3)
    parse_success_rate = round(
        (parse_stats["parsed_success"] / parse_stats["total_lines"]) if parse_stats["total_lines"] else 0.0,
        6,
    )
    report: dict[str, object] = {
        **parse_stats,
        "parse_success_rate": parse_success_rate,
        "total_messages": len(messages),
        "total_sessions": total_sessions,
        "filtered_sessions": total_filtered_sessions,
        "total_users": total_users,
        "qualified_users": qualified_users,
        "filtered_users": total_users - qualified_users,
        "filter_tag_counts": dict(sorted(tag_counts.items())),
        "input_files": [str(p) for p in files],
        "processing_seconds": elapsed,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5.0 chat data pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Raw log file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--min-messages", type=int, default=200)
    parser.add_argument("--min-partners", type=int, default=3)
    parser.add_argument(
        "--normalize",
        type=str,
        default="simplified",
        choices=["simplified", "traditional", "none"],
        help="Chinese normalization target",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N lines (0 to disable)",
    )
    args = parser.parse_args()
    normalize = None if args.normalize == "none" else args.normalize
    report = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        min_messages=args.min_messages,
        min_partners=args.min_partners,
        normalize=normalize,
        progress_every=args.progress_every,
    )
    acceptance = {
        "milestone": "M5.0",
        "status": "completed",
        "report_path": str((args.output / "pipeline_report.json").as_posix()),
        "summary": {
            "parse_success_rate": report["parse_success_rate"],
            "qualified_users": report["qualified_users"],
            "processing_seconds": report["processing_seconds"],
        },
    }
    artifacts_path = Path("artifacts/m50_pipeline_acceptance.json")
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
