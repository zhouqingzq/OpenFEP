from __future__ import annotations

import json
from pathlib import Path

from segmentum.chat_pipeline.exporter import export_user_dataset
from segmentum.chat_pipeline.parser import parse_file, parse_line
from segmentum.chat_pipeline.quality_filter import QualityFilter
from segmentum.chat_pipeline.session_builder import build_sessions
from segmentum.chat_pipeline.user_aggregator import aggregate_users
from scripts.run_m50_pipeline import run_pipeline


def _line(ts: str, sender: int, receiver: int, body: str, msg_type: int = 0) -> str:
    return (
        f"{ts} INFO   MessageSender::OnData message type: {msg_type}, "
        f"sender uid: {sender}, reciever uid: {receiver}, body: {body}"
    )


def test_parser_parse_line_and_invalid() -> None:
    ok = _line("2023-10-22-12:42:01", 1, 2, "a:b,c")
    msg = parse_line(ok)
    assert msg is not None
    assert msg.sender_uid == 1
    assert msg.receiver_uid == 2
    assert msg.body == "a:b,c"
    assert parse_line("bad log line") is None


def test_parser_parse_file_streaming(tmp_path: Path) -> None:
    path = tmp_path / "sample.log"
    path.write_text(
        "\n".join(
            [
                _line("2023/10/22 12:42:01", 1, 2, "hello"),
                "MALFORMED",
                _line("2023-10-22-12:42:02", 2, 1, "world"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(parse_file(path))
    assert len(rows) == 2
    assert rows[0].timestamp < rows[1].timestamp


def test_session_builder_gap_and_uid_normalization() -> None:
    messages = [
        parse_line(_line("2023-10-22-12:00:00", 9, 3, "a")),
        parse_line(_line("2023-10-22-12:10:00", 3, 9, "b")),
        parse_line(_line("2023-10-22-13:00:00", 9, 3, "c")),
    ]
    parsed = [m for m in messages if m is not None]
    sessions = build_sessions(parsed, gap_threshold_minutes=30)
    assert (3, 9) in sessions
    assert len(sessions[(3, 9)]) == 2
    assert sessions[(3, 9)][0].metadata["turn_count"] == 2


def test_quality_filter_spam_pii_and_ultra_short() -> None:
    msg = parse_line(
        _line(
            "2023-10-22-12:00:00",
            1,
            2,
            "嗯 联系我 13800138000 test@example.com https://a.com https://b.com",
        )
    )
    assert msg is not None
    result = QualityFilter(normalize_chinese=None).filter_message(msg)
    assert not result.kept
    assert "spam" in result.tags
    assert "pii_redacted" in result.tags
    assert "[PHONE]" in result.redacted_body
    assert "[EMAIL]" in result.redacted_body
    assert "[URL]" in result.redacted_body


def test_user_aggregator_threshold_and_stats() -> None:
    messages = [
        parse_line(_line("2023-10-22-12:00:00", 1, 2, "hello world")),
        parse_line(_line("2023-10-22-12:01:00", 2, 1, "ok")),
        parse_line(_line("2023-10-22-12:02:00", 1, 3, "ni hao")),
        parse_line(_line("2023-10-22-12:03:00", 3, 1, "ha")),
    ]
    sessions = build_sessions([m for m in messages if m is not None], gap_threshold_minutes=30)
    filtered = {
        pair: [QualityFilter(normalize_chinese=None).filter_session(s) for s in ss]
        for pair, ss in sessions.items()
    }
    profiles = aggregate_users(filtered, min_messages=2, min_partners=2)
    assert profiles[1].qualifies
    assert profiles[1].unique_partners == 2
    assert profiles[1].total_messages == 2
    assert len(profiles[1].temporal_pattern) == 24


def test_export_and_end_to_end_pipeline(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    (input_dir / "a.log").write_text(
        "\n".join(
            [
                _line("2023-10-22-12:00:00", 1, 2, "你好"),
                _line("2023-10-22-12:01:00", 2, 1, "哈哈"),
                _line("2023-10-22-12:01:30", 1, 2, "image payload", msg_type=2),
                _line("2023-10-22-12:02:00", 1, 3, "hello"),
                _line("2023-10-22-12:03:00", 3, 1, "world"),
                "BROKEN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_pipeline(
        input_path=input_dir,
        output_path=output_dir,
        min_messages=2,
        min_partners=2,
        normalize=None,
    )
    assert report["total_lines"] == 6
    assert report["parsed_failed"] == 1
    assert report["non_text_filtered"] == 1
    assert report["qualified_users"] == 1

    user_file = output_dir / "users" / "1.json"
    payload = json.loads(user_file.read_text(encoding="utf-8"))
    assert payload["uid"] == 1
    assert len(payload["sessions"]) == 2
    pipeline_report = json.loads((output_dir / "pipeline_report.json").read_text(encoding="utf-8"))
    assert pipeline_report["qualified_users"] == 1
