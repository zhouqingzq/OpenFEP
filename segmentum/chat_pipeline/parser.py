from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Iterator

LOGGER = logging.getLogger(__name__)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[- ]\d{2}[:\-]\d{2}[:\-]\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<source>[^ ]+)\s+"
    r"message type:\s*(?P<msg_type>-?\d+),\s*"
    r"sender uid:\s*(?P<sender_uid>\d+),\s*"
    r"reciever uid:\s*(?P<receiver_uid>\d+),\s*"
    r"body:\s*(?P<body>.*)$",
    re.DOTALL,
)

TS_FORMATS = (
    "%Y-%m-%d-%H:%M:%S",
    "%Y/%m/%d-%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d-%H-%M-%S",
)


@dataclass(frozen=True)
class ChatMessage:
    timestamp: datetime
    msg_type: int
    sender_uid: int
    receiver_uid: int
    body: str
    raw_line: str


def _parse_timestamp(raw_ts: str) -> datetime | None:
    normalized = raw_ts.strip()
    for ts_format in TS_FORMATS:
        try:
            return datetime.strptime(normalized, ts_format)
        except ValueError:
            continue
    return None


def parse_line(line: str) -> ChatMessage | None:
    """Parse a single chat log line. Returns None when unmatched."""
    raw_line = line.rstrip("\n")
    if not raw_line:
        return None
    cleaned_line = ANSI_ESCAPE_RE.sub("", raw_line)
    match = LINE_RE.match(cleaned_line)
    if match is None:
        return None
    timestamp = _parse_timestamp(match.group("timestamp"))
    if timestamp is None:
        return None
    try:
        return ChatMessage(
            timestamp=timestamp,
            msg_type=int(match.group("msg_type")),
            sender_uid=int(match.group("sender_uid")),
            receiver_uid=int(match.group("receiver_uid")),
            body=match.group("body"),
            raw_line=raw_line,
        )
    except ValueError:
        return None


def parse_file(path: Path, *, encoding: str = "utf-8") -> Iterator[ChatMessage]:
    """Stream-parse a whole log file and skip malformed lines."""
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = parse_line(line)
            if parsed is None:
                LOGGER.warning("Failed to parse line %s in %s", line_number, path)
                continue
            yield parsed
