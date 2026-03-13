"""Minimal predicate evaluator for serialized numeric predicates."""

from __future__ import annotations

import re

_PATTERN = re.compile(r"^(<=|>=|<|>|==|!=)\s*(-?\d+\.?\d*)$")
_OPS = {
    "<": float.__lt__,
    ">": float.__gt__,
    "<=": float.__le__,
    ">=": float.__ge__,
    "==": float.__eq__,
    "!=": float.__ne__,
}


def evaluate(value: float, predicate_str: str) -> bool:
    match = _PATTERN.match(str(predicate_str).strip())
    if not match:
        return False
    op, threshold = match.group(1), float(match.group(2))
    return _OPS[op](float(value), threshold)
