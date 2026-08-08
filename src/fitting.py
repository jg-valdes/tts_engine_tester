"""Word budget, duration classification. See docs/source.md Fitting."""

from __future__ import annotations

import math


def word_count(text: str) -> int:
    return len(text.split())


def word_budget(target_ms: int | None, words_per_second: float) -> int | None:
    if not target_ms:
        return None
    return math.floor(target_ms / 1000 * words_per_second)


def classify_fit(
    natural_ms: int, target_ms: int | None, max_compression_ratio: float
) -> tuple[float, str]:
    overflow_ratio = (natural_ms / target_ms) if target_ms else 0.0
    if overflow_ratio <= 1.0:
        fit = "NATURAL"
    elif overflow_ratio <= max_compression_ratio:
        fit = "TIGHT"
    else:
        fit = "OVERFLOW"
    return overflow_ratio, fit
