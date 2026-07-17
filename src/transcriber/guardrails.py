# SPDX-License-Identifier: MIT
"""Guardrails that prevent the LLM correction pass from altering critical content.

These are pure functions with no I/O so they are easy to unit-test and cheap to
call in a tight loop. The core invariants enforced:

- Numbers and negations must be identical in the raw and corrected texts.
- The corrected length must remain within a configurable ratio of the original.
- The edit distance between simplified forms must exceed a minimum similarity
  threshold, preventing wholesale rewrites from sneaking through.
"""

from __future__ import annotations

import difflib
import re

from transcriber.models import NEGATION_PATTERN, NUMBER_PATTERN
from transcriber.utils import normalise_whitespace


def extract_numbers(text: str) -> list[str]:
    return [match.group(0).lower() for match in NUMBER_PATTERN.finditer(text)]


def extract_negations(text: str) -> list[str]:
    return [match.group(0).lower().replace("'", "'") for match in NEGATION_PATTERN.finditer(text)]


def simplified_for_similarity(text: str) -> str:
    return re.sub(r"[^\w\s]", "", normalise_whitespace(text).lower())


def validate_correction(
    raw: str,
    corrected: str,
    aggressive: bool,
) -> tuple[bool, str]:
    """Return ``(accepted, reason)`` for a proposed ASR correction.

    ``aggressive=True`` widens the length-ratio and similarity thresholds for
    segments that the quality scorer flagged as low-confidence, where larger
    edits are more likely to be genuine fixes.
    """
    corrected = normalise_whitespace(corrected)
    if not corrected:
        return False, "empty correction"
    if extract_numbers(raw) != extract_numbers(corrected):
        return False, "numbers changed"
    if extract_negations(raw) != extract_negations(corrected):
        return False, "negation changed"

    raw_length = max(1, len(raw))
    ratio = len(corrected) / raw_length
    lower, upper = (0.35, 2.5) if aggressive else (0.55, 1.85)
    if not lower <= ratio <= upper:
        return False, f"length ratio {ratio:.2f} outside safe range"

    raw_simple = simplified_for_similarity(raw)
    corrected_simple = simplified_for_similarity(corrected)
    similarity = difflib.SequenceMatcher(None, raw_simple, corrected_simple).ratio()
    minimum = 0.25 if aggressive else 0.42
    if similarity < minimum:
        return False, f"edit similarity {similarity:.2f} below safe threshold"

    return True, "accepted"
