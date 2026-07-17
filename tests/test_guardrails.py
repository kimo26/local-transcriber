# SPDX-License-Identifier: MIT
"""Tests for the guardrails module (pure functions, no I/O)."""

from __future__ import annotations

from transcriber.guardrails import (
    extract_negations,
    extract_numbers,
    validate_correction,
)


class TestExtractNumbers:
    def test_integer(self) -> None:
        assert extract_numbers("there are 42 items") == ["42"]

    def test_decimal(self) -> None:
        assert extract_numbers("costs 3.14 dollars") == ["3.14"]

    def test_multiple(self) -> None:
        assert extract_numbers("between 1 and 100") == ["1", "100"]

    def test_no_numbers(self) -> None:
        assert extract_numbers("hello world") == []

    def test_version_number(self) -> None:
        assert extract_numbers("python 3.11.2") == ["3.11.2"]


class TestExtractNegations:
    def test_not(self) -> None:
        assert extract_negations("it is not working") == ["not"]

    def test_contraction(self) -> None:
        assert extract_negations("I can't do that") == ["can't"]

    def test_multiple_negations(self) -> None:
        result = extract_negations("no, never, not again")
        assert "no" in result
        assert "never" in result
        assert "not" in result

    def test_no_negation(self) -> None:
        assert extract_negations("everything is fine") == []


class TestValidateCorrection:
    def test_accepts_identical(self) -> None:
        ok, reason = validate_correction("hello world", "hello world", aggressive=False)
        assert ok
        assert reason == "accepted"

    def test_accepts_minor_fix(self) -> None:
        ok, reason = validate_correction("helo world", "hello world", aggressive=False)
        assert ok

    def test_rejects_empty(self) -> None:
        ok, reason = validate_correction("hello", "", aggressive=False)
        assert not ok
        assert "empty" in reason

    def test_rejects_changed_number(self) -> None:
        ok, reason = validate_correction("there are 5 items", "there are 6 items", aggressive=False)
        assert not ok
        assert "numbers" in reason

    def test_rejects_changed_negation(self) -> None:
        ok, reason = validate_correction("I cannot go", "I can go", aggressive=False)
        assert not ok
        assert "negation" in reason

    def test_rejects_too_long(self) -> None:
        raw = "short text"
        corrected = "this is a very very very very very long rewrite of the original text"
        ok, reason = validate_correction(raw, corrected, aggressive=False)
        assert not ok
        assert "length ratio" in reason

    def test_rejects_too_short(self) -> None:
        raw = "this is quite a long sentence with many words"
        corrected = "ok"
        ok, reason = validate_correction(raw, corrected, aggressive=False)
        assert not ok

    def test_rejects_dissimilar(self) -> None:
        # Same approximate length but completely different words → similarity fails.
        raw = "alpha beta gamma delta"
        corrected = "xyzzy quux frob bork"
        ok, reason = validate_correction(raw, corrected, aggressive=False)
        assert not ok
        assert "similarity" in reason

    def test_aggressive_accepts_wider_ratio(self) -> None:
        # Aggressive mode allows ratios down to 0.35 (tighter lower bound) and up to 2.5.
        # "hello world" (11) vs "hello" (5): ratio 0.45, accepted in aggressive, rejected otherwise.
        ok_aggressive, _ = validate_correction("hello world", "hello", aggressive=True)
        ok_normal, _ = validate_correction("hello world", "hello", aggressive=False)
        assert ok_aggressive
        assert not ok_normal

    def test_preserves_corrected_whitespace(self) -> None:
        ok, _ = validate_correction("hello   world", "hello world", aggressive=False)
        assert ok
