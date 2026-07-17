# SPDX-License-Identifier: MIT
"""Tests for pure utility functions in utils.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from transcriber.utils import (
    format_clock,
    normalise_whitespace,
    parse_glossary,
    read_optional_text,
    strip_think_tags,
)


class TestNormaliseWhitespace:
    def test_collapses_multiple_spaces(self) -> None:
        assert normalise_whitespace("hello   world") == "hello world"

    def test_strips_leading_and_trailing(self) -> None:
        assert normalise_whitespace("  hello  ") == "hello"

    def test_newlines_become_space(self) -> None:
        assert normalise_whitespace("line1\nline2") == "line1 line2"

    def test_tabs_become_space(self) -> None:
        assert normalise_whitespace("a\t\tb") == "a b"

    def test_empty_string(self) -> None:
        assert normalise_whitespace("") == ""

    def test_already_clean(self) -> None:
        assert normalise_whitespace("hello world") == "hello world"


class TestStripThinkTags:
    def test_removes_think_block(self) -> None:
        text = "before<think>internal reasoning</think>after"
        assert strip_think_tags(text) == "beforeafter"

    def test_multiline_think_block(self) -> None:
        text = "start<think>\nline1\nline2\n</think>end"
        assert strip_think_tags(text) == "startend"

    def test_no_think_tags_unchanged(self) -> None:
        text = "just a normal sentence"
        assert strip_think_tags(text) == text

    def test_strips_surrounding_whitespace(self) -> None:
        result = strip_think_tags("  <think>x</think>  ")
        assert result == ""

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>text<think>b</think>"
        assert strip_think_tags(text) == "text"


class TestFormatClock:
    def test_zero(self) -> None:
        assert format_clock(0.0) == "00:00:00.000"

    def test_one_second(self) -> None:
        assert format_clock(1.0) == "00:00:01.000"

    def test_one_minute(self) -> None:
        assert format_clock(60.0) == "00:01:00.000"

    def test_one_hour(self) -> None:
        assert format_clock(3600.0) == "01:00:00.000"

    def test_fractional_seconds(self) -> None:
        assert format_clock(1.501) == "00:00:01.501"

    def test_negative_clamps_to_zero(self) -> None:
        assert format_clock(-5.0) == "00:00:00.000"

    def test_comma_separator_for_srt(self) -> None:
        result = format_clock(1.0, decimal=",")
        assert result == "00:00:01,000"

    def test_complex_timestamp(self) -> None:
        # 1h 2m 3.456s
        assert format_clock(3600 + 120 + 3.456) == "01:02:03.456"


class TestReadOptionalText:
    def test_none_path_returns_empty(self) -> None:
        assert read_optional_text(None) == ""

    def test_reads_file_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  hello world  ")
            p = Path(f.name)
        assert read_optional_text(p) == "hello world"
        p.unlink()

    def test_missing_file_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Could not read"):
            read_optional_text(Path("/nonexistent/path/file.txt"))


class TestParseGlossary:
    def test_inline_terms(self) -> None:
        terms = parse_glossary(None, "React, TypeScript, Whisper")
        assert terms == ["React", "TypeScript", "Whisper"]

    def test_deduplicates(self) -> None:
        terms = parse_glossary(None, "A, B, A")
        assert terms == ["A", "B"]

    def test_strips_whitespace_from_inline(self) -> None:
        terms = parse_glossary(None, "  React ,  TypeScript ")
        assert terms == ["React", "TypeScript"]

    def test_empty_inline_returns_empty(self) -> None:
        assert parse_glossary(None, "") == []

    def test_file_terms(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("- React\n# comment\n* TypeScript\nfoo\n")
            p = Path(f.name)
        terms = parse_glossary(p, "")
        assert terms == ["React", "TypeScript", "foo"]
        p.unlink()

    def test_file_and_inline_merged(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("GPU\n")
            p = Path(f.name)
        terms = parse_glossary(p, "CPU")
        assert "GPU" in terms
        assert "CPU" in terms
        p.unlink()
