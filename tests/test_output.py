# SPDX-License-Identifier: MIT
"""Tests for output formatters (pure functions)."""

from __future__ import annotations

from transcriber.models import SegmentRecord
from transcriber.output import (
    paragraph_text,
    review_report,
    srt_text,
    timestamped_text,
    vtt_text,
)
from transcriber.utils import format_clock


def _seg(
    seg_id: int = 1,
    start: float = 0.0,
    end: float = 1.0,
    raw: str = "hello",
    corrected: str = "hello",
    review_reasons: list[str] | None = None,
    quality_score: float = 0.9,
) -> SegmentRecord:
    return SegmentRecord(
        id=seg_id,
        start=start,
        end=end,
        raw_text=raw,
        corrected_text=corrected,
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        compression_ratio=1.0,
        temperature=0.0,
        quality_score=quality_score,
        review_reasons=review_reasons or [],
    )


class TestFormatClock:
    def test_zero(self) -> None:
        assert format_clock(0.0) == "00:00:00.000"

    def test_one_hour(self) -> None:
        assert format_clock(3600.0) == "01:00:00.000"

    def test_millis(self) -> None:
        assert format_clock(1.5) == "00:00:01.500"

    def test_negative_clamps_to_zero(self) -> None:
        assert format_clock(-1.0) == "00:00:00.000"

    def test_comma_separator(self) -> None:
        result = format_clock(1.0, decimal=",")
        assert "," in result


class TestParagraphText:
    def test_empty(self) -> None:
        assert paragraph_text([], corrected=True) == ""

    def test_single_segment(self) -> None:
        result = paragraph_text([_seg(corrected="Hello world.")], corrected=True)
        assert "Hello world." in result
        assert result.endswith("\n")

    def test_uses_raw_when_not_corrected(self) -> None:
        seg = _seg(raw="raw text", corrected="corrected text")
        assert "raw text" in paragraph_text([seg], corrected=False)
        assert "corrected text" not in paragraph_text([seg], corrected=False)

    def test_double_newline_for_long_pause(self) -> None:
        seg1 = _seg(seg_id=1, start=0.0, end=1.0, corrected="First.")
        seg2 = _seg(seg_id=2, start=4.0, end=5.0, corrected="Second.")
        result = paragraph_text([seg1, seg2], corrected=True)
        assert "\n\n" in result

    def test_single_space_for_short_pause(self) -> None:
        seg1 = _seg(seg_id=1, start=0.0, end=1.0, corrected="First.")
        seg2 = _seg(seg_id=2, start=1.5, end=2.5, corrected="Second.")
        result = paragraph_text([seg1, seg2], corrected=True)
        assert " " in result
        assert "\n\n" not in result


class TestTimestampedText:
    def test_contains_timestamps(self) -> None:
        result = timestamped_text([_seg(start=0.0, end=1.0, corrected="Hi.")])
        assert "-->" in result
        assert "Hi." in result

    def test_empty_segments(self) -> None:
        assert timestamped_text([]) == ""


class TestSrtText:
    def test_contains_index(self) -> None:
        result = srt_text([_seg(corrected="Test.")], width=48)
        assert result.startswith("1\n")

    def test_uses_comma_separator(self) -> None:
        result = srt_text([_seg(corrected="Test.")], width=48)
        assert "," in result.split("\n")[1]

    def test_empty(self) -> None:
        assert srt_text([], width=48) == ""


class TestVttText:
    def test_starts_with_webvtt(self) -> None:
        result = vtt_text([_seg(corrected="Test.")], width=48)
        assert result.startswith("WEBVTT")

    def test_uses_dot_separator(self) -> None:
        result = vtt_text([_seg(start=1.5, end=2.0, corrected="Test.")], width=48)
        assert "." in result.split("\n")[2]


class TestReviewReport:
    def test_no_flagged(self) -> None:
        result = review_report([_seg()])
        assert "No segments" in result

    def test_flagged_segment_appears(self) -> None:
        seg = _seg(review_reasons=["low quality score"])
        result = review_report([seg])
        assert "low quality score" in result
        assert "Raw:" in result
