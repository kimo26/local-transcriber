# SPDX-License-Identifier: MIT
"""Tests for output formatters (pure functions)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from transcriber.models import OllamaConfig, SegmentRecord, TranscribeConfig
from transcriber.output import (
    paragraph_text,
    review_report,
    srt_text,
    subtitle_text,
    timestamped_text,
    vtt_text,
    write_outputs,
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


class TestSubtitleText:
    def test_wraps_long_line(self) -> None:
        long = "word " * 20
        result = subtitle_text(long, width=40)
        assert all(len(line) <= 40 for line in result.splitlines())

    def test_short_text_unchanged(self) -> None:
        assert subtitle_text("hello", width=80) == "hello"

    def test_normalises_whitespace(self) -> None:
        result = subtitle_text("  hello   world  ", width=80)
        assert result == "hello world"


class TestReviewReport:
    def test_no_flagged(self) -> None:
        result = review_report([_seg()])
        assert "No segments" in result

    def test_flagged_segment_appears(self) -> None:
        seg = _seg(review_reasons=["low quality score"])
        result = review_report([seg])
        assert "low quality score" in result
        assert "Raw:" in result


class TestWriteOutputs:
    def _default_cfg(self) -> TranscribeConfig:
        return TranscribeConfig(model="large-v3")

    def _default_ollama(self, model: str = "qwen3:7b") -> OllamaConfig:
        return OllamaConfig(model=model, enabled=True)

    def test_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            write_outputs(
                output_dir=out,
                source=Path("/tmp/audio.mp3"),
                segments=[_seg()],
                metadata={"detected_language": "en", "language_probability": 0.99},
                cfg=self._default_cfg(),
                ollama_cfg=self._default_ollama(),
                context="",
                glossary=[],
                asr_device="cpu",
                asr_compute_type="int8",
            )
            expected = [
                "transcript.txt",
                "transcript_raw.txt",
                "transcript_timestamped.txt",
                "transcript.srt",
                "transcript.vtt",
                "review_needed.txt",
                "transcript.json",
            ]
            for name in expected:
                assert (out / name).exists(), f"{name} missing"

    def test_json_ollama_model_uses_ollama_config(self) -> None:
        """The JSON document must record the Ollama model, not the ASR model."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            write_outputs(
                output_dir=out,
                source=Path("/tmp/audio.mp3"),
                segments=[_seg()],
                metadata={},
                cfg=self._default_cfg(),
                ollama_cfg=self._default_ollama(model="qwen3:7b"),
                context="",
                glossary=[],
                asr_device="cpu",
                asr_compute_type="int8",
            )
            doc = json.loads((out / "transcript.json").read_text())
            assert doc["pipeline"]["ollama_model"] == "qwen3:7b"
            assert doc["pipeline"]["ollama_model"] != "large-v3"

    def test_json_ollama_model_null_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            write_outputs(
                output_dir=out,
                source=Path("/tmp/audio.mp3"),
                segments=[_seg()],
                metadata={},
                cfg=self._default_cfg(),
                ollama_cfg=OllamaConfig(model="qwen3:7b", enabled=False),
                context="",
                glossary=[],
                asr_device="cpu",
                asr_compute_type="int8",
            )
            doc = json.loads((out / "transcript.json").read_text())
            assert doc["pipeline"]["ollama_model"] is None
