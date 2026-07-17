# SPDX-License-Identifier: MIT
"""Tests for dataclass defaults and round-trip serialisation."""

from __future__ import annotations

from dataclasses import asdict

from transcriber.models import (
    GpuInfo,
    OllamaConfig,
    SegmentRecord,
    TranscribeConfig,
    WordRecord,
)


class TestSegmentRecord:
    def test_default_words_are_empty_list(self) -> None:
        seg = SegmentRecord(
            id=1,
            start=0.0,
            end=1.0,
            raw_text="hello",
            corrected_text="hello",
            avg_logprob=-0.3,
            no_speech_prob=0.1,
            compression_ratio=1.0,
            temperature=0.0,
        )
        assert seg.words == []

    def test_default_review_reasons_are_empty(self) -> None:
        seg = SegmentRecord(
            id=1,
            start=0.0,
            end=1.0,
            raw_text="hello",
            corrected_text="hello",
            avg_logprob=-0.3,
            no_speech_prob=0.1,
            compression_ratio=1.0,
            temperature=None,
        )
        assert seg.review_reasons == []

    def test_correction_applied_defaults_false(self) -> None:
        seg = SegmentRecord(
            id=1,
            start=0.0,
            end=1.0,
            raw_text="x",
            corrected_text="x",
            avg_logprob=-0.5,
            no_speech_prob=0.0,
            compression_ratio=1.0,
            temperature=0.0,
        )
        assert seg.correction_applied is False

    def test_asdict_round_trip(self) -> None:
        word = WordRecord(start=0.1, end=0.5, text="hi", probability=0.99)
        seg = SegmentRecord(
            id=2,
            start=0.1,
            end=0.8,
            raw_text="hi",
            corrected_text="hi",
            avg_logprob=-0.2,
            no_speech_prob=0.05,
            compression_ratio=1.1,
            temperature=0.0,
            words=[word],
            quality_score=0.9,
        )
        d = asdict(seg)
        assert d["id"] == 2
        assert d["words"][0]["text"] == "hi"
        assert d["quality_score"] == 0.9


class TestTranscribeConfig:
    def test_default_model_is_large_v3(self) -> None:
        cfg = TranscribeConfig()
        assert cfg.model == "large-v3"

    def test_default_device_is_auto(self) -> None:
        cfg = TranscribeConfig()
        assert cfg.device == "auto"

    def test_custom_values_are_stored(self) -> None:
        cfg = TranscribeConfig(model="large-v3-turbo", language="en")
        assert cfg.model == "large-v3-turbo"
        assert cfg.language == "en"


class TestOllamaConfig:
    def test_enabled_by_default(self) -> None:
        assert OllamaConfig().enabled is True

    def test_can_disable(self) -> None:
        assert OllamaConfig(enabled=False).enabled is False


class TestGpuInfo:
    def test_fields(self) -> None:
        gpu = GpuInfo(compute_cap=12.0, cuda_major=13)
        assert gpu.compute_cap == 12.0
        assert gpu.cuda_major == 13
