# SPDX-License-Identifier: MIT
"""Tests for pure Ollama batch-construction and prompt-building functions."""

from __future__ import annotations

from transcriber.models import SegmentRecord
from transcriber.ollama import correction_prompt, make_batches


def _seg(i: int, text: str, quality: float = 0.9) -> SegmentRecord:
    return SegmentRecord(
        id=i,
        start=float(i),
        end=float(i + 1),
        raw_text=text,
        corrected_text=text,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        compression_ratio=1.0,
        temperature=0.0,
        quality_score=quality,
    )


class TestMakeBatches:
    def test_empty_returns_no_batches(self) -> None:
        assert make_batches([], max_segments=10, max_characters=4000) == []

    def test_single_segment(self) -> None:
        batches = make_batches([_seg(0, "hello")], max_segments=10, max_characters=4000)
        assert batches == [(0, 1)]

    def test_respects_max_segments(self) -> None:
        segs = [_seg(i, "word") for i in range(6)]
        batches = make_batches(segs, max_segments=2, max_characters=4000)
        assert all(end - start <= 2 for start, end in batches)
        # All segments covered
        covered = set()
        for start, end in batches:
            covered.update(range(start, end))
        assert covered == set(range(6))

    def test_respects_max_characters(self) -> None:
        long_text = "x" * 200
        segs = [_seg(i, long_text) for i in range(4)]
        batches = make_batches(segs, max_segments=10, max_characters=300)
        # Each batch covers at most 1 segment (200 chars + 80 overhead = 280 < 300, but
        # adding a second pushes to 560 which exceeds 300).
        assert all(end - start == 1 for start, end in batches)

    def test_all_segments_covered_exactly_once(self) -> None:
        segs = [_seg(i, f"segment text {i}") for i in range(10)]
        batches = make_batches(segs, max_segments=3, max_characters=4000)
        indices = [i for start, end in batches for i in range(start, end)]
        assert sorted(indices) == list(range(10))

    def test_exact_boundary_not_split(self) -> None:
        segs = [_seg(i, "a") for i in range(3)]
        batches = make_batches(segs, max_segments=3, max_characters=4000)
        assert batches == [(0, 3)]


class TestCorrectionPrompt:
    def test_contains_segment_json(self) -> None:
        batch = [_seg(1, "hello world")]
        prompt = correction_prompt(batch, before=[], after=[], context="", glossary=[])
        assert '"hello world"' in prompt
        assert '"id": 1' in prompt

    def test_context_included_when_provided(self) -> None:
        batch = [_seg(1, "test")]
        prompt = correction_prompt(batch, [], [], context="Machine learning lecture", glossary=[])
        assert "Machine learning lecture" in prompt

    def test_glossary_included_when_provided(self) -> None:
        batch = [_seg(1, "test")]
        prompt = correction_prompt(batch, [], [], context="", glossary=["PyTorch", "CUDA"])
        assert "PyTorch" in prompt
        assert "CUDA" in prompt

    def test_no_glossary_uses_fallback_text(self) -> None:
        batch = [_seg(1, "test")]
        prompt = correction_prompt(batch, [], [], context="", glossary=[])
        assert "No supplied glossary" in prompt

    def test_before_context_included(self) -> None:
        before = [_seg(0, "prior sentence")]
        batch = [_seg(1, "current")]
        prompt = correction_prompt(batch, before=before, after=[], context="", glossary=[])
        assert "prior sentence" in prompt

    def test_after_context_included(self) -> None:
        after = [_seg(2, "following sentence")]
        batch = [_seg(1, "current")]
        prompt = correction_prompt(batch, before=[], after=after, context="", glossary=[])
        assert "following sentence" in prompt

    def test_multiple_segments_all_present(self) -> None:
        batch = [_seg(i, f"segment {i}") for i in range(3)]
        prompt = correction_prompt(batch, [], [], context="", glossary=[])
        for i in range(3):
            assert f"segment {i}" in prompt
