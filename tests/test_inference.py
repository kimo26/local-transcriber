# SPDX-License-Identifier: MIT
"""Tests for the heuristic term extraction in inference.py."""

from __future__ import annotations

from transcriber.inference import _heuristic_terms
from transcriber.models import SegmentRecord, WordRecord


def _seg(text: str, words: list[WordRecord] | None = None) -> SegmentRecord:
    return SegmentRecord(
        id=0,
        start=0.0,
        end=1.0,
        raw_text=text,
        corrected_text=text,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        compression_ratio=1.0,
        temperature=0.0,
        words=words or [],
    )


class TestHeuristicTerms:
    def test_empty_segments_returns_empty(self) -> None:
        assert _heuristic_terms([]) == []

    def test_acronym_appearing_twice_included(self) -> None:
        segs = [_seg("The GPU was fast"), _seg("The GPU ran well")]
        terms = _heuristic_terms(segs)
        assert "GPU" in terms

    def test_acronym_appearing_once_excluded(self) -> None:
        segs = [_seg("The GPU was fast"), _seg("something else entirely")]
        terms = _heuristic_terms(segs)
        assert "GPU" not in terms

    def test_camelcase_appearing_twice_included(self) -> None:
        segs = [_seg("Use PyTorch for training"), _seg("PyTorch supports CUDA")]
        terms = _heuristic_terms(segs)
        assert "PyTorch" in terms

    def test_uncertain_proper_noun_included_once(self) -> None:
        word = WordRecord(start=0.0, end=0.5, text="Kubernetes", probability=0.4)
        segs = [_seg("deploy on Kubernetes", words=[word])]
        terms = _heuristic_terms(segs)
        assert "Kubernetes" in terms

    def test_high_confidence_word_not_added_via_uncertain_path(self) -> None:
        word = WordRecord(start=0.0, end=0.5, text="Hello", probability=0.95)
        segs = [_seg("Hello world", words=[word])]
        # "Hello" doesn't appear twice in acronym/camelcase, so shouldn't be in heuristics
        terms = _heuristic_terms(segs)
        assert "Hello" not in terms

    def test_result_length_capped(self) -> None:
        # Build many unique acronyms appearing ≥2 times
        texts = [f"TERM{i:03d} TERM{i:03d}" for i in range(200)]
        segs = [_seg(t) for t in texts]
        terms = _heuristic_terms(segs)
        from transcriber.models import _MAX_HEURISTIC_TERMS

        assert len(terms) <= _MAX_HEURISTIC_TERMS

    def test_preserves_insertion_order_for_deduplication(self) -> None:
        # GPU appears twice so gets included; ensure no duplicate entries
        segs = [_seg("GPU test"), _seg("GPU benchmark")]
        terms = _heuristic_terms(segs)
        assert terms.count("GPU") == 1
