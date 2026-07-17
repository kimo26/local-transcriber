# SPDX-License-Identifier: MIT
"""Tests for pure ASR functions: quality scoring, review reasons, device resolution."""

from __future__ import annotations

from transcriber.asr import quality_score, resolve_device_and_compute, review_reasons_for
from transcriber.models import GpuInfo


class TestQualityScore:
    def test_perfect_signal(self) -> None:
        score = quality_score(0.0, 0.0, [1.0, 1.0, 1.0])
        assert score == 1.0

    def test_no_words_uses_neutral_word_score(self) -> None:
        # With no word probabilities the word_score defaults to 0.55.
        # Perfect logprob (0.0) and no-speech (0.0):
        # 0.55 * 0.55 + 0.30 * 1.0 + 0.15 * 1.0 = 0.3025 + 0.30 + 0.15 = 0.7525
        score = quality_score(0.0, 0.0, [])
        assert abs(score - 0.7525) < 0.001

    def test_worst_case_low(self) -> None:
        score = quality_score(-2.0, 1.0, [0.0, 0.0])
        assert score == 0.0

    def test_clamped_to_zero(self) -> None:
        score = quality_score(-99.0, 1.0, [0.0])
        assert score >= 0.0

    def test_logprob_and_speech_components(self) -> None:
        # avg_logprob=-1.5 → logprob_score=0.0; no_speech=0.0 → speech_score=1.0
        # word=[1.0] → word_score=1.0
        # 0.55*1.0 + 0.30*0.0 + 0.15*1.0 = 0.70
        score = quality_score(-1.5, 0.0, [1.0])
        assert abs(score - 0.70) < 0.001

    def test_returns_rounded_to_4dp(self) -> None:
        score = quality_score(-0.5, 0.1, [0.8, 0.9])
        assert score == round(score, 4)


class TestReviewReasonsFor:
    def test_clean_segment_no_reasons(self) -> None:
        reasons = review_reasons_for(
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            compression_ratio=1.5,
            word_probabilities=[0.9, 0.95, 0.92],
            score=0.9,
        )
        assert reasons == []

    def test_low_logprob_flagged(self) -> None:
        reasons = review_reasons_for(-0.9, 0.01, 1.5, [0.9], 0.7)
        assert any("log-probability" in r for r in reasons)

    def test_high_no_speech_flagged(self) -> None:
        reasons = review_reasons_for(-0.1, 0.6, 1.5, [0.9], 0.7)
        assert any("no-speech" in r for r in reasons)

    def test_high_compression_ratio_flagged(self) -> None:
        reasons = review_reasons_for(-0.1, 0.01, 2.5, [0.9], 0.7)
        assert any("repetition" in r for r in reasons)

    def test_low_avg_word_probability_flagged(self) -> None:
        reasons = review_reasons_for(-0.1, 0.01, 1.5, [0.5, 0.6], 0.7)
        assert any("average word probability" in r for r in reasons)

    def test_single_uncertain_word_flagged(self) -> None:
        reasons = review_reasons_for(-0.1, 0.01, 1.5, [0.9, 0.3], 0.7)
        assert any("uncertain word" in r for r in reasons)

    def test_low_score_without_other_reasons_still_flags(self) -> None:
        reasons = review_reasons_for(-0.1, 0.01, 1.5, [0.9], 0.5)
        assert any("aggregate quality" in r for r in reasons)

    def test_no_words_no_word_reasons(self) -> None:
        # Empty probabilities: none of the word-based checks should fire.
        reasons = review_reasons_for(-0.1, 0.01, 1.5, [], 0.9)
        assert not any("word" in r for r in reasons)


class TestResolveDeviceAndCompute:
    def test_auto_with_gpu_gives_cuda_float16(self) -> None:
        gpu = GpuInfo(compute_cap=8.9, cuda_major=12)
        device, compute = resolve_device_and_compute("auto", "auto", gpu)
        assert device == "cuda"
        assert compute == "float16"

    def test_auto_without_gpu_gives_cpu_int8(self) -> None:
        device, compute = resolve_device_and_compute("auto", "auto", None)
        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_device_overrides_auto(self) -> None:
        gpu = GpuInfo(compute_cap=8.9, cuda_major=12)
        device, compute = resolve_device_and_compute("cpu", "auto", gpu)
        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_compute_type_overrides_auto(self) -> None:
        gpu = GpuInfo(compute_cap=8.9, cuda_major=12)
        device, compute = resolve_device_and_compute("auto", "int8_float16", gpu)
        assert compute == "int8_float16"

    def test_explicit_both(self) -> None:
        device, compute = resolve_device_and_compute("cpu", "float32", None)
        assert device == "cpu"
        assert compute == "float32"
