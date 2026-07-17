# SPDX-License-Identifier: MIT
"""ASR layer: device resolution, model loading, and the core transcription loop.

Depends on faster-whisper and ctranslate2, which are installed at bootstrap time.
The top-level try/except keeps the module importable before installation; any
attempt to call the functions before those packages are present will raise a clear
RuntimeError rather than a confusing NameError.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from transcriber.models import (
    _CUDA_FALLBACK_MARKERS,
    GpuInfo,
    SegmentRecord,
    TranscribeConfig,
    WordRecord,
)
from transcriber.utils import eprint, normalise_whitespace

try:
    import ctranslate2
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
    ctranslate2 = None


# Quality scoring


def quality_score(
    avg_logprob: float,
    no_speech_prob: float,
    word_probabilities: Sequence[float],
) -> float:
    word_score = mean(word_probabilities) if word_probabilities else 0.55
    logprob_score = max(0.0, min(1.0, (avg_logprob + 1.5) / 1.5))
    speech_score = max(0.0, min(1.0, 1.0 - no_speech_prob))
    return round(0.55 * word_score + 0.30 * logprob_score + 0.15 * speech_score, 4)


def review_reasons_for(
    avg_logprob: float,
    no_speech_prob: float,
    compression_ratio: float,
    word_probabilities: Sequence[float],
    score: float,
) -> list[str]:
    reasons: list[str] = []
    avg_word = mean(word_probabilities) if word_probabilities else None
    min_word = min(word_probabilities) if word_probabilities else None

    if avg_logprob < -0.85:
        reasons.append(f"low decoder log-probability ({avg_logprob:.2f})")
    if no_speech_prob > 0.50:
        reasons.append(f"high no-speech probability ({no_speech_prob:.2f})")
    if compression_ratio > 2.35:
        reasons.append(f"possible repetition/hallucination ({compression_ratio:.2f})")
    if avg_word is not None and avg_word < 0.68:
        reasons.append(f"low average word probability ({avg_word:.2f})")
    if min_word is not None and min_word < 0.35:
        reasons.append(f"at least one very uncertain word ({min_word:.2f})")
    if score < 0.62 and not reasons:
        reasons.append(f"low aggregate quality score ({score:.2f})")
    return reasons


# Device and compute-type resolution


def resolve_device_and_compute(
    device_arg: str,
    compute_type_arg: str,
    gpu_info: GpuInfo | None,
) -> tuple[str, str]:
    """Resolve 'auto' placeholders to concrete device and compute_type strings.

    Blackwell (sm_120+) note: the stock CTranslate2 wheel disables INT8 on sm_120.
    float16 is unconditionally used on CUDA devices for safety.
    """
    device = ("cuda" if gpu_info else "cpu") if device_arg == "auto" else device_arg

    if compute_type_arg == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = compute_type_arg

    return device, compute_type


def _is_cuda_fallback_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_FALLBACK_MARKERS)


# Model loading with transparent CPU fallback


def probe_and_load_model(
    model_name: str,
    device: str,
    device_index: int,
    compute_type: str,
    cpu_threads: int,
    download_root: str | None,
    local_files_only: bool,
) -> tuple[Any, str, str]:
    """Load WhisperModel, falling back to CPU int8 for unsupported GPU architectures.

    Prebuilt CTranslate2 wheels may not include kernels for newer GPU architectures
    (e.g. Blackwell sm_120). This function probes the device before committing to a
    load and catches runtime CUDA errors on the first attempt, transparently retrying
    on CPU.
    """
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Re-run without --no-auto-install.")

    original_device = device
    actual_device = device
    actual_compute = compute_type

    if device == "cuda":
        if ctranslate2 is None:
            eprint("ctranslate2 unavailable; falling back to CPU (int8).")
            actual_device = "cpu"
            actual_compute = "int8"
        else:
            try:
                gpu_count = ctranslate2.get_cuda_device_count()
                if gpu_count == 0:
                    raise RuntimeError("ctranslate2 reports no CUDA-capable device")
                supported = ctranslate2.get_supported_compute_types("cuda", device_index)
                if compute_type not in supported:
                    raise RuntimeError(
                        f"compute_type {compute_type!r} not supported on this GPU "
                        f"(supported: {sorted(supported)})"
                    )
            except RuntimeError as probe_exc:
                eprint(
                    f"GPU not usable ({probe_exc}). "
                    "Falling back to CPU (int8). "
                    "For speed consider --model large-v3-turbo."
                )
                actual_device = "cpu"
                actual_compute = "int8"

    try:
        model = WhisperModel(
            model_name,
            device=actual_device,
            device_index=device_index,
            compute_type=actual_compute,
            cpu_threads=cpu_threads,
            download_root=download_root,
            local_files_only=local_files_only,
        )
        return model, actual_device, actual_compute
    except Exception as exc:
        if original_device == "cuda" and actual_device == "cuda" and _is_cuda_fallback_error(exc):
            eprint(
                f"GPU init failed ({exc}). "
                "Falling back to CPU (int8). "
                "For speed consider --model large-v3-turbo."
            )
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
                download_root=download_root,
                local_files_only=local_files_only,
            )
            return model, "cpu", "int8"
        raise


# Transcription loop


def transcribe_audio(
    audio_path: Path,
    cfg: TranscribeConfig,
    context: str,
    glossary: list[str],
    device: str,
    compute_type: str,
) -> tuple[list[SegmentRecord], dict[str, Any]]:
    """Transcribe audio and return (segments, metadata).

    The WhisperModel is freed via del + gc.collect inside a try/finally block
    so VRAM is reclaimed before any subsequent Ollama inference call.
    """

    language = None if cfg.language.lower() == "auto" else cfg.language.lower()

    prompt_parts: list[str] = []
    if context:
        prompt_parts.append(f"Context: {context}")
    if glossary:
        prompt_parts.append("Expected vocabulary: " + ", ".join(glossary))
    initial_prompt = ". ".join(prompt_parts)[:1800] or None
    hotwords = ", ".join(glossary)[:1200] or None

    print(f"Loading Whisper model {cfg.model!r} on {device} ({compute_type})...")
    model, actual_device, actual_compute = probe_and_load_model(
        model_name=cfg.model,
        device=device,
        device_index=cfg.device_index,
        compute_type=compute_type,
        cpu_threads=cfg.cpu_threads,
        download_root=str(cfg.model_cache) if cfg.model_cache else None,
        local_files_only=cfg.local_files_only,
    )
    if actual_device != device:
        print(f"  Actual device: {actual_device} ({actual_compute})")

    vad_parameters = {
        "threshold": cfg.vad_threshold,
        "min_speech_duration_ms": cfg.vad_min_speech_ms,
        "min_silence_duration_ms": cfg.vad_min_silence_ms,
        "speech_pad_ms": cfg.vad_speech_pad_ms,
    }

    print("Transcribing with Silero VAD and word timestamps...")
    try:
        segment_generator, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            log_progress=True,
            beam_size=cfg.beam_size,
            best_of=cfg.best_of,
            patience=cfg.patience,
            repetition_penalty=cfg.repetition_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            temperature=(0.0, 0.2, 0.4, 0.6),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=not cfg.disable_previous_context,
            prompt_reset_on_temperature=0.5,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            multilingual=cfg.detect_language_per_segment,
            vad_filter=True,
            vad_parameters=vad_parameters,
            hallucination_silence_threshold=cfg.hallucination_silence_threshold,
            hotwords=hotwords,
            language_detection_threshold=cfg.language_detection_threshold,
            language_detection_segments=cfg.language_detection_segments,
        )

        records: list[SegmentRecord] = []
        for output_segment in segment_generator:
            raw_text = normalise_whitespace(output_segment.text)
            if not raw_text:
                continue

            words: list[WordRecord] = []
            probabilities: list[float] = []
            for word in output_segment.words or []:
                probability = float(word.probability)
                probabilities.append(probability)
                words.append(
                    WordRecord(
                        start=round(float(word.start), 3),
                        end=round(float(word.end), 3),
                        text=word.word,
                        probability=round(probability, 5),
                    )
                )

            score = quality_score(
                float(output_segment.avg_logprob),
                float(output_segment.no_speech_prob),
                probabilities,
            )
            reasons = review_reasons_for(
                float(output_segment.avg_logprob),
                float(output_segment.no_speech_prob),
                float(output_segment.compression_ratio),
                probabilities,
                score,
            )
            records.append(
                SegmentRecord(
                    id=len(records) + 1,
                    start=round(float(output_segment.start), 3),
                    end=round(float(output_segment.end), 3),
                    raw_text=raw_text,
                    corrected_text=raw_text,
                    avg_logprob=round(float(output_segment.avg_logprob), 5),
                    no_speech_prob=round(float(output_segment.no_speech_prob), 5),
                    compression_ratio=round(float(output_segment.compression_ratio), 5),
                    temperature=(
                        None
                        if output_segment.temperature is None
                        else round(float(output_segment.temperature), 3)
                    ),
                    words=words,
                    quality_score=score,
                    review_reasons=reasons,
                )
            )

        metadata: dict[str, Any] = {
            "detected_language": info.language,
            "language_probability": round(float(info.language_probability), 5),
            "duration_seconds": round(float(info.duration), 3),
            "duration_after_vad_seconds": round(float(info.duration_after_vad), 3),
            "vad_removed_seconds": round(
                max(0.0, float(info.duration) - float(info.duration_after_vad)), 3
            ),
            "vad_parameters": vad_parameters,
            "asr_device": actual_device,
            "asr_compute_type": actual_compute,
        }
        return records, metadata

    finally:
        del model
        gc.collect()
