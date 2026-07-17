# SPDX-License-Identifier: MIT
"""Two-pass transcription pipeline used by both the CLI and the API.

Extracting the orchestration here lets both entry points share identical
behaviour without duplicating the pass-1 → inference → pass-2 → Ollama logic.
"""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transcriber.asr import resolve_device_and_compute, transcribe_audio
from transcriber.inference import infer_context_and_glossary
from transcriber.models import GpuInfo, OllamaConfig, SegmentRecord, TranscribeConfig
from transcriber.ollama import correct_with_ollama
from transcriber.utils import eprint, run_ffmpeg

# Progress-event type: callers supply a callback that receives string events.
ProgressCallback = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


def run_pipeline(
    source: Path,
    cfg: TranscribeConfig,
    ollama_cfg: OllamaConfig,
    user_context: str,
    user_glossary: list[str],
    gpu_info: GpuInfo | None,
    progress: ProgressCallback = _noop,
) -> tuple[list[SegmentRecord], dict[str, Any], str, str, str, list[str]]:
    """Execute the full two-pass transcription pipeline.

    Returns:
        (segments, metadata, asr_device, asr_compute_type, final_context, final_glossary)
    """
    device, compute_type = resolve_device_and_compute(cfg.device, cfg.compute_type, gpu_info)

    with tempfile.TemporaryDirectory(prefix="local-transcriber-") as tmp:
        if cfg.skip_ffmpeg_conversion:
            asr_input = source
        else:
            asr_input = Path(tmp) / "decoded_16khz_mono.wav"
            progress("Decoding audio with FFmpeg…")
            run_ffmpeg(source, asr_input, cfg.normalise_audio)

        progress("Pass 1: transcribing without domain-specific hotwords…")
        segments_pass1, metadata = transcribe_audio(
            asr_input, cfg, user_context, user_glossary, device, compute_type
        )

        if not segments_pass1:
            raise RuntimeError(
                "No speech detected in pass 1. "
                "Try --vad-threshold 0.3, set --language explicitly, "
                "or inspect the audio track."
            )

        asr_device = str(metadata.get("asr_device", device))
        asr_compute_type = str(metadata.get("asr_compute_type", compute_type))
        progress(
            f"Pass 1 complete: {len(segments_pass1)} segments | "
            f"{metadata['detected_language']} "
            f"(p={metadata['language_probability']:.3f})"
        )

        if cfg.single_pass:
            segments = segments_pass1
            final_context = user_context
            final_glossary = user_glossary
        else:
            progress("Inferring context and hotwords from pass-1 transcript…")
            use_ollama_for_inference = ollama_cfg.enabled and not cfg.no_hotword_inference
            final_context, final_glossary = infer_context_and_glossary(
                segments_pass1,
                user_context,
                user_glossary,
                ollama_cfg.url,
                ollama_cfg.model,
                ollama_cfg.timeout,
                no_hotword_inference=not use_ollama_for_inference,
            )

            new_terms = [t for t in final_glossary if t not in user_glossary]
            context_changed = final_context != user_context
            if new_terms or context_changed:
                progress(
                    f"Pass 2: re-transcribing with {len(final_glossary)} hotwords "
                    f"({len(new_terms)} newly inferred)…"
                )
                segments, metadata = transcribe_audio(
                    asr_input,
                    cfg,
                    final_context,
                    final_glossary,
                    device,
                    compute_type,
                )
                asr_device = str(metadata.get("asr_device", device))
                asr_compute_type = str(metadata.get("asr_compute_type", compute_type))
            else:
                progress("No new terms inferred; skipping pass 2.")
                segments = segments_pass1

    if not segments:
        raise RuntimeError(
            "No speech was transcribed. Try a lower --vad-threshold, set "
            "--language explicitly, or inspect the audio track."
        )

    if ollama_cfg.enabled:
        gc.collect()
        try:
            progress(f"Applying Ollama correction ({ollama_cfg.model})…")
            correct_with_ollama(segments, ollama_cfg, final_context, final_glossary)
        except Exception as exc:
            eprint(
                f"Warning: Ollama correction was skipped: {exc}. "
                "The raw Whisper transcript will still be written."
            )
            for segment in segments:
                segment.review_reasons.append("Ollama correction was unavailable")
                if not segment.correction_note:
                    segment.correction_note = f"Raw text retained: {exc}"

    return segments, metadata, asr_device, asr_compute_type, final_context, final_glossary
