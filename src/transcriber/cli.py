# SPDX-License-Identifier: MIT
"""CLI entry point for the local transcription pipeline.

``bootstrap_startup()`` is called at the top of ``main()`` so it only runs
when the script is invoked directly, not when the package is imported by the
API server (which manages its own environment via ``uv sync``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transcriber.models import (
    AUDIO_EXTENSIONS,
    OllamaConfig,
    TranscribeConfig,
)
from transcriber.output import write_outputs
from transcriber.pipeline import run_pipeline
from transcriber.utils import eprint, normalise_whitespace, parse_glossary, read_optional_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe almost any audio format locally with faster-whisper, "
            "Silero VAD, automatic two-pass hotword inference, and an optional "
            "Ollama correction pass. GPU is auto-detected; falls back to CPU "
            "automatically when the GPU architecture is unsupported."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <input_stem>_transcript",
    )

    parser.add_argument("--model", default="large-v3", help="Whisper model name or path")
    parser.add_argument("--model-cache", type=Path, help="Optional model cache directory")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download model files; require an existing local cache/path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. 'auto' picks CUDA when a GPU is detected, else CPU",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "CTranslate2 compute type: auto, float16, int8_float16, int8, float32. "
            "'auto' selects float16 on GPU and int8 on CPU. "
            "INT8 is disabled on Blackwell (sm_120+) by the upstream wheel."
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument(
        "--language",
        default="auto",
        help="ISO language code such as en, de, fr, ar, or auto for detection",
    )
    parser.add_argument(
        "--detect-language-per-segment",
        action="store_true",
        help="Detect language repeatedly for code-switched audio",
    )
    parser.add_argument("--language-detection-threshold", type=float, default=0.5)
    parser.add_argument("--language-detection-segments", type=int, default=3)

    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--patience", type=float, default=1.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--disable-previous-context",
        action="store_true",
        help="Reduce repetition loops at the cost of less cross-window consistency",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=1.5,
        help="Skip long silent regions around likely hallucinations",
    )

    parser.add_argument("--vad-threshold", type=float, default=0.45)
    parser.add_argument("--vad-min-speech-ms", type=int, default=120)
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    parser.add_argument(
        "--normalise-audio",
        action="store_true",
        help="Apply FFmpeg loudness normalisation; leave off for already-clean audio",
    )
    parser.add_argument(
        "--skip-ffmpeg-conversion",
        action="store_true",
        help="Let faster-whisper decode the source directly instead of making a 16 kHz WAV",
    )

    parser.add_argument(
        "--context",
        default="",
        help="Subject, speakers or technical domain used as an ASR/correction hint",
    )
    parser.add_argument("--context-file", type=Path, help="UTF-8 context text file")
    parser.add_argument(
        "--glossary",
        type=Path,
        help="UTF-8 vocabulary file, one term per line; merged with inferred terms",
    )
    parser.add_argument(
        "--hotwords",
        default="",
        help="Comma-separated expected terms; merged with inferred terms",
    )

    parser.add_argument(
        "--single-pass",
        action="store_true",
        help=(
            "Skip hotword inference and the second ASR pass. "
            "Uses only the first transcription result."
        ),
    )
    parser.add_argument(
        "--no-hotword-inference",
        action="store_true",
        help=(
            "Derive hotwords from heuristics only; skip the Ollama inference call. "
            "A second ASR pass is still performed with the heuristic terms."
        ),
    )

    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip LLM correction entirely and keep the raw Whisper transcript",
    )
    parser.add_argument("--ollama-model", default="qwen3:30b-a3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--ollama-timeout", type=int, default=900)
    parser.add_argument("--ollama-retries", type=int, default=3)
    parser.add_argument("--ollama-keep-alive", default="10m")
    parser.add_argument("--ollama-batch-segments", type=int, default=16)
    parser.add_argument("--ollama-batch-characters", type=int, default=6500)
    parser.add_argument(
        "--allow-aggressive-corrections",
        action="store_true",
        help="Relax rewrite-safety thresholds; raw transcript is still preserved",
    )

    parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Do not attempt to install missing packages automatically",
    )

    parser.add_argument("--subtitle-width", type=int, default=48)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.input.exists() or not args.input.is_file():
        raise RuntimeError(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
        eprint(
            f"Warning: uncommon extension {args.input.suffix!r}; FFmpeg will still "
            "attempt format auto-detection."
        )
    if not 0.0 < args.vad_threshold < 1.0:
        raise RuntimeError("--vad-threshold must be between 0 and 1")
    if args.beam_size < 1 or args.best_of < 1:
        raise RuntimeError("--beam-size and --best-of must be positive")
    if args.ollama_retries < 1:
        raise RuntimeError("--ollama-retries must be at least 1")


def _args_to_transcribe_config(args: argparse.Namespace) -> TranscribeConfig:
    return TranscribeConfig(
        model=args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        model_cache=args.model_cache,
        local_files_only=args.local_files_only,
        language=args.language,
        detect_language_per_segment=args.detect_language_per_segment,
        language_detection_threshold=args.language_detection_threshold,
        language_detection_segments=args.language_detection_segments,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=args.patience,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        disable_previous_context=args.disable_previous_context,
        hallucination_silence_threshold=args.hallucination_silence_threshold,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        normalise_audio=args.normalise_audio,
        skip_ffmpeg_conversion=args.skip_ffmpeg_conversion,
        single_pass=args.single_pass,
        no_hotword_inference=args.no_hotword_inference,
        subtitle_width=args.subtitle_width,
    )


def _args_to_ollama_config(args: argparse.Namespace) -> OllamaConfig:
    return OllamaConfig(
        model=args.ollama_model,
        url=args.ollama_url,
        num_ctx=args.ollama_num_ctx,
        timeout=args.ollama_timeout,
        retries=args.ollama_retries,
        keep_alive=args.ollama_keep_alive,
        batch_segments=args.ollama_batch_segments,
        batch_characters=args.ollama_batch_characters,
        allow_aggressive_corrections=args.allow_aggressive_corrections,
        enabled=not args.no_ollama,
    )


def main() -> int:
    from transcriber.bootstrap import _DETECTED_GPU_INFO, bootstrap_startup

    bootstrap_startup()

    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)

        source = args.input.expanduser().resolve()
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else source.parent / f"{source.stem}_transcript"
        )

        user_context = normalise_whitespace(
            " ".join(p for p in [args.context.strip(), read_optional_text(args.context_file)] if p)
        )
        user_glossary = parse_glossary(args.glossary, args.hotwords)

        cfg = _args_to_transcribe_config(args)
        ollama_cfg = _args_to_ollama_config(args)

        gpu_desc = (
            f"compute_cap={_DETECTED_GPU_INFO.compute_cap}"
            if _DETECTED_GPU_INFO
            else "no GPU detected"
        )
        print(f"Device: {cfg.device} ({cfg.compute_type}) | {gpu_desc}")

        segments, metadata, asr_device, asr_compute_type, final_context, final_glossary = (
            run_pipeline(
                source=source,
                cfg=cfg,
                ollama_cfg=ollama_cfg,
                user_context=user_context,
                user_glossary=user_glossary,
                gpu_info=_DETECTED_GPU_INFO,
                progress=print,
            )
        )

        write_outputs(
            output_dir=output_dir,
            source=source,
            segments=segments,
            metadata=metadata,
            cfg=cfg,
            ollama_cfg=ollama_cfg,
            context=final_context,
            glossary=final_glossary,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
        )

        flagged_count = sum(bool(segment.review_reasons) for segment in segments)
        changed_count = sum(segment.correction_applied for segment in segments)
        print(f"\nFinished. Outputs: {output_dir}")
        print(f"Ollama changed {changed_count} segment(s).")
        print(f"{flagged_count} segment(s) are listed in review_needed.txt.")
        print("The untouched Whisper transcript is always kept in transcript_raw.txt.")
        return 0

    except KeyboardInterrupt:
        eprint("Interrupted.")
        return 130
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
