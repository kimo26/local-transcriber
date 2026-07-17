#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Local, open-source audio transcription pipeline with adaptive GPU/CPU execution.

Pipeline:
  1. Bootstrap: auto-detects the GPU compute capability and CUDA driver version,
     installs faster-whisper plus matching CUDA libraries via subprocess if absent,
     then re-execs the process with the correct LD_LIBRARY_PATH so the dynamic
     loader can find the pip-installed cuBLAS/cuDNN shared objects.
  2. FFmpeg decodes almost any audio/video container to mono 16 kHz PCM.
  3. Pass 1: faster-whisper large-v3 with Silero VAD and word timestamps.
     On Blackwell (sm_120+) GPUs, compute_type is forced to float16; if the
     prebuilt CTranslate2 wheel cannot use the GPU (missing sm_120 kernels), the
     pipeline falls back to CPU int8 automatically with a one-line notice.
  4. Context and hotword inference: acronyms, CamelCase tokens, and low-confidence
     proper nouns are extracted via heuristics, then optionally refined by a
     constrained Ollama JSON call that returns a context summary and glossary.
     The Ollama model is unloaded immediately (keep_alive: 0) to free VRAM.
  5. Pass 2: faster-whisper re-transcribes with the inferred hotwords and context
     as an initial_prompt, improving recognition of domain-specific vocabulary.
  6. Ollama correction pass: a local LLM (default: qwen3:30b-a3b) receives
     batches of segments with surrounding context and returns constrained JSON
     corrections. think:false is sent and any leaked <think> blocks are stripped.
  7. Guardrails: edits that alter numbers, negations, or too much of the original
     wording are rejected; the raw transcript is preserved in every case.

Outputs (in <filename>_transcript/):
  transcript.txt              Clean corrected transcript
  transcript_timestamped.txt  Corrected transcript with timestamps
  transcript_raw.txt          Original ASR text, never modified by the LLM
  transcript.srt              Corrected subtitles
  transcript.vtt              Corrected WebVTT subtitles
  transcript.json             Full metadata, raw/corrected text and word timings
  review_needed.txt           Low-confidence or uncertain regions

All processing is local. No cloud transcription or paid API is used.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

# These packages may be absent on the very first invocation; bootstrap_startup()
# installs them via subprocess and re-execs so they are available to every
# function that uses them. The top-level try/except avoids inline imports in
# function bodies while still allowing the module to load before installation.
try:
    from faster_whisper import WhisperModel
    import ctranslate2
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]
    ctranslate2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# GPU hardware descriptor
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    compute_cap: float  # e.g. 12.0 for Blackwell sm_120
    cuda_major: int     # inferred CUDA major version from driver


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WordRecord:
    start: float
    end: float
    text: str
    probability: float


@dataclass
class SegmentRecord:
    id: int
    start: float
    end: float
    raw_text: str
    corrected_text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float | None
    words: list[WordRecord] = field(default_factory=list)
    quality_score: float = 0.0
    review_reasons: list[str] = field(default_factory=list)
    uncertain_terms: list[str] = field(default_factory=list)
    correction_note: str = ""
    correction_applied: bool = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {
    ".aac", ".ac3", ".aiff", ".alac", ".amr", ".ape", ".caf", ".dts",
    ".flac", ".m4a", ".m4b", ".mka", ".mp2", ".mp3", ".oga", ".ogg",
    ".opus", ".ra", ".tta", ".wav", ".wma", ".wv", ".webm", ".3gp",
    ".mp4", ".mkv", ".mov", ".avi", ".mts", ".m2ts",
}

NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|none|nothing|neither|nor|without|cannot|can't|"
    r"don't|doesn't|didn't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|"
    r"wasn't|weren't|hasn't|haven't|hadn't)\b|n['']t\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d[\d.,:/-]*)(?!\w)")
SPACE_PATTERN = re.compile(r"\s+")
_THINK_TAGS_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# Heuristic term-extraction patterns for the hotword inference step.
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
_CAMELCASE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_TECH_TOKEN_PATTERN = re.compile(r"\b[A-Za-z][\w]*[-_][\w]+\b")

CORRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "corrected_text": {"type": "string"},
                    "uncertain_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "corrected_text", "uncertain_terms", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "context_summary": {"type": "string"},
        "glossary": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["context_summary", "glossary"],
    "additionalProperties": False,
}

# Error substrings that indicate a missing GPU kernel rather than a real failure.
# Catching these allows a transparent CPU fallback instead of a crash.
_CUDA_FALLBACK_MARKERS = (
    "CUBLAS_STATUS_NOT_SUPPORTED",
    "CUBLAS_STATUS_",
    "no CUDA-capable device",
    "no kernel image is available",
    "libcublas",
    "libcudnn",
    "libcudart",
    "cannot be loaded",
    "cannot open shared object",
)

# Linux: minimum driver version (major) that implies a given CUDA major version.
_DRIVER_CUDA_MAP: list[tuple[int, int]] = [(525, 12), (450, 11), (418, 10)]

_MAX_GLOSSARY_TERMS = 80
_MAX_HEURISTIC_TERMS = 80
_MAX_INFERENCE_CHARS = 6_000


# ---------------------------------------------------------------------------
# Bootstrap: GPU detection, dependency installation, LD_LIBRARY_PATH re-exec
# ---------------------------------------------------------------------------

_DETECTED_GPU_INFO: GpuInfo | None = None


def detect_gpu_info() -> GpuInfo | None:
    """Return the first GPU's compute capability and inferred CUDA version."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        compute_cap = float(parts[0])
        driver_major = int(parts[1].split(".")[0])
        cuda_major = 0
        for min_driver, cuda_ver in _DRIVER_CUDA_MAP:
            if driver_major >= min_driver:
                cuda_major = cuda_ver
                break
        return GpuInfo(compute_cap=compute_cap, cuda_major=cuda_major)
    except Exception:
        return None


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _ensure_venv_and_reexec() -> None:
    """If not inside a venv, create .venv in the script directory and re-exec.

    Ubuntu (and other modern distros) mark the system Python as externally-managed
    (PEP 668), which prevents pip from installing packages globally. Creating a
    project-local venv and re-execing into it gives pip a safe install target
    without requiring the user to manually set up an environment first.
    """
    if _in_venv() or os.environ.get("LOCAL_TRANSCRIBER_IN_VENV") == "1":
        return

    script_dir = Path(__file__).resolve().parent
    venv_dir = script_dir / ".venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"

    if not venv_python.exists():
        print(f"Creating virtual environment at {venv_dir} ...", file=sys.stderr)
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    env = os.environ.copy()
    env["LOCAL_TRANSCRIBER_IN_VENV"] = "1"
    env["VIRTUAL_ENV"] = str(venv_dir)
    os.execvpe(str(venv_python), [str(venv_python), *sys.argv], env)


def _pip_installer() -> list[str]:
    """Return the pip command prefix appropriate for this environment.

    uv pip is used when available inside a virtual environment. Outside a venv
    this function should not be called (ensure_dependencies is only reached
    after _ensure_venv_and_reexec guarantees we are inside one).
    """
    if shutil.which("uv"):
        return ["uv", "pip"]
    return [sys.executable, "-m", "pip"]


def ensure_dependencies(gpu_info: GpuInfo | None) -> None:
    """Install faster-whisper and CUDA-version-matched libraries if not present."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return

    print(
        "faster-whisper not found. Installing automatically. "
        "Pass --no-auto-install to skip.",
        file=sys.stderr,
    )

    packages = ["faster-whisper"]
    if gpu_info and gpu_info.cuda_major >= 12:
        packages += ["nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"]
    elif gpu_info and gpu_info.cuda_major == 11:
        # CUDA 11 + cuDNN 8 requires an older ctranslate2.
        packages += ["ctranslate2==4.4.0"]
    # CPU-only: no NVIDIA packages needed.

    installer = _pip_installer()
    cmd = installer + ["install"] + packages
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Dependency installation failed. Install manually:\n"
            f"  {' '.join(installer)} install {' '.join(packages)}"
        )


def _setup_library_path_and_reexec() -> None:
    """Prepend pip-installed cuBLAS/cuDNN directories to LD_LIBRARY_PATH and re-exec.

    CTranslate2 wheels expect cuBLAS and cuDNN to be discoverable by the Linux
    dynamic loader. When they are installed via pip inside a venv, their directories
    must appear in LD_LIBRARY_PATH before the process starts. We compute the correct
    path, update the environment, and re-exec so the loader sees the libraries from
    the very beginning of the next run.
    """
    if os.name != "posix":
        return

    library_dirs: list[str] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file:
                library_dirs.append(str(Path(module_file).resolve().parent))
        except ImportError:
            pass

    if not library_dirs:
        return

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = list(dict.fromkeys(library_dirs + current))
    if merged == current:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["LOCAL_TRANSCRIBER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def bootstrap_startup() -> None:
    """Run once at module import time to prepare the runtime environment.

    1. Detects GPU compute capability and CUDA version via nvidia-smi.
    2. If not inside a virtual environment, creates .venv in the script directory
       and re-execs into it (required on PEP 668 externally-managed systems).
    3. Installs faster-whisper and matching CUDA libraries if absent (unless
       --no-auto-install is in sys.argv).
    4. Re-execs the process with an updated LD_LIBRARY_PATH if pip-installed
       NVIDIA libraries need to be visible to the dynamic loader.
    """
    global _DETECTED_GPU_INFO

    _DETECTED_GPU_INFO = detect_gpu_info()

    if "--no-auto-install" not in sys.argv:
        _ensure_venv_and_reexec()

    already_ready = os.environ.get("LOCAL_TRANSCRIBER_LD_READY") == "1"
    if not already_ready and "--no-auto-install" not in sys.argv:
        ensure_dependencies(_DETECTED_GPU_INFO)

    _setup_library_path_and_reexec()


bootstrap_startup()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def normalise_whitespace(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text).strip()


def read_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def parse_glossary(path: Path | None, inline_terms: str) -> list[str]:
    terms: list[str] = []
    if path:
        content = read_optional_text(path)
        for line in content.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line and not line.startswith("#"):
                terms.append(line)
    if inline_terms:
        terms.extend(term.strip() for term in inline_terms.split(",") if term.strip())
    return list(dict.fromkeys(terms))


def format_clock(seconds: float, decimal: str = ".") -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000.0))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models may include."""
    return _THINK_TAGS_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# FFmpeg decode
# ---------------------------------------------------------------------------


def run_ffmpeg(
    source: Path,
    destination: Path,
    normalise_audio: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is not installed. Install it with: sudo apt install ffmpeg"
        )

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-nostdin", "-y", "-i", str(source),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
    ]
    if normalise_audio:
        command.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    command.extend(["-c:a", "pcm_s16le", str(destination)])

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg could not decode the input: {detail}")


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Device and compute-type resolution + GPU probe
# ---------------------------------------------------------------------------


def resolve_device_and_compute(
    device_arg: str,
    compute_type_arg: str,
    gpu_info: GpuInfo | None,
) -> tuple[str, str]:
    """Resolve 'auto' placeholders to concrete device and compute_type strings.

    Blackwell (sm_120+) note: INT8 is explicitly disabled in the stock CTranslate2
    wheel for sm_120 GPUs. float16 is used on all CUDA devices for safety.
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


def probe_and_load_model(
    model_name: str,
    device: str,
    device_index: int,
    compute_type: str,
    cpu_threads: int,
    download_root: str | None,
    local_files_only: bool,
) -> tuple[Any, str, str]:
    """Load WhisperModel with a transparent CPU fallback for unsupported GPU archs.

    The prebuilt CTranslate2 wheels may not contain kernels for newer GPU
    architectures such as Blackwell (sm_120). This function probes the device
    before committing to a model load and catches runtime CUDA errors on the
    first load attempt, falling back to CPU int8 in both cases.
    """
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Re-run without --no-auto-install."
        )

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


# ---------------------------------------------------------------------------
# ASR — callable twice (pass 1 and pass 2)
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_path: Path,
    args: argparse.Namespace,
    context: str,
    glossary: list[str],
    device: str,
    compute_type: str,
) -> tuple[list[SegmentRecord], dict[str, Any]]:
    """Transcribe audio and return (segments, metadata).

    The WhisperModel is freed (del + gc.collect) inside this function via
    try/finally so GPU VRAM is reclaimed before any subsequent Ollama call.
    """
    language = None if args.language.lower() == "auto" else args.language.lower()

    prompt_parts: list[str] = []
    if context:
        prompt_parts.append(f"Context: {context}")
    if glossary:
        prompt_parts.append("Expected vocabulary: " + ", ".join(glossary))
    initial_prompt = ". ".join(prompt_parts)[:1800] or None
    hotwords = ", ".join(glossary)[:1200] or None

    print(f"Loading Whisper model {args.model!r} on {device} ({compute_type})...")
    model, actual_device, actual_compute = probe_and_load_model(
        model_name=args.model,
        device=device,
        device_index=args.device_index,
        compute_type=compute_type,
        cpu_threads=args.cpu_threads,
        download_root=str(args.model_cache) if args.model_cache else None,
        local_files_only=args.local_files_only,
    )
    if actual_device != device:
        print(f"  Actual device: {actual_device} ({actual_compute})")

    vad_parameters = {
        "threshold": args.vad_threshold,
        "min_speech_duration_ms": args.vad_min_speech_ms,
        "min_silence_duration_ms": args.vad_min_silence_ms,
        "speech_pad_ms": args.vad_speech_pad_ms,
    }

    print("Transcribing with Silero VAD and word timestamps...")
    try:
        segment_generator, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            log_progress=True,
            beam_size=args.beam_size,
            best_of=args.best_of,
            patience=args.patience,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            temperature=(0.0, 0.2, 0.4, 0.6),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=not args.disable_previous_context,
            prompt_reset_on_temperature=0.5,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            multilingual=args.detect_language_per_segment,
            vad_filter=True,
            vad_parameters=vad_parameters,
            hallucination_silence_threshold=args.hallucination_silence_threshold,
            hotwords=hotwords,
            language_detection_threshold=args.language_detection_threshold,
            language_detection_segments=args.language_detection_segments,
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


# ---------------------------------------------------------------------------
# Ollama HTTP utilities
# ---------------------------------------------------------------------------


def get_json(url: str, timeout: int = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_ollama(ollama_url: str, model: str) -> None:
    try:
        data = get_json(f"{ollama_url.rstrip('/')}/api/tags")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start it with `ollama serve`, or use --no-ollama."
        ) from exc

    names = {
        item.get("name", "")
        for item in data.get("models", [])
        if isinstance(item, dict)
    }
    short_names = {name.split(":", 1)[0] for name in names}
    if model not in names and model.split(":", 1)[0] not in short_names:
        available = ", ".join(sorted(names)) or "none"
        raise RuntimeError(
            f"Ollama model {model!r} is not installed. Run `ollama pull {model}`. "
            f"Installed models: {available}"
        )


# ---------------------------------------------------------------------------
# Batch construction and correction prompt
# ---------------------------------------------------------------------------


def make_batches(
    segments: Sequence[SegmentRecord],
    max_segments: int,
    max_characters: int,
) -> list[tuple[int, int]]:
    batches: list[tuple[int, int]] = []
    start = 0
    while start < len(segments):
        end = start
        characters = 0
        while end < len(segments) and end - start < max_segments:
            next_size = len(segments[end].raw_text) + 80
            if end > start and characters + next_size > max_characters:
                break
            characters += next_size
            end += 1
        batches.append((start, end))
        start = end
    return batches


def correction_prompt(
    batch: Sequence[SegmentRecord],
    before: Sequence[SegmentRecord],
    after: Sequence[SegmentRecord],
    context: str,
    glossary: Sequence[str],
) -> str:
    payload = [
        {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "raw_text": segment.raw_text,
            "quality_score": segment.quality_score,
            "review_reasons": segment.review_reasons,
        }
        for segment in batch
    ]
    before_text = " ".join(s.corrected_text for s in before)
    after_text = " ".join(s.raw_text for s in after)
    glossary_text = ", ".join(glossary) if glossary else "No supplied glossary."

    return f"""
You are correcting automatic speech-recognition output, not rewriting prose.

Hard rules:
1. Preserve the speaker's exact meaning, register, uncertainty, repetition and technical detail.
2. Correct only likely ASR spelling, punctuation, casing, word-boundary and homophone errors.
3. Use the surrounding context and glossary to resolve technical terms and proper nouns.
4. Never invent facts or words that are not supported by the raw transcript.
5. Never change numbers, units, mathematical symbols, versions, commands, URLs, code, or negations unless the correction is unmistakable from context.
6. Do not summarise, censor, improve style, remove filler words, or combine/split segment IDs.
7. When unsure, keep the raw wording unchanged and list the uncertain expression.
8. Return every requested segment exactly once using the supplied JSON schema.

Overall subject/context:
{context or "No additional context supplied."}

Expected vocabulary / names / acronyms:
{glossary_text}

Immediately preceding context:
{before_text or "None."}

Immediately following context:
{after_text or "None."}

Segments to correct:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def extract_numbers(text: str) -> list[str]:
    return [match.group(0).lower() for match in NUMBER_PATTERN.finditer(text)]


def extract_negations(text: str) -> list[str]:
    return [
        match.group(0).lower().replace("'", "'")
        for match in NEGATION_PATTERN.finditer(text)
    ]


def simplified_for_similarity(text: str) -> str:
    return re.sub(r"[^\w\s]", "", normalise_whitespace(text).lower())


def validate_correction(
    raw: str,
    corrected: str,
    aggressive: bool,
) -> tuple[bool, str]:
    corrected = normalise_whitespace(corrected)
    if not corrected:
        return False, "empty correction"
    if extract_numbers(raw) != extract_numbers(corrected):
        return False, "numbers changed"
    if extract_negations(raw) != extract_negations(corrected):
        return False, "negation changed"

    raw_length = max(1, len(raw))
    ratio = len(corrected) / raw_length
    lower, upper = (0.35, 2.5) if aggressive else (0.55, 1.85)
    if not lower <= ratio <= upper:
        return False, f"length ratio {ratio:.2f} outside safe range"

    raw_simple = simplified_for_similarity(raw)
    corrected_simple = simplified_for_similarity(corrected)
    similarity = difflib.SequenceMatcher(None, raw_simple, corrected_simple).ratio()
    minimum = 0.25 if aggressive else 0.42
    if similarity < minimum:
        return False, f"edit similarity {similarity:.2f} below safe threshold"

    return True, "accepted"


# ---------------------------------------------------------------------------
# LLM correction pass
# ---------------------------------------------------------------------------


def correct_with_ollama(
    segments: list[SegmentRecord],
    args: argparse.Namespace,
    context: str,
    glossary: list[str],
) -> None:
    ollama_url = args.ollama_url.rstrip("/")
    verify_ollama(ollama_url, args.ollama_model)

    batches = make_batches(
        segments,
        max_segments=args.ollama_batch_segments,
        max_characters=args.ollama_batch_characters,
    )
    print(
        f"Correcting {len(segments)} segments with "
        f"{args.ollama_model!r} in {len(batches)} batches..."
    )

    for batch_number, (start, end) in enumerate(batches, start=1):
        batch = segments[start:end]
        before = segments[max(0, start - 2):start]
        after = segments[end:min(len(segments), end + 2)]
        prompt = correction_prompt(batch, before, after, context, glossary)

        payload: dict[str, Any] = {
            "model": args.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Act as a conservative ASR error corrector. Output only "
                        "schema-valid JSON. Never rewrite or add unsupported content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": CORRECTION_SCHEMA,
            "think": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "seed": 0,
                "num_ctx": args.ollama_num_ctx,
            },
            "keep_alive": args.ollama_keep_alive,
        }

        response_data: dict[str, Any] | None = None
        error: Exception | None = None
        for attempt in range(1, args.ollama_retries + 1):
            try:
                response_data = post_json(
                    f"{ollama_url}/api/chat",
                    payload,
                    timeout=args.ollama_timeout,
                )
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < args.ollama_retries:
                    time.sleep(min(2**attempt, 8))

        if response_data is None:
            for segment in batch:
                segment.review_reasons.append("Ollama correction request failed")
                segment.correction_note = f"Ollama failure: {error}"
            eprint(f"Batch {batch_number}/{len(batches)} failed; raw text retained.")
            continue

        try:
            message_content = strip_think_tags(
                response_data["message"]["content"]
            )
            parsed = json.loads(message_content)
            candidates = parsed["segments"]
            by_id = {
                int(candidate["id"]): candidate
                for candidate in candidates
                if isinstance(candidate, dict) and "id" in candidate
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            for segment in batch:
                segment.review_reasons.append("invalid Ollama structured output")
                segment.correction_note = f"Invalid Ollama output: {exc}"
            eprint(
                f"Batch {batch_number}/{len(batches)} returned invalid JSON; "
                "raw text retained."
            )
            continue

        expected_ids = {segment.id for segment in batch}
        if set(by_id) != expected_ids:
            missing = expected_ids - set(by_id)
            extra = set(by_id) - expected_ids
            eprint(
                f"Batch {batch_number}: ID mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)}). "
                "Invalid segments remain raw."
            )

        for segment in batch:
            candidate = by_id.get(segment.id)
            if candidate is None:
                segment.review_reasons.append("Ollama omitted this segment")
                segment.correction_note = "Raw text retained because segment was omitted."
                continue

            proposed = normalise_whitespace(str(candidate.get("corrected_text", "")))
            valid, validation_note = validate_correction(
                segment.raw_text,
                proposed,
                aggressive=args.allow_aggressive_corrections,
            )
            uncertain = candidate.get("uncertain_terms", [])
            if isinstance(uncertain, list):
                segment.uncertain_terms = [
                    normalise_whitespace(str(term))
                    for term in uncertain
                    if normalise_whitespace(str(term))
                ][:20]
            reason = normalise_whitespace(str(candidate.get("reason", "")))

            if valid:
                segment.corrected_text = proposed
                segment.correction_applied = proposed != segment.raw_text
                segment.correction_note = reason or validation_note
            else:
                segment.corrected_text = segment.raw_text
                segment.correction_applied = False
                segment.review_reasons.append(
                    f"rejected unsafe Ollama edit: {validation_note}"
                )
                segment.correction_note = (
                    f"Raw text retained. Proposed edit rejected: {validation_note}."
                )

            if segment.uncertain_terms:
                segment.review_reasons.append(
                    "Ollama marked uncertain: " + ", ".join(segment.uncertain_terms)
                )

        print(f"Corrected batch {batch_number}/{len(batches)}")


# ---------------------------------------------------------------------------
# Hotword and context inference from pass-1 transcript
# ---------------------------------------------------------------------------


def _heuristic_terms(segments: list[SegmentRecord]) -> list[str]:
    """Extract candidate technical terms using regex heuristics, without Ollama."""
    term_counts: dict[str, int] = {}
    uncertain_proper: list[str] = []

    for seg in segments:
        text = seg.raw_text
        for pattern in (_ACRONYM_PATTERN, _CAMELCASE_PATTERN, _TECH_TOKEN_PATTERN):
            for match in pattern.finditer(text):
                word = match.group(0)
                term_counts[word] = term_counts.get(word, 0) + 1

        for word_rec in seg.words:
            token = word_rec.text.strip()
            if token and word_rec.probability < 0.6 and token[0].isupper():
                uncertain_proper.append(token)

    frequent = [t for t, c in term_counts.items() if c >= 2]
    return list(dict.fromkeys(frequent + uncertain_proper))[:_MAX_HEURISTIC_TERMS]


def infer_context_and_glossary(
    segments: list[SegmentRecord],
    user_context: str,
    user_glossary: list[str],
    ollama_url: str,
    ollama_model: str,
    ollama_timeout: int,
    no_hotword_inference: bool,
) -> tuple[str, list[str]]:
    """Derive context summary and glossary from pass-1 segments.

    Always runs heuristic extraction. Unless no_hotword_inference is True, also
    calls Ollama with keep_alive: 0 (immediate model unload after the response)
    so VRAM is free before pass-2 Whisper reloads.
    """
    heuristic_terms = _heuristic_terms(segments)

    if no_hotword_inference:
        merged_glossary = list(dict.fromkeys(user_glossary + heuristic_terms))
        return user_context, merged_glossary[:_MAX_GLOSSARY_TERMS]

    raw_text = " ".join(seg.raw_text for seg in segments)[:_MAX_INFERENCE_CHARS]
    candidate_preview = ", ".join(heuristic_terms[:40]) or "none"

    prompt = (
        "You are analysing a raw automatic speech-recognition transcript to extract "
        "metadata for a second, higher-quality transcription pass.\n\n"
        "Task: return two fields:\n"
        "1. context_summary — one or two sentences describing the subject, domain, "
        "speakers, or setting inferred from the transcript.\n"
        "2. glossary — a list of proper nouns, technical terms, acronyms, product "
        "names, or domain vocabulary found or strongly implied. Include likely ASR "
        "misspellings as separate entries only if the intended word is inferable. "
        "Limit to 60 items. Do not include common English words.\n\n"
        "Rules:\n"
        "- Do not correct or rewrite the transcript text.\n"
        "- Return JSON matching the required schema exactly.\n\n"
        f"Heuristically detected candidate terms (may contain noise):\n{candidate_preview}\n\n"
        f"Raw transcript (first {_MAX_INFERENCE_CHARS} characters):\n{raw_text}"
    )

    payload: dict[str, Any] = {
        "model": ollama_model,
        "messages": [
            {
                "role": "system",
                "content": "Extract transcript metadata. Output only schema-valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": INFERENCE_SCHEMA,
        "think": False,
        "options": {"temperature": 0, "top_p": 0.1, "seed": 0},
        "keep_alive": "0",
    }

    inferred_context = ""
    inferred_glossary: list[str] = []

    try:
        data = post_json(
            f"{ollama_url.rstrip('/')}/api/chat", payload, timeout=ollama_timeout
        )
        content = strip_think_tags(data.get("message", {}).get("content", ""))
        parsed = json.loads(content)
        inferred_context = normalise_whitespace(str(parsed.get("context_summary", "")))
        inferred_glossary = [
            normalise_whitespace(str(t))
            for t in parsed.get("glossary", [])
            if normalise_whitespace(str(t))
        ][:60]
        print(
            f"  Context: {inferred_context[:100]!r} | "
            f"{len(inferred_glossary)} glossary terms inferred."
        )
    except Exception as exc:
        eprint(f"Hotword inference via Ollama failed ({exc}); using heuristics only.")
        inferred_glossary = heuristic_terms[:40]

    merged_context = normalise_whitespace(
        " ".join(p for p in [user_context, inferred_context] if p)
    )
    merged_glossary = list(
        dict.fromkeys(user_glossary + inferred_glossary + heuristic_terms)
    )
    return merged_context, merged_glossary[:_MAX_GLOSSARY_TERMS]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def paragraph_text(segments: Sequence[SegmentRecord], corrected: bool) -> str:
    if not segments:
        return ""
    pieces: list[str] = []
    previous_end: float | None = None
    for segment in segments:
        text = segment.corrected_text if corrected else segment.raw_text
        if previous_end is not None:
            pieces.append("\n\n" if segment.start - previous_end >= 2.5 else " ")
        pieces.append(text.strip())
        previous_end = segment.end
    return "".join(pieces).strip() + "\n"


def timestamped_text(segments: Sequence[SegmentRecord]) -> str:
    return "\n".join(
        f"[{format_clock(segment.start)} --> {format_clock(segment.end)}] "
        f"{segment.corrected_text}"
        for segment in segments
    ) + ("\n" if segments else "")


def subtitle_text(text: str, width: int) -> str:
    lines = textwrap.wrap(
        normalise_whitespace(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


def srt_text(segments: Sequence[SegmentRecord], width: int) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_clock(segment.start, ',')} --> "
            f"{format_clock(segment.end, ',')}\n"
            f"{subtitle_text(segment.corrected_text, width)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def vtt_text(segments: Sequence[SegmentRecord], width: int) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            f"{format_clock(segment.start)} --> {format_clock(segment.end)}\n"
            f"{subtitle_text(segment.corrected_text, width)}"
        )
    return "\n\n".join(blocks) + "\n"


def review_report(segments: Sequence[SegmentRecord]) -> str:
    flagged = [segment for segment in segments if segment.review_reasons]
    if not flagged:
        return "No segments were automatically flagged for review.\n"

    lines = [
        "Segments needing human review",
        "=============================",
        "",
        "Quality scores are heuristics, not calibrated probabilities.",
        "Listen to these regions before treating exact wording as authoritative.",
        "",
    ]
    for segment in flagged:
        lines.extend(
            [
                f"[{format_clock(segment.start)} --> {format_clock(segment.end)}] "
                f"segment {segment.id} | quality={segment.quality_score:.3f}",
                f"Raw:       {segment.raw_text}",
                f"Corrected: {segment.corrected_text}",
                "Reasons:    " + "; ".join(segment.review_reasons),
                (
                    "Uncertain:  " + ", ".join(segment.uncertain_terms)
                    if segment.uncertain_terms
                    else "Uncertain:  none explicitly listed"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_dir: Path,
    source: Path,
    segments: list[SegmentRecord],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    context: str,
    glossary: list[str],
    asr_device: str,
    asr_compute_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcript.txt").write_text(
        paragraph_text(segments, corrected=True), encoding="utf-8"
    )
    (output_dir / "transcript_raw.txt").write_text(
        paragraph_text(segments, corrected=False), encoding="utf-8"
    )
    (output_dir / "transcript_timestamped.txt").write_text(
        timestamped_text(segments), encoding="utf-8"
    )
    (output_dir / "transcript.srt").write_text(
        srt_text(segments, args.subtitle_width), encoding="utf-8"
    )
    (output_dir / "transcript.vtt").write_text(
        vtt_text(segments, args.subtitle_width), encoding="utf-8"
    )
    (output_dir / "review_needed.txt").write_text(
        review_report(segments), encoding="utf-8"
    )

    document: dict[str, Any] = {
        "source": str(source.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "asr": "faster-whisper",
            "asr_model": args.model,
            "asr_device": asr_device,
            "asr_compute_type": asr_compute_type,
            "two_pass": not args.single_pass,
            "vad": "Silero VAD through faster-whisper",
            "ollama_enabled": not args.no_ollama,
            "ollama_model": None if args.no_ollama else args.ollama_model,
            "correction_guardrails": {
                "preserve_numbers": True,
                "preserve_negations": True,
                "reject_large_rewrites": True,
            },
        },
        "audio": metadata,
        "context": context,
        "glossary": glossary,
        "segments": [asdict(segment) for segment in segments],
    }
    (output_dir / "transcript.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "--auto-install",
        action="store_true",
        default=True,
        help="Install faster-whisper automatically if missing (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-install",
        dest="auto_install",
        action="store_false",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
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
            " ".join(
                p for p in [args.context.strip(), read_optional_text(args.context_file)]
                if p
            )
        )
        user_glossary = parse_glossary(args.glossary, args.hotwords)

        device, compute_type = resolve_device_and_compute(
            args.device, args.compute_type, _DETECTED_GPU_INFO
        )
        gpu_desc = (
            f"compute_cap={_DETECTED_GPU_INFO.compute_cap}"
            if _DETECTED_GPU_INFO
            else "no GPU detected"
        )
        print(f"Device: {device} ({compute_type}) | {gpu_desc}")

        with tempfile.TemporaryDirectory(prefix="local-transcriber-") as temp_dir:
            if args.skip_ffmpeg_conversion:
                asr_input = source
            else:
                asr_input = Path(temp_dir) / "decoded_16khz_mono.wav"
                print("Decoding audio with FFmpeg...")
                run_ffmpeg(source, asr_input, args.normalise_audio)

            print("Pass 1: transcribing without domain-specific hotwords...")
            segments_pass1, metadata = transcribe_audio(
                asr_input, args, user_context, user_glossary, device, compute_type
            )

            if not segments_pass1:
                raise RuntimeError(
                    "No speech detected in pass 1. "
                    "Try --vad-threshold 0.3, set --language explicitly, "
                    "or inspect the audio track."
                )

            asr_device = str(metadata.get("asr_device", device))
            asr_compute_type = str(metadata.get("asr_compute_type", compute_type))
            print(
                f"Pass 1 complete: {len(segments_pass1)} segments | "
                f"{metadata['detected_language']} "
                f"(p={metadata['language_probability']:.3f})"
            )

            if args.single_pass:
                segments = segments_pass1
                final_context = user_context
                final_glossary = user_glossary
            else:
                print("Inferring context and hotwords from pass-1 transcript...")
                use_ollama_for_inference = not args.no_ollama and not args.no_hotword_inference
                final_context, final_glossary = infer_context_and_glossary(
                    segments_pass1,
                    user_context,
                    user_glossary,
                    args.ollama_url,
                    args.ollama_model,
                    args.ollama_timeout,
                    no_hotword_inference=not use_ollama_for_inference,
                )

                new_terms = [t for t in final_glossary if t not in user_glossary]
                context_changed = final_context != user_context
                if new_terms or context_changed:
                    print(
                        f"Pass 2: re-transcribing with {len(final_glossary)} hotwords "
                        f"({len(new_terms)} newly inferred)..."
                    )
                    segments, metadata = transcribe_audio(
                        asr_input, args, final_context, final_glossary,
                        device, compute_type,
                    )
                    asr_device = str(metadata.get("asr_device", device))
                    asr_compute_type = str(metadata.get("asr_compute_type", compute_type))
                else:
                    print("No new terms inferred; skipping pass 2.")
                    segments = segments_pass1

        if not segments:
            raise RuntimeError(
                "No speech was transcribed. Try a lower --vad-threshold, set "
                "--language explicitly, or inspect the audio track."
            )

        print(f"Whisper produced {len(segments)} segments.")

        if not args.no_ollama:
            gc.collect()
            try:
                correct_with_ollama(segments, args, final_context, final_glossary)
            except Exception as exc:
                eprint(
                    f"Warning: Ollama correction was skipped: {exc}. "
                    "The raw Whisper transcript will still be written."
                )
                for segment in segments:
                    segment.review_reasons.append("Ollama correction was unavailable")
                    if not segment.correction_note:
                        segment.correction_note = f"Raw text retained: {exc}"

        write_outputs(
            output_dir,
            source,
            segments,
            metadata,
            args,
            final_context,
            final_glossary,
            asr_device,
            asr_compute_type,
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
