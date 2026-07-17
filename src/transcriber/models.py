# SPDX-License-Identifier: MIT
"""Shared dataclasses, constants, and JSON schemas for the transcription pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hardware descriptor
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    compute_cap: float  # e.g. 12.0 for Blackwell sm_120
    cuda_major: int  # inferred CUDA major version from driver


# ---------------------------------------------------------------------------
# Transcript data model
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
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class TranscribeConfig:
    """All parameters governing a single transcription run."""

    model: str = "large-v3"
    device: str = "auto"
    device_index: int = 0
    compute_type: str = "auto"
    cpu_threads: int = 0
    model_cache: Path | None = None
    local_files_only: bool = False
    language: str = "auto"
    detect_language_per_segment: bool = False
    language_detection_threshold: float = 0.5
    language_detection_segments: int = 3
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.2
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 0
    disable_previous_context: bool = False
    hallucination_silence_threshold: float = 1.5
    vad_threshold: float = 0.45
    vad_min_speech_ms: int = 120
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 300
    normalise_audio: bool = False
    skip_ffmpeg_conversion: bool = False
    single_pass: bool = False
    no_hotword_inference: bool = False
    subtitle_width: int = 48


@dataclass
class OllamaConfig:
    """Parameters for the Ollama LLM correction pass."""

    model: str = "qwen3:30b-a3b"
    url: str = "http://127.0.0.1:11434"
    num_ctx: int = 16384
    timeout: int = 900
    retries: int = 3
    keep_alive: str = "10m"
    batch_segments: int = 16
    batch_characters: int = 6500
    allow_aggressive_corrections: bool = False
    enabled: bool = True


# ---------------------------------------------------------------------------
# Regex constants (used across multiple modules)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Structured JSON schemas for Ollama constrained outputs
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Other shared constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {
    ".aac",
    ".ac3",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".caf",
    ".dts",
    ".flac",
    ".m4a",
    ".m4b",
    ".mka",
    ".mp2",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".ra",
    ".tta",
    ".wav",
    ".wma",
    ".wv",
    ".webm",
    ".3gp",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".mts",
    ".m2ts",
}

# Substrings that indicate a missing GPU kernel rather than a real failure.
# Catching these enables transparent CPU fallback instead of a crash.
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

# Minimum driver version (major) that implies a given CUDA major version on Linux.
_DRIVER_CUDA_MAP: list[tuple[int, int]] = [(525, 12), (450, 11), (418, 10)]

_MAX_GLOSSARY_TERMS = 80
_MAX_HEURISTIC_TERMS = 80
_MAX_INFERENCE_CHARS = 6_000
