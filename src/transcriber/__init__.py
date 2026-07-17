# SPDX-License-Identifier: MIT
"""Local GPU-accelerated audio transcription with Whisper and LLM correction."""

from transcriber.models import (
    GpuInfo,
    OllamaConfig,
    SegmentRecord,
    TranscribeConfig,
    WordRecord,
)

__all__ = [
    "GpuInfo",
    "OllamaConfig",
    "SegmentRecord",
    "TranscribeConfig",
    "WordRecord",
]
