# SPDX-License-Identifier: MIT
"""Output formatters and file-writing for the transcription pipeline.

All public functions are pure or near-pure (file I/O only) so they are trivial
to unit-test without any GPU or network dependencies.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transcriber.models import OllamaConfig, SegmentRecord, TranscribeConfig
from transcriber.utils import format_clock, normalise_whitespace


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
        f"[{format_clock(segment.start)} --> {format_clock(segment.end)}] {segment.corrected_text}"
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
    cfg: TranscribeConfig,
    ollama_cfg: OllamaConfig,
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
        srt_text(segments, cfg.subtitle_width), encoding="utf-8"
    )
    (output_dir / "transcript.vtt").write_text(
        vtt_text(segments, cfg.subtitle_width), encoding="utf-8"
    )
    (output_dir / "review_needed.txt").write_text(review_report(segments), encoding="utf-8")

    document: dict[str, Any] = {
        "source": str(source.resolve()),
        "created_utc": datetime.now(UTC).isoformat(),
        "pipeline": {
            "asr": "faster-whisper",
            "asr_model": cfg.model,
            "asr_device": asr_device,
            "asr_compute_type": asr_compute_type,
            "two_pass": not cfg.single_pass,
            "vad": "Silero VAD through faster-whisper",
            "ollama_enabled": ollama_cfg.enabled,
            "ollama_model": cfg.model if ollama_cfg.enabled else None,
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
