# SPDX-License-Identifier: MIT
"""Hotword and context inference from pass-1 transcript.

The two-pass strategy: after pass 1 produces a rough transcript, this module
extracts candidate technical terms via regex heuristics and optionally asks
Ollama (with keep_alive: 0 so VRAM is immediately freed) to refine them into
a context summary and glossary. Pass 2 feeds those back to Whisper as hotwords
and an initial_prompt, improving recognition of domain-specific vocabulary.
"""

from __future__ import annotations

import json
from typing import Any

from transcriber.models import (
    _ACRONYM_PATTERN,
    _CAMELCASE_PATTERN,
    _MAX_GLOSSARY_TERMS,
    _MAX_HEURISTIC_TERMS,
    _MAX_INFERENCE_CHARS,
    _TECH_TOKEN_PATTERN,
    INFERENCE_SCHEMA,
    SegmentRecord,
)
from transcriber.ollama import post_json
from transcriber.utils import eprint, normalise_whitespace, strip_think_tags


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

    # Require a term to appear at least twice to filter out noise.
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

    Always runs heuristic extraction. Unless ``no_hotword_inference`` is True,
    also calls Ollama with ``keep_alive: 0`` (immediate model unload) so VRAM is
    free before pass-2 Whisper reloads.
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
        data = post_json(f"{ollama_url.rstrip('/')}/api/chat", payload, timeout=ollama_timeout)
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
    merged_glossary = list(dict.fromkeys(user_glossary + inferred_glossary + heuristic_terms))
    return merged_context, merged_glossary[:_MAX_GLOSSARY_TERMS]
