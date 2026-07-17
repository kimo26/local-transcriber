# SPDX-License-Identifier: MIT
"""Ollama HTTP client, batch construction, and the LLM correction pass.

All network calls use the stdlib ``urllib`` so no extra dependencies are needed
beyond the project's declared requirements.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, cast

from transcriber.guardrails import validate_correction
from transcriber.models import CORRECTION_SCHEMA, OllamaConfig, SegmentRecord
from transcriber.utils import eprint, normalise_whitespace, strip_think_tags

# HTTP primitives


def get_json(url: str, timeout: int = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


# Ollama availability check


def verify_ollama(ollama_url: str, model: str) -> None:
    try:
        data = get_json(f"{ollama_url.rstrip('/')}/api/tags")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start it with `ollama serve`, or use --no-ollama."
        ) from exc

    names = {item.get("name", "") for item in data.get("models", []) if isinstance(item, dict)}
    short_names = {name.split(":", 1)[0] for name in names}
    if model not in names and model.split(":", 1)[0] not in short_names:
        available = ", ".join(sorted(names)) or "none"
        raise RuntimeError(
            f"Ollama model {model!r} is not installed. Run `ollama pull {model}`. "
            f"Installed models: {available}"
        )


# Batch construction


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


# LLM correction pass


def correct_with_ollama(
    segments: list[SegmentRecord],
    cfg: OllamaConfig,
    context: str,
    glossary: list[str],
) -> None:
    """Apply Ollama correction to every segment in-place."""
    ollama_url = cfg.url.rstrip("/")
    verify_ollama(ollama_url, cfg.model)

    batches = make_batches(
        segments,
        max_segments=cfg.batch_segments,
        max_characters=cfg.batch_characters,
    )
    print(f"Correcting {len(segments)} segments with {cfg.model!r} in {len(batches)} batches...")

    for batch_number, (start, end) in enumerate(batches, start=1):
        batch = segments[start:end]
        before = segments[max(0, start - 2) : start]
        after = segments[end : min(len(segments), end + 2)]
        prompt = correction_prompt(batch, before, after, context, glossary)

        payload: dict[str, Any] = {
            "model": cfg.model,
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
                "num_ctx": cfg.num_ctx,
            },
            "keep_alive": cfg.keep_alive,
        }

        response_data: dict[str, Any] | None = None
        error: Exception | None = None
        for attempt in range(1, cfg.retries + 1):
            try:
                response_data = post_json(
                    f"{ollama_url}/api/chat",
                    payload,
                    timeout=cfg.timeout,
                )
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < cfg.retries:
                    time.sleep(min(2**attempt, 8))

        if response_data is None:
            for segment in batch:
                segment.review_reasons.append("Ollama correction request failed")
                segment.correction_note = f"Ollama failure: {error}"
            eprint(f"Batch {batch_number}/{len(batches)} failed; raw text retained.")
            continue

        try:
            message_content = strip_think_tags(response_data["message"]["content"])
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
            eprint(f"Batch {batch_number}/{len(batches)} returned invalid JSON; raw text retained.")
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
                aggressive=cfg.allow_aggressive_corrections,
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
                segment.review_reasons.append(f"rejected unsafe Ollama edit: {validation_note}")
                segment.correction_note = (
                    f"Raw text retained. Proposed edit rejected: {validation_note}."
                )

            if segment.uncertain_terms:
                segment.review_reasons.append(
                    "Ollama marked uncertain: " + ", ".join(segment.uncertain_terms)
                )

        print(f"Corrected batch {batch_number}/{len(batches)}")
