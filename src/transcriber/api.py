# SPDX-License-Identifier: MIT
"""FastAPI server exposing the transcription pipeline over HTTP.

Three endpoints:
  POST /api/jobs          — upload audio, receive job_id
  GET  /api/jobs/{id}     — SSE stream of progress events + final JSON
  GET  /api/health        — GPU/model status

Jobs run in background asyncio tasks. Progress events are pushed through
per-job asyncio.Queue instances and consumed by the SSE endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import traceback
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from transcriber.asr import resolve_device_and_compute
from transcriber.bootstrap import detect_gpu_info
from transcriber.models import OllamaConfig, SegmentRecord, TranscribeConfig
from transcriber.pipeline import run_pipeline

# Job state machine


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status = JobStatus.queued
        self.progress_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.segments: list[SegmentRecord] = []


_jobs: dict[str, Job] = {}
_gpu_info = detect_gpu_info()

app = FastAPI(title="Local Transcriber API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background transcription task


async def _run_job(
    job: Job,
    audio_path: Path,
    cfg: TranscribeConfig,
    ollama_cfg: OllamaConfig,
    context: str,
    glossary: list[str],
) -> None:
    job.status = JobStatus.running

    def progress(msg: str) -> None:
        job.progress_queue.put_nowait(json.dumps({"type": "progress", "message": msg}))

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                source=audio_path,
                cfg=cfg,
                ollama_cfg=ollama_cfg,
                user_context=context,
                user_glossary=glossary,
                gpu_info=_gpu_info,
                progress=progress,
            ),
        )
        segments, metadata, asr_device, asr_compute_type, final_context, final_glossary = result

        job.segments = segments
        job.result = {
            "job_id": job.id,
            "status": "done",
            "metadata": metadata,
            "asr_device": asr_device,
            "asr_compute_type": asr_compute_type,
            "context": final_context,
            "glossary": final_glossary,
            "segments": [asdict(s) for s in segments],
            "summary": {
                "total_segments": len(segments),
                "corrections_applied": sum(s.correction_applied for s in segments),
                "flagged_for_review": sum(bool(s.review_reasons) for s in segments),
            },
        }
        job.status = JobStatus.done
    except Exception:
        job.error = traceback.format_exc()
        job.status = JobStatus.failed
    finally:
        audio_path.unlink(missing_ok=True)
        job.progress_queue.put_nowait(None)  # sentinel: SSE stream should close


# Endpoints


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form("auto"),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
    single_pass: bool = Form(False),
    no_hotword_inference: bool = Form(False),
    vad_threshold: float = Form(0.45),
    normalise_audio: bool = Form(False),
    ollama_model: str = Form("qwen3:30b-a3b"),
    ollama_url: str = Form("http://127.0.0.1:11434"),
    no_ollama: bool = Form(False),
    context: str = Form(""),
    hotwords: str = Form(""),
) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    job = Job(job_id)
    _jobs[job_id] = job

    suffix = Path(file.filename or "audio.tmp").suffix or ".tmp"
    tmp_file = Path(tempfile.mktemp(suffix=suffix, prefix="transcriber-"))
    with tmp_file.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    cfg = TranscribeConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        single_pass=single_pass,
        no_hotword_inference=no_hotword_inference,
        vad_threshold=vad_threshold,
        normalise_audio=normalise_audio,
    )
    ollama_cfg = OllamaConfig(
        model=ollama_model,
        url=ollama_url,
        enabled=not no_ollama,
    )
    glossary = [t.strip() for t in hotwords.split(",") if t.strip()]

    asyncio.create_task(_run_job(job, tmp_file, cfg, ollama_cfg, context, glossary))

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def stream_job(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'status', 'status': job.status})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(job.progress_queue.get(), timeout=30.0)
            except TimeoutError:
                yield 'data: {"type": "ping"}\n\n'
                continue

            if event is None:
                break

            yield f"data: {event}\n\n"

        if job.status == JobStatus.done and job.result is not None:
            yield f"data: {json.dumps({'type': 'result', **job.result})}\n\n"
        elif job.status == JobStatus.failed:
            yield (f"data: {json.dumps({'type': 'error', 'detail': job.error})}\n\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    device, compute_type = resolve_device_and_compute("auto", "auto", _gpu_info)
    return {
        "status": "ok",
        "gpu": (
            {
                "compute_cap": _gpu_info.compute_cap,
                "cuda_major": _gpu_info.cuda_major,
            }
            if _gpu_info
            else None
        ),
        "device": device,
        "compute_type": compute_type,
    }


# Server entry point


def serve() -> None:
    """Called by the ``transcribe-server`` script defined in pyproject.toml."""
    host = os.environ.get("TRANSCRIBER_HOST", "0.0.0.0")
    port = int(os.environ.get("TRANSCRIBER_PORT", "8000"))
    uvicorn.run("transcriber.api:app", host=host, port=port, reload=False)
