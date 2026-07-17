# SPDX-License-Identifier: MIT
"""Tests for the FastAPI endpoints using httpx TestClient.

The pipeline itself is mocked so no GPU or Ollama is required.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from transcriber.api import app
from transcriber.models import SegmentRecord


def _make_segment(seg_id: int = 1) -> SegmentRecord:
    return SegmentRecord(
        id=seg_id,
        start=0.0,
        end=1.0,
        raw_text="hello",
        corrected_text="hello",
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        compression_ratio=1.0,
        temperature=0.0,
        quality_score=0.9,
    )


_MOCK_PIPELINE_RESULT = (
    [_make_segment()],
    {
        "detected_language": "en",
        "language_probability": 0.99,
        "duration_seconds": 1.0,
        "duration_after_vad_seconds": 0.9,
        "vad_removed_seconds": 0.1,
        "vad_parameters": {},
        "asr_device": "cpu",
        "asr_compute_type": "int8",
    },
    "cpu",
    "int8",
    "",
    [],
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data

    def test_has_gpu_field(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert "gpu" in response.json()


class TestCreateJob:
    def test_accepts_audio_upload(self, client: TestClient) -> None:
        fake_audio = io.BytesIO(b"\x00" * 100)
        response = client.post(
            "/api/jobs",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
        )
        assert response.status_code == 202
        assert "job_id" in response.json()

    def test_returns_uuid_job_id(self, client: TestClient) -> None:
        fake_audio = io.BytesIO(b"\x00" * 100)
        response = client.post(
            "/api/jobs",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
        )
        job_id = response.json()["job_id"]
        assert len(job_id) == 36  # UUID4 format


class TestStreamJob:
    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/jobs/does-not-exist")
        assert response.status_code == 404

    def test_known_job_returns_sse_content_type(self, client: TestClient) -> None:
        fake_audio = io.BytesIO(b"\x00" * 100)
        create_resp = client.post(
            "/api/jobs",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
        )
        job_id = create_resp.json()["job_id"]
        stream_resp = client.get(f"/api/jobs/{job_id}")
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")
