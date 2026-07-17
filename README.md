# local-transcriber

GPU-accelerated local audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Silero VAD, automatic two-pass hotword inference, and an optional LLM correction pass via [Ollama](https://ollama.com).

**All processing is local. No audio or text leaves your machine.**

## Features

- Adaptive GPU/CPU execution — auto-detects CUDA compute capability, falls back to CPU int8 transparently
- Two-pass ASR: pass 1 transcribes, then Ollama infers context and hotwords, pass 2 re-transcribes with those as hints
- LLM correction with hard guardrails: numbers, negations, and large rewrites are always rejected
- FastAPI backend with SSE progress streaming
- Next.js 15 frontend with drag-and-drop upload, live progress, tabbed transcript view (clean / raw / timestamped / SRT / segments), and one-click downloads
- Seven output files per transcription including `.srt`, `.vtt`, and full `.json` metadata

## Architecture

```mermaid
graph LR
    Browser -->|multipart upload| API["FastAPI :8000"]
    API -->|SSE progress| Browser
    API --> Pipeline
    Pipeline --> FFmpeg
    Pipeline --> Whisper["faster-whisper\n(GPU/CPU)"]
    Pipeline --> Ollama["Ollama\n(hotword inference + correction)"]
    Pipeline --> Outputs["transcript.txt / .srt / .vtt / .json"]
    subgraph "src/transcriber"
        Pipeline
        Whisper
        Ollama
        FFmpeg
    end
    subgraph "frontend/"
        Browser
    end
```

## Quick start

### Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org), Node 20+
- Optional: NVIDIA GPU with driver ≥ 525 (CUDA 12), [Ollama](https://ollama.com) with a model pulled

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/local-transcriber
cd local-transcriber
uv sync

# (Optional) Pull an Ollama correction model
ollama pull qwen3:30b-a3b

# Start the API server
uv run transcribe-server

# Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
# → open http://localhost:3000
```

### CLI usage

```bash
uv run transcribe audio.mp4
uv run transcribe audio.mp3 --model large-v3-turbo --language en
uv run transcribe lecture.wav --no-ollama --single-pass
uv run transcribe --help
```

Output lands in `<filename>_transcript/`:
| File | Contents |
|------|----------|
| `transcript.txt` | Clean corrected text |
| `transcript_raw.txt` | Original Whisper output, never modified |
| `transcript_timestamped.txt` | Corrected text with timestamps |
| `transcript.srt` | SRT subtitles |
| `transcript.vtt` | WebVTT subtitles |
| `transcript.json` | Full metadata + per-segment data |
| `review_needed.txt` | Low-confidence regions for human review |

## API reference

### `POST /api/jobs`

Upload an audio/video file to start a transcription job.

**Form fields** (all optional except `file`):

| Field | Default | Description |
|-------|---------|-------------|
| `file` | — | Audio/video file (required) |
| `model` | `large-v3` | Whisper model name |
| `language` | `auto` | ISO code or `auto` |
| `single_pass` | `false` | Skip hotword inference + pass 2 |
| `no_hotword_inference` | `false` | Heuristics only, no Ollama inference |
| `vad_threshold` | `0.45` | Silero VAD speech threshold |
| `normalise_audio` | `false` | FFmpeg loudnorm pre-processing |
| `ollama_model` | `qwen3:30b-a3b` | Ollama model for correction |
| `ollama_url` | `http://127.0.0.1:11434` | Ollama endpoint |
| `no_ollama` | `false` | Skip LLM correction entirely |
| `context` | `""` | Domain hint for ASR and correction |
| `hotwords` | `""` | Comma-separated vocabulary hints |

**Response:** `202 { "job_id": "uuid" }`

### `GET /api/jobs/{job_id}`

Server-Sent Events stream. Each event is a JSON object with `type`:

| `type` | Fields | Description |
|--------|--------|-------------|
| `status` | `status` | Initial state (`queued`/`running`) |
| `progress` | `message` | Human-readable progress line |
| `result` | all result fields | Final transcript data |
| `error` | `detail` | Error traceback if job failed |
| `ping` | — | Keep-alive every 30 s |

### `GET /api/health`

```json
{ "status": "ok", "gpu": { "compute_cap": 12.0, "cuda_major": 13 }, "device": "cuda", "compute_type": "float16" }
```

## Configuration

Copy `.env.example` to `.env.local` inside `frontend/` and adjust:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For remote access, expose the FastAPI server via a tunnel (e.g. `cloudflared tunnel`) and set `NEXT_PUBLIC_API_URL` on the Vercel dashboard.

## Development

```bash
# Run quality checks
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest --cov=src/transcriber

# Frontend
cd frontend && npm run build
```

## Contributing

1. Fork → feature branch → PR against `main`
2. All CI checks must pass (ruff, mypy, pytest, Next.js build)
3. Keep commit messages neutral and descriptive
