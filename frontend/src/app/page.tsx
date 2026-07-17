"use client";
import { useState, useEffect, useCallback } from "react";
import { Mic2, GitBranch, Cpu } from "lucide-react";
import { DropZone } from "@/components/DropZone";
import { SettingsPanel, DEFAULT_SETTINGS, type TranscribeSettings } from "@/components/SettingsPanel";
import { JobStatus, type JobResult } from "@/components/JobStatus";
import { TranscriptView } from "@/components/TranscriptView";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

interface HealthData {
  status: string;
  gpu: { compute_cap: number; cuda_major: number } | null;
  device: string;
  compute_type: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [settings, setSettings] = useState<TranscribeSettings>(DEFAULT_SETTINGS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);

  // Poll health on mount.
  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((r) => r.json())
      .then((d: HealthData) => setHealth(d))
      .catch(() => setHealth(null));
  }, []);

  const reset = useCallback(() => {
    setJobId(null);
    setResult(null);
    setError(null);
    setSubmitting(false);
    setFile(null);
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setResult(null);
    setJobId(null);

    const form = new FormData();
    form.append("file", file);
    form.append("model", settings.model);
    form.append("language", settings.language);
    form.append("single_pass", String(settings.singlePass));
    form.append("no_hotword_inference", String(settings.noHotwordInference));
    form.append("vad_threshold", String(settings.vadThreshold));
    form.append("normalise_audio", String(settings.normaliseAudio));
    form.append("ollama_model", settings.ollamaModel);
    form.append("ollama_url", settings.ollamaUrl);
    form.append("no_ollama", String(settings.noOllama));
    form.append("context", settings.context);
    form.append("hotwords", settings.hotwords);

    try {
      const res = await fetch(`${API_URL}/api/jobs`, { method: "POST", body: form });
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      }
      const data = await res.json() as { job_id: string };
      setJobId(data.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  const busy = submitting || (jobId !== null && result === null && error === null);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-sm">
        <div className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic2 className="h-5 w-5 text-primary" />
            <span className="font-semibold tracking-tight">Local Transcriber</span>
          </div>
          <div className="flex items-center gap-2">
            {health && (
              <Badge variant="outline" className="hidden sm:flex items-center gap-1 text-xs">
                <Cpu className="h-3 w-3" />
                {health.gpu
                  ? `GPU sm_${health.gpu.compute_cap.toString().replace(".", "")}`
                  : "CPU"}
              </Badge>
            )}
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md p-1.5 hover:bg-muted"
              aria-label="GitHub"
            >
              <GitBranch className="h-4 w-4" />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 space-y-5">
        {/* Upload area */}
        <section className="space-y-3">
          <DropZone onFile={setFile} disabled={busy} />
          <SettingsPanel settings={settings} onChange={setSettings} disabled={busy} />

          {error && !busy && (
            <p className="rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <div className="flex gap-2">
            <Button
              className="flex-1"
              onClick={handleSubmit}
              disabled={!file || busy}
            >
              {busy ? "Transcribing…" : "Transcribe"}
            </Button>
            {(jobId || result) && (
              <Button variant="outline" onClick={reset} disabled={submitting}>
                New
              </Button>
            )}
          </div>
        </section>

        {/* Job progress */}
        {jobId && !result && (
          <section>
            <JobStatus
              jobId={jobId}
              apiUrl={API_URL}
              onResult={(r) => {
                setResult(r);
                setSubmitting(false);
              }}
            />
          </section>
        )}

        {/* Transcript result */}
        {result && (
          <section>
            <TranscriptView result={result} />
          </section>
        )}
      </main>

      <footer className="border-t py-4 text-center text-xs text-muted-foreground">
        All processing is local — no audio leaves your machine.
      </footer>
    </div>
  );
}
