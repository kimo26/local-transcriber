"use client";
import { useEffect, useRef, useState } from "react";
import { Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export interface JobResult {
  job_id: string;
  status: string;
  metadata: Record<string, unknown>;
  asr_device: string;
  asr_compute_type: string;
  context: string;
  glossary: string[];
  segments: Segment[];
  summary: {
    total_segments: number;
    corrections_applied: number;
    flagged_for_review: number;
  };
}

export interface Segment {
  id: number;
  start: number;
  end: number;
  raw_text: string;
  corrected_text: string;
  quality_score: number;
  review_reasons: string[];
  correction_applied: boolean;
}

type JobState = "queued" | "running" | "done" | "failed";

interface Props {
  jobId: string;
  apiUrl: string;
  onResult: (result: JobResult) => void;
}

export function JobStatus({ jobId, apiUrl, onResult }: Props) {
  const [state, setState] = useState<JobState>("queued");
  const [messages, setMessages] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const startRef = useRef<number>(Date.now());
  const listRef = useRef<HTMLUListElement>(null);

  // Tick the elapsed timer every second.
  useEffect(() => {
    const id = setInterval(() => setElapsed(Date.now() - startRef.current), 1000);
    return () => clearInterval(id);
  }, []);

  // Scroll message list to bottom whenever messages change.
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const es = new EventSource(`${apiUrl}/api/jobs/${jobId}`);

    es.onmessage = (e) => {
      const data = JSON.parse(e.data) as Record<string, unknown>;

      if (data.type === "status") {
        setState(data.status as JobState);
      } else if (data.type === "progress" && typeof data.message === "string") {
        setState("running");
        setMessages((prev) => [...prev, data.message as string]);
      } else if (data.type === "result") {
        setState("done");
        es.close();
        onResult(data as unknown as JobResult);
      } else if (data.type === "error") {
        setState("failed");
        setError(typeof data.detail === "string" ? data.detail : "Unknown error");
        es.close();
      }
    };

    es.onerror = () => {
      if (state !== "done" && state !== "failed") {
        setError("Connection to server lost.");
        setState("failed");
      }
      es.close();
    };

    return () => es.close();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, apiUrl]);

  const formatElapsed = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
  };

  const progressPercent =
    state === "done" ? 100 : state === "running" ? undefined : 0;

  return (
    <div className="rounded-xl border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {state === "running" || state === "queued" ? (
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
          ) : state === "done" ? (
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          ) : (
            <XCircle className="h-4 w-4 text-destructive" />
          )}
          <span className="text-sm font-medium capitalize">{state}</span>
        </div>
        <span className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          {formatElapsed(elapsed)}
        </span>
      </div>

      {/* Indeterminate or determinate progress bar */}
      {state !== "failed" && (
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-primary/20">
          {progressPercent === undefined ? (
            <div className="absolute inset-y-0 w-1/3 bg-primary rounded-full animate-[slide_1.5s_ease-in-out_infinite]" />
          ) : (
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          )}
        </div>
      )}

      {/* Progress log */}
      {messages.length > 0 && (
        <ul
          ref={listRef}
          className="max-h-32 overflow-y-auto space-y-0.5 text-xs text-muted-foreground font-mono"
        >
          {messages.map((msg, i) => (
            <li key={i} className={cn(i === messages.length - 1 && "text-foreground")}>
              {msg}
            </li>
          ))}
        </ul>
      )}

      {error && (
        <p className="text-xs text-destructive font-mono whitespace-pre-wrap">{error}</p>
      )}
    </div>
  );
}
