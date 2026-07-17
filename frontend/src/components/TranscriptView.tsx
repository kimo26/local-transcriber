"use client";
import { useState } from "react";
import { Copy, Download, Check } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { type JobResult, type Segment } from "@/components/JobStatus";
import { cn } from "@/lib/utils";

interface Props {
  result: JobResult;
}

function formatClock(seconds: number) {
  const ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1_000);
  const msLeft = ms % 1_000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(msLeft).padStart(3, "0")}`;
}

function buildCleanText(segments: Segment[]) {
  if (!segments.length) return "";
  const parts: string[] = [];
  let prevEnd: number | null = null;
  for (const seg of segments) {
    if (prevEnd !== null) {
      parts.push(seg.start - prevEnd >= 2.5 ? "\n\n" : " ");
    }
    parts.push(seg.corrected_text.trim());
    prevEnd = seg.end;
  }
  return parts.join("");
}

function buildRawText(segments: Segment[]) {
  return segments.map((s) => s.raw_text).join(" ");
}

function buildTimestampedText(segments: Segment[]) {
  return segments
    .map((s) => `[${formatClock(s.start)} --> ${formatClock(s.end)}] ${s.corrected_text}`)
    .join("\n");
}

function buildSrtText(segments: Segment[]) {
  return segments
    .map((s, i) => {
      const toSrtTime = (sec: number) => formatClock(sec).replace(".", ",");
      return `${i + 1}\n${toSrtTime(s.start)} --> ${toSrtTime(s.end)}\n${s.corrected_text}`;
    })
    .join("\n\n");
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <Button variant="ghost" size="sm" onClick={copy} className="h-7 px-2 gap-1">
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function DownloadButton({ text, filename }: { text: string; filename: string }) {
  const download = () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <Button variant="ghost" size="sm" onClick={download} className="h-7 px-2 gap-1">
      <Download className="h-3.5 w-3.5" />
      Download
    </Button>
  );
}

function TextPane({ text, filename }: { text: string; filename: string }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-end gap-1">
        <CopyButton text={text} />
        <DownloadButton text={text} filename={filename} />
      </div>
      <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap font-sans text-sm leading-relaxed rounded-lg bg-muted/40 p-4">
        {text}
      </pre>
    </div>
  );
}

function SegmentRow({ seg }: { seg: Segment }) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3 text-sm space-y-1",
        seg.review_reasons.length > 0 && "border-amber-500/30 bg-amber-500/5"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-muted-foreground">
          {formatClock(seg.start)} → {formatClock(seg.end)}
        </span>
        <div className="flex items-center gap-1">
          {seg.correction_applied && (
            <Badge variant="secondary" className="text-[10px] py-0">edited</Badge>
          )}
          {seg.review_reasons.length > 0 && (
            <Badge variant="outline" className="text-[10px] py-0 border-amber-500/60 text-amber-600">
              review
            </Badge>
          )}
          <span className="text-xs text-muted-foreground">
            {(seg.quality_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <p>{seg.corrected_text}</p>
      {seg.review_reasons.length > 0 && (
        <p className="text-xs text-amber-600/80">{seg.review_reasons.join("; ")}</p>
      )}
    </div>
  );
}

export function TranscriptView({ result }: Props) {
  const clean = buildCleanText(result.segments);
  const raw = buildRawText(result.segments);
  const timestamped = buildTimestampedText(result.segments);
  const srt = buildSrtText(result.segments);

  const meta = result.metadata as Record<string, unknown>;

  return (
    <div className="space-y-4">
      {/* Summary chips */}
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline">{result.asr_device} · {result.asr_compute_type}</Badge>
        {typeof meta.detected_language === "string" && (
          <Badge variant="outline">
            lang: {meta.detected_language}
            {typeof meta.language_probability === "number" &&
              ` (${(meta.language_probability * 100).toFixed(0)}%)`}
          </Badge>
        )}
        {typeof meta.duration_seconds === "number" && (
          <Badge variant="outline">{Math.round(meta.duration_seconds)}s audio</Badge>
        )}
        <Badge variant="outline">{result.summary.total_segments} segments</Badge>
        {result.summary.corrections_applied > 0 && (
          <Badge variant="secondary">{result.summary.corrections_applied} corrections</Badge>
        )}
        {result.summary.flagged_for_review > 0 && (
          <Badge variant="outline" className="border-amber-500/60 text-amber-600">
            {result.summary.flagged_for_review} flagged
          </Badge>
        )}
      </div>

      <Tabs defaultValue="clean">
        <TabsList className="flex-wrap h-auto gap-1">
          <TabsTrigger value="clean">Clean</TabsTrigger>
          <TabsTrigger value="raw">Raw</TabsTrigger>
          <TabsTrigger value="timestamped">Timestamped</TabsTrigger>
          <TabsTrigger value="srt">SRT</TabsTrigger>
          <TabsTrigger value="segments">Segments</TabsTrigger>
        </TabsList>

        <TabsContent value="clean">
          <TextPane text={clean} filename="transcript.txt" />
        </TabsContent>

        <TabsContent value="raw">
          <TextPane text={raw} filename="transcript_raw.txt" />
        </TabsContent>

        <TabsContent value="timestamped">
          <TextPane text={timestamped} filename="transcript_timestamped.txt" />
        </TabsContent>

        <TabsContent value="srt">
          <TextPane text={srt} filename="transcript.srt" />
        </TabsContent>

        <TabsContent value="segments">
          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
            {result.segments.map((seg) => (
              <SegmentRow key={seg.id} seg={seg} />
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* JSON download */}
      <div className="flex justify-end">
        <DownloadButton
          text={JSON.stringify(result, null, 2)}
          filename="transcript.json"
        />
      </div>
    </div>
  );
}
