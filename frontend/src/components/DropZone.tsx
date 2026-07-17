"use client";
import { useCallback, useState } from "react";
import { Upload, FileAudio, X } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPTED_TYPES = [
  "audio/*", "video/mp4", "video/webm", "video/quicktime",
  "video/x-matroska", "video/x-msvideo",
].join(",");

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function DropZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      setSelected(file);
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelected(null);
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  };

  return (
    <label
      className={cn(
        "relative flex flex-col items-center justify-center w-full min-h-[200px] rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200",
        dragging
          ? "border-primary bg-primary/5 scale-[1.01]"
          : "border-border hover:border-primary/60 hover:bg-muted/50",
        disabled && "opacity-50 pointer-events-none",
        selected && "border-primary/40 bg-muted/30"
      )}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        accept={ACCEPTED_TYPES}
        className="sr-only"
        onChange={onInputChange}
        disabled={disabled}
      />

      {selected ? (
        <div className="flex items-center gap-3 px-6 py-4">
          <FileAudio className="h-8 w-8 text-primary shrink-0" />
          <div className="min-w-0">
            <p className="font-medium text-sm truncate">{selected.name}</p>
            <p className="text-xs text-muted-foreground">{formatBytes(selected.size)}</p>
          </div>
          <button
            type="button"
            onClick={clear}
            className="ml-2 rounded-full p-1 hover:bg-muted"
            aria-label="Remove file"
          >
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 px-6 py-8 text-center">
          <div className="rounded-full bg-muted p-4">
            <Upload className="h-6 w-6 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium">Drop an audio or video file here</p>
            <p className="mt-1 text-xs text-muted-foreground">
              MP3, WAV, FLAC, M4A, OGG, OPUS, MP4, MKV, MOV and more
            </p>
          </div>
          <p className="text-xs text-muted-foreground">or click to browse</p>
        </div>
      )}
    </label>
  );
}
