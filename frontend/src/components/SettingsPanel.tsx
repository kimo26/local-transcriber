"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, Settings2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TranscribeSettings {
  model: string;
  language: string;
  singlePass: boolean;
  noHotwordInference: boolean;
  vadThreshold: number;
  normaliseAudio: boolean;
  ollamaModel: string;
  ollamaUrl: string;
  noOllama: boolean;
  context: string;
  hotwords: string;
}

export const DEFAULT_SETTINGS: TranscribeSettings = {
  model: "large-v3",
  language: "auto",
  singlePass: false,
  noHotwordInference: false,
  vadThreshold: 0.45,
  normaliseAudio: false,
  ollamaModel: "qwen3:30b-a3b",
  ollamaUrl: "http://localhost:11434",
  noOllama: false,
  context: "",
  hotwords: "",
};

const MODELS = [
  "large-v3",
  "large-v3-turbo",
  "large-v2",
  "medium",
  "small",
  "base",
];

const LANGUAGES = [
  { value: "auto", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "es", label: "Spanish" },
  { value: "pt", label: "Portuguese" },
  { value: "it", label: "Italian" },
  { value: "nl", label: "Dutch" },
  { value: "pl", label: "Polish" },
  { value: "ru", label: "Russian" },
  { value: "zh", label: "Chinese" },
  { value: "ja", label: "Japanese" },
  { value: "ar", label: "Arabic" },
];

interface Props {
  settings: TranscribeSettings;
  onChange: (s: TranscribeSettings) => void;
  disabled?: boolean;
}

export function SettingsPanel({ settings, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);

  const set = (patch: Partial<TranscribeSettings>) =>
    onChange({ ...settings, ...patch });

  return (
    <div className="rounded-xl border bg-card">
      <button
        type="button"
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
        onClick={() => setOpen(!open)}
        disabled={disabled}
      >
        <span className="flex items-center gap-2">
          <Settings2 className="h-4 w-4 text-muted-foreground" />
          Settings
        </span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className={cn("border-t px-4 pb-4 pt-3 grid gap-4", disabled && "opacity-50 pointer-events-none")}>
          <div className="grid grid-cols-2 gap-3">
            {/* Model */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Whisper model
              </label>
              <select
                className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                value={settings.model}
                onChange={(e) => set({ model: e.target.value })}
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            {/* Language */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Language
              </label>
              <select
                className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                value={settings.language}
                onChange={(e) => set({ language: e.target.value })}
              >
                {LANGUAGES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* VAD threshold */}
          <div>
            <label className="flex justify-between text-xs font-medium text-muted-foreground mb-1">
              <span>VAD threshold</span>
              <span>{settings.vadThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={settings.vadThreshold}
              onChange={(e) => set({ vadThreshold: parseFloat(e.target.value) })}
              className="w-full accent-primary"
            />
          </div>

          {/* Context hint */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1">
              Context hint (optional)
            </label>
            <input
              type="text"
              placeholder="Subject, domain, speakers…"
              value={settings.context}
              onChange={(e) => set({ context: e.target.value })}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Hotwords */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1">
              Hotwords (comma-separated, optional)
            </label>
            <input
              type="text"
              placeholder="React, TypeScript, useEffect…"
              value={settings.hotwords}
              onChange={(e) => set({ hotwords: e.target.value })}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Toggles */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            {[
              { key: "singlePass" as const, label: "Single pass" },
              { key: "noHotwordInference" as const, label: "Skip hotword inference" },
              { key: "normaliseAudio" as const, label: "Normalise audio" },
              { key: "noOllama" as const, label: "Skip Ollama correction" },
            ].map(({ key, label }) => (
              <label key={key} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings[key] as boolean}
                  onChange={(e) => set({ [key]: e.target.checked })}
                  className="accent-primary"
                />
                {label}
              </label>
            ))}
          </div>

          {/* Ollama settings */}
          {!settings.noOllama && (
            <div className="grid grid-cols-2 gap-3 border-t pt-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Ollama model
                </label>
                <input
                  type="text"
                  value={settings.ollamaModel}
                  onChange={(e) => set({ ollamaModel: e.target.value })}
                  className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Ollama URL
                </label>
                <input
                  type="text"
                  value={settings.ollamaUrl}
                  onChange={(e) => set({ ollamaUrl: e.target.value })}
                  className="w-full rounded-md border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
