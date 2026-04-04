"use client";
import { useState } from "react";
import PlatformSelector from "./PlatformSelector";

interface Props {
  onScore: (message: string, persona: string, platform: string) => void;
  loading: boolean;
}

export default function ScoreForm({ onScore, loading }: Props) {
  const [message, setMessage] = useState("");
  const [persona, setPersona] = useState("");
  const [platform, setPlatform] = useState("general");
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (message.trim().length < 10) {
      setValidationError("Pitch message must be at least 10 characters");
      return;
    }
    if (persona.trim().length < 5) {
      setValidationError("Persona must be at least 5 characters");
      return;
    }
    setValidationError(null);
    onScore(message.trim(), persona.trim(), platform);
  }

  return (
    <form onSubmit={handleSubmit} className="panel p-6 space-y-5">
      <div className="space-y-1 pb-2 border-b border-[var(--color-line)]">
        <h2 className="text-base font-bold text-[var(--color-ink)]">Pitch Analyzer</h2>
        <p className="text-xs text-[var(--color-muted)]">
          Describe your target audience and paste your pitch. TRIBE neural analysis + AI interpretation will score your message for persuasion effectiveness.
        </p>
      </div>
      <div>
        <label className="block text-sm font-semibold text-[var(--color-ink)] mb-1.5">
          Who are you pitching to?
        </label>
        <textarea
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          placeholder="CTO, 40 years old, startup background, technical but pragmatic, values efficiency..."
          className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2.5 text-sm placeholder:text-[var(--color-faint)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)] min-h-[80px] resize-y"
          rows={3}
        />
        <p className="mt-1 text-xs text-[var(--color-faint)]">{persona.length} characters</p>
      </div>

      <div>
        <label className="block text-sm font-semibold text-[var(--color-ink)] mb-1.5">
          Your pitch
        </label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Paste your sales message, cold email, pitch deck text, ad copy..."
          className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2.5 text-sm placeholder:text-[var(--color-faint)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)] min-h-[160px] resize-y"
          rows={6}
        />
        <p className="mt-1 text-xs text-[var(--color-faint)]">{message.length} characters</p>
      </div>

      <PlatformSelector value={platform} onChange={setPlatform} />

      {validationError && (
        <p className="text-sm text-red-600" role="alert">{validationError}</p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-lg bg-[var(--color-pitch)] px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-pitch-light)] disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Scoring...
          </span>
        ) : (
          "Score My Pitch"
        )}
      </button>
    </form>
  );
}
