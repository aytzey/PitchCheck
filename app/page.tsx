"use client";
import { useState } from "react";
import ScoreForm from "@/components/ScoreForm";
import ScoreDisplay from "@/components/ScoreDisplay";
import type { PitchScoreReport } from "@/shared/types";

export default function Home() {
  const [report, setReport] = useState<PitchScoreReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleScore(message: string, persona: string, platform: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, persona, platform }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Scoring failed");
        return;
      }
      setReport(data.report);
    } catch {
      setError("Unable to reach scoring service");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="border-b border-[var(--color-line)] bg-[var(--color-surface)]">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--color-pitch)] text-white font-bold text-sm">PS</div>
            <div>
              <h1 className="text-lg font-bold text-[var(--color-ink)]">PitchScore</h1>
              <p className="text-xs text-[var(--color-muted)]">Neural Persuasion Intelligence</p>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="grid gap-8 lg:grid-cols-[1fr_1.2fr]">
          {/* Left: Input */}
          <div>
            <ScoreForm onScore={handleScore} loading={loading} />
            {error && (
              <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div>
            {report ? (
              <ScoreDisplay report={report} />
            ) : (
              <div className="panel flex min-h-[400px] items-center justify-center p-8 text-center">
                <div>
                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-[var(--color-pitch-faint)]">
                    <svg className="h-8 w-8 text-[var(--color-pitch)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                    </svg>
                  </div>
                  <p className="text-[var(--color-muted)]">Enter your pitch and target persona to get a persuasion score</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
