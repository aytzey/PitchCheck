import type { PitchScoreReport } from "@/shared/types";
import ScoreGauge from "./ScoreGauge";
import BreakdownBar from "./BreakdownBar";
import RewriteSuggestion from "./RewriteSuggestion";

export default function ScoreDisplay({ report }: { report: PitchScoreReport }) {
  return (
    <div className="space-y-6">
      {/* Score + Verdict */}
      <div className="panel p-6 text-center">
        <ScoreGauge score={report.persuasion_score} />
        <p className="mt-3 text-lg font-bold text-[var(--color-ink)]">{report.verdict}</p>
        <p className="mt-1 text-sm text-[var(--color-muted)]">{report.narrative}</p>
        {report.persona_summary && (
          <p className="mt-3 rounded-lg bg-[var(--color-pitch-faint)] px-3 py-2 text-xs text-[var(--color-pitch)]">
            Persona: {report.persona_summary}
          </p>
        )}
      </div>

      {/* Breakdown */}
      {report.breakdown.length > 0 && (
        <div className="panel p-6 space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)]">Persuasion Breakdown</h3>
          {report.breakdown.map((b) => (
            <BreakdownBar key={b.key} section={b} />
          ))}
        </div>
      )}

      {/* Neural Signals */}
      {report.neural_signals.length > 0 && (
        <div className="panel p-6">
          <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)] mb-3">Neural Signals</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {report.neural_signals.map((s) => (
              <div key={s.key} className="rounded-lg border border-[var(--color-line)] p-3 text-center">
                <p className="text-2xl font-bold" style={{ color: s.score >= 60 ? 'var(--color-success)' : s.score < 40 ? 'var(--color-danger)' : 'var(--color-warning)' }}>
                  {Math.round(s.score)}
                </p>
                <p className="mt-1 text-xs text-[var(--color-muted)]">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strengths & Risks */}
      <div className="grid gap-4 sm:grid-cols-2">
        {report.strengths.length > 0 && (
          <div className="panel p-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-success)] mb-2">Strengths</h3>
            <ul className="space-y-1.5">
              {report.strengths.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-ink)]">
                  <span className="text-[var(--color-success)] mt-0.5">+</span> {s}
                </li>
              ))}
            </ul>
          </div>
        )}
        {report.risks.length > 0 && (
          <div className="panel p-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-warning)] mb-2">Risks</h3>
            <ul className="space-y-1.5">
              {report.risks.map((r, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-ink)]">
                  <span className="text-[var(--color-warning)] mt-0.5">!</span> {r}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Rewrite Suggestions */}
      {report.rewrite_suggestions.length > 0 && (
        <div className="panel p-6 space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)]">Rewrite Suggestions</h3>
          {report.rewrite_suggestions.map((s, i) => (
            <RewriteSuggestion key={i} suggestion={s} />
          ))}
        </div>
      )}
    </div>
  );
}
