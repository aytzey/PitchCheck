import type { PitchScoreReport } from "@/shared/types";
import ScoreGauge from "./ScoreGauge";
import BreakdownBar from "./BreakdownBar";
import RewriteSuggestion from "./RewriteSuggestion";
import TemporalTrace from "./TemporalTrace";

export default function ScoreDisplay({ report }: { report: PitchScoreReport }) {
  const neuroAxes = report.robustness?.neuro_axes
    ? Object.entries(report.robustness.neuro_axes)
    : [];

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
            {report.neural_signals.map((s) => {
              // Cognitive friction is inverted: low = good (green), high = bad (red)
              const isInverted = s.key === "cognitive_friction";
              const displayScore = Math.round(s.score);
              const effectiveScore = isInverted ? 100 - s.score : s.score;
              const color = effectiveScore >= 60 ? 'var(--color-success)' : effectiveScore < 40 ? 'var(--color-danger)' : 'var(--color-warning)';
              return (
                <div key={s.key} className="rounded-lg border border-[var(--color-line)] p-3 text-center">
                  <p className="text-2xl font-bold" style={{ color }}>
                    {displayScore}
                  </p>
                  <p className="mt-1 text-xs text-[var(--color-muted)]">{s.label}</p>
                  {isInverted && <p className="text-[9px] text-[var(--color-faint)]">low = clear</p>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Evidence calibration + robustness */}
      {(report.persuasion_evidence || report.robustness) && (
        <div className="grid gap-4 sm:grid-cols-2">
          {report.robustness && (
            <div className="panel p-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)] mb-2">Robustness</h3>
              <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
                <Metric label="Neuro axes" value={Math.round(report.robustness.neural_score)} />
                <Metric label="Neural prior" value={Math.round(report.robustness.neural_prior_score ?? report.robustness.evidence_score)} />
                <Metric label="Quality" value={`${Math.round((report.robustness.prediction_quality_weight ?? 1) * 100)}%`} />
                <Metric label="Confidence" value={`${Math.round(report.robustness.confidence * 100)}%`} />
              </div>
              {neuroAxes.length > 0 && (
                <div className="mt-3 space-y-2">
                  {neuroAxes.slice(0, 5).map(([key, axis]) => (
                    <div key={key}>
                      <div className="flex items-center justify-between gap-3 text-xs">
                        <span className="font-medium text-[var(--color-ink)]">{axis.label}</span>
                        <span className="font-mono text-[var(--color-muted)]">{Math.round(axis.score)}</span>
                      </div>
                      <div className="mt-1 h-1.5 rounded-full bg-[var(--color-line)]">
                        <div
                          className="h-1.5 rounded-full bg-[var(--color-pitch)]"
                          style={{ width: `${Math.max(4, Math.min(100, axis.score))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {report.robustness.guardrails_applied.length > 0 && (
                <p className="mt-3 text-xs text-[var(--color-muted)]">
                  Guardrails: {report.robustness.guardrails_applied.join(", ")}
                </p>
              )}
              {(report.robustness.scientific_caveats ?? []).length > 0 && (
                <p className="mt-3 text-[10px] leading-relaxed text-[var(--color-faint)]">
                  {report.robustness.scientific_caveats?.[0]}
                </p>
              )}
            </div>
          )}
          {report.persuasion_evidence && (
            <div className="panel p-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)] mb-2">Semantic Context</h3>
              <p className="text-xs text-[var(--color-muted)]">
                Text heuristics are disabled; message and persona are used only as untrusted semantic context for LLM interpretation.
              </p>
              {report.persuasion_evidence.methodology && (
                <p className="mt-2 text-xs text-[var(--color-warning)]">
                  {report.persuasion_evidence.methodology.replaceAll("_", " ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Temporal Engagement Timeline */}
      {report.fmri_output && report.fmri_output.temporal_trace.length > 0 && (
        <TemporalTrace fmri={report.fmri_output} />
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

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-[var(--color-line)] p-2">
      <p className="text-lg font-bold text-[var(--color-ink)]">{value}</p>
      <p className="text-[10px] text-[var(--color-faint)]">{label}</p>
    </div>
  );
}
