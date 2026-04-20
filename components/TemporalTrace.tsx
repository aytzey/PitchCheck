import type { FmriOutput } from "@/shared/types";

interface Props {
  fmri: FmriOutput;
}

export default function TemporalTrace({ fmri }: Props) {
  const { temporal_trace, segments, voxel_count } = fmri;

  if (!temporal_trace.length) return null;

  const maxVal = Math.max(...temporal_trace, 0.001);
  const peakIdx = temporal_trace.indexOf(Math.max(...temporal_trace));
  const isSynthetic = fmri.temporal_trace_basis === "synthetic_word_order";

  // Segment labels: opener, body, close
  function segmentLabel(i: number, total: number): string {
    const pct = i / Math.max(total - 1, 1);
    if (pct <= 0.2) return "opener";
    if (pct >= 0.8) return "close";
    return "body";
  }

  return (
    <div className="panel p-6">
      <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--color-muted)] mb-1">
        {isSynthetic ? "Predicted Engagement Trace" : "Predicted Engagement Timeline"}
      </h3>
      <p className="text-xs text-[var(--color-faint)] mb-4">
        {segments} temporal segments analyzed on {voxel_count.toLocaleString()} cortical vertices
        &mdash; peak at segment {peakIdx + 1}/{segments} ({segmentLabel(peakIdx, segments)})
        {isSynthetic ? " (Direct text mode uses ordered text segments, not elapsed seconds)" : ""}
      </p>

      {/* SVG Bar Chart */}
      <div className="relative">
        <svg viewBox={`0 0 ${Math.max(temporal_trace.length * 32, 100)} 120`} className="w-full h-28" preserveAspectRatio="none">
          {/* Grid lines */}
          <line x1="0" y1="30" x2={temporal_trace.length * 32} y2="30" stroke="var(--color-line)" strokeWidth="0.5" strokeDasharray="4 4" />
          <line x1="0" y1="60" x2={temporal_trace.length * 32} y2="60" stroke="var(--color-line)" strokeWidth="0.5" strokeDasharray="4 4" />
          <line x1="0" y1="90" x2={temporal_trace.length * 32} y2="90" stroke="var(--color-line)" strokeWidth="0.5" strokeDasharray="4 4" />

          {/* Bars */}
          {temporal_trace.map((val, i) => {
            const barHeight = (val / maxVal) * 100;
            const isPeak = i === peakIdx;
            const section = segmentLabel(i, temporal_trace.length);
            const color = isPeak
              ? "var(--color-pitch)"
              : section === "opener"
                ? "var(--color-success)"
                : section === "close"
                  ? "var(--color-warning)"
                  : "var(--color-pitch-light)";
            return (
              <g key={i}>
                <rect
                  x={i * 32 + 4}
                  y={110 - barHeight}
                  width="24"
                  height={barHeight}
                  rx="3"
                  fill={color}
                  opacity={isPeak ? 1 : 0.7}
                />
                <text
                  x={i * 32 + 16}
                  y={120}
                  textAnchor="middle"
                  fill="var(--color-faint)"
                  fontSize="8"
                >
                  {i + 1}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-[var(--color-faint)]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: "var(--color-success)" }} />
            Opener
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: "var(--color-pitch-light)" }} />
            Body
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: "var(--color-warning)" }} />
            Close
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: "var(--color-pitch)" }} />
            Peak
          </span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mt-4 text-center">
        <div className="rounded-lg border border-[var(--color-line)] p-2">
          <p className="text-lg font-bold text-[var(--color-ink)]">{fmri.global_mean_abs.toFixed(3)}</p>
          <p className="text-[10px] text-[var(--color-faint)]">Mean Response</p>
        </div>
        <div className="rounded-lg border border-[var(--color-line)] p-2">
          <p className="text-lg font-bold text-[var(--color-ink)]">{fmri.global_peak_abs.toFixed(3)}</p>
          <p className="text-[10px] text-[var(--color-faint)]">Peak Response</p>
        </div>
        <div className="rounded-lg border border-[var(--color-line)] p-2">
          <p className="text-lg font-bold text-[var(--color-ink)]">{(fmri.global_peak_abs / Math.max(fmri.global_mean_abs, 0.001)).toFixed(1)}x</p>
          <p className="text-[10px] text-[var(--color-faint)]">Peak/Mean Ratio</p>
        </div>
      </div>
    </div>
  );
}
