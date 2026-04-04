import type { BreakdownSection } from "@/shared/types";

function barColor(score: number): string {
  if (score >= 70) return "bg-[var(--color-success)]";
  if (score >= 40) return "bg-[var(--color-warning)]";
  return "bg-[var(--color-danger)]";
}

export default function BreakdownBar({ section }: { section: BreakdownSection }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-[var(--color-ink)]">{section.label}</span>
        <span className="font-mono text-xs font-semibold text-[var(--color-muted)]">{Math.round(section.score)}</span>
      </div>
      <div className="h-2 rounded-full bg-[var(--color-line)]">
        <div
          className={`h-2 rounded-full ${barColor(section.score)} transition-all duration-500`}
          style={{ width: `${Math.min(100, Math.max(0, section.score))}%` }}
        />
      </div>
      <p className="text-xs text-[var(--color-muted)]">{section.explanation}</p>
    </div>
  );
}
