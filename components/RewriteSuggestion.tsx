import type { RewriteSuggestion as Suggestion } from "@/shared/types";

export default function RewriteSuggestion({ suggestion }: { suggestion: Suggestion }) {
  return (
    <div className="rounded-lg border border-[var(--color-line)] p-3 space-y-2">
      <p className="text-sm font-semibold text-[var(--color-ink)]">{suggestion.title}</p>
      <div className="grid gap-2 sm:grid-cols-2 text-xs">
        <div className="rounded bg-red-50 p-2">
          <p className="font-semibold text-red-600 mb-1">Before</p>
          <p className="text-red-800">{suggestion.before}</p>
        </div>
        <div className="rounded bg-green-50 p-2">
          <p className="font-semibold text-green-600 mb-1">After</p>
          <p className="text-green-800">{suggestion.after}</p>
        </div>
      </div>
      <p className="text-xs text-[var(--color-muted)]">{suggestion.why}</p>
    </div>
  );
}
