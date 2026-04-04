import { PLATFORMS } from "@/shared/constants";
import { platformValues } from "@/shared/types";

interface Props {
  value: string;
  onChange: (v: string) => void;
}

export default function PlatformSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="block text-sm font-semibold text-[var(--color-ink)] mb-1.5">Platform</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2.5 text-sm focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)]"
      >
        {platformValues.map((p) => (
          <option key={p} value={p}>{PLATFORMS[p].label}</option>
        ))}
      </select>
    </div>
  );
}
