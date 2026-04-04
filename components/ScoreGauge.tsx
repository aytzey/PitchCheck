interface Props {
  score: number;
}

function scoreColor(score: number): string {
  if (score >= 70) return "var(--color-success)";
  if (score >= 40) return "var(--color-warning)";
  return "var(--color-danger)";
}

export default function ScoreGauge({ score }: Props) {
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (score / 100) * circumference;
  const color = scoreColor(score);

  return (
    <div className="flex flex-col items-center">
      <svg width="140" height="140" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="54" fill="none" stroke="var(--color-line)" strokeWidth="8" />
        <circle
          cx="60" cy="60" r="54" fill="none"
          stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        <text x="60" y="55" textAnchor="middle" className="text-3xl font-bold" fill={color}>{Math.round(score)}</text>
        <text x="60" y="72" textAnchor="middle" className="text-xs" fill="var(--color-muted)">/100</text>
      </svg>
    </div>
  );
}
