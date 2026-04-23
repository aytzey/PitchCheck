import { PitchScoreReport, Platform } from "./types";

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string").slice(0, 8);
}

function asScore(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function normalizePitchScoreReport(data: unknown): PitchScoreReport {
  if (!data || typeof data !== "object") {
    throw new Error("Runtime returned an invalid payload.");
  }

  const raw = data as Record<string, unknown>;
  const score = asScore(raw.persuasion_score);
  const verdict = asString(raw.verdict, "No verdict provided.");
  const narrative = asString(raw.narrative, "No narrative provided.");
  const strengths = asStringArray(raw.strengths);
  const risks = asStringArray(raw.risks);
  const platformRaw = asString(raw.platform, "general");
  const allowed = new Set<Platform>(["email", "linkedin", "cold-call-script", "landing-page", "ad-copy", "general"]);
  const platform: Platform = allowed.has(platformRaw as Platform) ? (platformRaw as Platform) : "general";
  const scoredAt = asString(raw.scored_at, new Date().toISOString());

  return {
    persuasion_score: score,
    verdict,
    narrative,
    strengths,
    risks,
    platform,
    scored_at: scoredAt,
  };
}
