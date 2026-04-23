import { describe, expect, it } from "vitest";
import { normalizePitchScoreReport } from "@/mobile/src/report";

describe("normalizePitchScoreReport", () => {
  it("normalizes and clamps a valid payload", () => {
    const result = normalizePitchScoreReport({
      persuasion_score: 104.4,
      verdict: "Strong",
      narrative: "Great clarity",
      strengths: ["A", "B"],
      risks: ["X"],
      platform: "email",
      scored_at: "2026-04-23T00:00:00.000Z",
    });

    expect(result.persuasion_score).toBe(100);
    expect(result.platform).toBe("email");
    expect(result.strengths).toEqual(["A", "B"]);
  });

  it("falls back on malformed payload fields", () => {
    const result = normalizePitchScoreReport({
      persuasion_score: "bad",
      strengths: ["ok", 123, null],
      risks: "oops",
      platform: "unknown",
    });

    expect(result.persuasion_score).toBe(0);
    expect(result.platform).toBe("general");
    expect(result.strengths).toEqual(["ok"]);
    expect(result.risks).toEqual([]);
    expect(result.verdict.length).toBeGreaterThan(0);
  });

  it("throws for non-object payload", () => {
    expect(() => normalizePitchScoreReport(null)).toThrow(/invalid payload/i);
  });
});
