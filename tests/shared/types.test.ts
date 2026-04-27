import { describe, it, expect } from "vitest";
import { isPitchScoreReport, platformValues } from "@/shared/types";
import { BREAKDOWN_KEYS, NEURAL_SIGNAL_KEYS, PLATFORMS } from "@/shared/constants";

describe("shared/types", () => {
  it("platformValues has 6 entries", () => {
    expect(platformValues).toHaveLength(6);
    expect(platformValues).toContain("email");
    expect(platformValues).toContain("general");
  });

  it("isPitchScoreReport validates correct shape", () => {
    const valid = {
      persuasion_score: 75,
      verdict: "Good pitch",
      narrative: "Analysis here",
      breakdown: [{ key: "clarity", label: "Clarity", score: 80, explanation: "Clear" }],
      neural_signals: [{ key: "attention", label: "Attention", score: 70, direction: "up" }],
      strengths: ["Good opener"],
      risks: ["Weak CTA"],
      rewrite_suggestions: [],
      persona_summary: "CTO",
      platform: "email",
      scored_at: new Date().toISOString(),
    };
    expect(isPitchScoreReport(valid)).toBe(true);
  });

  it("isPitchScoreReport rejects invalid shapes", () => {
    expect(isPitchScoreReport(null)).toBe(false);
    expect(isPitchScoreReport({})).toBe(false);
    expect(isPitchScoreReport({ persuasion_score: 150, verdict: "x", breakdown: [], neural_signals: [], strengths: [], risks: [] })).toBe(false);
    expect(isPitchScoreReport({ persuasion_score: "not a number" })).toBe(false);
    expect(isPitchScoreReport({
      persuasion_score: 75,
      verdict: "Good pitch",
      narrative: "Analysis here",
      breakdown: [{ key: "clarity", label: "Clarity", score: "80", explanation: "Clear" }],
      neural_signals: [],
      strengths: [],
      risks: [],
      rewrite_suggestions: [],
      persona_summary: "CTO",
      platform: "email",
      scored_at: new Date().toISOString(),
    })).toBe(false);
  });

  it("BREAKDOWN_KEYS has 5 entries", () => {
    expect(BREAKDOWN_KEYS).toHaveLength(5);
  });

  it("NEURAL_SIGNAL_KEYS has 6 entries", () => {
    expect(NEURAL_SIGNAL_KEYS).toHaveLength(6);
  });

  it("PLATFORMS covers all platformValues", () => {
    for (const p of platformValues) {
      expect(PLATFORMS[p]).toBeDefined();
      expect(PLATFORMS[p].label).toBeTruthy();
    }
  });
});
