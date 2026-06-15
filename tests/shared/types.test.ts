import { describe, it, expect } from "vitest";
import { isPitchScoreReport, normalizePitchScoreReport, platformValues } from "@/shared/types";
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

  it("normalizePitchScoreReport repairs benign runtime drift", () => {
    const report = normalizePitchScoreReport({
      persuasion_score: "87.4",
      verdict: "  Strong fit  ",
      narrative: "",
      breakdown: [
        { key: "clarity", label: "Clarity", score: "105", explanation: "  Direct and readable.  " },
        { key: "broken", label: "Broken", score: "not-a-score", explanation: "drop me" },
      ],
      neural_signals: [
        { key: "attention_capture", label: "Attention Capture", score: "71", direction: "UP" },
      ],
      strengths: ["  Clear trigger  ", "", 42, "Specific CTA"],
      risks: ["  Could add proof  "],
      rewrite_suggestions: [
        { title: "", before: "old", after: "new", why: "clearer" },
        { title: "empty" },
      ],
      persona_summary: "",
      top_moves: [{ priority: "1", title: "Open stronger", do: "Lead with the migration.", because: "Attention is weak." }],
      platform: "LinkedIn",
      scored_at: "not-a-date",
      fmri_output: { segments: "bad" },
    });

    expect(report).not.toBeNull();
    expect(report?.persuasion_score).toBe(87);
    expect(report?.verdict).toBe("Strong fit");
    expect(report?.narrative).toBe("No narrative provided.");
    expect(report?.breakdown).toEqual([
      { key: "clarity", label: "Clarity", score: 100, explanation: "Direct and readable." },
    ]);
    expect(report?.neural_signals[0]).toEqual({
      key: "attention_capture",
      label: "Attention Capture",
      score: 71,
      direction: "up",
    });
    expect(report?.strengths).toEqual(["Clear trigger", "Specific CTA"]);
    expect(report?.rewrite_suggestions[0].title).toBe("Suggestion 1");
    expect(report?.persona_summary).toBe("Persona not summarized.");
    expect(report?.top_moves?.[0].title).toBe("Open stronger");
    expect(report?.platform).toBe("linkedin");
    expect(report?.fmri_output).toBeNull();
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
