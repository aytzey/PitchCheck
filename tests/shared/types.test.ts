import { describe, it, expect } from "vitest";
import { isPitchScoreReport, normalizePitchScoreReport, normalizeRefineEvidence, platformValues } from "@/shared/types";
import { BREAKDOWN_KEYS, NEURAL_SIGNAL_KEYS, PLATFORMS } from "@/shared/constants";
import { roiSummary, roiPrior } from "../fixtures/roi";

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

  it("keeps the full source registry, axis source_keys, caveats and guardrails", () => {
    const sourceKeys = Array.from({ length: 9 }, (_, index) => `source_${index}`);
    const report = normalizePitchScoreReport({
      persuasion_score: 70,
      persuasion_evidence: {
        research_sources: Array.from({ length: 37 }, (_, index) => ({
          key: `source_${index}`,
          citation: `Author ${index}`,
          url: `https://doi.org/10.0/${index}`,
          evidence_level: "individual_differences_in_sample",
          modality: "fMRI",
        })),
      },
      robustness: {
        final_score: 70,
        neuro_axes: { self_value: { label: "Self/value response proxy", score: 60, source_keys: sourceKeys } },
        scientific_caveats: Array.from({ length: 8 }, (_, index) => `caveat ${index}`),
        guardrails_applied: Array.from({ length: 10 }, (_, index) => `guardrail_${index}`),
        breakdown_anchor: { basis: "persona_read_facets_v1", items: {} },
      },
    });

    expect(report?.persuasion_evidence?.research_sources).toHaveLength(37);
    expect(report?.persuasion_evidence?.research_sources?.[36].modality).toBe("fMRI");
    expect(report?.robustness?.neuro_axes?.self_value.source_keys).toEqual(sourceKeys);
    expect(report?.robustness?.scientific_caveats).toHaveLength(8);
    expect(report?.robustness?.guardrails_applied).toHaveLength(10);
    expect(report?.robustness?.breakdown_anchor).toEqual({ basis: "persona_read_facets_v1" });
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

  it("normalizeRefineEvidence keeps only known candidate rows and scores", () => {
    const evidence = normalizeRefineEvidence({
      mode: "model",
      selected_id: "c2",
      improved: true,
      message: "never rendered",
      evaluations: [
        { id: "original", eligible: true, semantic_score: 61.4, neural_score: 55, selection_score: 58.2 },
        { id: "c2", eligible: true, semantic_score: "70", neural_score: 140, selection_score: 66 },
        { id: "<script>", eligible: true, semantic_score: 99 },
        "junk",
      ],
    });
    expect(evidence?.mode).toBe("model");
    expect(evidence?.selectedId).toBe("c2");
    expect(evidence?.jev).toBe(false);
    expect(evidence?.rows.map((row) => row.id)).toEqual(["original", "c2"]);
    expect(evidence?.rows[1]).toMatchObject({ semantic: 70, neural: 100, selection: 66 });
    expect(normalizeRefineEvidence({ mode: "model", evaluations: [] })).toBeNull();
    expect(normalizeRefineEvidence("nope")).toBeNull();
  });
});

it("preserves signed ROI fields and provenance while filtering malformed statistics", () => {
  const payload = {
    persuasion_score: 60,
    fmri_output: { segments: 6, voxel_count: 20484, global_mean_abs: 0.2, global_peak_abs: 0.5,
      roi_summary: { ...roiSummary, rois: { ...roiSummary.rois, invalid: { z: "no", n_vertices: 10 } } } },
    robustness: { neural_score: 60, roi_prior: roiPrior, roi_prior_points: -1.2 },
  };
  const normalized = normalizePitchScoreReport(payload)!;
  expect(normalized.fmri_output?.roi_summary).toEqual(roiSummary);
  expect(normalized.robustness?.roi_prior).toEqual(roiPrior);
  expect(normalized.robustness?.roi_prior_points).toBe(-1.2);
  const malformed = normalizePitchScoreReport({ ...payload,
    fmri_output: { ...payload.fmri_output, roi_summary: { ...roiSummary, cortex_sd_signed: Infinity } },
    robustness: { neural_score: 60, roi_prior: { ...roiPrior, raw: NaN }, roi_prior_points: "bad" },
  })!;
  expect(malformed.fmri_output?.roi_summary).toBeNull();
  expect(malformed.robustness?.roi_prior).toBeNull();
  expect(malformed.robustness?.roi_prior_points).toBeUndefined();
  expect(normalizePitchScoreReport({ persuasion_score: 60 })?.fmri_output).toBeNull();
});
