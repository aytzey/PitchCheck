import { describe, it, expect } from "vitest";
import { isPitchScoreReport, normalizePitchScoreReport, normalizeRefineEvidence, platformValues } from "@/shared/types";
import { BREAKDOWN_KEYS, NEURAL_SIGNAL_KEYS, PLATFORMS } from "@/shared/constants";

const contextFit = {
  persona_pain_alignment: { score: 70, note: "Pain" },
  objection_coverage: { score: 60, note: "Objections" },
  proof_credibility: { score: 40, note: "Proof" },
  cta_ease: { score: 80, note: "CTA" },
  channel_fit: { score: 75, note: "Channel" },
};

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
        context_fit_rubric_version: "v2",
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
    expect(report?.robustness?.context_fit_rubric_version).toBe("v2");
    expect(report?.robustness?.scientific_caveats).toHaveLength(8);
    expect(report?.robustness?.guardrails_applied).toHaveLength(10);
    expect(report?.robustness?.breakdown_anchor).toEqual({ basis: "persona_read_facets_v1" });
  });

  it.each([
    ["receptive", "generic"],
    ["neutral", "category"],
    ["skeptical", "recipient_specific"],
  ])("round-trips canonical persona labels %s / %s", (stance, level) => {
    const report = normalizePitchScoreReport({
      persuasion_score: 70,
      context_fit: { ...contextFit, recipient_stance: stance, personalization_level: level, unused_persona_facts: ["Owns the migration rollout"] },
    });

    expect(report?.context_fit).toMatchObject({ recipient_stance: stance, personalization_level: level, unused_persona_facts: ["Owns the migration rollout"] });
    expect(normalizePitchScoreReport(report)?.context_fit).toEqual(report?.context_fit);
  });

  it.each(["Skeptical", "duygusal", "unknown", 123, null, undefined])("keeps invalid persona labels %s unknown", (value) => {
    const report = normalizePitchScoreReport({
      persuasion_score: 70,
      context_fit: { ...contextFit, recipient_stance: value, personalization_level: value },
    });

    expect(report?.context_fit?.recipient_stance).toBeNull();
    expect(report?.context_fit?.personalization_level).toBeNull();
  });

  it("bounds persona facts and still requires every context-fit facet", () => {
    const report = normalizePitchScoreReport({
      persuasion_score: 70,
      context_fit: { ...contextFit, unused_persona_facts: [" x ", "a".repeat(300), "Third fact", "Fourth fact"] },
    });

    expect(report?.context_fit?.unused_persona_facts).toEqual(["x", "a".repeat(160), "Third fact"]);
    expect(normalizePitchScoreReport({ persuasion_score: 70, context_fit: { ...contextFit, channel_fit: null, recipient_stance: "skeptical" } })?.context_fit).toBeNull();
  });

  it("normalizes the persona prior and preserves its provenance", () => {
    const report = normalizePitchScoreReport({
      persuasion_score: 70,
      robustness: {
        final_score: 70,
        persona_read_prior: {
          version: "persona_read_prior_v1",
          enabled: true,
          basis: "literature_prior_unvalidated",
          recipient_stance: "skeptical",
          personalization_level: "recipient_specific",
          skeptic_facet_boost: "0.4",
          tailoring_points: "3",
          final_score_without_prior: "68.5",
          source_keys: ["petty_cacioppo_1986_elm", "chua_2009_tailoring"],
        },
      },
    });

    expect(report?.robustness?.persona_read_prior).toEqual({
      version: "persona_read_prior_v1",
      enabled: true,
      basis: "literature_prior_unvalidated",
      recipient_stance: "skeptical",
      personalization_level: "recipient_specific",
      skeptic_facet_boost: 0.4,
      tailoring_points: 3,
      final_score_without_prior: 68.5,
      source_keys: ["petty_cacioppo_1986_elm", "chua_2009_tailoring"],
    });
    const invalid = normalizePitchScoreReport({ persuasion_score: 70, robustness: { persona_read_prior: { enabled: "true", final_score_without_prior: "NaN" } } });
    expect(invalid?.robustness?.persona_read_prior).toMatchObject({ enabled: false, final_score_without_prior: null });
    expect(normalizePitchScoreReport({ persuasion_score: 70, robustness: {} })?.robustness?.persona_read_prior).toBeNull();
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
