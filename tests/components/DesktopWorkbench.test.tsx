import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DesktopWorkbench from "@/components/DesktopWorkbench";
import type { PitchScoreReport } from "@/shared/types";
import { roiSummary, roiPrior } from "../fixtures/roi";

const mockFetch = vi.fn();
const tauriMocks = vi.hoisted(() => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: tauriMocks.invoke,
}));

const mockReport: PitchScoreReport = {
  persuasion_score: 76,
  verdict: "Strong pitch for technical audience",
  narrative: "The pitch addresses deployment pain and offers a specific next step.",
  breakdown: [
    { key: "emotional_resonance", label: "Emotional Resonance", score: 72, explanation: "Clear pain framing" },
    { key: "clarity", label: "Clarity", score: 84, explanation: "Specific and concise" },
    { key: "urgency", label: "Urgency", score: 68, explanation: "Timely migration context" },
    { key: "credibility", label: "Credibility", score: 80, explanation: "Concrete outcome" },
    { key: "personalization_fit", label: "Personalization Fit", score: 75, explanation: "Matches the recipient" },
  ],
  neural_signals: [
    { key: "emotional_engagement", label: "Emotional Engagement", score: 74, direction: "up" },
    { key: "personal_relevance", label: "Personal Relevance", score: 79, direction: "up" },
    { key: "social_proof_potential", label: "Social Proof Potential", score: 63, direction: "neutral" },
    { key: "memorability", label: "Memorability", score: 70, direction: "up" },
    { key: "attention_capture", label: "Attention Capture", score: 78, direction: "up" },
    { key: "cognitive_friction", label: "Cognitive Friction", score: 28, direction: "down" },
  ],
  strengths: ["Specific trigger", "Clear operational pain", "Concrete promise"],
  risks: ["Could add proof", "Could tighten CTA", "Could quantify cost"],
  rewrite_suggestions: [
    {
      title: "Add sharper proof",
      before: "cuts setup time",
      after: "cuts dashboard setup from roughly 3 days to 10 minutes",
      why: "Proof makes the claim easier to trust.",
    },
  ],
  persona_summary: "Engineering leader handling a migration",
  top_moves: [
    {
      priority: 1,
      title: "Open inside her migration problem",
      do: "Start with the observability gap her team hit after Athena, then offer the dashboard.",
      because: "Attention is weakest in the opener and she ignores product-first outreach.",
    },
  ],
  fmri_output: null,
  platform: "email",
  scored_at: new Date().toISOString(),
};

const caveats = Array.from({ length: 7 }, (_, index) => `Scientific caveat number ${index + 1}.`);
const axisKeys = ["self_value", "reward_affect", "social_sharing", "encoding_attention", "processing_fluency"];
const span = (segment: number, text: string) => ({ segment, of: 8, position_pct: segment * 12, value: 0.3, percentile: 50, text });

function fullReport(llmScore: number | null, anchored: boolean) {
  return {
    ...mockReport,
    fmri_output: {
      segments: 8,
      voxel_count: 20484,
      global_mean_abs: 0.2,
      global_peak_abs: 0.6,
      temporal_trace: [0.2, 0.3, 0.5, 0.4, 0.2, 0.3, 0.45, 0.35],
      temporal_peaks: [0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.55, 0.45],
      top_voxel_indices: [1, 2, 3, 4, 5, 6],
      top_voxel_values: [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
      temporal_trace_basis: "synthetic_word_order",
    },
    persuasion_evidence: {
      research_sources: [
        { key: "falk_2010_persuasion_change", citation: "Falk et al. 2010", url: "https://doi.org/10.1523/x", modality: "fMRI", evidence_level: "individual_differences_in_sample" },
        { key: "chan_2024_ad_liking", citation: "Insecure Source 2024", url: "http://example.com/chan" },
        ...Array.from({ length: 35 }, (_, index) => ({ key: `unused_${index}`, citation: `Unused ${index}`, url: "https://doi.org/10.0/x" })),
      ],
    },
    robustness: {
      neural_score: 66,
      evidence_score: 66,
      llm_score: llmScore,
      final_score: 70,
      confidence: 0.7,
      calibration_basis: "test",
      neuro_axes: Object.fromEntries(axisKeys.map((key) => [key, { label: key, score: 60, caveat: `${key} caveat`, source_keys: ["falk_2010_persuasion_change"] }])),
      scientific_caveats: caveats,
      breakdown_anchor: anchored ? { basis: "persona_read_facets_v1" } : null,
      research_synthesis: {
        items: [],
        temporal_archetype: { key: "late_peak", label: "Slow burn, late peak", implication: "i", lever: "l", citation: "c", source_keys: ["chan_2024_ad_liking"] },
        localization: { opener: span(1, "Hi"), opener_strength_percentile: 40, closer_strength_percentile: 60, peak: span(3, "peak words"), weakest: span(5, "weak words"), attention_cliff: null },
      },
    },
  };
}

async function renderReport(report: unknown) {
  mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ report }) });
  render(<DesktopWorkbench />);
  fireEvent.click(screen.getByRole("button", { name: /Score message/ }));
  await screen.findByText("Variant re-rank");
}

describe("DesktopWorkbench", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
    mockFetch.mockReset();
    tauriMocks.invoke.mockReset();
    delete window.__TAURI_INTERNALS__;
    vi.stubGlobal("fetch", mockFetch);
  });

  it("opens the settings panel from the URL hash with OpenRouter defaults", () => {
    window.history.replaceState(null, "", "/#settings");

    render(<DesktopWorkbench />);

    expect(screen.getByText("Machine env")).toBeDefined();
    expect(screen.getByLabelText(/OpenRouter API key/)).toBeDefined();
    expect(screen.getByLabelText<HTMLInputElement>(/Evaluator model/).value).toBe(
      "google/gemini-3.8-flash",
    );
    expect(screen.getByLabelText<HTMLInputElement>(/Refiner model/).value).toBe(
      "google/gemini-3.8-flash",
    );
  });

  it("updates the URL hash when changing tabs", () => {
    render(<DesktopWorkbench />);

    fireEvent.click(screen.getByRole("button", { name: "Runtime" }));

    expect(window.location.hash).toBe("#runtime");
    expect(screen.getByText("03 / Runtime selection")).toBeDefined();
  });

  it("shows PitchServer as an unsaved runtime password option", () => {
    window.history.replaceState(null, "", "/#runtime");

    const { unmount } = render(<DesktopWorkbench />);

    expect(screen.getByText("PitchServer")).toBeDefined();

    unmount();
    window.history.replaceState(null, "", "/#settings");
    render(<DesktopWorkbench />);
    expect(screen.queryByLabelText(/PitchServer SSH password/)).toBeNull();
    expect(screen.queryByLabelText(/PitchServer username/)).toBeNull();
    expect(screen.queryByLabelText(/PitchServer password/)).toBeNull();
  });

  it("exposes refine and variant re-rank controls in the workspace", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ report: mockReport }),
    });
    render(<DesktopWorkbench />);

    fireEvent.click(screen.getByRole("button", { name: /Score message/ }));

    expect(await screen.findByText("Variant re-rank")).toBeDefined();
    expect(screen.getByText("Top moves")).toBeDefined();
    expect(screen.getByText(/Open inside her migration problem/)).toBeDefined();
    expect(screen.getByRole("button", { name: /Refine draft/ })).toBeDefined();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        refined_message: "Hey Jordan - tighter rewrite with one clear ask.",
        model: "deepseek/deepseek-v4-pro",
        needs_clarification: false,
        questions: [],
      }),
    });
    fireEvent.click(screen.getByRole("button", { name: /Refine draft/ }));

    await waitFor(() => expect(screen.getByText("After . refined")).toBeDefined());
    const refineCall = mockFetch.mock.calls[1];
    expect(refineCall[0]).toBe("/api/refine");
    const refineBody = JSON.parse(refineCall[1].body as string) as { suggestions: string[] };
    expect(refineBody.suggestions[0]).toContain("Baseline persuasion score 76/100");
    expect(screen.getByText("Hey Jordan - tighter rewrite with one clear ask.")).toBeDefined();
    expect(screen.getByRole("button", { name: "Accept & continue editing" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Accept & re-evaluate" })).toBeDefined();
  });

  it("labels a neural-only report honestly and exposes every caveat and source", async () => {
    await renderReport(fullReport(null, false));

    for (const label of ["MPFC", "HPC", "dlPFC", "AI/dACC", "mPFC/PCC"]) {
      expect(screen.queryByText(label, { exact: true })).toBeNull();
    }
    for (const caveat of caveats) expect(screen.getByText(caveat)).toBeDefined();
    expect(screen.getByText(/20,484 VERTICES/)).toBeDefined();
    expect(screen.queryByText(/P = /)).toBeNull();
    expect(screen.getByText(/Positions are decorative/)).toBeDefined();
    expect(screen.getByText("Falk et al. 2010").closest("a")?.getAttribute("href")).toBe("https://doi.org/10.1523/x");
    expect(screen.getByText("Insecure Source 2024").closest("a")).toBeNull();
    expect(screen.getByText("Sources (2)")).toBeDefined();
    expect(screen.getByText("Neural-only (hand-set mapping)")).toBeDefined();
    expect(screen.getByText("Early salience (placeholder)")).toBeDefined();
    expect(screen.queryByText("LLM interpreted")).toBeNull();
    expect(screen.queryByText(/anchored to persona pain alignment/)).toBeNull();
  });

  it("keeps LLM facet labels and notes the persona anchor", async () => {
    await renderReport(fullReport(78, true));

    expect(screen.getByText("LLM interpreted")).toBeDefined();
    expect(screen.getByText("Urgency")).toBeDefined();
    expect(screen.getByText(/Matches the recipient · anchored to persona pain alignment/)).toBeDefined();
  });

  it("shows desktop refiner clarification questions in the main editor", async () => {
    window.__TAURI_INTERNALS__ = {};
    let refineCalls = 0;
    tauriMocks.invoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "get_app_config") {
        return {
          openRouterApiKey: "sk-test",
          openRouterModel: "deepseek/deepseek-v4-pro",
          openRouterRefinerModel: "deepseek/deepseek-v4-pro",
        };
      }
      if (command === "get_runtime_status") {
        return {
          mode: "pitchserver",
          connected: true,
          service_url: "http://127.0.0.1:18090",
          local_gpu: { available: true },
          image: "ghcr.io/aytzey/pitchcheck-tribe:latest",
        };
      }
      if (command === "get_setup_status") {
        return {
          platform: "linux",
          ready_for_local: true,
          ready_for_cloud: true,
          steps: [],
        };
      }
      if (command === "score_pitch") {
        return { report: mockReport };
      }
      if (command === "refine_pitch") {
        refineCalls += 1;
        if (refineCalls === 1) {
          return {
            needsClarification: true,
            model: "deepseek/deepseek-v4-pro",
            questions: [
              "Does Jordan's team currently use OpenTelemetry, or would adoption be a prerequisite?",
            ],
            safetyNotes: ["No invented metrics, timelines, or credentials will be added."],
          };
        }
        const request = (args as {
          request?: {
            persona?: string;
            clarificationAnswers?: Array<{ answer: string }>;
            clarificationRound?: number;
            forceRewrite?: boolean;
          };
        } | undefined)?.request;
        expect(request?.persona).toContain("Yes, they already use OpenTelemetry.");
        expect(request?.clarificationAnswers?.[0]?.answer).toBe("Yes, they already use OpenTelemetry.");
        expect(request?.clarificationRound).toBe(1);
        expect(request?.forceRewrite).toBe(false);
        return {
          refinedMessage: "Hey Jordan - since your team already has OpenTelemetry in place, I can show a dashboard from one service without new instrumentation.",
          model: "deepseek/deepseek-v4-pro",
        };
      }
      throw new Error(`unexpected command ${command}`);
    });

    render(<DesktopWorkbench />);

    fireEvent.click(await screen.findByRole("button", { name: /Score message/ }));
    expect(await screen.findByText("Variant re-rank")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /Refine draft/ }));

    expect(await screen.findByText("Refiner needs context")).toBeDefined();
    expect(screen.getAllByText(/Does Jordan's team currently use OpenTelemetry/).length).toBeGreaterThan(0);
    expect(screen.queryByText("After . refined")).toBeNull();
    expect((screen.getAllByRole("button", { name: /Skip & refine/ })[0] as HTMLButtonElement).disabled).toBe(false);

    fireEvent.change(screen.getByLabelText("Answer"), {
      target: { value: "Yes, they already use OpenTelemetry." },
    });
    fireEvent.click(screen.getAllByRole("button", { name: /Use answers & refine/ })[0]);

    expect(await screen.findByText("After . refined")).toBeDefined();
    expect(refineCalls).toBe(2);
  });
});

describe("WS07 desktop ROI fields", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
    mockFetch.mockReset();
    tauriMocks.invoke.mockReset();
    delete window.__TAURI_INTERNALS__;
    vi.stubGlobal("fetch", mockFetch);
  });

  it("hides the ROI row for old servers and shares the signal explanation", async () => {
    await renderReport(fullReport(78, true));
    expect(screen.queryByText("ROI prior (literature prior)")).toBeNull();
    expect(screen.getAllByText("About these signals")).toHaveLength(2);
    expect(screen.queryByRole("img", { name: "Signed ROI analogues for the average reader" })).toBeNull();
  });

  it("fills the ROI row, uses atlas centroids and sends only levers to refine", async () => {
    const report = fullReport(78, true);
    await renderReport({ ...report,
      fmri_output: { ...report.fmri_output, roi_summary: roiSummary },
      robustness: { ...report.robustness, roi_prior: roiPrior, roi_prior_points: -1.2,
        research_synthesis: { items: [{ key: "roi_vmpfc_low", kind: "gap", axis: "cortical_roi",
          observation: "Predicted medial prefrontal response z=-0.5", finding: "Finding", citation: "Vezich et al. 2017",
          lever: "Say why the outcome matters to this reader." }] } },
    });
    expect(screen.getByText("ROI prior (literature prior)")).toBeDefined();
    expect(screen.getByText("-1.2 pts")).toBeDefined();
    expect(screen.getByText(/external fidelity: weak · in silico: null/)).toBeDefined();
    expect(screen.getByRole("img", { name: "Signed ROI analogues for the average reader" })).toBeDefined();
    expect(screen.queryByText(/Positions are decorative/)).toBeNull();
    expect(screen.getByText(/signed ROI analogues \(average reader\)/)).toBeDefined();
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ refined_message: "A concrete rewrite with a clear ask.", model: "test" }) });
    fireEvent.click(screen.getByRole("button", { name: /Refine draft/ }));
    await screen.findByText("After . refined");
    const suggestions = JSON.parse(mockFetch.mock.calls[1][1].body as string).suggestions.join(" ");
    expect(suggestions).toContain("Say why the outcome matters");
    expect(suggestions).not.toContain("z=-0.5");
    expect(suggestions).not.toContain("medial prefrontal");
  });

  it("labels an experimental research hypothesis HYP", async () => {
    const report = fullReport(78, true);
    await renderReport({ ...report, robustness: { ...report.robustness, research_synthesis: {
      items: [{ key: "falk2010jocn_core_low_relative", kind: "hypothesis", axis: null, experimental: true,
        reference: "default_prior_within_draft", observation: "Predicted average-reader response is relatively low.",
        finding: "Unvalidated literature prior; not this recipient.", citation: "Falk, Rameson et al. 2010 (JoCN)",
        lever: "State the claim and your reason in the reader's own terms." }],
    } } });
    expect(screen.getByText("HYP")).toBeDefined();
  });

  it.each([
    ["higher", true, "≤0.5 pt nudge"],
    ["higher", false, "not used for ranking"],
    ["lower", false, "not used for ranking"],
    ["no_reliable_difference", true, "experimental"],
  ])("shows the concise ROI readout %s (Jev=%s)", async (direction, jev, label) => {
    await renderReport(fullReport(78, true));
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({
      refined_message: "A concrete rewrite with a clear ask.", model: "test", tribe_guidance: {
        mode: "model", selected_id: "c1", improved: true,
        evaluations: [{ id: "original", eligible: true }, { id: "c1", eligible: true }],
        ...(jev ? { selection_policy: {} } : {}), roi_readout: { direction, used_in_selection: jev },
      },
    }) });
    fireEvent.click(screen.getByRole("button", { name: /Refine draft/ }));
    const line = await screen.findByText(/Medial-prefrontal response vs your draft:/);
    expect(line.textContent).toContain(label);
    expect(line.textContent!.split(/\s+/).length).toBeLessThanOrEqual(15);
    expect(line.getAttribute("title")).toContain("Predicted average-reader response");
    expect(screen.getAllByText(/not a mental state/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/vmPFC/)).toBeNull();
  });
});
