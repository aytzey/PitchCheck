import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DesktopWorkbench from "@/components/DesktopWorkbench";
import type { PitchScoreReport } from "@/shared/types";

const mockFetch = vi.fn();

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
  fmri_output: null,
  platform: "email",
  scored_at: new Date().toISOString(),
};

describe("DesktopWorkbench", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
    mockFetch.mockReset();
    vi.stubGlobal("fetch", mockFetch);
  });

  it("opens the settings panel from the URL hash with OpenRouter defaults", () => {
    window.history.replaceState(null, "", "/#settings");

    render(<DesktopWorkbench />);

    expect(screen.getByText("Machine env")).toBeDefined();
    expect(screen.getByLabelText(/OpenRouter API key/)).toBeDefined();
    expect(screen.getByLabelText<HTMLInputElement>(/Evaluator model/).value).toBe(
      "anthropic/claude-sonnet-4.6",
    );
    expect(screen.getByLabelText<HTMLInputElement>(/Refiner model/).value).toBe(
      "anthropic/claude-sonnet-4.6",
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
  });

  it("exposes refine and variant re-rank controls in the workspace", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ report: mockReport }),
    });
    render(<DesktopWorkbench />);

    fireEvent.click(screen.getByRole("button", { name: /Score message/ }));

    expect(await screen.findByText("Variant re-rank")).toBeDefined();
    expect(screen.getByRole("button", { name: /Refine & re-score/ })).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /Refine & re-score/ }));

    await waitFor(() => expect(screen.getByText("After . refined")).toBeDefined());
    expect(screen.getByRole("button", { name: "Accept & continue editing" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Accept & re-evaluate" })).toBeDefined();
  });
});
