import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ScoreDisplay from "@/components/ScoreDisplay";
import type { PitchScoreReport } from "@/shared/types";

const mockReport: PitchScoreReport = {
  persuasion_score: 75,
  verdict: "Strong pitch for technical audience",
  narrative: "The pitch effectively addresses deployment pain points.",
  breakdown: [
    { key: "emotional_resonance", label: "Emotional Resonance", score: 70, explanation: "Good emotional engagement" },
    { key: "clarity", label: "Clarity", score: 85, explanation: "Very clear messaging" },
    { key: "urgency", label: "Urgency", score: 60, explanation: "Moderate urgency" },
    { key: "credibility", label: "Credibility", score: 80, explanation: "Strong credibility signals" },
    { key: "personalization_fit", label: "Personalization Fit", score: 65, explanation: "Decent fit" },
  ],
  neural_signals: [
    { key: "emotional_engagement", label: "Emotional Engagement", score: 72, direction: "up" },
    { key: "personal_relevance", label: "Personal Relevance", score: 68, direction: "up" },
    { key: "social_proof_potential", label: "Social Proof", score: 55, direction: "neutral" },
    { key: "memorability", label: "Memorability", score: 60, direction: "neutral" },
    { key: "attention_capture", label: "Attention Capture", score: 78, direction: "up" },
    { key: "cognitive_friction", label: "Cognitive Friction", score: 30, direction: "down" },
  ],
  strengths: ["Clear value proposition", "Strong technical credibility", "Good opener"],
  risks: ["Missing social proof", "Could improve urgency", "Limited personalization"],
  rewrite_suggestions: [
    { title: "Strengthen opener", before: "Our platform", after: "[Name], your team is spending 5x too long on deploys", why: "Personal hook" },
  ],
  persona_summary: "Technical CTO at early-stage startup",
  platform: "email",
  scored_at: new Date().toISOString(),
};

describe("ScoreDisplay", () => {
  it("renders the score gauge", () => {
    render(<ScoreDisplay report={mockReport} />);
    expect(screen.getByText("75")).toBeDefined();
  });

  it("renders the verdict", () => {
    render(<ScoreDisplay report={mockReport} />);
    expect(screen.getByText("Strong pitch for technical audience")).toBeDefined();
  });

  it("renders breakdown bars", () => {
    render(<ScoreDisplay report={mockReport} />);
    expect(screen.getByText("Emotional Resonance")).toBeDefined();
    expect(screen.getByText("Clarity")).toBeDefined();
  });

  it("renders strengths and risks", () => {
    render(<ScoreDisplay report={mockReport} />);
    expect(screen.getByText("Clear value proposition")).toBeDefined();
    expect(screen.getByText("Missing social proof")).toBeDefined();
  });

  it("renders rewrite suggestions", () => {
    render(<ScoreDisplay report={mockReport} />);
    expect(screen.getByText("Strengthen opener")).toBeDefined();
  });
});
