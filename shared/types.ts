// Platform options for pitch context
export const platformValues = [
  "email",
  "linkedin",
  "cold-call-script",
  "landing-page",
  "ad-copy",
  "general",
] as const;

export type Platform = (typeof platformValues)[number];

// Request to score a pitch
export interface PitchScoreRequest {
  message: string;      // The pitch/sales message (min 10 chars)
  persona: string;      // Free-text target persona description (min 5 chars)
  platform?: Platform;  // Optional platform context (default: "general")
}

// Individual breakdown dimension
export interface BreakdownSection {
  key: string;           // e.g. "emotional_resonance"
  label: string;         // e.g. "Emotional Resonance"
  score: number;         // 0-100
  explanation: string;   // LLM-generated explanation for this persona
}

// Neural signal from TRIBE
export interface NeuralSignal {
  key: string;           // e.g. "emotional_engagement"
  label: string;         // e.g. "Emotional Engagement"
  score: number;         // 0-100
  direction: "up" | "down" | "neutral";
}

// Rewrite suggestion
export interface RewriteSuggestion {
  title: string;         // e.g. "Strengthen the opener"
  before: string;        // Original snippet
  after: string;         // Improved snippet
  why: string;           // Explanation
}

// Full score report
export interface PitchScoreReport {
  persuasion_score: number;        // 0-100 overall score
  verdict: string;                  // One-line verdict
  narrative: string;                // 2-3 sentence analysis
  breakdown: BreakdownSection[];    // 5 sections
  neural_signals: NeuralSignal[];   // 6 TRIBE-derived signals
  strengths: string[];              // 3 strengths
  risks: string[];                  // 3 risks
  rewrite_suggestions: RewriteSuggestion[]; // 2-3 suggestions
  persona_summary: string;          // LLM's understanding of the persona
  fmri_output?: FmriOutput | null;  // fMRI summary from TRIBE
  platform: Platform;
  scored_at: string;                // ISO timestamp
}

// fMRI output from TRIBE neural model
export interface FmriOutput {
  segments: number;
  voxel_count: number;
  global_mean_abs: number;
  global_peak_abs: number;
  temporal_trace: number[];     // per-segment mean activation
  temporal_peaks: number[];     // per-segment peak activation
  top_voxel_indices: number[];  // top 6 most-activated voxel indices
  top_voxel_values: number[];   // their mean activation values
}

// Type guard
export function isPitchScoreReport(obj: unknown): obj is PitchScoreReport {
  if (!obj || typeof obj !== "object") return false;
  const r = obj as Record<string, unknown>;
  return (
    typeof r.persuasion_score === "number" &&
    r.persuasion_score >= 0 &&
    r.persuasion_score <= 100 &&
    typeof r.verdict === "string" &&
    Array.isArray(r.breakdown) &&
    Array.isArray(r.neural_signals) &&
    Array.isArray(r.strengths) &&
    Array.isArray(r.risks)
  );
}
