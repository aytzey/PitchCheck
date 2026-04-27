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
  openRouterModel?: string; // Optional evaluator model override
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

// Compatibility metadata. Text heuristics are disabled; score calibration is neural-only.
export interface PersuasionEvidence {
  overall_text_score: number;
  feature_scores: Record<string, number>;
  detected_strategies: string[];
  missing_elements: string[];
  warnings: string[];
  prompt_injection_risk: number;
  readability: Record<string, number>;
  matched_persona_terms: string[];
  strategy_counts?: Record<string, number>;
  platform?: Platform | string;
  methodology_version?: string;
  methodology?: string;
  input_metadata?: Record<string, number>;
  research_basis?: string[];
  research_sources?: Array<Record<string, string>>;
  calibration_quality?: Record<string, unknown>;
}

export interface NeuroAxisReport {
  label: string;
  score: number;
  weight: number;
  contribution: number;
  analogue: string;
  evidence: string[];
  caveat: string;
  source_keys?: string[];
  unsupported_by_text?: boolean;
}

// Robustness/calibration metadata for the final persuasion score
export interface RobustnessReport {
  neural_prior_score?: number;
  neural_score: number;
  quality_adjusted_neural_score?: number;
  prediction_quality_weight?: number;
  text_score: number | null;
  evidence_score: number;
  llm_score: number | null;
  raw_llm_score?: number | null;
  llm_score_adjusted?: boolean;
  llm_model?: string | null;
  final_score: number;
  confidence: number;
  score_delta: number | null;
  prompt_injection_risk: number | null;
  guardrails_applied: string[];
  warnings: string[];
  neuro_axes?: Record<string, NeuroAxisReport>;
  confidence_reasons?: string[];
  scientific_caveats?: string[];
  calibration_basis: string;
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
  persuasion_evidence?: PersuasionEvidence | null; // Neural-only compatibility metadata
  robustness?: RobustnessReport | null; // Score calibration + guardrail metadata
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
  response_kind?: string;
  prediction_subject_basis?: string;
  cortical_mesh?: string;
  hemodynamic_lag_seconds?: number;
  temporal_trace_basis?: "real_time_seconds" | "synthetic_word_order";
  temporal_segment_label?: string;
  temporal_trace_note?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function isFiniteScore(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isBreakdownSection(value: unknown): value is BreakdownSection {
  if (!isRecord(value)) return false;
  return (
    typeof value.key === "string" &&
    typeof value.label === "string" &&
    isFiniteScore(value.score) &&
    typeof value.explanation === "string"
  );
}

function isNeuralSignal(value: unknown): value is NeuralSignal {
  if (!isRecord(value)) return false;
  return (
    typeof value.key === "string" &&
    typeof value.label === "string" &&
    isFiniteScore(value.score) &&
    (value.direction === "up" ||
      value.direction === "down" ||
      value.direction === "neutral")
  );
}

function isRewriteSuggestion(value: unknown): value is RewriteSuggestion {
  if (!isRecord(value)) return false;
  return (
    typeof value.title === "string" &&
    typeof value.before === "string" &&
    typeof value.after === "string" &&
    typeof value.why === "string"
  );
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item));
}

function isFmriOutput(value: unknown): value is FmriOutput {
  if (!isRecord(value)) return false;
  return (
    Number.isInteger(value.segments) &&
    Number.isInteger(value.voxel_count) &&
    typeof value.global_mean_abs === "number" &&
    Number.isFinite(value.global_mean_abs) &&
    typeof value.global_peak_abs === "number" &&
    Number.isFinite(value.global_peak_abs) &&
    isNumberArray(value.temporal_trace) &&
    isNumberArray(value.temporal_peaks) &&
    isNumberArray(value.top_voxel_indices) &&
    isNumberArray(value.top_voxel_values)
  );
}

// Type guard
export function isPitchScoreReport(obj: unknown): obj is PitchScoreReport {
  if (!isRecord(obj)) return false;
  const r = obj as Record<string, unknown>;
  return (
    isFiniteScore(r.persuasion_score) &&
    typeof r.verdict === "string" &&
    typeof r.narrative === "string" &&
    Array.isArray(r.breakdown) &&
    r.breakdown.every(isBreakdownSection) &&
    Array.isArray(r.neural_signals) &&
    r.neural_signals.every(isNeuralSignal) &&
    isStringArray(r.strengths) &&
    isStringArray(r.risks) &&
    Array.isArray(r.rewrite_suggestions) &&
    r.rewrite_suggestions.every(isRewriteSuggestion) &&
    typeof r.persona_summary === "string" &&
    platformValues.includes(r.platform as Platform) &&
    typeof r.scored_at === "string" &&
    (r.fmri_output === undefined ||
      r.fmri_output === null ||
      isFmriOutput(r.fmri_output))
  );
}
