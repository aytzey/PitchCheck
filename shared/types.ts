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

// One citation-anchored link between TRIBE geometry and a published finding
export interface ResearchSynthesisItem {
  key: string;
  kind: "gap" | "strength" | "pattern";
  axis?: string;
  observation: string;
  finding: string;
  citation: string;
  source_keys?: string[];
  lever: string;
}

export interface SegmentLocation {
  segment: number;
  of: number;
  position_pct: number;
  value: number;
  percentile: number;
  text: string;
}

export interface SegmentLocalization {
  opener: SegmentLocation;
  opener_strength_percentile: number;
  closer_strength_percentile: number;
  peak: SegmentLocation;
  weakest: SegmentLocation;
  attention_cliff?: {
    drop: number;
    drop_ratio: number;
    from: SegmentLocation;
    to: SegmentLocation;
  } | null;
  basis?: string;
}

export interface ResearchSynthesis {
  items: ResearchSynthesisItem[];
  temporal_archetype?: {
    key: string;
    label: string;
    implication: string;
    lever: string;
    citation: string;
    source_keys?: string[];
  } | null;
  route_hint?: string;
  localization?: SegmentLocalization | null;
  basis?: string;
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
  context_fit_score?: number | null;
  llm_score_adjusted?: boolean;
  llm_model?: string | null;
  final_score: number;
  confidence: number;
  score_delta: number | null;
  semantic_blend_weight?: number;
  prompt_injection_risk: number | null;
  guardrails_applied: string[];
  warnings: string[];
  neuro_axes?: Record<string, NeuroAxisReport>;
  research_synthesis?: ResearchSynthesis | null;
  confidence_reasons?: string[];
  scientific_caveats?: string[];
  calibration_basis: string;
}

// One of the 1-3 highest-leverage changes, ranked by expected impact
export interface TopMove {
  priority: number;     // 1-3
  title: string;        // short imperative
  do: string;           // concrete change, ideally paste-ready copy
  because: string;      // plain-language reason
  principle?: string;   // research principle it rests on, e.g. "loss aversion"
}

// Semantic context-fit diagnostics from the LLM (bounded by the neural band)
export interface ContextFitFacet {
  score: number;        // 0-100
  note: string;
}

export interface ContextFitReport {
  persona_pain_alignment: ContextFitFacet;
  objection_coverage: ContextFitFacet;
  proof_credibility: ContextFitFacet;
  cta_ease: ContextFitFacet;
  channel_fit: ContextFitFacet;
  decision_driver: string;
  top_unaddressed_objection: string;
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
  top_moves?: TopMove[];            // 1-3 highest-leverage changes, ranked
  context_fit?: ContextFitReport | null; // semantic context-fit diagnostics
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
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
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

const MAX_REPORT_TEXT_CHARS = 2_000;
const MAX_SHORT_TEXT_CHARS = 500;
const MAX_LIST_ITEMS = 8;
const MAX_BREAKDOWN_ITEMS = 12;
const MAX_SIGNAL_ITEMS = 12;
const MAX_REWRITE_ITEMS = 8;
const MAX_TEMPORAL_POINTS = 512;

function parseFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeScore(value: unknown): number | null {
  const parsed = parseFiniteNumber(value);
  return parsed === null ? null : Math.round(clamp(parsed, 0, 100));
}

function normalizeInteger(value: unknown): number | null {
  const parsed = parseFiniteNumber(value);
  return parsed === null ? null : Math.max(0, Math.round(parsed));
}

function normalizeUnit(value: unknown): number | null {
  const parsed = parseFiniteNumber(value);
  if (parsed === null) return null;
  return clamp(parsed > 1 ? parsed / 100 : parsed, 0, 1);
}

function normalizeText(value: unknown, fallback = "", maxChars = MAX_REPORT_TEXT_CHARS) {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return (trimmed || fallback).slice(0, maxChars);
}

function normalizeOptionalText(value: unknown, maxChars = MAX_REPORT_TEXT_CHARS) {
  return typeof value === "string" ? value.trim().slice(0, maxChars) : "";
}

function normalizeKey(value: unknown, fallback: string) {
  const key = normalizeOptionalText(value, 120)
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return key || fallback;
}

function labelFromKey(key: string) {
  return key
    .replace(/[-_]+/g, " ")
    .replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
}

function normalizeStringArray(value: unknown, maxItems = MAX_LIST_ITEMS, maxChars = MAX_SHORT_TEXT_CHARS) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => normalizeOptionalText(item, maxChars))
    .filter(Boolean)
    .slice(0, maxItems);
}

function normalizeNumberArray(value: unknown, maxItems = MAX_TEMPORAL_POINTS) {
  if (!Array.isArray(value)) return [];
  const numbers: number[] = [];
  for (const item of value) {
    const parsed = parseFiniteNumber(item);
    if (parsed !== null) numbers.push(parsed);
    if (numbers.length >= maxItems) break;
  }
  return numbers;
}

function normalizeNumberRecord(value: unknown) {
  if (!isRecord(value)) return {};
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(value).slice(0, 64)) {
    const parsed = parseFiniteNumber(item);
    if (parsed !== null) result[key.slice(0, 120)] = parsed;
  }
  return result;
}

function normalizeStringRecordArray(value: unknown) {
  if (!Array.isArray(value)) return undefined;
  const records: Array<Record<string, string>> = [];
  for (const item of value) {
    if (!isRecord(item)) continue;
    const record: Record<string, string> = {};
    for (const [key, entry] of Object.entries(item).slice(0, 12)) {
      const text = normalizeOptionalText(entry, MAX_SHORT_TEXT_CHARS);
      if (text) record[key.slice(0, 120)] = text;
    }
    if (Object.keys(record).length) records.push(record);
    if (records.length >= MAX_LIST_ITEMS) break;
  }
  return records.length ? records : undefined;
}

export function normalizePlatform(value: unknown): Platform {
  const platform = typeof value === "string" ? value.trim().toLowerCase() : "";
  return platformValues.includes(platform as Platform) ? (platform as Platform) : "general";
}

function normalizeTimestamp(value: unknown) {
  const text = normalizeOptionalText(value, 120);
  return text && !Number.isNaN(Date.parse(text)) ? text : new Date().toISOString();
}

function normalizeDirection(value: unknown, key: string, score: number): NeuralSignal["direction"] {
  const direction = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (direction === "up" || direction === "down" || direction === "neutral") return direction;
  if (key === "cognitive_friction") {
    if (score <= 35) return "down";
    if (score >= 65) return "up";
    return "neutral";
  }
  if (score >= 70) return "up";
  if (score <= 40) return "down";
  return "neutral";
}

function normalizeBreakdownSection(value: unknown, index: number): BreakdownSection | null {
  if (!isRecord(value)) return null;
  const score = normalizeScore(value.score);
  if (score === null) return null;
  const rawLabel = normalizeOptionalText(value.label, MAX_SHORT_TEXT_CHARS);
  const key = normalizeKey(value.key, normalizeKey(rawLabel, `section_${index + 1}`));
  return {
    key,
    label: rawLabel || labelFromKey(key),
    score,
    explanation: normalizeText(value.explanation, "No explanation provided.", MAX_REPORT_TEXT_CHARS),
  };
}

function normalizeNeuralSignal(value: unknown, index: number): NeuralSignal | null {
  if (!isRecord(value)) return null;
  const score = normalizeScore(value.score);
  if (score === null) return null;
  const rawLabel = normalizeOptionalText(value.label, MAX_SHORT_TEXT_CHARS);
  const key = normalizeKey(value.key, normalizeKey(rawLabel, `signal_${index + 1}`));
  return {
    key,
    label: rawLabel || labelFromKey(key),
    score,
    direction: normalizeDirection(value.direction, key, score),
  };
}

function normalizeRewriteSuggestion(value: unknown, index: number): RewriteSuggestion | null {
  if (!isRecord(value)) return null;
  const before = normalizeOptionalText(value.before, MAX_REPORT_TEXT_CHARS);
  const after = normalizeOptionalText(value.after, MAX_REPORT_TEXT_CHARS);
  const why = normalizeOptionalText(value.why, MAX_REPORT_TEXT_CHARS);
  if (!before && !after && !why) return null;
  return {
    title: normalizeText(value.title, `Suggestion ${index + 1}`, MAX_SHORT_TEXT_CHARS),
    before,
    after,
    why,
  };
}

function normalizeTopMove(value: unknown, index: number): TopMove | null {
  if (!isRecord(value)) return null;
  const title = normalizeOptionalText(value.title, MAX_SHORT_TEXT_CHARS);
  const action = normalizeOptionalText(value.do, MAX_REPORT_TEXT_CHARS);
  if (!title || !action) return null;
  return {
    priority: clamp(normalizeInteger(value.priority) ?? index + 1, 1, 3),
    title,
    do: action,
    because: normalizeOptionalText(value.because, MAX_REPORT_TEXT_CHARS),
    principle: normalizeOptionalText(value.principle, MAX_SHORT_TEXT_CHARS) || undefined,
  };
}

function normalizeFacet(value: unknown): ContextFitFacet | null {
  if (!isRecord(value)) return null;
  const score = normalizeScore(value.score);
  if (score === null) return null;
  return {
    score,
    note: normalizeText(value.note, "No note provided.", MAX_REPORT_TEXT_CHARS),
  };
}

function normalizeContextFit(value: unknown): ContextFitReport | null {
  if (!isRecord(value)) return null;
  const personaPainAlignment = normalizeFacet(value.persona_pain_alignment);
  const objectionCoverage = normalizeFacet(value.objection_coverage);
  const proofCredibility = normalizeFacet(value.proof_credibility);
  const ctaEase = normalizeFacet(value.cta_ease);
  const channelFit = normalizeFacet(value.channel_fit);
  if (!personaPainAlignment || !objectionCoverage || !proofCredibility || !ctaEase || !channelFit) {
    return null;
  }
  return {
    persona_pain_alignment: personaPainAlignment,
    objection_coverage: objectionCoverage,
    proof_credibility: proofCredibility,
    cta_ease: ctaEase,
    channel_fit: channelFit,
    decision_driver: normalizeOptionalText(value.decision_driver, MAX_REPORT_TEXT_CHARS),
    top_unaddressed_objection: normalizeOptionalText(value.top_unaddressed_objection, MAX_REPORT_TEXT_CHARS),
  };
}

function normalizeFmriOutput(value: unknown): FmriOutput | null {
  if (!isRecord(value)) return null;
  const segments = normalizeInteger(value.segments);
  const voxelCount = normalizeInteger(value.voxel_count);
  const globalMeanAbs = parseFiniteNumber(value.global_mean_abs);
  const globalPeakAbs = parseFiniteNumber(value.global_peak_abs);
  if (segments === null || voxelCount === null || globalMeanAbs === null || globalPeakAbs === null) return null;
  const temporalTraceBasis =
    value.temporal_trace_basis === "real_time_seconds" || value.temporal_trace_basis === "synthetic_word_order"
      ? value.temporal_trace_basis
      : undefined;
  return {
    segments,
    voxel_count: voxelCount,
    global_mean_abs: globalMeanAbs,
    global_peak_abs: globalPeakAbs,
    temporal_trace: normalizeNumberArray(value.temporal_trace),
    temporal_peaks: normalizeNumberArray(value.temporal_peaks),
    top_voxel_indices: normalizeNumberArray(value.top_voxel_indices, 24).map((item) => Math.round(item)),
    top_voxel_values: normalizeNumberArray(value.top_voxel_values, 24),
    response_kind: normalizeOptionalText(value.response_kind, MAX_SHORT_TEXT_CHARS) || undefined,
    prediction_subject_basis: normalizeOptionalText(value.prediction_subject_basis, MAX_SHORT_TEXT_CHARS) || undefined,
    cortical_mesh: normalizeOptionalText(value.cortical_mesh, MAX_SHORT_TEXT_CHARS) || undefined,
    hemodynamic_lag_seconds: parseFiniteNumber(value.hemodynamic_lag_seconds) ?? undefined,
    temporal_trace_basis: temporalTraceBasis,
    temporal_segment_label: normalizeOptionalText(value.temporal_segment_label, MAX_SHORT_TEXT_CHARS) || undefined,
    temporal_trace_note: normalizeOptionalText(value.temporal_trace_note, MAX_REPORT_TEXT_CHARS) || undefined,
  };
}

function normalizePersuasionEvidence(value: unknown): PersuasionEvidence | null {
  if (!isRecord(value)) return null;
  const evidence: PersuasionEvidence = {
    overall_text_score: normalizeScore(value.overall_text_score) ?? 0,
    feature_scores: normalizeNumberRecord(value.feature_scores),
    detected_strategies: normalizeStringArray(value.detected_strategies),
    missing_elements: normalizeStringArray(value.missing_elements),
    warnings: normalizeStringArray(value.warnings),
    prompt_injection_risk: normalizeScore(value.prompt_injection_risk) ?? 0,
    readability: normalizeNumberRecord(value.readability),
    matched_persona_terms: normalizeStringArray(value.matched_persona_terms),
  };
  const strategyCounts = normalizeNumberRecord(value.strategy_counts);
  if (Object.keys(strategyCounts).length) evidence.strategy_counts = strategyCounts;
  evidence.platform = normalizePlatform(value.platform);
  evidence.methodology_version = normalizeOptionalText(value.methodology_version, MAX_SHORT_TEXT_CHARS) || undefined;
  evidence.methodology = normalizeOptionalText(value.methodology, MAX_REPORT_TEXT_CHARS) || undefined;
  const inputMetadata = normalizeNumberRecord(value.input_metadata);
  if (Object.keys(inputMetadata).length) evidence.input_metadata = inputMetadata;
  const researchBasis = normalizeStringArray(value.research_basis);
  if (researchBasis.length) evidence.research_basis = researchBasis;
  evidence.research_sources = normalizeStringRecordArray(value.research_sources);
  evidence.calibration_quality = isRecord(value.calibration_quality) ? value.calibration_quality : undefined;
  return evidence;
}

function normalizeNeuroAxes(value: unknown) {
  if (!isRecord(value)) return undefined;
  const axes: Record<string, NeuroAxisReport> = {};
  for (const [axisKey, item] of Object.entries(value).slice(0, 16)) {
    if (!isRecord(item)) continue;
    const key = normalizeKey(axisKey, `axis_${Object.keys(axes).length + 1}`);
    axes[key] = {
      label: normalizeText(item.label, labelFromKey(key), MAX_SHORT_TEXT_CHARS),
      score: normalizeScore(item.score) ?? 0,
      weight: parseFiniteNumber(item.weight) ?? 0,
      contribution: parseFiniteNumber(item.contribution) ?? 0,
      analogue: normalizeText(item.analogue, "", MAX_SHORT_TEXT_CHARS),
      evidence: normalizeStringArray(item.evidence),
      caveat: normalizeText(item.caveat, "", MAX_REPORT_TEXT_CHARS),
      source_keys: normalizeStringArray(item.source_keys),
      unsupported_by_text: typeof item.unsupported_by_text === "boolean" ? item.unsupported_by_text : undefined,
    };
  }
  return Object.keys(axes).length ? axes : undefined;
}

function normalizeRobustnessReport(value: unknown): RobustnessReport | null {
  if (!isRecord(value)) return null;
  const neuralScore = normalizeScore(value.neural_score) ?? normalizeScore(value.final_score) ?? 0;
  const finalScore = normalizeScore(value.final_score) ?? neuralScore;
  const report: RobustnessReport = {
    neural_score: neuralScore,
    text_score: parseFiniteNumber(value.text_score),
    evidence_score: normalizeScore(value.evidence_score) ?? finalScore,
    llm_score: parseFiniteNumber(value.llm_score),
    final_score: finalScore,
    confidence: normalizeUnit(value.confidence) ?? 0,
    score_delta: parseFiniteNumber(value.score_delta),
    prompt_injection_risk: parseFiniteNumber(value.prompt_injection_risk),
    guardrails_applied: normalizeStringArray(value.guardrails_applied),
    warnings: normalizeStringArray(value.warnings),
    calibration_basis: normalizeText(value.calibration_basis, "Unavailable.", MAX_REPORT_TEXT_CHARS),
  };
  const neuralPriorScore = normalizeScore(value.neural_prior_score);
  if (neuralPriorScore !== null) report.neural_prior_score = neuralPriorScore;
  const qualityAdjustedNeuralScore = normalizeScore(value.quality_adjusted_neural_score);
  if (qualityAdjustedNeuralScore !== null) report.quality_adjusted_neural_score = qualityAdjustedNeuralScore;
  const predictionQualityWeight = normalizeUnit(value.prediction_quality_weight);
  if (predictionQualityWeight !== null) report.prediction_quality_weight = predictionQualityWeight;
  report.raw_llm_score = parseFiniteNumber(value.raw_llm_score);
  report.context_fit_score = parseFiniteNumber(value.context_fit_score);
  if (typeof value.llm_score_adjusted === "boolean") report.llm_score_adjusted = value.llm_score_adjusted;
  report.llm_model = normalizeOptionalText(value.llm_model, MAX_SHORT_TEXT_CHARS) || null;
  const semanticBlendWeight = normalizeUnit(value.semantic_blend_weight);
  if (semanticBlendWeight !== null) report.semantic_blend_weight = semanticBlendWeight;
  report.neuro_axes = normalizeNeuroAxes(value.neuro_axes);
  report.research_synthesis = isRecord(value.research_synthesis)
    ? (value.research_synthesis as unknown as ResearchSynthesis)
    : null;
  const confidenceReasons = normalizeStringArray(value.confidence_reasons);
  if (confidenceReasons.length) report.confidence_reasons = confidenceReasons;
  const scientificCaveats = normalizeStringArray(value.scientific_caveats);
  if (scientificCaveats.length) report.scientific_caveats = scientificCaveats;
  return report;
}

function normalizeItems<T>(
  value: unknown,
  maxItems: number,
  normalizeItem: (item: unknown, index: number) => T | null,
) {
  if (!Array.isArray(value)) return [];
  const items: T[] = [];
  for (const [index, item] of value.entries()) {
    const normalized = normalizeItem(item, index);
    if (normalized) items.push(normalized);
    if (items.length >= maxItems) break;
  }
  return items;
}

export function normalizePitchScoreReport(obj: unknown): PitchScoreReport | null {
  if (!isRecord(obj)) return null;
  const score = normalizeScore(obj.persuasion_score);
  if (score === null) return null;

  return {
    persuasion_score: score,
    verdict: normalizeText(obj.verdict, "No verdict provided.", MAX_SHORT_TEXT_CHARS),
    narrative: normalizeText(obj.narrative, "No narrative provided.", MAX_REPORT_TEXT_CHARS),
    breakdown: normalizeItems(obj.breakdown, MAX_BREAKDOWN_ITEMS, normalizeBreakdownSection),
    neural_signals: normalizeItems(obj.neural_signals, MAX_SIGNAL_ITEMS, normalizeNeuralSignal),
    strengths: normalizeStringArray(obj.strengths),
    risks: normalizeStringArray(obj.risks),
    rewrite_suggestions: normalizeItems(obj.rewrite_suggestions, MAX_REWRITE_ITEMS, normalizeRewriteSuggestion),
    persona_summary: normalizeText(obj.persona_summary, "Persona not summarized.", MAX_REPORT_TEXT_CHARS),
    top_moves: normalizeItems(obj.top_moves, 3, normalizeTopMove),
    context_fit: normalizeContextFit(obj.context_fit),
    fmri_output: normalizeFmriOutput(obj.fmri_output),
    persuasion_evidence: normalizePersuasionEvidence(obj.persuasion_evidence),
    robustness: normalizeRobustnessReport(obj.robustness),
    platform: normalizePlatform(obj.platform),
    scored_at: normalizeTimestamp(obj.scored_at),
  };
}
