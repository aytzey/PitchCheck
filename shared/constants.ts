export const PLATFORMS = {
  email: { label: "Email", icon: "mail" },
  linkedin: { label: "LinkedIn", icon: "linkedin" },
  "cold-call-script": { label: "Cold Call Script", icon: "phone" },
  "landing-page": { label: "Landing Page", icon: "globe" },
  "ad-copy": { label: "Ad Copy", icon: "megaphone" },
  general: { label: "General", icon: "message-square" },
} as const;

export const API_PATHS = {
  score: "/api/score",
  health: "/api/health",
} as const;

export const BREAKDOWN_KEYS = [
  "emotional_resonance",
  "clarity",
  "urgency",
  "credibility",
  "personalization_fit",
] as const;

export const NEURAL_SIGNAL_KEYS = [
  "emotional_engagement",
  "personal_relevance",
  "social_proof_potential",
  "memorability",
  "attention_capture",
  "cognitive_friction",
] as const;
