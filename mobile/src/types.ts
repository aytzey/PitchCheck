export const platformValues = [
  "email",
  "linkedin",
  "cold-call-script",
  "landing-page",
  "ad-copy",
  "general",
] as const;

export type Platform = (typeof platformValues)[number];
export type RuntimeKind = "pitchserver" | "vast";
export type TransportMode = "auto" | "next-api" | "direct";

export interface PitchScoreReport {
  persuasion_score: number;
  verdict: string;
  narrative: string;
  strengths: string[];
  risks: string[];
  platform: Platform;
  scored_at: string;
}

export interface RuntimeSettings {
  runtime: RuntimeKind;
  pitchserverUrl: string;
  vastUrl: string;
  vastApiKey: string;
  openRouterModel: string;
  transportMode: TransportMode;
  strictTransportSecurity: boolean;
}

export interface RuntimeProbe {
  ok: boolean;
  status: number | null;
  endpointTried: string;
  detail: string;
}
