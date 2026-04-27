const TRIBE_SERVICE_URL = (
  process.env.TRIBE_SERVICE_URL ?? "http://127.0.0.1:8090"
).replace(/\/$/, "");
const TRIBE_SCORE_TIMEOUT_MS = envInt("TRIBE_SERVICE_SCORE_TIMEOUT_MS", 920_000, 1_000);
const TRIBE_HEALTH_TIMEOUT_MS = envInt("TRIBE_SERVICE_HEALTH_TIMEOUT_MS", 5_000, 500);

function envInt(name: string, fallback: number, minimum: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<{ response: Response; data: unknown }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const data = await response
      .json()
      .catch(() => ({ error: "Invalid response from scoring service" }));
    return { response, data };
  } finally {
    clearTimeout(timeout);
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

export interface TribeScoreResponse {
  ok: boolean;
  data?: unknown;
  error?: string;
  status: number;
}

export async function scorePitch(request: {
  message: string;
  persona: string;
  platform: string;
  openRouterModel?: string;
}): Promise<TribeScoreResponse> {
  try {
    const { response, data } = await fetchJsonWithTimeout(
      `${TRIBE_SERVICE_URL}/score`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        cache: "no-store",
      },
      TRIBE_SCORE_TIMEOUT_MS,
    );
    return { ok: response.ok, data, status: response.status };
  } catch (error) {
    return {
      ok: false,
      error: isAbortError(error)
        ? "Scoring service timed out."
        : "Scoring service is unreachable.",
      status: 503,
    };
  }
}

export async function checkTribeHealth(): Promise<{
  ok: boolean;
  detail?: unknown;
}> {
  try {
    const { response, data } = await fetchJsonWithTimeout(
      `${TRIBE_SERVICE_URL}/health`,
      { cache: "no-store" },
      TRIBE_HEALTH_TIMEOUT_MS,
    );
    return { ok: response.ok, detail: data };
  } catch (error) {
    return {
      ok: false,
      detail: isAbortError(error)
        ? "Scoring service health check timed out"
        : "Scoring service unreachable",
    };
  }
}
