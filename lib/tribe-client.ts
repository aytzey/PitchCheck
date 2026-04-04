const TRIBE_SERVICE_URL = (
  process.env.TRIBE_SERVICE_URL ?? "http://127.0.0.1:8090"
).replace(/\/$/, "");

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
}): Promise<TribeScoreResponse> {
  try {
    const response = await fetch(`${TRIBE_SERVICE_URL}/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      cache: "no-store",
    });
    const data = await response
      .json()
      .catch(() => ({ error: "Invalid response from scoring service" }));
    return { ok: response.ok, data, status: response.status };
  } catch {
    return { ok: false, error: "Scoring service is unreachable.", status: 503 };
  }
}

export async function checkTribeHealth(): Promise<{
  ok: boolean;
  detail?: unknown;
}> {
  try {
    const response = await fetch(`${TRIBE_SERVICE_URL}/health`, {
      cache: "no-store",
    });
    const data = await response.json().catch(() => ({}));
    return { ok: response.ok, detail: data };
  } catch {
    return { ok: false, detail: "Scoring service unreachable" };
  }
}
