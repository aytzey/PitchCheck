import * as SecureStore from "expo-secure-store";
import { PitchScoreReport, RuntimeProbe, RuntimeSettings } from "./types";
import { normalizePitchScoreReport } from "./report";
import {
  createIdempotencyKey,
  createRequestId,
  fetchWithTimeout,
  healthEndpoints,
  isLocalRuntimeUrl,
  normalizeBaseUrl,
  parseHttpUrl,
  scoreEndpoints,
} from "./network";
import { logRuntimeEvent } from "./telemetry";

const KEY = "pitchcheck-mobile-settings";
const TIMEOUT_MS = 22000;

export const defaultSettings: RuntimeSettings = {
  runtime: "pitchserver",
  pitchserverUrl: "http://127.0.0.1:8090",
  vastUrl: "",
  vastApiKey: "",
  openRouterModel: "anthropic/claude-sonnet-4.6",
  transportMode: "auto",
  strictTransportSecurity: true,
};

export async function loadSettings(): Promise<RuntimeSettings> {
  const raw = await SecureStore.getItemAsync(KEY);
  if (!raw) return defaultSettings;
  try {
    const parsed = JSON.parse(raw) as Partial<RuntimeSettings>;
    return { ...defaultSettings, ...parsed };
  } catch {
    return defaultSettings;
  }
}

export async function saveSettings(settings: RuntimeSettings): Promise<void> {
  await SecureStore.setItemAsync(KEY, JSON.stringify(settings));
}

function runtimeBaseUrl(settings: RuntimeSettings) {
  return settings.runtime === "pitchserver"
    ? settings.pitchserverUrl.trim()
    : settings.vastUrl.trim();
}

function buildAuthHeaders(settings: RuntimeSettings): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (settings.runtime === "vast" && settings.vastApiKey.trim()) {
    headers.Authorization = `Bearer ${settings.vastApiKey.trim()}`;
  }
  return headers;
}

function validateRuntimeUrl(url: string, settings: RuntimeSettings) {
  if (!url) return "Runtime URL is required.";
  const parsed = parseHttpUrl(url);
  if (!parsed) return "Runtime URL must start with http:// or https://";

  if (
    settings.strictTransportSecurity &&
    parsed.protocol === "http:" &&
    !isLocalRuntimeUrl(url)
  ) {
    return "HTTPS is required for non-local runtimes when strict security is enabled.";
  }

  return null;
}

async function postScore(
  url: string,
  headers: Record<string, string>,
  payload: Record<string, string>,
  requestId: string,
): Promise<PitchScoreReport> {
  const response = await fetchWithTimeout(
    url,
    {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    },
    TIMEOUT_MS,
  );

  const data = (await response.json().catch(() => null)) as Record<string, unknown> | null;

  if (!response.ok) {
    const reason = data && typeof data.error === "string" ? data.error : `Scoring failed (${response.status}).`;
    logRuntimeEvent({
      at: new Date().toISOString(),
      level: "error",
      event: "score_http_error",
      requestId,
      details: { status: response.status, url },
    });
    throw new Error(reason);
  }

  return normalizePitchScoreReport(data);
}

async function retryOnce<T>(fn: () => Promise<T>): Promise<T> {
  const attempts = [0, 450, 900];
  let lastError: unknown = null;

  for (const wait of attempts) {
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    try {
      return await fn();
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

export async function probeRuntime(settings: RuntimeSettings): Promise<RuntimeProbe> {
  const baseUrl = normalizeBaseUrl(runtimeBaseUrl(settings));
  const invalidReason = validateRuntimeUrl(baseUrl, settings);
  if (invalidReason) {
    return {
      ok: false,
      status: null,
      endpointTried: "",
      detail: invalidReason,
    };
  }

  const headers = buildAuthHeaders(settings);

  for (const endpoint of healthEndpoints(baseUrl, settings.transportMode)) {
    try {
      const response = await fetchWithTimeout(endpoint, { headers }, 10000);
      if (response.ok) {
        return {
          ok: true,
          status: response.status,
          endpointTried: endpoint,
          detail: "Runtime reachable.",
        };
      }
      return {
        ok: false,
        status: response.status,
        endpointTried: endpoint,
        detail: `Runtime responded with ${response.status}.`,
      };
    } catch {
      continue;
    }
  }

  return {
    ok: false,
    status: null,
    endpointTried: `${baseUrl}/health`,
    detail: "Could not reach runtime.",
  };
}

export async function scorePitch(
  settings: RuntimeSettings,
  message: string,
  persona: string,
  platform: string,
): Promise<PitchScoreReport> {
  const baseUrl = normalizeBaseUrl(runtimeBaseUrl(settings));
  const invalidReason = validateRuntimeUrl(baseUrl, settings);
  if (invalidReason) {
    throw new Error(invalidReason);
  }

  const requestId = createRequestId();
  const idempotencyKey = createIdempotencyKey([
    settings.runtime,
    baseUrl,
    platform,
    message,
    persona,
    settings.openRouterModel,
  ]);

  const headers = {
    ...buildAuthHeaders(settings),
    "X-Request-Id": requestId,
    "X-Idempotency-Key": idempotencyKey,
  };
  const payload = {
    message,
    persona,
    platform,
    openRouterModel: settings.openRouterModel,
  };

  logRuntimeEvent({
    at: new Date().toISOString(),
    level: "info",
    event: "score_start",
    requestId,
    details: { runtime: settings.runtime, transportMode: settings.transportMode },
  });

  let lastError: Error | null = null;
  for (const endpoint of scoreEndpoints(baseUrl, settings.transportMode)) {
    try {
      const result = await retryOnce(() => postScore(endpoint, headers, payload, requestId));
      logRuntimeEvent({
        at: new Date().toISOString(),
        level: "info",
        event: "score_success",
        requestId,
        details: { endpoint, score: result.persuasion_score },
      });
      return result;
    } catch (caught) {
      lastError = caught instanceof Error ? caught : new Error("Scoring failed.");
      logRuntimeEvent({
        at: new Date().toISOString(),
        level: "warn",
        event: "score_endpoint_failed",
        requestId,
        details: { endpoint, reason: lastError.message },
      });
    }
  }

  logRuntimeEvent({
    at: new Date().toISOString(),
    level: "error",
    event: "score_failed",
    requestId,
    details: { error: lastError?.message ?? "Scoring failed." },
  });
  throw lastError ?? new Error("Scoring failed.");
}
