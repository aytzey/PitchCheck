import { TransportMode } from "./types";

export function normalizeBaseUrl(raw: string): string {
  return raw.trim().replace(/\/$/, "");
}

export function parseHttpUrl(raw: string): URL | null {
  try {
    const value = normalizeBaseUrl(raw);
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url;
  } catch {
    return null;
  }
}

export function isProbablyHttpUrl(raw: string): boolean {
  return Boolean(parseHttpUrl(raw));
}

export function isLocalRuntimeUrl(raw: string): boolean {
  const url = parseHttpUrl(raw);
  if (!url) return false;
  return url.hostname === "localhost" || url.hostname === "127.0.0.1";
}

export function scoreEndpoints(baseUrl: string, mode: TransportMode): string[] {
  if (mode === "next-api") return [`${baseUrl}/api/score`];
  if (mode === "direct") return [`${baseUrl}/score`];
  return [`${baseUrl}/api/score`, `${baseUrl}/score`];
}

export function healthEndpoints(baseUrl: string, mode: TransportMode): string[] {
  if (mode === "next-api") return [`${baseUrl}/api/health`];
  if (mode === "direct") return [`${baseUrl}/health`];
  return [`${baseUrl}/api/health`, `${baseUrl}/health`];
}

export function createRequestId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `pc_${Date.now().toString(36)}_${rand}`;
}

export function createIdempotencyKey(parts: string[]): string {
  const normalized = parts.map((part) => part.trim().toLowerCase()).join("|");
  let hash = 0;
  for (let i = 0; i < normalized.length; i += 1) {
    hash = (hash << 5) - hash + normalized.charCodeAt(i);
    hash |= 0;
  }
  return `pcid_${Math.abs(hash)}`;
}

export async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs = 18000,
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}
