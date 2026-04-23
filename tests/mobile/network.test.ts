import { describe, expect, it, vi } from "vitest";
import {
  createIdempotencyKey,
  createRequestId,
  healthEndpoints,
  isLocalRuntimeUrl,
  isProbablyHttpUrl,
  normalizeBaseUrl,
  scoreEndpoints,
  fetchWithTimeout,
} from "@/mobile/src/network";

describe("mobile network helpers", () => {
  it("normalizes url trailing slash", () => {
    expect(normalizeBaseUrl("https://x.y/z/")).toBe("https://x.y/z");
  });

  it("validates http urls", () => {
    expect(isProbablyHttpUrl("https://demo.example")).toBe(true);
    expect(isProbablyHttpUrl("http://127.0.0.1:8090")).toBe(true);
    expect(isProbablyHttpUrl("ftp://x")).toBe(false);
  });

  it("detects localhost runtimes", () => {
    expect(isLocalRuntimeUrl("http://127.0.0.1:8090")).toBe(true);
    expect(isLocalRuntimeUrl("http://localhost:3000")).toBe(true);
    expect(isLocalRuntimeUrl("https://cloud.example")).toBe(false);
  });

  it("builds score endpoints by transport", () => {
    expect(scoreEndpoints("https://a", "auto")).toEqual(["https://a/api/score", "https://a/score"]);
    expect(scoreEndpoints("https://a", "next-api")).toEqual(["https://a/api/score"]);
    expect(scoreEndpoints("https://a", "direct")).toEqual(["https://a/score"]);
  });

  it("builds health endpoints by transport", () => {
    expect(healthEndpoints("https://a", "auto")).toEqual(["https://a/api/health", "https://a/health"]);
    expect(healthEndpoints("https://a", "next-api")).toEqual(["https://a/api/health"]);
    expect(healthEndpoints("https://a", "direct")).toEqual(["https://a/health"]);
  });

  it("creates request id and deterministic idempotency keys", () => {
    expect(createRequestId()).toMatch(/^pc_/);
    const k1 = createIdempotencyKey(["A", "B"]);
    const k2 = createIdempotencyKey(["a", "b"]);
    expect(k1).toEqual(k2);
  });

  it("passes timeout signal to fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);
    await fetchWithTimeout("https://a", { method: "GET" }, 50);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, options] = fetchMock.mock.calls[0] as [string, { signal?: AbortSignal }];
    expect(options.signal).toBeDefined();
    vi.unstubAllGlobals();
  });
});
