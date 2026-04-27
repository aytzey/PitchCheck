import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// We need to test the route handler directly
// Import after mocking
import { POST } from "@/app/api/score/route";

function makeRequest(body: unknown): Request {
  return new Request("http://localhost:3000/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function makeReport(score = 75, platform = "email") {
  return {
    persuasion_score: score,
    verdict: "Accepted",
    narrative: "Accepted",
    breakdown: [],
    neural_signals: [],
    strengths: [],
    risks: [],
    rewrite_suggestions: [],
    persona_summary: "Persona",
    platform,
    scored_at: new Date().toISOString(),
  };
}

function setEnv(name: string, value: string | undefined) {
  const env = process.env as Record<string, string | undefined>;
  if (value === undefined) {
    delete env[name];
    return;
  }
  env[name] = value;
}

describe("POST /api/score", () => {
  const originalNodeEnv = process.env.NODE_ENV;
  const originalApiKey = process.env.SCORE_API_KEY;

  beforeEach(() => {
    vi.clearAllMocks();
    setEnv("NODE_ENV", originalNodeEnv);
    setEnv("SCORE_API_KEY", originalApiKey);
  });

  afterEach(() => {
    setEnv("NODE_ENV", originalNodeEnv);
    setEnv("SCORE_API_KEY", originalApiKey);
  });

  it("returns 503 in production when SCORE_API_KEY is not configured", async () => {
    setEnv("NODE_ENV", "production");
    setEnv("SCORE_API_KEY", undefined);

    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
      }),
    );

    expect(res.status).toBe(503);
  });

  it("returns 401 in production when API key is missing", async () => {
    setEnv("NODE_ENV", "production");
    setEnv("SCORE_API_KEY", "test-key");

    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
      }),
    );

    expect(res.status).toBe(401);
  });

  it("returns 400 when message is missing", async () => {
    const res = await POST(makeRequest({ persona: "CTO at startup" }));
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toContain("message");
  });

  it("returns 400 when message is too short", async () => {
    const res = await POST(
      makeRequest({ message: "Hi", persona: "CTO at startup" }),
    );
    expect(res.status).toBe(400);
  });

  it("returns 400 when persona is missing", async () => {
    const res = await POST(
      makeRequest({ message: "This is a valid pitch message for testing" }),
    );
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toContain("persona");
  });

  it("proxies valid request to TRIBE service", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ report: makeReport(75, "email") }),
    });

    const res = await POST(
      makeRequest({
        message:
          "Our platform reduces deployment time by 80% for enterprise teams",
        persona: "CTO, 40, startup background",
        platform: "email",
      }),
    );

    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.report.persuasion_score).toBe(75);
  });

  it("normalizes platform casing and drops invalid model ids", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ report: makeReport(72, "linkedin") }),
    });

    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
        platform: "LinkedIn",
        openRouterModel: "bad model\ninjected",
      }),
    );

    expect(res.status).toBe(200);
    const forwarded = JSON.parse(String(mockFetch.mock.calls[0][1]?.body));
    expect(forwarded.platform).toBe("linkedin");
    expect(forwarded.openRouterModel).toBeUndefined();
  });

  it("does not reject long message or persona text before proxying", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ report: makeReport(70, "email") }),
    });

    const res = await POST(
      makeRequest({
        message: "A".repeat(6000),
        persona: "Technical buyer ".repeat(120),
        platform: "email",
        openRouterModel: "openai/gpt-5.4",
      }),
    );

    expect(res.status).toBe(200);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const forwarded = JSON.parse(String(mockFetch.mock.calls[0][1]?.body));
    expect(forwarded.openRouterModel).toBe("openai/gpt-5.4");
  });

  it("rejects oversized message text before proxying", async () => {
    const res = await POST(
      makeRequest({
        message: "A".repeat(30_001),
        persona: "CTO at startup, technical",
      }),
    );

    expect(res.status).toBe(413);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("rejects oversized request bodies before parsing full JSON", async () => {
    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
        ignoredPadding: "A".repeat(140_000),
      }),
    );

    expect(res.status).toBe(413);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("rejects malformed successful TRIBE responses", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ report: { persuasion_score: "bad" } }),
    });

    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
      }),
    );

    expect(res.status).toBe(502);
    expect((await res.json()).error).toContain("invalid report");
  });

  it("returns 503 when TRIBE service is down", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

    const res = await POST(
      makeRequest({
        message: "Our platform reduces deployment time by 80%",
        persona: "CTO at startup, technical",
      }),
    );
    expect(res.status).toBe(503);
  });
});
