import { describe, it, expect, vi, beforeEach } from "vitest";

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

describe("POST /api/score", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
      json: async () => ({
        persuasion_score: 75,
        verdict: "Strong pitch",
        narrative: "Good analysis",
        breakdown: [],
        neural_signals: [],
        strengths: ["A"],
        risks: ["B"],
        rewrite_suggestions: [],
        persona_summary: "CTO",
        platform: "email",
        scored_at: new Date().toISOString(),
      }),
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
    expect(data.persuasion_score).toBe(75);
  });

  it("does not reject long message or persona text before proxying", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        persuasion_score: 70,
        verdict: "Accepted",
        narrative: "Long-form pitch accepted",
        breakdown: [],
        neural_signals: [],
        strengths: [],
        risks: [],
        rewrite_suggestions: [],
        persona_summary: "Long persona",
        platform: "email",
        scored_at: new Date().toISOString(),
      }),
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
