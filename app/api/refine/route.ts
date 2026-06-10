import { refinePitch } from "@/lib/tribe-client";
import { platformValues, type Platform } from "@/shared/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_MESSAGE_CHARS = parseEnvInt("PITCHCHECK_MAX_MESSAGE_CHARS", 30_000, 10);
const MAX_PERSONA_CHARS = parseEnvInt("PITCHCHECK_MAX_PERSONA_CHARS", 5_000, 5);
const MAX_REQUEST_BODY_BYTES = parseEnvInt(
  "PITCHCHECK_MAX_REQUEST_BODY_BYTES",
  128 * 1024,
  1024,
);

function parseEnvInt(name: string, fallback: number, minimum: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}

function isAuthorized(request: Request): { ok: true } | { ok: false; status: number; error: string } {
  if (process.env.NODE_ENV !== "production") {
    return { ok: true };
  }

  const configuredApiKey = process.env.SCORE_API_KEY?.trim();
  if (!configuredApiKey) {
    return {
      ok: false,
      status: 503,
      error: "Server misconfiguration: SCORE_API_KEY is not set.",
    };
  }

  const headerApiKey = request.headers.get("x-api-key")?.trim();
  const bearerToken = request.headers
    .get("authorization")
    ?.match(/^Bearer\s+(.+)$/i)?.[1]
    ?.trim();

  if (headerApiKey === configuredApiKey || bearerToken === configuredApiKey) {
    return { ok: true };
  }

  return { ok: false, status: 401, error: "Unauthorized." };
}

function sanitizeSuggestions(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .map((item) => item.trim().slice(0, 500))
    .slice(0, 12);
}

function sanitizeClarificationAnswers(
  value: unknown,
): Array<{ id: string; question: string; answer: string }> {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (item): item is { id?: unknown; question?: unknown; answer?: unknown } =>
        Boolean(item) && typeof item === "object",
    )
    .map((item) => ({
      id: typeof item.id === "string" ? item.id.slice(0, 80) : "",
      question: typeof item.question === "string" ? item.question.slice(0, 500) : "",
      answer: typeof item.answer === "string" ? item.answer.trim().slice(0, 1000) : "",
    }))
    .filter((item) => item.answer.length > 0)
    .slice(0, 6);
}

export async function POST(request: Request) {
  try {
    const auth = isAuthorized(request);
    if (!auth.ok) {
      return Response.json({ error: auth.error }, { status: auth.status });
    }

    const rawLength = request.headers.get("content-length");
    if (rawLength && Number.parseInt(rawLength, 10) > MAX_REQUEST_BODY_BYTES) {
      return Response.json(
        { error: `Request body must be at most ${MAX_REQUEST_BODY_BYTES} bytes.` },
        { status: 413 },
      );
    }

    const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body || typeof body !== "object") {
      return Response.json({ error: "Request body must be JSON." }, { status: 400 });
    }

    const trimmedMessage = typeof body.message === "string" ? body.message.trim() : "";
    const trimmedPersona = typeof body.persona === "string" ? body.persona.trim() : "";

    if (trimmedMessage.length < 10) {
      return Response.json(
        { error: "message is required (min 10 characters)." },
        { status: 400 },
      );
    }
    if (trimmedMessage.length > MAX_MESSAGE_CHARS) {
      return Response.json(
        { error: `message must be at most ${MAX_MESSAGE_CHARS} characters.` },
        { status: 413 },
      );
    }
    if (trimmedPersona.length < 5) {
      return Response.json(
        { error: "persona is required (min 5 characters)." },
        { status: 400 },
      );
    }
    if (trimmedPersona.length > MAX_PERSONA_CHARS) {
      return Response.json(
        { error: `persona must be at most ${MAX_PERSONA_CHARS} characters.` },
        { status: 413 },
      );
    }

    const normalizedPlatform =
      typeof body.platform === "string" ? body.platform.trim().toLowerCase() : "";
    const validPlatform = platformValues.includes(normalizedPlatform as Platform)
      ? normalizedPlatform
      : "general";

    const result = await refinePitch({
      message: trimmedMessage,
      persona: trimmedPersona,
      platform: validPlatform,
      suggestions: sanitizeSuggestions(body.suggestions),
      clarificationAnswers: sanitizeClarificationAnswers(body.clarificationAnswers),
    });

    if (!result.ok) {
      return Response.json(result.data ?? { error: result.error }, {
        status: result.status,
      });
    }

    return Response.json(result.data ?? {}, { status: result.status });
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Refine failed.";
    return Response.json({ error: msg }, { status: 500 });
  }
}
