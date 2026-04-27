import { scorePitch } from "@/lib/tribe-client";
import { isPitchScoreReport, platformValues, type Platform } from "@/shared/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const OPENROUTER_MODEL_RE = /^[A-Za-z0-9._:/@+-]{1,160}$/;
const MAX_MESSAGE_CHARS = parseEnvInt("PITCHCHECK_MAX_MESSAGE_CHARS", 30_000, 10);
const MAX_PERSONA_CHARS = parseEnvInt("PITCHCHECK_MAX_PERSONA_CHARS", 5_000, 5);
const MAX_REQUEST_BODY_BYTES = parseEnvInt(
  "PITCHCHECK_MAX_REQUEST_BODY_BYTES",
  128 * 1024,
  1024,
);
const REQUEST_BODY_TOO_LARGE = Symbol("REQUEST_BODY_TOO_LARGE");

function parseEnvInt(name: string, fallback: number, minimum: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}

function bodyIsTooLarge(request: Request): boolean {
  const rawLength = request.headers.get("content-length");
  if (!rawLength) return false;
  const length = Number.parseInt(rawLength, 10);
  return Number.isFinite(length) && length > MAX_REQUEST_BODY_BYTES;
}

async function readRequestTextWithLimit(
  request: Request,
  maxBytes: number,
): Promise<string | typeof REQUEST_BODY_TOO_LARGE> {
  if (!request.body) {
    return "";
  }

  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  const chunks: string[] = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > maxBytes) {
        await reader.cancel();
        return REQUEST_BODY_TOO_LARGE;
      }
      chunks.push(decoder.decode(value, { stream: true }));
    }
    chunks.push(decoder.decode());
    return chunks.join("");
  } finally {
    reader.releaseLock();
  }
}

async function readJsonBody(
  request: Request,
): Promise<
  | { ok: true; body: unknown }
  | { ok: false; status: number; error: string }
> {
  if (bodyIsTooLarge(request)) {
    return {
      ok: false,
      status: 413,
      error: `Request body must be at most ${MAX_REQUEST_BODY_BYTES} bytes.`,
    };
  }

  const rawBody = await readRequestTextWithLimit(request, MAX_REQUEST_BODY_BYTES);
  if (rawBody === REQUEST_BODY_TOO_LARGE) {
    return {
      ok: false,
      status: 413,
      error: `Request body must be at most ${MAX_REQUEST_BODY_BYTES} bytes.`,
    };
  }

  try {
    return { ok: true, body: JSON.parse(rawBody) };
  } catch {
    return { ok: true, body: null };
  }
}

function isAuthorized(request: Request): { ok: true } | { ok: false; status: number; error: string } {
  // In production, require a shared secret header to protect costly scoring compute.
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

export async function POST(request: Request) {
  try {
    const auth = isAuthorized(request);
    if (!auth.ok) {
      return Response.json({ error: auth.error }, { status: auth.status });
    }

    const parsedBody = await readJsonBody(request);
    if (!parsedBody.ok) {
      return Response.json(
        { error: parsedBody.error },
        { status: parsedBody.status },
      );
    }

    const body = parsedBody.body;
    if (!body || typeof body !== "object") {
      return Response.json(
        { error: "Request body must be JSON." },
        { status: 400 },
      );
    }

    const { message, persona, platform, openRouterModel } = body as Record<string, unknown>;

    const trimmedMessage = typeof message === "string" ? message.trim() : "";
    const trimmedPersona = typeof persona === "string" ? persona.trim() : "";

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
      typeof platform === "string" ? platform.trim().toLowerCase() : "";
    const validPlatform =
      platformValues.includes(normalizedPlatform as Platform)
        ? normalizedPlatform
        : "general";
    const sanitizedOpenRouterModel =
      typeof openRouterModel === "string" &&
      OPENROUTER_MODEL_RE.test(openRouterModel.trim())
        ? openRouterModel.trim()
        : undefined;

    const result = await scorePitch({
      message: trimmedMessage,
      persona: trimmedPersona,
      platform: validPlatform,
      openRouterModel: sanitizedOpenRouterModel,
    });

    if (!result.ok) {
      return Response.json(result.data ?? { error: result.error }, {
        status: result.status,
      });
    }

    const payload = result.data && typeof result.data === "object"
      ? result.data as Record<string, unknown>
      : {};
    const report = payload.report ?? payload;

    if (!isPitchScoreReport(report)) {
      return Response.json(
        { error: "Scoring service returned an invalid report." },
        { status: 502 },
      );
    }

    return Response.json({ report }, { status: result.status });
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Scoring failed.";
    return Response.json({ error: msg }, { status: 500 });
  }
}
