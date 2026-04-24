import { scorePitch } from "@/lib/tribe-client";
import { platformValues, type Platform } from "@/shared/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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

    const body = await request.json().catch(() => null);
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
    if (trimmedPersona.length < 5) {
      return Response.json(
        { error: "persona is required (min 5 characters)." },
        { status: 400 },
      );
    }

    const validPlatform =
      typeof platform === "string" &&
      platformValues.includes(platform as Platform)
        ? (platform as string)
        : "general";

    const result = await scorePitch({
      message: trimmedMessage,
      persona: trimmedPersona,
      platform: validPlatform,
      openRouterModel: typeof openRouterModel === "string" ? openRouterModel.trim() || undefined : undefined,
    });

    return Response.json(result.data ?? { error: result.error }, {
      status: result.status,
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Scoring failed.";
    return Response.json({ error: msg }, { status: 500 });
  }
}
