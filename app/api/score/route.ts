import { scorePitch } from "@/lib/tribe-client";
import { platformValues, type Platform } from "@/shared/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => null);
    if (!body || typeof body !== "object") {
      return Response.json(
        { error: "Request body must be JSON." },
        { status: 400 },
      );
    }

    const { message, persona, platform } = body as Record<string, unknown>;

    const trimmedMessage = typeof message === "string" ? message.trim() : "";
    const trimmedPersona = typeof persona === "string" ? persona.trim() : "";

    if (trimmedMessage.length < 10) {
      return Response.json(
        { error: "message is required (min 10 characters)." },
        { status: 400 },
      );
    }
    if (trimmedMessage.length > 5000) {
      return Response.json(
        { error: "message is too long (max 5000 characters)." },
        { status: 400 },
      );
    }
    if (trimmedPersona.length < 5) {
      return Response.json(
        { error: "persona is required (min 5 characters)." },
        { status: 400 },
      );
    }
    if (trimmedPersona.length > 1500) {
      return Response.json(
        { error: "persona is too long (max 1500 characters)." },
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
    });

    return Response.json(result.data ?? { error: result.error }, {
      status: result.status,
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Scoring failed.";
    return Response.json({ error: msg }, { status: 500 });
  }
}
