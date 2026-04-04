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

    if (!message || typeof message !== "string" || message.trim().length < 10) {
      return Response.json(
        { error: "message is required (min 10 characters)." },
        { status: 400 },
      );
    }
    if (!persona || typeof persona !== "string" || persona.trim().length < 5) {
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
      message: (message as string).trim(),
      persona: (persona as string).trim(),
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
