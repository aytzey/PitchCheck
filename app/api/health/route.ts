import { checkTribeHealth } from "@/lib/tribe-client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const tribe = await checkTribeHealth();
  return Response.json({
    ok: tribe.ok,
    service: "pitchscore",
    tribe: tribe,
    timestamp: new Date().toISOString(),
  }, { status: tribe.ok ? 200 : 503 });
}

export async function HEAD() {
  return new Response(null, { status: 200 });
}
