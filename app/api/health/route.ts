import { checkTribeHealth } from "@/lib/tribe-client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const tribe = await checkTribeHealth();
  return Response.json({
    ok: true,
    service: "pitchscore",
    tribe: tribe,
    timestamp: new Date().toISOString(),
  });
}

export async function HEAD() {
  return new Response(null, { status: 200 });
}
