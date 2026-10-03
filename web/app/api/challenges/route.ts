import { NextRequest, NextResponse } from "next/server";
import { makeChallenge, type ChallengeMode, type Difficulty } from "@/lib/challenges";
import { loadAcoustic, loadCorpus } from "@/lib/quran-server";
export const runtime = "nodejs";
export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") || "next";
  const difficulty = request.nextUrl.searchParams.get("difficulty") || "easy";
  const scope = request.nextUrl.searchParams.get("scope") || "amma";
  if (!["next", "audio", "surah", "missing", "order"].includes(mode) || !["easy", "medium", "hard"].includes(difficulty) || !["amma", "all"].includes(scope)) return NextResponse.json({ error: "Invalid challenge settings." }, { status: 400 });
  try {
    const [verses, acoustic] = await Promise.all([loadCorpus(), loadAcoustic()]);
    const pool = scope === "all" ? verses : verses.filter(v => v.surah === 1 || v.surah >= 78);
    return NextResponse.json(makeChallenge(pool, mode as ChallengeMode, difficulty as Difficulty, acoustic), { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Challenge unavailable." }, { status: 503 });
  }
}
