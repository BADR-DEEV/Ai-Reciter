import { NextRequest, NextResponse } from "next/server";
import { CHOICE_SOURCES, makeChallenge, type ChallengeMode, type ChoiceSource, type Difficulty } from "@/lib/challenges";
import { loadAcoustic, loadCorpus, loadTextIndex } from "@/lib/quran-server";
import { DEFAULT_RECITER, isReciter, RECITERS } from "@/lib/reciters";
export const runtime = "nodejs";
export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") || "next";
  const difficulty = request.nextUrl.searchParams.get("difficulty") || "easy";
  const scope = request.nextUrl.searchParams.get("scope") || "amma";
  const reciter = request.nextUrl.searchParams.get("reciter") || DEFAULT_RECITER;
  const lang = request.nextUrl.searchParams.get("lang") || "en";
  const from = request.nextUrl.searchParams.get("from") || "scope";
  if (lang !== "en" && lang !== "ar") return NextResponse.json({ error: "Invalid language." }, { status: 400 });
  if (!isReciter(reciter)) return NextResponse.json({ error: "Unknown reciter." }, { status: 400 });
  if (!["next", "audio", "surah", "missing", "order"].includes(mode) || !["easy", "medium", "hard"].includes(difficulty) || !["amma", "all"].includes(scope) || !CHOICE_SOURCES.includes(from as ChoiceSource)) return NextResponse.json({ error: "Invalid challenge settings." }, { status: 400 });
  try {
    const [verses, acoustic, text] = await Promise.all([loadCorpus(reciter), loadAcoustic(reciter), loadTextIndex()]);
    const pool = scope === "all" ? verses : verses.filter(v => v.surah === 1 || v.surah >= 78);
    const question = makeChallenge(pool, mode as ChallengeMode, difficulty as Difficulty, { acoustic, lang, text, from: from as ChoiceSource, quran: verses });
    const reader = RECITERS.find(r => r.id === reciter)!;
    if (mode === "audio") question.explanation = lang === "ar" ? `التسجيل المرجعي: ${reader.arabic} بقالون · ${question.surah}:${question.ayah}.` : `Reference recording: ${reader.name}, Qālūn · ${question.surah}:${question.ayah}.`;
    return NextResponse.json(question, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Challenge unavailable." }, { status: 503 });
  }
}
