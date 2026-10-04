import { NextRequest } from "next/server";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { referenceAudio } from "@/lib/quran-server";
import { DEFAULT_RECITER, isReciter } from "@/lib/reciters";
import { validTimings, type PlaybackTiming } from "@/lib/playback";
import type { Surah } from "@/lib/types";
export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const surah = Number(request.nextUrl.searchParams.get("surah")), ayah = Number(request.nextUrl.searchParams.get("ayah"));
  const reciter = request.nextUrl.searchParams.get("reciter") || DEFAULT_RECITER;
  if (!isReciter(reciter) || !Number.isInteger(surah) || surah < 1 || surah > 114 || !Number.isInteger(ayah) || ayah < 1 || ayah > 286)
    return Response.json({ error: "Invalid reference" }, { status: 400 });
  try {
    const file = await referenceAudio(surah, ayah, reciter);
    if (!file) throw new Error("No matching audio");
    const source: Surah = JSON.parse(await readFile(path.join(process.cwd(), `public/quran/surahs/${String(surah).padStart(3, "0")}.json`), "utf8"));
    const verse = source.ayahs.find(a => a.ayah === ayah);
    const display = (verse?.displayText || verse?.text || "").replace(/[\u0660-\u0669\d]+/g, "").trim();
    const hash = createHash("sha256").update(await readFile(file)).digest("hex");
    for (const folder of ["large-v3/", ""]) {
      try {
        const data = JSON.parse(await readFile(path.resolve(process.cwd(), `../data/playback_timings/${folder}${reciter}.json`), "utf8"));
        const clip = data.clips.find((c: { surah: number; ayah: number }) => c.surah === surah && c.ayah === ayah) as PlaybackTiming & { reciter: string; audio_sha256: string };
        if (data.reciter !== reciter || !clip || clip.reciter !== reciter || !validTimings(clip, display) || hash !== clip.audio_sha256) continue;
        return Response.json({ displayText: clip.displayText, duration: clip.duration, words: clip.words, model: data.model, status: "machine-timing-proposal" }, { headers: { "Cache-Control": "no-store" } });
      } catch { /* Try another timing proposal for the SAME reader/WAV only. */ }
    }
    throw new Error("Stale or unavailable timings");
  } catch {
    return Response.json({ status: "ayah-only", words: [], message: "Matching word timings unavailable; no inferred timings." }, { status: 404 });
  }
}
