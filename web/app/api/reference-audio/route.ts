import { NextRequest } from "next/server";
import { readFile } from "node:fs/promises";
import { referenceAudio } from "@/lib/quran-server";
export const runtime = "nodejs";
export async function GET(request: NextRequest) {
  const surah = Number(request.nextUrl.searchParams.get("surah")), ayah = Number(request.nextUrl.searchParams.get("ayah"));
  if (!Number.isInteger(surah) || surah < 1 || surah > 114 || !Number.isInteger(ayah) || ayah < 1 || ayah > 286) return new Response("Invalid reference", { status: 400 });
  try {
    const file = await referenceAudio(surah, ayah);
    if (!file) return new Response("Aligned local Qaloon recording unavailable", { status: 404 });
    const audio = await readFile(file);
    if (audio.byteLength > 4000000) return new Response("Clip too large", { status: 413 });
    return new Response(new Uint8Array(audio), { headers: { "Content-Type": "audio/wav", "Content-Length": String(audio.length), "Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff" } });
  } catch { return new Response("Reference audio unavailable", { status: 503 }); }
}
