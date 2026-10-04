import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { TajweedMushaf } from "@/lib/tajweed";
export const runtime = "nodejs";
let cached: Promise<TajweedMushaf> | null = null;
export async function GET(request: NextRequest) {
  const surah = Number(request.nextUrl.searchParams.get("surah"));
  if (!Number.isInteger(surah) || surah < 1 || surah > 114) return NextResponse.json({ error: "Invalid surah" }, { status: 400 });
  try {
    if (!cached) cached = readFile(path.join(process.cwd(), "public/quran/qalon_majwad_mushaf.json"), "utf8").then(text => JSON.parse(text)).catch(error => { cached = null; throw error; });
    const data = await cached!;
    return NextResponse.json({ status: data.status, approved_ayahs: data.approved_ayahs, rules: data.rules, legend: data.legend, basmalah: data.basmalah, surahs: data.surahs.filter(s => s.id === surah) });
  } catch { return NextResponse.json({ error: "Generate the reviewable tajweed JSON first. Canonical Quran text is unaffected." }, { status: 503 }); }
}
