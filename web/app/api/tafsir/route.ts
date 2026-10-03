import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Surah } from "@/lib/types";

export const runtime = "nodejs";
type Book = { id: number; name: string; language: string; author: string; book_name: string };
// This provider's documented HTTP endpoint currently works; HTTPS is not
// reliable. Never send user data/tokens to it or permit user-controlled URLs.
const ORIGIN = "http://api.quran-tafseer.com";
async function provider(endpoint: string) {
  const response = await fetch(`${ORIGIN}${endpoint}`, {
    signal: AbortSignal.timeout(8000), next: { revalidate: 86400 },
  });
  if (!response.ok) throw new Error("Tafsir provider unavailable");
  return response.json();
}

export async function GET(request: NextRequest) {
  const surah = Number(request.nextUrl.searchParams.get("surah"));
  const ayah = Number(request.nextUrl.searchParams.get("ayah"));
  const language = request.nextUrl.searchParams.get("lang") || "en";
  if (!Number.isInteger(surah) || surah < 1 || surah > 114 || !Number.isInteger(ayah) || ayah < 1 || ayah > 286 || !["ar", "en"].includes(language)) {
    return NextResponse.json({ error: "Choose a valid surah, ayah and language." }, { status: 400 });
  }
  try {
    const source: Surah = JSON.parse(await readFile(path.join(process.cwd(), "public/quran/surahs", `${String(surah).padStart(3, "0")}.json`), "utf8"));
    const selected = source.ayahs.find(a => a.ayah === ayah);
    if (!selected) return NextResponse.json({ error: "Qaloon ayah not found." }, { status: 404 });
    const ids = [...new Set(selected.regions.map(r => r.display_ayah))].sort((a, b) => a - b);
    if (!ids.length || ids.length > 8 || ids.some(id => !Number.isInteger(id) || id < 1 || id > 286)) {
      return NextResponse.json({ error: "No verified numbering alignment available; refusing to guess a tafsir verse." }, { status: 409 });
    }
    const books: Book[] = await provider("/tafseer/");
    if (!Array.isArray(books)) throw new Error("Invalid book list");
    const book = books.find(b => b.language === language && b.id === (language === "en" ? 10 : 1)) || books.find(b => b.language === language);
    if (!book || !Number.isInteger(book.id) || book.id < 1 || book.id > 100) throw new Error("Requested language not offered");
    const entries = await Promise.all(ids.map(async id => {
      const data = await provider(`/tafseer/${book.id}/${surah}/${id}`);
      if (typeof data.text !== "string" || data.text.length > 100000 || data.tafseer_id !== book.id || data.ayah_number !== id) throw new Error("Invalid verse response");
      return { providerAyah: id, text: data.text };
    }));
    return NextResponse.json({ surah, qaloonAyah: ayah, language, kind: language === "en" ? "Translation of meanings" : "Arabic tafsir",
      book: { name: book.name, author: book.author }, entries,
      source: "Quran Tafseer API", sourceUrl: `${ORIGIN}/en/docs/`, transport: "HTTP",
      mappingNote: "Provider uses Hafs-numbered verses. Text-aligned references may include more text than this Qaloon ayah; this is commentary, not a replacement recitation text." },
    { headers: { "Cache-Control": "private, max-age=3600" } });
  } catch {
    return NextResponse.json({ error: "Tafsir is unavailable right now. Quran reading and recitation still work; try again later." }, { status: 502 });
  }
}
