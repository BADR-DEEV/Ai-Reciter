import { readFile, access } from "node:fs/promises";
import path from "node:path";
import { corpusVerses, type Verse, type AcousticIndex, type EmbeddingTable, type TextIndex } from "./challenges";
import { DEFAULT_RECITER, RECITERS, referenceURL, type ReciterID } from "./reciters";
import type { Surah } from "./types";
const publicRoot = path.join(process.cwd(), "public/quran");
const rootFor = (reciter: ReciterID) => path.resolve(process.cwd(), "../src/dataset_collection", RECITERS.find(r => r.id === reciter)!.folder);
type AudioRow = { surah: number; ayah: number; relative_audio_path: string; text_asr_normalized: string; start_time: number; end_time: number };
const audioCache = new Map<ReciterID, Promise<Map<string, AudioRow>>>();
export function audioRows(reciter: ReciterID = DEFAULT_RECITER) {
  const root = rootFor(reciter);
  if (!audioCache.has(reciter)) audioCache.set(reciter, readFile(path.join(root, "metadata.jsonl"), "utf8").then(async text => {
    const rows: AudioRow[] = text.trim().split(/\r?\n/).map(line => JSON.parse(line));
    const existing = await Promise.all(rows.filter(r => typeof r.relative_audio_path === "string" && /^[a-zA-Z0-9_/.-]+\.wav$/.test(r.relative_audio_path)).map(async row => {
      const resolved = path.resolve(root, row.relative_audio_path);
      if (!resolved.startsWith(root + path.sep)) return null;
      try { await access(resolved); return row; } catch { return null; }
    }));
    return new Map(existing.filter((row): row is AudioRow => row !== null).map(row => [`${row.surah}:${row.ayah}`, row]));
  }).catch(() => { audioCache.delete(reciter); return new Map<string, AudioRow>(); }));
  return audioCache.get(reciter)!;
}
export async function referenceAudio(surah: number, ayah: number, reciter: ReciterID = DEFAULT_RECITER) {
  const row = (await audioRows(reciter)).get(`${surah}:${ayah}`);
  if (!row) return null;
  const source: Surah = JSON.parse(await readFile(path.join(publicRoot, "surahs", `${String(surah).padStart(3, "0")}.json`), "utf8"));
  if (source.ayahs.find(a => a.ayah === ayah)?.normalized !== row.text_asr_normalized) return null;
  return path.resolve(rootFor(reciter), row.relative_audio_path);
}
const corpusCache = new Map<ReciterID, Promise<Verse[]>>();
let texts: Promise<Surah[]> | null = null;
export function loadCorpus(reciter: ReciterID = DEFAULT_RECITER) {
  if (!texts) texts = Promise.all(Array.from({ length: 114 }, (_, i) => readFile(path.join(publicRoot, "surahs", `${String(i + 1).padStart(3, "0")}.json`), "utf8").then(text => JSON.parse(text) as Surah))).catch(error => { texts = null; throw error; });
  if (!corpusCache.has(reciter)) corpusCache.set(reciter, texts.then(async surahs => {
    const rows = await audioRows(reciter);
    // Recordings may still be downloading (lib/reader-audio.ts); don't cache an audio-less corpus forever.
    if (!rows.size) queueMicrotask(() => corpusCache.delete(reciter));
    return corpusVerses(surahs).map(v => {
      const row = rows.get(`${v.surah}:${v.ayah}`);
      return row && row.text_asr_normalized === v.normalized ? { ...v, audio: referenceURL(v.surah, v.ayah, reciter), duration: (row.end_time - row.start_time) / 1000 } : v;
    });
  }).catch(error => { corpusCache.delete(reciter); throw error; }));
  return corpusCache.get(reciter)!;
}
export async function loadAcoustic(reciter: ReciterID = DEFAULT_RECITER): Promise<AcousticIndex | null> {
  try {
    const data = JSON.parse(await readFile(path.join(publicRoot, `audio-similarity-${reciter}.json`), "utf8"));
    return data.reciter_key === reciter && data.method && data.neighbors ? data : null;
  } catch { return null; }
}

type TableFile = { scales: number[]; vectors: string };
function table(keys: string[], file: TableFile, dim: number): EmbeddingTable {
  const bytes = Buffer.from(file.vectors, "base64");
  if (!Number.isInteger(dim) || dim <= 0 || file.scales.length !== keys.length || bytes.length !== keys.length * dim) throw new Error("Malformed embedding table");
  return { index: new Map(keys.map((k, i) => [k, i])), dim, values: new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length), scales: Float32Array.from(file.scales) };
}
let textIndex: Promise<TextIndex | null> | null = null;
/** Optional (src/learning/build_text_embeddings.py); challenges fall back to spelling similarity without it. */
export function loadTextIndex(): Promise<TextIndex | null> {
  if (!textIndex) textIndex = readFile(path.join(publicRoot, "text-embeddings.json"), "utf8").then(raw => {
    const data = JSON.parse(raw);
    return { model: String(data.label || data.model), method: data.method, ayahs: table(data.ayahs.ids, data.ayahs, data.dim), words: table(data.words.keys, data.words, data.dim) };
  }).catch(() => {
    // Retry later: the first-run build may still be writing the index.
    setTimeout(() => { textIndex = null; }, 60_000);
    return null;
  });
  return textIndex;
}
