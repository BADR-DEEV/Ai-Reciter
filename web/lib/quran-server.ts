import { readFile, access } from "node:fs/promises";
import path from "node:path";
import { corpusVerses, type Verse, type AcousticIndex } from "./challenges";
import type { Surah } from "./types";







const publicRoot = path.join(process.cwd(), "public/quran");
const audioRoot = path.resolve(process.cwd(), "../src/dataset_collection/dataset_qaloon_hutafi");
type AudioRow = { surah: number; ayah: number; relative_audio_path: string; text_asr_normalized: string; start_time: number; end_time: number };
let audioCache: Promise<Map<string, AudioRow>> | null = null;



export function audioRows() {
  if (!audioCache) audioCache = readFile(path.join(audioRoot, "metadata.jsonl"), "utf8").then(async text => {
    const rows: AudioRow[] = text.trim().split(/\r?\n/).map(line => JSON.parse(line));
    const existing = await Promise.all(rows.filter(r => /^[a-zA-Z0-9_/.-]+\.wav$/.test(r.relative_audio_path)).map(async row => {
      const resolved = path.resolve(audioRoot, row.relative_audio_path);
      if (!resolved.startsWith(audioRoot + path.sep)) return null;
      try { await access(resolved); return row; } catch { return null; }
    }));
    return new Map(existing.filter((row): row is AudioRow => row !== null).map(row => [`${row.surah}:${row.ayah}`, row]));
  }).catch(() => { audioCache = null; return new Map<string, AudioRow>(); });
  return audioCache;
}
export async function referenceAudio(surah: number, ayah: number) {
  const row = (await audioRows()).get(`${surah}:${ayah}`);
  if (!row) return null;
  const source: Surah = JSON.parse(await readFile(path.join(publicRoot, "surahs", `${String(surah).padStart(3, "0")}.json`), "utf8"));
  if (source.ayahs.find(a => a.ayah === ayah)?.normalized !== row.text_asr_normalized) return null;
  return path.resolve(audioRoot, row.relative_audio_path);
}
let corpusCache: Promise<Verse[]> | null = null;
export function loadCorpus() {
  if (!corpusCache) corpusCache = Promise.all(Array.from({ length: 114 }, (_, i) => readFile(path.join(publicRoot, "surahs", `${String(i + 1).padStart(3, "0")}.json`), "utf8").then(text => JSON.parse(text) as Surah)))
    .then(async surahs => {
      const rows = await audioRows();
      return corpusVerses(surahs).map(v => {
        const row = rows.get(`${v.surah}:${v.ayah}`);
        return row && row.text_asr_normalized === v.normalized ? { ...v, audio: `/api/reference-audio?surah=${v.surah}&ayah=${v.ayah}`, duration: (row.end_time - row.start_time) / 1000 } : v;
      });
    }).catch(error => { corpusCache = null; throw error; });
  return corpusCache;
}
export async function loadAcoustic(): Promise<AcousticIndex | null> {
  try { const data = JSON.parse(await readFile(path.join(publicRoot, "audio-similarity.json"), "utf8")); return data.method && data.neighbors ? data : null; }
  catch { return null; }
}
