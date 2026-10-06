// "Which ayah is this?": one clip goes to the local model service, which
// transcribes it blind and returns the best places in the Quran.
import { API, pcmBase64 } from "./learn/api";
import type { WordResult } from "./types";

const RATE = 16000;
export const MAX_CLIP_SECONDS = 30; // one Whisper window, the server's limit

export type SearchPlace = { surah: number; ayah: number };
export type SearchAyah = SearchPlace & { juz: number; text: string; displayText: string; words: WordResult[] };
export type Confidence = "high" | "medium" | "low";
export type SearchResult = {
  rank: number; surah: number; name: string; arabic: string; trained: boolean;
  start: SearchPlace; end: SearchPlace; juz: number[];
  score: number; coverage: number; matched_words: number; confidence: Confidence; ayahs: SearchAyah[];
};
export type SearchResponse = {
  verdict: "found" | "none" | "silent" | "opening_only";
  transcript: string; heard: string[]; opening: ("taawwudh" | "basmala")[];
  results: SearchResult[]; juz: number[]; model_used?: string;
};

/** `juz` empty searches the whole Quran. */
export async function searchRecitation(pcm: Float32Array, juz: number[], signal?: AbortSignal): Promise<SearchResponse> {
  const response = await fetch(`${API}/api/search`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio: pcmBase64(pcm), juz }),
  });
  if (!response.ok) {
    const detail = (await response.json().catch(() => null))?.detail;
    throw new Error(typeof detail === "string" ? detail : `The model service returned ${response.status}.`);
  }
  return response.json();
}

function quiet(pcm: Float32Array, start: number, length: number) {
  let sum = 0;
  for (let i = start; i < start + length; i++) sum += pcm[i] * pcm[i];
  return Math.sqrt(sum / length) < 0.01;
}

/** A recording as 16 kHz mono PCM from just before its first sound, at most
 *  MAX_CLIP_SECONDS long. `trimmed` is true when the end was left out. */
export async function decodeClip(file: File): Promise<{ pcm: Float32Array; trimmed: boolean }> {
  if (file.size > 50 * 1024 * 1024) throw new Error("Choose an audio file smaller than 50 MB.");
  const context = new AudioContext();
  let recording: AudioBuffer;
  try { recording = await context.decodeAudioData(await file.arrayBuffer()); }
  catch { throw new Error("This file could not be read as audio. Try a WAV, MP3 or M4A recording."); }
  finally { void context.close(); }
  if (!recording.length) throw new Error("This recording contains no audio.");
  const offline = new OfflineAudioContext(1, Math.ceil(recording.duration * RATE), RATE);
  const source = offline.createBufferSource();
  source.buffer = recording; source.connect(offline.destination); source.start();
  const pcm = (await offline.startRendering()).getChannelData(0);
  // A quiet intro would use up the 30 seconds; keep a short lead-in before the first sound.
  const frame = RATE / 20;
  let start = 0;
  while (start + frame <= pcm.length && quiet(pcm, start, frame)) start += frame;
  start = Math.max(0, start - frame * 4);
  const end = Math.min(pcm.length, start + MAX_CLIP_SECONDS * RATE);
  const clip = pcm.slice(start, end).map(v => Math.max(-1, Math.min(1, v)));
  return { pcm: clip, trimmed: end < pcm.length };
}

/** "1–3, 30" for [1, 2, 3, 30]. */
export function juzRanges(juz: number[], locale: string) {
  const sorted = [...new Set(juz)].sort((a, b) => a - b), parts: string[] = [];
  for (let i = 0; i < sorted.length; i++) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    const [a, b] = [sorted[i].toLocaleString(locale), sorted[j].toLocaleString(locale)];
    parts.push(j > i ? `${a}–${b}` : a);
    i = j;
  }
  return parts.join(locale === "ar" ? "، " : ", ");
}
