// Lesson checks run on the same local Whisper service as the studio.
import type { ModelName, ModelUsage } from "@/lib/models";

const ws = process.env.NEXT_PUBLIC_RECITER_WS || "ws://127.0.0.1:8000/ws/recite";
export const API = (process.env.NEXT_PUBLIC_RECITER_API || ws.replace(/^ws/, "http").replace(/\/ws\/recite$/, "")).replace(/\/$/, "");

export type SoundResult = { verdict: "correct" | "close" | "other"; heard: number; confidence: number; probabilities: number[]; transcript: string };
// `tags`: tajweed tags the tajweed model heard on that word (absent for the plain model).
export type ReadingResult = { verdict: "correct" | "close" | "other"; score: number; transcript: string; words: { index: number; text: string; status: "correct" | "missed"; heard: string | null; tags?: string[] }[] };
export type PracticeResult = (SoundResult | ReadingResult | { verdict: "silent" }) & ModelUsage;

function base64(pcm: Float32Array) {
  const bytes = new Uint8Array(pcm.length * 2);
  const view = new DataView(bytes.buffer);
  pcm.forEach((v, i) => view.setInt16(i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/** `model` picks the engine (service default when omitted); "tajweed" falls back to plain with `fallback_reason` set. */
export async function checkPractice(pcm: Float32Array, body: { mode: "sound" | "reading"; target: string; alternatives?: string[]; model?: ModelName }, signal?: AbortSignal): Promise<PracticeResult> {
  const response = await fetch(`${API}/api/practice`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, alternatives: body.alternatives || [], audio: base64(pcm) }),
  });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `The model service returned ${response.status}`);
  return response.json();
}

export async function modelOnline(signal?: AbortSignal) {
  try { return (await fetch(`${API}/health`, { signal })).ok; } catch { return false; }
}
