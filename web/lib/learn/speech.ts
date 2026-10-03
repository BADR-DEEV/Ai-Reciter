"use client";

import SOUNDS from "./sounds.json";

// One Web Audio player for every sound in the lessons.
//
// Browsers only allow audio after a user gesture, and an <audio> element that
// waits for network metadata before calling play() can lose that permission
// or stall on a large remote file. Instead the AudioContext is created/resumed
// synchronously inside the click, files are fetched and decoded, and segments
// are played with sample-accurate start/end times.
let context: AudioContext | null = null;
let current: AudioBufferSourceNode | null = null;
let generation = 0;
const buffers = new Map<string, Promise<AudioBuffer>>();
const manifest = SOUNDS as Record<string, string>;

function unlock() {
  if (typeof window === "undefined") return null;
  if (!context) {
    const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    context = new Ctor();
  }
  if (context.state === "suspended") void context.resume();
  return context;
}

function load(ctx: AudioContext, url: string) {
  let pending = buffers.get(url);
  if (!pending) {
    pending = fetch(url, { signal: AbortSignal.timeout(20000) })
      .then(r => { if (!r.ok) throw new Error(`Audio ${r.status}`); return r.arrayBuffer(); })
      .then(data => ctx.decodeAudioData(data));
    pending.catch(() => buffers.delete(url));
    buffers.set(url, pending);
    // Quiz navigation can touch hundreds of clips; bound decoded-audio memory.
    if (buffers.size > 12) buffers.delete(buffers.keys().next().value!);
  }
  return pending;
}

export function stopAudio() {
  generation++;
  // The replaced source's promise still settles, as "stopped".
  if (current) { try { current.stop(); } catch { /* Already stopped. */ } }
  current = null;
  if (typeof window !== "undefined" && "speechSynthesis" in window) window.speechSynthesis.cancel();
}

/** Play a file (optionally a [start, end) segment in ms). Resolves when playback ends or is replaced. */
export async function playUrl(url: string, start = 0, end?: number, onStart?: () => void): Promise<"ended" | "stopped" | "failed"> {
  const ctx = unlock(); // Must run synchronously inside the click handler.
  stopAudio();
  const token = generation;
  if (!ctx) return "failed";
  try {
    const buffer = await load(ctx, url);
    if (token !== generation) return "stopped";
    if (ctx.state !== "running") await ctx.resume();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    current = source;
    onStart?.();
    const offset = start / 1000;
    source.start(0, offset, end !== undefined ? Math.max(0.05, (end - start) / 1000) : undefined);
    return await new Promise(resolve => {
      source.onended = () => { if (current === source) current = null; resolve(token === generation ? "ended" : "stopped"); };
    });
  } catch {
    return "failed";
  }
}

/** Say an Arabic letter, syllable or word: pre-rendered audio, else the browser voice. */
export function speak(text: string) {
  const file = manifest[text];
  if (file) { void playUrl(`/audio/sounds/${file}`); return true; }
  return speakWithBrowser(text);
}

function speakWithBrowser(text: string) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;
  const synth = window.speechSynthesis;
  stopAudio();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = synth.getVoices().find(v => v.lang.toLowerCase().startsWith("ar"));
  if (voice) { utterance.voice = voice; utterance.lang = voice.lang; } else utterance.lang = "ar";
  utterance.rate = 0.75;
  // Chrome drops an utterance queued in the same tick as cancel(), and can be
  // left paused; give it a moment and make sure it is running.
  setTimeout(() => { synth.resume(); synth.speak(utterance); }, 60);
  return true;
}

/** Play one ayah out of a full-surah recording using millisecond timings. */
export function playClip(src: string, start: number, end: number, onStart: () => void, onEnd: (result: "ended" | "stopped" | "failed") => void) {
  void playUrl(src, start, end, onStart).then(onEnd);
}

export const stopClip = stopAudio;
