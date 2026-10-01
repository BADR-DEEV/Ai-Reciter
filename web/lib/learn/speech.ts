"use client";

// Letter and syllable models use the browser's Arabic voice. This is a stopgap
// for single sounds only: synthetic speech is not a recitation reference, so
// Quran ayahs always play a human reciter instead (see SURAHS).
let voice: SpeechSynthesisVoice | null | undefined;

function arabicVoice() {
  if (voice !== undefined && voice !== null) return voice;
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  voice = voices.find(v => v.lang.toLowerCase().startsWith("ar-sa")) || voices.find(v => v.lang.toLowerCase().startsWith("ar")) || null;
  return voice;
}

export function canSpeak() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function speak(text: string, rate = 0.7) {
  if (!canSpeak()) return false;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ar-SA";
  utterance.rate = rate;
  const v = arabicVoice();
  if (v) utterance.voice = v;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
  return true;
}

if (typeof window !== "undefined" && "speechSynthesis" in window)
  window.speechSynthesis.addEventListener?.("voiceschanged", () => { voice = undefined; });

let clip: HTMLAudioElement | null = null;
let clipTimer: ReturnType<typeof setTimeout> | undefined;
/** Play one ayah out of a full-surah recording using millisecond timings. */
export function playClip(src: string, start: number, end: number, onEnd?: () => void) {
  stopClip();
  const audio = new Audio(src);
  clip = audio;
  audio.preload = "auto";
  const begin = () => {
    audio.currentTime = start / 1000;
    void audio.play().catch(() => onEnd?.());
    clipTimer = setTimeout(() => { audio.pause(); onEnd?.(); }, end - start + 150);
  };
  if (audio.readyState >= 1) begin(); else audio.addEventListener("loadedmetadata", begin, { once: true });
  audio.addEventListener("error", () => onEnd?.(), { once: true });
}

export function stopClip() {
  clearTimeout(clipTimer);
  clip?.pause();
  clip = null;
}
