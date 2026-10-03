import type { Surah } from "./types";

export type ChallengeMode = "next" | "audio" | "surah" | "missing" | "order";
export type Difficulty = "easy" | "medium" | "hard";
export type Verse = { surah: number; name: string; ayah: number; text: string; normalized: string; audio?: string; duration?: number };
export type Challenge = { id: string; mode: ChallengeMode; difficulty: Difficulty; prompt: string; reference: string; surah: number; ayah: number; options: { label: string; audio?: string }[]; answer: number; target: string; explanation: string; similarity: string };
export type AcousticIndex = { method: string; neighbors: Record<string, { id: string; score: number }[]> };
const key = (v: Verse) => `${v.surah}:${v.ayah}`;
export function corpusVerses(surahs: Surah[]): Verse[] {
  return surahs.flatMap(s => s.ayahs.map(a => ({ surah: s.id, name: s.name, ayah: a.ayah, text: a.displayText || a.text.replace(/[\d\u0660-\u0669]+/gu, "").trim(), normalized: a.normalized })));
}
const pick = <T,>(values: T[], random: () => number) => values[Math.floor(random() * values.length)];
export function shuffle<T>(values: T[], random: () => number = Math.random): T[] {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [result[i], result[j]] = [result[j], result[i]]; }
  return result;
}
export function textSimilarity(a: string, b: string) {
  const grams = (s: string) => new Set(Array.from({ length: Math.max(0, s.length - 1) }, (_, i) => s.slice(i, i + 2)));
  const x = grams(a), y = grams(b);
  return 2 * [...x].filter(g => y.has(g)).length / Math.max(1, x.size + y.size);
}
function distractors(target: Verse, pool: Verse[], difficulty: Difficulty, acoustic: AcousticIndex | null, random: () => number): Verse[] {
  const neighbors = acoustic?.neighbors[key(target)] || [];
  const scores = new Map(neighbors.map(n => [n.id, n.score]));
  // Compute a similarity once per candidate, not inside O(n log n) sort calls.
  const ranked = shuffle(pool.filter(v => v.normalized !== target.normalized), random)
    .map(verse => ({ verse, score: scores.get(key(verse)) ?? textSimilarity(target.normalized, verse.normalized) * (neighbors.length ? 0.5 : 1) }))
    .sort((a, b) => b.score - a.score).map(row => row.verse);
  const available = difficulty === "hard" ? ranked.slice(0, 12) : difficulty === "medium" ? ranked.slice(0, Math.max(6, Math.ceil(ranked.length / 2))) : ranked.slice(Math.floor(ranked.length / 2));
  const unique = new Map(shuffle(available.length >= 2 ? available : ranked, random).map(v => [v.normalized, v]));
  return [...unique.values()].slice(0, 2);
}

export function makeChallenge(verses: Verse[], mode: ChallengeMode, difficulty: Difficulty, acoustic: AcousticIndex | null = null, random: () => number = Math.random): Challenge {
  if (verses.length < 3) throw new Error("Cache the Quran before opening challenges.");
  let prompt: Verse, target: Verse;
  let options: { label: string; audio?: string }[];
  let right: { label: string; audio?: string };
  let explanation = "";
  const indexed = new Map(verses.map(v => [key(v), v]));
  if (mode === "next" || mode === "order") {
    const eligible = verses.filter(v => {
      const next = indexed.get(`${v.surah}:${v.ayah + 1}`);
      const third = indexed.get(`${v.surah}:${v.ayah + 2}`);
      return next && (mode !== "next" || next.normalized.split(" ").length <= 12) && (mode !== "order" || third && new Set([v.normalized, next.normalized, third.normalized]).size === 3);
    });
    if (!eligible.length) throw new Error("No suitable consecutive ayahs in this scope.");
    prompt = pick(eligible, random); target = indexed.get(`${prompt.surah}:${prompt.ayah + 1}`)!;
    if (mode === "order") {
      const third = indexed.get(`${prompt.surah}:${prompt.ayah + 2}`)!;
      const labels = [prompt.text, target.text, third.text];
      right = { label: labels.join(" → ") };
      options = [right, { label: [labels[1], labels[0], labels[2]].join(" → ") }, { label: [labels[2], labels[1], labels[0]].join(" → ") }];
      explanation = `Qālūn order: ${prompt.name}, ayahs ${prompt.ayah}–${third.ayah}.`;
    } else {
      right = { label: target.text };
      options = [right, ...distractors(target, verses, difficulty, null, random).map(v => ({ label: v.text }))];
      explanation = `The next Qālūn ayah is ${target.name} ${target.ayah}.`;
    }
  } else if (mode === "audio") {
    const playable = verses.filter(v => v.audio && v.duration && v.duration <= 30);
    if (new Set(playable.map(v => v.normalized)).size < 3) throw new Error("Local Al-Husary Qālūn clips are unavailable. Cache the training dataset or choose a text challenge.");
    target = prompt = pick(playable, random);
    right = { label: "", audio: target.audio };
    options = [right, ...distractors(target, playable, difficulty, acoustic, random).map(v => ({ label: "", audio: v.audio }))];
    explanation = `Reference recording: Al-Husary, Qālūn, ${target.name} ${target.ayah}. Similar recordings are not equivalent ayahs.`;
  } else if (mode === "surah") {
    const origins = new Map<string, Set<number>>();
    for (const v of verses) { const ids = origins.get(v.normalized) || new Set(); ids.add(v.surah); origins.set(v.normalized, ids); }
    // Repeated ayahs can belong to several surahs; never mark one valid origin wrong.
    const eligible = verses.filter(v => origins.get(v.normalized)!.size === 1);
    if (!eligible.length) throw new Error("No unambiguous surah questions in this scope.");
    target = prompt = pick(eligible, random);
    right = { label: target.name };
    const pool = new Map(distractors(target, verses.filter(v => v.surah !== target.surah), difficulty, null, random).map(v => [v.surah, v.name]));
    for (const v of shuffle(verses, random)) if (v.surah !== target.surah && pool.size < 2) pool.set(v.surah, v.name);
    options = [right, ...[...pool.values()].slice(0, 2).map(label => ({ label }))];
    explanation = `This ayah is ${target.name} ${target.ayah} in the Qālūn source.`;
  } else {
    const eligible = verses.filter(v => v.text.split(/\s+/).length >= 3);
    if (!eligible.length) throw new Error("No suitable word questions.");
    target = pick(eligible, random);
    const words = target.text.split(/\s+/), missing = Math.floor(random() * words.length);
    right = { label: words[missing] };
    prompt = { ...target, text: words.map((w, i) => i === missing ? "ـــــ" : w).join(" ") };
    const choices = [...new Set(verses.flatMap(v => v.text.split(/\s+/)).filter(w => w !== right.label))];
    const ranked = choices.map(label => ({ label, score: textSimilarity(right.label, label) })).sort((a, b) => b.score - a.score).map(row => row.label);
    options = [right, ...shuffle(difficulty === "hard" ? ranked.slice(0, 15) : ranked, random).slice(0, 2).map(label => ({ label }))];
    explanation = `The missing word is ${right.label}; ${target.name} ${target.ayah}.`;
  }
  options = shuffle(options, random);
  if (options.length !== 3) throw new Error("Not enough distinct answer options in this scope.");
  return { id: `${mode}:${key(target)}`, mode, difficulty, prompt: mode === "order" ? "Arrange these three consecutive ayahs by choosing the correct order." : prompt.text,
    reference: mode === "surah" ? "Find its surah" : `${prompt.name} · Qālūn ayah ${prompt.ayah}`,
    surah: target.surah, ayah: target.ayah, target: target.text, options, answer: options.indexOf(right), explanation,
    similarity: mode === "audio" && acoustic?.neighbors[key(target)]?.length ? "MFCC recording-similarity + text fallback (not phoneme correctness)" : "Text bigram similarity (not measured acoustic or semantic similarity)" };
}
