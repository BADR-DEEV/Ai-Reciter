import type { Surah } from "./types";

export type ChallengeMode = "next" | "audio" | "surah" | "missing" | "order";
export type Difficulty = "easy" | "medium" | "hard";
export type Verse = { surah: number; name: string; nameAr?: string; ayah: number; text: string; normalized: string; audio?: string; duration?: number };
export type ChallengeOption = { label: string; audio?: string; parts?: string[] };
export type Challenge = { id: string; mode: ChallengeMode; difficulty: Difficulty; prompt: string; reference: string; surah: number; ayah: number; options: ChallengeOption[]; answer: number; target: string; explanation: string; similarity: string };
export type AcousticIndex = { method: string; neighbors: Record<string, { id: string; score: number }[]> };
/** Unit vectors quantized to int8 with one scale per row (src/learning/build_text_embeddings.py). */
export type EmbeddingTable = { index: Map<string, number>; dim: number; values: Int8Array; scales: Float32Array };
export type TextIndex = { model: string; method: string; ayahs: EmbeddingTable; words: EmbeddingTable };
const key = (v: Verse) => `${v.surah}:${v.ayah}`;
export function corpusVerses(surahs: Surah[]): Verse[] {
  return surahs.flatMap(s => s.ayahs.map(a => ({ surah: s.id, name: s.name, nameAr: s.arabic, ayah: a.ayah, text: a.displayText || a.text.replace(/[\d\u0660-\u0669]+/gu, "").trim(), normalized: a.normalized })));
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
/** Letters only, as in word_key() of the Python builder: vowels, Qālūn marks and tatweel removed. */
export const wordKey = (word: string) => word.normalize("NFD").replace(/[\p{M}ـ]/gu, "");
export function cosine(table: EmbeddingTable, a: string, b: string): number | null {
  const i = table.index.get(a), j = table.index.get(b);
  if (i === undefined || j === undefined) return null;
  let sum = 0;
  for (let k = 0, x = i * table.dim, y = j * table.dim; k < table.dim; k++) sum += table.values[x + k] * table.values[y + k];
  return sum * table.scales[i] * table.scales[j];
}

/** Closeness of two strings for a learner: meaning (when embedded), shared letters, the same rhyme ending, similar length. */
function closeness(a: string, b: string, semantic: number | null) {
  const ending = (s: string) => s.replace(/\s+/gu, "").slice(-2);
  const rhyme = ending(a) === ending(b) ? 1 : ending(a).slice(-1) === ending(b).slice(-1) ? .5 : 0;
  // A much longer or shorter option stands out without being read.
  const length = Math.min(a.length, b.length) / Math.max(a.length, b.length, 1);
  const lexical = textSimilarity(a, b);
  return semantic === null ? .7 * lexical + .2 * rhyme + .1 * length : .55 * semantic + .2 * lexical + .15 * rhyme + .1 * length;
}
export const verseCloseness = (a: Verse, b: Verse, text: TextIndex | null) =>
  closeness(a.normalized, b.normalized, text && cosine(text.ayahs, key(a), key(b)));

/** Easy draws related but distinguishable choices, hard the closest ones. Bands are rank windows over a sorted list. */
const BANDS: Record<Difficulty, [number, number]> = { hard: [0, 6], medium: [6, 20], easy: [20, 60] };
function band<T>(ranked: T[], difficulty: Difficulty, scale = 1) {
  const [from, to] = BANDS[difficulty].map(n => Math.round(n * scale));
  const window = ranked.slice(from, to);
  return window.length >= 2 ? window : ranked.slice(Math.max(0, ranked.length - Math.max(2, to - from)));
}

function distractors(target: Verse, pool: Verse[], difficulty: Difficulty, acoustic: AcousticIndex | null, text: TextIndex | null, random: () => number, exclude: Verse[] = []): Verse[] {
  const neighbors = acoustic?.neighbors[key(target)] || [];
  const sound = new Map(neighbors.map(n => [n.id, n.score]));
  const skip = new Set([target, ...exclude].map(v => v.normalized));
  // Score each candidate once, not inside O(n log n) sort calls.
  const ranked = shuffle(pool.filter(v => !skip.has(v.normalized)), random)
    .map(verse => {
      const close = verseCloseness(target, verse, text) + (exclude.some(v => v.surah === verse.surah) ? .1 : 0);
      const heard = sound.get(key(verse));
      return { verse, score: heard === undefined ? close * (neighbors.length ? .5 : 1) : .5 * heard + .5 * close };
    })
    .sort((a, b) => b.score - a.score).map(row => row.verse);
  const unique = new Map(shuffle(band(ranked, difficulty), random).map(v => [v.normalized, v]));
  return [...unique.values()].slice(0, 2);
}

/** Option orders by difficulty: hard swaps two neighbours, easy scrambles the whole sequence. */
const ORDERS: Record<Difficulty, number[][]> = { hard: [[0, 2, 1], [1, 0, 2]], medium: [[1, 0, 2], [2, 0, 1]], easy: [[2, 1, 0], [1, 2, 0]] };

export function makeChallenge(verses: Verse[], mode: ChallengeMode, difficulty: Difficulty, acoustic: AcousticIndex | null = null, random: () => number = Math.random, lang: "en" | "ar" = "en", text: TextIndex | null = null): Challenge {
  const c = (en: string, ar: string) => lang === "ar" ? ar : en;
  if (lang === "ar") verses = verses.map(v => ({ ...v, name: v.nameAr || v.name }));
  if (verses.length < 3) throw new Error("Cache the Quran before opening challenges.");
  let prompt: Verse, target: Verse;
  let options: ChallengeOption[];
  let right: ChallengeOption;
  let explanation = "";
  const indexed = new Map(verses.map(v => [key(v), v]));
  if (mode === "next" || mode === "order") {
    const eligible = verses.filter(v => {
      const next = indexed.get(`${v.surah}:${v.ayah + 1}`);
      const third = indexed.get(`${v.surah}:${v.ayah + 2}`);
      return next && (mode !== "next" || next.normalized.split(" ").length <= 12) && (mode !== "order" || third && new Set([v.normalized, next.normalized, third.normalized]).size === 3);
    });
    if (!eligible.length) throw new Error("No suitable consecutive ayahs in this scope.");
    if (mode === "order") {
      // Ayahs that resemble each other are harder to put in order.
      const cohesion = (v: Verse) => {
        const [a, b, d] = [0, 1, 2].map(n => indexed.get(`${v.surah}:${v.ayah + n}`)!);
        return (verseCloseness(a, b, text) + verseCloseness(b, d, text) + verseCloseness(a, d, text)) / 3;
      };
      const sample = shuffle(eligible, random).slice(0, 240).map(v => ({ v, score: cohesion(v) })).sort((a, b) => b.score - a.score).map(row => row.v);
      const third = Math.max(1, Math.floor(sample.length / 3));
      prompt = pick(difficulty === "hard" ? sample.slice(0, third) : difficulty === "medium" ? sample.slice(third, 2 * third) : sample.slice(2 * third), random) || pick(sample, random);
    } else prompt = pick(eligible, random);
    target = indexed.get(`${prompt.surah}:${prompt.ayah + 1}`)!;
    if (mode === "order") {
      const third = indexed.get(`${prompt.surah}:${prompt.ayah + 2}`)!;
      const labels = [prompt.text, target.text, third.text];
      const option = (order: number[]) => ({ label: order.map(i => labels[i]).join(" → "), parts: order.map(i => labels[i]) });
      right = option([0, 1, 2]);
      options = [right, ...ORDERS[difficulty].map(option)];
      explanation = c(`Qālūn order: ${prompt.name}, ayahs ${prompt.ayah}–${third.ayah}.`, `ترتيب قالون: ${prompt.name}، الآيات ${prompt.ayah}–${third.ayah}.`);
    } else {
      right = { label: target.text };
      // The prompt is on screen, so it is never offered as its own successor.
      options = [right, ...distractors(target, verses, difficulty, null, text, random, [prompt]).map(v => ({ label: v.text }))];
      explanation = c(`The next Qālūn ayah is ${target.name} ${target.ayah}.`, `الآية التالية بقالون: ${target.name} ${target.ayah}.`);
    }
  } else if (mode === "audio") {
    const playable = verses.filter(v => v.audio && v.duration && v.duration <= 30);
    if (new Set(playable.map(v => v.normalized)).size < 3) throw new Error(c("This reader’s local Qālūn clips are unavailable. Choose a text challenge.", "مقاطع قالون المحلية للقارئ غير متاحة. اختر تحديًا نصيًا."));
    target = prompt = pick(playable, random);
    right = { label: "", audio: target.audio };
    options = [right, ...distractors(target, playable, difficulty, acoustic, text, random).map(v => ({ label: "", audio: v.audio }))];
    explanation = c(`Qālūn reference: ${target.name} ${target.ayah}.`, `مرجع قالون: ${target.name} ${target.ayah}.`);
  } else if (mode === "surah") {
    const origins = new Map<string, Set<number>>();
    for (const v of verses) { const ids = origins.get(v.normalized) || new Set(); ids.add(v.surah); origins.set(v.normalized, ids); }
    // Repeated ayahs can belong to several surahs; never mark one valid origin wrong.
    const eligible = verses.filter(v => origins.get(v.normalized)!.size === 1);
    if (!eligible.length) throw new Error("No unambiguous surah questions in this scope.");
    target = prompt = pick(eligible, random);
    right = { label: target.name };
    // A surah is as confusable as its closest ayah to the prompt.
    const best = new Map<number, { name: string; score: number }>();
    for (const v of verses) {
      if (v.surah === target.surah) continue;
      const score = verseCloseness(target, v, text), seen = best.get(v.surah);
      if (!seen || score > seen.score) best.set(v.surah, { name: v.name, score });
    }
    const ranked = shuffle([...best.values()], random).sort((a, b) => b.score - a.score).map(row => row.name);
    options = [right, ...shuffle(band(ranked, difficulty, .5), random).slice(0, 2).map(label => ({ label }))];
    explanation = c(`This ayah is ${target.name} ${target.ayah} in the Qālūn source.`, `الآية من ${target.name}، رقم ${target.ayah} في نص قالون.`);
  } else {
    const eligible = verses.filter(v => v.text.split(/\s+/).length >= 3);
    if (!eligible.length) throw new Error("No suitable word questions.");
    target = pick(eligible, random);
    const words = target.text.split(/\s+/), missing = Math.floor(random() * words.length);
    right = { label: words[missing] };
    prompt = { ...target, text: words.map((w, i) => i === missing ? "ـــــ" : w).join(" ") };
    // One spelling per letter sequence: a vowel-only or Qālūn-mark difference is never offered as a wrong answer.
    const rightKey = wordKey(right.label);
    const choices = new Map<string, string>();
    for (const v of shuffle(verses, random)) for (const w of v.text.split(/\s+/)) { const k = wordKey(w); if (k && k !== rightKey && !choices.has(k)) choices.set(k, w); }
    const ranked = [...choices].map(([k, label]) => ({ label, score: closeness(rightKey, k, text && cosine(text.words, rightKey, k)) }))
      .sort((a, b) => b.score - a.score).map(row => row.label);
    options = [right, ...shuffle(band(ranked, difficulty, 1.5), random).slice(0, 2).map(label => ({ label }))];
    explanation = c(`The missing word is ${right.label}; ${target.name} ${target.ayah}.`, `الكلمة الناقصة: ${right.label}؛ ${target.name} ${target.ayah}.`);
  }
  options = shuffle(options, random);
  if (options.length !== 3) throw new Error("Not enough distinct answer options in this scope.");
  const textMethod = text
    ? c(`Arabic sentence embeddings (${text.model}) with spelling, rhyme and length closeness. Plausibility only, not meaning or correctness`, `تضمينات دلالية عربية (${text.model}) مع تقارب الحروف والفاصلة والطول؛ للتقارب فقط، لا للمعنى أو الصحة`)
    : c("Text bigram, rhyme and length similarity (not measured acoustic or semantic similarity)", "تشابه الحروف والفاصلة والطول؛ ليس قياسًا صوتيًا أو دلاليًا");
  return { id: `${mode}:${key(target)}`, mode, difficulty, prompt: mode === "order" ? c("Arrange these three consecutive ayahs by choosing the correct order.", "اختر الترتيب الصحيح للآيات الثلاث المتتابعة.") : prompt.text,
    reference: mode === "surah" ? c("Find its surah", "من أي سورة؟") : c(`${prompt.name} · Qālūn ayah ${prompt.ayah}`, `${prompt.name} · الآية ${prompt.ayah} بقالون`),
    surah: target.surah, ayah: target.ayah, target: target.text, options, answer: options.indexOf(right), explanation,
    similarity: mode === "audio" && acoustic?.neighbors[key(target)]?.length ? c(`MFCC recording-similarity blended with ${text ? "Arabic sentence embeddings" : "text similarity"} (not phoneme correctness)`, `تشابه التسجيلات بـ MFCC مع ${text ? "التضمينات الدلالية العربية" : "التشابه النصي"}؛ ليس قياسًا لصحة النطق`) : textMethod };
}
