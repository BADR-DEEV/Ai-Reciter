"use client";

import { useLang } from "@/lib/i18n";
import { wordText, type TajweedAyah, type TajweedRules } from "@/lib/tajweed";

type Names = { en?: string; ar?: string; fix_en?: string; fix_ar?: string };
export type TajweedMistake = Names & { word: number; text: string; tag: string };
export type TajweedFeedback = {
  words: { word: number; text: string; expected: string[]; applied: string[]; missed: string[]; extra: string[] }[];
  mistakes?: TajweedMistake[];
  expected: number; applied: number; score: number | null;
  rules: Record<string, Names & { expected: number; applied: number }>;
};

/** Words whose tajweed tokens were expected but not heard, by word index. */
export function missedByWord(feedback?: TajweedFeedback | null) {
  const map = new Map<number, Set<string>>();
  for (const word of feedback?.words || []) if (word.missed.length) map.set(word.word, new Set(word.missed));
  return map;
}

function mistakesOf(feedback: TajweedFeedback, rules: TajweedRules | null): TajweedMistake[] {
  return feedback.mistakes ?? feedback.words.flatMap(w => w.missed.map(tag => ({ word: w.word, text: w.text, tag, ...(rules?.tags?.[tag] || {}) })));
}

/** "Check my ahkam": what was heard, and each rule the reciter missed with how to fix it. */
export function TajweedFeedbackCard({ entries, rules, ayahs, onPick }: {
  entries: [number, TajweedFeedback][]; rules: TajweedRules | null; ayahs: Map<number, TajweedAyah> | null;
  onPick: (ayah: number, word: number) => void;
}) {
  const { lang, c } = useLang();
  const expected = entries.reduce((n, [, f]) => n + f.expected, 0), applied = entries.reduce((n, [, f]) => n + f.applied, 0);
  const mistakes = entries.flatMap(([ayah, f]) => mistakesOf(f, rules).map(m => ({ ...m, ayah })));
  return <section className="tajweed-feedback-card">
    <div className="section-heading"><h3>{c("Ahkam check", "فحص الأحكام")}</h3><span className="mini-pill">{expected ? `${applied} / ${expected}` : "—"}</span></div>
    {expected === 0 ? <p>{c("Recite an ayah to see which tajweed rules the model heard.", "اتلُ آية لترى أي أحكام التجويد سمعها النموذج.")}</p> : <>
      <div className="progress-track"><span style={{ width: `${Math.round(applied / expected * 100)}%` }} /></div>
      {mistakes.length ? <>
        <h4>{c(`What to fix (${mistakes.length})`, `ما يحتاج إلى تصحيح (${mistakes.length})`)}</h4>
        <ol className="tajweed-mistakes">{mistakes.slice(0, 12).map((m, i) => {
          const word = ayahs?.get(m.ayah)?.w[m.word];
          return <li key={`${m.ayah}-${m.word}-${m.tag}-${i}`}><button onClick={() => onPick(m.ayah, m.word)}>
            <span className="tajweed-mistake-word" lang="ar" dir="rtl">{word ? wordText(word) : m.text}</span>
            <span className="tajweed-mistake-rule"><strong>{(lang === "ar" ? m.ar : m.en) || m.tag}</strong> · {c("ayah", "آية")} {m.ayah}</span>
            <span className="tajweed-mistake-fix">{lang === "ar" ? m.fix_ar : m.fix_en}</span>
          </button></li>;
        })}</ol>
        {mistakes.length > 12 && <p>{c(`…and ${mistakes.length - 12} more, underlined in the text.`, `…و${mistakes.length - 12} أخرى مسطّرة في النص.`)}</p>}
      </> : <p>{c("Every expected rule was heard. Well done!", "سُمعت كل الأحكام المتوقعة. أحسنت!")}</p>}
      <p className="tajweed-feedback-hint">{c("Wavy underlines mark missed rules in the text. Early tajweed model: let a teacher confirm.", "الخط المتموج يحدد الأحكام غير المسموعة في النص. نموذج تجويد أولي؛ استعن بمعلم للتأكد.")}</p>
    </>}
  </section>;
}
