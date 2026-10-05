"use client";

import { useLang } from "@/lib/i18n";

export type TajweedFeedback = {
  words: { word: number; text: string; expected: string[]; applied: string[]; missed: string[]; extra: string[] }[];
  expected: number; applied: number; score: number | null;
  rules: Record<string, { rule: string; en: string; ar: string; expected: number; applied: number }>;
};

/** Words whose tajweed tags were expected but not heard, by word index. */
export function missedByWord(feedback?: TajweedFeedback | null) {
  const map = new Map<number, Set<string>>();
  for (const word of feedback?.words || []) if (word.missed.length) map.set(word.word, new Set(word.missed));
  return map;
}

/** Session summary for "check my tajweed": applied vs expected audible rules. */
export function TajweedFeedbackCard({ feedback }: { feedback: TajweedFeedback[] }) {
  const { lang, c } = useLang();
  const totals = new Map<string, { en: string; ar: string; expected: number; applied: number }>();
  for (const f of feedback) for (const [tag, r] of Object.entries(f.rules)) {
    const row = totals.get(tag) || { en: r.en, ar: r.ar, expected: 0, applied: 0 };
    row.expected += r.expected; row.applied += r.applied; totals.set(tag, row);
  }
  const expected = [...totals.values()].reduce((n, r) => n + r.expected, 0), applied = [...totals.values()].reduce((n, r) => n + r.applied, 0);
  const missed = [...totals.values()].filter(r => r.applied < r.expected).sort((a, b) => (b.expected - b.applied) - (a.expected - a.applied));
  return <section className="tajweed-feedback-card">
    <div className="section-heading"><h3>{c("Ahkam check", "فحص الأحكام")}</h3><span className="mini-pill">{expected ? `${applied} / ${expected}` : "—"}</span></div>
    {expected === 0 ? <p>{c("Recite an ayah to see which tajweed rules the model heard.", "اتلُ آية لترى أي أحكام التجويد سمعها النموذج.")}</p> : <>
      <div className="progress-track"><span style={{ width: `${Math.round(applied / expected * 100)}%` }} /></div>
      {missed.length ? <ul>{missed.slice(0, 6).map(r => <li key={r.en}><span>{r[lang]}</span><strong>{r.expected - r.applied}</strong></li>)}</ul>
        : <p>{c("Every expected rule was heard. Well done!", "سُمعت كل الأحكام المتوقعة. أحسنت!")}</p>}
      <p className="tajweed-feedback-hint">{c("Missed rules are underlined in the text. This is an early tajweed model; let a teacher confirm.", "الأحكام غير المسموعة مسطّرة في النص. هذا نموذج تجويد أولي؛ استعن بمعلم للتأكد.")}</p>
    </>}
  </section>;
}
