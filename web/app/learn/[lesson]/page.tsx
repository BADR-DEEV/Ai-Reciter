"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { CheckCircle2, X } from "lucide-react";
import { Rich } from "@/components/learn/rich";
import { AyahStep, ChoiceStep, FormsStep, IntroStep, LetterStep, PairStep, SayStep, StudioStep, VowelStep } from "@/components/learn/steps";
import { useLang } from "@/lib/i18n";
import { modelOnline } from "@/lib/learn/api";
import { findLesson, nextLesson, type Step } from "@/lib/learn/curriculum";
import { useProgress } from "@/lib/learn/progress";
import { stopClip } from "@/lib/learn/speech";
import { SiteHeader } from "@/components/learn/site-header";

const SCORED = new Set<Step["kind"]>(["choice", "say", "ayah"]);

export default function LessonPage() {
  const { lesson: id } = useParams<{ lesson: string }>();
  const lesson = findLesson(id);
  const { t, lang, c } = useLang();
  const { complete } = useProgress();
  // Wrongly answered choices are repeated once at the end of the lesson.
  const [queue, setQueue] = useState<number[]>(() => lesson ? lesson.steps.map((_, i) => i) : []);
  const [position, setPosition] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [spoken, setSpoken] = useState<number | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [finished, setFinished] = useState(false);

  const needsModel = useMemo(() => lesson?.steps.some(s => s.kind === "say" || s.kind === "ayah"), [lesson]);
  useEffect(() => {
    if (!needsModel) return;
    const controller = new AbortController();
    void modelOnline(controller.signal).then(setOnline);
    return () => controller.abort();
  }, [needsModel]);
  useEffect(() => () => stopClip(), []);

  if (!lesson) return <div className="learn-shell"><main className="course"><h1>Lesson not found</h1><a className="btn-primary" href="/learn">{t("backToCourse")}</a></main></div>;

  const stepIndex = queue[position];
  const step = lesson.steps[stepIndex];
  const scoreKey = `${stepIndex}:${position}`;
  const values = Object.values(scores);
  const accuracy = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 1;
  const repeat = position >= lesson.steps.length;

  const advance = () => {
    stopClip();
    if (step.kind === "say" || step.kind === "ayah") {
      if (spoken !== null) setScores(s => ({ ...s, [scoreKey]: spoken }));
    }
    setPicked(null); setRevealed(false); setSpoken(null);
    if (position + 1 >= queue.length) {
      const finalValues = Object.values({ ...scores, ...(spoken !== null ? { [scoreKey]: spoken } : {}) });
      const final = finalValues.length ? finalValues.reduce((a, b) => a + b, 0) / finalValues.length : 1;
      complete(lesson.id, final, 10 + Math.round(final * 10));
      setFinished(true);
      return;
    }
    setPosition(p => p + 1);
  };

  const check = () => {
    if (step.kind !== "choice" || picked === null) return;
    const right = picked === step.answer;
    setRevealed(true);
    // Only the first attempt counts toward accuracy.
    if (!repeat) setScores(s => ({ ...s, [scoreKey]: right ? 1 : 0 }));
    if (!right && !repeat) setQueue(q => [...q, stepIndex]);
  };

  const next = nextLesson(lesson.id);
  if (finished) {
    return <div className="learn-shell" dir={lang === "ar" ? "rtl" : "ltr"} lang={lang}><SiteHeader active="learn" /><main className="lesson-done">
      <CheckCircle2 size={54} />
      <h1>{t("lessonComplete")}</h1>
      <p className="done-title">{lesson.title}</p>
      <div className="done-stats"><div><strong>{Math.round(accuracy * 100)}%</strong><span>{t("accuracy")}</span></div><div><strong>+{10 + Math.round(accuracy * 10)}</strong><span>{t("xp")}</span></div></div>
      <div className="hero-actions">
        {next && <a className="btn-primary" href={`/learn/${next.id}`}>{t("nextLesson")}: {next.title}</a>}
        <a className="btn-quiet" href="/learn">{t("backToCourse")}</a>
      </div>
    </main></div>;
  }

  const isChoice = step.kind === "choice";
  const correct = isChoice && revealed && picked === step.answer;
  const canSkip = (step.kind === "say" || step.kind === "ayah") && spoken === null;

  return <div className="learn-shell lesson-shell" lang={lang}>
    <SiteHeader active="learn" />
    <header className="lesson-top">
      <a href="/learn" className="btn-icon" aria-label={t("exit")}><X size={20} /></a>
      <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={queue.length} aria-valuenow={position}><span style={{ width: `${position / queue.length * 100}%` }} /></div>
      <span className="lesson-count">{position + 1}/{queue.length}</span>
    </header>
    {lang === "ar" && <p className="course-language-note" dir="rtl">المحتوى التعليمي التفصيلي متاح بالإنجليزية حاليًا؛ لم نعرض ترجمة عربية غير مراجعة. أدوات التلاوة والاستماع متاحة بالعربية.</p>}
    <main className="lesson-stage" dir="ltr" lang="en" key={`${position}-${stepIndex}`}>
      {repeat && <p className="repeat-note">{c("Let’s try this one again.", "لنحاول هذه مرة أخرى.")}</p>}
      {step.kind === "intro" && <IntroStep step={step} />}
      {step.kind === "letter" && <LetterStep step={step} />}
      {step.kind === "forms" && <FormsStep step={step} />}
      {step.kind === "pair" && <PairStep step={step} />}
      {step.kind === "vowel" && <VowelStep step={step} />}
      {step.kind === "choice" && <ChoiceStep step={step} picked={picked} revealed={revealed} onPick={setPicked} />}
      {step.kind === "say" && <SayStep step={step} online={online} onScore={setSpoken} />}
      {step.kind === "ayah" && <AyahStep step={step} online={online} onScore={setSpoken} />}
      {step.kind === "studio" && <StudioStep step={step} />}
    </main>
    <footer className={`lesson-foot ${revealed ? (correct ? "good" : "bad") : ""}`}>
      <div className="foot-message" role="status">
        {revealed && <><strong>{correct ? t("correct") : t("incorrect")}</strong>{isChoice && step.explain && <span><Rich text={step.explain} /></span>}{isChoice && !correct && <span>Answer: <b lang={step.options[step.answer].arabic ? "ar" : undefined}>{step.options[step.answer].label}</b></span>}</>}
      </div>
      <div className="foot-actions">
        {canSkip && <button className="btn-quiet" onClick={advance}>{t("skip")}</button>}
        {isChoice && !revealed
          ? <button className="btn-primary" disabled={picked === null} onClick={check}>{t("check")}</button>
          : (!canSkip || !SCORED.has(step.kind)) && <button className="btn-primary" onClick={advance} autoFocus>{t("continue")}</button>}
      </div>
    </footer>
  </div>;
}
