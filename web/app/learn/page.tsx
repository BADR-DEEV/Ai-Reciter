"use client";

import { Check, Play, RotateCcw } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { useLang } from "@/lib/i18n";
import { LESSONS, UNITS } from "@/lib/learn/curriculum";
import { streak, useProgress } from "@/lib/learn/progress";

export default function Course() {
  const { t, lang } = useLang();
  const { progress, loaded, resetAll } = useProgress();
  const doneCount = LESSONS.filter(l => progress.lessons[l.id]?.done).length;
  const next = LESSONS.find(l => !progress.lessons[l.id]?.done);

  return <div className="learn-shell" dir={lang === "ar" ? "rtl" : "ltr"} lang={lang}>
    <SiteHeader active="learn" />
    <main className="course">
      <section className="course-head">
        <div>
          <h1>{t("course")}</h1>
          <p>{doneCount} / {LESSONS.length} {t("lessons")} {t("lessonsDone")} · {progress.xp} {t("xp")} · {streak(progress.days)} {t("streak")}</p>
          <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={LESSONS.length} aria-valuenow={doneCount}><span style={{ width: `${doneCount / LESSONS.length * 100}%` }} /></div>
        </div>
        {loaded && next && <a className="btn-primary" href={`/learn/${next.id}`}><Play size={16} /> {doneCount ? t("continueLearning") : t("startLearning")}</a>}
      </section>

      {UNITS.map((unit, u) => <section className="unit" key={unit.id} aria-labelledby={`unit-${unit.id}`}>
        <header className="unit-head">
          <span className="unit-index">{u + 1}</span>
          <div><h2 id={`unit-${unit.id}`}>{lang === "ar" ? unit.titleAr : unit.title}</h2><p>{unit.summary}</p></div>
          <span className="unit-ar" lang="ar">{unit.titleAr}</span>
        </header>
        <ol className="lesson-list">
          {unit.lessons.map(lesson => {
            const record = progress.lessons[lesson.id];
            const isNext = next?.id === lesson.id;
            return <li key={lesson.id} className={`${record?.done ? "done" : ""} ${isNext ? "next" : ""}`}>
              <a href={`/learn/${lesson.id}`}>
                <span className="lesson-state" aria-hidden="true">{record?.done ? <Check size={15} /> : isNext ? <Play size={13} /> : null}</span>
                <span className="lesson-title"><strong>{lang === "ar" ? lesson.titleAr : lesson.title}</strong><span lang={/[؀-ۿ]/.test(lesson.summary) ? "ar" : undefined}>{lesson.summary}</span></span>
                <span className="lesson-meta">{record?.done ? `${Math.round(record.accuracy * 100)}%` : isNext ? t("next") : `${lesson.steps.length} steps`}</span>
              </a>
            </li>;
          })}
        </ol>
      </section>)}

      {doneCount > 0 && <button className="btn-text reset" onClick={() => { if (window.confirm("Reset all lesson progress on this device?")) resetAll(); }}><RotateCcw size={14} /> {t("resetProgress")}</button>}
    </main>
  </div>;
}
