"use client";

import { Ear, Mic, Volume2, BookOpenText } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { useLang } from "@/lib/i18n";
import { LESSONS, UNITS } from "@/lib/learn/curriculum";
import { LETTERS } from "@/lib/learn/letters";
import { useProgress } from "@/lib/learn/progress";
import { speak } from "@/lib/learn/speech";

export default function Landing() {
  const { t, lang } = useLang();
  const { progress } = useProgress();
  const started = Object.keys(progress.lessons).length > 0;
  const next = LESSONS.find(l => !progress.lessons[l.id]?.done) || LESSONS[0];

  return <div className="learn-shell" dir={lang === "ar" ? "rtl" : "ltr"} lang={lang}>
    <SiteHeader active="home" />
    <main className="landing">
      <section className="hero">
        <div className="hero-copy">
          <p className="hero-kicker">{t("heroEyebrow")}</p>
          <h1>{t("heroTitle")}</h1>
          <p className="hero-body">{t("heroBody")}</p>
          <div className="hero-actions">
            <a className="btn-primary" href={started ? `/learn/${next.id}` : "/learn/welcome"}>{started ? t("continueLearning") : t("startLearning")}</a>
            <a className="btn-quiet" href="/studio">{t("iCanRead")}</a>
          </div>
        </div>
        <div className="alphabet" dir="rtl" aria-label="The Arabic alphabet. Select a letter to hear it.">
          {LETTERS.map(l => <button key={l.char} className={`alpha-tile ${l.noEnglish ? "rare" : ""}`} onClick={() => speak(l.practice)} title={`${l.name}: ${l.sound}`}>
            <span className="alpha-glyph" lang="ar">{l.char}</span><span className="alpha-name">{l.name}</span>
          </button>)}
          <p className="alphabet-note" dir="ltr"><Volume2 size={14} /> Tap a letter to hear it. Gold letters have no English equivalent.</p>
        </div>
      </section>

      <section className="features" aria-label="How it works">
        <div><BookOpenText size={22} /><h2>{t("feature1Title")}</h2><p>{t("feature1Body")}</p></div>
        <div><Mic size={22} /><h2>{t("feature2Title")}</h2><p>{t("feature2Body")}</p></div>
        <div><Ear size={22} /><h2>{t("feature3Title")}</h2><p>{t("feature3Body")}</p></div>
      </section>

      <section className="path-preview">
        <h2>{t("pathTitle")}</h2>
        <ol>
          {UNITS.map(u => <li key={u.id}>
            <span className="path-ar" lang="ar">{u.titleAr}</span>
            <span><strong>{u.title}</strong><span>{u.summary}</span></span>
            <span className="path-count">{u.lessons.length} {u.lessons.length === 1 ? t("lesson") : t("lessons")}</span>
          </li>)}
        </ol>
        <p className="privacy-note">{t("privacy")}</p>
      </section>
    </main>
  </div>;
}
