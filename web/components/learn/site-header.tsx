"use client";

import { Flame, Languages } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { streak, useProgress } from "@/lib/learn/progress";

export function SiteHeader({ active }: { active: "home" | "learn" }) {
  const { t, toggle, lang } = useLang();
  const { progress, loaded } = useProgress();
  const days = streak(progress.days);
  return <header className="lh-header">
    <a className="lh-wordmark" href="/">tarteel<span lang="ar">ترتيل</span></a>
    <nav className="lh-nav" aria-label="Main">
      <a href="/" aria-current={active === "home" ? "page" : undefined}>{t("home")}</a>
      <a href="/learn" aria-current={active === "learn" ? "page" : undefined}>{t("learn")}</a>
      <a href="/studio">{t("studio")}</a>
    </nav>
    <div className="lh-right">
      {loaded && progress.xp > 0 && <span className="lh-stat" title={`${days} ${t("streak")}`}><Flame size={15} /> {days}<span className="lh-stat-xp">{progress.xp} {t("xp")}</span></span>}
      <button className="lh-lang" onClick={toggle} lang={lang === "en" ? "ar" : "en"}><Languages size={15} /> {t("language")}</button>
    </div>
  </header>;
}
