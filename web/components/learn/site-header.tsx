"use client";

import { Flame, Languages } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { DEV_MODE } from "@/lib/dev-mode";
import { streak, useProgress } from "@/lib/learn/progress";

export function SiteHeader({ active, children }: { active: "home" | "learn" | "games" | "profile" | "studio" | "dev"; children?: React.ReactNode }) {
  const { t, toggle, lang } = useLang();
  const { progress, loaded } = useProgress();
  const days = streak(progress.days);
  return <header className="lh-header">
    <a className="lh-wordmark" href="/">rattil<span lang="ar">رَتِّل</span></a>
    <nav className="lh-nav" aria-label={lang === "ar" ? "التنقل الرئيسي" : "Main"}>
      <a href="/" aria-current={active === "home" ? "page" : undefined}>{t("home")}</a>
      <a href="/learn" aria-current={active === "learn" ? "page" : undefined}>{t("learn")}</a>
      <a href="/studio" aria-current={active === "studio" ? "page" : undefined}>{t("studio")}</a>
      <a href="/games" aria-current={active === "games" ? "page" : undefined}>{lang === "ar" ? "تحديات" : "Challenges"}</a>
      <a href="/profile" aria-current={active === "profile" ? "page" : undefined}>{lang === "ar" ? "حسابي" : "Profile"}</a>
      {DEV_MODE && <a href="/dev" aria-current={active === "dev" ? "page" : undefined}>Dev</a>}
    </nav>
    <div className="lh-right">
      {children}
      {loaded && progress.xp > 0 && <span className="lh-stat" title={`${days} ${t("streak")}`}><Flame size={15} /> {days}<span className="lh-stat-xp">{progress.xp} {t("xp")}</span></span>}
      <button className="lh-lang" onClick={toggle} lang={lang === "en" ? "ar" : "en"}><Languages size={15} /> {t("language")}</button>
    </div>
  </header>;
}
