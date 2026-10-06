"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { usePathname } from "next/navigation";
import { BookOpen, Menu, Search, Trophy, X } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { DEV_MODE } from "@/lib/dev-mode";

/** Move the actual recording button into the phone dock, preserving its handlers and state. */
export function MobileAudioControl({ children, keepInline = false }: { children: ReactNode; keepInline?: boolean }) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const update = () => setTarget(media.matches ? document.getElementById("mobile-audio-slot") : null);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return target ? <>{keepInline && children}{createPortal(<div className="mobile-audio-control">{children}</div>, target)}</> : children;
}

export function MobileNavigation() {
  const path = usePathname();
  const { lang, c } = useLang();
  const [more, setMore] = useState(false);
  const menu = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => setMore(false), [path]);
  useEffect(() => {
    if (!more) return;
    menu.current?.querySelector<HTMLElement>("a")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setMore(false); trigger.current?.focus(); }
      if (event.key === "Tab") {
        const items = Array.from(menu.current?.querySelectorAll<HTMLElement>("a, button") || []);
        const first = items[0], last = items.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [more]);
  const current = (href: string) => path === href || (href !== "/" && path.startsWith(`${href}/`));
  const close = () => { setMore(false); trigger.current?.focus(); };
  const extra = [
    ["/", c("Home", "الرئيسية")],
    ["/learn", c("Learn · letters & reading", "تعلّم · الحروف والقراءة")],
    ["/tajweed", c("Tajweed", "التجويد")],
    ["/profile", c("Profile", "حسابي")],
    ...(DEV_MODE ? [["/dev", "Dev"]] : []),
  ];
  return <>
    {more && <div className="mobile-menu-backdrop" onClick={close}>
      <div className="mobile-menu" ref={menu} role="dialog" aria-modal="true" aria-label={c("More navigation", "المزيد من الصفحات")} dir={lang === "ar" ? "rtl" : "ltr"} onClick={event => event.stopPropagation()}>
        <div className="mobile-menu-heading"><strong>{c("Explore Rattil", "اكتشف رتّل")}</strong><button onClick={close} aria-label={c("Close menu", "أغلق القائمة")}><X size={22} /></button></div>
        {extra.map(([href, label]) => <a key={href} href={href} aria-current={current(href) ? "page" : undefined}>{label}</a>)}
      </div>
    </div>}
    <nav className="mobile-nav" aria-label={c("Mobile navigation", "التنقل على الهاتف")} dir={lang === "ar" ? "rtl" : "ltr"}>
      <a href="/studio" aria-current={current("/studio") ? "page" : undefined}><BookOpen size={23} /><span>{c("Studio", "التلاوة")}</span></a>
      <a href="/search" aria-current={current("/search") ? "page" : undefined}><Search size={22} /><span>{c("Find ayah", "ابحث")}</span></a>
      <div className="mobile-audio-slot" id="mobile-audio-slot" />
      <a href="/games" aria-current={current("/games") ? "page" : undefined}><Trophy size={22} /><span>{c("Challenges", "تحديات")}</span></a>
      <button ref={trigger} onClick={() => setMore(value => !value)} aria-expanded={more} aria-haspopup="dialog" className={extra.some(([href]) => current(href)) ? "active" : ""}><Menu size={22} /><span>{c("Menu", "القائمة")}</span></button>
    </nav>
  </>;
}
