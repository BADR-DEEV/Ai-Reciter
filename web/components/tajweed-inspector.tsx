"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { X } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { harakatLabel, type Note, type TajweedAyah, type TajweedRules } from "@/lib/tajweed";

type Open = { rules: string[]; notes: Note[]; word: string; anchor: HTMLElement; x: number; y: number; below: boolean };

function place(anchor: HTMLElement) {
  const box = anchor.getBoundingClientRect(), below = box.top < 260;
  return { x: Math.min(Math.max(box.left + box.width / 2, 170), window.innerWidth - 170), y: below ? box.bottom + 8 : box.top - 8, below,
    visible: box.bottom > 0 && box.top < window.innerHeight };
}

/** Tap or click a coloured letter to see its rule, Qālūn count, ways and book page.
 *  One listener on the reader (event delegation) instead of one per letter. */
export function TajweedInspector({ container, rules, ayahs, basmala }: {
  container: RefObject<HTMLElement | null>; rules: TajweedRules | null; ayahs: Map<number, TajweedAyah> | null; basmala?: { n: Note[] };
}) {
  const { lang, c } = useLang();
  const [open, setOpen] = useState<Open | null>(null);
  const card = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = container.current;
    if (!root || !rules) return;
    const onClick = (event: MouseEvent) => {
      const span = (event.target as HTMLElement).closest<HTMLElement>(".tajweed-span[data-tj]");
      if (!span) return;
      const indexes = span.dataset.tj!.split(",").map(Number);
      const word = Number(span.dataset.w);
      const ayah = Number(span.closest<HTMLElement>("[data-ayah]")?.dataset.ayah ?? 0);
      const source = ayah ? ayahs?.get(ayah)?.n : basmala?.n;
      setOpen({
        rules: indexes.map(i => rules.order[i]).filter(Boolean),
        notes: (source || []).filter(n => n[0] === word && indexes.includes(n[1])),
        word: span.closest(".quran-word")?.textContent || span.textContent || "", anchor: span, ...place(span),
      });
    };
    root.addEventListener("click", onClick);
    return () => root.removeEventListener("click", onClick);
  }, [container, rules, ayahs, basmala]);
  useEffect(() => {
    if (!open) return;
    const close = (event: Event) => {
      if (event instanceof KeyboardEvent ? event.key === "Escape" : !card.current?.contains(event.target as Node)
          && !(event.target as HTMLElement).closest?.(".tajweed-span")) setOpen(null);
    };
    // Follow the letter when the page or reader scrolls; close once it leaves the screen.
    let frame = 0;
    const follow = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => {
      const next = place(open.anchor);
      setOpen(current => current && (next.visible && open.anchor.isConnected ? { ...current, ...next } : null));
    }); };
    document.addEventListener("keydown", close); document.addEventListener("pointerdown", close);
    window.addEventListener("scroll", follow, true); window.addEventListener("resize", follow);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", close); document.removeEventListener("pointerdown", close);
      window.removeEventListener("scroll", follow, true); window.removeEventListener("resize", follow);
    };
  }, [open?.anchor]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open || !rules) return null;
  return <div ref={card} className={`tajweed-inspector ${open.below ? "below" : "above"}`} role="dialog" aria-label={c("Tajweed rule", "حكم التجويد")}
    style={{ left: open.x, top: open.y }} dir={lang === "ar" ? "rtl" : "ltr"}>
    <div className="tajweed-inspector-head"><span lang="ar" dir="rtl">{open.word}</span>
      <button className="icon-button" aria-label={c("Close", "إغلاق")} onClick={() => setOpen(null)}><X size={15} /></button></div>
    {open.rules.map(id => {
      const rule = rules.rules[id], group = rules.groups[rule.group];
      const notes = open.notes.filter(n => rules.order[n[1]] === id);
      const wajh = notes.find(n => n[5])?.[5] || rule.wajh;
      return <section key={id}>
        <h4><i style={{ background: group.light }} aria-hidden="true" />{rule[lang]}{rule.harakat && <small>{harakatLabel(rule, lang)}</small>}</h4>
        <p>{lang === "ar" ? rule.explain_ar : rule.explain_en}</p>
        {notes.map((n, i) => (n[3] || n[4]) && <p key={i} className="tajweed-note">{lang === "ar" ? n[4] || n[3] : n[3]}</p>)}
        {wajh && <p className="tajweed-wajh">{c("Qālūn's ways:", "الأوجه لقالون:")} {wajh.join(" · ")}</p>}
        <footer>{group[lang]}{rule.page ? ` · ${c("book p.", "ص")} ${rule.page}` : ""} · <a href={`/tajweed#${rule.group === "riwaya" ? "riwaya" : "rules"}`}>{c("Learn this rule", "تعلّم الحكم")}</a></footer>
      </section>;
    })}
  </div>;
}
