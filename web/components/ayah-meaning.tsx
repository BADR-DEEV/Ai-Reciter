"use client";
import { useEffect, useRef, useState } from "react";
import { TafsirPanel } from "./tafsir-panel";
import { useLang } from "@/lib/i18n";
/** Only fetch meanings actually scrolled into view, not a whole long surah. */
export function AyahMeaning({ surah, ayah }: { surah: number; ayah: number }) {
  const element = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const { c } = useLang();
  useEffect(() => {
    if (!element.current) return;
    const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) { setVisible(true); observer.disconnect(); } });
    observer.observe(element.current); return () => observer.disconnect();
  }, []);
  return <div ref={element} className="ayah-meaning-pane" dir="ltr">{visible ? <TafsirPanel surah={surah} ayah={ayah} inline /> : <span>{c("Meaning loads when visible", "يُحمّل المعنى عند ظهور الآية")}</span>}</div>;
}
