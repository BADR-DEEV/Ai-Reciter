"use client";
import { useEffect, useState } from "react";
import { useLang } from "@/lib/i18n";

type Tafsir = { kind: string; book: { name: string; author: string }; entries: { providerAyah: number; text: string }[]; mappingNote: string };
export function TafsirPanel({ surah, ayah, inline = false }: { surah: number; ayah: number; inline?: boolean }) {
  const { lang, c } = useLang();
  const [language, setLanguage] = useState<string>(lang);
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<Tafsir | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { setLanguage(lang); }, [lang]);
  useEffect(() => {
    if (!open && !inline) return;
    const controller = new AbortController();
    setData(null); setError("");
    fetch(`/api/tafsir?surah=${surah}&ayah=${ayah}&lang=${language}`, { signal: controller.signal })
      .then(async r => { const body = await r.json(); if (!r.ok) throw new Error(body.error); return body; })
      .then(setData).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [open, inline, surah, ayah, language]);
  return <section className="tafsir-panel">
    {!inline && <button className="btn-quiet" aria-expanded={open} onClick={() => setOpen(v => !v)}>{c(`Meaning & tafsir · Ayah ${ayah}`, `المعنى والتفسير · الآية ${ayah}`)}</button>}
    {(open || inline) && <div><label>{c("Language", "اللغة")} <select aria-label={c("Tafsir language", "لغة التفسير")} value={language} onChange={e => setLanguage(e.target.value)}><option value="en">English translation</option><option value="ar">التفسير العربي</option></select></label>
      {error ? <p role="alert">{c(error, "التفسير غير متاح الآن. يمكنك متابعة القراءة والمحاولة لاحقًا.")}</p> : data ? <><h3>{language === "ar" ? "التفسير العربي" : "Translation of meanings"} · {data.book.name}</h3><small>{data.book.author} · {c("Quran Tafseer API", "واجهة تفسير القرآن")}</small>
        {data.entries.map(entry => <p key={entry.providerAyah} lang={language} dir={language === "ar" ? "rtl" : "ltr"}>{entry.text}</p>)}<small>{c(data.mappingNote, "يستخدم المصدر ترقيم حفص. الربط نصي وقد يتضمن الشرح نصًا أوسع من هذه الآية برواية قالون؛ ليس بديلًا لنص التلاوة.")}</small>
      </> : <p role="status">{c("Loading commentary…", "جارٍ تحميل التفسير…")}</p>}</div>}
  </section>;
}
