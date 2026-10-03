"use client";
import { useEffect, useState } from "react";

type Tafsir = { kind: string; book: { name: string; author: string }; entries: { providerAyah: number; text: string }[]; mappingNote: string };
export function TafsirPanel({ surah, ayah }: { surah: number; ayah: number }) {
  const [language, setLanguage] = useState("en");
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<Tafsir | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setData(null); setError("");
    fetch(`/api/tafsir?surah=${surah}&ayah=${ayah}&lang=${language}`, { signal: controller.signal })
      .then(async r => { const body = await r.json(); if (!r.ok) throw new Error(body.error); return body; })
      .then(setData).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [open, surah, ayah, language]);
  return <section className="tafsir-panel">
    <button className="btn-quiet" aria-expanded={open} onClick={() => setOpen(v => !v)}>Meaning & tafsir · Ayah {ayah}</button>
    {open && <div><label>Language <select aria-label="Tafsir language" value={language} onChange={e => setLanguage(e.target.value)}><option value="en">English translation</option><option value="ar">التفسير العربي</option></select></label>
      {error ? <p role="alert">{error}</p> : data ? <><h3>{data.kind} · {data.book.name}</h3><small>{data.book.author} · Quran Tafseer API</small>
        {data.entries.map(entry => <p key={entry.providerAyah} lang={language} dir={language === "ar" ? "rtl" : "ltr"}>{entry.text}</p>)}<small>{data.mappingNote}</small>
      </> : <p role="status">Loading commentary…</p>}</div>}
  </section>;
}
