"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, BookOpen, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Expand, Headphones, Leaf, Mic, Play, RotateCcw, Search, ShieldCheck, Sparkles, Square, Upload, X } from "lucide-react";
import type { Manifest, Surah } from "@/lib/types";
import { useRecitation } from "@/lib/use-recitation";
import { useLang } from "@/lib/i18n";
import { useReferenceReciter } from "@/lib/use-reference-reciter";
import { qaloonG2P } from "@/lib/qaloon-g2p";
import { stopAudio } from "@/lib/learn/speech";
import type { TajweedMushaf } from "@/lib/tajweed";
import { SiteHeader } from "@/components/learn/site-header";
import { AyahWords } from "@/components/ayah-words";
import { AyahListen } from "@/components/ayah-listen";
import type { PlaybackCursor } from "@/lib/playback";
import { ReciterSelector } from "@/components/reciter-selector";
import { PhoneticAid } from "@/components/phonetic-aid";
import { TafsirPanel } from "@/components/tafsir-panel";
import { ServiceStatus } from "@/components/service-status";
import { TajweedLegend } from "@/components/tajweed-legend";

type View = "mushaf" | "text" | "phonetic" | "meaning";
function Ornament() { return <span className="ornament" aria-hidden="true"><span>✦</span></span>; }

export default function Studio() {
  const { lang, c } = useLang();
  const { reciter, selectReciter } = useReferenceReciter();
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [surah, setSurah] = useState<Surah | null>(null);
  const [selected, setSelected] = useState(1);
  const [loading, setLoading] = useState(true), [loadError, setLoadError] = useState("");
  const [picker, setPicker] = useState(false), [query, setQuery] = useState(""), [help, setHelp] = useState(false);
  const [pageIndex, setPageIndex] = useState(0), [mode, setMode] = useState<View>("text");
  const [fullscreen, setFullscreen] = useState(false), [startAyah, setStartAyah] = useState(1);
  const [showPhonetics, setShowPhonetics] = useState(false), [showTajweed, setShowTajweed] = useState(false);
  const [tajweed, setTajweed] = useState<TajweedMushaf | null>(null), [tajweedError, setTajweedError] = useState(false);
  const [playback, setPlayback] = useState<PlaybackCursor | null>(null), [focusAyah, setFocusAyah] = useState(1);
  const followPlayback = useCallback((ayah: number, cursor: PlaybackCursor | null) => {
    if (cursor) setFocusAyah(ayah);
    setPlayback(previous => cursor
      ? previous?.ayah === cursor.ayah && previous.word === cursor.word && previous.tracking === cursor.tracking ? previous : cursor
      : previous?.ayah === ayah ? null : previous);
  }, []);
  const textReader = useRef<HTMLDivElement>(null), audioInput = useRef<HTMLInputElement>(null);
  const recitation = useRecitation(surah, startAyah);
  const active = ["connecting", "listening", "stopping"].includes(recitation.state);

  useEffect(() => { const requested = Number(new URLSearchParams(window.location.search).get("surah")); if (Number.isInteger(requested) && requested >= 1 && requested <= 114) setSelected(requested); }, []);
  useEffect(() => { const controller = new AbortController(); fetch("/quran/manifest.json", { signal: controller.signal }).then(r => { if (!r.ok) throw new Error("Quran assets are not cached. Run the cache script in the README."); return r.json(); }).then(setManifest).catch(e => { if (e.name !== "AbortError") setLoadError(e.message); }); return () => controller.abort(); }, []);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setLoadError(""); setPageIndex(0); setFocusAyah(1); setPlayback(null); stopAudio();
    fetch(`/quran/surahs/${String(selected).padStart(3, "0")}.json`, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error("Could not load this surah. Check the local Quran cache."); return r.json(); }).then(data => { setSurah(data); setLoading(false); }).catch(e => { if (e.name !== "AbortError") { setLoadError(e.message); setLoading(false); } });
    return () => controller.abort();
  }, [selected]);
  useEffect(() => {
    if (!showTajweed) return;
    const controller = new AbortController(); setTajweed(null); setTajweedError(false);
    fetch(`/api/tajweed?surah=${selected}`, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(setTajweed).catch(e => { if (e.name !== "AbortError") setTajweedError(true); });
    return () => controller.abort();
  }, [selected, showTajweed]);
  useEffect(() => { const listener = () => setFullscreen(Boolean(document.fullscreenElement)); document.addEventListener("fullscreenchange", listener); return () => { document.removeEventListener("fullscreenchange", listener); stopAudio(); }; }, []);
  useEffect(() => {
    if (!picker && !help) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusable = () => Array.from(dialog?.querySelectorAll<HTMLElement>('button:not(:disabled), input, [tabindex="0"]') || []);
    focusable()[0]?.focus();
    const listener = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setPicker(false); setHelp(false); }
      if (e.key === "Tab") { const elements = focusable(), first = elements[0], last = elements[elements.length - 1]; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); } }
    };
    document.addEventListener("keydown", listener); return () => { document.removeEventListener("keydown", listener); previous?.focus(); };
  }, [picker, help]);

  const pages = useMemo(() => [...new Set(surah?.ayahs.flatMap(a => a.regions.map(r => r.page)) || [])], [surah]);
  const results = recitation.update?.results || {};
  const current = recitation.update?.current ?? (recitation.state === "complete" ? null : startAyah);
  const place = playback?.ayah ?? (active || recitation.state === "complete" ? current : focusAyah);
  const readingAyah = surah?.ayahs.find(a => a.ayah === place);
  const liveWords = current ? results[current]?.words || [] : [];
  const liveWord = active ? liveWords.map((word, index) => word.status === "correct" ? index : -1).filter(index => index >= 0).at(-1) ?? null : null;
  const wordCursor = (ayah: number) => playback?.ayah === ayah ? playback.word : !playback && active && current === ayah ? liveWord : null;
  const finalized = Object.values(results).filter(r => r.final);
  const wordResults = Object.values(results).flatMap(r => r.words || []);
  const heardWords = wordResults.filter(w => w.status === "correct").length, omittedWords = wordResults.filter(w => w.status === "missed").length;
  const progress = surah ? Math.round(finalized.length / surah.ayahs.length * 100) : 0;
  const pagePath = pages[pageIndex], pageInfo = pagePath ? manifest?.pages[pagePath.split("/").pop()!] : null;
  const filtered = manifest?.surahs.filter(s => `${s.id} ${s.name} ${s.arabic}`.toLowerCase().includes(query.toLowerCase())) || [];
  const annotations = new Map(tajweed?.surahs.find(s => s.id === selected)?.ayahs.map(a => [a.ayah, a]) || []);
  useEffect(() => { const page = readingAyah?.regions[0]?.page; if (page && pages.includes(page)) setPageIndex(pages.indexOf(page)); }, [readingAyah, pages]);
  useEffect(() => { if (active) { stopAudio(); setPlayback(null); } }, [active]);
  useEffect(() => {
    if ((!active && !playback) || mode === "mushaf" || !textReader.current || !place) return;
    const container = textReader.current, ayah = container.querySelector<HTMLElement>(`[data-ayah="${place}"]`);
    if (!ayah) return;
    const anchor = ayah.querySelector<HTMLElement>(".playback-word") || ayah;
    const box = anchor.getBoundingClientRect(), top = box.top - container.getBoundingClientRect().top;
    if (top < 0 || top + box.height > container.clientHeight) container.scrollTo({ top: container.scrollTop + top - container.clientHeight / 3, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
  }, [active, place, mode, playback?.word, recitation.update?.results]);
  const selectSurah = (id: number) => { recitation.reset(); stopAudio(); setStartAyah(1); setSelected(id); setPicker(false); setQuery(""); };
  const toggleFullscreen = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { /* Optional. */ } };

  return <div className="app-shell studio-shell" dir={lang === "ar" ? "rtl" : "ltr"}>
    <div className="workspace">
      <SiteHeader active="studio"><button className="icon-button" aria-label={c(fullscreen ? "Exit fullscreen" : "Enter fullscreen", fullscreen ? "الخروج من ملء الشاشة" : "ملء الشاشة")} onClick={toggleFullscreen}><Expand size={18} /></button></SiteHeader>
      <main>
        <section className="intro"><div><div className="eyebrow">{c("IN THE NAME OF ALLAH, THE MOST MERCIFUL", "بسم الله الرحمن الرحيم")}</div><h1>{c("A little practice. A deeper connection.", "تدريب قليل، وصلة أعمق بالقرآن.")}</h1><p>{c("Recite at your own pace. Let every ayah be a step closer.", "اتلُ على مهل؛ واجعل كل آية خطوة أقرب.")}</p></div><button className="btn-quiet" onClick={() => setHelp(true)}>{c("How it works", "كيف يعمل")} <CircleHelp size={16} /></button></section>
        <section className="session-bar" aria-label={c("Session settings", "إعدادات الجلسة")}>
          <div className="session-picker"><span className="field-label">{c("YOUR SURAH", "السورة")}</span><button disabled={active || !manifest} onClick={() => setPicker(true)} className="surah-select"><span className="surah-number">{String(selected).padStart(2, "0")}</span><strong>{lang === "ar" ? surah?.arabic || "الفاتحة" : surah?.name || "Al-Fātiḥah"}</strong><ChevronDown size={17} /></button></div>
          <div className="session-riwayah"><span className="field-label">{c("RECITATION TRADITION", "الرواية")}</span><span><BookOpen size={16} />{c("Qālūn ʿan Nāfiʿ", "قالون عن نافع")}</span></div>
          <ReciterSelector value={reciter} onChange={selectReciter} disabled={active} />
        </section>
        {(loadError || recitation.error) && <div className="error-message" role="alert">{c(loadError || recitation.error, "تعذّر تحميل النص أو الاتصال بالنموذج. تحقّق من الخادم المحلي؛ نتائجك الحالية محفوظة.")}</div>}
        {surah && !surah.trained && <div className="notice">{c("Reading is available; this surah is outside ASR training coverage. Live matching is experimental. Local ayah audio covers Fātiḥah/Juz ʿAmma only.", "القراءة متاحة، لكن السورة خارج نطاق تدريب التعرف الصوتي، والمطابقة تجريبية. المقاطع المحلية تشمل الفاتحة وجزء عمّ فقط.")}</div>}
        {recitation.demo && <div className="demo-banner"><Sparkles size={16} /><strong>{c("Presentation demo", "عرض توضيحي")}</strong>{c("Simulated word-by-word results. No microphone or model inference.", "نتائج محاكاة كلمة بكلمة؛ دون ميكروفون أو استدلال بالنموذج.")}</div>}
        <section className="reading-tools"><label>{c("Start / resume from ayah", "ابدأ أو استأنف من الآية")}<select aria-label={c("Starting ayah", "آية البداية")} disabled={active} value={startAyah} onChange={e => { setStartAyah(Number(e.target.value)); setFocusAyah(Number(e.target.value)); }}>{surah?.ayahs.map(a => <option key={a.ayah} value={a.ayah}>{a.ayah}</option>)}</select></label>
          <label><input type="checkbox" checked={showPhonetics} onChange={e => setShowPhonetics(e.target.checked)} />{c("Show draft Qālūn phonetics", "أظهر النقل الصوتي التجريبي لقالون")}</label>
          <label><input type="checkbox" checked={showTajweed} onChange={e => setShowTajweed(e.target.checked)} />{c("Draft tajweed colors", "ألوان التجويد التجريبية")}</label>
          <span role="status">{c(recitation.connection, active ? "الجلسة متصلة؛ راقب حالة الخادم" : "جاهز")}</span></section>
        {showTajweed && !tajweed && <p role="status">{tajweedError ? c("Tajweed file unavailable. It is generated when the app server first starts; reload in a moment, or run src/learning/build_qalon_tajweed.py. Canonical text is unchanged.", "ملف التجويد غير متاح. يُنشأ تلقائيًا عند أول تشغيل للخادم؛ أعد تحميل الصفحة بعد لحظة. النص الأصلي لم يتغير.") : c("Loading reviewable annotations…", "جارٍ تحميل العلامات للمراجعة…")}</p>}
        <div className="studio-grid">
          <aside className="commentary-sidebar" dir={lang === "ar" ? "rtl" : "ltr"} aria-label={c("Translation and tafsir", "الترجمة والتفسير")}>
            <div className="section-heading"><h2>{c("Translation & tafsir", "الترجمة والتفسير")}</h2><span className="mini-pill">{place ?? "—"}</span></div>
            <label>{c("Follow ayah", "تابع الآية")}<select aria-label={c("Commentary ayah", "آية التفسير")} value={place || focusAyah} disabled={active || Boolean(playback)} onChange={e => setFocusAyah(Number(e.target.value))}>{surah?.ayahs.map(a => <option key={a.ayah} value={a.ayah}>{a.ayah}</option>)}</select></label>
            {readingAyah && <div className="ayah-meaning-pane"><TafsirPanel surah={selected} ayah={readingAyah.ayah} inline fixedLanguage="en" /><TafsirPanel surah={selected} ayah={readingAyah.ayah} inline fixedLanguage="ar" /></div>}
          </aside>
          <section className="mushaf-card" dir={lang === "ar" ? "rtl" : "ltr"} aria-label={c("Quran reader", "قارئ القرآن")}>
            <div className="reader-toolbar"><span><BookOpen size={17} />{c("The noble Quran", "القرآن الكريم")}</span><div className="view-switch" aria-label={c("Reader view", "طريقة العرض")}>{([
              ["mushaf", c("Mushaf", "المصحف المصوّر")], ["text", c("Ayah view", "النص العربي")], ["phonetic", c("Phonetics", "النقل الصوتي")], ["meaning", c("Meaning + phonetics", "المعنى والنقل الصوتي")],
            ] as [View, string][]).map(([id, label]) => <button key={id} aria-pressed={mode === id} className={mode === id ? "active" : ""} onClick={() => setMode(id)}>{label}</button>)}</div></div>
            <div className={`quran-paper ${selected === 1 ? "fatiha" : ""}`}>
              <div className="surah-heading"><Ornament /><div><span className="surah-caption">{c("SURAH", "سورة")} {String(selected).padStart(3, "0")}</span><h2 lang="ar" dir="rtl">سُورَةُ {surah?.arabic || "الفَاتِحَة"}</h2><span>{surah?.name}</span></div><Ornament /></div>
              {selected !== 9 && mode === "text" && <div className="basmalah" lang="ar" dir="rtl"><AyahWords ayah={{ ayah: 0, text: "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", normalized: "بسم الله الرحمن الرحيم", regions: [] }} tajweed={showTajweed ? tajweed?.basmalah : undefined} rules={tajweed?.rules} lang={lang} /></div>}
              {(mode === "phonetic" || mode === "meaning") && <p className="reading-aid-warning">{c("Draft Qālūn reading aid; not teacher-approved. Listen and learn Arabic letters alongside it.", "نقل صوتي تجريبي لقالون لم يعتمده معلم. استمع وتعلّم الحروف العربية معه.")}</p>}
              {loading || mode === "mushaf" && !pageInfo ? <div className="reader-placeholder"><Leaf className="loading-leaf" />{c("Preparing your mushaf…", "جارٍ تجهيز المصحف…")}</div> : mode === "mushaf" && pageInfo ? <div className="svg-page" style={{ aspectRatio: pageInfo.viewBox.split(" ").slice(2).join(" / ") }}>
                {/* External SVG remains an image, never executable markup. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={pagePath} alt={c(`Quran page ${pagePath.split("/").pop()?.replace(".svg", "")}`, `صفحة القرآن ${pageIndex + 1}`)} draggable={false} />
                <svg viewBox={pageInfo.viewBox} aria-label={c("Ayah recitation highlights", "تمييز الآيات")} className="ayah-overlay">{surah?.ayahs.flatMap(a => a.regions.filter(r => r.page === pagePath).map((r, i) => <polygon key={`${a.ayah}-${i}`} data-ayah={a.ayah} points={r.polygon} onClick={() => { if (!active && !playback) setFocusAyah(a.ayah); }} className={`ayah-region ${surah.preciseGeometry ? results[a.ayah]?.status || "pending" : "pending"} ${a.ayah === place && (playback || active) ? "playback-ayah" : ""}`}><title>{c(`Ayah ${a.ayah}`, `الآية ${a.ayah}`)}</title></polygon>))}</svg>
              </div> : <div className={`text-mushaf word-mode ${mode === "phonetic" ? "phonetic-text" : mode === "meaning" ? "meaning-text" : ""}`} ref={textReader} lang={mode === "phonetic" ? "en" : "ar"} dir={mode === "phonetic" ? "ltr" : "rtl"}>{surah?.ayahs.map(ayah => {
                const phonetics = qaloonG2P(ayah.text);
                const cursor = wordCursor(ayah.ayah);
                const display = (ayah.displayText || ayah.text).replace(/[\u0660-\u0669\d]+/g, "").trim();
                const phoneticCursor = phonetics.words.length === display.split(/\s+/).length ? cursor : null;
                return <div key={ayah.ayah} data-ayah={ayah.ayah} onClick={() => { if (!active && !playback) setFocusAyah(ayah.ayah); }} className={`text-ayah ${results[ayah.ayah]?.status || (ayah.ayah === current && active ? "listening" : "pending")} ${playback?.ayah === ayah.ayah ? "playback-ayah" : ""}`}>
                  <div className="ayah-reading-line">{mode === "phonetic" ? phonetics.words.map((word, i) => <span key={i}><span className={`quran-word ${results[ayah.ayah]?.words?.[i]?.status || "pending"} ${phoneticCursor === i ? "playback-word" : ""}`} data-word-index={i} aria-current={phoneticCursor === i ? "true" : undefined}>{word}</span>{" "}</span>) : <AyahWords ayah={ayah} result={results[ayah.ayah]} tajweed={showTajweed ? annotations.get(ayah.ayah) : undefined} rules={tajweed?.rules} lang={lang} activeWord={cursor} />}<span className="ayah-medallion">{ayah.ayah.toLocaleString(lang)}</span></div>
                  {(showPhonetics && mode === "text" || mode === "meaning") && <PhoneticAid text={ayah.text} activeWord={phoneticCursor} />}
                  {mode === "phonetic" && phonetics.warnings.length > 0 && <details className="phonetic-review"><summary>{c("Pronunciation review notes", "ملاحظات مراجعة النطق")}</summary><ul>{phonetics.warnings.map(w => <li key={w}>{w}</li>)}</ul></details>}
                  <AyahListen surah={selected} ayah={ayah.ayah} reciter={reciter} disabled={active} displayText={display} onPlayback={followPlayback} />
                </div>;
              })}</div>}
              {mode === "mushaf" && <><p className="geometry-note">{c("Supplied SVG artwork is not certified Qālūn spelling. Whole-ayah overlays use draft text-aligned regions, not word timing.", "المصحف المصوّر ليس توثيقًا لرسم قالون. تمييز الآية يعتمد على مناطق ربط نصي تجريبية لا على توقيت الكلمات.")}</p>{readingAyah && <AyahListen surah={selected} ayah={readingAyah.ayah} reciter={reciter} disabled={active} onPlayback={followPlayback} />}</>}
            </div>
            <div className="reader-footer"><div className="legend"><span><i className="correct" />{c("Heard", "مطابق")}</span><span><i className="missed" />{c("Not matched", "غير مطابق")}</span><span><i className="pending" />{c("Not reached", "لم تصل إليه")}</span></div>{mode === "mushaf" && <div className="page-controls"><button aria-label={c("Previous Quran page", "الصفحة السابقة")} disabled={pageIndex === 0} onClick={() => setPageIndex(n => n - 1)}><ChevronLeft size={16} /></button><span>{c("Page", "صفحة")} {pagePath ? Number(pagePath.split("/").pop()?.replace(".svg", "")) : "—"}</span><button aria-label={c("Next Quran page", "الصفحة التالية")} disabled={pageIndex >= pages.length - 1} onClick={() => setPageIndex(n => n + 1)}><ChevronRight size={16} /></button></div>}</div>
            {showTajweed && tajweed && <TajweedLegend rules={tajweed.rules} legend={tajweed.legend} />}
          </section>
          <aside className="session-panel" dir={lang === "ar" ? "rtl" : "ltr"}>
            <ServiceStatus />
            <section className="listening-card"><div className="panel-heading"><span>{c(active ? "LIVE RECITATION" : "RECITATION COMPANION", active ? "تلاوة مباشرة" : "رفيق التلاوة")}</span><Headphones size={17} /></div><div className={`mic-orbit ${recitation.state === "listening" ? "is-listening" : ""}`}><div className="orbit-ring" /><div className="mic-core">{recitation.state === "complete" ? <Check size={31} /> : <Mic size={29} />}</div></div>
              <h2>{c(recitation.state === "connecting" ? "Connecting to your model…" : recitation.state === "stopping" ? "Finishing your recitation…" : recitation.state === "complete" ? "A beautiful step forward." : recitation.state === "listening" ? "We’re listening." : "Your voice. Your journey.", recitation.state === "connecting" ? "جارٍ الاتصال بالنموذج…" : recitation.state === "stopping" ? "جارٍ إنهاء التلاوة…" : recitation.state === "complete" ? "خطوة جميلة إلى الأمام." : recitation.state === "listening" ? "نستمع إليك." : "صوتك، ورحلتك.")}</h2><p>{c(active ? "Keep reciting naturally. Your place follows you, one ayah at a time." : "Take a breath and begin whenever you’re ready.", active ? "تابع التلاوة بطبيعتك؛ نتابع موضعك آية بآية." : "خذ نفسًا وابدأ حين تكون مستعدًا.")}</p>
              <div className="waveform" aria-hidden="true">{Array.from({ length: 39 }, (_, i) => <span key={i} style={{ height: `${5 + (active ? recitation.level * (14 + Math.sin(i * 1.7) * 12) : 0)}px` }} />)}</div>
              <button className={`start-button ${active ? "recording" : ""}`} disabled={loading || !surah || Boolean(loadError) || recitation.state === "stopping"} onClick={() => { stopAudio(); if (active) recitation.stop(); else void recitation.start(); }}>{active ? <Square size={15} /> : <Mic size={18} />}{c(recitation.state === "connecting" ? "Cancel" : recitation.state === "stopping" ? "Processing…" : active ? "Finish recitation" : recitation.state === "complete" ? "Recite again" : "Begin recitation", recitation.state === "connecting" ? "إلغاء" : recitation.state === "stopping" ? "جارٍ المعالجة…" : active ? "إنهاء التلاوة" : recitation.state === "complete" ? "اتلُ مرة أخرى" : "ابدأ التلاوة")}</button>
              <input ref={audioInput} type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" hidden aria-label={c("Choose audio recording", "اختر تسجيلًا صوتيًا")} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) { stopAudio(); void recitation.start(file); } }} />
              <button className="upload-button" disabled={active || loading || !surah || Boolean(loadError)} onClick={() => audioInput.current?.click()}><Upload size={15} />{c("Upload audio to test", "ارفع تسجيلًا للتجربة")}</button>
               {recitation.fileName && <div className="upload-file-note"><strong>{recitation.fileName}</strong><span>{c("Accelerated local processing · real model results", "معالجة محلية مُسرّعة · نتائج حقيقية للنموذج")}</span></div>}
              <div className="privacy-note"><ShieldCheck size={13} />{c("Inference is local. Reference listening uses cached audio.", "الاستدلال محلي، والاستماع يستخدم المقاطع المحفوظة.")}</div>
              {!active && recitation.state !== "complete" && <button className="demo-link" disabled={!surah || loading} onClick={recitation.startDemo}><Play size={12} />{c("Try the presentation demo", "جرّب العرض التوضيحي")}</button>}
            </section>
            <section className="progress-card"><div className="section-heading"><h3>{c("Your session", "جلستك")}</h3><button className="icon-button" aria-label={c("Reset session", "أعد ضبط الجلسة")} disabled={active} onClick={recitation.reset}><RotateCcw size={15} /></button></div><div className="progress-title"><strong>{finalized.length}<span> / {surah?.ayahCount || 7} {c("ayahs", "آيات")}</span></strong><span>{progress}%</span></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><div className="stat-row"><div><CheckCircle2 size={15} /><strong>{heardWords}</strong><span>{c("Heard words", "كلمات مطابقة")}</span></div><div><CircleHelp size={15} /><strong>{omittedWords}</strong><span>{c("Unmatched words", "كلمات غير مطابقة")}</span></div><div><strong>{Math.floor(recitation.seconds / 60)}:{String(recitation.seconds % 60).padStart(2, "0")}</strong><span>{c("Time", "الوقت")}</span></div></div></section>
            <section className="current-card"><div className="section-heading"><h3>{c("Your place", "موضعك")}</h3><span className="mini-pill">{place ? c(`Ayah ${place}`, `الآية ${place}`) : c("Complete", "اكتملت")}</span></div>{readingAyah && (mode === "phonetic" ? <p className="current-phonetic" lang="en" dir="ltr">{qaloonG2P(readingAyah.text).text}</p> : <p lang="ar" dir="rtl" className="current-text"><AyahWords ayah={readingAyah} result={results[readingAyah.ayah]} lang={lang} /></p>)}<p className="current-hint">{c("Matching tracks recognized text, not pronunciation or tajweed correctness.", "المطابقة تتابع النص المتعرَّف عليه، وليست حكمًا على النطق أو التجويد.")}</p></section>
          </aside>
        </div>
        {recitation.update?.transcript && <section className="transcript-card"><span className="field-label">{c(recitation.demo ? "SIMULATED TRANSCRIPT" : "WHAT THE MODEL HEARD", recitation.demo ? "نص محاكاة" : "ما تعرَّف عليه النموذج")}</span><p lang="ar" dir="rtl">{recitation.update.transcript}</p></section>}
        <footer className="footer"><span><Leaf size={13} />{c("Made for mindful recitation.", "للتلاوة بتدبر.")}</span><span>{c("Recognition aid, not a tajweed assessment.", "وسيلة تعرف صوتي، وليست تقييمًا للتجويد.")} <button onClick={() => setHelp(true)}>{c("Learn more", "اعرف المزيد")}</button></span></footer>
      </main>
    </div>
    {picker && <div className="modal-backdrop" onClick={() => setPicker(false)}><section className="modal surah-modal" role="dialog" aria-modal="true" aria-labelledby="surah-modal-title" onClick={e => e.stopPropagation()}><div className="modal-heading"><h2 id="surah-modal-title">{c("Choose your surah", "اختر السورة")}</h2><button className="icon-button" aria-label={c("Close surah picker", "أغلق قائمة السور")} onClick={() => setPicker(false)}><X size={21} /></button></div><label className="search-field"><Search size={18} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder={c("Search name or surah number…", "ابحث باسم السورة أو رقمها…")} aria-label={c("Search surahs", "ابحث عن سورة")} /></label><p className="picker-note">{c("All Quran text is local. Trained marks the 38-surah speech-model scope.", "نص القرآن محفوظ محليًا. «مدرَّب» يحدد نطاق النموذج الصوتي: ٣٨ سورة.")}</p><div className="surah-list">{filtered.map(s => <button className={selected === s.id ? "chosen" : ""} key={s.id} onClick={() => selectSurah(s.id)}><span className="list-number">{String(s.id).padStart(2, "0")}</span><div><strong>{lang === "ar" ? s.arabic : s.name}</strong><span>{s.ayahCount} {c("ayahs", "آيات")} {s.trained && <em>{c("Trained", "مدرَّب")}</em>}</span></div><span className="list-arabic" lang="ar">{s.arabic}</span>{selected === s.id ? <Check size={17} /> : <ChevronRight size={17} />}</button>)}</div></section></div>}
    {help && <div className="modal-backdrop" onClick={() => setHelp(false)}><section className="modal help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={e => e.stopPropagation()}><div className="modal-heading"><h2 id="help-title">{c("Keep your recitation flowing.", "حافظ على سلاسة تلاوتك.")}</h2><button className="icon-button" aria-label={c("Close help", "أغلق المساعدة")} onClick={() => setHelp(false)}><X size={21} /></button></div><ol className="help-steps">
      <li><span>01</span><div><h3>{c("Choose a surah", "اختر سورة")}</h3><p>{c("Choose your starting Qālūn ayah. Fātiḥah’s basmalah is an unnumbered introduction. Listen with Al-Huthaify by default or choose a different Qālūn reader.", "اختر آية البداية برواية قالون. البسملة في الفاتحة مقدمة غير مرقمة. القارئ الافتراضي الحذيفي، ويمكن اختيار قارئ آخر بقالون.")}</p></div></li>
      <li><span>02</span><div><h3>{c("Choose your reading view", "اختر طريقة القراءة")}</h3><p>{c("SVG artwork, Arabic, large phonetics, or meanings with phonetics. Recording disables reference playback to avoid scoring the reference audio.", "مصحف مصوّر، أو نص عربي، أو نقل صوتي كبير، أو معانٍ مع النقل الصوتي. يُوقف الاستماع المرجعي أثناء التسجيل حتى لا نقيس صوت القارئ بدلًا من صوتك.")}</p></div></li>
      <li><span>03</span><div><h3>{c("Understand the colors", "افهم الألوان")}</h3><p>{c("Speech-match backgrounds and draft tajweed letter colors are separate. Tajweed lengths are harakat, not seconds; natural madd is 2, not universally 4.", "خلفيات المطابقة منفصلة عن ألوان حروف التجويد التجريبية. أطوال المد بالحركات لا بالثواني؛ الطبيعي حركتان وليس دائمًا أربعًا.")}</p></div></li>
    </ol><div className="help-note"><ShieldCheck size={22} /><p>{c("Scores compare recognized words with expected text. They are not model probabilities, pronunciation verification, or tajweed judgments. Draft phonetics and tajweed require a qualified Qālūn teacher. SVG artwork is not a certified Qālūn mushaf.", "الدرجات تقارن النص المتعرَّف عليه بالنص المتوقع، وليست احتمالات النموذج أو إثباتًا لصحة النطق والتجويد. النقل الصوتي وعلامات التجويد بحاجة إلى معلم مؤهل بقالون. الصور ليست مصحف قالون معتمدًا.")}</p></div><button className="start-button" onClick={() => setHelp(false)}>{c("Return to your quiet space", "عُد إلى مساحة التلاوة")}<ArrowRight size={17} /></button></section></div>}
  </div>;
}
