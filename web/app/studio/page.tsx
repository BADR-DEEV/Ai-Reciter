"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowRight, BookOpen, GraduationCap, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Expand, Headphones, Leaf, Mic, Moon, Play, RotateCcw, Search, ShieldCheck, Sparkles, Square, Upload, X } from "lucide-react";
import type { Manifest, Surah } from "@/lib/types";
import { useRecitation } from "@/lib/use-recitation";
import { AyahWords } from "@/components/ayah-words";
import { PhoneticAid } from "@/components/phonetic-aid";
import { TafsirPanel } from "@/components/tafsir-panel";
import { ServiceStatus } from "@/components/service-status";

function Ornament({ small = false }: { small?: boolean }) {
  return <span className={`ornament ${small ? "small" : ""}`} aria-hidden="true"><span>✦</span></span>;
}

export default function Home() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [surah, setSurah] = useState<Surah | null>(null);
  const [selected, setSelected] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [picker, setPicker] = useState(false);
  const [query, setQuery] = useState("");
  const [help, setHelp] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);
  const [mode, setMode] = useState<"mushaf" | "text">("mushaf");
  const [fullscreen, setFullscreen] = useState(false);
  const textReader = useRef<HTMLDivElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const [startAyah, setStartAyah] = useState(1);
  const [showPhonetics, setShowPhonetics] = useState(false);
  const recitation = useRecitation(surah, startAyah);
  const active = ["connecting", "listening", "stopping"].includes(recitation.state);

  useEffect(() => {
    // Lessons link here with ?surah=N to recite a surah they just learned.
    const requested = Number(new URLSearchParams(window.location.search).get("surah"));
    if (Number.isInteger(requested) && requested >= 1 && requested <= 114) setSelected(requested);
  }, []);
  useEffect(() => {
    fetch("/quran/manifest.json").then(r => { if (!r.ok) throw new Error("Quran assets are not cached yet. Run the cache script in the README."); return r.json(); }).then(setManifest).catch(error => setLoadError(error.message));
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setLoadError(""); setPageIndex(0);
    fetch(`/quran/surahs/${String(selected).padStart(3, "0")}.json`, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error("Could not load this surah. Check your local Quran cache."); return r.json(); }).then(data => { setSurah(data); setMode("text"); setLoading(false); }).catch(error => { if (error.name !== "AbortError") { setLoadError(error.message); setLoading(false); } });
    return () => controller.abort();
  }, [selected]);
  useEffect(() => {
    const listener = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", listener);
    return () => document.removeEventListener("fullscreenchange", listener);
  }, []);
  useEffect(() => {
    if (!picker && !help) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusable = () => Array.from(dialog?.querySelectorAll<HTMLElement>('button:not(:disabled), input, [tabindex="0"]') || []);
    focusable()[0]?.focus();
    const listener = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setPicker(false); setHelp(false); }
      if (e.key === "Tab") {
        const elements = focusable(), first = elements[0], last = elements[elements.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", listener);
    return () => { document.removeEventListener("keydown", listener); previous?.focus(); };
  }, [picker, help]);

  const pages = useMemo(() => [...new Set(surah?.ayahs.flatMap(a => a.regions.map(r => r.page)) || [])], [surah]);
  const results = recitation.update?.results || {};
  const current = recitation.update?.current ?? (recitation.state === "complete" ? null : startAyah);
  const currentAyah = surah?.ayahs.find(a => a.ayah === current);
  const finalized = Object.values(results).filter(r => r.final);
  const wordResults = Object.values(results).flatMap(result => result.words || []);
  const heardWords = wordResults.filter(word => word.status === "correct").length;
  const omittedWords = wordResults.filter(word => word.status === "missed").length;
  const progress = surah ? Math.round(finalized.length / surah.ayahs.length * 100) : 0;
  const agreement = finalized.length ? Math.round(finalized.reduce((sum, r) => sum + r.score, 0) / finalized.length * 100) : null;
  const pagePath = pages[pageIndex];
  const pageInfo = pagePath ? manifest?.pages[pagePath.split("/").pop()!] : null;
  const filtered = manifest?.surahs.filter(s => `${s.id} ${s.name} ${s.arabic}`.toLowerCase().includes(query.toLowerCase())) || [];

  useEffect(() => {
    const page = currentAyah?.regions[0]?.page;
    if (page && pages.includes(page)) setPageIndex(pages.indexOf(page));
  }, [currentAyah, pages]);

  useEffect(() => {
    if (active) setMode("text");
  }, [active, surah]);

  useEffect(() => {
    if (!active || mode !== "text" || !textReader.current || !current) return;
    const container = textReader.current;
    const ayah = container.querySelector<HTMLElement>(`[data-ayah="${current}"]`);
    if (!ayah) return;
    const matched = ayah.querySelectorAll<HTMLElement>('.quran-word.correct');
    const anchor = matched[matched.length - 1] || ayah;
    const box = anchor.getBoundingClientRect();
    const top = box.top - container.getBoundingClientRect().top;
    if (top < 0 || top + box.height > container.clientHeight) {
      container.scrollTo({ top: container.scrollTop + top - container.clientHeight / 3, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
    }
  }, [active, current, mode, recitation.update?.results]);

  const toggleFullscreen = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { /* Fullscreen is optional in embedded browsers. */ }
  };
  const selectSurah = (id: number) => { recitation.reset(); setStartAyah(1); setSelected(id); setPicker(false); setQuery(""); };

  return <div className="app-shell">
    <aside className="rail" aria-label="Primary navigation">
      <a href="/" className="brand-symbol" aria-label="Rattil home"><Ornament small /></a>
      <div className="rail-divider" />
      <button className="rail-button selected" title="Recitation workspace" aria-label="Recitation workspace"><BookOpen size={21} /></button>
      <a className="rail-button" href="/learn" title="Learn to read" aria-label="Learn to read"><GraduationCap size={21} /></a>
      <button className="rail-button" title="Choose a surah" aria-label="Choose a surah" disabled={active} onClick={() => setPicker(true)}><Search size={20} /></button>
      <button className="rail-button" title="Presentation demo" aria-label="Start presentation demo" disabled={active || loading || !surah} onClick={recitation.startDemo}><Play size={20} /></button>
      <div className="rail-bottom"><button className="rail-button" title="How it works" aria-label="How it works" onClick={() => setHelp(true)}><CircleHelp size={21} /></button><div className="avatar">Q</div></div>
    </aside>

    <div className="workspace">
      <header className="topbar">
        <a className="wordmark" href="/">rattil<span>رَتِّل</span></a>
        <nav className="topnav" aria-label="Workspace"><a href="/learn">Learn to read</a><a href="/games">Challenges</a><a href="/profile">Profile</a><span className="topnav-active">Recitation studio</span><button onClick={() => setHelp(true)}>How it works <ArrowRight size={14} /></button></nav>
        <div className="topbar-right"><span className="local-pill"><span /> Local & private</span><button className="icon-button" title="Presentation fullscreen" aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"} onClick={toggleFullscreen}><Expand size={18} /></button></div>
      </header>

      <main>
        <section className="intro">
          <div><div className="eyebrow"><span className="tiny-star">✦</span> IN THE NAME OF ALLAH, THE MOST MERCIFUL</div><h1>A little practice. A deeper connection.</h1><p>Recite at your own pace. Let every ayah be a step closer.</p></div>
          <div className="intro-calligraphy" lang="ar" dir="rtl">وَرَتِّلِ الْقُرْآنَ تَرْتِيلًا<span>And recite the Quran with measured recitation. · 73:4</span></div>
        </section>

        <section className="session-bar" aria-label="Session settings">
          <div className="session-picker"><span className="field-label">YOUR SURAH</span><button onClick={() => setPicker(true)} disabled={active || !manifest} className="surah-select"><span className="surah-number">{String(selected).padStart(2, "0")}</span><strong>{surah?.name || "Al-Fātiḥah"}</strong><span className="surah-arabic" lang="ar">{surah?.arabic || "الفاتحة"}</span><ChevronDown size={17} /></button></div>
          <div className="session-divider" /><div className="session-riwayah"><span className="field-label">RECITATION TRADITION</span><span><BookOpen size={16} /> Qālūn ʿan Nāfiʿ <span className="mini-pill">Riwayah</span></span></div>
          <div className="session-ready"><span className={`status-dot ${active ? "pulsing" : ""}`} /><span>{recitation.demo ? "Presentation demo" : active ? "Session in progress" : "Your quiet space is ready"}</span></div>
        </section>

        {(loadError || recitation.error) && <div className="error-message" role="alert">{loadError || recitation.error}</div>}
        {surah && !surah.trained && <div className="notice"><CircleHelp size={16} /> This surah is available to read, but lies outside the model’s training coverage. Live matching is experimental.</div>}
        {recitation.demo && <div className="demo-banner"><Sparkles size={16} /><strong>Presentation demo</strong> Simulated word-by-word results, including an omitted word and a missed ayah. No microphone or model inference.</div>}
        <section className="reading-tools"><label>Start / resume from ayah <select aria-label="Starting ayah" disabled={active} value={startAyah} onChange={e => setStartAyah(Number(e.target.value))}>{surah?.ayahs.map(a => <option key={a.ayah} value={a.ayah}>{a.ayah}</option>)}</select></label>
          <label><input type="checkbox" checked={showPhonetics} onChange={e => setShowPhonetics(e.target.checked)} /> Show draft Qālūn phonetics</label><span role="status">{recitation.connection}</span></section>

        <div className="studio-grid">
          <section className="mushaf-card" aria-label="Quran reader">
            <div className="reader-toolbar"><div><BookOpen size={17} /><span>The noble Quran</span><span className="reader-tag">Word by word</span></div><div className="view-switch" aria-label="Reader view"><button className={mode === "mushaf" ? "active" : ""} disabled={active} title="SVG reading view; word highlights use Ayah view" onClick={() => setMode("mushaf")}>Mushaf</button><button className={mode === "text" ? "active" : ""} onClick={() => setMode("text")}>Ayah view</button></div></div>
            <div className={`quran-paper ${selected === 1 ? "fatiha" : ""}`}>
              <div className="paper-corner top-left" /><div className="paper-corner top-right" /><div className="paper-corner bottom-left" /><div className="paper-corner bottom-right" />
              <div className="surah-heading"><Ornament /><div><span className="surah-caption">SURAH {String(selected).padStart(3, "0")}</span><h2 lang="ar" dir="rtl">سُورَةُ {surah?.arabic || "الفَاتِحَة"}</h2><span>{surah?.name || "Al-Fātiḥah"}</span></div><Ornament /></div>
              {selected !== 9 && mode === "text" && <div className="basmalah" lang="ar" dir="rtl">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</div>}
              {loading ? <div className="reader-placeholder"><Leaf className="loading-leaf" /> Preparing your mushaf…</div> : mode === "mushaf" && pageInfo ? <div className="svg-page" style={{ aspectRatio: pageInfo.viewBox.split(" ").slice(2).join(" / ") }}>
                {/* Deliberately rendered as an image: remote SVG markup cannot execute. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={pagePath} alt={`Quran page ${pagePath.split("/").pop()?.replace(".svg", "")}`} draggable={false} />
                <svg viewBox={pageInfo.viewBox} aria-label="Ayah recitation highlights" className="ayah-overlay">
                  {surah?.ayahs.flatMap(ayah => ayah.regions.filter(region => region.page === pagePath).map((region, index) => {
                    const result = results[ayah.ayah];
                    const status = surah.preciseGeometry ? result?.status || (ayah.ayah === current && active ? "listening" : "pending") : "pending";
                    return <polygon key={`${ayah.ayah}-${index}`} points={region.polygon} className={`ayah-region ${status}`}><title>Ayah {ayah.ayah}: {status}{result ? ` · ${Math.round(result.score * 100)}% text agreement` : ""}</title></polygon>;
                  }))}
                </svg>
              </div> : <div className="text-mushaf word-mode" ref={textReader} lang="ar" dir="rtl">{surah?.ayahs.map(ayah => <div key={ayah.ayah} data-ayah={ayah.ayah} className={`text-ayah ${results[ayah.ayah]?.status || (ayah.ayah === current && active ? "listening" : "pending")}`}><AyahWords ayah={ayah} result={results[ayah.ayah]} /><span className="ayah-medallion">{ayah.ayah.toLocaleString("ar")}</span>{showPhonetics && <PhoneticAid text={ayah.text} />}</div>)}</div>}
              {mode === "mushaf" && <p className="geometry-note">Switch to Ayah view for individual heard and omitted word colors.</p>}
              {surah && !surah.preciseGeometry && <p className="geometry-note">Qālūn ayah divisions differ from this page artwork. Live highlighting uses Ayah view.</p>}
              {surah?.ayahs.some(a => !a.regions.length) && mode === "mushaf" && <p className="geometry-note">Some ayahs have no API polygon. Use Ayah view for complete text.</p>}
              <div className="paper-bottom"><span /><span>۞</span><span /></div>
            </div>
            <div className="reader-footer"><div className="legend"><span><i className="correct" />Heard</span><span><i className="missed" />Omitted</span><span><i className="pending" />Not reached</span></div><div className="page-controls"><button aria-label="Previous Quran page" disabled={pageIndex === 0 || !pages.length} onClick={() => setPageIndex(value => value - 1)}><ChevronLeft size={16} /></button><span>Page {pagePath ? Number(pagePath.split("/").pop()?.replace(".svg", "")) : "—"}</span><button aria-label="Next Quran page" disabled={pageIndex >= pages.length - 1} onClick={() => setPageIndex(value => value + 1)}><ChevronRight size={16} /></button></div></div>
          </section>

          <aside className="session-panel">
            <ServiceStatus />
            <section className="listening-card"><div className="panel-heading"><span><span className="status-dot" /> {recitation.state === "complete" ? "SESSION COMPLETE" : active ? "LIVE RECITATION" : "RECITATION COMPANION"}</span><Headphones size={17} /></div>
              <div className={`mic-orbit ${recitation.state === "listening" ? "is-listening" : ""}`}><div className="orbit-ring" /><div className="mic-core">{recitation.state === "complete" ? <Check size={31} /> : <Mic size={29} />}</div><span className="orbit-star">✦</span></div>
              <h2>{recitation.state === "connecting" ? "Connecting to your model…" : recitation.state === "stopping" ? "Finishing your recitation…" : recitation.state === "complete" ? "A beautiful step forward." : recitation.state === "listening" ? "We’re listening." : "Your voice. Your journey."}</h2>
              <p>{recitation.state === "complete" ? "Take a moment to reflect, then return to the ayahs that need a little care." : active ? "Keep reciting naturally. Your place follows you, one ayah at a time." : "Take a breath, make your intention, and begin whenever you’re ready."}</p>
              <div className="waveform" aria-hidden="true">{Array.from({ length: 39 }, (_, i) => <span key={i} style={{ height: `${5 + (active ? recitation.level * (14 + Math.sin(i * 1.7) * 12 + Math.sin(i * 0.4) * 12) : 0)}px`, opacity: active ? 0.45 + recitation.level * 0.55 : 0.25 }} />)}</div>
              <button className={`start-button ${active ? "recording" : ""}`} disabled={loading || !surah || Boolean(loadError) || recitation.state === "stopping"} onClick={() => active ? recitation.stop() : void recitation.start()}>{active ? <Square size={15} fill="currentColor" /> : <Mic size={18} />}{recitation.state === "connecting" ? "Cancel" : recitation.state === "stopping" ? "Processing…" : active ? "Finish recitation" : recitation.state === "complete" ? "Recite again" : "Begin recitation"}{!active && <ArrowRight size={17} />}</button>
              <input ref={audioInput} type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" hidden aria-label="Choose audio recording" onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void recitation.start(file); }} />
              <button className="upload-button" disabled={active || loading || !surah || Boolean(loadError)} onClick={() => audioInput.current?.click()}><Upload size={15} /> Upload audio to test</button>
              {recitation.fileName && <div className="upload-file-note" title={recitation.fileName}><strong>{recitation.fileName}</strong><span>Silent replay · real model results</span></div>}
              <div className="privacy-note"><ShieldCheck size={13} /> Audio stays on your machine. Always.</div>
              {!active && recitation.state !== "complete" && <button className="demo-link" disabled={!surah || loading} onClick={recitation.startDemo}><Play size={12} /> Try the presentation demo</button>}
            </section>

            <section className="progress-card"><div className="section-heading"><h3>Your session</h3><button className="icon-button" aria-label="Reset session" title="Reset session" disabled={active} onClick={recitation.reset}><RotateCcw size={15} /></button></div><div className="progress-title"><strong>{finalized.length}<span> / {surah?.ayahCount || 7} ayahs</span></strong><span>{progress}%</span></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><div className="stat-row"><div><span className="stat-icon green"><CheckCircle2 size={15} /></span><strong>{heardWords}</strong><span>Heard words</span></div><div><span className="stat-icon red"><CircleHelp size={15} /></span><strong>{omittedWords}</strong><span>Omitted words</span></div><div><span className="stat-icon gray"><Moon size={15} /></span><strong>{`${Math.floor(recitation.seconds / 60)}:${String(recitation.seconds % 60).padStart(2, "0")}`}</strong><span>Time</span></div></div></section>

            <section className="current-card"><div className="section-heading"><h3>{recitation.state === "complete" ? "Session reflection" : "Your place"}</h3><span className="mini-pill">{current ? `Ayah ${current}` : "Complete"}</span></div>{currentAyah ? <p lang="ar" dir="rtl" className="current-text"><AyahWords ayah={currentAyah} result={results[currentAyah.ayah]} /></p> : <p className="reflection-text">{heardWords} words matched. {omittedWords ? `${omittedWords} to revisit gently.` : "May your practice bring you closer."}</p>}<div className="current-hint"><ArrowDown size={14} /><span>{active ? "The last word moves you to the next ayah" : "Start with the first ayah, and follow your flow"}</span></div></section>
          </aside>
        </div>
        {currentAyah && <TafsirPanel surah={selected} ayah={currentAyah.ayah} />}

        <section className="insight-row"><div className="insight-card"><span className="insight-icon"><Sparkles size={19} /></span><div><h3>Presence, not perfection.</h3><p>A missed ayah won’t interrupt you. Keep your flow, then come back with care.</p></div></div><div className="model-status"><span className="model-dot" /><div><strong>{recitation.demo ? "Simulated presentation" : "Whisper base · Qālūn"}</strong><span>{recitation.device === "cuda" ? "GPU connected" : recitation.device === "cpu" ? "CPU connected" : "Local model"} · 65% text-match threshold{agreement !== null ? ` · ${agreement}% session agreement` : ""}</span></div></div></section>
        {recitation.update?.transcript && <section className="transcript-card"><span className="field-label">{recitation.demo ? "SIMULATED TRANSCRIPT" : "WHAT THE MODEL HEARD"}</span><p lang="ar" dir="rtl">{recitation.update.transcript}</p>{recitation.update.latency_ms && <span className="decode-time">Last decode: {recitation.update.latency_ms} ms</span>}</section>}
        <footer className="footer"><span><Leaf size={13} /> Made for mindful recitation.</span><span>Recognition aid, not a tajweed assessment. <button onClick={() => setHelp(true)}>Learn more</button></span></footer>
      </main>
    </div>

    {picker && <div className="modal-backdrop" onClick={() => setPicker(false)}><section className="modal surah-modal" role="dialog" aria-modal="true" aria-labelledby="surah-modal-title" onClick={e => e.stopPropagation()}><div className="modal-heading"><div><span className="eyebrow">BEGIN A NEW CHAPTER</span><h2 id="surah-modal-title">Choose your surah</h2></div><button className="icon-button" aria-label="Close surah picker" onClick={() => setPicker(false)}><X size={21} /></button></div><label className="search-field"><Search size={18} /><input autoFocus value={query} onChange={e => setQuery(e.target.value)} placeholder="Search name or surah number…" aria-label="Search surahs" /></label><p className="picker-note">All Quran text is local. <span className="trained-label">Trained</span> marks the model’s 38-surah coverage.</p><div className="surah-list">{filtered.map(s => <button className={selected === s.id ? "chosen" : ""} key={s.id} onClick={() => selectSurah(s.id)}><span className="list-number">{String(s.id).padStart(2, "0")}</span><div><strong>{s.name}</strong><span>{s.ayahCount} ayahs {s.trained && <em>Trained</em>}</span></div><span className="list-arabic" lang="ar">{s.arabic}</span>{selected === s.id ? <Check size={17} /> : <ChevronRight size={17} />}</button>)}{!filtered.length && <p className="empty-search">No surahs match your search.</p>}</div></section></div>}
    {help && <div className="modal-backdrop" onClick={() => setHelp(false)}>
      <section className="modal help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={e => e.stopPropagation()}>
        <div className="modal-heading"><div><span className="eyebrow">A GENTLE COMPANION</span><h2 id="help-title">Keep your recitation flowing.</h2></div><button className="icon-button" aria-label="Close help" onClick={() => setHelp(false)}><X size={21} /></button></div>
        <ol className="help-steps">
          <li><span>01</span><div><h3>Choose a surah</h3><p>Your session begins with the first Qālūn ayah. Fātiḥah’s basmalah is an unnumbered introduction. Recite live or upload a recording for silent testing.</p></div></li>
          <li><span>02</span><div><h3>Follow each word</h3><p>Matched words turn green individually. Confirmed omissions turn red; words you have not reached stay gray. Pausing halfway through an ayah does not mark its remaining words omitted.</p></div></li>
          <li><span>03</span><div><h3>Keep your flow</h3><p>A stable last-word match advances your place. Recognizing a later ayah confirms skipped words without stopping you. At 65% ordered word agreement, the ayah summary counts as matched, but individual omitted words still stay red.</p></div></li>
        </ol>
        <div className="help-note"><ShieldCheck size={22} /><p>Scores compare recognized words with the expected text. They are not model probabilities, pronunciation verification, or tajweed judgments. A qualified teacher remains essential. Word colors use Qālūn Ayah view; the supplied SVG artwork only has whole-ayah polygons.</p></div>
        <button className="start-button" onClick={() => setHelp(false)}>Return to your quiet space <ArrowRight size={17} /></button>
      </section>
    </div>}
  </div>;
}
