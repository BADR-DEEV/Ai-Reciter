"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Loader2, Mic, RotateCcw, Square, Upload } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { MobileAudioControl } from "@/components/mobile-navigation";
import { AyahWords } from "@/components/ayah-words";
import { AyahListen } from "@/components/ayah-listen";
import { TafsirPanel } from "@/components/tafsir-panel";
import { useLang } from "@/lib/i18n";
import { useRecorder, wavUrl } from "@/lib/learn/use-recorder";
import { stopAudio } from "@/lib/learn/speech";
import { useAvailableModels } from "@/lib/models";
import { useReferenceReciter } from "@/lib/use-reference-reciter";
import { qaloonG2P } from "@/lib/qaloon-g2p";
import { JUZ_NUMBERS, JUZ_STARTS } from "@/lib/juz";
import { decodeClip, juzRanges, MAX_CLIP_SECONDS, searchRecitation, type SearchResponse, type SearchResult } from "@/lib/search";
import type { Manifest, SurahInfo } from "@/lib/types";
import type { ReciterID } from "@/lib/reciters";

type Scope = { only: boolean; juz: number[] };
type Phase = "idle" | "listening" | "reading" | "searching" | "done";
type Names = Map<number, SurahInfo>;
const SCOPE_KEY = "rattil.search-scope.v1";
const RATE = 16000;
const CONFIDENCE = { high: ["Strong match", "تطابق قوي"], medium: ["Possible match", "تطابق محتمل"], low: ["Weak match", "تطابق ضعيف"] } as const;

function MatchCard({ result, heard, primary, names, reciter }: { result: SearchResult; heard: number; primary: boolean; names: Names; reciter: ReciterID }) {
  const { lang, c } = useLang();
  const n = (value: number) => value.toLocaleString(lang);
  const { start, end } = result, first = result.ayahs[0];
  const other = names.get(end.surah), otherName = other ? (lang === "ar" ? other.arabic : other.name) : String(end.surah);
  const place = start.surah !== end.surah
    ? c(`Ayah ${n(start.ayah)} to ${otherName} ${n(end.ayah)}`, `من الآية ${n(start.ayah)} إلى ${otherName} ${n(end.ayah)}`)
    : start.ayah === end.ayah ? c(`Ayah ${n(start.ayah)}`, `الآية ${n(start.ayah)}`) : c(`Ayahs ${n(start.ayah)}–${n(end.ayah)}`, `الآيات ${n(start.ayah)}–${n(end.ayah)}`);
  const Heading = primary ? "h2" : "h3";
  return <article className={`find-match ${primary ? "primary" : ""}`}>
    <header className="find-match-head">
      <div>
        <Heading>{lang === "ar" ? <><span lang="ar" className="find-surah-ar">{result.arabic}</span><span lang="en">{result.name}</span></>
          : <><span>{result.name}</span><span lang="ar" dir="rtl" className="find-surah-ar">{result.arabic}</span></>}</Heading>
        <p className="find-where">{place}<span className="find-juz-tag">{c(`Juz ${juzRanges(result.juz, lang)}`, `الجزء ${juzRanges(result.juz, lang)}`)}</span></p>
      </div>
      <span className={`find-confidence ${result.confidence}`}>{c(CONFIDENCE[result.confidence][0], CONFIDENCE[result.confidence][1])}</span>
    </header>
    <div className="find-text">{result.ayahs.map(ayah => <div key={`${ayah.surah}:${ayah.ayah}`} className="find-ayah">
      <p lang="ar" dir="rtl" className="find-arabic">
        <AyahWords ayah={{ ayah: ayah.ayah, text: ayah.text, displayText: ayah.displayText, normalized: "", regions: [] }}
          result={{ status: "correct", score: result.coverage, missing: [], final: true, words: ayah.words }} lang={lang} />
        <span className="ayah-medallion">{n(ayah.ayah)}</span>
      </p>
      <p lang="en" dir="ltr" className="find-latin">{qaloonG2P(ayah.text).text}</p>
    </div>)}</div>
    <p className="find-evidence">
      {c(`${n(Math.min(result.matched_words, heard))} of the ${n(heard)} words we heard line up with this passage.`, `${n(Math.min(result.matched_words, heard))} من ${n(heard)} كلمات سمعناها تطابق هذا الموضع.`)}
      {!result.trained && <> {c("The model wasn’t trained on this surah, so treat the match as a guide.", "لم يُدرَّب النموذج على هذه السورة، فاعتبر النتيجة إرشادية.")}</>}
    </p>
    <div className="find-actions">
      {primary && <AyahListen surah={first.surah} ayah={first.ayah} reciter={reciter} />}
      <a className="btn-quiet" href={`/studio?surah=${first.surah}&ayah=${first.ayah}`}>{c("Recite it in the studio", "اتلُها في الاستوديو")}</a>
    </div>
    {primary && <TafsirPanel surah={first.surah} ayah={first.ayah} />}
  </article>;
}

export default function FindAyah() {
  const { lang, c } = useLang();
  const recorder = useRecorder({ maxSeconds: 20, silenceSeconds: 2 });
  const { online } = useAvailableModels();
  const { reciter } = useReferenceReciter();
  const [names, setNames] = useState<Names>(new Map());
  const [scope, setScope] = useState<Scope>({ only: false, juz: [30] });
  const [phase, setPhase] = useState<Phase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState("");
  const [trimmed, setTrimmed] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [searchedScope, setSearchedScope] = useState("");
  const [showMore, setShowMore] = useState(false);
  const [clipUrl, setClipUrl] = useState("");
  const clip = useRef<Float32Array | null>(null);
  const request = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultsSection = useRef<HTMLElement>(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(SCOPE_KEY) || "null");
      if (saved && typeof saved.only === "boolean" && Array.isArray(saved.juz)) setScope({ only: saved.only, juz: saved.juz.filter((n: unknown) => JUZ_NUMBERS.includes(n as number)) });
    } catch { /* The default scope stays usable. */ }
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/quran/manifest.json", { signal: controller.signal }).then(r => r.ok ? r.json() : null)
      .then((manifest: Manifest | null) => { if (manifest) setNames(new Map(manifest.surahs.map(s => [s.id, s]))); }).catch(() => {});
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (phase !== "listening") return;
    const started = Date.now();
    setSeconds(0);
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250);
    return () => clearInterval(timer);
  }, [phase]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    if (!response) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    resultsSection.current?.scrollIntoView({ block: "start", behavior: motion });
  }, [response]);
  useEffect(() => () => { if (clipUrl) URL.revokeObjectURL(clipUrl); }, [clipUrl]);

  const updateScope = (next: Scope) => {
    setScope(next);
    try { localStorage.setItem(SCOPE_KEY, JSON.stringify(next)); } catch { /* Still applies in this tab. */ }
  };
  const scopeJuz = scope.only ? [...scope.juz].sort((a, b) => a - b) : [];
  const blocked = scope.only && !scope.juz.length;
  const busy = phase === "reading" || phase === "searching";
  const listening = phase === "listening";

  const search = useCallback(async (pcm: Float32Array, juz: number[]) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    clip.current = pcm;
    setClipUrl(wavUrl(pcm)); setPhase("searching"); setError(""); setShowMore(false);
    try {
      const body = await searchRecitation(pcm, juz, controller.signal);
      if (controller.signal.aborted) return;
      setResponse(body); setSearchedScope(juz.join(",")); setPhase("done");
    } catch (cause) {
      if (controller.signal.aborted) return;
      setResponse(null); setPhase("idle");
      setError(cause instanceof TypeError ? "Could not reach the listening model. Start it on port 8000 and try again." : (cause as Error).message);
    }
  }, []);

  const listen = async () => {
    if (recorder.recording) { recorder.stop(); return; }
    stopAudio(); request.current?.abort();
    setError(""); setTrimmed(false); setResponse(null); setPhase("listening");
    const pcm = await recorder.record();
    if (!pcm) { setPhase("idle"); return; }
    if (pcm.length < RATE * 0.6) { setError("That was too short. Recite at least a few words."); setPhase("idle"); return; }
    await search(pcm, scopeJuz);
  };
  const upload = async (file: File) => {
    stopAudio(); request.current?.abort();
    setError(""); setResponse(null); setPhase("reading");
    try {
      const decoded = await decodeClip(file);
      setTrimmed(decoded.trimmed);
      await search(decoded.pcm, scopeJuz);
    } catch (cause) { setError((cause as Error).message); setPhase("idle"); }
  };

  const [best, ...others] = response?.results || [];
  const heard = response?.heard.length || 0;
  const repeats = others.filter(r => r.score === best?.score && r.coverage === best.coverage).length;
  const status = listening ? c(`Listening… ${seconds}s. Tap again when you’re done.`, `نستمع… ${seconds.toLocaleString("ar")} ث. اضغط مرة أخرى عند الانتهاء.`)
    : phase === "reading" ? c("Reading your recording…", "جارٍ قراءة التسجيل…")
    : phase === "searching" ? c("Finding the ayah…", "نبحث عن الآية…")
    : phase === "done" ? c("Tap to search for another ayah.", "اضغط للبحث عن آية أخرى.")
    : c("Tap and recite, or play a recording near your microphone.", "اضغط واتلُ، أو شغّل تسجيلًا قرب الميكروفون.");
  const empty = response && !best ? {
    silent: c("We didn’t hear anything. Check your microphone, or that the recording has sound, and try again.", "لم نسمع شيئًا. تحقّق من الميكروفون أو من أن التسجيل فيه صوت، ثم حاول مجددًا."),
    opening_only: c("We heard the basmalah but nothing after it. Keep reciting into the ayah itself; a few words are enough.", "سمعنا البسملة فقط. تابع التلاوة في الآية نفسها؛ تكفي بضع كلمات."),
    none: response.juz.length
      ? c(`No match in juz ${juzRanges(response.juz, lang)}. Try the whole Quran, or recite a few more words.`, `لا تطابق في الجزء ${juzRanges(response.juz, lang)}. جرّب القرآن كله، أو اتلُ كلمات أكثر.`)
      : c("We couldn’t place these words. Recite a few more, a little closer to the microphone.", "لم نجد موضع هذه الكلمات. اتلُ كلمات أكثر، وأقرب قليلًا من الميكروفون."),
    found: "",
  }[response.verdict] : "";
  const shownError = error || recorder.error;

  return <div className="app-shell studio-shell" dir={lang === "ar" ? "rtl" : "ltr"}>
    <div className="workspace">
      <SiteHeader active="search" />
      <main className="find-page">
        <section className="find-hero">
          <h1>{c("Which ayah is this?", "ما هذه الآية؟")}</h1>
          <p>{c("Recite a few words, or let a recording play near your microphone. We’ll find where they are in the Quran.", "اتلُ بضع كلمات، أو شغّل تسجيلًا قرب الميكروفون، وسنجد موضعها في القرآن.")}</p>
        </section>

        {online === false && <div className="notice" role="status">{c("The listening model isn’t running, so search can’t hear you yet. Start it with", "نموذج الاستماع لا يعمل، لذا لا يمكن البحث الآن. شغّله بالأمر")} <code dir="ltr">python -m src.streaming.serve --model rattil-v3</code></div>}

        <section className="find-stage" aria-label={c("Listen", "الاستماع")}>
          <MobileAudioControl keepInline><button className={`find-star ${phase}`} style={{ "--level": listening ? recorder.level : 0 } as React.CSSProperties}
            disabled={busy || blocked} onClick={() => void listen()}
            aria-label={listening ? c("Stop and search", "توقّف وابحث") : c("Listen and find the ayah", "استمع وابحث عن الآية")}>
            <span className="find-star-ring" aria-hidden="true" /><span className="find-star-ring" aria-hidden="true" /><span className="find-star-ring" aria-hidden="true" />
            <span className="find-star-core" aria-hidden="true">{busy ? <Loader2 size={30} className="spin" /> : listening ? <Square size={24} fill="currentColor" /> : <Mic size={32} />}</span>
          </button></MobileAudioControl>
          <p className="find-status" aria-live="polite">{status}</p>
          <input ref={fileInput} type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" hidden aria-label={c("Choose a recording", "اختر تسجيلًا")}
            onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) void upload(file); }} />
          <button className="btn-quiet find-upload" disabled={busy || listening || blocked} onClick={() => fileInput.current?.click()}>
            <Upload size={15} />{c("Upload a recording", "ارفع تسجيلًا")}</button>
          <small className="find-hint">{c(`Up to ${MAX_CLIP_SECONDS} seconds are searched. Audio is processed on this project’s own server and not kept.`, `نبحث في أول ${MAX_CLIP_SECONDS.toLocaleString("ar")} ثانية. يُعالج الصوت على خادم المشروع ولا يُحفظ.`)}</small>
        </section>

        {shownError && <div className="error-message" role="alert">{c(shownError, "تعذّر البحث. تحقّق من الميكروفون أو الملف ومن تشغيل النموذج، ثم حاول مجددًا.")}</div>}

        <section className="find-scope" aria-labelledby="find-scope-title">
          <div className="find-scope-bar">
            <p id="find-scope-title">{c("Searching", "نبحث في")} <strong>{scope.only
              ? scope.juz.length ? c(`juz ${juzRanges(scope.juz, lang)}`, `الجزء ${juzRanges(scope.juz, lang)}`) : c("no juzʾ yet", "لا أجزاء بعد")
              : c("the whole Quran", "القرآن كله")}</strong></p>
            <label className="find-switch">
              <input type="checkbox" role="switch" checked={scope.only} disabled={listening || busy}
                onChange={e => updateScope({ only: e.target.checked, juz: e.target.checked && !scope.juz.length ? [30] : scope.juz })} />
              <span className="find-switch-track" aria-hidden="true" />{c("Only in juzʾ I choose", "في أجزاء أختارها فقط")}
            </label>
          </div>
          {scope.only && <div className="find-juz">
            <div className="find-juz-presets">
              {([[c("Juz ʿAmma", "جزء عمّ"), [30]], [c("Last three", "الثلاثة الأخيرة"), [28, 29, 30]], [c("Clear", "مسح"), []]] as const).map(([label, juz]) =>
                <button key={label} aria-pressed={juz.length ? scopeJuz.join() === juz.join() : undefined} disabled={listening || busy}
                  onClick={() => updateScope({ only: true, juz: [...juz] })}>{label}</button>)}
            </div>
            <div className="find-juz-grid" role="group" aria-label={c("Juzʾ to search", "الأجزاء التي نبحث فيها")}>
              {JUZ_NUMBERS.map(juz => {
                const [surah, ayah] = JUZ_STARTS[juz - 1], info = names.get(surah), on = scope.juz.includes(juz);
                const where = info ? `${lang === "ar" ? info.arabic : info.name}${ayah > 1 ? ` ${ayah.toLocaleString(lang)}` : ""}` : "";
                return <button key={juz} aria-pressed={on} className={on ? "on" : ""} disabled={listening || busy}
                  aria-label={c(`Juz ${juz}${where ? `, from ${where}` : ""}`, `الجزء ${juz.toLocaleString("ar")}${where ? `، من ${where}` : ""}`)}
                  onClick={() => updateScope({ only: true, juz: on ? scope.juz.filter(n => n !== juz) : [...scope.juz, juz].sort((a, b) => a - b) })}>
                  <strong>{juz.toLocaleString(lang)}</strong><small lang={lang === "ar" ? "ar" : undefined}>{where}</small></button>;
              })}
            </div>
            {blocked
              ? <p className="find-note" role="status">{c("Choose at least one juzʾ, or switch this off to search the whole Quran.", "اختر جزءًا واحدًا على الأقل، أو أوقف هذا الخيار للبحث في القرآن كله.")}</p>
              : <p className="find-note">{c("Recognition is most reliable in Al-Fātiḥah and Juz ʿAmma, where the model was trained.", "التعرف أدق في الفاتحة وجزء عمّ، حيث دُرّب النموذج.")}</p>}
          </div>}
        </section>

        {response && <section className="find-results" ref={resultsSection} aria-label={c("Search results", "نتائج البحث")}>
          {clipUrl && searchedScope !== scopeJuz.join(",") && !blocked && !busy && <div className="find-rescope">
            <p>{c("You changed where to search.", "غيّرت نطاق البحث.")}</p>
            <button className="btn-quiet" onClick={() => { if (clip.current) void search(clip.current, scopeJuz); }}><RotateCcw size={15} />{c("Search this clip again", "ابحث بالتسجيل نفسه مجددًا")}</button>
          </div>}
          {best ? <>
            <MatchCard result={best} heard={heard} primary names={names} reciter={reciter} />
            {repeats > 0 && <p className="find-note find-tip">{c(`These exact words appear in ${repeats + 1} places. The others are under “more matches”.`, `هذه الكلمات نفسها ترد في ${(repeats + 1).toLocaleString("ar")} مواضع، والبقية في «النتائج الأخرى».`)}</p>}
            {!repeats && (heard < 4 || best.confidence === "low") && <p className="find-note find-tip">{heard < 4
              ? c("Short clips can fit many places. Recite a few more words for a surer answer.", "المقاطع القصيرة قد تناسب مواضع كثيرة. اتلُ كلمات أكثر لنتيجة أدق.")
              : c("The words we heard don’t line up closely with any ayah. Try again a little closer to the microphone, or recite for longer.", "الكلمات التي سمعناها لا تطابق آية بدقة. حاول مجددًا أقرب من الميكروفون، أو اتلُ مدة أطول.")}</p>}
            {others.length > 0 && <button className="find-more" aria-expanded={showMore} aria-controls="find-others" onClick={() => setShowMore(v => !v)}>
              {showMore ? c("Hide other matches", "أخفِ النتائج الأخرى") : c(`Show ${others.length} more ${others.length === 1 ? "match" : "matches"}`, `أظهر النتائج الأخرى (${others.length.toLocaleString("ar")})`)}<ChevronDown size={16} /></button>}
            {showMore && <ol id="find-others" className="find-others">{others.map(result =>
              <li key={result.rank}><MatchCard result={result} heard={heard} primary={false} names={names} reciter={reciter} /></li>)}</ol>}
          </> : <p className="find-empty" role="status">{empty}</p>}
          <section className="find-heard">
            <h3>{c("What the model heard", "ما سمعه النموذج")}</h3>
            <p lang="ar" dir="rtl">{response.transcript || "—"}</p>
            {response.opening.length > 0 && <p className="find-note">{c(
              `We set aside the ${response.opening.map(o => o === "taawwudh" ? "taʿawwudh" : "basmalah").join(" and ")} at the start: they come before many passages, so they can’t tell us where you are.`,
              `تجاوزنا ${response.opening.map(o => o === "taawwudh" ? "الاستعاذة" : "البسملة").join(" و")} في البداية، فهي تسبق مواضع كثيرة ولا تدل على موضعك.`)}</p>}
            {trimmed && <p className="find-note">{c(`Only the first ${MAX_CLIP_SECONDS} seconds of your recording were searched.`, `بحثنا في أول ${MAX_CLIP_SECONDS.toLocaleString("ar")} ثانية من التسجيل فقط.`)}</p>}
            {clipUrl && <audio controls src={clipUrl} aria-label={c("Play the clip we searched", "شغّل المقطع الذي بحثنا به")} />}
            {best && <p className="find-note">{c("Latin lines are a draft Qālūn reading aid. Matching compares words, not pronunciation or tajweed.", "السطور اللاتينية نقل صوتي تجريبي لقالون. المطابقة تقارن الكلمات، لا النطق أو التجويد.")}</p>}
          </section>
        </section>}
      </main>
    </div>
  </div>;
}
