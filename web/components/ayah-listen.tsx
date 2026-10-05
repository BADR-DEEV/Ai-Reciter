"use client";
import { useEffect, useRef, useState } from "react";
import { Loader2, Mic2, Square } from "lucide-react";
import { playUrl, stopAudio } from "@/lib/learn/speech";
import { referenceURL, RECITERS, type ReciterID } from "@/lib/reciters";
import { useLang } from "@/lib/i18n";
import { ReaderAudioNotice } from "@/components/reader-audio-notice";
import { validTimings, wordAtTime, type PlaybackCursor, type PlaybackTiming } from "@/lib/playback";
export function AyahListen({ surah, ayah, reciter, disabled = false, displayText, onPlayback }: { surah: number; ayah: number; reciter: ReciterID; disabled?: boolean; displayText?: string; onPlayback?: (ayah: number, cursor: PlaybackCursor | null) => void }) {
  const { lang, c } = useLang();
  const [state, setState] = useState<"idle" | "loading" | "playing" | "failed">("idle");
  const version = useRef(0);
  const playing = useRef(false);
  const callback = useRef(onPlayback);
  callback.current = onPlayback;
  const [tracking, setTracking] = useState<"word" | "ayah">("ayah");
  useEffect(() => { setState("idle"); return () => { version.current++; if (playing.current) { stopAudio(); callback.current?.(ayah, null); } playing.current = false; }; }, [surah, ayah, reciter, disabled]);
  const reader = RECITERS.find(r => r.id === reciter)!;
  const name = lang === "ar" ? reader.arabic : reader.name;
  return <div className="ayah-listen"><button className="btn-listen" disabled={disabled} aria-label={c(`Listen to ayah ${ayah}`, `استمع إلى الآية ${ayah}`)} onClick={async () => {
    const token = ++version.current;
    if (state === "loading" || state === "playing") { stopAudio(); playing.current = false; callback.current?.(ayah, null); setState("idle"); return; }
    setState("loading"); playing.current = true;
    let timing: PlaybackTiming | null = null;
    setTracking("ayah");
    if (displayText) void fetch(`/api/reference-timing?surah=${surah}&ayah=${ayah}&reciter=${reciter}`, { signal: AbortSignal.timeout(10000) })
      .then(async response => response.ok ? await response.json() : null)
      .then(data => { if (data && version.current === token && validTimings(data, displayText)) timing = data; }).catch(() => {});
    const result = await playUrl(referenceURL(surah, ayah, reciter), 0, undefined,
      () => { if (version.current === token) { setState("playing"); callback.current?.(ayah, { ayah, word: null, tracking: "ayah" }); } },
      (seconds, duration) => { if (version.current !== token) return;
        const valid = timing && Math.abs(timing.duration - duration) < .05;
        setTracking(valid ? "word" : "ayah");
        callback.current?.(ayah, { ayah, word: valid ? wordAtTime(timing!.words, seconds) : null, tracking: valid ? "word" : "ayah" });
      });
    if (version.current === token) { playing.current = false; callback.current?.(ayah, null); setState(result === "failed" ? "failed" : "idle"); }
  }}>{state === "loading" ? <Loader2 size={16} className="spin" /> : state === "playing" ? <Square size={15} /> : <Mic2 size={17} />}
    {state === "loading" ? c("Loading…", "جارٍ التحميل…") : state === "playing" ? c("Stop", "إيقاف") : c(`Listen · ${name}`, `استمع · ${name}`)}</button>
    {state === "failed" && <ReaderAudioNotice reciter={reciter} />}
    {state === "playing" && <small className="playback-tracking" role="status">{tracking === "word" ? c("Word tracking · machine timing draft", "متابعة الكلمات · توقيت آلي تجريبي") : c("Ayah tracking · word timings unavailable", "متابعة الآية · توقيت الكلمات غير متاح")}</small>}
  </div>;
}
