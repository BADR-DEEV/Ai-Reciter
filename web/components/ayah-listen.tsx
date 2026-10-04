"use client";
import { useEffect, useRef, useState } from "react";
import { Loader2, Mic2, Square } from "lucide-react";
import { playUrl, stopAudio } from "@/lib/learn/speech";
import { referenceURL, RECITERS, type ReciterID } from "@/lib/reciters";
import { useLang } from "@/lib/i18n";
export function AyahListen({ surah, ayah, reciter, disabled = false }: { surah: number; ayah: number; reciter: ReciterID; disabled?: boolean }) {
  const { lang, c } = useLang();
  const [state, setState] = useState<"idle" | "loading" | "playing" | "failed">("idle");
  const version = useRef(0);
  const playing = useRef(false);
  useEffect(() => { setState("idle"); return () => { version.current++; if (playing.current) stopAudio(); playing.current = false; }; }, [surah, ayah, reciter, disabled]);
  const reader = RECITERS.find(r => r.id === reciter)!;
  const name = lang === "ar" ? reader.arabic : reader.name;
  return <div className="ayah-listen"><button className="btn-listen" disabled={disabled} aria-label={c(`Listen to ayah ${ayah}`, `استمع إلى الآية ${ayah}`)} onClick={async () => {
    const token = ++version.current;
    if (state === "loading" || state === "playing") { stopAudio(); playing.current = false; setState("idle"); return; }
    setState("loading"); playing.current = true;
    const result = await playUrl(referenceURL(surah, ayah, reciter), 0, undefined, () => { if (version.current === token) setState("playing"); });
    if (version.current === token) { playing.current = false; setState(result === "failed" ? "failed" : "idle"); }
  }}>{state === "loading" ? <Loader2 size={16} className="spin" /> : state === "playing" ? <Square size={15} /> : <Mic2 size={17} />}
    {state === "loading" ? c("Loading…", "جارٍ التحميل…") : state === "playing" ? c("Stop", "إيقاف") : c(`Listen · ${name}`, `استمع · ${name}`)}</button>
    {state === "failed" && <small role="alert">{c("This reader’s aligned ayah clip is unavailable. Choose another reader; no audio was substituted.", "لا يتوفر مقطع مواءَم لهذه الآية بصوت القارئ. اختر قارئًا آخر؛ لم نستبدل الصوت.")}</small>}
  </div>;
}
