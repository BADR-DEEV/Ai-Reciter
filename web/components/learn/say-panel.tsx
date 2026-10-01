"use client";

import { useEffect, useRef, useState } from "react";
import { Headphones, Loader2, Mic, RotateCcw, Square, Volume2 } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { checkPractice, type PracticeResult } from "@/lib/learn/api";
import { LETTERS } from "@/lib/learn/letters";
import { Rich } from "./rich";
import { speak } from "@/lib/learn/speech";
import { useRecorder, wavUrl } from "@/lib/learn/use-recorder";

export type SayProps = {
  arabic: string; translit: string; mode: "sound" | "reading"; alternatives?: string[];
  listen?: () => void; online: boolean | null; compact?: boolean;
  onScore: (score: number | null) => void;
};

const describe = (practice: string) => {
  const l = LETTERS.find(x => x.practice === practice);
  return l ? `${practice} (${l.practiceTranslit}, ${l.name})` : practice;
};

/** Record the learner, ask the model, and explain what it heard. */
export function SayPanel({ arabic, translit, mode, alternatives = [], listen, online, compact, onScore }: SayProps) {
  const { t } = useLang();
  const recorder = useRecorder();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PracticeResult | null>(null);
  const [error, setError] = useState("");
  const [playback, setPlayback] = useState("");
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => { controller.current?.abort(); }, []);
  useEffect(() => () => { if (playback) URL.revokeObjectURL(playback); }, [playback]);

  const attempt = async () => {
    setResult(null); setError("");
    const pcm = await recorder.record();
    if (!pcm) return;
    setPlayback(wavUrl(pcm));
    setBusy(true);
    controller.current = new AbortController();
    try {
      const checked = await checkPractice(pcm, { mode, target: arabic, alternatives }, controller.current.signal);
      setResult(checked);
      if (checked.verdict !== "silent") onScore(checked.verdict === "correct" ? 1 : checked.verdict === "close" ? 0.5 : 0);
    } catch (cause) {
      if ((cause as Error).name !== "AbortError") setError(cause instanceof Error ? cause.message : "Could not check this recording.");
    } finally { setBusy(false); }
  };

  const verdict = result?.verdict;
  let feedback: React.ReactNode = null;
  if (result && "probabilities" in result) {
    const heardAlt = result.heard > 0 ? alternatives[result.heard - 1] : null;
    feedback = verdict === "correct" ? <p>We clearly heard <span lang="ar" className="ar-inline">{arabic}</span>.</p>
      : verdict === "close" ? <p>Closest to <span lang="ar" className="ar-inline">{arabic}</span>, but not clearly. Exaggerate the sound and try again.</p>
      : <p>That sounded closer to <Rich text={heardAlt ? describe(heardAlt) : "another sound"} />. Listen again and compare.</p>;
  } else if (result && "words" in result) {
    feedback = <>
      <p className="say-words" lang="ar" dir="rtl">{result.words.map(w => <span key={w.index} className={`say-word ${w.status}`}>{w.text}</span>)}</p>
      <p>{verdict === "correct" ? "Every word was recognised." : `${result.words.filter(w => w.status === "correct").length} of ${result.words.length} words recognised. Grey words weren’t heard clearly.`}</p>
      <p className="say-heard">Model heard: <span lang="ar" dir="rtl">{result.transcript || "nothing"}</span></p>
    </>;
  }

  return <div className={`say-panel ${compact ? "compact" : ""}`}>
    {!compact && <>
      <p className="say-arabic" lang="ar" dir="rtl">{arabic}</p>
      <p className="say-translit">{translit}</p>
    </>}
    <div className="say-controls">
      {listen && <button className="btn-round" onClick={listen} aria-label={t("listen")}><Volume2 size={20} /></button>}
      {recorder.recording
        ? <button className="btn-mic live" onClick={recorder.stop}><Square size={18} /> {t("stop")}<span className="mic-level" style={{ transform: `scaleX(${0.15 + recorder.level * 0.85})` }} /></button>
        : <button className="btn-mic" onClick={attempt} disabled={busy || online === false}>{busy ? <Loader2 className="spin" size={18} /> : result ? <RotateCcw size={18} /> : <Mic size={18} />} {result ? t("tryAgain") : t("yourTurn")}</button>}
      {playback && !recorder.recording && <button className="btn-round" onClick={() => void new Audio(playback).play()} aria-label={t("hearYourself")} title={t("hearYourself")}><Headphones size={19} /></button>}
    </div>
    {recorder.recording && <p className="say-status">{t("recording")}</p>}
    {online === false && <p className="say-offline">{t("modelOffline")}</p>}
    {(recorder.error || error) && <p className="say-error" role="alert">{recorder.error || error}</p>}
    {verdict === "silent" && <p className="say-error" role="status">{t("silent")}</p>}
    {feedback && <div className={`say-feedback ${verdict}`} role="status">
      <strong>{verdict === "correct" ? t("correct") : verdict === "close" ? t("close") : t("incorrect")}</strong>{feedback}
    </div>}
  </div>;
}

export const speakText = (text: string) => () => { speak(text); };
