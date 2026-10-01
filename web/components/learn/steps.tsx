"use client";

import { useEffect, useState } from "react";
import { Mic2, Pause, Volume2 } from "lucide-react";
import type { Step } from "@/lib/learn/curriculum";
import { PLACE_LABEL, forms, letter } from "@/lib/learn/letters";
import { SURAHS } from "@/lib/learn/surahs";
import { playClip, speak, stopClip } from "@/lib/learn/speech";
import { useLang } from "@/lib/i18n";
import { Rich } from "./rich";
import { SayPanel, speakText } from "./say-panel";

const Listen = ({ text, label }: { text: string; label?: string }) =>
  <button className="btn-listen" onClick={() => speak(text)}><Volume2 size={17} /> {label || "Listen"}</button>;

export function IntroStep({ step }: { step: Extract<Step, { kind: "intro" }> }) {
  return <div className="step-intro">
    <h2>{step.title}</h2>
    {step.arabic && <figure><p className="intro-arabic" lang="ar" dir="rtl">{step.arabic}</p>{step.caption && <figcaption>{step.caption}</figcaption>}</figure>}
    {step.body.map((p, i) => <p key={i}><Rich text={p} /></p>)}
  </div>;
}

export function LetterStep({ step }: { step: Extract<Step, { kind: "letter" }> }) {
  const l = letter(step.char);
  const f = forms(l);
  return <div className="step-letter">
    <div className="letter-hero">
      <span className="letter-glyph" lang="ar">{l.char}</span>
      <div>
        <h2>{l.name} <span lang="ar">{l.arabicName}</span></h2>
        <p className="letter-sound">{l.sound}</p>
        <div className="letter-tags">
          <span>{PLACE_LABEL[l.place]}</span>
          {l.heavy && <span className="tag-heavy">Heavy letter</span>}
          {l.noEnglish && <span className="tag-rare">No English equivalent</span>}
        </div>
        <Listen text={l.practice} label={`Listen: ${l.practiceTranslit}`} />
      </div>
    </div>
    <p className="letter-tip"><Rich text={l.tip} /></p>
    <div className="letter-forms" dir="rtl">
      {(["isolated", "initial", "medial", "final"] as const).map(k => <div key={k}><span lang="ar">{f[k]}</span><small dir="ltr">{k}</small></div>)}
    </div>
  </div>;
}

export function FormsStep({ step }: { step: Extract<Step, { kind: "forms" }> }) {
  const l = letter(step.char);
  const f = forms(l);
  return <div className="step-forms">
    <h2>{l.name} in a word</h2>
    <div className="forms-row" dir="rtl">
      {(["initial", "medial", "final", "isolated"] as const).map(k => <div key={k}><span lang="ar">{f[k]}</span><small dir="ltr">{k === "initial" ? "Start" : k === "medial" ? "Middle" : k === "final" ? "End" : "Alone"}</small></div>)}
    </div>
    <p>{l.joinsAfter ? "The dots and the core shape stay the same; only the connecting strokes change." : `${l.name} never joins to the letter after it, so it only has two shapes.`}</p>
  </div>;
}

export function PairStep({ step }: { step: Extract<Step, { kind: "pair" }> }) {
  const [a, b] = [letter(step.a), letter(step.b)];
  return <div className="step-pair">
    <h2>{a.name} or {b.name}?</h2>
    <div className="pair-row">
      {[a, b].map(l => <button key={l.char} className="pair-card" onClick={() => speak(l.practice)}>
        <span lang="ar">{l.char}</span><strong>{l.practiceTranslit}</strong><small>{PLACE_LABEL[l.place]}{l.heavy ? " · heavy" : ""}</small><Volume2 size={16} />
      </button>)}
    </div>
    <p><Rich text={step.note} /></p>
  </div>;
}

export function VowelStep({ step }: { step: Extract<Step, { kind: "vowel" }> }) {
  return <div className="step-vowel">
    <h2>{step.name}: “{step.sound}”</h2>
    <p className="vowel-mark" lang="ar">ـ{step.mark}</p>
    <p>{step.hint}</p>
    <div className="pair-row">
      {step.examples.map(e => <button key={e} className="pair-card" onClick={() => speak(e)}><span lang="ar">{e}</span><Volume2 size={16} /></button>)}
    </div>
  </div>;
}

export function ChoiceStep({ step, picked, revealed, onPick }: { step: Extract<Step, { kind: "choice" }>; picked: number | null; revealed: boolean; onPick: (i: number) => void }) {
  useEffect(() => { if (step.listen) speak(step.listen); }, [step]);
  return <div className="step-choice">
    <h2>{step.question}</h2>
    {(step.prompt || step.listen) && <div className="choice-prompt">
      {step.listen && <button className="btn-round big" onClick={() => speak(step.listen!)} aria-label="Play the sound again"><Volume2 size={26} /></button>}
      {step.prompt && step.promptArabic && <span className="choice-arabic" lang="ar" dir="rtl">{step.prompt}</span>}
    </div>}
    <div className={`choice-options ${step.options.every(o => o.arabic) ? "arabic" : ""}`} role="radiogroup">
      {step.options.map((o, i) => {
        const state = revealed ? (i === step.answer ? "right" : i === picked ? "wrong" : "") : i === picked ? "picked" : "";
        return <button key={i} role="radio" aria-checked={picked === i} disabled={revealed} className={`choice ${state}`} onClick={() => onPick(i)}>
          <span lang={o.arabic ? "ar" : undefined}>{o.label}</span>
        </button>;
      })}
    </div>
  </div>;
}

export function AyahStep({ step, online, onScore }: { step: Extract<Step, { kind: "ayah" }>; online: boolean | null; onScore: (s: number | null) => void }) {
  const { t } = useLang();
  const surah = SURAHS.find(s => s.id === step.surah)!;
  const ayah = surah.ayahs.find(a => a.ayah === step.ayah)!;
  const [playing, setPlaying] = useState(false);
  useEffect(() => () => stopClip(), []);
  const play = () => {
    if (playing) { stopClip(); setPlaying(false); return; }
    setPlaying(true);
    playClip(surah.audio, ayah.start, ayah.end, () => setPlaying(false));
  };
  return <div className="step-ayah">
    <p className="ayah-ref">{surah.name} · Ayah {ayah.ayah}</p>
    <p className="ayah-text" lang="ar" dir="rtl">{ayah.text}</p>
    <p className="ayah-translit">{ayah.translit}</p>
    <p className="ayah-meaning"><span>{t("meaning")}:</span> {ayah.meaning}</p>
    <button className="btn-listen" onClick={play}>{playing ? <Pause size={17} /> : <Mic2 size={17} />} {playing ? "Pause" : "Listen to Al-Husary"}</button>
    <SayPanel compact arabic={ayah.text} translit={ayah.translit} mode="reading" online={online} onScore={onScore} />
  </div>;
}

export function StudioStep({ step }: { step: Extract<Step, { kind: "studio" }> }) {
  const { t } = useLang();
  const surah = SURAHS.find(s => s.id === step.surah)!;
  return <div className="step-intro">
    <h2>Now recite the whole surah</h2>
    <p className="intro-arabic" lang="ar">{surah.arabic}</p>
    <p>In the recitation studio, recite {surah.name} from start to finish. Each word turns green as the model recognises it, and skipped words are marked so you know where to look again.</p>
    <a className="btn-primary" href={`/studio?surah=${surah.id}`}>{t("openStudio")}</a>
  </div>;
}

export function SayStep({ step, online, onScore }: { step: Extract<Step, { kind: "say" }>; online: boolean | null; onScore: (s: number | null) => void }) {
  return <div className="step-say">
    <h2>Say it out loud</h2>
    <SayPanel arabic={step.arabic} translit={step.translit} mode={step.mode} alternatives={step.alternatives}
      listen={step.listen ? speakText(step.listen) : undefined} online={online} onScore={onScore} />
    {step.tip && <p className="say-tip"><Rich text={step.tip} /></p>}
  </div>;
}
