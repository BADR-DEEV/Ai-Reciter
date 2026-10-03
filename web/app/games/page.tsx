"use client";
import { useEffect, useState } from "react";
import { BookOpen, Headphones, Mic, Puzzle, Sparkles, Volume2 } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { PhoneticAid } from "@/components/phonetic-aid";
import { TafsirPanel } from "@/components/tafsir-panel";
import { SayPanel } from "@/components/learn/say-panel";
import { modelOnline } from "@/lib/learn/api";
import { playUrl, stopAudio } from "@/lib/learn/speech";
import { useProgress } from "@/lib/learn/progress";
import type { Challenge, ChallengeMode, Difficulty } from "@/lib/challenges";

const MODES = [
  { id: "next", title: "What comes next?", note: "Recall the next ayah. Recite it for blind AI text feedback, or use choices.", icon: Mic },
  { id: "audio", title: "Listen & match", note: "One ayah, three Qālūn recordings. Find the matching recitation.", icon: Headphones },
  { id: "surah", title: "Find the surah", note: "Recognize where an ayah belongs. Ambiguous repeated ayahs are excluded.", icon: BookOpen },
  { id: "missing", title: "Restore the word", note: "Fill a missing word in the displayed ayah. The source is never changed.", icon: Puzzle },
  { id: "order", title: "Ayah sequence", note: "Choose the correct order of three consecutive ayahs.", icon: Sparkles },
] as const;

export default function GamesPage() {
  const [mode, setMode] = useState<ChallengeMode>("next"), [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [scope, setScope] = useState("amma"), [round, setRound] = useState(0);
  const [question, setQuestion] = useState<Challenge | null>(null), [picked, setPicked] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false), [choices, setChoices] = useState(false), [phonetics, setPhonetics] = useState(false);
  const [error, setError] = useState(""), [playing, setPlaying] = useState<number | null>(null), [online, setOnline] = useState<boolean | null>(null);
  const [correct, setCorrect] = useState(0), [attempts, setAttempts] = useState(0);
  const [outcome, setOutcome] = useState<"correct" | "incorrect" | "review">("review");
  const { progress, complete } = useProgress();
  useEffect(() => { const c = new AbortController(); void modelOnline(c.signal).then(setOnline); return () => c.abort(); }, []);
  useEffect(() => {
    const c = new AbortController();
    stopAudio(); setPlaying(null); setQuestion(null); setPicked(null); setRevealed(false); setOutcome("review"); setChoices(mode !== "next"); setError("");
    fetch(`/api/challenges?mode=${mode}&difficulty=${difficulty}&scope=${scope}`, { signal: c.signal })
      .then(async r => { const body = await r.json(); if (!r.ok) throw new Error(body.error); return body; })
      .then(setQuestion).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => { c.abort(); stopAudio(); };
  }, [mode, difficulty, scope, round]);
  const award = (right: boolean) => {
    if (!question || revealed) return;
    setRevealed(true); setAttempts(n => n + 1);
    setOutcome(right ? "correct" : "incorrect");
    if (right) { setCorrect(n => n + 1); complete(`challenge:${question.id}:${difficulty}`, 1, difficulty === "hard" ? 30 : difficulty === "medium" ? 20 : 10); }
  };
  return <div className="learn-shell"><SiteHeader active="games" /><main className="challenge-main">
    <section className="challenge-hero"><div><p className="hero-kicker">REMEMBER · LISTEN · READ</p><h1>A little challenge. A stronger connection.</h1><p>Practice Quran reading and recall, not general Arabic. No lives lost, no rush. Return to a teacher for pronunciation.</p></div><div className="challenge-score"><strong>{progress.xp} XP</strong><span>{correct} / {attempts} this session</span><a href="/profile">Save under your local profile →</a></div></section>
    <div className="challenge-modes">{MODES.map(item => <button key={item.id} aria-pressed={mode === item.id} className={`challenge-mode ${mode === item.id ? "selected" : ""}`} onClick={() => setMode(item.id)}><item.icon size={23} /><strong>{item.title}</strong><span>{item.note}</span></button>)}</div>
    <section className="challenge-settings"><label>Difficulty<select aria-label="Challenge difficulty" value={difficulty} onChange={e => setDifficulty(e.target.value as Difficulty)}><option value="easy">Gentle · distinct choices</option><option value="medium">Growing · closer choices</option><option value="hard">Focused · similar choices</option></select></label>
      <label>Reading scope<select aria-label="Challenge scope" value={scope} onChange={e => setScope(e.target.value)}><option value="amma">Fātiḥah + Juz ʿAmma</option><option value="all">Whole Quran · text available, ASR experimental</option></select></label>
      <label><input type="checkbox" checked={phonetics} onChange={e => setPhonetics(e.target.checked)} /> Draft phonetic aid</label></section>
    <p className="safety-note">Audio questions use local Al-Husary Qālūn clips, currently Fātiḥah/Juz ʿAmma. Harder choices use similarity, not altered Quran audio. AI feedback compares recognized words; it does not certify tajweed. Latin aids still need qualified review.</p>
    {error ? <section className="challenge-card"><p role="alert">{error}</p><button className="btn-primary" onClick={() => setRound(n => n + 1)}>Try again</button></section> : !question ? <p role="status">Preparing a Qālūn challenge…</p> : <section className="challenge-card">
      <div className="challenge-question-head"><span className="hero-kicker">{MODES.find(m => m.id === mode)?.title}</span><span>{question.reference}</span></div>
      <p className={mode === "order" ? "" : "challenge-ayah"} lang={mode === "order" ? "en" : "ar"} dir={mode === "order" ? "ltr" : "rtl"}>{question.prompt}</p>
      {phonetics && mode !== "order" && mode !== "missing" && <PhoneticAid text={question.prompt} />}
      {mode === "next" && !revealed && <div><h2>Recite the next ayah from memory</h2><p>Short recording, up to 12 seconds. If the ayah needs longer, use answer choices or the studio. No answer text is sent to the decoder.</p>
        <SayPanel key={`${question.id}:${round}`} compact arabic={question.target} translit="" mode="reading" online={online} onScore={score => { if (score === 1) award(true); }} />
        {!choices && <button className="btn-quiet" onClick={() => setChoices(true)}>Use answer choices instead</button>}</div>}
      {choices && <div className="challenge-options" role="radiogroup" aria-label="Answer choices">{question.options.map((option, index) => <div key={index} className={`challenge-option ${revealed && index === question.answer ? "right" : revealed && picked === index ? "wrong" : picked === index ? "picked" : ""}`}>
        {option.audio && <button className="btn-listen" aria-label={`Play audio ${index + 1}`} onClick={async () => {
          setPlaying(index); setError("");
          const result = await playUrl(option.audio!);
          setPlaying(current => current === index ? null : current);
          if (result === "failed") setError("Reference audio could not load. Try another question; no recording was substituted.");
        }}><Volume2 size={16} />{playing === index ? "Playing…" : `Audio ${index + 1}`}</button>}
        <button role="radio" aria-checked={picked === index} disabled={revealed} onClick={() => setPicked(index)} lang={mode === "surah" || mode === "audio" ? "en" : "ar"} dir={mode === "surah" || mode === "audio" ? "ltr" : "rtl"}>{option.audio ? `Choose audio ${index + 1}` : option.label}</button>
      </div>)}</div>}
      <div className="challenge-actions">{!revealed && choices && <button className="btn-primary" disabled={picked === null} onClick={() => { stopAudio(); award(picked === question.answer); }}>Check answer</button>}
        {!revealed && <button className="btn-quiet" onClick={() => { setRevealed(true); stopAudio(); }}>Show answer · no XP</button>}
        <button className="btn-quiet" onClick={() => setRound(n => n + 1)}>{revealed ? "Next challenge" : "Skip"}</button></div>
      {revealed && <div className="challenge-feedback" role="status"><h2>{outcome === "correct" ? "Well remembered!" : outcome === "incorrect" ? "A chance to learn" : "Take a moment to review"}</h2><p>{question.explanation}</p><p className="challenge-ayah" lang="ar" dir="rtl">{question.target}</p>
        {phonetics && <PhoneticAid text={question.target} />}<a className="btn-quiet" href={`/studio?surah=${question.surah}`}>Practice this surah in the studio</a><TafsirPanel surah={question.surah} ayah={question.ayah} /></div>}
      <small className="similarity-note">Distractor method: {question.similarity}. XP is practice progress, not a religious competency score.</small>
    </section>}
  </main></div>;
}
