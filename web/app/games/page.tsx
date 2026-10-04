"use client";
import { useEffect, useRef, useState } from "react";
import { BookOpen, Headphones, Mic, Puzzle, Sparkles, Volume2 } from "lucide-react";
import { SiteHeader } from "@/components/learn/site-header";
import { PhoneticAid } from "@/components/phonetic-aid";
import { TafsirPanel } from "@/components/tafsir-panel";
import { SayPanel } from "@/components/learn/say-panel";
import { modelOnline } from "@/lib/learn/api";
import { playUrl, stopAudio } from "@/lib/learn/speech";
import { useProgress } from "@/lib/learn/progress";
import type { Challenge, ChallengeMode, Difficulty } from "@/lib/challenges";
import { useLang } from "@/lib/i18n";
import { useReferenceReciter } from "@/lib/use-reference-reciter";
import { ReciterSelector } from "@/components/reciter-selector";
import { PROFILE_EVENT, progressKey } from "@/lib/local-profile";
import { emptySkill, recordChoice, nextSkill, type SkillHistory } from "@/lib/adaptive-challenges";

const MODES = [
  { id: "next", title: "What comes next?", note: "Recall the next ayah. Recite it for blind AI text feedback, or use choices.", icon: Mic },
  { id: "audio", title: "Listen & match", note: "One ayah, three Qālūn recordings. Find the matching recitation.", icon: Headphones },
  { id: "surah", title: "Find the surah", note: "Recognize where an ayah belongs. Ambiguous repeated ayahs are excluded.", icon: BookOpen },
  { id: "missing", title: "Restore the word", note: "Fill a missing word in the displayed ayah. The source is never changed.", icon: Puzzle },
  { id: "order", title: "Ayah sequence", note: "Choose the correct order of three consecutive ayahs.", icon: Sparkles },
] as const;

export default function GamesPage() {
  const { lang, c } = useLang();
  const { reciter, selectReciter } = useReferenceReciter();
  const titles: Record<ChallengeMode, string> = { next: "ما الآية التالية؟", audio: "استمع وطابق", surah: "حدّد السورة", missing: "أكمل الكلمة", order: "ترتيب الآيات" };
  const notes: Record<ChallengeMode, string> = { next: "تذكّر الآية التالية. اتلُها لملاحظات نصية أو استخدم الخيارات.", audio: "آية واحدة وثلاثة تسجيلات بقالون. اختر المطابق.", surah: "حدّد سورة الآية. نستبعد الآيات المكررة التي تحتمل أكثر من سورة.", missing: "أكمل كلمة ناقصة. لا يتغير نص المصدر.", order: "اختر الترتيب الصحيح لثلاث آيات متتابعة." };
  const [mode, setMode] = useState<ChallengeMode>("next"), [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [scope, setScope] = useState("amma"), [round, setRound] = useState(0);
  const [question, setQuestion] = useState<Challenge | null>(null), [picked, setPicked] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false), [choices, setChoices] = useState(false), [phonetics, setPhonetics] = useState(false);
  const [error, setError] = useState(""), [playing, setPlaying] = useState<number | null>(null), [online, setOnline] = useState<boolean | null>(null);
  const [correct, setCorrect] = useState(0), [attempts, setAttempts] = useState(0);
  const [outcome, setOutcome] = useState<"correct" | "incorrect" | "review">("review");
  const [adaptive, setAdaptive] = useState(false), [skill, setSkill] = useState<SkillHistory>(emptySkill);
  const answered = useRef("");
  const [profileRevision, setProfileRevision] = useState(0);
  useEffect(() => {
    const changed = () => { answered.current = ""; setProfileRevision(n => n + 1); };
    window.addEventListener(PROFILE_EVENT, changed);
    return () => window.removeEventListener(PROFILE_EVENT, changed);
  }, []);
  const skillID = `${mode}:${scope}:${mode === "audio" ? reciter : "text"}`;
  useEffect(() => {
    const refresh = () => {
      try { const saved = JSON.parse(localStorage.getItem(`${progressKey()}:adaptive-v1:${skillID}`) || "null");
        const valid = saved && Array.isArray(saved.outcomes) && saved.outcomes.every((v: unknown) => typeof v === "boolean") && ["easy", "medium", "hard"].includes(saved.difficulty) && Number.isFinite(saved.attempts) && saved.attempts >= 0;
        setSkill(valid ? { ...saved, outcomes: saved.outcomes.slice(-12) } : emptySkill());
        if (adaptive) setDifficulty(valid ? saved.difficulty : "easy");
      } catch { setSkill(emptySkill()); if (adaptive) setDifficulty("easy"); }
    };
    refresh(); window.addEventListener(PROFILE_EVENT, refresh);
    return () => window.removeEventListener(PROFILE_EVENT, refresh);
  }, [skillID, adaptive]);
  const { progress, complete } = useProgress();
  useEffect(() => { const c = new AbortController(); void modelOnline(c.signal).then(setOnline); return () => c.abort(); }, []);
  useEffect(() => {
    const c = new AbortController();
    answered.current = "";
    stopAudio(); setPlaying(null); setQuestion(null); setPicked(null); setRevealed(false); setOutcome("review"); setChoices(mode !== "next"); setError("");
    fetch(`/api/challenges?mode=${mode}&difficulty=${difficulty}&scope=${scope}&reciter=${reciter}&lang=${lang}`, { signal: c.signal })
      .then(async r => { const body = await r.json(); if (!r.ok) throw new Error(body.error); return body; })
      .then(setQuestion).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => { c.abort(); stopAudio(); };
  }, [mode, difficulty, scope, round, reciter, lang, profileRevision]);
  const award = (right: boolean, evidence: "choice" | "asr" = "choice") => {
    if (!question || revealed || answered.current === `${question.id}:${round}`) return;
    answered.current = `${question.id}:${round}`;
    setRevealed(true); setAttempts(n => n + 1);
    setOutcome(right ? "correct" : "incorrect");
    if (adaptive && evidence === "choice") {
      const next = recordChoice({ ...skill, difficulty }, right);
      setSkill(next);
      try { localStorage.setItem(`${progressKey()}:adaptive-v1:${skillID}`, JSON.stringify(next)); } catch { /* Optional local personalization. */ }
    }
    if (right) { setCorrect(n => n + 1); complete(`challenge:${question.id}:${difficulty}`, 1, difficulty === "hard" ? 30 : difficulty === "medium" ? 20 : 10); }
  };
  return <div className="learn-shell" dir={lang === "ar" ? "rtl" : "ltr"}><SiteHeader active="games" /><main className="challenge-main">
    <section className="challenge-hero"><div><p className="hero-kicker">{c("REMEMBER · LISTEN · READ", "تذكّر · استمع · اقرأ")}</p><h1>{c("A little challenge. A stronger connection.", "تحدٍّ صغير، وصلة أقوى بالقرآن.")}</h1><p>{c("Practice Quran reading and recall, not general Arabic. No rush. Return to a teacher for pronunciation.", "تدرّب على قراءة القرآن وتذكّره، لا على العربية العامة. دون استعجال؛ واستعن بمعلم للنطق.")}</p></div><div className="challenge-score"><strong>{progress.xp} {c("XP", "نقطة")}</strong><span>{correct} / {attempts} {c("this session", "في هذه الجلسة")}</span><a href="/profile">{c("Save under your local profile →", "احفظ التقدّم في ملفك المحلي ←")}</a></div></section>
    <div className="challenge-modes">{MODES.map(item => <button key={item.id} aria-pressed={mode === item.id} className={`challenge-mode ${mode === item.id ? "selected" : ""}`} onClick={() => setMode(item.id)}><item.icon size={23} /><strong>{c(item.title, titles[item.id])}</strong><span>{c(item.note, notes[item.id])}</span></button>)}</div>
    <section className="challenge-settings"><label>{c("Difficulty", "الصعوبة")}<select aria-label={c("Challenge difficulty", "صعوبة التحدي")} disabled={adaptive} value={difficulty} onChange={e => setDifficulty(e.target.value as Difficulty)}><option value="easy">{c("Gentle · distinct choices", "سهل · خيارات متباينة")}</option><option value="medium">{c("Growing · closer choices", "متوسط · خيارات أقرب")}</option><option value="hard">{c("Focused · similar choices", "صعب · خيارات متشابهة")}</option></select></label>
      <label><input type="checkbox" checked={adaptive} onChange={e => setAdaptive(e.target.checked)} />{c("Personalized difficulty · prototype", "صعوبة شخصية · نموذج أولي")}</label>
      <label>{c("Reading scope", "نطاق القراءة")}<select aria-label={c("Challenge scope", "نطاق التحدي")} value={scope} onChange={e => setScope(e.target.value)}><option value="amma">{c("Fātiḥah + Juz ʿAmma", "الفاتحة وجزء عمّ")}</option><option value="all">{c("Whole Quran · ASR experimental", "القرآن كاملًا · التعرف الصوتي تجريبي")}</option></select></label>
      <ReciterSelector value={reciter} onChange={selectReciter} />
      <label><input type="checkbox" checked={phonetics} onChange={e => setPhonetics(e.target.checked)} />{c("Draft phonetic aid", "نقل صوتي تجريبي")}</label></section>
    {adaptive && <p className="safety-note">{c("Difficulty changes on the next question from your recent choice answers—not XP or ASR confidence. Closer alternatives follow steady success; mistakes bring gentler practice. Local, explainable policy; not a trained learner model.", "تتغير الصعوبة في السؤال التالي وفق إجابات الخيارات الأخيرة، لا النقاط ولا ثقة التعرف الصوتي. تقارب البدائل بعد النجاح المستمر، وتيسيرها بعد الأخطاء. سياسة محلية قابلة للتفسير وليست نموذج تعلم مدرّبًا.")}</p>}
    <p className="safety-note">{c("Reference audio uses your selected Qālūn reader, default Al-Huthaify, currently Fātiḥah/Juz ʿAmma. No altered Quran audio or silent voice substitution. AI text feedback and Latin aids do not certify tajweed.", "الاستماع بصوت قارئ قالون الذي تختاره، والافتراضي الحذيفي؛ المتاح حاليًا الفاتحة وجزء عمّ. لا نغيّر الصوت القرآني ولا نستبدل القارئ بصمت. الملاحظات النصية والنقل الصوتي لا يثبتان صحة التجويد.")}</p>
    {error ? <section className="challenge-card"><p role="alert">{c(error, "تعذّر تجهيز التحدي أو تشغيل المقطع. اختر نوعًا آخر أو حاول مجددًا.")}</p><button className="btn-primary" onClick={() => setRound(n => n + 1)}>{c("Try again", "حاول مجددًا")}</button></section> : !question ? <p role="status">{c("Preparing a Qālūn challenge…", "جارٍ تجهيز تحدٍّ بقالون…")}</p> : <section className="challenge-card">
      <div className="challenge-question-head"><span className="hero-kicker">{c(MODES.find(m => m.id === mode)!.title, titles[mode])}</span><span>{question.reference}</span></div>
      <p className={mode === "order" ? "" : "challenge-ayah"} lang={mode === "order" ? lang : "ar"} dir={mode === "order" && lang === "en" ? "ltr" : "rtl"}>{question.prompt}</p>
      {phonetics && mode !== "order" && mode !== "missing" && <PhoneticAid text={question.prompt} />}
      {mode === "next" && !revealed && <div><h2>{c("Recite the next ayah from memory", "اتلُ الآية التالية من الذاكرة")}</h2><p>{c("Record up to 12 seconds. Use choices or the studio for longer ayahs. No answer text is sent to the decoder.", "التسجيل حتى ١٢ ثانية. للآيات الأطول استخدم الخيارات أو الاستوديو. لا نرسل نص الإجابة إلى مفكك الصوت.")}</p>
        <SayPanel key={`${question.id}:${round}`} compact arabic={question.target} translit="" mode="reading" online={online} onScore={score => { if (score === 1) award(true, "asr"); }} />
        {!choices && <button className="btn-quiet" onClick={() => setChoices(true)}>{c("Use answer choices instead", "استخدم خيارات الإجابة بدلًا من ذلك")}</button>}</div>}
      {choices && <div className="challenge-options" role="radiogroup" aria-label={c("Answer choices", "خيارات الإجابة")}>{question.options.map((option, index) => <div key={index} className={`challenge-option ${revealed && index === question.answer ? "right" : revealed && picked === index ? "wrong" : picked === index ? "picked" : ""}`}>
        {option.audio && <button className="btn-listen" aria-label={c(`Play audio ${index + 1}`, `شغّل المقطع ${index + 1}`)} onClick={async () => {
          setPlaying(index); setError("");
          const result = await playUrl(option.audio!);
          setPlaying(current => current === index ? null : current);
          if (result === "failed") setError("Reference audio could not load. Try another question; no recording was substituted.");
        }}><Volume2 size={16} />{playing === index ? c("Playing…", "قيد التشغيل…") : c(`Audio ${index + 1}`, `المقطع ${index + 1}`)}</button>}
        <button role="radio" aria-checked={picked === index} disabled={revealed} onClick={() => setPicked(index)} lang={mode === "surah" || mode === "audio" ? lang : "ar"} dir={(mode === "surah" || mode === "audio") && lang === "en" ? "ltr" : "rtl"}>{option.audio ? c(`Choose audio ${index + 1}`, `اختر المقطع ${index + 1}`) : option.label}</button>
      </div>)}</div>}
      <div className="challenge-actions">{!revealed && choices && <button className="btn-primary" disabled={picked === null} onClick={() => { stopAudio(); award(picked === question.answer); }}>{c("Check answer", "تحقّق من الإجابة")}</button>}
        {!revealed && <button className="btn-quiet" onClick={() => { setRevealed(true); stopAudio(); }}>{c("Show answer · no XP", "أظهر الإجابة · دون نقاط")}</button>}
        <button className="btn-quiet" onClick={() => {
          if (adaptive && revealed && outcome !== "review") {
            const next = nextSkill(skill); setSkill(next); setDifficulty(next.difficulty);
            try { localStorage.setItem(`${progressKey()}:adaptive-v1:${skillID}`, JSON.stringify(next)); } catch { /* Optional storage. */ }
          }
          setRound(n => n + 1);
        }}>{revealed ? c("Next challenge", "التحدي التالي") : c("Skip", "تخطَّ")}</button></div>
      {revealed && <div className="challenge-feedback" role="status"><h2>{outcome === "correct" ? c("Well remembered!", "أحسنت التذكّر!") : outcome === "incorrect" ? c("A chance to learn", "فرصة للتعلّم") : c("Take a moment to review", "راجع على مهل")}</h2><p>{question.explanation}</p><p className="challenge-ayah" lang="ar" dir="rtl">{question.target}</p>
        {phonetics && <PhoneticAid text={question.target} />}<a className="btn-quiet" href={`/studio?surah=${question.surah}`}>{c("Practice this surah in the studio", "تدرّب على هذه السورة في الاستوديو")}</a><TafsirPanel surah={question.surah} ayah={question.ayah} /></div>}
      <small className="similarity-note">{c("Distractor method:", "طريقة اختيار البدائل:")} {question.similarity}. {c("XP is practice progress, not religious competency.", "النقاط للتقدّم التدريبي، وليست حكمًا على الإتقان الديني.")}</small>
    </section>}
  </main></div>;
}
