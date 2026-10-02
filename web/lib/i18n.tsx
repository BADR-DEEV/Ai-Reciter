"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

// English is the default: the learn track is built for people who can't read
// Arabic yet. Arabic switches the interface chrome; lesson teaching text stays
// English for now.
export type Lang = "en" | "ar";

const STRINGS = {
  en: {
    brandTagline: "Learn to recite the Quran, starting from the alphabet",
    heroEyebrow: "No Arabic needed",
    heroTitle: "Read the Quran aloud, sound by sound.",
    heroBody: "Whether you are curious about Islam, have just embraced it, or always wanted to read the Quran properly, start from the very first letter. Short lessons, real recitation, and a speech model that listens and helps you hear the difference.",
    startLearning: "Start learning",
    continueLearning: "Continue learning",
    iCanRead: "I can already read: open the studio",
    feature1Title: "Sounds, not vocabulary",
    feature1Body: "Learn the 28 letters, the vowel marks and the joining rules: everything you need to read aloud, nothing you need to memorise.",
    feature2Title: "A partner that listens",
    feature2Body: "Say a sound and the model tells you whether it heard a strong ḥ or a soft h, a deep q or a light k. It never stores your voice.",
    feature3Title: "Real recitation",
    feature3Body: "Follow Sheikh Al-Husary ayah by ayah, then recite whole surahs in the studio with word-by-word feedback.",
    pathTitle: "Your path",
    privacy: "Practice runs on this project’s own server. Recordings are processed and discarded.",
    learn: "Learn",
    studio: "Recitation studio",
    home: "Home",
    course: "Course",
    xp: "XP",
    streak: "day streak",
    lessons: "lessons",
    lesson: "lesson",
    lessonsDone: "completed",
    start: "Start",
    review: "Review",
    next: "Up next",
    continue: "Continue",
    check: "Check",
    skip: "Skip",
    tryAgain: "Try again",
    listen: "Listen",
    yourTurn: "Your turn: say it",
    recording: "Listening… stop when you’re done",
    stop: "Stop",
    hearYourself: "Hear yourself",
    correct: "Well done!",
    close: "Almost there",
    incorrect: "Not quite",
    silent: "We didn’t hear anything. Check your microphone and try again.",
    modelOffline: "The listening model isn’t running, so speaking exercises can be skipped. Listening and reading exercises still work.",
    lessonComplete: "Lesson complete",
    accuracy: "Accuracy",
    backToCourse: "Back to course",
    nextLesson: "Next lesson",
    exit: "Exit lesson",
    language: "العربية",
    resetProgress: "Reset progress",
    openStudio: "Recite it in the studio",
    meaning: "Meaning",
  },
  ar: {
    brandTagline: "تعلّم تلاوة القرآن بدءًا من الحروف",
    heroEyebrow: "لا تحتاج إلى معرفة العربية",
    heroTitle: "اقرأ القرآن بصوتك، حرفًا حرفًا.",
    heroBody: "سواء كنت مهتمًا بالإسلام، أو دخلت فيه حديثًا، أو أردت دائمًا قراءة القرآن قراءةً صحيحة، ابدأ من الحرف الأول. دروس قصيرة، وتلاوة حقيقية، ونموذج صوتي يستمع إليك ويساعدك على تمييز الأصوات.",
    startLearning: "ابدأ التعلّم",
    continueLearning: "تابع التعلّم",
    iCanRead: "أعرف القراءة: افتح الاستوديو",
    feature1Title: "أصوات لا مفردات",
    feature1Body: "تعلّم الحروف الثمانية والعشرين والحركات وقواعد الوصل: كل ما تحتاجه للقراءة، دون حفظ معانٍ.",
    feature2Title: "رفيق يستمع إليك",
    feature2Body: "انطق الصوت وسيخبرك النموذج هل سمع ح أم ه، ق أم ك. لا يُحفظ صوتك.",
    feature3Title: "تلاوة حقيقية",
    feature3Body: "تابع الشيخ الحصري آيةً آية، ثم اتلُ السورة كاملة في الاستوديو مع ملاحظات كلمةً كلمة.",
    pathTitle: "مسارك",
    privacy: "يعمل التدريب على خادم المشروع نفسه. تُعالَج التسجيلات ثم تُحذف.",
    learn: "تعلّم",
    studio: "استوديو التلاوة",
    home: "الرئيسية",
    course: "الدورة",
    xp: "نقطة",
    streak: "أيام متتالية",
    lessons: "دروس",
    lesson: "درس",
    lessonsDone: "مكتملة",
    start: "ابدأ",
    review: "راجع",
    next: "التالي",
    continue: "متابعة",
    check: "تحقّق",
    skip: "تخطَّ",
    tryAgain: "حاول مجددًا",
    listen: "استمع",
    yourTurn: "دورك: انطقها",
    recording: "نستمع إليك… توقّف عند الانتهاء",
    stop: "إيقاف",
    hearYourself: "استمع لنفسك",
    correct: "أحسنت!",
    close: "قريب جدًا",
    incorrect: "ليس تمامًا",
    silent: "لم نسمع شيئًا. تحقّق من الميكروفون وحاول مجددًا.",
    modelOffline: "نموذج الاستماع غير مشغّل، لذا يمكنك تخطّي تمارين النطق. تمارين الاستماع والقراءة تعمل.",
    lessonComplete: "اكتمل الدرس",
    accuracy: "الدقة",
    backToCourse: "العودة إلى الدورة",
    nextLesson: "الدرس التالي",
    exit: "الخروج من الدرس",
    language: "English",
    resetProgress: "إعادة ضبط التقدّم",
    openStudio: "اتلُها في الاستوديو",
    meaning: "المعنى",
  },
} as const;

export type StringKey = keyof typeof STRINGS.en;
const KEY = "rattil.lang";
const LangContext = createContext<{ lang: Lang; t: (key: StringKey) => string; toggle: () => void }>({
  lang: "en", t: key => STRINGS.en[key], toggle: () => undefined,
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");
  useEffect(() => { try { if (localStorage.getItem(KEY) === "ar") setLang("ar"); } catch { /* Default English. */ } }, []);
  const toggle = useCallback(() => setLang(current => {
    const next = current === "en" ? "ar" : "en";
    try { localStorage.setItem(KEY, next); } catch { /* Not persisted. */ }
    return next;
  }), []);
  const t = useCallback((key: StringKey) => STRINGS[lang][key], [lang]);
  return <LangContext.Provider value={{ lang, t, toggle }}>{children}</LangContext.Provider>;
}

export const useLang = () => useContext(LangContext);
