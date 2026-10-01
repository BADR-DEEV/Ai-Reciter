import { DAMMA, FATHA, KASRA, LETTERS, VOWELS, forms, letter, syllable, type Letter } from "./letters";
import { SURAHS } from "./surahs";

// A lesson is a list of steps. Teaching steps come first; practice steps that
// are answered wrongly are repeated at the end of the lesson.
export type Choice = { label: string; arabic?: boolean };
export type Step =
  | { kind: "intro"; title: string; body: string[]; arabic?: string; caption?: string }
  | { kind: "letter"; char: string }
  | { kind: "forms"; char: string }
  | { kind: "pair"; a: string; b: string; note: string }
  | { kind: "vowel"; mark: string; name: string; sound: string; hint: string; examples: string[] }
  | { kind: "choice"; question: string; prompt?: string; promptArabic?: boolean; listen?: string; options: Choice[]; answer: number; explain?: string }
  | { kind: "say"; arabic: string; translit: string; mode: "sound" | "reading"; alternatives?: string[]; listen?: string; tip?: string }
  | { kind: "ayah"; surah: number; ayah: number }
  | { kind: "studio"; surah: number };

export type Lesson = { id: string; title: string; titleAr: string; summary: string; steps: Step[] };
export type Unit = { id: string; title: string; titleAr: string; summary: string; lessons: Lesson[] };

// Deterministic shuffling keeps lessons identical between visits and renders.
function seeded(seed: string) {
  let h = 2166136261;
  for (const c of seed) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  return () => ((h = Math.imul(h ^ (h >>> 15), 2246822507) ^ Math.imul(h ^ (h >>> 13), 3266489909)) >>> 0) / 4294967296;
}
function shuffle<T>(items: T[], seed: string) {
  const rand = seeded(seed), out = [...items];
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
}
function choice(question: string, correct: Choice, wrong: Choice[], seed: string, extra: Partial<Extract<Step, { kind: "choice" }>> = {}): Step {
  const options = shuffle([correct, ...wrong.slice(0, 3)], seed);
  return { kind: "choice", question, options, answer: options.indexOf(correct), ...extra };
}
const others = (l: Letter, pool: Letter[], seed: string) =>
  shuffle([...l.confusables.map(letter), ...shuffle(pool, seed)].filter((x, i, all) => x !== l && all.indexOf(x) === i), seed).slice(0, 3);

const speakable = (l: Letter) => l.char !== "ا" && l.char !== "ء";

function alphabetLesson(id: string, title: string, titleAr: string, chars: string[]): Lesson {
  const group = chars.map(letter);
  const steps: Step[] = group.flatMap(l => [{ kind: "letter", char: l.char } as Step]);
  group.forEach((l, i) => {
    steps.push(choice("Which letter makes this sound?", { label: l.char, arabic: true }, others(l, group, id + i).map(x => ({ label: x.char, arabic: true })),
      `${id}-hear-${i}`, { listen: l.practice, prompt: l.practiceTranslit, explain: `${l.char} is ${l.name}: ${l.sound}` }));
    steps.push(choice("What sound does this letter make?", { label: l.translit }, others(l, LETTERS, id + "s" + i).map(x => ({ label: x.translit })),
      `${id}-see-${i}`, { prompt: l.char, promptArabic: true, explain: `${l.char} is ${l.name}.` }));
  });
  group.filter(speakable).forEach(l => steps.push({ kind: "say", arabic: l.practice, translit: l.practiceTranslit, mode: "sound", alternatives: l.confusables.map(c => letter(c).practice), listen: l.practice, tip: l.tip }));
  return { id, title, titleAr, summary: chars.join(" "), steps };
}

function pairLesson(id: string, title: string, titleAr: string, summary: string, pairs: [string, string, string][]): Lesson {
  const steps: Step[] = [];
  pairs.forEach(([a, b, note]) => steps.push({ kind: "pair", a, b, note }));
  pairs.forEach(([a, b], i) => {
    const [x, y] = [letter(a), letter(b)];
    for (const [target, other] of shuffle([[x, y], [y, x]], `${id}${i}`))
      steps.push(choice("Which one did you hear?", { label: target.char, arabic: true }, [{ label: other.char, arabic: true }], `${id}-${i}-${target.char}`,
        { listen: target.practice, explain: `That was ${target.name} (${target.practiceTranslit}).` }));
  });
  pairs.forEach(([a, b]) => {
    for (const [target, other] of [[letter(a), letter(b)], [letter(b), letter(a)]])
      steps.push({ kind: "say", arabic: target.practice, translit: target.practiceTranslit, mode: "sound", alternatives: [other.practice], listen: target.practice, tip: target.tip });
  });
  return { id, title, titleAr, summary, steps };
}

function formsLesson(): Lesson {
  const show = ["ب", "ج", "ع", "ه", "د"];
  const steps: Step[] = [
    { kind: "intro", title: "Letters join up", body: ["In Arabic, letters in a word are joined like cursive handwriting. Most letters change shape a little depending on where they sit: at the start, middle or end of a word.", "The good news: the core shape and the dots never change. Look for the dots first."], arabic: "بَ ← بـ ← ـبـ ← ـب", caption: "Bā’ alone, at the start, in the middle, at the end" },
    ...show.map(char => ({ kind: "forms", char }) as Step),
    { kind: "intro", title: "The six that never join forward", body: ["Six letters connect to the letter before them, but never to the letter after: ا د ذ ر ز و.", "After one of these, the next letter starts fresh, which is why you see small gaps inside words."], arabic: "ا د ذ ر ز و", caption: "Alif, Dāl, Dhāl, Rā’, Zāy, Wāw" },
  ];
  const pool = LETTERS.filter(l => l.joinsAfter);
  shuffle(pool, "forms").slice(0, 8).forEach((l, i) => {
    const f = forms(l), position = (["initial", "medial", "final"] as const)[i % 3];
    steps.push(choice(`Which letter is this, written at the ${position === "initial" ? "start" : position === "medial" ? "middle" : "end"} of a word?`,
      { label: l.char, arabic: true }, others(l, pool, `f${i}`).map(x => ({ label: x.char, arabic: true })), `forms-${i}`,
      { prompt: f[position], promptArabic: true, explain: `That is ${l.name}. Its dots and core shape stay the same.` }));
  });
  steps.push(choice("Which of these letters never joins to the letter after it?", { label: "د", arabic: true }, ["ب", "س", "م"].map(label => ({ label, arabic: true })), "nonjoin"));
  return { id: "joining", title: "How letters join", titleAr: "اتصال الحروف", summary: "Start, middle, end shapes", steps };
}

const SYLLABLE_LETTERS = ["ب", "ت", "ك", "ل", "م", "ن", "س", "ق", "ح", "ع", "د", "ر"].map(letter);

function vowelsLesson(): Lesson {
  const steps: Step[] = [
    { kind: "intro", title: "Vowels are small marks", body: ["Arabic letters are mostly consonants. Short vowels are written as small marks above or below the letter.", "There are three. Once you know them, you can sound out any syllable."] },
    ...VOWELS.map(v => ({ kind: "vowel", mark: v.mark, name: v.name, sound: v.sound, hint: v.hint, examples: ["ب", "ت", "م"].map(c => c + v.mark) }) as Step),
  ];
  SYLLABLE_LETTERS.slice(0, 9).forEach((l, i) => {
    const v = VOWELS[i % 3], correct = syllable(l, v.sound);
    const wrong = [...VOWELS.filter(x => x !== v).map(x => syllable(l, x.sound)), syllable(others(l, SYLLABLE_LETTERS, `v${i}`)[0], v.sound)];
    steps.push(choice("How do you read this?", { label: correct }, wrong.map(label => ({ label })), `vow-${i}`, { prompt: l.char + v.mark, promptArabic: true, listen: l.char + v.mark }));
  });
  return { id: "short-vowels", title: "Short vowels", titleAr: "الحركات", summary: "Fatḥah, kasrah, ḍammah", steps };
}

// Real words, mostly from the short surahs the course ends with.
const WORDS: { arabic: string; translit: string; wrong: string[] }[] = [
  { arabic: "قُلْ", translit: "qul", wrong: ["kul", "qil", "qal"] },
  { arabic: "لَمْ", translit: "lam", wrong: ["lim", "nam", "lum"] },
  { arabic: "مِنْ", translit: "min", wrong: ["man", "mun", "nim"] },
  { arabic: "هُوَ", translit: "huwa", wrong: ["ḥuwa", "hiya", "hawa"] },
  { arabic: "خَلَقَ", translit: "khalaqa", wrong: ["ḥalaqa", "khalaka", "khuliqa"] },
  { arabic: "حَسَدَ", translit: "ḥasada", wrong: ["hasada", "ḥaṣada", "ḥasuda"] },
  { arabic: "يَلِدْ", translit: "yalid", wrong: ["yalad", "yulid", "walid"] },
  { arabic: "أَحَدٌ", translit: "aḥadun", wrong: ["ahadun", "aḥadan", "akhadun"] },
];

function sukunLesson(): Lesson {
  const steps: Step[] = [
    { kind: "intro", title: "Sukūn: no vowel", body: ["A small circle (ْ) means the letter has no vowel. It closes the syllable.", "So مِنْ is “min”: m-i-n, with the n stopped cleanly."], arabic: "مِنْ  ·  قُلْ  ·  لَمْ", caption: "min · qul · lam" },
    { kind: "intro", title: "Qalqalah: the little bounce", body: ["Five letters (ق ط ب ج د) are strong stops. When they have a sukūn, let them echo with a small bounce instead of swallowing them.", "Think of the tiny release at the end of “lid” when you say it carefully: yalid → yalid’."], arabic: "قُطْبُ جَدٍّ", caption: "The five qalqalah letters, as a phrase to remember them" },
  ];
  WORDS.slice(0, 6).forEach((w, i) => steps.push(choice("How do you read this word?", { label: w.translit }, w.wrong.map(label => ({ label })), `suk-${i}`, { prompt: w.arabic, promptArabic: true, listen: w.arabic })));
  WORDS.slice(0, 4).forEach(w => steps.push({ kind: "say", arabic: w.arabic, translit: w.translit, mode: "reading", listen: w.arabic }));
  return { id: "sukun", title: "Sukūn & your first words", titleAr: "السكون", summary: "Closed syllables, qalqalah", steps };
}

function longVowelsLesson(): Lesson {
  const steps: Step[] = [
    { kind: "intro", title: "Long vowels: hold for two", body: ["Three letters stretch a vowel: ا after fatḥah gives “ā”, ي after kasrah gives “ī”, و after ḍammah gives “ū”.", "Hold a long vowel for two counts, about the time of two short vowels. This stretching is called madd (مَدّ) and it is one of the first rules of beautiful recitation."], arabic: "بَا  ·  بِي  ·  بُو", caption: "bā · bī · bū" },
  ];
  ["ب", "ق", "ن", "س"].map(letter).forEach((l, i) => {
    const long = [l.char + FATHA + "ا", l.char + KASRA + "ي", l.char + DAMMA + "و"];
    const read = [syllable(l, "ā"), syllable(l, "ī"), syllable(l, "ū")];
    const k = i % 3;
    steps.push(choice("Short or long?", { label: read[k] }, [syllable(l, ["a", "i", "u"][k]), ...read.filter((_, j) => j !== k)].map(label => ({ label })), `long-${i}`,
      { prompt: long[k], promptArabic: true, listen: long[k] }));
  });
  steps.push({ kind: "say", arabic: "بَا", translit: "bā", mode: "sound", alternatives: ["بِي", "بُو"], listen: "بَا", tip: "Open the mouth and hold “aa” for two counts." });
  steps.push({ kind: "say", arabic: "بُو", translit: "bū", mode: "sound", alternatives: ["بَا", "بِي"], listen: "بُو", tip: "Round the lips and hold “oo”." });
  steps.push({ kind: "say", arabic: "نِي", translit: "nī", mode: "sound", alternatives: ["نَا", "نُو"], listen: "نِي", tip: "Smile slightly and hold “ee”." });
  return { id: "long-vowels", title: "Long vowels (madd)", titleAr: "المد", summary: "ā · ī · ū, held for two counts", steps };
}

function shaddaLesson(): Lesson {
  const steps: Step[] = [
    { kind: "intro", title: "Shaddah: say it twice", body: ["The small “w” shape (ّ) doubles a letter: hold or press it twice as long.", "رَبِّ is “rab-bi”, not “rabi”. Doubling changes the meaning of words, so it matters."], arabic: "رَبِّ  ·  إِنَّ  ·  شَرِّ", caption: "rabbi · inna · sharri" },
    { kind: "intro", title: "Ghunnah: the nasal hum", body: ["When ن or م carries a shaddah, hum it through your nose for two counts. This is ghunnah (غُنَّة).", "Try إِنَّ: in-nnna, with a warm hum on the n."], arabic: "إِنَّ  ·  ثُمَّ", caption: "inna · thumma" },
    { kind: "intro", title: "Tanwīn: a hidden “n”", body: ["A doubled vowel mark at the end of a word adds an “n” sound: ـً “an”, ـٍ “in”, ـٌ “un”.", "أَحَدٌ is “aḥadun”. When you stop at the end of an ayah, drop it: “aḥad”."], arabic: "بًا  ·  بٍ  ·  بٌ", caption: "ban · bin · bun" },
  ];
  const items: [string, string, string[]][] = [
    ["رَبِّ", "rabbi", ["rabi", "rubbi", "rabba"]],
    ["إِنَّ", "inna", ["ina", "anna", "inni"]],
    ["شَرِّ", "sharri", ["shari", "sirri", "sharra"]],
    ["ثُمَّ", "thumma", ["thuma", "tumma", "thamma"]],
    ["بٌ", "bun", ["bu", "ban", "bin"]],
    ["بًا", "ban", ["bā", "bun", "bin"]],
  ];
  items.forEach(([arabic, correct, wrong], i) => steps.push(choice("How do you read this?", { label: correct }, wrong.map(label => ({ label })), `sh-${i}`, { prompt: arabic, promptArabic: true, listen: arabic })));
  steps.push({ kind: "say", arabic: "رَبِّ", translit: "rabbi", mode: "reading", listen: "رَبِّ" });
  steps.push({ kind: "say", arabic: "شَرِّ", translit: "sharri", mode: "reading", listen: "شَرِّ" });
  return { id: "shadda", title: "Shaddah, ghunnah & tanwīn", titleAr: "الشدة والتنوين", summary: "Doubled letters and hidden n", steps };
}

function allahLesson(): Lesson {
  const steps: Step[] = [
    { kind: "intro", title: "The word Allāh", body: ["اللَّه (Allāh) is the Arabic word for God, used by Arabic-speaking Muslims and Christians alike.", "Its lām is heavy and full (like the “l” in “full”) after an “a” or “u”, and light after an “i”: qul huwa llāh (heavy) but bismi llāh (light)."], arabic: "قُلْ هُوَ اَ۬للَّهُ", caption: "qul huwa llāhu" },
    { kind: "intro", title: "“The”: al- and the sun letters", body: ["ال means “the”. Before half of the letters (the moon letters) you say the l: al-qamar.", "Before the other half (the sun letters: ت ث د ذ ر ز س ش ص ض ط ظ ل ن) the l disappears and the next letter doubles: ash-shams, an-nās."], arabic: "اَ۬لْقَمَرُ  ·  اَ۬لشَّمْسُ", caption: "al-qamaru · ash-shamsu" },
    { kind: "intro", title: "Joining across words", body: ["When a word starts with ال after another word, the “a” of al- is dropped and the words flow together: rabbi + al-ʿālamīn → rabbi l-ʿālamīn.", "In the Qālūn mushaf a small mark over the alif reminds you of this silent connecting alif."] },
  ];
  const items: [string, string, string[]][] = [
    ["اَ۬لنَّاسِ", "an-nāsi", ["al-nāsi", "an-nasi", "al-nāsu"]],
    ["اَ۬لْفَلَقِ", "al-falaqi", ["af-falaqi", "al-falaki", "al-fulqi"]],
    ["اَ۬لصَّمَدُ", "aṣ-ṣamadu", ["al-ṣamadu", "as-samadu", "aṣ-ṣumudu"]],
    ["اَ۬لْكَوْثَرَ", "al-kawthara", ["ak-kawthara", "al-kawsara", "al-qawthara"]],
  ];
  items.forEach(([arabic, correct, wrong], i) => steps.push(choice("How do you read this?", { label: correct }, wrong.map(label => ({ label })), `al-${i}`, { prompt: arabic, promptArabic: true })));
  steps.push(choice("Which of these is a sun letter (the l of al- disappears)?", { label: "ن", arabic: true }, ["ق", "ب", "ع"].map(label => ({ label, arabic: true })), "sun"));
  steps.push({ kind: "say", arabic: "اَ۬لنَّاسِ", translit: "an-nāsi", mode: "reading" });
  return { id: "allah-al", title: "Allāh and “al-”", titleAr: "لفظ الجلالة وأل", summary: "Sun & moon letters", steps };
}

function surahLessons(): Lesson[] {
  return SURAHS.map(surah => ({
    id: `surah-${surah.id}`,
    title: surah.name,
    titleAr: surah.arabic,
    summary: `${surah.meaning} · ${surah.ayahs.length} ayahs`,
    steps: [
      { kind: "intro", title: `${surah.name}: ${surah.meaning}`, body: [surah.about, "For each ayah: listen to Sheikh Al-Husary, follow the transliteration, then read it aloud yourself. Don’t worry about speed. Clear letters matter more than pace."], arabic: surah.arabic },
      ...surah.ayahs.map(a => ({ kind: "ayah", surah: surah.id, ayah: a.ayah }) as Step),
      { kind: "studio", surah: surah.id },
    ],
  }));
}

export const UNITS: Unit[] = [
  {
    id: "start", title: "Start here", titleAr: "ابدأ هنا", summary: "What the Quran is and how this course works",
    lessons: [{
      id: "welcome", title: "Welcome", titleAr: "مرحبًا", summary: "No Arabic needed",
      steps: [
        { kind: "intro", title: "You don’t need to know any Arabic", body: ["The Quran is the holy book of Islam, revealed in Arabic more than 1,400 years ago. It is meant to be read aloud, and its recitation follows careful rules for how each letter should sound.", "This course teaches you to read the Arabic script and pronounce it well enough to recite. You won’t memorise vocabulary. You will learn to read sounds, the way a child learns phonics."] },
        { kind: "intro", title: "Arabic reads right to left", body: ["Start on the right side of the line and move left. There are 28 letters, and most of them are consonants.", "Short vowels are tiny marks above and below letters. The Quran is always printed with these marks, which makes it easier to read than everyday Arabic."], arabic: "بِسْمِ اِ۬للَّهِ", caption: "bismi llāh: “In the name of God” ← read this way" },
        { kind: "intro", title: "How the listening works", body: ["Some exercises ask you to read aloud. A speech model trained on Quran recitation listens and tells you which sound it heard. It runs on this project’s own server, not a commercial cloud service, and recordings are not stored.", "It is a practice partner, not a teacher. It can tell ح from ه, but it cannot certify your tajweed. When you are ready, read to a qualified teacher."] },
        { kind: "intro", title: "One way of reciting: Qālūn", body: ["The Quran has been passed down in several authentic recitation traditions (riwāyāt). They differ in small details of pronunciation.", "This app follows Qālūn ʿan Nāfiʿ, widely recited in Libya and parts of North and West Africa. Everything you learn about letters and vowels applies to every tradition."] },
        { kind: "choice", question: "Where do you start reading an Arabic line?", options: [{ label: "On the right" }, { label: "On the left" }], answer: 0, explain: "Arabic runs right to left." },
        { kind: "choice", question: "What are the small marks above and below Arabic letters?", options: [{ label: "Decorations" }, { label: "Short vowels" }, { label: "Punctuation" }], answer: 1, explain: "They are the short vowels: a, i and u." },
      ],
    }],
  },
  {
    id: "letters", title: "The letters", titleAr: "الحروف", summary: "All 28 letters, their sounds and where they are made",
    lessons: [
      alphabetLesson("letters-1", "Alif and the dotted family", "ا ب ت ث", ["ا", "ب", "ت", "ث"]),
      alphabetLesson("letters-2", "The bowl family", "ج ح خ", ["ج", "ح", "خ"]),
      alphabetLesson("letters-3", "Small letters that don’t join", "د ذ ر ز", ["د", "ذ", "ر", "ز"]),
      alphabetLesson("letters-4", "The teeth letters", "س ش ص ض", ["س", "ش", "ص", "ض"]),
      alphabetLesson("letters-5", "Tall letters and the eye", "ط ظ ع غ", ["ط", "ظ", "ع", "غ"]),
      alphabetLesson("letters-6", "Loops and hooks", "ف ق ك ل", ["ف", "ق", "ك", "ل"]),
      alphabetLesson("letters-7", "The last letters", "م ن ه و ي ء", ["م", "ن", "ه", "و", "ي", "ء"]),
    ],
  },
  {
    id: "sounds", title: "Sounds English doesn’t have", titleAr: "مخارج الحروف", summary: "Train your ear and mouth on the tricky pairs",
    lessons: [
      pairLesson("pairs-throat", "Throat letters", "حروف الحلق", "ح/ه · ع/ء · خ/غ", [
        ["ح", "ه", "ح is tight and strong from the middle of the throat; ه is soft and airy from the bottom. Try fogging a mirror (ح) versus sighing (ه)."],
        ["ع", "ء", "ء is a clean stop, like “uh-oh”. ع is a voiced squeeze that you can hold."],
        ["خ", "غ", "Same place at the back of the mouth. خ is whispered, غ is voiced like a gargle."],
      ]),
      pairLesson("pairs-heavy", "Heavy and light letters", "التفخيم والترقيق", "ص/س · ط/ت · ق/ك · ض/د", [
        ["ص", "س", "Heavy letters fill the mouth: raise the back of the tongue. The vowel after a heavy letter sounds deeper."],
        ["ط", "ت", "ط is a heavy, full t; ت is light and crisp."],
        ["ق", "ك", "ق comes from much further back, near where you swallow."],
        ["ض", "د", "ض uses the side of the tongue against the molars and is heavy; د is light at the front teeth."],
      ]),
      pairLesson("pairs-teeth", "Tongue between the teeth", "الحروف اللثوية", "ث/س · ذ/ز · ظ/ذ", [
        ["ث", "س", "ث puts the tongue tip out between the teeth (think); س keeps it behind them (sea)."],
        ["ذ", "ز", "ذ is “th” in “this”; ز is “z” in “zoo”."],
        ["ظ", "ذ", "Same tongue position. ظ is heavy, ذ is light."],
      ]),
    ],
  },
  {
    id: "reading", title: "Reading syllables", titleAr: "القراءة", summary: "Joining letters, vowels and your first words",
    lessons: [formsLesson(), vowelsLesson(), sukunLesson(), longVowelsLesson(), shaddaLesson(), allahLesson()],
  },
  {
    id: "surahs", title: "Your first surahs", titleAr: "سورك الأولى", summary: "Read real ayahs, then recite in the studio",
    lessons: surahLessons(),
  },
];

export const LESSONS = UNITS.flatMap(u => u.lessons);
export const findLesson = (id: string) => LESSONS.find(l => l.id === id);
export const nextLesson = (id: string) => LESSONS[LESSONS.findIndex(l => l.id === id) + 1];
