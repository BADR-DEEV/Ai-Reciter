// The Arabic letters taught for Quran reading. `practice` is the syllable the
// learner says aloud (the letter held with a long "aa"), which Whisper can
// transcribe far more reliably than a bare consonant. `confusables` are the
// letters a beginner most often reads or pronounces instead: same dot family
// or a nearby articulation point. They become the forced-choice alternatives.
export type Letter = {
  char: string;
  name: string;
  arabicName: string;
  translit: string;
  sound: string;
  tip: string;
  place: "throat" | "tongue-back" | "tongue-middle" | "tongue-tip" | "teeth" | "lips" | "long vowel";
  heavy?: boolean;
  noEnglish?: boolean;
  joinsAfter: boolean;
  practice: string;
  practiceTranslit: string;
  confusables: string[];
};

export const LETTERS: Letter[] = [
  { char: "ا", name: "Alif", arabicName: "أَلِف", translit: "ā", sound: "A long “aa”, as in “father”.", tip: "Alif has no consonant sound of its own. It stretches the vowel before it, or carries a hamzah.", place: "long vowel", joinsAfter: false, practice: "آ", practiceTranslit: "ʾā", confusables: [] },
  { char: "ب", name: "Bā’", arabicName: "بَاء", translit: "b", sound: "Like “b” in “boy”.", tip: "Press both lips together. One dot below.", place: "lips", joinsAfter: true, practice: "بَا", practiceTranslit: "bā", confusables: ["ت", "ث"] },
  { char: "ت", name: "Tā’", arabicName: "تَاء", translit: "t", sound: "Like “t” in “tea”, but softer.", tip: "The tongue tip touches the back of the upper front teeth, not the ridge behind them. Two dots above.", place: "tongue-tip", joinsAfter: true, practice: "تَا", practiceTranslit: "tā", confusables: ["ب", "ث", "ط"] },
  { char: "ث", name: "Thā’", arabicName: "ثَاء", translit: "th", sound: "Like “th” in “think”.", tip: "Let the tongue tip peek out between your teeth and blow gently. Three dots above.", place: "teeth", joinsAfter: true, practice: "ثَا", practiceTranslit: "thā", confusables: ["ت", "س", "ذ"] },
  { char: "ج", name: "Jīm", arabicName: "جِيم", translit: "j", sound: "Like “j” in “jam”.", tip: "The middle of the tongue presses the roof of the mouth. One dot inside the bowl.", place: "tongue-middle", joinsAfter: true, practice: "جَا", practiceTranslit: "jā", confusables: ["ح", "خ"] },
  { char: "ح", name: "Ḥā’", arabicName: "حَاء", translit: "ḥ", sound: "A strong, breathy “h” from the middle of the throat.", tip: "Like breathing hard to fog a mirror, but tighter in the throat. No voice, no scrape. No dot.", place: "throat", noEnglish: true, joinsAfter: true, practice: "حَا", practiceTranslit: "ḥā", confusables: ["ه", "خ", "ج"] },
  { char: "خ", name: "Khā’", arabicName: "خَاء", translit: "kh", sound: "Like “ch” in Scottish “loch” or German “Bach”.", tip: "A gentle scrape where the back of the tongue nears the soft palate. One dot above.", place: "throat", heavy: true, noEnglish: true, joinsAfter: true, practice: "خَا", practiceTranslit: "khā", confusables: ["ح", "غ"] },
  { char: "د", name: "Dāl", arabicName: "دَال", translit: "d", sound: "Like “d” in “door”, but softer.", tip: "Tongue tip at the back of the upper front teeth. Does not join to the next letter.", place: "tongue-tip", joinsAfter: false, practice: "دَا", practiceTranslit: "dā", confusables: ["ذ", "ض"] },
  { char: "ذ", name: "Dhāl", arabicName: "ذَال", translit: "dh", sound: "Like “th” in “this”.", tip: "The tongue tip between the teeth, with voice. One dot above.", place: "teeth", joinsAfter: false, practice: "ذَا", practiceTranslit: "dhā", confusables: ["د", "ز", "ظ"] },
  { char: "ر", name: "Rā’", arabicName: "رَاء", translit: "r", sound: "A single tapped “r”, like Spanish “pero”.", tip: "Flick the tongue tip once against the ridge behind your teeth. Do not roll it on and on.", place: "tongue-tip", joinsAfter: false, practice: "رَا", practiceTranslit: "rā", confusables: ["ز", "ل"] },
  { char: "ز", name: "Zāy", arabicName: "زَاي", translit: "z", sound: "Like “z” in “zoo”.", tip: "A buzzing “s”. One dot above.", place: "tongue-tip", joinsAfter: false, practice: "زَا", practiceTranslit: "zā", confusables: ["ر", "ذ", "س"] },
  { char: "س", name: "Sīn", arabicName: "سِين", translit: "s", sound: "Like “s” in “sea”.", tip: "A light, thin whistle. No dots.", place: "tongue-tip", joinsAfter: true, practice: "سَا", practiceTranslit: "sā", confusables: ["ش", "ص", "ث"] },
  { char: "ش", name: "Shīn", arabicName: "شِين", translit: "sh", sound: "Like “sh” in “ship”.", tip: "Let the air spread across the middle of the tongue. Three dots above.", place: "tongue-middle", joinsAfter: true, practice: "شَا", practiceTranslit: "shā", confusables: ["س"] },
  { char: "ص", name: "Ṣād", arabicName: "صَاد", translit: "ṣ", sound: "A heavy, deep “s”.", tip: "Say س but raise the back of the tongue and round the mouth slightly. The vowel after it sounds like “o” in “sod”, not “a” in “sad”.", place: "tongue-tip", heavy: true, noEnglish: true, joinsAfter: true, practice: "صَا", practiceTranslit: "ṣā", confusables: ["س", "ض"] },
  { char: "ض", name: "Ḍād", arabicName: "ضَاد", translit: "ḍ", sound: "A heavy “d” made with the side of the tongue.", tip: "Press the side of the tongue against the upper molars and raise the back of the tongue. Arabic is called “the language of ḍād” because this sound is so rare.", place: "tongue-back", heavy: true, noEnglish: true, joinsAfter: true, practice: "ضَا", practiceTranslit: "ḍā", confusables: ["د", "ظ"] },
  { char: "ط", name: "Ṭā’", arabicName: "طَاء", translit: "ṭ", sound: "A heavy, deep “t”.", tip: "Say ت with the back of the tongue raised and a full mouth. No puff of air.", place: "tongue-tip", heavy: true, noEnglish: true, joinsAfter: true, practice: "طَا", practiceTranslit: "ṭā", confusables: ["ت", "ظ"] },
  { char: "ظ", name: "Ẓā’", arabicName: "ظَاء", translit: "ẓ", sound: "A heavy “th” as in “this”.", tip: "Like ذ, tongue between the teeth, but with the back of the tongue raised.", place: "teeth", heavy: true, noEnglish: true, joinsAfter: true, practice: "ظَا", practiceTranslit: "ẓā", confusables: ["ذ", "ض", "ز"] },
  { char: "ع", name: "ʿAyn", arabicName: "عَيْن", translit: "ʿ", sound: "A voiced squeeze in the middle of the throat.", tip: "Tighten the throat as if starting to gently gag, while humming. It is a sound, not a silent letter.", place: "throat", noEnglish: true, joinsAfter: true, practice: "عَا", practiceTranslit: "ʿā", confusables: ["ء", "غ"] },
  { char: "غ", name: "Ghayn", arabicName: "غَيْن", translit: "gh", sound: "Like a French “r” in “Paris”, a soft gargle.", tip: "The voiced partner of خ. One dot above.", place: "throat", heavy: true, noEnglish: true, joinsAfter: true, practice: "غَا", practiceTranslit: "ghā", confusables: ["خ", "ع", "ر"] },
  { char: "ف", name: "Fā’", arabicName: "فَاء", translit: "f", sound: "Like “f” in “fish”.", tip: "Upper teeth on the lower lip. One dot above.", place: "lips", joinsAfter: true, practice: "فَا", practiceTranslit: "fā", confusables: ["ق", "ث"] },
  { char: "ق", name: "Qāf", arabicName: "قَاف", translit: "q", sound: "A deep “k” from the very back of the mouth.", tip: "Touch the back of the tongue to the soft palate, near where you swallow. Two dots above.", place: "tongue-back", heavy: true, noEnglish: true, joinsAfter: true, practice: "قَا", practiceTranslit: "qā", confusables: ["ك", "ف"] },
  { char: "ك", name: "Kāf", arabicName: "كَاف", translit: "k", sound: "Like “k” in “kite”.", tip: "Further forward than ق. Light and crisp.", place: "tongue-back", joinsAfter: true, practice: "كَا", practiceTranslit: "kā", confusables: ["ق"] },
  { char: "ل", name: "Lām", arabicName: "لَام", translit: "l", sound: "Like “l” in “light”.", tip: "Keep it light and clear, never the dark “l” of “full”, except in the word Allāh after a or u.", place: "tongue-tip", joinsAfter: true, practice: "لَا", practiceTranslit: "lā", confusables: ["ن", "ر"] },
  { char: "م", name: "Mīm", arabicName: "مِيم", translit: "m", sound: "Like “m” in “moon”.", tip: "Lips together, sound through the nose.", place: "lips", joinsAfter: true, practice: "مَا", practiceTranslit: "mā", confusables: ["ن"] },
  { char: "ن", name: "Nūn", arabicName: "نُون", translit: "n", sound: "Like “n” in “noon”.", tip: "Tongue tip on the ridge behind the teeth. One dot above.", place: "tongue-tip", joinsAfter: true, practice: "نَا", practiceTranslit: "nā", confusables: ["م", "ل"] },
  { char: "ه", name: "Hā’", arabicName: "هَاء", translit: "h", sound: "Like “h” in “hat”, light and airy.", tip: "From the deepest part of the throat, with no tightness. Compare with the strong ح.", place: "throat", joinsAfter: true, practice: "هَا", practiceTranslit: "hā", confusables: ["ح"] },
  { char: "و", name: "Wāw", arabicName: "وَاو", translit: "w / ū", sound: "Like “w” in “wet”, or a long “oo”.", tip: "Round the lips. Does not join to the next letter.", place: "lips", joinsAfter: false, practice: "وَا", practiceTranslit: "wā", confusables: ["ي"] },
  { char: "ي", name: "Yā’", arabicName: "يَاء", translit: "y / ī", sound: "Like “y” in “yes”, or a long “ee”.", tip: "Two dots below.", place: "tongue-middle", joinsAfter: true, practice: "يَا", practiceTranslit: "yā", confusables: ["و"] },
  { char: "ء", name: "Hamzah", arabicName: "هَمْزَة", translit: "ʾ", sound: "A glottal stop: the catch in “uh-oh”.", tip: "A clean stop of the breath at the bottom of the throat. Often sits on an alif (أ), wāw (ؤ) or yā’ (ئ).", place: "throat", noEnglish: true, joinsAfter: false, practice: "آ", practiceTranslit: "ʾā", confusables: ["ع"] },
];

export const letter = (char: string) => {
  const found = LETTERS.find(l => l.char === char);
  if (!found) throw new Error(`Unknown letter ${char}`);
  return found;
};

export const PLACE_LABEL: Record<Letter["place"], string> = {
  throat: "Throat",
  "tongue-back": "Back of the tongue",
  "tongue-middle": "Middle of the tongue",
  "tongue-tip": "Tip of the tongue",
  teeth: "Tongue and teeth",
  lips: "Lips",
  "long vowel": "Open mouth (vowel)",
};

const ZWJ = "‍";
/** Contextual shapes, rendered by the font through zero-width joiners. */
export function forms(l: Letter) {
  return {
    isolated: l.char,
    initial: l.joinsAfter ? l.char + ZWJ : l.char,
    medial: l.joinsAfter ? ZWJ + l.char + ZWJ : ZWJ + l.char,
    final: ZWJ + l.char,
  };
}

export const FATHA = "َ", KASRA = "ِ", DAMMA = "ُ", SUKUN = "ْ", SHADDA = "ّ";
export const VOWELS = [
  { mark: FATHA, name: "Fatḥah", sound: "a", hint: "a small slash above → “a” as in “cup”" },
  { mark: KASRA, name: "Kasrah", sound: "i", hint: "a small slash below → “i” as in “sit”" },
  { mark: DAMMA, name: "Ḍammah", sound: "u", hint: "a small wāw above → “u” as in “put”" },
] as const;

/** Romanize a consonant + vowel syllable, e.g. بَ → "ba". */
export function syllable(l: Letter, vowel: string) {
  const base = l.translit.split(" ")[0].replace("ā", "");
  return `${base || "ʾ"}${vowel}`;
}
