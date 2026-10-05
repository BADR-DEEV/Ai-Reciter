// Articulation points (makhārij) and qualities (ṣifāt) of the letters, from
// chapter 2 of the Awqaf curriculum (pp. 33–42), following Ibn al-Jazarī's 17 places.

export type Place = { id: string; area: string; en: string; ar: string; tip: string; letters: string };
export const AREAS: Record<string, { en: string; ar: string }> = {
  jawf: { en: "The mouth's empty space", ar: "الجوف" },
  halq: { en: "The throat", ar: "الحلق" },
  lisan: { en: "The tongue", ar: "اللسان" },
  shafatan: { en: "The lips", ar: "الشفتان" },
  khayshum: { en: "The nose", ar: "الخيشوم" },
};

export const PLACES: Place[] = [
  { id: "jawf", area: "jawf", en: "Empty space of throat and mouth", ar: "الجوف", letters: "اوي", tip: "The three long vowels ā, ū, ī flow out with nothing touching." },
  { id: "deep_throat", area: "halq", en: "Deepest part of the throat", ar: "أقصى الحلق", letters: "ءه", tip: "Hamza is a clean catch in the throat; hā is a soft breath from the same place." },
  { id: "mid_throat", area: "halq", en: "Middle of the throat", ar: "وسط الحلق", letters: "عح", tip: "Squeeze the middle of the throat: voiced for ʿayn, breathy for ḥā." },
  { id: "near_throat", area: "halq", en: "Top of the throat", ar: "أدنى الحلق", letters: "غخ", tip: "Like gargling: voiced for ghayn, rough breath for khāʾ." },
  { id: "q", area: "lisan", en: "Back of the tongue against the soft palate (deepest)", ar: "أقصى اللسان مما يلي الحلق", letters: "ق", tip: "Further back than k, with a full, heavy sound." },
  { id: "k", area: "lisan", en: "Back of the tongue, a little forward", ar: "أقصى اللسان أسفل مخرج القاف", letters: "ك", tip: "Like English k, light and with a puff of air." },
  { id: "mid_tongue", area: "lisan", en: "Middle of the tongue against the hard palate", ar: "وسط اللسان", letters: "جشي", tip: "Jīm, shīn and consonant yā all touch here." },
  { id: "d_side", area: "lisan", en: "Side of the tongue against the upper molars", ar: "إحدى حافتي اللسان", letters: "ض", tip: "Unique to Arabic: the sound stretches along the side of the tongue (istiṭāla)." },
  { id: "l", area: "lisan", en: "Front edges of the tongue against the gums", ar: "أدنى حافتي اللسان إلى منتهى طرفه", letters: "ل", tip: "Air escapes around the sides of the tongue." },
  { id: "n", area: "lisan", en: "Tip of the tongue, just under the l", ar: "طرف اللسان تحت مخرج اللام", letters: "ن", tip: "Always carries some nasal sound (ghunna)." },
  { id: "r", area: "lisan", en: "Tip of the tongue, slightly further back", ar: "طرف اللسان مع ظهره قليلًا", letters: "ر", tip: "A single tap; do not roll it (takrīr must be held back)." },
  { id: "tip_ridge", area: "lisan", en: "Tip of the tongue against the roots of the upper front teeth", ar: "طرف اللسان مع أصول الثنايا العليا", letters: "طدت", tip: "Ṭāʾ is the heavy partner of tāʾ; dāl is the voiced one." },
  { id: "tip_whistle", area: "lisan", en: "Tip of the tongue near the lower front teeth", ar: "طرف اللسان مع الثنايا السفلى", letters: "صسز", tip: "The whistling letters (ṣafīr); ṣād is the heavy one." },
  { id: "tip_teeth", area: "lisan", en: "Tip of the tongue against the edges of the upper front teeth", ar: "طرف اللسان مع أطراف الثنايا العليا", letters: "ظذث", tip: "Let the tongue tip show slightly between the teeth." },
  { id: "f", area: "shafatan", en: "Inside of the lower lip against the upper teeth", ar: "بطن الشفة السفلى مع أطراف الثنايا العليا", letters: "ف", tip: "Like English f." },
  { id: "lips", area: "shafatan", en: "Both lips", ar: "الشفتان", letters: "بمو", tip: "Lips close for b and m; they round without closing for consonant w." },
  { id: "nose", area: "khayshum", en: "The nasal passage", ar: "الخيشوم", letters: "نم", tip: "The ghunna of nūn and mīm comes from here; pinch your nose and it stops." },
];

type Quality = { id: string; en: string; ar: string; letters: string; opposite?: string; explain: string };
// Paired qualities (each letter has one of each pair), then the unpaired ones.
export const QUALITIES: Quality[] = [
  { id: "hams", en: "Whispered (hams)", ar: "الهمس", letters: "فحثهشخصسكت", opposite: "jahr", explain: "Breath keeps flowing: «فحثه شخص سكت»." },
  { id: "jahr", en: "Voiced (jahr)", ar: "الجهر", letters: "ءبجدذرزضطظعغقلمنوي", opposite: "hams", explain: "The breath is held back and the voice is full." },
  { id: "shidda", en: "Stopped (shidda)", ar: "الشدة", letters: "ءجدقطبكت", opposite: "rakhawa", explain: "The sound stops completely: «أجد قط بكت»." },
  { id: "tawassut", en: "In between (tawassuṭ)", ar: "التوسط", letters: "لنعمر", explain: "Neither fully stopped nor flowing: «لن عمر»." },
  { id: "rakhawa", en: "Flowing (rakhāwa)", ar: "الرخاوة", letters: "ثحخذزسشصضظغفهوي", opposite: "shidda", explain: "The sound keeps running." },
  { id: "istila", en: "Raised (istiʿlāʾ)", ar: "الاستعلاء", letters: "خصضغطقظ", opposite: "istifal", explain: "The back of the tongue rises, so the letter is always heavy: «خص ضغط قظ»." },
  { id: "istifal", en: "Lowered (istifāl)", ar: "الاستفال", letters: "ءبتثجحدذرزسشعفكلمنهوي", opposite: "istila", explain: "The tongue stays low; these letters are light (except rāʾ and lām in some places)." },
  { id: "itbaq", en: "Covered (iṭbāq)", ar: "الإطباق", letters: "صضطظ", opposite: "infitah", explain: "The tongue presses up against the palate: the heaviest letters." },
  { id: "infitah", en: "Open (infitāḥ)", ar: "الانفتاح", letters: "ءبتثجحخدذرزسشعغفقكلمنهوي", opposite: "itbaq", explain: "The tongue stays away from the palate." },
  { id: "idhlaq", en: "Fluent (idhlāq)", ar: "الإذلاق", letters: "فرمنلب", opposite: "ismat", explain: "Said with the tip of the tongue or lips: «فر من لب»." },
  { id: "ismat", en: "Restrained (iṣmāt)", ar: "الإصمات", letters: "ءتثجحخدذزسشصضطظعغقكهوي", opposite: "idhlaq", explain: "Heavier to say; a long Arabic root always mixes in a fluent letter." },
  { id: "safir", en: "Whistle (ṣafīr)", ar: "الصفير", letters: "صزس", explain: "A sharp whistle accompanies the letter." },
  { id: "qalqala", en: "Echo (qalqala)", ar: "القلقلة", letters: "قطبجد", explain: "When sākin it bounces with a small echo: «قطب جد»." },
  { id: "lin", en: "Soft (līn)", ar: "اللين", letters: "وي", explain: "Wāw and yāʾ sākin after a fatḥa (خَوْف، بَيْت)." },
  { id: "inhiraf", en: "Leaning (inḥirāf)", ar: "الانحراف", letters: "لر", explain: "The sound leans away from its place." },
  { id: "takrir", en: "Repetition (takrīr)", ar: "التكرير", letters: "ر", explain: "The tongue tends to trill; hold it to a single tap." },
  { id: "tafashshi", en: "Spreading (tafashshī)", ar: "التفشي", letters: "ش", explain: "The air spreads through the mouth." },
  { id: "istitala", en: "Stretching (istiṭāla)", ar: "الاستطالة", letters: "ض", explain: "The sound stretches along the side of the tongue." },
];

export const LETTERS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".split("");
export const LETTER_NAMES: Record<string, string> = {
  ء: "hamza", ب: "bāʾ", ت: "tāʾ", ث: "thāʾ", ج: "jīm", ح: "ḥāʾ", خ: "khāʾ", د: "dāl", ذ: "dhāl", ر: "rāʾ", ز: "zāy", س: "sīn",
  ش: "shīn", ص: "ṣād", ض: "ḍād", ط: "ṭāʾ", ظ: "ẓāʾ", ع: "ʿayn", غ: "ghayn", ف: "fāʾ", ق: "qāf", ك: "kāf", ل: "lām", م: "mīm",
  ن: "nūn", ه: "hāʾ", و: "wāw", ي: "yāʾ",
};

export const placeOf = (letter: string) => PLACES.find(p => p.id !== "jawf" && p.id !== "nose" && p.letters.includes(letter))!;
export const qualitiesOf = (letter: string) => QUALITIES.filter(q => q.letters.includes(letter));
