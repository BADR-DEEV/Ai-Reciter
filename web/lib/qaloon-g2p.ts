/** Source-driven, conservative orthographic G2P. NOT a tajweed oracle.
 * Qālūn source vowels are retained, never replaced with a Hafs transliteration.
 * Every result remains draft until a qualified Qālūn reviewer signs it off.
 */
export const G2P_VERSION = "qaloon-orthographic-0.1";
export type Phonetics = { text: string; words: string[]; warnings: string[]; status: "draft"; version: string };
const consonants: Record<string, string> = {
  "ء": "ʾ", "أ": "ʾ", "إ": "ʾ", "ؤ": "ʾ", "ئ": "ʾ", "ب": "b", "ت": "t", "ث": "th",
  "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s",
  "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ", "ع": "ʿ", "غ": "gh", "ف": "f",
  "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y", "ة": "t",
};
const vowels: Record<string, string> = { "َ": "a", "ِ": "i", "ُ": "u", "ً": "an", "ٍ": "in", "ٌ": "un" };
const sun = new Set(Array.from("تثدذرزسشصضطظلن"));
type Glyph = { base: string; marks: string };

function glyphs(word: string, warnings: Set<string>): Glyph[] {
  const result: Glyph[] = [];
  // A recitation madd sign on an ordinary alif is NOT a hamza. NFC alone
  // would turn ضآل into ضآل and incorrectly introduce a glottal stop.
  const prepared = word.replace(/ا\u0653/gu, "ا").replace(/\u0656/gu, "ٍ").replace(/\u0657/gu, "ً").replace(/\u065e/gu, "ٌ");
  for (const char of prepared.normalize("NFC")) {
    if (/\p{M}/u.test(char)) {
      if (!/[\u064b-\u0655\u0670\u06d6\u06df\u06e2\u06ea\u06ec]/u.test(char)) warnings.add(`Source mark U+${char.codePointAt(0)!.toString(16).toUpperCase()} needs an explicit reviewed rule.`);
      if (result.length) result[result.length - 1].marks += char;
    } else if (char === "ـ") continue;
    else if (/[\u06e5\u06e6]/u.test(char)) {
      warnings.add("Small waw/ya: pronoun silah and chosen Qaloon transmission need audio review.");
      result.push({ base: char === "ۥ" ? "و" : "ي", marks: "" });
    } else if (consonants[char] || "اٱآىے".includes(char)) {
      result.push({ base: char === "ے" ? "ي" : char, marks: "" });
    } else if (!/[\s\d\u0660-\u0669\u06dd\u06de]/u.test(char)) {
      warnings.add(`Unsupported character U+${char.codePointAt(0)!.toString(16).toUpperCase()}; do not infer its sound.`);
    }
  }
  return result;
}

function renderWord(word: string, joined: boolean, pause: boolean, warnings: Set<string>): string {
  const letters = glyphs(word, warnings);
  if (!letters.length) return "";
  if (/[\u06e2\u06ed]/u.test(word) || /[ًٌٍ]/u.test(word)) warnings.add("Nun/tanwin articulation across words is not encoded; review with the reference audio.");
  if (/[\u06e4\u0653]/u.test(word)) warnings.add("Madd sign: duration is not represented by Latin spelling.");
  if (/[\u06ea\u06eb]/u.test(word)) warnings.add("Maghrebi vowel/recitation sign: review the source-specific realization.");
  // The Maghrebi source's leading vowel on this imperative is not a literal
  // standalone /u/. Start the imperative with /i/; elide in connected reading.
  const bare = letters.map(l => l.base).join("");
  const disjoint = new Set(["الم", "الر", "المر", "كهيعص", "طه", "طسم", "طس", "يس", "ص", "حم", "عسق", "ق", "ن"]);
  const names: Record<string, string> = { "ا": "alif", "ل": "lām", "م": "mīm", "ر": "rā", "ك": "kāf", "ه": "hā", "ي": "yā", "ع": "ʿayn", "ص": "ṣād", "ط": "ṭā", "س": "sīn", "ح": "ḥā", "ق": "qāf", "ن": "nūn" };
  const letterSequence = bare.replace(/[أإٱ]/gu, "ا");
  if (disjoint.has(letterSequence)) {
    warnings.add("Disjoint letters are letter names, not a word; madd timing requires teacher review.");
    return Array.from(letterSequence).map(c => names[c]).join(" ");
  }
  if (bare === "اهدنا") return `${joined ? "" : "i"}hdinā`;
  const relative = bare.match(/^([وفبك]?)(الذي|الذين)$/u);
  if (relative) {
    const prefix = relative[1] ? consonants[relative[1]] + (relative[1] === "ب" ? "i" : "a") : "";
    return prefix + (joined || prefix ? "lladhī" : "alladhī") + (relative[2] === "الذين" ? pause ? "n" : "na" : "");
  }
  if (bare === "لله" || bare === "الله" || /^[وفبك]الله$/u.test(bare)) {
    const last = letters[letters.length - 1];
    const suffix = pause ? "" : Object.keys(vowels).map(v => last.marks.includes(v) ? vowels[v] : "").find(Boolean) || "";
    if (bare === "لله") return "lillāh" + suffix;
    if (bare === "الله") return (joined ? "llāh" : "allāh") + suffix;
    const first = letters[0];
    const prefix = consonants[first.base] + (Object.keys(vowels).map(v => first.marks.includes(v) ? vowels[v] : "").find(Boolean) || "");
    return prefix + "llāh" + suffix;
  }
  if (/ه[ُِ]?م/u.test(word.replace(/[\u0651\u0652]/gu, ""))) warnings.add("Mim al-jam: sukun/silah depends on the selected Qaloon route; orthography alone is insufficient.");
  let output = "";
  let from = 0;
  let article = -1;
  if ("اٱ".includes(letters[0].base) && letters[1]?.base === "ل") article = 0;
  else if ("وفبك".includes(letters[0].base) && "اٱ".includes(letters[1]?.base || "") && letters[2]?.base === "ل") {
    const first = letters[0];
    output = consonants[first.base] + (Object.keys(vowels).map(v => first.marks.includes(v) ? vowels[v] : "").find(Boolean) || "");
    article = 1;
  }
  if (article >= 0) {
    const next = letters[article + 2];
    const onset = joined || article > 0 ? "" : "a";
    // The following sun letter's shadda supplies the second consonant.
    output += onset + (next && sun.has(next.base) ? consonants[next.base] : "l") + "-";
    from = article + 2;
  }
  for (let i = from; i < letters.length; i++) {
    const { base, marks } = letters[i];
    const last = i === letters.length - 1;
    const previous = letters[i - 1];
    const next = letters[i + 1];
    const vowelMark = Object.keys(vowels).find(v => marks.includes(v));
    let vowel = vowelMark ? vowels[vowelMark] : "";
    if (marks.includes("ٓ") && "أإؤئء".includes(base) && !vowel) vowel = "ā";
    if (base === "آ") { output += "ʾā"; continue; }
    if ("اٱ".includes(base)) {
      // Silent spelling alif after fatḥatayn and after plural wāw.
      if (previous?.marks.includes("ً") || (last && previous?.base === "و" && previous.marks.includes("ْ"))) continue;
      if (i === 0) {
        if (joined && (base === "ٱ" || marks.includes("۬") || !vowelMark)) continue;
        if (!vowel) warnings.add("Unvowelled initial alif/hamzat al-wasl needs a reviewed starting vowel.");
        output += vowel || "⟨?⟩";
      } else if (output.endsWith("a")) output = output.slice(0, -1) + "ā";
      else if (vowel) { output += "ʾ" + vowel; warnings.add("Medial alif onset needs hamza/wasl review."); }
      else warnings.add("Unresolved alif (silent versus long vowel); review the word.");
      continue;
    }
    if (base === "ى") {
      output = output.replace(/a$/, "") + (marks.includes("ً") && !pause ? "an" : "ā");
      continue;
    }
    if (!vowelMark && !marks.includes("ّ") && "وي".includes(base)) {
      if (base === "و" && output.endsWith("u")) { output = output.slice(0, -1) + "ū"; continue; }
      if (base === "ي" && output.endsWith("i")) { output = output.slice(0, -1) + "ī"; continue; }
    }
    let sound = consonants[base] || "⟨?⟩";
    if (marks.includes("ّ") && !(article >= 0 && i === from && sun.has(base))) sound += sound;
    if (marks.includes("ٰ")) vowel = "ā";
    if (pause && last) {
      if (base === "ة") { sound = "h"; vowel = ""; }
      else if (vowel === "an") vowel = "ā";
      else if (["a", "i", "u", "in", "un"].includes(vowel)) vowel = "";
    }
    // In the common fatḥatayn + spelling alif, pause applies to the consonant.
    if (pause && next?.base === "ا" && i === letters.length - 2 && vowel === "an") vowel = "ā";
    if (!vowel && !marks.includes("ْ") && !marks.includes("ّ") && !(pause && last) && !"وي".includes(base)) {
      warnings.add("An unvowelled consonant may involve assimilation or source notation; review rather than guessing.");
    }
    output += sound + vowel;
  }
  return output;
}

export function qaloonG2P(text: string, ending: "pause" | "connect" = "pause"): Phonetics {
  const warnings = new Set<string>();
  const source = text.replace(/[\u0660-\u0669\d\u06dd\u06de]+/gu, "").trim().split(/\s+/u).filter(Boolean);
  const words = source.map((word, i) => renderWord(word, i > 0, ending === "pause" && i === source.length - 1, warnings));
  return { text: words.join(" "), words, warnings: [...warnings], status: "draft", version: G2P_VERSION };
}
