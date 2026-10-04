"""Generate reviewable Qaloon tajweed spans without replacing canonical text.

NOT a certified mujawwad mushaf. Context is wasl within an ayah, waqf at its
end. Route-dependent rules are flagged and never assigned a universal duration.
Offsets are UTF-16 code units, matching the browser's String.slice.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
VERSION = "qaloon-tajweed-candidates-0.1"
RULES = {
    "madd_lazim": {"en": "Necessary madd · 6 harakat", "ar": "مد لازم · ٦ حركات", "color": "#c1121f", "harakat": [6]},
    "madd_muttasil": {"en": "Connected madd · route review", "ar": "مد متصل · راجع الطريق", "color": "#df3a32", "harakat": None},
    "madd_munfasil": {"en": "Separate madd · Qaloon route/choice review", "ar": "مد منفصل · راجع طريق قالون والوجه", "color": "#df3a32", "harakat": None},
    "madd_arid": {"en": "Pause-induced madd · 2 / 4 / 6 harakat", "ar": "مد عارض للسكون عند الوقف · ٢ / ٤ / ٦", "color": "#df3a32", "harakat": [2, 4, 6]},
    "madd_tabii": {"en": "Natural madd · 2 harakat", "ar": "مد طبيعي · حركتان", "color": "#c94b41", "harakat": [2]},
    "idgham": {"en": "Nun/tanwin idgham · joined reading", "ar": "إدغام النون أو التنوين · عند الوصل", "color": "#16743a", "harakat": None},
    "ikhfa": {"en": "Nun/tanwin ikhfa · joined reading", "ar": "إخفاء النون أو التنوين · عند الوصل", "color": "#646b73", "harakat": None},
    "iqlab": {"en": "Iqlab before ba · joined reading", "ar": "إقلاب قبل الباء · عند الوصل", "color": "#8a4c9c", "harakat": None},
    "qalqala": {"en": "Qalqalah · sukun/ending pause", "ar": "قلقلة · عند السكون أو الوقف آخر الآية", "color": "#007e8c", "harakat": None},
    "tafkhim": {"en": "Emphatic isti'la letter", "ar": "تفخيم حرف استعلاء", "color": "#164acb", "harakat": None},
    "ghunna": {"en": "Doubled nun/mim · ghunna", "ar": "غنة النون أو الميم المشددتين", "color": "#795e17", "harakat": None},
}
PRIORITY = {rule: index for index, rule in enumerate(RULES)}


def glyphs(text):
    result = []
    word = 0
    for i, char in enumerate(text):
        if char.isspace():
            if i and not text[i - 1].isspace():
                word += 1
        elif unicodedata.category(char).startswith("M") or char == "ـ":
            if result:
                result[-1]["marks"] += char
                result[-1]["end"] = i + 1
        elif "\u0621" <= char <= "\u064a" or char in "ٱے":
            result.append({"base": char, "marks": "", "start": i, "end": i + 1, "word": word})
    return result


def annotate(text, disjoint=False):
    letters = glyphs(text)
    spans = []
    warnings = ["Machine candidates; qualified Qaloon review required.",
                "Context: joined words within this ayah, stop at ayah end; internal waqf/next-ayah wasl are not modeled.",
                "Qaloon tariq not selected: mim al-jam, pronoun silah, hamza variants and disjoint-letter madd are not resolved."]
    tanwin = "ًٌٍٖٗٞ"
    def add(index, rule, note=""):
        letter = letters[index]
        spans.append({"start": len(text[:letter["start"]].encode("utf-16-le")) // 2,
                      "end": len(text[:letter["end"]].encode("utf-16-le")) // 2,
                      "rule": rule, "status": "needs-review", "context_note": note})
    # Context, not spelling alone: أَلَمْ in an ordinary verse is NOT the
    # disjoint letters alif-lam-mim. The caller supplies the known ayah context.
    if disjoint:
        warnings.append("Disjoint letters: letter-name madd requires a separate reviewed lexical rule.")
        return {"text": text, "spans": [], "warnings": warnings, "status": "needs-review"}
    for i, letter in enumerate(letters):
        base, marks = letter["base"], letter["marks"]
        nxt = letters[i + 1] if i + 1 < len(letters) else None
        prev = letters[i - 1] if i else None
        at_end = i == len(letters) - 1
        if base in "خصضغطقظ":
            add(i, "tafkhim", "Only inherently emphatic letters; ra/lam/alif contextual heaviness is not inferred.")
        if base in "قطبجد" and ("ْ" in marks or at_end):
            add(i, "qalqala", "At ayah end this applies only when actually stopping; no fixed added vowel.")
        if base in "نم" and "ّ" in marks:
            add(i, "ghunna")
        if nxt:
            is_nun = base == "ن" and ("ْ" in marks or not any(v in marks for v in "َُِّ"))
            is_tanwin = any(v in marks for v in tanwin)
            # Spelling alif after fatḥatayn is not the following pronounced sound.
            following = nxt
            if is_tanwin and nxt["base"] in "اى" and nxt["word"] == letter["word"] and i + 2 < len(letters):
                following = letters[i + 2]
            if is_nun or is_tanwin:
                other_word = following["word"] != letter["word"]
                if following["base"] in "يرملون" and other_word:
                    add(i, "idgham", "Nun-sakin idgham is across words, not dunya/bunyan/qinwan/sinwan within a word.")
                elif following["base"] in "تثجدذزسشصضطظفقك":
                    add(i, "ikhfa")
                elif following["base"] == "ب":
                    add(i, "iqlab")
        # Conservative long-vowel recognition; source exception signs are kept.
        long_vowel = base == "آ" or "ٰ" in marks or (prev and letter["word"] == prev["word"] and (
            base == "ا" and "َ" in prev["marks"] and not any(v in prev["marks"] for v in tanwin)
            or base in "يے" and "ِ" in prev["marks"] and not any(v in marks for v in "َُِّ")
            or base == "و" and "ُ" in prev["marks"] and not any(v in marks for v in "َُِّ")
            or base == "ى" and "َ" in prev["marks"]))
        if long_vowel:
            if "۟" in marks:
                warnings.append("Zero/silent-letter notation encountered; long-vowel candidate withheld.")
                continue
            if nxt and nxt["word"] == letter["word"] and ("ّ" in nxt["marks"] or "ْ" in nxt["marks"]):
                add(i, "madd_lazim", "Permanent sukun/shadda candidate; spelling exceptions still require review.")
            elif nxt and nxt["base"] in "ءأإؤئآ" and nxt["word"] == letter["word"]:
                add(i, "madd_muttasil", "Common Qaloon/Shatibiyyah tawassut is 4 harakat; exact tariq must be confirmed before fixing a count.")
            elif nxt and nxt["base"] in "ءأإآ" and nxt["word"] != letter["word"]:
                add(i, "madd_munfasil", "Qaloon qasr/tawassut choices must be selected consistently; not a universal 4.")
            elif nxt and i + 1 == len(letters) - 1 and "ّ" not in nxt["marks"] and "ْ" not in nxt["marks"]:
                add(i, "madd_arid", "Length options arise on stopping, not on connected reading.")
            else:
                add(i, "madd_tabii")
    # Flags survive even when the generic classifier found a plausible span.
    if any(c in text for c in "ٕٔۧۨ"):
        warnings.append("Small/combining hamza or small letter notation: manual riwayah-specific review required.")
    spans.sort(key=lambda s: (s["start"], PRIORITY[s["rule"]]))
    return {"text": text, "spans": spans, "warnings": list(dict.fromkeys(warnings)), "status": "needs-review"}


def build(source, output):
    surahs = []
    digest = hashlib.sha256()
    counts = Counter()
    for file in sorted(source.glob("*.json")):
        raw = file.read_bytes()
        digest.update(file.name.encode()); digest.update(raw)
        surah = json.loads(raw)
        ayahs = []
        for ayah in surah["ayahs"]:
            # Offsets must address precisely the existing reader representation,
            # not silently switch it to differently marked raw-source spelling.
            text = re.sub(r"[\d\u0660-\u0669]+", "", ayah.get("displayText") or ayah["text"]).strip()
            starts_with_letters = surah["id"] in {2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30, 31, 32, 36, 38, 40, 41, 42, 43, 44, 45, 46, 50, 68} and ayah["ayah"] == 1
            data = annotate(text, disjoint=starts_with_letters or (surah["id"], ayah["ayah"]) == (42, 2))
            counts.update(s["rule"] for s in data["spans"])
            data["warnings"].append("Spans address the reader displayText; original Qaloon source is retained separately. Simplified typography cannot establish every riwayah-specific rule.")
            ayahs.append({"ayah": ayah["ayah"], "source_text": ayah["text"], **data})
        surahs.append({"id": surah["id"], "ayahs": ayahs})
    if len(surahs) != 114:
        raise ValueError("Cache all 114 Qaloon surahs before generating the whole-Quran JSON")
    payload = {"schema_version": 1, "version": VERSION, "riwayah": "Qaloon an Nafi", "tariq": "unselected-needs-qualified-review",
               "status": "machine-candidates-not-certified-mushaf", "approved_ayahs": 0,
               "duration_unit": "harakat-not-seconds", "offset_unit": "utf16", "source_sha256": digest.hexdigest(),
               "context": "wasl-within-ayah-waqf-at-end", "rules": RULES, "counts": dict(counts), "surahs": surahs}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Generated {sum(len(s['ayahs']) for s in surahs)} ayahs / {sum(counts.values())} candidate spans. 0 approved. {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "web/public/quran/surahs")
    parser.add_argument("--output", type=Path, default=ROOT / "web/public/quran/qalon_majwad_mushaf.json")
    args = parser.parse_args()
    build(args.source, args.output)
