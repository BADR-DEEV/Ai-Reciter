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
VERSION = "qaloon-shatibiyyah-candidates-0.3"
LEGEND = {
    "necessary": {"en": "Necessary madd · 6 counts", "ar": "مد لازم · ٦ حركات", "color": "#ae0088"},
    "connected": {"en": "Ordinary connected madd · Qaloon 4", "ar": "مد متصل عادي · قالون ٤ حركات", "color": "#dd007f"},
    "permitted": {"en": "Permitted madd · see rule/options", "ar": "مد جائز · بحسب الحكم والوجه", "color": "#d18a00"},
    "natural": {"en": "Natural madd · 2 counts", "ar": "مد طبيعي · حركتان", "color": "#a98924"},
    "nasal": {"en": "Ikhfa / iqlab / ghunna · 2 counts", "ar": "إخفاء وإقلاب ومواقع الغنة · حركتان", "color": "#009b64"},
    "merged": {"en": "Merged or unpronounced letter · in wasl", "ar": "إدغام وما لا يلفظ · عند الوصل", "color": "#8c9188"},
    "heavy": {"en": "Tafkhim · heavy articulation", "ar": "تفخيم", "color": "#007692"},
    "echo": {"en": "Qalqalah · echo on a sakin letter", "ar": "قلقلة", "color": "#00a6bd"},
}
RULES = {
    "madd_lazim": {"en": "Necessary madd · 6 harakat", "ar": "مد لازم · ٦ حركات", "group": "necessary", "harakat": [6]},
    "madd_muttasil": {"en": "Ordinary connected madd · Qaloon/Shatibiyyah 4 harakat", "ar": "مد متصل عادي · قالون من الشاطبية ٤ حركات", "group": "connected", "harakat": [4]},
    "madd_changed_hamza": {"en": "Madd before changed adjacent hamza · combination review", "ar": "مد قبل همزة متغيرة · مراجعة الأوجه والتركيب", "group": "permitted", "harakat": None},
    "madd_munfasil": {"en": "Separate madd · Qaloon/Shatibiyyah 2 or 4 harakat", "ar": "مد منفصل · قالون من الشاطبية ٢ أو ٤ حركات", "group": "permitted", "harakat": [2, 4]},
    "madd_arid": {"en": "Pause-induced madd · 2 / 4 / 6 harakat", "ar": "مد عارض للسكون عند الوقف · ٢ / ٤ / ٦", "group": "permitted", "harakat": [2, 4, 6]},
    "madd_tabii": {"en": "Natural madd · 2 harakat", "ar": "مد طبيعي · حركتان", "group": "natural", "harakat": [2]},
    "madd_iwad": {"en": "Fathatayn replacement on stopping · 2 harakat (not ta marbuta)", "ar": "مد عوض عند الوقف على تنوين الفتح · حركتان، دون التاء المربوطة", "group": "natural", "harakat": [2]},
    "madd_lin": {"en": "Lin before final pause-induced sukun · 2 / 4 / 6 harakat", "ar": "مد لين قبل السكون العارض عند الوقف · ٢ / ٤ / ٦ حركات", "group": "permitted", "harakat": [2, 4, 6]},
    "idgham": {"en": "Nun/tanwin merged · wasl; ghunna only with its nasal letters", "ar": "إدغام النون أو التنوين · عند الوصل؛ الغنة مع حروفها", "group": "merged", "harakat": None},
    "silent": {"en": "Unpronounced source-marked letter · wasl context", "ar": "حرف لا يلفظ بحسب علامة المصدر · في سياق الوصل", "group": "merged", "harakat": None},
    "ikhfa": {"en": "Nun/tanwin ikhfa · ghunna 2 harakat in wasl", "ar": "إخفاء النون أو التنوين · غنة حركتين عند الوصل", "group": "nasal", "harakat": [2]},
    "iqlab": {"en": "Iqlab before ba · ghunna 2 harakat in wasl", "ar": "إقلاب قبل الباء · غنة حركتين عند الوصل", "group": "nasal", "harakat": [2]},
    "ikhfa_shafawi": {"en": "Sakin mim before ba · ghunna 2; conditional on mim sukun, not silah", "ar": "إخفاء شفوي للميم الساكنة قبل الباء · غنة حركتين؛ مع وجه السكون لا الصلة", "group": "nasal", "harakat": [2]},
    "idgham_shafawi": {"en": "Sakin mim into mim · ghunna 2; conditional on mim sukun, not silah", "ar": "إدغام شفوي للميم الساكنة في الميم · غنة حركتين؛ مع وجه السكون لا الصلة", "group": "nasal", "harakat": [2]},
    "qalqala": {"en": "Qalqalah · sukun/ending pause", "ar": "قلقلة · عند السكون أو الوقف آخر الآية", "group": "echo", "harakat": None},
    "tafkhim": {"en": "Tafkhim · supported letter/context", "ar": "تفخيم · للحرف والسياق المدعوم", "group": "heavy", "harakat": None},
    "ghunna": {"en": "Ghunna · 2 harakat", "ar": "غنة · حركتان", "group": "nasal", "harakat": [2]},
}
for rule in RULES.values():
    rule["color"] = LEGEND[rule["group"]]["color"]
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


def annotate(text, disjoint=False, source_text=None):
    letters = glyphs(text)
    source_letters = glyphs(source_text or text)
    same_letters = len(source_letters) == len(letters) and all(a["base"].replace("ٱ", "ا") == b["base"].replace("ٱ", "ا") for a, b in zip(source_letters, letters))
    spans = []
    warnings = ["Machine candidates; qualified Qaloon review required.",
                "Context: joined words within this ayah, stop at ayah end; internal waqf/next-ayah wasl are not modeled.",
                "Shatibiyyah madd options are listed, not an arbitrary Hafs color-key length. Consistent performance choices still require review.",
                "Mim al-jam, pronoun silah, hamza variants and disjoint-letter madd require specialized review; unresolved cases are not assigned a fixed color/count."]
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
        raw_marks = source_letters[i]["marks"] if same_letters else marks
        at_end = i == len(letters) - 1
        if "۟" in raw_marks or (prev and prev["word"] != letter["word"] and (base == "ٱ" or "۬" in raw_marks)):
            add(i, "silent", "Source silent/wasl notation; wasl marking applies only when joining the preceding word.")
        if base == "ل" and prev and prev["base"] in "اٱ" and nxt and nxt["base"] in "تثدذرزسشصضطظلن" and "ّ" in nxt["marks"]:
            add(i, "silent", "Lam of the sun-letter article is assimilated, not pronounced separately.")
        if base in "خصضغطقظ":
            add(i, "tafkhim", "Inherently emphatic isti'la letter; precise heaviness strength still depends on vowel/context.")
        if base == "ر" and not at_end:
            if any(v in marks for v in "ًٌَُ") or ("ْ" in marks and prev and any(v in prev["marks"] for v in "َُ") and prev["base"] not in "اٱ"):
                add(i, "tafkhim", "Non-final ra with fatha/damma, or sakin ra after an explicit preceding fatha/damma; other ra/waqf exceptions need review.")
        if base == "ل" and "ّ" in marks:
            word_letters = [x for x in letters if x["word"] == letter["word"]]
            word_base = "".join(x["base"] for x in word_letters).replace("ٱ", "ا")
            if word_base == "الله":
                preceding = next((x for x in reversed(letters[:i]) if x["word"] != letter["word"]), None)
                if preceding is None or any(v in preceding["marks"] for v in "َُ"):
                    add(i, "tafkhim", "Lam of Allah is heavy initially or after an explicit fatha/damma; not after kasra.")
        if base in "قطبجد" and ("ْ" in marks or (at_end and not any(v in marks for v in "ًٗ"))):
            add(i, "qalqala", "At ayah end this applies only when actually stopping; no fixed added vowel.")
        if base in "نم" and "ّ" in marks:
            add(i, "ghunna")
        if base == "م" and "ْ" in marks and nxt and nxt["word"] != letter["word"]:
            if nxt["base"] == "ب":
                add(i, "ikhfa_shafawi", "Applies when this mim is actually read sakin. Qaloon mim al-jam silah is a separate unselected performance choice, not inferred from this color.")
            elif nxt["base"] == "م":
                add(i, "idgham_shafawi", "Applies to the sukun realization; mim al-jam silah must be reviewed separately.")
                add(i + 1, "ghunna", "Recipient of mim-sakin assimilation, conditional on sukun realization.")
        if (base in "وي" and "ْ" in marks and prev and "َ" in prev["marks"]
                and prev["word"] == letter["word"] and nxt and i + 1 == len(letters) - 1
                and nxt["word"] == letter["word"] and nxt["base"] not in "ةاىويے"
                and not any(v in nxt["marks"] for v in "ًٗ")):
            add(i, "madd_lin", "Only on stopping here; in wasl this is a lin sound, not natural madd.")
        # The Maghrebi source can attach fathatayn to the spelling ALIF,
        # whereas other fonts place it on the preceding consonant. Read both
        # layouts without changing the source text or display offsets.
        if at_end and base in "اى" and prev and any(v in prev["marks"] + marks for v in "ًٗ"):
            add(i, "madd_iwad", "Replace final fathatayn by a long a on stopping, not sukun on its consonant.")
            continue
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
                    if following["base"] in "ينمو":
                        add(letters.index(following), "ghunna", "Nasal recipient of nun/tanwin idgham; no ghunna for lam/ra.")
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
            if "۟" in raw_marks:
                warnings.append("Zero/silent-letter notation encountered; long-vowel candidate withheld.")
                continue
            if nxt and nxt["word"] == letter["word"] and ("ّ" in nxt["marks"] or "ْ" in nxt["marks"]):
                add(i, "madd_lazim", "Permanent sukun/shadda candidate; spelling exceptions still require review.")
            elif nxt and nxt["base"] in "ءأإؤئآ" and nxt["word"] == letter["word"]:
                next_next = letters[i + 2] if i + 2 < len(letters) else None
                if next_next and next_next["word"] != nxt["word"] and next_next["base"] in "ءأإؤئآ":
                    add(i, "madd_changed_hamza", "Adjacent hamzas across words can change the madd cause in Qaloon; tas-hil/omission and permissible combinations need lexical/route review.")
                else:
                    add(i, "madd_muttasil", "Ordinary unchanged hamzat-qat in the same word: Qaloon from Shatibiyyah reads 4 harakat.")
            elif nxt and nxt["base"] in "ءأإآ" and nxt["word"] != letter["word"]:
                add(i, "madd_munfasil", "Qaloon qasr/tawassut choices must be selected consistently; not a universal 4.")
            elif (nxt and i + 1 == len(letters) - 1 and "ْ" not in nxt["marks"]
                  and nxt["base"] not in "ةاىويے" and not any(v in nxt["marks"] for v in "ًٗ")):
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
            data = annotate(text, disjoint=starts_with_letters or (surah["id"], ayah["ayah"]) == (42, 2), source_text=ayah["text"])
            counts.update(s["rule"] for s in data["spans"])
            data["warnings"].append("Spans address the reader displayText; original Qaloon source is retained separately. Simplified typography cannot establish every riwayah-specific rule.")
            ayahs.append({"ayah": ayah["ayah"], "source_text": ayah["text"], **data})
        surahs.append({"id": surah["id"], "ayahs": ayahs})
    if len(surahs) != 114:
        raise ValueError("Cache all 114 Qaloon surahs before generating the whole-Quran JSON")
    basmalah = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    payload = {"schema_version": 2, "version": VERSION, "riwayah": "Qaloon an Nafi", "tariq": "Shatibiyyah-options-performance-choice-unselected",
               "status": "machine-candidates-not-certified-mushaf", "approved_ayahs": 0,
               "duration_unit": "harakat-not-seconds", "offset_unit": "utf16", "source_sha256": digest.hexdigest(),
               "context": "wasl-within-ayah-waqf-at-end", "rules": RULES, "legend": LEGEND,
               "basmalah": {"ayah": 0, **annotate(basmalah, source_text="بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")},
               "sources": [{"url": "https://www.scribd.com/document/652847888/Qaloon-Tajweed-Rules", "access": "client-challenge-not-read"},
                           {"url": "https://archive.org/details/UsulRewayatQalun", "title": "The Secure Way to Rewayat Qalun", "pages": "8–14, 18–22", "access": "public-secondary-reference-read"},
                           {"source": "user-supplied mushaf image", "use": "palette/categories only; printed lengths are not copied as Qaloon rulings"}],
               "counts": dict(counts), "surahs": surahs}
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
