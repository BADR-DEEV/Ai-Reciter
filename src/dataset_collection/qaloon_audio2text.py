import os
import re
import json
import time
import base64
import unicodedata
import requests

INPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "QaloonData_v10(1).json")
# OUT_DIR = "dataset_qaloon_final"
# AUDIO_DIR = os.path.join(OUT_DIR, "audio")
# METADATA = os.path.join(OUT_DIR, "metadata.jsonl")
# MISSING = os.path.join(OUT_DIR, "missing_audio.json")

# os.makedirs(AUDIO_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nquran.com/ar/quranplayer/",
}

RECITERS = [
    "moshaf/qaloon/waleed_allebi",
    "moshaf/qaloon/Husary",
    "moshaf/qaloon/Menshawi",
    "moshaf/qaloon/Huthaify",
]

ARABIC = set("ءاأإآؤئبتثجحخدذرزسشصضطظعغفقكلمنهويىة")


# ------------------------------------------------------------------
# TEXT NORMALIZATION
# ------------------------------------------------------------------

def strip_harakat(text):
    return re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)


def fix_combining_hamza(text):
    """
    Qur'anic orthography frequently represents hamza as a combining
    character. Convert it BEFORE stripping harakat.

    Examples:
        قُرِۓَ    -> قرئ
        سَنُقْرِئُكَ -> سنقرئك
        يَسْـَٔلُونَكَ -> يسألونك
    """

    # ي / ے + combining hamza above -> ئ
    text = re.sub(r"[يىے]\u0654", "ئ", text)

    # waw + combining hamza
    text = text.replace("ؤ", "ؤ")

    # alef + hamza
    text = text.replace("أ", "أ")
    text = text.replace("إ", "إ")

    # tatweel carrying hamza
    text = re.sub(r"ـ[\u064B-\u0652]*ٔ", "أ", text)
    text = re.sub(r"ـ[\u064B-\u0652]*ٕ", "إ", text)

    return text


def normalize_word(word):
    # Alef maqsura + dagger alef: موسىٰ -> موسى
    word = word.replace("ىٰ", "ى")

    # Qalun final-aa pattern:
    # بنيٰها -> بناها
    # مرسيٰها -> مرساها
    word = re.sub(
        r"يٰ(?=(?:ها|هما|هم|هن|ه|كما|كم|كن|ك|نا)$)",
        "ا",
        word,
    )

    # Elsewhere يٰ represents يا
    word = word.replace("يٰ", "يا")

    # Remaining dagger alef normally materializes as alif
    word = word.replace("ٰ", "ا")

    return word


def normalize_uthmani(text):
    text = str(text)

    text = re.sub(
        r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00A0]",
        " ",
        text,
    )

    text = re.sub(r"[\u0660-\u0669\d]+", "", text)

    text = fix_combining_hamza(text)

    # Base character normalization
    text = (
        text.replace("ٱ", "ا")
            .replace("ے", "ي")
            .replace("ی", "ي")
            .replace("ک", "ك")
    )

    # Pronunciation annotations: REMOVE, don't materialize.
    text = (
        text.replace("ۥ", "")
            .replace("ۦ", "")
            .replace("ۨ", "")
            .replace("۬", "")
    )

    # Quranic stop symbols
    text = re.sub(r"[\u06D6-\u06ED\u06EE-\u06EF]", "", text)

    # Remaining tatweel
    text = text.replace("ـ", "")

    text = strip_harakat(text)

    words = [normalize_word(w) for w in text.split()]
    text = " ".join(words)

    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_asr(text):
    return (
        text.replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ؤ", "و")
            .replace("ئ", "ي")
    )


def normalize_quran_for_asr_v1(text: str) -> str:
    """Normalize Qālūn Uthmani spelling for unvowelled, MSA-style ASR labels.

    Resolve dagger-alif seats before removing the dagger; preserve medial
    hamzas and word-final alif maqṣūra rather than phoneticizing the spelling.
    """
    text = str(text)
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00A0]", " ", text)
    text = re.sub(r"[\u0660-\u0669\u06F0-\u06F9\d]", "", text)
    text = fix_combining_hamza(text)
    text = text.replace("ے", "ي").replace("ی", "ي").replace("ک", "ك")
    text = re.sub(r"[\u06D6-\u06ED\u06EE-\u06EF\u08F0-\u08F3]", "", text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0674]", "", text)
    text = text.replace("ـ", "")

    # Suffixal yaa-seat: أتيٰك -> أتاك, but موسىٰ -> موسى.
    text = re.sub(r"[يى]\u0670(?=[\u0621-\u064A])", "ا", text)
    text = re.sub(r"ى\u0670(?![\u0621-\u064A])", "ى", text)
    # The waw is an orthographic seat, not a pronounced consonant.
    text = re.sub(r"و\u0670(?=ة)", "ا", text)

    # Closed-list exceptions to productive dagger-alif expansion. Correct
    # both annotated and accidentally expanded spellings at word boundaries.
    exceptions = {
        "الرحمان": "الرحمن", "هاذا": "هذا", "هاذه": "هذه",
        "هاؤلاء": "هؤلاء", "هاذان": "هذان", "ذالك": "ذلك",
        "ذالكم": "ذلكم", "لاكن": "لكن", "إلاه": "إله",
        "اللاه": "الله", "طاها": "طه", "ياس": "يس",
    }

    def normalize_word(match):
        word = match.group()
        expanded = word.replace("\u0670", "ا")
        return exceptions.get(expanded, expanded)

    text = re.sub(r"[\u0621-\u064A\u0670\u0671]+", normalize_word, text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"[^\u0621-\u063A\u0641-\u064A\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NORMALIZER_VERSION = "qaloon-asr-v2-vocative-consonantal-yaa"


def normalize_quran_for_asr(text: str) -> str:
    """V2: distinguish consonantal yaa from a silent dagger-alif seat.

    Resolve while fatha is still visible: قِيَٰمَة -> قيامة, but أَتَيٰكَ -> أتاك.
    Initial Uthmani vocatives are separate words even when graphically joined.
    Final alif/maqsura and medial hamza are NOT folded into fuzzy equivalence.
    V1 stays callable solely to audit frozen historical labels/results.
    """
    text = str(text)
    marks = r"[\u064B-\u065F]*"
    # A token-initial (optionally conjoined) vocalized vocative, not an arbitrary
    # medial yaa. Insert a real word boundary instead of deleting the vocative.
    text = re.sub(r"(?<![\u0621-\u0671\u06D6-\u06EF])([وف]" + marks + r")?يَ\u0670" + marks,
                  lambda m: (m.group(1) or "") + "يا ", text)
    # Yaa carrying fatha is pronounced; replacing it with alif destroys قیامة,
    # آيات, ديار and similar words. Consume the dagger before v1 seat resolution.
    text = re.sub(r"ي([\u064B-\u065F]*َ[\u064B-\u065F]*)\u0670", lambda m: "ي" + m.group(1) + "ا", text)
    # Uthmani initial standalone hamza+alif is equivalent to normalized آ, NOT
    # a rule deleting arbitrary medial/final hamza or final alif.
    text = re.sub(r"(?<![\u0621-\u064A\u064B-\u065F])ء" + marks + r"ا", "ا", text)
    normalized = normalize_quran_for_asr_v1(text)
    # Includes an initial hamza bearing a dagger itself (ءَٰا...) and is
    # idempotent. Preserve the resulting two alifs for interrogative أأ forms.
    return re.sub(r"(?<![\u0621-\u064A])ء(?=ا)", "", normalized)


def normalizer_for_version(version):
    if version in (None, "qaloon-asr-v1"):
        return normalize_quran_for_asr_v1
    if version == NORMALIZER_VERSION:
        return normalize_quran_for_asr
    raise ValueError(f"Unknown frozen normalizer version: {version}")


# Backward-compatible name used by existing dataset builders.
normalize_qaloon_for_asr = normalize_quran_for_asr


def normalize_with_harakat(text: str) -> str:
    """Clean Qālūn annotations without removing attested vowels/shaddah/madd.

    This is a *diacritized transcription*, not a tajweed-error label. Dagger
    alif is kept as written; do not invent pronunciation from silent text.
    """
    text = str(text)
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00A0]", " ", text)
    text = re.sub(r"[\u0660-\u0669\u06F0-\u06F9\d]", "", text)
    text = fix_combining_hamza(text)
    text = text.replace("ے", "ي").replace("ی", "ي").replace("ک", "ك").replace("ـ", "")
    text = re.sub(r"[\u06D6-\u06EF\u08F0-\u08F3]", "", text)
    text = re.sub(r"[^\u0621-\u063A\u0641-\u065F\u0670\u0671\s]", " ", text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _test_normalize_quran_for_asr():
    examples = {
        "هَلْ أَتَيٰكَ حَدِيثُ مُوسَىٰ": "هل اتاك حديث موسى",
        "إِذْ نَادَيٰهُ رَبُّهُۥ": "اذ ناداه ربه",
        "وَنَهَى ٱلنَّفْسَ عَنِ ٱلْهَوَىٰ": "ونهى النفس عن الهوى",
        "وَالْمَلَائِكَةُ صَفًّا": "والملائكة صفا",
        "اِ۬لْحَمْدُ لِلهِ رَبِّ اِ۬لْعَٰلَمِينَ": "الحمد لله رب العالمين",
        "اَ۬لرَّحْمَٰنِ اِ۬لرَّحِيمِ": "الرحمن الرحيم",
        "مَلِكِ يَوْمِ اِ۬لدِّينِ": "ملك يوم الدين",
        "اِ۬لذِے هُمْ فِيهِ مُخْتَلِفُونَ": "الذي هم فيه مختلفون",
        "وَأَقِيمُواْ اَ۬لصَّلَوٰةَ": "واقيموا الصلاة",
        "الزكوٰة الحيوٰة مشكوٰة غدوٰة منوٰة": "الزكاة الحياة مشكاة غداة مناة",
        "تَوَفَّيٰهُم وَمُؤْمِنٌ وَعِيسَىٰ": "توفاهم ومؤمن وعيسى",
    }
    for raw, expected in examples.items():
        actual = normalize_quran_for_asr(raw)
        assert actual == expected, (raw, actual, expected)


def validate(text, s, a):
    bad = {c for c in text if not c.isspace() and c not in ARABIC}

    if bad:
        raise ValueError(
            f"{s}:{a} invalid chars: "
            + str({c: f"U+{ord(c):04X}" for c in bad})
            + f"\n{text}"
        )

    for x in ("ـ", "ٰ", "ۥ", "ۦ", "ۨ", "ىا"):
        if x in text:
            raise ValueError(f"{s}:{a} suspicious '{x}': {text}")


# ------------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------------

def audio_url(reciter, surah, ayah):
    path = f"{reciter}/{surah:03d}/{ayah:03d}.mp3"
    raw = f's:{len(path)}:"{path}";|_*7H_'
    code = base64.b64encode(raw.encode()).decode()

    return f"https://www.nquran.com/globals/readaudio.php?mp3={code}"


def download_audio(surah, ayah, dest):
    for reciter in RECITERS:
        try:
            r = requests.get(
                audio_url(reciter, surah, ayah),
                headers=HEADERS,
                timeout=15,
            )

            if (
                r.status_code == 200
                and len(r.content) > 1000
                and "text" not in r.headers.get("Content-Type", "").lower()
            ):
                with open(dest, "wb") as f:
                    f.write(r.content)

                return reciter

        except requests.RequestException:
            pass

    return None


# ------------------------------------------------------------------
# LOAD QALUN TEXT
# ------------------------------------------------------------------

def load_quran():
    with open(INPUT_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    db = {}

    for row in raw:
        s = int(row["sura_no"])
        a = int(row["aya_no"])

        clean = normalize_uthmani(row["aya_text"])
        validate(clean, s, a)

        db[(s, a)] = {
            "text": clean,
            # Derive recognition labels from intact Uthmani evidence, NOT the
            # legacy simplified display text, which already discarded vowels.
            "text_asr": normalize_quran_for_asr(row["aya_text"]),
            "normalizer_version": NORMALIZER_VERSION,
            "raw": row["aya_text"],
        }

    # Fix incorrect ayah splits + renumber automatically
    db = apply_ayah_merges(db)

    return db


# ------------------------------------------------------------------
# AUDIO/TEXT ALIGNMENT
# ------------------------------------------------------------------

def aligned_text(db, surah, audio_ayah):
    # nQuran Fatiha audio #1 = Basmalah.
    # Our Qalun JSON begins with الحمد لله, so skip audio #1.
    if surah == 1:
        if audio_ayah == 1:
            return None

        # audio 2..6 -> textual 1..5
        if 2 <= audio_ayah <= 6:
            return db.get((1, audio_ayah - 1))

        # final audio contains the final two Qalun textual segments
        if audio_ayah == 7:
            x = db.get((1, 6))
            y = db.get((1, 7))

            if not x or not y:
                return None

            text = f"{x['text']} {y['text']}"

            return {
                "text": text,
                "text_asr": normalize_quran_for_asr(f"{x['raw']} {y['raw']}"),
                "normalizer_version": NORMALIZER_VERSION,
                "raw": f"{x['raw']} {y['raw']}",
            }

    return db.get((surah, audio_ayah))


def targets(db):
    # Fatiha: deliberately skip Basmalah audio
    yield from ((1, a) for a in range(2, 8))

    # Change to range(2, 115) when you're ready for the whole Quran.
    for surah in range(78, 115):
        ayahs = sorted(a for s, a in db if s == surah)

        for ayah in ayahs:
            yield surah, ayah


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------


AYAH_MERGES = {
    80: [(24, 25)],
    81: [(26, 27)],
    86: [(15, 16)],
    99: [(6, 7)],
}


def apply_ayah_merges(db):
    """
    Fix known incorrect ayah splits in the source JSON.

    Example:
        80:24 + 80:25 -> corrected 80:24

    Then:
        old 80:26 -> corrected 80:25
        old 80:27 -> corrected 80:26
        ...
        old 80:42 -> corrected 80:41
    """

    fixed = {}

    surahs = sorted({s for s, _ in db})

    for surah in surahs:
        ayahs = sorted(
            (a, db[(surah, a)])
            for s, a in db
            if s == surah
        )

        merges = {
            start: end
            for start, end in AYAH_MERGES.get(surah, [])
        }

        i = 0
        new_ayah = 1

        while i < len(ayahs):
            old_ayah, item = ayahs[i]

            # Merge current + next ayah
            if old_ayah in merges:
                end_ayah = merges[old_ayah]

                parts = []
                source_ayahs = []

                while (
                    i < len(ayahs)
                    and ayahs[i][0] <= end_ayah
                ):
                    src_ayah, part = ayahs[i]
                    parts.append(part)
                    source_ayahs.append(src_ayah)
                    i += 1

                text = " ".join(
                    p["text"] for p in parts
                )

                raw = " ".join(
                    p["raw"] for p in parts
                )

                fixed[(surah, new_ayah)] = {
                    "text": text,
                    "text_asr": normalize_quran_for_asr(raw),
                    "normalizer_version": NORMALIZER_VERSION,
                    "raw": raw,
                    "source_ayahs": source_ayahs,
                }

            else:
                fixed[(surah, new_ayah)] = {
                    **item,
                    "source_ayahs": [old_ayah],
                }

                i += 1

            new_ayah += 1

    return fixed



def main():
    from quran_geometry import geometry_fields
    db = load_quran()
    rows = []
    missing = []

    todo = list(targets(db))
    print(f"Processing {len(todo)} audio segments...")

    for surah, ayah in todo:
        data = aligned_text(db, surah, ayah)

        if not data:
            continue

        filename = f"{surah:03d}{ayah:03d}.mp3"
        dest = os.path.join(AUDIO_DIR, filename)

        reciter = None

        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            reciter = download_audio(surah, ayah, dest)

            if not reciter:
                missing.append({
                    "surah": surah,
                    "ayah": ayah,
                    "text": data["text"],
                })

                print(f"⚠️ missing {surah}:{ayah}")
                continue

            time.sleep(0.25)

        rows.append({
            "surah": surah,
            "ayah": ayah,
            "audio_filename": filename,
            "relative_audio_path": f"audio/{filename}",
            "text": data["text"],
            "text_asr_normalized": data["text_asr"],
            "normalizer_version": NORMALIZER_VERSION,
            "normalized_with_harakat": normalize_with_harakat(data["raw"]),
            "text_raw_uthmani": data["raw"],
            "reciter": reciter,
            **geometry_fields(surah, [ayah - 1] if surah == 1 and ayah < 7
                              else [6, 7] if surah == 1 else [ayah]),
        })

    # with open(METADATA, "w", encoding="utf-8") as f:
    #     for row in rows:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # with open(MISSING, "w", encoding="utf-8") as f:
    #     json.dump(missing, f, ensure_ascii=False, indent=2)

    print(
        f"Done: {len(rows)} samples | "
        f"{len(missing)} missing audio"
    )


if __name__ == "__main__":
    main()
