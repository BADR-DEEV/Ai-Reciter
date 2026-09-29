import os
import io
import re
import json
import base64
import shutil
import unicodedata
import requests
from qaloon_audio2text import normalize_with_harakat
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ------------------------------------------------------------------
# FFMPEG SETUP
# ------------------------------------------------------------------
if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    raise RuntimeError("❌ ffmpeg/ffprobe missing. Run: pip install static-ffmpeg pydub")

from pydub import AudioSegment
AudioSegment.converter = shutil.which("ffmpeg")

# ------------------------------------------------------------------
# CONFIG & PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(BASE_DIR, "QaloonData_v10(1).json")

OUT_DIR = os.path.join(BASE_DIR, "dataset_qaloon_final")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
METADATA = os.path.join(OUT_DIR, "metadata.jsonl")
MISSING = os.path.join(OUT_DIR, "missing_audio.json")

os.makedirs(AUDIO_DIR, exist_ok=True)

RECITERS = [
    "moshaf/qaloon/waleed_allebi",
    "moshaf/qaloon/Husary",
    "moshaf/qaloon/Menshawi",
    "moshaf/qaloon/Huthaify",
]

ARABIC = set("ءاأإآؤئبتثجحخدذرزسشصضطظعغفقكلمنهويىة")

# Fast HTTP Session with Connection Pooling (No TLS reconnect overhead)
SESSION = requests.Session()
retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retries)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nquran.com/ar/quranplayer/",
})

# ------------------------------------------------------------------
# TEXT NORMALIZATION
# ------------------------------------------------------------------
def strip_harakat(text):
    return re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)

def fix_combining_hamza(text):
    text = re.sub(r"[يىے]\u0654", "ئ", text)
    text = text.replace("ؤ", "ؤ").replace("أ", "أ").replace("إ", "إ")
    text = re.sub(r"ـ[\u064B-\u0652]*ٔ", "أ", text)
    text = re.sub(r"ـ[\u064B-\u0652]*ٕ", "إ", text)
    return text

def normalize_word(word):
    word = word.replace("ىٰ", "ى")
    word = re.sub(r"يٰ(?=(?:ها|هما|هم|هن|ه|كما|كم|كن|ك|نا)$)", "ا", word)
    word = word.replace("يٰ", "يا").replace("ٰ", "ا")
    return word

def normalize_uthmani(text):
    text = str(text)
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00A0]", " ", text)
    text = re.sub(r"[\u0660-\u0669\d]+", "", text)
    text = fix_combining_hamza(text)
    text = text.replace("ٱ", "ا").replace("ے", "ي").replace("ی", "ي").replace("ک", "ك")
    text = text.replace("ۥ", "").replace("ۦ", "").replace("ۨ", "").replace("۬", "")
    text = re.sub(r"[\u06D6-\u06ED\u06EE-\u06EF]", "", text)
    text = text.replace("ـ", "")
    text = strip_harakat(text)
    words = [normalize_word(w) for w in text.split()]
    text = " ".join(words)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_quran_for_asr(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00A0]", " ", text)
    text = re.sub(r"[\u0660-\u0669\u06F0-\u06F9\d]", "", text)
    text = fix_combining_hamza(text)
    text = text.replace("ے", "ي").replace("ی", "ي").replace("ک", "ك")
    text = re.sub(r"[\u06D6-\u06ED\u06EE-\u06EF\u08F0-\u08F3]", "", text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0674]", "", text)
    text = text.replace("ـ", "")

    text = re.sub(r"[يى]\u0670(?=[\u0621-\u064A])", "ا", text)
    text = re.sub(r"ى\u0670(?![\u0621-\u064A])", "ى", text)
    text = re.sub(r"و\u0670(?=ة)", "ا", text)

    exceptions = {
        "الرحمان": "الرحمن", "هاذا": "هذا", "هاذه": "هذه",
        "هاؤلاء": "هؤلاء", "هاذان": "هذان", "ذالك": "ذلك",
        "ذالكم": "ذلكم", "لاكن": "لكن", "إلاه": "إله",
        "اللاه": "الله", "طاها": "طه", "ياس": "يس",
    }

    def normalize_word_token(match):
        word = match.group()
        expanded = word.replace("\u0670", "ا")
        return exceptions.get(expanded, expanded)

    text = re.sub(r"[\u0621-\u064A\u0670\u0671]+", normalize_word_token, text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"[^\u0621-\u063A\u0641-\u064A\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def validate(text, s, a):
    bad = {c for c in text if not c.isspace() and c not in ARABIC}
    if bad:
        raise ValueError(f"{s}:{a} invalid chars: {bad}\n{text}")

# ------------------------------------------------------------------
# FAST AUDIO WORKER (16kHz Mono WAV)
# ------------------------------------------------------------------
def audio_url(reciter, surah, ayah):
    path = f"{reciter}/{surah:03d}/{ayah:03d}.mp3"
    raw = f's:{len(path)}:"{path}";|_*7H_'
    code = base64.b64encode(raw.encode()).decode()
    return f"https://www.nquran.com/globals/readaudio.php?mp3={code}"

def download_and_convert(surah, ayah, wav_dest):
    for reciter in RECITERS:
        try:
            r = SESSION.get(audio_url(reciter, surah, ayah), timeout=8)
            if r.status_code == 200 and len(r.content) > 1000 and "text" not in r.headers.get("Content-Type", "").lower():
                # Direct in-memory decode & resample to 16kHz mono 16-bit
                audio = AudioSegment.from_file(io.BytesIO(r.content), format="mp3")
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                audio.export(wav_dest, format="wav")
                return reciter
        except Exception:
            continue
    return None

# ------------------------------------------------------------------
# AYAH MERGES & FIXED FATIHA ALIGNMENT
# ------------------------------------------------------------------
AYAH_MERGES = {
    80: [(24, 25)],
    81: [(26, 27)],
    86: [(15, 16)],
    99: [(6, 7)],
}

def apply_ayah_merges(db):
    fixed = {}
    surahs = sorted({s for s, _ in db})

    for surah in surahs:
        ayahs = sorted((a, db[(surah, a)]) for s, a in db if s == surah)
        merges = {start: end for start, end in AYAH_MERGES.get(surah, [])}

        i = 0
        new_ayah = 1
        while i < len(ayahs):
            old_ayah, item = ayahs[i]
            if old_ayah in merges:
                end_ayah = merges[old_ayah]
                parts, source_ayahs = [], []
                while i < len(ayahs) and ayahs[i][0] <= end_ayah:
                    src_ayah, part = ayahs[i]
                    parts.append(part)
                    source_ayahs.append(src_ayah)
                    i += 1
                text = " ".join(p["text"] for p in parts)
                raw = " ".join(p["raw"] for p in parts)
                fixed[(surah, new_ayah)] = {
                    "text": text,
                    "text_asr": normalize_quran_for_asr(raw),
                    "raw": raw,
                    "source_ayahs": source_ayahs,
                }
            else:
                fixed[(surah, new_ayah)] = {**item, "source_ayahs": [old_ayah]}
                i += 1
            new_ayah += 1
    return fixed

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
            "text_asr": normalize_quran_for_asr(row["aya_text"]),
            "raw": row["aya_text"],
        }
    return apply_ayah_merges(db)

def aligned_text(db, surah, audio_ayah):
    # FIXED: Handled 001001 cleanly instead of skipping it!
    if surah == 1:
        if audio_ayah == 1:
            raw = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
            return {
                "text": "بسم الله الرحمن الرحيم",
                "text_asr": "بسم الله الرحمن الرحيم",
                "raw": raw,
                "source_ayahs": [0]
            }
        if 2 <= audio_ayah <= 6:
            return db.get((1, audio_ayah - 1))
        if audio_ayah == 7:
            x, y = db.get((1, 6)), db.get((1, 7))
            if not x or not y:
                return None
            text = f"{x['text']} {y['text']}"
            raw = f"{x['raw']} {y['raw']}"
            return {
                "text": text,
                "text_asr": normalize_quran_for_asr(raw),
                "raw": raw,
                "source_ayahs": [6, 7]
            }
    return db.get((surah, audio_ayah))

def targets(db):
    # FIXED: Starts from 1 now! (Includes 001001.wav)
    yield from ((1, a) for a in range(1, 8))
    for surah in range(78, 115):
        ayahs = sorted(a for s, a in db if s == surah)
        for ayah in ayahs:
            yield surah, ayah

# ------------------------------------------------------------------
# MULTI-THREADED WORKER FUNCTION
# ------------------------------------------------------------------
def process_single_ayah(item, db):
    surah, ayah = item
    data = aligned_text(db, surah, ayah)
    if not data:
        return None

    wav_filename = f"{surah:03d}{ayah:03d}.wav"
    dest_wav = os.path.join(AUDIO_DIR, wav_filename)

    # Check cache
    if os.path.exists(dest_wav) and os.path.getsize(dest_wav) > 1000:
        reciter = "cached"
    else:
        reciter = download_and_convert(surah, ayah, dest_wav)
        if not reciter:
            return {"missing": True, "surah": surah, "ayah": ayah, "text": data["text"]}

    return {
        "missing": False,
        "record": {
            "surah": surah,
            "ayah": ayah,
            "audio_filename": wav_filename,
            "relative_audio_path": f"audio/{wav_filename}",
            "text": data["text"],
            "text_asr_normalized": data["text_asr"],
            "normalized_with_harakat": normalize_with_harakat(data["raw"]),
            "text_raw_uthmani": data["raw"],
            "source_ayahs": data.get("source_ayahs", [ayah]),
            "reciter": reciter,
        }
    }

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    print("📖 Loading Qalun text...")
    db = load_quran()
    todo = list(targets(db))
    print(f"🚀 Processing {len(todo)} clips using 10 Parallel Worker Threads...")

    rows = []
    missing = []
    completed = 0

    # 10 Concurrent Threads for blazing fast speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ayah, item, db): item for item in todo}

        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res:
                if res["missing"]:
                    missing.append(res)
                    print(f"  ⚠️ Missing audio: {res['surah']:03d}:{res['ayah']:03d}")
                else:
                    rows.append(res["record"])

            if completed % 50 == 0 or completed == len(todo):
                print(f"  ⚡ [{completed}/{len(todo)}] processed...")

    # Sort rows so metadata.jsonl stays in exact Quranic order
    rows.sort(key=lambda x: (x["surah"], x["ayah"]))

    # Write output
    with open(METADATA, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(MISSING, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"🎉 COMPLETED IN RECORD TIME!")
    print(f"Saved to: {METADATA}")
    print(f"Total Clips: {len(rows)} (16kHz Mono WAV)")
    print(f"Missing: {len(missing)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
