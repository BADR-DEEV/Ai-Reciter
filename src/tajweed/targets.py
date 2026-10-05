"""Targets for the tajweed model: the plain model's words plus tag tokens.

The plain model writes unvowelled words (`text_asr_normalized`). The tajweed
model writes the same words, each followed by one `<tj:RULE>` token per audible
rule on it (ghunna, ikhfāʾ, qalqala, madd lāzim/muttaṣil, tas-hīl, imāla...),
so stripping the tags gives exactly the plain target. Only rules whose
realisation can be heard get a tag; optional ways (munfaṣil length, mīm
al-jamʿ ṣilah) and spelling-only notes do not.

    python -m src.tajweed.targets --quran web/public/quran \
        --reader-root data/hf/qaloon-reciter-dataset --out data/tajweed/labels.jsonl
"""
import argparse
import glob
import json
from pathlib import Path

from src.dataset_collection.qaloon_audio2text import normalize_quran_for_asr

from .engine import annotate
from .rules import MODEL_TAGS, PRIORITY, RULES

ROOT = Path(__file__).resolve().parents[2]
TOKENS_FILE = Path(__file__).with_name("model_tokens.txt")


def token(tag):
    return f"<tj:{tag}>"


def tagged_words(text, surah=None, ayah=None, hafs=None):
    """[(source_word_index, normalized_word, [tags])]. A source word can
    normalize to two words (Uthmani vocative يٰأيها → يا ايها); tags go on the
    last of them."""
    result = annotate(text, surah, ayah, hafs)
    out = []
    for w, (display, _) in enumerate(result.words):
        words = normalize_quran_for_asr(display).split()
        if not words:
            continue
        found = sorted((a for a in result.annotations if a.word == w and RULES[a.rule]["tag"]),
                       key=lambda a: (a.start, PRIORITY[a.rule]))
        tags = list(dict.fromkeys(RULES[a.rule]["tag"] for a in found))
        out.extend((w, word, tags if i == len(words) - 1 else []) for i, word in enumerate(words))
    return out


def tagged_target(text, surah=None, ayah=None, hafs=None):
    return " ".join(word + "".join(map(token, tags)) for _, word, tags in tagged_words(text, surah, ayah, hafs))


def write_tokens(path=TOKENS_FILE):
    lines = ["# Tajweed tag tokens added to the Whisper tokenizer (src/tajweed/targets.py).", *map(token, MODEL_TAGS)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hafs_index(quran):
    hafs_path = Path(quran) / "hafs-reference.json"
    if not hafs_path.is_file():
        return None
    from .hafs import HafsIndex
    surahs = {}
    for path in sorted(glob.glob(str(Path(quran) / "surahs/*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        surahs[data["id"]] = [(a["ayah"], a["text"]) for a in data["ayahs"]]
    return HafsIndex(hafs_path, surahs), surahs


def hafs_for(index, surah, ayah, count):
    return [index.word(surah, ayah, w) for w in range(count)] if index else None


def build_labels(quran, reader_roots, out):
    """One line per ayah (covers every reader) and one per clip whose text
    spans several ayahs or differs from the cached ayah."""
    loaded = _hafs_index(quran)
    index, surahs = loaded if loaded else (None, {})
    from .text import ayah_words
    rows, by_ayah = [], {}
    for surah, ayahs in surahs.items():
        for ayah, text in ayahs:
            target = tagged_target(text, surah, ayah, hafs_for(index, surah, ayah, len(ayah_words(text))))
            by_ayah[(surah, ayah)] = (text, target)
            rows.append({"surah": surah, "ayah": ayah, "text": target})
    clips = 0
    for root in reader_roots:
        for meta in sorted(Path(root).glob("dataset_qaloon_*/metadata.jsonl")):
            reciter = meta.parent.name.removeprefix("dataset_qaloon_").lower()
            for line in meta.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                raw = row.get("text_raw_uthmani")
                cached = by_ayah.get((row["surah"], row["ayah"]))
                if not raw or (cached and len(row.get("source_ayahs") or [1]) == 1
                               and normalize_quran_for_asr(raw) == normalize_quran_for_asr(cached[0])):
                    continue
                key = row.get("reciter_key") or reciter
                rows.append({"reciter_key": key, "relative_audio_path": row["relative_audio_path"],
                             "text": tagged_target(raw, row["surah"], row["ayah"])})
                clips += 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(by_ayah)} ayah targets, {clips} clip-specific targets → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quran", type=Path, default=ROOT / "web/public/quran")
    parser.add_argument("--reader-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "data/tajweed/labels.jsonl")
    args = parser.parse_args()
    write_tokens()
    build_labels(args.quran, args.reader_root, args.out)
