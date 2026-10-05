"""Write the reader's tajweed data: one small file per surah plus the rules.

    python -m src.tajweed.build            # web/public/quran/tajweed/

rules.json   rule registry, colour groups, riwāyah topics with occurrences
NNN.json     {"s", "ayahs": [{"a", "w": [[[text, [rule indexes]]...]...], "n": [notes]}]}
             Each word is a list of segments whose texts join to the word, so
             the reader colours letters without offsets or Arabic re-shaping.
"""
import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re

from .engine import annotate, segments
from .rules import GROUPS, PRIORITY, RULES, TAGS
from .text import ayah_words
from .topics import TOPICS

ROOT = Path(__file__).resolve().parents[2]
VERSION = "qaloon-tajweed-1.0"
ORDER = sorted(RULES, key=PRIORITY.__getitem__)
INDEX = {rule: i for i, rule in enumerate(ORDER)}
BASMALA = "بِسْمِ اِ۬للَّهِ اِ۬لرَّحْمَٰنِ اِ۬لرَّحِيمِ"
SOURCE = {"title_ar": "المنهج العلمي في أحكام التجويد وأصول رواية الإمام قالون",
          "publisher": "الهيئة العامة للأوقاف والشؤون الإسلامية، ليبيا", "edition": "2nd, 1443/2022",
          "isbn": "978-9959-58-025-2"}
TOPIC_LIMIT = 400
EXAMPLES = 3


def encode(result):
    words = [[[text, [INDEX[r] for r in rules]] if rules else [text] for text, rules in parts] for parts in segments(result)]
    notes = []
    for a in result.annotations:
        if a.note_en or a.wajh:
            note = [a.word, INDEX[a.rule], a.start, a.note_en, a.note_ar]
            if a.wajh:
                note.append(a.wajh)
            if note not in notes:
                notes.append(note)
    return words, notes


def load_surahs(quran):
    surahs = {}
    for path in sorted(glob.glob(str(Path(quran) / "surahs/*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        surahs[data["id"]] = data
    return surahs


def build(quran, out):
    surahs = load_surahs(quran)
    if len(surahs) != 114:
        raise SystemExit(f"Cache all 114 surahs first ({len(surahs)} found): python src/dataset_collection/cache_quran_pages.py")
    index = None
    hafs_path = Path(quran) / "hafs-reference.json"
    if hafs_path.is_file():
        from .hafs import HafsIndex
        index = HafsIndex(hafs_path, {s: [(a["ayah"], a["text"]) for a in d["ayahs"]] for s, d in surahs.items()})
    out.mkdir(parents=True, exist_ok=True)
    counts, digest = Counter(), hashlib.sha256()
    found = {t["id"]: [] for t in TOPICS}
    totals = Counter()
    examples = {rule: [] for rule in ORDER}
    for sid, data in surahs.items():
        ayahs = []
        for ayah in data["ayahs"]:
            n = len(ayah_words(ayah["text"]))
            hafs = [index.word(sid, ayah["ayah"], w) for w in range(n)] if index else None
            result = annotate(ayah["text"], sid, ayah["ayah"], hafs)
            words, notes = encode(result)
            entry = {"a": ayah["ayah"], "w": words}
            if notes:
                entry["n"] = notes
            ayahs.append(entry)
            counts.update(a.rule for a in result.annotations)
            for a in result.annotations:
                if len(examples[a.rule]) < EXAMPLES and all(e[:2] != [sid, ayah["ayah"]] for e in examples[a.rule]):
                    examples[a.rule].append([sid, ayah["ayah"], a.word, words[a.word]])
            for topic in TOPICS:
                for a in result.annotations:
                    pattern = topic.get("split", {}).get(a.rule)
                    if a.rule in topic["rules"] and (pattern is None or re.search(pattern, a.note_en)):
                        totals[topic["id"]] += 1
                        if len(found[topic["id"]]) < TOPIC_LIMIT:
                            found[topic["id"]].append([sid, ayah["ayah"], a.word, result.words[a.word][0], a.rule])
        payload = json.dumps({"v": VERSION, "s": sid, "ayahs": ayahs}, ensure_ascii=False, separators=(",", ":"))
        digest.update(payload.encode())
        (out / f"{sid:03d}.json").write_text(payload + "\n", encoding="utf-8")
    basmala_words, basmala_notes = encode(annotate(BASMALA))
    rules = {"v": VERSION, "riwayah": "Qālūn ʿan Nāfiʿ", "tariq": "al-Shāṭibiyyah", "source": SOURCE,
             "context": "waṣl inside the ayah, waqf at its end", "duration_unit": "harakat",
             "order": ORDER, "groups": GROUPS, "rules": RULES, "counts": dict(counts), "digest": digest.hexdigest(),
             "examples": examples, "tags": TAGS,
             "hafs_compared": index is not None, "basmala": {"w": basmala_words, "n": basmala_notes},
             "topics": [{**{k: v for k, v in t.items() if k != "split"}, "total": totals[t["id"]], "found": found[t["id"]]}
                        for t in TOPICS]}
    (out / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"{sum(len(d['ayahs']) for d in surahs.values())} ayahs, {sum(counts.values())} annotations, "
          f"{size / 1e6:.1f} MB in {out} (Ḥafṣ comparison: {'on' if index else 'off'})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quran", type=Path, default=ROOT / "web/public/quran")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    build(args.quran, args.out or args.quran / "tajweed")


if __name__ == "__main__":
    main()
