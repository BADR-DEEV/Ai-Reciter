"""Compare the tajweed model's tags with the rules expected for an ayah.

The server calls `compare` with the ayah's Qālūn text and the words/tags the
tajweed model heard for it. Only words that were recognised are judged (a
missing word is a reading error, reported by the matcher), and only audible
rules are expected (see targets.py).
"""
from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path

from .rules import TAGS
from .targets import tagged_words
from .text import ayah_words

QURAN = Path(__file__).resolve().parents[2] / "web/public/quran"

@lru_cache(maxsize=1)
def _hafs_index():
    """Ḥafṣ alignment for isqāṭ and the yāʾāt, built once (~1 s) like the training labels."""
    if not (QURAN / "hafs-reference.json").is_file():
        return None
    from .hafs import HafsIndex
    surahs = {}
    for path in sorted((QURAN / "surahs").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        surahs[data["id"]] = [(a["ayah"], a["text"]) for a in data["ayahs"]]
    return HafsIndex(QURAN / "hafs-reference.json", surahs)


def compare(target_text, hyp_words, hyp_tags, surah=None, ayah=None):
    index = _hafs_index() if surah and ayah else None
    hafs = [index.word(surah, ayah, w) for w in range(len(ayah_words(target_text)))] if index else None
    expected = tagged_words(target_text, surah, ayah, hafs)
    if not expected:
        return None
    matcher = SequenceMatcher(None, [w for _, w, _ in expected], list(hyp_words), autojunk=False)
    words, per_rule = [], {}
    for op, i1, i2, j1, _ in matcher.get_opcodes():
        if op != "equal":
            continue
        for k in range(i2 - i1):
            source, text, wanted = expected[i1 + k]
            heard = set(hyp_tags[j1 + k]) if j1 + k < len(hyp_tags) else set()
            if not wanted and not heard:
                continue
            for tag in wanted:
                counts = per_rule.setdefault(tag, [0, 0])
                counts[0] += 1
                counts[1] += tag in heard
            words.append({"word": source, "text": text, "expected": wanted,
                          "applied": [t for t in wanted if t in heard],
                          "missed": [t for t in wanted if t not in heard],
                          "extra": sorted(heard - set(wanted))})
    total = sum(e for e, _ in per_rule.values())
    applied = sum(a for _, a in per_rule.values())
    names = {tag: {k: TAGS[tag][k] for k in ("en", "ar", "fix_en", "fix_ar")} for tag in per_rule if tag in TAGS}
    mistakes = [{"word": w["word"], "text": w["text"], "tag": tag, **names.get(tag, {})} for w in words for tag in w["missed"]]
    return {"words": words, "mistakes": mistakes, "expected": total, "applied": applied,
            "score": round(applied / total, 3) if total else None,
            "rules": {tag: {**names.get(tag, {}), "expected": e, "applied": a} for tag, (e, a) in sorted(per_rule.items())}}
