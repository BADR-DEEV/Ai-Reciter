"""Compare the tajweed model's tags with the rules expected for an ayah.

The server calls `compare` with the ayah's Qālūn text and the words/tags the
tajweed model heard for it. Only words that were recognised are judged (a
missing word is a reading error, reported by the matcher), and only audible
rules are expected (see targets.py).
"""
from difflib import SequenceMatcher

from .rules import RULES
from .targets import tagged_words

TAG_RULE = {}
for rule_id, rule in RULES.items():
    if rule["tag"]:
        TAG_RULE.setdefault(rule["tag"], rule_id)


def compare(target_text, hyp_words, hyp_tags, surah=None, ayah=None):
    expected = tagged_words(target_text, surah, ayah)
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
    return {"words": words, "expected": total, "applied": applied, "score": round(applied / total, 3) if total else None,
            "rules": {tag: {"rule": TAG_RULE[tag], "en": RULES[TAG_RULE[tag]]["en"], "ar": RULES[TAG_RULE[tag]]["ar"],
                            "expected": e, "applied": a} for tag, (e, a) in sorted(per_rule.items())}}
