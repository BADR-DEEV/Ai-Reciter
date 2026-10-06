"""Find where in the Quran a short recitation comes from ("which ayah is this?").

The clip is transcribed blind (no expected text). Its words are then looked up in
the whole Qālūn text: rare words vote for candidate places, and an ordered word
alignment ranks them. Like the live matcher, this compares recognised text, not
sound; it says nothing about pronunciation or tajweed.
"""
from bisect import bisect_right
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
import json
import math
from pathlib import Path
import threading

import numpy as np

from .matcher import alignment, normalize_quran_for_asr, words

# First ayah of each juzʾ in this app's Qālūn (Madanī) numbering: the standard Ḥafṣ
# boundaries carried over through web/public/quran/text-mapping.json.
# web/lib/juz.ts mirrors this table for the search page's labels.
JUZ_STARTS = ((1, 1), (2, 141), (2, 251), (3, 93), (4, 24), (4, 147), (5, 84), (6, 112), (7, 87), (8, 41),
              (9, 94), (11, 6), (12, 53), (15, 1), (17, 1), (18, 74), (21, 1), (23, 1), (25, 21), (27, 58),
              (29, 46), (33, 31), (36, 27), (39, 31), (41, 46), (46, 1), (51, 31), (58, 1), (67, 1), (78, 1))
# Recited before many passages but part of none (the basmalah is not a numbered Qālūn ayah).
OPENINGS = (("taawwudh", tuple(words("أعوذ بالله من الشيطان الرجيم"))),
            ("basmala", tuple(words("بسم الله الرحمن الرحيم"))))
MAX_WORDS = 90      # ~30 s of recitation; longer transcripts are cut, never searched slowly
BUCKET = 4          # words per diagonal bucket when voting
SEEDS = 24          # candidate places aligned exactly
RESCORE = 8         # best exact candidates re-aligned with near-miss credit
SKIP_HEARD, SKIP_TEXT, MISMATCH = .6, .4, .5
# Below this rarity-weighted share of the heard words, a place is noise: sharing قل and
# الله says little, one rare word such as وشاهد says a lot.
MIN_EVIDENCE = .45


def juz_of(surah, ayah):
    return bisect_right(JUZ_STARTS, (surah, ayah)) or 1


def strip_openings(heard):
    """Drop a leading taʿawwudh and basmalah. Returns (remaining words, names removed)."""
    removed = []
    for name, phrase in OPENINGS:
        # Room for additions such as «السميع العليم» inside the taʿawwudh.
        pairs = alignment(list(phrase), heard[:len(phrase) + 3])
        if len(pairs) >= 3 and pairs[0][1] <= 1:
            heard = heard[pairs[-1][1] + 1:]
            removed.append(name)
    return heard, removed


def _exact(a, b):
    return 1.0 if a == b else 0.0


@lru_cache(maxsize=65536)
def _near(a, b):
    """Full credit for the same word, partial for a likely mis-hearing (يعلمون/تعلمون)."""
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 3 or abs(len(a) - len(b)) > 2:
        return 0.0
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return round(.7 * ratio, 3) if ratio >= .8 else 0.0


def fit(heard, text, similar=_exact, allowed=None, near=None):
    """Align every heard word inside the best stretch of `text` (free ends in `text` only).

    Returns (score, [(heard index, text index, credit)]). `allowed[j]` False makes
    text word j unmatchable, which keeps results inside the chosen juzʾ. Equal
    scores (a wording repeated in `text`) end nearest to text index `near`."""
    n, m = len(heard), len(text)
    previous = [0.0] * (m + 1)
    moves = [None]
    for i in range(1, n + 1):
        word, row, move = heard[i - 1], [previous[0] - SKIP_HEARD] + [0.0] * m, [1] + [0] * m
        for j in range(1, m + 1):
            credit = similar(word, text[j - 1]) if allowed is None or allowed[j - 1] else 0.0
            best, step = previous[j - 1] + (credit or -MISMATCH), 0
            if previous[j] - SKIP_HEARD > best:
                best, step = previous[j] - SKIP_HEARD, 1
            if row[j - 1] - SKIP_TEXT > best:
                best, step = row[j - 1] - SKIP_TEXT, 2
            row[j], move[j] = best, step
        previous = row
        moves.append(move)
    end = max(range(m + 1), key=lambda j: (previous[j], -abs(j - near) if near is not None else 0))
    pairs, i, j = [], n, end
    while i > 0:
        step = moves[i][j]
        if step == 0:
            credit = similar(heard[i - 1], text[j - 1]) if allowed is None or allowed[j - 1] else 0.0
            if credit:
                pairs.append((i - 1, j - 1, credit))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return previous[end], pairs[::-1]


def confidence(coverage, matched):
    if coverage >= .75 and matched >= 3:
        return "high"
    if coverage >= .5 and matched >= 2:
        return "medium"
    return "low"


class QuranIndex:
    """Every word of the Qālūn text in reading order, with an inverted index."""

    def __init__(self, surahs):
        self.ayahs, self.surahs, tokens, owner = [], {}, [], []
        for surah in sorted(surahs, key=lambda s: s["id"]):
            self.surahs[surah["id"]] = {"name": surah.get("name", ""), "arabic": surah.get("arabic", ""),
                                        "trained": bool(surah.get("trained"))}
            for row in surah["ayahs"]:
                # Same matching text as the live tracker: re-derived from the source spelling.
                found = words(normalize_quran_for_asr(row["text"]) if row.get("text") else row["normalized"])
                self.ayahs.append({"surah": surah["id"], "ayah": row["ayah"], "start": len(tokens),
                                   "juz": juz_of(surah["id"], row["ayah"]), "text": row.get("text") or row["normalized"],
                                   "displayText": row.get("displayText") or row.get("text") or row["normalized"]})
                owner.extend([len(self.ayahs) - 1] * len(found))
                tokens.extend(found)
        self.tokens = tokens
        self.owner = np.array(owner, dtype=np.int32)
        self.word_juz = np.array([self.ayahs[row]["juz"] for row in owner], dtype=np.int8)
        spots = defaultdict(list)
        for position, word in enumerate(tokens):
            spots[word].append(position)
        self.postings = {word: np.array(found, dtype=np.int32) for word, found in spots.items()}
        self.idf = {word: math.log(1 + len(tokens) / len(found)) for word, found in spots.items()}
        self.unseen_idf = math.log(1 + len(tokens))  # a heard word that is nowhere in the text

    def _seeds(self, heard, allowed):
        """Start positions where many rare heard words line up on one diagonal."""
        n, offsets, weights, query = len(heard), [], [], []
        for j, word in enumerate(heard):
            found = self.postings.get(word)
            if found is None:
                continue
            if allowed is not None:
                found = found[allowed[found]]
            if not len(found):
                continue
            offsets.append(found - j + n)
            weights.append(np.full(len(found), self.idf[word]))
            query.append(np.full(len(found), j))
        if not offsets:
            return []
        bucket = np.concatenate(offsets) // BUCKET
        weights, query = np.concatenate(weights), np.concatenate(query)
        _, first = np.unique(bucket * n + query, return_index=True)  # one vote per heard word per place
        raw = np.bincount(bucket[first], weights=weights[first])
        after, before = np.r_[raw[1:], 0], np.r_[0, raw[:-1]]
        # Rank by the neighbourhood (a place may straddle two buckets), but seed only
        # at raw peaks: two nearby repeats must not merge into one seed between them.
        votes = np.where((raw > 0) & (raw >= after) & (raw >= before), raw + after + before, 0)
        reach, seeds = 1 + n // (4 * BUCKET), []
        for b in np.argsort(-votes, kind="stable"):
            if votes[b] <= 0 or len(seeds) == SEEDS:
                break
            if all(abs(int(b) - s) > reach for s in seeds):
                seeds.append(int(b))
        return [b * BUCKET - n for b in seeds]

    def _passage(self, score, pairs, lo, heard):
        n = len(heard)
        matched = {lo + j for _, j, _ in pairs}
        first, last = min(matched), max(matched)
        coverage = sum(credit for *_, credit in pairs) / n
        weights = [self.idf.get(word, self.unseen_idf) for word in heard]
        evidence = sum(credit * weights[i] for i, _, credit in pairs) / sum(weights)
        ayahs = []
        for row in range(self.owner[first], self.owner[last] + 1):
            ayah = self.ayahs[row]
            end = self.ayahs[row + 1]["start"] if row + 1 < len(self.ayahs) else len(self.tokens)
            ayahs.append({key: ayah[key] for key in ("surah", "ayah", "juz", "text", "displayText")} | {
                "words": [{"index": k, "text": self.tokens[p], "status": "correct" if p in matched else "pending"}
                          for k, p in enumerate(range(ayah["start"], end))]})
        start, stop = ayahs[0], ayahs[-1]
        # Ties between identical wordings favour a whole ayah over the tail of a longer one.
        fill = len(matched) / sum(len(a["words"]) for a in ayahs)
        return {"surah": start["surah"], **self.surahs[start["surah"]],
                "start": {"surah": start["surah"], "ayah": start["ayah"]}, "end": {"surah": stop["surah"], "ayah": stop["ayah"]},
                "juz": sorted({a["juz"] for a in ayahs}), "trained": all(self.surahs[a["surah"]]["trained"] for a in ayahs),
                "score": round(score, 3), "coverage": round(coverage, 3), "matched_words": len(pairs),
                "confidence": confidence(min(coverage, evidence), len(pairs)), "ayahs": ayahs,
                "_span": (first, last), "_rows": (int(self.owner[first]), int(self.owner[last])), "_fill": fill, "_evidence": evidence}

    def search(self, heard, juz=(), limit=3):
        """Up to `limit` distinct passages for these heard words, best first; `juz` limits the scope."""
        heard = list(heard)[:MAX_WORDS]
        if not heard:
            return []
        allowed = np.isin(self.word_juz, sorted(juz)) if juz else None
        n, size, pad = len(heard), len(self.tokens), 6 + len(heard) // 3
        candidates = []
        for start in self._seeds(heard, allowed):
            lo, hi = max(0, start - pad), min(size, start + n + BUCKET + pad)
            mask = None if allowed is None else allowed[lo:hi].tolist()
            score, pairs = fit(heard, self.tokens[lo:hi], _exact, mask, start + n - lo)
            if pairs:
                # Equal scores: the place matching rarer words goes first (وشاهد over وما).
                rarity = sum(self.idf[heard[i]] for i, _, _ in pairs)
                candidates.append((score, rarity, lo, hi, mask, start + n - lo))
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
        passages = []
        for _, _, lo, hi, mask, near in candidates[:RESCORE]:
            score, pairs = fit(heard, self.tokens[lo:hi], _near, mask, near)
            if pairs:
                passage = self._passage(score, pairs, lo, heard)
                if passage["_evidence"] >= MIN_EVIDENCE:
                    passages.append(passage)
        passages.sort(key=lambda p: (-p["score"], -p["coverage"], -p["_evidence"], -p["_fill"], p["_span"][0]))
        def apart(a, b):  # distinct places never share an ayah
            return a["_rows"][1] < b["_rows"][0] or a["_rows"][0] > b["_rows"][1]
        chosen = []
        for passage in passages:
            if all(apart(passage, other) for other in chosen):
                chosen.append(passage)
            if len(chosen) == limit:
                break
        for passage in chosen:
            # A short ayah heard whole (الله الصمد) is certain when nothing else comes close.
            rival = max((p["score"] for p in passages if apart(p, passage)), default=0)
            if passage["confidence"] == "medium" and passage["coverage"] >= .75 and passage["_fill"] >= .99 and passage["score"] - rival >= 1:
                passage["confidence"] = "high"
        return [{key: value for key, value in p.items() if not key.startswith("_")} | {"rank": i + 1} for i, p in enumerate(chosen)]


_built = {"key": None, "index": None}
_building = threading.Lock()


def load_index(folder):
    """The index of every cached surah file, rebuilt only when that set of files changes."""
    folder = Path(folder)
    files = sorted(folder.glob("[0-9][0-9][0-9].json"))
    key = (str(folder), len(files), max((f.stat().st_mtime_ns for f in files), default=0))
    with _building:
        if _built["key"] != key:
            if not files:
                raise FileNotFoundError(f"No cached surahs in {folder}")
            _built["index"] = QuranIndex([json.loads(f.read_text(encoding="utf-8")) for f in files])
            _built["key"] = key
        return _built["index"]
