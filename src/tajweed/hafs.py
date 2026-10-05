"""Align Qālūn words to the Ḥafṣ reference to find riwāyah differences.

Many learners already read Ḥafṣ, so the places where Qālūn differs are the
ones worth flagging. The alignment is per surah on letter skeletons (ayah
numbering differs between the Madanī and Kūfī counts); each aligned pair is
then compared on letters and vowels.
"""
import ast
from difflib import SequenceMatcher
import json
from pathlib import Path

from .text import DAGGER, IQLAB_MEEM, LETTERS, is_prefix, SMALL_HIGH_YEH, SMALL_WAW, SMALL_YEH, TATWEEL, parse_word

HAFS_MARKS = {"ۡ": "ْ"}  # KFGQPC Ḥafṣ draws sukūn as a dotless khāʾ head
COARSE = {"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ے": "ي", "ۓ": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": ""}


def coarse(word):
    """Letter skeleton; a dagger alif counts as alif so قٰل = قال."""
    out = []
    for c in word.replace(TATWEEL + DAGGER, "ا"):
        if c in LETTERS:
            out.append(COARSE.get(c, c))
        elif c == DAGGER:
            out.append("ا")
        elif c == SMALL_HIGH_YEH:
            out.append("ي")
    return "".join(out)


def hafs_words(text):
    words = []
    for token in text.split():
        if any(c in LETTERS for c in token):
            words.append("".join(HAFS_MARKS.get(c, c) for c in token))
    return words


def load_hafs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for surah in data:
        verses = surah["verses"]
        verses = ast.literal_eval(verses) if isinstance(verses, str) else verses
        out[int(surah["id"])] = [(int(v["id"]), v["text"]) for v in verses]
    return out


def signature(word):
    """(letter, vowel, long) per letter, ignoring spelling-only differences:
    waṣl alifs, alif drawn full or as a dagger, maddah signs, sukūn style,
    shadda, the doubled article lām, and tanwīn drawn on the seat alif."""
    sig = []
    clusters = parse_word(word, 0)
    for i, c in enumerate(clusters):
        prev = clusters[i - 1] if i else None
        nxt = clusters[i + 1] if i + 1 < len(clusters) else None
        if c.base in "اى" and (c.tanwin or IQLAB_MEEM in c.marks) and sig:
            sig[-1] = (sig[-1][0], c.vowel, sig[-1][2])
            continue
        if c.base in "اٱى" and not c.vowel and DAGGER not in c.marks:
            wasl = c.base == "ٱ" or i == 0 or (nxt is not None and (not nxt.vowel or nxt.shadda) and is_prefix(clusters[:i]))
            seat = prev is not None and (prev.tanwin or IQLAB_MEEM in prev.marks)
            if not (wasl or seat) and sig and sig[-1][1] == "a":
                sig[-1] = (sig[-1][0], "a", "a")
            continue
        if i == 0 and c.base == "ا":
            continue
        if c.letter in "وي" and not (c.vowel or c.sukun or c.shadda) and sig and sig[-1][1] == {"و": "u", "ي": "i"}[c.letter] \
                and not sig[-1][2]:
            sig[-1] = (sig[-1][0], sig[-1][1], sig[-1][1])  # a full long vowel letter = the small one
            continue
        if c.hamza and c.base in "ئيے" and c.sukun and nxt is not None and nxt.base == "ا" and nxt.tanwin:
            sig.append(("ي", "", ""))  # Maghribi شَئْاً = شَيْـًٔا
            sig.append(("ء", nxt.vowel, ""))
            break
        letter = "ء" if c.hamza else (COARSE.get(c.base, c.base) or "ء")
        vowel = c.vowel
        after_wasl = prev is not None and (prev.base == "ٱ" or (prev.base == "ا" and (
            prev.index == 0 or (not prev.vowel and is_prefix(clusters[:prev.index])))))
        if letter == "ل" and not vowel and not c.sukun and (after_wasl or (sig and sig[-1] == ("ل", "i", ""))):
            vowel = "" if nxt is not None and nxt.shadda else "a"  # اَ۬لذِينَ: one written lām for the doubled lām
        long = "a" if DAGGER in c.marks else "u" if SMALL_WAW in c.marks else "i" if (SMALL_YEH in c.marks or SMALL_HIGH_YEH in c.marks) else ""
        sig.append((letter, vowel, long))
    return sig


def align_surah(qaloon_words, hafs_word_list):
    """qaloon_words / hafs_word_list: lists of words in reading order.
    Returns {qaloon_index: hafs_index} for 1:1 aligned words."""
    a = [coarse(w) for w in qaloon_words]
    b = [coarse(w) for w in hafs_word_list]
    pairs = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("equal", "replace") and i2 - i1 == j2 - j1:
            pairs.update({i1 + k: j1 + k for k in range(i2 - i1)})
    return pairs


class HafsIndex:
    """Lookup of the aligned Ḥafṣ word for each (surah, ayah, word index)."""

    def __init__(self, hafs_path, qaloon_surahs):
        hafs = load_hafs(hafs_path)
        self.pairs = {}
        for surah, ayahs in qaloon_surahs.items():
            from .text import ayah_words
            q_ids, q_words = [], []
            for ayah, text in ayahs:
                for w, (display, _) in enumerate(ayah_words(text)):
                    q_ids.append((ayah, w))
                    q_words.append(display)
            h_words = [w for _, text in hafs.get(surah, []) for w in hafs_words(text)]
            for qi, hi in align_surah(q_words, h_words).items():
                self.pairs[(surah, *q_ids[qi])] = h_words[hi]

    def word(self, surah, ayah, w):
        return self.pairs.get((surah, ayah, w))


SPELLING_ONLY = {"الن", "فالن"}  # Maghribi اُ۬ءَلْٰنَ writes the hamza before the lām: same reading


def classify(display, clusters, hafs_word, next_clusters):
    """Return (rule, cluster_index_or_None, note) for a Qālūn/Ḥafṣ difference,
    or None when the two read alike."""
    if coarse(display) == coarse(hafs_word) and (signature(display) == signature(hafs_word)
                                                 or coarse(display) in SPELLING_ONLY):
        return None
    last = clusters[-1]
    h_last = parse_word(hafs_word, 0)[-1]
    if h_last.hamza and not last.hamza and last.letter == "ا" and next_clusters and next_clusters[0].hamza \
            and next_clusters[0].vowel == "a":
        return "isqat", last.index, ""
    if last.letter != "ه" and SMALL_YEH in last.marks and SMALL_YEH not in h_last.marks:
        return "ya_zaida", last.index, ""
    if last.letter == "ي" and h_last.letter == "ي" and (last.vowel == "a") != (h_last.vowel == "a"):
        return "ya_idafa", last.index, ""
    return "riwaya_diff", None, hafs_word
