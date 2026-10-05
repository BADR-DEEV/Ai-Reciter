"""Parse KFGQPC-style Qālūn text into words of letter clusters.

A cluster is one base letter plus every mark drawn on it. Rules are applied to
clusters and spans never split a letter from its marks, so colouring cannot
detach a ḥaraka from its letter or break Arabic shaping inside a cluster.
"""
from dataclasses import dataclass, field
import re
import unicodedata

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SUKUN, SHADDA = "ْ", "ّ"
MADDAH, DAGGER = "ٓ", "ٰ"
HAMZA_MARKS = "ٕٔ"
# KFGQPC encodes "open" (staggered) tanwīn with look-alike code points; Unicode
# has dedicated ones that fonts such as Amiri draw correctly.
OPEN_TANWIN = {"ٗ": "ࣰ", "ٞ": "ࣱ", "ٖ": "ࣲ"}
TANWIN = {"ً": "a", "ࣰ": "a", "ٌ": "u", "ࣱ": "u", "ٍ": "i", "ࣲ": "i"}
SMALL_WAW, SMALL_YEH, SMALL_HIGH_YEH, SMALL_HIGH_NOON = "ۥ", "ۦ", "ۧ", "ۨ"
IQLAB_MEEM = "ۢ"
# Qālūn sila/tas-hīl dots: on a word-initial alif they mark hamzat al-waṣl,
# elsewhere they mark a changed hamza, imāla, ishmām or ikhtilās.
DOT_HIGH, DOT_LOW, ROUND_ZERO = "۬", "۪", "۟"
TATWEEL = "ـ"

HAMZA_LETTERS = set("ءأإؤئٱآۓ")
THROAT = set("ءهعحغخ")
ISTILA = set("خصضغطقظ")
QALQALA = set("قطبجد")
IKHFA = set("تثجدذزسشصضطظفقك")
SUN = set("تثدذرزسشصضطظلن")
LETTERS = set(chr(c) for c in range(0x0621, 0x064B) if c != 0x0640) | set("ٱےۓ")
MARK_EXTRA = set(f"{SMALL_WAW}{SMALL_YEH}{TATWEEL}۩")
NON_TEXT = re.compile(r"[٠-٩۰-۹\d ​-‏‪-‮⁦-⁩﻿]")

# Makhraj groups (Ibn al-Jazarī's 17 places) used to classify adjacent letters.
MAKHRAJ = {**dict.fromkeys("ءه", "deep_throat"), **dict.fromkeys("عح", "mid_throat"), **dict.fromkeys("غخ", "near_throat"),
           "ق": "deep_tongue", "ك": "deep_tongue_lower", **dict.fromkeys("جشي", "mid_tongue"), "ض": "tongue_side",
           "ل": "tongue_edge", "ن": "tongue_tip_n", "ر": "tongue_tip_r", **dict.fromkeys("طدت", "tongue_tip_ridge"),
           **dict.fromkeys("صسز", "tongue_tip_whistle"), **dict.fromkeys("ظذث", "tongue_tip_teeth"),
           "ف": "lip_teeth", **dict.fromkeys("بمو", "lips")}


def is_prefix(clusters):
    """Clusters form a proclitic chain (وَ فَ بِ كَ لِ تَ أَ) before an alif."""
    return 0 < len(clusters) <= 3 and all(
        c.letter in "وفبكلتء" and c.vowel in ("a", "i") and not (c.shadda and c.index) and not c.tanwin for c in clusters)


def is_mark(char):
    return char in MARK_EXTRA or unicodedata.category(char) == "Mn"


@dataclass
class Cluster:
    base: str          # first code point as written
    marks: str         # all following marks, original order
    word: int          # word index within the ayah (letter words only)
    index: int         # cluster index within the word
    text: str = ""     # display text (open tanwīn remapped)
    letter: str = ""   # logical letter: ے→ي, hamza seats→ء
    flags: dict = field(default_factory=dict)

    def has(self, chars):
        return any(c in self.marks for c in chars)

    @property
    def vowel(self):
        if self.has(FATHA) or self.tanwin == "a":
            return "a"
        if self.has(DAMMA) or self.tanwin == "u":
            return "u"
        if self.has(KASRA) or self.tanwin == "i":
            return "i"
        return ""

    @property
    def tanwin(self):
        return next((v for c, v in TANWIN.items() if c in self.marks), "")

    @property
    def staggered(self):
        return any(c in self.marks for c in "ࣰࣱࣲ")

    @property
    def sukun(self):
        return SUKUN in self.marks

    @property
    def shadda(self):
        return SHADDA in self.marks

    @property
    def bare(self):
        """No vowel, sukūn, shadda or tanwīn drawn on the letter."""
        return not (self.vowel or self.sukun or self.shadda)

    @property
    def hamza(self):
        return self.letter == "ء"


def _logical(base, marks):
    if base in HAMZA_LETTERS and base not in "ٱآ":
        return "ء"
    if base in "اوےيى" + TATWEEL and any(c in marks for c in HAMZA_MARKS):
        return "ء"
    if base == "آ":
        return "ا"
    return {"ے": "ي", "ٱ": "ا"}.get(base, base)


def parse_word(token, word_index):
    clusters = []
    for char in token:
        if char in LETTERS or (char == TATWEEL and not clusters):
            clusters.append(Cluster(char, "", word_index, len(clusters)))
        elif char == TATWEEL and clusters:
            # A tatweel carrying a hamza is the hamza's own seat (يَٰـَٔادَمُ).
            clusters.append(Cluster(char, "", word_index, len(clusters)))
        elif is_mark(char) and clusters:
            clusters[-1].marks += OPEN_TANWIN.get(char, char)
    merged = []
    for cluster in clusters:
        if cluster.base == TATWEEL and not any(c in cluster.marks for c in HAMZA_MARKS) and merged:
            merged[-1].marks += TATWEEL + cluster.marks
            continue
        cluster.index = len(merged)
        merged.append(cluster)
    for cluster in merged:
        if cluster.base == "آ":
            cluster.marks = MADDAH + cluster.marks
        cluster.letter = _logical(cluster.base, cluster.marks)
        cluster.text = (cluster.base if cluster.base != "آ" else "ا") + cluster.marks
    return merged


def ayah_words(text):
    """Letter words of an ayah: [(display_word, [Cluster])]. Numbers, ۞ and
    bidi controls are dropped so word indexes match the reader and matcher."""
    words = []
    for token in NON_TEXT.sub(" ", text).split():
        if not any(c in LETTERS for c in token):
            continue
        clusters = parse_word(token, len(words))
        words.append(("".join(c.text for c in clusters), clusters))
    return words


def display_text(text):
    return " ".join(word for word, _ in ayah_words(text))


def bare_letters(text):
    """Skeleton used for lexical matching: letters only, seats folded."""
    out = []
    for char in text:
        if char in LETTERS:
            out.append({"ٱ": "ا", "آ": "ا", "أ": "ا", "إ": "ا", "ے": "ي", "ۓ": "ي", "ئ": "ي", "ؤ": "و"}.get(char, char))
    return "".join(out)
