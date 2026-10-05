"""Annotate a Qālūn ayah with tajweed and riwāyah rules.

Context is waṣl inside the ayah and waqf at its end, matching how the app
plays and checks one ayah at a time. Cross-word rules look through silent
letters (hamzat al-waṣl, sun-letter lām) to the next pronounced letter.
"""
from dataclasses import dataclass, field
import re

from . import hafs as hafs_mod
from .rules import PRIORITY, RULES
from .text import (DAGGER, DOT_HIGH, DOT_LOW, IKHFA, IQLAB_MEEM, ISTILA, MADDAH, MAKHRAJ, QALQALA, ROUND_ZERO,
                   SMALL_HIGH_NOON, SMALL_HIGH_YEH, SMALL_WAW, SMALL_YEH, THROAT, ayah_words, bare_letters, is_prefix)

OPENING_SURAHS = {2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30, 31, 32, 36, 38, 40, 41, 42, 43, 44,
                  45, 46, 50, 68}
MUQATTAAT = {"الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس", "يس", "ص", "حم", "عسق", "ق", "ن"}
ALLAH = re.compile(r"^[وف]?[اء]?[بتك]?ا?لله(م)?$")
DOTS = DOT_HIGH + DOT_LOW + ROUND_ZERO


@dataclass
class Annotation:
    rule: str
    word: int
    start: int
    end: int
    note_en: str = ""
    note_ar: str = ""
    wajh: list = None

    def as_dict(self):
        out = {"rule": self.rule, "word": self.word, "start": self.start, "end": self.end}
        if self.note_en:
            out.update(note_en=self.note_en, note_ar=self.note_ar)
        if self.wajh:
            out["wajh"] = self.wajh
        return out


@dataclass
class AyahResult:
    words: list
    annotations: list = field(default_factory=list)

    def rules_at(self, word, index):
        found = [a.rule for a in self.annotations if a.word == word and a.start <= index < a.end]
        return sorted(dict.fromkeys(found), key=PRIORITY.__getitem__)


# (skeleton regex, surah or None, rule, letter to mark (or None for the whole word), note en, note ar, wajh)
LEXICAL = [
    (r"^[وف]?[يم]اجوج$", None, "ibdal", "ا", "Qālūn: يَاجُوجَ / مَاجُوجَ without hamza.", "قالون: بإبدال الهمزة ألفًا.", None),
    (r"^[وف]?موصده$", None, "ibdal", "و", "Qālūn: مُوصَدَة, the hamza becomes wāw.", "قالون: بإبدال الهمزة واوًا.", None),
    (r"^بيس$", 7, "ibdal", "ي", "Qālūn: بِيسٍ, the hamza becomes yāʾ (al-Aʿrāf only).", "قالون: «بيسٍ» بالأعراف بإبدال الهمزة ياء.", None),
    (r"^وريا$", 19, "ibdal", "ي", "Qālūn: وَرِيًّا, hamza changed to yāʾ and merged.", "قالون: «وريًّا» بإبدال الهمزة ياء وإدغامها.", None),
    (r"^منساته$", 34, "ibdal", "ا", "Qālūn: مِنسَاتَهُ, hamza changed to alif.", "قالون: «منساته» بإبدال الهمزة ألفًا.", None),
    (r"^سال$", 70, "ibdal", "ا", "Qālūn: سَالَ, hamza changed to alif.", "قالون: «سال» بإبدال الهمزة ألفًا.", None),
    (r"^لاهب$", 19, "ibdal", "ء", "Qālūn: hamza kept (preferred) or changed to yāʾ (لِيَهَبَ).", "لقالون التحقيق وهو المقدم، والإبدال ياء.",
     ["tahqīq (preferred)", "ibdāl: لِيَهَبَ"]),
    (r"^بالسو$", 12, "ibdal", "و", "Qālūn: wāw doubled (preferred) or the first hamza eased.", "قالون: الإبدال مع الإدغام وهو المقدم، والتسهيل.",
     ["ibdāl + idghām (preferred)", "tas-hīl"]),
    (r"^[وف]?الصبين$|^[وف]?الصبون$|^يضهون$", None, "hadhf", None, "Qālūn reads it without the hamza.", "حذف قالون الهمزة.", None),
    (r"^ءالن$", 10, "naql", None, "Qālūn: naql onto the lām; the alif may be held 6 or 2, or the hamza eased.",
     "لقالون فيها النقل مع ثلاثة أوجه: الإبدال مع الإشباع، والإبدال مع القصر، والتسهيل.",
     ["ibdāl, 6", "ibdāl, 2", "tas-hīl"]),
    (r"^ردا$", 28, "naql", None, "Qālūn: the hamza's vowel moves to the dāl: رِدًا.", "نقل قالون حركة الهمز إلى الدال: «رِدًا».", None),
    (r"^[وفب]?التوري[ةه]$", None, "taqlil", "ي", "", "", None),
    (r"^يوده$", 3, "ha_kinaya_qasr", "ه", "", "", None),
    (r"^نوته$", (3, 42), "ha_kinaya_qasr", "ه", "", "", None),
    (r"^[وف]?نوله$|^[وف]?نصله$", 4, "ha_kinaya_qasr", "ه", "", "", None),
    (r"^ارجه$", (7, 26), "ha_kinaya_qasr", "ه", "", "", None),
    (r"^فالقه$", 27, "ha_kinaya_qasr", "ه", "", "", None),
    (r"^ويتقه$", 24, "ha_kinaya_qasr", "ه", "", "", None),
    (r"^يرضه$", 39, "ha_kinaya_qasr", "ه", "", "", None),
    (r"^ياته$", 20, "ha_kinaya_both", "ه", "", "", None),
    (r"^مكني$", 18, "idgham_riwaya", "ن", "Two nūns merged into one (idghām kabīr).", "إدغام النون الأولى في الثانية.", None),
]


EXPLAINED = {"tasheel", "ibdal", "naql", "hadhf", "imala", "ishmam", "ikhtilas", "ha_sukun", "ha_kinaya_qasr",
             "ha_kinaya_both", "taqlil", "idgham_riwaya", "isqat"}


class _Ayah:
    def __init__(self, text, surah, ayah):
        self.words = ayah_words(text)
        self.seq = [c for _, cs in self.words for c in cs]
        for i, c in enumerate(self.seq):
            c.flags["gi"] = i
        self.surah, self.ayah = surah, ayah
        self.out = []
        self.muq = set()
        if surah in OPENING_SURAHS and ayah == 1 or (surah, ayah) == (42, 2):
            for w, (display, _) in enumerate(self.words):
                if bare_letters(display) not in MUQATTAAT:
                    break
                self.muq.add(w)

    # ── navigation ──────────────────────────────────────────────────────────
    def add(self, rule, c, end=None, note=("", ""), wajh=None):
        end_index = (end.index if end is not None else c.index) + 1
        self.out.append(Annotation(rule, c.word, c.index, end_index, note[0], note[1], wajh))

    def word(self, c):
        return self.words[c.word][1]

    def prev_in_word(self, c):
        return self.word(c)[c.index - 1] if c.index else None

    def next_in_word(self, c):
        cs = self.word(c)
        return cs[c.index + 1] if c.index + 1 < len(cs) else None

    def next_p(self, c):
        crossed = False
        for d in self.seq[c.flags["gi"] + 1:]:
            if d.flags.get("silent"):
                crossed |= d.flags["silent"] == "wasl"
                continue
            return d, crossed
        return None, crossed

    def prev_p(self, c, same_word=False):
        for d in reversed(self.seq[:c.flags["gi"]]):
            if same_word and d.word != c.word:
                return None
            if not d.flags.get("silent"):
                return d
        return None

    # ── classification ──────────────────────────────────────────────────────
    def classify(self):
        for w, (_, cs) in enumerate(self.words):
            if w in self.muq:
                continue
            for c in cs:
                prev, nxt = self.prev_in_word(c), self.next_in_word(c)
                plain = not (c.vowel or c.sukun or c.shadda or c.has(MADDAH + DAGGER + DOTS))
                if c.index == 0 and c.base == "ا":
                    if self.changed_hamza(c):
                        c.flags["changed_hamza"] = True
                    else:
                        c.flags["silent"] = "wasl"
                elif c.base == "ا" and c.sukun:
                    c.flags["silent"] = "letter"
                elif c.letter == "و" and c.sukun and prev and prev.vowel == "u":
                    c.flags["silent"] = "letter"
                elif c.base == "ا" and plain and is_prefix(cs[:c.index]) and nxt and (not nxt.vowel or nxt.shadda):
                    c.flags["silent"] = "wasl"
                elif c.base in "اى" and c.tanwin:
                    c.flags["seat"] = True
                elif c.base in "اى" and plain and prev and prev.tanwin:
                    c.flags["seat"] = True
            for c in cs:
                prev, nxt = self.prev_in_word(c), self.next_in_word(c)
                if c.letter != "ل" or not prev or not nxt:
                    continue
                article = prev.flags.get("silent") == "wasl" or (
                    prev.letter == "ل" and prev.vowel == "i" and (prev.index == 0 or is_prefix(cs[:prev.index])))
                if not article or c.vowel:
                    continue
                if c.sukun:
                    c.flags["article"] = "qamari"
                elif nxt.shadda:
                    c.flags["silent"] = "shamsi"
                else:
                    c.flags["article"] = "geminate"  # الذين / التي / لله: the written lām is doubled
            for c in cs:
                if c.flags.get("silent") or c.flags.get("seat"):
                    continue
                prev = self.prev_in_word(c)
                if c.has(SMALL_WAW):
                    c.flags["silah"] = "u"
                elif c.has(SMALL_YEH + SMALL_HIGH_YEH):
                    c.flags["silah"] = "i"
                if c.has(DAGGER) and not c.has(DOTS):
                    c.flags["long"] = "a"
                elif c.has(DOTS) and not (c.has(MADDAH) and c.has(ROUND_ZERO)):
                    continue
                elif c.hamza and c.has(MADDAH):
                    c.flags["long"] = "a"
                elif c.vowel or c.sukun or c.shadda:
                    if c.letter in "وي" and c.sukun and prev and prev.vowel == "a":
                        c.flags["lin"] = True
                    continue
                elif c.letter == "ا" and (c.has(MADDAH) or (prev and prev.vowel == "a" and not prev.tanwin)):
                    c.flags["long"] = "a"
                elif c.letter == "ى" and prev and prev.vowel == "a":
                    c.flags["long"] = "a"
                elif c.letter == "و" and prev and prev.vowel == "u":
                    c.flags["long"] = "u"
                elif c.letter == "ي" and prev and prev.vowel == "i":
                    c.flags["long"] = "i"
        pronounced = [c for c in self.seq if not c.flags.get("silent")]
        self.last = pronounced[-1] if pronounced else None

    def changed_hamza(self, c):
        """Word-initial alif after a hamza-final word: the eased/changed second
        of two hamzas (يَشَآءُ اِ۪لَىٰ), not hamzat al-waṣl (no article lām)."""
        if c.word == 0 or not c.has(DOT_HIGH + DOT_LOW):
            return False
        nxt = self.next_in_word(c)
        if nxt is None or (nxt.letter == "ل" and not nxt.vowel):
            return False
        prev_word = self.words[c.word - 1][1]
        last = prev_word[-1]
        return last.hamza and bool(last.vowel)

    # ── rules ───────────────────────────────────────────────────────────────
    def run(self, hafs=None):
        self.classify()
        for w, (display, cs) in enumerate(self.words):
            if w in self.muq:
                self.muqattaat(w, cs)
                continue
            for c in cs:
                silent = c.flags.get("silent")
                if silent:
                    self.add({"wasl": "hamzat_wasl", "letter": "silent", "shamsi": "idgham_shamsi"}[silent], c)
                    continue
                for rule in (self.nun_tanwin, self.mim, self.ghunna, self.qalqala, self.ra, self.madd,
                             self.idgham_harf, self.marks):
                    rule(c)
                if c.letter in ISTILA:
                    self.add("tafkhim_letter", c)
                if c.flags.get("article") == "qamari":
                    self.add("izhar_qamari", c)
                if c is self.last and (c.flags.get("seat") or (c.tanwin == "a" and c.letter != "ة")):
                    self.add("madd_iwad", c)
            self.allah(w, display, cs)
            self.lexical(w, display, cs)
            self.ha_sukun(cs, display)
        self.phrases()
        if hafs:
            self.hafs_diffs(hafs)
        return AyahResult(self.words, self.out)

    def hafs_diffs(self, hafs):
        for w, (display, cs) in enumerate(self.words):
            h = hafs[w] if w < len(hafs) else None
            if not h or w in self.muq or ALLAH.match(bare_letters(display)):
                continue
            if {a.rule for a in self.out if a.word == w} & EXPLAINED:
                continue
            found = hafs_mod.classify(display, cs, h, self.words[w + 1][1] if w + 1 < len(self.words) else None)
            if not found:
                continue
            rule, index, hafs_word = found
            if rule == "isqat":
                self.out = [a for a in self.out if not (a.word == w and a.start == index and a.rule.startswith("madd_"))]
                self.add(rule, cs[index], wajh=RULES["isqat"]["wajh"])
            elif index is not None:
                self.add(rule, cs[index])
            else:
                self.add(rule, cs[0], cs[-1], (f"Ḥafṣ reads: {hafs_word}", f"رواية حفص: {hafs_word}"))

    def muqattaat(self, w, cs):
        for c in cs:
            if c.has(MADDAH):
                if c.letter == "ع" and self.surah in (19, 42):
                    self.add("madd_ayn", c, wajh=RULES["madd_ayn"]["wajh"])
                elif c.letter == "م" and self.surah == 3 and len(self.words) > 1:
                    self.add("madd_lazim_harfi", c, note=("Joined to اَ۬للَّهُ the mīm may be held 6 or 2.",
                                                          "عند وصل «الم» بلفظ الجلالة يجوز في الميم المد والقصر."), wajh=["6", "2"])
                else:
                    self.add("madd_lazim_harfi", c)
            elif c.letter in "حيطهر":
                self.add("madd_harfi_tabii", c)
            if c.letter in "نم" and c.shadda:
                self.add("ghunna", c)
        if bare_letters(self.words[w][0]) == "طسم":
            sin = next((c for c in cs if c.letter == "س"), None)
            if sin:
                self.add("idgham_riwaya", sin, note=("Qālūn merges the nūn of sīn into mīm.", "أدغم قالون نون «سين» في الميم."))

    def nun_tanwin(self, c):
        is_nun = (c.letter == "ن" and not c.vowel and not c.shadda) or c.has(SMALL_HIGH_NOON)
        if not (is_nun or c.tanwin or c.has(IQLAB_MEEM)):
            return
        after, crossed = self.next_p(c)
        if after is None or crossed:
            return
        if c.has(IQLAB_MEEM) or after.letter == "ب":
            self.add("iqlab", c)
        elif after.letter in THROAT:
            self.add("izhar_halqi", c)
        elif after.letter in "ينمو":
            self.add("izhar_mutlaq" if after.word == c.word else "idgham_ghunna", c)
        elif after.letter in "لر":
            self.add("idgham_no_ghunna", c)
        elif after.letter in IKHFA:
            self.add("ikhfa", c)

    def mim(self, c):
        if c.letter != "م" or c.shadda or c.tanwin:
            return
        after, crossed = self.next_p(c)
        if not c.vowel and after is not None and not crossed:
            if after.letter == "ب":
                self.add("ikhfa_shafawi", c)
            elif after.letter == "م":
                self.add("idgham_shafawi", c)
            elif after.letter in "وف":
                self.add("izhar_shafawi", c)
        prev = self.prev_in_word(c)
        is_jam = (self.next_in_word(c) is None and prev is not None and prev.letter in "هكتء"
                  and (prev.vowel == "u" or (prev.vowel == "i" and prev.letter == "ه")))
        if is_jam and not c.vowel and after is not None and not crossed:
            before_hamza = after.hamza and after.index == 0
            self.add("mim_jam", c, wajh=RULES["mim_jam"]["wajh"],
                     note=("Before a hamza the ṣilah way becomes a separated madd (2 or 4)." if before_hamza else "",
                           "قبل الهمز تكون الصلة من المد المنفصل: القصر والتوسط." if before_hamza else ""))

    def ghunna(self, c):
        if c.letter in "نم" and c.shadda:
            self.add("ghunna", c)

    def qalqala(self, c):
        if c.letter not in QALQALA:
            return
        if c is self.last and c.tanwin != "a":
            self.add("qalqala_waqf", c)
        elif c.sukun:
            self.add("qalqala", c)

    def ra(self, c):
        if c.letter != "ر":
            return
        waqf = c is self.last and c.tanwin != "a"
        if not waqf and c.vowel:
            return self.add("ra_tarqiq" if c.vowel == "i" else "ra_tafkhim", c)
        prev = self.prev_p(c, same_word=True)
        if not waqf:
            before = self.prev_in_word(c)
            if before is not None and before.flags.get("silent") == "wasl":
                return self.add("ra_tafkhim", c, note=("The kasra before it is only for starting (ʿāriḍ).", "الكسر قبلها عارض."))
            if prev is not None and prev.vowel == "i":
                nxt = self.next_in_word(c)
                if nxt is not None and nxt.letter in ISTILA:
                    if nxt.vowel == "i":
                        return self.add("ra_both", c, wajh=RULES["ra_both"]["wajh"])
                    return self.add("ra_tafkhim", c, note=("A heavy letter follows it in the word.", "بعدها حرف استعلاء غير مكسور."))
                return self.add("ra_tarqiq", c)
            return self.add("ra_tafkhim", c)
        # Stopping: the rāʾ becomes sākin and takes its quality from what precedes it.
        if prev is None:
            return self.add("ra_tafkhim", c)
        before = self.prev_p(prev, same_word=True)
        if prev.flags.get("long") == "i" or (prev.letter == "ي" and prev.sukun) or prev.vowel == "i" \
                or (prev.flags.get("long") == "a" and before is not None and before.has(DOT_LOW)):
            return self.add("ra_tarqiq", c)
        if prev.sukun and before is not None and before.vowel == "i":
            if prev.letter in ISTILA:
                preferred = ("light (preferred)", "heavy") if prev.letter == "ط" else ("heavy (preferred)", "light")
                return self.add("ra_both", c, wajh=list(preferred))
            return self.add("ra_tarqiq", c)
        self.add("ra_tafkhim", c)

    def madd(self, c):
        silah = c.flags.get("silah")
        if silah and not (c.has(SMALL_HIGH_YEH) and c.hamza):
            after, crossed = self.next_p(c)
            if after is None or crossed:
                return
            if after.hamza and after.index == 0:
                self.add("madd_munfasil", c, note=("Ṣilah before a hamza counts as a separated madd.",
                                                   "الصلة قبل الهمز من المد المنفصل."), wajh=RULES["madd_munfasil"]["wajh"])
            elif c.letter == "ه":
                self.add("madd_silah", c)
            else:
                self.add("madd_tabii", c)
            return
        if c.flags.get("lin"):
            after, _ = self.next_p(c)
            if after is not None and after is self.last and after.word == c.word:
                self.add("madd_lin", c, wajh=["2", "4", "6"])
            return
        if not c.flags.get("long"):
            return
        after, crossed = self.next_p(c)
        prev = self.prev_in_word(c)
        badal = (c.hamza and c.has(MADDAH)) or (prev is not None and prev.hamza)
        same = after is not None and after.word == c.word
        if c.has(ROUND_ZERO) and c.has(MADDAH):
            return self.add("madd_changed_hamza", c, wajh=RULES["madd_changed_hamza"]["wajh"])
        if c.hamza and c.has(DAGGER) and after is not None and same and after.has(DOTS):
            return  # idkhāl alif between two hamzas: natural length, covered by tas-hīl
        vocative = c.index == 0 and c.letter in "يه" and c.has(DAGGER)
        if vocative and same and (after.hamza or after.has(DOTS)):
            return self.add("madd_munfasil", c, note=("Yā of calling / hā of attention before a hamza: a separated madd.",
                                                      "ياء النداء أو هاء التنبيه قبل الهمز: مد منفصل حكمي."),
                            wajh=RULES["madd_munfasil"]["wajh"])
        if after is None:
            rule = "madd_badal" if badal else "madd_tabii"
        elif same and after.hamza:
            if after.has(DOTS):
                return self.add("madd_changed_hamza", c, wajh=RULES["madd_changed_hamza"]["wajh"])
            rule = "madd_muttasil_waqf" if after is self.last else "madd_muttasil"
        elif same and (after.shadda or after.sukun):
            rule = "madd_lazim"
        elif not same:
            if crossed:
                rule = "wasl_drop"
            elif after.hamza and after.index == 0:
                rule = "madd_munfasil"
            else:
                rule = "madd_badal" if badal else "madd_tabii"
        elif after is self.last:
            rule = "madd_arid"
        else:
            rule = "madd_badal" if badal else "madd_tabii"
        self.add(rule, c, wajh=RULES[rule].get("wajh") if rule in ("madd_munfasil", "madd_muttasil_waqf", "madd_arid") else None)

    def idgham_harf(self, c):
        if c.letter in "نم" or not c.bare or c.flags.get("long") or c.flags.get("seat") or c.flags.get("article") \
                or c.letter in "اىءوي" or c.has(DOTS + DAGGER + MADDAH):
            return
        after, crossed = self.next_p(c)
        if after is None or crossed or not after.shadda or not (after.word == c.word or after.index == 0):
            return
        if after.letter == c.letter:
            kind = ("Same letter (mithlayn).", "متماثلان.")
        elif MAKHRAJ.get(after.letter) == MAKHRAJ.get(c.letter):
            kind = ("Same articulation point (mutajānisayn)." + (" The heaviness of ṭāʾ stays (incomplete idghām)." if c.letter == "ط" else ""),
                    "متجانسان." + (" مع بقاء صفة الإطباق (إدغام ناقص)." if c.letter == "ط" else ""))
        else:
            kind = ("Close articulation points (mutaqāribayn)." + (" Complete idghām is preferred." if c.letter == "ق" else ""),
                    "متقاربان." + (" والإدغام الكامل هو المقدم." if c.letter == "ق" else ""))
        self.add("idgham_harf", c, note=kind)

    def marks(self, c):
        if c.flags.get("changed_hamza"):
            return self.second_hamza(c)
        if not c.has(DOTS) or c.flags.get("silent"):
            return
        prev = self.prev_in_word(c)
        if c.has(ROUND_ZERO):
            final = self.next_in_word(c) is None
            note = ("The first of two hamzas meeting across words is eased.", "تُسهّل الأولى من الهمزتين المتفقتين من كلمتين.") if final \
                else ("Two hamzas in one word: the second is eased, with an alif between them (idkhāl).", "تسهيل الثانية مع الإدخال.")
            return self.add("tasheel", c, note=note)
        if c.has(DOT_HIGH):
            if c.base == "ا" or c.has(DAGGER):
                if prev is not None and prev.hamza and prev.has(DAGGER):
                    note = ("Two hamzas in one word: the second is eased, with an alif between them (idkhāl).", "تسهيل الثانية مع الإدخال.")
                elif c.letter == "ر" or bare_letters(self.words[c.word][0]).startswith("هان"):
                    note = ("Qālūn eases this hamza (أرأيت / ها أنتم).", "سهّل قالون هذه الهمزة.")
                else:
                    note = ("Second hamza eased without idkhāl.", "تسهيل الثانية بلا إدخال.")
                return self.add("tasheel", c, note=note)
            if c.letter == "س":
                return self.add("ishmam", c, note=("Kasra of sīn blended with ḍamma (سِيءَ / سِيئَتْ).", "إشمام كسرة السين الضم."))
            if c.letter == "م" and bare_letters(self.words[c.word][0]).endswith("امنا"):
                return self.add("ikhtilas", c, note=("تأمنّا: ikhtilās of the first nūn (preferred) or idghām with ishmām.",
                                                    "«تأمنّا»: الاختلاس وهو المقدم، والإدغام مع الإشمام."),
                                wajh=["ikhtilās (preferred)", "idghām + ishmām"])
            return self.add("ikhtilas", c)
        if c.letter == "ه":
            return self.add("imala", c)
        if c.letter == "ي":
            return self.add("tasheel", c, note=("أئمة: the second hamza is eased without idkhāl.", "«أئمة»: تسهيل الثانية بلا إدخال."))
        self.add("ikhtilas", c)

    def second_hamza(self, c):
        first = self.words[c.word - 1][1][-1]
        pv = first.vowel
        v = c.vowel or ("i" if c.has(DOT_LOW) else "a")
        if pv == "a":
            self.add("tasheel", c, note=("Second hamza eased (two hamzas across words).", "تسهيل الهمزة الثانية من كلمتين."))
        elif pv == "i" and v == "a":
            self.add("ibdal", c, note=("Second hamza becomes a pure yāʾ.", "إبدال الهمزة الثانية ياء خالصة."))
        elif pv == "u" and v == "a":
            self.add("ibdal", c, note=("Second hamza becomes a pure wāw.", "إبدال الهمزة الثانية واوًا خالصة."))
        elif pv == "u" and v == "i":
            self.add("ibdal", c, note=("Second hamza becomes wāw (preferred) or is eased.", "إبدالها واوًا وهو المقدم، أو تسهيلها."),
                     wajh=["ibdāl: wāw (preferred)", "tas-hīl"])
        else:
            self.add("tasheel", c)

    def allah(self, w, display, cs):
        if not ALLAH.match(bare_letters(display)):
            return
        lam = [c for c in cs if c.letter == "ل"][-1]
        prev = next((c for c in reversed(cs[:lam.index]) if not c.flags.get("silent")), None)
        if prev is not None:
            vowel = prev.vowel or prev.flags.get("long", "")
        elif w == 0:
            vowel = "a"
        else:
            vowel = cs[0].vowel or self.words[w - 1][1][-1].vowel or "a"
        self.add("lam_tarqiq" if vowel == "i" else "lam_tafkhim", lam)

    def lexical(self, w, display, cs):
        skeleton = bare_letters(display)
        for pattern, surah, rule, letter, en, ar, wajh in LEXICAL:
            if (surah is None or self.surah in (surah if isinstance(surah, tuple) else (surah,))) and re.search(pattern, skeleton):
                targets = [c for c in cs if c.letter == letter or (letter == "ء" and c.hamza)] if letter else []
                start = targets[0] if targets else cs[0]
                end = targets[0] if targets else cs[-1]
                self.add(rule, start, end, (en, ar), wajh or RULES[rule].get("wajh"))

    def ha_sukun(self, cs, display):
        skeleton = bare_letters(display)
        if skeleton in ("وهو", "فهو", "لهو", "وهي", "فهي", "لهي", "هو"):
            ha = next((c for c in cs if c.letter == "ه"), None)
            if ha is not None and ha.sukun:
                self.add("ha_sukun", ha)

    def phrases(self):
        """Two-word riwāyah points: idghām/iẓhār choices and Ḥafṣ's sakt."""
        skeletons = [bare_letters(d) for d, _ in self.words]
        for w in range(len(self.words)):
            here, nxt = skeletons[w], skeletons[w + 1] if w + 1 < len(skeletons) else ""
            cs = self.words[w][1]
            if here == "يلهث" and nxt == "ذلك" or here == "اركب" and nxt == "معنا":
                self.add("idgham_riwaya", cs[-1], note=("Qālūn: idghām (preferred) or iẓhār.", "لقالون الإدغام وهو المقدم والإظهار."),
                         wajh=["idghām (preferred)", "iẓhār"])
            elif here == "ويعذب" and nxt == "من" and cs[-1].bare:
                self.add("idgham_riwaya", cs[-1], note=("Qālūn merges this bāʾ into the mīm.", "أدغم قالون الباء في الميم."))
            elif re.search(r"خذت", here) and any(c.letter == "ذ" and c.bare for c in cs):
                dhal = next(c for c in cs if c.letter == "ذ" and c.bare)
                self.add("idgham_riwaya", dhal, note=("Qālūn merges ذ into ت (أخذتم، اتخذتم).", "أدغم قالون الذال في التاء."))
            elif re.search(r"لبثت", here) and any(c.letter == "ث" and c.sukun for c in cs):
                tha = next(c for c in cs if c.letter == "ث" and c.sukun)
                self.add("idgham_riwaya", tha, note=("Qālūn keeps ث clear before ت (iẓhār).", "أظهر قالون الثاء عند التاء."))
            elif w == 0 and here in ("ن", "يس") and nxt in ("والقلم", "والقران"):
                self.add("idgham_riwaya", cs[-1], note=("Qālūn keeps the nūn of the letter-name clear before wāw.", "أظهر قالون النون عند الواو."))
            if (self.surah, here) in ((18, "عوجا"), (36, "مرقدنا")) or (self.surah, here, nxt) in ((75, "من", "راق"), (83, "بل", "ران")):
                self.add("no_sakt", cs[-1])


def annotate(text, surah=None, ayah=None, hafs=None):
    """hafs: optional list of aligned Ḥafṣ words (None where unaligned)."""
    return _Ayah(text, surah, ayah).run(hafs)


def segments(result):
    """Per word: [(text, [rule ids by priority])], merging equal neighbours.
    Lām-alif and the Allāh ligature are never split across segments."""
    out = []
    for w, (display, cs) in enumerate(result.words):
        groups = []
        for c in cs:
            rules = result.rules_at(w, c.index)
            join = groups and (
                (c.letter == "ا" and groups[-1][2] == "ل")
                or (ALLAH.match(bare_letters(display)) and c.letter in "له" and groups[-1][2] in "ل"))
            if join or (groups and groups[-1][1] == rules):
                text, prev_rules, _ = groups[-1]
                merged = prev_rules if prev_rules == rules else sorted(set(prev_rules) | set(rules), key=PRIORITY.__getitem__)
                groups[-1] = (text + c.text, merged, c.letter)
            else:
                groups.append((c.text, rules, c.letter))
        out.append([(text, rules) for text, rules, _ in groups])
    return out
